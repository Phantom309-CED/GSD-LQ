"""
GSD-LQ v2.0 Full 32-Layer End-to-End Native OpenCL GPU Chat Engine (Mistral-7B-Instruct-v0.3)
Target Hardware: AMD Radeon RX 5600 XT (gfx1010, 5.98 GB VRAM) via Windows OpenCL.dll

Runs 100% of the 32-layer Transformer forward pass inside GPU VRAM:
  - All 224 Attention & MLP weights + lm_head remain packed in hex2d (~3.6 GB VRAM)
  - Unpacks weights on the fly inside GPU LDS shared memory during matvec
  - RMSNorm, RoPE, 2048-token FP16 KV-Cache, GQA Attention, SwiGLU, and LM Head all execute on GPU
  - Zero intermediate PCIe CPU<->GPU transfers during the 32-layer forward pass
  - Continuous multi-turn memory + anti-hallucination prompt + automatic markdown logging
"""

import ctypes
import datetime
import json
import os
import re
import struct
import sys
import time
import numpy as np

V2_DIR = os.path.dirname(os.path.abspath(__file__))
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)
def _get_turbo_codebook(radius: int = 8):
    sqrt3_2 = np.sqrt(3.0) / 2.0
    xy_pts = []
    for r0 in range(-radius, radius + 1):
        for r1 in range(-radius, radius + 1):
            if abs(r0 + r1) <= radius:
                xy_pts.append([r0 + 0.5 * r1, r1 * sqrt3_2])
    return np.array(xy_pts, dtype=np.float32), len(xy_pts)

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(V2_DIR, "mistral_v2_native_hex2d.gsd2")
LOG_FILE = os.path.join(V2_DIR, "GPU_NATIVE_CHAT_LOG.md")

CL_DEVICE_TYPE_GPU = 1 << 2
CL_MEM_READ_WRITE = 1 << 0
CL_MEM_WRITE_ONLY = 1 << 1
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_COPY_HOST_PTR = 1 << 5
CL_TRUE = 1

MAX_SEQ_LEN = 2048

OPENCL_ALL_KERNELS = r"""
inline float half_to_float(ushort h) {
    uint sign = (uint)(h & 0x8000u) << 16;
    uint exp  = (uint)(h >> 10) & 0x1Fu;
    uint mant = (uint)(h & 0x03FFu);
    uint f;
    if (exp == 0u) {
        if (mant == 0u) {
            f = sign;
        } else {
            exp = 1u;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            f = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return as_float(f);
}

inline ushort float_to_half(float val) {
    uint f = as_uint(val);
    uint sign = (f >> 16) & 0x8000u;
    int exp = (int)((f >> 23) & 0xFFu) - 127 + 15;
    uint mant = f & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (ushort)sign;
        mant = (mant | 0x00800000u) >> (1 - exp);
        return (ushort)(sign | ((mant + 0x00001000u) >> 13));
    } else if (exp >= 31) {
        return (ushort)(sign | 0x7C00u);
    }
    return (ushort)(sign | ((uint)exp << 10) | ((mant + 0x00001000u) >> 13));
}

__kernel void rmsnorm_kernel(
    __global const float* x,
    __global const float* weight,
    __global float* out,
    const int n,
    const float eps,
    __local float* lsum
) {
    int tid = get_local_id(0);
    int lsize = get_local_size(0);
    float ss = 0.0f;
    for (int i = tid; i < n; i += lsize) {
        float v = x[i];
        ss += v * v;
    }
    lsum[tid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (tid < s) lsum[tid] += lsum[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_rms = rsqrt(lsum[0] / (float)n + eps);
    for (int i = tid; i < n; i += lsize) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

__kernel void gsd_lowrank_proj(
    __global const float* x,
    __global const ushort* Vt,
    __global const float* S,
    __global float* tmp_k,
    const int cols,
    __local float* lsum
) {
    int j = get_group_id(0);
    int tid = get_local_id(0);
    int lsize = get_local_size(0);

    __global const ushort* vt_row = Vt + (size_t)j * (size_t)cols;
    float acc = 0.0f;
    for (int c = tid; c < cols; c += lsize) {
        acc += x[c] * half_to_float(vt_row[c]);
    }
    lsum[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) lsum[tid] += lsum[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) tmp_k[j] = lsum[0] * S[j];
}

__kernel void gsd_hex2d_matvec_fused(
    __global const float* x,
    __global const uchar* codes,
    __global const ushort* scales,
    __global const float2* xy_table,
    __global const ushort* U,
    __global const float* tmp_k,
    __global const uint* kink_row_ptr,
    __global const ushort* kink_cols,
    __global const float* kink_deltas,
    __global float* y,
    const int rows,
    const int cols,
    const int k_dim,
    const int n_pts,
    __local float* lsum,
    __local float2* l_xy
) {
    int r = get_group_id(0);
    if (r >= rows) return;
    int tid = get_local_id(0);
    int lsize = get_local_size(0);

    for (int i = tid; i < n_pts; i += lsize) {
        l_xy[i] = xy_table[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    size_t row_base = (size_t)r * (size_t)cols;
    float acc = 0.0f;

    if ((cols & 63) == 0) {
        size_t pair_base = row_base >> 1;
        size_t blk_base  = row_base >> 6;
        int num_pairs = cols >> 1;
        for (int p = tid; p < num_pairs; p += lsize) {
            float scale = half_to_float(scales[blk_base + (size_t)(p >> 5)]);
            float2 pt = l_xy[codes[pair_base + (size_t)p]];
            float2 xv = vload2(0, x + (p << 1));
            acc += (pt.x * xv.x + pt.y * xv.y) * scale;
        }
    } else {
        for (int c = tid; c < cols; c += lsize) {
            size_t idx = row_base + (size_t)c;
            float scale = half_to_float(scales[idx >> 6]);
            float2 pt = l_xy[codes[idx >> 1]];
            float w = ((idx & 1u) == 0u) ? pt.x : pt.y;
            acc += (w * scale) * x[c];
        }
    }

    uint k_start = kink_row_ptr[r];
    uint k_end   = kink_row_ptr[r + 1];
    for (uint ki = k_start + (uint)tid; ki < k_end; ki += (uint)lsize) {
        acc += kink_deltas[ki] * x[kink_cols[ki]];
    }

    if (k_dim > 0) {
        size_t u_base = (size_t)r * (size_t)k_dim;
        for (int j = tid; j < k_dim; j += lsize) {
            acc += half_to_float(U[u_base + (size_t)j]) * tmp_k[j];
        }
    }

    lsum[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) lsum[tid] += lsum[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) y[r] = lsum[0];
}

__kernel void rope_and_store_kv_kernel(
    __global float* q,
    __global const float* k,
    __global const float* v,
    __global ushort* k_cache,
    __global ushort* v_cache,
    const int layer_idx,
    const int pos,
    const int max_seq,
    const int n_heads,
    const int n_kv_heads,
    const int head_dim,
    const float theta,
    const int interleaved
) {
    int gid = get_global_id(0);
    int half_d = head_dim >> 1;
    int total_q_pairs = n_heads * half_d;
    int total_kv_pairs = n_kv_heads * half_d;

    if (gid < total_q_pairs) {
        int h = gid / half_d;
        int i = gid % half_d;
        float freq = 1.0f / pow(theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cos(angle);
        float s = sin(angle);
        int base = h * head_dim;
        int i0 = interleaved ? (base + 2 * i) : (base + i);
        int i1 = interleaved ? (base + 2 * i + 1) : (base + i + half_d);
        float x0 = q[i0];
        float x1 = q[i1];
        q[i0] = x0 * c - x1 * s;
        q[i1] = x0 * s + x1 * c;
    }

    if (gid < total_kv_pairs) {
        int h = gid / half_d;
        int i = gid % half_d;
        float freq = 1.0f / pow(theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cos(angle);
        float s = sin(angle);
        int base = h * head_dim;
        int i0 = interleaved ? (base + 2 * i) : (base + i);
        int i1 = interleaved ? (base + 2 * i + 1) : (base + i + half_d);
        float k0 = k[i0];
        float k1 = k[i1];
        float rk0 = k0 * c - k1 * s;
        float rk1 = k0 * s + k1 * c;

        // KV cache layout: (n_layers, max_seq, n_kv_heads, head_dim)
        size_t cache_base = (((size_t)layer_idx * (size_t)max_seq + (size_t)pos) * (size_t)n_kv_heads + (size_t)h) * (size_t)head_dim;
        k_cache[cache_base + (size_t)(i0 - base)] = float_to_half(rk0);
        k_cache[cache_base + (size_t)(i1 - base)] = float_to_half(rk1);
        v_cache[cache_base + (size_t)(i0 - base)] = float_to_half(v[i0]);
        v_cache[cache_base + (size_t)(i1 - base)] = float_to_half(v[i1]);
    }
}

__kernel void gqa_attention_kernel(
    __global const float* q,
    __global const ushort* k_cache,
    __global const ushort* v_cache,
    __global float* attn_out,
    const int layer_idx,
    const int pos,
    const int max_seq,
    const int n_heads,
    const int n_kv_heads,
    const int head_dim,
    __local float* sdata,
    __local float* scores
) {
    int h = get_group_id(0);
    if (h >= n_heads) return;
    int tid = get_local_id(0); // 0..127 (head_dim == 128)
    int kv_h = h / (n_heads / n_kv_heads);

    float q_val = q[h * head_dim + tid];
    float inv_sqrt_d = rsqrt((float)head_dim);
    size_t layer_base = (size_t)layer_idx * (size_t)max_seq * (size_t)n_kv_heads * (size_t)head_dim;
    size_t kv_stride = (size_t)n_kv_heads * (size_t)head_dim;
    size_t head_off = (size_t)kv_h * (size_t)head_dim + (size_t)tid;

    for (int t = 0; t <= pos; t++) {
        float k_val = half_to_float(k_cache[layer_base + (size_t)t * kv_stride + head_off]);
        sdata[tid] = q_val * k_val;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) {
            scores[t] = sdata[0] * inv_sqrt_d;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Numerically stable softmax across scores[0..pos]
    float max_s = -1e30f;
    for (int t = tid; t <= pos; t += head_dim) {
        max_s = fmax(max_s, scores[t]);
    }
    sdata[tid] = max_s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float g_max = sdata[0];

    float sum_e = 0.0f;
    for (int t = tid; t <= pos; t += head_dim) {
        float e = exp(scores[t] - g_max);
        scores[t] = e;
        sum_e += e;
    }
    sdata[tid] = sum_e;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_sum = 1.0f / sdata[0];
    for (int t = tid; t <= pos; t += head_dim) {
        scores[t] *= inv_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Weighted sum of V
    float out_val = 0.0f;
    for (int t = 0; t <= pos; t++) {
        float v_val = half_to_float(v_cache[layer_base + (size_t)t * kv_stride + head_off]);
        out_val += scores[t] * v_val;
    }
    attn_out[h * head_dim + tid] = out_val;
}

__kernel void vec_add_kernel(__global float* x, __global const float* delta, const int n) {
    int i = get_global_id(0);
    if (i < n) x[i] += delta[i];
}

__kernel void swiglu_mul_kernel(__global const float* gate, __global const float* up, __global float* out, const int n) {
    int i = get_global_id(0);
    if (i < n) {
        float g = clamp(gate[i], -30.0f, 30.0f);
        float silu = gate[i] / (1.0f + exp(-g));
        out[i] = silu * up[i];
    }
}
"""


class OpenCLRuntime:
    def __init__(self):
        self.cl = ctypes.WinDLL("OpenCL.dll")
        self._bind_api()

        n_plat = ctypes.c_uint()
        self.cl.clGetPlatformIDs(0, None, ctypes.byref(n_plat))
        plats = (ctypes.c_void_p * n_plat.value)()
        self.cl.clGetPlatformIDs(n_plat.value, plats, None)

        devs = (ctypes.c_void_p * 4)()
        n_dev = ctypes.c_uint()
        err = self.cl.clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_GPU, 4, devs, ctypes.byref(n_dev))
        if err != 0 or n_dev.value == 0:
            raise RuntimeError(f"No OpenCL GPU found (err={err})")
        self.device = devs[0]

        buf = ctypes.create_string_buffer(256)
        self.cl.clGetDeviceInfo(self.device, 0x102B, 256, buf, None)
        self.device_name = buf.value.decode("utf-8", errors="replace")

        mem = ctypes.c_ulonglong()
        self.cl.clGetDeviceInfo(self.device, 0x101F, 8, ctypes.byref(mem), None)
        self.total_vram_mb = mem.value / (1024 * 1024)

        err_c = ctypes.c_int()
        self.ctx = self.cl.clCreateContext(None, 1, ctypes.byref(ctypes.c_void_p(self.device)), None, None, ctypes.byref(err_c))
        self.queue = self.cl.clCreateCommandQueue(self.ctx, self.device, 0, ctypes.byref(err_c))

        src_bytes = OPENCL_ALL_KERNELS.encode("utf-8")
        src_ptr = ctypes.c_char_p(src_bytes)
        src_len = ctypes.c_size_t(len(src_bytes))
        self.program = self.cl.clCreateProgramWithSource(self.ctx, 1, ctypes.byref(src_ptr), ctypes.byref(src_len), ctypes.byref(err_c))
        err = self.cl.clBuildProgram(self.program, 1, ctypes.byref(ctypes.c_void_p(self.device)), b"-cl-fast-relaxed-math", None, None)
        if err != 0:
            log_size = ctypes.c_size_t()
            self.cl.clGetProgramBuildInfo(self.program, self.device, 0x1183, 0, None, ctypes.byref(log_size))
            log_buf = ctypes.create_string_buffer(log_size.value)
            self.cl.clGetProgramBuildInfo(self.program, self.device, 0x1183, log_size.value, log_buf, None)
            raise RuntimeError(f"OpenCL Build Failed ({err}):\n{log_buf.value.decode()}")

        self.k_rmsnorm = self.cl.clCreateKernel(self.program, b"rmsnorm_kernel", ctypes.byref(err_c))
        self.k_lowrank = self.cl.clCreateKernel(self.program, b"gsd_lowrank_proj", ctypes.byref(err_c))
        self.k_fused = self.cl.clCreateKernel(self.program, b"gsd_hex2d_matvec_fused", ctypes.byref(err_c))
        self.k_rope_kv = self.cl.clCreateKernel(self.program, b"rope_and_store_kv_kernel", ctypes.byref(err_c))
        self.k_gqa = self.cl.clCreateKernel(self.program, b"gqa_attention_kernel", ctypes.byref(err_c))
        self.k_add = self.cl.clCreateKernel(self.program, b"vec_add_kernel", ctypes.byref(err_c))
        self.k_swiglu = self.cl.clCreateKernel(self.program, b"swiglu_mul_kernel", ctypes.byref(err_c))

        self.allocated_bytes = 0
        self.all_buffers = []

    def _bind_api(self):
        c = self.cl
        vp, u32, i32, u64, sz = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_size_t
        c.clGetPlatformIDs.argtypes = [u32, ctypes.POINTER(vp), ctypes.POINTER(u32)]
        c.clGetDeviceIDs.argtypes = [vp, u64, u32, ctypes.POINTER(vp), ctypes.POINTER(u32)]
        c.clGetDeviceInfo.argtypes = [vp, u32, sz, vp, ctypes.POINTER(sz)]
        c.clCreateContext.argtypes = [vp, u32, ctypes.POINTER(vp), vp, vp, ctypes.POINTER(i32)]
        c.clCreateContext.restype = vp
        c.clCreateCommandQueue.argtypes = [vp, vp, u64, ctypes.POINTER(i32)]
        c.clCreateCommandQueue.restype = vp
        c.clCreateProgramWithSource.argtypes = [vp, u32, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(sz), ctypes.POINTER(i32)]
        c.clCreateProgramWithSource.restype = vp
        c.clBuildProgram.argtypes = [vp, u32, ctypes.POINTER(vp), ctypes.c_char_p, vp, vp]
        c.clGetProgramBuildInfo.argtypes = [vp, vp, u32, sz, vp, ctypes.POINTER(sz)]
        c.clCreateKernel.argtypes = [vp, ctypes.c_char_p, ctypes.POINTER(i32)]
        c.clCreateKernel.restype = vp
        c.clCreateBuffer.argtypes = [vp, u64, sz, vp, ctypes.POINTER(i32)]
        c.clCreateBuffer.restype = vp
        c.clEnqueueWriteBuffer.argtypes = [vp, vp, u32, sz, sz, vp, u32, vp, vp]
        c.clEnqueueReadBuffer.argtypes = [vp, vp, u32, sz, sz, vp, u32, vp, vp]
        c.clSetKernelArg.argtypes = [vp, u32, sz, vp]
        c.clEnqueueNDRangeKernel.argtypes = [vp, vp, u32, vp, ctypes.POINTER(sz), ctypes.POINTER(sz), u32, vp, vp]
        c.clFinish.argtypes = [vp]
        c.clReleaseMemObject.argtypes = [vp]

    def upload_bytes(self, raw_bytes: bytes, flags: int = CL_MEM_READ_ONLY):
        nbytes = max(len(raw_bytes), 4)
        if len(raw_bytes) == 0:
            return self.alloc_empty(4, flags)
        buf_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        err_c = ctypes.c_int()
        b = self.cl.clCreateBuffer(self.ctx, flags | CL_MEM_COPY_HOST_PTR, len(raw_bytes), buf_arr.ctypes.data_as(ctypes.c_void_p), ctypes.byref(err_c))
        if err_c.value != 0:
            raise RuntimeError(f"clCreateBuffer failed: {err_c.value} (nbytes={len(raw_bytes)})")
        self.allocated_bytes += len(raw_bytes)
        self.all_buffers.append(b)
        return b

    def alloc_empty(self, nbytes: int, flags: int = CL_MEM_READ_WRITE):
        nbytes = max(nbytes, 4)
        err_c = ctypes.c_int()
        b = self.cl.clCreateBuffer(self.ctx, flags, nbytes, None, ctypes.byref(err_c))
        if err_c.value != 0:
            raise RuntimeError(f"clCreateBuffer(empty) failed: {err_c.value} (nbytes={nbytes})")
        self.allocated_bytes += nbytes
        self.all_buffers.append(b)
        return b

    def write_f32(self, buf, arr: np.ndarray):
        arr_c = np.ascontiguousarray(arr, dtype=np.float32)
        self.cl.clEnqueueWriteBuffer(self.queue, buf, CL_TRUE, 0, arr_c.nbytes, arr_c.ctypes.data_as(ctypes.c_void_p), 0, None, None)

    def read_f32(self, buf, out_arr: np.ndarray):
        self.cl.clEnqueueReadBuffer(self.queue, buf, CL_TRUE, 0, out_arr.nbytes, out_arr.ctypes.data_as(ctypes.c_void_p), 0, None, None)

    def set_args(self, kernel, *args):
        for idx, val in enumerate(args):
            if isinstance(val, tuple) and val[0] == "local":
                self.cl.clSetKernelArg(kernel, idx, val[1], None)
            elif isinstance(val, ctypes.c_void_p):
                self.cl.clSetKernelArg(kernel, idx, ctypes.sizeof(val), ctypes.byref(val))
            elif isinstance(val, float):
                v = ctypes.c_float(val)
                self.cl.clSetKernelArg(kernel, idx, ctypes.sizeof(v), ctypes.byref(v))
            elif isinstance(val, int):
                v = ctypes.c_int(val)
                self.cl.clSetKernelArg(kernel, idx, ctypes.sizeof(v), ctypes.byref(v))
            else:
                raise TypeError(f"Unsupported arg type: {type(val)}")

    def launch(self, kernel, global_size: int, local_size: int):
        g = (ctypes.c_size_t * 1)(global_size)
        l = (ctypes.c_size_t * 1)(local_size)
        err = self.cl.clEnqueueNDRangeKernel(self.queue, kernel, 1, None, g, l, 0, None, None)
        if err != 0:
            raise RuntimeError(f"clEnqueueNDRangeKernel failed: {err}")

    def finish(self):
        self.cl.clFinish(self.queue)

    def release_all(self):
        for b in self.all_buffers:
            self.cl.clReleaseMemObject(b)
        self.all_buffers.clear()


class MistralSentencePieceTokenizer:
    """Self-contained offline SentencePiece BPE tokenizer loaded from GGUF metadata."""

    def __init__(self, tokens, scores, token_types, bos_id=1, eos_id=2):
        self.tokens = tokens
        self.scores = scores
        self.token_types = token_types
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.specials = [t for i, t in enumerate(tokens) if token_types[i] == 3]
        self.specials.sort(key=len, reverse=True)
        self.byte_tokens = {i: int(t[3:5], 16) for i, t in enumerate(tokens) if token_types[i] == 6 and len(t) == 6 and t.startswith("<0x")}

    def _bpe_piece(self, text: str) -> list[int]:
        if not text:
            return []
        syms = []
        for ch in text:
            if ch in self.vocab:
                syms.append(ch)
            else:
                for b in ch.encode("utf-8"):
                    syms.append(f"<0x{b:02X}>")

        while len(syms) > 1:
            best_score = -1e30
            best_pos = -1
            for i in range(len(syms) - 1):
                cand = syms[i] + syms[i + 1]
                tid = self.vocab.get(cand)
                if tid is not None and self.scores[tid] > best_score:
                    best_score = self.scores[tid]
                    best_pos = i
            if best_pos < 0:
                break
            syms = syms[:best_pos] + [syms[best_pos] + syms[best_pos + 1]] + syms[best_pos + 2:]

        return [self.vocab.get(s, 0) for s in syms]

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = [self.bos_id] if add_bos else []
        pattern = "(" + "|".join(re.escape(s) for s in self.specials) + ")"
        parts = re.split(pattern, text)
        first_text = True
        for part in parts:
            if not part:
                continue
            if part in self.vocab and self.token_types[self.vocab[part]] == 3:
                ids.append(self.vocab[part])
                first_text = False
            else:
                norm = part.replace(" ", "\u2581")
                if first_text and not norm.startswith("\u2581"):
                    norm = "\u2581" + norm
                first_text = False
                ids.extend(self._bpe_piece(norm))
        return ids

    def decode(self, ids: list[int]) -> str:
        raw_bytes = bytearray()
        for tid in ids:
            if tid in (self.bos_id, self.eos_id):
                continue
            if tid in self.byte_tokens:
                raw_bytes.append(self.byte_tokens[tid])
            else:
                s = self.tokens[tid].replace("\u2581", " ")
                raw_bytes.extend(s.encode("utf-8"))
        return raw_bytes.decode("utf-8", errors="replace")


class GPUMistralV2Model:
    def __init__(self, gsd2_path: str, rope_interleaved: int = 1):
        self.rope_interleaved = rope_interleaved
        self.cl = OpenCLRuntime()

        print(f"GPU Device Detected : {self.cl.device_name} ({self.cl.total_vram_mb:.0f} MB VRAM)", flush=True)
        print(f"Loading packed model: {gsd2_path}", flush=True)
        t0 = time.time()

        with open(gsd2_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GSD2":
                raise ValueError(f"Invalid GSD2 magic: {magic}")
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen).decode("utf-8"))
            data_base = 12 + hlen

            self.cfg = header["config"]
            self.n_layers = self.cfg["n_layers"]
            self.hidden = self.cfg["hidden_size"]
            self.ffn_size = self.cfg["ffn_size"]
            self.n_heads = self.cfg["n_heads"]
            self.n_kv_heads = self.cfg["n_kv_heads"]
            self.head_dim = self.cfg["head_dim"]
            self.rope_theta = self.cfg["rope_theta"]
            self.rms_eps = self.cfg["rms_eps"]
            self.vocab_size = self.cfg["vocab_size"]
            self.avg_bpw = header["avg_bpw"]

            self.tokenizer = MistralSentencePieceTokenizer(
                self.cfg["tokens"], self.cfg["scores"], self.cfg["token_types"],
                self.cfg["bos_id"], self.cfg["eos_id"]
            )

            tensors = header["tensors"]

            def read_blob(meta: dict) -> bytes:
                f.seek(data_base + meta["offset"])
                return f.read(meta["nbytes"])

            # Keep token embeddings in system RAM (256 MB FP16, 1 row lookup per token)
            emb_meta = tensors["token_embd.weight"]["data"]
            self.embed = np.frombuffer(read_blob(emb_meta), dtype=np.float16).reshape(emb_meta["shape"])

            # Upload A2 hex codebook tables (radius 6 and radius 8)
            self.xy_tables = {}
            for r in (6, 8):
                table_pts, num_pts = _get_turbo_codebook(r)
                self.xy_tables[r] = (
                    self.cl.upload_bytes(table_pts.tobytes()),
                    int(num_pts),
                )

            self.dummy_u16 = self.cl.upload_bytes(np.zeros(2, dtype=np.uint16).tobytes())
            self.dummy_f32 = self.cl.upload_bytes(np.zeros(1, dtype=np.float32).tobytes())

            def load_hex_gpu(tname: str) -> dict:
                m = tensors[tname]
                r = m["radius"]
                xy_buf, n_pts = self.xy_tables[r]
                k_dim = m["k_dim"]
                if k_dim > 0:
                    d_U = self.cl.upload_bytes(read_blob(m["U"]))
                    d_S = self.cl.upload_bytes(read_blob(m["S"]))
                    d_Vt = self.cl.upload_bytes(read_blob(m["Vt"]))
                else:
                    d_U = d_Vt = self.dummy_u16
                    d_S = self.dummy_f32
                return {
                    "rows": m["rows"],
                    "cols": m["cols"],
                    "k_dim": k_dim,
                    "n_pts": n_pts,
                    "xy_table": xy_buf,
                    "codes": self.cl.upload_bytes(read_blob(m["codes"])),
                    "scales": self.cl.upload_bytes(read_blob(m["scales"])),
                    "U": d_U,
                    "S": d_S,
                    "Vt": d_Vt,
                    "row_ptr": self.cl.upload_bytes(read_blob(m["row_ptr"])),
                    "kink_cols": self.cl.upload_bytes(read_blob(m["kink_cols"])),
                    "kink_deltas": self.cl.upload_bytes(read_blob(m["kink_deltas"])),
                }

            self.layers = []
            for l in range(self.n_layers):
                p = f"blk.{l}."
                self.layers.append({
                    "attn_norm": self.cl.upload_bytes(read_blob(tensors[p + "attn_norm.weight"]["data"])),
                    "ffn_norm": self.cl.upload_bytes(read_blob(tensors[p + "ffn_norm.weight"]["data"])),
                    "q": load_hex_gpu(p + "attn_q.weight"),
                    "k": load_hex_gpu(p + "attn_k.weight"),
                    "v": load_hex_gpu(p + "attn_v.weight"),
                    "o": load_hex_gpu(p + "attn_output.weight"),
                    "gate": load_hex_gpu(p + "ffn_gate.weight"),
                    "up": load_hex_gpu(p + "ffn_up.weight"),
                    "down": load_hex_gpu(p + "ffn_down.weight"),
                })

            self.final_norm = self.cl.upload_bytes(read_blob(tensors["output_norm.weight"]["data"]))
            self.lm_head = load_hex_gpu("output.weight")

        # Allocate GPU KV-cache (FP16) and activation scratch buffers
        kv_bytes = self.n_layers * MAX_SEQ_LEN * self.n_kv_heads * self.head_dim * 2
        self.d_k_cache = self.cl.alloc_empty(kv_bytes, CL_MEM_READ_WRITE)
        self.d_v_cache = self.cl.alloc_empty(kv_bytes, CL_MEM_READ_WRITE)

        self.d_x = self.cl.alloc_empty(self.hidden * 4, CL_MEM_READ_WRITE)
        self.d_xb = self.cl.alloc_empty(self.hidden * 4, CL_MEM_READ_WRITE)
        self.d_q = self.cl.alloc_empty(self.hidden * 4, CL_MEM_READ_WRITE)
        self.d_k = self.cl.alloc_empty(self.n_kv_heads * self.head_dim * 4, CL_MEM_READ_WRITE)
        self.d_v = self.cl.alloc_empty(self.n_kv_heads * self.head_dim * 4, CL_MEM_READ_WRITE)
        self.d_attn_out = self.cl.alloc_empty(self.hidden * 4, CL_MEM_READ_WRITE)
        self.d_gate = self.cl.alloc_empty(self.ffn_size * 4, CL_MEM_READ_WRITE)
        self.d_up = self.cl.alloc_empty(self.ffn_size * 4, CL_MEM_READ_WRITE)
        self.d_tmp_k = self.cl.alloc_empty(16 * 4, CL_MEM_READ_WRITE)
        self.d_logits = self.cl.alloc_empty(self.vocab_size * 4, CL_MEM_WRITE_ONLY)

        self.vram_used_mb = self.cl.allocated_bytes / (1024 * 1024)
        print(
            f"Model Loaded 100% Resident in GPU VRAM in {time.time() - t0:.2f}s!\n"
            f"  Packed VRAM Footprint (32 Layers + LM Head + 2K KV Cache): {self.vram_used_mb:.1f} MB "
            f"({self.vram_used_mb / 1024:.2f} GB / {self.cl.total_vram_mb / 1024:.2f} GB)",
            flush=True,
        )

    def _matvec_gpu(self, d_in, layer_gpu: dict, d_out):
        if layer_gpu["k_dim"] > 0:
            self.cl.set_args(
                self.cl.k_lowrank,
                ctypes.c_void_p(d_in),
                ctypes.c_void_p(layer_gpu["Vt"]),
                ctypes.c_void_p(layer_gpu["S"]),
                ctypes.c_void_p(self.d_tmp_k),
                layer_gpu["cols"],
                ("local", 64 * 4),
            )
            self.cl.launch(self.cl.k_lowrank, layer_gpu["k_dim"] * 64, 64)

        self.cl.set_args(
            self.cl.k_fused,
            ctypes.c_void_p(d_in),
            ctypes.c_void_p(layer_gpu["codes"]),
            ctypes.c_void_p(layer_gpu["scales"]),
            ctypes.c_void_p(layer_gpu["xy_table"]),
            ctypes.c_void_p(layer_gpu["U"]),
            ctypes.c_void_p(self.d_tmp_k),
            ctypes.c_void_p(layer_gpu["row_ptr"]),
            ctypes.c_void_p(layer_gpu["kink_cols"]),
            ctypes.c_void_p(layer_gpu["kink_deltas"]),
            ctypes.c_void_p(d_out),
            layer_gpu["rows"],
            layer_gpu["cols"],
            layer_gpu["k_dim"],
            layer_gpu["n_pts"],
            ("local", 64 * 4),
            ("local", layer_gpu["n_pts"] * 8),
        )
        self.cl.launch(self.cl.k_fused, layer_gpu["rows"] * 64, 64)

    def forward_token(self, token_id: int, pos: int, return_logits: bool = True):
        # Copy single 4096-dim embedding vector (16 KB) to GPU
        x_emb = self.embed[token_id].astype(np.float32)
        self.cl.write_f32(self.d_x, x_emb)

        half_d = self.head_dim // 2
        rope_global = ((self.n_heads * half_d + 63) // 64) * 64
        hidden_global = ((self.hidden + 255) // 256) * 256
        ffn_global = ((self.ffn_size + 255) // 256) * 256

        for l in range(self.n_layers):
            lyr = self.layers[l]

            # 1. Attention RMSNorm
            self.cl.set_args(self.cl.k_rmsnorm, ctypes.c_void_p(self.d_x), ctypes.c_void_p(lyr["attn_norm"]), ctypes.c_void_p(self.d_xb), self.hidden, float(self.rms_eps), ("local", 256 * 4))
            self.cl.launch(self.cl.k_rmsnorm, 256, 256)

            # 2. Q, K, V projections (unpack-on-the-fly)
            self._matvec_gpu(self.d_xb, lyr["q"], self.d_q)
            self._matvec_gpu(self.d_xb, lyr["k"], self.d_k)
            self._matvec_gpu(self.d_xb, lyr["v"], self.d_v)

            # 3. RoPE + store K, V into GPU cache
            self.cl.set_args(
                self.cl.k_rope_kv,
                ctypes.c_void_p(self.d_q),
                ctypes.c_void_p(self.d_k),
                ctypes.c_void_p(self.d_v),
                ctypes.c_void_p(self.d_k_cache),
                ctypes.c_void_p(self.d_v_cache),
                l, pos, MAX_SEQ_LEN, self.n_heads, self.n_kv_heads, self.head_dim,
                float(self.rope_theta), self.rope_interleaved,
            )
            self.cl.launch(self.cl.k_rope_kv, rope_global, 64)

            # 4. GQA Attention over GPU KV-cache
            self.cl.set_args(
                self.cl.k_gqa,
                ctypes.c_void_p(self.d_q),
                ctypes.c_void_p(self.d_k_cache),
                ctypes.c_void_p(self.d_v_cache),
                ctypes.c_void_p(self.d_attn_out),
                l, pos, MAX_SEQ_LEN, self.n_heads, self.n_kv_heads, self.head_dim,
                ("local", 128 * 4),
                ("local", MAX_SEQ_LEN * 4),
            )
            self.cl.launch(self.cl.k_gqa, self.n_heads * 128, 128)

            # 5. Output projection + residual
            self._matvec_gpu(self.d_attn_out, lyr["o"], self.d_xb)
            self.cl.set_args(self.cl.k_add, ctypes.c_void_p(self.d_x), ctypes.c_void_p(self.d_xb), self.hidden)
            self.cl.launch(self.cl.k_add, hidden_global, 256)

            # 6. FFN RMSNorm
            self.cl.set_args(self.cl.k_rmsnorm, ctypes.c_void_p(self.d_x), ctypes.c_void_p(lyr["ffn_norm"]), ctypes.c_void_p(self.d_xb), self.hidden, float(self.rms_eps), ("local", 256 * 4))
            self.cl.launch(self.cl.k_rmsnorm, 256, 256)

            # 7. SwiGLU MLP (gate, up, silu*up, down + residual)
            self._matvec_gpu(self.d_xb, lyr["gate"], self.d_gate)
            self._matvec_gpu(self.d_xb, lyr["up"], self.d_up)
            self.cl.set_args(self.cl.k_swiglu, ctypes.c_void_p(self.d_gate), ctypes.c_void_p(self.d_up), ctypes.c_void_p(self.d_gate), self.ffn_size)
            self.cl.launch(self.cl.k_swiglu, ffn_global, 256)
            self._matvec_gpu(self.d_gate, lyr["down"], self.d_xb)
            self.cl.set_args(self.cl.k_add, ctypes.c_void_p(self.d_x), ctypes.c_void_p(self.d_xb), self.hidden)
            self.cl.launch(self.cl.k_add, hidden_global, 256)

        if not return_logits:
            self.cl.finish()
            return None

        # Final norm + LM head (packed hex2d on GPU)
        self.cl.set_args(self.cl.k_rmsnorm, ctypes.c_void_p(self.d_x), ctypes.c_void_p(self.final_norm), ctypes.c_void_p(self.d_xb), self.hidden, float(self.rms_eps), ("local", 256 * 4))
        self.cl.launch(self.cl.k_rmsnorm, 256, 256)
        self._matvec_gpu(self.d_xb, self.lm_head, self.d_logits)

        logits = np.empty(self.vocab_size, dtype=np.float32)
        self.cl.read_f32(self.d_logits, logits)
        return logits

    def sample_token(self, logits: np.ndarray, temperature: float = 0.4, top_p: float = 0.9) -> int:
        if temperature <= 1e-4:
            return int(np.argmax(logits))
        scaled = logits / temperature
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        probs /= np.sum(probs)
        sorted_idx = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[sorted_idx])
        cutoff = int(np.searchsorted(cum, top_p)) + 1
        top_idx = sorted_idx[:cutoff]
        p = probs[top_idx]
        p /= np.sum(p)
        return int(np.random.choice(top_idx, p=p))


SYSTEM_RULES = (
    "You are Mistral-7B-GSD-LQ-v2.0 running 100% natively inside the user's AMD Radeon RX 5600 XT GPU "
    "using packed hex2d on-the-fly unpacking. "
    "You have continuous memory of this entire conversation. "
    "If the user asks about facts or hardware details mentioned earlier, quote them accurately. "
    "If they ask about something not mentioned yet, say you do not know rather than guessing."
)


def format_mistral_prompt(history: list[tuple[str, str]], new_user_msg: str) -> str:
    parts = []
    if not history:
        parts.append(f"[INST] {SYSTEM_RULES}\n\nUser: {new_user_msg} [/INST]")
    else:
        for idx, (u, a) in enumerate(history):
            if idx == 0:
                parts.append(f"[INST] {SYSTEM_RULES}\n\nUser: {u} [/INST] {a}</s>")
            else:
                parts.append(f"[INST] {u} [/INST] {a}</s>")
        parts.append(f"[INST] {new_user_msg} [/INST]")
    return "".join(parts)


def ensure_log_header(vram_mb: float, total_vram_mb: float, avg_bpw: float):
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("# NATIVE GPU OPENCL CHAT LOG: MISTRAL-7B GSD-LQ v2.0\n\n")
            f.write(f"**Session Initialized**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Runtime Engine**: Custom Zero-Dependency OpenCL (`OpenCL.dll`, `gfx1010` RX 5600 XT)\n")
            f.write(f"**Average Weight Footprint**: {avg_bpw:.3f} bits/weight (Native Packed Hex2D)\n")
            f.write(f"**Live VRAM Footprint**: {vram_mb:.1f} MB / {total_vram_mb:.0f} MB\n")
            f.write("**Environment**: 100% Offline, 100% On-GPU End-to-End\n\n---\n\n")


def log_turn(turn_num: int, prompt: str, response: str, prefill_s: float, p_toks: int, gen_s: float, g_toks: int, vram_mb: float):
    tps = g_toks / max(gen_s, 0.001)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"### Turn {turn_num} at {datetime.datetime.now().strftime('%H:%M:%S')}\n\n")
        f.write(f"**User Prompt**:\n> {prompt}\n\n")
        f.write(f"**Mistral-7B Native GPU Response**:\n{response}\n\n")
        f.write(
            f"*Stats*: Prefill {p_toks} tok in {prefill_s:.2f}s | Generated {g_toks} tok in {gen_s:.2f}s "
            f"(**{tps:.2f} tok/s**) | *GPU VRAM*: {vram_mb:.1f} MB\n\n---\n\n"
        )


def run_smoke_verify(model: GPUMistralV2Model):
    prompt = "The capital of France is"
    ids = model.tokenizer.encode(prompt, add_bos=True)
    for pos, tid in enumerate(ids):
        logits = model.forward_token(tid, pos, return_logits=(pos == len(ids) - 1))
    out_ids = []
    pos = len(ids)
    for _ in range(8):
        nxt = model.sample_token(logits, temperature=0.0)
        if nxt == model.tokenizer.eos_id:
            break
        out_ids.append(nxt)
        logits = model.forward_token(nxt, pos, return_logits=True)
        pos += 1
    decoded = model.tokenizer.decode(out_ids)
    print(f"[SMOKE TEST] '{prompt}' -> '{decoded}'", flush=True)
    return decoded


def main():
    if "--smoke" in sys.argv:
        # Test exact RoPE convention on GPU
        m = GPUMistralV2Model(MODEL_PATH, rope_interleaved=1)
        run_smoke_verify(m)
        m.cl.release_all()
        return

    model = GPUMistralV2Model(MODEL_PATH, rope_interleaved=1)
    ensure_log_header(model.vram_used_mb, model.cl.total_vram_mb, model.avg_bpw)

    print("=" * 84)
    print("MISTRAL-7B GSD-LQ v2.0 NATIVE OPENCL GPU CHAT (100% END-TO-END ON RX 5600 XT)")
    print(f"Logging all turns to: {LOG_FILE}")
    print("Commands: type 'reset' to clear chat memory, 'quit' or 'exit' to stop.")
    print("=" * 84)

    history: list[tuple[str, str]] = []
    cached_tokens: list[int] = []
    turn_num = 1

    while True:
        try:
            print(f"\n[Turn {turn_num}] You > ", end="", flush=True)
            line = sys.stdin.readline()
            if not line:
                break
            user_msg = line.strip()
            if user_msg.lower() in ("quit", "exit", "q"):
                print("Closing native GPU engine. Chat log saved!")
                break
            if user_msg.lower() == "reset":
                history.clear()
                cached_tokens.clear()
                print("[Memory cleared]")
                continue
            if not user_msg:
                continue

            full_prompt = format_mistral_prompt(history, user_msg)
            prompt_ids = model.tokenizer.encode(full_prompt, add_bos=True)

            # Reuse KV cache prefix for tokens already evaluated in previous turns
            prefix_len = 0
            while prefix_len < len(cached_tokens) and prefix_len < len(prompt_ids) - 1 and cached_tokens[prefix_len] == prompt_ids[prefix_len]:
                prefix_len += 1

            if len(prompt_ids) >= MAX_SEQ_LEN - 64:
                print("[Context window full, resetting conversation history to fit 2048 tokens]")
                history.clear()
                cached_tokens.clear()
                prefix_len = 0
                full_prompt = format_mistral_prompt(history, user_msg)
                prompt_ids = model.tokenizer.encode(full_prompt, add_bos=True)

            # Prefill new prompt tokens into GPU KV cache
            t_pre = time.time()
            logits = None
            for pos in range(prefix_len, len(prompt_ids)):
                is_last = (pos == len(prompt_ids) - 1)
                logits = model.forward_token(prompt_ids[pos], pos, return_logits=is_last)
            prefill_s = time.time() - t_pre
            new_p_toks = len(prompt_ids) - prefix_len

            print("\nMistral-7B (GPU Native) > ", end="", flush=True)
            t_gen = time.time()
            gen_ids = []
            prev_text = ""
            pos = len(prompt_ids)
            max_new = min(512, MAX_SEQ_LEN - pos - 1)

            for _ in range(max_new):
                nxt = model.sample_token(logits, temperature=0.4, top_p=0.9)
                if nxt == model.tokenizer.eos_id:
                    break
                gen_ids.append(nxt)
                cur_text = model.tokenizer.decode(gen_ids)
                delta = cur_text[len(prev_text):]
                print(delta, end="", flush=True)
                prev_text = cur_text

                logits = model.forward_token(nxt, pos, return_logits=True)
                pos += 1

            gen_s = time.time() - t_gen
            response_text = model.tokenizer.decode(gen_ids).strip()
            tps = len(gen_ids) / max(gen_s, 0.001)
            print(
                f"\n\n[GPU Stats: Prefill {new_p_toks} tok ({prefill_s:.2f}s) | "
                f"Gen {len(gen_ids)} tok in {gen_s:.2f}s ({tps:.2f} tok/s) | "
                f"VRAM: {model.vram_used_mb:.1f} MB / {model.cl.total_vram_mb:.0f} MB]"
            )

            history.append((user_msg, response_text))
            cached_tokens = prompt_ids + gen_ids
            log_turn(turn_num, user_msg, response_text, prefill_s, new_p_toks, gen_s, len(gen_ids), model.vram_used_mb)
            turn_num += 1

        except KeyboardInterrupt:
            print("\nInterrupted. Log saved.")
            break

    model.cl.release_all()


if __name__ == "__main__":
    main()
