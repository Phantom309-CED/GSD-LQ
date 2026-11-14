"""
GSD-LQ v2.0 High-Performance CPU Engine & Runtime
Phantom Lattice Labs (c) 2026. U.S. Patent Application No. 64/195,346.

Provides universal fallback inference on CPU / ARM NEON when GPU OpenCL is unavailable.
Loads identical .gsd2 binary files directly into unified system memory.
"""

import ctypes
import json
import os
import struct
import subprocess
import sys
import time
import numpy as np

V2_DIR = os.path.dirname(os.path.abspath(__file__))
C_SRC = os.path.join(V2_DIR, "gsd_lq_cpu_engine.c")
LIB_NAME = "libgsd_cpu.dll" if sys.platform == "win32" else "libgsd_cpu.so"
LIB_PATH = os.path.join(V2_DIR, LIB_NAME)
MAX_SEQ_LEN = 2048


def build_c_library() -> str:
    """Compiles gsd_lq_cpu_engine.c into a shared library using available C compiler."""
    if os.path.exists(LIB_PATH) and os.path.exists(C_SRC):
        if os.path.getmtime(LIB_PATH) >= os.path.getmtime(C_SRC):
            return LIB_PATH

    compilers = ["clang", "gcc"]
    compiler = None
    for c in compilers:
        try:
            subprocess.run([c, "--version"], capture_output=True, check=True)
            compiler = c
            break
        except Exception:
            continue

    if not compiler:
        return ""

    print(f"[*] Compiling native GSD-LQ CPU engine with {compiler}...")
    compile_cmd = [
        compiler,
        "-O3",
        "-shared",
        "-fPIC",
        "-fopenmp",
        C_SRC,
        "-o",
        LIB_PATH,
        "-lm",
    ]
    if sys.platform != "win32":
        import platform
        if "arm" in platform.machine().lower() or "aarch64" in platform.machine().lower():
            compile_cmd.extend(["-march=armv8-a+simd"])

    try:
        subprocess.run(compile_cmd, check=True)
        print(f"[+] Successfully compiled: {LIB_PATH}")
        return LIB_PATH
    except Exception as e:
        try:
            compile_cmd.remove("-fopenmp")
            subprocess.run(compile_cmd, check=True)
            print(f"[+] Successfully compiled (single-threaded): {LIB_PATH}")
            return LIB_PATH
        except Exception as e2:
            print(f"[!] Compilation failed: {e2}")
            return ""


class CPUEngine:
    def __init__(self):
        self.lib_path = build_c_library()
        self.lib = None
        if self.lib_path and os.path.exists(self.lib_path):
            try:
                self.lib = ctypes.CDLL(self.lib_path)
                self._bind()
                print("[+] Loaded compiled native C / ARM NEON acceleration engine.")
            except Exception as e:
                print(f"[!] Warning: Could not load C engine ({e}), using vectorized NumPy fallback.")
                self.lib = None
        else:
            print("[*] Running in pure vectorized NumPy CPU mode.")

    def _bind(self):
        c = self.lib
        vp, i32, f32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_float

        c.gsd_cpu_rmsnorm.argtypes = [vp, vp, vp, i32, f32]
        c.gsd_cpu_vec_add.argtypes = [vp, vp, vp, i32]
        c.gsd_cpu_swiglu.argtypes = [vp, vp, vp, i32]
        c.gsd_cpu_lowrank_proj.argtypes = [vp, vp, vp, vp, i32, i32]
        c.gsd_cpu_hex2d_matvec.argtypes = [
            vp, vp, vp, vp, vp, vp, vp, vp, vp, vp,
            i32, i32, i32, i32
        ]
        c.gsd_cpu_rope_and_store_kv.argtypes = [
            vp, vp, vp, vp, vp, i32, i32, i32, i32, i32, i32, f32, i32
        ]
        c.gsd_cpu_gqa_attention.argtypes = [
            vp, vp, vp, vp, i32, i32, i32, i32, i32, i32, f32
        ]


def _get_turbo_codebook(radius=6):
    sqrt3_2 = float(np.sqrt(3.0) / 2.0)
    xy_pts = []
    for r0 in range(-radius, radius + 1):
        for r1 in range(-radius, radius + 1):
            if abs(r0 + r1) <= radius:
                xy_pts.append([r0 + 0.5 * r1, r1 * sqrt3_2])
    return np.array(xy_pts, dtype=np.float32), len(xy_pts)


class CPUMistralV2Model:
    def __init__(self, gsd2_path: str, rope_interleaved: int = 1):
        t0 = time.time()
        self.rope_interleaved = rope_interleaved
        self.cpu = CPUEngine()

        # Load GSD2 binary format
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
            self.n_heads = self.cfg["n_heads"]
            self.n_kv_heads = self.cfg["n_kv_heads"]
            self.head_dim = self.cfg["head_dim"]
            self.ffn_size = self.cfg["ffn_size"]
            self.rms_eps = self.cfg["rms_eps"]
            self.rope_theta = self.cfg["rope_theta"]
            self.vocab_size = self.cfg["vocab_size"]
            self.avg_bpw = header.get("avg_bpw", 3.79)

            from run_gpu_native_chat import MistralSentencePieceTokenizer
            self.tokenizer = MistralSentencePieceTokenizer(
                self.cfg["tokens"], self.cfg["scores"], self.cfg["token_types"],
                self.cfg["bos_id"], self.cfg["eos_id"]
            )

            tensors = header["tensors"]

            def read_blob(meta: dict) -> bytes:
                f.seek(data_base + meta["offset"])
                return f.read(meta["nbytes"])

            # Token embeddings
            emb_meta = tensors["token_embd.weight"]["data"]
            self.embed = np.frombuffer(read_blob(emb_meta), dtype=np.float16).astype(np.float32).reshape(emb_meta["shape"])

            # Codebook tables
            self.xy_tables = {}
            for r in (6, 8):
                table_pts, num_pts = _get_turbo_codebook(r)
                self.xy_tables[r] = (np.ascontiguousarray(table_pts, dtype=np.float32), int(num_pts))

            self.dummy_u16 = np.zeros(2, dtype=np.uint16)
            self.dummy_f32 = np.zeros(1, dtype=np.float32)

            def load_hex_cpu(tname: str) -> dict:
                m = tensors[tname]
                r = m["radius"]
                xy_table, n_pts = self.xy_tables[r]
                k_dim = m["k_dim"]
                if k_dim > 0:
                    U = np.frombuffer(read_blob(m["U"]), dtype=np.uint16)
                    S = np.frombuffer(read_blob(m["S"]), dtype=np.float32)
                    Vt = np.frombuffer(read_blob(m["Vt"]), dtype=np.uint16)
                else:
                    U = Vt = self.dummy_u16
                    S = self.dummy_f32
                return {
                    "rows": m["rows"],
                    "cols": m["cols"],
                    "k_dim": k_dim,
                    "n_pts": n_pts,
                    "xy_table": xy_table,
                    "codes": np.frombuffer(read_blob(m["codes"]), dtype=np.uint8),
                    "scales": np.frombuffer(read_blob(m["scales"]), dtype=np.uint16),
                    "U": U,
                    "S": S,
                    "Vt": Vt,
                    "row_ptr": np.frombuffer(read_blob(m["row_ptr"]), dtype=np.uint32),
                    "kink_cols": np.frombuffer(read_blob(m["kink_cols"]), dtype=np.uint16),
                    "kink_deltas": np.frombuffer(read_blob(m["kink_deltas"]), dtype=np.float32),
                }

            self.layers = []
            for l in range(self.n_layers):
                p = f"blk.{l}."
                self.layers.append({
                    "attn_norm": np.frombuffer(read_blob(tensors[p + "attn_norm.weight"]["data"]), dtype=np.float32),
                    "ffn_norm": np.frombuffer(read_blob(tensors[p + "ffn_norm.weight"]["data"]), dtype=np.float32),
                    "q": load_hex_cpu(p + "attn_q.weight"),
                    "k": load_hex_cpu(p + "attn_k.weight"),
                    "v": load_hex_cpu(p + "attn_v.weight"),
                    "o": load_hex_cpu(p + "attn_output.weight"),
                    "gate": load_hex_cpu(p + "ffn_gate.weight"),
                    "up": load_hex_cpu(p + "ffn_up.weight"),
                    "down": load_hex_cpu(p + "ffn_down.weight"),
                })

            self.final_norm = np.frombuffer(read_blob(tensors["output_norm.weight"]["data"]), dtype=np.float32)
            self.lm_head = load_hex_cpu("output.weight")

        # KV Cache & Scratchpad Buffers
        self.k_cache = np.zeros((self.n_layers, MAX_SEQ_LEN, self.n_kv_heads, self.head_dim), dtype=np.uint16)
        self.v_cache = np.zeros((self.n_layers, MAX_SEQ_LEN, self.n_kv_heads, self.head_dim), dtype=np.uint16)

        self.buf_x = np.zeros(self.hidden, dtype=np.float32)
        self.buf_xb = np.zeros(self.hidden, dtype=np.float32)
        self.buf_q = np.zeros(self.hidden, dtype=np.float32)
        self.buf_k = np.zeros(self.n_kv_heads * self.head_dim, dtype=np.float32)
        self.buf_v = np.zeros(self.n_kv_heads * self.head_dim, dtype=np.float32)
        self.buf_attn_out = np.zeros(self.hidden, dtype=np.float32)
        self.buf_gate = np.zeros(self.ffn_size, dtype=np.float32)
        self.buf_up = np.zeros(self.ffn_size, dtype=np.float32)
        self.buf_tmp_k = np.zeros(16, dtype=np.float32)
        self.buf_logits = np.zeros(self.vocab_size, dtype=np.float32)

        file_size_mb = os.path.getsize(gsd2_path) / (1024 * 1024)
        self.vram_used_mb = file_size_mb
        class _MockCL:
            total_vram_mb = file_size_mb
            def release_all(self):
                pass
        self.cl = _MockCL()
        print(
            f"[+] GSD-LQ Model Loaded 100% in RAM in {time.time() - t0:.2f}s!\n"
            f"    Unified Memory Footprint: {file_size_mb:.1f} MB ({file_size_mb / 1024:.2f} GB)\n"
            f"    Effective Quantization:  {self.avg_bpw:.2f} bpw",
            flush=True,
        )

    def _ptr(self, arr):
        return arr.ctypes.data_as(ctypes.c_void_p)

    def _matvec_cpu(self, x: np.ndarray, lyr: dict, out: np.ndarray):
        if self.cpu.lib is not None:
            if lyr["k_dim"] > 0:
                self.cpu.lib.gsd_cpu_lowrank_proj(
                    self._ptr(x),
                    self._ptr(lyr["Vt"]),
                    self._ptr(lyr["S"]),
                    self._ptr(self.buf_tmp_k),
                    lyr["k_dim"],
                    lyr["cols"]
                )
            self.cpu.lib.gsd_cpu_hex2d_matvec(
                self._ptr(x),
                self._ptr(lyr["codes"]),
                self._ptr(lyr["scales"]),
                self._ptr(lyr["xy_table"]),
                self._ptr(lyr["U"]),
                self._ptr(self.buf_tmp_k),
                self._ptr(lyr["row_ptr"]),
                self._ptr(lyr["kink_cols"]),
                self._ptr(lyr["kink_deltas"]),
                self._ptr(out),
                lyr["rows"],
                lyr["cols"],
                lyr["k_dim"],
                lyr["n_pts"]
            )
        else:
            # Vectorized NumPy Fallback
            rows, cols = lyr["rows"], lyr["cols"]
            scales_f32 = lyr["scales"].view(np.float16).astype(np.float32)
            pts = lyr["xy_table"][lyr["codes"]].reshape(rows, cols)
            scales_expanded = np.repeat(scales_f32.reshape(rows, cols // 64), 64, axis=1)
            W = pts * scales_expanded
            res = W @ x
            # Kinks
            row_ptr = lyr["row_ptr"]
            k_cols = lyr["kink_cols"]
            k_deltas = lyr["kink_deltas"]
            for r in range(rows):
                ks, ke = row_ptr[r], row_ptr[r + 1]
                if ke > ks:
                    res[r] += np.dot(k_deltas[ks:ke], x[k_cols[ks:ke]])
            # Lowrank
            if lyr["k_dim"] > 0:
                U = lyr["U"].view(np.float16).astype(np.float32).reshape(rows, lyr["k_dim"])
                S = lyr["S"]
                Vt = lyr["Vt"].view(np.float16).astype(np.float32).reshape(lyr["k_dim"], cols)
                res += U @ (S * (Vt @ x))
            np.copyto(out, res)

    def forward_token(self, token_id: int, pos: int, return_logits: bool = True):
        self.buf_x[:] = self.embed[token_id]

        for l in range(self.n_layers):
            lyr = self.layers[l]

            # 1. Attn Norm
            if self.cpu.lib is not None:
                self.cpu.lib.gsd_cpu_rmsnorm(
                    self._ptr(self.buf_x), self._ptr(lyr["attn_norm"]), self._ptr(self.buf_xb),
                    self.hidden, float(self.rms_eps)
                )
            else:
                scale = 1.0 / np.sqrt(np.mean(self.buf_x ** 2) + self.rms_eps)
                self.buf_xb[:] = self.buf_x * scale * lyr["attn_norm"]

            # 2. Q, K, V
            self._matvec_cpu(self.buf_xb, lyr["q"], self.buf_q)
            self._matvec_cpu(self.buf_xb, lyr["k"], self.buf_k)
            self._matvec_cpu(self.buf_xb, lyr["v"], self.buf_v)

            # 3. RoPE & Cache
            if self.cpu.lib is not None:
                self.cpu.lib.gsd_cpu_rope_and_store_kv(
                    self._ptr(self.buf_q), self._ptr(self.buf_k), self._ptr(self.buf_v),
                    self._ptr(self.k_cache), self._ptr(self.v_cache),
                    l, pos, self.n_heads, self.n_kv_heads, self.head_dim, MAX_SEQ_LEN,
                    float(self.rope_theta), self.rope_interleaved
                )
            else:
                half_d = self.head_dim // 2
                freqs = 1.0 / (self.rope_theta ** (np.arange(0, half_d * 2, 2, dtype=np.float32) / self.head_dim))
                angles = pos * freqs
                cos_v, sin_v = np.cos(angles), np.sin(angles)
                q_reshaped = self.buf_q.reshape(self.n_heads, self.head_dim)
                k_reshaped = self.buf_k.reshape(self.n_kv_heads, self.head_dim)
                for h in range(self.n_heads):
                    q0 = q_reshaped[h, 0::2]
                    q1 = q_reshaped[h, 1::2]
                    q_reshaped[h, 0::2] = q0 * cos_v - q1 * sin_v
                    q_reshaped[h, 1::2] = q0 * sin_v + q1 * cos_v
                for h in range(self.n_kv_heads):
                    k0 = k_reshaped[h, 0::2]
                    k1 = k_reshaped[h, 1::2]
                    rk0 = k0 * cos_v - k1 * sin_v
                    rk1 = k0 * sin_v + k1 * cos_v
                    k_reshaped[h, 0::2] = rk0
                    k_reshaped[h, 1::2] = rk1
                self.k_cache[l, pos] = k_reshaped.astype(np.float16).view(np.uint16)
                self.v_cache[l, pos] = self.buf_v.reshape(self.n_kv_heads, self.head_dim).astype(np.float16).view(np.uint16)

            # 4. GQA Attention
            if self.cpu.lib is not None:
                sm_scale = 1.0 / float(np.sqrt(self.head_dim))
                self.cpu.lib.gsd_cpu_gqa_attention(
                    self._ptr(self.buf_q), self._ptr(self.k_cache), self._ptr(self.v_cache),
                    self._ptr(self.buf_attn_out), l, pos, self.n_heads, self.n_kv_heads,
                    self.head_dim, MAX_SEQ_LEN, sm_scale
                )
            else:
                sm_scale = 1.0 / np.sqrt(self.head_dim)
                group_size = self.n_heads // self.n_kv_heads
                q_heads = self.buf_q.reshape(self.n_heads, self.head_dim)
                out_heads = self.buf_attn_out.reshape(self.n_heads, self.head_dim)
                k_active = self.k_cache[l, :pos + 1].view(np.float16).astype(np.float32)
                v_active = self.v_cache[l, :pos + 1].view(np.float16).astype(np.float32)
                for h in range(self.n_heads):
                    kv_h = h // group_size
                    scores = (k_active[:, kv_h] @ q_heads[h]) * sm_scale
                    max_s = np.max(scores)
                    exp_s = np.exp(scores - max_s)
                    probs = exp_s / np.sum(exp_s)
                    out_heads[h] = probs @ v_active[:, kv_h]

            # 5. Output projection + residual
            self._matvec_cpu(self.buf_attn_out, lyr["o"], self.buf_xb)
            self.buf_x += self.buf_xb

            # 6. FFN Norm
            if self.cpu.lib is not None:
                self.cpu.lib.gsd_cpu_rmsnorm(
                    self._ptr(self.buf_x), self._ptr(lyr["ffn_norm"]), self._ptr(self.buf_xb),
                    self.hidden, float(self.rms_eps)
                )
            else:
                scale = 1.0 / np.sqrt(np.mean(self.buf_x ** 2) + self.rms_eps)
                self.buf_xb[:] = self.buf_x * scale * lyr["ffn_norm"]

            # 7. SwiGLU MLP
            self._matvec_cpu(self.buf_xb, lyr["gate"], self.buf_gate)
            self._matvec_cpu(self.buf_xb, lyr["up"], self.buf_up)

            if self.cpu.lib is not None:
                self.cpu.lib.gsd_cpu_swiglu(
                    self._ptr(self.buf_gate), self._ptr(self.buf_up), self._ptr(self.buf_gate),
                    self.ffn_size
                )
            else:
                silu = self.buf_gate / (1.0 + np.exp(-self.buf_gate))
                self.buf_gate[:] = silu * self.buf_up

            self._matvec_cpu(self.buf_gate, lyr["down"], self.buf_xb)
            self.buf_x += self.buf_xb

        if not return_logits:
            return None

        # Final norm + LM head
        if self.cpu.lib is not None:
            self.cpu.lib.gsd_cpu_rmsnorm(
                self._ptr(self.buf_x), self._ptr(self.final_norm), self._ptr(self.buf_xb),
                self.hidden, float(self.rms_eps)
            )
        else:
            scale = 1.0 / np.sqrt(np.mean(self.buf_x ** 2) + self.rms_eps)
            self.buf_xb[:] = self.buf_x * scale * self.final_norm

        self._matvec_cpu(self.buf_xb, self.lm_head, self.buf_logits)
        return self.buf_logits.copy()

    def sample_token(self, logits, temperature: float = 0.0, top_p: float = 0.9):
        logits = np.nan_to_num(logits, nan=-1e4, posinf=100.0, neginf=-1e4)
        if temperature <= 0.0:
            return int(np.argmax(logits))
        logits = logits / max(temperature, 1e-4)
        logits -= np.max(logits)
        probs = np.exp(logits)
        s = np.sum(probs)
        if s <= 0 or np.isnan(s):
            return int(np.argmax(logits))
        probs /= s
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_index = int(np.argmax(cumulative_probs > top_p)) + 1
        top_idx = sorted_indices[:cutoff_index]
        p = sorted_probs[:cutoff_index]
        p_sum = np.sum(p)
        if p_sum <= 0 or np.isnan(p_sum):
            return int(top_idx[0])
        p /= p_sum
        return int(np.random.choice(top_idx, p=p))
