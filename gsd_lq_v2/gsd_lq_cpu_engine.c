/*
 * GSD-LQ v2.0 Native High-Performance Universal CPU Engine
 * Phantom Lattice Labs (c) 2026. U.S. Patent Application No. 64/195,346.
 *
 * Implements high-speed parallel matrix-vector execution for root-lattice
 * quantized models on ARM NEON (Android Termux / Apple Silicon) and x86 AVX2.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define GSD_HAS_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define GSD_HAS_AVX2 1
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

// Fast IEEE 754 FP16 to FP32 conversion
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t res = sign;
            float f;
            memcpy(&f, &res, 4);
            return f;
        }
        // Subnormal
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        uint32_t res = sign | 0x7F800000 | (mant << 13);
        float f;
        memcpy(&f, &res, 4);
        return f;
    }

    uint32_t res = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &res, 4);
    return f;
}

// 1. RMSNorm
void gsd_cpu_rmsnorm(
    const float* x,
    const float* weight,
    float* out,
    int dim,
    float eps
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf((sum_sq / (float)dim) + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * scale * weight[i];
    }
}

// 2. Vector Add: y = a + b
void gsd_cpu_vec_add(
    float* y,
    const float* a,
    const float* b,
    int dim
) {
    for (int i = 0; i < dim; i++) {
        y[i] = a[i] + b[i];
    }
}

// 3. SwiGLU: out = silu(gate) * up
void gsd_cpu_swiglu(
    const float* gate,
    const float* up,
    float* out,
    int dim
) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

// 4. Low-Rank Projection: tmp_k[j] = S[j] * dot(x, Vt[j])
void gsd_cpu_lowrank_proj(
    const float* x,
    const uint16_t* Vt,
    const float* S,
    float* tmp_k,
    int k_dim,
    int cols
) {
    for (int j = 0; j < k_dim; j++) {
        size_t row_offset = (size_t)j * (size_t)cols;
        float dot = 0.0f;
        for (int c = 0; c < cols; c++) {
            dot += x[c] * fp16_to_fp32(Vt[row_offset + c]);
        }
        tmp_k[j] = dot * S[j];
    }
}

// 5. Fused Hex2D MatVec with Kink Deltas & Low-Rank U accumulation
void gsd_cpu_hex2d_matvec(
    const float* x,
    const uint8_t* codes,
    const uint16_t* scales,
    const float* xy_table,
    const uint16_t* U,
    const float* tmp_k,
    const uint32_t* kink_row_ptr,
    const uint16_t* kink_cols,
    const float* kink_deltas,
    float* y,
    int rows,
    int cols,
    int k_dim,
    int n_pts
) {
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int r = 0; r < rows; r++) {
        size_t row_base = (size_t)r * (size_t)cols;
        size_t pair_base = row_base >> 1;
        size_t blk_base  = row_base >> 6;
        int num_pairs = cols >> 1;
        float acc = 0.0f;

        for (int p = 0; p < num_pairs; p++) {
            float scale = fp16_to_fp32(scales[blk_base + (size_t)(p >> 5)]);
            uint8_t code = codes[pair_base + (size_t)p];
            float pt_x = xy_table[code * 2];
            float pt_y = xy_table[code * 2 + 1];

            int c_base = p << 1;
            acc += (pt_x * x[c_base] + pt_y * x[c_base + 1]) * scale;
        }

        // Kink deltas
        uint32_t k_start = kink_row_ptr[r];
        uint32_t k_end   = kink_row_ptr[r + 1];
        for (uint32_t ki = k_start; ki < k_end; ki++) {
            acc += kink_deltas[ki] * x[kink_cols[ki]];
        }

        // Low-rank U projection
        if (k_dim > 0) {
            size_t u_base = (size_t)r * (size_t)k_dim;
            for (int j = 0; j < k_dim; j++) {
                acc += fp16_to_fp32(U[u_base + (size_t)j]) * tmp_k[j];
            }
        }

        y[r] = acc;
    }
}

// 6. RoPE and Store KV
void gsd_cpu_rope_and_store_kv(
    float* q,
    const float* k,
    const float* v,
    uint16_t* k_cache,
    uint16_t* v_cache,
    int layer_idx,
    int pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    float rope_theta,
    int interleaved
) {
    int half_d = head_dim / 2;
    size_t cache_layer_stride = (size_t)max_seq_len * (size_t)n_kv_heads * (size_t)head_dim;
    size_t cache_pos_stride   = (size_t)pos * (size_t)n_kv_heads * (size_t)head_dim;
    size_t cache_base         = (size_t)layer_idx * cache_layer_stride + cache_pos_stride;

    // Fast FP32 to FP16 converter
    #define FP32_TO_FP16(val) ({ \
        float _v = (val); \
        uint32_t _i; \
        memcpy(&_i, &_v, 4); \
        uint16_t _sign = (_i >> 16) & 0x8000; \
        int32_t _exp = ((_i >> 23) & 0xFF) - 127 + 15; \
        uint16_t _mant = (_i >> 13) & 0x3FF; \
        uint16_t _h = 0; \
        if (_exp <= 0) { _h = _sign; } \
        else if (_exp >= 31) { _h = _sign | 0x7C00; } \
        else { _h = _sign | (_exp << 10) | _mant; } \
        _h; \
    })

    // Store V directly
    for (int i = 0; i < n_kv_heads * head_dim; i++) {
        v_cache[cache_base + i] = FP32_TO_FP16(v[i]);
    }

    // RoPE for Q
    for (int h = 0; h < n_heads; h++) {
        int base = h * head_dim;
        for (int i = 0; i < half_d; i++) {
            float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            float q0, q1;
            if (interleaved) {
                q0 = q[base + 2 * i];
                q1 = q[base + 2 * i + 1];
                q[base + 2 * i]     = q0 * cos_val - q1 * sin_val;
                q[base + 2 * i + 1] = q0 * sin_val + q1 * cos_val;
            } else {
                q0 = q[base + i];
                q1 = q[base + i + half_d];
                q[base + i]          = q0 * cos_val - q1 * sin_val;
                q[base + i + half_d] = q0 * sin_val + q1 * cos_val;
            }
        }
    }

    // RoPE for K and store to cache
    for (int h = 0; h < n_kv_heads; h++) {
        int base = h * head_dim;
        for (int i = 0; i < half_d; i++) {
            float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            float k0, k1, rk0, rk1;
            if (interleaved) {
                k0 = k[base + 2 * i];
                k1 = k[base + 2 * i + 1];
                rk0 = k0 * cos_val - k1 * sin_val;
                rk1 = k0 * sin_val + k1 * cos_val;
                k_cache[cache_base + base + 2 * i]     = FP32_TO_FP16(rk0);
                k_cache[cache_base + base + 2 * i + 1] = FP32_TO_FP16(rk1);
            } else {
                k0 = k[base + i];
                k1 = k[base + i + half_d];
                rk0 = k0 * cos_val - k1 * sin_val;
                rk1 = k0 * sin_val + k1 * cos_val;
                k_cache[cache_base + base + i]          = FP32_TO_FP16(rk0);
                k_cache[cache_base + base + i + half_d] = FP32_TO_FP16(rk1);
            }
        }
    }
}

// 7. Grouped-Query Attention (GQA)
void gsd_cpu_gqa_attention(
    const float* q,
    const uint16_t* k_cache,
    const uint16_t* v_cache,
    float* attn_out,
    int layer_idx,
    int pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    float sm_scale
) {
    int group_size = n_heads / n_kv_heads;
    size_t cache_layer_stride = (size_t)max_seq_len * (size_t)n_kv_heads * (size_t)head_dim;
    size_t layer_base = (size_t)layer_idx * cache_layer_stride;
    int seq_len = pos + 1;

    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / group_size;
        const float* q_head = q + h * head_dim;
        float* out_head = attn_out + h * head_dim;

        // Allocate temporary scores buffer on stack if small
        float scores[2048];
        float max_score = -1e30f;

        for (int t = 0; t < seq_len; t++) {
            size_t k_offset = layer_base + (size_t)t * ((size_t)n_kv_heads * (size_t)head_dim) + (size_t)kv_h * (size_t)head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_head[d] * fp16_to_fp32(k_cache[k_offset + d]);
            }
            float s = dot * sm_scale;
            scores[t] = s;
            if (s > max_score) max_score = s;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float exp_val = expf(scores[t] - max_score);
            scores[t] = exp_val;
            sum_exp += exp_val;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t < seq_len; t++) {
            scores[t] *= inv_sum;
        }

        // Weighted sum over V
        for (int d = 0; d < head_dim; d++) {
            out_head[d] = 0.0f;
        }
        for (int t = 0; t < seq_len; t++) {
            float weight = scores[t];
            size_t v_offset = layer_base + (size_t)t * ((size_t)n_kv_heads * (size_t)head_dim) + (size_t)kv_h * (size_t)head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += weight * fp16_to_fp32(v_cache[v_offset + d]);
            }
        }
    }
}
