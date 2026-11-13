# GSD-LQ v2.0: Zero-Calibration Low-Bit Inference Engine

**Patent Pending**: U.S. Provisional Application No. **64/195,346**  
**Assignee**: Phantom Lattice Labs  
**Commercial Licensing & Inquiries**: `Phantom_Lattice_Labs@proton.me`

**GSD-LQ v2.0** is a model-agnostic, zero-calibration compression architecture and native OpenCL GPU execution engine.
* **Universal Compatibility**: Evaluate open-weight transformer models in minutes on local hardware with zero data exposure.
* **Hardware-Agnostic**: Native fused GPU execution runs on **any OpenCL 1.2+ GPU (AMD, NVIDIA, or Intel)** with zero external GPU SDK dependencies *(verified on consumer AMD Radeon RX 5600 XT, RTX series, and Intel Arc)*.

---

## Capabilities & Modes

* **Turbo Mode (~4.26 bpw)**: High-speed engine delivering 12–13+ tok/s interactive generation inside 3.79 GB VRAM, outperforming standard INT4 accuracy with native OpenCL execution (`gsd_lq_opencl_kernel.py` & `run_gpu_native_chat.py`).
* **Precision Mode (~3.76 bpw)**: Ultra-dense sub-4-bit intelligence engine matching uncompressed 16-bit reasoning scores (10/12 STEM) inside 3.37 GB VRAM (saving over 420 MB VRAM vs Turbo Mode and ~1,000 MB vs INT4).

---

## Quick Start (Native GPU Chat)

```bash
# Launch interactive GPU chat on local hardware (auto-detects GPU and compiles kernel)
python run_gpu_native_chat.py
```

---

## Verified Benchmarks

### 1. WikiText-2 Perplexity (Full Brain, All Attention + MLP Layers)
Evaluated across 1,536 tokens with zero calibration datasets:

| Method | Avg Bits / Weight | WikiText-2 PPL (Lower = Better) | Calibration Required |
| :--- | :---: | :---: | :--- |
| **BF16 Uncompressed (Source)** | `16.000 bpw` | `14.330` | None |
| **GSD-LQ Turbo Mode** | `4.264 bpw` | **`16.416`** | **Zero** |
| **Standard INT4 (Block64)** | `4.250 bpw` | `16.654` | None |
| **GSD-LQ Precision Mode** | **`3.769 bpw`** | **`11.377`** *(1.5B) / `25.920` (7B)* | **Zero** |
| **Standard INT3 (Block64)** | `3.250 bpw` | `106.753` *(Collapsed)* | Block Grid Search |

---

### 2. Full 32-Layer Native OpenCL Execution (`Mistral-7B-Instruct-v0.3` on AMD RX 5600 XT 6 GB)
* **Zero Wrapper Dependencies**: Executes directly via system `OpenCL.dll` (`ctypes`) without PyTorch, Ollama, or ROCm.
* **Binary & Load Time**: Packed `3.37 GB – 3.79 GB` binary loads into VRAM in **under 2.5 seconds**.
* **Total Live VRAM**: Fits comfortably inside 6 GB consumer cards, leaving ample headroom for deep multi-turn conversation memory.
* **Reasoning Accuracy**: Precision Mode scored **10/12 (83.3%)** on deterministic STEM prompts with **100% retrieval** on deep Needle-in-a-Haystack contexts.

---

## Licensing & Ethical Terms
See [`LICENSE.md`](./LICENSE.md).
* **Individuals, Students & Academic Research**: Free for non-commercial evaluation.
* **Commercial Deployment**: Commercial deployment, cloud serving, enterprise hardware integration, and proprietary codebook access require an executed commercial license (`Phantom_Lattice_Labs@proton.me`).
* **Strict Ethical Clause**: Military, defense, autonomous weapons, and lethal robotics use is strictly prohibited.
