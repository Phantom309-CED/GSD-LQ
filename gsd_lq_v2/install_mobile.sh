#!/usr/bin/env bash
# ==============================================================================
# GSD-LQ v2.0: Universal Mobile & Tablet Installer
# Phantom Lattice Labs
# ==============================================================================

set -e

echo "===================================================================="
echo "  GSD-LQ v2.0: Mobile Hardware Scanner & Runner"
echo "  Phantom Lattice Labs (U.S. Provisional Patent No. 64/195,346)"
echo "===================================================================="

# 1. Setup Termux or Linux environment
if command -v pkg &> /dev/null; then
    echo "[1/4] Setting up Termux packages & mobile GPU drivers..."
    # Ensure reliable DNS resolution in Termux
    if [ -n "$PREFIX" ]; then
        mkdir -p "$PREFIX/etc"
        echo "nameserver 1.1.1.1" > "$PREFIX/etc/resolv.conf" 2>/dev/null || true
        echo "nameserver 8.8.8.8" >> "$PREFIX/etc/resolv.conf" 2>/dev/null || true
    fi
    pkg update -y
    pkg install -y ca-certificates python python-numpy clang git curl ocl-icd opencl-vendor-driver
    pip install --upgrade pip --quiet 2>/dev/null || true
    pip install huggingface_hub gguf --quiet 2>/dev/null || true
elif command -v apt-get &> /dev/null; then
    echo "[1/4] Setting up Linux packages..."
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates python3 python3-pip python3-numpy git curl
    pip3 install huggingface_hub gguf --quiet 2>/dev/null || true
fi

# 2. Check Android vendor drivers (Mali / Adreno)
echo "[2/4] Detecting GPU driver..."
DRIVER_PATHS=(
    "/vendor/lib64/libOpenCL.so"
    "/vendor/lib64/egl/libGLES_mali.so"
    "/system/vendor/lib64/libOpenCL.so"
    "/vendor/lib/libOpenCL.so"
    "/vendor/lib/egl/libGLES_mali.so"
)

for p in "${DRIVER_PATHS[@]}"; do
    if [ -f "$p" ]; then
        echo "      Found mobile GPU driver: $p"
        if [ -n "$PREFIX" ]; then
            mkdir -p "$PREFIX/lib"
            ln -sf "$p" "$PREFIX/lib/libOpenCL.so" 2>/dev/null || true
            ln -sf "$p" "$PREFIX/lib/libOpenCL.so.1" 2>/dev/null || true
        fi
        break
    fi
done

# 3. Pull or update GSD-LQ repository
echo "[3/4] Downloading latest GSD-LQ engine..."
TARGET_DIR="$HOME/GSD-LQ"
if [ -d "$TARGET_DIR/.git" ]; then
    cd "$TARGET_DIR"
    git pull --quiet
else
    cd "$HOME"
    git clone --depth 1 https://github.com/Phantom309-CED/GSD-LQ.git
fi

# 4. Launch Hardware Detection and Engine
echo "[4/4] Starting Hardware Scanner..."
cd "$TARGET_DIR/gsd_lq_v2"
python run_gpu_native_chat.py "$@"
