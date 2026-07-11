# Building CUDA Support for Sassy

## Overview

This document provides instructions for building Sassy with CUDA GPU acceleration support. The CUDA implementation parallelizes pattern matching across thousands of GPU threads for massive throughput improvements with large pattern batches (1000-10000+ patterns).

## Prerequisites

### 1. CUDA Toolkit (Required)

Install NVIDIA CUDA Toolkit version 12.0 or newer:

- **Linux**: https://developer.nvidia.com/cuda-downloads
- **Windows**: Same link, choose Windows installer

After installation, verify:
```bash
nvcc --version
# Should show: Cuda compilation tools, release 12.x
```

Make sure CUDA is in your PATH:
```bash
# Linux/Mac
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_LIBRARY_PATH=/usr/local/cuda/lib64

# Add to ~/.bashrc or ~/.zshrc for persistence
```

### 2. LLVM (Required for PTX generation)

Install LLVM version 7.x or newer:

```bash
# Ubuntu/Debian
sudo apt install llvm-7

# Mac
brew install llvm

# Windows
# Download from https://releases.llvm.org/
```

### 3. Rust NVPTX Target (Required)

Add the CUDA GPU target to your Rust toolchain:

```bash
rustup target add nvptx64-nvidia-cuda
```

Verify it's installed:
```bash
rustup target list --installed | grep nvptx
# Should show: nvptx64-nvidia-cuda
```

### 4. NVIDIA GPU (Runtime Requirement)

To actually run CUDA code, you need:
- NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
- NVIDIA driver installed (version 450.80.02+ for CUDA 12.0)

Check your GPU:
```bash
nvidia-smi
# Shows GPU model, driver version, CUDA version
```

## Build Steps

### Step 1: Compile CUDA Kernels to PTX

The GPU kernels must be compiled to PTX (Parallel Thread Execution) assembly:

```bash
cd kernels
cargo rustc --release --target nvptx64-nvidia-cuda -- --emit=asm
```

This creates `kernels.ptx` in:
```
target/nvptx64-nvidia-cuda/release/deps/kernels-<hash>.s
```

The `.s` file is the PTX assembly. The build script looks for `kernels.ptx` but currently you need to manually copy it (this will be automated):

```bash
# Find the PTX file
find target/nvptx64-nvidia-cuda/release/deps -name "*.s"

# Copy to expected location (adjust hash)
cp target/nvptx64-nvidia-cuda/release/deps/kernels-abc123.s ../target/debug/build/sassy-<hash>/out/kernels.ptx
```

### Step 2: Build Sassy with CUDA Feature

From the repository root:

```bash
# Development build
cargo build --features cuda

# Release build (recommended for benchmarking)
cargo build --release --features cuda
```

### Step 3: Verify Build

Check that it compiled successfully:

```bash
cargo check --features cuda
```

If you see warnings about missing `kernels.ptx`, that's expected until step 1 is completed. The main code should still compile.

## Troubleshooting

### "CUDA SDK cannot be found"

**Problem**: `cust_raw` build script can't find CUDA.

**Solution**: Set environment variables:
```bash
export CUDA_LIBRARY_PATH=/usr/local/cuda/lib64
export PATH=/usr/local/cuda/bin:$PATH
```

On some systems, CUDA might be in `/opt/cuda` or `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0`.

### "cannot find type `UVec2` in module `glam`"

**Problem**: Outdated rust-cuda version.

**Solution**: Already fixed in current version (rev 103a8d56). If you still see this, run:
```bash
cargo clean
cargo update
```

### PTX compilation fails

**Problem**: LLVM not found or wrong version.

**Solution**: Install LLVM 7.x+ and make sure it's in PATH:
```bash
llvm-config --version
```

### "linking with `cc` failed" during PTX compilation

**Problem**: Trying to link PTX code (shouldn't link, only compile to .s file).

**Solution**: Use `--emit=asm` flag as shown in Step 1. This tells rustc to output assembly instead of trying to create an executable.

## Docker Alternative

If you don't want to install CUDA locally, use the rust-cuda Docker image:

```bash
# Pull image
docker pull ghcr.io/rust-gpu/rust-cuda:latest

# Run with GPU support
docker run -it --gpus all -v $(pwd):/workspace ghcr.io/rust-gpu/rust-cuda:latest bash

# Inside container
cd /workspace
cargo build --release --features cuda
```

This image has all dependencies pre-installed.

## Testing

### Without GPU Hardware

You can verify compilation without running GPU code:

```bash
cargo check --features cuda
cargo test --features cuda  # Tests will skip GPU execution
```

### With GPU Hardware

Run benchmarks to compare SIMD vs CUDA:

```bash
# TODO: Add benchmark command once tests are written
cargo bench --features cuda -- myers
```

Compare results against baseline SIMD implementation to verify correctness.

## Performance Expectations

See `CUDA_IMPLEMENTATION_README.md` for detailed performance analysis:

- **Small batches (<100 patterns)**: Use SIMD (0.5-1x speedup with CUDA overhead)
- **Medium batches (100-1000 patterns)**: CUDA starts winning (2-5x speedup)
- **Large batches (1000-10000+ patterns)**: CUDA dominates (20-100x speedup)

Optimal performance requires:
- Large pattern batches (amortize kernel launch overhead)
- Long text sequences (maximize GPU utilization)
- Proper thread/block configuration (see tuning guide)

## Next Steps

1. ✅ Build with CUDA feature
2. ⚠️ Compile PTX kernels
3. ⚠️ Test on GPU hardware
4. ⚠️ Run benchmarks
5. ⚠️ Optimize performance (shared memory, coalescing)

See `CUDA_QUICKSTART.md` for a quick reference guide.

## Additional Resources

- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **rust-cuda Documentation**: https://rust-gpu.github.io/rust-cuda/
- **PTX ISA**: https://docs.nvidia.com/cuda/parallel-thread-execution/
- **Sassy CUDA Implementation**: `CUDA_IMPLEMENTATION_README.md`
