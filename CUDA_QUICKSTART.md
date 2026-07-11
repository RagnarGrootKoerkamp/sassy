# CUDA Implementation Quickstart

This guide provides a quick path to getting the CUDA implementation working.

## Current Status

The CUDA generalization is **scaffolded but not yet active**. The code structure is in place with:

✅ **Complete**:
- Migration plan (`CUDA_MIGRATION_PLAN.md`)
- Architecture documentation (`CUDA_IMPLEMENTATION_README.md`)
- Backend abstraction (`src/pattern_tiling/backend_cuda.rs`)
- Kernel template (`kernels/src/lib.rs`)
- Host interface (`src/pattern_tiling/search_cuda.rs`)
- Build system (`build.rs`)

🚧 **To Activate**:
1. Uncomment rust-cuda dependencies
2. Set up CUDA toolkit and LLVM 7.x
3. Enable kernel compilation
4. Test and benchmark

## Quick Setup (Docker - Recommended)

The easiest way to get started is using Docker:

```bash
# 1. Install NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 2. Pull rust-cuda dev image
docker pull ghcr.io/rust-gpu/rust-cuda:latest

# 3. Run container with your project mounted
cd /path/to/sassy
docker run --gpus all -it -v $(pwd):/workspace ghcr.io/rust-gpu/rust-cuda:latest bash

# 4. Inside container, uncomment dependencies and build
cd /workspace
# Edit files to uncomment CUDA code (see below)
cargo build --features cuda
```

## Manual Setup (Advanced)

### Prerequisites

1. **CUDA Toolkit** (≥ 12.0):
   ```bash
   # Download from NVIDIA
   wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
   sudo sh cuda_12.3.0_545.23.06_linux.run
   ```

2. **LLVM 7.x**:
   ```bash
   # Option 1: Let rustc_codegen_nvvm download it automatically (Windows only)
   # Option 2: Build from source
   git clone --depth 1 --branch release/7.x https://github.com/llvm/llvm-project.git
   cd llvm-project
   mkdir build && cd build
   cmake -G Ninja ../llvm -DCMAKE_BUILD_TYPE=Release
   ninja install
   export LLVM_CONFIG=/usr/local/bin/llvm-config
   ```

3. **Rust nightly** (specific version):
   ```bash
   # Copy rust-toolchain.toml from rust-cuda repo
   # https://github.com/Rust-GPU/rust-cuda/blob/main/rust-toolchain.toml
   ```

### Activation Steps

1. **Add dependencies to `Cargo.toml`**:
   ```toml
   [dependencies]
   cust = { git = "https://github.com/rust-gpu/rust-cuda", rev = "7fa76f3d" }
   kernels = { path = "kernels" }
   
   [build-dependencies]
   cuda_builder = { 
       git = "https://github.com/rust-gpu/rust-cuda",
       rev = "7fa76f3d",
       features = ["rustc_codegen_nvvm"]
   }
   
   [features]
   cuda = ["cust"]
   ```

2. **Uncomment in `kernels/Cargo.toml`**:
   ```toml
   [dependencies]
   cuda_std = { git = "https://github.com/rust-gpu/rust-cuda", rev = "7fa76f3d" }
   ```

3. **Uncomment in `build.rs`**:
   ```rust
   use cuda_builder::CudaBuilder;
   
   fn main() {
       CudaBuilder::new("kernels")
           .copy_to(concat!(env!("OUT_DIR"), "/kernels.ptx"))
           .build()
           .unwrap();
   }
   ```

4. **Uncomment in `kernels/src/lib.rs`**:
   - Uncomment `use cuda_std::prelude::*;`
   - Uncomment the `#[kernel]` function

5. **Uncomment in `src/pattern_tiling/search_cuda.rs`**:
   - Uncomment all CUDA implementation code
   - Remove placeholder struct

6. **Add module to `src/pattern_tiling/mod.rs`**:
   ```rust
   #[cfg(feature = "cuda")]
   pub mod backend_cuda;
   
   #[cfg(feature = "cuda")]
   pub mod search_cuda;
   ```

7. **Build**:
   ```bash
   cargo build --features cuda
   ```

## Testing

```bash
# Run tests (requires GPU)
cargo test --features cuda

# Run benchmarks
cargo bench --features cuda

# Compare CUDA vs SIMD
cargo test --features cuda cuda_vs_simd -- --ignored
```

## Minimal Example

Once activated, use it like this:

```rust
use sassy::pattern_tiling::search_cuda::MyersCuda;
use sassy::profiles::Iupac;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU
    let mut gpu_searcher = MyersCuda::new()?;
    
    // Prepare patterns (100-10000 patterns recommended)
    let patterns: Vec<Vec<u8>> = (0..1000)
        .map(|i| format!("ACGT{:04}", i).into_bytes())
        .collect();
    
    // Large text to search
    let text = b"ACGTACGTACGT...";  // Your sequence here
    
    // Search with k=2 edits allowed
    let hits = gpu_searcher.search_ranges::<Iupac>(&patterns, text, 2, None)?;
    
    println!("Found {} hits on GPU", hits.len());
    
    Ok(())
}
```

## Verification

Check that GPU search matches SIMD:

```rust
use sassy::pattern_tiling::search::Myers;
use sassy::pattern_tiling::backend::U64;

// CPU/SIMD search
let mut cpu_searcher = Myers::<U64, Iupac>::new(None);
let cpu_hits = cpu_searcher.search_ranges(&tqueries, text, k, true, true);

// GPU search
let mut gpu_searcher = MyersCuda::new()?;
let gpu_hits = gpu_searcher.search_ranges::<Iupac>(&patterns, text, k, None)?;

// Compare
assert_eq!(cpu_hits.len(), gpu_hits.len());
```

## Performance Tuning

After basic functionality works:

1. **Adjust block size** in `search_cuda.rs`:
   ```rust
   let block_size = 128;  // Try 128, 256, 512
   ```

2. **Add shared memory** for text caching
3. **Profile** with Nsight Compute:
   ```bash
   ncu --set full target/release/examples/cuda_search
   ```

4. **Batch multiple texts** for better GPU utilization

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No CUDA device" | Run `nvidia-smi` to verify GPU is detected |
| "PTX compilation failed" | Check LLVM 7.x is installed: `llvm-config --version` |
| "Launch failed" | Enable blocking: `CUDA_LAUNCH_BLOCKING=1` |
| Incorrect results | Verify PEQ encoding matches profile |

## Next Steps

1. **Get it working**: Follow setup above, start with Docker
2. **Validate correctness**: Compare with SIMD reference
3. **Optimize**: Add shared memory, tune block size
4. **Benchmark**: Measure speedup vs SIMD
5. **Integrate**: Add to main CLI/API

## Resources

- **Detailed plan**: `CUDA_MIGRATION_PLAN.md`
- **Full docs**: `CUDA_IMPLEMENTATION_README.md`
- **Rust CUDA Guide**: https://rust-gpu.github.io/rust-cuda/
- **Example projects**: https://github.com/Rust-GPU/rust-cuda/tree/main/examples

## Questions?

- Check existing issues: https://github.com/Rust-GPU/rust-cuda/issues
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Rust GPU Discord: https://discord.gg/rust-gpu

---

**Status**: 🚧 Ready to activate. Uncomment code and dependencies to begin.
