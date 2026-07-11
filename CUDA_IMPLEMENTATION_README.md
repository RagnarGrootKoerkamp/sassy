# CUDA Implementation for Sassy Pattern Matching

This directory contains the CUDA GPU implementation of the Myers' bit-parallel algorithm for approximate string matching.

## Overview

The CUDA implementation follows the rust-cuda framework (https://rust-gpu.github.io/rust-cuda/) to parallelize pattern matching across GPU threads. Each GPU thread processes one query pattern independently, enabling massive parallelism for large batches of patterns.

## Architecture

### Key Design Decisions

1. **Thread-per-Pattern Mapping**: Each CUDA thread handles one query pattern
   - Enables natural parallelization across thousands of patterns
   - Leverages warp-level synchronization (32 threads/warp)
   - Allows coalesced memory access for shared text data

2. **Scalar Backend**: Unlike SIMD where lanes are explicit, CUDA threads are implicit
   - `CudaBackend` treats "SIMD" operations as scalar operations
   - Each thread maintains its own VP/VN/cost state
   - Parallelism comes from launching thousands of threads

3. **Memory Layout**:
   ```
   Text:  [shared across all threads] → Read-only, broadcast to all
   PEQs:  [per-pattern, 256 * u64]   → Pattern-specific equality vectors
   State: [per-thread registers]      → VP, VN, cost (in registers)
   Output: [global memory + atomics]  → Hit ranges written atomically
   ```

## File Structure

```
.
├── CUDA_MIGRATION_PLAN.md          # Detailed migration plan and rationale
├── CUDA_IMPLEMENTATION_README.md   # This file
├── build.rs                        # Build script to compile PTX
├── kernels/                        # GPU kernel code
│   ├── Cargo.toml                 # Kernel crate manifest
│   └── src/
│       └── lib.rs                 # CUDA kernel implementation
└── src/
    └── pattern_tiling/
        ├── backend_cuda.rs        # CudaBackend trait impl
        └── search_cuda.rs         # Host-side CUDA interface
```

## Implementation Status

### ✅ Completed
- [x] Migration plan and documentation
- [x] `CudaBackend` trait implementation (scalar operations)
- [x] Kernel template with `myers_step_scalar`
- [x] Host-side interface structure (`MyersCuda`)
- [x] Build system scaffolding

### 🚧 In Progress
- [ ] rust-cuda dependency configuration
- [ ] PTX compilation setup
- [ ] Kernel launch and memory management
- [ ] Testing and validation

### 📋 TODO
- [ ] Shared memory optimizations
- [ ] Warp-level primitives
- [ ] Multi-text batching
- [ ] Performance benchmarking
- [ ] Integration with main API

## Getting Started

### Prerequisites

1. **NVIDIA GPU** with Compute Capability ≥ 5.0 (Maxwell or later)
2. **CUDA Toolkit** ≥ 12.0
   - Download from: https://developer.nvidia.com/cuda-downloads
3. **NVIDIA Driver**
   - For CUDA 12: version 525-579
   - For CUDA 13: version ≥ 580
4. **LLVM 7.x** (7.0-7.4)
   - Required by `rustc_codegen_nvvm`
   - Set `LLVM_CONFIG` environment variable if not in PATH

### Installation Steps

1. **Add rust-cuda dependencies** (currently commented out in `Cargo.toml`):
   ```toml
   [dependencies]
   cust = { git = "https://github.com/rust-gpu/rust-cuda", rev = "7fa76f3d" }
   
   [build-dependencies]
   cuda_builder = { 
       git = "https://github.com/rust-gpu/rust-cuda",
       rev = "7fa76f3d",
       features = ["rustc_codegen_nvvm"]
   }
   ```

2. **Create `rust-toolchain.toml`** (copy from rust-cuda repo):
   ```toml
   [toolchain]
   channel = "nightly-YYYY-MM-DD"  # Match rust-cuda requirements
   ```

3. **Uncomment CUDA code**:
   - Uncomment dependencies in `kernels/Cargo.toml`
   - Uncomment build script in `build.rs`
   - Uncomment kernel code in `kernels/src/lib.rs`
   - Uncomment host interface in `src/pattern_tiling/search_cuda.rs`

4. **Build**:
   ```bash
   cargo build --features cuda
   ```

### Docker Alternative

To avoid complex LLVM/CUDA setup, use the rust-cuda Docker images:

```bash
# Pull rust-cuda development image
docker pull ghcr.io/rust-gpu/rust-cuda:latest

# Run container with GPU support
docker run --gpus all -it -v $(pwd):/workspace ghcr.io/rust-gpu/rust-cuda:latest

# Inside container
cd /workspace
cargo build --features cuda
```

## Usage Example

```rust
use sassy::pattern_tiling::search_cuda::MyersCuda;
use sassy::profiles::Iupac;

// Initialize CUDA context
let mut searcher = MyersCuda::new().expect("Failed to initialize CUDA");

// Prepare data
let queries = vec![
    b"ACGTACGT".to_vec(),
    b"TGCATGCA".to_vec(),
    // ... thousands more patterns
];
let text = b"AAACGTACGTTTGCATGCAAA";
let k = 2;  // Allow up to 2 edits

// Search on GPU
let hits = searcher
    .search_ranges::<Iupac>(&queries, text, k, None)
    .expect("GPU search failed");

// Process results
for hit in hits {
    println!(
        "Pattern {} matched at [{}, {}]",
        hit.pattern_idx, hit.start, hit.end
    );
}
```

## Performance Characteristics

### Expected Performance

| Batch Size | Speedup vs SIMD | Best Use Case |
|------------|-----------------|---------------|
| < 100      | 0.5-1x (slower) | Not recommended |
| 100-1000   | 5-20x           | Small to medium batches |
| 1000-10000 | 20-100x         | **Sweet spot** |
| > 10000    | 50-100x         | Large-scale screening |

### Optimization Opportunities

1. **Shared Memory for Text**
   - Load text chunks into shared memory
   - Reduces global memory bandwidth
   - ~2-3x speedup expected

2. **Warp Shuffles**
   - Exchange data within warp (32 threads)
   - Useful for reductions and pattern comparisons
   - Eliminates shared memory synchronization

3. **Texture Memory**
   - Cache text data with spatial locality
   - Beneficial for long texts with repeated access

4. **Stream Compaction**
   - Efficiently collect sparse hit ranges
   - Use parallel prefix sum (scan) for output

5. **Multi-Stream Execution**
   - Overlap computation with memory transfers
   - Process multiple texts concurrently

## Testing Strategy

### Unit Tests
```bash
# Test CUDA backend (CPU-side)
cargo test backend_cuda

# Test kernel logic (CPU emulation)
cargo test kernels
```

### Integration Tests
```bash
# Compare CUDA results with SIMD reference
cargo test --features cuda cuda_integration

# Fuzzing for correctness
cargo fuzz run cuda_vs_simd
```

### Benchmarks
```bash
# Throughput benchmarks
cargo bench --features cuda pattern_search_cuda

# Profile with nvprof/Nsight
nvprof target/release/deps/pattern_search_cuda-*
```

## Debugging

### Enable CUDA Error Checking
```rust
std::env::set_var("CUDA_LAUNCH_BLOCKING", "1");
```

### Print Kernel Output
Add debug prints in kernel (limited support):
```rust
// In kernel
cuda_std::vprintf(b"tid=%d, cost=%llu\n\0", tid, cost);
```

### Use cuda-gdb
```bash
cuda-gdb target/release/sassy
```

## Known Limitations

1. **Pattern Length**: Currently limited to 64 bp (64-bit word size)
   - Could extend with multi-word support
   
2. **Output Buffer**: Pre-allocated, may overflow with many hits
   - Need dynamic allocation or stream compaction
   
3. **Single Text**: Processes one text at a time
   - Could batch multiple texts for better GPU utilization

4. **No Reverse Complement**: Not yet implemented
   - Would require separate kernel invocations or doubling patterns

## Troubleshooting

### Common Issues

1. **"No CUDA device available"**
   - Check: `nvidia-smi`
   - Ensure driver is installed and GPU is visible

2. **"PTX compilation failed"**
   - Verify LLVM 7.x is installed
   - Check `LLVM_CONFIG` environment variable
   - Try using Docker image instead

3. **"Kernel launch failed"**
   - Check grid/block dimensions (must be > 0)
   - Verify memory allocations succeeded
   - Enable `CUDA_LAUNCH_BLOCKING=1` for better errors

4. **Incorrect results**
   - Validate PEQ pre-computation
   - Compare with SIMD reference implementation
   - Check for race conditions in output writes

## Contributing

When adding features or optimizations:

1. **Maintain compatibility** with existing SIMD interface
2. **Add tests** comparing CUDA vs SIMD results
3. **Benchmark** to verify performance improvements
4. **Document** any CUDA-specific behavior or limitations

## References

- **Rust CUDA Guide**: https://rust-gpu.github.io/rust-cuda/
- **CUDA C Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Myers' Algorithm**: https://doi.org/10.1145/316542.316550
- **GPU-Accelerated Myers**: https://doi.org/10.1186/1471-2105-11-S12-S12

## License

Same as parent project (MIT).
