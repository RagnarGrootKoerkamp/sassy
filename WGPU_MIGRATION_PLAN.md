# Migration Plan: CUDA → WGPU for GPU Acceleration

## Why WGPU?

### Advantages over CUDA
1. **Cross-platform**: Works on NVIDIA, AMD, Intel GPUs
2. **Cross-OS**: Windows, Linux, macOS, Web (WebGPU)
3. **Better Rust support**: rust-gpu project is actively maintained
4. **No CUDA SDK required**: Uses Vulkan/Metal/DX12 backends
5. **Works today**: No dependency issues like rust-cuda's half/nvptx64 problem

### What We Lose
- Potentially lower peak performance on NVIDIA GPUs (CUDA is optimized for them)
- Less direct control over GPU hardware features

### Net Benefit
**Huge win** - portability + working toolchain >> small performance difference

## Architecture Comparison

### CUDA Approach (Currently Blocked)
```
src/pattern_tiling/
├── backend_cuda.rs          # CudaBackend trait
├── search_cuda.rs           # MyersCuda with cust
└── ...

kernels/                     # Separate crate
└── src/lib.rs              # cuda_std → BLOCKED by half crate
```

### WGPU Approach (Following rust-gpu-chimera)
```
src/pattern_tiling/
├── backend_wgpu.rs          # WgpuBackend trait
├── search_wgpu.rs           # MyersWgpu with wgpu
└── ...

gpu_kernel/                  # New separate crate (like chimera's "kernel")
├── Cargo.toml              # Uses spirv-std for GPU target
└── src/lib.rs              # Shader code compiled to SPIR-V

build.rs                     # Compiles GPU kernel to SPIR-V using spirv-builder
```

## Implementation Plan

### Phase 1: Setup WGPU Infrastructure

1. **Create gpu_kernel crate** (analogous to chimera's kernel/)
   - Cargo.toml with spirv-std dependency
   - conditional compilation for `target_arch = "spirv"`
   - lib.rs with Myers algorithm for GPU

2. **Update root Cargo.toml**
   - Add `wgpu` feature
   - Dependencies: wgpu, pollster, futures
   - Build dependency: spirv-builder

3. **Create build.rs**
   - Compile gpu_kernel to SPIR-V when wgpu feature enabled
   - Embed SPIR-V in binary (like chimera does)

### Phase 2: Implement GPU Kernel

**File**: `gpu_kernel/src/lib.rs`

Key differences from CUDA:
- Use `spirv-std` instead of `cuda_std`
- Entry point: `#[spirv(compute(threads(256)))]` instead of `#[kernel]`
- Thread indexing: `spirv_std::arch::global_invocation_id()` instead of CUDA thread API
- Atomics: spirv-std atomics instead of CUDA atomics
- No `#[no_std]` needed - spirv-std provides what we need

Structure:
```rust
#![cfg_attr(target_arch = "spirv", no_std)]

#[cfg(target_arch = "spirv")]
use spirv_std::{spirv, glam::UVec3};

// Myers step function (same logic as CUDA version)
fn myers_step_scalar(...) -> (...) {
    // Bit-parallel algorithm - works identically
}

// Main compute shader entry point
#[spirv(compute(threads(256)))]  // Workgroup size
pub fn myers_search_kernel(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] text: &[u8],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] peqs: &[u64],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hits: &mut [HitRange],
    // Push constants for parameters
    #[spirv(push_constant)] params: &MyersParams,
) {
    let thread_id = global_id.x as usize;
    // Rest of Myers algorithm...
}
```

### Phase 3: Host-Side WGPU Runner

**File**: `src/pattern_tiling/search_wgpu.rs`

Following chimera's pattern:

```rust
pub struct MyersWgpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl MyersWgpu {
    pub async fn new() -> Result<Self, String> {
        // Request adapter (auto-detect Vulkan/Metal/DX12)
        // Create device
        // Load embedded SPIR-V kernel
        // Create compute pipeline
    }
    
    pub fn search_ranges(&self, ...) -> Result<...> {
        // Create GPU buffers for text, PEQs, output
        // Create bind group
        // Set push constants
        // Dispatch compute shader
        // Read results back
    }
}
```

### Phase 4: Backend Trait Implementation

**File**: `src/pattern_tiling/backend_wgpu.rs`

```rust
pub struct WgpuBackend;

impl SimdBackend for WgpuBackend {
    type Simd = u64;  // Scalar like CUDA (each thread = one pattern)
    const LANES: usize = 1;
    
    // Same scalar operations as CudaBackend
}
```

### Phase 5: Feature Flag Integration

Update `Cargo.toml`:
```toml
[features]
default = []
cuda = ["cust", "kernels"]           # Keep CUDA (blocked)
wgpu = ["dep:wgpu", "pollster", "futures", "spirv-builder"]  # NEW

[dependencies]
wgpu = { version = "26.0", features = ["spirv"], optional = true }
pollster = { version = "0.3", optional = true }
futures = { version = "0.3", optional = true }
gpu_kernel = { path = "gpu_kernel", optional = true }

[build-dependencies]
spirv-builder = { git = "https://github.com/rust-gpu/rust-gpu", branch = "main", optional = true }
```

## Timeline

**Day 1-2**: Setup infrastructure
- Create gpu_kernel crate
- Setup build.rs for SPIR-V compilation
- Test basic kernel compilation

**Day 2-3**: Implement GPU kernel
- Port Myers algorithm to spirv-std
- Handle thread indexing, atomics
- Test kernel builds successfully

**Day 3-4**: Implement host runner
- WGPU device setup
- Buffer management
- Pipeline creation
- Kernel dispatch

**Day 4-5**: Testing & optimization
- Verify correctness vs SIMD
- Benchmark performance
- Tune workgroup sizes
- Documentation

## Key Differences from CUDA

| Aspect | CUDA | WGPU |
|--------|------|------|
| Compilation target | nvptx64 PTX | SPIR-V |
| Std library | cuda_std | spirv-std |
| Entry point | `#[kernel]` | `#[spirv(compute(...))]` |
| Thread ID | `thread::index_1d()` | `global_invocation_id.x` |
| Atomics | `cuda_std::atomics` | `spirv_std::arch::atomic_*` |
| Memory | Pointers | Storage buffers + descriptors |
| Parameters | Direct args | Push constants + bindings |
| Backend | CUDA only | Vulkan/Metal/DX12/OpenGL |
| Platform | NVIDIA only | All GPUs |

## Expected Challenges

1. **SPIR-V atomics**: May have different semantics than CUDA
2. **Memory layout**: SPIR-V storage buffers vs CUDA global memory
3. **Synchronization**: Workgroup barriers vs CUDA thread blocks
4. **Performance tuning**: Different optimal workgroup sizes

## Testing Strategy

1. **Unit tests**: Test Myers step function in CPU mode
2. **Integration tests**: Compare WGPU results vs SIMD baseline
3. **Benchmarks**: Measure vs SIMD across different pattern counts
4. **Cross-platform**: Test on Vulkan (Linux), Metal (Mac), DX12 (Windows)

## Fallback Plan

If WGPU has issues:
1. Keep SIMD as primary (already works great)
2. WGPU as optional feature
3. Can revisit CUDA when rust-cuda ecosystem matures

## Success Criteria

✅ GPU kernel compiles to SPIR-V successfully
✅ Host code runs on at least one backend (Vulkan or Metal)
✅ Results match SIMD implementation
✅ Performance > SIMD for large pattern batches (1000+)
✅ Works on multiple platforms

## References

- rust-gpu-chimera: `/home/philae/git/eth/git/clones/rust-gpu-chimera/`
- rust-gpu docs: https://github.com/EmbarkStudios/rust-gpu
- wgpu docs: https://wgpu.rs/
- SPIR-V spec: https://registry.khronos.org/SPIR-V/
