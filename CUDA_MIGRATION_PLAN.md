# CUDA Migration Plan for Lane-wise SIMD Code

## Overview
This document outlines the plan to generalize the lane-wise SIMD code in `src/pattern_tiling/search.rs` to CUDA Rust GPU code using the rust-cuda framework.

## Current Architecture

### SIMD Abstraction
The code uses a `SimdBackend` trait to abstract over different SIMD implementations:
- **Trait**: `SimdBackend` with associated types `Simd`, `Scalar`, and `LaneArray`
- **Key Operations**: arithmetic, bitwise, shifts, comparisons, lane extraction
- **Implementations**: U8 (32 lanes), U16 (16 lanes), U32 (8 lanes), U64 (4 lanes), plus AVX-512 variants

### Core Algorithm (Myers' Bit-parallel)
The `Myers` struct processes multiple query patterns in parallel using SIMD lanes:
1. **Initialization**: Set up blocks with initial VP/VN/cost states
2. **Main Loop**: Iterate over text, performing `myers_step` for each character
3. **myers_step**: Core bitwise operations computing edit distance updates
4. **Range Tracking**: Detect when patterns enter/exit valid cost range

### Key Data Structures
```rust
struct BlockState<S: Copy> {
    vp: S,      // Vertical positive (matches + insertions)
    vn: S,      // Vertical negative (deletions)
    cost: S,    // Current edit distance
    active_mask: u64,  // Bitmask of active lanes
}

struct Myers<B: SimdBackend, P: Profile> {
    blocks: Vec<BlockState<B::Simd>>,
    // ... other fields
}
```

## CUDA Architecture Design

### 1. Thread/Block Mapping

**Option A: Thread-per-Query (Recommended)**
- Each GPU thread handles one query pattern
- Block of threads = SIMD "lane" equivalent
- Grid dimension = number of patterns / block size

```
Grid:    [Block 0] [Block 1] ... [Block N]
         ↓         ↓             ↓
Block:   [T0 T1 ... T31] [T32 T33 ... T63] ...
         ↑
Thread:  handles 1 query pattern
```

**Advantages**:
- Natural 1:1 mapping of SIMD lane → GPU thread
- Warp-level synchronization already built-in (32 threads/warp)
- Can use warp shuffles for reductions
- Memory coalescing for pattern data access

### 2. GPU Backend Implementation

Create a new backend trait implementation:

```rust
// File: src/pattern_tiling/backend_cuda.rs

pub struct CudaBackend;

impl SimdBackend for CudaBackend {
    type Simd = u64;     // Each "SIMD value" is scalar in GPU thread
    type Scalar = u64;
    type LaneArray = [u64; 1];  // Single element
    
    const LANES: usize = 1;  // Each thread processes 1 lane
    const LIMB_BITS: usize = 64;
    
    // Operations become identity functions since each thread has scalar values
    fn splat_scalar(value: Self::Scalar) -> Self::Simd { value }
    fn simd_gt(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
        if lhs > rhs { !0u64 } else { 0u64 }
    }
    // ...
}
```

### 3. Kernel Structure

The main search kernel:

```rust
// File: kernels/src/lib.rs

use cuda_std::prelude::*;

#[kernel]
pub unsafe fn myers_search_kernel(
    // Input: text
    text: &[u8],
    text_len: usize,
    
    // Input: pattern encodings (pre-computed PEQs)
    peqs: &[u64],  // Flattened: [pattern0_char0, pattern0_char1, ..., pattern1_char0, ...]
    pattern_length: usize,
    num_patterns: usize,
    
    // Input: parameters
    k: u32,
    alpha_pattern: u64,
    
    // Output: hit ranges
    hit_ranges: *mut HitRange,
    hit_count: *mut u32,
) {
    let tid = thread::index_1d() as usize;
    
    if tid >= num_patterns {
        return;
    }
    
    // Each thread processes one pattern
    let pattern_peqs = &peqs[tid * 256..(tid + 1) * 256];  // 256 chars
    
    // Initialize state
    let mut vp = alpha_pattern;
    let mut vn = 0u64;
    let mut cost = alpha_pattern.count_ones() as u64;
    
    let mut active = false;
    let mut range_start = -1isize;
    
    // Process text
    for pos in 0..text_len {
        let c = text[pos];
        let eq = pattern_peqs[c as usize];
        
        // Myers step
        let (vp_new, vn_new, cost_new) = myers_step_scalar(
            vp, vn, cost, eq, !0u64, pattern_length - 1
        );
        
        vp = vp_new;
        vn = vn_new;
        cost = cost_new;
        
        // Track ranges
        let was_active = active;
        active = cost <= k as u64;
        
        if active && !was_active {
            range_start = pos as isize;
        } else if !active && was_active {
            // Write hit range atomically
            let idx = atomics::add(hit_count, 1, Ordering::Relaxed);
            if idx < MAX_HITS {
                *hit_ranges.add(idx as usize) = HitRange {
                    pattern_idx: tid,
                    start: range_start,
                    end: (pos - 1) as isize,
                };
            }
        }
    }
    
    // Finalize if still active
    if active {
        let idx = atomics::add(hit_count, 1, Ordering::Relaxed);
        if idx < MAX_HITS {
            *hit_ranges.add(idx as usize) = HitRange {
                pattern_idx: tid,
                start: range_start,
                end: (text_len - 1) as isize,
            };
        }
    }
}

#[inline(always)]
fn myers_step_scalar(
    vp: u64,
    vn: u64,
    cost: u64,
    eq: u64,
    all_ones: u64,
    last_bit_pos: usize,
) -> (u64, u64, u64) {
    let eq_and_pv = eq & vp;
    let xh = ((eq_and_pv.wrapping_add(vp)) ^ vp) | eq;
    let mh = vp & xh;
    let ph = vn | (all_ones ^ (xh | vp));
    
    let ph_shifted = ph << 1;
    let mh_shifted = mh << 1;
    
    let xv = eq | vn;
    let vp_out = mh_shifted | (all_ones ^ (xv | ph_shifted));
    let vn_out = ph_shifted & xv;
    
    let ph_bit = (ph >> last_bit_pos) & 1;
    let mh_bit = (mh >> last_bit_pos) & 1;
    
    let cost_out = cost.wrapping_add(ph_bit).wrapping_sub(mh_bit);
    
    (vp_out, vn_out, cost_out)
}
```

### 4. CPU-Side Interface

```rust
// File: src/pattern_tiling/search_cuda.rs

use cust::prelude::*;
use cust::memory::DeviceBuffer;

pub struct MeyersCuda {
    context: Context,
    module: Module,
    stream: Stream,
}

impl MeyersCuda {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let device = Device::get_device(0)?;
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device
        )?;
        
        // Load PTX compiled from kernels crate
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
        let module = Module::load_from_string(ptx)?;
        
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        Ok(Self { context, module, stream })
    }
    
    pub fn search_ranges(
        &mut self,
        queries: &[Vec<u8>],
        text: &[u8],
        k: u32,
    ) -> Result<Vec<HitRange>, Box<dyn std::error::Error>> {
        // Prepare input data
        let peqs = precompute_peqs(queries);
        let num_patterns = queries.len();
        let pattern_length = queries[0].len();
        
        // Allocate device memory
        let d_text = DeviceBuffer::from_slice(text)?;
        let d_peqs = DeviceBuffer::from_slice(&peqs)?;
        let mut d_hit_ranges = DeviceBuffer::<HitRange>::zeroed(num_patterns * 10)?;
        let mut d_hit_count = DeviceBuffer::<u32>::zeroed(1)?;
        
        // Launch kernel
        let kernel = self.module.get_function("myers_search_kernel")?;
        
        let block_size = 256;
        let grid_size = (num_patterns + block_size - 1) / block_size;
        
        unsafe {
            launch!(
                kernel<<<grid_size as u32, block_size, 0, self.stream>>>(
                    d_text.as_device_ptr(),
                    text.len(),
                    d_peqs.as_device_ptr(),
                    pattern_length,
                    num_patterns,
                    k,
                    0xFFFFFFFFFFFFFFFF_u64, // alpha_pattern
                    d_hit_ranges.as_device_ptr(),
                    d_hit_count.as_device_ptr()
                )
            )?;
        }
        
        self.stream.synchronize()?;
        
        // Copy results back
        let hit_count = d_hit_count.copy_to(&mut vec![0u32; 1])?[0] as usize;
        let mut hit_ranges = vec![HitRange::default(); hit_count];
        d_hit_ranges.copy_to(&mut hit_ranges)?;
        
        Ok(hit_ranges)
    }
}
```

### 5. Build System

```toml
# File: Cargo.toml (root)
[dependencies]
cust = { git = "https://github.com/rust-gpu/rust-cuda", rev = "7fa76f3d" }
kernels = { path = "kernels" }

[build-dependencies]
cuda_builder = { 
    git = "https://github.com/rust-gpu/rust-cuda", 
    rev = "7fa76f3d",
    features = ["rustc_codegen_nvvm"] 
}

# File: kernels/Cargo.toml
[package]
name = "kernels"
version = "0.1.0"
edition = "2024"

[dependencies]
cuda_std = { 
    git = "https://github.com/rust-gpu/rust-cuda",
    rev = "7fa76f3d"
}

[lib]
crate-type = ["cdylib", "rlib"]

# File: build.rs
use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("kernels")
        .copy_to(concat!(env!("OUT_DIR"), "/kernels.ptx"))
        .build()
        .unwrap();
}

# File: rust-toolchain.toml
[toolchain]
channel = "nightly-2024-XX-XX"  # Match rust-cuda requirements
```

## Migration Steps

### Phase 1: Setup & Infrastructure
1. ✅ Add rust-cuda dependencies to Cargo.toml
2. ✅ Create `kernels/` subcrate
3. ✅ Set up build.rs for PTX compilation
4. ✅ Create CudaBackend trait implementation

### Phase 2: Core Kernel
1. ✅ Port `myers_step` to scalar GPU function
2. ✅ Implement basic search kernel
3. ✅ Add range tracking logic
4. ✅ Test with simple patterns

### Phase 3: Optimizations
1. Use shared memory for text chunks
2. Implement warp-level reductions
3. Add support for large pattern batches
4. Profile and optimize memory access patterns

### Phase 4: Integration
1. Create unified interface supporting both SIMD and CUDA
2. Add feature flag for CUDA support
3. Benchmarking and validation
4. Documentation

## Performance Considerations

### Memory Access Patterns
- **Text**: Read-only, broadcasted to all threads → use texture memory or shared memory
- **PEQs**: Read-only, per-pattern → coalesced access if organized properly
- **Results**: Write-only, sparse → atomic operations required

### Optimization Opportunities
1. **Shared Memory**: Load text chunks into shared memory for repeated access
2. **Warp Shuffles**: Exchange data between threads in same warp
3. **Texture Memory**: Cache text data for better spatial locality
4. **Stream Compaction**: Efficiently collect hit ranges
5. **Multiple Texts**: Batch process multiple texts concurrently

### Expected Performance
- **Throughput**: 10-100x improvement for large batches (1000+ patterns)
- **Latency**: Higher overhead for small batches (< 100 patterns)
- **Sweet Spot**: Medium to large batches (100-10000 patterns) against long texts

## Testing Strategy

1. **Unit Tests**: Validate `myers_step_scalar` matches CPU version
2. **Integration Tests**: Compare CUDA results with SIMD results
3. **Fuzzing**: Random pattern/text combinations
4. **Benchmarks**: Measure throughput vs SIMD for various batch sizes

## Open Questions

1. **Block Size**: What's optimal? (128, 256, 512 threads?)
2. **Memory Layout**: Row-major vs column-major for PEQs?
3. **Large Patterns**: How to handle patterns > 64bp?
4. **Dynamic Parallelism**: Useful for hierarchical search?

## References

- [Rust CUDA Guide](https://rust-gpu.github.io/rust-cuda/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Myers' Bit-parallel Algorithm](https://doi.org/10.1145/316542.316550)
- [GPU-accelerated Myers](https://doi.org/10.1186/1471-2105-11-S12-S12)
