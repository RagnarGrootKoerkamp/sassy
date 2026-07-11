//! CUDA-accelerated pattern search using Myers' bit-parallel algorithm
//! 
//! This module provides a GPU-accelerated version of the Myers algorithm for
//! approximate string matching. It offloads the search to NVIDIA GPUs using CUDA.

use cust::prelude::*;
use cust::memory::{DeviceBuffer, DeviceCopy};
use std::error::Error;
use std::fmt;

use crate::pattern_tiling::search::HitRange;
use crate::profiles::Profile;

/// Error types for CUDA operations
#[derive(Debug)]
pub enum CudaError {
    CudaError(cust::error::CudaError),
    NoDevice,
    KernelLaunchFailed(String),
    MemoryError(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CudaError::CudaError(e) => write!(f, "CUDA error: {}", e),
            CudaError::NoDevice => write!(f, "No CUDA device available"),
            CudaError::KernelLaunchFailed(msg) => write!(f, "Kernel launch failed: {}", msg),
            CudaError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
        }
    }
}

impl Error for CudaError {}

impl From<cust::error::CudaError> for CudaError {
    fn from(e: cust::error::CudaError) -> Self {
        CudaError::CudaError(e)
    }
}

/// CUDA-accelerated Myers searcher
pub struct MyersCuda {
    _context: Context,
    module: Module,
    stream: Stream,
}

impl MyersCuda {
    /// Initialize CUDA context and load kernel module
    pub fn new() -> Result<Self, CudaError> {
        // Initialize CUDA
        cust::init(CudaFlags::empty())?;
        
        // Get first available device
        let device = Device::get_device(0).map_err(|_| CudaError::NoDevice)?;
        
        // Create context
        let _context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device,
        )?;
        
        // Load PTX module (compiled by build.rs)
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
        let module = Module::load_from_string(ptx)?;
        
        // Create stream for asynchronous operations
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        Ok(Self {
            _context,
            module,
            stream,
        })
    }
    
    /// Search for patterns in text using GPU acceleration
    /// 
    /// # Arguments
    /// * `queries` - Vector of query patterns (byte sequences)
    /// * `text` - Text to search in
    /// * `k` - Maximum edit distance allowed
    /// * `alpha` - Optional alpha value for overhang (None = 1.0)
    /// 
    /// # Returns
    /// Vector of hit ranges indicating where patterns were found
    pub fn search_ranges<P: Profile>(
        &mut self,
        queries: &[Vec<u8>],
        text: &[u8],
        k: u32,
        alpha: Option<f32>,
    ) -> Result<Vec<HitRange>, CudaError> {
        if queries.is_empty() || text.is_empty() {
            return Ok(Vec::new());
        }
        
        let num_patterns = queries.len();
        let pattern_length = queries[0].len();
        
        // Compute alpha pattern for overhang handling
        let alpha_val = alpha.unwrap_or(1.0);
        let alpha_pattern = generate_alpha_mask(alpha_val, pattern_length);
        
        // Pre-compute PEQs (Pattern Equality vectors) for all patterns
        let peqs = precompute_peqs::<P>(queries);
        
        // Allocate device memory
        let d_text = DeviceBuffer::from_slice(text)?;
        let d_peqs = DeviceBuffer::from_slice(&peqs)?;
        
        // Estimate maximum number of hits (heuristic: 10x patterns)
        let max_hits = num_patterns * 10;
        let mut d_hit_ranges = DeviceBuffer::<HitRange>::zeroed(max_hits)?;
        let mut d_hit_count = DeviceBuffer::<u32>::zeroed(1)?;
        
        // Get kernel function
        let kernel = self.module.get_function("myers_search_kernel")?;
        
        // Calculate launch configuration
        let block_size = 256; // Threads per block
        let grid_size = (num_patterns + block_size - 1) / block_size;
        
        // Launch kernel
        let last_bit_shift = (pattern_length - 1) as u32;
        
        unsafe {
            launch!(
                kernel<<<grid_size as u32, block_size, 0, self.stream>>>(
                    d_text.as_device_ptr(),
                    text.len(),
                    d_peqs.as_device_ptr(),
                    pattern_length,
                    num_patterns,
                    k,
                    alpha_pattern,
                    last_bit_shift,
                    d_hit_ranges.as_device_ptr(),
                    max_hits,
                    d_hit_count.as_device_ptr()
                )
            )?;
        }
        
        // Wait for kernel to complete
        self.stream.synchronize()?;
        
        // Copy results back to host
        let mut hit_count_host = vec![0u32; 1];
        d_hit_count.copy_to(&mut hit_count_host)?;
        let hit_count = hit_count_host[0] as usize;
        
        // Copy hit ranges
        let mut hit_ranges = vec![HitRange::default(); hit_count.min(max_hits)];
        d_hit_ranges.copy_to(&mut hit_ranges)?;
        
        Ok(hit_ranges)
    }
}

/// Pre-compute Pattern Equality vectors for all patterns
/// 
/// For each pattern and each possible character value (0-255), compute a 64-bit
/// vector indicating which positions in the pattern match that character.
fn precompute_peqs<P: Profile>(queries: &[Vec<u8>]) -> Vec<u64> {
    let num_patterns = queries.len();
    
    // Allocate: num_patterns * 256 u64 values
    let mut peqs = vec![0u64; num_patterns * 256];
    
    for (pattern_idx, query) in queries.iter().enumerate() {
        let base_offset = pattern_idx * 256;
        
        // For each position in pattern
        for (pos, &byte) in query.iter().enumerate() {
            if pos >= 64 {
                break; // Limit to 64-bit patterns
            }
            
            let encoded = P::encode_char(byte) as usize;
            peqs[base_offset + encoded] |= 1u64 << pos;
        }
    }
    
    peqs
}

/// Generate alpha mask for overhang handling
fn generate_alpha_mask(alpha: f32, length: usize) -> u64 {
    let mut mask = 0u64;
    let limit = length.min(64);
    
    for i in 0..limit {
        let val = ((i + 1) as f32 * alpha).floor() as u64 
                - (i as f32 * alpha).floor() as u64;
        if val >= 1 {
            mask |= 1u64 << i;
        }
    }
    
    mask
}

impl Default for HitRange {
    fn default() -> Self {
        HitRange {
            pattern_idx: 0,
            start: 0,
            end: 0,
        }
    }
}

// Implement DeviceCopy for HitRange to allow GPU memory transfers
unsafe impl DeviceCopy for HitRange {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::Iupac;
    
    #[test]
    #[ignore] // Only run when CUDA is available
    fn test_cuda_search_basic() {
        let mut searcher = MyersCuda::new().expect("Failed to initialize CUDA");
        
        let queries = vec![b"ACGT".to_vec(), b"TGCA".to_vec()];
        let text = b"AAACGTTTGCAAA";
        let k = 0;
        
        let hits = searcher
            .search_ranges::<Iupac>(&queries, text, k, None)
            .expect("Search failed");
        
        assert_eq!(hits.len(), 2, "Should find 2 exact matches");
    }
}
