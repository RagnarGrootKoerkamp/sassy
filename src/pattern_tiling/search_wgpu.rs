//! WGPU-accelerated pattern search using Myers' bit-parallel algorithm
//!
//! This module offloads the Myers algorithm to the GPU via WGPU, which works
//! across Vulkan, Metal, DX12 and OpenGL - i.e. on NVIDIA, AMD, Intel and Apple
//! GPUs, unlike the CUDA backend which is NVIDIA-only and currently blocked by
//! an upstream PTX compilation issue (see CUDA_PTX_COMPILATION_ISSUE.md).
//!
//! The actual compute kernel lives in the separate `gpu_kernel` crate, which is
//! compiled to SPIR-V by `build.rs` using `spirv-builder`, following the
//! pattern used by the rust-gpu-chimera example project.

use std::error::Error;
use std::fmt;

use gpu_kernel::{HitRange as GpuHitRange, MyersParams};
use wgpu::util::DeviceExt;

use crate::pattern_tiling::search::HitRange;
use crate::profiles::Profile;

/// The compiled SPIR-V bytes for the Myers search kernel, embedded at build time.
static MYERS_KERNEL_SPV: &[u8] = include_bytes!(env!("MYERS_KERNEL_SPV_PATH"));

/// Entry point name of the compiled kernel.
const MYERS_KERNEL_ENTRY: &str = env!("MYERS_KERNEL_SPV_ENTRY");

/// Error types for WGPU operations.
#[derive(Debug)]
pub enum WgpuSearchError {
    NoAdapter,
    RequestDevice(String),
    Other(String),
}

impl fmt::Display for WgpuSearchError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WgpuSearchError::NoAdapter => write!(f, "No suitable WGPU adapter found"),
            WgpuSearchError::RequestDevice(msg) => write!(f, "Failed to request device: {}", msg),
            WgpuSearchError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for WgpuSearchError {}

/// WGPU-accelerated Myers searcher.
pub struct MyersWgpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    backend_name: String,
    adapter_name: String,
}

impl MyersWgpu {
    pub fn new() -> Self {
        pollster::block_on(Self::new_async()).unwrap()
    }
    /// Initialize the WGPU device and compute pipeline. We require the Vulkan
    /// backend (matching the `spirv-unknown-vulkan1.2` target used to compile
    /// the kernel): other backends (GL, DX12 without passthrough) go through
    /// naga's SPIR-V -> IR translation, which does not support all the
    /// constructs emitted by rustc_codegen_spirv (e.g. narrow/bool types).
    pub async fn new_async() -> Result<Self, WgpuSearchError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| WgpuSearchError::NoAdapter)?;

        let info = adapter.get_info();
        let backend_name = match info.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DirectX 12",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "Unknown",
        }
        .to_string();
        let adapter_name = info.name.clone();

        // Prefer raw SPIR-V passthrough when the adapter supports it, so the
        // shader bytes we compiled are used as-is instead of being
        // round-tripped through naga (which can choke on some rust-gpu
        // output, e.g. narrow/bool types).
        let adapter_features = adapter.features();
        let required_features =
            if adapter_features.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH) {
                wgpu::Features::PUSH_CONSTANTS | wgpu::Features::SPIRV_SHADER_PASSTHROUGH
            } else {
                wgpu::Features::PUSH_CONSTANTS
            };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Myers WGPU Device"),
                required_features,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 128,
                    ..Default::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::default(),
            })
            .await
            .map_err(|e| WgpuSearchError::RequestDevice(format!("{e:?}")))?;

        let (pipeline, bind_group_layout) = Self::create_pipeline(&device);

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            backend_name,
            adapter_name,
        })
    }

    /// Human-readable info about the active backend (for logging/diagnostics).
    pub fn backend_info(&self) -> (&str, &str) {
        (self.backend_name.as_str(), self.adapter_name.as_str())
    }

    fn create_pipeline(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        // Prefer raw SPIR-V passthrough (bypasses naga's SPIR-V -> IR
        // translation, which does not support all constructs emitted by
        // rustc_codegen_spirv, e.g. narrow/bool types). Fall back to the
        // naga-validated path if the device doesn't support passthrough.
        let shader_module = if device
            .features()
            .contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH)
        {
            let spirv_words: std::borrow::Cow<[u32]> = wgpu::util::make_spirv_raw(MYERS_KERNEL_SPV);
            unsafe {
                device.create_shader_module_passthrough(
                    wgpu::ShaderModuleDescriptorPassthrough::SpirV(
                        wgpu::ShaderModuleDescriptorSpirV {
                            label: Some("Myers Search Kernel"),
                            source: spirv_words,
                        },
                    ),
                )
            }
        } else {
            let spirv_data = wgpu::util::make_spirv(MYERS_KERNEL_SPV);
            unsafe {
                device.create_shader_module_trusted(
                    wgpu::ShaderModuleDescriptor {
                        label: Some("Myers Search Kernel"),
                        source: spirv_data,
                    },
                    wgpu::ShaderRuntimeChecks::unchecked(),
                )
            }
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Myers Bind Group Layout"),
            entries: &[
                // binding 0: text (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: PEQs (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: hit ranges (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: atomic hit counter (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Myers Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<MyersParams>() as u32,
            }],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Myers Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(MYERS_KERNEL_ENTRY),
            compilation_options: Default::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    /// Search for patterns in text using GPU acceleration.
    ///
    /// # Arguments
    /// * `queries` - Vector of query patterns (byte sequences)
    /// * `text` - Text to search in
    /// * `k` - Maximum edit distance allowed
    /// * `alpha` - Optional alpha value for overhang (None = 1.0)
    ///
    /// # Returns
    /// Vector of hit ranges indicating where patterns were found.
    pub fn search_ranges<P: Profile>(
        &self,
        queries: &[Vec<u8>],
        text: &[u8],
        k: u32,
        alpha: Option<f32>,
    ) -> Result<Vec<HitRange>, WgpuSearchError> {
        pollster::block_on(self.search_ranges_async::<P>(queries, text, k, alpha))
    }

    async fn search_ranges_async<P: Profile>(
        &self,
        queries: &[Vec<u8>],
        text: &[u8],
        k: u32,
        alpha: Option<f32>,
    ) -> Result<Vec<HitRange>, WgpuSearchError> {
        if queries.is_empty() || text.is_empty() {
            return Ok(Vec::new());
        }

        let num_patterns = queries.len() as u32;
        let pattern_length = queries[0].len() as u32;

        let alpha_val = alpha.unwrap_or(1.0);
        let alpha_pattern = generate_alpha_mask(alpha_val, pattern_length as usize);

        let peqs = precompute_peqs::<P>(queries);

        // Text and PEQ buffers are independent of max_hits, so build them once
        // and reuse them across the grow-and-retry dispatches below.
        let text_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Myers Text Buffer"),
                contents: text,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let peqs_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Myers PEQs Buffer"),
                contents: bytemuck::cast_slice(&peqs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let num_workgroups = num_patterns.div_ceil(gpu_kernel::WORKGROUP_SIZE);

        // The kernel counts every hit atomically but only writes ranges while
        // idx < max_hits. If the true count exceeds the buffer, the surplus is
        // dropped, so grow to the reported count and re-dispatch until every hit
        // fits (converges in <= 2 iterations: the retry sizes to the exact count).
        let mut max_hits = (num_patterns as usize * 10).max(1);
        let (hit_count, hit_ranges_staging) = loop {
            let params = MyersParams {
                text_len: text.len() as u32,
                pattern_length,
                num_patterns,
                k,
                alpha_pattern,
                last_bit_shift: pattern_length.saturating_sub(1),
                max_hits: max_hits as u32,
            };
            let (hit_count, hit_ranges_staging) = self
                .dispatch_and_count(&params, &text_buffer, &peqs_buffer, num_workgroups)
                .await?;
            match next_hit_capacity(hit_count, max_hits as u32) {
                Some(bigger) => max_hits = bigger as usize, // overflow: grow and retry
                None => break (hit_count, hit_ranges_staging),
            }
        };

        let actual_hits = (hit_count as usize).min(max_hits);
        if actual_hits == 0 {
            return Ok(Vec::new());
        }

        // Read back hit ranges.
        let ranges_slice = hit_ranges_staging.slice(..);
        let (ranges_sender, ranges_receiver) = futures::channel::oneshot::channel();
        ranges_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = ranges_sender.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        ranges_receiver
            .await
            .map_err(|e| WgpuSearchError::Other(format!("Channel error: {e:?}")))?
            .map_err(|e| WgpuSearchError::Other(format!("Buffer async error: {e:?}")))?;

        let hit_ranges = {
            let view = ranges_slice.get_mapped_range();
            let gpu_ranges: &[GpuHitRange] = bytemuck::cast_slice(&view);
            gpu_ranges[..actual_hits]
                .iter()
                .map(|r| HitRange {
                    pattern_idx: r.pattern_idx as usize,
                    start: r.start as isize,
                    end: r.end as isize,
                })
                .collect()
        };
        hit_ranges_staging.unmap();

        Ok(hit_ranges)
    }

    /// Dispatch the kernel once with `params` and read back the atomic hit count.
    /// Returns the count and the hit-ranges staging buffer (mapped-readable), so
    /// the caller can read ranges from the final, non-overflowing dispatch. Fresh
    /// hit-ranges/hit-count buffers are created each call (their size depends on
    /// `params.max_hits`); the `text`/`peqs` buffers are passed in and reused.
    async fn dispatch_and_count(
        &self,
        params: &MyersParams,
        text_buffer: &wgpu::Buffer,
        peqs_buffer: &wgpu::Buffer,
        num_workgroups: u32,
    ) -> Result<(u32, wgpu::Buffer), WgpuSearchError> {
        let hit_ranges_size =
            (params.max_hits as usize * std::mem::size_of::<GpuHitRange>()) as u64;

        let hit_ranges_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Myers Hit Ranges Buffer"),
            size: hit_ranges_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let hit_count_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Myers Hit Count Buffer"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let hit_ranges_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Myers Hit Ranges Staging"),
            size: hit_ranges_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hit_count_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Myers Hit Count Staging"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Myers Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: text_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: peqs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: hit_ranges_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hit_count_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Myers Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Myers Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(params));
            compute_pass.dispatch_workgroups(num_workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&hit_ranges_buffer, 0, &hit_ranges_staging, 0, hit_ranges_size);
        encoder.copy_buffer_to_buffer(
            &hit_count_buffer,
            0,
            &hit_count_staging,
            0,
            std::mem::size_of::<u32>() as u64,
        );

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);

        // Read back the hit count.
        let count_slice = hit_count_staging.slice(..);
        let (count_sender, count_receiver) = futures::channel::oneshot::channel();
        count_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = count_sender.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        count_receiver
            .await
            .map_err(|e| WgpuSearchError::Other(format!("Channel error: {e:?}")))?
            .map_err(|e| WgpuSearchError::Other(format!("Buffer async error: {e:?}")))?;

        let hit_count = {
            let view = count_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u32>(&view)[0]
        };
        hit_count_staging.unmap();

        Ok((hit_count, hit_ranges_staging))
    }
}

/// Given the atomically-counted hit total reported by the kernel and the
/// capacity of the output buffer it wrote into, return `Some(new_capacity)` if
/// the buffer overflowed (so the caller must re-dispatch with a bigger buffer),
/// or `None` if all hits fit. The kernel counts every hit but only writes while
/// `idx < max_hits`, so `reported > capacity` means hits were dropped.
fn next_hit_capacity(reported: u32, current_capacity: u32) -> Option<u32> {
    if reported > current_capacity {
        Some(reported)
    } else {
        None
    }
}

/// Pre-compute Pattern Equality vectors for all patterns.
///
/// For each pattern and each possible character value (0-255), compute a
/// 64-bit vector indicating which positions in the pattern match that
/// character.
fn precompute_peqs<P: Profile>(queries: &[Vec<u8>]) -> Vec<u64> {
    let num_patterns = queries.len();
    let mut peqs = vec![0u64; num_patterns * 256];

    // Pre-compute the IUPAC-style encoding for every possible raw text byte
    // (0..256) once, since `P::encode_char` returns a small bitmask (e.g. one
    // bit per base for the Iupac profile) rather than a literal index.
    let mut encoded_byte = [0u8; 256];
    for (v, slot) in encoded_byte.iter_mut().enumerate() {
        *slot = P::encode_char(v as u8);
    }

    for (pattern_idx, query) in queries.iter().enumerate() {
        let base_offset = pattern_idx * 256;
        for (pos, &pattern_byte) in query.iter().enumerate() {
            if pos >= 64 {
                break; // Limit to 64-bit patterns.
            }
            let enc_pattern = P::encode_char(pattern_byte);
            if enc_pattern == 0 {
                continue; // Skip wildcard/unmatched pattern chars, as in TQueries::new.
            }
            let bit = 1u64 << pos;
            // A pattern position matches a text byte whenever their IUPAC
            // encodings share at least one bit (ambiguity-code compatible),
            // mirroring the bitwise-overlap check in `TQueries::new` /
            // `Profile::is_match`.
            for (byte, &enc_text) in encoded_byte.iter().enumerate() {
                if (enc_pattern & enc_text) != 0 {
                    peqs[base_offset + byte] |= bit;
                }
            }
        }
    }

    peqs
}

/// Generate alpha mask for overhang handling.
fn generate_alpha_mask(alpha: f32, length: usize) -> u64 {
    let mut mask = 0u64;
    let limit = length.min(64);

    for i in 0..limit {
        let val = ((i + 1) as f32 * alpha).floor() as u64 - (i as f32 * alpha).floor() as u64;
        if val >= 1 {
            mask |= 1u64 << i;
        }
    }

    mask
}

#[cfg(test)]
mod max_hits_tests {
    use super::next_hit_capacity;

    #[test]
    fn detects_overflow_and_requests_true_count() {
        assert_eq!(next_hit_capacity(5, 10), None); // fits: no retry
        assert_eq!(next_hit_capacity(10, 10), None); // exactly full: no drop
        assert_eq!(next_hit_capacity(23, 10), Some(23)); // overflow: grow to 23
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::Iupac;

    #[test]
    #[ignore] // Only run when a WGPU-compatible GPU/driver is available.
    fn test_wgpu_search_basic() {
        let searcher = MyersWgpu::new();

        let queries = vec![b"ACGT".to_vec(), b"TGCA".to_vec()];
        let text = b"AAACGTTTGCAAA";
        let k = 0;

        let hits = searcher
            .search_ranges::<Iupac>(&queries, text, k, None)
            .expect("Search failed");

        assert_eq!(hits.len(), 2, "Should find 2 exact matches");
    }
}
