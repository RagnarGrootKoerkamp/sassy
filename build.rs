// Build script for compiling GPU kernels
//
// - CUDA (feature "cuda"): PTX compilation is currently done manually, see
//   CUDA_PTX_COMPILATION_ISSUE.md for details on why automated compilation is blocked.
// - WGPU (feature "wgpu"): compiles the `gpu_kernel` crate to SPIR-V via spirv-builder,
//   following the pattern used in rust-gpu-chimera.

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda_kernel();

    #[cfg(feature = "wgpu")]
    build_spirv_kernel();
}

#[cfg(feature = "cuda")]
fn build_cuda_kernel() {
    // Check if pre-compiled PTX exists
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = std::path::Path::new(&out_dir).join("kernels.ptx");

    if !ptx_path.exists() {
        println!("cargo:warning=CUDA feature enabled but kernels.ptx not found.");
        println!("cargo:warning=To compile CUDA kernels, manually run:");
        println!(
            "cargo:warning=  cd kernels && cargo rustc --release --target nvptx64-nvidia-cuda -- --emit=asm"
        );
        println!(
            "cargo:warning=Then copy the PTX file to: {}",
            ptx_path.display()
        );
    }

    // Tell cargo to rerun if kernel source changes
    println!("cargo:rerun-if-changed=kernels/src/lib.rs");
}

#[cfg(feature = "wgpu")]
fn build_spirv_kernel() {
    use spirv_builder::{Capability, SpirvBuilder};
    use std::path::PathBuf;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_path = PathBuf::from(manifest_dir).join("gpu_kernel");

    println!("cargo:rerun-if-changed=gpu_kernel/src/lib.rs");
    println!("cargo:rerun-if-changed=gpu_kernel/Cargo.toml");

    let result = SpirvBuilder::new(crate_path, "spirv-unknown-vulkan1.2")
        .capability(Capability::Int8)
        .capability(Capability::Int64)
        .spirv_metadata(spirv_builder::SpirvMetadata::Full)
        .build()
        .expect("Failed to build gpu_kernel to SPIR-V");

    // Export the compiled SPIR-V module path so the runtime can embed/load it.
    println!(
        "cargo:rustc-env=MYERS_KERNEL_SPV_PATH={}",
        result.module.unwrap_single().display()
    );

    // Export the kernel entry point name.
    println!(
        "cargo:rustc-env=MYERS_KERNEL_SPV_ENTRY={}",
        result
            .entry_points
            .iter()
            .find(|e| e.contains("myers_search_kernel"))
            .cloned()
            .unwrap_or_else(|| result.entry_points.first().unwrap().clone())
    );
}
