// Build script for compiling CUDA kernels
// 
// This script uses cuda_builder to compile the kernels crate to PTX
// and makes the resulting PTX file available to the main crate at compile time.
//
// NOTE: Due to dependency conflicts with lzma-sys, automated PTX compilation
// is currently disabled. To use CUDA:
//
// 1. Manually compile kernels to PTX:
//    cd kernels
//    cargo rustc --release --target nvptx64-nvidia-cuda -- --emit=asm
//    cp target/nvptx64-nvidia-cuda/release/*.ptx ../target/kernels.ptx
//
// 2. Or use a separate build environment with rust-cuda

#[cfg(feature = "cuda")]
fn main() {
    // Check if pre-compiled PTX exists
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = std::path::Path::new(&out_dir).join("kernels.ptx");
    
    if !ptx_path.exists() {
        println!("cargo:warning=CUDA feature enabled but kernels.ptx not found.");
        println!("cargo:warning=To compile CUDA kernels, manually run:");
        println!("cargo:warning=  cd kernels && cargo rustc --release --target nvptx64-nvidia-cuda -- --emit=asm");
        println!("cargo:warning=Then copy the PTX file to: {}", ptx_path.display());
    }
    
    // Tell cargo to rerun if kernel source changes
    println!("cargo:rerun-if-changed=kernels/src/lib.rs");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    // No-op when CUDA feature is not enabled
}
