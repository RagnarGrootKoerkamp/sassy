//! Generates `python/sassy/__init__.pyi` using pyo3-introspection.
//!
//! First build the extension. On macOS, `-undefined dynamic_lookup` is required to
//! defer Python symbol resolution to load time (Linux does not need this):
//!
//!   # macOS
//!   RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" cargo build --features python
//!   # Linux
//!   cargo build --features python
//!
//! Then run from the repo root, passing the path to the compiled extension:
//!   cargo run --features python-stubs --bin gen_stubs -- target/debug/libsassy.dylib   # macOS
//!   cargo run --features python-stubs --bin gen_stubs -- target/debug/libsassy.so      # Linux

use std::path::{Path, PathBuf};

fn main() {
    let lib_path: PathBuf = std::env::args()
        .nth(1)
        .expect("Usage: gen_stubs <path_to_compiled_extension>")
        .into();

    let module = pyo3_introspection::introspect_cdylib(&lib_path, "sassy")
        .unwrap_or_else(|e| panic!("Failed to introspect {}: {e}", lib_path.display()));

    let stubs = pyo3_introspection::module_stub_files(&module);

    for (rel_path, content) in &stubs {
        let output = Path::new("python/sassy").join(rel_path);
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("Failed to create {}: {e}", parent.display()));
        }
        std::fs::write(&output, content)
            .unwrap_or_else(|e| panic!("Failed to write {}: {e}", output.display()));
        eprintln!("Wrote {}", output.display());
    }
}
