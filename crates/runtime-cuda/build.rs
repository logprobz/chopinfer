use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CHOPINFER_CUDA_HOME");
    let cuda_home = env::var_os("CHOPINFER_CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let include = cuda_home.join("targets/x86_64-linux/include");
    let cuda_header = include.join("cuda.h");
    let cuda_stub = cuda_home.join("targets/x86_64-linux/lib/stubs/libcuda.so");
    // The no-device stub stands in for the driver itself, so a build that asks
    // for it links that stub and needs no installed driver on any platform.
    // Linux is where the device-free CI runs these tests, and there the real
    // driver link would leave the stub's own controls undefined.
    if env::var_os("CARGO_FEATURE_FAKE_DRIVER_TESTS").is_some() {
        build_no_gpu_link_stub();
        return;
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")
        .expect("Cargo did not provide the target architecture to runtime-cuda");
    assert_eq!(
        target_arch, "x86_64",
        "runtime-cuda supports only Linux x86_64 CUDA targets; got Linux {target_arch}"
    );

    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let binding_check = manifest.join("bindings/verify-bindings.sh");
    let status = Command::new("bash")
        .arg(&binding_check)
        .arg(&cuda_header)
        .current_dir(&manifest)
        .status()
        .expect("cannot execute the checked-in CUDA binding provenance check");
    assert!(
        status.success(),
        "checked-in CUDA binding differs from its pinned header provenance"
    );

    let nvdec_check = manifest.join("bindings/nvdec/verify-headers.sh");
    let status = Command::new("bash")
        .arg(&nvdec_check)
        .status()
        .expect("cannot execute the vendored NVDEC header provenance check");
    assert!(
        status.success(),
        "vendored NVDEC headers differ from their pinned provenance"
    );

    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let helper_object = output.join("tensor_map.o");
    let helper_library = output.join("libruntime_cuda_tensor_map.a");
    let helper_source = manifest.join("csrc/tensor_map.c");
    let helper_status = Command::new("/usr/bin/gcc")
        .args(["-std=c11", "-O2", "-fPIC", "-c"])
        .arg(&helper_source)
        .arg("-I")
        .arg(&include)
        .arg("-o")
        .arg(&helper_object)
        .status()
        .expect("cannot compile the CUDA tensor-map helper");
    assert!(
        helper_status.success(),
        "CUDA tensor-map helper compilation failed"
    );
    // The NVDEC helper resolves libnvcuvid at run time, so the build links
    // nothing of it; the vendored headers only supply its declarations.
    let nvdec_object = output.join("nvdec.o");
    let nvdec_status = Command::new("/usr/bin/gcc")
        .args(["-std=c11", "-O2", "-fPIC", "-c"])
        .arg(manifest.join("csrc/nvdec.c"))
        .arg("-I")
        .arg(&include)
        .arg(format!("-I{}", manifest.join("bindings/nvdec").display()))
        .arg("-o")
        .arg(&nvdec_object)
        .status()
        .expect("cannot compile the NVDEC helper");
    assert!(nvdec_status.success(), "NVDEC helper compilation failed");
    let archive_status = Command::new("/usr/bin/ar")
        .arg("crs")
        .arg(&helper_library)
        .arg(&helper_object)
        .arg(&nvdec_object)
        .status()
        .expect("cannot archive the CUDA tensor-map helper");
    assert!(
        archive_status.success(),
        "CUDA tensor-map helper archive failed"
    );
    // Stage the stub under the link-time name only. Cargo puts this directory
    // on the dynamic library path of `cargo run` and `cargo test`, and a stub
    // named libcuda.so.1 there would shadow the real driver at load time.
    fs::copy(&cuda_stub, output.join("libcuda.so"))
        .expect("cannot stage the NVIDIA Driver API link stub");
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=runtime_cuda_tensor_map");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rerun-if-changed=csrc/tensor_map.c");
    println!("cargo:rerun-if-changed=csrc/tensor_map.h");
    println!("cargo:rerun-if-changed=csrc/nvdec.c");
    println!("cargo:rerun-if-changed=csrc/nvdec.h");
    println!("cargo:rerun-if-changed=bindings/nvdec");
    println!("cargo:rerun-if-changed=bindings/PROVENANCE.md");
    println!("cargo:rerun-if-changed=bindings/allowlist-functions.txt");
    println!("cargo:rerun-if-changed=bindings/cuda-driver-13.2.rs.inc");
    println!("cargo:rerun-if-changed=bindings/cuda-driver-wrapper.h");
    println!("cargo:rerun-if-changed=bindings/regenerate-bindings.sh");
    println!("cargo:rerun-if-changed=bindings/verify-bindings.sh");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed={}", cuda_header.display());
    println!("cargo:rerun-if-changed={}", cuda_stub.display());
}

fn build_no_gpu_link_stub() {
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = manifest.join("csrc/no_gpu_cuda_stub.c");
    let object = output.join("no_gpu_cuda_stub.o");
    let archive = output.join("libruntime_cuda_no_gpu.a");
    let compile = Command::new("/usr/bin/cc")
        .args(["-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("cannot compile the no-GPU CUDA link stub");
    assert!(
        compile.success(),
        "no-GPU CUDA link stub compilation failed"
    );
    let status = Command::new("/usr/bin/ar")
        .args(["crs"])
        .arg(&archive)
        .arg(&object)
        .status()
        .expect("cannot create the no-GPU CUDA link stub");
    assert!(status.success(), "no-GPU CUDA link stub creation failed");
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=runtime_cuda_no_gpu");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rerun-if-changed=csrc/no_gpu_cuda_stub.c");
}
