"""A compiled, weightless release fixture using the production producer."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

KERNEL_BUILD = r'''
use std::{env, fs, path::{Path, PathBuf}};
use kernel_build::{BuildOptions, CubinSpec, EntryParam, EntrySpec, LockDir, Toolchain, build, build_in_families};
fn spec(root: &Path, name: &str) -> CubinSpec {
    CubinSpec { name: name.into(), sources_root: root.join("cuda"), include_dirs: vec![PathBuf::from(".")],
        headers: vec!["#include \"value.cuh\"".into()], defines: vec![],
        entries: vec![EntrySpec::new(format!("{name}_v1"), vec![EntryParam::DevicePointer {
            name: "out".into(), cxx_type: "int".into() }], "out[threadIdx.x] = fixture_value();".into(), 32, 1)] }
}
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let toolkit = Toolchain::detect().unwrap();
    let mut options = BuildOptions::from_env();
    if env::var_os("FIXTURE_CREATE_LOCK").is_some() { options.lock_update = true; }
    else {
        for name in ["KERNELS_LOCK_UPDATE", "CHOPINFER_KERNEL_LINEINFO", "CHOPINFER_MODEL_RELOCK_DIR",
                     "EXTERNAL_MODEL_CREATE_KERNEL_LOCK", "EXTERNAL_MODEL_CREATE_PRECISION_LOCK",
                     "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTC_BOOTSTRAP"] {
            assert_eq!(env::var(name).unwrap(), "", "bypass was not forced empty: {name}");
        }
    }
    if env!("CARGO_PKG_NAME") == "runtime-kernels" {
        for (name, lock, directory) in [("sampling_sm120", "kernels.lock", "sampling"),
                                        ("sampling_mapped_sm120", "sampling-mapped.lock", "mapped-sampling")] {
            build(&toolkit, &[spec(&root, name)], &out.join(directory), &root.join(lock), options).unwrap();
        }
    } else {
        build(&toolkit, &[spec(&root, "fixture_main")], &out.join("main"), &root.join("kernels.lock"), options).unwrap();
        fs::create_dir_all(root.join("kernels.lock.d")).unwrap();
        build_in_families(&toolkit, &[spec(&root, "fixture_child")], &out.join("child"),
            &LockDir::new(root.join("kernels.lock.d")).family("child", ["fixture_child"]), options, None).unwrap();
    }
}
'''

HOST = r'''
#[link(name = "libcuda.so.1", kind = "dylib", modifiers = "+verbatim")]
unsafe extern "C" { fn cuInit(flags: u32) -> i32; }
fn main() {
    // Retain the CUDA SONAME dependency without calling the driver.
    std::hint::black_box(cuInit as unsafe extern "C" fn(u32) -> i32);
    std::hint::black_box(runtime_kernels::proof());
    let args: Vec<String> = std::env::args().collect();
    if args[1..] == ["--version"] {
        let name = std::path::Path::new(&args[0]).file_name().unwrap().to_str().unwrap();
        println!("{} {} ({})", name, env!("CARGO_PKG_VERSION"), env!("CHOPINFER_BUILD_REVISION"));
    } else if args.get(1).map(String::as_str) == Some("eval") {
        println!("models/fixture/checkpoint-files.json");
        println!("tools/qual/modules.json");
    } else { std::process::exit(2); }
}
'''


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def create(directory):
    """Compile initial tiny locks separately. The producer must verify them fresh."""
    directory = Path(directory)
    root = directory / 'source'
    root.mkdir()
    members = ['crates/kernel-build', 'crates/runtime-host', 'crates/runtime-kernels', 'crates/choptop',
               'models/fixture/bundle', 'models/fixture/compiler']
    dependencies = '\n'.join(line for line in (ROOT / 'Cargo.toml').read_text().splitlines()
                             if re.match(r'^(object|serde|serde_json|sha2) = ', line))
    write(root, 'Cargo.toml', '[workspace]\nresolver = "3"\nmembers = ' + json.dumps(members) + '\n'
          '[workspace.package]\nversion = "0.1.0-alpha.1"\nedition = "2024"\nrust-version = "1.89"\n'
          'publish = false\nlicense = "Apache-2.0"\n[workspace.dependencies]\n' + dependencies + '\n'
          'kernel-build = { path = "crates/kernel-build" }\n[workspace.lints.rust]\nunsafe_code = "allow"\n')
    shutil.copytree(ROOT / 'crates/kernel-build', root / 'crates/kernel-build', ignore=shutil.ignore_patterns('target', '__pycache__'))
    for source in ('tools/release', 'tools/model_sdk'):
        destination = root / source
        destination.mkdir(parents=True)
        for path in (ROOT / source).iterdir():
            if path.suffix in ('.py', '.sh', '.md'):
                shutil.copyfile(path, destination / path.name)
    write(root, 'ci/kernel_reports.py', (ROOT / 'ci/kernel_reports.py').read_text())
    for name in ('rust-toolchain.toml', 'LICENSE', 'THIRD_PARTY_NOTICES', 'fixtures/GSM8K-LICENSE'):
        write(root, name, (ROOT / name).read_text())
    write(root, '.git/HEAD', 'a' * 40 + '\n')
    for package, path, extra in [
        ('runtime-host', 'crates/runtime-host', '[features]\ndefault = ["fake-host-services"]\nfake-host-services = []\nlinux-cuda-host = []\n'
         '[dependencies]\nruntime-kernels = { path = "../runtime-kernels" }\n'
         '[[bin]]\nname = "chopinfer"\npath = "src/main.rs"\n[[bin]]\nname = "chopinfer-host"\npath = "src/main.rs"\n'),
        ('runtime-kernels', 'crates/runtime-kernels', '[build-dependencies]\nkernel-build.workspace = true\n'),
        ('choptop', 'crates/choptop', ''),
        ('bundle-fixture', 'models/fixture/bundle', '[lib]\ncrate-type = ["rlib", "cdylib"]\n[build-dependencies]\nkernel-build.workspace = true\n'),
        ('model-fixture', 'models/fixture/compiler', '[[bin]]\nname = "fixture-package"\npath = "src/main.rs"\n'
         '[[bin]]\nname = "chopinfer-model-fixture"\npath = "src/main.rs"\n')]:
        write(root, path + '/Cargo.toml', f'[package]\nname = "{package}"\nversion.workspace = true\nedition.workspace = true\n'
              'license.workspace = true\n' + extra)
    write(root, 'crates/runtime-host/src/main.rs', HOST)
    write(root, 'crates/choptop/src/main.rs', 'fn main() { println!("choptop {}", env!("CARGO_PKG_VERSION")); }\n')
    write(root, 'models/fixture/compiler/src/main.rs', 'fn main() { println!("fixture helper"); }\n')
    for path in ('crates/runtime-kernels', 'models/fixture/bundle'):
        write(root, path + '/build.rs', KERNEL_BUILD)
        write(root, path + '/cuda/value.cuh', '__device__ inline int fixture_value() { return 7; }\n')
    write(root, 'crates/runtime-kernels/src/lib.rs', r'''
pub static ORDINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sampling/sampling_sm120.cubin"));
pub static MAPPED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mapped-sampling/sampling_mapped_sm120.cubin"));
pub fn proof() -> u8 { std::hint::black_box(ORDINARY)[0] ^ std::hint::black_box(MAPPED)[0] }
''')
    write(root, 'models/fixture/bundle/src/lib.rs', r'''
static MAIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/main/fixture_main.cubin"));
static CHILD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/child/fixture_child.cubin"));
#[unsafe(no_mangle)]
pub extern "C" fn fixture_value() -> u8 { std::hint::black_box(MAIN)[0] ^ std::hint::black_box(CHILD)[0] }
''')
    write(root, 'models/fixture/model.toml', 'schema = "chopinfer-model-inventory-v1"\n'
          '[bundle]\npackage = "bundle-fixture"\ncdylib = ["bundle_fixture"]\n'
          '[compiler]\npackage = "model-fixture"\nbin = ["fixture-package", "chopinfer-model-fixture"]\n'
          '[release]\nenabled = true\nbins = ["fixture-package", "chopinfer-model-fixture"]\ncdylibs = ["bundle_fixture"]\n')
    write(root, 'models/fixture/checkpoint-files.json', '{}\n')
    write(root, 'models/fixture/profiles.json', '{}\n')
    write(root, 'tools/qual/modules.json', '{}\n')
    env = {key: value for key, value in os.environ.items() if key not in ('CARGO_ENCODED_RUSTFLAGS', 'RUSTFLAGS')}
    env.update(FIXTURE_CREATE_LOCK='1', CARGO_TARGET_DIR=str(directory / 'initial-target'), CHOPINFER_BUILD_REVISION='a' * 40)
    subprocess.run(['cargo', 'generate-lockfile', '--offline'], cwd=root, env=env, check=True, capture_output=True)
    result = subprocess.run(['cargo', 'build', '--locked', '--offline', '-p', 'runtime-kernels', '-p', 'bundle-fixture'],
                            cwd=root, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    bypasses = ('KERNELS_LOCK_UPDATE', 'CHOPINFER_KERNEL_LINEINFO', 'CHOPINFER_MODEL_RELOCK_DIR',
                'EXTERNAL_MODEL_CREATE_KERNEL_LOCK', 'EXTERNAL_MODEL_CREATE_PRECISION_LOCK')
    write(root, '.cargo/config.toml', '[env]\n' + ''.join(
        f'{name} = {{ value = "1", force = true, relative = true }}\n' for name in bypasses)
        + 'CHOPINFER_BUILD_REVISION = { value = "' + 'b' * 40 + '", force = true }\n'
        + 'RUSTC_WRAPPER = { value = "untrusted-wrapper", force = true, relative = true }\n')
    return root
