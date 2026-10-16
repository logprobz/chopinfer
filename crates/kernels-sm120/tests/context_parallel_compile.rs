//! Builds the context-parallel entries at the served latent shape (16 heads
//! merged per rank, latent width 512, 64-token pages, key lists of 2051) and
//! checks the audits and the SASS.
use kernel_build::{BuildOptions, Toolchain, build};
use kernels_sm120::context_parallel::{Geometry, cubin_spec, entries};
use kernels_sm120::{Profile, ProfileNames};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "context-parallel-compile",
    tag: "context_parallel_compile",
});

#[test]
fn context_parallel_entries_compile_without_spills() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into())).join(
        format!("context-parallel-cubin-{}-{stamp}", std::process::id()),
    );
    fs::create_dir_all(&root).unwrap();
    let geometry = Geometry {
        heads: 16,
        width: 512,
        page_size: 64,
        list_width: 2051,
        capacity: 8192,
    };
    let spec = cubin_spec(PROFILE, &entries(PROFILE, geometry)).unwrap();
    let cubins = build(
        &toolchain,
        &[spec],
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("context-parallel cubin builds and audits");
    assert_eq!(cubins.len(), 1);
    assert_eq!(cubins[0].entries.len(), 6);
    assert!(cubins[0].entries.iter().all(|entry| entry.local_bytes == 0));
    for entry in &cubins[0].entries {
        eprintln!(
            "{}: {} registers, {} static shared bytes",
            entry.symbol, entry.registers, entry.shared_static_bytes
        );
    }
    let sass = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&cubins[0].path)
        .output()
        .expect("inspect context-parallel SASS");
    assert!(sass.status.success());
    let text = String::from_utf8_lossy(&sass.stdout);
    fs::write(root.join("sass.txt"), text.as_bytes()).unwrap();
    // exp2 on the MUFU, FMA folds, the BF16 conversion and the PDL pair.
    for opcode in ["MUFU.EX2", "FFMA", "F2FP.BF16", "ACQBULK", "PREEXIT"] {
        assert!(text.contains(opcode), "the SASS has no {opcode}");
    }
    // No float atomics anywhere.
    assert!(!text.contains("RED.") && !text.contains("ATOM"));
    // Every entry waits on its producer (ACQBULK, griddepcontrol.wait)
    // before its first global load, the live row count among them.
    let functions = text.split("Function : ").skip(1).collect::<Vec<_>>();
    assert_eq!(functions.len(), 6);
    for function in functions {
        let name = function.lines().next().unwrap_or_default();
        let wait = function
            .find("ACQBULK")
            .unwrap_or_else(|| panic!("{name} has no dependency wait"));
        let load = function
            .find("LDG")
            .unwrap_or_else(|| panic!("{name} has no global load"));
        assert!(
            wait < load,
            "{name} loads from global memory before its wait"
        );
    }
    fs::remove_dir_all(root).unwrap();
}
