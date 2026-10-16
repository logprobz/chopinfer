//! Builds the slice-widened latent attention entries (16 local heads, latent
//! 512, 64-token pages, 2051-entry key lists, FP8 pages) and each lab
//! mutant, and checks the audits and the SASS: no spills or local memory,
//! registers and shared memory for two CTAs per SM, BF16 MMA on both
//! ldmatrix forms, the exact FP8 widening, cp.async staging, and the PDL
//! release and wait.

use kernel_build::{BuildOptions, Toolchain, build};
use kernels_sm120::latent::slices::{self, lab_cubin_specs};
use kernels_sm120::latent::{Geometry, entries};
use kernels_sm120::{Profile, ProfileNames};
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "latent-slices-build",
    tag: "latent_slices_build",
});

const SERVED: Geometry = Geometry {
    heads: 16,
    latent: 512,
    page_size: 64,
    max_keys: 2051,
    softmax_scale: 0.0625,
    nope: 256,
    v_dim: 256,
};

/// The SASS of one function, from `cuobjdump -sass` output.
fn function_sass<'a>(sass: &'a str, symbol: &str) -> &'a str {
    let header = format!("Function : {symbol}\n");
    let start = sass
        .find(&header)
        .unwrap_or_else(|| panic!("{symbol} has no SASS"));
    let body = &sass[start + header.len()..];
    &body[..body.find("Function : ").unwrap_or(body.len())]
}

#[test]
fn slice_entries_build_for_two_ctas_per_sm() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "chopinfer-latent-slices-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir_all(&root).expect("create test output directory");
    let catalog = slices::entries(&entries(PROFILE, SERVED));
    assert_eq!(catalog.len(), 2);
    let specs = lab_cubin_specs(PROFILE, &catalog).expect("slice entries are valid");
    let cubins = build(
        &toolchain,
        &specs,
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("slice cubins build and audit");
    assert_eq!(cubins.len(), 1 + slices::MUTANTS.len());
    for cubin in &cubins {
        assert_eq!(cubin.entries.len(), 2);
        for entry in &cubin.entries {
            eprintln!(
                "{}: {} registers, {} static shared bytes",
                entry.symbol, entry.registers, entry.shared_static_bytes
            );
            assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
            assert_eq!(entry.stack_bytes, 0, "{} uses a stack", entry.symbol);
            // Two CTAs of 128 threads share the 64K-register file and the
            // 100 KiB of shared memory, with the 1 KiB ptxas reports as
            // static and the 1 KiB the driver reserves per CTA.
            assert!(entry.registers <= 256, "{}", entry.symbol);
            assert!(
                2 * (entry.shared_static_bytes + catalog[0].dynamic_shared_bytes() + 1024)
                    <= 102_400,
                "{}",
                entry.symbol
            );
        }
    }
    let served = cubins
        .iter()
        .find(|cubin| cubin.name == slices::cubin_name(PROFILE, ""))
        .expect("the served cubin");
    let output = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&served.path)
        .output()
        .expect("cuobjdump SASS");
    assert!(output.status.success(), "cuobjdump failed");
    let sass = String::from_utf8(output.stdout).expect("SASS is UTF-8");
    for entry in &catalog {
        let symbol = entry.symbol();
        let code = function_sass(&sass, &symbol);
        for opcode in [
            "HMMA.16816.F32.BF16",
            "LDGSTS",
            "LDSM.16.M88.4",
            "LDSM.16.MT88.4",
            "F2FP.BF16.E4M3.UNPACK_B",
            "PREEXIT",
            "ACQBULK",
        ] {
            assert!(code.contains(opcode), "{symbol} lacks {opcode}");
        }
        for opcode in ["LDL", "STL", "TF32"] {
            assert!(!code.contains(opcode), "{symbol} uses {opcode}");
        }
    }
    fs::remove_dir_all(root).expect("remove hermetic test output");
}
