//! Builds the one-split selection forms (`pooled_indexer::rows`) of the
//! served indexer shape (32 heads of 128, pools of 4, top 2048 with the
//! tail, 64-token pages), global and per rank, one and two rows per CTA,
//! and each lab mutant, and checks the audits and the SASS: no spills or
//! local memory, registers and shared memory for the form's CTAs per SM,
//! BF16 MMA on the ldmatrix key tiles, cp.async staging, and the PDL
//! release and wait.

use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::contract::DType;
use kernels_sm120::pooled_indexer::rows::{self, lab_cubin_specs};
use kernels_sm120::pooled_indexer::{Geometry, entries};
use kernels_sm120::{Profile, ProfileNames};
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "pooled-rows-build",
    tag: "pooled_rows_build",
});

const SERVED: Geometry = Geometry {
    heads: 32,
    head_dim: 128,
    group: 4,
    page_size: 64,
    max_top_k: 2048,
    always_tail: true,
    position_dtype: DType::Bf16,
    key_norm_eps: 1e-6,
    verify_rows: 6,
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

/// Builds every form and lab mutant of `geometry`'s selections into a fresh
/// directory and audits them for spills, stacks, registers and shared
/// memory; returns the toolchain, the directory and the built cubins.
fn build_and_audit(
    geometry: Geometry,
) -> (
    Toolchain,
    std::path::PathBuf,
    Vec<rows::Entry>,
    Vec<kernel_build::BuiltCubin>,
) {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "chopinfer-pooled-rows-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir_all(&root).expect("create test output directory");
    let selects = entries(PROFILE, geometry);
    let catalog = rows::FORMS
        .iter()
        .flat_map(|&form| rows::entries(&selects, form))
        .collect::<Vec<_>>();
    assert_eq!(catalog.len(), 6);
    let specs = lab_cubin_specs(PROFILE, &catalog).expect("one-split entries are valid");
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
    .expect("one-split cubins build and audit");
    assert_eq!(cubins.len(), 2 + rows::MUTANTS.len());
    for cubin in &cubins {
        for built in &cubin.entries {
            let entry = catalog
                .iter()
                .find(|entry| built.symbol.starts_with(&entry.symbol().replace("_v1", "")))
                .unwrap_or_else(|| panic!("{} is no catalog form", built.symbol));
            eprintln!(
                "{}: {} registers, {} static shared bytes",
                built.symbol, built.registers, built.shared_static_bytes
            );
            assert_eq!(built.local_bytes, 0, "{} spills", built.symbol);
            assert_eq!(built.stack_bytes, 0, "{} uses a stack", built.symbol);
            // The form's CTAs share the 64K-register file and the 100 KiB of
            // shared memory, with the 1 KiB the driver reserves per CTA.
            let ctas = entry.ctas_per_sm();
            assert!(
                built.registers * entry.threads() * ctas <= 65_536,
                "{}",
                built.symbol
            );
            assert!(
                ctas * (built.shared_static_bytes + entry.dynamic_shared_bytes() + 1024) <= 102_400,
                "{}",
                built.symbol
            );
        }
    }
    (toolchain, root, catalog, cubins)
}

#[test]
fn one_split_forms_build_for_their_ctas_per_sm() {
    let (toolchain, root, catalog, cubins) = build_and_audit(SERVED);
    let served = cubins
        .iter()
        .find(|cubin| cubin.name == rows::cubin_name(PROFILE, ""))
        .expect("the served cubin");
    assert_eq!(served.entries.len(), 6);
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

/// The smallest admitted maximum (128 kept pools: a trim's buffer of 256
/// keys, one per row thread) and pools of 2 (whose rows past 512K tokens
/// sort their lists) build and audit as the served shape does.
#[test]
fn the_smallest_maximum_and_pools_of_two_build() {
    for geometry in [
        Geometry {
            max_top_k: 512,
            ..SERVED
        },
        Geometry {
            group: 2,
            max_top_k: 1024,
            always_tail: false,
            ..SERVED
        },
    ] {
        let (_, root, catalog, cubins) = build_and_audit(geometry);
        assert_eq!(catalog.len(), 6);
        assert_eq!(cubins.len(), 2 + rows::MUTANTS.len());
        fs::remove_dir_all(root).expect("remove hermetic test output");
    }
}
