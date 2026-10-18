//! The vision encoder's component conformance, through the shared encoder
//! runner (`kernel_harness::encoder`): every catalog entry the declaration
//! runs, from the encoder cubins rebuilt against the bundle's
//! `kernels.lock.d`, checked against its component oracle with poisoned
//! outputs. The patch projection (the first GEMM, over the payload padded
//! from 588 to 640 columns, also at full capacity) and the 3 x 3
//! space-to-depth merge over odd grids run first, as their own tests; the
//! BF16 cast after the aligner's last GEMM, which writes F32, is one too.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_harness::encoder::{self as conformance, Case, Runner};
use kernels_sm120::encoder::{ATTENTION_CUBIN, ROWS_GEMM_CUBIN, Shared};
use model_deepseek41_contract::encoder::{FAMILY, declarations};

use crate::encoder_catalog::{cubins, kernels};

/// The encoder cubins, rebuilt from the bundle's catalog and verified
/// against its lock family (an ordinary build: a mismatch fails).
fn cubin(name: &str) -> Option<&'static [u8]> {
    static BUILT: OnceLock<BTreeMap<String, Vec<u8>>> = OnceLock::new();
    BUILT
        .get_or_init(|| {
            let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bundle");
            let bundle = std::fs::canonicalize(bundle).expect("bundle directory");
            let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("deepseek41-encoder");
            std::fs::create_dir_all(&out).expect("cubin output directory");
            let lock = LockDir::new(bundle.join("kernels.lock.d"))
                .family(FAMILY, [ROWS_GEMM_CUBIN, ATTENTION_CUBIN]);
            let specs: Vec<_> = cubins()
                .expect("encoder cubin specs")
                .into_iter()
                .map(|(_, spec)| spec)
                .collect();
            let built = kernel_build::build_in_families(
                &Toolchain::detect().expect("CUDA 13.2 toolchain"),
                &specs,
                &out,
                &lock,
                BuildOptions {
                    lock_update: false,
                    emit_cargo_directives: false,
                    profile_line_info: false,
                },
                None,
            )
            .expect("the encoder cubins match their lock");
            built
                .iter()
                .map(|cubin| {
                    let bytes = std::fs::read(&cubin.path).expect("built cubin");
                    (cubin.name.clone(), bytes)
                })
                .collect()
        })
        .get(name)
        .map(Vec::as_slice)
}

/// Each variant the declaration runs, with its runner.
const RUNNERS: &[(&str, Runner)] = &[
    ("pad.zero-columns.v1", conformance::rows),
    ("linear.pairs.v1", conformance::gemm),
    ("norm.rms-f32.v1", conformance::norm),
    ("attention.varlen.pairs.v1", conformance::attention),
    ("rotary.axial-2d.v1", conformance::rotary),
    ("mlp.swiglu.v1", conformance::rows),
    ("merge.space-to-depth.v1", conformance::rows),
    ("layout.learned-delimiters.v1", conformance::layout),
    ("cast.bf16.v1", conformance::rows),
];

fn cases() -> Vec<Case> {
    let [vision] = declarations().expect("the vision declaration");
    conformance::cases(&[(vision, &Shared)], kernels().unwrap(), cubin).unwrap()
}

fn run(id: &str) {
    conformance::run(id, &cases(), RUNNERS).expect(id);
}

/// Every case runs under a test below, and every runner has a case. CPU.
#[test]
fn every_entry_has_a_runner() {
    let cases = cases();
    let variants: BTreeSet<&str> = cases.iter().map(|case| case.variant.as_str()).collect();
    let runners: BTreeSet<&str> = RUNNERS.iter().map(|(id, _)| *id).collect();
    assert_eq!(variants, runners);
    let inputs: Vec<&str> = cases
        .iter()
        .filter(|case| case.input_linear)
        .map(|case| case.entry.symbol.as_str())
        .collect();
    assert_eq!(inputs.len(), 1, "{inputs:?}");
    assert!(inputs[0].starts_with("chop_enc_linear_pairs_n1024_k640_bias_f32_"));
}

/// The rebuilt cubins match the lock and hold every catalog entry. CPU
/// (nvcc only).
#[test]
fn the_cubins_rebuild_to_their_lock() {
    for entry in kernels().unwrap() {
        let name = if entry.is_attention() {
            ATTENTION_CUBIN
        } else {
            ROWS_GEMM_CUBIN
        };
        let bytes = cubin(name).expect("built cubin");
        assert!(
            bytes
                .windows(entry.symbol.len())
                .any(|window| window == entry.symbol.as_bytes()),
            "{name} lacks {}",
            entry.symbol
        );
    }
}

/// The patch projection alone: K 640 over the padded payload, at 300 rows
/// and at the full 9216-row capacity.
#[test]
fn patch_projection() {
    let cases: Vec<Case> = cases()
        .into_iter()
        .filter(|case| case.input_linear)
        .collect();
    conformance::run("linear.pairs.v1", &cases, RUNNERS).expect("patch projection");
}

/// The 3 x 3 space-to-depth merge over 7 x 5, 31 x 51 and 1 x 1 patch
/// grids, whose last cells run past the grid.
#[test]
fn space_to_depth_odd_grids() {
    run("merge.space-to-depth.v1");
}

#[test]
fn pad_columns() {
    run("pad.zero-columns.v1");
}

#[test]
fn linear_pairs() {
    run("linear.pairs.v1");
}

#[test]
fn norm_rms() {
    run("norm.rms-f32.v1");
}

#[test]
fn attention_varlen() {
    run("attention.varlen.pairs.v1");
}

#[test]
fn rotary_axial() {
    run("rotary.axial-2d.v1");
}

#[test]
fn mlp_swiglu() {
    run("mlp.swiglu.v1");
}

#[test]
fn layout_learned_delimiters() {
    run("layout.learned-delimiters.v1");
}

/// The aligner's F32 rows rounded once to BF16, bit for bit, before the
/// arena placement copies them.
#[test]
fn cast_bf16() {
    run("cast.bf16.v1");
}
