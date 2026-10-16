//! Lab-only instances of the strip-chain stage-release race (fixed 2026-10-09),
//! shared by the `strip_chain_race` lock family and
//! `tests/strip_chain_race.rs`.
//!
//! `run_a16_strips`, `run_a16_strips_fp8` (strip_chains.cuh,
//! strip_chains_fp8.cuh) and `run_a16_warps` (tiled_warps.cuh) release a
//! TMA ring stage right after issuing the step's shared reads of it, and
//! ptxas issues that release before the last reads return, so the producer
//! can refill the stage while a math warp still reads it. Served entries now
//! run the `_ordered` kernels (strip_chains_ordered.cuh,
//! strip_chains_fp8_ordered.cuh, tiled_warps_ordered.cuh), which fence those
//! reads first.
//!
//! Each case is one chain schedule over one shape and bucket with three
//! entries: the one-split `Tiled` entry whose bits every chain schedule
//! keeps (`one`), the schedule on the locked kernel (`locked`: the generated
//! body calling the locked function) and on the served one (`ordered`, the
//! generated body unchanged). The cases are the vocabulary head's two-split
//! strip chains (BF16 38720 x 4096, F32 out, 4 strips, 2 splits, 1210 CTAs;
//! PERF-B14b saw them differ) at 4, 16 and 32 rows, every multi-split strip
//! chain a served model's contract schedules (its BF16 and FP8 strip
//! schedules) at the widest bucket it serves it at, and two
//! prefill warp-tile entries (`TiledChains` at one split over 2048 rows,
//! BF16 and FP8 128x128 weights). One cubin holds every case; a second
//! holds the contention probe (`kernel-lab/cuda/probes/strip_race.cuh`).

use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernel_core::contract::{DType, DenseKernel, GemmQuery, InstantiationSpec, Target};
use kernels_sm120::gemm::{self, DenseRequest};
use std::path::PathBuf;

/// The lock family of the cubins in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "strip_chain_race";

/// Every case's entries.
pub const CUBIN: &str = "strip_chain_race_sm120";

/// The contention probe's cubin and entry.
pub const PROBE_CUBIN: &str = "probe_strip_race_sm120";
pub const CONTEND: &str = "chop_probe_strip_race_contend_v1";

/// Threads per contention CTA (`contend` keeps eight words per thread of a
/// 1024-word pad).
pub const CONTEND_THREADS: u32 = 128;

/// One schedule at one bucket.
pub struct Case {
    pub tag: String,
    pub query: GemmQuery,
    pub kernel: DenseKernel,
    /// One-split `Tiled`, the bits every strip chain keeps.
    pub one: InstantiationSpec,
    /// The schedule on the locked kernel (its body is rewritten in `cubins`).
    pub locked: InstantiationSpec,
    /// The schedule as served.
    pub ordered: InstantiationSpec,
}

fn make(query: GemmQuery, kernel: DenseKernel, symbol: String) -> InstantiationSpec {
    gemm::instantiate(DenseRequest {
        query,
        kernel,
        target: Target::Sm120f,
        symbol,
        knobs: None,
    })
    .expect("lab strip-chain race instances are valid")
}

/// The cases' shapes: tag, query, strips and splits.
fn schedules() -> Vec<(String, GemmQuery, DenseKernel)> {
    let bf16 = |capacity, n, k, output| gemm::bf16_query(capacity, n, k, output);
    let fp8 = |capacity, n, k, output| {
        let mut query = gemm::fp8_block_query(capacity, n, k);
        query.output = output;
        query
    };
    let strips = |strips, splits| DenseKernel::StripChains { strips, splits };
    let mut schedules = Vec::new();
    for capacity in [4, 16, 32] {
        schedules.push((
            format!("head_n38720_k4096_f32_m{capacity}_s4x2"),
            bf16(capacity, 38720, 4096, DType::F32),
            strips(4, 2),
        ));
    }
    let warps = DenseKernel::TiledChains { splits: 1 };
    for (tag, query, kernel) in [
        (
            "n128_k4096_m8_s1x23",
            bf16(8, 128, 4096, DType::Bf16),
            strips(1, 23),
        ),
        (
            "n128_k4096_m32_s1x47",
            bf16(32, 128, 4096, DType::Bf16),
            strips(1, 47),
        ),
        (
            "n64_k4096_m32_s1x64",
            bf16(32, 64, 4096, DType::Bf16),
            strips(1, 64),
        ),
        (
            "n320_k4096_f32_m32_s1x18",
            bf16(32, 320, 4096, DType::F32),
            strips(1, 18),
        ),
        (
            "fp8_n4096_k4096_f32_m8_s2x2",
            fp8(8, 4096, 4096, DType::F32),
            strips(2, 2),
        ),
        (
            "fp8_n1536_k4096_m8_s2x7",
            fp8(8, 1536, 4096, DType::Bf16),
            strips(2, 7),
        ),
        (
            "fp8_n1536_k4096_m32_s1x3",
            fp8(32, 1536, 4096, DType::Bf16),
            strips(1, 3),
        ),
        (
            "fp8_n512_k4096_m32_s1x11",
            fp8(32, 512, 4096, DType::Bf16),
            strips(1, 11),
        ),
        (
            "warps_n4096_k2048_m2048",
            bf16(2048, 4096, 2048, DType::Bf16),
            warps,
        ),
        (
            "warps_fp8_n4096_k4096_f32_m2048",
            fp8(2048, 4096, 4096, DType::F32),
            warps,
        ),
    ] {
        schedules.push((tag.to_owned(), query, kernel));
    }
    schedules
}

/// Every case.
pub fn cases() -> Vec<Case> {
    schedules()
        .into_iter()
        .map(|(tag, query, kernel)| {
            let name = |what: &str| format!("chop_lab_strip_race_{tag}_{what}_v1");
            Case {
                one: make(query, DenseKernel::Tiled { splits: 1 }, name("one")),
                locked: make(query, kernel, name("locked")),
                ordered: make(query, kernel, name("ordered")),
                tag,
                query,
                kernel,
            }
        })
        .collect()
}

/// The served kernels and the locked ones they replace.
const RENAMES: [(&str, &str); 3] = [
    ("run_a16_strips_ordered<", "run_a16_strips<"),
    ("run_a16_strips_fp8_ordered<", "run_a16_strips_fp8<"),
    ("run_a16_warps_ordered<", "run_a16_warps<"),
];

/// The locked kernel's call in place of the served one's: the generated
/// body with its `_ordered` function renamed to the locked one (the
/// ordered headers include the locked ones).
pub fn locked_body(body: &str) -> String {
    let calls: usize = RENAMES
        .iter()
        .map(|(served, _)| body.matches(served).count())
        .sum();
    assert_eq!(calls, 1, "{body}");
    RENAMES
        .iter()
        .fold(body.to_owned(), |body, (served, locked)| {
            body.replace(served, locked)
        })
}

/// The family's cubins: every case's entries, the locked ones' bodies
/// rewritten (`locked_body`), then the contention probe.
#[allow(dead_code)] // Read by the build script only.
pub fn cubins() -> Vec<CubinSpec> {
    let cases = cases();
    let specs: Vec<InstantiationSpec> = cases
        .iter()
        .flat_map(|case| [case.one.clone(), case.locked.clone(), case.ordered.clone()])
        .collect();
    let mut cubin = gemm::cubin_spec(&specs).expect("lab strip-chain race cubin is valid");
    cubin.name = CUBIN.into();
    let locked: Vec<&str> = cases
        .iter()
        .map(|case| case.locked.symbol.as_str())
        .collect();
    for entry in &mut cubin.entries {
        if locked.contains(&entry.symbol.as_str()) {
            entry.body = locked_body(&entry.body);
        }
    }
    let pointer = |name: &str, cxx_type: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let probe = CubinSpec {
        name: PROBE_CUBIN.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_dirs: vec![],
        headers: vec!["#include \"kernel-lab/cuda/probes/strip_race.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            CONTEND.into(),
            vec![
                pointer("sink", "float"),
                P::U64 {
                    name: "words".into(),
                },
                P::U64 { name: "ns".into() },
            ],
            "chop::probe::strip_race::contend(sink, words, ns);".into(),
            CONTEND_THREADS,
            1,
        )],
    };
    vec![cubin, probe]
}

/// Every cubin name the family declares.
#[allow(dead_code)] // Read by the family file only.
pub fn cubin_names() -> Vec<String> {
    vec![CUBIN.to_owned(), PROBE_CUBIN.to_owned()]
}
