//! Lab-only strip-chain GEMM instances (`DenseKernel::StripChains`,
//! strip_chains.cuh), shared by the `strip_chains` lock family and the
//! tests that check and time them (`tests/tiled_chain_splits.rs`,
//! `tests/projection_latency.rs`).
//!
//! - Sweep: BF16 decode projections of eight shapes (n x K: 2048 x 4096,
//!   4096 x 2048, 128 x 4096, 64 x 4096, 4096 x 1536 and 2048 x 128 with
//!   BF16 outputs, 320 x 4096 and the vocabulary head's 38720 x 4096 with
//!   F32 outputs) at 1, 4, 8, 16 and 32 rows, each beside the entry a
//!   decode bucket of that shape runs today (`current`: `TiledChains` at
//!   `gemm::chain_splits`, else one-split `Tiled`) and the one-split
//!   `Tiled` entry whose bits every chain schedule keeps (`one`), in 1, 2
//!   and 4 strips per CTA at the split counts `sweep_splits` gives. One
//!   cubin per shape.
//! - Mutants: two shapes, each unmodified and with every deliberate defect
//!   of `chop::gemm::StripsMutation` that applies to it (no or a late
//!   dependency wait at one split; those, temporaries shared across tiles
//!   and a ticket never reset over K splits). The checks of
//!   `tests/tiled_chain_splits.rs` must pass on the unmodified instances
//!   and fail on each mutant.

use kernel_build::CubinSpec;
use kernel_core::contract::{DType, DenseKernel, GemmQuery, InstantiationSpec, Target};
use kernels_sm120::gemm::{self, DenseRequest};

/// The lock family of the cubins in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "strip_chains";

/// The mutant instances' cubin.
pub const MUTANTS_CUBIN: &str = "strip_chain_mutants_sm120";

/// Row capacities of the sweep: each runs at its capacity.
pub const CAPACITIES: [u32; 5] = [1, 4, 8, 16, 32];

/// `chop::gemm::StripsMutation` values; the first is the unmodified kernel.
pub const MUTATIONS: [&str; 5] = [
    "none",
    "drop_wait",
    "late_wait",
    "shared_slots",
    "keep_ticket",
];

/// The mutations that apply at one split: no or a late wait.
pub const ONE_SPLIT_MUTATIONS: [&str; 3] = ["none", "drop_wait", "late_wait"];

/// The swept shapes by tag: n, K and the output type.
pub fn shapes() -> Vec<(&'static str, u32, u32, DType)> {
    vec![
        ("n2048_k4096", 2048, 4096, DType::Bf16),
        ("n4096_k2048", 4096, 2048, DType::Bf16),
        ("n128_k4096", 128, 4096, DType::Bf16),
        ("n64_k4096", 64, 4096, DType::Bf16),
        ("n4096_k1536", 4096, 1536, DType::Bf16),
        ("n2048_k128", 2048, 128, DType::Bf16),
        ("n320_k4096_f32", 320, 4096, DType::F32),
        ("n38720_k4096_f32", 38720, 4096, DType::F32),
    ]
}

/// The cubin of the sweep shape `tag`.
pub fn cubin_name(tag: &str) -> String {
    format!("strip_chains_{tag}_sm120")
}

/// The K splits swept at `strips` strips per CTA over `query`: the counts
/// that fill one and two waves of `gemm::SM_COUNT` CTAs, and one split
/// where its tiles alone fill a third of them, each at most one per 64-K
/// chunk.
pub fn sweep_splits(query: &GemmQuery, strips: u8) -> Vec<u8> {
    let tiles = query.n.div_ceil(16 * u32::from(strips));
    let chunks = query.k.div_ceil(64);
    let mut splits: Vec<u8> = [gemm::SM_COUNT / tiles, 2 * gemm::SM_COUNT / tiles]
        .into_iter()
        .chain((tiles >= gemm::SM_COUNT / 3).then_some(1))
        .map(|s| s.clamp(1, chunks).min(u32::from(u8::MAX)) as u8)
        .collect();
    splits.sort_unstable();
    splits.dedup();
    splits
}

/// One swept shape at one capacity.
pub struct Case {
    /// The shape's tag (`shapes`), which names its cubin.
    pub shape: &'static str,
    pub tag: String,
    pub query: GemmQuery,
    /// The schedule a decode bucket of this shape runs today.
    pub current: InstantiationSpec,
    /// One-split `Tiled`, the bits every chain schedule keeps.
    pub one: InstantiationSpec,
    /// Strip-chain candidates.
    pub candidates: Vec<InstantiationSpec>,
}

fn make(query: GemmQuery, kernel: DenseKernel, symbol: String) -> InstantiationSpec {
    gemm::instantiate(DenseRequest {
        query,
        kernel,
        target: Target::Sm120f,
        symbol,
        knobs: None,
    })
    .expect("lab strip-chain instances are valid")
}

/// Every sweep case, shapes then capacities.
pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (tag, n, k, output) in shapes() {
        for capacity in CAPACITIES {
            let query = gemm::bf16_query(capacity, n, k, output);
            let name = |what: &str| format!("chop_lab_strips_{tag}_m{capacity}_{what}_v1");
            let current = match gemm::chain_splits(&query) {
                1 => DenseKernel::Tiled { splits: 1 },
                splits => DenseKernel::TiledChains { splits },
            };
            let candidates = gemm::STRIP_COUNTS
                .iter()
                .flat_map(|&strips| {
                    sweep_splits(&query, strips)
                        .into_iter()
                        .map(move |splits| (strips, splits))
                })
                .map(|(strips, splits)| {
                    make(
                        query,
                        DenseKernel::StripChains { strips, splits },
                        name(&format!("s{strips}x{splits}")),
                    )
                })
                .collect();
            cases.push(Case {
                shape: tag,
                tag: format!("{tag}_m{capacity}"),
                query,
                current: make(query, current, name("current")),
                one: make(query, DenseKernel::Tiled { splits: 1 }, name("one")),
                candidates,
            });
        }
    }
    cases
}

/// The mutant shapes: BF16 n512 x K 4096 at 1 row in 2 strips over 8
/// splits (16 tiles), and n2048 x K 4096 at 4 rows in 1 strip at one split.
pub fn mutant_queries() -> Vec<(&'static str, GemmQuery, u8, u8, &'static [&'static str])> {
    vec![
        (
            "n512_m1_s2x8",
            gemm::bf16_query(1, 512, 4096, DType::Bf16),
            2,
            8,
            &MUTATIONS[..],
        ),
        (
            "n2048_m4_s1x1",
            gemm::bf16_query(4, 2048, 4096, DType::Bf16),
            1,
            1,
            &ONE_SPLIT_MUTATIONS[..],
        ),
    ]
}

/// The mutant shape `tag`'s instance with `mutation`.
pub fn mutant(tag: &str, query: GemmQuery, strips: u8, splits: u8, mutation: &str) -> InstantiationSpec {
    make(
        query,
        DenseKernel::StripChains { strips, splits },
        format!("chop_lab_strips_{mutation}_{tag}_v1"),
    )
}

/// Every mutant instance with its mutation, shapes then mutations.
pub fn mutants() -> Vec<(String, InstantiationSpec)> {
    mutant_queries()
        .into_iter()
        .flat_map(|(tag, query, strips, splits, mutations)| {
            mutations.iter().map(move |&mutation| {
                (
                    mutation.to_owned(),
                    mutant(tag, query, strips, splits, mutation),
                )
            })
        })
        .collect()
}

/// The family's cubins: one per sweep shape (its cases' current, one-split
/// and candidate entries), then the mutants', each body given its mutation
/// as the kernel's last template argument.
#[allow(dead_code)] // Read by the build script only.
pub fn cubins() -> Vec<CubinSpec> {
    let cases = cases();
    let mut cubins: Vec<CubinSpec> = shapes()
        .into_iter()
        .map(|(tag, ..)| {
            let specs: Vec<InstantiationSpec> = cases
                .iter()
                .filter(|case| case.shape == tag)
                .flat_map(|case| {
                    [case.current.clone(), case.one.clone()]
                        .into_iter()
                        .chain(case.candidates.iter().cloned())
                })
                .collect();
            let mut cubin = gemm::cubin_spec(&specs).expect("lab strip-chain cubin is valid");
            cubin.name = cubin_name(tag);
            cubin
        })
        .collect();
    let mutants = mutants();
    let specs: Vec<InstantiationSpec> = mutants.iter().map(|(_, spec)| spec.clone()).collect();
    let mut cubin = gemm::cubin_spec(&specs).expect("lab strip-chain mutant cubin is valid");
    cubin.name = MUTANTS_CUBIN.into();
    for (entry, (mutation, _)) in cubin.entries.iter_mut().zip(&mutants) {
        assert_eq!(entry.body.matches(">(map_a").count(), 1, "{}", entry.symbol);
        entry.body = entry.body.replace(
            ">(map_a",
            &format!(", chop::gemm::StripsMutation::{mutation}>(map_a"),
        );
    }
    cubins.push(cubin);
    cubins
}

/// Every cubin name the family declares.
#[allow(dead_code)] // Read by the family file only.
pub fn cubin_names() -> Vec<String> {
    shapes()
        .into_iter()
        .map(|(tag, ..)| cubin_name(tag))
        .chain(std::iter::once(MUTANTS_CUBIN.to_owned()))
        .collect()
}
