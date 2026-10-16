//! Lab-only decoded-tile GEMM instances (`DenseKernel::TiledChains` over
//! FP8 128x128 block weights at `gemm::decoded_tile_knobs`,
//! decoded_tiles.cuh), shared by the `decoded_tiles` lock family and the
//! tests that check and time them (`tests/tiled_chain_splits.rs`,
//! `tests/projection_latency.rs`).
//!
//! - Sweep: the FP8 prefill projections of six shapes (n x K: 4096 x 4096,
//!   4096 x 3072 and 4096 x 512 with a residual, all three with F32
//!   outputs; 1536 x 4096, 512 x 4096 and 4096 x 1536 with BF16 outputs)
//!   at a prefill capacity of `CAPACITY` rows, each beside the entry a
//!   prefill of that shape runs today (`current`: `TiledChains { splits: 1
//!   }` at its default knobs, warp tiles) and the one-split `Tiled` entry
//!   whose bits both keep (`one`). One candidate, the decoded tiles. One
//!   cubin per shape. The cases share `strip_chain_shapes.rs`' `Case`.
//! - Mutants: one shape (512 x 4096 at 256 rows), unmodified and with every
//!   deliberate defect of `chop::gemm::DecodedMutation` (no dependency
//!   wait, the math warps' wait after the K loop, 256-K chains). The checks
//!   of `tests/tiled_chain_splits.rs` must pass on the unmodified instance
//!   and fail on each mutant.

use crate::strip_chains::shapes::Case;
use kernel_build::CubinSpec;
use kernel_core::contract::{
    CompileKnobs, DType, DenseKernel, GemmQuery, InstantiationSpec, Target,
};
use kernels_sm120::gemm::{self, DenseRequest};

/// The lock family of the cubins in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "decoded_tiles";

/// The mutant instances' cubin.
pub const MUTANTS_CUBIN: &str = "decoded_tile_mutants_sm120";

/// The row capacity of the sweep: a prefill step's.
pub const CAPACITY: u32 = 8192;

/// The rows each sweep case is timed at: a teacher-scored step, a short
/// prompt, a prefill part (a 2,048-token chunk) and a whole step.
pub const TIMED_ROWS: [u32; 4] = [64, 512, 2048, 8192];

/// `chop::gemm::DecodedMutation` values; the first is the unmodified
/// kernel.
pub const MUTATIONS: [&str; 4] = ["none", "drop_wait", "late_wait", "long_chain"];

/// The swept shapes by tag: n, K, the output type and a residual add.
pub fn shapes() -> Vec<(&'static str, u32, u32, DType, bool)> {
    vec![
        ("n4096_k4096_f32", 4096, 4096, DType::F32, false),
        ("n1536_k4096", 1536, 4096, DType::Bf16, false),
        ("n512_k4096", 512, 4096, DType::Bf16, false),
        ("n4096_k1536", 4096, 1536, DType::Bf16, false),
        ("n4096_k3072_f32", 4096, 3072, DType::F32, false),
        ("n4096_k512_f32_residual", 4096, 512, DType::F32, true),
    ]
}

/// An FP8 128x128 query of `n` x `k` at `capacity` rows.
pub fn query(capacity: u32, n: u32, k: u32, output: DType, residual: bool) -> GemmQuery {
    let mut query = gemm::fp8_block_query(capacity, n, k);
    query.output = output;
    query.epilogues.residual_add = residual;
    query
}

/// The cubin of the sweep shape `tag`.
pub fn cubin_name(tag: &str) -> String {
    format!("decoded_tiles_{tag}_sm120")
}

fn make(
    query: GemmQuery,
    kernel: DenseKernel,
    knobs: Option<CompileKnobs>,
    symbol: String,
) -> InstantiationSpec {
    gemm::instantiate(DenseRequest {
        query,
        kernel,
        target: Target::Sm120f,
        symbol,
        knobs,
    })
    .expect("lab decoded-tile instances are valid")
}

/// Every sweep case, in shape order.
pub fn cases() -> Vec<Case> {
    shapes()
        .into_iter()
        .map(|(tag, n, k, output, residual)| {
            let query = query(CAPACITY, n, k, output, residual);
            let name = |what: &str| format!("chop_lab_decoded_{tag}_m{CAPACITY}_{what}_v1");
            let chains = DenseKernel::TiledChains { splits: 1 };
            Case {
                shape: tag,
                tag: format!("fp8_{tag}_m{CAPACITY}"),
                query,
                current: make(query, chains, None, name("current")),
                one: make(query, DenseKernel::Tiled { splits: 1 }, None, name("one")),
                candidates: vec![make(
                    query,
                    chains,
                    Some(gemm::decoded_tile_knobs()),
                    name("decoded"),
                )],
            }
        })
        .collect()
}

/// The mutant shape: FP8 512 x 4096 with BF16 outputs at 256 rows (8
/// tiles).
pub fn mutant_query() -> (&'static str, GemmQuery) {
    ("n512_m256", query(256, 512, 4096, DType::Bf16, false))
}

/// The mutant shape's instance with `mutation`.
pub fn mutant(mutation: &str) -> InstantiationSpec {
    let (tag, query) = mutant_query();
    make(
        query,
        DenseKernel::TiledChains { splits: 1 },
        Some(gemm::decoded_tile_knobs()),
        format!("chop_lab_decoded_{mutation}_{tag}_v1"),
    )
}

/// Every mutant instance with its mutation.
pub fn mutants() -> Vec<(String, InstantiationSpec)> {
    MUTATIONS
        .iter()
        .map(|&mutation| (mutation.to_owned(), mutant(mutation)))
        .collect()
}

/// The family's cubins: one per sweep shape (its case's current, one-split
/// and decoded entries), then the mutants', each body given its mutation
/// as the kernel's last template argument.
#[allow(dead_code)] // Read by the build script only.
pub fn cubins() -> Vec<CubinSpec> {
    let mut cubins: Vec<CubinSpec> = cases()
        .into_iter()
        .map(|case| {
            let specs: Vec<InstantiationSpec> = [case.current, case.one]
                .into_iter()
                .chain(case.candidates)
                .collect();
            let mut cubin = gemm::cubin_spec(&specs).expect("lab decoded-tile cubin is valid");
            cubin.name = cubin_name(case.shape);
            cubin
        })
        .collect();
    let mutants = mutants();
    let specs: Vec<InstantiationSpec> = mutants.iter().map(|(_, spec)| spec.clone()).collect();
    let mut cubin = gemm::cubin_spec(&specs).expect("lab decoded-tile mutant cubin is valid");
    cubin.name = MUTANTS_CUBIN.into();
    for (entry, (mutation, _)) in cubin.entries.iter_mut().zip(&mutants) {
        assert_eq!(entry.body.matches(">(map_a").count(), 1, "{}", entry.symbol);
        entry.body = entry.body.replace(
            ">(map_a",
            &format!(", chop::gemm::DecodedMutation::{mutation}>(map_a"),
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
