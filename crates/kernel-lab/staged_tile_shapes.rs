//! Lab-only staged-tile GEMM instances (`DenseKernel::TiledChains` over
//! BF16 weights at `gemm::staged_tile_knobs`, staged_tiles.cuh), shared by
//! the `staged_tiles` lock family and the tests that check and time them
//! (`tests/tiled_chain_splits.rs`, `tests/projection_latency.rs`).
//!
//! - Sweep: the BF16 prefill projections of seven shapes (n x K: the linear
//!   attention q, k and v 2048 x 4096, its output 4096 x 2048 with F32
//!   outputs, the gate and forget gates' up projections 2048 x 128, the
//!   indexer's query 4096 x 1536, the drafter's input projection 4096 x
//!   8192; 4096 x 2048 with F32 outputs onto a residual, which no served
//!   projection takes; and the gate and forget low-rank inputs 128 x 4096,
//!   which stay on their warp tiles) at a prefill capacity of `CAPACITY` rows,
//!   each beside the entry a prefill of that shape runs today (`current`:
//!   warp tiles at its default knobs) and the one-split `Tiled` entry whose
//!   bits every schedule keeps (`one`). Candidates: the staged tiles, then
//!   the warp tiles at 64 x 64 for comparison. Each output element's 64-K chains and their K
//!   order sums are the same in every candidate. One cubin per shape.
//! - Mutants: BF16 512 x 4096 at 256 rows (two row tiles, so 8 tiles and a
//!   wave of 16 halves), unmodified and with every deliberate defect of
//!   `chop::gemm::DecodedMutation` (no dependency wait, the math warps'
//!   wait after the K loop, 128-K chains).

use crate::strip_chains::shapes::Case;
use kernel_build::CubinSpec;
use kernel_core::contract::{
    CompileKnobs, DType, DenseKernel, GemmQuery, InstantiationSpec, Target,
};
use kernels_sm120::gemm::{self, DenseRequest};

/// The lock family of the cubins in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "staged_tiles";

/// The mutant instances' cubin.
pub const MUTANTS_CUBIN: &str = "staged_tile_mutants_sm120";

/// The row capacity of the sweep: a prefill step's.
pub const CAPACITY: u32 = 8192;

/// The rows each sweep case is timed at: a 3K prompt's second part, a
/// teacher text, a prefill part (a 2,048-token chunk), a whole 3K prompt
/// and a whole step.
pub const TIMED_ROWS: [u32; 5] = [960, 1040, 2048, 3008, 8192];

/// The warp tiles compared beside the staged tiles (whose 128 x 128 tiles
/// the warp tiles no longer take).
pub const WARP_TILES: [(u16, u16); 1] = [(64, 64)];

/// `chop::gemm::DecodedMutation` values; the first is the unmodified
/// kernel.
pub const MUTATIONS: [&str; 4] = ["none", "drop_wait", "late_wait", "long_chain"];

/// The swept shapes by tag: n, K, the output type and a residual add.
pub fn shapes() -> Vec<(&'static str, u32, u32, DType, bool)> {
    vec![
        ("n2048_k4096", 2048, 4096, DType::Bf16, false),
        ("n4096_k2048_f32", 4096, 2048, DType::F32, false),
        ("n2048_k128", 2048, 128, DType::Bf16, false),
        ("n4096_k1536", 4096, 1536, DType::Bf16, false),
        ("n4096_k8192", 4096, 8192, DType::Bf16, false),
        ("n4096_k2048_f32_residual", 4096, 2048, DType::F32, true),
        ("n128_k4096", 128, 4096, DType::Bf16, false),
    ]
}

/// A BF16 query of `n` x `k` at `capacity` rows.
pub fn query(capacity: u32, n: u32, k: u32, output: DType, residual: bool) -> GemmQuery {
    let mut query = gemm::bf16_query(capacity, n, k, output);
    query.epilogues.residual_add = residual;
    query
}

/// The cubin of the sweep shape `tag`.
pub fn cubin_name(tag: &str) -> String {
    format!("staged_tiles_{tag}_sm120")
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
    .expect("lab staged-tile instances are valid")
}

/// Every sweep case, in shape order.
pub fn cases() -> Vec<Case> {
    shapes()
        .into_iter()
        .map(|(tag, n, k, output, residual)| {
            let query = query(CAPACITY, n, k, output, residual);
            let name = |what: &str| format!("chop_lab_staged_{tag}_m{CAPACITY}_{what}_v1");
            let chains = DenseKernel::TiledChains { splits: 1 };
            let current = make(query, chains, None, name("current"));
            let defaults = current.key.knobs;
            let warp_tiles = WARP_TILES
                .iter()
                .filter(|&&(m, n)| (m, n) != (defaults.tile_m, defaults.tile_n))
                .map(|&(tile_m, tile_n)| {
                    // Three 64-K stages fit every admitted tile.
                    let knobs = CompileKnobs {
                        tile_m,
                        tile_n,
                        ..defaults
                    };
                    make(query, chains, Some(knobs), name(&format!("t{tile_m}x{tile_n}")))
                });
            Case {
                shape: tag,
                tag: format!("bf16_{tag}_m{CAPACITY}"),
                query,
                current,
                one: make(query, DenseKernel::Tiled { splits: 1 }, None, name("one")),
                candidates: std::iter::once(make(
                    query,
                    chains,
                    Some(gemm::staged_tile_knobs()),
                    name("staged"),
                ))
                .chain(warp_tiles)
                .collect(),
            }
        })
        .collect()
}

/// The mutant shape: BF16 512 x 4096 with BF16 outputs at 256 rows.
pub fn mutant_query() -> (&'static str, GemmQuery) {
    ("n512_m256", query(256, 512, 4096, DType::Bf16, false))
}

/// The mutant shape's instance with `mutation`.
pub fn mutant(mutation: &str) -> InstantiationSpec {
    let (tag, query) = mutant_query();
    make(
        query,
        DenseKernel::TiledChains { splits: 1 },
        Some(gemm::staged_tile_knobs()),
        format!("chop_lab_staged_{mutation}_{tag}_v1"),
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
/// and candidate entries), then the mutants', each body given its mutation
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
            let mut cubin = gemm::cubin_spec(&specs).expect("lab staged-tile cubin is valid");
            cubin.name = cubin_name(case.shape);
            cubin
        })
        .collect();
    let mutants = mutants();
    let specs: Vec<InstantiationSpec> = mutants.iter().map(|(_, spec)| spec.clone()).collect();
    let mut cubin = gemm::cubin_spec(&specs).expect("lab staged-tile mutant cubin is valid");
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
