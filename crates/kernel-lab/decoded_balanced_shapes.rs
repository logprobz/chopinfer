//! Lab-only balanced decoded-tile GEMM instances (`DenseKernel::TiledChains`
//! over FP8 128x128 block weights at `gemm::balanced_decoded_tile_knobs`,
//! decoded_balanced.cuh), shared by the `decoded_balanced` lock family and
//! the tests that check and time them (`tests/tiled_chain_splits.rs`,
//! `tests/projection_latency.rs`).
//!
//! - Sweep: the six FP8 prefill projections of the decoded-tile sweep
//!   (`decoded_tile_shapes.rs`) at its capacity, each beside the decoded
//!   tiles a prefill of that shape runs today (`current`) and the one-split
//!   `Tiled` entry whose bits both keep (`one`). One candidate, the balanced
//!   decoded tiles. One cubin per shape.
//! - Mutants: the decoded-tile mutant shape (512 x 4096 at 256 rows, 16
//!   halves on 16 CTAs), unmodified and with every deliberate defect of
//!   `chop::gemm::DecodedMutation` (no dependency wait, the math warps' wait
//!   after their first item's K loop, 256-K chains).

use crate::decoded_tiles::shapes as decoded;
use crate::strip_chains::shapes::Case;
use kernel_build::CubinSpec;
use kernel_core::contract::{CompileKnobs, DenseKernel, GemmQuery, InstantiationSpec, Target};
use kernels_sm120::gemm::{self, DenseRequest};

/// The lock family of the cubins in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "decoded_balanced";

/// The mutant instances' cubin.
pub const MUTANTS_CUBIN: &str = "decoded_balanced_mutants_sm120";

/// The rows each sweep case is timed at: a teacher-scored step, a short
/// prompt, a teacher text, a prefill part (a 2,048-token chunk), a 3K
/// prompt's step and a whole step.
pub const TIMED_ROWS: [u32; 6] = [64, 512, 1040, 2048, 3008, 8192];

/// The cubin of the sweep shape `tag`.
pub fn cubin_name(tag: &str) -> String {
    format!("decoded_balanced_{tag}_sm120")
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
    .expect("lab balanced decoded-tile instances are valid")
}

/// Every sweep case, in the decoded-tile sweep's shape order.
pub fn cases() -> Vec<Case> {
    decoded::shapes()
        .into_iter()
        .map(|(tag, n, k, output, residual)| {
            let capacity = decoded::CAPACITY;
            let query = decoded::query(capacity, n, k, output, residual);
            let name = |what: &str| format!("chop_lab_balanced_{tag}_m{capacity}_{what}_v1");
            let chains = DenseKernel::TiledChains { splits: 1 };
            Case {
                shape: tag,
                tag: format!("fp8_{tag}_m{capacity}"),
                query,
                current: make(
                    query,
                    chains,
                    Some(gemm::decoded_tile_knobs()),
                    name("decoded"),
                ),
                one: make(query, DenseKernel::Tiled { splits: 1 }, None, name("one")),
                candidates: vec![make(
                    query,
                    chains,
                    Some(gemm::balanced_decoded_tile_knobs()),
                    name("balanced"),
                )],
            }
        })
        .collect()
}

/// The mutant shape's instance with `mutation`.
pub fn mutant(mutation: &str) -> InstantiationSpec {
    let (tag, query) = decoded::mutant_query();
    make(
        query,
        DenseKernel::TiledChains { splits: 1 },
        Some(gemm::balanced_decoded_tile_knobs()),
        format!("chop_lab_balanced_{mutation}_{tag}_v1"),
    )
}

/// Every mutant instance with its mutation.
pub fn mutants() -> Vec<(String, InstantiationSpec)> {
    decoded::MUTATIONS
        .iter()
        .map(|&mutation| (mutation.to_owned(), mutant(mutation)))
        .collect()
}

/// The family's cubins: one per sweep shape (its case's current, one-split
/// and balanced entries), then the mutants', each body given its mutation
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
            let mut cubin =
                gemm::cubin_spec(&specs).expect("lab balanced decoded-tile cubin is valid");
            cubin.name = cubin_name(case.shape);
            cubin
        })
        .collect();
    let mutants = mutants();
    let specs: Vec<InstantiationSpec> = mutants.iter().map(|(_, spec)| spec.clone()).collect();
    let mut cubin =
        gemm::cubin_spec(&specs).expect("lab balanced decoded-tile mutant cubin is valid");
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
    decoded::shapes()
        .into_iter()
        .map(|(tag, ..)| cubin_name(tag))
        .chain(std::iter::once(MUTANTS_CUBIN.to_owned()))
        .collect()
}
