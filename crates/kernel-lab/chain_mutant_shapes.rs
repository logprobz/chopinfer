//! Lab-only chain-split tiled GEMM instances (`DenseKernel::TiledChains`)
//! at two decode shapes, each built unmodified and with every deliberate
//! defect of `chop::gemm::ChainsMutation` (no or a late dependency wait,
//! temporaries shared across tiles, a ticket never reset), and at two
//! prefill shapes (warp tiles, `tiled_warps.cuh`) unmodified and with the
//! defects that apply to them (no or a late dependency wait). Shared by
//! build.rs, which builds their cubin in the `tiled_chain_mutants` lock
//! family, and `tests/tiled_chain_splits.rs`, whose PDL, workspace and
//! ticket checks must pass on the unmodified instances and fail on each
//! mutant.

use kernel_build::CubinSpec;
use kernel_core::contract::{
    DType, DenseKernel, GemmQuery, InstantiationSpec, ScaleMode, Target, WeightFormat,
};
use kernels_sm120::gemm::{self, DenseRequest};

/// The lock family of the cubin in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "tiled_chain_mutants";

/// The instances' cubin.
pub const CUBIN: &str = "tiled_chain_mutants_sm120";

/// `chop::gemm::ChainsMutation` values; the first is the unmodified kernel.
pub const MUTATIONS: [&str; 5] = [
    "none",
    "drop_wait",
    "late_wait",
    "shared_slots",
    "keep_ticket",
];

/// The warp-tile mutations: the first is the unmodified kernel.
pub const WARP_MUTATIONS: [&str; 3] = ["none", "drop_wait", "late_wait"];

/// BF16 weights at 1 row and FP8 128x128 weights at 4 rows, both n512 by
/// K 4096: eight tiles of 24 K splits each.
pub fn queries() -> Vec<(&'static str, GemmQuery)> {
    let bf16 = gemm::bf16_query(1, 512, 4096, DType::Bf16);
    let mut fp8 = gemm::bf16_query(4, 512, 4096, DType::Bf16);
    fp8.weight = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);
    vec![("bf16_m1", bf16), ("fp8_m4", fp8)]
}

/// Warp-tile prefill shapes: BF16 and FP8 128x128 weights, n512 by K 4096,
/// at 64 rows (one row of 64 x 128 tiles).
pub fn warp_queries() -> Vec<(&'static str, GemmQuery)> {
    let bf16 = gemm::bf16_query(64, 512, 4096, DType::Bf16);
    let mut fp8 = gemm::bf16_query(64, 512, 4096, DType::Bf16);
    fp8.weight = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);
    vec![("bf16_m64", bf16), ("fp8_m64", fp8)]
}

/// The instance of `query` with `mutation`, at the served split count (one
/// split in warp tiles above 32 rows).
pub fn instance(tag: &str, query: GemmQuery, mutation: &str) -> InstantiationSpec {
    let splits = if query.m_capacity > 32 {
        1
    } else {
        gemm::chain_splits(&query)
    };
    gemm::instantiate(DenseRequest {
        query,
        kernel: DenseKernel::TiledChains { splits },
        target: Target::Sm120f,
        symbol: format!("chop_lab_chains_{mutation}_{tag}_v1"),
        knobs: None,
    })
    .expect("lab chain-split instances are valid")
}

/// Every instance, shapes then mutations: chain splits, then warp tiles.
pub fn instances() -> Vec<(String, InstantiationSpec)> {
    let each = |queries: Vec<(&'static str, GemmQuery)>, mutations: &'static [&'static str]| {
        queries.into_iter().flat_map(move |(tag, query)| {
            mutations
                .iter()
                .map(move |&mutation| (mutation.to_owned(), instance(tag, query, mutation)))
        })
    };
    each(queries(), &MUTATIONS)
        .chain(each(warp_queries(), &WARP_MUTATIONS))
        .collect()
}

/// The cubin: `gemm::cubin_spec`'s entries, each body given its mutation
/// as the kernel's last template argument.
#[allow(dead_code)] // Read by the build script only.
pub fn cubin() -> CubinSpec {
    let instances = instances();
    let specs = instances
        .iter()
        .map(|(_, spec)| spec.clone())
        .collect::<Vec<_>>();
    let mut cubin = gemm::cubin_spec(&specs).expect("lab chain-split cubin is valid");
    cubin.name = CUBIN.into();
    for (entry, (mutation, _)) in cubin.entries.iter_mut().zip(&instances) {
        assert_eq!(entry.body.matches(">(map_a").count(), 1, "{}", entry.symbol);
        entry.body = entry.body.replace(
            ">(map_a",
            &format!(", chop::gemm::ChainsMutation::{mutation}>(map_a"),
        );
    }
    cubin
}
