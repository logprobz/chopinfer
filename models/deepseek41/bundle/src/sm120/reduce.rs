//! The all-reduces' entries the bundle registers on SM120 and its build
//! compiles. Every sublayer's F32 rank partials are summed in F32 in rank
//! order and rounded once to BF16 (`CommQuery::AllReduceRounded`, the
//! vendor's F32 all-reduce of `RowParallelLinear` before its BF16 cast),
//! and the embedding's widened shard rows the same way. Until create admits
//! the low-latency and phased entries against the devices' residency, the
//! rounded all-reduce takes the composed form: the comm engine's F32
//! reduce-scatter and all-gather, then the round row of the sum: the
//! shared rounding rows (`kernels_sm120::row::rounding`) compiled under
//! each geometry's symbols, in one cubin per geometry. The comm cubin also
//! holds the F32 row exchange and row gather the logits move through.

use kernel_build::CubinSpec;
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::rounding::{self, PartialRounding, Rounding};
use model_deepseek41_contract::geometry::Geometry;

/// The lock family and cubin of the comm engine's entries.
pub const COMM_FAMILY: &str = "comm";
pub const COMM_CUBIN: &str = "comm_deepseek41_sm120";
/// The lock family of the round and widen rows, one cubin per geometry.
pub const ROUNDING_FAMILY: &str = "row_rounding";

/// The comm engine's F32 reduce-scatter and all-gather, which the rounded
/// all-reduce composes with the round row.
pub const SCATTER_SYMBOL: &str = "chop_comm_reduce_scatter_f32_v1";
pub const GATHER_SYMBOL: &str = "chop_comm_allgather_scattered_f32_v1";
/// The F32 logits moves: decode and verify rows exchanged to their block
/// owners, prefill rows gathered to rank 0.
pub const EXCHANGE_SYMBOL: &str = "chop_comm_exchange_rows_f32_v1";
pub const GATHER_ROWS_SYMBOL: &str = "chop_comm_gather_rows_f32_v1";
/// Every comm entry the bundle embeds.
pub const COMM_SYMBOLS: [&str; 4] = [
    SCATTER_SYMBOL,
    GATHER_SYMBOL,
    EXCHANGE_SYMBOL,
    GATHER_ROWS_SYMBOL,
];

/// The name of `g`'s rounding cubin.
pub fn rounding_cubin(g: &Geometry) -> String {
    format!("row_rounding_{}_sm120", g.kernel_tag)
}

/// `g`'s rounding row `rounding` of its hidden rows.
pub fn rounding_symbol(g: &Geometry, rounding: Rounding) -> String {
    let name = match rounding {
        Rounding::Round => "round",
        Rounding::Widen => "widen",
    };
    format!(
        "chop_row_rounding_{name}_{}_w{}_v1",
        g.kernel_tag, g.shape.hidden
    )
}

/// `g`'s round and widen rows: the shared entries' bodies under `g`'s
/// symbols, which declare what the shared entries do.
pub fn pair(g: &Geometry) -> PartialRounding {
    let width = g.shape.hidden;
    let entry = |kind| rounding::entry(kind, width, rounding_symbol(g, kind));
    PartialRounding {
        width,
        round: entry(Rounding::Round),
        widen: entry(Rounding::Widen),
        invariance: rounding::shared_invariance(),
    }
}

/// The comm cubin and each geometry's rounding cubin, with their lock
/// families.
pub fn cubins(geometries: &[Geometry]) -> Vec<(&'static str, CubinSpec)> {
    let mut comm = kernels_sm120::comm::cubin_spec();
    comm.name = COMM_CUBIN.into();
    comm.entries
        .retain(|entry| COMM_SYMBOLS.contains(&entry.symbol.as_str()));
    let mut phased =
        kernels_sm120::comm::phased_f32::cubin_spec("comm_rounded_phased_deepseek41_sm120".into());
    phased.entries.retain(|entry| {
        [
            kernels_sm120::comm::phased_f32::ROUND_SYMBOL,
            kernels_sm120::comm::phased_f32::ONE_SHOT_ROUND_SYMBOL,
        ]
        .contains(&entry.symbol.as_str())
    });
    let mut specs = vec![
        (COMM_FAMILY, comm),
        ("comm_rounded_phased", phased),
        (
            "comm_dma_sum",
            kernels_sm120::comm::owned_sum_f32::cubin_spec("comm_dma_sum_deepseek41_sm120".into()),
        ),
        (
            "comm_dma_bf16",
            kernels_sm120::comm::owned_sum_bf16::cubin_spec(
                "comm_dma_bf16_deepseek41_sm120".into(),
            ),
        ),
        (
            "comm_bf16_partials",
            kernels_sm120::comm::phased_bf16::cubin_spec(
                "comm_bf16_partials_deepseek41_sm120".into(),
            ),
        ),
        (
            "comm_bf16_partials",
            kernels_sm120::comm::phased_turns_bf16::cubin_spec(
                "comm_bf16_partials_turns_deepseek41_sm120".into(),
            ),
        ),
    ];
    for g in geometries {
        specs.push((
            ROUNDING_FAMILY,
            rounding::cubin_spec(rounding_cubin(g), &[pair(g)]),
        ));
    }
    specs
}

/// `sm120` with the round and widen rows of `g`'s hidden width, which the
/// rounded all-reduce's composed form launches.
pub fn register(sm120: Sm120, g: &Geometry) -> Sm120 {
    sm120.with_partial_rounding(pair(g))
}
