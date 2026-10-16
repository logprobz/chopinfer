//! MiMo V2.6 Flash's communication on the shared transport
//! (`kernels_sm120::comm`): its declared row shapes, including the per-rank
//! vocabulary shard, and its verify all-reduce cubin.

use super::geometry::{self, GeometryModel};
use crate::geometry::V26_FLASH as GEOMETRY;

use kernel_build::CubinSpec;
use kernel_core::contract::{DType, Refusal};
use kernels_sm120::comm::{self, Entry, Shape};
use kernels_sm120::names::intern;

/// The state MiMo's comm buffers size from the shared transport: the
/// fused all-reduces' (`fused_state_buffers`), and the flag and epoch
/// bytes of the logits row gather and row exchange.
pub use kernels_sm120::comm::{
    EXCHANGE_ROWS_EPOCH_BYTES, EXCHANGE_ROWS_FLAG_BYTES, GATHER_ROWS_EPOCH_BYTES,
    GATHER_ROWS_FLAG_BYTES, PushBuffers, fused_state_buffers,
};

/// The cubin of MiMo's exact verify all-reduces (`comm::VERIFY_SYMBOL`,
/// `comm::VERIFY_COMBINE_SYMBOL`).
pub const VERIFY_CUBIN: &str =
    kernels_sm120::const_name!("comm_", GEOMETRY.kernel_tag, "_verify_sm120");

/// The verify all-reduces' cubin.
pub fn verify_cubin_spec() -> CubinSpec {
    comm::verify_cubin_spec(VERIFY_CUBIN)
}

/// Policy: select prepared communication capacities.
const ROWS: [u64; 7] = [1, 2, 4, 8, 16, 32, 8192];

/// The shapes at `ranks` ranks (`ModelCatalog::comm_shapes`): the hidden
/// all-reduce, its prefill gather and the vocabulary shard's gathers.
pub fn shapes(ranks: u32) -> Vec<Entry> {
    geometry::MODELS
        .iter()
        .flat_map(|owner| shapes_for_geometry(*owner, ranks).expect("a catalog rank count"))
        .collect()
}

/// The shapes at `ranks` ranks, refusing a rank count that does not divide
/// the vocabulary.
pub fn shapes_for(ranks: u32) -> Result<Vec<Entry>, Refusal> {
    shapes_for_geometry(geometry::MODELS[0], ranks)
}

pub fn shapes_for_geometry(owner: GeometryModel, ranks: u32) -> Result<Vec<Entry>, Refusal> {
    let g = owner.geometry;
    if g.hidden() != comm::FUSED_EPILOGUE.width {
        return Err(Refusal("comm fused epilogue hidden width"));
    }
    let vocabulary = u64::from(g.vocab());
    let hidden = Shape {
        profile: owner.profile(4),
        family: intern(format!("{}_hidden", g.kernel_tag)),
        width: u64::from(g.hidden()),
        dtype: DType::Bf16,
    };
    if ranks == 0 || !vocabulary.is_multiple_of(u64::from(ranks)) {
        return Err(Refusal("comm catalog rank count"));
    }
    let vocab_per_rank = vocabulary / u64::from(ranks);
    let mut entries = Vec::new();
    for rows in ROWS {
        let reduce_symbol = if rows <= 32 {
            "chop_comm_allreduce_oneshot_norm_bf16_v2"
        } else {
            "chop_comm_reduce_scatter_bf16_v1"
        };
        entries.push(hidden.entry(ranks, rows, reduce_symbol));
        if rows > 32 {
            entries.push(
                Shape {
                    family: intern(format!("{}_hidden_gather", g.kernel_tag)),
                    ..hidden
                }
                .entry(ranks, rows, "chop_comm_allgather_scattered_bf16_v1"),
            );
        }
        for (family, dtype, symbol) in [
            (
                intern(format!("{}_vocab", g.kernel_tag)),
                DType::F32,
                "chop_comm_allgather_f32_v1",
            ),
            (
                intern(format!("{}_vocab_bf16", g.kernel_tag)),
                DType::Bf16,
                "chop_comm_allgather_bf16_v1",
            ),
        ] {
            entries.push(
                Shape {
                    profile: owner.profile(4),
                    family,
                    width: vocab_per_rank,
                    dtype,
                }
                .entry(ranks, rows, symbol),
            );
        }
    }
    Ok(entries)
}

// Verify symbols are geometry-owned so membership resolves them unambiguously.
// Each topology binds its own names to the shared verify implementation.
pub fn verify_for(owner: GeometryModel) -> CubinSpec {
    let mut spec =
        comm::verify_cubin_spec(&format!("comm_{}_verify_sm120", owner.geometry.kernel_tag));
    if !geometry::pinned_names(owner.geometry) {
        for entry in &mut spec.entries {
            entry.symbol = entry.symbol.replacen(
                "chop_comm_",
                &format!("chop_comm_{}_", owner.geometry.kernel_tag),
                1,
            );
        }
    }
    spec
}
pub fn cubins() -> Vec<CubinSpec> {
    geometry::MODELS
        .iter()
        .map(|owner| verify_for(*owner))
        .collect()
}
