//! The direct MoE queues of Flash TP4 verify steps above 64 rows
//! (`kernels_sm120::moe::chained::wide`): the backbone bank's direct
//! entries compiled at the package's verify rows, over the generic route's
//! metadata, with its gate/up bank in quad-lane row tiles.
use kernel_build::CubinSpec;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::moe::chained::{floor::Entry, quantized::Projection, wide};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
use model_deepseek41_contract::moe;
use model_deepseek41_contract::serving::{DECODE_BATCH, MAX_VERIFY_ROWS};

pub const FAMILY: &str = "routed_moe_direct_wide";
pub const CUBIN: &str = "bandwidth_moe_deepseek41_wide_sm120";

/// The compiled capacity: the package's verify rows.
pub const CAPACITY: u32 = DECODE_BATCH * MAX_VERIFY_ROWS;

/// The backbone bank's direct entry at [`CAPACITY`]. The ceil floor is the
/// 64-row direct entries' (`moe_direct_kernels::entries`).
pub fn entry(g: &Geometry) -> Result<Entry, String> {
    let set = moe::expert_sets(g)[0];
    let projection = |kind| -> Result<Projection, String> {
        Projection::new(moe::projection(g, 4, set, kind, CAPACITY)?, 64, CAPACITY)
            .and_then(Projection::with_direct_chains)
            .map_err(|e| e.to_string())
    };
    Entry::new(
        projection(GatedMlpProjection::GateUp)?,
        projection(GatedMlpProjection::Down)?,
        1e-4_f32.to_bits(),
        format!(
            "chop_{}_moe_floor_t{CAPACITY}_e{}_k{}",
            g.kernel_tag, set.count, set.top_k
        ),
    )
    .map_err(|e| e.to_string())
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        wide::cubin_spec(CUBIN, &[entry(&FLASH)?]).map_err(|e| e.to_string())?,
    )])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wide entry is the 64-row direct entry of the backbone bank at
    /// the package's verify rows: the same geometry, clamp and ceil floor.
    #[test]
    fn the_wide_entry_is_the_direct_entry_at_the_package_rows() {
        let wide = entry(&FLASH).unwrap();
        let direct = super::super::moe_direct_kernels::entries(&FLASH)
            .unwrap()
            .into_iter()
            .find(|e| e.experts() == 384)
            .unwrap();
        assert_eq!(CAPACITY, 192);
        assert_eq!(wide.capacity(), 192);
        assert_eq!(direct.capacity(), 64);
        assert_eq!(
            (
                wide.experts(),
                wide.top_k(),
                wide.hidden(),
                wide.intermediate()
            ),
            (
                direct.experts(),
                direct.top_k(),
                direct.hidden(),
                direct.intermediate()
            )
        );
        assert_eq!(wide.floor_bits, direct.floor_bits);
        assert_eq!(
            wide.gate.query.projection.swiglu_clamp,
            direct.gate.query.projection.swiglu_clamp
        );
        let spec = cubins(&[FLASH]).unwrap().remove(0).1;
        assert_eq!(spec.name, CUBIN);
        assert_eq!(spec.entries.len(), 12);
        assert!(cubins(&[]).unwrap().is_empty());
    }
}
