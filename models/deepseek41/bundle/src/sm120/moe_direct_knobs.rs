//! The knob candidates of the backbone bank's absorbing direct path
//! (`kernels_sm120::moe::chained::knobs`): the 32- to 64-row buckets' gate
//! queue, two-strip down queue and finishers under each cache-policy knob
//! set, and the gate queue that finishes its own strips, with raw and
//! width-1 tile scales; and the same entries at the package's verify rows
//! for the wide verify buckets (`moe_direct_wide`).
use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::{floor::Entry, knobs};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "routed_moe_direct_knobs";
pub const CUBIN: &str = "bandwidth_moe_deepseek41_knobs_sm120";
pub const WIDE_CUBIN: &str = "bandwidth_moe_deepseek41_wide_knobs_sm120";

/// The backbone bank's 64-row direct entry.
pub fn entry(g: &Geometry) -> Result<Entry, String> {
    super::moe_direct_kernels::entries(g)?
        .into_iter()
        .find(|e| e.experts() == model_deepseek41_contract::moe::expert_sets(g)[0].count)
        .ok_or_else(|| "missing backbone direct entry".into())
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    let spec = |name, e: Entry| knobs::cubin_spec(name, &[e]).map_err(|e| e.to_string());
    Ok(vec![
        (FAMILY, spec(CUBIN, entry(&FLASH)?)?),
        (
            FAMILY,
            spec(WIDE_CUBIN, super::moe_direct_wide::entry(&FLASH)?)?,
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both modules hold every knob entry of their bank: the 64-row direct
    /// entry and the wide entry at the package's verify rows.
    #[test]
    fn the_knob_modules_hold_each_bank_once() {
        let specs = cubins(&[FLASH]).unwrap();
        assert_eq!(
            specs
                .iter()
                .map(|(f, s)| (*f, s.name.as_str(), s.entries.len()))
                .collect::<Vec<_>>(),
            [(FAMILY, CUBIN, 31), (FAMILY, WIDE_CUBIN, 31)]
        );
        assert!(
            specs[0]
                .1
                .entries
                .iter()
                .all(|e| e.symbol.contains("_e384_k6_"))
        );
        assert!(
            specs[1]
                .1
                .entries
                .iter()
                .all(|e| e.symbol.contains("_t192_e384_k6_"))
        );
        assert!(cubins(&[]).unwrap().is_empty());
    }

    /// No instance loads a knob module before an option selects it.
    #[test]
    fn no_instance_loads_the_knob_family_yet() {
        use bundle_runtime::geometry::Ranks;
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = crate::shape::Shape::new(&FLASH, ranks).unwrap();
            for rows in [32, 64, 65, 128, 192] {
                let shape = shape.with_verify_rows(rows);
                for wide in [false, true] {
                    let shape = shape.with_wide_verify(wide);
                    assert!(!crate::Kernels::loads(CUBIN, &shape));
                    assert!(!crate::Kernels::loads(WIDE_CUBIN, &shape));
                }
            }
        }
    }
}
