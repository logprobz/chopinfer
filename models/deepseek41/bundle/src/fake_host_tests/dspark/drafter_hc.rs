//! The `dspark_hc_fusion` candidate: a drafter stage's attention boundary
//! and the boundary into the next stage take the backbone's fused round,
//! expansion, next mixer and input norm where the finish route admits the
//! forward's rows, and keep their chains elsewhere.
use super::*;
use model_deepseek41_contract::execution_options::DSPARK_HC_FUSION;

const MIX: &str = "chop_sinkhorn_hc_deepseek41_lagged_mix_collapse";

/// A drafter launch as the program export lists it.
type Launch = (String, u32, [u32; 3], [u32; 3], u32, bool);

fn count(program: &[Launch], what: &str) -> usize {
    program
        .iter()
        .filter(|(symbol, ..)| symbol.starts_with(what))
        .count()
}

/// Each forward of 7 to 12 items (35 to 60 block rows, the finish route's
/// admitted verify rows) fuses every boundary but the one into the head:
/// each stage's attention into its MoE and each stage's MoE into the next
/// stage's attention. Each fused boundary drops its lagged mixer, input
/// norm, quantizer, round and expansion (and an MoE input's widening) for
/// one producer and one finisher on every rank, and keeps its collectives.
/// Forwards of fewer items lower as without the option.
#[test]
fn drafter_attention_boundaries_fuse_at_admitted_rows() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (_, mut served) = create_with_limits(shape, 32, 64);
    let (_, mut fused) = create_with_options(shape, 32, 64, &[(DSPARK_HC_FUSION, true.into())]);
    let stages = FLASH.dspark_stages() as usize;
    for items in 1..=12 {
        let lowered = |instance: &BundleInstanceV1| {
            crate::instance::dspark_drafter_launches(instance.handle(), items)
                .unwrap_or_else(|e| panic!("{items} items: {e}"))
        };
        let (before, after) = (lowered(&served), lowered(&fused));
        if items < 7 {
            assert_eq!(before, after, "{items} items");
            continue;
        }
        let fused_boundaries = 2 * stages - 1;
        for (old, new) in before.iter().zip(&after) {
            assert_eq!(
                count(old, MIX) - count(new, MIX),
                fused_boundaries,
                "{items} items"
            );
            for gone in [
                "chop_row_rmsnorm_deepseek41_w5120_v1",
                "chop_deepseek41_moe_ceil_w5120",
                "chop_sinkhorn_hc_deepseek41_expand_f32_v1",
                "chop_row_rounding_round_deepseek41_w5120_v1",
            ] {
                assert_eq!(
                    count(old, gone) - count(new, gone),
                    fused_boundaries,
                    "{items} items {gone}"
                );
            }
            // Only the MoE inputs widen for the router.
            assert_eq!(
                count(old, "chop_row_rounding_widen_deepseek41_w5120_v1")
                    - count(new, "chop_row_rounding_widen_deepseek41_w5120_v1"),
                stages,
                "{items} items"
            );
            assert_eq!(
                count(new, "chop_deepseek41_hc_round_cols")
                    + count(new, "chop_deepseek41_hc_round_chain"),
                fused_boundaries,
                "{items} items: one producer per fused boundary"
            );
            assert_eq!(
                count(new, "chop_deepseek41_hc_finish"),
                fused_boundaries,
                "{items} items: one finisher per fused boundary"
            );
            assert_eq!(
                old.len() - new.len(),
                3 * fused_boundaries + stages,
                "{items} items"
            );
            // The collectives keep their count and order.
            let collectives = |p: &[Launch]| {
                p.iter()
                    .filter(|l| l.5)
                    .map(|l| l.0.clone())
                    .collect::<Vec<_>>()
            };
            assert_eq!(collectives(old), collectives(new), "{items} items");
        }
    }
    served.destroy().unwrap();
    fused.destroy().unwrap();
}
