//! The `dspark_moe_direct` candidate: drafter forwards whose block rows the
//! drafter table admits run each stage's routed experts through the direct
//! path instead of the expert chains, and keep every other launch.
use super::*;
use model_deepseek41_contract::execution_options::DSPARK_MOE_DIRECT;

const CHAIN: &str = "deepseek41_chop_moe_chain_mxfp4_a8ceil_before_";
const DIRECT: &str = "chop_deepseek41_moe_bandwidth_direct_e128_k3_";

/// A drafter launch as the program export lists it.
type Launch = (String, u32, [u32; 3], [u32; 3], u32, bool);

fn count(program: &[Launch], what: &str) -> usize {
    program
        .iter()
        .filter(|(symbol, ..)| symbol.starts_with(what))
        .count()
}

/// Every forward of 2 to 12 items (10 to 60 block rows) trades each
/// stage's expert chains for the direct path; one item (5 rows) is direct
/// already. The launches before each stage's routed experts and after its
/// expert sum keep their symbols and order, and so do the collectives.
#[test]
fn drafter_moe_takes_the_direct_path_at_admitted_rows() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (_, mut served) = create_with_limits(shape, 32, 64);
    let (_, mut direct) = create_with_options(shape, 32, 64, &[(DSPARK_MOE_DIRECT, true.into())]);
    let stages = FLASH.dspark_stages() as usize;
    for items in 1..=12 {
        let lowered = |instance: &BundleInstanceV1| {
            crate::instance::dspark_drafter_launches(instance.handle(), items)
                .unwrap_or_else(|e| panic!("{items} items: {e}"))
        };
        let (before, after) = (lowered(&served), lowered(&direct));
        if items == 1 {
            assert_eq!(before, after);
            continue;
        }
        for (old, new) in before.iter().zip(&after) {
            assert_eq!(count(new, CHAIN), 0, "{items} items");
            assert!(count(old, CHAIN) >= 2 * stages, "{items} items");
            assert_eq!(count(old, DIRECT), 0, "{items} items");
            assert!(count(new, DIRECT) >= 2 * stages, "{items} items");
            let collectives = |p: &[Launch]| {
                p.iter()
                    .filter(|l| l.5)
                    .map(|l| l.0.clone())
                    .collect::<Vec<_>>()
            };
            assert_eq!(collectives(old), collectives(new), "{items} items");
            // Outside the routed experts every launch is unchanged: the
            // attention sublayers, the dense shared expert and the head.
            let outside = |p: &[Launch]| {
                p.iter()
                    .filter(|l| {
                        !l.0.contains("_moe_")
                            && !l.0.starts_with("chop_moe_")
                            && !l.0.contains("dspark_router")
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            };
            assert_eq!(outside(old), outside(new), "{items} items");
        }
    }
    served.destroy().unwrap();
    direct.destroy().unwrap();
}
