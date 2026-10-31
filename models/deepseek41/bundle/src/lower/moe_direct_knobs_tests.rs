use crate::shape::Shape;
use crate::sm120::moe_direct_knobs::{CUBIN, WIDE_CUBIN};
use bundle_runtime::geometry::Ranks;
use model_deepseek41_contract::execution_options::MoeDirect;
use model_deepseek41_contract::geometry::FLASH;

/// An instance loads a knob module exactly where a `moe_direct_*`
/// option other than loads alone meets a bucket of the absorbing path:
/// the 64-row module on Flash TP4, the wide one with the wide verify
/// buckets as well.
#[test]
fn instances_load_the_knob_modules_only_under_an_option() {
    let options = [
        MoeDirect::default(),
        MoeDirect {
            retire: true,
            ..MoeDirect::default()
        },
        MoeDirect {
            stream: true,
            keep: true,
            retire: true,
            loads: true,
        },
        MoeDirect {
            loads: true,
            ..MoeDirect::default()
        },
    ];
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        for rows in [32, 64, 65, 128, 192] {
            for wide in [false, true] {
                for option in options {
                    let shape = shape
                        .with_verify_rows(rows)
                        .with_wide_verify(wide)
                        .with_moe_knobs(option);
                    let on = (option.retire || option.stream) && ranks == Ranks::TP4;
                    assert_eq!(crate::Kernels::loads(CUBIN, &shape), on);
                    assert_eq!(
                        crate::Kernels::loads(WIDE_CUBIN, &shape),
                        on && wide && rows > 64
                    );
                }
            }
        }
    }
}
