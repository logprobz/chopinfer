use crate::shape::Shape;
use crate::sm120::moe_direct_knobs::{CUBIN, WIDE_CUBIN};
use bundle_runtime::geometry::Ranks;
use model_deepseek41_contract::execution_options::MoeDirect;
use model_deepseek41_contract::geometry::FLASH;

/// An instance loads the 64-row knob module exactly where the tuning
/// table's knob set meets a 32- to 64-row bucket of the absorbing path,
/// whatever the `moe_direct_*` options: on Flash TP4. The wide verify
/// buckets keep the plain route, so only an option loads the wide one.
#[test]
fn instances_load_the_knob_modules_with_the_absorbing_path() {
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
                    let on = ranks == Ranks::TP4;
                    assert_eq!(crate::Kernels::loads(CUBIN, &shape), on);
                    assert_eq!(
                        crate::Kernels::loads(WIDE_CUBIN, &shape),
                        on && wide && rows > 64 && (option.retire || option.stream)
                    );
                }
            }
        }
    }
}
