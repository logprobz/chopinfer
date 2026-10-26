//! The served split quantizers handle one-row prefill; decode uses its folds.
use super::query_window::{decode_steps, symbol};
use super::*;

fn prefill_case(layout: KvLayout) -> Vec<String> {
    {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, layout);
        let mut instance = create(&host, 8);
        assert!(
            host.device
                .module_cubins
                .borrow()
                .values()
                .any(|name| crate::sm120::ceil_split::is_cubin(name))
        );
        let from = host.device.journal.borrow().len();
        step(
            &mut instance,
            &prompt(1, 0, 0, 1, true),
            &mut Views::default(),
        );
        let lines = host.device.journal.borrow()[from..]
            .iter()
            .filter(|l| l.starts_with("launch "))
            .cloned()
            .collect();
        instance.retire_request(1).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        lines
    }
}

#[test]
fn one_row_prefill_quantizes_both_wo_inputs_with_split_twins() {
    for layout in KV_LAYOUTS {
        let lines = prefill_case(layout);
        for name in [
            "chop_deepseek41_dense_ceil_w8192_split8_v1",
            "chop_deepseek41_dense_ceil_f32_w2048_split8_v1",
        ] {
            assert_eq!(
                lines.iter().filter(|line| symbol(line) == name).count(),
                40 * 4
            );
        }
    }
}

#[test]
fn one_row_decode_quantizes_wo_inputs_in_its_status_folds() {
    for layout in KV_LAYOUTS {
        let (_, steps) = decode_steps(layout);
        assert_eq!(
            steps[0]
                .iter()
                .filter(|line| symbol(line).starts_with("chop_deepseek41_dense_ceil_w8192_sliced_"))
                .count(),
            40 * 4
        );
        assert!(
            steps
                .iter()
                .flatten()
                .all(|line| !symbol(line).ends_with("_split8_v1"))
        );
    }
}
