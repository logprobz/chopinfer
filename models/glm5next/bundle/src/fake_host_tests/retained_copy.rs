//! Regular retained and prompt-ending rows use one copy launch per rank.

use super::*;
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

fn prompts() -> Vec<u8> {
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..32)
            .map(|id| BatchItemV1::new(id + 1, false, 8, true, vec![1; 8]))
            .collect(),
        (0..32)
            .map(|id| BatchAppendV1::new(0, 8, 8, vec![id], None))
            .collect(),
    )
    .with_state_slots((0..32).collect())
    .encode()
    .unwrap()
}

#[test]
fn regular_prompt_rows_and_retains_batch_per_rank() {
    let format = Shape::TP4.kv_formats()[1];
    let host = FakeHost::with_format((3, 32), format);
    let mut instance = create_over(&host, format);
    crate::instance::run_lowered_stages_alone(instance.handle());
    let start = host.device.journal.borrow().len();
    instance
        .execute(&prompts(), &mut DraftSink::default())
        .unwrap();
    let copies = |start| {
        issued(&host, start)
            .into_iter()
            .filter(|line| {
                line.starts_with("launch ")
                    && line.contains(model_glm5next_contract::sm120::engines::SLICE_SYMBOL)
                    && line.contains("grid_x: 32,")
            })
            .count()
    };
    // Other row slices compact the router's expert scores. The prompt's
    // staging launch alone copies the hidden width at 32 rows here.
    let staged = issued(&host, start).into_iter().filter(|line| {
        line.starts_with("launch ")
            && line.contains(model_glm5next_contract::sm120::engines::SLICE_SYMBOL)
            && line.contains("ScalarU32(4096), ScalarU32(32768), ScalarU32(1), ScalarU32(0), ScalarU32(4096)")
    }).count();
    assert_eq!(staged, 4, "one strided staging launch per rank");
    let start = host.device.journal.borrow().len();
    instance.commit().unwrap();
    assert_eq!(copies(start), 4, "one retained-history launch per rank");
    assert_eq!(count(&host, start, "copy "), 0);
    instance.destroy().unwrap();
}

#[test]
fn a_partially_submitted_retained_copy_poisons_commit() {
    let format = Shape::TP4.kv_formats()[1];
    let host = FakeHost::with_format((3, 32), format);
    let mut instance = create_over(&host, format);
    crate::instance::run_lowered_stages_alone(instance.handle());
    instance
        .execute(&prompts(), &mut DraftSink::default())
        .unwrap();
    let symbol = model_glm5next_contract::sm120::engines::SLICE_SYMBOL;
    *host.fail_launch_on_rank.borrow_mut() = Some((symbol.into(), 1));
    let start = host.device.journal.borrow().len();
    assert_eq!(
        instance.commit().unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::HostError)
    );
    assert!(host.fail_launch_on_rank.borrow().is_none());
    assert_eq!(
        issued(&host, start)
            .iter()
            .filter(|line| line.contains(symbol))
            .count(),
        1
    );
    let end = host.device.journal.borrow().len();
    assert!(instance.commit().is_err());
    assert!(instance.cancel().is_err());
    assert!(
        instance
            .execute(&prompts(), &mut DraftSink::default())
            .is_err()
    );
    assert!(instance.destroy().is_err());
    assert_eq!(host.device.journal.borrow().len(), end);
}
