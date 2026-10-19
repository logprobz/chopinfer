use super::*;
use crate::ops::{with_bf16_gather, with_dma_setting};

fn capture_default(setting: Option<&str>, rows: u64, bf16: bool) -> Vec<String> {
    let host = host();
    let mut instance = with_bf16_gather(bf16, || {
        with_dma_setting(setting, || with_mode(0, || create(&host, 4)))
    });
    host.device.journal.borrow_mut().clear();
    step(
        &mut instance,
        &prompt(7, 0, 0, rows, true),
        &mut Views::default(),
    );
    let journal = host.device.journal.borrow().clone();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    journal
}

#[test]
fn automatic_full_chunk_matches_the_qualified_explicit_dma_schedule() {
    let automatic = capture_default(None, 8192, true);
    let qualified = capture_default(Some("1"), 8192, true);
    assert_eq!(automatic.len(), qualified.len());
    for (index, (actual, expected)) in automatic.iter().zip(&qualified).enumerate() {
        assert_eq!(actual, expected, "journal entry {index}");
    }
    assert_eq!(audit(&automatic).unwrap(), 81 * (24 * 2 + 4 * 3));
    assert_eq!(
        automatic
            .iter()
            .filter(|line| line.starts_with("peer-copy "))
            .count(),
        81 * 24
    );
}

#[test]
fn automatic_dma_keeps_unmeasured_rows_and_both_transport_controls() {
    for (setting, rows, bf16) in [
        (None, 4096, true),
        (None, 8191, true),
        (Some("0"), 8192, true),
        (None, 8192, false),
    ] {
        let journal = capture_default(setting, rows, bf16);
        assert!(
            !journal
                .iter()
                .any(|line| line.starts_with("peer-copy ") || line.contains(SUM)),
            "setting={setting:?}, rows={rows}, bf16={bf16}"
        );
    }
}

#[test]
fn automatic_dma_avoids_unqualified_instances_and_unused_skip_buffers() {
    for (ranks, layout, chunk) in [
        (Ranks::TP2, KV_LAYOUTS[0], 0),
        (Ranks::TP4, KV_LAYOUTS[1], 0),
        (Ranks::TP4, KV_LAYOUTS[0], 8192),
    ] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let mut record = crate::config::tests::record(&shape, layout, 4, false);
        let at = record
            .windows(MODEL_OPTIONS_TAG_V1.len())
            .position(|window| window == MODEL_OPTIONS_TAG_V1)
            .unwrap();
        let begin = at + MODEL_OPTIONS_TAG_V1.len() + 4;
        let mut options: serde_json::Value = serde_json::from_slice(&record[begin..]).unwrap();
        options["prefill_chunk_tokens"] = chunk.into();
        options["prefill_upper_skip"] = (chunk != 0).into();
        let options = serde_json::to_vec(&options).unwrap();
        record.truncate(at + MODEL_OPTIONS_TAG_V1.len());
        record.extend_from_slice(&(options.len() as u32).to_le_bytes());
        record.extend_from_slice(&options);
        let created = |setting| {
            let host = FakeHost::new(shape, layout);
            let mut instance = with_bf16_gather(true, || {
                with_dma_setting(setting, || {
                    api().create(HostServicesTableV1::new(Rc::clone(&host)), &record)
                })
            })
            .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
            let journal = host.device.journal.borrow().clone();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
            journal
        };
        let automatic = created(None);
        let disabled = created(Some("0"));
        // Exact creation journals also exclude an unused inbox, lifetime
        // stream, events or DMA module in the negotiated skip instance.
        assert_eq!(automatic.len(), disabled.len());
        for (index, (actual, expected)) in automatic.iter().zip(&disabled).enumerate() {
            assert_eq!(actual, expected, "creation entry {index}");
        }
        assert!(!automatic.iter().any(|line| line.contains(SUM)));
    }
}
