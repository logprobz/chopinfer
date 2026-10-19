use super::*;
fn identity() -> Identity {
    Identity {
        request: 7,
        slot: 2,
        lease: 3,
    }
}
fn request() -> Request {
    Request {
        identity: identity(),
        k: 5,
        remaining: 6,
        terminal_stop_ids: vec![],
        top_logprobs: 1,
        constrained: false,
        history_penalties: false,
    }
}
fn packet(batch: &Batch, proposed: &[[u32; 5]]) -> (Vec<u8>, Vec<u8>) {
    let mut samples = vec![0x5a; batch.op.sample_rows as usize * OUTPUT_BYTES];
    for item in &batch.items {
        for r in 0..=item.k {
            let row =
                &mut samples[(item.first_sample + r) as usize * OUTPUT_BYTES..][..OUTPUT_BYTES];
            row[..4].copy_from_slice(&(10 + r).to_le_bytes());
            row[4..8].fill(0);
            row[8..16].copy_from_slice(&(-0.25_f64).to_le_bytes());
            row[16..20].copy_from_slice(&13_u32.to_le_bytes());
            row[96..104].copy_from_slice(&(-0.5_f64).to_le_bytes());
        }
    }
    let result = oracle(
        batch.op,
        Input {
            items: &batch.items,
            proposals: proposed,
            samples: &samples,
            stops: &batch.stops,
        },
    )
    .unwrap()
    .into_iter()
    .flat_map(|row| row.encode())
    .collect();
    (result, samples)
}
#[test]
fn matched_eos_preserves_exact_count_and_sampler_payload_bits() {
    let mut adapter = Adapter::default();
    let mut req = request();
    req.terminal_stop_ids = vec![10];
    let batch = adapter.begin(100, &[req]).unwrap();
    let epoch = batch.epoch;
    assert_eq!(batch.item_bytes().len(), 40);
    let (result, samples) = packet(batch, &[[10, 11, 12, 13, 14]]);
    let complete = adapter
        .finish(epoch, &[identity()], &result, &samples)
        .unwrap();
    assert_eq!(
        (complete[0].decision.accepted, complete[0].decision.kept),
        (1, 1)
    );
    assert_eq!(complete[0].records, &samples[..OUTPUT_BYTES]);
    assert!(
        adapter
            .finish(epoch, &[identity()], &result, &samples)
            .is_err()
    );
}
#[test]
fn cancellation_and_slot_restore_refuse_stale_completions() {
    let mut adapter = Adapter::default();
    let batch = adapter.begin(100, &[request()]).unwrap();
    let old = batch.epoch;
    let (result, samples) = packet(batch, &[[10, 11, 12, 13, 14]]);
    adapter.cancel();
    let epoch = adapter.begin(100, &[request()]).unwrap().epoch;
    assert_ne!(old, epoch);
    assert_eq!(
        adapter
            .finish(old, &[identity()], &result, &samples)
            .unwrap_err(),
        Refusal("stale DSpark acceptance epoch")
    );
    assert_eq!(adapter.pending.as_ref().unwrap().epoch, epoch);
    let (result, samples) = packet(adapter.pending.as_ref().unwrap(), &[[10, 11, 12, 13, 14]]);
    let moved = Identity {
        lease: 4,
        ..identity()
    };
    assert_eq!(
        adapter
            .finish(epoch, &[moved], &result, &samples)
            .unwrap_err(),
        Refusal("DSpark acceptance slot lease changed")
    );
    assert!(adapter.pending.is_none());
}
#[test]
fn rejected_suffix_still_validates_top_logprob_payload() {
    let mut adapter = Adapter::default();
    let batch = adapter.begin(100, &[request()]).unwrap();
    let epoch = batch.epoch;
    let (result, mut samples) = packet(batch, &[[99, 11, 12, 13, 14]]);
    assert_eq!(Decision::decode(&result).unwrap().accepted, 0);
    samples[5 * OUTPUT_BYTES + 96..5 * OUTPUT_BYTES + 104].copy_from_slice(&f64::NAN.to_le_bytes());
    assert_eq!(
        adapter
            .finish(epoch, &[identity()], &result, &samples)
            .unwrap_err(),
        Refusal("DSpark sampler payload is invalid")
    );
}
#[test]
fn unsupported_history_sampling_and_corrupt_result_refuse_before_commit() {
    assert!(!crate::Kernels::loads(
        crate::sm120::acceptance_entries::CUBIN
    ));
    for constrained in [false, true] {
        let mut req = request();
        req.constrained = constrained;
        req.history_penalties = !constrained;
        assert!(
            Adapter::default()
                .begin(100, &[req])
                .err()
                .unwrap()
                .0
                .contains("identity-penalty")
        );
    }
    let mut adapter = Adapter::default();
    let batch = adapter.begin(100, &[request()]).unwrap();
    let epoch = batch.epoch;
    let (mut result, samples) = packet(batch, &[[10, 11, 12, 13, 14]]);
    result[12..16].copy_from_slice(&7_u32.to_le_bytes());
    assert_eq!(
        adapter
            .finish(epoch, &[identity()], &result, &samples)
            .unwrap_err(),
        Refusal("DSpark acceptance counts are invalid")
    );
}
