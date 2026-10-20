//! The candidate changes only transfer scheduling and explicit layer joins.
use super::*;
use crate::engram_rows::{EngramRows, Step};
use crate::instance::prefill_engram::{indices, with_enabled};
use model_deepseek41_contract::own_ops::engram::HashTables;
use runtime_bundle_sdk::prefill::{Boundary, encode};

fn create_mapped(host: &Rc<Host>, on: bool, skip: bool) -> BundleInstanceV1 {
    try_create_mapped(host, on, skip).unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()))
}

fn try_create_mapped(
    host: &Rc<Host>,
    on: bool,
    skip: bool,
) -> Result<BundleInstanceV1, BundleAbiError> {
    host.drafter.set(true);
    let mut record = crate::config::tests::record(&host.shape, host.kv, 1, false);
    let tag = record
        .windows(MODEL_OPTIONS_TAG_V1.len())
        .position(|x| x == MODEL_OPTIONS_TAG_V1)
        .unwrap();
    let start = tag + MODEL_OPTIONS_TAG_V1.len();
    let mut options: serde_json::Value = serde_json::from_slice(&record[start + 4..]).unwrap();
    options["prefill_chunk_tokens"] = serde_json::json!(8192);
    options["prefill_upper_skip"] = serde_json::json!(skip);
    record.truncate(start);
    let options = serde_json::to_vec(&options).unwrap();
    record.extend_from_slice(&(options.len() as u32).to_le_bytes());
    record.extend_from_slice(&options);
    with_enabled(on, || {
        api()
            .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
            .unwrap()
            .create(
                HostServicesTableV1::new(Rc::new(dspark::host::MappedHost::new(Rc::clone(host)))),
                &record,
            )
    })
}

fn launches(host: &Host) -> BTreeMap<String, Vec<String>> {
    let mut result = BTreeMap::<String, Vec<String>>::new();
    for line in host
        .journal
        .borrow()
        .iter()
        .filter(|s| s.starts_with("launch "))
    {
        let stream = line.split_whitespace().nth(1).unwrap().to_owned();
        result.entry(stream).or_default().push(line.clone());
    }
    result
}

#[test]
fn eager_and_upper_skip_join_each_layer_without_changing_launches() {
    for (ranks, kv, rows, skip) in [
        (Ranks::TP1, KV_LAYOUTS[0], 64, false),
        (Ranks::TP2, KV_LAYOUTS[1], 257, true),
        (Ranks::TP4, KV_LAYOUTS[0], 8192, true),
        (Ranks::TP1, KV_LAYOUTS[1], 8191, true),
    ] {
        let mut reference = None;
        for on in [false, true] {
            let host = FakeHost::new(Shape::new(&FLASH, ranks).unwrap(), kv);
            let mut instance = create_mapped(&host, on, skip);
            let spans = indices(instance.handle());
            assert_eq!(spans.len(), if on { 2 } else { 0 });
            let begin = host.journal.borrow().len();
            let request = encode(
                &prompt(7, 0, 0, rows, true),
                &[Boundary {
                    request_id: 7,
                    prompt_tokens: rows,
                    chunk_end: rows,
                    chunk_tokens: 8192,
                }],
            )
            .unwrap();
            instance
                .execute_prefill(&request, &mut Views::default())
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
            if on {
                let logs = host.logs.borrow();
                let metric: serde_json::Value = serde_json::from_str(
                    logs.iter()
                        .find(|s| s.contains("\"phase\":\"prefill-engram\""))
                        .expect("candidate actually executed"),
                )
                .unwrap();
                assert_eq!(metric["rows"], rows);
                assert_eq!(metric["start"], 0);
                assert_eq!(metric["waits"], ranks.get() * 2);
                assert_eq!(metric["layers"][0]["layer"], 1);
                assert_eq!(metric["layers"][1]["layer"], 14);
                let journal = host.journal.borrow();
                let step = &journal[begin..];
                let first_launch = step.iter().position(|s| s.starts_with("launch ")).unwrap();
                assert_eq!(
                    step[..first_launch]
                        .iter()
                        .filter(|s| s.starts_with("async mapped ids "))
                        .count(),
                    2
                );
                let first = step
                    .iter()
                    .position(|s| s.contains("layers.1.engram.embed.weight"))
                    .unwrap();
                let second = step
                    .iter()
                    .position(|s| s.contains("layers.14.engram.embed.weight"))
                    .unwrap();
                assert!(step[..first].iter().any(|s| s.starts_with("launch ")));
                assert!(
                    step[first..second]
                        .iter()
                        .filter(|s| s.starts_with("launch "))
                        .count()
                        > 100
                );
                assert_eq!(
                    step.iter()
                        .filter(|s| s.starts_with("async mapped complete"))
                        .count(),
                    2
                );
                assert_eq!(
                    step.iter()
                        .filter(|s| s.starts_with("async mapped wait "))
                        .count(),
                    2 * ranks.count()
                );
                let hash = HashTables::of(&FLASH).unwrap();
                let columns = hash.columns();
                let oracle = EngramRows::new(hash);
                let tokens: Vec<_> = (0..rows as u32).collect();
                let hashed = oracle
                    .hash(&[Step {
                        request: 7,
                        tokens: &tokens,
                        text: None,
                    }])
                    .unwrap();
                for (layer, (_, span)) in spans.iter().enumerate() {
                    let wanted: Vec<_> = hashed
                        .ids()
                        .chunks_exact(columns * 2)
                        .flat_map(|row| &row[layer * columns..(layer + 1) * columns])
                        .flat_map(|id| id.to_le_bytes())
                        .collect();
                    assert_eq!(
                        host.uploads.borrow()[&(span.buffer.raw, span.offset)],
                        wanted
                    );
                }
            }
            instance.commit().unwrap();
            let before_decode = host.journal.borrow().len();
            instance
                .execute(&decode(&[(7, 0, rows, 7)]), &mut Views::default())
                .unwrap();
            instance.commit().unwrap();
            assert!(
                !host.journal.borrow()[before_decode..]
                    .iter()
                    .any(|s| s.starts_with("async mapped ids "))
            );
            let observed = launches(&host);
            if let Some(reference) = &reference {
                assert_eq!(&observed, reference, "ranks={} rows={rows}", ranks.get());
            } else {
                reference = Some(observed);
            }
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

#[test]
fn layer_handles_reuse_after_partial_prefill_and_cancel() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create_mapped(&host, true, true);
    let initial = indices(instance.handle());
    for (start, rows) in [(0, 8192), (8192, 17)] {
        let request = encode(
            &prompt(7, 0, start, rows, start + rows == 8209),
            &[Boundary {
                request_id: 7,
                prompt_tokens: 8209,
                chunk_end: start + rows,
                chunk_tokens: 8192,
            }],
        )
        .unwrap();
        instance
            .execute_prefill(&request, &mut Views::default())
            .unwrap();
        assert_eq!(indices(instance.handle()), initial);
        if start == 0 {
            instance.commit().unwrap();
        } else {
            instance.cancel().unwrap();
        }
    }
    let request = encode(
        &prompt(8, 0, 0, 17, true),
        &[Boundary {
            request_id: 8,
            prompt_tokens: 17,
            chunk_end: 17,
            chunk_tokens: 8192,
        }],
    )
    .unwrap();
    instance
        .execute_prefill(&request, &mut Views::default())
        .unwrap();
    instance.commit().unwrap();
    assert_eq!(
        host.logs
            .borrow()
            .iter()
            .filter(|s| s.contains("\"phase\":\"prefill-engram\""))
            .count(),
        3
    );
    instance.retire_request(8).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn private_id_allocation_failures_release_prior_resources() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create_mapped(&host, true, true);
    let count = host
        .journal
        .borrow()
        .iter()
        .filter(|s| s.starts_with("allocate "))
        .count();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    for at in [count - 2, count - 1] {
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.device.fail_allocate.set(Some(at as u32));
        assert!(try_create_mapped(&host, true, true).is_err());
        assert_eq!(host.device.fail_allocate.get(), None);
        // The fake increments acquisition before injecting allocation failure;
        // no handle is returned for that one counter increment.
        assert_eq!(host.device.allocation_failures.get(), 1);
        assert_eq!(host.acquired.get(), host.released.get() + 1);
        assert_eq!(host.device.buffer_order.borrow().len(), at);
        for id in host.device.buffer_order.borrow().iter() {
            assert!(!host.device.extents.borrow().contains_key(id));
        }
    }
}

#[test]
fn enabled_prefill_flag_preserves_dspark_resources() {
    let mut reference = None;
    for on in [false, true] {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        host.speculation.set(Some((5, 32)));
        let mut instance = create_mapped(&host, on, true);
        assert!(indices(instance.handle()).is_empty());
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        let journal = host.journal.borrow().clone();
        if let Some(reference) = &reference {
            assert_eq!(&journal, reference);
        } else {
            reference = Some(journal);
        }
    }
}
