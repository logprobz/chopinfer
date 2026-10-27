use super::*;

#[test]
fn dspark_prefill_defaults_to_batched_commit_and_prepared_gathers() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    host.speculation.set(Some((5, 32)));
    let mut instance = create_backend_with_options(
        &host,
        true,
        dspark::host::MappedHost::with_delayed_completion(Rc::clone(&host)),
        &[(
            model_deepseek41_contract::execution_options::DIAGNOSTIC_STEP_LOGS,
            true.into(),
        )],
    )
    .unwrap();
    assert_eq!(indices(instance.handle()).len(), 2);
    for line in host.logs.borrow().iter() {
        if let Some(record) = line.strip_prefix("startup memory: ") {
            let record: serde_json::Value = serde_json::from_str(record).unwrap();
            let bytes = record["bytes"]["prefill_engram"].as_u64().unwrap_or(0);
            let expected = if record["rank"] == 0 { 2 * 786_432 } else { 0 };
            assert_eq!(bytes, expected);
            assert_eq!(record["after_kv"], 0, "gather IDs must precede KV sizing");
        }
    }
    let request = encode(
        &prompt(7, 0, 0, 193, true),
        &[Boundary {
            request_id: 7,
            prompt_tokens: 193,
            chunk_end: 193,
            chunk_tokens: 8192,
        }],
    )
    .unwrap();
    instance
        .execute_prefill(&request, &mut Views::default())
        .unwrap();
    instance.commit().unwrap();
    let record = host
        .logs
        .borrow()
        .iter()
        .find_map(|line| {
            let row: serde_json::Value = serde_json::from_str(line).ok()?;
            (row["schema"] == "chopinfer.deepseek41.commit-shape.v1").then_some(row)
        })
        .unwrap();
    assert_eq!(record["projection_passes"], 3);
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("\"phase\":\"prefill-engram\""))
    );
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn autoregressive_prefill_defaults_to_prepared_gathers() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance =
        create_backend(&host, true, dspark::host::MappedHost::new(Rc::clone(&host))).unwrap();
    assert_eq!(indices(instance.handle()).len(), 2);
    let request = encode(
        &prompt(7, 0, 0, 193, true),
        &[Boundary {
            request_id: 7,
            prompt_tokens: 193,
            chunk_end: 193,
            chunk_tokens: 8192,
        }],
    )
    .unwrap();
    instance
        .execute_prefill(&request, &mut Views::default())
        .unwrap();
    instance.commit().unwrap();
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("\"phase\":\"prefill-engram\""))
    );
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
