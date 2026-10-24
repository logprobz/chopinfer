use super::*;
use crate::dspark::device::{batch_commit_enabled, with_batch_commit_setting};

#[test]
fn dspark_prefill_defaults_match_explicit_opt_ins() {
    assert!(batch_commit_enabled(), "batched commit must default on");
    let run = |setting| {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        host.speculation.set(Some((5, 32)));
        let mut instance = with_batch_commit_setting(setting, || {
            create_backend_with_setting(
                &host,
                setting,
                true,
                dspark::host::MappedHost::with_delayed_completion(Rc::clone(&host)),
            )
            .unwrap()
        });
        let enabled = setting != Some(false);
        assert_eq!(
            indices(instance.handle()).len(),
            if enabled { 2 } else { 0 }
        );
        for line in host.logs.borrow().iter() {
            if let Some(record) = line.strip_prefix("startup memory: ") {
                let record: serde_json::Value = serde_json::from_str(record).unwrap();
                let bytes = record["bytes"]["prefill_engram"].as_u64().unwrap_or(0);
                let expected = if enabled && record["rank"] == 0 {
                    2 * 786_432
                } else {
                    0
                };
                assert_eq!(bytes, expected);
                assert_eq!(record["after_kv"], 0, "gather IDs must precede KV sizing");
            }
        }
        crate::instance::dspark_step_trace_for_test(instance.handle(), true);
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
        assert_eq!(record["projection_passes"], if enabled { 3 } else { 6 });
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        host.journal.borrow().clone()
    };
    let default = run(None);
    assert_eq!(
        default,
        run(Some(true)),
        "default must select the GPU-gated paths"
    );
    assert_ne!(
        default,
        run(Some(false)),
        "explicit opt-outs must retain the previous paths"
    );
}

#[test]
fn autoregressive_prefill_keeps_gathers_opt_in() {
    for setting in [None, Some(false), Some(true)] {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let mut instance = create_backend_with_setting(
            &host,
            setting,
            true,
            dspark::host::MappedHost::new(Rc::clone(&host)),
        )
        .unwrap();
        assert_eq!(
            indices(instance.handle()).len(),
            if setting == Some(true) { 2 } else { 0 }
        );
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
