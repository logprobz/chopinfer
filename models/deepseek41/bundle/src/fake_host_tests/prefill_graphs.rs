//! Short text prefills split into graph segments
//! (`instance::execute::prefill_graphs`).
use super::indexer_workloads::skip_host;
use super::*;
use bundle_runtime::fake_host::RecordingSink;
use model_deepseek41_contract::execution_options::DIAGNOSTIC_STEP_LOGS;
use runtime_bundle_sdk::prefill::{Boundary, encode};

type Sink = RecordingSink<super::conformance::Limits>;

/// A whole prompt of `rows` tokens of `request` in `slot`, after `previous`
/// tokens, ending its prompt. Requests take slots of the four served items
/// in turn.
fn whole_prompt(request: u64, slot: u64, previous: u64, rows: u64) -> Vec<u8> {
    encode(
        &prompt(request, slot, previous, rows, true),
        &[Boundary {
            request_id: request,
            prompt_tokens: previous + rows,
            chunk_end: previous + rows,
            chunk_tokens: 8192,
        }],
    )
    .unwrap()
}

fn records(host: &Host) -> Vec<serde_json::Value> {
    host.logs
        .borrow()
        .iter()
        .filter(|line| line.contains("chopinfer.deepseek41.prefill-program.v1"))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// With diagnostic step logs, every upper-skip prefill logs its program
/// record: an eligible one its split and keys, a longer one only its rows.
/// Without them, none is logged.
#[test]
fn diagnostic_logs_record_each_prefill_program() {
    for logs in [false, true] {
        let (host, mut instance) = skip_host(
            Ranks::TP4,
            false,
            true,
            4,
            &[(DIAGNOSTIC_STEP_LOGS, logs.into())],
        );
        for (request, rows) in [(1, 31), (2, 31), (3, 600)] {
            instance
                .execute_prefill(
                    &whole_prompt(request, request, 0, rows),
                    &mut Sink::default(),
                )
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
            instance.commit().unwrap();
            instance.retire_request(request).unwrap();
        }
        let records = records(&host);
        if !logs {
            assert!(records.is_empty());
        } else {
            assert_eq!(records.len(), 3, "{records:?}");
            let [first, second, long] = &records[..] else {
                unreachable!()
            };
            assert_eq!(first["rows"], 31);
            assert_eq!(first["eligible"], true);
            assert_eq!(first["segments"], 7);
            assert_eq!(first["handles"], 7);
            assert_eq!(first["joins"], 2);
            assert!(first["eager_launches"].as_u64().unwrap() > 0);
            assert!(first["graph_launches"].as_u64().unwrap() > 1000);
            // The same rows in another slot are the same program.
            assert_eq!(first["body"], second["body"]);
            assert_eq!(first["head"], second["head"]);
            assert_eq!(long["rows"], 600);
            assert_eq!(long["eligible"], false);
            assert!(long.get("body").is_none());
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
