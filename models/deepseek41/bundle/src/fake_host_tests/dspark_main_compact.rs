use super::*;
use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::prefill::{Boundary, encode};

type Sink = RecordingSink<super::conformance::Limits>;

#[test]
fn candidate_changes_only_the_qualified_main_projection() {
    for enabled in [false, true] {
        let (host, mut instance) = super::indexer_workloads::skip_host(
            Ranks::TP4,
            true,
            true,
            1,
            &[("dspark_main_compact", enabled.into())],
        );
        let first = host.device.journal.borrow().len();
        let request = encode(
            &prompt(1, 0, 0, 192, true),
            &[Boundary {
                request_id: 1,
                prompt_tokens: 192,
                chunk_end: 192,
                chunk_tokens: 8192,
            }],
        )
        .unwrap();
        instance
            .execute_prefill(&request, &mut Sink::default())
            .unwrap();
        instance.commit().unwrap();
        let lines = host.device.journal.borrow();
        let main: Vec<_> = lines[first..]
            .iter()
            .filter(|line| {
                line.starts_with("launch ") && line.contains("_m8192_n5120_k15360_g1_f32_v2")
            })
            .collect();
        assert!(
            !main.is_empty(),
            "the served step must execute main projections"
        );
        assert!(
            main.iter()
                .all(|line| line.contains("_compact_") == enabled),
            "{main:#?}"
        );
        assert_eq!(main.len(), if enabled { 24 } else { 12 });
        drop(lines);
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
