use super::*;
use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::prefill::{Boundary, encode};

type Sink = RecordingSink<super::conformance::Limits>;

// Cold prompt lengths whose whole retained dispatches pack to each main width.
// Every prompt exceeds the 134-row history capacity and projects at commit.
pub(super) const BOUNDARIES: &[(u32, &[u32])] = &[
    (128, &[256]),
    (129, &[257]),
    (130, &[258]),
    (192, &[192]),
    (255, &[255]),
    (256, &[256, 256]),
    (257, &[256, 257]),
    (383, &[255, 256]),
    (384, &[192, 192]),
    (385, &[192, 193]),
    (510, &[255, 255]),
    (511, &[255, 256, 256]),
    (512, &[192, 192, 256]),
];

#[test]
fn served_main_projection_uses_every_qualified_inner_boundary() {
    for &(inner_rows, lengths) in BOUNDARIES {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.kv_pages.set(32 * 4200);
        host.drafter.set(true);
        host.speculation.set(Some((5, 6)));
        let mut record = crate::config::tests::record(&shape, host.kv, lengths.len() as u32, false);
        for (name, value) in [
            ("prefill_chunk_tokens", serde_json::json!(8192)),
            ("prefill_upper_skip", serde_json::json!(true)),
            ("diagnostic_step_logs", true.into()),
        ] {
            record = crate::config::tests::with_option(record, name, value);
        }
        let services = HostServicesTableV1::new(Rc::new(super::dspark::host::MappedHost::new(
            Rc::clone(&host),
        )));
        let mut instance = api()
            .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
            .unwrap()
            .create(services, &record)
            .unwrap_or_else(|error| panic!("create: {error:?} {:?}", host.logs.borrow()));
        let first = host.device.journal.borrow().len();
        let tokens: Vec<Vec<u32>> = lengths.iter().map(|&n| vec![7; n as usize]).collect();
        let items: Vec<_> = tokens
            .iter()
            .enumerate()
            .map(|(index, values)| {
                (
                    index as u64 + 1,
                    index as u64,
                    0,
                    values.as_slice(),
                    bundle_runtime::fake_host::conformance::Phase::Prefill,
                )
            })
            .collect();
        let boundaries: Vec<_> = lengths
            .iter()
            .enumerate()
            .map(|(index, &rows)| Boundary {
                request_id: index as u64 + 1,
                prompt_tokens: rows.into(),
                chunk_end: rows.into(),
                chunk_tokens: 8192,
            })
            .collect();
        let request = encode(&super::conformance::step(&items, false), &boundaries).unwrap();
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
        let selected = (129..=511).contains(&inner_rows);
        assert!(
            main.iter()
                .all(|line| line.contains("_compact_") == selected),
            "{main:#?}"
        );
        assert_eq!(main.len(), if selected { 24 } else { 12 });
        drop(lines);
        let records: Vec<serde_json::Value> = host
            .logs
            .borrow()
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|row| row["schema"] == "chopinfer.deepseek41.commit-shape.v1")
            .collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["projection_passes"], 3);
        assert_eq!(records[0]["projection_rows"], 3 * inner_rows);
        for request in 1..=lengths.len() as u64 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
