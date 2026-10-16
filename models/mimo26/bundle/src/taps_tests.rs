//! Taps on the fake host. A serving bundle, built without the
//! `diagnostic-taps` feature, refuses to create while the environment asks
//! for taps. A diagnostic bundle reads the filters at create, and a tapped
//! step launches what an untapped one does and writes every declared tap.

use super::*;

/// The create runs in a child test process, because the variable is
/// process-wide.
#[cfg(not(feature = "diagnostic-taps"))]
#[test]
fn serving_create_refuses_taps() {
    const CHILD: &str = "CHOPINFER_TEST_TAPS_CHILD";
    const NAME: &str = "fake_host_tests::taps::serving_create_refuses_taps";
    let variable = runtime_bundle_sdk::taps::TAPS_ENV.dir;
    if std::env::var_os(CHILD).is_some() {
        let host = Rc::new(FakeHost::default());
        let refused = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
            .unwrap()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &configuration())
            .err();
        assert!(
            matches!(
                refused,
                Some(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
            ),
            "{refused:?}"
        );
        let logs = host.logs.borrow();
        assert!(
            logs.iter()
                .any(|line| line.contains(variable) && line.contains("diagnostic-taps")),
            "{logs:?}"
        );
        assert!(
            host.journal.borrow().is_empty(),
            "the refused create reached the device"
        );
        return;
    }
    let directory = std::env::temp_dir().join(format!("mimo-taps-serving-{}", std::process::id()));
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([NAME, "--exact", "--test-threads=1"])
        .env(CHILD, "1")
        .env(variable, &directory)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!directory.exists(), "a refused create wrote taps");
}

#[cfg(feature = "diagnostic-taps")]
const CHILD: &str = "CHOPINFER_TEST_TAPS_CHILD";

/// The journal's device work but the taps' own: every line except their
/// stream synchronizations and readbacks.
#[cfg(feature = "diagnostic-taps")]
fn device_work(host: &FakeHost, from: usize) -> Vec<String> {
    host.journal.borrow()[from..]
        .iter()
        .filter(|line| !line.starts_with("readback ") && !line.starts_with("synchronize "))
        .cloned()
        .collect()
}

/// Device work by stream: each stream's queued work in its order (launches,
/// graph captures and replays, copies, uploads, zeroes, and event records
/// and waits), and the rest (allocations, symbols, event creation) in the
/// journal's. Rank 0's block taps add commands to its stream alone, which
/// shifts how a relocated program's ranks interleave on the host but not
/// what any stream runs or waits for.
#[cfg(feature = "diagnostic-taps")]
fn on_each_stream(work: Vec<String>) -> std::collections::BTreeMap<String, Vec<String>> {
    let mut streams = std::collections::BTreeMap::<String, Vec<String>>::new();
    for line in work {
        let words = line.split(' ').collect::<Vec<_>>();
        let stream = match words.as_slice() {
            ["peer", "copy", stream, ..] => stream.to_string(),
            [
                "launch" | "row" | "replay" | "capture" | "captured" | "copy" | "upload" | "zero"
                | "record" | "wait" | "synchronize" | "readback",
                stream,
                ..,
            ] => stream.to_string(),
            _ => "host".to_owned(),
        };
        streams.entry(stream).or_default().push(line);
    }
    streams
}

/// Moving an event record or wait, a replay or a peer copy across a launch
/// of its stream changes the stream's work; interleaving streams does not.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn stream_work_keeps_each_streams_event_order() {
    let streams = |lines: &[&str]| on_each_stream(lines.iter().map(|&l| l.to_owned()).collect());
    let served = [
        "event 9",
        "launch 1 producer",
        "record 1 9",
        "wait 2 9",
        "launch 2 consumer",
        "replay 2 4",
        "peer copy 1 0 a 1 b",
    ];
    assert_eq!(
        streams(&served),
        streams(&[
            "event 9",
            "launch 1 producer",
            "record 1 9",
            "peer copy 1 0 a 1 b",
            "wait 2 9",
            "launch 2 consumer",
            "replay 2 4",
        ])
    );
    for moved in [
        [
            "event 9",
            "record 1 9",
            "launch 1 producer",
            "launch 2 consumer",
            "wait 2 9",
            "replay 2 4",
            "peer copy 1 0 a 1 b",
        ],
        [
            "event 9",
            "launch 1 producer",
            "record 1 9",
            "launch 2 consumer",
            "wait 2 9",
            "replay 2 4",
            "peer copy 1 0 a 1 b",
        ],
        [
            "event 9",
            "launch 1 producer",
            "record 1 9",
            "wait 2 9",
            "replay 2 4",
            "launch 2 consumer",
            "peer copy 1 0 a 1 b",
        ],
        [
            "event 9",
            "peer copy 1 0 a 1 b",
            "launch 1 producer",
            "record 1 9",
            "wait 2 9",
            "launch 2 consumer",
            "replay 2 4",
        ],
    ] {
        assert_ne!(streams(&moved), streams(&served), "{moved:?}");
    }
}

/// A diagnostic bundle creates with taps on, and refuses filters that select
/// no tap, naming the variable. Each case creates in a child process with
/// the case's variables set.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_create_reads_the_filters() {
    const NAME: &str = "fake_host_tests::taps::diagnostic_create_reads_the_filters";
    if let Some(expected) = std::env::var_os(CHILD) {
        let expected = expected.into_string().unwrap();
        let host = Rc::new(FakeHost::default());
        let created = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
            .unwrap()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &configuration());
        if expected == "created" {
            created.unwrap().destroy().unwrap();
            return;
        }
        assert!(created.is_err(), "the create took {expected}");
        let logs = host.logs.borrow();
        assert!(logs.iter().any(|line| line.contains(&expected)), "{logs:?}");
        return;
    }
    let directory = std::env::temp_dir().join(format!("mimo-taps-cases-{}", std::process::id()));
    let cases: &[(&[(&str, &str)], &str)] = &[
        (&[], "created"),
        (
            &[
                ("CHOPINFER_TAPS_SITES", "out,enc.vision.out,attn.core"),
                ("CHOPINFER_TAPS_LAYERS", "3"),
            ],
            "created",
        ),
        (
            &[("CHOPINFER_TAPS_SITES", "out,x.hc.*")],
            "CHOPINFER_TAPS_SITES selects no tap: x.hc.*",
        ),
        (
            &[("CHOPINFER_TAPS_LAYERS", "47,48")],
            "CHOPINFER_TAPS_LAYERS names layers without taps: 48",
        ),
        (
            &[("CHOPINFER_TAPS_FROM", "-1")],
            "CHOPINFER_TAPS_FROM is not a position: -1",
        ),
    ];
    for (variables, expected) in cases {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([NAME, "--exact", "--test-threads=1"])
            .env(CHILD, expected)
            .env(runtime_bundle_sdk::taps::TAPS_ENV.dir, &directory)
            .envs(variables.iter().copied())
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{variables:?}: {stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert!(!directory.exists(), "a create wrote taps before a step");
}

/// A media prefill with taps does the device work it does without them on
/// every stream, adding only synchronizations and readbacks, and writes the image's
/// encoder rows and every declared tap, in declaration order, into one
/// complete step.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn a_tapped_prefill_does_the_served_device_work_and_writes_every_tap() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    let root = std::env::temp_dir().join(format!("mimo-taps-step-{}", std::process::id()));
    let launches = |tapped: bool| {
        let (backend, mut instance, request) = encoder_fault_fixture();
        if tapped {
            let taps = Taps::new(root.clone(), TapFilter::default()).with_model("mimo26");
            crate::instance::force_taps(instance.handle(), taps);
            backend.tap_readbacks.set(true);
        }
        instance
            .execute(&request.encode().unwrap(), &mut ViewSink::default())
            .unwrap();
        instance.commit().unwrap();
        instance.destroy().unwrap();
        device_work(&backend, 0)
    };
    let plain = on_each_stream(launches(false));
    assert_eq!(on_each_stream(launches(true)), plain);

    let dir = root.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("completion.json")["completion"], "full");
    let capture = read("capture.json");
    assert_eq!(capture["model"], "mimo26");
    assert_eq!(capture["program"], "Prefill");
    let expected = capture["expected"].as_array().unwrap();
    let index = std::fs::read_to_string(dir.join("index.jsonl")).unwrap();
    let written = index
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(written.len(), expected.len());
    for (written, expected) in written.iter().zip(expected) {
        assert_eq!(
            (&written["name"], &written["rank"]),
            (&expected["name"], &expected["rank"])
        );
        let file = dir.join(format!(
            "{}.r{}.bin",
            written["name"].as_str().unwrap(),
            written["rank"]
        ));
        assert_eq!(
            std::fs::metadata(file).unwrap().len(),
            expected["bytes"].as_u64().unwrap()
        );
    }
    // The image's one row first, on the encoder axis, then the sequence's.
    assert_eq!(
        written[0],
        serde_json::json!({"name": "enc.vision.out", "rank": 0, "first_row": 0, "rows": 1,
            "cols": 4096, "dtype": "bf16", "axis": "encoder"})
    );
    assert_eq!(
        read("items.json")["encoder"]["enc.vision.out"],
        serde_json::json!([{"request": 7, "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 1}])
    );
    // Then the image's blocks, its four patch rows each.
    let blocks = 1 + 28;
    assert_eq!(written[1]["name"], "enc.vision.B00.in");
    assert_eq!(written[blocks]["name"], "enc.vision.B27.out");
    assert_eq!(
        (&written[blocks]["rows"], &written[blocks]["cols"]),
        (&4.into(), &1280.into())
    );
    assert_eq!(written[blocks + 1]["name"], "embed.out");
    assert_eq!(written.last().unwrap()["name"], "final.norm");
    let ranks = |name: &str| written.iter().filter(|entry| entry["name"] == name).count();
    assert_eq!((ranks("L05.attn.out"), ranks("L05.mlp.in")), (4, 1));
    std::fs::remove_dir_all(root).unwrap();
}

/// A pipelined prefill with taps keeps serving's halves, lanes and events:
/// its device work is the served one's but the taps' synchronizations and
/// readbacks. Request 8's first 184 rows are in the first half, so that
/// half plans its attention at a context of 184 while the whole chunk's
/// is 500. Each tap is written once both halves are copied.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn a_tapped_pipelined_prefill_does_the_served_device_work() {
    use runtime_bundle_sdk::taps::{TapFilter, TapLayers, Taps};
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let root = std::env::temp_dir().join(format!("mimo-taps-pipelined-{}", std::process::id()));
    let (first, second) = (200_u64, 500_u64);
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![
            BatchItemV1::new(7, false, first as u32, true, vec![40; first as usize]),
            BatchItemV1::new(8, false, second as u32, true, vec![41; second as usize]),
        ],
        vec![
            BatchAppendV1::new(
                0,
                first,
                first,
                (0..first.div_ceil(PAGE_TOKENS)).collect(),
                None,
            ),
            BatchAppendV1::new(
                0,
                second,
                second,
                (8..8 + second.div_ceil(PAGE_TOKENS)).collect(),
                None,
            ),
        ],
    )
    .encode()
    .unwrap();
    let config = crate::overlap::Config {
        sms: 188,
        reserved_sms: 16,
        comm_ctas: Some(24),
        min_tokens: 512,
    };
    let work = |tapped: bool| {
        let (backend, mut instance) = fault_instance();
        crate::instance::force_overlap(instance.handle(), Some(config));
        if tapped {
            let filter = TapFilter {
                layers: Some(TapLayers::List(vec![1])),
                sites: Some(vec!["L01.*".into(), "final.norm".into()]),
                from: None,
            };
            let taps = Taps::new(root.clone(), filter).with_model("mimo26");
            crate::instance::force_taps(instance.handle(), taps);
            backend.tap_readbacks.set(true);
        }
        let from = backend.journal.borrow().len();
        instance
            .execute(&request, &mut ViewSink::default())
            .unwrap();
        instance.commit().unwrap();
        let work = device_work(&backend, from);
        instance.destroy().unwrap();
        work
    };
    let served = work(false);
    assert!(
        served.iter().any(|line| line.starts_with("wait ")),
        "not pipelined"
    );
    assert_eq!(work(true), served);

    let dir = root.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("completion.json")["completion"], "full");
    let expected = read("capture.json")["expected"].as_array().unwrap().clone();
    // Layer 1's replicated taps once, its partials and head shards per
    // rank, then final.norm.
    assert_eq!(expected.len(), 6 + 3 * 4 + 1);
    for entry in &expected {
        let file = dir.join(format!(
            "{}.r{}.bin",
            entry["name"].as_str().unwrap(),
            entry["rank"]
        ));
        assert_eq!(
            std::fs::metadata(file).unwrap().len(),
            entry["bytes"].as_u64().unwrap()
        );
        assert_eq!(entry["rows"], 700);
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// A tapped decode runs stage by stage, every stream launching what the
/// served decode's graph captured, and writes a complete decode step;
/// buckets that sum the routed rows inside the all-reduce tap no MoE
/// layer's MLP output.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn a_tapped_decode_launches_the_served_graph_kernels_and_writes_its_step() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let root = std::env::temp_dir().join(format!("mimo-taps-decode-{}", std::process::id()));
    let request = |start: u64, tokens: Vec<u32>| {
        let rows = tokens.len() as u32;
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::new(7, start > 0, rows, true, tokens)],
            vec![BatchAppendV1::new(
                start,
                u64::from(rows),
                start + u64::from(rows),
                vec![1],
                None,
            )],
        )
        .encode()
        .unwrap()
    };
    let launches = |tapped: bool| {
        let (backend, mut instance) = fault_instance();
        if tapped {
            let taps = Taps::new(root.clone(), TapFilter::default()).with_model("mimo26");
            crate::instance::force_taps(instance.handle(), taps);
            backend.tap_readbacks.set(true);
        }
        instance
            .execute(&request(0, vec![40, 41, 42]), &mut ViewSink::default())
            .unwrap();
        instance.commit().unwrap();
        let from = backend.journal.borrow().len();
        instance
            .execute(&request(3, vec![43]), &mut ViewSink::default())
            .unwrap();
        instance.commit().unwrap();
        instance.destroy().unwrap();
        let journal = &backend.journal.borrow()[from..];
        assert_eq!(
            journal.iter().any(|line| line.starts_with("captured ")),
            !tapped,
            "the served decode captures its graph, the tapped one none"
        );
        let mut streams = BTreeMap::<String, Vec<String>>::new();
        for line in &backend.journal.borrow()[from..] {
            if let Some(rest) = line.strip_prefix("launch ") {
                let (stream, launch) = rest.split_once(' ').unwrap();
                streams
                    .entry(stream.into())
                    .or_default()
                    .push(launch.into());
            }
        }
        streams
    };
    let served = launches(false);
    assert!(!served.is_empty());
    assert_eq!(launches(true), served);

    let dir = root.join("step0001");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("completion.json")["completion"], "full");
    let capture = read("capture.json");
    assert_eq!(capture["program"], "Decode(1)");
    assert_eq!(read("items.json")["positions"], serde_json::json!([3]));
    let names = capture["expected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["name"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    assert!(names.contains(&"L00.mlp.out".to_owned()));
    assert!(!names.contains(&"L01.mlp.out".to_owned()));
    assert!(names.contains(&"L01.router.ids".to_owned()));
    assert_eq!(names.last().unwrap(), "final.norm");
    std::fs::remove_dir_all(root).unwrap();
}

/// Two requests that inject the rows of one encode each hold them in the
/// encoder tap, indexed from their own first encoder row.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn requests_sharing_an_encode_each_hold_its_rows() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchMediaInjectV3, BatchRequestV1};
    let root = std::env::temp_dir().join(format!("mimo-taps-shared-{}", std::process::id()));
    let (backend, mut instance, single) = encoder_fault_fixture();
    let filter = TapFilter {
        sites: Some(vec!["enc.vision.out".into(), "embed.out".into()]),
        ..TapFilter::default()
    };
    let taps = Taps::new(root.clone(), filter).with_model("mimo26");
    crate::instance::force_taps(instance.handle(), taps);
    backend.tap_readbacks.set(true);
    let mut media = single.media().clone();
    media.injects.push(BatchMediaInjectV3 {
        item_index: 1,
        ..media.injects[0]
    });
    let request = BatchRequestV1::new(
        runtime_bundle_sdk::BatchWireVersionV1::V3,
        vec![
            BatchItemV1::new(7, false, 1, true, vec![151655]),
            BatchItemV1::new(8, false, 1, true, vec![151655]),
        ],
        vec![
            BatchAppendV1::new(0, 1, 1, vec![3], None),
            BatchAppendV1::new(0, 1, 1, vec![4], None),
        ],
    )
    .with_media(media);
    instance
        .execute(&request.encode().unwrap(), &mut ViewSink::default())
        .unwrap();
    instance.commit().unwrap();
    instance.destroy().unwrap();
    let dir = root.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("completion.json")["completion"], "full");
    assert_eq!(
        read("items.json")["encoder"]["enc.vision.out"],
        serde_json::json!([{"request": 7, "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 1},
            {"request": 8, "start": 1, "kind": "image", "item": 0, "index": 0, "rows": 1}])
    );
    let index = std::fs::read_to_string(dir.join("index.jsonl")).unwrap();
    let first: serde_json::Value = serde_json::from_str(index.lines().next().unwrap()).unwrap();
    assert_eq!(
        (&first["name"], &first["rows"]),
        (&"enc.vision.out".into(), &2.into())
    );
    std::fs::remove_dir_all(root).unwrap();
}

/// An image, a video and another image in one request: each encoder row is
/// recorded with its media kind and that kind's occurrence, so the second
/// image is image 1, not the request's third encoder item.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn interleaved_images_and_video_are_recorded_by_kind() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchMediaInjectV3, BatchRequestV1};
    let root = std::env::temp_dir().join(format!("mimo-taps-interleaved-{}", std::process::id()));
    let (backend, mut instance, single) = encoder_fault_fixture();
    let filter = TapFilter {
        sites: Some(vec!["enc.vision.out".into()]),
        ..TapFilter::default()
    };
    let taps = Taps::new(root.clone(), filter).with_model("mimo26");
    crate::instance::force_taps(instance.handle(), taps);
    backend.tap_readbacks.set(true);
    let mut media = single.media().clone();
    let record = media.encodes[0].clone();
    media.encodes = (0..3)
        .map(|page| runtime_bundle_sdk::BatchMediaEncodeV3 {
            pages: vec![page],
            payload_offset: page * record.payload_bytes,
            ..record.clone()
        })
        .collect();
    media.payload = vec![0; 3 * record.payload_bytes as usize];
    // Prompt rows 0..3 inject arena pages 0, 1 and 2 in turn.
    media.injects = (0..3)
        .map(|row| BatchMediaInjectV3 {
            item_index: 0,
            item_row: row,
            rows: 1,
            arena_row: u64::from(row) * 64,
        })
        .collect();
    let request = BatchRequestV1::new(
        runtime_bundle_sdk::BatchWireVersionV1::V3,
        vec![BatchItemV1::new(
            7,
            false,
            3,
            true,
            vec![151655, 151656, 151655],
        )],
        vec![BatchAppendV1::new(0, 3, 3, vec![3], None)],
    )
    .with_media(media);
    instance
        .execute(&request.encode().unwrap(), &mut ViewSink::default())
        .unwrap();
    instance.commit().unwrap();
    instance.destroy().unwrap();
    let dir = root.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("completion.json")["completion"], "full");
    assert_eq!(
        read("items.json")["encoder"]["enc.vision.out"],
        serde_json::json!([
            {"request": 7, "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 1},
            {"request": 7, "start": 1, "kind": "video", "item": 0, "index": 0, "rows": 1},
            {"request": 7, "start": 2, "kind": "image", "item": 1, "index": 0, "rows": 1}])
    );
    std::fs::remove_dir_all(root).unwrap();
}

/// Block taps through serving: an image and a 16-frame audio clip encoded in
/// one prefill write every block of both towers on the encoder axis, each
/// tower's transformer inputs and block outputs under the item's request,
/// kind and occurrence: the image's 4 patch rows, the clip's 8 tokenizer
/// rows and 4 grouped codes.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn block_taps_write_each_encodes_blocks_by_kind_and_item() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchMediaInjectV3, BatchRequestV1};
    let root = std::env::temp_dir().join(format!("mimo-taps-blocks-{}", std::process::id()));
    let (backend, mut instance, single) = encoder_fault_fixture();
    let filter = TapFilter {
        sites: Some(vec!["enc.*.B*".into()]),
        ..TapFilter::default()
    };
    let taps = Taps::new(root.clone(), filter).with_model("mimo26");
    crate::instance::force_taps(instance.handle(), taps);
    backend.tap_readbacks.set(true);
    let mut media = single.media().clone();
    let image = media.encodes[0].clone();
    let frames = 16_u32;
    let audio = runtime_bundle_sdk::BatchMediaEncodeV3 {
        encoder: 1,
        geometry: [frames, 128, 0, 0],
        output_rows: 1,
        pages: vec![1],
        payload_offset: image.payload_bytes,
        payload_bytes: u64::from(frames) * 128 * 4,
    };
    media.payload = vec![0; (image.payload_bytes + audio.payload_bytes) as usize];
    media.encodes = vec![image, audio];
    media.injects = (0..2)
        .map(|row| BatchMediaInjectV3 {
            item_index: 0,
            item_row: row,
            rows: 1,
            arena_row: u64::from(row) * 64,
        })
        .collect();
    let request = BatchRequestV1::new(
        runtime_bundle_sdk::BatchWireVersionV1::V3,
        vec![BatchItemV1::new(7, false, 2, true, vec![151655, 151669])],
        vec![BatchAppendV1::new(0, 2, 2, vec![3], None)],
    )
    .with_media(media);
    instance
        .execute(&request.encode().unwrap(), &mut ViewSink::default())
        .unwrap();
    instance.commit().unwrap();
    instance.destroy().unwrap();
    let dir = root.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("completion.json")["completion"], "full");
    let index = std::fs::read_to_string(dir.join("index.jsonl")).unwrap();
    let written = index
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    let mut expected = vec!["enc.vision.B00.in".to_owned()];
    expected.extend((0..28).map(|block| format!("enc.vision.B{block:02}.out")));
    expected.push("enc.audio.B00.in".into());
    expected.extend((0..24).map(|block| format!("enc.audio.B{block:02}.out")));
    expected.push("enc.audio.B24.in".into());
    expected.extend((24..30).map(|block| format!("enc.audio.B{block:02}.out")));
    let names: Vec<&str> = written
        .iter()
        .map(|entry| entry["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, expected);
    let items = read("items.json");
    let segment = |kind: &str, rows: u32| serde_json::json!([{"request": 7, "start": 0, "kind": kind, "item": 0, "index": 0, "rows": rows}]);
    for entry in &written {
        let name = entry["name"].as_str().unwrap();
        let (kind, rows, cols) = match name {
            name if name.starts_with("enc.vision") => ("image", 4, 1280),
            name if name >= "enc.audio.B24" => ("audio", 4, 1024),
            _ => ("audio", 8, 1024),
        };
        assert_eq!(entry["axis"], "encoder", "{name}");
        assert_eq!(
            (&entry["rows"], &entry["cols"]),
            (&rows.into(), &cols.into()),
            "{name}"
        );
        assert_eq!(items["encoder"][name], segment(kind, rows), "{name}");
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// The encoder taps of every step under `root`, in step order: per step,
/// each tap's rows and its items.json segments.
#[cfg(feature = "diagnostic-taps")]
fn encoder_steps(root: &std::path::Path) -> Vec<BTreeMap<String, (u64, serde_json::Value)>> {
    let mut steps = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    steps.sort();
    steps
        .iter()
        .map(|dir| {
            let read = |name: &str| -> serde_json::Value {
                serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
            };
            let items = read("items.json");
            let index = std::fs::read_to_string(dir.join("index.jsonl")).unwrap();
            index
                .lines()
                .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                .filter(|entry| entry["axis"] == "encoder")
                .map(|entry| {
                    let name = entry["name"].as_str().unwrap().to_owned();
                    let segments = items["encoder"][&name].clone();
                    (name, (entry["rows"].as_u64().unwrap(), segments))
                })
                .collect()
        })
        .collect()
}

/// The image's encode is shown twice by request 7 and once by request 8
/// in one prefill: its block rows (4 patch rows) are written under each of
/// the three occurrences, as its output row is.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn block_taps_name_every_occurrence_of_a_shared_encode() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchMediaInjectV3, BatchRequestV1};
    let root = std::env::temp_dir().join(format!("mimo-taps-shared-blocks-{}", std::process::id()));
    let (backend, mut instance, single) = encoder_fault_fixture();
    let filter = TapFilter {
        sites: Some(vec![
            "enc.vision.out".into(),
            "enc.vision.B00.in".into(),
            "enc.vision.B27.out".into(),
        ]),
        ..TapFilter::default()
    };
    let taps = Taps::new(root.clone(), filter).with_model("mimo26");
    crate::instance::force_taps(instance.handle(), taps);
    backend.tap_readbacks.set(true);
    let mut media = single.media().clone();
    media.injects = [(0, 0), (0, 1), (1, 0)]
        .map(|(item_index, item_row)| BatchMediaInjectV3 {
            item_index,
            item_row,
            rows: 1,
            arena_row: 0,
        })
        .to_vec();
    let request = BatchRequestV1::new(
        runtime_bundle_sdk::BatchWireVersionV1::V3,
        vec![
            BatchItemV1::new(7, false, 2, true, vec![151655, 151655]),
            BatchItemV1::new(8, false, 1, true, vec![151655]),
        ],
        vec![
            BatchAppendV1::new(0, 2, 2, vec![3], None),
            BatchAppendV1::new(0, 1, 1, vec![4], None),
        ],
    )
    .with_media(media);
    instance
        .execute(&request.encode().unwrap(), &mut ViewSink::default())
        .unwrap();
    instance.commit().unwrap();
    instance.destroy().unwrap();
    let steps = encoder_steps(&root);
    assert_eq!(steps.len(), 1);
    let segments = |rows: u32| {
        let segment = |start: u32, request: u64, item: u32| {
            serde_json::json!({"request": request,
            "start": start, "kind": "image", "item": item, "index": 0, "rows": rows})
        };
        serde_json::json!([
            segment(0, 7, 0),
            segment(rows, 7, 1),
            segment(2 * rows, 8, 0)
        ])
    };
    for (name, rows) in [
        ("enc.vision.out", 1),
        ("enc.vision.B00.in", 4),
        ("enc.vision.B27.out", 4),
    ] {
        assert_eq!(
            steps[0][name],
            (u64::from(3 * rows), segments(rows)),
            "{name}"
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// Request 7's prompt over three prefill chunks: text only, then image A,
/// then image B; request 8 joins the third step showing A, resident from
/// the second. Each chunk's rows keep their request and the image's place
/// in its prompt (B is 7's image 1), and A's block rows, read back when it
/// was encoded, are written again for request 8.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn taps_keep_a_requests_items_across_its_prefill_chunks() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    use runtime_bundle_sdk::{
        BatchAppendV1, BatchItemV1, BatchMediaInjectV3, BatchMediaSectionV3, BatchRequestV1,
    };
    let root = std::env::temp_dir().join(format!("mimo-taps-chunks-{}", std::process::id()));
    let (backend, mut instance, single) = encoder_fault_fixture();
    let filter = TapFilter {
        sites: Some(vec!["enc.vision.out".into(), "enc.vision.B27.out".into()]),
        ..TapFilter::default()
    };
    let taps = Taps::new(root.clone(), filter).with_model("mimo26");
    crate::instance::force_taps(instance.handle(), taps);
    backend.tap_readbacks.set(true);
    let image = single.media().encodes[0].clone();
    let inject = |item_index: u32, item_row: u32, arena_row: u64| BatchMediaInjectV3 {
        item_index,
        item_row,
        rows: 1,
        arena_row,
    };
    let mut execute = |items: Vec<BatchItemV1>, appends: Vec<BatchAppendV1>, media| {
        let request =
            BatchRequestV1::new(runtime_bundle_sdk::BatchWireVersionV1::V3, items, appends)
                .with_media(media);
        instance
            .execute(&request.encode().unwrap(), &mut ViewSink::default())
            .unwrap();
        instance.commit().unwrap();
    };
    execute(
        vec![BatchItemV1::new(7, false, 2, false, vec![11, 12])],
        vec![BatchAppendV1::new(0, 2, 2, vec![3], None)],
        BatchMediaSectionV3::default(),
    );
    execute(
        vec![BatchItemV1::new(7, false, 2, false, vec![151655, 13])],
        vec![BatchAppendV1::new(2, 2, 4, vec![3], None)],
        BatchMediaSectionV3 {
            encodes: vec![image.clone()],
            injects: vec![inject(0, 0, 0)],
            payload: single.media().payload.clone(),
        },
    );
    execute(
        vec![
            BatchItemV1::new(7, false, 2, true, vec![14, 151655]),
            BatchItemV1::new(8, false, 1, true, vec![151655]),
        ],
        vec![
            BatchAppendV1::new(4, 2, 6, vec![3], None),
            BatchAppendV1::new(0, 1, 1, vec![4], None),
        ],
        BatchMediaSectionV3 {
            encodes: vec![runtime_bundle_sdk::BatchMediaEncodeV3 {
                pages: vec![1],
                ..image
            }],
            injects: vec![inject(0, 1, 64), inject(1, 0, 0)],
            payload: single.media().payload.clone(),
        },
    );
    instance.destroy().unwrap();
    let segment = |start: u32, request: u64, item: u32, rows: u32| {
        serde_json::json!({"request": request, "start": start, "kind": "image", "item": item,
            "index": 0, "rows": rows})
    };
    let encoded = encoder_steps(&root)
        .into_iter()
        .filter(|step| !step.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(encoded.len(), 2, "{encoded:?}");
    for (name, rows) in [("enc.vision.out", 1), ("enc.vision.B27.out", 4)] {
        assert_eq!(
            encoded[0][name],
            (u64::from(rows), serde_json::json!([segment(0, 7, 0, rows)])),
            "{name}"
        );
        assert_eq!(
            encoded[1][name],
            (
                u64::from(2 * rows),
                serde_json::json!([segment(0, 7, 1, rows), segment(rows, 8, 0, rows)])
            ),
            "{name}"
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "diagnostic-taps")]
#[test]
fn cancelled_continuation_retries_the_same_encoder_occurrence() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchMediaSectionV3, BatchRequestV1};
    let root = std::env::temp_dir().join(format!("mimo-taps-cancel-{}", std::process::id()));
    let (backend, mut instance, single) = encoder_fault_fixture();
    let taps = Taps::new(
        root.clone(),
        TapFilter {
            sites: Some(vec!["enc.vision.out".into(), "enc.vision.B27.out".into()]),
            ..TapFilter::default()
        },
    );
    crate::instance::force_taps(instance.handle(), taps);
    backend.tap_readbacks.set(true);
    let mut media = single.media().clone();
    media.encodes[0].geometry = [1, 2, 4, 0];
    media.encodes[0].output_rows = 2;
    media.encodes[0].payload_bytes *= 2;
    media.payload.resize(media.payload.len() * 2, 0);
    let request = |previous, tokens: Vec<u32>, media| {
        let rows = tokens.len() as u64;
        BatchRequestV1::new(
            runtime_bundle_sdk::BatchWireVersionV1::V3,
            vec![BatchItemV1::new(7, false, rows as u32, true, tokens)],
            vec![BatchAppendV1::new(
                previous,
                rows,
                previous + rows,
                vec![3],
                None,
            )],
        )
        .with_media(media)
        .encode()
        .unwrap()
    };
    let first = request(0, vec![151655], media);
    instance.execute(&first, &mut ViewSink::default()).unwrap();
    instance.commit().unwrap();
    let continuation = request(
        1,
        vec![151655, 12],
        BatchMediaSectionV3 {
            injects: vec![runtime_bundle_sdk::BatchMediaInjectV3 {
                item_index: 0,
                item_row: 0,
                rows: 1,
                arena_row: 1,
            }],
            ..BatchMediaSectionV3::default()
        },
    );
    instance
        .execute(&continuation, &mut ViewSink::default())
        .unwrap();
    instance.cancel().unwrap();
    instance
        .execute(&continuation, &mut ViewSink::default())
        .unwrap();
    instance.commit().unwrap();
    let read = |step: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(root.join(step).join("items.json")).unwrap()).unwrap()
    };
    assert_eq!(
        read("step0002")["encoder"]["enc.vision.out"],
        serde_json::json!([{"request": 7, "start": 0, "kind": "image", "item": 0,
            "index": 1, "rows": 1}])
    );
    assert!(
        !root.join("step0001").exists(),
        "cancelled rows remain published"
    );
    assert_eq!(
        read("step0000")["encoder"]["enc.vision.B27.out"][0]["rows"],
        8
    );
    instance.destroy().unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "diagnostic-taps")]
#[test]
fn retired_prompts_forget_labels_but_keep_resident_encoder_rows() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchMediaSectionV3, BatchRequestV1};
    for text_only in [false, true] {
        let root = std::env::temp_dir().join(format!(
            "mimo-taps-retire-{text_only}-{}",
            std::process::id()
        ));
        let (backend, mut instance, single) = encoder_fault_fixture();
        crate::instance::force_taps(
            instance.handle(),
            Taps::new(
                root.clone(),
                TapFilter {
                    sites: Some(vec!["enc.vision.out".into(), "enc.vision.B27.out".into()]),
                    ..TapFilter::default()
                },
            ),
        );
        backend.tap_readbacks.set(true);
        let request = |id, previous, tokens: Vec<u32>, media| {
            let rows = tokens.len() as u64;
            BatchRequestV1::new(
                runtime_bundle_sdk::BatchWireVersionV1::V3,
                vec![BatchItemV1::new(id, false, rows as u32, true, tokens)],
                vec![BatchAppendV1::new(
                    previous,
                    rows,
                    previous + rows,
                    vec![3],
                    None,
                )],
            )
            .with_media(media)
            .encode()
            .unwrap()
        };
        let tokens = if text_only {
            vec![11, 12]
        } else {
            vec![151655, 12]
        };
        let media = if text_only {
            BatchMediaSectionV3::default()
        } else {
            single.media().clone()
        };
        instance
            .execute(
                &request(7, 0, tokens.clone(), media),
                &mut ViewSink::default(),
            )
            .unwrap();
        instance.commit().unwrap();
        let pages = (100..100 + instance.describe_prefix_state().unwrap()).collect::<Vec<_>>();
        instance.capture_prefix_state(7, 2, &pages).unwrap();
        instance
            .execute(
                &request(7, 2, vec![13, 14], BatchMediaSectionV3::default()),
                &mut ViewSink::default(),
            )
            .unwrap();
        assert!(
            instance.retire_request(7).is_err(),
            "cannot retire an open request"
        );
        instance.cancel().unwrap();
        instance.retire_request(7).unwrap();
        instance.retire_request(7).unwrap();
        instance.restore_prefix_state(7, &tokens, &pages).unwrap();
        let mut resident = single.media().clone();
        if !text_only {
            resident.encodes.clear();
            resident.payload.clear();
        }
        instance
            .execute(
                &request(7, 2, vec![151655, 15], resident.clone()),
                &mut ViewSink::default(),
            )
            .unwrap();
        instance.commit().unwrap();
        let steps = encoder_steps(&root);
        assert_eq!(
            steps.last().unwrap()["enc.vision.out"].1[0]["request"],
            serde_json::Value::Null
        );
        instance.retire_request(7).unwrap();
        resident.encodes.clear();
        resident.payload.clear();
        instance
            .execute(
                &request(8, 0, vec![151655], resident),
                &mut ViewSink::default(),
            )
            .unwrap();
        instance.commit().unwrap();
        let steps = encoder_steps(&root);
        for name in ["enc.vision.out", "enc.vision.B27.out"] {
            assert_eq!(steps.last().unwrap()[name].1[0]["request"], 8);
            assert_eq!(steps.last().unwrap()[name].1[0]["item"], 0);
        }
        instance.retire_request(8).unwrap();
        instance.destroy().unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}

/// An audio item injected in two parts: each part's rows take the codes
/// they merge (output row r merges codes 4r..4r + 3, the last row only the
/// real codes), on the encoder row axis under the same kind and item.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn audio_injects_tap_the_codes_their_rows_merge() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchMediaInjectV3, BatchRequestV1};
    let root = std::env::temp_dir().join(format!("mimo-taps-codes-{}", std::process::id()));
    let (backend, mut instance, single) = encoder_fault_fixture();
    let filter = TapFilter {
        sites: Some(vec!["enc.audio.out".into(), "enc.audio.x.codes".into()]),
        ..TapFilter::default()
    };
    let taps = Taps::new(root.clone(), filter).with_model("mimo26");
    crate::instance::force_taps(instance.handle(), taps);
    backend.tap_readbacks.set(true);
    // 40 frames: 20 attention rows, 10 codes, 3 output rows.
    let frames = 40;
    let declaration = &model_mimo26_contract::encoders::spec::declarations_flash().unwrap()[1];
    let input = declaration.admit_record([frames, 128, 0, 0]).unwrap();
    let output_rows = declaration.trace(input).unwrap().last().unwrap().rows as u32;
    assert_eq!(output_rows, 3);
    let mut media = single.media().clone();
    media.encodes = vec![runtime_bundle_sdk::BatchMediaEncodeV3 {
        encoder: 1,
        geometry: [frames, 128, 0, 0],
        output_rows,
        pages: vec![0],
        payload_offset: 0,
        payload_bytes: u64::from(frames) * 128 * 4,
    }];
    media.payload = vec![0; frames as usize * 128 * 4];
    media.injects = vec![
        BatchMediaInjectV3 {
            item_index: 0,
            item_row: 0,
            rows: 2,
            arena_row: 0,
        },
        BatchMediaInjectV3 {
            item_index: 0,
            item_row: 2,
            rows: 1,
            arena_row: 2,
        },
    ];
    let request = BatchRequestV1::new(
        runtime_bundle_sdk::BatchWireVersionV1::V3,
        vec![BatchItemV1::new(7, false, 3, true, vec![151669; 3])],
        vec![BatchAppendV1::new(0, 3, 3, vec![3], None)],
    )
    .with_media(media);
    instance
        .execute(&request.encode().unwrap(), &mut ViewSink::default())
        .unwrap();
    instance.commit().unwrap();
    instance.destroy().unwrap();
    let dir = root.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("completion.json")["completion"], "full");
    let encoder = &read("items.json")["encoder"];
    assert_eq!(
        encoder["enc.audio.out"],
        serde_json::json!([
            {"request": 7, "start": 0, "kind": "audio", "item": 0, "index": 0, "rows": 2},
            {"request": 7, "start": 2, "kind": "audio", "item": 0, "index": 2, "rows": 1}])
    );
    assert_eq!(
        encoder["enc.audio.x.codes"],
        serde_json::json!([
            {"request": 7, "start": 0, "kind": "audio", "item": 0, "index": 0, "rows": 8},
            {"request": 7, "start": 8, "kind": "audio", "item": 0, "index": 8, "rows": 2}])
    );
    let index = std::fs::read_to_string(dir.join("index.jsonl")).unwrap();
    let codes = index
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .find(|entry| entry["name"] == "enc.audio.x.codes")
        .unwrap();
    assert_eq!(
        (
            &codes["rows"],
            &codes["cols"],
            &codes["dtype"],
            &codes["axis"]
        ),
        (&10.into(), &20.into(), &"u32".into(), &"encoder".into())
    );
    assert_eq!(
        std::fs::metadata(dir.join("enc.audio.x.codes.r0.bin"))
            .unwrap()
            .len(),
        10 * 20 * 4
    );
    let expected = read("capture.json")["expected"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "enc.audio.x.codes")
        .cloned()
        .unwrap();
    assert_eq!(expected["compare"], "codes");
    std::fs::remove_dir_all(root).unwrap();
}

/// Codes retained with an encode follow its request across chunks and a
/// second request reusing the same pages. Each step also names its producer.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn audio_codes_keep_request_binding_across_chunks_and_resident_reuse() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchMediaInjectV3, BatchRequestV1};
    let root = std::env::temp_dir().join(format!("mimo-codes-chunks-{}", std::process::id()));
    let (backend, mut instance, single) = encoder_fault_fixture();
    let nonce = "ab".repeat(16);
    crate::instance::force_taps(
        instance.handle(),
        Taps::new(
            root.clone(),
            TapFilter {
                sites: Some(vec!["enc.audio.x.codes".into(), "enc.audio.B00.in".into()]),
                ..TapFilter::default()
            },
        )
        .with_model("mimo26")
        .with_producer(Some(nonce.clone())),
    );
    backend.tap_readbacks.set(true);
    // Ten real codes merge into three rows; the last row has two codes.
    for (step, (request, previous, first, rows)) in [(7, 0, 0, 2), (7, 2, 2, 1), (8, 0, 0, 3)]
        .into_iter()
        .enumerate()
    {
        let mut media = single.media().clone();
        media.encodes.clear();
        media.payload.clear();
        if step == 0 {
            media.encodes.push(runtime_bundle_sdk::BatchMediaEncodeV3 {
                encoder: 1,
                geometry: [40, 128, 0, 0],
                output_rows: 3,
                pages: vec![0],
                payload_offset: 0,
                payload_bytes: 40 * 128 * 4,
            });
            media.payload = vec![0; 40 * 128 * 4];
        }
        media.injects = vec![BatchMediaInjectV3 {
            item_index: 0,
            item_row: 0,
            rows,
            arena_row: first,
        }];
        let batch = BatchRequestV1::new(
            runtime_bundle_sdk::BatchWireVersionV1::V3,
            vec![BatchItemV1::new(
                request,
                false,
                rows,
                step != 0,
                vec![151669; rows as usize],
            )],
            vec![BatchAppendV1::new(
                previous,
                u64::from(rows),
                previous + u64::from(rows),
                vec![if request == 7 { 3 } else { 4 }],
                None,
            )],
        )
        .with_media(media);
        instance
            .execute(&batch.encode().unwrap(), &mut ViewSink::default())
            .unwrap();
        instance.commit().unwrap();
        let dir = root.join(format!("step{step:04}"));
        let read = |name: &str| -> serde_json::Value {
            serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
        };
        let items = read("items.json");
        assert_eq!(items["items"][0]["request"], request);
        assert_eq!(read("capture.json")["producer"]["startup_nonce"], nonce);
        let codes = &items["encoder"]["enc.audio.x.codes"][0];
        let count = if step == 0 {
            8
        } else if step == 1 {
            2
        } else {
            10
        };
        assert_eq!(
            codes,
            &serde_json::json!({
                "request": request, "kind": "audio", "item": 0,
                "start": 0, "index": first * 4, "rows": count,
            })
        );
        assert_eq!(
            std::fs::metadata(dir.join("enc.audio.x.codes.r0.bin"))
                .unwrap()
                .len(),
            count * 20 * 4
        );
        if step != 1 {
            assert_eq!(items["encoder"]["enc.audio.B00.in"][0]["request"], request);
        }
    }
    instance.destroy().unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
