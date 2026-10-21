//! Encoder and text taps on the fake host. Each create runs in a child test
//! process, because the taps variables are process-wide: a serving bundle
//! refuses to create while one is set, and a diagnostic bundle writes the
//! taps of a committed step and removes those of a cancelled one.

use super::*;

const CHILD: &str = "CHOPINFER_TEST_TAPS_CHILD";

/// [`image_prompt`]'s step of an 8 x 8 patch image (16 rows) between two
/// text rows, its image rows holding the image placeholder as the host's
/// expansion writes it, so its occurrence is named by request and kind.
#[cfg(feature = "diagnostic-taps")]
fn tapped_image_prompt(request: u64) -> Vec<u8> {
    use runtime_bundle_sdk::{
        BatchAppendV1, BatchCodecV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1,
    };
    let image = Shape::TP4.geometry.shape.tokens.image;
    let bytes = image_prompt(request, [1, 8, 8]);
    let batch = BatchCodecV1::decode(&bytes).unwrap();
    let rows = batch.items()[0].input_tokens().len() as u32;
    let tokens = (0..rows)
        .map(|row| {
            if row == 0 || row == rows - 1 {
                10
            } else {
                image
            }
        })
        .collect();
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(request, false, rows, true, tokens)],
        vec![BatchAppendV1::new(
            0,
            u64::from(rows),
            u64::from(rows),
            (0..u64::from(rows).div_ceil(64)).collect(),
            None,
        )],
    )
    .with_media(batch.media().clone())
    .with_state_slots(vec![0])
    .encode()
    .unwrap()
}

/// A diagnostic instance with the encoder, created under the producer
/// nonce `run-7`, issuing the lowered stages alone.
#[cfg(feature = "diagnostic-taps")]
fn create_tapped(host: &Rc<Host>) -> BundleInstanceV1 {
    let mut tail = options(2048);
    tail.extend_from_slice(runtime_bundle_sdk::taps::PRODUCER_TAG);
    tail.extend_from_slice(&5u32.to_le_bytes());
    tail.extend_from_slice(b"run-7");
    let roles = crate::config::tests::encoder_roles();
    let instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(host)),
            &crate::config::tests::record_naming(DECODE_BATCH, &tail, host.format, &roles),
        )
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    crate::instance::run_lowered_stages_alone(instance.handle());
    instance
}

/// [`create_one_slot`]'s served program under the producer nonce `run-7`.
#[cfg(feature = "diagnostic-taps")]
fn create_one_slot_tapped(host: &Rc<Host>) -> BundleInstanceV1 {
    host.items.set(1);
    let mut tail = options(2048);
    tail.extend_from_slice(runtime_bundle_sdk::taps::PRODUCER_TAG);
    tail.extend_from_slice(&5u32.to_le_bytes());
    tail.extend_from_slice(b"run-7");
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let record = crate::config::tests::record_naming(1, &tail, host.format, &roles);
    api()
        .create(HostServicesTableV1::new(Rc::clone(host)), &record)
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
}

/// Runs the test `name` in a child process with `CHOPINFER_TAPS_DIR` set to
/// `directory` and the encoder taps selected, and fails unless it passed.
fn in_child(name: &str, directory: &std::path::Path) {
    in_child_with(name, directory, &[("CHOPINFER_TAPS_SITES", "enc.*")]);
}

/// [`in_child`] with the taps variables `env` instead.
fn in_child_with(name: &str, directory: &std::path::Path, env: &[(&str, &str)]) {
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([name, "--exact", "--test-threads=1"])
        .env(CHILD, "1")
        .env(runtime_bundle_sdk::taps::TAPS_ENV.dir, directory);
    for variable in ["SITES", "LAYERS", "FROM", "BYTES", "SCOPE_JSON"] {
        command.env_remove(format!("CHOPINFER_TAPS_{variable}"));
    }
    let output = command.envs(env.iter().copied()).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(not(feature = "diagnostic-taps"))]
#[test]
fn serving_create_refuses_taps() {
    if std::env::var_os(CHILD).is_some() {
        let host = FakeHost::new((3, 32));
        let refused = api()
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &record(DECODE_BATCH, &options(2048)),
            )
            .err();
        assert!(refused.is_some(), "the create was not refused");
        let logs = host.logs.borrow();
        assert!(
            logs.iter().any(|line| line.contains("CHOPINFER_TAPS_DIR")
                && line.contains("diagnostic-taps")),
            "{logs:?}"
        );
        assert_eq!(host.released.get(), host.acquired.get());
        return;
    }
    let directory =
        std::env::temp_dir().join(format!("glm5next-taps-serving-{}", std::process::id()));
    in_child(
        "fake_host_tests::taps::serving_create_refuses_taps",
        &directory,
    );
    assert!(!directory.exists(), "a refused create wrote taps");
}

/// A diagnostic bundle's image step writes `enc.vision.out` (the 16 rows
/// the prompt injects, BF16 from rank 0's arena) and every block tap (the
/// image's 64 patch rows in F32) into one tap step under the record's
/// producer nonce, named by the prompt's request, published by the commit;
/// a cancelled step's capture is removed.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_image_steps_write_the_encoder_taps() {
    if std::env::var_os(CHILD).is_some() {
        let host = FakeHost::new((3, 32));
        let mut instance = create_tapped(&host);
        instance
            .execute(&tapped_image_prompt(7), &mut DraftSink::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        instance.retire_request(7).unwrap();
        instance
            .execute(&tapped_image_prompt(8), &mut DraftSink::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.cancel().unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        return;
    }
    let directory = std::env::temp_dir().join(format!("glm5next-taps-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    in_child(
        "fake_host_tests::taps::diagnostic_image_steps_write_the_encoder_taps",
        &directory,
    );
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(directory.join("step0000").join(name)).unwrap())
            .unwrap()
    };
    let capture = read("capture.json");
    assert_eq!(capture["model"], "glm5next");
    assert_eq!(capture["producer"]["startup_nonce"], "run-7");
    assert_eq!(read("completion.json")["completion"], "full");
    let items = read("items.json");
    assert_eq!(items["items"][0]["request"], 7);
    let specs = crate::encoders::taps().unwrap();
    let encoder = items["encoder"].as_object().unwrap();
    assert_eq!(encoder.len(), specs.len());
    for spec in &specs {
        let segments = encoder[&spec.name].as_array().unwrap();
        assert_eq!(segments.len(), 1, "{}", spec.name);
        assert_eq!(segments[0]["request"], 7);
        assert_eq!(segments[0]["kind"], "image");
        let rows = if spec.name == "enc.vision.out" {
            16
        } else {
            64
        };
        assert_eq!(segments[0]["rows"], rows, "{}", spec.name);
        let bytes = std::fs::metadata(
            directory
                .join("step0000")
                .join(format!("{}.r0.bin", spec.name)),
        )
        .unwrap()
        .len();
        assert_eq!(
            bytes,
            rows * u64::from(spec.cols) * spec.dtype.bytes(),
            "{}",
            spec.name
        );
    }
    // The cancelled step left no capture.
    assert!(!directory.join("step0001").exists());
    std::fs::remove_dir_all(directory).unwrap();
}

/// An encoder run that fails after its block rows were read back (rank 0's
/// source-row upload) writes no capture, and the next request's step holds
/// only its own rows: one segment per tap, named by that request, its
/// capture complete.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn a_failed_encoder_run_leaves_no_rows_to_the_next_request() {
    if std::env::var_os(CHILD).is_some() {
        let host = FakeHost::new((3, 32));
        let mut instance = create_tapped(&host);
        let sources = crate::instance::encoder_region(instance.handle(), 0, "source-rows");
        host.fail_upload_to.set(Some(sources));
        assert!(
            instance
                .execute(&tapped_image_prompt(7), &mut DraftSink::default())
                .is_err()
        );
        assert!(host.fail_upload_to.get().is_none(), "the upload ran");
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|line| line.contains("encoder execution")),
            "{:?}",
            host.logs.borrow()
        );
        instance
            .execute(&tapped_image_prompt(8), &mut DraftSink::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        instance.retire_request(8).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        return;
    }
    let directory =
        std::env::temp_dir().join(format!("glm5next-taps-failed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    in_child(
        "fake_host_tests::taps::a_failed_encoder_run_leaves_no_rows_to_the_next_request",
        &directory,
    );
    let step = directory.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(step.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("completion.json")["completion"], "full");
    let items = read("items.json");
    assert_eq!(items["items"][0]["request"], 8);
    let specs = crate::encoders::taps().unwrap();
    let encoder = items["encoder"].as_object().unwrap();
    assert_eq!(encoder.len(), specs.len());
    for spec in &specs {
        let segments = encoder[&spec.name].as_array().unwrap();
        assert_eq!(segments.len(), 1, "{}: {segments:?}", spec.name);
        assert_eq!(segments[0]["request"], 8, "{}", spec.name);
        let rows = if spec.name == "enc.vision.out" {
            16
        } else {
            64
        };
        assert_eq!(segments[0]["rows"], rows, "{}", spec.name);
        let bytes = std::fs::metadata(step.join(format!("{}.r0.bin", spec.name)))
            .unwrap()
            .len();
        assert_eq!(
            bytes,
            rows * u64::from(spec.cols) * spec.dtype.bytes(),
            "{}",
            spec.name
        );
    }
    // The failed run wrote no step.
    assert!(!directory.join("step0001").exists());
    std::fs::remove_dir_all(directory).unwrap();
}

/// The text taps' sites the tests select: the localization set and
/// layer 3's query latent as a detail tap.
#[cfg(feature = "diagnostic-taps")]
const TEXT_SITES: &[(&str, &str)] = &[
    (
        "CHOPINFER_TAPS_SITES",
        "x.hc.attn,attn.out,x.latent,x.q_latent,x.indexer.ids,router.*,final.norm",
    ),
    ("CHOPINFER_TAPS_LAYERS", "3"),
];

/// A diagnostic bundle's text prefill runs eagerly, stage by stage with
/// every rank queued before any waits, and issues exactly the launches of
/// an untapped prefill; its committed step holds every selected tap of the
/// prompt's five rows in program order, from rank 0, under the record's
/// producer nonce. A repeated prefill shape is never captured, and a
/// cancelled step's capture is removed.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_prefills_write_the_text_taps() {
    let launches = |host: &FakeHost, from: usize| -> Vec<Vec<String>> {
        let stream_ranks = host.device.stream_ranks.borrow();
        let mut ranks = vec![Vec::new(); 4];
        for (stream, lines) in launches_by_stream(host, from) {
            ranks[stream_ranks[&stream] as usize].extend(lines.iter().map(|l| without_spans(l)));
        }
        ranks
    };
    if std::env::var_os(CHILD).is_some() {
        let host = FakeHost::new((3, 32));
        let mut instance = create_one_slot_tapped(&host);
        for request in [7, 8] {
            let start = host.device.journal.borrow().len();
            instance
                .execute(&prompt_chunk(request, 0, 5, true), &mut NoRows::default())
                .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
            assert_ranks_advance_together(&host, start);
            if request == 7 {
                let directory = std::env::var_os(runtime_bundle_sdk::taps::TAPS_ENV.dir).unwrap();
                std::fs::write(
                    std::path::Path::new(&directory).join("launches.json"),
                    serde_json::to_vec(&launches(&host, start)).unwrap(),
                )
                .unwrap();
                instance.commit().unwrap();
            } else {
                instance.cancel().unwrap();
            }
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        return;
    }
    // The launches of the same prefill without taps.
    let host = FakeHost::new((3, 32));
    let mut instance = create_one_slot(&host);
    let start = host.device.journal.borrow().len();
    instance
        .execute(&prompt_chunk(7, 0, 5, true), &mut NoRows::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    let untapped = launches(&host, start);
    instance.commit().unwrap();
    instance.destroy().unwrap();
    let directory = std::env::temp_dir().join(format!("glm5next-taps-text-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    in_child_with(
        "fake_host_tests::taps::diagnostic_prefills_write_the_text_taps",
        &directory,
        TEXT_SITES,
    );
    let tapped: Vec<Vec<String>> =
        serde_json::from_slice(&std::fs::read(directory.join("launches.json")).unwrap()).unwrap();
    assert!(untapped[0].len() > 1024, "{}", untapped[0].len());
    assert_eq!(tapped, untapped);
    let step = directory.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(step.join(name)).unwrap()).unwrap()
    };
    let capture = read("capture.json");
    assert_eq!(capture["model"], "glm5next");
    assert_eq!(capture["program"], "Prefill");
    assert_eq!(capture["producer"]["startup_nonce"], "run-7");
    assert_eq!(read("completion.json")["completion"], "full");
    let items = read("items.json");
    assert_eq!(items["items"][0]["request"], 7);
    assert_eq!(
        items["items"][0]["positions"],
        serde_json::json!([0, 1, 2, 3, 4])
    );
    let index = std::fs::read_to_string(step.join("index.jsonl")).unwrap();
    let written = index
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    let sites = [
        "x.hc.attn",
        "attn.out",
        "x.latent",
        "x.q_latent",
        "x.indexer.ids",
        "router.logits",
        "router.weights",
        "router.ids",
        "norm",
    ];
    let expected = crate::taps::text_specs(Shape::TP4)
        .into_iter()
        .filter(|spec| {
            sites.contains(&runtime_bundle_sdk::taps::site(&spec.name))
                && (!spec.detail || spec.layer == Some(3))
        })
        .collect::<Vec<_>>();
    assert_eq!(expected.len(), 45 + 45 + 11 + 1 + 11 + 3 * 42 + 1);
    assert_eq!(written.len(), expected.len());
    for (line, spec) in written.iter().zip(&expected) {
        assert_eq!(line["name"], spec.name.as_str());
        assert_eq!(line["rank"], 0);
        assert_eq!(line["rows"], 5);
        assert_eq!(line["dtype"], spec.dtype.name());
        let bytes = std::fs::metadata(step.join(format!("{}.r0.bin", spec.name)))
            .unwrap()
            .len();
        assert_eq!(
            bytes,
            5 * u64::from(spec.cols) * spec.dtype.bytes(),
            "{}",
            spec.name
        );
    }
    // The cancelled step left no capture.
    assert!(!directory.join("step0001").exists());
    std::fs::remove_dir_all(directory).unwrap();
}

/// With text taps selected, an image step's encoder rows are written into
/// its prefill's text step, ahead of the text taps, in one complete step.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn image_steps_write_their_encoder_rows_into_the_text_step() {
    if std::env::var_os(CHILD).is_some() {
        let host = FakeHost::new((3, 32));
        let mut instance = create_tapped(&host);
        instance
            .execute(&tapped_image_prompt(7), &mut DraftSink::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        return;
    }
    let directory =
        std::env::temp_dir().join(format!("glm5next-taps-image-text-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    in_child_with(
        "fake_host_tests::taps::image_steps_write_their_encoder_rows_into_the_text_step",
        &directory,
        &[("CHOPINFER_TAPS_SITES", "enc.vision.out,final.norm")],
    );
    let step = directory.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(step.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("completion.json")["completion"], "full");
    assert_eq!(
        read("items.json")["encoder"]["enc.vision.out"][0]["rows"],
        16
    );
    let index = std::fs::read_to_string(step.join("index.jsonl")).unwrap();
    let names = index
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()["name"].clone())
        .collect::<Vec<_>>();
    assert_eq!(names, ["enc.vision.out", "final.norm"]);
    assert!(!directory.join("step0001").exists());
    std::fs::remove_dir_all(directory).unwrap();
}
