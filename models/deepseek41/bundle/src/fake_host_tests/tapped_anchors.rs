//! A tapped instance offers no prompt-end anchors: text and KV-state taps
//! run every prefill stage by stage, which keeps none, so a scheduler with
//! the prefix cache on plans none and an unaligned prompt runs.
use super::*;
use std::path::Path;

use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::prefill::{Boundary, encode_with_cuts};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

type Sink = RecordingSink<super::conformance::Limits>;

/// An unaligned prompt: a multiple of 64 lies strictly inside it.
const PROMPT: u64 = 141;
const ITEMS: u32 = 4;

/// An instance of the native Flash TP4 skip schedule, which keeps
/// prompt-end anchors untapped, with the prefill, cuts and anchors APIs.
fn create(host: &Rc<Host>) -> BundleInstanceV1 {
    let mut record = crate::config::tests::record(&host.shape, host.kv, ITEMS, false);
    for (name, value) in [
        ("prefill_chunk_tokens", 8192.into()),
        ("prefill_upper_skip", true.into()),
    ] {
        record = crate::config::tests::with_option(record, name, value);
    }
    api()
        .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
        .unwrap()
        .with_prefill_cuts_api(crate::instance::prefill::PREFILL_CUTS_API)
        .unwrap()
        .with_prefill_anchors_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_ANCHORS_API_V2)
        .unwrap()
        .create(HostServicesTableV1::new(Rc::clone(host)), &record)
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()))
}

/// Request 7's whole prompt in slot 0 as a scheduler plans it when the
/// instance keeps `anchors` anchors: anchored at its last 64-aligned
/// position when it may keep one.
fn prompt(anchors: u32) -> Vec<u8> {
    let item = BatchItemV1::new(7, false, PROMPT as u32, true, (0..PROMPT as u32).collect())
        .with_anchor(anchors > 0);
    let append = BatchAppendV1::new(0, PROMPT, PROMPT, (0..PROMPT.div_ceil(64)).collect(), None);
    let original = BatchRequestV1::new(BatchWireVersionV1::V3, vec![item], vec![append])
        .with_state_slots(vec![0])
        .encode()
        .unwrap();
    let boundary = Boundary {
        request_id: 7,
        prompt_tokens: PROMPT,
        chunk_end: PROMPT,
        chunk_tokens: 8192,
    };
    encode_with_cuts(&original, &[boundary]).unwrap()
}

/// Runs the prompt as planned for the anchors the instance offers, and
/// returns them.
fn run(host: &Rc<Host>) -> u32 {
    let mut instance = create(host);
    let anchors = instance.prefill_anchor_items().unwrap().unwrap_or(0);
    instance
        .execute_prefill_cuts(&prompt(anchors), &mut Sink::default())
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    anchors
}

/// Untapped, the instance keeps anchors and the unaligned prompt keeps one
/// at token 128. With text taps or KV-state taps selected it offers none,
/// so the same prompt is planned without one and runs, writing its taps.
#[test]
fn a_tapped_instance_offers_no_anchors_and_runs_an_unaligned_prompt() {
    let host = || FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    if std::env::var_os(TAPS_CHILD).is_some() {
        let anchors = run(&host());
        let directory = std::env::var_os(runtime_bundle_sdk::taps::TAPS_ENV.dir).unwrap();
        std::fs::write(Path::new(&directory).join("anchors"), anchors.to_string()).unwrap();
        return;
    }
    assert_eq!(run(&host()), ITEMS);
    for (name, sites, program) in [
        ("text", "out", "Prefill"),
        ("kv", "x.kv.window", crate::taps::kv::PROGRAM),
    ] {
        let directory = std::env::temp_dir().join(format!(
            "deepseek41-tapped-anchors-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        in_child(
            "fake_host_tests::tapped_anchors::a_tapped_instance_offers_no_anchors_and_runs_an_unaligned_prompt",
            &directory,
            &[
                ("CHOPINFER_TAPS_SITES", sites),
                ("CHOPINFER_TAPS_LAYERS", "0"),
            ],
        );
        assert_eq!(
            std::fs::read_to_string(directory.join("anchors")).unwrap(),
            "0",
            "{name}"
        );
        let step = directory.join("step0000");
        let read = |file: &str| -> serde_json::Value {
            serde_json::from_slice(&std::fs::read(step.join(file)).unwrap()).unwrap()
        };
        assert_eq!(read("capture.json")["program"], program, "{name}");
        assert_eq!(read("completion.json")["completion"], "full", "{name}");
        assert_eq!(read("items.json")["items"][0]["request"], 7, "{name}");
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
