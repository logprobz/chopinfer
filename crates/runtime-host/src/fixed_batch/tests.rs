use super::diagnostic::{self, Budget, Description, Limits};
use super::record::{Capture, CapturedRow, Frame, Golden, SampleBits, metadata};
use super::trace::{Item, MIXED_DECODE, Mode, Step, Trace};
use crate::executor::SamplingDispatch;
use runtime_engine::{RequestId, SampledRow};
use runtime_sampling::TokenLogprob;
use serde_json::json;
use std::path::PathBuf;

struct Scratch(PathBuf);

#[test]
fn native_replay_command_preserves_the_config_path_and_refuses_surplus_options() {
    let parse = |words: &[&str]| {
        crate::parse_host_command_v1(&words.iter().map(|word| (*word).into()).collect::<Vec<_>>())
    };
    assert_eq!(
        parse(&["fixed-batch", "replay", "--config", "trace.json"]).unwrap(),
        crate::HostCommandV1::FixedBatchReplay {
            configuration: "trace.json".into(),
        }
    );
    for words in [
        vec!["fixed-batch"],
        vec!["fixed-batch", "replay", "--config"],
        vec!["fixed-batch", "replay", "--config", ""],
        vec![
            "fixed-batch",
            "replay",
            "--config",
            "trace.json",
            "--unknown",
        ],
    ] {
        assert!(parse(&words).is_err(), "{words:?}");
    }
}

#[test]
fn replay_temperature_keeps_the_old_default_and_greedy_mode() {
    let trace = Trace::construct(vec![vec![3; 128]], 1024, false);
    let step = trace
        .steps
        .iter()
        .find(|step| step.mode == Mode::Seeded)
        .unwrap();
    let default = step.model_items(None);
    let explicit_default = step.model_items_at_temperature(None, 0.8);
    let sampled = step.model_items_at_temperature(None, 1.0);
    assert_eq!(
        default[0]
            .sampling()
            .unwrap()
            .parameters
            .temperature()
            .to_bits(),
        0.8_f64.to_bits()
    );
    assert_eq!(
        metadata(
            &default[0],
            SamplingDispatch {
                rows: 1,
                simple: true,
                top: false,
                greedy: false
            }
        ),
        metadata(
            &explicit_default[0],
            SamplingDispatch {
                rows: 1,
                simple: true,
                top: false,
                greedy: false
            }
        )
    );
    assert_eq!(
        sampled[0]
            .sampling()
            .unwrap()
            .parameters
            .temperature()
            .to_bits(),
        1.0_f64.to_bits()
    );
    let greedy = trace.steps[0].model_items_at_temperature(None, 1.0);
    assert_eq!(greedy[0].sampling().unwrap().parameters.temperature(), 0.0);
}

#[test]
fn fixed_depth_replay_covers_wide_buckets_and_rejects_unsupported_depth() {
    for concurrency in [16, 32] {
        for depth in [2, 3] {
            let trace = Trace::construct_fixed_depth(vec![vec![3; 128]; concurrency], 1024, depth);
            trace.validate().unwrap();
            for step in trace.steps.iter().skip(1) {
                assert_eq!(step.rows(), concurrency * (depth + 1));
                assert!(
                    step.items
                        .iter()
                        .all(|item| item.drafts == depth && item.kept == 1)
                );
            }
            let replay: Trace =
                serde_json::from_slice(&serde_json::to_vec(&trace).unwrap()).unwrap();
            assert_eq!(trace, replay);
        }
        let invalid = Trace::construct_fixed_depth(vec![vec![3; 128]; concurrency], 1024, 4);
        assert!(invalid.validate().is_err());
    }
}
impl Scratch {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "fixed-batch-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn sample() -> SampledRow {
    SampledRow::new(RequestId::new(1), 3, -0.0, vec![TokenLogprob::new(3, -0.0)])
}
fn rows() -> Vec<CapturedRow> {
    let mut capture = Capture::default();
    capture
        .insert(
            0,
            CapturedRow {
                rank: 0,
                element_type: 1,
                width: 2,
                bytes: [0x7fc00001_u32.to_le_bytes(), 0x80000000_u32.to_le_bytes()].concat(),
                metadata: json!({"seed":71}),
                sample: None,
            },
        )
        .unwrap();
    assert!(capture.take(1).is_err());
    capture.samples(0, &[sample()]).unwrap();
    assert!(capture.samples(0, &[sample()]).is_err());
    capture.take(1).unwrap()
}
fn frame(rows: &[CapturedRow]) -> Frame {
    Frame::from_rows(vec![1, 2], vec![(1, 1)], 1, vec![], rows)
}

#[test]
fn exact_evidence_rejects_each_kind_of_bit_or_trace_drift() {
    let scratch = Scratch::new();
    let baseline = scratch.0.join("baseline");
    let identity = json!({"trace":1});
    let original = rows();
    let mut golden = Golden::open(&baseline, None, &identity).unwrap();
    golden.step(&frame(&original), &original).unwrap();
    golden.finish(1).unwrap();
    std::fs::write(baseline.join("receipt.json"), r#"{"passed":true}"#).unwrap();
    for case in 0..9 {
        let mut actual = rows();
        actual[0].rank = 3; // Ownership is telemetry, never numerical evidence.
        let mut record = frame(&actual);
        match case {
            1 => actual[0].bytes[0] ^= 1,
            2 => record.rows[0].sample.logprob ^= 1 << 63,
            3 => record.rows[0].sample.top[0].1 ^= 1,
            4 => record.rows[0].sample.token += 1,
            5 => record.request.push(3),
            6 => record.kept[0].1 += 1,
            7 => record.drafts.push((1, vec![7])),
            8 => record.rows[0].metadata["seed"] = json!(72),
            _ => (),
        }
        let mut compare = Golden::open(
            &scratch.0.join(format!("arm{case}")),
            Some(&baseline),
            &identity,
        )
        .unwrap();
        let result = compare.step(&record, &actual);
        if case == 0 {
            result.unwrap();
            compare.finish(1).unwrap();
            assert_eq!(compare.bytes_checked, 8);
        } else {
            assert!(result.is_err(), "case {case} escaped exact comparison");
        }
    }
    assert_eq!(SampleBits::from(&sample()).logprob, (-0.0_f64).to_bits());
    assert!(
        Golden::open(
            &scratch.0.join("wrong-identity"),
            Some(&baseline),
            &json!(2)
        )
        .is_err()
    );
}

#[test]
fn incomplete_and_trailing_evidence_cannot_pass() {
    let scratch = Scratch::new();
    let baseline = scratch.0.join("baseline");
    let identity = json!(1);
    let data = rows();
    let mut golden = Golden::open(&baseline, None, &identity).unwrap();
    assert!(golden.finish(0).is_err());
    golden.step(&frame(&data), &data).unwrap();
    golden.finish(1).unwrap();
    assert!(Golden::open(&scratch.0.join("uncommitted"), Some(&baseline), &identity).is_err());
    std::fs::write(baseline.join("receipt.json"), r#"{"passed":true}"#).unwrap();
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(baseline.join("rows.bin"))
        .unwrap()
        .write_all(&[0])
        .unwrap();
    let mut compare = Golden::open(&scratch.0.join("extra"), Some(&baseline), &identity).unwrap();
    compare.step(&frame(&data), &data).unwrap();
    assert!(compare.finish(1).is_err());
}

/// Prefills `prompts` in `groups` of requests, then forces `decode` tokens
/// for the requests listed in every decode step.
fn mixed_trace(prompts: &[Vec<u32>], groups: &[&[usize]], decode: &[usize], steps: u32) -> Trace {
    let item = |slot, prefill, inputs: Vec<u32>, history: &Vec<u32>, emitted| Item {
        slot,
        prefill,
        anchor: false,
        inputs,
        drafts: 0,
        fixed_drafts: Vec::new(),
        kept: if prefill { prompts[slot].len() } else { 1 },
        history: history.clone(),
        emitted,
    };
    let mut histories = prompts.to_vec();
    let mut trace = Trace {
        lifecycle: Vec::new(),
        version: MIXED_DECODE,
        concurrency: prompts.len(),
        vocabulary: 1024,
        steps: groups
            .iter()
            .map(|group| Step {
                mode: Mode::Greedy,
                items: group
                    .iter()
                    .map(|&slot| item(slot, true, prompts[slot].clone(), &prompts[slot], 0))
                    .collect(),
            })
            .collect(),
    };
    for token in 0..steps {
        let items = decode
            .iter()
            .map(|&slot| {
                histories[slot].push(token + 7);
                let emitted = histories[slot].len() - prompts[slot].len();
                item(slot, false, vec![token + 7], &histories[slot], emitted)
            })
            .collect();
        trace.steps.push(Step {
            mode: Mode::Greedy,
            items,
        });
    }
    trace
}

#[test]
fn mixed_decode_traces_prefill_once_then_decode_with_idle_holders() {
    let prompts = vec![vec![3; 5], vec![4; 300], vec![5; 40], vec![6; 1]];
    // Request 0 holds its slot without decoding; prefill spans two steps.
    let trace = mixed_trace(&prompts, &[&[0, 3], &[2, 1]], &[1, 2, 3], 4);
    trace.validate().unwrap();
    assert_eq!(trace.final_tokens(), 304);
    assert_eq!(trace.prefill_tokens(), 340);
    let replay: Trace = serde_json::from_slice(&serde_json::to_vec(&trace).unwrap()).unwrap();
    assert_eq!(replay, trace);
    let decode = trace.steps.last().unwrap();
    assert_eq!(decode.rows(), 3);
    assert!(
        decode
            .model_items(None)
            .iter()
            .all(|item| item.phase() == runtime_scheduler::BatchPhase::Decode)
    );

    let invalid = |change: &dyn Fn(&mut Trace)| {
        let mut trace = trace.clone();
        change(&mut trace);
        trace.validate().unwrap_err()
    };
    // A request decodes before its prefill, or prefills twice.
    invalid(&|t| t.steps.swap(1, 2));
    invalid(&|t| {
        let again = t.steps[0].items[0].clone();
        t.steps[1].items.push(again);
    });
    // Prefill after decoding, a request never prefilled, no decode step.
    invalid(&|t| {
        let step = t.steps.remove(1);
        t.steps.push(step);
    });
    invalid(&|t| t.concurrency = 5);
    invalid(&|t| t.steps.truncate(2));
    // Drafts, multi-token decode inputs, a phase mix and history drift.
    invalid(&|t| t.steps[2].items[0].drafts = 1);
    invalid(&|t| t.steps[3].items[0].inputs.push(7));
    invalid(&|t| t.steps[2].items[0].prefill = true);
    invalid(&|t| t.steps[4].items[1].emitted = 99);
    invalid(&|t| t.steps[4].items[1].history[0] = 9);
    // Other trace versions keep their own geometry.
    invalid(&|t| t.version = 1);
}

#[test]
fn recorded_traces_preserve_inputs_histories_and_force_verify_owner_splits() {
    for (concurrency, verify) in [(1, false), (16, false), (32, false), (16, true)] {
        let trace = Trace::construct(vec![vec![3; 128]; concurrency], 1024, verify);
        trace.validate().unwrap();
        let replay: Trace = serde_json::from_slice(&serde_json::to_vec(&trace).unwrap()).unwrap();
        assert_eq!(replay, trace);
        if verify {
            let step = &trace.steps[2];
            assert_eq!(step.rows(), 32);
            let mut start = 0;
            let mut splits = 0;
            for item in &step.items {
                let end = start + item.drafts + 1;
                splits += usize::from(start / 8 != (end - 1) / 8);
                start = end;
            }
            assert_eq!(splits, 2);
        }
        for step in &trace.steps {
            if step.mode == Mode::Grammar {
                continue;
            }
            let items = step.model_items(None);
            let expanded = items
                .iter()
                .flat_map(|item| {
                    let count = match item.phase() {
                        runtime_scheduler::BatchPhase::Verify { draft_tokens } => draft_tokens + 1,
                        _ => 1,
                    };
                    (0..count).map(move |row| {
                        item.clone()
                            .with_sampling(item.sampling().unwrap().for_verify_row(&[3, 9], row))
                    })
                })
                .collect::<Vec<_>>();
            let refs = expanded.iter().collect::<Vec<_>>();
            let dispatch = SamplingDispatch::for_items(&refs).unwrap();
            for item in &expanded {
                let meta = metadata(item, dispatch);
                assert_eq!(
                    meta["history"].as_array().unwrap().len()
                        - meta["emitted"].as_u64().unwrap() as usize,
                    128
                );
            }
            #[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
            for part in refs.chunks(8) {
                crate::sampling::validate_trace_metadata_for_test(part, 1024).unwrap();
            }
        }
        let mut invalid = trace.clone();
        invalid.steps[1].items[0].emitted = 999;
        assert!(invalid.validate().is_err());
        let mut invalid = trace;
        invalid.steps[1].items[0].history.push(5);
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn bundle_proposals_cover_full_and_partial_verify_buckets() {
    for concurrency in [16, 32] {
        for width in 2..=8 {
            let prompts = (0..concurrency)
                .map(|slot| (0..128).map(|i| (i + slot) as u32).collect())
                .collect();
            let trace = Trace::construct_verify(prompts, 1024, width).unwrap();
            let replay: Trace =
                serde_json::from_slice(&serde_json::to_vec(&trace).unwrap()).unwrap();
            assert_eq!(trace, replay);
            assert!(trace.matches_verify_rows(width));
            assert_eq!(trace.verify_capacity(32).unwrap(), 32 * width);
            assert_eq!(
                trace.verify_capacity(concurrency).unwrap(),
                concurrency * width
            );
            assert!(trace.verify_capacity(concurrency - 1).is_err());
            assert!(trace.verify_capacity(33).is_err());
            assert!(!trace.matches_verify_rows(if width == 8 { 7 } else { 8 }));
            for (index, step) in trace.steps.iter().enumerate().skip(1) {
                assert_eq!(
                    step.rows(),
                    concurrency * (width - usize::from(index % 2 == 0))
                );
                if step.mode == Mode::Grammar {
                    continue;
                }
                // Grammar rows share the filtered sampling parameters; their compiled
                // constraint only adds the mask, written here at its widest.
                let mut unconstrained = step.clone();
                if step.mode == Mode::Grammar {
                    unconstrained.mode = Mode::Filtered;
                }
                for (item, model) in step.items.iter().zip(unconstrained.model_items(None)) {
                    assert_eq!(&model.input_tokens()[1..], item.fixed_drafts);
                    if item.drafts > 0 {
                        assert_eq!(
                            model.draft().unwrap().source(),
                            runtime_engine::DraftSource::Bundle
                        );
                        assert_eq!(model.draft().unwrap().tokens(), item.drafts);
                    }
                    assert_eq!(model.sampling().unwrap().history.committed(), item.history);
                    assert_eq!(item.kept, 1);
                }
            }
            for change in 0..4 {
                let mut invalid = trace.clone();
                match change {
                    0 => invalid.steps[1].items[0].fixed_drafts.push(1),
                    1 => invalid.steps[1].items[0].inputs[0] = 1024,
                    2 => invalid.steps[1].items[0].drafts = 8,
                    _ => invalid.version = 0,
                }
                assert!(
                    invalid.validate().is_err(),
                    "invalid bundle proposal trace {change}"
                );
            }
        }
    }
    for width in [0, 1, 9] {
        assert!(Trace::construct_verify(vec![vec![1; 128]; 16], 1024, width).is_err());
    }
}

#[test]
fn live_row_owners_exclude_graph_padding_and_match_each_row() {
    for (live, block) in [
        (48, 16),
        (96, 32),
        (1, 1),
        (16, 4),
        (32, 8),
        (64, 16),
        (128, 32),
    ] {
        let owners = (0..live)
            .map(|row| (row / block) as u32)
            .collect::<Vec<_>>();
        super::trace::validate_row_owners(&owners, 4, true).unwrap();
        super::trace::validate_row_owners(&vec![0; live], 4, false).unwrap();
        let mut misplaced = owners.clone();
        misplaced[0] = 3;
        assert!(super::trace::validate_row_owners(&misplaced, 4, true).is_err());
        assert!(super::trace::validate_row_owners(&misplaced, 4, false).is_err());
    }
}

fn description(root: &std::path::Path) -> Description {
    Description {
        invariants: json!({"host":{"build":"fixture","binary_sha256":"fixture"},"trace":"two-steps",
            "vocabulary":4,"topology":"fixture","ranks":1,"fold":false,"capacity":1,"sampling":"fixture"}),
        manifest: json!({"native_module":{"sha256":"original"},"components":{
            "frontend":{"sha256":"original"},"weights":{"sha256":"original"}},"state_layout":1}),
        environment: json!({}),
        source_checkpoint: Some(json!("fixture-checkpoint")),
        declared_differences: vec![],
        expected_steps: 2,
        expected_rows: 2,
        limits: Limits::default(),
        context: json!({"fixture":true,"observed_graph_execution":null}),
        reproduction: vec![
            "env".into(),
            format!(
                "CHOPINFER_DIAGNOSTIC_FIXTURE_DIR={}",
                root.join("reproduced").display()
            ),
            std::env::current_exe()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            "--exact".into(),
            "executor::fixed_batch::tests::diagnostic_fixture_reports_first_divergence".into(),
            "--nocapture".into(),
        ],
    }
}

fn fixture_rows() -> Vec<CapturedRow> {
    let mut data = rows();
    data[0].width = 4;
    data[0].bytes = [1.0_f32, 2.0, -0.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    data
}

fn fixture_frame(data: &[CapturedRow], step: u64) -> Frame {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let wire = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(1, true, 1, true, vec![3])],
        vec![BatchAppendV1::new(step, 1, step + 1, vec![0], None)],
    )
    .with_state_slots(vec![7])
    .encode()
    .unwrap();
    Frame::from_rows(wire, vec![(1, 1)], 1, vec![], data)
}

fn finish_record(golden: &mut Golden, root: &std::path::Path, baseline: bool) {
    golden.finish(2).unwrap();
    diagnostic::write_json(&root.join("receipt.json"), &json!({"passed":true}), 1024).unwrap();
    golden
        .publish(
            if baseline { "baseline" } else { "match" },
            "fixture complete",
        )
        .unwrap();
}

fn report(root: &std::path::Path, name: &str, code: i32) -> serde_json::Value {
    let tool =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/dev/diagnostics.py");
    let output = std::process::Command::new("python3")
        .arg("-B")
        .arg(tool)
        .arg("compare")
        .arg("--reference")
        .arg(root.join("baseline"))
        .arg("--candidate")
        .arg(root.join(name))
        .args(["--mode", "fixed-schedule"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(code),
        "{} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn diagnostic_fixture_reports_first_divergence() {
    let scratch = std::env::var_os("CHOPINFER_DIAGNOSTIC_FIXTURE_DIR")
        .is_none()
        .then(Scratch::new);
    let root = scratch.as_ref().map(|s| s.0.clone()).unwrap_or_else(|| {
        let path = PathBuf::from(std::env::var_os("CHOPINFER_DIAGNOSTIC_FIXTURE_DIR").unwrap());
        diagnostic::external(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        path
    });
    let desc = description(&root);
    let identity = json!({"fixture":"two steps"});
    let baseline = root.join("baseline");
    let mut golden =
        Golden::open_diagnostic(&baseline, None, None, &identity, desc.clone()).unwrap();
    for step in 0..2 {
        let data = fixture_rows();
        golden.step(&fixture_frame(&data, step), &data).unwrap();
    }
    finish_record(&mut golden, &baseline, true);
    for (name, fault) in [("matching", false), ("mismatch", true)] {
        let path = root.join(name);
        let control = fault.then(|| root.join("matching"));
        let mut golden = Golden::open_diagnostic(
            &path,
            Some(&baseline),
            control.as_deref(),
            &identity,
            desc.clone(),
        )
        .unwrap();
        let data = fixture_rows();
        golden.step(&fixture_frame(&data, 0), &data).unwrap();
        let mut data = fixture_rows();
        if fault {
            data[0].bytes[8..12].copy_from_slice(&1.25_f32.to_le_bytes());
        }
        let result = golden.step(&fixture_frame(&data, 1), &data);
        if fault {
            assert!(result.is_err());
        } else {
            result.unwrap();
            finish_record(&mut golden, &path, false);
        }
    }
    let matching = report(&root, "matching", 0);
    assert_eq!(matching["status"], "match");
    let mismatch = report(&root, "mismatch", 1);
    assert_eq!(mismatch["first"]["stage"], "logits");
    for (key, value) in [
        ("step", 1),
        ("row", 0),
        ("request", 1),
        ("element", 2),
        ("item", 0),
        ("state_slot", 7),
    ] {
        assert_eq!(mismatch["first"][key], value, "{key}");
    }
    assert_eq!(mismatch["first"]["shape"], json!([4]));
    assert_eq!(mismatch["first"]["dtype"], "f32");
    assert_eq!(mismatch["first"]["expected"]["bits"], "0x80000000");
    assert_eq!(mismatch["first"]["observed"]["value"], 1.25);
    assert_eq!(mismatch["reproduction"], json!(desc.reproduction));
    let mut declared = desc.clone();
    declared.manifest["native_module"]["sha256"] = json!("candidate");
    declared.manifest["components"]["frontend"]["sha256"] = json!("candidate");
    declared.declared_differences =
        diagnostic::changes("manifest", &desc.manifest, &declared.manifest);
    let changed_path = root.join("declared-change");
    let mut changed = Golden::open_diagnostic(
        &changed_path,
        Some(&baseline),
        None,
        &identity,
        declared.clone(),
    )
    .unwrap();
    for step in 0..2 {
        let data = fixture_rows();
        changed.step(&fixture_frame(&data, step), &data).unwrap();
    }
    finish_record(&mut changed, &changed_path, false);
    assert_eq!(
        report(&root, "declared-change", 0)["declared_differences"],
        json!(declared.declared_differences)
    );
    let error = Golden::open_diagnostic(
        &root.join("bad-repeat"),
        Some(&baseline),
        Some(&changed_path),
        &identity,
        desc.clone(),
    )
    .err()
    .expect("a package experiment cannot stand in for the baseline repeat");
    assert!(error.to_string().contains("repeat"));
    assert_eq!(report(&root, "bad-repeat", 3)["status"], "insufficient");
    for name in ["schedule", "sample"] {
        let mut changed = Golden::open_diagnostic(
            &root.join(name),
            Some(&baseline),
            None,
            &identity,
            desc.clone(),
        )
        .unwrap();
        let data = fixture_rows();
        changed.step(&fixture_frame(&data, 0), &data).unwrap();
        let mut frame = fixture_frame(&data, if name == "schedule" { 999 } else { 1 });
        if name == "sample" {
            frame.rows[0].sample.token = 2;
        }
        assert!(changed.step(&frame, &data).is_err());
        let found = report(&root, name, 1);
        assert_eq!(found["first"]["stage"], name);
        assert_eq!(found["first"]["step"], 1);
        assert_eq!(found["first"]["request"], 1);
    }
    let mut incompatible = desc.clone();
    incompatible.invariants["capacity"] = json!(2);
    assert!(
        Golden::open_diagnostic(
            &root.join("incompatible"),
            Some(&baseline),
            None,
            &identity,
            incompatible
        )
        .is_err()
    );
    assert_eq!(report(&root, "incompatible", 2)["status"], "incompatible");
    let mut over_budget = desc.clone();
    over_budget.limits.step_bytes = 8;
    let mut limited = Golden::open_diagnostic(
        &root.join("over-budget"),
        Some(&baseline),
        None,
        &identity,
        over_budget,
    )
    .unwrap();
    let data = fixture_rows();
    assert!(limited.step(&fixture_frame(&data, 0), &data).is_err());
    assert_eq!(report(&root, "over-budget", 3)["status"], "insufficient");
    let mut incomplete = Golden::open_diagnostic(
        &root.join("incomplete"),
        Some(&baseline),
        None,
        &identity,
        desc,
    )
    .unwrap();
    let data = fixture_rows();
    incomplete.step(&fixture_frame(&data, 0), &data).unwrap();
    drop(incomplete);
    assert_eq!(report(&root, "incomplete", 3)["status"], "insufficient");
    for name in [
        "stale",
        "truncated",
        "trailing",
        "legacy",
        "invalid-table",
        "invalid-count",
        "wrong-repeat",
    ] {
        let path = root.join(name);
        std::fs::create_dir(&path).unwrap();
        for entry in std::fs::read_dir(root.join("matching")).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), path.join(entry.file_name())).unwrap();
        }
        if matches!(
            name,
            "legacy" | "invalid-table" | "invalid-count" | "wrong-repeat"
        ) {
            let file = path.join("diagnostic.json");
            let mut terminal = diagnostic::read_json(&file, 16 << 20).unwrap();
            match name {
                "legacy" => {
                    terminal.as_object_mut().unwrap().remove("description");
                }
                "invalid-table" => terminal["artifacts"] = json!([]),
                "wrong-repeat" => terminal["control"] = diagnostic::binding(&changed_path).unwrap(),
                _ => terminal["description"]["expected_rows"] = json!(false),
            }
            std::fs::write(file, serde_json::to_vec(&terminal).unwrap()).unwrap();
        } else {
            let file = path.join("frames.jsonl");
            let mut data = std::fs::read(&file).unwrap();
            match name {
                "truncated" => {
                    data.pop();
                }
                "trailing" => data.push(b'\n'),
                _ => data[1] ^= 1,
            }
            std::fs::write(&file, data).unwrap();
        }
        assert_eq!(report(&root, name, 3)["status"], "insufficient");
    }
    eprintln!("diagnostic fixture: {}", root.display());
}

#[test]
fn diagnostic_compatibility_requires_pinned_differences_and_replay_invariants() {
    let scratch = Scratch::new();
    let a = description(&scratch.0);
    for key in [
        "host",
        "trace",
        "vocabulary",
        "topology",
        "ranks",
        "fold",
        "capacity",
        "sampling",
    ] {
        let mut b = a.clone();
        b.invariants[key] = json!("changed");
        assert!(
            diagnostic::compatible(&a, &b)
                .unwrap_err()
                .to_string()
                .contains(key)
        );
    }
    let mut b = a.clone();
    b.source_checkpoint = Some(json!("other"));
    assert!(diagnostic::compatible(&a, &b).is_err());
    b.source_checkpoint = None;
    diagnostic::compatible(&a, &b).unwrap();
    b.manifest["components"]["frontend"]["sha256"] = json!("changed");
    b.manifest["components"]["weights"]["sha256"] = json!("repacked");
    b.manifest["native_module"]["sha256"] = json!("candidate");
    b.manifest["state_layout"] = json!(2);
    assert!(diagnostic::compatible(&a, &b).is_err());
    b.declared_differences = diagnostic::changes("manifest", &a.manifest, &b.manifest);
    diagnostic::compatible(&a, &b).unwrap();
    b.declared_differences[0].observed = json!("stale");
    assert!(diagnostic::compatible(&a, &b).is_err());
}

#[test]
fn diagnostic_budgets_and_explicit_trace_are_checked_without_a_device() {
    let scratch = Scratch::new();
    let path = scratch.0.join("trace.json");
    let trace = Trace::select(&path, false, || {
        Ok(Trace::construct(vec![vec![3; 128]; 16], 1024, false))
    })
    .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let replay =
        Trace::select(&path, true, || panic!("explicit trace must not regenerate")).unwrap();
    assert_eq!(trace, replay);
    assert_eq!(bytes, std::fs::read(&path).unwrap());
    assert!(Trace::select(&path, false, || Ok(trace.clone())).is_err());
    let limits = Limits {
        step_bytes: 16,
        logits_bytes: 24,
        ..Limits::default()
    };
    assert!(limits.estimate([2].into_iter(), 4).is_err());
    let mut budget = Budget::new(limits);
    budget.reserve(1, 4, 1).unwrap();
    assert!(budget.reserve(1, 4, 1).is_err());
    budget.next_step();
    budget.reserve(2, 4, 1).unwrap();
    assert!(budget.reserve(2, 4, 1).is_err());
    assert!(diagnostic::row_bytes(9, 4).is_err());
    let options = diagnostic::Options {
        environment: json!({"CHOPINFER_TEST_ROUTE":"1"}),
        declared_differences: vec![],
        reproduction: vec!["fixture".into()],
        limits: Limits::default(),
    };
    options
        .validate_environment([("CHOPINFER_TEST_ROUTE".into(), "1".into())].into_iter())
        .unwrap();
    assert!(
        options
            .validate_environment([("CHOPINFER_TEST_ROUTE".into(), "0".into())].into_iter())
            .is_err()
    );
}

/// Worst-case row evidence of every step: histories extended by the item's
/// drafts with ten-digit tokens, grammar masks of ten-digit words, and sample
/// bits with 20 top logprobs of twenty-digit bits. Each step's frame must fit
/// the trace's metadata bound.
#[test]
fn metadata_bound_covers_drafter_frames_with_grammar_masks() {
    let vocabulary = 248_320;
    let trace = Trace::construct_verify(vec![vec![7; 128]; 16], vocabulary, 8).unwrap();
    let dispatch = SamplingDispatch {
        rows: u32::MAX,
        simple: false,
        top: true,
        greedy: false,
    };
    let mut written = 0_u64;
    for step in &trace.steps {
        let mut rows = Vec::new();
        // Grammar rows share the filtered sampling parameters; their compiled
        // constraint only adds the mask, written here at its widest.
        let mut unconstrained = step.clone();
        if step.mode == Mode::Grammar {
            unconstrained.mode = Mode::Filtered;
        }
        for (item, model) in step.items.iter().zip(unconstrained.model_items(None)) {
            for _ in 0..=item.drafts {
                let mut meta = metadata(&model, dispatch);
                let history = meta["history"].as_array_mut().unwrap();
                history.extend((0..=item.drafts).map(|_| json!(u32::MAX)));
                if step.mode == Mode::Grammar {
                    meta["mask"] = json!(vec![u32::MAX; vocabulary.div_ceil(32)]);
                }
                rows.push(CapturedRow {
                    rank: u32::MAX,
                    element_type: 1,
                    width: u32::MAX,
                    bytes: Vec::new(),
                    metadata: meta,
                    sample: Some(SampleBits {
                        request: u64::MAX,
                        token: u32::MAX,
                        logprob: u64::MAX,
                        top: vec![(u32::MAX, u64::MAX); 20],
                    }),
                });
            }
        }
        let drafts = step
            .items
            .iter()
            .filter(|item| item.drafts > 0)
            .map(|item| (u64::MAX, vec![u32::MAX; item.drafts]))
            .collect();
        let kept = step.items.iter().map(|_| (u64::MAX, usize::MAX)).collect();
        let frame = Frame::from_rows(Vec::new(), kept, usize::MAX, drafts, &rows);
        written += serde_json::to_vec(&frame).unwrap().len() as u64 + 1;
    }
    let bound = trace.metadata_bound();
    assert!(
        written <= bound,
        "frames {written} exceed the bound {bound}"
    );
    // The previous 16 MiB default could not hold this C16 trace; the
    // replay now refuses it before any step instead of at step 8.
    assert!(written > 16 << 20);
    let limits = Limits::default();
    assert!(
        limits
            .check_metadata(bound)
            .unwrap_err()
            .to_string()
            .contains("--metadata-bytes")
    );
}

/// The largest package comparison the runner prepares (32 requests, eight
/// verify rows each) at the sampling contract's widest vocabulary fits the
/// hard metadata cap, which is the runner's default budget.
#[test]
fn widest_comparison_trace_fits_the_metadata_cap() {
    let trace = Trace::construct_verify(vec![vec![7; 128]; 32], 1 << 20, 8).unwrap();
    let identity = serde_json::to_vec(&json!({"trace": trace})).unwrap().len() as u64;
    let bound = trace.metadata_bound() + identity + 64 * 1024;
    assert!(bound <= diagnostic::MAX_METADATA_BYTES, "{bound}");
    let limits = Limits {
        metadata_bytes: diagnostic::MAX_METADATA_BYTES,
        ..Limits::default()
    };
    limits.check_metadata(bound).unwrap();
}

#[test]
fn prefill_only_traces_preserve_anchors_and_bound_work() {
    for lengths in [[1025, 1023], [1030, 1000], [100, 1000]] {
        let mut trace = Trace::construct(lengths.map(|n| vec![3; n]).to_vec(), 1024, false);
        assert!(!serde_json::to_string(&trace).unwrap().contains("anchor"));
        trace.version = 4;
        trace.steps.truncate(1);
        trace.steps[0].items[0].anchor = true;
        trace.validate().unwrap();
        let model = trace.steps[0].model_items(None);
        assert!(model[0].anchor());
        assert!(!model[1].anchor());
        assert_eq!(model[0].input_tokens().len(), lengths[0]);
        let encoded = serde_json::to_vec(&trace).unwrap();
        assert_eq!(serde_json::from_slice::<Trace>(&encoded).unwrap(), trace);
        let mut invalid = trace.clone();
        invalid.steps.push(trace.steps[0].clone());
        assert!(invalid.validate().is_err());
        invalid = trace.clone();
        invalid.version = 1;
        assert!(invalid.validate().is_err());
        invalid = trace.clone();
        invalid.steps[0].items[0].inputs = vec![3; 8193];
        invalid.steps[0].items[0].history = vec![3; 8193];
        invalid.steps[0].items[0].kept = 8193;
        assert!(invalid.validate().is_err());
    }
}
