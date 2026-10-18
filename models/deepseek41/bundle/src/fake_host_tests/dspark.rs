//! Full synchronous instances with drafter rows and mapped Engram tables.
//! These journals prove ordering and spans; fake kernels do not compute logits.
use super::*;
use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

pub(super) mod host;
mod package;
type Sink = RecordingSink<super::conformance::Limits>;

pub(super) fn index(
    shape: &Shape,
    rank: u32,
    layouts: &BTreeMap<String, String>,
) -> (Vec<u8>, u64) {
    let mut all = super::views(shape);
    let g = shape.geometry;
    let hidden = u64::from(g.shape.hidden);
    for stage in 0..g.dspark_stages() {
        for (name, mut bytes, dtype) in super::views(shape) {
            let Some(suffix) = name.strip_prefix("layers.0.") else {
                continue;
            };
            if suffix.starts_with("ffn.gate.") || suffix.starts_with("ffn.experts.") {
                bytes = bytes / u64::from(g.shape.moe.experts) * u64::from(g.shape.dspark.experts);
            }
            all.push((format!("mtp.{stage}.{suffix}"), bytes, dtype));
        }
    }
    let main = hidden * hidden * g.shape.dspark.targets.len() as u64;
    all.extend([
        ("mtp.0.main_proj.weight".into(), main, "U8"),
        ("mtp.0.main_proj.scale".into(), main / 32, "U8"),
        ("mtp.0.main_norm.weight".into(), hidden * 2, "BF16"),
    ]);
    let last = g.dspark_stages() - 1;
    let markov = u64::from(g.shape.dspark.markov_rank);
    all.push((format!("mtp.{last}.norm.weight"), hidden * 2, "BF16"));
    for name in ["embed", "head"] {
        all.push((
            format!("mtp.{last}.markov_head.{name}.weight"),
            u64::from(shape.vocab_rows()) * markov * 2,
            "BF16",
        ));
    }
    all.push((
        format!("mtp.{last}.confidence_head.proj.weight"),
        (hidden + markov) * 2,
        "BF16",
    ));
    for (i, layer) in g.shape.engram.layers.iter().enumerate() {
        let bytes = u64::from(g.shape.engram.rows[i]) * u64::from(g.shape.engram.head_dim);
        all.push((format!("layers.{layer}.engram.embed.weight"), bytes, "U8"));
        all.push((
            format!("layers.{layer}.engram.embed.scale"),
            bytes / 32,
            "U8",
        ));
    }
    let mut index = FakeIndex::new(INDEX_SCHEMA, rank);
    for (name, bytes, dtype) in all {
        let (element, layout) = match dtype {
            "F32" => (4, "f32_row_major_v1"),
            "U8" => (1, "row-major"),
            _ => (2, "bf16_row_major_v1"),
        };
        let layout = layouts.get(&name).map_or(layout, String::as_str);
        index.push(
            name.clone(),
            bytes,
            dtype,
            vec![bytes / element],
            layout,
            VIEW_ALIGNMENT,
        );
    }
    let bytes = index.end(VIEW_ALIGNMENT);
    (index.bytes(bytes), bytes)
}

fn create(shape: Shape) -> (Rc<Host>, BundleInstanceV1) {
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.drafter.set(true);
    host.speculation.set(Some((5, 32)));
    let record = crate::config::tests::record(&shape, host.kv, 4, false);
    let instance = api()
        .create(
            HostServicesTableV1::new(Rc::new(host::MappedHost(Rc::clone(&host)))),
            &record,
        )
        .unwrap_or_else(|e| panic!("drafter create: {e:?} {:?}", host.logs.borrow()));
    (host, instance)
}

fn verify(request: u64, slot: u64, start: u64, drafts: u32) -> Vec<u8> {
    if drafts == 0 {
        return super::decode(&[(request, slot, start, 7)]);
    }
    let rows = u64::from(drafts + 1);
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::verify(request, drafts + 1, true, vec![7])],
        vec![BatchAppendV1::new(
            start,
            rows,
            start + rows,
            (0..(start + rows).div_ceil(64))
                .map(|page| slot * 8 + page)
                .collect(),
            None,
        )],
    )
    .with_state_slots(vec![slot])
    .encode()
    .unwrap()
}

#[track_caller]
fn execute(instance: &mut BundleInstanceV1, host: &Host, batch: &[u8]) -> Sink {
    let mut sink = Sink::default();
    instance.execute(batch, &mut sink).unwrap_or_else(|e| {
        panic!(
            "drafter execute: {e:?}, {:?}, {:?}",
            instance.last_execute_refusal(),
            host.logs.borrow()
        )
    });
    sink
}

fn state(instance: &BundleInstanceV1, request: u64) -> crate::dspark::state::Context {
    crate::instance::dspark_context(instance.handle(), request).expect("committed drafter context")
}

#[test]
fn instance_configures_draft_attention_before_execution() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let (host, mut instance) = create(shape);
        let journal = host.journal.borrow();
        let mut attention = 0;
        let symbol = format!(
            "chop_latent_sources_deepseek41_h{}_d512_window_normalized_s1_r64_verify_draft_block_v3",
            shape.local_heads()
        );
        for line in journal.iter() {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.len() == 3 && fields[0] == "resolve" && fields[2] == symbol {
                let prefix = format!("configure {} ", fields[1]);
                let bytes = journal
                    .iter()
                    .find_map(|line| line.strip_prefix(&prefix))
                    .and_then(|bytes| bytes.parse::<u32>().ok())
                    .unwrap_or(48 * 1024);
                assert!(
                    bytes >= 83_200,
                    "{} leaves {} at {bytes} dynamic shared bytes",
                    shape.topology(),
                    fields[2]
                );
                attention += 1;
            }
        }
        assert_eq!(attention, shape.ranks());
        drop(journal);
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn full_instance_drafts_and_retains_all_topologies() {
    for shape in crate::shape::topologies() {
        let (host, mut instance) = create(shape);
        execute(&mut instance, &host, &super::prompt(7, 0, 0, 3, true));
        instance.commit().unwrap();
        let before = host.journal.borrow().len();
        let sink = execute(&mut instance, &host, &verify(7, 0, 3, 5));
        assert_eq!(sink.draft_tokens(), vec![(7, vec![0; 5])]);
        assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 6);
        instance.commit_accepted(&[kept(7, 6)]).unwrap();
        let context = state(&instance, 7);
        assert_eq!(context.end, 9);
        assert_eq!(
            context.projected(),
            vec![3; shape.geometry.dspark_stages() as usize]
        );
        assert!(
            host.journal.borrow()[before..]
                .iter()
                .any(|line| line.starts_with("mapped gather"))
        );
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|line| line.contains(crate::dspark::calibration::SCHEMA))
        );
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

fn kept(request_id: u64, kept_rows: u32) -> BundleKeptRowsV1 {
    BundleKeptRowsV1 {
        struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
        reserved: 0,
        request_id,
        kept_rows,
        reserved_tail: 0,
    }
}

fn records(host: &Host) -> Vec<serde_json::Value> {
    host.logs
        .borrow()
        .iter()
        .filter_map(|line| {
            let value: serde_json::Value = serde_json::from_str(line).ok()?;
            (value["schema"] == crate::dspark::calibration::SCHEMA).then_some(value)
        })
        .collect()
}

#[test]
fn instance_partial_acceptance_ar_wrap_cancel_restore_and_slot_reuse() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (host, mut instance) = create(shape);
    let prompt_rows = 141;
    let ids = (0..prompt_rows as u32).collect::<Vec<_>>();
    execute(
        &mut instance,
        &host,
        &super::prompt(1, 0, 0, prompt_rows, true),
    );
    instance.commit().unwrap();
    let pages = (0..instance.describe_prefix_state().unwrap())
        .map(|i| 40 + i)
        .collect::<Vec<_>>();
    instance
        .capture_prefix_state(1, prompt_rows, &pages)
        .unwrap();
    assert_eq!(state(&instance, 1).projected(), vec![prompt_rows; 3]);
    instance.retire_request(1).unwrap();
    for accepted in 0..=5 {
        let request = 10 + u64::from(accepted);
        instance
            .restore_prefix_state_in_slot(request, &ids, &pages, 2)
            .unwrap();
        execute(&mut instance, &host, &verify(request, 2, prompt_rows, 5));
        instance
            .commit_accepted(&[kept(request, accepted + 1)])
            .unwrap();
        let mut end = prompt_rows + u64::from(accepted + 1);
        assert_eq!(state(&instance, request).end, end);
        for _ in 0..3 {
            let sink = execute(&mut instance, &host, &verify(request, 2, end, 0));
            assert!(sink.drafts.is_empty());
            instance.commit().unwrap();
            end += 1;
        }
        execute(&mut instance, &host, &verify(request, 2, end, 5));
        assert_eq!(state(&instance, request).projected(), vec![end; 3]);
        instance.cancel().unwrap();
        assert!(crate::instance::dspark_context(instance.handle(), request).is_none());
        assert_eq!(
            crate::instance::ledger_state(instance.handle(), 2),
            (None, false)
        );
        instance.retire_request(request).unwrap();
        let replacement = request + 100;
        execute(
            &mut instance,
            &host,
            &super::prompt(replacement, 2, 0, 2, true),
        );
        instance.commit().unwrap();
        assert_eq!(state(&instance, replacement).end, 2);
        instance.retire_request(replacement).unwrap();
    }
    let diagnostics = records(&host);
    assert_eq!(diagnostics.len(), 12);
    for (accepted, pair) in diagnostics.chunks_exact(2).enumerate() {
        let labels = pair[0]["outcomes"].as_array().unwrap();
        assert_eq!(
            labels.iter().filter(|v| **v == "accepted").count(),
            accepted
        );
        assert!(labels[accepted..].iter().all(|v| *v == "unknown"));
        assert_eq!(pair[1]["completion"], "cancelled");
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn instance_budget_prefixes_compute_the_same_full_drafter_program() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (host, mut instance) = create(shape);
    execute(&mut instance, &host, &super::prompt(1, 0, 0, 3, true));
    instance.commit().unwrap();
    let pages = (0..instance.describe_prefix_state().unwrap())
        .map(|i| 40 + i)
        .collect::<Vec<_>>();
    instance.capture_prefix_state(1, 3, &pages).unwrap();
    instance.retire_request(1).unwrap();
    let mut full_program = None;
    let mut full_tokens = Vec::new();
    for budget in [5, 1, 2, 3, 4] {
        let request = u64::from(budget) + 10;
        instance
            .restore_prefix_state_in_slot(request, &[0, 1, 2], &pages, 0)
            .unwrap();
        let start = host.journal.borrow().len();
        let sink = execute(&mut instance, &host, &verify(request, 0, 3, budget));
        let tokens = &sink.drafts[0].tokens;
        let journal = host.journal.borrow();
        let draft = journal[start..]
            .iter()
            .take_while(|line| !line.starts_with("mapped gather"));
        let program = draft
            .filter(|line| line.starts_with("launch "))
            .cloned()
            .collect::<Vec<_>>();
        assert!(!program.is_empty());
        if let Some(full) = &full_program {
            assert_eq!(&program, full, "K={budget} changed a full-block launch");
            assert_eq!(tokens, &full_tokens[..budget as usize]);
        } else {
            full_tokens = tokens.clone();
            full_program = Some(program);
        }
        drop(journal);
        instance.commit_accepted(&[kept(request, 1)]).unwrap();
        let record = records(&host).pop().unwrap();
        assert_eq!(record["computed_positions"], 5);
        assert_eq!(record["emitted_positions"], budget);
        assert_eq!(record["verify_rows"], budget + 1);
        assert_eq!(record["confidence_f32_bits"].as_array().unwrap().len(), 5);
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn instance_mixed_phases_keep_disjoint_features_and_refuse_slot_aliases_before_io() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (host, mut instance) = create(shape);
    execute(&mut instance, &host, &super::prompt(1, 0, 0, 3, true));
    instance.commit().unwrap();
    let batch = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![
            BatchItemV1::new(2, false, 4, true, vec![1, 2, 3, 4]),
            BatchItemV1::verify(1, 6, true, vec![7]),
        ],
        vec![
            BatchAppendV1::new(0, 4, 4, vec![8], None),
            BatchAppendV1::new(3, 6, 9, vec![0], None),
        ],
    )
    .with_state_slots(vec![1, 0])
    .encode()
    .unwrap();
    let sink = execute(&mut instance, &host, &batch);
    assert_eq!(sink.draft_tokens(), vec![(1, vec![0; 5])]);
    instance.commit_accepted(&[kept(1, 3)]).unwrap();
    assert_eq!(state(&instance, 1).end, 6);
    assert_eq!(state(&instance, 2).end, 4);
    let before = host.journal.borrow().len();
    let mut sink = Sink::default();
    assert!(instance.execute(&verify(1, 1, 6, 5), &mut sink).is_err());
    assert_eq!(host.journal.borrow().len(), before);
    assert_eq!(state(&instance, 1).end, 6);
    let sink = execute(&mut instance, &host, &verify(1, 0, 6, 1));
    assert_eq!(sink.draft_tokens(), vec![(1, vec![0])]);
    instance.commit_accepted(&[kept(1, 2)]).unwrap();
    instance.retire_request(1).unwrap();
    instance.retire_request(2).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
