//! Full synchronous instances with drafter rows and mapped Engram tables.
//! These journals prove ordering and spans; fake kernels do not compute logits.
use super::*;
use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

mod boundary;
mod catch_up;
mod coupled;
mod early;
mod graph_budget;
pub(super) mod host;
mod item_keys;
mod layers;
#[cfg(test)]
mod native_package;
mod prefetch;
mod results;
mod round_trips;
mod undrafted;
pub(super) use host::MappedHost;
type Sink = RecordingSink<super::conformance::Limits>;

#[test]
fn fold_scratch_keeps_target_and_drafter_allocations_disjoint() {
    let (host, mut instance) = create(Shape::new(&FLASH, Ranks::TP4).unwrap());
    super::fold_stream::audit_allocations(&host, &instance, true);
    // Drafter blocks lower whole five-row blocks, which no stream bucket
    // serves; only catch-up projections hold drafter fold storage.
    for views in crate::instance::fold_views_for_test(instance.handle()) {
        for (name, _) in views.iter().filter(|(name, _)| name.starts_with("mtp.")) {
            assert!(
                name.starts_with("mtp.0.main_proj.") || name.contains(".attn.wkv."),
                "{name}"
            );
        }
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn target_graphs_capture_each_exact_verify_width() {
    let (host, mut instance) = create(Shape::new(&FLASH, Ranks::TP4).unwrap());
    execute(&mut instance, &host, &super::prompt(7, 0, 0, 3, true));
    instance.commit().unwrap();
    let mut graphs = std::collections::BTreeMap::new();
    for rows in [3, 4, 5, 6, 3, 4, 5, 6] {
        let start = state(&instance, 7).end;
        let begin = host.journal.borrow().len();
        let sink = execute(&mut instance, &host, &verify(7, 0, start, rows - 1));
        assert_eq!(sink.draft_tokens(), [(7, vec![0; (rows - 1) as usize])]);
        let journal = host.journal.borrow();
        let step = &journal[begin..];
        // Each Engram gather joins immediately before its target consumer.
        // Hashing separates the drafter graphs from the full target schedule.
        let hashed = step
            .iter()
            .position(|line| line.contains(crate::sm120::engram_hash_rows::SYMBOL))
            .unwrap();
        let target = &step[hashed + 1..];
        let handles = |event: &str| {
            let mut ids: Vec<_> = target
                .iter()
                .filter(|line| line.starts_with(event))
                .map(|line| line.split_whitespace().nth(2).unwrap().to_owned())
                .collect();
            ids.sort();
            ids
        };
        let captured = handles("captured ");
        let replayed = handles("replay ");
        if let Some(previous) = graphs.get(&rows) {
            assert!(captured.is_empty());
            assert_eq!(&replayed, previous);
        } else {
            assert_eq!(
                captured.len(),
                12,
                "width {rows} must capture three target pieces on every rank"
            );
            assert_eq!(replayed, captured);
            graphs.insert(rows, captured);
        }
        drop(journal);
        instance.commit_accepted(&[kept(7, 1)]).unwrap();
    }
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_state_sidecar_contains_each_committed_snapshot_piece() {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let root =
        std::path::PathBuf::from(std::env::var_os("CHOPINFER_DATA").expect("external test data"))
            .join("deepseek41-tests")
            .join(format!(
                "dspark-state-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
    std::fs::create_dir_all(root.join("first")).unwrap();
    std::fs::create_dir_all(root.join("second")).unwrap();
    let shape = Shape::new(&FLASH, Ranks::TP1).unwrap();
    let (host, mut instance) = create(shape);
    crate::instance::set_dspark_state_capture_root_for_test(instance.handle(), Some(root.clone()));
    std::fs::write(root.join("enabled"), []).unwrap();
    execute(&mut instance, &host, &super::prompt(7, 0, 0, 3, true));
    instance.commit().unwrap();
    execute(&mut instance, &host, &verify(7, 0, 3, 2));
    instance.commit_accepted(&[kept(7, 1)]).unwrap();
    for name in ["first", "second"] {
        crate::instance::capture_dspark_state_for_test(instance.handle(), 7, &root.join(name))
            .unwrap();
    }
    let first = root.join("first/request-7");
    let second = root.join("second/request-7");
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(first.join("state.json")).unwrap()).unwrap();
    assert_eq!(metadata["end"], 4);
    let pieces = metadata["pieces"].as_array().unwrap();
    for name in ["dspark.features", "dspark.ring"] {
        assert!(pieces.iter().any(|piece| piece["buffer"] == name));
    }
    let bytes: u64 = pieces
        .iter()
        .map(|piece| piece["bytes"].as_u64().unwrap())
        .sum();
    assert_eq!(
        std::fs::metadata(first.join("rank-0.bin")).unwrap().len(),
        bytes
    );
    let paged = metadata["paged_pieces"].as_array().unwrap();
    assert!(!paged.is_empty());
    assert!(paged.iter().all(|piece| piece["logical_page"] == 0));
    let paged_bytes: u64 = paged
        .iter()
        .map(|piece| piece["bytes"].as_u64().unwrap())
        .sum();
    assert_eq!(
        std::fs::metadata(first.join("kv-rank-0.bin"))
            .unwrap()
            .len(),
        paged_bytes
    );
    assert!(
        paged_bytes
            < model_deepseek41_contract::state::request_state(&FLASH, KvStorage::Native, true)
                .page_bytes()
    );
    for name in ["state.json", "rank-0.bin", "kv-rank-0.bin"] {
        assert_eq!(
            std::fs::read(first.join(name)).unwrap(),
            std::fs::read(second.join(name)).unwrap()
        );
    }
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "diagnostic-taps")]
#[test]
fn disabled_state_capture_does_not_retain_block_tables() {
    let root = std::path::PathBuf::from(std::env::var_os("CHOPINFER_DATA").unwrap())
        .join("deepseek41-tests")
        .join(format!("dspark-disabled-state-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let shape = Shape::new(&FLASH, Ranks::TP1).unwrap();
    let (host, mut instance) = create(shape);
    let handle = instance.handle();
    let captured = || crate::instance::dspark_state_capture_requests_for_test(handle);
    crate::instance::set_dspark_state_capture_root_for_test(instance.handle(), None);
    execute(&mut instance, &host, &super::prompt(7, 0, 0, 3, true));
    instance.commit().unwrap();
    assert!(
        captured().is_empty(),
        "capture without a directory retained block tables"
    );
    crate::instance::set_dspark_state_capture_root_for_test(instance.handle(), Some(root.clone()));
    for (start, active) in [(3, false), (4, true), (5, false)] {
        if active {
            std::fs::write(root.join("enabled"), []).unwrap();
        }
        execute(&mut instance, &host, &verify(7, 0, start, 2));
        instance.commit_accepted(&[kept(7, 1)]).unwrap();
        assert_eq!(captured(), if active { vec![7] } else { vec![] });
        if active {
            std::fs::remove_file(root.join("enabled")).unwrap();
        }
    }
    instance.retire_request(7).unwrap();
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    std::fs::remove_dir(root).unwrap();
}

#[test]
fn mixed_prefill_and_draft_replay_the_same_target_graph() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (host, mut instance) = create(shape);
    execute(&mut instance, &host, &super::prompt(7, 0, 0, 3, true));
    instance.commit().unwrap();
    let mut target_graphs = Vec::new();
    for (prefill_start, prefill_rows, verify_start) in [(0, 2, 3), (2, 5, 4), (7, 0, 5)] {
        let mut requests = Vec::new();
        let mut appends = Vec::new();
        let mut slots = Vec::new();
        if prefill_rows > 0 {
            requests.push(BatchItemV1::new(
                8,
                false,
                prefill_rows,
                true,
                vec![9; prefill_rows as usize],
            ));
            appends.push(BatchAppendV1::new(
                prefill_start,
                u64::from(prefill_rows),
                prefill_start + u64::from(prefill_rows),
                vec![8],
                None,
            ));
            slots.push(1);
        }
        requests.push(BatchItemV1::verify(7, 6, true, vec![7]));
        appends.push(BatchAppendV1::new(
            verify_start,
            6,
            verify_start + 6,
            vec![0],
            None,
        ));
        slots.push(0);
        let batch = BatchRequestV1::new(BatchWireVersionV1::V3, requests, appends)
            .with_state_slots(slots)
            .encode()
            .unwrap();
        let start = host.journal.borrow().len();
        execute(&mut instance, &host, &batch);
        let journal = host.journal.borrow();
        let step = &journal[start..];
        let boundary = step
            .iter()
            .position(|line| line.starts_with("async mapped ids "))
            .or_else(|| {
                step.iter()
                    .rposition(|line| line.starts_with("mapped gather"))
            })
            .expect("target rows must have an Engram producer");
        let target = &step[boundary + 1..];
        let handles = |event: &str| {
            let mut ids: Vec<_> = target
                .iter()
                .filter(|line| line.starts_with(event))
                .map(|line| line.split_whitespace().nth(2).unwrap().to_owned())
                .collect();
            ids.sort();
            ids
        };
        let captured = handles("captured ");
        let replayed = handles("replay ");
        if target_graphs.is_empty() {
            assert!(
                !captured.is_empty(),
                "DSpark target did not capture a graph"
            );
            target_graphs = captured;
        } else {
            assert!(captured.is_empty(), "prefill length changed the graph key");
        }
        assert_eq!(
            replayed, target_graphs,
            "target graph handles were not reused"
        );
        drop(journal);
        instance.commit_accepted(&[kept(7, 1)]).unwrap();
    }
    assert_eq!(state(&instance, 7).end, 6);
    assert_eq!(state(&instance, 8).end, 7);
    instance.retire_request(7).unwrap();
    instance.retire_request(8).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn drafter_graph_cache_tracks_exact_items_and_reordered_slots() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (host, mut instance) = create(shape);
    for slot in 0..3 {
        execute(
            &mut instance,
            &host,
            &super::prompt(slot + 1, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    let mut catch_up = std::collections::BTreeSet::new();
    for (slots, fresh) in [
        (vec![0], true),
        (vec![2, 0, 1], true),
        (vec![1], false),
        (vec![1, 2, 0], false),
    ] {
        let batch = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            slots
                .iter()
                .map(|&slot| BatchItemV1::verify(slot + 1, 6, true, vec![7]))
                .collect(),
            slots
                .iter()
                .map(|&slot| {
                    let start = state(&instance, slot + 1).end;
                    BatchAppendV1::new(start, 6, start + 6, vec![slot * 8], None)
                })
                .collect(),
        )
        .with_state_slots(slots.clone())
        .encode()
        .unwrap();
        let start = host.journal.borrow().len();
        execute(&mut instance, &host, &batch);
        let journal = host.journal.borrow();
        catch_up.extend(catch_up::captured_catch_up(&journal[start..]));
        let draft: Vec<_> = journal[start..]
            .iter()
            .take_while(|line| {
                !line.contains(crate::sm120::engram_hash_rows::SYMBOL)
                    && !line.starts_with("mapped gather")
            })
            .filter(|line| {
                !(line.starts_with("captured ") || line.starts_with("replay "))
                    || !catch_up.contains(line.split_whitespace().nth(2).unwrap())
            })
            .collect();
        assert_eq!(
            draft
                .iter()
                .filter(|line| line.starts_with("captured "))
                .count(),
            if fresh { 4 } else { 0 }
        );
        assert_eq!(
            draft
                .iter()
                .filter(|line| line.starts_with("replay "))
                .count(),
            4
        );
        drop(journal);
        instance
            .commit_accepted(
                &slots
                    .iter()
                    .map(|&slot| kept(slot + 1, 1))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }
    for request in 1..=3 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

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
        let (element, layout) = super::plan_layout(shape, &name, dtype);
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
    create_with_limits(shape, 4, 32)
}

fn create_with_limits(
    shape: Shape,
    items: u32,
    verify_tokens: u32,
) -> (Rc<Host>, BundleInstanceV1) {
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.drafter.set(true);
    host.speculation.set(Some((5, verify_tokens)));
    let record = crate::config::tests::record(&shape, host.kv, items, false);
    let instance = api()
        .create(
            HostServicesTableV1::new(Rc::new(MappedHost::new(Rc::clone(&host)))),
            &record,
        )
        .unwrap_or_else(|e| panic!("drafter create: {e:?} {:?}", host.logs.borrow()));
    (host, instance)
}

/// Create refuses a drafter expert bank stored in row tiles: the
/// drafter's routed ops read rows at every rank count.
#[test]
fn create_refuses_a_drafter_expert_bank_in_row_tiles() {
    use super::RoutedLayout;
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.drafter.set(true);
    host.speculation.set(Some((5, 32)));
    let view = "mtp.0.ffn.experts.gate_up.weight";
    let [tiles, _] = RoutedLayout::Tiles.view_layouts().unwrap();
    let [rows, _] = RoutedLayout::Rows.view_layouts().unwrap();
    host.layouts.borrow_mut().insert(view.into(), tiles.into());
    let record = crate::config::tests::record(&shape, host.kv, 4, false);
    let created = api().create(
        HostServicesTableV1::new(Rc::new(MappedHost::new(Rc::clone(&host)))),
        &record,
    );
    assert!(created.is_err());
    let expected = format!("{view} is stored as {tiles}, but its op reads {rows}");
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains(&expected)),
        "{:?}",
        host.logs.borrow()
    );
}

#[test]
fn full_budget_prefill_then_maximum_dspark_verify() {
    use model_deepseek41_contract::serving::{
        DECODE_BATCH, DeepSeek, MAX_VERIFY_ROWS, PREFILL_TOKENS,
    };

    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let verify_tokens = <DeepSeek as bundle_runtime::BundleGeometry>::VERIFY_TOKENS;
    let (host, mut instance) = create_with_limits(shape, DECODE_BATCH, verify_tokens);
    let prompt_rows = PREFILL_TOKENS / DECODE_BATCH;
    let slots: Vec<_> = (0..u64::from(DECODE_BATCH)).collect();
    let batch = |draft: bool, items: &[(u64, u64, u32)]| {
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            items
                .iter()
                .map(|&(slot, _, rows)| {
                    if draft {
                        BatchItemV1::verify(slot + 1, rows, true, vec![7])
                    } else {
                        BatchItemV1::new(slot + 1, false, rows, true, (0..rows).collect())
                    }
                })
                .collect(),
            items
                .iter()
                .map(|&(slot, start, rows)| {
                    let end = start + u64::from(rows);
                    BatchAppendV1::new(
                        start,
                        u64::from(rows),
                        end,
                        (0..end.div_ceil(64)).map(|page| slot * 8 + page).collect(),
                        None,
                    )
                })
                .collect(),
        )
        .with_state_slots(items.iter().map(|&(slot, _, _)| slot).collect())
        .encode()
        .unwrap()
    };
    let prompts: Vec<_> = slots.iter().map(|&slot| (slot, 0, prompt_rows)).collect();
    let prefill = execute(&mut instance, &host, &batch(false, &prompts));
    assert_eq!(
        prefill.views.iter().map(|v| v.raw().row_count).sum::<u32>(),
        DECODE_BATCH
    );
    instance.commit().unwrap();
    // The package admits 32 verify rows total, up to six per item. Fill
    // that budget with maximum-length items, then with the most drafters
    // it admits (16 two-row items). Both include the highest state slot.
    let cases = [
        vec![
            (0, MAX_VERIFY_ROWS),
            (6, MAX_VERIFY_ROWS),
            (12, MAX_VERIFY_ROWS),
            (18, MAX_VERIFY_ROWS),
            (24, MAX_VERIFY_ROWS),
            (31, 2),
        ],
        (0..16).map(|i| (i * 2 + 1, 2)).collect(),
    ];
    for case in cases {
        assert_eq!(
            case.iter().map(|&(_, rows)| rows).sum::<u32>(),
            verify_tokens
        );
        let items: Vec<_> = case
            .iter()
            .map(|&(slot, rows)| (slot, state(&instance, slot + 1).end, rows))
            .collect();
        let verify = execute(&mut instance, &host, &batch(true, &items));
        assert_eq!(
            verify.draft_tokens(),
            items
                .iter()
                .map(|&(slot, _, rows)| (slot + 1, vec![0; (rows - 1) as usize]))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            verify.views.iter().map(|v| v.raw().row_count).sum::<u32>(),
            verify_tokens,
        );
        let accepted: Vec<_> = items
            .iter()
            .map(|&(slot, _, rows)| kept(slot + 1, rows))
            .collect();
        instance.commit_accepted(&accepted).unwrap();
        for (slot, start, rows) in items {
            assert_eq!(state(&instance, slot + 1).end, start + u64::from(rows));
        }
    }
    for slot in slots {
        instance.retire_request(slot + 1).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
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
    graph_lifecycle(None);
}

#[test]
fn graph_pressure_preserves_partial_acceptance_cancel_restore_and_slot_reuse() {
    graph_lifecycle(Some(8 * Ranks::TP4.count()));
}

fn graph_lifecycle(capacity: Option<usize>) {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (host, mut instance) = create(shape);
    crate::instance::dspark_graphs_for_test(instance.handle(), capacity);
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
    if let Some(capacity) = capacity {
        let stats = crate::instance::dspark_graphs_for_test(instance.handle(), None);
        // A full ledger lowers new shapes eagerly and keeps warmed graphs.
        assert!(stats.denied > 0);
        assert_eq!((stats.released, stats.epoch), (0, 0));
        assert!(!stats.pressure);
        assert!(stats.peak <= capacity);
        assert_eq!(stats.reserved, 0);
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
    let mut catch_up = catch_up::captured_catch_up(&host.journal.borrow())
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    for budget in [5, 1, 2, 3, 4] {
        let request = u64::from(budget) + 10;
        instance
            .restore_prefix_state_in_slot(request, &[0, 1, 2], &pages, 0)
            .unwrap();
        let start = host.journal.borrow().len();
        let sink = execute(&mut instance, &host, &verify(request, 0, 3, budget));
        let tokens = &sink.drafts[0].tokens;
        let journal = host.journal.borrow();
        catch_up.extend(catch_up::captured_catch_up(&journal[start..]));
        let draft = journal[start..].iter().take_while(|line| {
            !line.contains(crate::sm120::engram_hash_rows::SYMBOL)
                && !line.starts_with("mapped gather")
        });
        let mut capturing = false;
        let program = draft
            .filter_map(|line| {
                if line.starts_with("capture ") {
                    capturing = true;
                } else if line.starts_with("captured ") {
                    capturing = false;
                } else if !capturing
                    && (line.starts_with("launch ") || line.starts_with("replay "))
                    && !catch_up.contains(line.split_whitespace().nth(2).unwrap())
                {
                    return Some(line.clone());
                }
                None
            })
            .collect::<Vec<_>>();
        assert!(!program.is_empty());
        assert_eq!(
            program
                .iter()
                .filter(|line| line.starts_with("replay "))
                .count(),
            4
        );
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

/// One verify step drafting `drafts` for every (request, slot, start) item.
fn verify_all(items: &[(u64, u64, u64)], drafts: u32) -> Vec<u8> {
    let rows = u64::from(drafts + 1);
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    for &(request, slot, start) in items {
        batch.push(BatchItemV1::verify(request, drafts + 1, true, vec![7]));
        appends.push(BatchAppendV1::new(
            start,
            rows,
            start + rows,
            (0..(start + rows).div_ceil(64))
                .map(|page| slot * 8 + page)
                .collect(),
            None,
        ));
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(items.iter().map(|item| item.1).collect())
        .encode()
        .unwrap()
}

/// Catch-up programs launched since `from`, one per rank each: only
/// catch-up quantizes raw feature rows (three 5120-wide target taps).
fn catch_up_launches(host: &Host, from: usize) -> usize {
    host.journal.borrow()[from..]
        .iter()
        .filter(|line| {
            line.starts_with("launch ")
                && line.split_whitespace().nth(2)
                    == Some("chop_deepseek41_dspark_quantize_w15360_bf16_parallel_v1")
        })
        .count()
}

#[test]
fn ar_steps_defer_catch_up_and_drafting_items_share_one_batched_launch() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (host, mut instance) = create(shape);
    for (request, slot) in [(7, 0), (8, 1)] {
        execute(
            &mut instance,
            &host,
            &super::prompt(request, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    // The three-row prefills are within the batch bound and defer.
    let mut end = 3;
    for _ in 0..3 {
        let from = host.journal.borrow().len();
        execute(
            &mut instance,
            &host,
            &super::decode(&[(7, 0, end, 7), (8, 1, end, 7)]),
        );
        instance.commit().unwrap();
        assert_eq!(catch_up_launches(&host, from), 0);
        end += 1;
        for request in [7, 8] {
            assert_eq!(state(&instance, request).projected(), vec![0; 3]);
        }
    }
    // Two drafting items' deferred rows share one batched program: all
    // three stages share its main projection.
    let from = host.journal.borrow().len();
    execute(
        &mut instance,
        &host,
        &verify_all(&[(7, 0, end), (8, 1, end)], 1),
    );
    assert_eq!(catch_up_launches(&host, from), shape.ranks() as usize);
    for request in [7, 8] {
        assert_eq!(state(&instance, request).projected(), vec![end; 3]);
    }
    instance.commit_accepted(&[kept(7, 2), kept(8, 1)]).unwrap();
    for request in [7, 8] {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// The package's declared drafter chunk reaches the drafter: the host
/// resolves the declaration's default into the create record, and a profile
/// or flag override replaces it.
#[test]
fn the_declared_draft_chunk_reaches_the_drafter() {
    use model_deepseek41_contract::dspark_options::{DEFAULT_DRAFT_CHUNK_ITEMS, DRAFT_CHUNK_ITEMS};
    use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1};

    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let declared = ModelOptionsV1 {
        schema: CAPABILITY.into(),
        options: shape
            .geometry
            .model_options(shape.geometry.runtime_defaults()),
    };
    for (overrides, expected) in [
        (vec![], DEFAULT_DRAFT_CHUNK_ITEMS),
        (vec![(DRAFT_CHUNK_ITEMS.to_owned(), "8".to_owned())], 8),
    ] {
        let resolved = declared.resolve(&overrides.into_iter().collect()).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.drafter.set(true);
        host.speculation.set(Some((5, 32)));
        let record = crate::config::tests::with_options(
            crate::config::tests::record(&shape, host.kv, 4, false),
            &resolved,
        );
        let instance = api()
            .create(
                HostServicesTableV1::new(Rc::new(MappedHost::new(Rc::clone(&host)))),
                &record,
            )
            .unwrap_or_else(|e| panic!("drafter create: {e:?} {:?}", host.logs.borrow()));
        assert_eq!(
            crate::instance::dspark_draft_chunk_items(instance.handle()),
            Some(expected as usize)
        );
    }
}
