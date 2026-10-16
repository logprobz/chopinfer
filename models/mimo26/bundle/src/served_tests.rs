//! Instances serving more than 32 sequences (CONC-MIMO): what their buffers
//! hold and which kernels their decode, verify and prefill phases run.

use super::*;
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

/// An instance at TP4 serving `served` sequences, with `speculation` when
/// given.
fn serving(
    served: u32,
    speculation: Option<(u32, u32)>,
) -> (Rc<FakeHost>, runtime_bundle_abi::BundleInstanceV1) {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::default());
    backend.served.set(served);
    backend.speculation.set(speculation);
    let record = configuration_serving(Tp::TP4, &KV_FORMATS[0], false, served);
    let instance = api
        .create(HostServicesTableV1::new(Rc::clone(&backend)), &record)
        .expect("fake-host create");
    (backend, instance)
}

/// One step of `items` requests from 100: prefill three tokens each, or
/// decode one token after `start` committed ones.
fn step(items: u64, decode: bool, start: u64) -> Vec<u8> {
    let tokens = if decode { 1 } else { 3 };
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..items)
            .map(|item| {
                BatchItemV1::new(100 + item, decode, tokens, true, vec![7; tokens as usize])
            })
            .collect(),
        (0..items)
            .map(|item| {
                BatchAppendV1::new(
                    start,
                    u64::from(tokens),
                    start + u64::from(tokens),
                    // The fake host's pool holds 128 pages.
                    vec![item % 128],
                    None,
                )
            })
            .collect(),
    )
    .encode()
    .unwrap()
}

/// Runs `request` and returns the symbols it launched and its logits rows.
fn run(
    backend: &FakeHost,
    instance: &mut runtime_bundle_abi::BundleInstanceV1,
    request: &[u8],
) -> (Vec<String>, u32) {
    let before = backend.launch_symbols.borrow().len();
    let mut sink = ViewSink::default();
    instance.execute(request, &mut sink).expect("step");
    let rows = sink.0.iter().map(|view| view.raw().row_count).sum();
    (backend.launch_symbols.borrow()[before..].to_vec(), rows)
}

fn count(launched: &[String], name: &str) -> usize {
    launched
        .iter()
        .filter(|symbol| symbol.contains(name))
        .count()
}

#[test]
fn served_counts_size_the_ring_slot_tables_and_item_buffers() {
    for served in [8, 33, 64, 100, 256] {
        let (backend, mut instance) = serving(served, None);
        let items = bundle_runtime::geometry::Items::new(served).unwrap();
        // Instances with buckets above 32 rows log their all-reduce and
        // dense GEMM settings for runners.
        let wide = u8::from(std::env::var(crate::switches::WIDE_COMM).as_deref() != Ok("0"));
        let said = backend.logs.borrow().iter().any(|line| {
            line.contains(&format!(
                "wide all-reduce {} (CHOPINFER_MIMO_WIDE_COMM={wide})",
                if wide == 1 { "on" } else { "off" }
            ))
        });
        assert_eq!(said, items.capacity() > 32, "{served}");
        // `allocate_state` checked the ring state against the served count's
        // largest bucket; the block tables hold one row per sequence.
        let allocations = backend.allocations.borrow().clone();
        assert!(
            allocations.contains(&crate::upload::BlockTableMirror::device_bytes(
                crate::contract::MODEL_LIMITS,
                items
            )),
            "{served}"
        );
        assert!(allocations.contains(&crate::upload::InputLayout::decode_within(items).end));
        // The ring entries of the slot count are loaded, and only those
        // other than 32 come from the slots cubin.
        let names = backend.symbol_names.borrow();
        let ring = crate::sm120::ring_entries::ring_backup_symbol(Tp::TP4, items.capacity());
        assert!(names.values().any(|name| *name == ring), "{ring}");
        assert_eq!(
            names.values().any(|name| name.ends_with("_s256_v1")),
            items.capacity() != 32
        );
        drop(names);
        instance.destroy().expect("destroy");
    }
}

#[test]
fn decode_steps_above_32_rows_run_the_prefill_entries_in_their_bucket_graphs() {
    let (backend, mut instance) = serving(256, None);
    let (_, rows) = run(&backend, &mut instance, &step(256, false, 0));
    assert_eq!(rows, 256);
    instance.commit().expect("commit prefill");
    let ranks = RANKS as usize;
    let moe_layers = crate::program::LAYERS as usize - 1;
    let start = 3;
    // The first step of each bucket captures its graph (the launches below);
    // a later step of the same bucket replays it.
    for (live, bucket, captures) in [
        (33, 64, true),
        (64, 64, false),
        (65, 128, true),
        (128, 128, false),
        (129, 256, true),
        (256, 256, false),
    ] {
        let begins = backend.graph_begins.get();
        let (launched, rows) = run(&backend, &mut instance, &step(live, true, start));
        assert_eq!(rows as u64, live, "{live} decode rows emit logits");
        instance.cancel().expect("cancel decode");
        let captured = backend.graph_begins.get() - begins;
        if !captures {
            assert_eq!(captured, 0, "{live} replays its bucket's graph");
            continue;
        }
        assert!(captured > 0 && captured % RANKS == 0, "{live}: {captured}");
        // Token-indexed stages run the prefill kernels at the bucket's rows,
        // or, by default (CHOPINFER_MIMO_VERIFY_DENSE_MID), the medium-M entries
        // that stand in for the dense ones at up to 64 rows.
        let medium = crate::switches::verify_dense_mid_enabled() && bucket <= 64;
        assert_eq!(
            count(&launched, "chop_gemm_mimo_qkv_swa_prefill_v1"),
            if medium {
                0
            } else {
                SWA_LAYERS as usize * ranks
            },
            "{live}"
        );
        assert_eq!(count(&launched, "chop_gemm_mid_") > 0, medium, "{live}");
        assert_eq!(count(&launched, "_m32_v1"), 0);
        assert_eq!(count(&launched, "permute_tile_scatter"), moe_layers * ranks);
        assert_eq!(
            count(
                &launched,
                model_mimo26_contract::sm120::moe::DECODE_ROUTE_SYMBOL
            ),
            0
        );
        assert_eq!(
            count(&launched, "chop_row_combine_mimo4096_v1"),
            moe_layers * ranks
        );
        assert_eq!(count(&launched, "_verify_mxfp4_a16_t"), 0);
        assert_eq!(
            count(&launched, "_prefill_mxfp4_a16_t8192_v1"),
            2 * moe_layers * ranks
        );
        // The hidden all-reduces take CONC-COMM's wide entry unless
        // CHOPINFER_MIMO_WIDE_COMM=0 keeps the phased stages.
        let wide = std::env::var(crate::switches::WIDE_COMM).as_deref() != Ok("0");
        assert_eq!(count(&launched, crate::sm120::comm::WIDE_SYMBOL) > 0, wide);
        assert_eq!(
            count(&launched, crate::sm120::launch::comm::PHASED_SYMBOLS[0]) > 0,
            !wide
        );
        // Decode attention still folds RoPE and the cache write in, so the
        // row engine's RoPE and ring write launch nothing.
        assert!(count(&launched, "chop_attn_mimo_ga_bf16_decode_rope_table_v1") > 0);
        assert!(count(&launched, "chop_attn_mimo_swa_decode_rope_table_v1") > 0);
        assert_eq!(count(&launched, "chop_row_rope_mimo"), 0);
        assert_eq!(count(&launched, "chop_row_ring_write_mimo"), 0);
        // The ring backup runs the 256-slot entry.
        assert_eq!(count(&launched, "chop_mimo_ring_backup_s256_v1"), ranks);
        assert_eq!(count(&launched, "chop_mimo_ring_backup_v1"), 0);
    }
    // A bucket's second step replays its graph.
    let begins = backend.graph_begins.get();
    run(&backend, &mut instance, &step(100, true, start));
    assert_eq!(backend.graph_begins.get(), begins);
    instance.cancel().expect("cancel decode");
    instance.destroy().expect("destroy");
}

#[test]
fn steps_above_the_served_count_are_refused() {
    let (backend, mut instance) = serving(64, None);
    run(&backend, &mut instance, &step(64, false, 0));
    instance.commit().expect("commit prefill");
    let mut sink = ViewSink::default();
    assert!(instance.execute(&step(65, true, 3), &mut sink).is_err());
    // 64 decode rows still run.
    let (_, rows) = run(&backend, &mut instance, &step(64, true, 3));
    assert_eq!(rows, 64);
    instance.cancel().expect("cancel");
    instance.destroy().expect("destroy");
}

/// DFlash, MiMo's default speculation, serves the topology's limit: an
/// instance serving 33, 100 or 256 sequences sizes its cache planes by the
/// ring's slots, catches up the verifying items' committed rows in their own
/// (high) slots and drafts for every item of a verify step.
#[test]
fn dflash_instances_serve_and_draft_above_32_sequences() {
    use crate::dflash::state::HISTORY_CAPACITY;
    for served in [33_u32, 100, 256] {
        let (backend, mut instance) = {
            let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
                .expect("bundle API");
            let backend = Rc::new(FakeHost::default());
            backend.served.set(served);
            backend.speculation.set(Some((7, 256)));
            backend.dflash.set(true);
            let record = configuration_serving(Tp::TP4, &KV_FORMATS[0], true, served);
            let instance = api
                .create(HostServicesTableV1::new(Rc::clone(&backend)), &record)
                .unwrap_or_else(|error| panic!("DFlash at {served}: {error:?}"));
            (backend, instance)
        };
        let slots = served.next_power_of_two();
        assert!(
            backend
                .allocations
                .borrow()
                .contains(&crate::dflash::program::cache_bytes(Tp::TP4, slots)),
            "{served}"
        );
        // Every sequence prefills three tokens into slots 0 to served - 1.
        let mut sink = DraftSink::default();
        instance
            .execute(&step(u64::from(served), false, 0), &mut sink)
            .expect("prefill");
        instance.commit().expect("commit prefill");
        // The last `verifying` sequences, in the highest slots, verify four
        // rows each: one bonus token and three drafts.
        let verifying = served.min(64);
        let first = served - verifying;
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            (first..served)
                .map(|item| BatchItemV1::verify(100 + u64::from(item), 4, true, vec![43]))
                .collect(),
            (first..served)
                .map(|item| BatchAppendV1::new(3, 4, 7, vec![u64::from(item) % 128], None))
                .collect(),
        )
        .encode()
        .unwrap();
        *backend.readback_tokens.borrow_mut() = (0..4 * verifying).map(|row| 1000 + row).collect();
        let mut sink = DraftSink::default();
        instance.execute(&request, &mut sink).expect("verify");
        // One catch-up gathers the verifying items' three committed rows
        // from their own slots on each rank.
        let window = (first..served)
            .flat_map(|slot| {
                (0..3).map(move |position| crate::retain::cell(slot, position, HISTORY_CAPACITY))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            backend.histories_gathering(&window).len(),
            RANKS as usize,
            "{served}"
        );
        assert_eq!(sink.drafts.len(), verifying as usize, "{served}");
        for (index, drafts) in sink.drafts.iter().enumerate() {
            assert_eq!(drafts.request_id, 100 + u64::from(first) + index as u64);
            assert_eq!(drafts.tokens.len(), 3);
        }
        assert_eq!(
            sink.views
                .iter()
                .map(|view| view.raw().row_count)
                .sum::<u32>(),
            4 * verifying
        );
        let keep = (first..served)
            .map(|item| kept(100 + u64::from(item), 1 + item % 4))
            .collect::<Vec<_>>();
        instance.commit_accepted(&keep).expect("keep rows");
        instance.destroy().expect("destroy");
    }
}

#[test]
fn mtp_instances_serve_at_most_32_sequences() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    for (served, accepted) in [(32, true), (33, false), (64, false)] {
        let backend = Rc::new(FakeHost::default());
        backend.served.set(served);
        backend.speculation.set(Some((7, 32)));
        let record = configuration_serving(Tp::TP4, &KV_FORMATS[0], true, served);
        let created = api.create(HostServicesTableV1::new(Rc::clone(&backend)), &record);
        assert_eq!(created.is_ok(), accepted, "{served}");
        if let Ok(mut instance) = created {
            instance.destroy().expect("destroy");
        }
    }
}

#[test]
fn verify_steps_carry_more_than_32_items_with_widened_attention() {
    let (backend, mut instance) = serving(128, Some((7, 256)));
    run(&backend, &mut instance, &step(100, false, 0));
    instance.commit().expect("commit prefill");
    // 100 items verifying two rows each: 200 rows in the 256 bucket over the
    // 128-item attention tables.
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..100)
            .map(|item| BatchItemV1::verify(100 + item, 2, false, vec![8, 9]))
            .collect(),
        (0..100)
            .map(|item| BatchAppendV1::new(3, 2, 5, vec![item % 128], None))
            .collect(),
    )
    .encode()
    .unwrap();
    let (launched, rows) = run(&backend, &mut instance, &request);
    assert_eq!(rows, 200);
    let ranks = RANKS as usize;
    assert!(count(&launched, "chop_attn_mimo_ga_bf16_verify_partial_v1") > 0);
    assert_eq!(count(&launched, "chop_mimo_ring_backup_s128_v1"), ranks);
    let keep = (0..100).map(|item| kept(100 + item, 1)).collect::<Vec<_>>();
    instance.commit_accepted(&keep).expect("keep one row each");
    instance.destroy().expect("destroy");
}

/// Below 32 served sequences the catch-up graph's item bucket holds at most
/// eight rows per served item. A longer catch-up, such as a prompt's first
/// projection, runs eagerly instead of asking for a graph the cache refuses.
#[test]
fn dflash_catch_ups_beyond_the_served_graph_run_eagerly() {
    for (served, prompt) in [
        (8_u32, 64_usize),
        (8, 70),
        (16, 128),
        (16, 150),
        (16, 250),
        (32, 250),
    ] {
        let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
            .expect("bundle API");
        let backend = Rc::new(FakeHost::default());
        backend.served.set(served);
        backend.speculation.set(Some((3, 256)));
        backend.dflash.set(true);
        let record = configuration_serving(Tp::TP4, &KV_FORMATS[0], true, served);
        let mut instance = api
            .create(HostServicesTableV1::new(Rc::clone(&backend)), &record)
            .expect("fake-host create");
        instance
            .execute(
                &dflash_step(0, vec![40; prompt], None),
                &mut DraftSink::default(),
            )
            .expect("prefill");
        instance.commit().expect("commit prefill");
        *backend.readback_tokens.borrow_mut() = vec![40; 4];
        let mut sink = DraftSink::default();
        instance
            .execute(&dflash_step(prompt as u64, vec![40], Some(4)), &mut sink)
            .unwrap_or_else(|error| {
                panic!(
                    "{served} served, {prompt} rows: {error:?} {:?}",
                    backend.logs.borrow()
                )
            });
        assert_eq!(sink.drafts.len(), 1, "{served} served, {prompt} rows");
        instance.cancel().expect("cancel");
        instance.destroy().expect("destroy");
    }
}
