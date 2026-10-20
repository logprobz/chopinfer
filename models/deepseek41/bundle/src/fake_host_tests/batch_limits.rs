//! Full-budget steps on the real launch planners. The fake host validates
//! spans and lifecycle calls; it does not compute device outputs.
use super::*;
use model_deepseek41_contract::serving::{DECODE_BATCH, PAGE_TOKENS, PREFILL_TOKENS};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

struct Item {
    request: u64,
    slot: u64,
    start: u64,
    rows: u32,
    decode: bool,
    pages: Vec<u64>,
}

fn batch(items: &[Item]) -> Vec<u8> {
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        items
            .iter()
            .map(|item| {
                BatchItemV1::new(
                    item.request,
                    item.decode,
                    item.rows,
                    item.decode || !u64::from(item.rows).is_multiple_of(PAGE_TOKENS),
                    (item.start..item.start + u64::from(item.rows))
                        .map(|token| token as u32)
                        .collect(),
                )
            })
            .collect(),
        items
            .iter()
            .map(|item| {
                let end = item.start + u64::from(item.rows);
                BatchAppendV1::new(
                    item.start,
                    u64::from(item.rows),
                    end,
                    item.pages[..end.div_ceil(PAGE_TOKENS) as usize].to_vec(),
                    None,
                )
            })
            .collect(),
    )
    .with_state_slots(items.iter().map(|item| item.slot).collect())
    .encode()
    .unwrap()
}

fn full_budget(starts: &[u64], counts: &[u32], decode_last: bool) {
    assert_eq!(starts.len(), counts.len());
    assert_eq!(counts.iter().sum::<u32>(), PREFILL_TOKENS);
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let record = crate::config::tests::record(&shape, host.kv, DECODE_BATCH, false);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap_or_else(|e| panic!("create: {e:?}: {:?}", host.logs.borrow()));
    let mut next_page = 0;
    let items: Vec<_> = starts
        .iter()
        .zip(counts)
        .enumerate()
        .map(|(slot, (&start, &rows))| {
            let end_page = next_page + (start + u64::from(rows)).div_ceil(PAGE_TOKENS);
            let item = Item {
                request: slot as u64 + 100,
                slot: slot as u64,
                start,
                rows,
                decode: decode_last && slot + 1 == starts.len(),
                pages: (next_page..end_page).collect(),
            };
            next_page = end_page;
            item
        })
        .collect();
    // Disjoint KV tables, including the snapshot pages. Reuse a snapshot
    // only after its restore has completed.
    let snapshots = instance.describe_prefix_state().unwrap();
    assert!(next_page + snapshots <= KV_PAGES);
    let snapshot_pages: Vec<_> = (next_page..next_page + snapshots).collect();
    for item in items.iter().filter(|item| item.start > 0) {
        let seed = Item {
            request: item.request + 100,
            slot: item.slot,
            start: 0,
            rows: item.start as u32,
            decode: false,
            pages: item.pages.clone(),
        };
        step(&mut instance, &batch(&[seed]), &mut Views::default());
        instance
            .capture_prefix_state(item.request + 100, item.start, &snapshot_pages)
            .unwrap();
        instance.retire_request(item.request + 100).unwrap();
        let tokens: Vec<_> = (0..item.start as u32).collect();
        instance
            .restore_prefix_state_in_slot(item.request, &tokens, &snapshot_pages, item.slot)
            .unwrap();
    }
    let begin = host.device.journal.borrow().len();
    let mut views = Views::default();
    instance
        .execute(&batch(&items), &mut views)
        .unwrap_or_else(|e| panic!("full-budget execute: {e:?}: {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    assert_eq!(views.rows(), if decode_last { 2 } else { 0 });
    let journal = host.device.journal.borrow();
    let launches = |family: &str| {
        journal[begin..]
            .iter()
            .filter(|line| line.starts_with("launch ") && line.contains(family))
            .count()
    };
    // Every layer and rank reaches its source and merge. Mixed batches
    // also execute one-row decode, with its selected-key cache producer.
    assert_eq!(
        launches("chop_latent_sources_"),
        FLASH.layers() as usize * 4 * if decode_last { 5 } else { 2 },
    );
    assert_eq!(
        launches("_cache_decode_control0_"),
        if decode_last {
            FLASH.layers() as usize * 4
        } else {
            0
        },
    );
    for family in [
        "chop_position_indexer_",
        "chop_position_blocks_",
        "compressor",
        "moe",
    ] {
        assert!(launches(family) > 0, "no {family} launches");
    }
    drop(journal);
    for item in items {
        instance.retire_request(item.request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn full_budget_four_prefills() {
    full_budget(&[0; 4], &[2048; 4], false);
}

#[test]
fn full_budget_four_prefills_after_prefix_hits() {
    // Before, at, and beyond the 128-token window, on 64-token pages.
    full_budget(&[64, 128, 192, 256], &[2048; 4], false);
}

#[test]
fn full_budget_every_item() {
    full_budget(&[0; 32], &[256; 32], false);
}

#[test]
fn full_budget_prefill_and_decode() {
    full_budget(&[0, 0, 0, 0, 128], &[2048, 2048, 2048, 2047, 1], true);
}
