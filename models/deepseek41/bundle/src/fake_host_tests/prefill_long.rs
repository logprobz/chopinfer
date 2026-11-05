//! The long band on eager prefill: full-source batches past the short band,
//! up to a 256K extent, take the tile4 scorer at the table's CTA count and
//! the early-stop selector; other rows, longer extents and the opt-outs keep
//! the guarded scorer at 192 CTAs and the live or cached selector.
use super::*;

/// Each chunk's launches, by its first position, for one prompt prefilled
/// to `end`: 4096 tokens, 8192-token chunks to 36864 (one straddles the
/// short band's end at 32768), 4096 tokens, then 8192-token chunks, which
/// end on the 128K and 256K extents.
fn prefill(shape: Shape, kv: KvLayout, end: u64) -> BTreeMap<u64, Vec<String>> {
    let host = FakeHost::new(shape, kv);
    host.kv_pages.set(8192);
    let record = crate::config::tests::record(&shape, kv, 1, false);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    let mut chunks = BTreeMap::new();
    let mut previous = 0;
    while previous < end {
        let count = if matches!(previous, 0 | 36864) {
            4096
        } else {
            8192
        };
        let start = host.device.journal.borrow().len();
        let batch = chunk(previous, count, previous + count >= end);
        if let Err(error) = instance.execute(&batch, &mut Views::default()) {
            panic!(
                "chunk {previous}: {error:?} {:?} {:?}",
                instance.last_execute_refusal(),
                host.logs.borrow()
            );
        }
        instance.commit().unwrap();
        let launches = host.device.journal.borrow()[start..]
            .iter()
            .filter(|line| line.starts_with("launch "))
            .cloned()
            .collect();
        chunks.insert(previous, launches);
        previous += count;
    }
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    chunks
}

/// [`prompt`] of request 7 in slot 0 with token IDs inside the vocabulary.
fn chunk(previous: u64, count: u64, last: bool) -> Vec<u8> {
    packed(&[(7, 0, previous, count, last)])
}

/// One prefill step of `(request, slot, previous, count, last)` items, each
/// slot on its own 4096 pages, with token IDs inside the vocabulary.
fn packed(items: &[(u64, u64, u64, u64, bool)]) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let (batch, appends) = items
        .iter()
        .map(|&(request, slot, previous, count, last)| {
            let end = previous + count;
            let inputs = (previous..end)
                .map(|token| (token % 65536) as u32)
                .collect();
            (
                BatchItemV1::new(request, false, count as u32, last, inputs),
                BatchAppendV1::new(
                    previous,
                    count,
                    end,
                    (0..end.div_ceil(64))
                        .map(|page| slot * 4096 + page)
                        .collect(),
                    None,
                ),
            )
        })
        .unzip();
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(items.iter().map(|item| item.1).collect())
        .encode()
        .unwrap()
}

/// Checks every full-source batch's scorer, local selector and merge: the
/// band's tile4 grid and early-stop entries where `band` holds and the table
/// admits the batch's rows, the guarded scorer at 192 CTAs and the live or
/// cached selector otherwise. Returns the batches the band took and the
/// batches checked.
fn full_source_batches(lines: &[String], band: bool) -> (usize, usize) {
    let (mut taken, mut checked) = (0, 0);
    for (at, line) in lines.iter().enumerate() {
        if !(line.contains("chop_position_indexer_")
            && line.contains("_scores_v3_live_v1_mma_fp4_")
            && !line.contains("_pack_fp4_"))
        {
            continue;
        }
        let n = super::shared_overlap::journal_spans(line)[0].2 / 8192;
        let (local, merge) = (&lines[at + 1], &lines[at + 2]);
        assert!(local.contains("_all_select_v3_"), "{local}");
        assert!(merge.contains("_all_select_v3_live_merge"), "{merge}");
        checked += 1;
        let admitted = band
            .then(|| crate::Kernels::prefill_long_selected(!line.contains("_r2_"), n as u32))
            .flatten();
        if let Some(ctas) = admitted {
            taken += 1;
            assert!(line.contains("_mma_fp4_tile4_mode0_trace0_v1 "), "{line}");
            assert!(line.contains(&format!("grid_x: {ctas}, grid_y: {n}, grid_z: 1")));
            assert!(local.contains("_select_v3_live_early_v1 "), "{local}");
            assert!(merge.contains("_select_v3_live_merge_early_v1 "), "{merge}");
        } else {
            assert!(line.contains("_mma_fp4_guard_fast_v1 "), "{line}");
            assert!(line.contains(&format!("grid_x: 192, grid_y: {n}, grid_z: 1")));
            assert!(
                local.contains("_select_v3_live_v1 ") || local.contains("_select_v3_cached4_v1 "),
                "{local}"
            );
            assert!(merge.contains("_select_v3_live_merge_v1 "), "{merge}");
        }
    }
    (taken, checked)
}

/// Chunks whose last position reaches 32768 and whose page-rounded extent
/// is at most 262144 take the band for every full-source batch the table
/// admits, including a chunk that straddles the short band's end and the
/// 128K extent's batches, which would otherwise select with the cached
/// selector; the chunk past 256K keeps the guarded route throughout.
#[test]
fn long_band_takes_every_admitted_full_source_batch() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let end = 262144 + 8192;
    let mut previous = 0;
    for (first, lines) in prefill(shape, KV_LAYOUTS[0], end) {
        let last = first
            + if matches!(first, 0 | 36864) {
                4096
            } else {
                8192
            }
            - 1;
        assert_eq!(first, previous);
        previous = last + 1;
        if last < 32768 {
            // The short band (`position_valid`).
            continue;
        }
        let band = (last + 1).next_multiple_of(64) <= 262144;
        let (taken, checked) = full_source_batches(&lines, band);
        assert!(checked > 0, "chunk {first}");
        if [28672, 98304, 122880, 196608, 253952].contains(&first) {
            // Every batch of these chunks has at least 32 rows.
            assert_eq!(taken, checked, "chunk {first}");
        }
        if !band {
            assert_eq!(taken, 0);
        }
    }
    assert_eq!(previous, end);
}

/// `text_config.indexer_prefill_tile=false` and BF16 keys take no long band.
#[test]
fn opted_out_prefill_takes_no_long_band() {
    let mut untiled = Shape::new(&FLASH, Ranks::TP4).unwrap();
    untiled.runtime.indexer_prefill_tile = false;
    let native = Shape::new(&FLASH, Ranks::TP4).unwrap();
    for (shape, kv) in [(untiled, KV_LAYOUTS[0]), (native, KV_LAYOUTS[1])] {
        for (first, lines) in prefill(shape, kv, 98304 + 8192) {
            if first + 8192 <= 32768 {
                continue;
            }
            assert!(
                lines
                    .iter()
                    .all(|line| !line.contains("_mma_fp4_tile4_mode0_trace0_v1 ")
                        && !line.contains("_early_v1 ")),
                "{:?} chunk {first}",
                kv.storage
            );
            if kv.storage == KvStorage::Native {
                assert!(full_source_batches(&lines, false).1 > 0);
            }
        }
    }
}

/// A packed phase whose largest position is long takes the band for all of
/// its batches, including those of a new prompt's short rows; a later
/// 16-row chunk, below the table's rows, keeps the guarded route.
#[test]
fn long_band_takes_a_packed_phase_whole_and_leaves_small_batches() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.kv_pages.set(8192);
    let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 2, false);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    let mut previous = 0;
    while previous < 36864 {
        let count = if previous == 0 { 4096 } else { 8192 };
        step(
            &mut instance,
            &chunk(previous, count, false),
            &mut Views::default(),
        );
        previous += count;
    }
    let mut steps = Vec::new();
    for batch in [
        packed(&[(7, 0, 36864, 4096, false), (8, 1, 0, 4096, true)]),
        chunk(40960, 16, true),
    ] {
        let start = host.device.journal.borrow().len();
        step(&mut instance, &batch, &mut Views::default());
        steps.push(
            host.device.journal.borrow()[start..]
                .iter()
                .filter(|line| line.starts_with("launch "))
                .cloned()
                .collect::<Vec<_>>(),
        );
    }
    for request in [7, 8] {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let (taken, checked) = full_source_batches(&steps[0], true);
    assert!(taken > 0);
    assert_eq!(taken, checked);
    let (taken, checked) = full_source_batches(&steps[1], true);
    assert_eq!(taken, 0);
    assert!(checked > 0);
}
