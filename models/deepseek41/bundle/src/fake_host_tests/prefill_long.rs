//! The `indexer_prefill_long` candidate on eager prefill: full-source batches
//! past the short band, up to a 256K extent, take the tile4 scorer at the
//! table's CTA count and the table's selector. Every other launch, and every
//! launch outside the band or without the option, is unchanged.
use super::*;
use model_deepseek41_contract::execution_options::INDEXER_PREFILL_LONG;

/// Each chunk's launches, by its first position, for one prompt prefilled
/// to `end`: 4096 tokens, 8192-token chunks to 36864 (one straddles the
/// short band's end at 32768), 4096 tokens, then 8192-token chunks, which
/// end on the 128K and 256K extents.
fn prefill(
    shape: Shape,
    kv: KvLayout,
    enabled: Option<bool>,
    end: u64,
) -> BTreeMap<u64, Vec<String>> {
    let host = FakeHost::new(shape, kv);
    host.kv_pages.set(8192);
    let mut record = crate::config::tests::record(&shape, kv, 1, false);
    if let Some(enabled) = enabled {
        record = crate::config::tests::with_option(record, INDEXER_PREFILL_LONG, enabled.into());
    }
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

/// `off` with each full-source guarded scorer retargeted to its band's tile4
/// grid, and its local selector and merge to the early-stop entries; the
/// number of batches the band took.
fn expected(off: &[String]) -> (Vec<String>, usize) {
    let mut lines = off.to_vec();
    let mut taken = 0;
    for at in 0..lines.len() {
        let line = &off[at];
        if !(line.contains("chop_position_indexer_")
            && line.contains("_scores_v3_live_v1_mma_fp4_guard_fast_v1 "))
        {
            continue;
        }
        let n = super::shared_overlap::journal_spans(line)[0].2 / 8192;
        let Some(ctas) = crate::Kernels::prefill_long_selected(!line.contains("_r2_"), n as u32)
        else {
            continue;
        };
        taken += 1;
        lines[at] = line
            .replacen(
                "_mma_fp4_guard_fast_v1 ",
                "_mma_fp4_tile4_mode0_trace0_v1 ",
                1,
            )
            .replacen("grid_x: 192, ", &format!("grid_x: {ctas}, "), 1);
        // Today's route may already select with the 128K cached selector.
        let local = off[at + 1].replacen("_select_v3_cached4_v1 ", "_select_v3_live_v1 ", 1);
        assert!(local.contains("_all_select_v3_live_v1 "), "{local}");
        assert!(off[at + 2].contains("_all_select_v3_live_merge_v1 "));
        lines[at + 1] = local.replacen("_select_v3_live_v1 ", "_select_v3_live_early_v1 ", 1);
        lines[at + 2] = off[at + 2].replacen(
            "_select_v3_live_merge_v1 ",
            "_select_v3_live_merge_early_v1 ",
            1,
        );
    }
    (lines, taken)
}

/// With the option, chunks whose last position reaches 32768 and whose
/// page-rounded extent is at most 262144 change exactly the full-source
/// batches the table admits, including a chunk that straddles the short
/// band's end and the 128K extent's batches, which select with the cached
/// selector today. Earlier and later chunks keep today's launches, as does
/// an explicit false.
#[test]
fn long_band_retargets_only_admitted_full_source_batches() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let end = 262144 + 8192;
    let off = prefill(shape, KV_LAYOUTS[0], None, end);
    assert_eq!(prefill(shape, KV_LAYOUTS[0], Some(false), end), off);
    let on = prefill(shape, KV_LAYOUTS[0], Some(true), end);
    assert_eq!(
        on.keys().collect::<Vec<_>>(),
        off.keys().collect::<Vec<_>>()
    );
    let mut previous = 0;
    for (&first, off) in &off {
        let last = first
            + if matches!(first, 0 | 36864) {
                4096
            } else {
                8192
            }
            - 1;
        let band = last >= 32768 && (last + 1).next_multiple_of(64) <= 262144;
        let (lines, taken) = if band {
            expected(off)
        } else {
            (off.clone(), 0)
        };
        if [28672, 98304, 122880, 196608, 253952].contains(&first) {
            assert!(taken > 0, "chunk {first}");
        }
        assert_eq!(on[&first].len(), lines.len(), "chunk {first}");
        for (on, expected) in on[&first].iter().zip(&lines) {
            assert_eq!(on, expected, "chunk {first}");
        }
        assert_eq!(first, previous);
        previous = last + 1;
    }
    assert_eq!(previous, end);
}

/// The tiled-route opt-out and BF16 keys keep today's launches with the
/// option.
#[test]
fn long_band_keeps_opted_out_prefill_on_todays_launches() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let mut untiled = shape;
    untiled.runtime.indexer_prefill_tile = false;
    for (shape, kv) in [(untiled, KV_LAYOUTS[0]), (shape, KV_LAYOUTS[1])] {
        let end = 98304 + 8192;
        assert_eq!(
            prefill(shape, kv, Some(true), end),
            prefill(shape, kv, None, end),
            "{:?}",
            kv.storage
        );
    }
}

/// A packed phase whose largest position is long takes the band for all of
/// its batches, including those of a new prompt's short rows; a later
/// 16-row chunk, below the table's rows, keeps today's launches.
#[test]
fn long_band_takes_a_packed_phase_whole_and_leaves_small_batches() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let run = |enabled: Option<bool>| {
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.kv_pages.set(8192);
        let mut record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 2, false);
        if let Some(enabled) = enabled {
            record =
                crate::config::tests::with_option(record, INDEXER_PREFILL_LONG, enabled.into());
        }
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap();
        let mut steps = Vec::new();
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
        steps
    };
    let (off, on) = (run(None), run(Some(true)));
    let full = |lines: &[String], symbol: &str| {
        lines
            .iter()
            .filter(|line| line.contains("chop_position_indexer_") && line.contains(symbol))
            .count()
    };
    let (lines, taken) = expected(&off[0]);
    assert_eq!(on[0], lines);
    assert!(taken > 0);
    assert_eq!(taken, full(&off[0], "_mma_fp4_guard_fast_v1 "));
    assert_eq!(full(&on[0], "_mma_fp4_guard_fast_v1 "), 0);
    assert_eq!(expected(&off[1]).1, 0);
    assert_eq!(on[1], off[1]);
    assert!(full(&on[1], "_mma_fp4_guard_fast_v1 ") > 0);
}
