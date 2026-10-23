//! Drafted DSpark steps gather each item's first Engram rows before the
//! drafter unless `CHOPINFER_DSPARK_ENGRAM_EARLY=0`. Every stream launches the
//! same kernels in the same order, the late gathers keep the device hash,
//! the early gathers' ids are the host's hashes of each item's first row,
//! every rank joins both gathers of a layer before its consumer, both
//! complete at the boundary, and every resource is released.
use super::*;
use crate::dspark::cycle::with_early;
use crate::engram_rows::{EngramRows, Step};
use crate::fake_host_tests::ar_engram::order;
use crate::fake_host_tests::query_window::canonical;
use model_deepseek41_contract::own_ops::engram::HashTables;

/// The bonus token every drafted step verifies, and keeps.
const BONUS: u32 = 7;

/// The ids the early gathers upload for one drafted step of every slot:
/// each item's first row, oracle-hashed from its committed tokens, laid out
/// as each layer's `stride` rows in turn.
fn early_ids(histories: &[Vec<u32>], stride: u32, bytes: u64) -> Vec<u8> {
    let mut rows = EngramRows::new(HashTables::of(&FLASH).unwrap());
    for (slot, history) in histories.iter().enumerate() {
        rows.restore(7 + slot as u64, history, None).unwrap();
    }
    let steps: Vec<Step<'_>> = (0..histories.len())
        .map(|slot| Step {
            request: 7 + slot as u64,
            tokens: &[BONUS],
            text: None,
        })
        .collect();
    let hashed = rows.hash(&steps).unwrap();
    let layers = FLASH.shape.engram.layers.len();
    let columns = hashed.ids().len() / hashed.rows() / layers;
    let mut wanted = vec![0_u8; bytes as usize];
    for (row, row_ids) in hashed.ids().chunks_exact(layers * columns).enumerate() {
        for layer in 0..layers {
            for (column, id) in row_ids[layer * columns..(layer + 1) * columns]
                .iter()
                .enumerate()
            {
                let at = ((layer * stride as usize + row) * columns + column) * 4;
                wanted[at..at + 4].copy_from_slice(&id.to_le_bytes());
            }
        }
    }
    wanted
}

struct Run {
    host: Rc<Host>,
    from: usize,
    /// Each drafted step's execute journal lines.
    journals: Vec<Vec<String>>,
}

/// Four three-token prompts, then `steps` drafted steps of every slot, each
/// keeping its bonus row.
fn schedule(ranks: Ranks, steps: usize, on: bool) -> Run {
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let (host, mut instance) = with_early(on, || create(shape));
    let early = crate::instance::dspark_early_ids(instance.handle());
    assert_eq!(early.is_some(), on);
    let mut histories = Vec::new();
    for slot in 0..4 {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(7 + slot, slot, 0, 3, true),
        );
        instance.commit().unwrap();
        histories.push((0..3).collect::<Vec<u32>>());
    }
    let from = host.journal.borrow().len();
    let mut journals = Vec::new();
    for _ in 0..steps {
        let items: Vec<(u64, u64, u64)> = (0..4)
            .map(|slot| (7 + slot, slot, histories[slot as usize].len() as u64))
            .collect();
        let begin = host.journal.borrow().len();
        let sink = execute(&mut instance, &host, &verify_all(&items, 1));
        journals.push(host.journal.borrow()[begin..].to_vec());
        assert_eq!(sink.draft_tokens().len(), 4);
        if let Some((span, stride)) = early {
            assert_eq!(
                host.uploads.borrow()[&(span.buffer.raw, span.offset)],
                early_ids(&histories, stride, span.length)
            );
        }
        let kept: Vec<_> = (7..11).map(|request| kept(request, 1)).collect();
        instance.commit_accepted(&kept).unwrap();
        for history in &mut histories {
            history.push(BONUS);
        }
    }
    for request in 7..11 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    Run {
        host,
        from,
        journals,
    }
}

/// The step's lines of one kind.
fn count(lines: &[String], prefix: &str) -> usize {
    lines.iter().filter(|line| line.starts_with(prefix)).count()
}

/// The step's order without id uploads, as runs of one kind, with the joins
/// and completions of each run divided by `gathers`, the gathers per layer.
fn runs(lines: &[String], gathers: usize) -> Vec<(&'static str, usize)> {
    let mut result: Vec<(&'static str, usize)> = Vec::new();
    for kind in order(lines).into_iter().filter(|&kind| kind != "ids") {
        match result.last_mut() {
            Some((last, n)) if *last == kind => *n += 1,
            _ => result.push((kind, 1)),
        }
    }
    for (kind, n) in &mut result {
        if matches!(*kind, "wait" | "complete") {
            assert_eq!(*n % gathers, 0, "{kind} x{n}");
            *n /= gathers;
        }
    }
    result
}

#[test]
fn early_first_rows_keep_launches_and_join_before_each_consumer() {
    for ranks in [Ranks::TP4, Ranks::TP2, Ranks::TP1] {
        let off = schedule(ranks, 3, false);
        let on = schedule(ranks, 3, true);
        // The early scratch and side stream shift raw handle numbers, so
        // launches compare with handles numbered by first use.
        let launches = |run: &Run| {
            let lines = run.host.device.journal.borrow()[run.from..]
                .iter()
                .filter(|line| line.starts_with("launch "))
                .cloned()
                .collect::<Vec<_>>();
            canonical(&[lines], |_| false)
        };
        assert_eq!(launches(&off), launches(&on), "ranks={}", ranks.get());
        let count_ranks = ranks.count();
        for (off, on) in off.journals.iter().zip(&on.journals) {
            // Two more id uploads, a wait per rank and layer more, two more
            // completions; nothing else changes order.
            assert_eq!(count(off, "async mapped ids "), 2);
            assert_eq!(count(on, "async mapped ids "), 4);
            assert_eq!(count(off, "async mapped wait "), 2 * count_ranks);
            assert_eq!(count(on, "async mapped wait "), 4 * count_ranks);
            assert_eq!(count(off, "async mapped complete"), 2);
            assert_eq!(count(on, "async mapped complete"), 4);
            // The early ids go up before the device hash runs, the late
            // ones after it.
            let hash = |lines: &[String]| {
                lines
                    .iter()
                    .position(|line| line.contains(crate::sm120::engram_hash_rows::SYMBOL))
                    .unwrap()
            };
            assert_eq!(count(&on[..hash(on)], "async mapped ids "), 2);
            assert_eq!(count(&off[..hash(off)], "async mapped ids "), 0);
            // Apart from the ids, every rank's joins and the completions
            // double in place: the graphs and the join points are the late
            // route's.
            assert_eq!(runs(on, 2), runs(off, 1), "ranks={}", ranks.get());
        }
        let logs = on.host.logs.borrow();
        let early: Vec<serde_json::Value> = logs
            .iter()
            .filter(|line| line.contains("\"phase\":\"dspark-engram\""))
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()["early"].clone())
            .collect();
        assert_eq!(early.len(), 3);
        for layers in early {
            let layers = layers.as_array().unwrap();
            assert_eq!(layers.len(), FLASH.shape.engram.layers.len());
            assert!(layers.iter().all(|layer| layer["rows"] == 4));
        }
        assert!(
            !off.host
                .logs
                .borrow()
                .iter()
                .any(|line| line.contains("\"early\""))
        );
    }
}
