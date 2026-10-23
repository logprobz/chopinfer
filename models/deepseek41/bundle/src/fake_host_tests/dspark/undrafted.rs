//! A DSpark instance's decode steps without drafts upload their Engram
//! rows per layer through the gathers its drafted steps use, as AR decode
//! does (`instance::ar_engram`): every stream launches the same kernels in
//! the same order as the synchronous route, the rows land in the same
//! spans, the gathers' ids are the host's hashes, and each rank joins a
//! layer before the piece holding its consumer. Drafted steps between them
//! keep the device hash.
use super::*;
use crate::engram_rows::{EngramRows, Step};
use crate::fake_host_tests::ar_engram::{launches, order};
use crate::instance::ar_engram::{with_enabled, with_switch};
use model_deepseek41_contract::own_ops::engram::HashTables;

/// The token request `7 + slot` decodes at `position`.
fn token(slot: u64, position: u64) -> u32 {
    (slot * 37 + position * 11 + 5) as u32
}

/// A decode step without drafts of `(slot, position)` items.
fn undrafted(items: &[(u64, u64)]) -> Vec<u8> {
    let items: Vec<_> = items
        .iter()
        .map(|&(slot, position)| (7 + slot, slot, position, token(slot, position)))
        .collect();
    super::super::decode(&items)
}

/// The ids the synchronous route gathers for a decode step of `items`,
/// oracle-hashed from each request's committed tokens.
fn oracle_ids(histories: &[Vec<u32>], items: &[(u64, u64)]) -> crate::engram_rows::Hashed {
    let mut rows = EngramRows::new(HashTables::of(&FLASH).unwrap());
    for &(slot, _) in items {
        rows.restore(7 + slot, &histories[slot as usize], None)
            .unwrap();
    }
    let tokens: Vec<[u32; 1]> = items
        .iter()
        .map(|&(slot, position)| [token(slot, position)])
        .collect();
    let steps: Vec<Step<'_>> = items
        .iter()
        .zip(&tokens)
        .map(|(&(slot, _), tokens)| Step {
            request: 7 + slot,
            tokens,
            text: None,
        })
        .collect();
    rows.hash(&steps).unwrap()
}

/// Each row write a gather made: the upload that follows a synchronous
/// gather line, sorted.
fn row_writes(lines: &[String]) -> Vec<String> {
    let mut writes: Vec<String> = lines
        .windows(2)
        .filter(|pair| pair[0].starts_with("mapped gather ") && pair[1].starts_with("upload "))
        .map(|pair| format!("{} {}", pair[0], pair[1]))
        .collect();
    writes.sort();
    writes
}

/// One route's run of the schedule.
struct Run {
    host: Rc<Host>,
    /// Where the first decode step's journal starts.
    from: usize,
    /// Each step's execute journal lines.
    journals: Vec<Vec<String>>,
}

/// A step of the schedule: a decode without drafts of `(slot, position)`
/// items, or a drafted step of every slot.
enum Planned {
    Undrafted(&'static [(u64, u64)]),
    Drafted,
}

/// Four three-token prompts, then each planned step, committed.
fn schedule(ranks: Ranks, steps: &[Planned], on: bool) -> Run {
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let (host, mut instance) = with_enabled(on, || create(shape));
    let ids = crate::instance::dspark_hash_ids(instance.handle()).unwrap();
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
    for planned in steps {
        let begin = host.journal.borrow().len();
        match planned {
            Planned::Undrafted(items) => {
                let sink = execute(&mut instance, &host, &undrafted(items));
                journals.push(host.journal.borrow()[begin..].to_vec());
                assert!(sink.draft_tokens().is_empty());
                if on {
                    let (span, stride) = ids;
                    let hashed = oracle_ids(&histories, items);
                    let layers = FLASH.shape.engram.layers.len();
                    let columns = hashed.ids().len() / hashed.rows() / layers;
                    let mut wanted = vec![0_u8; span.length as usize];
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
                    assert_eq!(
                        host.uploads.borrow()[&(span.buffer.raw, span.offset)],
                        wanted,
                        "{items:?}"
                    );
                }
                instance.commit().unwrap();
                for &(slot, position) in items.iter() {
                    histories[slot as usize].push(token(slot, position));
                }
            }
            Planned::Drafted => {
                let items: Vec<(u64, u64, u64)> = (0..4)
                    .map(|slot| (7 + slot, slot, histories[slot as usize].len() as u64))
                    .collect();
                let sink = execute(&mut instance, &host, &verify_all(&items, 1));
                journals.push(host.journal.borrow()[begin..].to_vec());
                assert_eq!(sink.draft_tokens().len(), 4);
                let kept: Vec<_> = (7..11).map(|request| kept(request, 1)).collect();
                instance.commit_accepted(&kept).unwrap();
                for history in &mut histories {
                    history.push(7);
                }
            }
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

#[test]
fn undrafted_dspark_steps_join_each_engram_layer_without_changing_launches() {
    // Every slot, its replay, a smaller bucket, a drafted step, every slot
    // after it, and one row.
    let steps = [
        Planned::Undrafted(&[(0, 3), (1, 3), (2, 3), (3, 3)]),
        Planned::Undrafted(&[(0, 4), (1, 4), (2, 4), (3, 4)]),
        Planned::Undrafted(&[(0, 5), (1, 5)]),
        Planned::Drafted,
        Planned::Undrafted(&[(0, 7), (1, 7), (2, 6), (3, 6)]),
        Planned::Undrafted(&[(2, 7)]),
    ];
    for ranks in [Ranks::TP4, Ranks::TP2, Ranks::TP1] {
        let off = schedule(ranks, &steps, false);
        let on = schedule(ranks, &steps, true);
        assert_eq!(
            launches(&off.host, off.from),
            launches(&on.host, on.from),
            "ranks={}",
            ranks.get()
        );
        let count = ranks.count();
        for ((planned, off), on) in steps.iter().zip(&off.journals).zip(&on.journals) {
            assert_eq!(row_writes(off), row_writes(on));
            let (off, on) = (order(off), order(on));
            if matches!(planned, Planned::Drafted) {
                assert_eq!(on, off, "a drafted step keeps the device hash");
                continue;
            }
            assert!(!off.contains(&"ids") && !off.contains(&"wait"));
            assert!(off.contains(&"sync"));
            // Both layers enqueued, the prefix pieces; every rank joins
            // layer 1, the middle pieces; every rank joins layer 14, the
            // last pieces; both gathers complete.
            assert!(!on.contains(&"sync"), "{on:?}");
            let first = on.iter().position(|&k| k == "ids").unwrap();
            assert_eq!(&on[first..first + 2], ["ids", "ids"]);
            let groups: Vec<(&str, usize)> =
                on[first + 2..].iter().fold(Vec::new(), |mut acc, &k| {
                    match acc.last_mut() {
                        Some((last, n)) if *last == k => *n += 1,
                        _ => acc.push((k, 1)),
                    }
                    acc
                });
            assert_eq!(
                groups.iter().map(|g| g.0).collect::<Vec<_>>(),
                ["graph", "wait", "graph", "wait", "graph", "complete"],
                "{groups:?}"
            );
            assert_eq!(groups[1].1, count);
            assert_eq!(groups[3].1, count);
            assert_eq!(groups[5].1, 2);
        }
        let logs = on.host.logs.borrow();
        let rows: Vec<u64> = logs
            .iter()
            .filter(|line| line.contains("chopinfer.deepseek41.ar-engram.v1"))
            .map(|line| {
                serde_json::from_str::<serde_json::Value>(line).unwrap()["rows"]
                    .as_u64()
                    .unwrap()
            })
            .collect();
        assert_eq!(rows, [4, 4, 2, 4, 1]);
    }
}

/// `0` keeps the synchronous route for DSpark's undrafted steps too, and
/// an unset switch serves them per layer.
#[test]
fn the_switch_governs_undrafted_dspark_steps() {
    for (value, deferred) in [(None, true), (Some("1"), true), (Some("0"), false)] {
        let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
        let (host, mut instance) = with_switch(value, || create(shape));
        execute(
            &mut instance,
            &host,
            &super::super::prompt(7, 0, 0, 3, true),
        );
        instance.commit().unwrap();
        let begin = host.journal.borrow().len();
        execute(&mut instance, &host, &undrafted(&[(0, 3)]));
        instance.commit().unwrap();
        let step = order(&host.journal.borrow()[begin..]);
        assert_eq!(step.contains(&"ids"), deferred, "{value:?}");
        assert_eq!(!step.contains(&"sync"), deferred, "{value:?}");
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
