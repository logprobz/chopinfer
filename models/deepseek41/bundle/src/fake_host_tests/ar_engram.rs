//! AR decode's per-layer Engram uploads change only when the rows copy:
//! every stream launches the same kernels with the same operands in the
//! same order as the synchronous route, the uploaded ids are the host's
//! hashes, and each rank joins a layer after the graph pieces before its
//! consumer and before the piece that holds it.
use super::*;
use crate::engram_rows::{EngramRows, Step};
use crate::instance::ar_engram::{indices, with_switch};
use model_deepseek41_contract::own_ops::engram::HashTables;

fn create_mapped(host: &Rc<Host>, items: u32, on: bool) -> BundleInstanceV1 {
    create_switched(host, items, Some(if on { "1" } else { "0" }))
        .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()))
}

/// An AR instance on mapped tables created as if the switch held `value`.
fn create_switched(
    host: &Rc<Host>,
    items: u32,
    value: Option<&'static str>,
) -> Result<BundleInstanceV1, BundleAbiError> {
    host.drafter.set(true);
    let record = crate::config::tests::record(&host.shape, host.kv, items, false);
    with_switch(value, || {
        api().create(
            HostServicesTableV1::new(Rc::new(dspark::host::MappedHost::new(Rc::clone(host)))),
            &record,
        )
    })
}

/// Each stream's launch lines from `from` on, in journal order.
pub(super) fn launches(host: &Host, from: usize) -> BTreeMap<String, Vec<String>> {
    let mut result = BTreeMap::<String, Vec<String>>::new();
    for line in host.device.journal.borrow()[from..]
        .iter()
        .filter(|line| line.starts_with("launch "))
    {
        let stream = line.split_whitespace().nth(1).unwrap().to_owned();
        result.entry(stream).or_default().push(line.clone());
    }
    result
}

/// The tokens of request `request`'s prompt: three, as `prompt` writes.
const PROMPT: u64 = 3;

/// The token request `7 + slot` decodes at `position`.
fn token(slot: u64, position: u64) -> u32 {
    (slot * 37 + position * 11 + 5) as u32
}

/// One decode step of the first `live` requests, request `7 + slot` in
/// `slot` at its committed length `positions[slot]`.
fn decode_step(positions: &[u64], live: u64) -> Vec<u8> {
    let items: Vec<_> = (0..live)
        .map(|slot| {
            let position = positions[slot as usize];
            (7 + slot, slot, position, token(slot, position))
        })
        .collect();
    decode(&items)
}

/// The ids the synchronous route gathers for that step, oracle-hashed
/// from each request's committed tokens.
fn oracle_ids(positions: &[u64], live: u64) -> crate::engram_rows::Hashed {
    let mut rows = EngramRows::new(HashTables::of(&FLASH).unwrap());
    for slot in 0..live {
        let mut history: Vec<u32> = (0..PROMPT as u32).collect();
        history.extend((PROMPT..positions[slot as usize]).map(|at| token(slot, at)));
        rows.restore(7 + slot, &history, None).unwrap();
    }
    let tokens: Vec<[u32; 1]> = (0..live)
        .map(|slot| [token(slot, positions[slot as usize])])
        .collect();
    let steps: Vec<Step<'_>> = tokens
        .iter()
        .enumerate()
        .map(|(slot, tokens)| Step {
            request: 7 + slot as u64,
            tokens,
            text: None,
        })
        .collect();
    rows.hash(&steps).unwrap()
}

/// One route's run of a schedule.
struct Run {
    host: Rc<Host>,
    /// Where the first decode step's journal starts.
    from: usize,
    /// The id span and Engram layers, when the route prepared its gathers.
    spans: Option<(DeviceBufferSpanV1, Vec<u32>)>,
    /// Each decode step's journal lines.
    journals: Vec<Vec<String>>,
}

/// Runs the schedule on one route: `items` prompts, then a decode step of
/// the first `live` requests for each of `steps`.
fn schedule(ranks: Ranks, items: u32, steps: &[u64], on: bool) -> Run {
    let host = FakeHost::new(Shape::new(&FLASH, ranks).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create_mapped(&host, items, on);
    let spans = indices(instance.handle());
    assert_eq!(spans.is_some(), on);
    for slot in 0..u64::from(items) {
        step(
            &mut instance,
            &prompt(7 + slot, slot, 0, PROMPT, true),
            &mut Views::default(),
        );
    }
    let from = host.device.journal.borrow().len();
    let mut journals = Vec::new();
    let mut positions = vec![PROMPT; items as usize];
    for &live in steps {
        let begin = host.device.journal.borrow().len();
        let mut views = Views::default();
        step(&mut instance, &decode_step(&positions, live), &mut views);
        assert_eq!(u64::from(views.rows()), live);
        journals.push(host.device.journal.borrow()[begin..].to_vec());
        if on {
            let (span, layers) = spans.as_ref().unwrap();
            let hashed = oracle_ids(&positions, live);
            let columns = hashed.ids().len() / hashed.rows() / layers.len();
            let mut wanted = vec![0_u8; span.length as usize];
            let per_layer = span.length as usize / layers.len();
            for (row, ids) in hashed
                .ids()
                .chunks_exact(columns * layers.len())
                .enumerate()
            {
                for layer in 0..layers.len() {
                    for (column, id) in ids[layer * columns..(layer + 1) * columns]
                        .iter()
                        .enumerate()
                    {
                        let at = layer * per_layer + (row * columns + column) * 4;
                        wanted[at..at + 4].copy_from_slice(&id.to_le_bytes());
                    }
                }
            }
            assert_eq!(
                host.uploads.borrow()[&(span.buffer.raw, span.offset)],
                wanted,
                "live={live}"
            );
        }
        for position in &mut positions[..live as usize] {
            *position += 1;
        }
    }
    for slot in 0..u64::from(items) {
        instance.retire_request(7 + slot).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    Run {
        host,
        from,
        spans,
        journals,
    }
}

/// The step's id uploads, gather waits and completions, synchronous
/// gathers and graph captures or replays, in order, each reduced to its
/// kind. The fake host stages a waited layer's rows as synchronous gathers
/// at the wait; those belong to the wait.
pub(super) fn order(lines: &[String]) -> Vec<&'static str> {
    let mut kinds = Vec::new();
    let mut waiting = false;
    for line in lines {
        let kind = if line.starts_with("async mapped ids ") {
            "ids"
        } else if line.starts_with("async mapped wait ") {
            "wait"
        } else if line.starts_with("async mapped complete") {
            "complete"
        } else if line.starts_with("mapped gather ") {
            if waiting {
                continue;
            }
            "sync"
        } else if line.starts_with("replay ") || line.starts_with("capture ") {
            "graph"
        } else {
            continue;
        };
        waiting = kind == "wait";
        kinds.push(kind);
    }
    kinds
}

#[test]
fn ar_decode_joins_each_engram_layer_without_changing_launches() {
    // (ranks, items, live rows of each decode step): a capture, a replay,
    // a smaller bucket, a one-row step, then the first bucket again.
    for (ranks, items, steps) in [
        (Ranks::TP4, 32, &[32, 32, 13, 1, 32][..]),
        (Ranks::TP4, 16, &[16, 16, 2][..]),
        (Ranks::TP2, 8, &[8, 5, 1, 8][..]),
        (Ranks::TP1, 4, &[3, 4][..]),
        // C1: a one-row capture and its replay, a wider step, one row again.
        (Ranks::TP4, 4, &[1, 1, 4, 1][..]),
        (Ranks::TP1, 2, &[1, 1][..]),
    ] {
        let Run {
            host: off_host,
            from: off_from,
            journals: off_steps,
            ..
        } = schedule(ranks, items, steps, false);
        let Run {
            host: on_host,
            from: on_from,
            spans,
            journals: on_steps,
        } = schedule(ranks, items, steps, true);
        assert_eq!(
            launches(&off_host, off_from),
            launches(&on_host, on_from),
            "ranks={} items={items}",
            ranks.get()
        );
        let (_, layers) = spans.unwrap();
        assert_eq!(layers, FLASH.shape.engram.layers);
        let ranks = ranks.count();
        for ((&live, off), on) in steps.iter().zip(&off_steps).zip(&on_steps) {
            let off = order(off);
            let on = order(on);
            assert!(!off.contains(&"ids") && !off.contains(&"wait"));
            // Ids, both layers enqueued, then the prefix pieces; every rank
            // joins layer 1, the middle pieces run; every rank joins layer
            // 14, the last pieces run; both gathers complete.
            assert!(!on.contains(&"sync"));
            let first = on.iter().position(|&k| k == "ids").unwrap();
            assert_eq!(&on[first..first + 2], ["ids", "ids"]);
            let rest = &on[first + 2..];
            let groups: Vec<(&str, usize)> = rest.iter().fold(Vec::new(), |mut acc, &k| {
                match acc.last_mut() {
                    Some((last, count)) if *last == k => *count += 1,
                    _ => acc.push((k, 1)),
                }
                acc
            });
            assert_eq!(
                groups.iter().map(|g| g.0).collect::<Vec<_>>(),
                ["graph", "wait", "graph", "wait", "graph", "complete"],
                "live={live}: {groups:?}"
            );
            assert_eq!(groups[1].1, ranks);
            assert_eq!(groups[3].1, ranks);
            assert_eq!(groups[5].1, 2);
            assert!(off.contains(&"sync"));
        }
        let logs = on_host.logs.borrow();
        let traced: Vec<serde_json::Value> = logs
            .iter()
            .filter(|line| line.contains("chopinfer.deepseek41.ar-engram.v1"))
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            traced
                .iter()
                .map(|value| value["rows"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            steps
        );
        for value in &traced {
            let layers: Vec<u64> = value["layers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|layer| layer["layer"].as_u64().unwrap())
                .collect();
            assert_eq!(layers, [1, 14]);
        }
    }
}

/// A step refused after its gathers were enqueued leaves them pending; the
/// next step completes them before it writes any staging row, and decodes
/// on the per-layer route.
#[test]
fn a_refused_step_leaves_its_gathers_for_the_next_step_to_settle() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP2).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create_mapped(&host, 8, true);
    for slot in 0..2 {
        step(
            &mut instance,
            &prompt(7 + slot, slot, 0, PROMPT, true),
            &mut Views::default(),
        );
    }
    // Rank 1 flags row 1 of the decode: its read back follows the index
    // sources' and both of rank 0's status planes.
    let mut status = vec![0_u8; 2 * 4];
    status[4] = 1;
    host.injected.replace(Some(status));
    host.injected_after
        .set(FLASH.shape.csa.index_sources.len() + 2);
    let begin = host.device.journal.borrow().len();
    let error = instance
        .execute(&decode_step(&[PROMPT, PROMPT], 2), &mut Views::default())
        .unwrap_err();
    assert_eq!(error, BundleAbiError::Status(BundleStatusV1::BundleError));
    let refused = order(&host.device.journal.borrow()[begin..]);
    assert_eq!(refused.iter().filter(|&&k| k == "ids").count(), 2);
    assert!(!refused.contains(&"complete"));
    assert!(instance.commit().is_err());
    instance.retire_request(7).unwrap();
    instance.retire_request(8).unwrap();
    step(
        &mut instance,
        &prompt(9, 0, 0, PROMPT, true),
        &mut Views::default(),
    );
    let settled = order(&host.device.journal.borrow()[begin..]);
    let completes = settled.iter().position(|&k| k == "complete").unwrap();
    assert_eq!(&settled[completes..completes + 2], ["complete", "complete"]);
    // The prefill's synchronous gathers follow the settled uploads.
    assert!(settled[completes..].contains(&"sync"));
    step(
        &mut instance,
        &prompt(10, 1, 0, PROMPT, true),
        &mut Views::default(),
    );
    let mut views = Views::default();
    let begin = host.device.journal.borrow().len();
    step(
        &mut instance,
        &decode(&[(9, 0, PROMPT, 5), (10, 1, PROMPT, 6)]),
        &mut views,
    );
    assert_eq!(views.rows(), 2);
    let next = order(&host.device.journal.borrow()[begin..]);
    assert_eq!(next.iter().filter(|&&k| k == "complete").count(), 2);
    instance.retire_request(9).unwrap();
    instance.retire_request(10).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// The per-layer route serves AR decode when the switch is unset or 1, `0`
/// keeps the synchronous route, and any other value refuses creation.
#[test]
fn the_per_layer_route_is_the_default_and_zero_disables_it() {
    for (value, prepared) in [(None, true), (Some("1"), true), (Some("0"), false)] {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP2).unwrap(), KV_LAYOUTS[0]);
        let mut instance = create_switched(&host, 8, value)
            .unwrap_or_else(|error| panic!("{value:?}: {error:?}: {:?}", host.logs.borrow()));
        assert_eq!(indices(instance.handle()).is_some(), prepared, "{value:?}");
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP2).unwrap(), KV_LAYOUTS[0]);
    let error = create_switched(&host, 8, Some("yes"))
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        error,
        BundleAbiError::Status(BundleStatusV1::InvalidArgument)
    );
    assert_eq!(host.acquired.get(), host.released.get());
}
