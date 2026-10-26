//! Drafted DSpark steps gather each item's first Engram rows before the
//! drafter. The late gathers keep the device hash,
//! the early gathers' ids are the host's hashes of each item's first row,
//! every rank joins both gathers of a layer before its consumer, both
//! complete at the boundary, and every resource is released.
use super::*;
use crate::engram_rows::{EngramRows, Step};
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
    /// Each drafted step's execute journal lines.
    journals: Vec<Vec<String>>,
}

/// Four three-token prompts, then `steps` drafted steps of every slot, each
/// keeping its bonus row.
fn schedule(ranks: Ranks, steps: usize) -> Run {
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let (host, mut instance) = create(shape);
    let (span, stride) = crate::instance::dspark_early_ids(instance.handle()).expect("early IDs");
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
    let mut journals = Vec::new();
    for _ in 0..steps {
        let items: Vec<(u64, u64, u64)> = (0..4)
            .map(|slot| (7 + slot, slot, histories[slot as usize].len() as u64))
            .collect();
        let begin = host.journal.borrow().len();
        let sink = execute(&mut instance, &host, &verify_all(&items, 1));
        journals.push(host.journal.borrow()[begin..].to_vec());
        assert_eq!(sink.draft_tokens().len(), 4);
        assert_eq!(
            host.uploads.borrow()[&(span.buffer.raw, span.offset)],
            early_ids(&histories, stride, span.length)
        );
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
    Run { host, journals }
}

/// The step's lines of one kind.
fn count(lines: &[String], prefix: &str) -> usize {
    lines.iter().filter(|line| line.starts_with(prefix)).count()
}

#[test]
fn early_first_rows_precede_the_drafter_and_complete_at_the_boundary() {
    for ranks in [Ranks::TP4, Ranks::TP2, Ranks::TP1] {
        let on = schedule(ranks, 3);
        for step in &on.journals {
            assert_eq!(count(step, "async mapped ids "), 4);
            assert_eq!(count(step, "async mapped wait "), 4 * ranks.count());
            assert_eq!(count(step, "async mapped complete"), 4);
            let hash = step
                .iter()
                .position(|line| line.contains(crate::sm120::engram_hash_rows::SYMBOL))
                .unwrap();
            let drafter = step
                .iter()
                .position(|line| line.starts_with("capture ") || line.starts_with("replay "))
                .unwrap();
            assert!(drafter < hash);
            assert_eq!(count(&step[..drafter], "async mapped ids "), 2);
            assert_eq!(count(&step[..hash], "async mapped ids "), 2);
            assert_eq!(count(&step[..hash], "async mapped wait "), 0);
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
    }
}
