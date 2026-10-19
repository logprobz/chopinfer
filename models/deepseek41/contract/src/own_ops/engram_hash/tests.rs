use super::*;
use crate::geometry::GEOMETRIES;

fn fixture() -> (EngramHash, HashTables) {
    let g = GEOMETRIES.iter().find(|g| g.id == "flash").unwrap();
    let hash = HashTables::of(g).unwrap();
    let e = g.shape.engram;
    (
        EngramHash {
            vocab: g.shape.vocab,
            ngram: e.max_ngram,
            heads: e.heads,
            layers: e.layers.len() as u32,
            items: 2,
            stride: 12,
        },
        hash,
    )
}

#[test]
fn every_budget_matches_full_sequence_with_dead_boundaries_and_restored_lookback() {
    let (op, hash) = fixture();
    let past = [21, 81, 17].map(|token| hash.compress(token, true).unwrap());
    for count in 1..=6_usize {
        let tokens: Vec<_> = (0..count)
            .map(|i| 51 + i as u32)
            .chain((0..count).map(|i| 70 + i as u32))
            .collect();
        let batch: Vec<_> = (0..2)
            .flat_map(|item| std::iter::repeat_n(item, count))
            .collect();
        let start = [0, count as i32];
        let history: Vec<_> = past.into_iter().chain([DEAD, past[1], past[2]]).collect();
        let mut text = vec![1; tokens.len()];
        text[count] = 0;
        let mut ids = vec![u32::MAX; op.output_bytes() as usize / 4];
        let mut status = vec![0; tokens.len()];
        oracle(
            op,
            &hash,
            HashInput {
                tokens: &tokens,
                token_batch: &batch,
                q_start: &start,
                history: &history,
                text: &text,
            },
            &mut ids,
            &mut status,
        )
        .unwrap();
        assert!(status.iter().all(|&s| s == 0));
        for item in 0..2 {
            let mut sequence = history[item * 3..item * 3 + 3].to_vec();
            for at in item * count..(item + 1) * count {
                sequence.push(hash.compress(tokens[at], text[at] != 0).unwrap());
            }
            let expected = hash.sequence(&sequence).unwrap();
            let columns = op.columns() as usize;
            for row in 0..count {
                for layer in 0..op.layers as usize {
                    let from = ((row + 3) * op.layers as usize + layer) * columns;
                    let to = (layer * op.stride as usize + item * count + row) * columns;
                    assert_eq!(&ids[to..to + columns], &expected[from..from + columns]);
                }
            }
        }
        for layer in 0..op.layers as usize {
            let start = (layer * op.stride as usize + tokens.len()) * op.columns() as usize;
            let end = (layer + 1) * op.stride as usize * op.columns() as usize;
            assert!(ids[start..end].iter().all(|&id| id == u32::MAX));
        }
    }
}

#[test]
fn malformed_device_metadata_sets_sticky_status_and_safe_indices() {
    let (op, hash) = fixture();
    for case in 0..6 {
        let mut tokens = [10, 11];
        let mut batch = [0, 1];
        let mut start = [0, 1];
        let mut history = [0_i64; 6];
        let mut text = [1_u8; 2];
        let mut status = [0_u32; 2];
        match case {
            0 => tokens[0] = op.vocab,
            1 => batch[0] = -1,
            2 => start[0] = 1,
            3 => history[2] = i64::MAX,
            4 => text[0] = 2,
            _ => status[0] = 4,
        }
        let mut ids = vec![u32::MAX; op.output_bytes() as usize / 4];
        oracle(
            op,
            &hash,
            HashInput {
                tokens: &tokens,
                token_batch: &batch,
                q_start: &start,
                history: &history,
                text: &text,
            },
            &mut ids,
            &mut status,
        )
        .unwrap();
        assert_ne!(status[0] & 1, 0);
        if case == 5 {
            assert_eq!(status[0], 5);
        }
        assert_eq!(status[1], 0);
        for layer in 0..op.layers as usize {
            let at = layer * op.stride as usize * op.columns() as usize;
            assert!(
                ids[at..at + op.columns() as usize]
                    .iter()
                    .all(|&id| id == 0)
            );
        }
    }
}

#[test]
fn packed_constants_preserve_the_authenticated_map_and_integer_products() {
    let (op, hash) = fixture();
    let words = hash.device_words();
    assert_eq!(words.len() as u64, op.constant_words());
    for token in 0..op.vocab {
        assert_eq!(
            words[2 + token as usize] as i64,
            hash.compress(token, true).unwrap()
        );
    }
    let multipliers = &words[2 + op.vocab as usize..][..(op.layers * op.ngram) as usize];
    assert!(
        multipliers
            .iter()
            .all(|&m| m & 1 == 1 && m.checked_mul(words[0] - 1).unwrap() <= i64::MAX as u64)
    );
    assert!(
        EngramHash {
            stride: u32::MAX,
            layers: u32::MAX,
            ..op
        }
        .validate(Need::rows(1))
        .is_err()
    );
}
