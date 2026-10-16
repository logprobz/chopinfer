use kernel_core::select::{NanKeys, Ties, topk};

#[test]
fn topk_breaks_ties_by_lower_index_and_honors_the_mask() {
    let scores = [1.0, 2.0, 2.0, -0.0, 0.0, 2.0, f64::NEG_INFINITY];
    assert_eq!(
        topk(&scores, 4, Ties::LowerIndex, NanKeys::Refuse, None).unwrap(),
        [1, 2, 5, 0]
    );
    // total_cmp ranks +0 above -0, whatever their indexes.
    assert_eq!(
        topk(&scores, 7, Ties::LowerIndex, NanKeys::Refuse, None).unwrap(),
        [1, 2, 5, 0, 4, 3, 6]
    );
    let mask = [true, false, true, true, true, false, true];
    assert_eq!(
        topk(&scores, 3, Ties::LowerIndex, NanKeys::Refuse, Some(&mask)).unwrap(),
        [2, 0, 4]
    );
    // Fewer selectable keys than k: all of them.
    assert_eq!(
        topk(&scores, 9, Ties::LowerIndex, NanKeys::Refuse, Some(&mask))
            .unwrap()
            .len(),
        5
    );
    assert!(topk(&[1.0, f64::NAN], 1, Ties::LowerIndex, NanKeys::Refuse, None).is_err());
    assert!(
        topk(
            &scores,
            1,
            Ties::LowerIndex,
            NanKeys::Refuse,
            Some(&mask[..3])
        )
        .is_err()
    );
}

fn values(len: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0) * scale
        })
        .collect()
}

#[test]
fn streaming_head_matches_the_whole_row_log_softmax() {
    use kernel_core::head::StreamingHead;
    let mut logits = values(1000, 9, 40.0);
    // Ties across chunk boundaries keep the lower token id.
    logits[17] = 60.0;
    logits[700] = 60.0;
    logits[3] = 59.0;
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_z = logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln() + max;
    let mut ranked = (0..logits.len()).collect::<Vec<_>>();
    ranked.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    for chunk in [1000, 333, 64, 1] {
        let mut head = StreamingHead::new(5, Ties::LowerIndex, Some(700));
        for part in logits.chunks(chunk) {
            head.push(part).unwrap();
        }
        let score = head.finish().unwrap();
        assert!((score.log_z - log_z).abs() <= 1e-12 * log_z.abs());
        let ids = score.top.iter().map(|&(id, _)| id).collect::<Vec<_>>();
        assert_eq!(ids, ranked[..5]);
        assert_eq!(ids[..3], [17, 700, 3]);
        for &(id, logprob) in &score.top {
            assert_eq!(logprob, logits[id] - score.log_z);
        }
        assert_eq!(score.target, Some(60.0 - score.log_z));
        assert_eq!(score.margin, Some(0.0));
        if chunk == 1000 {
            // One chunk is the whole-row arithmetic exactly.
            assert_eq!(
                score.log_z,
                max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln()
            );
            assert_eq!(
                ids,
                topk(&logits, 5, Ties::LowerIndex, NanKeys::Refuse, None).unwrap()
            );
        }
    }
}

#[test]
fn streaming_head_refuses_nonfinite_logits_and_unseen_targets() {
    use kernel_core::head::StreamingHead;
    let mut head = StreamingHead::new(2, Ties::LowerIndex, None);
    assert!(head.push(&[1.0, f64::INFINITY]).is_err());
    assert!(head.push(&[1.0, f64::NAN]).is_err());
    assert!(
        StreamingHead::new(2, Ties::LowerIndex, None)
            .finish()
            .is_err()
    );
    let mut head = StreamingHead::new(2, Ties::LowerIndex, Some(4));
    head.push(&[1.0, 2.0, 3.0]).unwrap();
    assert!(head.finish().is_err());
    let mut head = StreamingHead::new(1, Ties::LowerIndex, None);
    head.push(&[1.0, 2.0]).unwrap();
    let score = head.finish().unwrap();
    assert_eq!(score.margin, None);
    assert_eq!(score.top, [(1, 2.0 - score.log_z)]);
}

#[test]
fn empty_head_chunks_change_nothing() {
    use kernel_core::head::StreamingHead;
    let logits = values(50, 11, 8.0);
    let score = |chunks: &[&[f64]]| {
        let mut head = StreamingHead::new(3, Ties::LowerIndex, Some(20));
        for chunk in chunks {
            head.push(chunk).unwrap();
        }
        head.finish().unwrap()
    };
    let reference = score(&[&logits[..20], &logits[20..]]);
    assert!(reference.log_z.is_finite());
    for chunks in [
        vec![&[][..], &logits[..20], &logits[20..]],
        vec![&logits[..20], &[], &[], &logits[20..]],
        vec![&logits[..20], &logits[20..], &[]],
    ] {
        assert_eq!(score(&chunks), reference);
    }
    // Only empty chunks: still no logits.
    let mut head = StreamingHead::new(1, Ties::LowerIndex, None);
    head.push(&[]).unwrap();
    assert!(head.finish().is_err());
}

#[test]
fn ieee_ties_take_the_first_strict_maximum() {
    use kernel_core::head::StreamingHead;
    let scores = [-0.0, 0.0, 1.0, 1.0, -0.0];
    assert_eq!(
        topk(&scores, 5, Ties::IeeeLowerIndex, NanKeys::Refuse, None).unwrap(),
        [2, 3, 0, 1, 4]
    );
    assert_eq!(
        topk(&[-0.0, 0.0], 1, Ties::IeeeLowerIndex, NanKeys::Refuse, None).unwrap(),
        [0]
    );
    assert_eq!(
        topk(&[-0.0, 0.0], 1, Ties::LowerIndex, NanKeys::Refuse, None).unwrap(),
        [1]
    );
    let mut head = StreamingHead::new(2, Ties::IeeeLowerIndex, None);
    head.push(&[-0.0, 0.0, -1.0]).unwrap();
    let ids = head
        .finish()
        .unwrap()
        .top
        .iter()
        .map(|t| t.0)
        .collect::<Vec<_>>();
    assert_eq!(ids, [0, 1]);
}

#[test]
fn nan_keys_are_refused_or_ranked_by_total_order() {
    let nan = f64::NAN;
    let scores = [1.0, nan, f64::INFINITY, -nan, 2.0];
    assert!(topk(&scores, 2, Ties::LowerIndex, NanKeys::Refuse, None).is_err());
    assert!(topk(&scores, 2, Ties::IeeeLowerIndex, NanKeys::TotalOrder, None).is_err());
    assert_eq!(
        topk(&scores, 5, Ties::LowerIndex, NanKeys::TotalOrder, None).unwrap(),
        [1, 2, 4, 0, 3]
    );
}

#[test]
fn partial_selection_ranks_as_a_full_sort() {
    let scores = values(500, 9, 1.0)
        .iter()
        .map(|v| (v * 8.0).round() / 8.0)
        .collect::<Vec<_>>();
    for ties in [Ties::LowerIndex, Ties::IeeeLowerIndex] {
        let full = topk(&scores, scores.len(), ties, NanKeys::Refuse, None).unwrap();
        for k in [1, 2, 7, 64, 499] {
            assert_eq!(
                topk(&scores, k, ties, NanKeys::Refuse, None).unwrap(),
                full[..k]
            );
        }
    }
}
