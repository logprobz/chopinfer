//! Prefill DMA over BF16 partials sends each rank's partial rounded once to
//! BF16, as the BF16-partial collectives it replaces do.
use super::*;
use crate::ops::with_partial_setting;

fn capture_bf16(mode: u8, rows: u64) -> Vec<String> {
    // Residency that admits the BF16-partial entries' grids.
    let host = super::super::bf16_partials::host(true);
    host.device
        .ordinary_symbols
        .borrow_mut()
        .extend([SUM, BF16_ROUND, BF16_SUM].map(String::from));
    let mut instance = with_partial_setting(Some("prefill"), || {
        with_dma(true, || with_mode(mode, || create(&host, 4)))
    });
    let created = host.device.journal.borrow().clone();
    assert!(
        created
            .iter()
            .any(|line| line.starts_with("resolve ") && line.ends_with(BF16_SUM))
    );
    host.device.journal.borrow_mut().clear();
    let batch = prompt(7, 0, 0, rows, true);
    with_partial_setting(Some("prefill"), || {
        instance.execute(&batch, &mut Views::default())
    })
    .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    let journal = host.device.journal.borrow().clone();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    journal
}

fn launches(journal: &[String], symbol: &str) -> Vec<String> {
    journal
        .iter()
        .filter(|s| s.starts_with("launch ") && s.contains(&format!(" {symbol} ")))
        .cloned()
        .collect()
}

#[test]
fn bf16_prefill_dma_rounds_once_and_sends_half_the_scatter_bytes() {
    let hidden = u64::from(FLASH.shape.hidden);
    for (mode, rows, calls) in [(0, 8192, 81), (0, 8191, 81), (2, 8191, 162)] {
        let journal = capture_bf16(mode, rows);
        // No replaced collective remains, and no F32 payload is sent.
        assert!(
            !journal
                .iter()
                .any(|s| s.starts_with("launch ") && s.contains("chop_comm_allreduce_phased_"))
        );
        assert!(launches(&journal, SUM).is_empty());
        assert_eq!(launches(&journal, BF16_SUM).len(), calls * 4);
        // Each rank rounds the rows it sends: everything but its own rows,
        // one interval for the first and last owner and two between.
        let rounds = launches(&journal, BF16_ROUND);
        assert_eq!(rounds.len(), calls * 6);
        let rounded = rounds
            .iter()
            .map(|s| {
                let spans = super::shared_overlap::journal_spans(s);
                assert_eq!(spans.len(), 2);
                assert_eq!(spans[0].2, 2 * spans[1].2, "F32 rows into BF16 rows");
                spans[1].2
            })
            .sum::<u64>();
        // Three of four ranks round each row: 2 bytes per element.
        assert_eq!(rounded, 81 * 3 * rows * hidden * 2);
        let copies = journal
            .iter()
            .filter(|s| s.starts_with("peer-copy "))
            .collect::<Vec<_>>();
        assert_eq!(copies.len(), calls * 24);
        let bytes = copies
            .iter()
            .map(|s| {
                let spans = super::shared_overlap::journal_spans(s);
                assert_eq!(spans.len(), 2);
                assert_eq!(spans[0].2, spans[1].2);
                spans[0].2
            })
            .sum::<u64>();
        // BF16 scatter and gather: 3 + 3 row copies of 2 bytes per element,
        // against 3 * 4 + 3 * 2 for F32 partials.
        assert_eq!(bytes, 81 * 12 * rows * hidden);
        assert_eq!(audit(&journal).unwrap(), calls * (24 * 2 + 4 * 3));
        let missing_waits = journal
            .iter()
            .filter(|s| !s.starts_with("wait "))
            .cloned()
            .collect::<Vec<_>>();
        assert!(audit(&missing_waits).is_err());
        // A round submitted after its first scatter is unordered.
        let mut late = journal.clone();
        let first_round = late.iter().position(|s| s.contains(BF16_ROUND)).unwrap();
        let round = late.remove(first_round);
        let first_copy = late
            .iter()
            .position(|s| s.starts_with("peer-copy "))
            .unwrap();
        late.insert(first_copy + 1, round);
        assert!(audit(&late).is_err());
    }
}

#[test]
fn bf16_prefill_dma_keeps_the_owner_rows_and_the_rounded_rows_disjoint() {
    let journal = capture_bf16(0, 8191);
    let rounds = launches(&journal, BF16_ROUND);
    let sums = launches(&journal, BF16_SUM);
    let spans = super::shared_overlap::journal_spans;
    // Per call, ranks in order: each rank's rounded output rows and its
    // BF16 inbox lie in one allocation and never overlap.
    for (call, sums) in sums.chunks_exact(4).enumerate() {
        let rounds = &rounds[call * 6..call * 6 + 6];
        let mut at = 0;
        for (rank, sum) in sums.iter().enumerate() {
            let s = spans(sum);
            let (input, inbox, output) = (s[0], s[1], s[2]);
            assert_eq!(input.2, 2 * output.2);
            assert_eq!(inbox.2, 4 * 2048 * u64::from(FLASH.shape.hidden) * 2);
            let pieces = if rank == 0 || rank == 3 { 1 } else { 2 };
            for round in &rounds[at..at + pieces] {
                let r = spans(round);
                assert_eq!(r[1].0, inbox.0, "rounded rows share the inbox allocation");
                assert!(r[1].1 >= inbox.1 + inbox.2, "rounded rows follow the inbox");
                assert!(
                    !intersects(r[0], input),
                    "the owner's rows are not rounded early"
                );
            }
            at += pieces;
        }
    }
}

#[test]
fn f32_prefill_dma_stays_where_partials_are_off() {
    let journal = capture(0, 8192);
    assert!(launches(&journal, BF16_ROUND).is_empty());
    assert!(launches(&journal, BF16_SUM).is_empty());
    assert_eq!(launches(&journal, SUM).len(), 81 * 4);
}
