//! Full prefill uses the same live/valid planners as decode and verify.
use super::*;

#[test]
fn prefill_indexer_blocks_use_live_and_valid_grids() {
    for kv in KV_LAYOUTS {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, kv);
        let record = crate::config::tests::record(&shape, kv, 1, false);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap();
        instance
            .execute(&prompt(7, 0, 0, 8192, true), &mut Views::default())
            .unwrap();
        let journal = host.device.journal.borrow();
        let scores = |family: &str| {
            journal
                .iter()
                .filter(|line| {
                    line.starts_with("launch ")
                        && line.contains(family)
                        && line.contains("_scores_v3")
                        && !line.contains("_pack_fp4_")
                })
                .collect::<Vec<_>>()
        };
        let dense = scores("chop_position_indexer_");
        let native = kv.storage == KvStorage::Native;
        assert_eq!(
            dense.len(),
            if native {
                4 * (3 * 2 + 16)
            } else {
                4 * 4 * 128
            }
        );
        for line in dense {
            assert_eq!(line.contains("_mma_fp4_guard_fast_v1"), native);
            assert!(line.contains("_scores_v3_live_v1"), "{line}");
            assert!(
                line.contains(&format!(
                    "grid_x: 192, grid_y: {}, grid_z: 1",
                    if !native {
                        64
                    } else if line.contains("_r2_") {
                        4096
                    } else {
                        512
                    }
                )),
                "{line}"
            );
        }
        let compact = scores("chop_position_blocks_");
        assert_eq!(compact.len(), 4 * 4 * 2);
        for line in compact {
            assert_eq!(line.contains("_mma_fp4_guard_fast_v1"), native);
            assert!(line.contains("_scores_v3_valid_v1"), "{line}");
            assert!(
                line.contains("grid_x: 192, grid_y: 4096, grid_z: 1"),
                "{line}"
            );
        }
        let candidates = scores("chop_candidate_block_ids_");
        assert_eq!(candidates.len(), 4 * if native { 16 } else { 128 });
        for line in candidates {
            assert!(line.contains("_scores_v3_valid_v1"), "{line}");
            assert!(
                line.contains(&format!(
                    "grid_x: 32, grid_y: {}, grid_z: 1",
                    if native { 512 } else { 64 }
                )),
                "{line}"
            );
        }
        if native {
            audit_batched(&host, &journal, false).unwrap();
            let at = journal
                .iter()
                .position(|line| {
                    line.starts_with("launch ")
                        && line.contains("_r2_all_scores_v3_live_v1_mma_fp4_guard_fast_v1 ")
                })
                .unwrap();
            let count = super::shared_overlap::journal_spans(&journal[at])[7];
            for (before, after) in [
                ("ScalarU32(4096)".to_owned(), "ScalarU32(2048)".to_owned()),
                (
                    format!("offset: {}, length: 4", count.1),
                    "offset: 0, length: 4".to_owned(),
                ),
            ] {
                let mut wrong = journal.clone();
                wrong[at] = wrong[at].replacen(&before, &after, 1);
                assert_ne!(wrong[at], journal[at]);
                assert!(audit_batched(&host, &wrong, false).is_err());
            }
        }
        drop(journal);
        instance.commit().unwrap();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

type Span = (u64, u64, u64);
fn uploaded(host: &Host, span: Span) -> Result<Vec<u8>, String> {
    host.uploads
        .borrow()
        .iter()
        .filter(|((buffer, base), data)| {
            *buffer == span.0 && *base <= span.1 && span.1 + span.2 <= *base + data.len() as u64
        })
        .max_by_key(|((_, base), _)| *base)
        .map(|((_, base), data)| {
            data[(span.1 - base) as usize..(span.1 + span.2 - base) as usize].to_vec()
        })
        .ok_or_else(|| format!("no uploaded bytes for {span:?}"))
}

/// Independently check the actual calls' positions, causal bound, immutable
/// counts, transient origins and carried output origins for the cold 8K case.
pub(super) fn audit_batched(host: &Host, journal: &[String], skip: bool) -> Result<usize, String> {
    let require = |ok: bool, message: &str| if ok { Ok(()) } else { Err(message.to_owned()) };
    let spans = super::shared_overlap::journal_spans;
    let mut checked = 0;
    for (at, line) in journal.iter().enumerate().filter(|(_, line)| {
        line.starts_with("launch ")
            && line.contains("chop_position_indexer_")
            && line.contains("_scores_v3_live_v1_mma_fp4_guard_fast_v1 ")
    }) {
        let b = spans(line);
        let n = b[0].2 / 8192;
        let local = b[0].1 / 8192;
        let ratio = if line.contains("_r2_") { 2 } else { 1 };
        let origin = if skip && ratio == 1 { 5696 } else { 0 };
        require(n > 0 && b[0].1.is_multiple_of(8192), "query row stride")?;
        require(
            b[1].1 == local * 128 && b[1].2 == n * 128,
            "weight row slice",
        )?;
        let actual = uploaded(host, b[6])?;
        let positions: Vec<_> = actual
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        require(
            positions
                == (origin + local..origin + local + n)
                    .map(|p| p as i32)
                    .collect::<Vec<_>>(),
            "wrong original positions",
        )?;
        let width: u64 = line
            .split("ScalarU32(")
            .nth(1)
            .unwrap()
            .split(')')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        require(width == 8192 / ratio, "wrong score stride")?;
        require(n * width * 4 <= b[8].2, "score allocation exceeded")?;
        let count = uploaded(host, b[7])?;
        require(
            count == u32::try_from(n).unwrap().to_le_bytes(),
            "wrong immutable batch count",
        )?;
        let prefix = u64::from(crate::buffers::blocks(64) + crate::buffers::blocks(4096)) * 4;
        require(
            b[7].1 == prefix + n * 4 && b[7].2 == 4,
            "count aliases the mutable prefix",
        )?;
        let partial = journal.get(at + 1).ok_or("missing partial selector")?;
        let merge = journal.get(at + 2).ok_or("missing merge selector")?;
        require(
            partial.contains("_all_select_v3_live_v1 ")
                && merge.contains("_all_select_v3_live_merge_v1 "),
            "selection ordering",
        )?;
        for select in [partial, merge] {
            let s = spans(select);
            require(
                s[0] == b[8] && s[3] == b[6] && s[4] == b[7],
                "selector changed scores, positions or counts",
            )?;
            require(
                s[5].1 % (8192 * 512 * 4) == (origin + local) * 512 * 4 && s[5].2 == n * 512 * 4,
                "selected row origin",
            )?;
            require(
                s[6].1 % (8192 * 4) == (origin + local) * 4 && s[6].2 == n * 4,
                "status row origin",
            )?;
        }
        if ratio == 1 {
            let gather = journal.get(at + 5).ok_or("missing candidate merge")?;
            require(
                gather.contains("chop_candidate_block_ids_") && gather.contains("merge"),
                "candidate publication ordering",
            )?;
            let c = spans(gather);
            let tags = uploaded(host, c[2])?;
            let original: Vec<_> = tags
                .chunks_exact(8)
                .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as u32)
                .collect();
            require(
                original
                    == (origin + local..origin + local + n)
                        .map(|p| p as u32)
                        .collect::<Vec<_>>(),
                "candidate tag origin",
            )?;
            for (span, stride) in [(c[5], 8192), (c[6], 8), (c[7], 4), (c[8], 4)] {
                require(
                    span.1 == (origin + local) * stride && span.2 == n * stride,
                    "candidate output origin",
                )?;
            }
        }
        checked += 1;
    }
    require(
        checked == host.shape.ranks() as usize * if skip { 11 } else { 22 },
        "missing batched calls",
    )?;
    Ok(checked)
}
