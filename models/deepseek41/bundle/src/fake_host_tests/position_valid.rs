//! Full prefill uses the same live/valid planners as decode and verify.
use super::*;

#[test]
fn qualified_prefill_tile_binds_all_four_chunk_offsets() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    assert!(shape.runtime.indexer_prefill_tile);
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.kv_pages.set(1024);
    let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 1, false);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    for offset in [0, 8192, 16384, 24576] {
        let first = host.device.journal.borrow().len();
        instance
            .execute(
                &prompt(7, 0, offset, 8192, offset == 24576),
                &mut Views::default(),
            )
            .unwrap();
        let journal = host.device.journal.borrow();
        let lines = &journal[first..];
        let scores: Vec<_> = lines
            .iter()
            .filter(|line| {
                line.starts_with("launch ")
                    && (line.contains("chop_position_indexer_")
                        || line.contains("chop_position_blocks_"))
                    && line.contains("_scores_v3")
                    && !line.contains("_pack_fp4_")
            })
            .collect();
        assert_eq!(scores.len(), 4 * (3 * 2 + 16 + 4 * 2));
        for line in scores {
            let compact = line.contains("chop_position_blocks_");
            assert!(
                line.contains(if compact {
                    "_mma_fp4_prefill_tile_v1 "
                } else {
                    "_mma_fp4_tile4_mode0_trace0_v1 "
                }),
                "{line}"
            );
            let spans = super::shared_overlap::journal_spans(line);
            let n = spans[0].2 / 8192;
            assert!(matches!(n, 512 | 4096));
            assert!(line.contains(&format!("grid_x: 8, grid_y: {n}, grid_z: 1")));
            let positions = uploaded(&host, spans[6]).unwrap();
            let positions: Vec<_> = positions
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(positions.len(), n as usize);
            assert!(
                positions
                    .iter()
                    .all(|&p| p >= offset as i32 && p < offset as i32 + 8192)
            );
            assert!(positions.windows(2).all(|p| p[1] == p[0] + 1));
        }
        if offset == 0 {
            audit_batched(&host, lines, false).unwrap();
        }
        drop(journal);
        instance.commit().unwrap();
    }
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn prefill_tile_keeps_unmeasured_batch_sizes_on_the_previous_route() {
    for rows in [32, 65, 512, 4096, 8191] {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 1, false);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap();
        instance
            .execute(&prompt(7, 0, 0, rows, true), &mut Views::default())
            .unwrap();
        let journal = host.device.journal.borrow();
        let mut checked = 0;
        for line in journal.iter().filter(|line| {
            line.starts_with("launch ")
                && (line.contains("chop_position_indexer_")
                    || line.contains("chop_position_blocks_"))
                && line.contains("_scores_v3")
                && !line.contains("_pack_fp4_")
        }) {
            let n = super::shared_overlap::journal_spans(line)[0].2 / 8192;
            let compact = line.contains("chop_position_blocks_");
            let expected = n
                == if compact || line.contains("_r2_") {
                    4096
                } else {
                    512
                };
            let tiled = line.contains("_mma_fp4_prefill_tile_v1 ")
                || line.contains("_mma_fp4_tile4_mode0_trace0_v1 ");
            assert_eq!(tiled, expected, "rows={rows}: {line}");
            assert!(line.contains(&format!(
                "grid_x: {}, grid_y: {n}, grid_z: 1",
                if expected { 8 } else { 192 }
            )));
            checked += 1;
        }
        assert!(checked >= 4 * 8);
        drop(journal);
        instance.commit().unwrap();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn prefill_indexer_blocks_use_live_and_valid_grids() {
    for kv in KV_LAYOUTS {
        let mut shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        shape.runtime.indexer_prefill_tile = false;
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
            && (line.contains("_scores_v3_live_v1_mma_fp4_guard_fast_v1 ")
                || line.contains("_scores_v3_live_v1_mma_fp4_tile4_mode0_trace0_v1 "))
    }) {
        let b = spans(line);
        let n = b[0].2 / 8192;
        let local = b[0].1 / 8192;
        let ratio = if line.contains("_r2_") { 2 } else { 1 };
        let tiled = line.contains("_mma_fp4_tile4_mode0_trace0_v1 ");
        let expected_tile =
            host.shape.runtime.indexer_prefill_tile && n == if ratio == 2 { 4096 } else { 512 };
        require(tiled == expected_tile, "wrong prefill score route")?;
        require(
            line.contains(&format!(
                "grid_x: {}, grid_y: {n}, grid_z: 1",
                if tiled { 8 } else { 192 }
            )),
            "wrong prefill score grid",
        )?;
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
            // The admitted schedule publishes with the fused producer, which
            // reads the selector's scores directly.
            let fused = host.shape.runtime.indexer_prefill_tile;
            let gather = journal
                .get(at + if fused { 3 } else { 5 })
                .ok_or("missing candidate publication")?;
            require(
                gather.contains("chop_candidate_block_ids_")
                    && if fused {
                        gather.contains("_gather_v3_cached4_v1 ")
                    } else {
                        gather.contains("merge")
                    },
                "candidate publication ordering",
            )?;
            let c = spans(gather);
            require(!fused || c[0] == b[8], "fused candidates changed scores")?;
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
