//! Prepared position scoring and exact selection. Registration uses the
//! ordinary capacity/exactness chooser; no family-specific ordering rule.

use super::own::{buffer, launch};
use super::select::{Candidate, choose};
use crate::pooled_indexer::position::{CandidateEntry, Entry};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, Refusal, Validate};
use kernel_core::pooled_indexer::IndexerKeyFormat;
use kernel_core::pooled_indexer::{
    CandidateBlocksBuffers, CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery,
    PositionKeyPages, PositionSelectBuffers,
};

pub(super) fn select(
    entries: &[Entry],
    query: &PositionIndexerQuery,
    top_k: u32,
    width: u32,
    need: Need,
) -> Result<Prepared<PositionSelectBuffers<View>>, Refused> {
    select_with_storage(entries, query, top_k, width, need, IndexerKeyFormat::Fp4)
}

pub(super) fn select_with_storage(
    entries: &[Entry],
    query: &PositionIndexerQuery,
    top_k: u32,
    width: u32,
    need: Need,
    storage: IndexerKeyFormat,
) -> Result<Prepared<PositionSelectBuffers<View>>, Refused> {
    query.validate()?;
    capacity(need.capacity, query.capacity)?;
    let chosen = choose(
        "position indexer",
        entries
            .iter()
            .filter(|entry| {
                PositionIndexerQuery {
                    capacity: query.capacity,
                    ..entry.query
                } == *query
            })
            .map(|entry| {
                Candidate::new(
                    entry.clone(),
                    entry.score_symbol(),
                    entry_key(entry),
                    entry.guarantees(),
                )
                .capacity(entry.query.capacity)
            }),
        need,
    )?;
    let entry = chosen.entry;
    super::check_plan(&entry.score_plan(need.capacity, width)?)?;
    super::check_plan(&entry.select_plan(need.capacity, width, top_k)?)?;
    let digest = kernel_build::digest(format!("{} {top_k} {width}", entry_key(&entry)).as_bytes());
    let mut info = OpInfo::new(
        format!(
            "sm120.position-indexer.f32-weights-warp128-radix8.v3#{}",
            &digest[..16]
        ),
        chosen.guarantees,
        vec![(entry.score_symbol(), 0), (entry.select_symbol(), 0)],
    );
    let rows = u64::from(need.capacity);
    info.bytes = vec![
        (
            "q".into(),
            rows * u64::from(query.heads) * u64::from(query.head_dim) * 2,
        ),
        ("head_weights".into(), rows * u64::from(query.heads) * 4),
        ("status".into(), rows * 4),
        ("positions".into(), rows * 4),
        ("row_batch".into(), rows * 4),
        ("live_rows".into(), 4),
        ("scores".into(), rows * u64::from(width) * 4),
        ("output".into(), rows * u64::from(query.max_top_k) * 4),
    ];
    if query.candidates == IndexerCandidates::Mask {
        info.bytes
            .push(("candidate_mask".into(), rows * u64::from(width)));
    }
    need.check(&info)?;
    Ok(Box::new(Select {
        storage,
        entry,
        info,
        top_k,
        width,
        capacity: need.capacity,
    }))
}

fn entry_key(entry: &Entry) -> String {
    // Name launches; query geometry selects, and capacity only bounds rows.
    let Entry { name, query } = entry;
    format!(
        "{name} {:?}",
        PositionIndexerQuery {
            capacity: 0,
            ..*query
        }
    )
}

pub(super) fn candidates(
    entries: &[CandidateEntry],
    query: &CandidateBlocksQuery,
    width: u32,
    need: Need,
) -> Result<Prepared<CandidateBlocksBuffers<View>>, Refused> {
    query.validate()?;
    capacity(need.capacity, query.capacity)?;
    let chosen = choose(
        "indexer candidate blocks",
        entries
            .iter()
            .filter(|entry| {
                CandidateBlocksQuery {
                    capacity: query.capacity,
                    ..entry.query
                } == *query
            })
            .map(|entry| {
                Candidate::new(
                    entry.clone(),
                    entry.score_symbol(),
                    candidate_key(entry),
                    entry.guarantees(),
                )
                .capacity(entry.query.capacity)
            }),
        need,
    )?;
    let entry = chosen.entry;
    super::check_plan(&entry.score_plan(need.capacity, width)?)?;
    super::check_plan(&entry.mask_plan(need.capacity, width)?)?;
    let digest = kernel_build::digest(format!("{} {width}", candidate_key(&entry)).as_bytes());
    let mut info = OpInfo::new(
        format!("sm120.candidate-blocks.max-radix8.v1#{}", &digest[..16]),
        chosen.guarantees,
        vec![(entry.score_symbol(), 0), (entry.mask_symbol(), 0)],
    );
    info.workspace = entry.workspace_bytes(need.capacity, width)?;
    let rows = u64::from(need.capacity);
    info.bytes = vec![
        ("scores".into(), rows * u64::from(width) * 4),
        ("visible_positions".into(), rows * 4),
        ("live_rows".into(), 4),
        ("output".into(), rows * u64::from(width)),
    ];
    need.check(&info)?;
    Ok(Box::new(Filter {
        entry,
        info,
        width,
        capacity: need.capacity,
    }))
}

fn candidate_key(entry: &CandidateEntry) -> String {
    let CandidateEntry { name, query } = entry;
    format!(
        "{name} {:?}",
        CandidateBlocksQuery {
            capacity: 0,
            ..*query
        }
    )
}

fn capacity(rows: u32, capacity: u32) -> Result<(), Refusal> {
    if rows == 0 || rows > capacity {
        return Err(Refusal("position indexer capacity exceeded"));
    }
    Ok(())
}

pub(super) fn pages(
    keys: &PositionKeyPages<View>,
    query: PositionIndexerQuery,
    width: u32,
) -> Result<(), Refusal> {
    keys.page_tables.check()?;
    let (code_bytes, scale_bytes) = query.page_bytes();
    if keys.pages == 0 || keys.page_tables.max_pages < width.div_ceil(query.page_size) {
        return Err(Refusal(
            "position indexer key pages do not cover the position capacity",
        ));
    }
    for (view, stride, bytes) in [
        (keys.codes, keys.code_page_stride, code_bytes),
        (keys.scales, keys.scale_page_stride, scale_bytes),
    ] {
        if stride < bytes {
            return Err(Refusal("position indexer page stride is below its plane"));
        }
        if stride % 4 != 0 {
            return Err(Refusal(
                "position indexer page stride is not 4-byte aligned",
            ));
        }
        let end = u64::from(keys.pages - 1)
            .checked_mul(stride)
            .and_then(|start| start.checked_add(bytes))
            .ok_or(Refusal("position indexer page address overflow"))?;
        view.covers(end)?;
    }
    Ok(())
}

pub(super) fn pages_with_storage(
    keys: &PositionKeyPages<View>,
    query: PositionIndexerQuery,
    width: u32,
    storage: IndexerKeyFormat,
) -> Result<(), Refusal> {
    if storage == IndexerKeyFormat::Fp4 {
        return pages(keys, query, width);
    }
    keys.page_tables.check()?;
    if keys.pages == 0 || keys.page_tables.max_pages < width.div_ceil(query.page_size) {
        return Err(Refusal(
            "position indexer key pages do not cover the position capacity",
        ));
    }
    if keys.scales.bytes != 0 || keys.scale_page_stride != 0 {
        return Err(Refusal("BF16 position keys have no scale plane"));
    }
    let (bytes, _) = query.storage_page_bytes(storage)?;
    let stride = keys.code_page_stride;
    if stride < bytes || stride % 2 != 0 {
        return Err(Refusal(
            "BF16 position key page stride is short or unaligned",
        ));
    }
    let end = u64::from(keys.pages - 1)
        .checked_mul(stride)
        .and_then(|start| start.checked_add(bytes))
        .ok_or(Refusal("position indexer page address overflow"))?;
    keys.codes.covers(end)
}

struct Select {
    storage: IndexerKeyFormat,
    entry: Entry,
    info: OpInfo,
    top_k: u32,
    width: u32,
    capacity: u32,
}

impl PreparedOp<PositionSelectBuffers<View>> for Select {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &PositionSelectBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        capacity(counts.rows, self.capacity)?;
        workspace.covers(self.info.workspace)?;
        if b.position_capacity != self.width {
            return Err(Refusal("position indexer score width changed"));
        }
        let q = self.entry.query;
        let rows = u64::from(counts.rows);
        b.q.covers(rows * u64::from(q.heads) * u64::from(q.head_dim) * 2)?;
        b.head_weights.covers(rows * u64::from(q.heads) * 4)?;
        b.positions.covers(rows * 4)?;
        b.row_batch.covers(rows * 4)?;
        b.live_rows.covers(4)?;
        b.scores.covers(rows * u64::from(self.width) * 4)?;
        b.status.covers(rows * 4)?;
        if [
            b.q.handle,
            b.head_weights.handle,
            b.keys.codes.handle,
            b.keys.scales.handle,
            b.keys.page_tables.tables.handle,
            b.positions.handle,
            b.row_batch.handle,
            b.live_rows.handle,
            b.scores.handle,
            b.output.handle,
        ]
        .contains(&b.status.handle)
            || b.candidate_mask
                .is_some_and(|v| v.handle == b.status.handle)
        {
            return Err(Refusal(
                "position indexer status must have distinct storage",
            ));
        }
        b.output.covers(rows * u64::from(q.max_top_k) * 4)?;
        pages_with_storage(&b.keys, q, self.width, self.storage)?;
        let mask = match (q.candidates, b.candidate_mask) {
            (IndexerCandidates::All, None) => Argument::NullDevicePointer,
            (IndexerCandidates::Mask, Some(mask)) => {
                mask.covers(rows * u64::from(self.width))?;
                buffer(mask)
            }
            _ => return Err(Refusal("position indexer candidate mask mismatch")),
        };
        Ok(vec![
            launch(
                &self.entry.score_symbol(),
                counts.rows,
                self.entry.score_plan(counts.rows, self.width)?,
                vec![
                    buffer(b.q),
                    buffer(b.head_weights),
                    buffer(b.keys.codes),
                    if self.storage == IndexerKeyFormat::Fp4 {
                        buffer(b.keys.scales)
                    } else {
                        Argument::NullDevicePointer
                    },
                    buffer(b.keys.page_tables.tables),
                    buffer(b.row_batch),
                    buffer(b.positions),
                    buffer(b.live_rows),
                    mask.clone(),
                    buffer(b.scores),
                    Argument::U32(self.width),
                    Argument::U32(b.keys.page_tables.max_pages),
                    Argument::U64(b.keys.code_page_stride),
                    Argument::U64(b.keys.scale_page_stride),
                ],
            ),
            launch(
                &self.entry.select_symbol(),
                counts.rows,
                self.entry
                    .select_plan(counts.rows, self.width, self.top_k)?,
                vec![
                    buffer(b.scores),
                    buffer(b.keys.page_tables.tables),
                    buffer(b.row_batch),
                    buffer(b.positions),
                    buffer(b.live_rows),
                    mask,
                    buffer(b.output),
                    buffer(b.status),
                    Argument::U32(self.width),
                    Argument::U32(b.keys.page_tables.max_pages),
                    Argument::U32(self.top_k),
                ],
            ),
        ])
    }
}

struct Filter {
    entry: CandidateEntry,
    info: OpInfo,
    width: u32,
    capacity: u32,
}

impl PreparedOp<CandidateBlocksBuffers<View>> for Filter {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &CandidateBlocksBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        capacity(counts.rows, self.capacity)?;
        workspace.covers(self.info.workspace)?;
        if b.position_capacity != self.width {
            return Err(Refusal("candidate block score width changed"));
        }
        let rows = u64::from(counts.rows);
        b.scores.covers(rows * u64::from(self.width) * 4)?;
        b.output.covers(rows * u64::from(self.width))?;
        b.visible_positions.covers(rows * 4)?;
        b.live_rows.covers(4)?;
        Ok(vec![
            launch(
                &self.entry.score_symbol(),
                counts.rows,
                self.entry.score_plan(counts.rows, self.width)?,
                vec![
                    buffer(b.scores),
                    buffer(b.visible_positions),
                    buffer(b.live_rows),
                    buffer(workspace),
                    Argument::U32(self.width),
                ],
            ),
            launch(
                &self.entry.mask_symbol(),
                counts.rows,
                self.entry.mask_plan(counts.rows, self.width)?,
                vec![
                    buffer(workspace),
                    buffer(b.live_rows),
                    buffer(b.output),
                    Argument::U32(self.width),
                ],
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Sm120;
    use kernel_core::{attention::BlockTables, backend::Backend};

    fn entry(name: &str, capacity: u32) -> Entry {
        Entry {
            name: name.into(),
            query: PositionIndexerQuery {
                heads: 3,
                head_dim: 160,
                compression_ratio: 2,
                tail_rope_dims: 64,
                key_norm_eps_bits: 1e-20_f32.to_bits(),
                max_top_k: 7,
                page_size: 64,
                candidates: IndexerCandidates::Mask,
                capacity,
            },
        }
    }

    fn buffers(query: PositionIndexerQuery, width: u32) -> PositionSelectBuffers<View> {
        let view = |handle, bytes| View { handle, bytes };
        let rows = 4;
        PositionSelectBuffers {
            q: view(
                1,
                rows * u64::from(query.heads) * u64::from(query.head_dim) * 2,
            ),
            head_weights: view(2, rows * u64::from(query.heads) * 4),
            keys: PositionKeyPages {
                codes: view(3, 3 * (1 << 32) + query.page_bytes().0),
                scales: view(4, 3 * (1 << 33) + query.page_bytes().1),
                pages: 4,
                code_page_stride: 1 << 32,
                scale_page_stride: 1 << 33,
                page_tables: BlockTables {
                    tables: view(5, 2 * 3 * 4),
                    sequences: 2,
                    max_pages: 3,
                },
            },
            positions: view(6, rows * 4),
            row_batch: view(7, rows * 4),
            live_rows: view(8, 4),
            candidate_mask: Some(view(9, rows * u64::from(width))),
            position_capacity: width,
            scores: view(10, rows * u64::from(width) * 4),
            status: view(12, rows * 4),
            output: view(11, rows * 7 * 4),
        }
    }

    #[test]
    fn position_selection_matches_all_semantics_and_uses_the_capacity_chooser() {
        let small = entry("small", 4);
        let large = entry("large", 8);
        let query = large.query;
        for entries in [
            [small.clone(), large.clone()],
            [large.clone(), small.clone()],
        ] {
            let backend = Sm120::new().with_position_indexer(entries);
            let prepared = backend
                .position_indexer(&query, 3, 129, Need::rows(4))
                .unwrap();
            assert_eq!(prepared.info().symbols[0].0, small.score_symbol());
            assert_eq!(prepared.info().bytes_of("candidate_mask"), Some(4 * 129));
            for mismatch in [
                PositionIndexerQuery {
                    compression_ratio: 1,
                    ..query
                },
                PositionIndexerQuery {
                    tail_rope_dims: 32,
                    ..query
                },
                PositionIndexerQuery {
                    key_norm_eps_bits: 1e-6_f32.to_bits(),
                    ..query
                },
                PositionIndexerQuery {
                    candidates: IndexerCandidates::All,
                    ..query
                },
            ] {
                assert!(
                    backend
                        .position_indexer(&mismatch, 3, 129, Need::rows(4))
                        .is_err()
                );
            }
        }
        let backend = Sm120::new().with_position_indexer([small.clone(), entry("other", 4)]);
        assert!(
            backend
                .position_indexer(&query, 3, 129, Need::rows(4))
                .is_err()
        );
        let backend = Sm120::new().with_position_indexer([small.clone(), small]);
        assert!(
            backend
                .position_indexer(&query, 3, 129, Need::rows(4))
                .is_ok()
        );
    }

    #[test]
    fn position_launches_keep_64_bit_planes_and_refuse_missing_masks_and_short_views() {
        let entry = entry("checked", 8);
        let query = entry.query;
        let backend = Sm120::new().with_position_indexer([entry]);
        let prepared = backend
            .position_indexer(&query, 3, 129, Need::rows(4))
            .unwrap();
        let b = buffers(query, 129);
        let launches = prepared.launches(&b, View::EMPTY, Counts::rows(4)).unwrap();
        assert_eq!(launches.len(), 2);
        assert!(launches[0].arguments.0.contains(&Argument::U64(1 << 32)));
        assert!(launches[0].arguments.0.contains(&Argument::U64(1 << 33)));
        assert!(launches.iter().all(|launch| launch.plan.pdl));
        for status in [
            View {
                bytes: 15,
                ..b.status
            },
            b.scores,
        ] {
            assert!(
                prepared
                    .launches(
                        &PositionSelectBuffers { status, ..b },
                        View::EMPTY,
                        Counts::rows(4)
                    )
                    .is_err()
            );
        }
        let legacy_weights = PositionSelectBuffers {
            head_weights: View {
                bytes: b.head_weights.bytes / 2,
                ..b.head_weights
            },
            ..b
        };
        assert!(
            prepared
                .launches(&legacy_weights, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        assert!(prepared.launches(&b, View::EMPTY, Counts::rows(5)).is_err());
        assert!(
            prepared
                .launches(
                    &PositionSelectBuffers {
                        candidate_mask: None,
                        ..b
                    },
                    View::EMPTY,
                    Counts::rows(4)
                )
                .is_err()
        );
        assert!(
            prepared
                .launches(
                    &PositionSelectBuffers {
                        position_capacity: 128,
                        ..b
                    },
                    View::EMPTY,
                    Counts::rows(4)
                )
                .is_err()
        );
        assert!(
            prepared
                .launches(
                    &PositionSelectBuffers {
                        scores: View {
                            bytes: 1,
                            ..b.scores
                        },
                        ..b
                    },
                    View::EMPTY,
                    Counts::rows(4)
                )
                .is_err()
        );
        let overflow = PositionSelectBuffers {
            keys: PositionKeyPages {
                code_page_stride: u64::MAX,
                ..b.keys
            },
            ..b
        };
        assert!(
            prepared
                .launches(&overflow, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        let short_page = PositionSelectBuffers {
            keys: PositionKeyPages {
                code_page_stride: 1,
                ..b.keys
            },
            ..b
        };
        assert!(
            prepared
                .launches(&short_page, View::EMPTY, Counts::rows(4))
                .is_err()
        );
    }

    #[test]
    fn position_fp4_pages_refuse_unaligned_strides() {
        let query = entry("aligned", 4).query;
        let mut keys = buffers(query, 129).keys;
        // Padding must cover the views, so alignment is the only refusal.
        keys.codes.bytes = u64::MAX;
        keys.scales.bytes = u64::MAX;
        for pages_count in [1, 4] {
            keys.pages = pages_count;
            for offset in 1..4 {
                for codes in [false, true] {
                    let mut bad = keys;
                    if codes {
                        bad.code_page_stride += offset;
                    } else {
                        bad.scale_page_stride += offset;
                    }
                    assert_eq!(
                        pages(&bad, query, 129),
                        Err(Refusal(
                            "position indexer page stride is not 4-byte aligned"
                        ))
                    );
                }
            }
            let mut padded = keys;
            padded.code_page_stride += 4;
            padded.scale_page_stride += 4;
            assert!(pages(&padded, query, 129).is_ok());
        }
    }

    #[test]
    fn candidate_filter_sizes_workspace_and_refuses_changed_width() {
        let query = CandidateBlocksQuery {
            block_size: 8,
            top_k_blocks: 2048,
            capacity: 8,
        };
        let backend = Sm120::new().with_indexer_candidate_blocks([CandidateEntry {
            name: "checked".into(),
            query,
        }]);
        let prepared = backend
            .indexer_candidate_blocks(&query, 16385, Need::rows(4))
            .unwrap();
        assert_eq!(prepared.info().workspace, 4 * 2049 * 4);
        let b = CandidateBlocksBuffers {
            scores: View {
                handle: 1,
                bytes: 4 * 16385 * 4,
            },
            visible_positions: View {
                handle: 2,
                bytes: 16,
            },
            live_rows: View {
                handle: 3,
                bytes: 4,
            },
            position_capacity: 16385,
            output: View {
                handle: 4,
                bytes: 4 * 16385,
            },
        };
        let workspace = View {
            handle: 5,
            bytes: prepared.info().workspace,
        };
        assert_eq!(
            prepared
                .launches(&b, workspace, Counts::rows(4))
                .unwrap()
                .len(),
            2
        );
        assert!(prepared.launches(&b, View::EMPTY, Counts::rows(4)).is_err());
        assert!(
            prepared
                .launches(
                    &CandidateBlocksBuffers {
                        position_capacity: 16384,
                        ..b
                    },
                    workspace,
                    Counts::rows(4)
                )
                .is_err()
        );
    }
    #[test]
    fn bf16_position_reader_requires_registration_and_validates_the_value_plane() {
        let e = entry("comparison", 4);
        let q = e.query;
        let need = Need::rows(4);
        let width = 1 << 20;
        let native = Sm120::new().with_position_indexer([e.clone()]);
        assert!(
            native
                .position_indexer_with_storage(&q, IndexerKeyFormat::Bf16, 7, width, need)
                .is_err()
        );
        let backend = native.with_position_indexer_bf16([e]);
        let op = backend
            .position_indexer_with_storage(&q, IndexerKeyFormat::Bf16, 7, width, need)
            .unwrap();
        let mut b = buffers(q, width);
        b.keys.scales = View {
            handle: 0,
            bytes: 0,
        };
        b.keys.scale_page_stride = 0;
        let page_bytes = q.storage_page_bytes(IndexerKeyFormat::Bf16).unwrap().0;
        b.keys.codes.bytes = 3 * (1 << 32) + page_bytes;
        b.keys.page_tables.max_pages = width.div_ceil(q.page_size);
        b.keys.page_tables.tables.bytes = u64::from(b.keys.page_tables.max_pages) * 2 * 4;
        let calls = op.launches(&b, View::default(), Counts::rows(4)).unwrap();
        assert!(calls[0].symbol.contains("_bf16_scores_v3"));
        assert_eq!(calls[0].arguments.0[3], Argument::NullDevicePointer);
        assert_eq!(calls[0].arguments.0[12], Argument::U64(1 << 32));
        let mut bad = b;
        bad.keys.codes.bytes -= 1;
        assert!(op.launches(&bad, View::default(), Counts::rows(4)).is_err());
        let mut bad = b;
        bad.keys.scales.bytes = 1;
        assert!(op.launches(&bad, View::default(), Counts::rows(4)).is_err());
        let mut bad = b;
        bad.keys.scale_page_stride = 1;
        assert!(op.launches(&bad, View::default(), Counts::rows(4)).is_err());
        let mut bad = b;
        bad.keys.code_page_stride = page_bytes - 2;
        assert!(op.launches(&bad, View::default(), Counts::rows(4)).is_err());
        let mut bad = b;
        bad.keys.code_page_stride = page_bytes + 1;
        assert!(op.launches(&bad, View::default(), Counts::rows(4)).is_err());
        let mut bad = b;
        bad.keys.code_page_stride = u64::MAX - 1;
        assert!(op.launches(&bad, View::default(), Counts::rows(4)).is_err());
        let mut bad = b;
        bad.status = bad.keys.codes;
        assert!(op.launches(&bad, View::default(), Counts::rows(4)).is_err());
        let mut bad = b;
        bad.candidate_mask = None;
        assert!(op.launches(&bad, View::default(), Counts::rows(4)).is_err());
        assert!(op.launches(&b, View::default(), Counts::rows(5)).is_err());
    }
}
