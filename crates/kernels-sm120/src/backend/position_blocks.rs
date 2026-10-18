//! Compact candidate preparation and launch validation. Physical key extent
//! and compact score extent are separate, including at a 1M key capacity.

use super::own::{buffer, launch};
use super::select::{Candidate, choose};
use crate::pooled_indexer::position_blocks::{CandidateEntry, Entry};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, Refusal, Validate};
use kernel_core::pooled_indexer::CandidateBlocksQuery;
use kernel_core::position_blocks::{
    CandidateBlockIds, CandidateBlockIdsBuffers, PositionBlockIndexerQuery,
    PositionBlockSelectBuffers,
};

fn capacity(rows: u32, bound: u32) -> Result<(), Refusal> {
    if rows == 0 || rows > bound {
        return Err(Refusal("compact position row capacity exceeded"));
    }
    Ok(())
}

fn distinct(outputs: &[View], inputs: &[View]) -> Result<(), Refusal> {
    for (i, out) in outputs.iter().enumerate() {
        if outputs[..i]
            .iter()
            .chain(inputs)
            .any(|v| v.handle == out.handle)
        {
            return Err(Refusal("compact position outputs require distinct storage"));
        }
    }
    Ok(())
}

fn record(b: &CandidateBlockIds<View>, rows: u64, kept: u32, width: u32) -> Result<(), Refusal> {
    if b.position_capacity != width {
        return Err(Refusal("compact position physical capacity changed"));
    }
    distinct(&[b.ids, b.row_tags, b.visible_positions, b.status], &[])?;
    b.ids.covers(rows * u64::from(kept) * 4)?;
    b.row_tags.covers(rows * 8)?;
    b.visible_positions.covers(rows * 4)?;
    b.status.covers(rows * 4)
}

pub(super) fn select(
    entries: &[Entry],
    query: &PositionBlockIndexerQuery,
    top_k: u32,
    width: u32,
    need: Need,
) -> Result<Prepared<PositionBlockSelectBuffers<View>>, Refused> {
    query.validate()?;
    capacity(need.capacity, query.indexer.capacity)?;
    let chosen = choose(
        "compact position indexer",
        entries
            .iter()
            .filter(|entry| {
                let mut q = entry.query;
                q.indexer.capacity = query.indexer.capacity;
                q == *query
            })
            .map(|entry| {
                Candidate::new(
                    entry.clone(),
                    entry.score_symbol(),
                    format!("{:?}", entry.query),
                    entry.guarantees(),
                )
                .capacity(entry.query.indexer.capacity)
            }),
        need,
    )?;
    let entry = chosen.entry;
    super::check_plan(&entry.score_plan(need.capacity, width)?)?;
    super::check_plan(&entry.select_plan(need.capacity, width, top_k)?)?;
    let digest = kernel_build::digest(format!("{:?} {top_k} {width}", entry.query).as_bytes());
    let mut info = OpInfo::new(
        format!("sm120.position-blocks.warp128-radix8.v3#{}", &digest[..16]),
        chosen.guarantees,
        vec![(entry.score_symbol(), 0), (entry.select_symbol(), 0)],
    );
    let rows = u64::from(need.capacity);
    let q = query.indexer;
    info.bytes = vec![
        (
            "q".into(),
            rows * u64::from(q.heads) * u64::from(q.head_dim) * 2,
        ),
        ("head_weights".into(), rows * u64::from(q.heads) * 4),
        ("positions".into(), rows * 4),
        ("row_batch".into(), rows * 4),
        ("row_tags".into(), rows * 8),
        ("live_rows".into(), 4),
        (
            "candidates.ids".into(),
            rows * u64::from(query.top_k_blocks) * 4,
        ),
        ("candidates.row_tags".into(), rows * 8),
        ("candidates.visible_positions".into(), rows * 4),
        ("candidates.status".into(), rows * 4),
        ("scores".into(), rows * u64::from(query.score_slots()?) * 4),
        ("status".into(), rows * 4),
        ("output".into(), rows * u64::from(q.max_top_k) * 4),
    ];
    need.check(&info)?;
    Ok(Box::new(Select {
        entry,
        info,
        top_k,
        width,
        capacity: need.capacity,
    }))
}

pub(super) fn candidates(
    entries: &[CandidateEntry],
    query: &CandidateBlocksQuery,
    width: u32,
    need: Need,
) -> Result<Prepared<CandidateBlockIdsBuffers<View>>, Refused> {
    query.validate()?;
    capacity(need.capacity, query.capacity)?;
    let chosen = choose(
        "candidate block IDs",
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
                    format!("{:?}", entry.query),
                    entry.guarantees(),
                )
                .capacity(entry.query.capacity)
            }),
        need,
    )?;
    let entry = chosen.entry;
    super::check_plan(&entry.score_plan(need.capacity, width)?)?;
    super::check_plan(&entry.gather_plan(need.capacity, width)?)?;
    let digest = kernel_build::digest(format!("{:?} {width}", entry.query).as_bytes());
    let mut info = OpInfo::new(
        format!("sm120.candidate-block-ids.max-radix8.v3#{}", &digest[..16]),
        chosen.guarantees,
        vec![(entry.score_symbol(), 0), (entry.gather_symbol(), 0)],
    );
    info.workspace = entry.workspace_bytes(need.capacity, width)?;
    let rows = u64::from(need.capacity);
    info.bytes = vec![
        ("scores".into(), rows * u64::from(width) * 4),
        ("visible_positions".into(), rows * 4),
        ("row_tags".into(), rows * 8),
        ("source_status".into(), rows * 4),
        ("live_rows".into(), 4),
        (
            "output.ids".into(),
            rows * u64::from(query.top_k_blocks) * 4,
        ),
        ("output.row_tags".into(), rows * 8),
        ("output.visible_positions".into(), rows * 4),
        ("output.status".into(), rows * 4),
    ];
    need.check(&info)?;
    Ok(Box::new(Filter {
        entry,
        info,
        width,
        capacity: need.capacity,
    }))
}

struct Select {
    entry: Entry,
    info: OpInfo,
    top_k: u32,
    width: u32,
    capacity: u32,
}
impl PreparedOp<PositionBlockSelectBuffers<View>> for Select {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &PositionBlockSelectBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        capacity(counts.rows, self.capacity)?;
        workspace.covers(self.info.workspace)?;
        let rows = u64::from(counts.rows);
        let q = self.entry.query.indexer;
        let c = b.candidates;
        record(&c, rows, self.entry.query.top_k_blocks, self.width)?;
        if b.row_tags.handle == c.row_tags.handle {
            return Err(Refusal(
                "current compact row tags must be distinct from retained tags",
            ));
        }
        b.q.covers(rows * u64::from(q.heads) * u64::from(q.head_dim) * 2)?;
        b.head_weights.covers(rows * u64::from(q.heads) * 4)?;
        b.positions.covers(rows * 4)?;
        b.row_batch.covers(rows * 4)?;
        b.row_tags.covers(rows * 8)?;
        b.live_rows.covers(4)?;
        b.scores
            .covers(rows * u64::from(self.entry.query.score_slots()?) * 4)?;
        b.status.covers(rows * 4)?;
        b.output.covers(rows * u64::from(q.max_top_k) * 4)?;
        super::position_indexer::pages(&b.keys, q, self.width)?;
        distinct(
            &[b.scores, b.status, b.output],
            &[
                b.q,
                b.head_weights,
                b.keys.codes,
                b.keys.scales,
                b.keys.page_tables.tables,
                b.positions,
                b.row_batch,
                b.row_tags,
                b.live_rows,
                c.ids,
                c.row_tags,
                c.visible_positions,
                c.status,
            ],
        )?;
        let common = vec![
            buffer(b.keys.page_tables.tables),
            buffer(b.row_batch),
            buffer(b.positions),
            buffer(b.live_rows),
            buffer(c.ids),
            buffer(b.row_tags),
            buffer(c.row_tags),
            buffer(c.visible_positions),
            buffer(c.status),
        ];
        let mut scores = vec![
            buffer(b.q),
            buffer(b.head_weights),
            buffer(b.keys.codes),
            buffer(b.keys.scales),
        ];
        scores.extend(common.clone());
        scores.extend([
            buffer(b.scores),
            Argument::U32(self.width),
            Argument::U32(b.keys.page_tables.max_pages),
            Argument::U64(b.keys.code_page_stride),
            Argument::U64(b.keys.scale_page_stride),
        ]);
        let mut select = vec![buffer(b.scores)];
        select.extend(common);
        select.extend([
            buffer(b.output),
            buffer(b.status),
            Argument::U32(self.width),
            Argument::U32(b.keys.page_tables.max_pages),
            Argument::U32(self.top_k),
        ]);
        Ok(vec![
            launch(
                &self.entry.score_symbol(),
                counts.rows,
                self.entry.score_plan(counts.rows, self.width)?,
                scores,
            ),
            launch(
                &self.entry.select_symbol(),
                counts.rows,
                self.entry
                    .select_plan(counts.rows, self.width, self.top_k)?,
                select,
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
impl PreparedOp<CandidateBlockIdsBuffers<View>> for Filter {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &CandidateBlockIdsBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        capacity(counts.rows, self.capacity)?;
        workspace.covers(self.info.workspace)?;
        let rows = u64::from(counts.rows);
        let out = b.output;
        record(&out, rows, self.entry.query.top_k_blocks, self.width)?;
        b.scores.covers(rows * u64::from(self.width) * 4)?;
        b.visible_positions.covers(rows * 4)?;
        b.row_tags.covers(rows * 8)?;
        b.source_status.covers(rows * 4)?;
        b.live_rows.covers(4)?;
        distinct(
            &[
                workspace,
                out.ids,
                out.row_tags,
                out.visible_positions,
                out.status,
            ],
            &[
                b.scores,
                b.visible_positions,
                b.row_tags,
                b.source_status,
                b.live_rows,
            ],
        )?;
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
                &self.entry.gather_symbol(),
                counts.rows,
                self.entry.gather_plan(counts.rows, self.width)?,
                vec![
                    buffer(workspace),
                    buffer(b.live_rows),
                    buffer(b.row_tags),
                    buffer(b.visible_positions),
                    buffer(b.source_status),
                    buffer(out.ids),
                    buffer(out.row_tags),
                    buffer(out.visible_positions),
                    buffer(out.status),
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
    use kernel_core::attention::BlockTables;
    use kernel_core::backend::Backend;
    use kernel_core::pooled_indexer::{IndexerCandidates, PositionIndexerQuery, PositionKeyPages};

    fn query() -> PositionBlockIndexerQuery {
        PositionBlockIndexerQuery {
            indexer: PositionIndexerQuery {
                heads: 32,
                head_dim: 128,
                compression_ratio: 1,
                tail_rope_dims: 64,
                key_norm_eps_bits: 1e-20_f32.to_bits(),
                max_top_k: 512,
                page_size: 64,
                candidates: IndexerCandidates::All,
                capacity: 8192,
            },
            block_size: 8,
            top_k_blocks: 2048,
        }
    }
    fn view(handle: u64, bytes: u64) -> View {
        View { handle, bytes }
    }
    fn retained(rows: u64) -> CandidateBlockIds<View> {
        CandidateBlockIds {
            ids: view(1, rows * 8192),
            row_tags: view(2, rows * 8),
            visible_positions: view(3, rows * 4),
            status: view(4, rows * 4),
            position_capacity: 1 << 20,
        }
    }
    fn backend() -> Sm120 {
        Sm120::new()
            .with_position_block_indexer([Entry {
                name: "checked".into(),
                query: query(),
            }])
            .with_indexer_candidate_block_ids([CandidateEntry {
                name: "checked".into(),
                query: query().filter(),
            }])
    }
    #[test]
    fn compact_launches_use_1m_key_extent_and_4096_row_score_blocks() {
        let q = query();
        let backend = backend();
        let op = backend
            .position_block_indexer(&q, 512, 1 << 20, Need::rows(4096))
            .unwrap();
        assert_eq!(op.info().workspace, 0);
        assert_eq!(
            op.info()
                .bytes
                .iter()
                .find(|(n, _)| n == "scores")
                .unwrap()
                .1,
            256 << 20
        );
        assert_eq!(retained(8192).ids.bytes, 64 << 20);
        assert_eq!(retained(131072).ids.bytes, 1 << 30);
        let rows = 4096;
        let b = PositionBlockSelectBuffers {
            q: view(5, rows * 32 * 128 * 2),
            head_weights: view(6, rows * 32 * 4),
            keys: PositionKeyPages {
                codes: view(7, (1 << 32) + 4096),
                scales: view(8, (1 << 32) + 256),
                pages: 2,
                code_page_stride: 1 << 32,
                scale_page_stride: 1 << 32,
                page_tables: BlockTables {
                    tables: view(9, 16384 * 4),
                    sequences: 1,
                    max_pages: 16384,
                },
            },
            positions: view(10, rows * 4),
            row_batch: view(11, rows * 4),
            row_tags: view(12, rows * 8),
            live_rows: view(13, 4),
            candidates: retained(rows),
            scores: view(14, 256 << 20),
            status: view(15, rows * 4),
            output: view(16, rows * 512 * 4),
        };
        let launches = op
            .launches(&b, View::EMPTY, Counts::rows(rows as u32))
            .unwrap();
        assert_eq!(launches.len(), 2);
        for broken in [
            PositionBlockSelectBuffers {
                scores: view(14, (256 << 20) - 1),
                ..b
            },
            PositionBlockSelectBuffers {
                row_tags: view(12, rows * 8 - 1),
                ..b
            },
            PositionBlockSelectBuffers {
                candidates: CandidateBlockIds {
                    ids: view(1, rows * 8192 - 1),
                    ..b.candidates
                },
                ..b
            },
            PositionBlockSelectBuffers {
                candidates: CandidateBlockIds {
                    position_capacity: 16384,
                    ..b.candidates
                },
                ..b
            },
            PositionBlockSelectBuffers {
                status: b.candidates.status,
                ..b
            },
            PositionBlockSelectBuffers {
                row_tags: b.candidates.row_tags,
                ..b
            },
            PositionBlockSelectBuffers {
                candidates: CandidateBlockIds {
                    status: b.candidates.visible_positions,
                    ..b.candidates
                },
                ..b
            },
            PositionBlockSelectBuffers {
                keys: PositionKeyPages {
                    page_tables: BlockTables {
                        max_pages: 256,
                        ..b.keys.page_tables
                    },
                    ..b.keys
                },
                ..b
            },
            PositionBlockSelectBuffers {
                keys: PositionKeyPages {
                    code_page_stride: u64::MAX,
                    ..b.keys
                },
                ..b
            },
        ] {
            assert!(
                op.launches(&broken, View::EMPTY, Counts::rows(rows as u32))
                    .is_err()
            );
        }
        assert!(op.launches(&b, View::EMPTY, Counts::rows(4097)).is_err());
    }
    #[test]
    fn full_filter_is_bounded_to_64_rows_at_1m_capacity() {
        let backend = backend();
        let op = backend
            .indexer_candidate_block_ids(&query().filter(), 1 << 20, Need::rows(64))
            .unwrap();
        assert_eq!(op.info().workspace, 32 << 20);
        let b = CandidateBlockIdsBuffers {
            scores: view(5, 256 << 20),
            visible_positions: view(6, 256),
            row_tags: view(7, 512),
            source_status: view(8, 256),
            live_rows: view(9, 4),
            output: retained(64),
        };
        let workspace = view(10, 32 << 20);
        assert_eq!(
            op.launches(&b, workspace, Counts::rows(64)).unwrap().len(),
            2
        );
        assert!(
            op.launches(&b, view(10, (32 << 20) - 1), Counts::rows(64))
                .is_err()
        );
        assert!(
            op.launches(
                &CandidateBlockIdsBuffers {
                    output: CandidateBlockIds {
                        row_tags: b.row_tags,
                        ..b.output
                    },
                    ..b
                },
                workspace,
                Counts::rows(64)
            )
            .is_err()
        );
        assert!(
            op.launches(
                &CandidateBlockIdsBuffers {
                    source_status: view(8, 255),
                    ..b
                },
                workspace,
                Counts::rows(64)
            )
            .is_err()
        );
    }
}
