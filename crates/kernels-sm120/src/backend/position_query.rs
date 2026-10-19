//! F32 query rotation followed by the backend's shared trained quantizer.

use super::own::{buffer, launch};
use super::select::{Candidate, choose};
use crate::pooled_indexer::position_query::Entry;
use crate::pooled_indexer::position_query_parallel as parallel;
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::{Argument, Refusal, Validate};
use kernel_core::pooled_indexer::{PositionIndexerQuery, PositionQueryBuffers};
use kernel_core::precision::quantizer::BlockQuantizeBuffers;

pub(super) fn prepare(
    entries: &[Entry],
    backend: &dyn Backend,
    query: &PositionIndexerQuery,
    need: Need,
    parallel_queries: bool,
) -> Result<Prepared<PositionQueryBuffers<View>>, Refused> {
    query.validate()?;
    if need.capacity == 0 || need.capacity > query.capacity {
        return Err(Refusal("position query preparation capacity exceeded").into());
    }
    let chosen = choose(
        "position query",
        entries
            .iter()
            .filter(|entry| entry.matches(query))
            .map(|entry| {
                Candidate::new(
                    entry.clone(),
                    entry.symbol(),
                    format!("{:?}", entry.spec()),
                    entry.guarantees(),
                )
                .capacity(entry.capacity)
            }),
        need,
    )?;
    let entry = chosen.entry;
    let parallel = parallel_queries && parallel::supports(&entry);
    super::check_plan(&entry.plan(need.capacity)?)?;
    let quantizer = backend.block_quantize(&query.query_quantization()?, need)?;
    if quantizer.info().workspace != 0 {
        return Err(
            Refusal("position query composition needs a workspace-free row quantizer").into(),
        );
    }
    let mut symbols = vec![(entry.symbol(), 0)];
    symbols.extend(quantizer.info().symbols.iter().cloned());
    if parallel {
        if need.capacity <= parallel::MAX_ROWS {
            symbols.clear();
        }
        symbols.push((parallel::symbol(&entry), 0));
    }
    let identity = format!("{:?} {}", entry, quantizer.info().implementation);
    let identity = if parallel {
        format!("{identity} parallel-v1")
    } else {
        identity
    };
    let digest = kernel_build::digest(identity.as_bytes());
    let mut info = OpInfo::new(
        format!(
            "sm120.position-query.f32-tail-rope-mxfp4.v3#{}",
            &digest[..16]
        ),
        chosen.guarantees.meet(quantizer.info().guarantees),
        symbols,
    );
    info.workspace = u64::from(need.capacity) * entry.row_bytes();
    info.bytes = vec![
        ("q".into(), info.workspace),
        ("positions".into(), u64::from(need.capacity) * 4),
        ("live_rows".into(), 4),
        ("quantizer_status".into(), u64::from(need.capacity) * 4),
        ("output".into(), info.workspace / 2),
    ];
    need.check(&info)?;
    Ok(Box::new(Query {
        entry,
        quantizer,
        info,
        capacity: need.capacity,
        parallel,
    }))
}

struct Query {
    entry: Entry,
    quantizer: Prepared<BlockQuantizeBuffers<View>>,
    info: OpInfo,
    capacity: u32,
    parallel: bool,
}

impl PreparedOp<PositionQueryBuffers<View>> for Query {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        b: &PositionQueryBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows == 0
            || counts.rows > self.capacity
            || counts.positions == 0
            || counts.positions > i32::MAX as u32
        {
            return Err(Refusal(
                "position query row or rotary table extent is invalid",
            ));
        }
        let bytes = u64::from(counts.rows) * self.entry.row_bytes();
        b.q.covers(bytes)?;
        b.output.covers(bytes / 2)?;
        b.positions.covers(u64::from(counts.rows) * 4)?;
        b.live_rows.covers(4)?;
        b.quantizer_status
            .covers(kernel_core::precision::quantizer::status_extent(
                b.quantizer_status_row_offset,
                counts.rows,
            )?)?;
        if [
            b.q.handle,
            b.cos_sin.handle,
            b.positions.handle,
            b.live_rows.handle,
            b.output.handle,
            workspace.handle,
        ]
        .contains(&b.quantizer_status.handle)
        {
            return Err(Refusal("position query status must not alias another role"));
        }
        b.cos_sin
            .covers(u64::from(counts.positions) * u64::from(self.entry.rotary_dim) * 4)?;
        workspace.covers(self.info.workspace)?;
        let inputs = [
            b.q.handle,
            b.cos_sin.handle,
            b.positions.handle,
            b.live_rows.handle,
        ];
        if inputs.contains(&workspace.handle)
            || inputs.contains(&b.output.handle)
            || workspace.handle == b.output.handle
        {
            return Err(Refusal(
                "position query inputs, output and workspace must not overlap",
            ));
        }
        if self.parallel && counts.rows <= parallel::MAX_ROWS {
            let mut plan = self.entry.plan(counts.rows)?;
            plan.grid = [counts.rows, 1, 1];
            plan.block = [parallel::THREADS, 1, 1];
            return Ok(vec![launch(
                &parallel::symbol(&self.entry),
                counts.rows,
                plan,
                vec![
                    buffer(b.q),
                    buffer(b.cos_sin),
                    buffer(b.positions),
                    buffer(b.live_rows),
                    buffer(b.output),
                    buffer(b.quantizer_status),
                    Argument::U64(b.quantizer_status_row_offset),
                    Argument::U32(counts.positions),
                ],
            )]);
        }
        let mut launches = vec![launch(
            &self.entry.symbol(),
            counts.rows,
            self.entry.plan(counts.rows)?,
            vec![
                buffer(b.q),
                buffer(b.cos_sin),
                buffer(b.positions),
                buffer(b.live_rows),
                buffer(workspace),
                buffer(b.quantizer_status),
                Argument::U64(b.quantizer_status_row_offset),
                Argument::U32(counts.positions),
            ],
        )];
        launches.extend(self.quantizer.launches(
            &BlockQuantizeBuffers {
                input: workspace,
                output: b.output,
                scales: None,
                live_rows: b.live_rows,
                status: b.quantizer_status,
                status_row_offset: b.quantizer_status_row_offset,
            },
            View::EMPTY,
            counts,
        )?);
        Ok(launches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Sm120;
    use kernel_core::pooled_indexer::IndexerCandidates;
    use kernel_core::properties::Invariance;

    fn query() -> PositionIndexerQuery {
        PositionIndexerQuery {
            heads: 3,
            head_dim: 160,
            compression_ratio: 2,
            tail_rope_dims: 64,
            key_norm_eps_bits: 1e-20_f32.to_bits(),
            max_top_k: 7,
            page_size: 64,
            candidates: IndexerCandidates::All,
            capacity: 4,
        }
    }

    fn backend() -> Sm120 {
        let q = query();
        Sm120::new()
            .with_position_queries([Entry::new("query", q).unwrap()])
            .with_block_quantize(
                crate::row::block_quantize::Entry::new(
                    q.query_quantization().unwrap(),
                    4,
                    "chop_query_quantize_v2",
                )
                .unwrap(),
            )
    }

    fn buffers() -> (PositionQueryBuffers<View>, View, Counts) {
        let view = |handle, bytes| View { handle, bytes };
        (
            PositionQueryBuffers {
                quantizer_status_row_offset: 0,
                q: view(1, 4 * 3 * 160 * 4),
                cos_sin: view(2, 8 * 64 * 4),
                positions: view(3, 16),
                live_rows: view(4, 4),
                output: view(5, 4 * 3 * 160 * 2),
                quantizer_status: view(7, 16),
            },
            view(6, 4 * 3 * 160 * 4),
            Counts {
                positions: 8,
                ..Counts::rows(4)
            },
        )
    }

    #[test]
    fn position_query_requires_the_shared_quantizer_and_preserves_its_launch() {
        let q = query();
        let missing = Sm120::new().with_position_queries([Entry::new("query", q).unwrap()]);
        assert!(missing.position_indexer_query(&q, Need::rows(4)).is_err());
        let backend = backend();
        let need = Need::rows(4).with_exact(Invariance::any_rows(4));
        let op = backend.position_indexer_query(&q, need).unwrap();
        assert_eq!(op.info().workspace, 4 * 3 * 160 * 4);
        let (b, workspace, counts) = buffers();
        let launches = op.launches(&b, workspace, counts).unwrap();
        assert_eq!(launches.len(), 2);
        assert_eq!(
            &launches[0].arguments.0[5..],
            &[
                buffer(b.quantizer_status),
                Argument::U64(0),
                Argument::U32(counts.positions)
            ]
        );

        assert_eq!(launches[0].symbol, "chop_position_query_query_rotate_v3");
        let quantize = backend
            .block_quantize(&q.query_quantization().unwrap(), need)
            .unwrap();
        let shared = quantize
            .launches(
                &BlockQuantizeBuffers {
                    input: workspace,
                    output: b.output,
                    scales: None,
                    live_rows: b.live_rows,
                    status: b.quantizer_status,
                    status_row_offset: b.quantizer_status_row_offset,
                },
                View::EMPTY,
                counts,
            )
            .unwrap();
        assert_eq!(launches[1].symbol, shared[0].symbol);
        assert_eq!(launches[1].arguments, shared[0].arguments);
        assert_eq!(launches[1].plan, shared[0].plan);
        assert!(op.info().guarantees.provides(need.exact));
    }

    #[test]
    fn position_query_refuses_short_f32_inputs_tables_workspace_and_aliases() {
        let op = backend()
            .position_indexer_query(&query(), Need::rows(4))
            .unwrap();
        let (b, workspace, counts) = buffers();
        let mut shifted = b;
        shifted.quantizer_status_row_offset = (1_u64 << 32) + 9;
        shifted.quantizer_status.bytes = (shifted.quantizer_status_row_offset + 4) * 4;
        let launches = op.launches(&shifted, workspace, counts).unwrap();
        assert_eq!(
            launches[1].arguments.0.last(),
            Some(&kernel_core::contract::Argument::U64(
                shifted.quantizer_status_row_offset
            ))
        );
        assert_eq!(
            launches[0].arguments.0[6],
            Argument::U64(shifted.quantizer_status_row_offset)
        );
        shifted.quantizer_status.bytes -= 1;
        assert!(op.launches(&shifted, workspace, counts).is_err());
        shifted.quantizer_status_row_offset = u64::MAX;
        assert!(op.launches(&shifted, workspace, counts).is_err());
        for alias in [b.q, b.output, b.positions, b.live_rows, workspace] {
            assert!(
                op.launches(
                    &PositionQueryBuffers {
                        quantizer_status: alias,
                        ..b
                    },
                    workspace,
                    counts
                )
                .is_err()
            );
        }
        for bad in [
            PositionQueryBuffers {
                q: View {
                    bytes: b.q.bytes / 2,
                    ..b.q
                },
                ..b
            },
            PositionQueryBuffers {
                cos_sin: View {
                    bytes: b.cos_sin.bytes - 1,
                    ..b.cos_sin
                },
                ..b
            },
            PositionQueryBuffers { output: b.q, ..b },
            PositionQueryBuffers {
                output: workspace,
                ..b
            },
        ] {
            assert!(op.launches(&bad, workspace, counts).is_err());
        }
        assert!(
            op.launches(
                &b,
                View {
                    bytes: workspace.bytes - 1,
                    ..workspace
                },
                counts
            )
            .is_err()
        );
        assert!(op.launches(&b, b.q, counts).is_err());
        for bad in [
            Counts::rows(4),
            Counts { rows: 5, ..counts },
            Counts { rows: 0, ..counts },
        ] {
            assert!(op.launches(&b, workspace, bad).is_err());
        }
    }
}
