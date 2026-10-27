use super::select::{Candidate, choose};
use crate::gemm::dense_decode::{Entry, Finish};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, Refusal};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
use kernel_core::properties::Invariance;

fn prepare_op(
    entries: &[(Entry, Option<u32>)],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
    paired: bool,
) -> Result<Op, Refused> {
    let reuse = entries
        .iter()
        .any(|(e, _)| e.query == *query && !paired && e.knobs.row_groups == 2);
    if reuse
        && entries
            .iter()
            .filter(|(e, _)| e.query == *query && e.knobs.row_groups == 2)
            .count()
            != 1
    {
        return Err(Refused::Rule(
            "row-reuse folds require one exact row bucket".into(),
        ));
    }
    let chosen = choose(
        "dense decode folds",
        entries
            .iter()
            .filter(|(e, _)| {
                e.query == *query && e.paired == paired && (e.knobs.row_groups == 2) == reuse
            })
            .map(|(e, folds)| {
                Candidate::new(
                    (e.clone(), *folds),
                    &e.symbol,
                    format!("{e:?}:folds={folds:?}"),
                    Invariance::any_rows(e.query.m_capacity),
                )
                .capacity(e.query.m_capacity)
            }),
        need,
    )?;
    let (entry, folds) = chosen.entry;
    entry.validate()?;
    if reuse {
        if paired || need.capacity != query.m_capacity || !matches!(folds, Some(2 | 4 | 8)) {
            return Err(Refused::Rule(
                "row-reuse folds require one exact row bucket and partition".into(),
            ));
        }
    } else if folds.is_some() {
        return Err(Refused::Rule(
            "runtime fold overrides are only for row reuse".into(),
        ));
    }
    entry.plans(
        need.capacity,
        folds.unwrap_or_else(|| entry.folds_per_cta(need.capacity)),
    )?;
    let legacy = super::mxfp8_fold::prepare(&[entry.legacy()?], query, need)?;
    let mut info = legacy.info().clone();
    info.implementation = format!("sm120.dense-decode.v1:{entry:?}:folds={folds:?}");
    info.symbols = entry.specs()?.into_iter().map(|s| (s.symbol, 0)).collect();
    info.workspace = entry.workspace(need.capacity)?;
    if paired {
        info.bytes = ["0", "1"]
            .into_iter()
            .flat_map(|bank| {
                legacy
                    .info()
                    .bytes
                    .iter()
                    .map(move |(role, bytes)| (format!("{bank}.{role}"), *bytes))
            })
            .collect();
        info.weight_layouts = ["0", "1"]
            .into_iter()
            .flat_map(|bank| {
                legacy
                    .info()
                    .weight_layouts
                    .iter()
                    .map(move |(role, layout)| (format!("{bank}.{role}"), layout.clone()))
            })
            .collect();
    }
    need.check(&info)?;
    Ok(Op {
        entry,
        folds,
        legacy,
        info,
    })
}

pub(super) fn prepare(
    entries: &[(Entry, Option<u32>)],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
    Ok(Box::new(prepare_op(entries, query, need, false)?))
}

pub(super) fn prepare_pair(
    entries: &[(Entry, Option<u32>)],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<[Mxfp8FoldGemmBuffers<View>; 2]>, Refused> {
    Ok(Box::new(prepare_op(entries, query, need, true)?))
}

struct Op {
    entry: Entry,
    folds: Option<u32>,
    legacy: Prepared<Mxfp8FoldGemmBuffers<View>>,
    info: OpInfo,
}

pub(super) fn validate_launch(
    legacy: &Prepared<Mxfp8FoldGemmBuffers<View>>,
    workspace_bytes: u64,
    b: &Mxfp8FoldGemmBuffers<View>,
    second: Option<&Mxfp8FoldGemmBuffers<View>>,
    workspace: View,
    counts: Counts,
) -> Result<(), Refusal> {
    // Reuse the original op's complete buffer, alias, count and status
    // offset checks without launching its device entry.
    legacy.launches(b, View::EMPTY, counts)?;
    workspace.covers(workspace_bytes)?;
    let handles = |b: &Mxfp8FoldGemmBuffers<View>| {
        [
            b.activations.handle,
            b.activation_scales.handle,
            b.weights.handle,
            b.weight_scales.handle,
            b.output.handle,
            b.live_rows.handle,
            b.status.handle,
        ]
    };
    if handles(b).contains(&workspace.handle) {
        return Err(Refusal("split MXFP8 workspace aliases a role"));
    }
    if let Some(other) = second {
        legacy.launches(other, View::EMPTY, counts)?;
        if b.activations != other.activations
            || b.activation_scales != other.activation_scales
            || b.live_rows != other.live_rows
            || b.status != other.status
            || b.status_row_offset != other.status_row_offset
        {
            return Err(Refusal(
                "paired MXFP8 folds require the same activations and status",
            ));
        }
        if handles(other).contains(&workspace.handle)
            || handles(other).contains(&b.output.handle)
            || handles(b).contains(&other.output.handle)
            || other.weights.handle == b.status.handle
            || other.weight_scales.handle == b.status.handle
        {
            return Err(Refusal(
                "paired MXFP8 outputs, workspace or status alias another role",
            ));
        }
    }
    Ok(())
}

impl Op {
    fn lower(
        &self,
        b: &Mxfp8FoldGemmBuffers<View>,
        second: Option<&Mxfp8FoldGemmBuffers<View>>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if self.folds.is_some() && counts.rows != self.entry.query.m_capacity {
            return Err(Refusal(
                "row-reuse launch differs from its selected row bucket",
            ));
        }
        validate_launch(
            &self.legacy,
            self.info.workspace,
            b,
            second,
            workspace,
            counts,
        )?;
        let d = |view: View| Argument::DeviceBuffer(view.handle);
        let tail = || {
            vec![
                d(b.live_rows),
                d(b.status),
                Argument::U64(b.status_row_offset),
                Argument::U32(counts.rows),
            ]
        };
        let mut admit = vec![d(b.activations), d(b.activation_scales), d(workspace)];
        admit.extend(tail());
        let mut partials = vec![
            d(b.activations),
            d(b.activation_scales),
            d(b.weights),
            d(b.weight_scales),
            second.map_or(Argument::NullDevicePointer, |s| d(s.weights)),
            second.map_or(Argument::NullDevicePointer, |s| d(s.weight_scales)),
            d(workspace),
        ];
        partials.extend(tail());
        let mut merge = vec![
            d(workspace),
            d(b.output),
            second.map_or(Argument::NullDevicePointer, |s| d(s.output)),
        ];
        merge.extend(tail());
        let folds = self
            .folds
            .unwrap_or_else(|| self.entry.folds_per_cta(counts.rows));
        let arguments = if self.entry.knobs.finish == Finish::Merge {
            vec![admit, partials, merge]
        } else {
            let compute = vec![
                d(b.activations),
                d(b.activation_scales),
                d(b.weights),
                d(b.weight_scales),
                second.map_or(Argument::NullDevicePointer, |s| d(s.weights)),
                second.map_or(Argument::NullDevicePointer, |s| d(s.weight_scales)),
                d(b.output),
                second.map_or(Argument::NullDevicePointer, |s| d(s.output)),
                d(workspace),
                d(b.live_rows),
                Argument::U32(counts.rows),
                Argument::U32(folds),
            ];
            vec![admit, compute]
        };
        Ok(self
            .entry
            .specs()?
            .into_iter()
            .zip(self.entry.plans(counts.rows, folds)?)
            .zip(arguments)
            .map(|((spec, plan), arguments)| Launch {
                symbol: spec.symbol,
                plan,
                arguments: ArgumentList(arguments),
                rows: counts.rows,
                collective: false,
                persistent: false,
            })
            .collect())
    }
}

impl PreparedOp<Mxfp8FoldGemmBuffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &Mxfp8FoldGemmBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        self.lower(b, None, workspace, counts)
    }
}

impl PreparedOp<[Mxfp8FoldGemmBuffers<View>; 2]> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &[Mxfp8FoldGemmBuffers<View>; 2],
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        self.lower(&b[0], Some(&b[1]), workspace, counts)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::backend::Sm120;
    use kernel_core::backend::Backend;

    fn buffers(entry: &Entry, rows: u32) -> Mxfp8FoldGemmBuffers<View> {
        let b = entry.legacy().unwrap().bytes(rows);
        let v = |handle, bytes| View { handle, bytes };
        Mxfp8FoldGemmBuffers {
            activations: v(1, b.activations),
            activation_scales: v(2, b.activation_scales),
            weights: v(3, b.weights),
            weight_scales: v(4, b.weight_scales),
            output: v(5, b.output),
            live_rows: v(6, 4),
            status: v(7, b.status + 12),
            status_row_offset: 3,
        }
    }

    #[test]
    fn split_folds_cover_tails_buckets_and_workspace_bounds() {
        for baseline in crate::gemm::mxfp8_fold::lab_entries() {
            let query = Mxfp8FoldGemmQuery {
                m_capacity: 32,
                ..baseline.query
            };
            let entry = Entry::new(
                query,
                "split_fold_test_v1",
                false,
                crate::gemm::dense_decode::Knobs::SPLIT,
            )
            .unwrap();
            let backend = Sm120::new().with_dense_decode(entry.clone(), None);
            for capacity in [1, 2, 4, 8, 16, 32] {
                let op = backend
                    .mxfp8_fold_gemm(&query, Need::rows(capacity))
                    .unwrap();
                let bytes = 256
                    + 4 * u64::from(query.k.div_ceil(128))
                        * u64::from(capacity)
                        * u64::from(query.groups)
                        * u64::from(query.n);
                assert_eq!(op.info().workspace, bytes);
                let workspace = View { handle: 8, bytes };
                let b = buffers(&entry, capacity);
                for rows in [0, 1, capacity] {
                    let calls = op.launches(&b, workspace, Counts::rows(rows)).unwrap();
                    assert_eq!(calls.len(), 3);
                    assert_eq!(
                        calls[1].plan.grid,
                        [
                            query.n.div_ceil(16),
                            rows.max(1).div_ceil(16),
                            query.groups * query.k.div_ceil(128).div_ceil(2)
                        ]
                    );
                    assert!(calls.iter().all(|call| call.plan.pdl));
                }
                assert!(
                    op.launches(
                        &b,
                        View {
                            bytes: bytes - 1,
                            ..workspace
                        },
                        Counts::rows(1)
                    )
                    .is_err()
                );
                assert!(
                    op.launches(
                        &b,
                        View {
                            handle: 5,
                            ..workspace
                        },
                        Counts::rows(1)
                    )
                    .is_err()
                );
                assert!(
                    op.launches(&b, workspace, Counts::rows(capacity + 1))
                        .is_err()
                );
                let mut bad = b;
                bad.status.bytes -= 1;
                assert!(
                    op.launches(&bad, workspace, Counts::rows(capacity))
                        .is_err()
                );
            }
            assert!(
                Entry::new(
                    Mxfp8FoldGemmQuery {
                        m_capacity: 33,
                        ..query
                    },
                    "too_wide_v1",
                    false,
                    crate::gemm::dense_decode::Knobs::SPLIT
                )
                .is_err()
            );
        }
    }

    #[test]
    fn paired_folds_share_admission_and_refuse_cross_bank_aliases() {
        let base = crate::gemm::mxfp8_fold::lab_entries().remove(0);
        let entry = Entry::new(
            base.query,
            "split_pair_test_v1",
            true,
            crate::gemm::dense_decode::Knobs::SPLIT,
        )
        .unwrap();
        let backend = Sm120::new().with_dense_decode(entry.clone(), None);
        let op = backend
            .mxfp8_fold_gemm_pair(&entry.query, Need::rows(1))
            .unwrap();
        let a = buffers(&entry, 1);
        let mut b = a;
        b.weights.handle = 9;
        b.weight_scales.handle = 10;
        b.output.handle = 11;
        let workspace = View {
            handle: 8,
            bytes: op.info().workspace,
        };
        let pair = [a, b];
        let calls = op.launches(&pair, workspace, Counts::rows(1)).unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].plan.grid, [1, 1, 1]);
        assert_eq!(calls[1].plan.grid[2], 2);
        b.activations.handle = 12;
        assert!(op.launches(&[a, b], workspace, Counts::rows(1)).is_err());
        let mut bad = pair;
        bad[1].output.handle = a.weights.handle;
        assert!(op.launches(&bad, workspace, Counts::rows(1)).is_err());
        let mut bad = pair;
        bad[1].weights.handle = a.output.handle;
        assert!(op.launches(&bad, workspace, Counts::rows(1)).is_err());
        assert!(
            backend
                .mxfp8_fold_gemm(&entry.query, Need::rows(1))
                .is_err()
        );
    }

    #[test]
    fn prefetch_uses_only_the_supplied_workspace_without_growing_it() {
        for old in crate::gemm::mxfp8_fold::lab_entries() {
            let q = Mxfp8FoldGemmQuery {
                m_capacity: 64,
                ..old.query
            };
            for paired in [false, true] {
                let ticket = Entry::new(
                    q,
                    "ticket_test_v2",
                    paired,
                    crate::gemm::dense_decode::Knobs::ticket(q.n),
                )
                .unwrap();
                let mut prefetch = Entry::new(
                    q,
                    "prefetch_test_v3",
                    paired,
                    crate::gemm::dense_decode::Knobs::ticket(q.n),
                )
                .unwrap();
                prefetch.knobs.weight_prefetch = true;
                for rows in 0..=64 {
                    assert_eq!(prefetch.workspace(rows), ticket.workspace(rows));
                    assert_eq!(
                        prefetch.plans(rows, prefetch.folds_per_cta(rows)),
                        ticket.plans(rows, ticket.folds_per_cta(rows))
                    );
                }
                if paired {
                    continue;
                }
                let backend = Sm120::new().with_dense_decode(prefetch, None);
                let op = backend.mxfp8_fold_gemm(&q, Need::rows(64)).unwrap();
                let bytes = old.bytes(64);
                let view = |handle, bytes| View { handle, bytes };
                let buffers = Mxfp8FoldGemmBuffers {
                    activations: view(1, bytes.activations),
                    activation_scales: view(2, bytes.activation_scales),
                    weights: view(3, bytes.weights),
                    weight_scales: view(4, bytes.weight_scales),
                    output: view(5, bytes.output),
                    live_rows: view(6, 4),
                    status: view(7, bytes.status),
                    status_row_offset: 0,
                };
                for workspace in [11, 31] {
                    let calls = op
                        .launches(
                            &buffers,
                            view(workspace, op.info().workspace),
                            Counts::rows(64),
                        )
                        .unwrap();
                    assert_eq!(calls[0].arguments.0[2], Argument::DeviceBuffer(workspace));
                    assert_eq!(calls[1].arguments.0[8], Argument::DeviceBuffer(workspace));
                    assert!(calls.iter().all(|call| !call.arguments.0.contains(
                        &Argument::DeviceBuffer(if workspace == 11 { 31 } else { 11 })
                    )));
                }
                assert!(
                    op.launches(&buffers, view(5, op.info().workspace), Counts::rows(64))
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn tickets_cover_every_row_count_and_refuse_uninitialized_workspace() {
        for old in crate::gemm::mxfp8_fold::lab_entries() {
            let q = Mxfp8FoldGemmQuery {
                m_capacity: 64,
                ..old.query
            };
            for paired in [false, true] {
                let entry = Entry::new(
                    q,
                    "ticket_test_v2",
                    paired,
                    crate::gemm::dense_decode::Knobs::ticket(q.n),
                )
                .unwrap();
                let backend = Sm120::new().with_dense_decode(entry.clone(), None);
                let sizes = entry.legacy().unwrap().bytes(64);
                let v = |handle, bytes| View { handle, bytes };
                let b = Mxfp8FoldGemmBuffers {
                    activations: v(1, sizes.activations),
                    activation_scales: v(2, sizes.activation_scales),
                    weights: v(3, sizes.weights),
                    weight_scales: v(4, sizes.weight_scales),
                    output: v(5, sizes.output),
                    live_rows: v(6, 4),
                    status: v(7, sizes.status + 12),
                    status_row_offset: 3,
                };
                let second = Mxfp8FoldGemmBuffers {
                    weights: v(8, sizes.weights),
                    weight_scales: v(9, sizes.weight_scales),
                    output: v(10, sizes.output),
                    ..b
                };
                for rows in 0..=64 {
                    let need = Need::rows(rows.max(1));
                    let workspace = v(11, entry.workspace(rows.max(1)).unwrap());
                    let calls = if paired {
                        backend
                            .mxfp8_fold_gemm_pair(&q, need)
                            .unwrap()
                            .launches(&[b, second], workspace, Counts::rows(rows))
                            .unwrap()
                    } else {
                        backend
                            .mxfp8_fold_gemm(&q, need)
                            .unwrap()
                            .launches(&b, workspace, Counts::rows(rows))
                            .unwrap()
                    };
                    assert_eq!(calls.len(), 2);
                    assert_eq!(
                        calls[1].plan.grid[2],
                        q.groups
                            * if paired { 2 } else { 1 }
                            * q.k.div_ceil(128).div_ceil(entry.folds_per_cta(rows))
                    );
                    assert!(matches!(entry.folds_per_cta(rows), 2 | 4 | 8 | 16));
                    assert!(entry.header(rows).unwrap() >= 256 + 4 * entry.tiles(rows));
                }
                if !paired {
                    let op = backend.mxfp8_fold_gemm(&q, Need::rows(64)).unwrap();
                    assert!(op.launches(&b, View::EMPTY, Counts::rows(1)).is_err());
                    assert!(
                        op.launches(&b, v(1, op.info().workspace), Counts::rows(1))
                            .is_err()
                    );
                    assert!(
                        op.launches(&b, v(11, op.info().workspace), Counts::rows(65))
                            .is_err()
                    );
                }
            }
        }
    }
}
