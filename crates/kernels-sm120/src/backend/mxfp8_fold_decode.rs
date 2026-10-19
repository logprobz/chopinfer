use super::select::{Candidate, choose};
use crate::gemm::mxfp8_fold_decode::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, Refusal};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
use kernel_core::properties::Invariance;

fn prepare_op(
    entries: &[Entry],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
    paired: bool,
) -> Result<Op, Refused> {
    let chosen = choose(
        "split MXFP8 folds",
        entries
            .iter()
            .filter(|e| e.query == *query && e.paired == paired)
            .map(|e| {
                Candidate::new(
                    e.clone(),
                    &e.symbol,
                    format!("{e:?}"),
                    Invariance::any_rows(e.query.m_capacity),
                )
                .capacity(e.query.m_capacity)
            }),
        need,
    )?;
    let entry = chosen.entry;
    entry.validate()?;
    let legacy = super::mxfp8_fold::prepare(&[entry.legacy()?], query, need)?;
    let mut info = legacy.info().clone();
    info.implementation = format!("sm120.mxfp8-fold.split.v1:{entry:?}");
    info.symbols = entry.symbols().into_iter().map(|s| (s, 0)).collect();
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
        legacy,
        info,
    })
}

pub(super) fn prepare(
    entries: &[Entry],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
    Ok(Box::new(prepare_op(entries, query, need, false)?))
}

pub(super) fn prepare_pair(
    entries: &[Entry],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<[Mxfp8FoldGemmBuffers<View>; 2]>, Refused> {
    Ok(Box::new(prepare_op(entries, query, need, true)?))
}

struct Op {
    entry: Entry,
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
        Ok(self
            .entry
            .symbols()
            .into_iter()
            .zip(self.entry.plans(counts.rows)?)
            .zip([admit, partials, merge])
            .map(|((symbol, plan), arguments)| Launch {
                symbol,
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
            let entry = Entry::new(query, "split_fold_test_v1", false).unwrap();
            let backend = Sm120::new().with_mxfp8_fold_decode(entry.clone());
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
                    false
                )
                .is_err()
            );
        }
    }

    #[test]
    fn paired_folds_share_admission_and_refuse_cross_bank_aliases() {
        let base = crate::gemm::mxfp8_fold::lab_entries().remove(0);
        let entry = Entry::new(base.query, "split_pair_test_v1", true).unwrap();
        let backend = Sm120::new().with_mxfp8_fold_decode(entry.clone());
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
}
