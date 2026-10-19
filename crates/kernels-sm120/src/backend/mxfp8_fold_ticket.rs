use super::select::{Candidate, choose};
use crate::gemm::mxfp8_fold_ticket::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, Refusal};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
use kernel_core::properties::Invariance;

fn op(
    entries: &[Entry],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
    paired: bool,
) -> Result<Op, Refused> {
    let selected = choose(
        "ticket MXFP8 folds",
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
    let entry = selected.entry;
    entry.validate()?;
    let legacy = super::mxfp8_fold::prepare(&[entry.legacy()?], query, need)?;
    let mut info = legacy.info().clone();
    info.implementation = format!("sm120.mxfp8-fold.ticket.v2:{entry:?}");
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
    Ok(Box::new(op(entries, query, need, false)?))
}
pub(super) fn prepare_pair(
    entries: &[Entry],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<[Mxfp8FoldGemmBuffers<View>; 2]>, Refused> {
    Ok(Box::new(op(entries, query, need, true)?))
}

struct Op {
    entry: Entry,
    legacy: Prepared<Mxfp8FoldGemmBuffers<View>>,
    info: OpInfo,
}
impl Op {
    fn lower(
        &self,
        b: &Mxfp8FoldGemmBuffers<View>,
        second: Option<&Mxfp8FoldGemmBuffers<View>>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        super::mxfp8_fold_decode::validate_launch(
            &self.legacy,
            self.info.workspace,
            b,
            second,
            workspace,
            counts,
        )?;
        let d = |v: View| Argument::DeviceBuffer(v.handle);
        let init = vec![
            d(b.activations),
            d(b.activation_scales),
            d(workspace),
            d(b.live_rows),
            d(b.status),
            Argument::U64(b.status_row_offset),
            Argument::U32(counts.rows),
        ];
        let compute = vec![
            d(b.activations),
            d(b.activation_scales),
            d(b.weights),
            d(b.weight_scales),
            second.map_or(Argument::NullDevicePointer, |b| d(b.weights)),
            second.map_or(Argument::NullDevicePointer, |b| d(b.weight_scales)),
            d(b.output),
            second.map_or(Argument::NullDevicePointer, |b| d(b.output)),
            d(workspace),
            d(b.live_rows),
            Argument::U32(counts.rows),
            Argument::U32(self.entry.folds_per_cta(counts.rows)),
        ];
        Ok(self
            .entry
            .symbols()
            .into_iter()
            .zip(self.entry.plans(counts.rows)?)
            .zip([init, compute])
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

    #[test]
    fn tickets_cover_every_row_count_and_refuse_uninitialized_workspace() {
        for old in crate::gemm::mxfp8_fold::lab_entries() {
            let q = Mxfp8FoldGemmQuery {
                m_capacity: 64,
                ..old.query
            };
            for paired in [false, true] {
                let entry = Entry::new(q, "ticket_test_v2", paired).unwrap();
                let backend = Sm120::new().with_mxfp8_fold_ticket(entry.clone());
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
