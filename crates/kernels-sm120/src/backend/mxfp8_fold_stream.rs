use super::select::{Candidate, choose};
use crate::gemm::mxfp8_fold_stream::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, Refusal};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery, Mxfp8FoldStreamBuffers};
use kernel_core::properties::Invariance;

pub(super) fn prepare(
    entries: &[Entry],
    query: &Mxfp8FoldGemmQuery,
    paired: bool,
    need: Need,
) -> Result<Prepared<Mxfp8FoldStreamBuffers<View>>, Refused> {
    let chosen = choose(
        "packed MXFP8 folds",
        entries
            .iter()
            .filter(|e| e.ticket.query == *query && e.ticket.paired == paired)
            .map(|e| {
                Candidate::new(
                    e.clone(),
                    &e.ticket.symbol,
                    format!("{e:?}"),
                    Invariance::any_rows(query.m_capacity),
                )
                .capacity(query.m_capacity)
            }),
        need,
    )?;
    let entry = chosen.entry;
    entry.validate()?;
    let legacy = super::mxfp8_fold::prepare(&[entry.ticket.legacy()?], query, need)?;
    let mut info = legacy.info().clone();
    info.implementation = format!("sm120.mxfp8-fold.stream.v1:{entry:?}");
    info.symbols = entry.symbols().into_iter().map(|s| (s, 0)).collect();
    info.workspace = entry.ticket.workspace(need.capacity)?;
    let banks: &[&str] = if paired {
        &["first", "second"]
    } else {
        &["first"]
    };
    info.bytes = banks
        .iter()
        .flat_map(|bank| {
            legacy
                .info()
                .bytes
                .iter()
                .map(move |(role, bytes)| (format!("{bank}.{role}"), *bytes))
        })
        .collect();
    info.weight_layouts = banks
        .iter()
        .flat_map(|bank| {
            legacy
                .info()
                .weight_layouts
                .iter()
                .map(move |(role, layout)| (format!("{bank}.{role}"), layout.clone()))
        })
        .collect();

    info.bytes
        .push(("packed_weights.0".into(), entry.packed_bytes()));
    if paired {
        info.bytes
            .push(("packed_weights.1".into(), entry.packed_bytes()));
    }
    info.bytes
        .push(("state".into(), entry.state_bytes(need.capacity)?));
    need.check(&info)?;
    Ok(Box::new(Op {
        entry,
        legacy,
        info,
        capacity: need.capacity,
    }))
}
struct Op {
    entry: Entry,
    legacy: Prepared<Mxfp8FoldGemmBuffers<View>>,
    info: OpInfo,
    capacity: u32,
}
impl Op {
    fn persistent(
        &self,
        b: &Mxfp8FoldStreamBuffers<View>,
        workspace: Option<View>,
    ) -> Result<(), Refusal> {
        if b.second.is_some() != self.entry.ticket.paired {
            return Err(Refusal("packed fold bank count differs from its entry"));
        }
        b.state.covers(self.entry.state_bytes(self.capacity)?)?;
        b.packed_weights[0].covers(self.entry.packed_bytes())?;
        if self.entry.ticket.paired {
            b.packed_weights[1].covers(self.entry.packed_bytes())?;
        } else if b.packed_weights[1] != View::EMPTY {
            return Err(Refusal("single packed fold has a second packed bank"));
        }
        let mut occupied = vec![];
        for x in std::iter::once(&b.first).chain(b.second.as_ref()) {
            occupied.extend([
                x.activations.handle,
                x.activation_scales.handle,
                x.weights.handle,
                x.weight_scales.handle,
                x.output.handle,
                x.live_rows.handle,
                x.status.handle,
            ]);
        }
        if let Some(w) = workspace {
            occupied.push(w.handle);
        }
        for v in std::iter::once(b.state).chain(
            b.packed_weights
                .iter()
                .copied()
                .take(if self.entry.ticket.paired { 2 } else { 1 }),
        ) {
            if occupied.contains(&v.handle) {
                return Err(Refusal(
                    "packed fold persistent storage aliases another role",
                ));
            }
            occupied.push(v.handle);
        }
        Ok(())
    }
}
impl PreparedOp<Mxfp8FoldStreamBuffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn initialize(&self, b: &Mxfp8FoldStreamBuffers<View>) -> Result<Vec<Launch>, Refusal> {
        self.persistent(b, None)?;
        let bytes = self.entry.ticket.legacy()?.bytes(0);
        b.first.weights.covers(bytes.weights)?;
        if let Some(second) = b.second {
            second.weights.covers(bytes.weights)?;
        }
        let d = |v: View| Argument::DeviceBuffer(v.handle);
        Ok(vec![Launch {
            symbol: self.entry.symbols()[0].clone(),
            plan: self.entry.initialize_plan(),
            arguments: ArgumentList(vec![
                d(b.first.weights),
                b.second
                    .map_or(Argument::NullDevicePointer, |s| d(s.weights)),
                d(b.packed_weights[0]),
                if self.entry.ticket.paired {
                    d(b.packed_weights[1])
                } else {
                    Argument::NullDevicePointer
                },
                d(b.state),
                Argument::U32((self.entry.state_bytes(self.capacity)? / 4) as u32),
            ]),
            rows: 0,
            collective: false,
            persistent: false,
        }])
    }
    fn launches(
        &self,
        b: &Mxfp8FoldStreamBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        self.persistent(b, Some(workspace))?;
        super::mxfp8_fold_decode::validate_launch(
            &self.legacy,
            self.info.workspace,
            &b.first,
            b.second.as_ref(),
            workspace,
            counts,
        )?;
        let d = |v: View| Argument::DeviceBuffer(v.handle);
        let first = b.first;
        Ok(vec![Launch {
            symbol: self.entry.ticket.symbol.clone(),
            plan: self.entry.ticket.plans(counts.rows)?[1],
            arguments: ArgumentList(vec![
                d(first.activations),
                d(first.activation_scales),
                d(b.packed_weights[0]),
                d(first.weight_scales),
                if self.entry.ticket.paired {
                    d(b.packed_weights[1])
                } else {
                    Argument::NullDevicePointer
                },
                b.second
                    .map_or(Argument::NullDevicePointer, |s| d(s.weight_scales)),
                d(first.output),
                b.second
                    .map_or(Argument::NullDevicePointer, |s| d(s.output)),
                d(workspace),
                d(b.state),
                d(first.live_rows),
                d(first.status),
                Argument::U64(first.status_row_offset),
                Argument::U32(counts.rows),
                Argument::U32(self.entry.ticket.folds_per_cta(counts.rows)),
            ]),
            rows: counts.rows,
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    #[test]
    fn packed_fold_initializes_once_and_keeps_state_out_of_disposable_workspace() {
        let query = Mxfp8FoldGemmQuery {
            m_capacity: 64,
            n: 576,
            k: 576,
            groups: 2,
            activation_row_stride: 1152,
            activation_group_stride: 576,
            output: kernel_core::contract::DType::F32,
            execution:
                kernel_core::precision::execution::BlockScaledExecution::NativeBlockScaledMma,
        };
        for paired in [false, true] {
            let entry = Entry::new(query, "packed_fold", paired).unwrap();
            let backend = crate::backend::Sm120::new().with_mxfp8_fold_stream(entry.clone());
            for rows in [1, 2, 4, 6, 8, 16, 32, 64] {
                let op = backend
                    .mxfp8_fold_gemm_stream(&query, paired, Need::rows(rows))
                    .unwrap();
                let bytes = entry.ticket.legacy().unwrap().bytes(rows);
                assert_eq!(entry.packed_bytes(), bytes.weights);
                assert_eq!(op.info().bytes_of("first.weights"), Some(bytes.weights));
                assert_eq!(
                    op.info().bytes_of("second.output"),
                    paired.then_some(bytes.output)
                );
                assert_eq!(op.info().layout_of("second.weights").is_some(), paired);
                let v = |handle, bytes| View { handle, bytes };
                let first = Mxfp8FoldGemmBuffers {
                    activations: v(1, bytes.activations),
                    activation_scales: v(2, bytes.activation_scales),
                    weights: v(3, bytes.weights),
                    weight_scales: v(4, bytes.weight_scales),
                    output: v(5, bytes.output),
                    live_rows: v(6, 4),
                    status: v(7, bytes.status),
                    status_row_offset: 0,
                };
                let second = Mxfp8FoldGemmBuffers {
                    weights: v(8, bytes.weights),
                    weight_scales: v(9, bytes.weight_scales),
                    output: v(10, bytes.output),
                    ..first
                };
                let b = Mxfp8FoldStreamBuffers {
                    first,
                    second: paired.then_some(second),
                    packed_weights: [
                        v(11, entry.packed_bytes()),
                        if paired {
                            v(12, entry.packed_bytes())
                        } else {
                            View::EMPTY
                        },
                    ],
                    state: v(13, entry.state_bytes(rows).unwrap()),
                };
                assert_eq!(op.initialize(&b).unwrap().len(), 1);
                let workspace = v(14, op.info().workspace);
                let calls = op.launches(&b, workspace, Counts::rows(rows)).unwrap();
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].symbol, "packed_fold");
                assert!(calls[0].arguments.0.contains(&Argument::DeviceBuffer(13)));
                for alias in [1, 5, 11, 14] {
                    let bad = Mxfp8FoldStreamBuffers {
                        state: v(alias, b.state.bytes),
                        ..b
                    };
                    assert!(op.launches(&bad, workspace, Counts::rows(rows)).is_err());
                }
                let bad = Mxfp8FoldStreamBuffers {
                    state: v(13, b.state.bytes - 1),
                    ..b
                };
                assert!(op.initialize(&bad).is_err());
                let other = Mxfp8FoldStreamBuffers {
                    state: v(31, b.state.bytes),
                    ..b
                };
                let second_calls = op
                    .launches(&other, v(32, workspace.bytes), Counts::rows(rows))
                    .unwrap();
                assert!(
                    second_calls[0]
                        .arguments
                        .0
                        .contains(&Argument::DeviceBuffer(31))
                );
                assert!(
                    !second_calls[0]
                        .arguments
                        .0
                        .contains(&Argument::DeviceBuffer(13))
                );
            }
        }
    }
}
