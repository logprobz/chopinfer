//! Prepared direct folds reuse the packed-weight contract without global partials.
use super::select::{Candidate, choose};
use crate::gemm::mxfp8_fold_direct::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery, Mxfp8FoldStreamBuffers};
use kernel_core::properties::Invariance;

pub(super) fn prepare(
    entries: &[Entry],
    query: &Mxfp8FoldGemmQuery,
    paired: bool,
    need: Need,
) -> Result<Prepared<Mxfp8FoldStreamBuffers<View>>, Refused> {
    let entry = choose(
        "direct MXFP8 folds",
        entries
            .iter()
            .filter(|e| e.packed.ticket.query == *query && e.packed.ticket.paired == paired)
            .map(|e| {
                Candidate::new(
                    e.clone(),
                    &e.symbol,
                    format!("{e:?}"),
                    Invariance::any_rows(query.m_capacity),
                )
                .capacity(query.m_capacity)
            }),
        need,
    )?
    .entry;
    entry.validate()?;
    if entry.defer_admission || entry.omit_first {
        return Err(Refusal("diagnostic direct folds cannot serve").into());
    }
    let initializer = super::mxfp8_fold_stream::prepare(
        std::slice::from_ref(&entry.packed),
        query,
        paired,
        need,
    )?;
    let legacy = super::mxfp8_fold::prepare(&[entry.packed.ticket.legacy()?], query, need)?;
    let mut info = initializer.info().clone();
    info.implementation = format!("sm120.mxfp8-fold.direct.v1:{entry:?}");
    let [init, compute, _] = entry.symbols();
    info.symbols = vec![
        (init, 0),
        (compute, entry.plan(need.capacity)?.dynamic_shared_bytes),
    ];
    info.workspace = 0;
    let state = info
        .bytes
        .iter_mut()
        .find(|(role, _)| role == "state")
        .unwrap();
    state.1 = state.1.max(entry.state_bytes(query.m_capacity)?);
    need.check(&info)?;
    Ok(Box::new(Op {
        entry,
        initializer,
        legacy,
        info,
    }))
}
struct Op {
    entry: Entry,
    initializer: Prepared<Mxfp8FoldStreamBuffers<View>>,
    legacy: Prepared<Mxfp8FoldGemmBuffers<View>>,
    info: OpInfo,
}
impl PreparedOp<Mxfp8FoldStreamBuffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn initialize(&self, b: &Mxfp8FoldStreamBuffers<View>) -> Result<Vec<Launch>, Refusal> {
        b.state.covers(self.info.bytes_of("state").unwrap())?;
        let mut launches = self.initializer.initialize(b)?;
        launches[0].symbol = self.entry.symbols()[0].clone();
        Ok(launches)
    }
    fn launches(
        &self,
        b: &Mxfp8FoldStreamBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        // Keep the original contract's count, alias and status-offset checks.
        self.legacy.launches(&b.first, View::EMPTY, counts)?;
        if let Some(second) = &b.second {
            self.legacy.launches(second, View::EMPTY, counts)?;
        }
        Ok(vec![self.entry.launch(b, counts.rows, None)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::{mxfp8_fold_direct::Schedule, mxfp8_fold_stream};
    use kernel_core::backend::Backend;
    use kernel_core::contract::{Argument, DType};
    use kernel_core::precision::execution::BlockScaledExecution;

    #[test]
    fn direct_fold_prepared_launch_matches_qualified_abi_and_clears_all_state() {
        let q = Mxfp8FoldGemmQuery {
            m_capacity: 2,
            n: 5120,
            k: 2048,
            groups: 1,
            activation_row_stride: 2048,
            activation_group_stride: 2048,
            output: DType::F32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        };
        for paired in [false, true] {
            let entry = Entry {
                packed: mxfp8_fold_stream::Entry::new(q, "packed", paired).unwrap(),
                symbol: "direct_v1".into(),
                schedule: Schedule::LocalSplit,
                defer_admission: false,
                omit_first: false,
            };
            let op = crate::backend::Sm120::new()
                .with_mxfp8_fold_direct(entry.clone())
                .mxfp8_fold_gemm_stream(&q, paired, Need::rows(2))
                .unwrap();
            let bytes = entry.packed.ticket.legacy().unwrap().bytes(2);
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
            let b = Mxfp8FoldStreamBuffers {
                first,
                second: paired.then_some(Mxfp8FoldGemmBuffers {
                    weights: v(8, bytes.weights),
                    weight_scales: v(9, bytes.weight_scales),
                    output: v(10, bytes.output),
                    ..first
                }),
                packed_weights: [
                    v(11, bytes.weights),
                    if paired {
                        v(12, bytes.weights)
                    } else {
                        View::EMPTY
                    },
                ],
                // A later row bucket can need more state than this initializer.
                state: v(13, op.info().bytes_of("state").unwrap() + 4096),
            };
            let init = op.initialize(&b).unwrap();
            assert_eq!(init[0].symbol, entry.symbols()[0]);
            assert_eq!(
                init[0].arguments.0[5],
                Argument::U32((b.state.bytes / 4) as u32)
            );
            assert!(!init[0].plan.pdl);
            let calls = op.launches(&b, View::EMPTY, Counts::rows(2)).unwrap();
            let qualified = entry.launch(&b, 2, None).unwrap();
            assert_eq!(calls[0].arguments, qualified.arguments);
            assert_eq!(calls[0].plan, qualified.plan);
            assert_eq!(op.info().symbols[1].1, qualified.plan.dynamic_shared_bytes);
            assert_eq!(op.info().workspace, 0);
            assert!(op.launches(&b, View::EMPTY, Counts::rows(3)).is_err());
            let mut bad = b;
            bad.state.handle = b.first.output.handle;
            assert!(op.initialize(&bad).is_err());
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(2)).is_err());
        }
    }
}
