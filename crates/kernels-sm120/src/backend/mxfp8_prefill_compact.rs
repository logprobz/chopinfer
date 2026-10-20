//! Exact-row opt-ins layered over the caller's existing prefill fallback.
use crate::gemm::mxfp8_prefill_compact::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers as Buffers, Mxfp8FoldGemmQuery};
use std::collections::BTreeMap;

pub(super) fn prepare(
    fallback: Prepared<Buffers<View>>,
    entries: &[(Entry, u32)],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<Buffers<View>>, Refused> {
    let mut selected = BTreeMap::<u32, &Entry>::new();
    for (entry, rows) in entries.iter().filter(|(e, _)| e.base.query == *query) {
        if *rows == 0 || *rows > query.m_capacity {
            return Err(Refusal("compact prefill registration exceeds query capacity").into());
        }
        if let Some(old) = selected.insert(*rows, entry)
            && (old.base.symbol != entry.base.symbol || old.stages != entry.stages)
        {
            return Err(Refusal("ambiguous compact prefill row registration").into());
        }
    }
    selected.retain(|rows, _| *rows <= need.capacity);
    if selected.is_empty() {
        return Ok(fallback);
    }
    let mut info = fallback.info().clone();
    let mut plans = BTreeMap::new();
    let mut choices = Vec::new();
    for (rows, entry) in selected {
        let op = entry.prepare(rows)?;
        info.symbols.extend(op.info().symbols.clone());
        choices.push((rows, entry.stages));
        plans.insert(rows, op);
    }
    info.symbols.sort();
    info.symbols.dedup();
    info.implementation
        .push_str(&format!(";compact-v1:{choices:?}"));
    need.check(&info)?;
    Ok(Box::new(Op {
        fallback,
        plans,
        info,
    }))
}

struct Op {
    fallback: Prepared<Buffers<View>>,
    plans: BTreeMap<u32, Prepared<Buffers<View>>>,
    info: OpInfo,
}
impl PreparedOp<Buffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        buffers: &Buffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        self.plans
            .get(&counts.rows)
            .unwrap_or(&self.fallback)
            .launches(buffers, workspace, counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::Sm120, gemm::mxfp8_fold};
    use kernel_core::{
        backend::Backend, contract::DType, precision::execution::BlockScaledExecution,
    };

    #[test]
    fn keep_fallback_capacity_and_reject_conflicting_row_routes() {
        let q = Mxfp8FoldGemmQuery {
            m_capacity: 512,
            n: 128,
            k: 128,
            groups: 1,
            activation_row_stride: 128,
            activation_group_stride: 128,
            output: DType::F32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        };
        let base = mxfp8_fold::Entry::new(q, "compact_route_test").unwrap();
        let old = || Sm120::new().with_mxfp8_prefill(base.clone());
        let e = Entry::new(base.clone(), 2).unwrap();
        let backend = old().with_mxfp8_prefill_compact(e.clone(), 128);
        let op = backend.mxfp8_prefill_gemm(&q, Need::rows(512)).unwrap();
        let fallback = old().mxfp8_prefill_gemm(&q, Need::rows(512)).unwrap();
        let v = |handle, bytes| View { handle, bytes };
        let sizes = base.bytes(512);
        let b = Buffers {
            activations: v(1, sizes.activations),
            activation_scales: v(2, sizes.activation_scales),
            weights: v(3, sizes.weights),
            weight_scales: v(4, sizes.weight_scales),
            output: v(5, sizes.output),
            live_rows: v(6, 4),
            status: v(7, sizes.status),
            status_row_offset: 0,
        };
        for rows in [0, 1, 127, 129, 256, 512] {
            assert_eq!(
                op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap(),
                fallback
                    .launches(&b, View::EMPTY, Counts::rows(rows))
                    .unwrap()
            );
        }
        assert_eq!(
            op.launches(&b, View::EMPTY, Counts::rows(128)).unwrap(),
            e.prepare(128)
                .unwrap()
                .launches(&b, View::EMPTY, Counts::rows(128))
                .unwrap()
        );
        assert!(op.launches(&b, View::EMPTY, Counts::rows(513)).is_err());
        let shorter = backend.mxfp8_prefill_gemm(&q, Need::rows(64)).unwrap();
        assert_eq!(
            shorter.info(),
            old().mxfp8_prefill_gemm(&q, Need::rows(64)).unwrap().info()
        );
        assert!(
            backend
                .with_mxfp8_prefill_compact(Entry::new(base.clone(), 3).unwrap(), 128)
                .mxfp8_prefill_gemm(&q, Need::rows(512))
                .is_err()
        );
        for rows in [0, 513] {
            assert!(
                old()
                    .with_mxfp8_prefill_compact(e.clone(), rows)
                    .mxfp8_prefill_gemm(&q, Need::rows(512))
                    .is_err()
            );
        }
    }
}
