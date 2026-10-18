//! Dense and grouped native MXFP8 fold entries behind the backend contract.

use super::select::{Candidate, choose};
use crate::gemm::mxfp8_fold::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Refusal, Validate};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
use kernel_core::properties::Invariance;

fn key(entry: &Entry) -> String {
    format!("{entry:?}")
}

pub(super) fn prepare(
    entries: &[Entry],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
    query.validate()?;
    if need.capacity == 0 || need.capacity > query.m_capacity || need.capacity > super::MAX_CAPACITY
    {
        return Err(Refusal("MXFP8 fold preparation exceeds the query capacity").into());
    }
    let chosen = choose(
        "MXFP8 fold GEMM",
        entries.iter().filter(|e| e.query == *query).map(|entry| {
            Candidate::new(entry.clone(), &entry.symbol, key(entry), entry.invariance())
                .capacity(entry.query.m_capacity)
        }),
        need,
    )?;
    let entry = chosen.entry;
    entry.validate()?;
    let b = entry.bytes(need.capacity);
    let mut info = OpInfo::new(
        format!("sm120.mxfp8-fold.native.v2:{query:?}"),
        chosen.guarantees.meet(Invariance::any_rows(need.capacity)),
        vec![(entry.symbol.clone(), 0)],
    );
    info.bytes = vec![
        ("activations".into(), b.activations),
        ("activation_scales".into(), b.activation_scales),
        ("weights".into(), b.weights),
        ("weight_scales".into(), b.weight_scales),
        ("output".into(), b.output),
        ("live_rows".into(), 4),
        ("status".into(), b.status),
    ];
    info.weight_layouts = vec![("weights".into(), entry.weight_layout())];
    need.check(&info)?;
    Ok(Box::new(Op {
        entry,
        capacity: need.capacity,
        info,
    }))
}

struct Op {
    entry: Entry,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<Mxfp8FoldGemmBuffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &Mxfp8FoldGemmBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows > self.capacity {
            return Err(Refusal("MXFP8 fold launch exceeds prepared capacity"));
        }
        let bytes = self.entry.bytes(counts.rows);
        b.activations.covers(bytes.activations)?;
        b.activation_scales.covers(bytes.activation_scales)?;
        b.weights.covers(bytes.weights)?;
        b.weight_scales.covers(bytes.weight_scales)?;
        b.output.covers(bytes.output)?;
        b.live_rows.covers(4)?;
        let status_bytes = b
            .status_row_offset
            .checked_add(u64::from(counts.rows))
            .and_then(|rows| rows.checked_mul(4))
            .filter(|&bytes| bytes <= kernel_core::precision::MAX_TENSOR_BYTES)
            .ok_or(Refusal(
                "MXFP8 fold status row offset exceeds the supported bound",
            ))?;
        b.status.covers(status_bytes)?;
        if [
            b.activations.handle,
            b.activation_scales.handle,
            b.weights.handle,
            b.weight_scales.handle,
            b.live_rows.handle,
            b.status.handle,
        ]
        .contains(&b.output.handle)
        {
            return Err(Refusal("MXFP8 fold output aliases an input"));
        }
        if [
            b.activations.handle,
            b.activation_scales.handle,
            b.weights.handle,
            b.weight_scales.handle,
            b.live_rows.handle,
        ]
        .contains(&b.status.handle)
        {
            return Err(Refusal("MXFP8 fold status aliases an input"));
        }
        let handles = Mxfp8FoldGemmBuffers {
            activations: b.activations.handle,
            activation_scales: b.activation_scales.handle,
            weights: b.weights.handle,
            weight_scales: b.weight_scales.handle,
            output: b.output.handle,
            live_rows: b.live_rows.handle,
            status: b.status.handle,
            status_row_offset: b.status_row_offset,
        };
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows: counts.rows,
            plan: self.entry.plan(counts.rows)?,
            arguments: self.entry.arguments(handles, counts.rows),
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Sm120;
    use kernel_core::backend::Backend;
    use kernel_core::precision::execution::BlockScaledExecution;

    fn buffers(e: &Entry) -> Mxfp8FoldGemmBuffers<View> {
        let b = e.bytes(e.query.m_capacity);
        let v = |handle, bytes| View { handle, bytes };
        Mxfp8FoldGemmBuffers {
            activations: v(1, b.activations),
            activation_scales: v(2, b.activation_scales),
            weights: v(3, b.weights),
            weight_scales: v(4, b.weight_scales),
            output: v(5, b.output),
            live_rows: v(6, 4),
            status: v(7, b.status),
            status_row_offset: 0,
        }
    }

    #[test]
    fn prepares_native_rows_and_groups_and_refuses_false_execution() {
        for e in crate::gemm::mxfp8_fold::lab_entries() {
            let q = e.query;
            let backend = Sm120::new().with_mxfp8_fold(e.clone());
            let op = backend
                .mxfp8_fold_gemm(&q, Need::rows(q.m_capacity))
                .unwrap();
            assert_eq!(op.info().layout_of("weights"), Some(&e.weight_layout()));
            assert_eq!(
                op.info().bytes_of("weight_scales"),
                Some(u64::from(q.groups) * u64::from(q.n) * u64::from(q.k / 32))
            );
            let mut b = buffers(&e);
            for rows in [0, 1, q.m_capacity] {
                let launch = op
                    .launches(&b, View::EMPTY, Counts::rows(rows))
                    .unwrap()
                    .remove(0);
                assert!(launch.plan.pdl);
                assert_eq!(launch.plan.grid[2], q.groups);
                assert_eq!(
                    launch.arguments.0.last(),
                    Some(&kernel_core::contract::Argument::U32(rows))
                );
            }
            assert!(
                op.launches(&b, View::EMPTY, Counts::rows(q.m_capacity + 1))
                    .is_err()
            );
            b.weight_scales.bytes -= 1;
            assert!(op.launches(&b, View::EMPTY, Counts::rows(1)).is_err());
            b = buffers(&e);
            b.output.handle = b.activations.handle;
            assert!(op.launches(&b, View::EMPTY, Counts::rows(1)).is_err());
            b = buffers(&e);
            b.status_row_offset = 1;
            assert!(
                op.launches(&b, View::EMPTY, Counts::rows(q.m_capacity))
                    .is_err()
            );
            b.status.bytes += 4;
            assert!(
                op.launches(&b, View::EMPTY, Counts::rows(q.m_capacity))
                    .is_ok()
            );
            b.status_row_offset = u64::MAX;
            assert!(op.launches(&b, View::EMPTY, Counts::rows(1)).is_err());
            b = buffers(&e);
            b.status.handle = b.weights.handle;
            assert!(op.launches(&b, View::EMPTY, Counts::rows(1)).is_err());
            let mut false_claim = q;
            false_claim.execution = BlockScaledExecution::ExactDecodedBf16Mma;
            assert!(Entry::new(false_claim, "false_native").is_err());
            assert!(
                backend
                    .mxfp8_fold_gemm(&false_claim, Need::rows(1))
                    .is_err()
            );
        }
    }

    #[test]
    fn selection_checks_offsets_output_and_ambiguous_entries() {
        let e = crate::gemm::mxfp8_fold::lab_entries()
            .into_iter()
            .find(|e| e.query.groups == 2)
            .unwrap();
        let mut other = e.clone();
        other.symbol = "other_native_fold".into();
        for pair in [[e.clone(), other.clone()], [other, e.clone()]] {
            let backend = pair.into_iter().fold(Sm120::new(), Sm120::with_mxfp8_fold);
            assert!(backend.mxfp8_fold_gemm(&e.query, Need::rows(1)).is_err());
        }
        let backend = Sm120::new().with_mxfp8_fold(e.clone());
        for q in [
            Mxfp8FoldGemmQuery {
                activation_group_stride: e.query.k,
                ..e.query
            },
            Mxfp8FoldGemmQuery {
                n: e.query.n + 1,
                ..e.query
            },
        ] {
            q.validate().unwrap();
            assert!(backend.mxfp8_fold_gemm(&q, Need::rows(1)).is_err());
        }
    }
}
