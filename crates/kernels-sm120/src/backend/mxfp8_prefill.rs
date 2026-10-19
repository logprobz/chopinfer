//! Preserve the fold contract's validation and layouts, replacing only execution.
use super::mxfp8_fold;
use crate::gemm::{mxfp8_fold::Entry, mxfp8_prefill as tiled};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, Refusal};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};

pub(super) fn prepare(
    entries: &[Entry],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
    let inner = mxfp8_fold::prepare(entries, query, need)?;
    let entry = entries
        .iter()
        .find(|e| e.query == *query)
        .expect("prepared entry")
        .clone();
    tiled::plan(&entry, need.capacity)?;
    let mut info = inner.info().clone();
    info.implementation = format!("sm120.mxfp8-prefill.tma128-fold128.v1:{query:?}");
    info.symbols = vec![
        (tiled::validation_symbol(&entry), 0),
        (tiled::symbol(&entry), tiled::SHARED),
    ];
    need.check(&info)?;
    Ok(Box::new(Op { inner, entry, info }))
}
struct Op {
    inner: Prepared<Mxfp8FoldGemmBuffers<View>>,
    entry: Entry,
    info: OpInfo,
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
        self.inner.launches(b, workspace, counts)?;
        if counts.rows == 0 {
            return Ok(Vec::new());
        }
        let v = |x: View| Argument::DeviceBuffer(x.handle);
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
        Ok(vec![
            Launch {
                symbol: tiled::validation_symbol(&self.entry),
                rows: counts.rows,
                plan: tiled::validation_plan(&self.entry, counts.rows)?,
                arguments: ArgumentList(vec![
                    v(b.activations),
                    v(b.activation_scales),
                    v(b.live_rows),
                    v(b.status),
                    Argument::U64(b.status_row_offset),
                    Argument::U32(counts.rows),
                ]),
                collective: false,
                persistent: false,
            },
            Launch {
                symbol: tiled::symbol(&self.entry),
                rows: counts.rows,
                plan: tiled::plan(&self.entry, counts.rows)?,
                arguments: tiled::arguments(&self.entry, handles, counts.rows)?,
                collective: false,
                persistent: false,
            },
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::DType;
    use kernel_core::precision::execution::BlockScaledExecution;

    #[test]
    fn refuse_prefill_shapes_past_signed_tma_and_fold_bounds() {
        for (n, k, groups) in [(1 << 30, 32, 2), (1, i32::MAX as u32 - 31, 1)] {
            let q = Mxfp8FoldGemmQuery {
                m_capacity: 16,
                n,
                k,
                groups,
                activation_row_stride: k * groups,
                activation_group_stride: k,
                output: DType::F32,
                execution: BlockScaledExecution::NativeBlockScaledMma,
            };
            // These are legal for the old pointer-based engine, but not
            // for the staged engine's signed coordinates and fold count.
            let entry = Entry::new(q, "chop_prefill_bound_v1").unwrap();
            assert!(prepare(std::slice::from_ref(&entry), &q, Need::rows(16)).is_err());
            assert!(tiled::specs(&entry).is_err());
        }
    }
}
