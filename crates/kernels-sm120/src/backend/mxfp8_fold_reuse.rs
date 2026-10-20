//! Checked binding of row-reuse folds to one explicitly selected row bucket.
use crate::gemm::mxfp8_fold_reuse::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument as A, ArgumentList, Refusal};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};

pub(super) fn prepare(
    entries: &[(Entry, u32)],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
    let mut matches = entries.iter().filter(|(e, _)| e.ticket.query == *query);
    let (entry, folds) = matches.next().ok_or(Refused::Unsupported)?;
    if matches.next().is_some() || entry.ticket.paired || need.capacity != query.m_capacity {
        return Err(Refused::Rule(
            "row-reuse folds require one exact row bucket".into(),
        ));
    }
    entry.plans(need.capacity, *folds)?;
    let legacy = super::mxfp8_fold::prepare(&[entry.ticket.legacy()?], query, need)?;
    let mut info = legacy.info().clone();
    info.implementation = format!("sm120.mxfp8-fold.reuse32.v1:{entry:?}:folds={folds}");
    info.symbols = entry
        .specs()
        .iter()
        .map(|s| (s.symbol.clone(), 0))
        .collect();
    info.workspace = entry.workspace(need.capacity)?;
    need.check(&info)?;
    Ok(Box::new(Op {
        entry: entry.clone(),
        folds: *folds,
        legacy,
        info,
    }))
}

struct Op {
    entry: Entry,
    folds: u32,
    legacy: Prepared<Mxfp8FoldGemmBuffers<View>>,
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
        if counts.rows != self.entry.ticket.query.m_capacity {
            return Err(Refusal(
                "row-reuse launch differs from its selected row bucket",
            ));
        }
        super::mxfp8_fold_decode::validate_launch(
            &self.legacy,
            self.info.workspace,
            b,
            None,
            workspace,
            counts,
        )?;
        let d = |v: View| A::DeviceBuffer(v.handle);
        let args = [
            vec![
                d(b.activations),
                d(b.activation_scales),
                d(workspace),
                d(b.live_rows),
                d(b.status),
                A::U64(b.status_row_offset),
                A::U32(counts.rows),
            ],
            vec![
                d(b.activations),
                d(b.activation_scales),
                d(b.weights),
                d(b.weight_scales),
                A::NullDevicePointer,
                A::NullDevicePointer,
                d(b.output),
                A::NullDevicePointer,
                d(workspace),
                d(b.live_rows),
                A::U32(counts.rows),
                A::U32(self.folds),
            ],
        ];
        Ok(self
            .entry
            .specs()
            .into_iter()
            .zip(self.entry.plans(counts.rows, self.folds)?)
            .zip(args)
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
