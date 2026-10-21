//! TMA code delivery and producer-staged packed scales for ordered FP8 folds.
use super::{mxfp8_fold, mxfp8_prefill};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::backend::{Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, View};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers as Buffers;

#[derive(Clone, Debug)]
pub struct Entry {
    pub base: mxfp8_fold::Entry,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        let q = self.base.query;
        if q.k % 128 != 0
            || q.activation_row_stride % 128 != 0
            || q.activation_group_stride % 128 != 0
        {
            return Err(Refusal(
                "staged scales require complete aligned 128-K folds",
            ));
        }
        mxfp8_prefill::specs(&self.base)?;
        let mut plan = mxfp8_prefill::plan(&self.base, 1)?;
        plan.dynamic_shared_bytes = self.shared_bytes();
        crate::backend::check_plan(&plan)
    }
    pub fn shared_bytes(&self) -> u32 {
        2 * (32768 + 1024) + 1024
    }
    pub fn symbol(&self) -> String {
        format!("{}_tma_scales_s2_v1", self.base.symbol)
    }
    pub fn specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.validate()?;
        let mut specs = mxfp8_prefill::specs(&self.base)?;
        specs[0].symbol = format!("{}_admit_v1", self.symbol());
        specs[1].symbol = self.symbol();
        for spec in &mut specs {
            spec.body = spec
                .body
                .replace("mxfp8_prefill::", "mxfp8_prefill_tma_scales::");
        }
        specs[1].body = specs[1]
            .body
            .replace(",false>(activation_map", ",false,2>(activation_map");
        let mut control = specs[1].clone();
        control.symbol.push_str("_omit_first_v1");
        control.body = control
            .body
            .replace(",false,2>(activation_map", ",true,2>(activation_map");
        specs.push(control);
        Ok(specs)
    }
    pub fn prepare(&self, rows: u32) -> Result<Prepared<Buffers<View>>, Refusal> {
        self.validate()?;
        let fallback = crate::backend::Sm120::new()
            .with_mxfp8_prefill(self.base.clone())
            .mxfp8_prefill_gemm(&self.base.query, Need::rows(rows))
            .map_err(|_| Refusal("TMA scales fallback rejected the request"))?;
        let mut info = fallback.info().clone();
        info.implementation = format!("sm120.mxfp8-prefill.tma-scales.v1:{self:?}");
        info.symbols = vec![
            (format!("{}_admit_v1", self.symbol()), 0),
            (self.symbol(), self.shared_bytes()),
        ];
        Ok(Box::new(Op {
            entry: self.clone(),
            fallback,
            info,
        }))
    }
}
struct Op {
    entry: Entry,
    fallback: Prepared<Buffers<View>>,
    info: OpInfo,
}
impl PreparedOp<Buffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &Buffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut calls = self.fallback.launches(b, workspace, counts)?;
        for (i, call) in calls.iter_mut().enumerate() {
            call.symbol = self.info.symbols[i].0.clone();
            if i == 1 {
                call.plan.dynamic_shared_bytes = self.entry.shared_bytes();
            }
            crate::backend::check_plan(&call.plan)?;
        }
        Ok(calls)
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("TMA scales cubin requires entries"));
    }
    let mut bases = entries.iter().map(|e| e.base.clone()).collect::<Vec<_>>();
    bases.sort_by(|a, b| a.symbol.cmp(&b.symbol));
    bases.dedup_by(|a, b| a.symbol == b.symbol && a.query == b.query);
    let mut spec = mxfp8_fold::cubin_spec(name, &bases)?;
    spec.headers = vec!["#include \"gemm/mxfp8_prefill_tma_scales.cuh\"".into()];
    spec.entries = entries
        .iter()
        .map(Entry::specs)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{
        contract::DType, mxfp8_gemm::Mxfp8FoldGemmQuery, precision::execution::BlockScaledExecution,
    };
    #[test]
    fn staged_scales_preserve_tma_maps_and_live_row_contracts() {
        let q = Mxfp8FoldGemmQuery {
            m_capacity: 8192,
            n: 1024,
            k: 4096,
            groups: 2,
            activation_row_stride: 8192,
            activation_group_stride: 4096,
            output: DType::F32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        };
        let base = mxfp8_fold::Entry::new(q, "tma_matrix_test").unwrap();
        let b = base.bytes(8192);
        let v = |handle, bytes| View { handle, bytes };
        let buffers = Buffers {
            activations: v(1, b.activations),
            activation_scales: v(2, b.activation_scales),
            weights: v(3, b.weights),
            weight_scales: v(4, b.weight_scales),
            output: v(5, b.output),
            live_rows: v(6, 4),
            status: v(7, b.status),
            status_row_offset: 0,
        };
        {
            let entry = Entry { base: base.clone() };
            let op = entry.prepare(8192).unwrap();
            let old = crate::backend::Sm120::new()
                .with_mxfp8_prefill(base.clone())
                .mxfp8_prefill_gemm(&q, Need::rows(8192))
                .unwrap();
            for rows in [0, 65, 128, 192, 2496, 8192] {
                let expected = old
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let calls = op
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                assert_eq!(calls.len(), expected.len());
                for (a, b) in calls.iter().zip(expected) {
                    assert_eq!(a.arguments, b.arguments);
                    assert_eq!(a.plan.grid, b.plan.grid);
                    assert_eq!(a.plan.block, b.plan.block);
                    assert!(a.plan.dynamic_shared_bytes <= 101376);
                }
            }
            assert!(
                op.launches(&buffers, View::EMPTY, Counts::rows(8193))
                    .is_err()
            );
        }
        let mut unaligned = base;
        unaligned.query.k = 4000;
        assert!(Entry { base: unaligned }.validate().is_err());
    }
}
