//! Two independent row tiles over the unchanged local-wide fold arithmetic.
use super::{mxfp8_fold_local_wide, mxfp8_fold_packed};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{
    backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View},
    contract::{Argument, LaunchPlan, Refusal},
    mxfp8_gemm::Mxfp8FoldGemmBuffers,
    properties::Invariance,
};

pub const CAPACITY: u32 = 64;
#[derive(Clone, Debug)]
pub struct Candidate {
    entry: mxfp8_fold_local_wide::Entry,
}
impl Candidate {
    pub fn new(entry: mxfp8_fold_local_wide::Entry) -> Result<Self, Refusal> {
        entry.validate()?;
        if entry.base.query.m_capacity != 32 || entry.stages < 3 {
            return Err(Refusal("tiled local folds require a 32-row deep ring"));
        }
        Ok(Self { entry })
    }
    pub fn entry(&self) -> &mxfp8_fold_local_wide::Entry {
        &self.entry
    }
    pub fn query(&self) -> kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
        kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
            m_capacity: CAPACITY,
            ..self.entry.base.query
        }
    }
    pub fn symbol(&self) -> String {
        format!("{}_rows64_v1", self.entry.symbol())
    }
    pub fn packed_symbol(&self) -> String {
        mxfp8_fold_packed::symbol(&self.symbol())
    }
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if !(33..=CAPACITY).contains(&rows) {
            return Err(Refusal("tiled local folds require 33 to 64 planned rows"));
        }
        let mut plan = self.entry.plan(32)?;
        plan.grid[1] = rows.div_ceil(32);
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }
    fn spec_impl(&self, packed: bool) -> Result<EntrySpec, Refusal> {
        let q = self.entry.base.query;
        let mut spec = if packed {
            self.entry.packed_spec()?
        } else {
            self.entry.spec()
        };
        spec.symbol = if packed {
            self.packed_symbol()
        } else {
            self.symbol()
        };
        // Wait before reading the live count. The old task publishes its
        // dependency and retains its own wait and K-fold ordering.
        spec.body = format!(
            "__shared__ uint32_t bounded_live;chop::ptx::griddep_wait();const uint32_t first=blockIdx.y*32u;const uint32_t count=min(32u,rows-first);if(threadIdx.x==0){{const uint32_t live=min(rows,*live_rows);bounded_live=live>first?min(count,live-first):0u;}}__syncthreads();chop::gemm::mxfp8_fold_local_wide::run_deep<{},{},{},{},{},{},{},true,false,{}>(activations+uint64_t(first)*{},activation_scales+uint64_t(first)*{},weights,weight_scales,output+uint64_t(first)*{},workspace+uint64_t(first)*4,&bounded_live,count);",
            q.n,
            q.k,
            q.groups,
            q.activation_row_stride,
            q.activation_group_stride,
            self.entry.tile_n,
            self.entry.stages,
            packed,
            q.activation_row_stride,
            q.activation_row_stride / 32,
            u64::from(q.n) * u64::from(q.groups)
        );
        Ok(spec)
    }
    pub fn spec(&self) -> EntrySpec {
        self.spec_impl(false).expect("validated local tile")
    }
    pub fn packed_spec(&self) -> Result<EntrySpec, Refusal> {
        self.spec_impl(true)
    }
    pub fn launch(&self, b: &Mxfp8FoldGemmBuffers<View>, rows: u32) -> Result<Launch, Refusal> {
        let plan = self.plan(rows)?;
        let size = self.entry.base.bytes(rows);
        b.activations.covers(size.activations)?;
        b.activation_scales.covers(size.activation_scales)?;
        b.output.covers(size.output)?;
        b.status.covers(u64::from(rows) * 4)?;
        // The original validator checks the remaining roles, distinct
        // handles and the zero status-row offset required by this task.
        let mut call = self.entry.launch(b, 32)?;
        if call.arguments.0.len() != 12 || call.arguments.0[10] != Argument::U32(32) {
            return Err(Refusal("local fold entry argument layout changed"));
        }
        call.symbol = self.symbol();
        call.plan = plan;
        call.rows = rows;
        call.arguments.0[10] = Argument::U32(rows);
        Ok(call)
    }
    pub fn prepare(&self, need: Need) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
        self.plan(need.capacity)?;
        let mut info = self.entry.prepare(Need::rows(32))?.info().clone();
        info.implementation = "sm120.mxfp8-fold.local-tiled.v1".into();
        info.guarantees = Invariance::any_rows(CAPACITY);
        info.symbols = vec![(self.symbol(), 0)];
        need.check(&info)?;
        Ok(Box::new(Op {
            candidate: self.clone(),
            capacity: need.capacity,
            info,
        }))
    }
}
struct Op {
    candidate: Candidate,
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
            return Err(Refusal("tiled local fold exceeds prepared rows"));
        }
        Ok(vec![self.candidate.launch(b, counts.rows)?])
    }
}
pub fn cubin_spec(name: &str, candidates: &[Candidate]) -> Result<CubinSpec, Refusal> {
    if candidates.is_empty() {
        return Err(Refusal("tiled local folds require entries"));
    }
    let mut entries = Vec::new();
    for c in candidates {
        entries.extend([c.spec(), c.packed_spec()?]);
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/mxfp8_fold_local_wide.cuh\"".into()],
        defines: vec![],
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::super::mxfp8_fold;
    use super::*;
    use kernel_core::{
        contract::DType, mxfp8_gemm::Mxfp8FoldGemmQuery, precision::execution::BlockScaledExecution,
    };
    #[test]
    fn tiled_rows_cover_full_buffers_and_preserve_the_entry_abi() {
        let q = Mxfp8FoldGemmQuery {
            m_capacity: 32,
            n: 104,
            k: 384,
            groups: 1,
            activation_row_stride: 384,
            activation_group_stride: 384,
            output: DType::F32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        };
        let c = Candidate::new(mxfp8_fold_local_wide::Entry {
            base: mxfp8_fold::Entry::new(q, "test_local_tiled").unwrap(),
            tile_n: 48,
            stages: 3,
        })
        .unwrap();
        for rows in 33..=64 {
            let size = c.entry.base.bytes(rows);
            let v = |handle, bytes| View { handle, bytes };
            let b = Mxfp8FoldGemmBuffers {
                activations: v(1, size.activations),
                activation_scales: v(2, size.activation_scales),
                weights: v(3, size.weights),
                weight_scales: v(4, size.weight_scales),
                output: v(5, size.output),
                live_rows: v(6, 4),
                status: v(7, u64::from(rows) * 4),
                status_row_offset: 0,
            };
            let call = c.launch(&b, rows).unwrap();
            assert_eq!(call.plan.grid, [3, 2, 1]);
            assert_eq!(call.arguments.0[10], Argument::U32(rows));
            let old = c.entry.launch(&b, 32).unwrap();
            assert_eq!(&call.arguments.0[..10], &old.arguments.0[..10]);
            assert_eq!(call.arguments.0[11], old.arguments.0[11]);
            let mut short = b;
            short.output.bytes -= 1;
            assert!(c.launch(&short, rows).is_err());
            let mut short = b;
            short.status.bytes -= 1;
            assert!(c.launch(&short, rows).is_err());
            let mut offset = b;
            offset.status_row_offset = 1;
            assert!(c.launch(&offset, rows).is_err());
            let op = c.prepare(Need::rows(rows)).unwrap();
            assert_eq!(
                op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap(),
                [call]
            );
            assert!(
                op.launches(&b, View::EMPTY, Counts::rows(rows + 1))
                    .is_err()
            );
        }
        for rows in [0, 1, 32, 65] {
            assert!(c.plan(rows).is_err());
        }
        let spec = c.packed_spec().unwrap();
        assert_eq!(spec.params.len(), 12);
        assert!(spec.body.find("griddep_wait()").unwrap() < spec.body.find("*live_rows").unwrap());
        assert!(spec.body.contains("workspace+uint64_t(first)*4"));
        assert!(spec.body.contains("output+uint64_t(first)*104"));
    }
}
