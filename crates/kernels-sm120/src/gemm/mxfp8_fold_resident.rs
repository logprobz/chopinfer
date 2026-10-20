//! Register-resident ordered folds for small decode rows.
use super::mxfp8_fold_stream;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, Refusal};

#[derive(Clone, Debug)]
pub struct Entry {
    pub stream: mxfp8_fold_stream::Entry,
    pub tile_n: u32,
    /// Order violation used only by a lab negative control.
    pub reverse: bool,
}

impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.stream.validate()?;
        if self.stream.ticket.paired
            || self.stream.ticket.query.output != DType::F32
            || ![8, 16, 32].contains(&self.tile_n)
        {
            return Err(Refusal(
                "resident folds require single F32 outputs and 8/16/32 columns",
            ));
        }
        Ok(())
    }

    pub fn symbol(&self) -> String {
        format!(
            "{}_resident_n{}_{}",
            self.stream.ticket.symbol,
            self.tile_n,
            if self.reverse { "reverse_v1" } else { "v1" }
        )
    }

    pub fn plans(&self, rows: u32) -> Result<[LaunchPlan; 2], Refusal> {
        self.validate()?;
        let q = self.stream.ticket.query;
        if rows == 0 || rows > 16 || rows > q.m_capacity {
            return Err(Refusal("resident fold rows must be within 1..=16"));
        }
        let plan = |grid, threads| LaunchPlan {
            grid,
            block: [threads, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 64,
            cluster: [1, 1, 1],
            pdl: true,
        };
        Ok([
            plan([rows, 1, 1], 128),
            plan([q.n.div_ceil(self.tile_n), 1, q.groups], self.tile_n * 4),
        ])
    }

    pub fn specs(&self) -> Vec<EntrySpec> {
        let q = self.stream.ticket.query;
        let mut compute = self.stream.specs().pop().unwrap();
        compute.symbol = self.symbol();
        compute.max_threads = self.tile_n * 4;
        let mut admit = compute.clone();
        admit.symbol.push_str("_admit_v1");
        admit.max_threads = 128;
        let ns = "chop::gemm::mxfp8_fold_resident";
        admit.body = format!(
            "chop::ptx::griddep_launch_dependents();chop::ptx::griddep_wait();{ns}::admit_task<{},{},{},{}>(blockIdx.x,activations,activation_scales,reinterpret_cast<uint32_t*>(workspace),live_rows,status,status_row_offset,rows);",
            q.k, q.groups, q.activation_row_stride, q.activation_group_stride
        );
        compute.body = format!(
            "using namespace {ns};const Task task{{blockIdx.x,blockIdx.y,blockIdx.z,gridDim.x,gridDim.y}};__shared__ Shared arena;chop::ptx::griddep_launch_dependents();auto first=chop::gemm::mxfp8_fold_stream::weight_fragments<{},{},{},{},false>(task,0,{},1,packed,weight_scales,nullptr,nullptr);chop::ptx::griddep_wait();run_task<{},{},{},{},{},{},{}>(task,arena,first,activations,activation_scales,packed,weight_scales,reinterpret_cast<float*>(output),reinterpret_cast<const uint32_t*>(workspace),live_rows,rows);",
            q.n,
            q.k,
            q.groups,
            self.tile_n,
            q.k.div_ceil(128),
            q.n,
            q.k,
            q.groups,
            q.activation_row_stride,
            q.activation_group_stride,
            self.tile_n,
            self.reverse
        );
        vec![admit, compute]
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("resident fold cubin requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        if !symbols.insert(entry.symbol()) {
            return Err(Refusal("duplicate resident fold symbol"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/mxfp8_fold_resident.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().flat_map(Entry::specs).collect(),
    })
}
