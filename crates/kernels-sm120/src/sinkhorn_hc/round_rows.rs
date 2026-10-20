//! Exact rounded projection chains sharing weights across independent rows.
use super::ptr;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, Refusal};

#[derive(Clone, Debug)]
pub struct Entry {
    pub reference: super::round_chain_partials::Entry,
    pub prefix: String,
}

impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.validate()
    }
    pub fn symbols(&self) -> [String; 3] {
        [2, 4, 8].map(|rows| format!("{}_r{rows}_v1", self.prefix))
    }
    pub fn plan(&self, rows: u32, live: u32, tile: u32) -> Result<LaunchPlan, Refusal> {
        if ![2, 4, 8].contains(&tile) {
            return Err(Refusal("unsupported rounded producer row tile"));
        }
        let mut plan = self.reference.plan(rows, live, 4)?;
        plan.grid[1] = rows.div_ceil(tile);
        plan.block = [128 * tile, 1, 1];
        Ok(plan)
    }
    pub fn cubin_spec(&self, name: &str) -> Result<CubinSpec, Refusal> {
        self.validate()?;
        let m = self.reference.reference.query.mix;
        let traits = format!(
            "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,{}>",
            m.streams,
            m.width,
            m.iterations,
            m.eps_bits,
            m.norm_eps_bits,
            m.fn_dtype == DType::F32
        );
        let entries = self
            .symbols()
            .into_iter()
            .zip([2, 4, 8])
            .map(|(symbol, rows)| {
                let threads = rows * 128;
                EntrySpec::new(
                    symbol,
                    vec![
                        ptr("x", "const uint16_t"),
                        ptr("sum", "const float"),
                        ptr("rounded", "uint16_t"),
                        ptr("post_in", "const float"),
                        ptr("comb_in", "const float"),
                        ptr("expanded", "uint16_t"),
                        ptr("fn", "const void"),
                        ptr("workspace", "float"),
                        ptr("live_rows", "const uint32_t"),
                    ],
                    format!(
                        "using M={traits};
                __shared__ chop::sinkhorn_hc::round_rows::Arena<M,{rows}> arena;
                if (blockDim.x!={threads} || gridDim.x!=M::kChains) asm volatile(\"trap;\");
                chop::ptx::griddep_wait();
                chop::sinkhorn_hc::round_rows::task<M,{rows}>(x,sum,rounded,post_in,comb_in,
                    expanded,fn,workspace,live_rows,blockIdx.x,blockIdx.y,
                    reinterpret_cast<unsigned char*>(&arena));
                chop::ptx::griddep_launch_dependents();"
                    ),
                    threads,
                    1,
                )
            })
            .collect();
        Ok(CubinSpec {
            name: name.into(),
            sources_root: crate::cuda_source_root(),
            include_dirs: vec![],
            headers: vec!["#include \"sinkhorn_hc/round_rows.cuh\"".into()],
            defines: vec![],
            entries,
        })
    }
}
