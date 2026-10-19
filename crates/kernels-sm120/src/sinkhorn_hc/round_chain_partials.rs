//! Per-chain rounded HC producers that keep the served workspace layout.
use super::ptr;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, Refusal};
#[derive(Clone, Debug)]
pub struct Entry {
    pub reference: super::expand_mix::Entry,
    pub prefix: String,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.validate()?;
        if self.reference.probe || !self.reference.query.mix.width.is_multiple_of(128) {
            return Err(Refusal(
                "chain producer needs complete hidden chains and no probe outputs",
            ));
        }
        Ok(())
    }
    pub fn symbols(&self) -> [String; 2] {
        [4, 8].map(|w| format!("{}_w{w}_v1", self.prefix))
    }
    pub fn plan(&self, rows: u32, live: u32, warps: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if live > rows || ![4, 8].contains(&warps) {
            return Err(Refusal("invalid chain producer rows or warp count"));
        }
        let mut p = self.reference.plans(rows)?[0];
        p.grid[0] = self.reference.query.mix.streams * self.reference.query.mix.width / 128;
        p.block = [warps * 32, 1, 1];
        Ok(p)
    }
    pub fn cubin_spec(&self, name: &str) -> Result<CubinSpec, Refusal> {
        self.validate()?;
        let m = self.reference.query.mix;
        let traits = format!(
            "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,{}>",
            m.streams,
            m.width,
            m.iterations,
            m.eps_bits,
            m.norm_eps_bits,
            m.fn_dtype == DType::F32
        );
        let entries=self.symbols().into_iter().zip([4,8]).map(|(symbol,warps)|{
            let threads=warps*32;
            EntrySpec::new(symbol,vec![ptr("x","const uint16_t"),ptr("sum","const float"),ptr("rounded","uint16_t"),ptr("post_in","const float"),ptr("comb_in","const float"),ptr("expanded","uint16_t"),ptr("fn","const void"),ptr("workspace","float"),ptr("live_rows","const uint32_t")],format!("using M={traits};
                if (blockDim.x!={threads} || gridDim.x!=M::kChains) asm volatile(\"trap;\");
                chop::ptx::griddep_wait();
                chop::sinkhorn_hc::round_chain::task<M,{warps}>(x,sum,rounded,post_in,comb_in,expanded,fn,workspace,live_rows,blockIdx.x,blockIdx.y,nullptr);
                chop::ptx::griddep_launch_dependents();"),threads,1)
        }).collect();
        Ok(CubinSpec {
            name: name.into(),
            sources_root: crate::cuda_source_root(),
            include_dirs: vec![],
            headers: vec!["#include \"sinkhorn_hc/round_chain_partials.cuh\"".into()],
            defines: vec![],
            entries,
        })
    }
}
