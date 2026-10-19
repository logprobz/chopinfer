//! Lab fusion of a rounded HC finisher and the following cached input norm.
use super::ptr;
use crate::row::rmsnorm_quantize_cached;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, Refusal};

#[derive(Clone, Debug)]
pub struct Entry {
    pub mixer: super::expand_mix::Entry,
    pub norm: rmsnorm_quantize_cached::Entry,
    pub prefix: String,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.mixer.validate()?;
        self.norm.validate()?;
        let m = self.mixer.query.mix;
        if self.mixer.probe
            || m.streams != 4
            || m.width != self.norm.reference.query.norm.width
            || m.width > 8192
            || self.mixer.capacity != self.norm.reference.capacity
        {
            return Err(Refusal(
                "finish norm needs matching four-stream BF16 row contracts",
            ));
        }
        Ok(())
    }
    pub fn symbols(&self) -> [String; 2] {
        [16, 32].map(|w| format!("{}_w{w}_v1", self.prefix))
    }
    pub fn plan(&self, rows: u32, live: u32, warps: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if ![16, 32].contains(&warps) {
            return Err(Refusal("invalid finish norm warp count"));
        }
        let mut p = self.norm.plan(rows, live, warps)?;
        p.workspace_bytes = self.mixer.plans(rows)?[1].workspace_bytes;
        Ok(p)
    }
    pub fn cubin_spec(&self, name: &str) -> Result<CubinSpec, Refusal> {
        self.validate()?;
        let m = self.mixer.query.mix;
        let floor = self.norm.reference.query.amax_floor_bits;
        let traits = format!(
            "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,{}>",
            m.streams,
            m.width,
            m.iterations,
            m.eps_bits,
            m.norm_eps_bits,
            m.fn_dtype == DType::F32
        );
        let entries=self.symbols().into_iter().zip([16,32]).map(|(symbol,warps)| {
            let mut params=vec![ptr("expanded","const uint16_t"),ptr("base","const float"),ptr("scale","const float"),ptr("pre_in","const float"),ptr("collapsed","uint16_t"),ptr("pre","float"),ptr("post","float"),ptr("comb","float"),ptr("workspace","const float"),ptr("live_rows","const uint32_t"),ptr("weight","const uint16_t"),ptr("normed","uint16_t"),ptr("widened","float"),ptr("codes","uint8_t"),ptr("scales","uint8_t"),ptr("status","uint32_t")];
            params.extend([EntryParam::U64 { name: "status_row_offset".into() },EntryParam::U32 { name: "eps_bits".into() }]);
            EntrySpec::new(symbol,params,format!("using M={traits};
                using namespace chop::sinkhorn_hc::finish_norm;
                if (blockDim.x!={warps}*32 || gridDim.x!=1) asm volatile(\"trap;\");
                __shared__ __align__(16) unsigned char arena[sizeof(Arena<M,{warps}>)];
                chop::ptx::griddep_wait();
                task<M,{warps},{floor:#x}u>(expanded,base,scale,pre_in,collapsed,pre,post,comb,workspace,live_rows,weight,normed,widened,codes,scales,status,status_row_offset,__uint_as_float(eps_bits),blockIdx.x,blockIdx.y,arena);
                chop::ptx::griddep_launch_dependents();"),warps*32,1)
        }).collect();
        Ok(CubinSpec {
            name: name.into(),
            sources_root: crate::cuda_source_root(),
            include_dirs: vec![],
            headers: vec!["#include \"sinkhorn_hc/finish_norm.cuh\"".into()],
            defines: vec![],
            entries,
        })
    }
}
