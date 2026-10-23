//! The served HC finisher and cached input norm on two CTAs per row.
//!
//! The 1024-thread fused tail (`finish_norm`) runs the fold and Sinkhorn on
//! one warp, then the whole CTA runs the cached norm, although the norm reads
//! only the collapse, which uses the previous `pre`. This launch keeps the
//! tail's ABI and arithmetic and runs the mixer half and the collapse plus
//! norm on separate CTAs of each row, so neither waits for the other. It
//! replaces only the tail; the chain producer before it is unchanged.
use super::{completion::Mutation, finish_norm};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{
    backend::Launch,
    contract::{DType, LaunchPlan, Refusal},
};

/// Warp counts per CTA; the mixer half needs the 256 original RMS lanes.
pub const WARPS: [u32; 3] = [8, 16, 32];

#[derive(Clone, Debug)]
pub struct Entry {
    /// The served 1024-thread tail this launch replaces.
    pub reference: finish_norm::Entry,
    pub prefix: String,
    pub mutation: Mutation,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.validate()
    }
    pub fn symbol(&self, warps: u32) -> String {
        format!("{}_w{warps}_v1", self.prefix)
    }
    pub fn plan(&self, rows: u32, live: u32, warps: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if !WARPS.contains(&warps) {
            return Err(Refusal("invalid two-CTA finish warp count"));
        }
        let mut p = self.reference.plan(rows, live, 32)?;
        p.grid = [2, rows, 1];
        p.block = [warps * 32, 1, 1];
        Ok(p)
    }
    /// Replace the served tail, already validated by its own op, with the
    /// same arguments on two CTAs per row.
    pub fn compose(&self, tail: &Launch, warps: u32) -> Result<Launch, Refusal> {
        self.validate()?;
        let rows = tail.rows;
        if tail.symbol != self.reference.symbols()[1]
            || tail.plan != self.reference.plan(rows, rows, 32)?
            || tail.arguments.0.len() != 18
            || tail.collective
            || tail.persistent
        {
            return Err(Refusal("two-CTA finish tail differs from the served tail"));
        }
        Ok(Launch {
            symbol: self.symbol(warps),
            plan: self.plan(rows, rows, warps)?,
            rows,
            arguments: tail.arguments.clone(),
            collective: false,
            persistent: false,
        })
    }
    pub fn spec(&self, warps: u32) -> Result<EntrySpec, Refusal> {
        self.plan(1, 1, warps)?;
        let m = self.reference.mixer.query.mix;
        let traits = format!(
            "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,{}>",
            m.streams,
            m.width,
            m.iterations,
            m.eps_bits,
            m.norm_eps_bits,
            m.fn_dtype == DType::F32
        );
        let params = self.reference.cubin_spec("tail")?.entries.remove(1).params;
        let floor = self.reference.norm.reference.query.amax_floor_bits;
        let mutation = self.mutation.name();
        Ok(EntrySpec::new(self.symbol(warps),params,format!("using M={traits};
            using namespace chop::sinkhorn_hc;
            if (blockDim.x!={warps}*32 || gridDim.x!=2) asm volatile(\"trap;\");
            __shared__ finish_norm::Arena<M,{warps}> arena;
            chop::ptx::griddep_wait();
            finish_twin::task<M,{warps},{floor:#x}u,finish_twin::Mutation::{mutation}>(expanded,base,scale,pre_in,collapsed,pre,post,comb,workspace,live_rows,weight,normed,widened,codes,scales,status,status_row_offset,__uint_as_float(eps_bits),blockIdx.x,blockIdx.y,arena);
            chop::ptx::griddep_launch_dependents();"),warps*32,1))
    }
}

/// One cubin holding `(entry, warps)` pairs.
pub fn cubin_spec(name: &str, entries: &[(Entry, u32)]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("two-CTA finish cubin requires entries"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/finish_twin.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .map(|(e, w)| e.spec(*w))
            .collect::<Result<_, _>>()?,
    })
}
