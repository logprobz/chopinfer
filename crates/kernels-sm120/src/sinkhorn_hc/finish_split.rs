//! The HC finish/norm tail after the column-block producer
//! (`cuda/sinkhorn_hc/finish_split.cuh`).
//!
//! The two-CTA tail of `finish_twin`, with the mixer CTA summing the 256
//! original RMS lanes from the expanded streams, because the column-block
//! producer (`round_cols`) leaves them out of the workspace. It keeps the
//! served tail's ABI and every expression and order of the served pair.
use super::finish_norm;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{
    backend::Launch,
    contract::{DType, LaunchPlan, Refusal},
};

/// Warp counts per CTA; the mixer needs the 256 original RMS lanes.
pub const WARPS: [u32; 3] = [8, 16, 32];

/// A measured bucket of the column-block pair (`round_cols` then this
/// finish).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bucket {
    pub rows: u32,
    /// Rows per producer CTA.
    pub tile: u32,
    /// Whether the producer uses the transposed butterfly.
    pub transposed: bool,
    /// Warps per finish CTA.
    pub warps: u32,
}

/// Lab controls: each must change an output of real rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mutation {
    None,
    /// The norm CTA zeroes the row status before the norm.
    ClearStatus,
    /// The norm's quantizer uses a wrong amax floor.
    WrongFloor,
    /// The mixer folds each mix's chains in decreasing K order.
    ReverseFold,
    /// Each RMS lane adds its packs in decreasing order.
    ReverseSquares,
}
impl Mutation {
    fn name(self) -> &'static str {
        match self {
            Mutation::None => "none",
            Mutation::ClearStatus => "clear_status",
            Mutation::WrongFloor => "wrong_floor",
            Mutation::ReverseFold => "reverse_fold",
            Mutation::ReverseSquares => "reverse_squares",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Entry {
    /// The served 1024-thread tail whose ABI and roles this keeps.
    pub reference: finish_norm::Entry,
    pub prefix: String,
    pub mutation: Mutation,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.validate()?;
        let m = self.reference.mixer.query.mix;
        if m.streams != 4 || !(m.streams * m.width / 8).is_multiple_of(256) {
            return Err(Refusal(
                "split finish needs four streams and whole RMS packs",
            ));
        }
        Ok(())
    }
    pub fn symbol(&self, warps: u32) -> String {
        format!("{}_w{warps}_v1", self.prefix)
    }
    pub fn plan(&self, rows: u32, live: u32, warps: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if !WARPS.contains(&warps) {
            return Err(Refusal("invalid split finish warp count"));
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
            return Err(Refusal("split finish tail differs from the served tail"));
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
            finish_split::task<M,{warps},{floor:#x}u,finish_split::Mutation::{mutation}>(expanded,base,scale,pre_in,collapsed,pre,post,comb,workspace,live_rows,weight,normed,widened,codes,scales,status,status_row_offset,__uint_as_float(eps_bits),blockIdx.x,blockIdx.y,arena);
            chop::ptx::griddep_launch_dependents();"),warps*32,1))
    }
}

/// One cubin holding `(entry, warps)` pairs.
pub fn cubin_spec(name: &str, entries: &[(Entry, u32)]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("split finish cubin requires entries"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/finish_split.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .map(|(e, w)| e.spec(*w))
            .collect::<Result<_, _>>()?,
    })
}
