//! Column-block rounded HC producers for wide decode buckets
//! (`cuda/sinkhorn_hc/round_cols.cuh`).
//!
//! The served chain producers give each 128-K chain (or chain and row
//! pair) its own CTA, which reads the four input streams of its columns,
//! and the first eight CTAs of every row recompute a row's expansion for
//! the mixer's RMS lanes. This producer gives each CTA one 128-column block
//! of all four streams for a tile of rows: every column is expanded once,
//! its four chains are summed with the served products and butterfly, and
//! the RMS lanes move to the finisher that reads the expanded streams
//! (`finish_split`). It writes every output of the served producer and the
//! chain sums of the served workspace, but not the workspace's RMS lanes,
//! so it pairs only with that finisher.
use super::{ptr, round_chain_partials};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{
    backend::Launch,
    contract::{DType, LaunchPlan, Refusal},
};

/// Candidate (rows per CTA, transposed butterfly) pairs.
pub const TILES: [(u32, bool); 6] = [
    (2, false),
    (4, false),
    (8, false),
    (2, true),
    (4, true),
    (8, true),
];

/// Lab controls: each must change the chain sums of real rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mutation {
    None,
    /// The butterfly's levels in increasing offset order.
    ReverseLevels,
}
impl Mutation {
    fn name(self) -> &'static str {
        match self {
            Mutation::None => "none",
            Mutation::ReverseLevels => "reverse_levels",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Entry {
    /// The served chain producer whose ABI and workspace this keeps.
    pub reference: round_chain_partials::Entry,
    pub prefix: String,
    pub mutation: Mutation,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.validate()?;
        let m = self.reference.reference.query.mix;
        if m.streams != 4 || m.mixes() != 24 {
            return Err(Refusal("column-block producer needs four streams"));
        }
        Ok(())
    }
    pub fn symbol(&self, tile: u32, transposed: bool) -> String {
        let tx = if transposed { "_tx" } else { "" };
        format!("{}_r{tile}{tx}_v1", self.prefix)
    }
    pub fn plan(
        &self,
        rows: u32,
        live: u32,
        tile: u32,
        transposed: bool,
    ) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if !TILES.contains(&(tile, transposed)) || (transposed && self.mutation != Mutation::None) {
            return Err(Refusal("invalid column-block producer tile"));
        }
        let mut p = self.reference.plan(rows, live, 8)?;
        p.grid = [
            self.reference.reference.query.mix.width / 128,
            rows.div_ceil(tile),
            1,
        ];
        p.block = [256, 1, 1];
        crate::backend::check_plan(&p)?;
        Ok(p)
    }
    /// Replace the served producer, already validated by its own op, with
    /// this one over the same nine arguments.
    pub fn compose(
        &self,
        producer: &Launch,
        tile: u32,
        transposed: bool,
    ) -> Result<Launch, Refusal> {
        let rows = producer.rows;
        if producer.arguments.0.len() != 9
            || producer.plan.workspace_bytes
                != self.plan(rows, rows, tile, transposed)?.workspace_bytes
            || producer.collective
            || producer.persistent
        {
            return Err(Refusal(
                "column-block producer differs from the served producer",
            ));
        }
        Ok(Launch {
            symbol: self.symbol(tile, transposed),
            plan: self.plan(rows, rows, tile, transposed)?,
            rows,
            arguments: producer.arguments.clone(),
            collective: false,
            persistent: false,
        })
    }
    pub fn spec(&self, tile: u32, transposed: bool) -> Result<EntrySpec, Refusal> {
        self.plan(tile, tile, tile, transposed)?;
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
        let mutation = self.mutation.name();
        Ok(EntrySpec::new(
            self.symbol(tile, transposed),
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
                using namespace chop::sinkhorn_hc;
                if (blockDim.x!=256 || gridDim.x!=M::kWidth/128) asm volatile(\"trap;\");
                __shared__ round_cols::Arena<M,{tile}> arena;
                float w[round_cols::Layout<M>::kMixesPerWarp][4];
                round_cols::load_weights<M>(fn,blockIdx.x,w);
                chop::ptx::griddep_wait();
                round_cols::task<M,{tile},{transposed},round_cols::Mutation::{mutation}>(x,sum,rounded,post_in,comb_in,expanded,workspace,live_rows,blockIdx.x,blockIdx.y,w,arena);
                chop::ptx::griddep_launch_dependents();"
            ),
            256,
            1,
        ))
    }
}

/// One cubin holding `(entry, rows per CTA, transposed)` triples.
pub fn cubin_spec(name: &str, entries: &[(Entry, u32, bool)]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("column-block producer cubin requires entries"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/round_cols.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .map(|(e, tile, transposed)| e.spec(*tile, *transposed))
            .collect::<Result<_, _>>()?,
    })
}
