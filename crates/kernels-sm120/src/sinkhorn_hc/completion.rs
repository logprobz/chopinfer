//! One-launch decode HC boundaries: the rounded chain producer, the lagged
//! mixer finish and the cached input norm, joined by a per-row ticket.
//!
//! Both layouts reproduce the served producer plus fused tail bit for bit.
//! `Tail` runs the unchanged finisher in the last chain CTA. `Split` adds a
//! norm CTA that recomputes the expansion for its collapse, so the norm runs
//! beside the chains while the last arrival runs only the mixer.
use super::{finish_norm, ptr};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{
    backend::{Launch, View},
    contract::{Argument, ArgumentList, DType, LaunchPlan, Refusal},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layout {
    Tail,
    Split,
}
impl Layout {
    fn name(self) -> &'static str {
        match self {
            Layout::Tail => "tail",
            Layout::Split => "split",
        }
    }
    pub fn warps(self) -> &'static [u32] {
        match self {
            Layout::Tail => &[8, 16, 32],
            Layout::Split => &[8, 16],
        }
    }
}

/// Lab controls that must fail the bitwise comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mutation {
    None,
    ClearStatus,
    WrongFloor,
    ReverseFold,
}
impl Mutation {
    pub(super) fn name(self) -> &'static str {
        match self {
            Mutation::None => "none",
            Mutation::ClearStatus => "clear_status",
            Mutation::WrongFloor => "wrong_floor",
            Mutation::ReverseFold => "reverse_fold",
        }
    }
}

/// Producer and tail arguments in served order, then the ticket view.
pub const ARGUMENTS: usize = 25;

#[derive(Clone, Debug)]
pub struct Entry {
    /// The served 1024-thread finisher this launch replaces.
    pub reference: finish_norm::Entry,
    /// The served chain producer's symbol prefix.
    pub producer_prefix: String,
    pub prefix: String,
    pub layout: Layout,
    pub mutation: Mutation,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.validate()?;
        let m = self.reference.mixer.query.mix;
        if m.streams != 4 || !m.flat_width().is_multiple_of(128) {
            return Err(Refusal("HC completion needs four streams and whole chains"));
        }
        if self.layout == Layout::Tail && self.mutation == Mutation::ReverseFold {
            return Err(Refusal("the tail layout has no separate fold control"));
        }
        Ok(())
    }
    pub fn symbol(&self, warps: u32) -> String {
        format!("{}_{}_w{warps}_v1", self.prefix, self.layout.name())
    }
    pub fn symbols(&self) -> Vec<String> {
        self.layout
            .warps()
            .iter()
            .map(|&w| self.symbol(w))
            .collect()
    }
    fn chains(&self) -> u32 {
        self.reference.mixer.query.mix.flat_width() / 128
    }
    pub fn plan(&self, rows: u32, live: u32, warps: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if live > rows || !self.layout.warps().contains(&warps) {
            return Err(Refusal("invalid HC completion geometry"));
        }
        let mut p = self.reference.norm.plan(rows, live, warps)?;
        p.grid = [
            self.chains() + u32::from(self.layout == Layout::Split),
            rows,
            1,
        ];
        p.workspace_bytes = self.reference.mixer.workspace(rows)?;
        Ok(p)
    }
    /// Replace the served producer and fused tail, both already validated
    /// by their own op, with one launch. The ticket view must not alias any
    /// constituent role and covers one word per row.
    pub fn compose(
        &self,
        producer: &Launch,
        tail: &Launch,
        tickets: View,
        warps: u32,
    ) -> Result<Launch, Refusal> {
        self.validate()?;
        let rows = producer.rows;
        let chain = super::round_chain_partials::Entry {
            reference: self.reference.mixer.clone(),
            prefix: self.producer_prefix.clone(),
        };
        let a = &producer.arguments.0;
        let b = &tail.arguments.0;
        if rows != tail.rows
            || a.len() != 9
            || b.len() != 18
            || a[5] != b[0]
            || a[7] != b[8]
            || a[8] != b[9]
            || producer.collective
            || tail.collective
            || producer.persistent
            || tail.persistent
            || ![4, 8].into_iter().any(|w| {
                producer.symbol == format!("{}_w{w}_v1", self.producer_prefix)
                    && chain.plan(rows, rows, w).ok() == Some(producer.plan)
            })
            || tail.plan != self.reference.plan(rows, rows, 32)?
            || tail.symbol != self.reference.symbols()[1]
            || tickets.bytes < u64::from(rows) * 4
            || a.iter()
                .chain(b)
                .any(|v| *v == Argument::DeviceBuffer(tickets.handle))
        {
            return Err(Refusal(
                "HC completion constituents or ticket binding differ",
            ));
        }
        let mut args = a[..7].to_vec();
        args.extend_from_slice(&b[1..8]);
        args.extend_from_slice(&a[7..9]);
        args.extend_from_slice(&b[10..]);
        args.push(Argument::DeviceBuffer(tickets.handle));
        debug_assert_eq!(args.len(), ARGUMENTS);
        Ok(Launch {
            symbol: self.symbol(warps),
            plan: self.plan(rows, rows, warps)?,
            rows,
            arguments: ArgumentList(args),
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
        let original = self.reference.cubin_spec("tail")?.entries.remove(1);
        let mut params = [
            ("x", "const uint16_t"),
            ("sum", "const float"),
            ("rounded", "uint16_t"),
            ("post_in", "const float"),
            ("comb_in", "const float"),
            ("expanded", "uint16_t"),
            ("fn", "const void"),
        ]
        .into_iter()
        .map(|(n, t)| ptr(n, t))
        .collect::<Vec<_>>();
        params.extend_from_slice(&original.params[1..8]);
        params.push(ptr("workspace", "float"));
        params.extend_from_slice(&original.params[9..]);
        params.push(ptr("tickets", "uint32_t"));
        let floor = self.reference.norm.reference.query.amax_floor_bits;
        let (arena, task, ctas) = match self.layout {
            Layout::Tail => ("TailArena", "tail_task", "M::kChains"),
            Layout::Split => ("SplitArena", "split_task", "M::kChains+1"),
        };
        let mutation = self.mutation.name();
        Ok(EntrySpec::new(self.symbol(warps),params,format!("using M={traits};
            using namespace chop::sinkhorn_hc::completion;
            if (blockDim.x!={warps}*32 || gridDim.x!={ctas}) asm volatile(\"trap;\");
            __shared__ {arena}<M,{warps}> arena;
            chop::ptx::griddep_wait();
            {task}<M,{warps},{floor:#x}u,Mutation::{mutation}>(blockIdx.x,blockIdx.y,arena,x,sum,rounded,post_in,comb_in,expanded,fn,base,scale,pre_in,collapsed,pre,post,comb,workspace,live_rows,weight,normed,widened,codes,scales,status,status_row_offset,__uint_as_float(eps_bits),tickets);
            chop::ptx::griddep_launch_dependents();"),warps*32,1))
    }
}

/// One cubin holding `(entry, warps)` pairs.
pub fn cubin_spec(name: &str, entries: &[(Entry, u32)]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("completion cubin requires entries"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/completion.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .map(|(e, w)| e.spec(*w))
            .collect::<Result<_, _>>()?,
    })
}
