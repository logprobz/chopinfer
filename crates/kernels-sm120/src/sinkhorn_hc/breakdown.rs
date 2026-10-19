//! Lab-only ablations of the existing split mixer. No serving backend selects them.
use super::{THREADS, lagged, lagged_split, ptr};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, Refusal};
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery;

pub const STAMP_FIELDS: u64 = 12;
pub const SAMPLED_LANES: u64 = 32;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Full,
    Loads,
    Resident,
    Sinkhorn,
    Ticket,
}
impl Mode {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Loads => "loads",
            Self::Resident => "resident",
            Self::Sinkhorn => "sinkhorn",
            Self::Ticket => "ticket",
        }
    }
}
#[derive(Clone, Debug)]
pub struct Entry {
    pub reference: lagged_split::Entry,
    pub prefix: String,
    pub mode: Mode,
    pub sampled: bool,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.contract()?;
        let SinkhornHyperConnectionQuery::MixCollapse { mix, .. } = self.reference.query() else {
            unreachable!()
        };
        if mix.fn_dtype != DType::F32
            || self.reference.reference.op != (lagged::Op::MixCollapse { probe: false })
            || (!self.sampled && self.mode != Mode::Full)
        {
            return Err(Refusal(
                "split breakdown requires F32 weights, no reference probes, and sampled ablations",
            ));
        }
        Ok(())
    }
    pub fn symbol(&self) -> String {
        format!(
            "{}_{}{}_v1",
            self.prefix,
            self.mode.name(),
            if self.sampled { "" } else { "_plain" }
        )
    }
    pub fn plan(&self, rows: u32, live: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        self.reference.plan(rows, live)
    }
    pub fn stamps_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        let plan = self.plan(rows, rows)?;
        Ok(u64::from(rows) * u64::from(plan.grid[0]) * SAMPLED_LANES * STAMP_FIELDS * 8)
    }
    pub fn spec(&self) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        let SinkhornHyperConnectionQuery::MixCollapse { mix: m, .. } = self.reference.query()
        else {
            unreachable!()
        };
        let traits = format!(
            "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,true>",
            m.streams, m.width, m.iterations, m.eps_bits, m.norm_eps_bits
        );
        let mut params = self.reference.params();
        params.extend([
            ptr("seed_logits", "const float"),
            ptr("observed_logits", "float"),
            ptr("loaded_weights", "float"),
            ptr("stamps", "uint64_t"),
        ]);
        Ok(EntrySpec::new(self.symbol(),params,format!("using M = {traits};
            using namespace chop::sinkhorn_hc::breakdown;
            if (blockDim.x != 256 || gridDim.x != M::kChains/8) asm volatile(\"trap;\");
            __shared__ __align__(16) unsigned char arena[sizeof(Arena<M>)];
            Clock<{}> clock(stamps);
            chop::ptx::griddep_launch_dependents();
            clock.mark(0);
            chop::ptx::griddep_wait();
            clock.mark(1);
            task<M,Mode::{},{}>(x,fn,base,scale,pre_in,collapsed,pre,post,comb,partials,tickets,live_rows,seed_logits,observed_logits,loaded_weights,stamps,blockIdx.x,blockIdx.y,arena,clock);",self.sampled,self.mode.name(),self.sampled),THREADS,1))
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("split breakdown needs entries"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/breakdown.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect::<Result<_, _>>()?,
    })
}

/// Payload accounting, not measured cache misses or transaction bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Traffic {
    pub weights_unique: u64,
    pub source_unique: u64,
    pub unique_input: u64,
    pub projection_weight_requests: u64,
    pub source_read_requests: u64,
    pub partial_reads: u64,
    pub partial_writes: u64,
    pub output_writes: u64,
    pub ticket_atomics: u64,
}
pub fn traffic(entry: &Entry, rows: u32) -> Result<Traffic, Refusal> {
    let plan = entry.plan(rows, rows)?;
    let SinkhornHyperConnectionQuery::MixCollapse { mix: m, .. } = entry.reference.query() else {
        unreachable!()
    };
    let r = u64::from(rows);
    let flat = u64::from(m.flat_width());
    let mixes = u64::from(m.mixes());
    let n = u64::from(m.streams);
    let weights = mixes * flat * 4;
    let source = r * flat * 2;
    let partials = r * mixes * (flat / 128) * 4;
    Ok(Traffic {
        weights_unique: weights,
        source_unique: source,
        unique_input: weights + source + r * n * 4 + mixes * 4 + 12 + 4 + r * 4,
        projection_weight_requests: r * weights,
        source_read_requests: 3 * source,
        partial_reads: partials,
        partial_writes: partials,
        output_writes: r * (u64::from(m.width) * 2 + (2 * n + n * n) * 4),
        ticket_atomics: r * u64::from(plan.grid[0]),
    })
}
