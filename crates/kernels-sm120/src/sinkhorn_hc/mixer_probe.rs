//! Finalizer entries, including the served plain warp finalizer.
use super::{THREADS, lagged, lagged_split, ptr};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{
    contract::{DType, Refusal},
    sinkhorn_hyper_connection::SinkhornHyperConnectionQuery,
};
#[derive(Clone, Debug)]
pub struct Entry {
    pub reference: lagged_split::Entry,
    pub prefix: String,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.contract()?;
        let SinkhornHyperConnectionQuery::MixCollapse { mix, .. } = self.reference.query() else {
            unreachable!()
        };
        if mix.fn_dtype != DType::F32
            || mix.streams > 4
            || !(mix.flat_width() / 128).is_multiple_of(16)
            || self.reference.reference.op != (lagged::Op::MixCollapse { probe: false })
        {
            return Err(Refusal(
                "mixer probes require at most four streams and F32 projection weights",
            ));
        }
        Ok(())
    }
    pub fn finalizer_symbols(&self) -> Vec<String> {
        ["shared", "warp", "converged"]
            .into_iter()
            .flat_map(|kind| {
                [false, true].map(move |sampled| {
                    format!(
                        "{}_{}_{}v1",
                        self.prefix,
                        kind,
                        if sampled { "stamped_" } else { "" }
                    )
                })
            })
            .collect()
    }
    pub fn specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.validate()?;
        let SinkhornHyperConnectionQuery::MixCollapse { mix: m, .. } = self.reference.query()
        else {
            unreachable!()
        };
        let traits = format!(
            "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,true>",
            m.streams, m.width, m.iterations, m.eps_bits, m.norm_eps_bits
        );
        // Retain the mixed module's shared extent after removing its cluster entries.
        let mut entries = Vec::new();
        for (index, symbol) in self.finalizer_symbols().into_iter().enumerate() {
            let mut params = self.reference.params();
            params.push(ptr("stamps", "uint64_t"));
            entries.push(EntrySpec::new(symbol,params,format!("using M={traits};
                using namespace chop::sinkhorn_hc::finalizer_probe;
                if (blockDim.x!=256 || gridDim.x!=M::kChains/8) asm volatile(\"trap;\");
                __shared__ __align__(16) unsigned char arena[((sizeof(Arena<M>) + 15) / 16) * 16];
                Clock<{}> clock(stamps);
                chop::ptx::griddep_launch_dependents();clock.mark(0);
                chop::ptx::griddep_wait();clock.mark(1);
                task<M,{},{}>(x,fn,base,scale,pre_in,collapsed,pre,post,comb,partials,tickets,live_rows,stamps,blockIdx.x,blockIdx.y,arena,clock);",index%2==1,index/2,index%2==1),THREADS,1));
        }
        Ok(entries)
    }
    pub fn cubin_spec(&self, name: &str) -> Result<CubinSpec, Refusal> {
        Ok(CubinSpec {
            name: name.into(),
            sources_root: crate::cuda_source_root(),
            include_dirs: vec![],
            headers: vec!["#include \"sinkhorn_hc/finalizer_probe.cuh\"".into()],
            defines: vec![],
            entries: self.specs()?,
        })
    }
}
