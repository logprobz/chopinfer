//! The all-reduce round, expansion and next mixer, with reusable CTA tasks.
use super::{THREADS, ptr};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, Refusal};

pub use super::expand_mix::Entry;

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("round-expand-mix cubin needs entries"));
    }
    let mut specs = Vec::new();
    for e in entries {
        e.validate()?;
        if !e.query.mix.width.is_multiple_of(4) {
            return Err(Refusal("rounded expansion needs whole four-element stores"));
        }
        if e.probe {
            return Err(Refusal("round-expand-mix has no probe outputs"));
        }
        let m = e.query.mix;
        let traits = format!(
            "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,{}>",
            m.streams,
            m.width,
            m.iterations,
            m.eps_bits,
            m.norm_eps_bits,
            m.fn_dtype == DType::F32
        );
        let [partial, finish] = e.symbols();
        specs.push(EntrySpec::new(
            partial,
            vec![
                ptr("x", "const uint16_t"), ptr("sum", "const float"),
                ptr("rounded", "uint16_t"), ptr("post_in", "const float"),
                ptr("comb_in", "const float"), ptr("expanded", "uint16_t"),
                ptr("fn", "const void"), ptr("workspace", "float"),
                ptr("live_rows", "const uint32_t"),
            ],
            format!(
                "using M = {traits};
                 if (blockDim.x != 256 || gridDim.x != M::kChains / 8) asm volatile(\"trap;\");
                 chop::ptx::griddep_wait();
                 chop::sinkhorn_hc::round_mix::partials_task<M>(x,sum,post_in,comb_in,rounded,expanded,fn,workspace,live_rows,blockIdx.x,blockIdx.y,nullptr);
                 chop::ptx::griddep_launch_dependents();"
            ), THREADS, 1,
        ));
        specs.push(EntrySpec::new(
            finish,
            vec![
                ptr("expanded", "const uint16_t"), ptr("base", "const float"),
                ptr("scale", "const float"), ptr("pre_in", "const float"),
                ptr("collapsed", "uint16_t"), ptr("pre", "float"),
                ptr("post", "float"), ptr("comb", "float"),
                ptr("workspace", "const float"), ptr("live_rows", "const uint32_t"),
            ],
            format!(
                "using M = {traits};
                 if (blockDim.x != 256 || gridDim.x != (M::kWidth + 511) / 512) asm volatile(\"trap;\");
                 __shared__ __align__(16) unsigned char arena[sizeof(chop::sinkhorn_hc::round_mix::FinishArena<M>)];
                 chop::ptx::griddep_wait();
                 chop::sinkhorn_hc::round_mix::finish_task<M,false>(expanded,base,scale,pre_in,collapsed,pre,post,comb,nullptr,nullptr,nullptr,workspace,live_rows,blockIdx.x,blockIdx.y,arena);
                 chop::ptx::griddep_launch_dependents();"
            ), THREADS, 1,
        ));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/round_expand_mix.cuh\"".into()],
        defines: vec![],
        entries: specs,
    })
}
