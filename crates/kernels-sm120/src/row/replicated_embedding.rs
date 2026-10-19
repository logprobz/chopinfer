//! Direct immutable peer lookup, exact reduction round and stream broadcast.
use super::ptr;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};
use kernel_core::replicated_embedding::ReplicatedEmbedding;

#[derive(Clone, Debug)]
pub struct Entry {
    pub query: ReplicatedEmbedding,
    pub capacity: u32,
    pub symbol: String,
}
impl Entry {
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.query.validate()?;
        if self.capacity > 32 || rows == 0 || rows > self.capacity {
            return Err(Refusal("embedding rows exceed capacity"));
        }
        let plan = LaunchPlan {
            grid: [rows, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1; 3],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("replicated embedding needs entries"));
    }
    let mut specs = Vec::new();
    for e in entries {
        e.plan(e.capacity)?;
        let q = e.query;
        specs.push(EntrySpec::new(e.symbol.clone(), vec![
            EntryParam::PointerArray {name:"tables".into(),slots:8,cxx_type:"const uint16_t".into()},
            ptr("ids","const int32_t"), ptr("shard","uint16_t"), ptr("wide","float"),
            ptr("output","uint16_t"), ptr("expanded","uint16_t"), ptr("live_rows","const uint32_t"),
            EntryParam::U32 {name:"rank".into()},
        ], format!("if (blockDim.x != 256) asm volatile(\"trap;\");
            chop::ptx::griddep_wait();
            chop::row::replicated_embedding::task<{},{},{}>(tables,ids,shard,wide,output,expanded,live_rows,{},rank,blockIdx.x,nullptr);
            chop::ptx::griddep_launch_dependents();",q.width,q.ranks,q.streams,q.vocab/q.ranks),256,1));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"row/replicated_embedding.cuh\"".into()],
        defines: vec![],
        entries: specs,
    })
}
