//! Experimental cached ceil groups following the unchanged 256-lane RMS tree.
use super::rmsnorm_quantize;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, Refusal};

#[derive(Clone, Debug)]
pub struct Entry {
    pub reference: rmsnorm_quantize::Entry,
    pub prefix: String,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.validate()?;
        let n = self.reference.query.norm;
        if n.input != DType::Bf16 || !n.width.is_multiple_of(256) || n.width > 16384 {
            return Err(Refusal(
                "cached norm needs BF16 input and complete 256-column tiles",
            ));
        }
        Ok(())
    }
    pub fn symbols(&self) -> [String; 3] {
        [8, 16, 32].map(|w| format!("{}_cached{w}_v1", self.prefix))
    }
    pub fn plan(&self, rows: u32, live: u32, warps: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if live > rows || ![8, 16, 32].contains(&warps) {
            return Err(Refusal("invalid cached norm rows or warp count"));
        }
        let mut p = self.reference.plan(rows)?;
        p.grid = [1, rows, 1];
        p.block = [warps * 32, 1, 1];
        Ok(p)
    }
    pub fn cubin_spec(&self, name: &str) -> Result<CubinSpec, Refusal> {
        self.validate()?;
        let width = self.reference.query.norm.width;
        let floor = self.reference.query.amax_floor_bits;
        let entries = self.symbols().into_iter().enumerate().map(|(i, symbol)| {
            let warps = [8,16,32][i];
            let threads = warps * 32;
            EntrySpec::new(symbol,self.reference.spec().params,format!("using namespace chop::row::rmsnorm_quantize_cached;
                if (blockDim.x!={threads} || gridDim.x!=1) asm volatile(\"trap;\");
                __shared__ __align__(16) unsigned char arena[sizeof(Arena<{width},{warps}>)];
                chop::ptx::griddep_wait();
                task<{width},{warps},{floor:#x}u>(static_cast<const uint16_t*>(input),weight,static_cast<uint16_t*>(normed),widened,codes,scales,live_rows,status,status_row_offset,__uint_as_float(eps_bits),blockIdx.x,blockIdx.y,arena);
                chop::ptx::griddep_launch_dependents();"),threads,1)
        }).collect();
        Ok(CubinSpec {
            name: name.into(),
            sources_root: crate::cuda_source_root(),
            include_dirs: vec![],
            headers: vec!["#include \"row/rmsnorm_quantize_cached.cuh\"".into()],
            defines: vec![],
            entries,
        })
    }
}
