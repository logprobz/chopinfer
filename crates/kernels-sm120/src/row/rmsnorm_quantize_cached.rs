//! Experimental cached ceil groups following the unchanged 256-lane RMS tree.
//! BF16 rows use `row/rmsnorm_quantize_cached.cuh`; F32 rows (the F32 query
//! norm) use `row/rmsnorm_quantize_cached_f32.cuh`, which also keeps the
//! served F32 norm's status commit.
use super::rmsnorm_quantize;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, Refusal};

#[derive(Clone, Debug)]
pub struct Entry {
    pub reference: rmsnorm_quantize::Entry,
    pub prefix: String,
    /// F32 rows only: signal dependents once the producer wait returns, so
    /// a dependent loads its immutable operands during the norm. Values and
    /// the status protocol are unchanged.
    pub early_release: bool,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.validate()?;
        let n = self.reference.query.norm;
        let admitted = match n.input {
            DType::Bf16 => n.width.is_multiple_of(256) && n.width <= 16384,
            // Eight values per lane, one chunk per lane: wider rows exceed
            // the 1024-thread register budget.
            DType::F32 => n.output == DType::F32 && n.width.is_multiple_of(32) && n.width <= 2048,
            _ => false,
        };
        if !admitted {
            return Err(Refusal(
                "cached norm needs BF16 rows in complete 256-column tiles or F32 rows of at most 2048 columns",
            ));
        }
        if self.early_release && n.input != DType::F32 {
            return Err(Refusal(
                "early dependent release is limited to F32 cached norms",
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
        if self.reference.query.norm.input == DType::F32 {
            return self.cubin_spec_f32(name);
        }
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
    /// The F32 body waits on its producer itself, after loading the weight.
    fn cubin_spec_f32(&self, name: &str) -> Result<CubinSpec, Refusal> {
        let width = self.reference.query.norm.width;
        let floor = self.reference.query.amax_floor_bits;
        let entries = self
            .symbols()
            .into_iter()
            .zip([8, 16, 32])
            .map(|(symbol, warps)| {
                let threads = warps * 32;
                let early = if self.early_release { ",true" } else { "" };
                EntrySpec::new(
                    symbol,
                    self.reference.spec().params,
                    format!(
                        "using namespace chop::row::rmsnorm_quantize_cached_f32;
                if (blockDim.x!={threads} || gridDim.x!=1 || widened) asm volatile(\"trap;\");
                __shared__ __align__(16) unsigned char arena[sizeof(Arena<{width},{warps}>)];
                run<{width},{warps},{floor:#x}u{early}>(static_cast<const float*>(input),weight,static_cast<float*>(normed),codes,scales,live_rows,status,status_row_offset,__uint_as_float(eps_bits),arena);"
                    ),
                    threads,
                    1,
                )
            })
            .collect();
        Ok(CubinSpec {
            name: name.into(),
            sources_root: crate::cuda_source_root(),
            include_dirs: vec![],
            headers: vec!["#include \"row/rmsnorm_quantize_cached_f32.cuh\"".into()],
            defines: vec![],
            entries,
        })
    }
}
