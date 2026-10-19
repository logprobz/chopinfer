//! Experimental independent output tiles with a duplicated exact RMS tree.
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
                "tiled norm needs BF16 input and complete 256-column tiles",
            ));
        }
        Ok(())
    }
    pub fn symbols(&self) -> [String; 2] {
        [
            format!("{}_local_v1", self.prefix),
            format!("{}_tiles_v1", self.prefix),
        ]
    }
    pub fn plan(&self, rows: u32, live: u32, tiled: bool) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if live > rows {
            return Err(Refusal("live norm rows exceed the grid"));
        }
        let mut p = self.reference.plan(rows)?;
        p.grid = [
            if tiled {
                self.reference.query.norm.width / 256
            } else {
                1
            },
            rows,
            1,
        ];
        Ok(p)
    }
    pub fn cubin_spec(&self, name: &str) -> Result<CubinSpec, Refusal> {
        self.validate()?;
        let width = self.reference.query.norm.width;
        let floor = self.reference.query.amax_floor_bits;
        let entries = self.symbols().into_iter().enumerate().map(|(i, symbol)| {
            let tiles = if i == 0 { 1 } else { width / 256 };
            EntrySpec::new(symbol,self.reference.spec().params,format!("using namespace chop::row::rmsnorm_quantize_tiles;
                if (blockDim.x!=256 || gridDim.x!={tiles}) asm volatile(\"trap;\");
                __shared__ __align__(16) unsigned char arena[sizeof(Arena<{width}>)];
                chop::ptx::griddep_wait();
                task<{width},{tiles},{floor:#x}u>(static_cast<const uint16_t*>(input),weight,static_cast<uint16_t*>(normed),widened,codes,scales,live_rows,status,status_row_offset,__uint_as_float(eps_bits),blockIdx.x,blockIdx.y,arena);
                chop::ptx::griddep_launch_dependents();"),256,1)
        }).collect();
        Ok(CubinSpec {
            name: name.into(),
            sources_root: crate::cuda_source_root(),
            include_dirs: vec![],
            headers: vec!["#include \"row/rmsnorm_quantize_tiles.cuh\"".into()],
            defines: vec![],
            entries,
        })
    }
}
