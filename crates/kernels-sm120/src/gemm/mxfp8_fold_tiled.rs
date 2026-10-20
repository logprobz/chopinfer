//! Ordered folds held in registers or in CTA-local shared memory.
use super::mxfp8_fold_stream;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::{Launch, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
use kernel_core::mxfp8_gemm::Mxfp8FoldStreamBuffers;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    NarrowRows,
    WideColumns,
    ReuseRows,
}
impl Schedule {
    pub fn name(self) -> &'static str {
        match self {
            Self::NarrowRows => "m8n16k4",
            Self::WideColumns => "m8n32k2",
            Self::ReuseRows => "m32n8k4",
        }
    }
    pub fn warps(self) -> (u32, u32) {
        match self {
            Self::NarrowRows => (2, 4),
            Self::WideColumns => (4, 2),
            Self::ReuseRows => (1, 4),
        }
    }
    pub fn rows(self) -> u32 {
        match self {
            Self::NarrowRows | Self::WideColumns => 8,
            Self::ReuseRows => 32,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Entry {
    pub packed: mxfp8_fold_stream::Entry,
    pub symbol: String,
    pub schedule: Schedule,
    pub defer_admission: bool,
    pub omit_first: bool,
}
impl Entry {
    pub fn state_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        let grid = self.plan(rows)?.grid;
        let leaves = (u64::from(grid[0]) * u64::from(grid[2])).div_ceil(32);
        let bytes = 1024 + u64::from(grid[1]) * leaves * 128;
        if bytes / 4 > u64::from(u32::MAX) {
            return Err(Refusal("tiled fold state exceeds device index bounds"));
        }
        Ok(bytes.div_ceil(256) * 256)
    }
    pub fn validate(&self) -> Result<(), Refusal> {
        self.packed.validate()?;
        if self.packed.ticket.query.m_capacity > 32 {
            return Err(Refusal("tiled folds require capacity at most 32"));
        }
        for rows in [0, 1, self.packed.ticket.query.m_capacity] {
            crate::backend::check_plan(&self.plan(rows)?)?;
            self.state_bytes(rows)?;
        }
        Ok(())
    }
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        let q = self.packed.ticket.query;
        if rows > q.m_capacity {
            return Err(Refusal("tiled fold rows exceed capacity"));
        }
        let (nw, kw) = self.schedule.warps();
        Ok(LaunchPlan {
            grid: [
                q.n.div_ceil(nw * 8),
                rows.max(1).div_ceil(self.schedule.rows()),
                q.groups * if self.packed.ticket.paired { 2 } else { 1 },
            ],
            block: [nw * kw * 32, 1, 1],
            dynamic_shared_bytes: if kw == 1 {
                0
            } else {
                q.k.div_ceil(128) * self.schedule.rows() * nw * 8 * 4
            },
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        })
    }
    pub fn stamp_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        Ok(self
            .plan(rows)?
            .grid
            .into_iter()
            .map(u64::from)
            .product::<u64>()
            * 64)
    }
    pub fn symbols(&self) -> [String; 3] {
        let base = self.symbol.trim_end_matches("_v1");
        [
            format!("{base}_prepare_v1"),
            self.symbol.clone(),
            format!("{base}_clock_v1"),
        ]
    }
    pub fn specs(&self) -> Vec<EntrySpec> {
        let q = self.packed.ticket.query;
        let [init, compute, clock] = self.symbols();
        let mut old = self.packed.specs();
        old[0].symbol = init;
        let template = old.pop().unwrap();
        old.extend([(compute, false), (clock, true)].map(|(symbol, stamped)| {
            let (nw, kw) = self.schedule.warps();
            EntrySpec::new(symbol, template.params.clone(), format!(
                "chop::gemm::mxfp8_fold_tiled::entry<{},{},{},{},{},{nw},{kw},{},{},{},{stamped},{},{}>(activations,activation_scales,packed,weight_scales,packed2,weight_scales2,output,output2,workspace,state,live_rows,status,status_row_offset,rows,folds_per_cta);",
                q.n,q.k,q.groups,q.activation_row_stride,q.activation_group_stride,
                self.schedule.rows(),q.output == kernel_core::contract::DType::F32,self.packed.ticket.paired,
                self.defer_admission,self.omit_first),nw*kw*32,1)
        }));
        old
    }
    fn buffers(&self, b: &Mxfp8FoldStreamBuffers<View>) -> Result<(), Refusal> {
        self.validate()?;
        if self.packed.ticket.paired != b.second.is_some() {
            return Err(Refusal("tiled fold bank count differs"));
        }
        if b.second.is_none() && b.packed_weights[1] != View::EMPTY {
            return Err(Refusal("single tiled fold has a second packed bank"));
        }
        b.state
            .covers(self.state_bytes(self.packed.ticket.query.m_capacity)?)?;
        for v in b
            .packed_weights
            .iter()
            .take(if b.second.is_some() { 2 } else { 1 })
        {
            v.covers(self.packed.packed_bytes())?;
        }
        let mut occupied: Vec<_> = std::iter::once(b.first)
            .chain(b.second)
            .flat_map(|s| {
                [
                    s.activations,
                    s.activation_scales,
                    s.output,
                    s.weights,
                    s.weight_scales,
                    s.status,
                    s.live_rows,
                ]
            })
            .map(|v| v.handle)
            .collect();
        for view in std::iter::once(b.state).chain(
            b.packed_weights
                .into_iter()
                .take(if b.second.is_some() { 2 } else { 1 }),
        ) {
            if view.handle == 0 || occupied.contains(&view.handle) {
                return Err(Refusal(
                    "tiled fold persistent storage aliases another role",
                ));
            }
            occupied.push(view.handle);
        }
        Ok(())
    }
    fn operands(&self, b: &Mxfp8FoldStreamBuffers<View>, rows: u32) -> Result<(), Refusal> {
        let sizes = self.packed.ticket.legacy()?.bytes(rows);
        for operand in std::iter::once(b.first).chain(b.second) {
            operand.activations.covers(sizes.activations)?;
            operand.activation_scales.covers(sizes.activation_scales)?;
            operand.weights.covers(sizes.weights)?;
            operand.weight_scales.covers(sizes.weight_scales)?;
            operand.output.covers(sizes.output)?;
            operand.live_rows.covers(4)?;
            let status = operand
                .status_row_offset
                .checked_add(u64::from(rows))
                .and_then(|v| v.checked_mul(4))
                .filter(|&v| v <= kernel_core::precision::MAX_TENSOR_BYTES)
                .ok_or(Refusal("tiled fold status offset overflows"))?;
            operand.status.covers(status)?;
            if operand.activations != b.first.activations
                || operand.activation_scales != b.first.activation_scales
                || operand.live_rows != b.first.live_rows
                || operand.status != b.first.status
                || operand.status_row_offset != b.first.status_row_offset
            {
                return Err(Refusal(
                    "paired tiled folds require common activation and status views",
                ));
            }
        }
        if b.second
            .is_some_and(|s| s.output.handle == b.first.output.handle)
        {
            return Err(Refusal("tiled fold output banks alias"));
        }
        let inputs: Vec<_> = std::iter::once(b.first)
            .chain(b.second)
            .flat_map(|s| {
                [
                    s.activations.handle,
                    s.activation_scales.handle,
                    s.weights.handle,
                    s.weight_scales.handle,
                    s.live_rows.handle,
                ]
            })
            .collect();
        if inputs.contains(&b.first.status.handle) {
            return Err(Refusal("tiled fold status aliases an input"));
        }
        for operand in std::iter::once(b.first).chain(b.second) {
            if inputs.contains(&operand.output.handle)
                || operand.output.handle == b.first.status.handle
            {
                return Err(Refusal("tiled fold output aliases an input"));
            }
        }
        Ok(())
    }
    pub fn initialize(&self, b: &Mxfp8FoldStreamBuffers<View>) -> Result<Launch, Refusal> {
        self.buffers(b)?;
        self.operands(b, 0)?;
        let d = |v: View| Argument::DeviceBuffer(v.handle);
        Ok(Launch {
            symbol: self.symbols()[0].clone(),
            plan: self.packed.initialize_plan(),
            arguments: ArgumentList(vec![
                d(b.first.weights),
                b.second
                    .map_or(Argument::NullDevicePointer, |s| d(s.weights)),
                d(b.packed_weights[0]),
                b.second
                    .map_or(Argument::NullDevicePointer, |_| d(b.packed_weights[1])),
                d(b.state),
                Argument::U32((self.state_bytes(self.packed.ticket.query.m_capacity)? / 4) as u32),
            ]),
            rows: 0,
            collective: false,
            persistent: false,
        })
    }
    pub fn launch(
        &self,
        b: &Mxfp8FoldStreamBuffers<View>,
        rows: u32,
        stamps: Option<View>,
    ) -> Result<Launch, Refusal> {
        let plan = self.plan(rows)?;
        self.buffers(b)?;
        self.operands(b, rows)?;
        if let Some(v) = stamps {
            v.covers(self.stamp_bytes(rows)?)?;
            if v.handle == 0
                || std::iter::once(b.first)
                    .chain(b.second)
                    .flat_map(|s| {
                        [
                            s.activations,
                            s.activation_scales,
                            s.output,
                            s.weights,
                            s.weight_scales,
                            s.status,
                            s.live_rows,
                        ]
                    })
                    .chain(b.packed_weights)
                    .chain([b.state])
                    .any(|s| s.handle == v.handle)
            {
                return Err(Refusal(
                    "tiled fold diagnostic storage aliases another role",
                ));
            }
        }
        let d = |v: View| Argument::DeviceBuffer(v.handle);
        Ok(Launch {
            symbol: self.symbols()[if stamps.is_some() { 2 } else { 1 }].clone(),
            plan,
            arguments: ArgumentList(vec![
                d(b.first.activations),
                d(b.first.activation_scales),
                d(b.packed_weights[0]),
                d(b.first.weight_scales),
                b.second
                    .map_or(Argument::NullDevicePointer, |_| d(b.packed_weights[1])),
                b.second
                    .map_or(Argument::NullDevicePointer, |s| d(s.weight_scales)),
                d(b.first.output),
                b.second
                    .map_or(Argument::NullDevicePointer, |s| d(s.output)),
                stamps.map_or(Argument::NullDevicePointer, d),
                d(b.state),
                d(b.first.live_rows),
                d(b.first.status),
                Argument::U64(b.first.status_row_offset),
                Argument::U32(rows),
                Argument::U32(0),
            ]),
            rows,
            collective: false,
            persistent: false,
        })
    }
}
pub const PUBLISH: &str = "chop_mxfp8_tiled_publish_live_v1";
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut seen = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        for symbol in entry.symbols() {
            if !seen.insert(symbol) {
                return Err(Refusal("duplicate tiled fold symbol"));
            }
        }
    }
    if seen.is_empty() {
        return Err(Refusal("tiled fold cubin requires entries"));
    }
    let mut specs: Vec<_> = entries.iter().flat_map(Entry::specs).collect();
    specs.push(EntrySpec::new(PUBLISH.into(),vec![
        EntryParam::DevicePointer{name:"source".into(),cxx_type:"const uint8_t".into()},
        EntryParam::DevicePointer{name:"activation".into(),cxx_type:"uint8_t".into()},
        EntryParam::U64{name:"bytes".into()},
        EntryParam::DevicePointer{name:"live".into(),cxx_type:"uint32_t".into()},
        EntryParam::U32{name:"rows".into()},EntryParam::U32{name:"delay".into()}],
        "chop::ptx::griddep_launch_dependents();chop::ptx::griddep_wait();const uint64_t start=clock64();while(clock64()-start<delay)__nanosleep(64);for(uint64_t i=threadIdx.x;i<bytes;i+=blockDim.x)activation[i]=source[i];__syncthreads();if(threadIdx.x==0)*live=rows;".into(),32,1));
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/mxfp8_fold_tiled.cuh\"".into()],
        defines: vec![],
        entries: specs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{
        contract::DType, mxfp8_gemm::Mxfp8FoldGemmQuery, precision::execution::BlockScaledExecution,
    };
    #[test]
    fn row_tiles_bound_private_flags_and_keep_individual_fold_storage() {
        for schedule in [
            Schedule::NarrowRows,
            Schedule::WideColumns,
            Schedule::ReuseRows,
        ] {
            let q = Mxfp8FoldGemmQuery {
                m_capacity: 32,
                n: 128,
                k: 576,
                groups: 2,
                activation_row_stride: 1152,
                activation_group_stride: 576,
                output: DType::F32,
                execution: BlockScaledExecution::NativeBlockScaledMma,
            };
            let e = Entry {
                packed: mxfp8_fold_stream::Entry::new(q, "test", false).unwrap(),
                symbol: "test_tiled".into(),
                schedule,
                defer_admission: false,
                omit_first: false,
            };
            e.validate().unwrap();
            for rows in 1..=32 {
                let p = e.plan(rows).unwrap();
                assert!(p.grid[1] * 32 <= 128, "root counters overlap row flags");
                assert_eq!(
                    p.dynamic_shared_bytes,
                    5 * schedule.rows() * schedule.warps().0 * 8 * 4
                );
                assert!(e.state_bytes(rows).unwrap() >= 1024 + p.grid[1] as u64 * 128);
            }
            assert!(e.plan(33).is_err());
            let mut invalid = e.clone();
            invalid.packed.ticket.query.m_capacity = 64;
            assert!(invalid.validate().is_err());
        }
    }
}
