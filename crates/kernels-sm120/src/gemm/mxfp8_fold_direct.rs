//! Ordered folds held in registers or in CTA-local shared memory.
use super::mxfp8_fold_stream;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::{Launch, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
use kernel_core::mxfp8_gemm::Mxfp8FoldStreamBuffers;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Warp,
    Block,
    LocalSplit,
}
impl Schedule {
    pub fn name(self) -> &'static str {
        match self {
            Self::Warp => "n8k1",
            Self::Block => "n32k1",
            Self::LocalSplit => "n8k4",
        }
    }
    pub fn warps(self) -> (u32, u32) {
        match self {
            Self::Warp => (1, 1),
            Self::Block => (4, 1),
            Self::LocalSplit => (1, 4),
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
            return Err(Refusal("direct fold state exceeds device index bounds"));
        }
        Ok(bytes.div_ceil(256) * 256)
    }
    pub fn validate(&self) -> Result<(), Refusal> {
        self.packed.validate()?;
        for rows in [0, 1, self.packed.ticket.query.m_capacity] {
            crate::backend::check_plan(&self.plan(rows)?)?;
            self.state_bytes(rows)?;
        }
        Ok(())
    }
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        let q = self.packed.ticket.query;
        if rows > q.m_capacity {
            return Err(Refusal("direct fold rows exceed capacity"));
        }
        let (nw, kw) = self.schedule.warps();
        Ok(LaunchPlan {
            grid: [
                q.n.div_ceil(nw * 8),
                rows.max(1).div_ceil(16),
                q.groups * if self.packed.ticket.paired { 2 } else { 1 },
            ],
            block: [nw * kw * 32, 1, 1],
            dynamic_shared_bytes: if kw == 1 {
                0
            } else {
                q.k.div_ceil(128) * 16 * nw * 8 * 4
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
                "chop::gemm::mxfp8_fold_direct::entry<{},{},{},{},{},{nw},{kw},{},{},{stamped},{},{}>(activations,activation_scales,packed,weight_scales,packed2,weight_scales2,output,output2,workspace,state,live_rows,status,status_row_offset,rows,folds_per_cta);",
                q.n,q.k,q.groups,q.activation_row_stride,q.activation_group_stride,
                q.output == kernel_core::contract::DType::F32,self.packed.ticket.paired,
                self.defer_admission,self.omit_first),nw*kw*32,1)
        }));
        old
    }
    fn buffers(&self, b: &Mxfp8FoldStreamBuffers<View>) -> Result<(), Refusal> {
        self.validate()?;
        if self.packed.ticket.paired != b.second.is_some() {
            return Err(Refusal("direct fold bank count differs"));
        }
        if b.second.is_none() && b.packed_weights[1] != View::EMPTY {
            return Err(Refusal("single direct fold has a second packed bank"));
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
                    "direct fold persistent storage aliases another role",
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
                .ok_or(Refusal("direct fold status offset overflows"))?;
            operand.status.covers(status)?;
            if operand.activations != b.first.activations
                || operand.activation_scales != b.first.activation_scales
                || operand.live_rows != b.first.live_rows
                || operand.status != b.first.status
                || operand.status_row_offset != b.first.status_row_offset
            {
                return Err(Refusal(
                    "paired direct folds require common activation and status views",
                ));
            }
        }
        if b.second
            .is_some_and(|s| s.output.handle == b.first.output.handle)
        {
            return Err(Refusal("direct fold output banks alias"));
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
            return Err(Refusal("direct fold status aliases an input"));
        }
        for operand in std::iter::once(b.first).chain(b.second) {
            if inputs.contains(&operand.output.handle)
                || operand.output.handle == b.first.status.handle
            {
                return Err(Refusal("direct fold output aliases an input"));
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
                    "direct fold diagnostic storage aliases another role",
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
pub const PUBLISH: &str = "chop_mxfp8_direct_publish_live_v1";
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut seen = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        for symbol in entry.symbols() {
            if !seen.insert(symbol) {
                return Err(Refusal("duplicate direct fold symbol"));
            }
        }
    }
    if seen.is_empty() {
        return Err(Refusal("direct fold cubin requires entries"));
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
        headers: vec!["#include \"gemm/mxfp8_fold_direct.cuh\"".into()],
        defines: vec![],
        entries: specs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
    #[test]
    fn direct_fold_bounds_private_state_and_stamped_storage() {
        let q = Mxfp8FoldGemmQuery {
            m_capacity: 64,
            n: 576,
            k: 576,
            groups: 2,
            activation_row_stride: 1152,
            activation_group_stride: 576,
            output: kernel_core::contract::DType::F32,
            execution:
                kernel_core::precision::execution::BlockScaledExecution::NativeBlockScaledMma,
        };
        let v = |handle, bytes| View { handle, bytes };
        for paired in [false, true] {
            for schedule in [Schedule::Warp, Schedule::Block, Schedule::LocalSplit] {
                let entry = Entry {
                    packed: mxfp8_fold_stream::Entry::new(q, "packed_v1", paired).unwrap(),
                    symbol: "direct_v1".into(),
                    schedule,
                    defer_admission: false,
                    omit_first: false,
                };
                let bytes = entry.packed.ticket.legacy().unwrap().bytes(64);
                let first = Mxfp8FoldGemmBuffers {
                    activations: v(1, bytes.activations),
                    activation_scales: v(2, bytes.activation_scales),
                    weights: v(3, bytes.weights),
                    weight_scales: v(4, bytes.weight_scales),
                    output: v(5, bytes.output),
                    live_rows: v(6, 4),
                    status: v(7, bytes.status),
                    status_row_offset: 0,
                };
                let b = Mxfp8FoldStreamBuffers {
                    first,
                    second: paired.then_some(Mxfp8FoldGemmBuffers {
                        weights: v(8, bytes.weights),
                        weight_scales: v(9, bytes.weight_scales),
                        output: v(10, bytes.output),
                        ..first
                    }),
                    packed_weights: [
                        v(11, entry.packed.packed_bytes()),
                        if paired {
                            v(12, entry.packed.packed_bytes())
                        } else {
                            View::EMPTY
                        },
                    ],
                    state: v(13, entry.state_bytes(64).unwrap()),
                };
                entry.initialize(&b).unwrap();
                for rows in [0, 1, 2, 4, 6, 8, 16, 32, 64] {
                    let call = entry.launch(&b, rows, None).unwrap();
                    assert_eq!(call.plan.workspace_bytes, 0);
                    assert_eq!(call.symbol, "direct_v1");
                    let stamp = v(14, entry.stamp_bytes(rows).unwrap());
                    entry.launch(&b, rows, Some(stamp)).unwrap();
                    assert!(
                        entry
                            .launch(&b, rows, Some(v(14, stamp.bytes - 1)))
                            .is_err()
                    );
                    for alias in [1, 3, 5, 7, 11, 13] {
                        assert!(entry.launch(&b, rows, Some(v(alias, stamp.bytes))).is_err());
                    }
                }
                assert!(entry.launch(&b, 65, None).is_err());
                let short = Mxfp8FoldStreamBuffers {
                    state: v(13, b.state.bytes - 1),
                    ..b
                };
                assert!(entry.initialize(&short).is_err());
                for alias in [0, 1, 3, 5, 7, 11] {
                    let bad = Mxfp8FoldStreamBuffers {
                        state: v(alias, b.state.bytes),
                        ..b
                    };
                    assert!(entry.initialize(&bad).is_err());
                    assert!(entry.launch(&bad, 64, None).is_err());
                }
                for alias in [0, 1, 3, 5, 7, 13] {
                    let bad = Mxfp8FoldStreamBuffers {
                        packed_weights: [v(alias, b.packed_weights[0].bytes), b.packed_weights[1]],
                        ..b
                    };
                    assert!(entry.initialize(&bad).is_err());
                }
                let mut overflow = b;
                overflow.first.status_row_offset = u64::MAX;
                assert!(entry.launch(&overflow, 1, None).is_err());
                for input in std::iter::once(first).chain(b.second).flat_map(|s| {
                    [
                        s.activations.handle,
                        s.activation_scales.handle,
                        s.weights.handle,
                        s.weight_scales.handle,
                        s.live_rows.handle,
                    ]
                }) {
                    let mut output_alias = b;
                    output_alias.first.output.handle = input;
                    assert!(entry.launch(&output_alias, 64, None).is_err());
                    let mut status_alias = b;
                    status_alias.first.status.handle = input;
                    if let Some(second) = status_alias.second.as_mut() {
                        second.status.handle = input;
                    }
                    assert!(entry.launch(&status_alias, 64, None).is_err());
                }
                if let Some(second) = b.second {
                    let bad = Mxfp8FoldStreamBuffers {
                        second: Some(Mxfp8FoldGemmBuffers {
                            output: first.output,
                            ..second
                        }),
                        ..b
                    };
                    assert!(entry.launch(&bad, 64, None).is_err());
                }
            }
        }
    }
}
