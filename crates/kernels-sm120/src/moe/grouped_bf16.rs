//! Active expert tiles with exact BF16 MMA chains and asynchronous operand stages.
//! Queue tickets assign disjoint outputs. Every output still folds 128-K chains
//! in ascending K, including zero padding in the last chain.
use super::chained::quantized::Projection;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::Launch;
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub rows: u32,
    pub columns: u32,
    pub stages: u32,
    /// Decode a full chain before MMA, reducing live temporary accumulators.
    pub whole_chain: bool,
}
#[derive(Clone, Debug)]
pub struct Entry {
    pub projection: Projection,
    pub tile: Tile,
    pub symbol: String,
}
impl Entry {
    pub fn new(projection: Projection, tile: Tile) -> Result<Self, Refusal> {
        projection.query.validate()?;
        if projection.row_tile != 64 || projection.tokens_capacity == 0 {
            return Err(Refusal(
                "grouped BF16 requires a positive capacity and 64-row routing",
            ));
        }
        projection.padded_rows(projection.tokens_capacity)?;
        let g = projection.query.projection.gemm;
        let routed = g
            .grouped
            .ok_or(Refusal("grouped BF16 needs expert metadata"))?;
        let routes = u64::from(projection.tokens_capacity) * u64::from(routed.top_k);
        if routes + u64::from(routed.experts) * 64 + 256 > i32::MAX as u64
            || u64::from(routed.experts) * u64::from(g.n) > i32::MAX as u64
            || g.k > i32::MAX as u32
        {
            return Err(Refusal("grouped BF16 geometry exceeds integer indexing"));
        }
        let Tile {
            rows,
            columns,
            stages,
            ..
        } = tile;
        if projection.row_tile != 64
            || projection.active
            || projection.streamed
            || projection.prefill
            || ![64, 128, 256].contains(&rows)
            || ![32, 64, 128].contains(&columns)
            || ![2, 3].contains(&stages)
            || !projection.query.projection.gemm.k.is_multiple_of(64)
            || rows / 32 * columns / 16 * 32 > 1024
        {
            return Err(Refusal("unsupported grouped BF16 tile"));
        }
        let symbol = format!(
            "{}_group_m{rows}n{columns}s{stages}c{}_v1",
            projection.symbol.trim_end_matches("_v1"),
            u8::from(tile.whole_chain)
        );
        let result = Self {
            projection,
            tile,
            symbol,
        };
        if result.shared_bytes() > 101376 || result.tasks_bound() > i32::MAX as u64 {
            return Err(Refusal("grouped BF16 launch exceeds device bounds"));
        }
        Ok(result)
    }
    pub fn threads(&self) -> u32 {
        self.tile.rows * self.tile.columns / 16
    }
    pub fn queue_slots(&self) -> u32 {
        let p = &self.projection;
        let g = p.query.projection.gemm.grouped.unwrap();
        (u64::from(p.tokens_capacity) * u64::from(g.top_k)).div_ceil(u64::from(self.tile.rows))
            as u32
            + g.experts
    }
    pub fn workspace_bytes(&self) -> u64 {
        (8 + u64::from(self.queue_slots()) * 12).next_multiple_of(256)
    }
    pub fn tasks_bound(&self) -> u64 {
        u64::from(self.queue_slots())
            * u64::from(self.projection.columns().div_ceil(self.tile.columns))
    }
    pub fn shared_bytes(&self) -> u32 {
        let Tile {
            rows,
            columns,
            stages,
            whole_chain,
        } = self.tile;
        let w = columns * if self.projection.gate_up() { 2 } else { 1 };
        let raw = (w * 32 + rows * 64 + (w + rows) * 4).next_multiple_of(16);
        (stages * raw + (w + rows) * (if whole_chain { 136 } else { 72 }) * 2 + rows * 4 + 4)
            .next_multiple_of(16)
    }
    pub fn pack_symbol(&self) -> String {
        format!("{}_pack_v1", self.symbol.trim_end_matches("_v1"))
    }
    pub fn negative_symbol(&self) -> String {
        format!("{}_omit_v1", self.symbol.trim_end_matches("_v1"))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn launches(
        &self,
        rows: u32,
        h: [u64; 10],
        counts: u64,
        active: u64,
        queue: u64,
        ctas: u32,
        negative: bool,
    ) -> Result<Vec<Launch>, Refusal> {
        self.projection.arguments(h)?;
        if rows == 0
            || rows > self.projection.tokens_capacity
            || counts == 0
            || active == 0
            || queue == 0
            || ctas == 0
        {
            return Err(Refusal(
                "grouped BF16 needs live capacity and queue buffers",
            ));
        }
        let make = |symbol, handles: Vec<u64>, grid, block, shared| Launch {
            symbol,
            rows,
            plan: LaunchPlan {
                grid: [grid, 1, 1],
                block: [block, 1, 1],
                cluster: [1; 3],
                dynamic_shared_bytes: shared,
                workspace_bytes: self.workspace_bytes(),
                pdl: true,
            },
            arguments: ArgumentList(
                handles
                    .into_iter()
                    .map(|h| {
                        if h == 0 {
                            Argument::NullDevicePointer
                        } else {
                            Argument::DeviceBuffer(h)
                        }
                    })
                    .collect(),
            ),
            collective: false,
            persistent: false,
        };
        let ops = vec![
            make(
                self.pack_symbol(),
                vec![counts, h[4], active, h[6], h[7], queue],
                1,
                32,
                0,
            ),
            make(
                if negative {
                    self.negative_symbol()
                } else {
                    self.symbol.clone()
                },
                vec![h[0], h[1], h[2], h[3], h[5], h[6], h[7], queue, h[8], h[9]],
                ctas,
                self.threads(),
                self.shared_bytes(),
            ),
        ];
        for op in &ops {
            op.plan.validate_geometry()?;
        }
        Ok(ops)
    }
    fn config(&self) -> String {
        let p = &self.projection;
        let q = p.query.projection;
        let g = q.gemm.grouped.unwrap();
        let clamp = q.swiglu_clamp.map_or_else(
            || "chop::moe::clamp::SwigluClamp<false,0u,0u>".into(),
            |c| {
                format!(
                    "chop::moe::clamp::SwigluClamp<true,{}u,{}u>",
                    c.gate_max_bits, c.up_limit_bits
                )
            },
        );
        format!(
            "chop::moe::chained::quantized::Config<chop::moe::chained::Projection::{},{},{},{},{},64,{},{}>",
            if p.gate_up() { "gate_up" } else { "down" },
            g.experts,
            g.top_k,
            p.columns(),
            q.gemm.k,
            p.tokens_capacity,
            clamp
        )
    }
    pub fn entries(&self, controls: bool) -> Vec<EntrySpec> {
        let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let c = self.config();
        let t = self.tile;
        let dependency = "chop::ptx::griddep_launch_dependents(); chop::ptx::griddep_wait();";
        let mut entries = vec![EntrySpec::new(
            self.pack_symbol(),
            vec![
                ptr("counts", "const int"),
                ptr("offsets", "const int"),
                ptr("active", "const int"),
                ptr("padded", "const int"),
                ptr("live", "const int"),
                ptr("queue", "int"),
            ],
            format!(
                "{dependency} chop::moe::grouped_bf16::pack<{c},{}>(counts,offsets,active,padded,live,queue);",
                t.rows
            ),
            32,
            1,
        )];
        for negative in [false, true].into_iter().filter(|v| !v || controls) {
            let params = vec![
                ptr("input", "const uint8_t"),
                ptr("weight", "const uint8_t"),
                ptr("weight_scales", "const uint8_t"),
                ptr("activation_scales", "const uint8_t"),
                ptr("sorted", "const int"),
                ptr("padded", "const int"),
                ptr("live", "const int"),
                ptr("queue", "int"),
                ptr("route_weights", "const float"),
                ptr("output", "void"),
            ];
            let template = format!(
                "{c},{},{},{},{}",
                t.rows, t.columns, t.stages, t.whole_chain
            );
            let body = format!(
                "static_assert(sizeof(chop::moe::grouped_bf16::Shared<{template}>) == {}); extern __shared__ __align__(16) uint8_t arena[]; {dependency} chop::moe::grouped_bf16::run<{template},{negative}>(arena,input,weight,weight_scales,activation_scales,sorted,padded,live,queue,route_weights,output);",
                self.shared_bytes()
            );
            entries.push(EntrySpec::new(
                if negative {
                    self.negative_symbol()
                } else {
                    self.symbol.clone()
                },
                params,
                body,
                self.threads(),
                1,
            ));
        }
        entries
    }
}
pub fn cubin(name: &str, entries: &[Entry], controls: bool) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec!["common".into()],
        headers: vec!["#include \"moe/grouped_bf16.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().flat_map(|e| e.entries(controls)).collect(),
    }
}
