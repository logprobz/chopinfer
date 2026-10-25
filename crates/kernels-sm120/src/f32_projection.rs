//! Exact F32-input projections. BF16 weights use three separate tensor
//! chains; raw F32 weights and values outside the tensor domain use SIMT.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    ActivationFormat, DType, GemmQuery, LaunchPlan, NumericalContract, Refusal, Target, Validate,
    WeightFormat,
};
use kernel_core::precision::capability::{
    ActivationCapability, CapabilityDocument, GlobalScaleCapability, OperationCapability,
};
use kernel_core::precision::{
    ArithmeticSpec, DecodeOrder, MatrixGeometry, OperationKind, OperationUse, Rounding,
    WeightLayout,
};

pub const HEADER: &str = "gemm/exact_f32.cuh";
pub const IMPLEMENTATION: &str = "sm120.exact-f32.fold128.v1";

pub mod blocked;
pub mod bounded_reuse;
pub mod cached_spans;
pub mod certified;
mod decode;
pub mod last;
mod prefill;
pub mod reuse;
mod split;
pub mod wide;
pub mod zero_decode;
pub mod zero_prefill;

/// An inclusive prefill row band selecting an existing feature tile.
/// Exact row entries take precedence over a covering band.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillBand {
    pub lo: u32,
    pub hi: u32,
    pub features: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    query: GemmQuery,
    usage: OperationUse,
    symbol: String,
    cached: bool,
    fused: bool,
    prefill_tile_n: u32,
    prefill_buckets: Vec<(u32, u32)>,
    prefill_bands: Vec<PrefillBand>,
    blocked_tile_m: u32,
    blocked_buckets: Vec<(u32, u32)>,
    wide_tile_n: u32,
    wide_buckets: Vec<(u32, u32)>,
    reuse_rows: Vec<u32>,
    last_rows: Vec<u32>,
    bounded_reuse_rows: Vec<u32>,
}

impl Entry {
    /// The fallback and the rows admitted by this registration's tile routes.
    pub fn route_tag(&self, symbol: &str) -> Option<kernel_core::route::RouteTag> {
        use kernel_core::route::{RouteClass, RouteTag};
        let rows: Vec<_> = self
            .prefill_buckets
            .iter()
            .chain(&self.blocked_buckets)
            .chain(&self.wide_buckets)
            .map(|&(r, _)| r..=r)
            .collect();
        if rows.is_empty() {
            return None;
        }
        let class = if symbol == self.baseline_symbol() {
            RouteClass::Baseline
        } else {
            RouteClass::Fast
        };
        Some(RouteTag::new(symbol, class, "f32_projection", rows))
    }

    pub fn route_at(&self, rows: u32, symbol: &str) -> Option<kernel_core::route::RouteTag> {
        use kernel_core::route::{RouteClass, branch};
        let admitted: Vec<_> = self
            .reuse_rows
            .iter()
            .chain(&self.last_rows)
            .chain(&self.bounded_reuse_rows)
            .map(|&r| r..=r)
            .collect();
        if admitted.is_empty() {
            return self.route_tag(symbol);
        }
        let entry = self.entry_for_rows(rows);
        let early = self.uses_bounded_reuse(rows) || entry.decode_last(rows).is_some();
        // Launch selection reaches registered prefill tiles before decode reuse.
        if rows > 0 && !early && entry.uses_prefill_tiles() {
            return self.route_tag(symbol);
        }
        let selected = early || entry.uses_decode_reuse(rows);
        Some(branch(
            symbol,
            if selected {
                RouteClass::Fast
            } else {
                RouteClass::Baseline
            },
            "f32_reuse",
            if selected { "reuse" } else { "reuse-fallback" },
            admitted,
        ))
    }

    pub fn route_tags(&self) -> Vec<kernel_core::route::RouteTag> {
        let mut tags: Vec<_> = self
            .runtime_symbols()
            .iter()
            .filter_map(|symbol| self.route_tag(symbol))
            .collect();
        if let Some(tag) = self.route_at(0, self.baseline_symbol()) {
            tags.push(tag);
        }
        tags
    }

    pub fn new(query: GemmQuery, usage: OperationUse, stem: &str) -> Result<Self, Refusal> {
        query.validate()?;
        if query.activation != ActivationFormat::F32
            || query.m_capacity != usage.capacity
            || query.m_capacity > crate::backend::MAX_CAPACITY
        {
            return Err(Refusal(
                "exact F32 projection requires matching bounded F32 capacity",
            ));
        }
        if query.n > i32::MAX as u32 || query.k > (i32::MAX as u32 - 127) {
            return Err(Refusal(
                "exact F32 projection dimensions exceed CUDA loop indices",
            ));
        }
        if stem.is_empty()
            || !stem
                .bytes()
                .enumerate()
                .all(|(i, c)| c == b'_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
        {
            return Err(Refusal("exact F32 projection needs a C identifier"));
        }
        Ok(Self {
            query,
            usage,
            symbol: format!("{stem}_v1"),
            cached: false,
            fused: false,
            prefill_tile_n: 0,
            prefill_buckets: Vec::new(),
            prefill_bands: Vec::new(),
            blocked_tile_m: 0,
            blocked_buckets: Vec::new(),
            wide_tile_n: 0,
            wide_buckets: Vec::new(),
            reuse_rows: Vec::new(),
            last_rows: Vec::new(),
            bounded_reuse_rows: Vec::new(),
        })
    }
    pub fn query(&self) -> GemmQuery {
        self.query
    }
    pub fn usage(&self) -> OperationUse {
        self.usage
    }
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
    pub fn weight_bytes(&self) -> u64 {
        u64::from(self.query.n)
            * u64::from(self.query.k)
            * if self.query.weight == WeightFormat::Bf16 {
                2
            } else {
                4
            }
    }
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if rows == 0 || rows > self.query.m_capacity {
            return Err(Refusal("exact F32 projection rows exceed capacity"));
        }
        let plan = LaunchPlan {
            grid: [self.query.n.div_ceil(16), rows.div_ceil(8), 1],
            block: [32, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }
    pub fn entry_spec(&self) -> EntrySpec {
        if self.uses_prefill_tiles() {
            return self.prefill_specs().pop().expect("prefill compute entry");
        }
        if self.fused {
            return self.split_specs().pop().expect("split compute entry");
        }
        if self.cached {
            return self.cached_spec();
        }
        let q = self.query;
        let f32_weight = q.weight == WeightFormat::F32;
        EntrySpec::new(
            self.symbol.clone(),
            vec![
                pointer("activation", "const float"),
                pointer(
                    "weight",
                    if f32_weight {
                        "const float"
                    } else {
                        "const uint16_t"
                    },
                ),
                pointer("output", "float"),
                pointer("live_rows", "const uint32_t"),
                EntryParam::U32 {
                    name: "rows".into(),
                },
            ],
            format!(
                "chop::exact_f32::run<{},{},{}>(activation,weight,output,live_rows,rows);",
                q.n, q.k, f32_weight
            ),
            32,
            1,
        )
    }
    pub fn capability(&self) -> OperationCapability {
        OperationCapability {
            kind: OperationKind::Dense,
            target: Target::Sm120f,
            usage: self.usage,
            geometry: MatrixGeometry {
                rows: u64::from(self.query.n),
                k: u64::from(self.query.k),
            },
            operands: 1,
            weight: self.query.weight,
            activation: Some(ActivationCapability::F32),
            epilogues: self.query.epilogues,
            layout: WeightLayout::row_major(),
            local_scale_dtypes: vec![],
            globals: GlobalScaleCapability::Absent,
            shared_local_scale: false,
            arithmetic: ArithmeticSpec {
                weight_operand: DType::F32,
                accumulator: DType::F32,
                output: DType::F32,
                decode_order: DecodeOrder::ExactF32Operands,
                rounding: Rounding::NearestEven,
                numerical_contract: self
                    .query
                    .numerical_contract_id()
                    .expect("validated query")
                    .to_string(),
            },
            implementation: IMPLEMENTATION.into(),
            stages: self.runtime_symbols(),
        }
    }
}

fn pointer(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}

pub fn capabilities(entries: &[Entry]) -> CapabilityDocument {
    CapabilityDocument {
        version: 1,
        operations: entries.iter().map(Entry::capability).collect(),
    }
}

pub fn cubin_spec(name: impl Into<String>, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("exact F32 cubin requires entries"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: {
            let mut headers = vec![format!(
                "#include \"{}\"",
                if entries.iter().any(Entry::uses_prefill_tiles) {
                    "gemm/exact_f32_prefill.cuh"
                } else if entries.iter().any(Entry::uses_fused_split) {
                    "gemm/exact_f32_split.cuh"
                } else if entries.iter().any(Entry::uses_cached_weights) {
                    "gemm/exact_f32_decode.cuh"
                } else {
                    HEADER
                }
            )];
            if entries.iter().any(Entry::uses_prefill_tiles)
                && entries.iter().any(Entry::uses_fused_split)
            {
                headers.push("#include \"gemm/exact_f32_split.cuh\"".into());
            }
            if entries.iter().any(|entry| entry.wide_tile_n != 0) {
                headers.push("#include \"gemm/exact_f32_wide.cuh\"".into());
            }
            if entries.iter().any(|entry| entry.blocked_tile_m != 0) {
                headers.push("#include \"gemm/exact_f32_blocked.cuh\"".into());
            }
            headers
        },
        defines: vec![],
        entries: entries.iter().flat_map(Entry::entry_specs).collect(),
    })
}
