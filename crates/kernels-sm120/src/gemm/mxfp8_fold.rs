//! Native E4M3/UE8M0 matrix products with independent 128-K F32 chains.

use kernel_build::{CubinSpec, EntryDeclaration, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, DType, LaunchPlan, NumericalContract, Refusal, Validate,
};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
use kernel_core::precision::WeightLayout;
use kernel_core::precision::execution::BlockScaledExecution;
use kernel_core::properties::Invariance;

pub const LAB_CUBIN: &str = "mxfp8_fold_sm120";
pub const CODES_LAYOUT: &str = "mxfp8_e4m3_rows_v1";
pub const SCALES_LAYOUT: &str = "mxfp8_ue8m0_scale_rows_v1";

#[derive(Clone, Debug)]
pub struct Entry {
    pub query: Mxfp8FoldGemmQuery,
    pub symbol: String,
}

/// Payload admission before registration. Both native and decoded paths use
/// the same exact scaled domain; this engine never narrows a scaled operand.
pub use kernel_core::mxfp8_gemm::admit_operand;

impl Entry {
    pub fn new(query: Mxfp8FoldGemmQuery, symbol: impl Into<String>) -> Result<Self, Refusal> {
        let entry = Self {
            query,
            symbol: symbol.into(),
        };
        entry.validate()?;
        Ok(entry)
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if self.query.execution != BlockScaledExecution::NativeBlockScaledMma {
            return Err(Refusal(
                "native MXFP8 fold engine refuses an exact-decoded BF16 execution claim",
            ));
        }
        if self.query.m_capacity > crate::backend::MAX_CAPACITY {
            return Err(Refusal("MXFP8 fold capacity exceeds SM120 bounds"));
        }
        if [
            self.query.n,
            self.query.k,
            self.query.groups,
            self.query.activation_row_stride,
            self.query.activation_group_stride,
        ]
        .into_iter()
        .any(|n| n > i32::MAX as u32)
        {
            return Err(Refusal(
                "MXFP8 fold dimensions exceed signed template bounds",
            ));
        }
        if self.symbol.is_empty()
            || !self
                .symbol
                .bytes()
                .enumerate()
                .all(|(i, c)| c == b'_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
        {
            return Err(Refusal("MXFP8 fold symbol must be a C identifier"));
        }
        self.plan(self.query.m_capacity)?;
        Ok(())
    }

    pub fn decode(&self) -> bool {
        self.query.m_capacity <= 8
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if rows > self.query.m_capacity {
            return Err(Refusal("MXFP8 fold rows exceed capacity"));
        }
        let plan = LaunchPlan {
            grid: [
                self.query.n.div_ceil(if self.decode() { 64 } else { 32 }),
                rows.max(1).div_ceil(if self.decode() { 8 } else { 16 }),
                self.query.groups,
            ],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub fn invariance(&self) -> Invariance {
        Invariance::any_rows(self.query.m_capacity)
    }

    pub fn weight_layout(&self) -> WeightLayout {
        WeightLayout {
            id: if self.query.groups == 1 {
                "row-major"
            } else {
                "row-major-groups"
            }
            .into(),
            version: 1,
        }
    }

    pub fn bytes(&self, rows: u32) -> Mxfp8FoldGemmBuffers<u64> {
        let q = self.query;
        let a = u64::from(rows) * u64::from(q.activation_row_stride);
        let w = u64::from(q.groups) * u64::from(q.n) * u64::from(q.k);
        Mxfp8FoldGemmBuffers {
            activations: a,
            activation_scales: a / 32,
            weights: w,
            weight_scales: w / 32,
            output: u64::from(rows)
                * u64::from(q.groups)
                * u64::from(q.n)
                * if q.output == DType::F32 { 4 } else { 2 },
            live_rows: 4,
            status: u64::from(rows) * 4,
            status_row_offset: 0,
        }
    }

    fn spec_impl(
        &self,
        symbol: String,
        single_chain: bool,
        group_stride: u32,
        fold_k: u32,
        modulo_scale_rows: bool,
    ) -> EntrySpec {
        let q = self.query;
        let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: cxx_type.into(),
        };
        EntrySpec::new(
            symbol,
            vec![
                ptr("activations", "const uint8_t"),
                ptr("activation_scales", "const uint8_t"),
                ptr("weights", "const uint8_t"),
                ptr("weight_scales", "const uint8_t"),
                ptr(
                    "output",
                    if q.output == DType::F32 {
                        "float"
                    } else {
                        "uint16_t"
                    },
                ),
                ptr("live_rows", "const uint32_t"),
                ptr("status", "uint32_t"),
                EntryParam::U64 {
                    name: "status_row_offset".into(),
                },
                EntryParam::U32 {
                    name: "rows".into(),
                },
            ],
            format!(
                "chop::gemm::mxfp8_fold::run<{},{},{},{},{},{},{},{},{},{},{}>(activations,activation_scales,weights,weight_scales,output,live_rows,status,status_row_offset,rows);",
                q.n,
                q.k,
                q.groups,
                q.activation_row_stride,
                group_stride,
                q.output == DType::F32,
                self.decode(),
                single_chain,
                fold_k,
                modulo_scale_rows,
                kernel_core::mxfp8_gemm::NON_FINITE_INPUT
            ),
            128,
            1,
        )
    }

    pub fn spec(&self) -> EntrySpec {
        self.spec_impl(
            self.symbol.clone(),
            false,
            self.query.activation_group_stride,
            128,
            false,
        )
    }

    /// A deliberately incorrect lab-only body. No backend registration or
    /// successful contract declaration exists for this symbol.
    pub fn single_chain_control_spec(&self) -> EntrySpec {
        self.spec_impl(
            format!("{}_single_chain_control_v1", self.symbol),
            true,
            self.query.activation_group_stride,
            128,
            false,
        )
    }

    pub fn wide_fold_control_spec(&self) -> EntrySpec {
        self.spec_impl(
            format!("{}_fold256_control_v1", self.symbol),
            false,
            self.query.activation_group_stride,
            256,
            false,
        )
    }

    pub fn scale_row_control_spec(&self) -> EntrySpec {
        self.spec_impl(
            format!("{}_scale_row_mod8_control_v1", self.symbol),
            false,
            self.query.activation_group_stride,
            128,
            true,
        )
    }

    pub fn arguments(&self, b: Mxfp8FoldGemmBuffers<u64>, rows: u32) -> ArgumentList {
        ArgumentList(vec![
            Argument::DeviceBuffer(b.activations),
            Argument::DeviceBuffer(b.activation_scales),
            Argument::DeviceBuffer(b.weights),
            Argument::DeviceBuffer(b.weight_scales),
            Argument::DeviceBuffer(b.output),
            Argument::DeviceBuffer(b.live_rows),
            Argument::DeviceBuffer(b.status),
            Argument::U64(b.status_row_offset),
            Argument::U32(rows),
        ])
    }

    pub fn declaration(&self) -> Result<EntryDeclaration, Refusal> {
        self.validate()?;
        Ok(EntryDeclaration {
            symbol: self.symbol.clone(),
            operation: "MXFP8 fold GEMM".into(),
            contract: self.query.numerical_contract_id()?.to_string(),
            implementation: "sm120.mxfp8-fold.native.v2".into(),
            requirements: [
                ("query".into(), format!("{:?}", self.query)),
                ("entry".into(), self.spec().body),
                ("codes".into(), CODES_LAYOUT.into()),
                ("scales".into(), SCALES_LAYOUT.into()),
                ("fold_k".into(), "128".into()),
                ("pdl".into(), "true".into()),
            ]
            .into(),
            correctness_case: Some("mxfp8-fold-f64".into()),
        })
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("MXFP8 fold cubin requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        if !symbols.insert(&entry.symbol) {
            return Err(Refusal("duplicate MXFP8 fold symbol"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/mxfp8_fold.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    })
}

/// Generic tails, grouped offsets, both output forms and the accumulation
/// witness. Production shapes belong to the consuming model's catalog.
pub fn lab_entries() -> Vec<Entry> {
    let mut entries = Vec::new();
    for (capacity, n, k, groups, pad) in [
        (1, 7, 32, 1, 0),
        (8, 65, 160, 1, 0),
        (16, 33, 576, 1, 0),
        (33, 65, 4096, 1, 0),
        (8, 17, 160, 2, 32),
        (17, 9, 576, 3, 32),
        (1, 1, 65536, 1, 0),
    ] {
        for output in [DType::F32, DType::Bf16] {
            let q = Mxfp8FoldGemmQuery {
                m_capacity: capacity,
                n,
                k,
                groups,
                activation_group_stride: k + pad,
                activation_row_stride: groups * (k + pad) + pad,
                output,
                execution: BlockScaledExecution::NativeBlockScaledMma,
            };
            entries.push(
                Entry::new(
                    q,
                    format!(
                        "chop_mxfp8_fold_m{capacity}_n{n}_k{k}_g{groups}_{}_v2",
                        if output == DType::F32 { "f32" } else { "bf16" }
                    ),
                )
                .unwrap(),
            );
        }
    }
    entries
}

pub fn group_offset_control() -> Entry {
    let mut entry = lab_entries()
        .into_iter()
        .find(|e| e.query.groups == 2 && e.query.output == DType::F32)
        .unwrap();
    entry.query.activation_group_stride = entry.query.k;
    entry.symbol.push_str("_wrong_offset_control_v1");
    entry
}
