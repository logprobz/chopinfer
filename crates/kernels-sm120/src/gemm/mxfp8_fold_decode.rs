//! Opt-in small-row MXFP8 entries with two staged folds per CTA.

use super::mxfp8_fold;
use kernel_build::{CubinSpec, EntryDeclaration, EntryParam, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, NumericalContract, Refusal};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;

#[derive(Clone, Debug)]
pub struct Entry {
    pub query: Mxfp8FoldGemmQuery,
    pub symbol: String,
    /// Two independent weight/output banks share activation admission and
    /// one partial grid. Each bank retains the single projection's layout.
    pub paired: bool,
}

impl Entry {
    pub fn new(
        query: Mxfp8FoldGemmQuery,
        symbol: impl Into<String>,
        paired: bool,
    ) -> Result<Self, Refusal> {
        let entry = Self {
            query,
            symbol: symbol.into(),
            paired,
        };
        entry.validate()?;
        Ok(entry)
    }

    pub fn legacy(&self) -> Result<mxfp8_fold::Entry, Refusal> {
        mxfp8_fold::Entry::new(self.query, &self.symbol)
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        self.legacy()?;
        if self.query.m_capacity > 32 {
            return Err(Refusal("split MXFP8 folds require at most 32 rows"));
        }
        self.workspace(self.query.m_capacity)?;
        self.plans(self.query.m_capacity)?;
        Ok(())
    }

    pub fn workspace(&self, rows: u32) -> Result<u64, Refusal> {
        [
            u64::from(rows),
            u64::from(self.query.groups),
            u64::from(self.query.n),
            u64::from(self.query.k.div_ceil(128)),
            if self.paired { 8 } else { 4 },
        ]
        .into_iter()
        .try_fold(1_u64, |n, v| n.checked_mul(v))
        .and_then(|n| n.checked_add(256))
        .filter(|&n| n <= kernel_core::precision::MAX_TENSOR_BYTES)
        .ok_or(Refusal("split MXFP8 workspace exceeds the supported bound"))
    }

    pub fn symbols(&self) -> [String; 3] {
        [
            format!("{}_admit_v1", self.symbol),
            format!("{}_partials_v1", self.symbol),
            self.symbol.clone(),
        ]
    }

    pub fn plans(&self, rows: u32) -> Result<[LaunchPlan; 3], Refusal> {
        if rows > self.query.m_capacity {
            return Err(Refusal("split MXFP8 rows exceed capacity"));
        }
        let q = self.query;
        let banks = if self.paired { 2 } else { 1 };
        let grid_z = q
            .groups
            .checked_mul(q.k.div_ceil(128).div_ceil(2))
            .and_then(|v| v.checked_mul(banks))
            .ok_or(Refusal("split MXFP8 grid overflows"))?;
        let outputs = u64::from(rows) * u64::from(q.groups) * u64::from(q.n) * u64::from(banks);
        let merge_grid = u32::try_from(outputs.max(1).div_ceil(128))
            .map_err(|_| Refusal("split MXFP8 merge grid overflows"))?;
        let plan = |grid, threads| LaunchPlan {
            grid,
            block: [threads, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        let plans = [
            plan([rows.max(1), 1, 1], 128),
            plan([q.n.div_ceil(16), rows.max(1).div_ceil(16), grid_z], 64),
            plan([merge_grid, 1, 1], 128),
        ];
        for p in &plans {
            crate::backend::check_plan(p)?;
        }
        Ok(plans)
    }

    pub fn specs(&self) -> Vec<EntrySpec> {
        let q = self.query;
        let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let tail = || {
            vec![
                ptr("live_rows", "const uint32_t"),
                ptr("status", "const uint32_t"),
                EntryParam::U64 {
                    name: "status_row_offset".into(),
                },
                EntryParam::U32 {
                    name: "rows".into(),
                },
            ]
        };
        let [admit, partials, merge] = self.symbols();
        let mut admission = vec![
            ptr("activations", "const uint8_t"),
            ptr("activation_scales", "const uint8_t"),
            ptr("partials", "uint32_t"),
        ];
        admission.extend(tail());
        admission[4] = ptr("status", "uint32_t");
        let mut partial = vec![
            ptr("activations", "const uint8_t"),
            ptr("activation_scales", "const uint8_t"),
            ptr("weights", "const uint8_t"),
            ptr("weight_scales", "const uint8_t"),
            ptr("weights2", "const uint8_t"),
            ptr("weight_scales2", "const uint8_t"),
            ptr("partials", "float"),
        ];
        partial.extend(tail());
        let output_type = if q.output == DType::F32 {
            "float"
        } else {
            "uint16_t"
        };
        let mut final_params = vec![
            ptr("partials", "const float"),
            ptr("output", output_type),
            ptr("output2", output_type),
        ];
        final_params.extend(tail());
        vec![
            EntrySpec::new(
                admit,
                admission,
                format!(
                    "chop::gemm::mxfp8_fold_decode::admit<{},{},{},{}>(activations,activation_scales,partials,live_rows,status,status_row_offset,rows);",
                    q.k, q.groups, q.activation_row_stride, q.activation_group_stride
                ),
                128,
                1,
            ),
            EntrySpec::new(
                partials,
                partial,
                format!(
                    "chop::gemm::mxfp8_fold_decode::partials<{},{},{},{},{},{}>(activations,activation_scales,weights,weight_scales,weights2,weight_scales2,partials,live_rows,status,status_row_offset,rows);",
                    q.n,
                    q.k,
                    q.groups,
                    q.activation_row_stride,
                    q.activation_group_stride,
                    self.paired
                ),
                64,
                1,
            ),
            EntrySpec::new(
                merge,
                final_params,
                format!(
                    "chop::gemm::mxfp8_fold_decode::merge<{},{},{},{},{}>(partials,output,output2,live_rows,status,status_row_offset,rows);",
                    q.n,
                    q.k,
                    q.groups,
                    q.output == DType::F32,
                    self.paired
                ),
                128,
                1,
            ),
        ]
    }

    pub fn declarations(&self) -> Result<Vec<EntryDeclaration>, Refusal> {
        self.validate()?;
        self.specs()
            .into_iter()
            .map(|spec| {
                Ok(EntryDeclaration {
                    symbol: spec.symbol,
                    operation: "MXFP8 split fold GEMM".into(),
                    contract: self.query.numerical_contract_id()?.to_string(),
                    implementation: "sm120.mxfp8-fold.split.v1".into(),
                    requirements: [
                        ("entry".into(), spec.body),
                        ("fold_k".into(), "128".into()),
                        ("folds_per_cta".into(), "2".into()),
                        ("stages".into(), "2".into()),
                        ("paired".into(), self.paired.to_string()),
                    ]
                    .into(),
                    correctness_case: Some("mxfp8-fold-bitwise".into()),
                })
            })
            .collect()
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("split MXFP8 cubin requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        for symbol in entry.symbols() {
            if !symbols.insert(symbol) {
                return Err(Refusal("duplicate split MXFP8 symbol"));
            }
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/mxfp8_fold_decode.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().flat_map(Entry::specs).collect(),
    })
}
