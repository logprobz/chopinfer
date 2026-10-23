//! Three-stage operand loads and a last-CTA ordered fold merge.
use super::mxfp8_fold;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, Refusal};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;

#[derive(Clone)]
pub struct Entry {
    pub query: Mxfp8FoldGemmQuery,
    pub symbol: String,
    pub paired: bool,
    pub prefetch_weights: bool,
}

impl std::fmt::Debug for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut entry = f.debug_struct("Entry");
        entry
            .field("query", &self.query)
            .field("symbol", &self.symbol)
            .field("paired", &self.paired);
        if self.prefetch_weights {
            entry.field("prefetch_weights", &true);
        }
        entry.finish()
    }
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
            prefetch_weights: false,
        };
        entry.validate()?;
        Ok(entry)
    }
    /// Opt in for persistent weight codes and scales that are ready before
    /// any PDL predecessor launches. Other inputs and workspace accesses
    /// remain after the dependency wait.
    pub fn with_weight_prefetch(mut self) -> Self {
        self.prefetch_weights = true;
        self
    }
    pub fn legacy(&self) -> Result<mxfp8_fold::Entry, Refusal> {
        mxfp8_fold::Entry::new(self.query, &self.symbol)
    }
    pub fn tile_n(&self) -> u32 {
        if self.query.n <= 2048 { 32 } else { 64 }
    }
    pub fn tiles(&self, rows: u32) -> u64 {
        u64::from(self.query.n.div_ceil(self.tile_n()))
            * u64::from(rows.max(1).div_ceil(16))
            * u64::from(self.query.groups)
            * if self.paired { 2 } else { 1 }
    }
    pub fn header(&self, rows: u32) -> Result<u64, Refusal> {
        self.tiles(rows)
            .checked_mul(4)
            .map(|n| 256 + n.div_ceil(256) * 256)
            .filter(|&n| n <= u64::from(u32::MAX))
            .ok_or(Refusal("MXFP8 ticket header exceeds device index bounds"))
    }
    /// An exact graph size fixes the partition. Each partition still writes
    /// every 128-K fold separately; no partition sum reaches the output.
    pub fn folds_per_cta(&self, rows: u32) -> u32 {
        let wanted = (u64::from(super::SM_COUNT) * 2)
            .div_ceil(self.tiles(rows))
            .max(1);
        let available = (u64::from(self.query.k.div_ceil(128)) / wanted).clamp(2, 16) as u32;
        1 << available.ilog2()
    }
    pub fn workspace(&self, rows: u32) -> Result<u64, Refusal> {
        let partials = [
            u64::from(rows),
            u64::from(self.query.groups),
            u64::from(self.query.n),
            u64::from(self.query.k.div_ceil(128)),
            if self.paired { 8 } else { 4 },
        ]
        .into_iter()
        .try_fold(1_u64, |n, v| n.checked_mul(v))
        .ok_or(Refusal("MXFP8 ticket partials overflow"))?;
        self.header(rows)?
            .checked_add(partials)
            .filter(|&n| n <= kernel_core::precision::MAX_TENSOR_BYTES)
            .ok_or(Refusal(
                "MXFP8 ticket workspace exceeds the supported bound",
            ))
    }
    pub fn validate(&self) -> Result<(), Refusal> {
        self.legacy()?;
        if self.query.m_capacity > 64 {
            return Err(Refusal("MXFP8 ticket folds require at most 64 rows"));
        }
        self.workspace(self.query.m_capacity)?;
        for rows in [0, 1, self.query.m_capacity] {
            self.plans(rows)?;
        }
        Ok(())
    }
    pub fn symbols(&self) -> [String; 2] {
        [
            format!(
                "{}_initialize_v{}",
                self.symbol,
                if self.prefetch_weights { 3 } else { 2 }
            ),
            self.symbol.clone(),
        ]
    }
    pub fn plans(&self, rows: u32) -> Result<[LaunchPlan; 2], Refusal> {
        if rows > self.query.m_capacity {
            return Err(Refusal("MXFP8 ticket rows exceed capacity"));
        }
        let q = self.query;
        let splits = q.k.div_ceil(128).div_ceil(self.folds_per_cta(rows));
        let z = q
            .groups
            .checked_mul(splits)
            .and_then(|v| v.checked_mul(if self.paired { 2 } else { 1 }))
            .ok_or(Refusal("MXFP8 ticket grid overflows"))?;
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
            plan(
                [q.n.div_ceil(self.tile_n()), rows.max(1).div_ceil(16), z],
                self.tile_n() * 4,
            ),
        ];
        for plan in &plans {
            crate::backend::check_plan(plan)?;
        }
        Ok(plans)
    }
    /// The weight reader on packed codes. Admission keeps its original symbol.
    pub fn packed_spec(&self) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        super::mxfp8_fold_packed::check(self.query.groups, self.query.n, self.query.k)?;
        let mut spec = self.specs().remove(1);
        spec.symbol = super::mxfp8_fold_packed::symbol(&spec.symbol);
        spec.body = spec.body.replace(">(activations", ",true>(activations");
        Ok(spec)
    }

    pub fn specs(&self) -> Vec<EntrySpec> {
        let q = self.query;
        let namespace = if self.prefetch_weights {
            "chop::gemm::mxfp8_fold_prefetch"
        } else {
            "chop::gemm::mxfp8_fold_ticket"
        };
        let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let rows = || EntryParam::U32 {
            name: "rows".into(),
        };
        let [init, compute] = self.symbols();
        vec![
            EntrySpec::new(
                init,
                vec![
                    ptr("activations", "const uint8_t"),
                    ptr("activation_scales", "const uint8_t"),
                    ptr("workspace", "uint8_t"),
                    ptr("live_rows", "const uint32_t"),
                    ptr("status", "uint32_t"),
                    EntryParam::U64 {
                        name: "status_row_offset".into(),
                    },
                    rows(),
                ],
                format!(
                    "{namespace}::initialize<{},{},{},{},{},{},{}>(activations,activation_scales,workspace,live_rows,status,status_row_offset,rows);",
                    q.n,
                    q.k,
                    q.groups,
                    q.activation_row_stride,
                    q.activation_group_stride,
                    self.tile_n(),
                    self.paired
                ),
                128,
                1,
            ),
            EntrySpec::new(
                compute,
                vec![
                    ptr("activations", "const uint8_t"),
                    ptr("activation_scales", "const uint8_t"),
                    ptr("weights", "const uint8_t"),
                    ptr("weight_scales", "const uint8_t"),
                    ptr("weights2", "const uint8_t"),
                    ptr("weight_scales2", "const uint8_t"),
                    ptr(
                        "output",
                        if q.output == DType::F32 {
                            "float"
                        } else {
                            "uint16_t"
                        },
                    ),
                    ptr(
                        "output2",
                        if q.output == DType::F32 {
                            "float"
                        } else {
                            "uint16_t"
                        },
                    ),
                    ptr("workspace", "uint8_t"),
                    ptr("live_rows", "const uint32_t"),
                    rows(),
                    EntryParam::U32 {
                        name: "folds_per_cta".into(),
                    },
                ],
                format!(
                    "{namespace}::run<{},{},{},{},{},{},{},{}>(activations,activation_scales,weights,weight_scales,weights2,weight_scales2,output,output2,workspace,live_rows,rows,folds_per_cta);",
                    q.n,
                    q.k,
                    q.groups,
                    q.activation_row_stride,
                    q.activation_group_stride,
                    self.tile_n(),
                    q.output == DType::F32,
                    self.paired
                ),
                self.tile_n() * 4,
                if self.tile_n() == 32 { 4 } else { 1 },
            ),
        ]
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("MXFP8 ticket cubin requires entries"));
    }
    let mut seen = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        for symbol in entry.symbols() {
            if !seen.insert(symbol) {
                return Err(Refusal("duplicate MXFP8 ticket symbol"));
            }
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![if entries.iter().any(|entry| entry.prefetch_weights) {
            "#include \"gemm/mxfp8_fold_prefetch.cuh\"".into()
        } else {
            "#include \"gemm/mxfp8_fold_ticket.cuh\"".into()
        }],
        defines: vec![],
        entries: entries.iter().flat_map(Entry::specs).collect(),
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn packed_admission_interval_matches_exact_bf16_operands() {
        use kernel_core::contract::{DType, WeightFormat};
        use kernel_core::precision::NibbleOrder;
        use kernel_core::precision::packed::{MatrixView, ScaleBytes};
        for scale in 3_u8..=246 {
            for code in 0_u8..=255 {
                for lane in 0..4 {
                    let mut codes = [0_u8; 32];
                    codes[..4].copy_from_slice(&[0x80, 0x7e, 0xfe, 0]);
                    codes[lane] = code;
                    let word = u32::from_le_bytes(codes[..4].try_into().unwrap());
                    let packed = ((word & 0x7f7f7f7f) + 0x01010101) & 0x80808080 == 0;
                    let scales = [scale];
                    let oracle = kernel_core::mxfp8_gemm::admit_operand(MatrixView {
                        tensor: "packed admission",
                        format: WeightFormat::Mxfp8,
                        rows: 1,
                        k: 32,
                        row_origin: 0,
                        k_origin: 0,
                        nibble_order: NibbleOrder::EvenLow,
                        codes: &codes,
                        local: Some(ScaleBytes {
                            dtype: DType::Ue8M0,
                            bytes: &scales,
                        }),
                        global: None,
                    })
                    .is_ok();
                    assert_eq!(packed, oracle, "scale {scale} code {code} lane {lane}");
                }
            }
        }
    }
}
