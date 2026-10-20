//! BF16-rounded block quantizers, isolated from legacy quantizer cubins.

use super::precision::ShapeBinding;
use super::{ptr, u32_arg};
use kernel_build::{CubinSpec, EntryDeclaration, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, DType, LaunchPlan, NumericalContract, Refusal, Validate,
};
use kernel_core::precision::quantizer::{BlockQuantization, BlockQuantizeQuery, QuantizeOutput};
use kernel_core::properties::Invariance;

pub const HEADER: &str = "row/block_quantize.cuh";
pub const LAB_CUBIN: &str = "block_quantize_sm120";
pub const LAB_CAPACITY: u32 = 16;

#[derive(Clone, Debug)]
pub struct Entry {
    pub query: BlockQuantizeQuery,
    pub capacity: u32,
    pub symbol: String,
    /// Opt-in warp-parallel groups with a barrier before row output.
    pub parallel_groups: bool,
    /// The parallel groups computed once and kept in registers across the
    /// row-status barrier (`row/ceil_cached.cuh`); implies `parallel_groups`.
    pub cached_groups: bool,
    /// Prefill-only warp count, zero for the existing implementations.
    pub prefill_warps: u32,
}

impl Entry {
    pub fn new(
        query: BlockQuantizeQuery,
        capacity: u32,
        symbol: impl Into<String>,
    ) -> Result<Self, Refusal> {
        let entry = Self {
            query,
            capacity,
            symbol: symbol.into(),
            parallel_groups: false,
            cached_groups: false,
            prefill_warps: 0,
        };
        entry.validate()?;
        Ok(entry)
    }

    /// Select the parallel implementation for a separately named entry.
    pub fn parallel(mut self) -> Result<Self, Refusal> {
        self.parallel_groups = true;
        self.validate()?;
        Ok(self)
    }

    /// Select the one-pass parallel implementation for a separately named
    /// entry: the parallel entry's thread mapping and group arithmetic.
    pub fn cached(mut self) -> Result<Self, Refusal> {
        self.parallel_groups = true;
        self.cached_groups = true;
        self.validate()?;
        Ok(self)
    }

    /// Use a separately named prefill entry with bounded row parallelism.
    pub fn prefill(mut self, warps: u32) -> Result<Self, Refusal> {
        if ![4, 8, 16].contains(&warps) {
            return Err(Refusal("prefill ceil requires 4, 8 or 16 warps"));
        }
        self.prefill_warps = warps;
        self.validate()?;
        Ok(self)
    }

    pub fn threads(&self) -> u32 {
        if self.prefill_warps != 0 {
            self.prefill_warps * 32
        } else if self.parallel_groups {
            (self.query.width / 32).min(32) * 32
        } else {
            32
        }
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if self.prefill_warps != 0
            && (self.parallel_groups || ![4, 8, 16].contains(&self.prefill_warps))
        {
            return Err(Refusal(
                "prefill ceil needs a separate 4, 8 or 16-warp entry",
            ));
        }
        if self.cached_groups
            && (!self.parallel_groups
                || self.prefill_warps != 0
                || !self.query.width.is_multiple_of(32)
                || self.query.width > 8192)
        {
            return Err(Refusal(
                "cached ceil groups need the parallel decode mapping",
            ));
        }
        if (self.parallel_groups || self.prefill_warps != 0)
            && (!matches!(self.query.format, BlockQuantization::E4M3Ceil32 { .. })
                || self.query.output != QuantizeOutput::Codes)
        {
            return Err(Refusal("parallel groups require ceil E4M3 code output"));
        }
        if self.capacity == 0 || self.capacity > crate::backend::MAX_CAPACITY {
            return Err(Refusal("block quantization capacity exceeds SM120 bounds"));
        }
        if self.symbol.is_empty()
            || !self
                .symbol
                .bytes()
                .enumerate()
                .all(|(i, c)| c == b'_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
        {
            return Err(Refusal("block quantization symbol must be a C identifier"));
        }
        self.plan(self.capacity)?;
        Ok(())
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if rows > self.capacity {
            return Err(Refusal("block quantization rows exceed capacity"));
        }
        let plan = LaunchPlan {
            grid: [rows.max(1), 1, 1],
            block: [self.threads(), 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub fn invariance(&self) -> Invariance {
        Invariance::any_rows(self.capacity)
    }

    pub fn input_bytes(&self, rows: u32) -> u64 {
        u64::from(rows)
            * u64::from(self.query.width)
            * if self.query.input == DType::F32 { 4 } else { 2 }
    }

    pub fn output_bytes(&self, rows: u32) -> u64 {
        let elements = u64::from(rows) * u64::from(self.query.width);
        match (self.query.output, self.query.format) {
            (QuantizeOutput::Bf16, _) => elements * 2,
            (_, BlockQuantization::E4M3Ceil32 { .. }) => elements,
            _ => elements / 2,
        }
    }

    pub fn scale_bytes(&self, rows: u32) -> u64 {
        if self.query.output == QuantizeOutput::Bf16 {
            0
        } else {
            u64::from(rows) * u64::from(self.query.width / self.query.format.group_size())
        }
    }

    pub fn spec(&self) -> EntrySpec {
        self.spec_with_shapes(ShapeBinding::Compiled)
    }

    pub(crate) fn spec_with_shapes(&self, shapes: ShapeBinding) -> EntrySpec {
        if self.parallel_groups || self.prefill_warps != 0 {
            assert_eq!(shapes, ShapeBinding::Compiled);
        }
        let engine = if self.prefill_warps != 0 {
            "block_quantize_prefill"
        } else if self.cached_groups {
            "block_quantize_cached"
        } else if self.parallel_groups {
            "block_quantize_parallel"
        } else {
            "block_quantize"
        };
        let query = self.query;
        let (format, floor) = match query.format {
            BlockQuantization::E4M3Ceil32 { amax_floor_bits } => (0, amax_floor_bits),
            BlockQuantization::E2M1E4M3Scale16 => (1, 0),
            BlockQuantization::E2M1Ceil32 => (2, 0),
        };
        let mut params = vec![
            ptr(
                "input",
                if query.input == DType::F32 {
                    "const float"
                } else {
                    "const uint16_t"
                },
            ),
            ptr(
                "output",
                if query.output == QuantizeOutput::Bf16 {
                    "uint16_t"
                } else {
                    "uint8_t"
                },
            ),
            ptr("scales", "uint8_t"),
            ptr("live_rows", "const uint32_t"),
            ptr("status", "uint32_t"),
            kernel_build::EntryParam::U64 {
                name: "status_row_offset".into(),
            },
        ];
        let (width, extra) = if shapes == ShapeBinding::Launch {
            params.push(u32_arg("width"));
            (0, ",width")
        } else {
            (query.width, "")
        };
        let prefill = if self.prefill_warps == 0 {
            String::new()
        } else {
            format!(",{}", self.prefill_warps)
        };
        EntrySpec::new(
            self.symbol.clone(),
            params,
            format!(
                "chop::row::{engine}::run<{width},{format},{},{},{floor}u{prefill}>(input,output,scales,live_rows,status,status_row_offset{extra});",
                query.input == DType::F32,
                query.output == QuantizeOutput::Bf16
            ),
            self.threads(),
            1,
        )
    }

    pub fn declaration(&self) -> Result<EntryDeclaration, Refusal> {
        self.validate()?;
        let entry = self.spec();
        Ok(EntryDeclaration {
            symbol: self.symbol.clone(),
            operation: "block quantize".into(),
            contract: self.query.numerical_contract_id()?.to_string(),
            implementation: "sm120.block-quantize.status-or-zero.v2".into(),
            requirements: [
                ("query".into(), format!("{:?}", self.query)),
                ("arguments".into(), format!("{:?}", entry.params)),
                ("entry".into(), entry.body),
                ("pdl".into(), "true".into()),
            ]
            .into(),
            correctness_case: Some("block-quantize-bitwise".into()),
        })
    }

    pub fn arguments(
        &self,
        input: u64,
        output: u64,
        scales: Option<u64>,
        live: u64,
        status: u64,
        status_row_offset: u64,
    ) -> Result<ArgumentList, Refusal> {
        if scales.is_some() != (self.query.output == QuantizeOutput::Codes) {
            return Err(Refusal(
                "block quantization scales must match its output form",
            ));
        }
        super::pack_arguments(
            &self.spec().params,
            vec![
                Argument::DeviceBuffer(input),
                Argument::DeviceBuffer(output),
                scales.map_or(Argument::NullDevicePointer, Argument::DeviceBuffer),
                Argument::DeviceBuffer(live),
                Argument::DeviceBuffer(status),
                Argument::U64(status_row_offset),
            ],
        )
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("block quantization cubin requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        if !symbols.insert(&entry.symbol) {
            return Err(Refusal("duplicate block quantization symbol"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: [
            Some(if entries.iter().any(|e| e.prefill_warps != 0) {
                "row/ceil_prefill.cuh"
            } else if entries
                .iter()
                .any(|e| e.parallel_groups && !e.cached_groups)
            {
                "row/ceil_parallel.cuh"
            } else if entries.iter().any(|e| e.cached_groups) {
                "row/ceil_cached.cuh"
            } else {
                HEADER
            }),
            // Cubins without cached entries keep their single header.
            entries
                .iter()
                .any(|e| e.cached_groups)
                .then_some("row/ceil_cached.cuh"),
        ]
        .into_iter()
        .flatten()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|h| format!("#include \"{h}\""))
        .collect(),
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    })
}

pub fn lab_entries() -> Vec<Entry> {
    let mut entries = Vec::new();
    for width in [32, 128, 512, 576] {
        for (tag, format) in [
            ("e4m3", BlockQuantization::E4M3_CEIL32),
            ("e2m1_e4m3", BlockQuantization::E2M1E4M3Scale16),
            ("e2m1_ue8m0", BlockQuantization::E2M1Ceil32),
        ] {
            for (input_name, input) in [("bf16", DType::Bf16), ("f32", DType::F32)] {
                for (output_name, output) in [
                    ("codes", QuantizeOutput::Codes),
                    ("fake", QuantizeOutput::Bf16),
                ] {
                    entries.push(
                        Entry::new(
                            BlockQuantizeQuery {
                                width,
                                input,
                                format,
                                output,
                            },
                            LAB_CAPACITY,
                            format!(
                                "chop_block_quantize_{tag}_{input_name}_{output_name}_w{width}_v2"
                            ),
                        )
                        .unwrap(),
                    );
                }
            }
        }
    }
    entries
}

#[cfg(test)]
mod prefill_tests {
    use super::*;
    #[test]
    fn prefill_warps_preserve_the_abi_and_validate_launch_bounds() {
        let baseline = Entry::new(
            BlockQuantizeQuery {
                width: 32768,
                input: DType::F32,
                format: BlockQuantization::E4M3_CEIL32,
                output: QuantizeOutput::Codes,
            },
            8192,
            "chop_ceil_reference_v1",
        )
        .unwrap();
        for warps in [4, 8, 16] {
            let e = baseline.clone().prefill(warps).unwrap();
            assert_eq!(e.spec().params, baseline.spec().params);
            assert_eq!(e.invariance(), baseline.invariance());
            for rows in [0, 1, 32, 1024, 2048, 8192] {
                let p = e.plan(rows).unwrap();
                assert_eq!(p.grid, [rows.max(1), 1, 1]);
                assert_eq!(p.block, [warps * 32, 1, 1]);
                assert_eq!(p.dynamic_shared_bytes, 0);
                assert_eq!(p.cluster, [1, 1, 1]);
                assert_eq!(p.workspace_bytes, 0);
            }
            assert!(e.plan(8193).is_err());
            assert!(e.clone().parallel().is_err());
        }
        for warps in [0, 1, 2, 3, 32, 64] {
            assert!(baseline.clone().prefill(warps).is_err());
        }
        let mut fake = baseline.clone();
        fake.query.output = QuantizeOutput::Bf16;
        assert!(fake.prefill(8).is_err());
        let mut fp4 = baseline;
        fp4.query.format = BlockQuantization::E2M1Ceil32;
        assert!(fp4.prefill(8).is_err());
    }
}

#[cfg(test)]
mod cached_tests {
    use super::*;
    fn entry(width: u32, input: DType) -> Entry {
        Entry::new(
            BlockQuantizeQuery {
                width,
                input,
                format: BlockQuantization::E4M3_CEIL32,
                output: QuantizeOutput::Codes,
            },
            32,
            format!("q{width}"),
        )
        .unwrap()
    }
    #[test]
    fn cached_groups_keep_the_parallel_mapping_and_abi() {
        for (width, input) in [(8192, DType::Bf16), (2048, DType::F32), (1280, DType::F32)] {
            let parallel = entry(width, input).parallel().unwrap();
            let cached = entry(width, input).cached().unwrap();
            assert_eq!(cached.threads(), parallel.threads());
            assert_eq!(cached.plan(8).unwrap(), parallel.plan(8).unwrap());
            let (a, b) = (parallel.spec(), cached.spec());
            assert_eq!(a.params, b.params);
            assert!(b.body.starts_with("chop::row::block_quantize_cached::run<"));
            assert_eq!(
                a.body
                    .replace("block_quantize_parallel", "block_quantize_cached"),
                b.body
            );
        }
        let mut fake = entry(512, DType::Bf16);
        fake.query.output = QuantizeOutput::Bf16;
        assert!(fake.cached().is_err());
        let mut prefill = entry(512, DType::Bf16).prefill(8).unwrap();
        prefill.cached_groups = true;
        assert!(prefill.validate().is_err());
    }
    #[test]
    fn existing_cubins_keep_their_single_header() {
        let headers = |entries: &[Entry]| cubin_spec("m", entries).unwrap().headers;
        let p = entry(8192, DType::Bf16).parallel().unwrap();
        let c = entry(2048, DType::F32).cached().unwrap();
        let mut c2 = c.clone();
        c2.symbol = "other".into();
        assert_eq!(
            headers(std::slice::from_ref(&p)),
            ["#include \"row/ceil_parallel.cuh\""]
        );
        assert_eq!(
            headers(&[entry(32, DType::Bf16)]),
            [format!("#include \"{HEADER}\"")]
        );
        assert_eq!(
            headers(std::slice::from_ref(&c)),
            ["#include \"row/ceil_cached.cuh\""]
        );
        assert_eq!(
            headers(&[p, c2]),
            [
                "#include \"row/ceil_cached.cuh\"",
                "#include \"row/ceil_parallel.cuh\""
            ]
        );
    }
}
