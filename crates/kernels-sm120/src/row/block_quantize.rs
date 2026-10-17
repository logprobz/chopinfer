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
        };
        entry.validate()?;
        Ok(entry)
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
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
            grid: [
                rows.max(1),
                self.query.width / self.query.format.group_size(),
                1,
            ],
            block: [32, 1, 1],
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
        ];
        let (width, extra) = if shapes == ShapeBinding::Launch {
            params.push(u32_arg("width"));
            (0, ",width")
        } else {
            (query.width, "")
        };
        EntrySpec::new(
            self.symbol.clone(),
            params,
            format!(
                "chop::row::block_quantize::run<{width},{format},{},{},{floor}u>(input,output,scales,live_rows{extra});",
                query.input == DType::F32,
                query.output == QuantizeOutput::Bf16
            ),
            32,
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
            implementation: "sm120.block-quantize.v1".into(),
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
        headers: vec![format!("#include \"{HEADER}\"")],
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
                                "chop_block_quantize_{tag}_{input_name}_{output_name}_w{width}_v1"
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
