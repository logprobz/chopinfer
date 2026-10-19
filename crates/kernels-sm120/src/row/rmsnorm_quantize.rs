//! Plain RMSNorm with its original output, optional widening, and ceil codes.
use super::{pack_arguments, ptr};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan, Refusal, Validate};
use kernel_core::rmsnorm::{RmsNormNumerics, RmsNormRequirement};
use kernel_core::rmsnorm_quantize::{RmsNormQuantizeBuffers, RmsNormQuantizeQuery};

#[derive(Clone, Debug)]
pub struct Entry {
    pub query: RmsNormQuantizeQuery,
    pub capacity: u32,
    pub symbol: String,
}

/// Resolve the same default numerics as the standalone row norm.
pub fn resolved(query: RmsNormQuantizeQuery) -> RmsNormQuantizeQuery {
    RmsNormQuantizeQuery {
        norm: RmsNormRequirement {
            numerics: Some(query.norm.numerics.unwrap_or(RmsNormNumerics::TREE_256)),
            ..query.norm
        },
        ..query
    }
}

impl Entry {
    pub fn new(
        query: RmsNormQuantizeQuery,
        capacity: u32,
        symbol: impl Into<String>,
    ) -> Result<Self, Refusal> {
        let entry = Self {
            query: resolved(query),
            capacity,
            symbol: symbol.into(),
        };
        entry.validate()?;
        Ok(entry)
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        super::rmsnorm::RmsNorm::new(
            self.query.norm,
            self.capacity,
            super::rmsnorm::RmsNormTarget::SM120F,
        )?;
        if self.query.norm.numerics != Some(RmsNormNumerics::TREE_256) {
            return Err(Refusal("fused norm requires the original 256-thread tree"));
        }
        // Reuse the quantizer's capacity and C-identifier admission.
        super::block_quantize::Entry::new(self.query.quantizer(), self.capacity, &self.symbol)?;
        Ok(())
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if rows == 0 || rows > self.capacity {
            return Err(Refusal("fused norm rows exceed capacity or are zero"));
        }
        let plan = LaunchPlan {
            grid: [rows, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            // The wrapper defers the reused norm's early signal.
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub fn spec(&self) -> EntrySpec {
        let width = self.query.norm.width;
        let f32 = self.query.norm.input == DType::F32;
        let floor = self.query.amax_floor_bits;
        EntrySpec::new(
            self.symbol.clone(),
            vec![
                ptr("input", "const void"),
                ptr("weight", "const uint16_t"),
                ptr("normed", "void"),
                ptr("widened", "float"),
                ptr("codes", "uint8_t"),
                ptr("scales", "uint8_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("status", "uint32_t"),
                EntryParam::U64 {
                    name: "status_row_offset".into(),
                },
                EntryParam::U32 {
                    name: "eps_bits".into(),
                },
            ],
            format!(
                "chop::row::rmsnorm_quantize::run<{width},{f32},{floor:#x}u>(input,weight,normed,widened,codes,scales,live_rows,status,status_row_offset,__uint_as_float(eps_bits));"
            ),
            256,
            1,
        )
    }

    pub fn arguments(&self, b: RmsNormQuantizeBuffers<u64>) -> Result<ArgumentList, Refusal> {
        let widened = b
            .widened
            .map_or(Argument::NullDevicePointer, Argument::DeviceBuffer);
        pack_arguments(
            &self.spec().params,
            vec![
                Argument::DeviceBuffer(b.input),
                Argument::DeviceBuffer(b.weight),
                Argument::DeviceBuffer(b.normed),
                widened,
                Argument::DeviceBuffer(b.codes),
                Argument::DeviceBuffer(b.scales),
                Argument::DeviceBuffer(b.live_rows),
                Argument::DeviceBuffer(b.status),
                Argument::U64(b.status_row_offset),
                Argument::U32(self.query.norm.epsilon_bits),
            ],
        )
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("fused norm cubin requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        if !symbols.insert(&entry.symbol) {
            return Err(Refusal("duplicate fused norm symbol"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"row/rmsnorm_quantize.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    })
}
