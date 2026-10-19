//! One-CTA composition of the existing window norm, quantizer, and store.
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};
use kernel_core::latent_attention::LatentKvFormat;

#[derive(Clone, Debug)]
pub struct Entry {
    pub kv: LatentKvFormat,
    pub capacity: u32,
    pub symbol: String,
}
impl Entry {
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if self.capacity == 0
            || self.capacity > crate::backend::MAX_CAPACITY
            || rows > self.capacity
        {
            return Err(Refusal("fused window rows exceed capacity"));
        }
        if !matches!(
            self.kv,
            LatentKvFormat::Bf16 | LatentKvFormat::Fp8E4M3Ue8M0Block32
        ) {
            return Err(Refusal("fused window stores ceil E4M3 products only"));
        }
        let plan = LaunchPlan {
            grid: [rows.max(1), 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1; 3],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }
    pub fn spec(&self) -> EntrySpec {
        let ptr = |name: &str, ty: &str| P::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let u32_arg = |name: &str| P::U32 { name: name.into() };
        let u64_arg = |name: &str| P::U64 { name: name.into() };
        EntrySpec::new(
            self.symbol.clone(),
            vec![
                ptr("input", "const float"),
                ptr("weight", "const uint16_t"),
                ptr("latent", "float"),
                ptr("cos_sin", "const float"),
                ptr("live_rows", "const uint32_t"),
                u32_arg("eps_bits"),
                ptr("status", "uint32_t"),
                u64_arg("status_row_offset"),
                ptr("quantized_codes", "uint8_t"),
                ptr("quantized_scales", "uint8_t"),
                ptr("positions", "const int32_t"),
                ptr("row_batch", "const int32_t"),
                ptr("first_positions", "const int32_t"),
                ptr("last_positions", "const int32_t"),
                ptr("codes", "uint8_t"),
                ptr("scales", "uint8_t"),
                u32_arg("sequences"),
                u32_arg("slots"),
                u32_arg("circular"),
                u64_arg("code_stride"),
                u64_arg("scale_stride"),
            ],
            format!(
                "chop::latent::window_fused::run<{}>(input,weight,latent,cos_sin,live_rows,eps_bits,status,status_row_offset,quantized_codes,quantized_scales,positions,row_batch,first_positions,last_positions,codes,scales,sequences,slots,circular,code_stride,scale_stride);",
                self.kv == LatentKvFormat::Bf16
            ),
            256,
            1,
        )
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("fused window cubin requires entries"));
    }
    for entry in entries {
        entry.plan(entry.capacity)?;
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"latent/window_fused.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    })
}
