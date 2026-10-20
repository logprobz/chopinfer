//! Exact widened-BF16 router rows, qualified at explicit prefill counts.
use kernel_build::CubinSpec;
use kernel_core::{
    contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat},
    precision::{ExecutionPhase, OperationUse},
};
use kernels_sm120::f32_projection::{Entry, zero_prefill};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
pub const FAMILY: &str = "f32_zero_prefill_serving";
pub const CUBIN: &str = "f32_zero_prefill_deepseek41_serving_sm120";
/// Exact bucket qualification on two devices at 250 W. M32 is selected;
/// M64's lower occupancy loses despite its additional activation reuse.
pub const ROWS: &[u32] = &[
    320, 448, 512, 576, 704, 832, 960, 1000, 1024, 1088, 1216, 1344, 1472, 1600, 1728, 1856, 1984,
    2048, 2112, 2240, 2368, 2496, 4095, 4096, 8150, 8191, 8192,
];
pub fn selected(g: &Geometry, ranks: u32, experts: u32) -> bool {
    g.id == FLASH.id && ranks == 4 && experts == 384
}
pub fn entry() -> Entry {
    Entry::new(
        GemmQuery {
            m_capacity: 8192,
            n: 384,
            k: 5120,
            weight: WeightFormat::Bf16,
            activation: ActivationFormat::F32,
            output: DType::F32,
            epilogues: Epilogues::default(),
            grouped: None,
        },
        OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        },
        "chop_deepseek41_dense_f32_m8192_n384_k5120",
    )
    .unwrap()
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<CubinSpec>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(vec![]);
    }
    let e = entry();
    let mut spec =
        zero_prefill::cubin_spec(CUBIN, std::slice::from_ref(&e)).map_err(|e| e.to_string())?;
    spec.entries.retain(|s| {
        s.symbol == zero_prefill::prepare_symbol(&e) || s.symbol == zero_prefill::symbol(&e, 32)
    });
    Ok(vec![spec])
}
