//! Per-operation CPU checks of Qwen3.8 Flash bundle dumps: QSA selections
//! recomputed from a dump's own projection rows, batch drift between dumps
//! and against the GPU reference runner's taps, and prepared projections on
//! captured operands. Whole-model references come from the GPU reference
//! runner (`models/qwen38/reference.py`), not from this module.

pub mod drift;
pub mod linalg;
pub mod prepared_capture;
pub mod prompts;
pub mod selection;
pub mod tap_names;
pub mod trace;
pub mod weights;

pub const LAYERS: usize = model_qwen38_contract::contract::LAYERS as usize;
/// `rms_norm_eps` 1e-6 as the F32 every reference norm adds.
pub const EPS_BITS: u32 = model_qwen38_contract::contract::EPS.to_bits();
pub const EPS: f64 = model_qwen38_contract::contract::EPS as f64;
