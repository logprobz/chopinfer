//! The grouped-MoE template (`cuda/moe/grouped*.cuh`) against the per-model
//! engines of MiMo V2.6 (MXFP4) and Qwen3.8 (FP8 block-128) grouped entries.
//!
//! `kernels_sm120::moe::template::lab_instances` pairs every served A16
//! entry, at each served rank count, every decode bucket and the prefill
//! capacity, with its engine and template implementations at
//! the same shape, schedule, ABI and launch plan. The template is served;
//! the engine is its frozen kernel-lab reference (`chop_moe_eng_` cubins
//! built from `kernel-lab/cuda/baselines`). Both run on identical buffers:
//! random weights over every finite code with wide-range scales (some FP8
//! blocks zero), activations over 31 binades, skewed and hot routes, at
//! each live count of `poison::live_counts`. Outputs and workspaces are
//! poisoned before every launch; each launch must write exactly its live
//! routed rows and leave its workspace untouched, and the two outputs must
//! agree bit for bit.
//! `CHOPINFER_MIMO_CHECKPOINT` and `CHOPINFER_QWEN38_CHECKPOINT` replace
//! local experts 0-7 with layer 1's real experts (the hot route targets
//! them); without them the real-weight cases are skipped and the receipt
//! says so.
//!
//! The ignored `grouped_moe_template_timing` alternates the two entries
//! (A B A B per repetition, L2 flushed before each batch of launches) and
//! gates the median ratio of adjacent pairs per shape.
//!
//! `grouped_moe_packed_ring_matches_the_packed_engine` does the same for the
//! packed NVFP4 and MXFP8 decode entries at Qwen3.8 TP4's shape
//! (`template::packed_lab_instances`): the packed engine that serves them
//! outside the template against the template's packed ring, at every live
//! count from 0 through each bucket, on NVFP4 banks of random codes, every
//! admitted UE4M3 local scale (2% zero) and per-expert F32 globals, on MXFP8
//! banks of random codes with UE8M0 scales 115-139 and of every UE8M0 code
//! 0-254 (with the E4M3 codes whose scaled products are exact, and
//! activations that cancel each scale block's base exponent so the outputs
//! stay finite and nonzero), adding crowded routes that give an expert
//! several rows of each token. The receipt records these domains, the
//! written and finite nonzero output elements per bank, and the cases with
//! finite nonzero outputs.
//! `CHOPINFER_QWEN38_NVFP4_CHECKPOINT` (NVIDIA's mixed export) replaces
//! local experts 0-7 with layer 1's real NVFP4 experts. The ignored
//! `grouped_moe_packed_ring_timing` times them A B A B with captured graphs
//! of 32 launches that rotate through eight banks and 32 route sets, so
//! weights come from DRAM as in decode and host-side launch preparation
//! stays out of the timing.
//!
//! `CHOPINFER_MOE_TEMPLATE=<substring>` selects instances by served symbol
//! (packed instances by their ring symbol).
//! `CHOPINFER_MOE_TIMING_LOCAL=1,2,3` makes the packed timing and sweep route
//! sets give every token exactly those counts of local experts in turn, as
//! one-token decode spreads its routes over four ranks. The ignored
//! `grouped_moe_one_token_floor` times the served FP8 one-token entries
//! against the NVFP4 packed ring on such routes, cold and warm in L2.

use kernel_build::tuning::{self, Comparison, Knob, Table};
use kernel_build::{BuildOptions, CubinSpec, Toolchain};
use kernel_core::codec::{bf16_to_f32, decode_ue8m0_f64};
use kernel_core::contract::{
    ArgumentList, DType, LaunchPlan, RoutedCapacity, RuntimeCounts, WeightFormat,
};
use kernel_core::select::{NanKeys, Ties, topk};
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_core::transform::{Mxfp4MoeFragmentLayout, Transform};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::{LabResult, Session, gpu_test, time};
use kernels_sm120::Profile;
use kernels_sm120::moe::template::{self, Instance, PackedInstance, Schedule, ScheduleKnobs};
use kernels_sm120::moe::{GroupedOperation, Projection, packed_scale_offset};
use kernels_sm120::{Model, RankShape};
use runtime_cuda::{DeviceBuffer, Function};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

const HANDLES: [u64; 15] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const WEIGHT_HANDLE: u64 = 16;
/// The Qwen3.8 rank whose experts the cases run.
const QWEN_RANK: u32 = 1;
const QWEN_GLOBAL_EXPERTS: usize = 512;
/// Local experts replaced by real weights when a checkpoint is present.
const REAL_EXPERTS: usize = 8;
const REAL_LAYER: usize = 1;

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn selected(instance: &Instance) -> bool {
    std::env::var("CHOPINFER_MOE_TEMPLATE")
        .ok()
        .is_none_or(|filter| instance.baseline.symbol.contains(&filter))
}

/// The selected instances. A filter that selects none is an error, so a
/// mistyped `CHOPINFER_MOE_TEMPLATE` cannot pass vacuously.
fn instances() -> Vec<Instance> {
    let selected = template::lab_instances(kernel_lab::catalogs())
        .into_iter()
        .filter(selected)
        .collect::<Vec<_>>();
    assert!(
        !selected.is_empty(),
        "CHOPINFER_MOE_TEMPLATE={:?} selects no template instance",
        std::env::var("CHOPINFER_MOE_TEMPLATE").unwrap_or_default()
    );
    selected
}

/// splitmix64 of a seed: independent 64-bit words.
fn mix(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn uniform(seed: u64) -> f64 {
    (mix(seed) >> 11) as f64 / (1u64 << 53) as f64
}

/// xorshift bytes, fast enough for gigabyte weight banks.
fn random_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = mix(seed) | 1;
    let mut bytes = Vec::with_capacity(len + 8);
    while bytes.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        bytes.extend_from_slice(&state.to_le_bytes());
    }
    bytes.truncate(len);
    bytes
}

fn u32_bytes(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_ne_bytes).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(16))?;
    if !bytes.is_empty() {
        buffer
            .copy_from_bytes(bytes, &session.stream)
            .map_err(err)?;
    }
    Ok(buffer)
}

/// BF16 bits over 31 binades (2^-24 to 2^6), both signs, with 1% zeros.
fn wide_bf16(seed: u64, len: usize) -> Vec<u8> {
    bf16_binades(seed, len, -24, 31)
}

/// BF16 bits over `binades` binades from 2^`lowest`, both signs, with 1%
/// zeros.
fn bf16_binades(seed: u64, len: usize, lowest: i32, binades: u32) -> Vec<u8> {
    let words = random_bytes(seed, len * 4);
    words
        .chunks_exact(4)
        .flat_map(|w| {
            let bits = u32::from_le_bytes(w.try_into().unwrap());
            let value = if bits % 100 == 0 {
                0u16
            } else {
                let exponent = (127 + lowest) as u32 + (bits >> 8) % binades;
                let mantissa = (bits >> 16) & 0x7f;
                let sign = (bits >> 23) & 1;
                ((sign << 15) | (exponent << 7) | mantissa) as u16
            };
            value.to_le_bytes()
        })
        .collect()
}

/// A bank's weight format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    /// MiMo: fragment E2M1 codes, UE8M0 scale tiles.
    Mxfp4,
    /// Qwen3.8: fragment E4M3 codes, F32 128 x 128 block scales.
    Fp8,
    /// Packed row-major E2M1 codes, UE4M3 scale tiles, F32 expert globals.
    Nvfp4,
    /// Packed row-major E4M3 codes, UE8M0 scale tiles.
    Mxfp8,
}

/// One projection's served shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Geometry {
    format: Format,
    experts: usize,
    top_k: usize,
    n: usize,
    k: usize,
    gate_up: bool,
}

impl Geometry {
    fn of(instance: &Instance) -> Self {
        Self::of_query(&instance.baseline.query, instance.baseline.projection)
    }

    fn of_query(query: &kernel_core::contract::GemmQuery, projection: Projection) -> Self {
        let group = query.grouped.expect("grouped GEMM");
        Self {
            format: match query.weight {
                WeightFormat::Mxfp4 => Format::Mxfp4,
                WeightFormat::Fp8Block128x128(_) => Format::Fp8,
                WeightFormat::Nvfp4 => Format::Nvfp4,
                WeightFormat::Mxfp8 => Format::Mxfp8,
                other => panic!("no template bank for {other:?}"),
            },
            experts: group.experts as usize,
            top_k: group.top_k as usize,
            n: query.n as usize,
            k: query.k as usize,
            gate_up: projection == Projection::GateUp,
        }
    }

    fn out_n(&self) -> usize {
        if self.gate_up { self.n / 2 } else { self.n }
    }

    fn code_bytes(&self) -> usize {
        self.experts * self.n * self.row_bytes()
    }

    fn row_bytes(&self) -> usize {
        match self.format {
            Format::Fp8 | Format::Mxfp8 => self.k,
            Format::Mxfp4 | Format::Nvfp4 => self.k / 2,
        }
    }

    /// Local scale blocks per row of the packed formats.
    fn scale_blocks(&self) -> usize {
        self.k / if self.format == Format::Nvfp4 { 16 } else { 32 }
    }

    /// Bytes of the packed formats' 16-row by 8-block scale tiles.
    fn scale_tile_bytes(&self) -> usize {
        self.experts * self.n / 16 * self.scale_blocks().div_ceil(8) * 128
    }

    /// NVFP4 F32 globals: one per expert and projection.
    fn globals(&self) -> usize {
        self.experts * if self.gate_up { 2 } else { 1 }
    }
}

/// One projection's weights on the device, shared by its buckets.
struct Bank {
    codes: DeviceBuffer,
    /// MXFP4 or MXFP8 UE8M0 and NVFP4 UE4M3 scale tiles, or a dummy for FP8.
    scales: DeviceBuffer,
    /// FP8 F32 block scales, NVFP4 F32 expert globals, or a dummy.
    f32_scales: DeviceBuffer,
    real: Option<String>,
}

/// FP8 E4M3 codes over every finite value: the two NaN codes become the
/// largest finite magnitudes.
fn fp8_codes(seed: u64, len: usize) -> Vec<u8> {
    random_bytes(seed, len)
        .into_iter()
        .map(|b| if b & 0x7f == 0x7f { b - 1 } else { b })
        .collect()
}

/// F32 block scales from 2^-20 to 2^4, with 3% zero blocks (prefill skips
/// them, decode promotes them by zero).
fn fp8_block_scales(seed: u64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let u = uniform(seed ^ (i as u64) << 1);
            if u < 0.03 {
                0.0
            } else {
                (-20.0 + 24.0 * uniform(seed ^ (i as u64) << 1 ^ 1)).exp2() as f32
            }
        })
        .collect()
}

/// UE8M0 scales 2^-12 to 2^12.
fn ue8m0_scales(seed: u64, len: usize) -> Vec<u8> {
    random_bytes(seed, len)
        .into_iter()
        .map(|b| 115 + b % 25)
        .collect()
}

/// UE4M3 local scales over every admitted code (0 to 0x7e: zero, the
/// subnormals and every normal value up to 448), 2% of them zero.
fn ue4m3_scales(seed: u64, len: usize) -> Vec<u8> {
    random_bytes(seed, len * 2)
        .chunks_exact(2)
        .map(|b| if b[1] < 5 { 0 } else { b[0] % 0x7f })
        .collect()
}

/// E4M3 codes whose product with UE8M0 scale `s` is zero or a finite
/// normal BF16 value: exactly the pairs the scaled conversion admits
/// without its inexact-product trap.
fn exact_mxfp8_codes(s: u8) -> Vec<u8> {
    const BF16_MAX: f64 = 3.389_531_389_251_535_5e38;
    (0..=255u8)
        .filter(|&code| code & 0x7f != 0x7f)
        .filter(|&code| {
            let v = f64::from(kernel_core::codec::SmallFloat::E4M3.decode_f32(code));
            let p = v * decode_ue8m0_f64(s);
            v == 0.0 || (p.abs() >= 2f64.powi(-126) && p.abs() <= BF16_MAX)
        })
        .collect()
}

/// The every-code MXFP8 bank's base UE8M0 code of scale block `block`:
/// 0 through 254 spread evenly over a row's `blocks`.
fn boundary_base(block: usize, blocks: usize) -> i32 {
    ((block * 254 + (blocks - 1) / 2) / (blocks - 1)) as i32
}

/// How far a row's scale strays from its block's base: half the widest gap
/// between neighbouring bases, so the rows cover every code 0 to 254.
fn boundary_spread(blocks: usize) -> i32 {
    254usize.div_ceil(blocks - 1).div_ceil(2) as i32
}

/// The UE8M0 scale of row `row`, block `block` in the every-code bank.
fn boundary_scale(row: usize, block: usize, blocks: usize) -> u8 {
    let d = boundary_spread(blocks);
    let delta = ((7 * row + 3 * block) % (2 * d as usize + 1)) as i32 - d;
    (boundary_base(block, blocks) + delta).clamp(0, 254) as u8
}

/// The binade of the activations that meet scale block `block` of the
/// every-code bank: 2^(118 - spread - base), within the normal BF16 range,
/// so a product (E4M3 at most 448, activation below 2) is at most 2^3.
fn boundary_activation_exponent(block: usize, blocks: usize) -> i32 {
    (118 - boundary_spread(blocks) - boundary_base(block, blocks)).clamp(-126, 127)
}

/// An MXFP8 bank over every admitted UE8M0 scale code (0 to 254): block b
/// of every row has a scale within `boundary_spread` of its base code (the
/// bases run from 0 to 254 over the blocks), and each code of the block is
/// drawn from the E4M3 codes whose scaled product is exact. Activations
/// from `boundary_activations` cancel each block's base exponent, so every
/// product is at most 2^3 and gate/up sums stay small enough for SwiGLU to
/// give finite, nonzero outputs.
fn mxfp8_boundary_bank(g: Geometry, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let rows = g.experts * g.n;
    let blocks = g.scale_blocks();
    let allowed: Vec<Vec<u8>> = (0..=254u8).map(exact_mxfp8_codes).collect();
    let mut scales = vec![0u8; g.scale_tile_bytes()];
    let mut codes = vec![0u8; g.code_bytes()];
    let noise = random_bytes(seed, g.code_bytes());
    for row in 0..rows {
        for block in 0..blocks {
            let s = boundary_scale(row, block, blocks);
            scales[packed_scale_offset(row, block, blocks)] = s;
            let choices = &allowed[usize::from(s)];
            for k in block * 32..(block + 1) * 32 {
                let at = row * g.k + k;
                codes[at] = choices[usize::from(noise[at]) % choices.len()];
            }
        }
    }
    (codes, scales)
}

/// Activations for the every-code MXFP8 bank, `len` BF16 values of rows of
/// K: column k is +-(1 to 2) x 2^`boundary_activation_exponent` of its
/// scale block, with 1% zeros.
fn boundary_activations(g: Geometry, seed: u64, len: usize) -> Vec<u8> {
    let blocks = g.scale_blocks();
    random_bytes(seed, len * 4)
        .chunks_exact(4)
        .enumerate()
        .flat_map(|(i, w)| {
            let bits = u32::from_le_bytes(w.try_into().unwrap());
            let value = if bits % 100 == 0 {
                0u16
            } else {
                let exponent = (127 + boundary_activation_exponent(i % g.k / 32, blocks)) as u32;
                let mantissa = (bits >> 16) & 0x7f;
                let sign = (bits >> 23) & 1;
                ((sign << 15) | (exponent << 7) | mantissa) as u16
            };
            value.to_le_bytes()
        })
        .collect()
}

/// Positive F32 NVFP4 globals from 2^-14 to 2^2 with random mantissas.
fn nvfp4_globals(seed: u64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (-14.0 + 16.0 * uniform(seed ^ (i as u64) << 3)).exp2() as f32)
        .collect()
}

impl Bank {
    fn new(session: &Session, g: Geometry, profile: Profile, seed: u64) -> LabResult<Self> {
        let codes = match g.format {
            Format::Fp8 | Format::Mxfp8 => fp8_codes(seed, g.code_bytes()),
            Format::Mxfp4 | Format::Nvfp4 => random_bytes(seed, g.code_bytes()),
        };
        let codes = upload(session, &codes)?;
        let (scales, f32_scales) = match g.format {
            Format::Fp8 => {
                let f32 = fp8_block_scales(seed ^ 0x5ca1e, g.experts * g.n / 128 * (g.k / 128));
                (upload(session, &[0; 16])?, session.upload_f32(&f32)?)
            }
            Format::Mxfp4 => {
                let ue8m0 = ue8m0_scales(seed ^ 0x5ca1e, g.experts * g.n * (g.k / 32));
                (upload(session, &ue8m0)?, session.upload_f32(&[1.0])?)
            }
            Format::Mxfp8 => {
                let ue8m0 = ue8m0_scales(seed ^ 0x5ca1e, g.scale_tile_bytes());
                (upload(session, &ue8m0)?, session.upload_f32(&[1.0])?)
            }
            Format::Nvfp4 => {
                let ue4m3 = ue4m3_scales(seed ^ 0x5ca1e, g.scale_tile_bytes());
                let globals = nvfp4_globals(seed ^ 0x610b, g.globals());
                (upload(session, &ue4m3)?, session.upload_f32(&globals)?)
            }
        };
        let mut bank = Bank {
            codes,
            scales,
            f32_scales,
            real: None,
        };
        bank.real = load_real(session, g, profile, &bank)?;
        Ok(bank)
    }
}

// ---------------------------------------------------------------------------
// Real-weight slices: layer 1's experts 0-7 of the rank (MiMo: rank 0's
// intermediate slice of each expert; Qwen3.8: rank `QWEN_RANK`'s experts).

fn load_real(
    session: &Session,
    g: Geometry,
    profile: Profile,
    bank: &Bank,
) -> LabResult<Option<String>> {
    if g.format == Format::Nvfp4 {
        return load_real_nvfp4(session, g, bank);
    }
    if g.format == Format::Mxfp8 {
        return Ok(None);
    }
    if g.format == Format::Fp8 {
        let Some(root) = std::env::var_os("CHOPINFER_QWEN38_CHECKPOINT").map(PathBuf::from) else {
            return Ok(None);
        };
        let first = QWEN_RANK as usize * g.experts;
        let (codes, scales) = qwen_real(&root, g, first)?;
        let packed = FragmentWeightLayout {
            rows: REAL_EXPERTS * g.n,
            k: g.k,
            format: FragmentWeightFormat::Fp8E4M3,
        }
        .pack(&codes)
        .map_err(err)?;
        poison::write(session, &bank.codes, 0, &packed.bytes)?;
        let scale_bytes = scales
            .iter()
            .flat_map(|s| s.to_ne_bytes())
            .collect::<Vec<_>>();
        poison::write(session, &bank.f32_scales, 0, &scale_bytes)?;
        Ok(Some(format!(
            "Qwen3.8 layer {REAL_LAYER} experts {first}-{} as local 0-{}",
            first + REAL_EXPERTS - 1,
            REAL_EXPERTS - 1
        )))
    } else {
        let Some(root) = std::env::var_os("CHOPINFER_MIMO_CHECKPOINT").map(PathBuf::from) else {
            return Ok(None);
        };
        let ranks = profile
            .ranks_of(model_mimo26_contract::sm120::config::MODEL)
            .ok_or("not a MiMo profile")?;
        let (codes, scales) = mimo_real(&root, g)?;
        let layout = Mxfp4MoeFragmentLayout {
            experts: REAL_EXPERTS,
            rows_per_expert: g.n,
            k: g.k,
        };
        poison::write(
            session,
            &bank.codes,
            0,
            &layout.pack_codes(&codes).map_err(err)?.bytes,
        )?;
        poison::write(
            session,
            &bank.scales,
            0,
            &layout.pack_scales(&scales).map_err(err)?.bytes,
        )?;
        Ok(Some(format!(
            "MiMo layer {REAL_LAYER} experts 0-{}, rank 0 of {ranks}",
            REAL_EXPERTS - 1
        )))
    }
}

/// Layer 1's experts of rank `QWEN_RANK` from NVIDIA's mixed NVFP4 export
/// (`CHOPINFER_QWEN38_NVFP4_CHECKPOINT`) as local experts 0-7, in the packed
/// layout the preparation writes: row-major codes (gate rows, then up rows),
/// UE4M3 scales in 16 x 8 tiles and each projection's `weight_scale_2`.
fn load_real_nvfp4(session: &Session, g: Geometry, bank: &Bank) -> LabResult<Option<String>> {
    let Some(root) = std::env::var_os("CHOPINFER_QWEN38_NVFP4_CHECKPOINT").map(PathBuf::from)
    else {
        return Ok(None);
    };
    let first = QWEN_RANK as usize * g.experts;
    let projections: &[&str] = if g.gate_up {
        &["gate_proj", "up_proj"]
    } else {
        &["down_proj"]
    };
    let rows = if g.gate_up { g.n / 2 } else { g.n };
    let blocks = g.scale_blocks();
    let mut codes = Vec::new();
    let mut scales = vec![0u8; REAL_EXPERTS * g.n / 16 * blocks.div_ceil(8) * 128];
    let mut globals = Vec::new();
    for (local, expert) in (first..first + REAL_EXPERTS).enumerate() {
        let prefix = format!("model.language_model.layers.{REAL_LAYER}.mlp.experts.{expert}");
        for (part, projection) in projections.iter().enumerate() {
            codes.extend(safetensors_bytes(
                &root,
                &format!("{prefix}.{projection}.weight"),
                "U8",
                &[rows, g.k / 2],
            )?);
            let local_scales = safetensors_bytes(
                &root,
                &format!("{prefix}.{projection}.weight_scale"),
                "F8_E4M3",
                &[rows, blocks],
            )?;
            for row in 0..rows {
                let physical = local * g.n + part * rows + row;
                for block in 0..blocks {
                    scales[packed_scale_offset(physical, block, blocks)] =
                        local_scales[row * blocks + block];
                }
            }
            let global = safetensors_bytes(
                &root,
                &format!("{prefix}.{projection}.weight_scale_2"),
                "F32",
                &[],
            )?;
            globals.extend_from_slice(&global);
        }
    }
    poison::write(session, &bank.codes, 0, &codes)?;
    poison::write(session, &bank.scales, 0, &scales)?;
    poison::write(session, &bank.f32_scales, 0, &globals)?;
    Ok(Some(format!(
        "NVIDIA NVFP4 Qwen3.8 layer {REAL_LAYER} experts {first}-{} as local 0-{}",
        first + REAL_EXPERTS - 1,
        REAL_EXPERTS - 1
    )))
}

/// Row-major E2M1 codes [8 experts * n, k / 2] and UE8M0 scales
/// [8 experts * n, k / 32] of rank 0's slice.
fn mimo_real(root: &Path, g: Geometry) -> LabResult<(Vec<u8>, Vec<u8>)> {
    let config = model_mimo26::config::MiMoConfig::load(root).map_err(err)?;
    let catalog = model_mimo26::catalog::Catalog::load(root, &config).map_err(err)?;
    // Row range and byte columns of rank 0's slice of a [rows, width] tensor.
    let read = |name: &str, rows: std::ops::Range<usize>, columns: std::ops::Range<usize>| {
        let tensor = catalog.tensor(name).map_err(err)?;
        if tensor.dtype.safetensors_name() != "U8" || tensor.shape.len() != 2 {
            return Err(format!("unexpected checkpoint dtype or rank for {name}"));
        }
        let width = tensor.shape[1] as usize;
        if rows.end > tensor.shape[0] as usize || columns.end > width {
            return Err(format!("rank slice outside {name}"));
        }
        let source = catalog.source(&tensor.file).map_err(err)?;
        let mut file = source.try_clone().map_err(err)?;
        let mut all = vec![0; rows.len() * width];
        file.seek(SeekFrom::Start(
            tensor.file_start + (rows.start * width) as u64,
        ))
        .map_err(err)?;
        file.read_exact(&mut all).map_err(err)?;
        source.verify().map_err(err)?;
        Ok(all
            .chunks_exact(width)
            .flat_map(|row| row[columns.clone()].to_vec())
            .collect::<Vec<u8>>())
    };
    let (mut codes, mut scales) = (Vec::new(), Vec::new());
    for expert in 0..REAL_EXPERTS {
        let prefix = format!("model.layers.{REAL_LAYER}.mlp.experts.{expert}");
        if g.gate_up {
            // Rank 0 holds intermediate rows [0, n / 2) of gate and of up.
            for projection in ["gate_proj", "up_proj"] {
                codes.extend(read(
                    &format!("{prefix}.{projection}.weight"),
                    0..g.n / 2,
                    0..g.k / 2,
                )?);
                scales.extend(read(
                    &format!("{prefix}.{projection}.weight_scale"),
                    0..g.n / 2,
                    0..g.k / 32,
                )?);
            }
        } else {
            // Rank 0 holds intermediate columns [0, k) of down.
            codes.extend(read(
                &format!("{prefix}.down_proj.weight"),
                0..g.n,
                0..g.k / 2,
            )?);
            scales.extend(read(
                &format!("{prefix}.down_proj.weight_scale"),
                0..g.n,
                0..g.k / 32,
            )?);
        }
    }
    Ok((codes, scales))
}

/// A safetensors file's header entries: dtype, shape and data offsets.
fn safetensors_entry(path: &Path, name: &str) -> LabResult<(String, Vec<usize>, u64, u64)> {
    let file = File::open(path).map_err(err)?;
    let mut length = [0u8; 8];
    file.read_exact_at(&mut length, 0).map_err(err)?;
    let length = u64::from_le_bytes(length);
    let mut header = vec![0u8; length as usize];
    file.read_exact_at(&mut header, 8).map_err(err)?;
    let header: Value = serde_json::from_slice(&header).map_err(err)?;
    let entry = &header[name];
    let dtype = entry["dtype"].as_str().ok_or("tensor dtype")?.to_string();
    let shape = entry["shape"]
        .as_array()
        .ok_or("tensor shape")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();
    let offsets = entry["data_offsets"].as_array().ok_or("tensor offsets")?;
    let begin = 8 + length + offsets[0].as_u64().ok_or("offset")?;
    let end = 8 + length + offsets[1].as_u64().ok_or("offset")?;
    Ok((dtype, shape, begin, end))
}

fn safetensors_bytes(root: &Path, name: &str, dtype: &str, shape: &[usize]) -> LabResult<Vec<u8>> {
    let index: Value = serde_json::from_slice(
        &std::fs::read(root.join("model.safetensors.index.json")).map_err(err)?,
    )
    .map_err(err)?;
    let file = index["weight_map"][name]
        .as_str()
        .ok_or_else(|| format!("{name} is not in the checkpoint index"))?;
    let path = root.join(file);
    let (actual, actual_shape, begin, end) = safetensors_entry(&path, name)?;
    if actual != dtype || actual_shape != shape {
        return Err(format!(
            "{name}: {actual} {actual_shape:?}, expected {dtype} {shape:?}"
        ));
    }
    let mut bytes = vec![0u8; (end - begin) as usize];
    File::open(&path)
        .map_err(err)?
        .read_exact_at(&mut bytes, begin)
        .map_err(err)?;
    Ok(bytes)
}

/// Row-major E4M3 codes [8 experts * n, k] and F32 block scales
/// [8 experts * n / 128, k / 128] (the BF16 `weight_scale_inv`, exact in F32).
fn qwen_real(root: &Path, g: Geometry, first: usize) -> LabResult<(Vec<u8>, Vec<f32>)> {
    let (mut codes, mut scales) = (Vec::new(), Vec::new());
    for expert in first..first + REAL_EXPERTS {
        let prefix = format!("model.language_model.layers.{REAL_LAYER}.mlp.experts.{expert}");
        let projections: &[&str] = if g.gate_up {
            &["gate_proj", "up_proj"]
        } else {
            &["down_proj"]
        };
        let rows = if g.gate_up { g.n / 2 } else { g.n };
        for projection in projections {
            codes.extend(safetensors_bytes(
                root,
                &format!("{prefix}.{projection}.weight"),
                "F8_E4M3",
                &[rows, g.k],
            )?);
            let bf16 = safetensors_bytes(
                root,
                &format!("{prefix}.{projection}.weight_scale_inv"),
                "BF16",
                &[rows / 128, g.k / 128],
            )?;
            scales.extend(
                bf16.chunks_exact(2)
                    .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
            );
        }
    }
    Ok((codes, scales))
}

// ---------------------------------------------------------------------------
// Routes.

#[derive(Clone, Copy, Debug)]
enum Routing {
    /// Power-law expert popularity with per-token noise.
    Skewed,
    /// Every token's first half of routes goes to local experts 0-7 (the
    /// real ones when loaded), so they hold full 128-row prefill tiles.
    Hot,
    /// Route k of every token goes to local expert k % 3, so an expert holds
    /// several rows of each token, as synthetic oracle routes do. Model
    /// routes never repeat an expert within a token; only the packed engine
    /// and the packed ring promise every slot of any metadata.
    Crowded,
    /// Exactly `experts` of every token's routes go to local experts, token
    /// t's to `first + t * experts + j` (modulo the local experts), and the
    /// rest to other ranks: Qwen3.8 decode at one token puts about 2.5 of
    /// its 10 routes on each of the four ranks.
    Local { experts: usize, first: usize },
}

impl Routing {
    fn name(self) -> &'static str {
        match self {
            Routing::Skewed => "skewed",
            Routing::Hot => "hot",
            Routing::Crowded => "crowded",
            Routing::Local { .. } => "local",
        }
    }

    /// A seed word: the variant's order (the skewed, hot and crowded cases'
    /// seeds predate the local routes).
    fn seed(self) -> u64 {
        match self {
            Routing::Skewed => 0,
            Routing::Hot => 1,
            Routing::Crowded => 2,
            Routing::Local { experts, first } => 3 ^ (experts as u64) << 8 ^ (first as u64) << 16,
        }
    }
}

/// Route metadata as the model routes write it: stable expert ranges over
/// the local experts padded to `block`, route order within each expert.
struct Routes {
    live: usize,
    offsets: Vec<i32>,
    sorted: Vec<i32>,
    padded: i32,
    active: Vec<i32>,
    /// Routed rows this rank writes.
    written: Vec<usize>,
}

fn routes(
    g: Geometry,
    qwen: bool,
    live: usize,
    block: usize,
    routing: Routing,
    seed: u64,
) -> Routes {
    let global = if qwen { QWEN_GLOBAL_EXPERTS } else { g.experts };
    let first_local = if qwen {
        QWEN_RANK as usize * g.experts
    } else {
        0
    };
    let popularity = (0..global)
        .map(|e| {
            let u = 2.0 * uniform(seed ^ 0x9091 ^ (e as u64) << 8) - 1.0;
            1.5 * u.signum() * u.abs().powf(1.5)
        })
        .collect::<Vec<_>>();
    let mut expert_rows = vec![Vec::new(); g.experts];
    for token in 0..live {
        let scores = (0..global)
            .map(|e| {
                let noise = uniform(seed ^ (token as u64) << 20 ^ e as u64);
                let hot = matches!(routing, Routing::Hot)
                    && (e + global - first_local) % global < REAL_EXPERTS;
                popularity[e] + noise + if hot { 100.0 } else { 0.0 }
            })
            .collect::<Vec<_>>();
        let ranked = topk(&scores, global, Ties::LowerIndex, NanKeys::Refuse, None)
            .expect("finite route scores");
        let mut chosen = ranked[..g.top_k].to_vec();
        if matches!(routing, Routing::Hot) {
            // Keep half the routes hot; the rest follow popularity.
            let rest = ranked
                .iter()
                .copied()
                .filter(|&e| (e + global - first_local) % global >= REAL_EXPERTS)
                .take(g.top_k - g.top_k / 2);
            chosen.truncate(g.top_k / 2);
            chosen.extend(rest);
        }
        if matches!(routing, Routing::Crowded) {
            chosen = (0..g.top_k).map(|k| first_local + k % 3).collect();
        }
        if let Routing::Local { experts, first } = routing {
            assert!(qwen && experts <= g.top_k, "local routes need other ranks");
            let remote = first_local + g.experts;
            chosen = (0..g.top_k)
                .map(|k| {
                    if k < experts {
                        first_local + (first + token * experts + k) % g.experts
                    } else {
                        (remote + token * g.top_k + k) % global
                    }
                })
                .collect();
        }
        for (k, e) in chosen.into_iter().enumerate() {
            if e >= first_local && e < first_local + g.experts {
                expert_rows[e - first_local].push((token * g.top_k + k) as i32);
            }
        }
    }
    assemble(g, live, block, expert_rows)
}

/// Route metadata of given routes: `chosen[token * top_k + k]` is route k's
/// expert, every expert local to the rank.
fn given_routes(g: Geometry, live: usize, block: usize, chosen: &[u32]) -> Routes {
    assert_eq!(
        chosen.len(),
        live * g.top_k,
        "given routes hold top_k per live token"
    );
    let mut expert_rows = vec![Vec::new(); g.experts];
    for (row, &e) in chosen.iter().enumerate() {
        expert_rows[e as usize].push(row as i32);
    }
    assemble(g, live, block, expert_rows)
}

/// Offsets, padded sorted rows and the active list of per-expert routed
/// rows, each expert's range padded to `block` rows.
fn assemble(g: Geometry, live: usize, block: usize, mut expert_rows: Vec<Vec<i32>>) -> Routes {
    let mut offsets = Vec::with_capacity(g.experts + 1);
    let mut sorted = Vec::new();
    let mut active = vec![0];
    let mut written = Vec::new();
    for (e, rows) in expert_rows.iter_mut().enumerate() {
        offsets.push(sorted.len() as i32);
        rows.sort_unstable();
        written.extend(rows.iter().map(|&r| r as usize));
        sorted.extend(rows.iter().copied());
        let padded = rows.len().div_ceil(block) * block;
        sorted.resize(offsets[e] as usize + padded, -1);
        if !rows.is_empty() {
            active.push(e as i32);
        }
    }
    offsets.push(sorted.len() as i32);
    active[0] = active.len() as i32 - 1;
    active.resize(g.experts + 1, -1);
    written.sort_unstable();
    Routes {
        live,
        padded: sorted.len() as i32,
        offsets,
        sorted,
        active,
        written,
    }
}

// ---------------------------------------------------------------------------
// One instance on shared buffers.

struct Entry {
    function: Function,
}

fn entry(session: &Session, cubin: &str, symbol: &str, shared: u32) -> LabResult<Entry> {
    let function = session.module(cubin)?.function(symbol).map_err(err)?;
    if shared > 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(shared)
            .map_err(err)?;
    }
    Ok(Entry { function })
}

fn counts(instance: &Instance, live: usize, top_k: usize) -> RuntimeCounts {
    RuntimeCounts {
        m: live as u32,
        batch: 0,
        context: 0,
        routed_rows: (live * top_k) as u32,
        split_k: instance.baseline.knobs.split_k_capacity,
    }
}

/// A sweep candidate: a template instance at other knobs and its entry,
/// loaded from a cubin the sweep compiled.
struct Candidate {
    instance: Instance,
    entry: Entry,
}

/// Which entry a case launches, each at its own launch plan.
#[derive(Clone, Copy)]
enum Arm<'a> {
    /// The per-model engine, at the engine's plan: the bitwise reference.
    Engine,
    /// The template at the instance's table row (the served entry once the
    /// model is swapped).
    Template,
    Candidate(&'a Candidate),
}

impl Arm<'_> {
    fn label(&self, case: &Case) -> String {
        match self {
            Arm::Engine => format!("{} engine", case.instance.engine_symbol),
            Arm::Template => format!("{} template", case.instance.template_symbol),
            Arm::Candidate(c) => format!("{} candidate", c.instance.template_symbol),
        }
    }
}

/// Buffers of one instance at its capacity.
struct Case<'a> {
    instance: &'a Instance,
    g: Geometry,
    capacity: usize,
    engine: Entry,
    template: Entry,
    activation: DeviceBuffer,
    output: DeviceBuffer,
    workspace: DeviceBuffer,
    topk_weights: DeviceBuffer,
    offsets: DeviceBuffer,
    sorted: DeviceBuffer,
    padded: DeviceBuffer,
    live: DeviceBuffer,
    active: DeviceBuffer,
    dummy: DeviceBuffer,
    sorted_capacity: usize,
}

impl<'a> Case<'a> {
    fn new(session: &Session, instance: &'a Instance) -> LabResult<Self> {
        let g = Geometry::of(instance);
        let v = &instance.baseline;
        let capacity = v.tokens_capacity as usize;
        let at_capacity = counts(instance, capacity, g.top_k);
        let engine_plan = v.engine_plan(at_capacity).map_err(err)?;
        let plan = instance.plan(at_capacity).map_err(err)?;
        let engine = entry(
            session,
            &instance.engine_cubin,
            &instance.engine_symbol,
            engine_plan.dynamic_shared_bytes,
        )?;
        let template = entry(
            session,
            &instance.template_cubin,
            &instance.template_symbol,
            plan.dynamic_shared_bytes,
        )?;
        let rows = capacity * g.top_k;
        let input_rows = if g.gate_up { capacity } else { rows };
        let sorted_capacity = match v.query.grouped.expect("grouped").capacity {
            RoutedCapacity::TotalRows(rows) => rows as usize,
            RoutedCapacity::PerExpert(_) => unreachable!(),
        };
        let workspace_bytes = plan.workspace_bytes.max(engine_plan.workspace_bytes) as usize;
        let workspace = session.allocate(workspace_bytes.max(16))?;
        workspace
            .zero_at_async(0, workspace.byte_len(), &session.stream)
            .map_err(err)?;
        Ok(Self {
            instance,
            g,
            capacity,
            engine,
            template,
            activation: session.allocate(input_rows * g.k * 2)?,
            output: session.allocate(rows * g.out_n() * 2)?,
            workspace,
            topk_weights: session.allocate(rows * 4)?,
            offsets: session.allocate((g.experts + 1) * 4)?,
            sorted: session.allocate(sorted_capacity * 4)?,
            padded: session.allocate(16)?,
            live: session.allocate(16)?,
            active: session.allocate((g.experts + 1) * 4)?,
            dummy: upload(session, &[0; 64])?,
            sorted_capacity,
        })
    }

    fn qwen(&self) -> bool {
        self.instance.baseline.fp8_fragment_weights()
    }

    /// Writes one live count's activations, route weights and metadata.
    fn prepare(
        &self,
        session: &Session,
        live: usize,
        routing: Routing,
        seed: u64,
    ) -> LabResult<Routes> {
        let r = routes(self.g, self.qwen(), live, self.block(), routing, seed);
        self.write_routes(session, r, seed)
    }

    /// The padded row block of the entry's expert ranges.
    fn block(&self) -> usize {
        usize::from(self.instance.baseline.knobs.tile_m)
    }

    /// Writes given routes (`given_routes`) with activations and route
    /// weights from `seed`.
    fn prepare_given(
        &self,
        session: &Session,
        live: usize,
        chosen: &[u32],
        seed: u64,
    ) -> LabResult<Routes> {
        let r = given_routes(self.g, live, self.block(), chosen);
        self.write_routes(session, r, seed)
    }

    fn write_routes(&self, session: &Session, r: Routes, seed: u64) -> LabResult<Routes> {
        let live = r.live;
        if r.sorted.len() > self.sorted_capacity {
            return Err(format!(
                "{} padded rows exceed the {} prepared",
                r.sorted.len(),
                self.sorted_capacity
            ));
        }
        let mut sorted = r.sorted.clone();
        sorted.resize(self.sorted_capacity, -1);
        poison::write(
            session,
            &self.activation,
            0,
            &wide_bf16(seed ^ 0xac71, self.activation.byte_len() / 2),
        )?;
        let weights = (0..self.topk_weights.byte_len() / 4).map(|i| {
            ((-8.0 + 8.0 * uniform(seed ^ 0x7e1 ^ (i as u64) << 4)).exp2() as f32).to_bits()
        });
        poison::write(session, &self.topk_weights, 0, &u32_bytes(weights))?;
        poison::write(session, &self.offsets, 0, &i32_bytes(&r.offsets))?;
        poison::write(session, &self.sorted, 0, &i32_bytes(&sorted))?;
        poison::write(session, &self.padded, 0, &i32_bytes(&[r.padded]))?;
        poison::write(session, &self.live, 0, &i32_bytes(&[live as i32]))?;
        poison::write(session, &self.active, 0, &i32_bytes(&r.active))?;
        Ok(r)
    }

    fn launch(
        &self,
        session: &Session,
        bank: &Bank,
        arm: Arm,
        live: usize,
        pdl: bool,
    ) -> LabResult<()> {
        let v = &self.instance.baseline;
        let buffers = [
            &self.activation,
            &self.dummy,
            &self.dummy,
            &bank.scales,
            &bank.f32_scales,
            &self.output,
            &self.topk_weights,
            &self.offsets,
            &self.sorted,
            &self.padded,
            &self.live,
            &self.active,
            &self.dummy,
            &self.dummy,
            &self.workspace,
            &bank.codes,
        ];
        let bindings = Bindings {
            buffers: buffers
                .into_iter()
                .enumerate()
                .map(|(i, b)| (i as u64 + 1, b))
                .collect(),
        };
        let args = v
            .arguments(HANDLES, v.weight_tensor_map(WEIGHT_HANDLE).map_err(err)?)
            .map_err(err)?;
        let live_counts = counts(self.instance, live, self.g.top_k);
        let (function, plan) = match arm {
            Arm::Engine => (&self.engine.function, v.engine_plan(live_counts)),
            Arm::Template => (&self.template.function, self.instance.plan(live_counts)),
            Arm::Candidate(c) => (&c.entry.function, c.instance.plan(live_counts)),
        };
        let mut plan = plan.map_err(err)?;
        if plan.workspace_bytes > self.workspace.byte_len() as u64 {
            return Err(format!("{} needs a larger workspace", arm.label(self)));
        }
        plan.pdl = pdl;
        launch::launch(
            &session.context,
            function,
            &session.stream,
            plan,
            &args,
            &bindings,
        )
    }

    /// Poisons the output and the workspace, launches one arm and returns
    /// its output after checking the write map and the untouched workspace.
    fn run(&self, session: &Session, bank: &Bank, arm: Arm, r: &Routes) -> LabResult<Vec<u8>> {
        self.checked(session, &arm.label(self), r, || {
            self.launch(session, bank, arm, r.live, true)
        })
    }

    /// Poisons the output and the workspace, runs `launch` (one launch or a
    /// graph replay of one) and returns the output after the same checks.
    fn checked(
        &self,
        session: &Session,
        label: &str,
        r: &Routes,
        launch: impl FnOnce() -> LabResult<()>,
    ) -> LabResult<Vec<u8>> {
        let output = self.poisoned(session, label, launch)?;
        let map = WriteMap::of(DType::Bf16).rows(self.g.out_n(), r.written.iter().copied())?;
        poison::check(
            &format!("{label} output"),
            &output,
            &map,
            Fill::Pattern(&[POISON]),
        )?;
        Ok(output)
    }

    /// Poisons the output and the workspace, runs `launch` and returns the
    /// whole output, poison included, after checking the untouched
    /// workspace. Without a write-map check, for routes whose written rows
    /// the entry does not promise.
    fn poisoned(
        &self,
        session: &Session,
        label: &str,
        launch: impl FnOnce() -> LabResult<()>,
    ) -> LabResult<Vec<u8>> {
        poison::poison(session, &[&self.output, &self.workspace])?;
        launch()?;
        session.stream.synchronize().map_err(err)?;
        let workspace = poison::read(session, &self.workspace)?;
        if workspace.iter().any(|&b| b != POISON) {
            return Err(format!("{label} wrote its workspace"));
        }
        poison::read(session, &self.output)
    }
}

fn first_difference(a: &[u8], b: &[u8], element: usize) -> Option<usize> {
    a.chunks(element)
        .zip(b.chunks(element))
        .position(|(x, y)| x != y)
}

fn live_counts(instance: &Instance) -> Vec<usize> {
    poison::live_counts(instance.baseline.tokens_capacity)
        .into_iter()
        .map(|live| live as usize)
        .collect()
}

/// Instances grouped by profile and projection, which share a weight bank.
fn banks_of(instances: &[Instance]) -> Vec<(Profile, Projection, Vec<&Instance>)> {
    let mut groups: Vec<(Profile, Projection, Vec<&Instance>)> = Vec::new();
    for instance in instances {
        let key = (instance.baseline.profile, instance.baseline.projection);
        match groups.iter_mut().find(|g| (g.0, g.1) == key) {
            Some(group) => group.2.push(instance),
            None => groups.push((key.0, key.1, vec![instance])),
        }
    }
    groups
}

fn schedule_name(schedule: Schedule) -> &'static str {
    match schedule {
        Schedule::Ring => "ring",
        Schedule::Prefill => "prefill",
    }
}

#[test]
fn grouped_moe_template_matches_served_entries() {
    let instances = instances();
    gpu_test(12 << 30, |session| {
        let mut results = Vec::new();
        let mut failures = Vec::new();
        let mut real_skipped = BTreeMap::new();
        for (index, (profile, projection, group)) in banks_of(&instances).into_iter().enumerate() {
            let g = Geometry::of(group[0]);
            let bank = Bank::new(session, g, profile, 0xba7c ^ (index as u64) << 32)?;
            if bank.real.is_none() {
                real_skipped.insert(
                    format!("{}_{projection:?}", profile.as_str()),
                    if g.format == Format::Fp8 {
                        "CHOPINFER_QWEN38_CHECKPOINT unset"
                    } else {
                        "CHOPINFER_MIMO_CHECKPOINT unset"
                    },
                );
            }
            for instance in group {
                let case = Case::new(session, instance)?;
                for live in live_counts(instance) {
                    for routing in [Routing::Skewed, Routing::Hot] {
                        let seed = mix(index as u64 ^ (live as u64) << 16 ^ routing.seed());
                        let r = case.prepare(session, live, routing, seed)?;
                        let engine = case.run(session, &bank, Arm::Engine, &r)?;
                        let template = case.run(session, &bank, Arm::Template, &r)?;
                        let output_equal = engine == template;
                        let result = json!({
                            "symbol": instance.baseline.symbol,
                            "engine": instance.engine_symbol,
                            "template": instance.template_symbol,
                            "schedule": schedule_name(instance.schedule),
                            "capacity": case.capacity,
                            "live": live,
                            "routing": routing.name(),
                            "real_weights": bank.real,
                            "written_rows": r.written.len(),
                            "active_experts": r.active[0],
                            "padded_rows": r.padded,
                            "output_bitwise": output_equal,
                        });
                        if !output_equal {
                            let at = first_difference(&engine, &template, 2);
                            failures.push(format!(
                                "{} live {live} {}: output differs (first BF16 {at:?})",
                                instance.baseline.symbol,
                                routing.name(),
                            ));
                        }
                        eprintln!("{result}");
                        results.push(result);
                    }
                }
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("instances".into(), json!(instances.len()));
        receipt.insert("executed_cases".into(), json!(results.len()));
        receipt.insert("cases".into(), json!(results));
        receipt.insert("failures".into(), json!(failures));
        receipt.insert("real_weights_skipped".into(), json!(real_skipped));
        receipt.insert("poisoned".into(), json!(["output", "workspace"]));
        let path = session.receipt("moe_template_bitwise", receipt)?;
        eprintln!("receipt {}", path.display());
        if results.is_empty() {
            return Err("no bitwise case ran".into());
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{} template cases differ:\n{}",
                failures.len(),
                failures.join("\n")
            ))
        }
    })
    .expect("grouped-MoE template bitwise equality");
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

/// Alternates the engine entry (A) and the template (B), A B A B per
/// repetition. Each timing is the median over `ROUNDS` of a batch of
/// back-to-back launches (32, or 4 at the prefill capacity) after one L2
/// flush, divided by the batch: single-launch events resolve only about
/// 2 us on this GPU, so times are averages of warm repeated launches. An
/// unrecorded repetition warms the clocks.
///
/// Each repetition gives two adjacent pairs (A1, B1) and (A2, B2). A shape
/// regresses when the median of its pair ratios B/A exceeds
/// `1 + REGRESSION`. The noise band is the median over repetitions of the
/// A1-to-A2 difference; above `NOISE_LIMIT` the shape is inconclusive and
/// the test fails asking for a rerun, so no single outlier widens the
/// threshold.
#[test]
#[ignore = "timing; run on an exclusive GPU with --ignored"]
fn grouped_moe_template_timing() {
    const REPETITIONS: usize = 7;
    const ROUNDS: usize = 15;
    const REGRESSION: f32 = 0.01;
    const NOISE_LIMIT: f32 = 0.02;
    let instances = instances();
    gpu_test(12 << 30, |session| {
        let mut rows = Vec::new();
        let (mut slower, mut inconclusive) = (Vec::new(), Vec::new());
        for (index, (profile, _, group)) in banks_of(&instances).into_iter().enumerate() {
            let g = Geometry::of(group[0]);
            let bank = Bank::new(session, g, profile, 0x7133 ^ (index as u64) << 32)?;
            for instance in group {
                let case = Case::new(session, instance)?;
                let live = case.capacity;
                case.prepare(session, live, Routing::Skewed, mix(0x7133 ^ index as u64))?;
                let batch = if instance.schedule == Schedule::Prefill { 4 } else { 32 };
                let arm = |template: bool| -> LabResult<f32> {
                    let side = if template { Arm::Template } else { Arm::Engine };
                    let timing = time(session, 3, ROUNDS, |_| {
                        for _ in 0..batch {
                            case.launch(session, &bank, side, live, false)?;
                        }
                        Ok(())
                    })?;
                    Ok(timing.median_ms * 1000.0 / batch as f32)
                };
                // One unrecorded repetition first: the first seconds of
                // sustained prefill load run at higher clocks.
                for template in [false, true, false, true] {
                    arm(template)?;
                }
                let (mut a, mut b, mut noise, mut pairs) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for _ in 0..REPETITIONS {
                    let a1 = arm(false)?;
                    let b1 = arm(true)?;
                    let a2 = arm(false)?;
                    let b2 = arm(true)?;
                    noise.push((a1 - a2).abs() / a1.min(a2));
                    pairs.extend([b1 / a1, b2 / a2]);
                    a.extend([a1, a2]);
                    b.extend([b1, b2]);
                }
                let (a_min, a_max) = (a.iter().copied().fold(f32::MAX, f32::min), a.iter().copied().fold(0.0, f32::max));
                let (b_min, b_max) = (b.iter().copied().fold(f32::MAX, f32::min), b.iter().copied().fold(0.0, f32::max));
                let (a_median, b_median) = (median(&mut a), median(&mut b));
                let band = noise.iter().copied().fold(0.0_f32, f32::max);
                let band_median = median(&mut noise);
                let (pair_min, pair_max) = (
                    pairs.iter().copied().fold(f32::MAX, f32::min),
                    pairs.iter().copied().fold(0.0, f32::max),
                );
                let ratio = median(&mut pairs);
                let row = json!({
                    "symbol": instance.baseline.symbol,
                    "engine": instance.engine_symbol,
                    "template": instance.template_symbol,
                    "schedule": schedule_name(instance.schedule),
                    "live": live,
                    "launches_per_timing": batch,
                    "engine_us": {"median": a_median, "min": a_min, "max": a_max},
                    "template_us": {"median": b_median, "min": b_min, "max": b_max},
                    "template_over_engine": ratio,
                    "pair_ratio": {"median": ratio, "min": pair_min, "max": pair_max},
                    "a_to_a_noise": {"median": band_median, "max": band},
                });
                eprintln!(
                    "{:<60} A {a_median:9.2} us [{a_min:.2}, {a_max:.2}]  B {b_median:9.2} us [{b_min:.2}, {b_max:.2}]  B/A {ratio:.4}  A-A noise median {:.2}% max {:.2}%",
                    instance.baseline.symbol,
                    100.0 * band_median,
                    100.0 * band,
                );
                if band_median > NOISE_LIMIT {
                    inconclusive.push(format!(
                        "{} A-A noise median {:.2}%",
                        instance.baseline.symbol,
                        100.0 * band_median
                    ));
                } else if ratio > 1.0 + REGRESSION {
                    slower.push(format!("{} median pair B/A {ratio:.4}", instance.baseline.symbol));
                }
                rows.push(row);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("repetitions".into(), json!(REPETITIONS));
        receipt.insert("rounds_per_timing".into(), json!(ROUNDS));
        receipt.insert("order".into(), json!("one unrecorded A B A B, then A B A B per repetition; L2 flushed before each timed batch; microseconds per launch"));
        receipt.insert("shapes".into(), json!(rows));
        receipt.insert("executed_shapes".into(), json!(rows.len()));
        receipt.insert(
            "gate".into(),
            json!({"median_pair_ratio_max": 1.0 + REGRESSION, "a_to_a_noise_median_max": NOISE_LIMIT}),
        );
        receipt.insert("slower".into(), json!(slower));
        receipt.insert("inconclusive_rerun".into(), json!(inconclusive));
        let path = session.receipt("moe_template_timing", receipt)?;
        eprintln!("receipt {}", path.display());
        if rows.is_empty() {
            Err("no timing shape ran".into())
        } else if !slower.is_empty() {
            Err(format!("template slower:\n{}", slower.join("\n")))
        } else if !inconclusive.is_empty() {
            Err(format!("inconclusive, rerun:\n{}", inconclusive.join("\n")))
        } else {
            Ok(())
        }
    })
    .expect("grouped-MoE template timing");
}

// ---------------------------------------------------------------------------
// Packed NVFP4 and MXFP8 decode: the packed engine (`moe.cuh`), which serves
// these formats outside the template, against the template's packed ring
// (`grouped_packed.cuh`) at its table row. Both are grouped operations with
// their own launch plans on the same bank, route metadata and entry ABI.

/// The selected packed instances: NVFP4 and MXFP8 at Qwen3.8 TP4's shape.
/// `CHOPINFER_MOE_TEMPLATE` may list several comma-separated substrings;
/// an instance is selected when its ring symbol contains any of them.
fn packed_instances() -> Vec<PackedInstance> {
    let filters = env_list("CHOPINFER_MOE_TEMPLATE");
    let selected = model_qwen38_contract::sm120::moe::packed_lab_instances()
        .into_iter()
        .filter(|i| {
            filters
                .as_ref()
                .is_none_or(|list| list.iter().any(|f| i.template.symbol.contains(f.as_str())))
        })
        .collect::<Vec<_>>();
    assert!(
        !selected.is_empty(),
        "CHOPINFER_MOE_TEMPLATE={:?} selects no packed instance",
        std::env::var("CHOPINFER_MOE_TEMPLATE").unwrap_or_default()
    );
    selected
}

/// Packed instances grouped by format and projection, which share a bank.
fn packed_banks(instances: &[PackedInstance]) -> Vec<Vec<&PackedInstance>> {
    let mut groups: Vec<Vec<&PackedInstance>> = Vec::new();
    for instance in instances {
        let key = (instance.template.query.weight, instance.template.projection);
        match groups
            .iter_mut()
            .find(|g| (g[0].template.query.weight, g[0].template.projection) == key)
        {
            Some(group) => group.push(instance),
            None => groups.push(vec![instance]),
        }
    }
    groups
}

fn packed_geometry(instance: &PackedInstance) -> Geometry {
    Geometry::of_query(&instance.template.query, instance.template.projection)
}

/// A packed case's activations.
#[derive(Clone, Copy, Debug)]
enum Activations {
    /// Random BF16 over `.1` binades from 2^`.0`, both signs, 1% zeros.
    Binades(i32, u32),
    /// `boundary_activations`, for the every-code MXFP8 bank.
    Boundary,
}

/// One live count's route metadata and route weights on the device.
struct RouteSet {
    topk_weights: DeviceBuffer,
    offsets: DeviceBuffer,
    sorted: DeviceBuffer,
    padded: DeviceBuffer,
    live: DeviceBuffer,
    active: DeviceBuffer,
}

impl RouteSet {
    fn new(session: &Session, g: Geometry, rows: usize, sorted_capacity: usize) -> LabResult<Self> {
        Ok(Self {
            topk_weights: session.allocate(rows * 4)?,
            offsets: session.allocate((g.experts + 1) * 4)?,
            sorted: session.allocate(sorted_capacity * 4)?,
            padded: session.allocate(16)?,
            live: session.allocate(16)?,
            active: session.allocate((g.experts + 1) * 4)?,
        })
    }
}

/// Buffers of one packed instance at its capacity. Route set 0 serves the
/// bitwise cases; the timing rotates through more.
struct PackedCase<'a> {
    instance: &'a PackedInstance,
    /// How the activations are drawn.
    activations: Activations,
    g: Geometry,
    capacity: usize,
    engine: Entry,
    ring: Entry,
    activation: DeviceBuffer,
    output: DeviceBuffer,
    route_sets: Vec<RouteSet>,
    dummy: DeviceBuffer,
    sorted_capacity: usize,
}

impl<'a> PackedCase<'a> {
    fn new(session: &Session, instance: &'a PackedInstance, sets: usize) -> LabResult<Self> {
        let g = packed_geometry(instance);
        let capacity = instance.template.tokens_capacity as usize;
        let ring = entry(
            session,
            &instance.template_cubin,
            &instance.template.symbol,
            Self::plan_of(&instance.template, g, capacity)?.dynamic_shared_bytes,
        )?;
        Self::with_ring(session, instance, ring, sets)
    }

    /// A case whose ring arm is `ring`, an entry of `instance.template`
    /// loaded from anywhere (a sweep candidate's cubin).
    fn with_ring(
        session: &Session,
        instance: &'a PackedInstance,
        ring: Entry,
        sets: usize,
    ) -> LabResult<Self> {
        let g = packed_geometry(instance);
        let capacity = instance.template.tokens_capacity as usize;
        let engine = entry(
            session,
            &instance.engine_cubin,
            &instance.engine.symbol,
            Self::plan_of(&instance.engine, g, capacity)?.dynamic_shared_bytes,
        )?;
        let rows = capacity * g.top_k;
        let input_rows = if g.gate_up { capacity } else { rows };
        let sorted_capacity = match instance.template.query.grouped.expect("grouped").capacity {
            RoutedCapacity::TotalRows(rows) => rows as usize,
            RoutedCapacity::PerExpert(_) => unreachable!(),
        };
        Ok(Self {
            instance,
            activations: Activations::Binades(-24, 31),
            g,
            capacity,
            engine,
            ring,
            activation: session.allocate(input_rows * g.k * 2)?,
            output: session.allocate(rows * g.out_n() * 2)?,
            route_sets: (0..sets)
                .map(|_| RouteSet::new(session, g, rows, sorted_capacity))
                .collect::<LabResult<_>>()?,
            dummy: upload(session, &[0; 64])?,
            sorted_capacity,
        })
    }

    fn plan_of(operation: &GroupedOperation, g: Geometry, live: usize) -> LabResult<LaunchPlan> {
        operation
            .plan(RuntimeCounts {
                m: live as u32,
                batch: 0,
                context: 0,
                routed_rows: (live * g.top_k) as u32,
                split_k: operation.knobs.split_k_capacity,
            })
            .map_err(err)
    }

    fn operation(&self, ring: bool) -> &GroupedOperation {
        if ring {
            &self.instance.template
        } else {
            &self.instance.engine
        }
    }

    fn label(&self, ring: bool) -> String {
        format!(
            "{} {}",
            self.operation(ring).symbol,
            if ring { "ring" } else { "engine" }
        )
    }

    /// Writes the activations and route set 0 for one live count.
    fn prepare(
        &self,
        session: &Session,
        live: usize,
        routing: Routing,
        seed: u64,
    ) -> LabResult<Routes> {
        poison::write(
            session,
            &self.activation,
            0,
            &match self.activations {
                Activations::Binades(lowest, binades) => bf16_binades(
                    seed ^ 0xac71,
                    self.activation.byte_len() / 2,
                    lowest,
                    binades,
                ),
                Activations::Boundary => {
                    boundary_activations(self.g, seed ^ 0xac71, self.activation.byte_len() / 2)
                }
            },
        )?;
        self.prepare_routes(session, 0, live, routing, seed)
    }

    fn prepare_routes(
        &self,
        session: &Session,
        set: usize,
        live: usize,
        routing: Routing,
        seed: u64,
    ) -> LabResult<Routes> {
        let block = usize::from(self.instance.engine.knobs.tile_m);
        let r = routes(self.g, true, live, block, routing, seed);
        if r.sorted.len() > self.sorted_capacity {
            return Err(format!(
                "{} padded rows exceed the {} prepared",
                r.sorted.len(),
                self.sorted_capacity
            ));
        }
        let mut sorted = r.sorted.clone();
        sorted.resize(self.sorted_capacity, -1);
        let routes = &self.route_sets[set];
        let weights = (0..routes.topk_weights.byte_len() / 4).map(|i| {
            ((-8.0 + 8.0 * uniform(seed ^ 0x7e1 ^ (i as u64) << 4)).exp2() as f32).to_bits()
        });
        poison::write(session, &routes.topk_weights, 0, &u32_bytes(weights))?;
        poison::write(session, &routes.offsets, 0, &i32_bytes(&r.offsets))?;
        poison::write(session, &routes.sorted, 0, &i32_bytes(&sorted))?;
        poison::write(session, &routes.padded, 0, &i32_bytes(&[r.padded]))?;
        poison::write(session, &routes.live, 0, &i32_bytes(&[live as i32]))?;
        poison::write(session, &routes.active, 0, &i32_bytes(&r.active))?;
        Ok(r)
    }

    fn arguments(&self, ring: bool) -> LabResult<ArgumentList> {
        let operation = self.operation(ring);
        operation
            .arguments(
                HANDLES,
                operation.weight_tensor_map(WEIGHT_HANDLE).map_err(err)?,
            )
            .map_err(err)
    }

    fn launch(
        &self,
        session: &Session,
        bank: &Bank,
        ring: bool,
        live: usize,
        pdl: bool,
        set: usize,
    ) -> LabResult<()> {
        let routes = &self.route_sets[set];
        let buffers = [
            &self.activation,
            &self.dummy,
            &self.dummy,
            &bank.scales,
            &bank.f32_scales,
            &self.output,
            &routes.topk_weights,
            &routes.offsets,
            &routes.sorted,
            &routes.padded,
            &routes.live,
            &routes.active,
            &self.dummy,
            &self.dummy,
            &self.dummy,
            &bank.codes,
        ];
        let bindings = Bindings {
            buffers: buffers
                .into_iter()
                .enumerate()
                .map(|(i, b)| (i as u64 + 1, b))
                .collect(),
        };
        let mut plan = Self::plan_of(self.operation(ring), self.g, live)?;
        if plan.workspace_bytes != 0 {
            return Err(format!("{} needs a workspace", self.label(ring)));
        }
        plan.pdl = pdl;
        let function = if ring {
            &self.ring.function
        } else {
            &self.engine.function
        };
        launch::launch(
            &session.context,
            function,
            &session.stream,
            plan,
            &self.arguments(ring)?,
            &bindings,
        )
    }

    /// Poisons the output, launches one arm and returns its output after
    /// checking that it wrote exactly the live routed rows.
    fn run(&self, session: &Session, bank: &Bank, ring: bool, r: &Routes) -> LabResult<Vec<u8>> {
        poison::poison(session, &[&self.output])?;
        self.launch(session, bank, ring, r.live, true, 0)?;
        session.stream.synchronize().map_err(err)?;
        let output = poison::read(session, &self.output)?;
        let map = WriteMap::of(DType::Bf16).rows(self.g.out_n(), r.written.iter().copied())?;
        poison::check(
            &format!("{} output", self.label(ring)),
            &output,
            &map,
            Fill::Pattern(&[POISON]),
        )?;
        Ok(output)
    }
}

/// Of the written rows' BF16 outputs: how many there are, and how many are
/// finite and nonzero.
fn informative_elements(output: &[u8], r: &Routes, g: Geometry) -> (usize, usize) {
    let width = g.out_n();
    let mut informative = 0;
    for &row in &r.written {
        for e in 0..width {
            let at = 2 * (row * width + e);
            let v = bf16_to_f32(u16::from_le_bytes([output[at], output[at + 1]]));
            informative += usize::from(v.is_finite() && v != 0.0);
        }
    }
    (r.written.len() * width, informative)
}

fn packed_skip_reason(g: Geometry) -> &'static str {
    if g.format == Format::Nvfp4 {
        "CHOPINFER_QWEN38_NVFP4_CHECKPOINT unset"
    } else {
        "no MXFP8 expert checkpoint is inspected"
    }
}

/// The template's packed ring at its table row against the packed engine,
/// bit for bit, at every live count, with skewed, hot and crowded routes,
/// poisoned outputs and, with `CHOPINFER_QWEN38_NVFP4_CHECKPOINT`, layer 1's
/// real NVIDIA experts; the banks are listed in the module comment.
#[test]
fn grouped_moe_packed_ring_matches_the_packed_engine() {
    let instances = packed_instances();
    gpu_test(12 << 30, |session| {
        let mut results = Vec::new();
        let mut failures = Vec::new();
        let mut real_skipped = BTreeMap::new();
        for (index, group) in packed_banks(&instances).into_iter().enumerate() {
            let g = packed_geometry(group[0]);
            let seed = 0xbac4 ^ (index as u64) << 32;
            let bank = Bank::new(session, g, group[0].profile, seed)?;
            if bank.real.is_none() {
                real_skipped.insert(
                    format!("{:?}_{:?}", g.format, group[0].template.projection),
                    packed_skip_reason(g),
                );
            }
            // Every bank with its activations: the generated (and real)
            // bank with activations over 31 binades, and for MXFP8 a bank
            // over every UE8M0 code with activations that cancel each scale
            // block's base exponent, so its outputs stay finite and nonzero.
            let mut banks = vec![("generated", bank, Activations::Binades(-24, 31))];
            if g.format == Format::Mxfp8 {
                let (codes, scales) = mxfp8_boundary_bank(g, seed ^ 0xb0d);
                banks.push((
                    "ue8m0_every_code",
                    Bank {
                        codes: upload(session, &codes)?,
                        scales: upload(session, &scales)?,
                        f32_scales: session.upload_f32(&[1.0])?,
                        real: None,
                    },
                    Activations::Boundary,
                ));
            }
            for (bank_name, bank, activations) in &banks {
                for instance in &group {
                    let mut case = PackedCase::new(session, instance, 1)?;
                    case.activations = *activations;
                    // Every live count of the bucket, 0 through capacity.
                    for live in 0..=case.capacity {
                        for routing in [Routing::Skewed, Routing::Hot, Routing::Crowded] {
                            let seed = mix(index as u64 ^ (live as u64) << 16 ^ routing.seed());
                            let r = case.prepare(session, live, routing, seed)?;
                            let engine = case.run(session, bank, false, &r)?;
                            let ring = case.run(session, bank, true, &r)?;
                            let equal = engine == ring;
                            let (written, informative) = informative_elements(&engine, &r, g);
                            let result = json!({
                                "symbol": instance.template.symbol,
                                "engine": instance.engine.symbol,
                                "knobs": format!("{:?}", instance.template.knobs),
                                "bank": bank_name,
                                "capacity": case.capacity,
                                "live": live,
                                "routing": routing.name(),
                                "real_weights": bank.real,
                                "written_rows": r.written.len(),
                                "written_elements": written,
                                "finite_nonzero_elements": informative,
                                "active_experts": r.active[0],
                                "padded_rows": r.padded,
                                "output_bitwise": equal,
                            });
                            if !equal {
                                let at = first_difference(&engine, &ring, 2);
                                failures.push(format!(
                                    "{} {bank_name} live {live} {}: output differs (first BF16 {at:?})",
                                    instance.template.symbol,
                                    routing.name(),
                                ));
                            }
                            eprintln!("{result}");
                            results.push(result);
                        }
                    }
                }
            }
        }
        // A case compares arithmetic only where it wrote finite, nonzero
        // outputs; infinities and zeros could hide a changed sum.
        let arithmetic = results
            .iter()
            .filter(|c| c["finite_nonzero_elements"].as_u64().unwrap_or(0) > 0)
            .count();
        let mut coverage = BTreeMap::<String, [u64; 3]>::new();
        for c in &results {
            let entry = coverage
                .entry(c["bank"].as_str().unwrap_or("?").to_string())
                .or_default();
            entry[0] += c["written_elements"].as_u64().unwrap_or(0);
            entry[1] += c["finite_nonzero_elements"].as_u64().unwrap_or(0);
            entry[2] += u64::from(c["finite_nonzero_elements"].as_u64().unwrap_or(0) > 0);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("instances".into(), json!(instances.len()));
        receipt.insert("executed_cases".into(), json!(results.len()));
        receipt.insert("arithmetic_cases".into(), json!(arithmetic));
        receipt.insert(
            "finite_nonzero_coverage".into(),
            json!(coverage
                .iter()
                .map(|(bank, [written, informative, cases])| (
                    bank.clone(),
                    json!({"written_elements": written, "finite_nonzero_elements": informative, "cases_with_finite_nonzero_outputs": cases})
                ))
                .collect::<BTreeMap<_, _>>()),
        );
        receipt.insert(
            "domains".into(),
            json!({
                "live_counts": "every count from 0 through each bucket's capacity",
                "routings": ["skewed", "hot", "crowded"],
                "nvfp4": "random E2M1 codes, UE4M3 local scales 0x00-0x7e (2% zero), F32 globals 2^-14 to 2^2, activations 2^-24 to 2^6",
                "mxfp8_generated": "E4M3 codes except NaN, UE8M0 scales 115-139, activations 2^-24 to 2^6",
                "mxfp8_ue8m0_every_code": "UE8M0 scales 0-254 (each scale block's rows within a few codes of a base that runs from 0 to 254 over the blocks), each with the E4M3 codes whose scaled product is exact and finite; activations +-(1 to 2) x 2^(118 - spread - base) of their block within the normal BF16 range, so every product is at most 2^3 and outputs stay finite and nonzero",
            }),
        );
        let executed = results.len();
        receipt.insert("cases".into(), json!(results));
        receipt.insert("failures".into(), json!(failures));
        receipt.insert("real_weights_skipped".into(), json!(real_skipped));
        receipt.insert("poisoned".into(), json!(["output"]));
        let path = session.receipt("moe_packed_ring_bitwise", receipt)?;
        eprintln!("receipt {}", path.display());
        if executed == 0 || arithmetic == 0 {
            Err("no bitwise arithmetic case ran".into())
        } else if failures.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{} packed ring cases differ:\n{}",
                failures.len(),
                failures.join("\n")
            ))
        }
    })
    .expect("packed ring bitwise equality with the packed engine");
}

// Decode reads each layer's experts from DRAM, and one projection's touched
// experts fit in L2, so a timed batch rotates through `TIMING_BANKS` weight
// banks and `TIMING_ROUTES` route sets: launch i uses bank i % 8 and route
// set i % 32, and the batch reads more distinct expert weights than L2
// holds.

const TIMING_BANKS: usize = 8;
const TIMING_ROUTES: usize = 32;
const TIMING_BATCH: usize = 32;

/// A case with every route set written for the full bucket.
fn rotating_case<'a>(
    session: &Session,
    instance: &'a PackedInstance,
    seed: u64,
) -> LabResult<PackedCase<'a>> {
    let case = PackedCase::new(session, instance, TIMING_ROUTES)?;
    write_rotating_routes(session, &case, seed)?;
    Ok(case)
}

/// The routes of timing set `set`: skewed, or with
/// `CHOPINFER_MOE_TIMING_LOCAL=1,2,3` exactly the listed local expert
/// counts per token in turn, each set on its own experts (12 per set covers
/// four tokens of three), so a batch's launches read distinct weights.
fn timing_routing(set: usize) -> Routing {
    match env_list("CHOPINFER_MOE_TIMING_LOCAL") {
        None => Routing::Skewed,
        Some(counts) => Routing::Local {
            experts: counts[set % counts.len()]
                .parse()
                .expect("CHOPINFER_MOE_TIMING_LOCAL lists counts"),
            first: set * 12,
        },
    }
}

/// The timing route sets' name for receipts.
fn timing_routes_name() -> String {
    std::env::var("CHOPINFER_MOE_TIMING_LOCAL").map_or_else(
        |_| "skewed".into(),
        |counts| format!("local experts {counts}"),
    )
}

/// Writes the activations and every route set of a timing case for the
/// full bucket, from seeds shared by every arm of an instance.
fn write_rotating_routes(session: &Session, case: &PackedCase, seed: u64) -> LabResult<()> {
    let live = case.capacity;
    case.prepare(session, live, timing_routing(0), seed)?;
    for set in 1..TIMING_ROUTES {
        case.prepare_routes(
            session,
            set,
            live,
            timing_routing(set),
            mix(seed ^ set as u64),
        )?;
    }
    Ok(())
}

/// One arm's timed batch as a CUDA graph, so that host-side argument and
/// tensor-map encoding stay outside the timing of kernels of a few
/// microseconds.
fn rotating_graph(
    session: &Session,
    case: &PackedCase,
    banks: &[Bank],
    ring: bool,
) -> LabResult<runtime_cuda::GraphExec> {
    let live = case.capacity;
    session.capture(|_| {
        for i in 0..TIMING_BATCH {
            case.launch(
                session,
                &banks[i % banks.len()],
                ring,
                live,
                false,
                i % TIMING_ROUTES,
            )?;
        }
        Ok(())
    })
}

/// Microseconds per launch: the median over `rounds` graph launches of a
/// batch, each after an L2 flush.
fn time_graph(session: &Session, graph: &runtime_cuda::GraphExec, rounds: usize) -> LabResult<f32> {
    let timing = time(session, 3, rounds, |stream| {
        graph.launch(stream).map_err(err)
    })?;
    Ok(timing.median_ms * 1000.0 / TIMING_BATCH as f32)
}

/// The packed engine (A) against the packed ring at its table row (B),
/// A B A B per repetition after one unrecorded repetition, with rotating
/// banks and route sets. The ring must not be slower than the engine it
/// replaces (median pair ratio at most 1), and the A-to-A noise gate is the
/// served timing's.
#[test]
#[ignore = "timing; run on an exclusive GPU with --ignored"]
fn grouped_moe_packed_ring_timing() {
    const REPETITIONS: usize = 7;
    const ROUNDS: usize = 9;
    const NOISE_LIMIT: f32 = 0.02;
    let instances = packed_instances();
    gpu_test(12 << 30, |session| {
        let mut rows = Vec::new();
        let (mut slower, mut inconclusive) = (Vec::new(), Vec::new());
        for (index, instance) in instances.iter().enumerate() {
            let g = packed_geometry(instance);
            let banks = (0..TIMING_BANKS)
                .map(|b| {
                    Bank::new(
                        session,
                        g,
                        instance.profile,
                        0x7134 ^ (index as u64) << 32 ^ (b as u64) << 40,
                    )
                })
                .collect::<LabResult<Vec<_>>>()?;
            let case = rotating_case(session, instance, mix(0x7134 ^ index as u64))?;
            let graphs = [
                rotating_graph(session, &case, &banks, false)?,
                rotating_graph(session, &case, &banks, true)?,
            ];
            let arm = |ring: bool| time_graph(session, &graphs[usize::from(ring)], ROUNDS);
            for ring in [false, true, false, true] {
                arm(ring)?;
            }
            let (mut a, mut b, mut noise, mut pairs) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for _ in 0..REPETITIONS {
                let a1 = arm(false)?;
                let b1 = arm(true)?;
                let a2 = arm(false)?;
                let b2 = arm(true)?;
                noise.push((a1 - a2).abs() / a1.min(a2));
                pairs.extend([b1 / a1, b2 / a2]);
                a.extend([a1, a2]);
                b.extend([b1, b2]);
            }
            let (a_median, b_median) = (median(&mut a), median(&mut b));
            let band_median = median(&mut noise);
            let ratio = median(&mut pairs);
            let plan = PackedCase::plan_of(&instance.template, g, case.capacity)?;
            let symbol = &instance.template.symbol;
            rows.push(json!({
                "symbol": symbol,
                "engine": instance.engine.symbol,
                "knobs": format!("{:?}", instance.template.knobs),
                "grid": plan.grid[0],
                "block": plan.block[0],
                "dynamic_shared_bytes": plan.dynamic_shared_bytes,
                "live": case.capacity,
                "engine_us": a_median,
                "ring_us": b_median,
                "ring_over_engine": ratio,
                "pair_ratio": {
                    "min": pairs.iter().copied().fold(f32::MAX, f32::min),
                    "max": pairs.iter().copied().fold(0.0, f32::max),
                },
                "a_to_a_noise_median": band_median,
            }));
            eprintln!(
                "{symbol:<56} engine {a_median:9.2} us  ring {b_median:9.2} us  ring/engine {ratio:.4}  A-A noise {:.2}%",
                100.0 * band_median
            );
            if band_median > NOISE_LIMIT {
                inconclusive.push(format!(
                    "{symbol} A-A noise median {:.2}%",
                    100.0 * band_median
                ));
            } else if ratio > 1.0 {
                slower.push(format!("{symbol} median pair ring/engine {ratio:.4}"));
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("repetitions".into(), json!(REPETITIONS));
        receipt.insert("rounds_per_timing".into(), json!(ROUNDS));
        receipt.insert(
            "order".into(),
            json!("one unrecorded A B A B, then A B A B per repetition; each timing replays a captured graph of 32 launches rotating 8 banks and 32 route sets after an L2 flush; microseconds per launch"),
        );
        let executed = rows.len();
        receipt.insert("timing_routes".into(), json!(timing_routes_name()));
        receipt.insert("shapes".into(), json!(rows));
        receipt.insert("slower".into(), json!(slower));
        receipt.insert("inconclusive_rerun".into(), json!(inconclusive));
        let path = session.receipt("moe_packed_ring_timing", receipt)?;
        eprintln!("receipt {}", path.display());
        if executed == 0 {
            Err("no timing shape ran".into())
        } else if !slower.is_empty() {
            Err(format!("packed ring slower than the engine:\n{}", slower.join("\n")))
        } else if !inconclusive.is_empty() {
            Err(format!("inconclusive, rerun:\n{}", inconclusive.join("\n")))
        } else {
            Ok(())
        }
    })
    .expect("packed ring timing");
}

/// Where one-token NVFP4 decode loses to FP8: the served FP8 entry (A, the
/// template at its table row) against the NVFP4 packed ring (B) at one
/// token, gate/up and down, on the same routes with exactly 1, 2 or 3 local
/// experts, A B A B. Cold graphs rotate 8 banks and 32 route sets on
/// distinct experts, so every launch reads DRAM after the L2 flush; warm
/// graphs reread bank 0's set 0, so all but the first launch read L2. A
/// warm ring close to its cold time while FP8 stays below it puts the gap in
/// the ring's dependent MMA chain rather than in memory. Measurement only.
#[test]
#[ignore = "timing; run on an exclusive GPU with --ignored"]
fn grouped_moe_one_token_floor() {
    const REPETITIONS: usize = 5;
    let packed = model_qwen38_contract::sm120::moe::packed_lab_instances()
        .into_iter()
        .filter(|i| {
            i.template.tokens_capacity == 1 && i.template.query.weight == WeightFormat::Nvfp4
        })
        .collect::<Vec<_>>();
    let fp8 = template::lab_instances(kernel_lab::catalogs());
    gpu_test(12 << 30, |session| {
        let mut rows = Vec::new();
        for (index, instance) in packed.iter().enumerate() {
            let g = packed_geometry(instance);
            let fp8_g = Geometry { format: Format::Fp8, ..g };
            let matching = fp8
                .iter()
                .filter(|i| i.baseline.tokens_capacity == 1 && Geometry::of(i) == fp8_g)
                .collect::<Vec<_>>();
            let [fp8_instance] = matching[..] else {
                return Err(format!(
                    "{} FP8 one-token entries match {}",
                    matching.len(),
                    instance.template.symbol
                ));
            };
            let seed = mix(0xf100 ^ index as u64);
            let banks = (0..TIMING_BANKS)
                .map(|b| Bank::new(session, g, instance.profile, seed ^ (b as u64) << 40))
                .collect::<LabResult<Vec<_>>>()?;
            let fp8_banks = (0..TIMING_BANKS)
                .map(|b| {
                    Bank::new(session, fp8_g, fp8_instance.baseline.profile, seed ^ 0xf8 ^ (b as u64) << 40)
                })
                .collect::<LabResult<Vec<_>>>()?;
            let ring = PackedCase::new(session, instance, TIMING_ROUTES)?;
            // FP8 cases hold one route set each.
            let fp8_cases = (0..TIMING_ROUTES)
                .map(|_| Case::new(session, fp8_instance))
                .collect::<LabResult<Vec<_>>>()?;
            for experts in 1..=3 {
                let routing = |set: usize| Routing::Local { experts, first: set * 12 };
                ring.prepare(session, 1, routing(0), seed)?;
                for set in 1..TIMING_ROUTES {
                    ring.prepare_routes(session, set, 1, routing(set), mix(seed ^ set as u64))?;
                }
                for (set, case) in fp8_cases.iter().enumerate() {
                    case.prepare(session, 1, routing(set), mix(seed ^ set as u64))?;
                }
                for warm in [false, true] {
                    let pick = |i: usize| {
                        if warm {
                            (0, 0)
                        } else {
                            (i % TIMING_BANKS, i % TIMING_ROUTES)
                        }
                    };
                    let a = session.capture(|_| {
                        for i in 0..TIMING_BATCH {
                            let (bank, set) = pick(i);
                            fp8_cases[set].launch(session, &fp8_banks[bank], Arm::Template, 1, false)?;
                        }
                        Ok(())
                    })?;
                    let b = session.capture(|_| {
                        for i in 0..TIMING_BATCH {
                            let (bank, set) = pick(i);
                            ring.launch(session, &banks[bank], true, 1, false, set)?;
                        }
                        Ok(())
                    })?;
                    let (fp8_us, ring_us, c) = packed_abab(session, &a, &b, REPETITIONS)?;
                    let memory = if warm { "warm" } else { "cold" };
                    eprintln!(
                        "{:<48} {experts} local {memory}: FP8 {fp8_us:7.2} us  NVFP4 ring {ring_us:7.2} us  ring-FP8 {:+6.2} us  ratio {:.3}  A-A noise {:.2}%",
                        instance.template.symbol,
                        ring_us - fp8_us,
                        c.ratio,
                        100.0 * c.noise
                    );
                    rows.push(json!({
                        "projection": if g.gate_up { "gate_up" } else { "down" },
                        "fp8": fp8_instance.template_symbol,
                        "ring": instance.template.symbol,
                        "ring_knobs": format!("{:?}", instance.template.knobs),
                        "local_experts": experts,
                        "memory": memory,
                        "fp8_us": fp8_us,
                        "ring_us": ring_us,
                        "ring_minus_fp8_us": ring_us - fp8_us,
                        "pair_ratio": c.ratio,
                        "a_to_a_noise_max": c.noise,
                    }));
                }
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "order".into(),
            json!("A (FP8) B (NVFP4 ring), one unrecorded A B then A B A B per repetition; each timing replays a captured graph of 32 one-token launches after an L2 flush; microseconds per launch"),
        );
        receipt.insert("repetitions".into(), json!(REPETITIONS));
        let executed = rows.len();
        receipt.insert("rows".into(), json!(rows));
        let path = session.receipt("moe_one_token_floor", receipt)?;
        eprintln!("receipt {}", path.display());
        if executed == 0 {
            Err("no one-token case ran".into())
        } else {
            Ok(())
        }
    })
    .expect("one-token floor");
}

/// Local routes put exactly the requested routes of each token on this
/// rank's experts, none twice, and a cold timing batch (launch i on bank
/// i % 8 and set i % 32, up to four tokens of three local experts) never
/// reads one bank's expert twice. CPU only.
#[test]
fn local_routes_hold_the_requested_counts_on_distinct_weights() {
    let instance = model_qwen38_contract::sm120::moe::packed_lab_instances()
        .into_iter()
        .find(|i| i.template.tokens_capacity == 4)
        .expect("a four-token packed instance");
    let g = packed_geometry(&instance);
    for experts in 1..=3 {
        for live in 1..=4 {
            let mut read = std::collections::BTreeSet::new();
            for i in 0..TIMING_BATCH {
                let set = i % TIMING_ROUTES;
                let r = routes(
                    g,
                    true,
                    live,
                    1,
                    Routing::Local {
                        experts,
                        first: set * 12,
                    },
                    7,
                );
                assert_eq!(r.written.len(), live * experts);
                for token in 0..live {
                    let local = r
                        .written
                        .iter()
                        .filter(|&&row| row / g.top_k == token)
                        .count();
                    assert_eq!(local, experts, "token {token}");
                }
                let active = r.active[0] as usize;
                assert_eq!(active, live * experts, "{experts} experts, {live} tokens");
                for &e in &r.active[1..=active] {
                    assert!(
                        read.insert((i % TIMING_BANKS, e)),
                        "launch {i} rereads expert {e}"
                    );
                }
            }
        }
    }
}

/// Given routes rebuilt from a synthetic route set's own choices give the
/// same metadata, and the sweep's wide counts cover prefill rows only. CPU
/// only.
#[test]
fn given_routes_rebuild_synthetic_metadata() {
    let instances = template::lab_instances(kernel_lab::catalogs());
    let prefill = instances
        .iter()
        .find(|i| i.schedule == Schedule::Prefill && !i.baseline.fp8_fragment_weights())
        .expect("a prefill instance with every expert local");
    assert_eq!(wide_live_counts(prefill), WIDE_LIVE);
    assert!(
        instances
            .iter()
            .filter(|i| i.schedule == Schedule::Ring)
            .all(|i| wide_live_counts(i).is_empty())
    );
    let g = Geometry::of(prefill);
    let block = usize::from(prefill.baseline.knobs.tile_m);
    for live in WIDE_LIVE.into_iter().chain([0, 1]) {
        for routing in [Routing::Skewed, Routing::Hot] {
            let r = routes(g, false, live, block, routing, 11);
            let mut chosen = vec![u32::MAX; live * g.top_k];
            for e in 0..g.experts {
                let range = r.offsets[e] as usize..r.offsets[e + 1] as usize;
                for &row in r.sorted[range].iter().filter(|&&row| row >= 0) {
                    chosen[row as usize] = e as u32;
                }
            }
            assert!(
                chosen.iter().all(|&e| e != u32::MAX),
                "every route is written"
            );
            let given = given_routes(g, live, block, &chosen);
            assert_eq!(given.offsets, r.offsets);
            assert_eq!(given.sorted, r.sorted);
            assert_eq!(given.padded, r.padded);
            assert_eq!(given.active, r.active);
            assert_eq!(given.written, r.written);
        }
    }
}

/// Route set files load only when every set has top_k IDs per row, each
/// below the expert count. CPU only.
#[test]
fn route_sets_check_their_shape() {
    let dir = std::env::temp_dir().join(format!("tune-routes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("routes.json");
    let write = |routes: Value| {
        let file = json!({"schema": "chopinfer.tune-routes.v1", "experts": 4, "top_k": 2, "routes": routes});
        std::fs::write(&path, file.to_string()).unwrap();
    };
    write(json!({"2": [[0, 1, 2, 3], [3, 2, 1, 0]]}));
    let sets = RouteSets::load(&path).expect("valid route sets");
    assert_eq!(sets.sets[&2].len(), 2);
    assert_eq!(sets.sets[&2][1], [3, 2, 1, 0]);
    write(json!({"2": [[0, 1, 2]]}));
    assert!(RouteSets::load(&path).is_err(), "too few IDs");
    write(json!({"2": [[0, 1, 2, 4]]}));
    assert!(RouteSets::load(&path).is_err(), "expert out of range");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Every UE8M0 code has admitted E4M3 codes, and each admitted product is
/// exact in BF16 and finite. CPU only.
#[test]
fn mxfp8_boundary_codes_are_exact() {
    for s in 0..=254u8 {
        let codes = exact_mxfp8_codes(s);
        assert!(
            codes.iter().any(|&c| c & 0x7f != 0),
            "scale {s} has only zeros"
        );
        for code in codes {
            let v = kernel_core::codec::SmallFloat::E4M3.decode_f32(code);
            let p = f64::from(v) * decode_ue8m0_f64(s);
            let bf16 = f64::from(bf16_to_f32(kernel_core::codec::f32_to_bf16(p as f32)));
            assert_eq!(bf16, p, "scale {s} code {code:#x}");
            assert!(p.is_finite());
        }
    }
}

/// The every-code MXFP8 bank with its activations, on two experts of each
/// projection: every UE8M0 code 0 to 254 occurs with nonzero products,
/// every product is at most 2^3, and at least 99% of the rows' sums lie
/// strictly between 0 and 80 in magnitude, so gate/up's SwiGLU (which
/// overflows or collapses to zero only beyond about 88) and down's route
/// weight keep the outputs finite and nonzero. CPU only; F64 sums.
#[test]
fn mxfp8_boundary_bank_keeps_outputs_finite_and_nonzero() {
    let decode = |c: u8| f64::from(kernel_core::codec::SmallFloat::E4M3.decode_f32(c));
    for instance in model_qwen38_contract::sm120::moe::packed_lab_instances()
        .iter()
        .filter(|i| {
            i.template.query.weight == WeightFormat::Mxfp8 && i.template.tokens_capacity == 1
        })
    {
        let g = Geometry {
            experts: 2,
            ..packed_geometry(instance)
        };
        let blocks = g.scale_blocks();
        let (codes, scales) = mxfp8_boundary_bank(g, 0xb0d);
        let x = boundary_activations(g, 0xac71, g.k)
            .chunks_exact(2)
            .map(|b| f64::from(bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))))
            .collect::<Vec<_>>();
        let mut seen = [false; 255];
        let (mut moderate, mut largest) = (0, 0.0_f64);
        for row in 0..g.experts * g.n {
            let mut sum = 0.0;
            for block in 0..blocks {
                let s = scales[packed_scale_offset(row, block, blocks)];
                for k in block * 32..(block + 1) * 32 {
                    let product = decode(codes[row * g.k + k]) * decode_ue8m0_f64(s) * x[k];
                    largest = largest.max(product.abs());
                    seen[usize::from(s)] |= product != 0.0;
                    sum += product;
                }
            }
            moderate += usize::from(sum != 0.0 && sum.abs() < 80.0);
        }
        let label = &instance.template.symbol;
        let missing = (0..255).filter(|&s| !seen[s]).collect::<Vec<_>>();
        assert!(
            missing.is_empty(),
            "{label}: scale codes without nonzero products {missing:?}"
        );
        assert!(largest <= 8.0, "{label}: product {largest}");
        assert!(
            moderate * 100 >= 99 * g.experts * g.n,
            "{label}: {moderate} of {} sums in (0, 80)",
            g.experts * g.n
        );
    }
}

mod engine_references {
    use kernel_lab::catalogs;
    include!("../engine_references.rs");
}

/// The frozen engine references kernel-lab compiles (`engine_references.rs`)
/// run at their engines' own launch plans: each entry's thread bound is the
/// engine plan's block and its parameters are the entry ABI, whatever the
/// schedule table gives the served template entry, and the reference specs
/// are the same whether built from the tuned rows or from the engines' own
/// schedules. CPU only.
#[test]
fn engine_references_keep_their_engine_launch_bounds() {
    let references = engine_references::moe_engine_reference_specs();
    let entries = references
        .iter()
        .flat_map(|spec| spec.entries.iter())
        .map(|entry| (entry.symbol.clone(), entry))
        .collect::<BTreeMap<_, _>>();
    let served = kernels_sm120::moe::cubin_specs(kernel_lab::catalogs(), None);
    let abi = |symbol: &str| {
        served
            .iter()
            .flat_map(|spec| spec.entries.iter())
            .find(|entry| entry.symbol == symbol)
            .map(|entry| entry.params.clone())
            .expect("served entry")
    };
    let instances = template::lab_instances(kernel_lab::catalogs());
    assert_eq!(entries.len(), instances.len());
    let mut failures = Vec::new();
    for instance in &instances {
        let v = &instance.baseline;
        let entry = entries[&instance.engine_symbol];
        let block = v
            .engine_plan(counts(
                instance,
                v.tokens_capacity as usize,
                v.query.grouped.expect("grouped").top_k as usize,
            ))
            .expect("engine plan")
            .block[0];
        if entry.max_threads != block || entry.min_blocks != 1 || entry.params != abi(&v.symbol) {
            failures.push(format!(
                "{}: bound {} x {}, engine block {block}",
                instance.engine_symbol, entry.max_threads, entry.min_blocks
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} references do not fit their engine plans:\n{}",
        failures.len(),
        instances.len(),
        failures.join("\n")
    );
    let at_engine_schedules = engine_references::moe_engine_reference_specs_for(
        &template::engine_schedule_instances(kernel_lab::catalogs()),
    );
    assert_eq!(
        format!("{references:?}"),
        format!("{at_engine_schedules:?}")
    );
}

// ---------------------------------------------------------------------------
// Offline schedule sweep (`CHOPINFER_TUNE_*`).
//
// For one model configuration and rank count, the rank shape
// (`RankShape::new`, `moe::Shape::classes`) selects the table rows; every
// row's legal knob sets (`template::candidates`: same K splits and portions,
// so the same arithmetic) are compiled into cubins outside the lab lock,
// checked bit for bit against the per-model engine with the poisoned cases
// above, and the survivors timed A B A B against the row's current (served)
// template entry. A candidate replaces the row only if it is faster by more
// than the A-to-A noise band plus the margin, in a screening pass and again
// in a fresh confirmation pass (`kernel_build::tuning::{best, accepts}`), and
// only if the row's null control (the current entry timed against itself)
// stays inside that band. Candidates the build or the driver refuses are
// reported with their reason. The sweep writes the resulting rows as a
// canonical partial table; the merge test folds them into the owning
// model's `tuning/schedules.json`.
//
// Prefill rows are also checked at the wide live counts (`WIDE_LIVE`) and
// in a graph replayed at decreasing counts, and with `CHOPINFER_TUNE_LIVE`
// timed there instead of at capacity, without changing the row
// (`wide_timing`).
//
// Compilation, A B A B timing, the replacement rule and table output take
// no MoE types. A dense GEMM or attention sweep supplies its own selection,
// candidate knobs, bitwise cases and launch closure and reuses them.

/// What a sweep invocation does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SweepPhase {
    /// Lists the selected rows and their candidate counts.
    Plan,
    /// Also compiles every candidate (no GPU).
    Compile,
    /// Also runs the bitwise cases and timing on `CHOPINFER_KERNEL_GPU`.
    Run,
}

struct SweepConfig {
    phase: SweepPhase,
    out: PathBuf,
    model: Model,
    rank: RankShape,
    ops: Option<Vec<String>>,
    buckets: Option<Vec<u32>>,
    margin: f64,
    repetitions: usize,
    /// Wide live counts to time prefill rows at (`CHOPINFER_TUNE_LIVE`),
    /// instead of screening at capacity; the rows themselves do not change.
    live: Option<Vec<usize>>,
    routes: Option<RouteSets>,
}

fn env_list(name: &str) -> Option<Vec<String>> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
}

impl SweepConfig {
    fn from_env() -> LabResult<Self> {
        let var = |name: &str| std::env::var(name).map_err(|_| format!("{name} is required"));
        let phase = match std::env::var("CHOPINFER_TUNE_PHASE").as_deref() {
            Ok("plan") => SweepPhase::Plan,
            Ok("compile") => SweepPhase::Compile,
            Ok("run") | Err(_) => SweepPhase::Run,
            Ok(other) => {
                return Err(format!(
                    "CHOPINFER_TUNE_PHASE {other}: plan, compile or run"
                ));
            }
        };
        let name = var("CHOPINFER_TUNE_MODEL")?;
        let model = match name.as_str() {
            "qwen38" => model_qwen38_contract::sm120::config::MODEL,
            "mimo26" => model_mimo26_contract::sm120::config::MODEL,
            _ => {
                return Err(format!(
                    "no grouped-MoE template instances for model {name}: the template \
                     covers the configurations of qwen38 and mimo26 (served formats \
                     mxfp4 and fp8_block128)"
                ));
            }
        };
        let ranks: u32 = var("CHOPINFER_TUNE_RANKS")?.parse().map_err(err)?;
        let rank = RankShape::new(model.config(), ranks)
            .map_err(|e| format!("{name} does not split over {ranks} ranks: {}", e.0))?;
        let buckets = env_list("CHOPINFER_TUNE_BUCKETS")
            .map(|list| list.iter().map(|b| b.parse().map_err(err)).collect())
            .transpose()?;
        let number = |name: &str, default: f64| -> LabResult<f64> {
            std::env::var(name)
                .ok()
                .map_or(Ok(default), |v| v.parse().map_err(err))
        };
        Ok(Self {
            phase,
            out: PathBuf::from(var("CHOPINFER_TUNE_OUT")?),
            model,
            rank,
            ops: env_list("CHOPINFER_TUNE_OPS"),
            buckets,
            margin: number("CHOPINFER_TUNE_MARGIN", 0.01)?,
            repetitions: number("CHOPINFER_TUNE_REPETITIONS", 7.0)? as usize,
            live: env_list("CHOPINFER_TUNE_LIVE")
                .map(|list| {
                    list.iter()
                        .map(|l| {
                            let live: usize = l.parse().map_err(err)?;
                            if WIDE_LIVE.contains(&live) {
                                Ok(live)
                            } else {
                                Err(format!(
                                    "CHOPINFER_TUNE_LIVE {live}: the wide counts are {WIDE_LIVE:?}"
                                ))
                            }
                        })
                        .collect::<LabResult<Vec<_>>>()
                })
                .transpose()?,
            routes: std::env::var_os("CHOPINFER_TUNE_ROUTES")
                .map(|path| RouteSets::load(Path::new(&path)))
                .transpose()?,
        })
    }

    /// The instances whose rows have one of the rank's expert shapes (one
    /// per MoE block class), filtered by operation and bucket. A class
    /// shape without rows is an error that names the shape: its rows (and
    /// served entries) must be added first.
    fn selection(&self) -> LabResult<Vec<Instance>> {
        let shapes = kernels_sm120::moe::Shape::classes(self.rank);
        if shapes.is_empty() {
            return Err("the model has no experts".into());
        }
        let instances = template::lab_instances(kernel_lab::catalogs());
        let mut at_shape = Vec::new();
        for (class, shape) in shapes {
            let rows: Vec<Instance> = instances
                .iter()
                .filter(|i| template::shape(&i.baseline) == shape)
                .cloned()
                .collect();
            if rows.is_empty() {
                return Err(format!(
                    "no schedule rows for the {class} rank shape {shape:?} ({} ranks): add \
                     its served entries and table rows first",
                    self.rank.ranks()
                ));
            }
            at_shape.extend(rows);
        }
        let selected: Vec<Instance> = at_shape
            .into_iter()
            .filter(|i| {
                let key = template::key(&i.baseline);
                self.ops.as_ref().is_none_or(|ops| ops.contains(&key.op))
                    && self
                        .buckets
                        .as_ref()
                        .is_none_or(|b| b.contains(&key.bucket))
            })
            .collect();
        if selected.is_empty() {
            return Err("CHOPINFER_TUNE_OPS and CHOPINFER_TUNE_BUCKETS select no row".into());
        }
        Ok(selected)
    }
}

fn knobs_json(knobs: &ScheduleKnobs) -> Value {
    Value::Object(
        knobs
            .row_knobs()
            .into_iter()
            .map(|(name, knob)| {
                let value = match knob {
                    Knob::Count(v) => json!(v),
                    Knob::Flag(v) => json!(v),
                };
                (name, value)
            })
            .collect(),
    )
}

fn write_json(path: &Path, value: &Value) -> LabResult<()> {
    let text = serde_json::to_string_pretty(value).map_err(err)? + "\n";
    std::fs::write(path, text).map_err(err)
}

/// One compiled candidate cubin.
struct Compiled {
    path: PathBuf,
    registers: u32,
}

/// Why a candidate did not reach the GPU checks.
fn refusal_kind(reason: &str) -> &'static str {
    if reason.starts_with("illegal:") {
        "illegal: the template cannot run these knobs correctly (template::legality)"
    } else if reason.starts_with("does not fit:") {
        "does not fit: shared memory or planner limits"
    } else if reason.starts_with("bitwise:") {
        "illegal: output differs from the engine"
    } else if reason.contains("exceeds allowed local bytes") {
        "audit: local or stack bytes (spills)"
    } else if reason.starts_with("load:") {
        "driver: module load or shared-memory opt-in"
    } else {
        "build: nvcc or audit"
    }
}

/// Compiles `specs` into `dir` against a scratch lock beside it (never a
/// committed lock). A failed spec is compiled alone to isolate it; cubins
/// already built are cache hits.
fn compile_candidates(
    toolchain: &Toolchain,
    dir: &Path,
    specs: &[CubinSpec],
) -> BTreeMap<String, Result<Compiled, String>> {
    let options = BuildOptions {
        lock_update: true,
        emit_cargo_directives: false,
        profile_line_info: false,
    };
    let collect = |built: Vec<kernel_build::BuiltCubin>| {
        built
            .into_iter()
            .map(|cubin| {
                let registers = cubin.entries.iter().map(|e| e.registers).max().unwrap_or(0);
                (
                    cubin.name,
                    Ok(Compiled {
                        path: cubin.path,
                        registers,
                    }),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };
    match kernel_build::build(toolchain, specs, dir, &dir.join("scratch.lock"), options) {
        Ok(built) => collect(built),
        Err(_) => specs
            .iter()
            .map(|spec| {
                let lock = dir.join(format!("{}.lock", spec.name));
                let result =
                    kernel_build::build(toolchain, std::slice::from_ref(spec), dir, &lock, options)
                        .map_err(|e| e.to_string())
                        .and_then(|built| {
                            collect(built)
                                .remove(&spec.name)
                                .ok_or_else(|| "no cubin".to_string())?
                        });
                (spec.name.clone(), result)
            })
            .collect(),
    }
}

/// A B A B timing of two launch sequences: one unrecorded repetition, then
/// `repetitions` recorded ones. Each timing is the median of `ROUNDS`
/// L2-flushed batches of `batch` launches, per launch in microseconds. The
/// ratio is the median of the adjacent pair ratios B1/A1 and B2/A2; the
/// noise is the widest A1-to-A2 difference, so an outlier widens the band
/// and keeps the current row.
struct Abab {
    a_us: f32,
    b_us: f32,
    comparison: Comparison,
}

fn abab(
    session: &Session,
    repetitions: usize,
    batch: usize,
    mut launch: impl FnMut(bool) -> LabResult<()>,
) -> LabResult<Abab> {
    const ROUNDS: usize = 15;
    let mut arm = |b: bool| -> LabResult<f32> {
        let timing = time(session, 3, ROUNDS, |_| {
            for _ in 0..batch {
                launch(b)?;
            }
            Ok(())
        })?;
        Ok(timing.median_ms * 1000.0 / batch as f32)
    };
    for b in [false, true, false, true] {
        arm(b)?;
    }
    let (mut a, mut b, mut pairs, mut noise) = (Vec::new(), Vec::new(), Vec::new(), 0.0_f32);
    for _ in 0..repetitions.max(1) {
        let a1 = arm(false)?;
        let b1 = arm(true)?;
        let a2 = arm(false)?;
        let b2 = arm(true)?;
        noise = noise.max((a1 - a2).abs() / a1.min(a2));
        pairs.extend([b1 / a1, b2 / a2]);
        a.extend([a1, a2]);
        b.extend([b1, b2]);
    }
    Ok(Abab {
        a_us: median(&mut a),
        b_us: median(&mut b),
        comparison: Comparison {
            ratio: f64::from(median(&mut pairs)),
            noise: f64::from(noise),
        },
    })
}

fn abab_json(timing: &Abab) -> Value {
    json!({
        "current_us": timing.a_us,
        "candidate_us": timing.b_us,
        "median_pair_ratio": timing.comparison.ratio,
        "a_to_a_noise": timing.comparison.noise,
    })
}

fn load_candidate(session: &Session, instance: Instance, path: &Path) -> LabResult<Candidate> {
    let v = &instance.baseline;
    let capacity = counts(
        &instance,
        v.tokens_capacity as usize,
        v.query.grouped.expect("grouped").top_k as usize,
    );
    let shared = instance.plan(capacity).map_err(err)?.dynamic_shared_bytes;
    let module = session.load_cubin(&std::fs::read(path).map_err(err)?)?;
    let function = module.function(&instance.template_symbol).map_err(err)?;
    if shared > 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(shared)
            .map_err(err)?;
    }
    Ok(Candidate {
        instance,
        entry: Entry { function },
    })
}

/// Live rows above the decode buckets, where wide decode and verify steps
/// run a model's prefill entries: the sweep checks every prefill row's
/// candidates at these counts too, and times them there on request
/// (`CHOPINFER_TUNE_LIVE`).
const WIDE_LIVE: [usize; 4] = [32, 64, 128, 256];

/// Weight banks a wide-count timing rotates through, so consecutive
/// launches read other weights from DRAM, as consecutive layers do.
const WIDE_BANKS: usize = 4;

/// The wide counts at or below a prefill instance's capacity; none for ring
/// instances, whose buckets are the decode counts.
fn wide_live_counts(instance: &Instance) -> Vec<usize> {
    if instance.schedule != Schedule::Prefill {
        return Vec::new();
    }
    let capacity = instance.baseline.tokens_capacity as usize;
    WIDE_LIVE
        .into_iter()
        .filter(|&live| live <= capacity)
        .collect()
}

/// Given route sets per live count (`CHOPINFER_TUNE_ROUTES`): a JSON object
/// `{"schema": "chopinfer.tune-routes.v1", "experts": E, "top_k": K,
/// "routes": {"<live>": [[expert of route k of token t at t * K + k, ...],
/// ...]}}`, for example route choices recorded from a model's hidden
/// states. The first set of a count screens, the next ones confirm.
struct RouteSets {
    experts: usize,
    top_k: usize,
    sets: BTreeMap<usize, Vec<Vec<u32>>>,
}

impl RouteSets {
    fn load(path: &Path) -> LabResult<Self> {
        let value: Value =
            serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)?;
        if value["schema"] != "chopinfer.tune-routes.v1" {
            return Err(format!("{}: not chopinfer.tune-routes.v1", path.display()));
        }
        let field = |name: &str| {
            value[name]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("{}: missing {name}", path.display()))
        };
        let (experts, top_k) = (field("experts")?, field("top_k")?);
        let mut sets = BTreeMap::new();
        for (live, list) in value["routes"]
            .as_object()
            .ok_or("routes must be an object")?
        {
            let live: usize = live.parse().map_err(err)?;
            let list = list
                .as_array()
                .ok_or("route sets must be arrays")?
                .iter()
                .map(|set| {
                    let ids = set
                        .as_array()
                        .ok_or("a route set must be an array")?
                        .iter()
                        .map(|e| {
                            e.as_u64()
                                .filter(|&e| (e as usize) < experts)
                                .map(|e| e as u32)
                        })
                        .collect::<Option<Vec<_>>>()
                        .ok_or("route expert IDs must be below experts")?;
                    if ids.len() != live * top_k {
                        return Err(format!("a {live}-row route set needs {} IDs", live * top_k));
                    }
                    Ok(ids)
                })
                .collect::<LabResult<Vec<_>>>()?;
            sets.insert(live, list);
        }
        Ok(Self {
            experts,
            top_k,
            sets,
        })
    }

    /// The sets of one live count, after checking they fit the instance:
    /// its experts, all local to the rank, and its top_k.
    fn of(&self, case: &Case, live: usize) -> LabResult<&[Vec<u32>]> {
        if self.experts != case.g.experts || self.top_k != case.g.top_k {
            return Err(format!(
                "given routes are for {} experts, top {}; {} has {} local experts, top {}",
                self.experts,
                self.top_k,
                case.instance.baseline.symbol,
                case.g.experts,
                case.g.top_k
            ));
        }
        Ok(self.sets.get(&live).map_or(&[], Vec::as_slice))
    }
}

/// The routes of one bitwise or timing input.
#[derive(Clone, Copy)]
enum Input<'a> {
    Synthetic(Routing),
    Given(usize, &'a [u32]),
}

impl Input<'_> {
    fn name(&self) -> String {
        match self {
            Input::Synthetic(routing) => routing.name().into(),
            Input::Given(set, _) => format!("given set {set}"),
        }
    }

    fn prepare(&self, session: &Session, case: &Case, live: usize, seed: u64) -> LabResult<Routes> {
        match *self {
            Input::Synthetic(routing) => case.prepare(session, live, routing, seed),
            Input::Given(_, chosen) => case.prepare_given(session, live, chosen, seed),
        }
    }
}

/// Bit-for-bit checks of the candidates against the per-model engine over
/// the poisoned cases: `poison::live_counts`, and for prefill rows the wide
/// counts, on skewed and hot routes and the first given route set of each
/// count. Returns each candidate's first failure. The current template
/// entry must match first, or the row is not swept.
fn sweep_bitwise(
    session: &Session,
    bank: &Bank,
    case: &Case,
    candidates: &[&Candidate],
    given: Option<&RouteSets>,
    index: usize,
) -> LabResult<Vec<Option<String>>> {
    let mut failures: Vec<Option<String>> = vec![None; candidates.len()];
    let mut counts = live_counts(case.instance);
    for live in wide_live_counts(case.instance) {
        if !counts.contains(&live) {
            counts.push(live);
        }
    }
    for live in counts {
        let mut inputs = vec![
            Input::Synthetic(Routing::Skewed),
            Input::Synthetic(Routing::Hot),
        ];
        if let Some(given) = given
            && let Some(first) = given.of(case, live)?.first()
        {
            inputs.push(Input::Given(0, first));
        }
        for (position, input) in inputs.into_iter().enumerate() {
            let seed = mix(index as u64 ^ (live as u64) << 16 ^ position as u64);
            let r = input.prepare(session, case, live, seed)?;
            let engine = case.run(session, bank, Arm::Engine, &r)?;
            let same = |other: &Vec<u8>| -> LabResult<()> {
                if let Some(at) = first_difference(&engine, other, 2) {
                    return Err(format!(
                        "output differs at live {live} {}: first BF16 {at}",
                        input.name()
                    ));
                }
                Ok(())
            };
            let template = case.run(session, bank, Arm::Template, &r)?;
            same(&template).map_err(|e| {
                format!(
                    "{}: the current row does not match the engine: {e}",
                    case.instance.baseline.symbol
                )
            })?;
            for (&candidate, failure) in candidates.iter().zip(failures.iter_mut()) {
                if failure.is_some() {
                    continue;
                }
                *failure = case
                    .run(session, bank, Arm::Candidate(candidate), &r)
                    .and_then(|output| same(&output))
                    .err();
            }
        }
    }
    Ok(failures)
}

/// Graph replay at decreasing live counts: each prefill entry (the current
/// row and every candidate still equal) is captured once in a graph at its
/// capacity plan, as a serving graph is, then replayed after the route
/// metadata shrinks through the wide counts to zero. Every replay must
/// write exactly its live rows, leave the workspace untouched and equal the
/// engine launched directly at that count, so no state of a larger replay
/// leaks into a smaller one. Returns each candidate's first failure.
fn sweep_graph_replay(
    session: &Session,
    bank: &Bank,
    case: &Case,
    candidates: &[&Candidate],
    failures: &[Option<String>],
    index: usize,
) -> LabResult<Vec<Option<String>>> {
    let mut found: Vec<Option<String>> = vec![None; candidates.len()];
    let mut counts = wide_live_counts(case.instance);
    if counts.is_empty() {
        return Ok(found);
    }
    counts.reverse();
    counts.push(0);
    let capture =
        |arm: Arm| session.capture(|_| case.launch(session, bank, arm, case.capacity, true));
    let template = capture(Arm::Template)?;
    let mut graphs = Vec::new();
    for ((&candidate, failure), found) in candidates.iter().zip(failures).zip(found.iter_mut()) {
        if failure.is_some() {
            graphs.push(None);
            continue;
        }
        match capture(Arm::Candidate(candidate)) {
            Ok(graph) => graphs.push(Some(graph)),
            Err(e) => {
                *found = Some(format!("graph capture: {e}"));
                graphs.push(None);
            }
        }
    }
    for live in counts {
        let seed = mix(index as u64 ^ (live as u64) << 16 ^ 0x9a7);
        let r = case.prepare(session, live, Routing::Skewed, seed)?;
        let engine = case.run(session, bank, Arm::Engine, &r)?;
        let replay = |graph: &runtime_cuda::GraphExec, label: &str| -> LabResult<()> {
            let output = case.checked(session, label, &r, || {
                graph.launch(&session.stream).map_err(err)
            })?;
            match first_difference(&engine, &output, 2) {
                Some(at) => Err(format!(
                    "graph replay differs at live {live} after larger counts: first BF16 {at}"
                )),
                None => Ok(()),
            }
        };
        replay(
            &template,
            &format!("{} graph", case.instance.template_symbol),
        )
        .map_err(|e| {
            format!(
                "{}: the current row fails decreasing-count replay: {e}",
                case.instance.baseline.symbol
            )
        })?;
        for ((graph, candidate), found) in graphs.iter().zip(candidates).zip(found.iter_mut()) {
            if let Some(graph) = graph
                && found.is_none()
            {
                *found = replay(
                    graph,
                    &format!("{} graph", candidate.instance.template_symbol),
                )
                .err();
            }
        }
    }
    Ok(found)
}

/// Times a prefill row's bitwise-equal candidates at each wide live count
/// against the row's current entry, A B A B with 32-launch batches that
/// rotate through `banks`. Per count: a null control (the current entry
/// against itself) on the screening input, every candidate screened on it,
/// and the best (`tuning::best`) confirmed on the other inputs. Inputs are
/// the given route sets of the count (first screens) plus fresh skewed
/// routes, or without given sets skewed, fresh skewed and hot routes.
/// Returns the report and the candidates confirmed at some count.
fn wide_timing<'c>(
    session: &Session,
    banks: &[&Bank],
    case: &Case,
    equal: &[&'c Candidate],
    live_counts: &[usize],
    config: &SweepConfig,
    index: usize,
) -> LabResult<(Value, Vec<&'c Candidate>)> {
    let mut report = Vec::new();
    let mut winners: Vec<&'c Candidate> = Vec::new();
    for &live in live_counts {
        if live > case.capacity {
            continue;
        }
        let given = match &config.routes {
            Some(routes) => routes.of(case, live)?,
            None => &[],
        };
        let mut inputs = given
            .iter()
            .enumerate()
            .take(3)
            .map(|(set, chosen)| (Input::Given(set, chosen.as_slice()), 0x7133 ^ set as u64))
            .collect::<Vec<_>>();
        if inputs.is_empty() {
            inputs.push((Input::Synthetic(Routing::Skewed), 0x7133));
            inputs.push((Input::Synthetic(Routing::Hot), 0xc1f));
        }
        inputs.push((Input::Synthetic(Routing::Skewed), 0xc0f));
        let timed = |input: Input, seed: u64, b: Arm| -> LabResult<Abab> {
            input.prepare(session, case, live, mix(seed ^ index as u64))?;
            let mut turn = 0;
            abab(session, config.repetitions, 32, |second| {
                turn += 1;
                let arm = if second { b } else { Arm::Template };
                case.launch(session, banks[turn % banks.len()], arm, live, false)
            })
        };
        let (screen, seed) = inputs[0];
        let null = timed(screen, seed, Arm::Template)?;
        let stable = !tuning::accepts(null.comparison, config.margin)
            && null.comparison.ratio <= 1.0 + null.comparison.noise + config.margin;
        let mut screened = Vec::new();
        let mut rows = Vec::new();
        for &candidate in equal {
            let timing = timed(screen, seed, Arm::Candidate(candidate))?;
            eprintln!(
                "live {live:>3} {:<80} {:.4} (noise {:.4})",
                candidate.instance.template_symbol,
                timing.comparison.ratio,
                timing.comparison.noise
            );
            rows.push(json!({
                "symbol": candidate.instance.template_symbol,
                "knobs": knobs_json(&candidate.instance.knobs),
                "timing": abab_json(&timing),
            }));
            screened.push((candidate, timing.comparison));
        }
        let row_knobs = screened
            .iter()
            .map(|(c, comparison)| (c.instance.knobs.row_knobs(), *comparison))
            .collect::<Vec<_>>();
        let best =
            tuning::best(row_knobs.iter().map(|(k, c)| (k, *c)), config.margin).map(|knobs| {
                screened
                    .iter()
                    .find(|(c, _)| c.instance.knobs.row_knobs() == *knobs)
                    .expect("best is screened")
                    .0
            });
        let mut entry = json!({
            "live": live,
            "screen": screen.name(),
            "null_control": abab_json(&null),
            "stable": stable,
            "accepted": false,
            "candidates": rows,
        });
        if stable && let Some(best) = best {
            let mut accepted = true;
            let mut runs = Vec::new();
            for &(input, seed) in &inputs[1..] {
                let run = timed(input, seed, Arm::Candidate(best))?;
                accepted &= tuning::accepts(run.comparison, config.margin);
                let mut timing = abab_json(&run);
                timing["input"] = json!(input.name());
                runs.push(timing);
                if !accepted {
                    break;
                }
            }
            entry["best"] = json!(best.instance.template_symbol);
            entry["confirmation"] = json!(runs);
            entry["accepted"] = json!(accepted);
            if accepted
                && !winners
                    .iter()
                    .any(|w| w.instance.template_symbol == best.instance.template_symbol)
            {
                winners.push(best);
            }
        }
        report.push(entry);
    }
    Ok((json!(report), winners))
}

/// Candidates that never reached the GPU checks, with their reasons, and
/// counts by kind.
fn refusal_report(refused: &[(String, String)]) -> Value {
    let mut kinds = BTreeMap::<&str, usize>::new();
    for (_, reason) in refused {
        *kinds.entry(refusal_kind(reason)).or_default() += 1;
    }
    json!({
        "count": refused.len(),
        "by_kind": kinds,
        "candidates": refused
            .iter()
            .map(|(symbol, reason)| json!({
                "symbol": symbol,
                "kind": refusal_kind(reason),
                "reason": reason,
            }))
            .collect::<Vec<_>>(),
    })
}

fn write_refusals(out: &Path, refused: &[(String, String)]) -> LabResult<()> {
    let lines = refused
        .iter()
        .map(|(symbol, reason)| format!("{symbol}\t{}\t{reason}", refusal_kind(reason)))
        .collect::<Vec<_>>();
    std::fs::write(out.join("refused.txt"), lines.join("\n") + "\n").map_err(err)?;
    let report = refusal_report(refused);
    eprintln!(
        "{} candidates refused before the GPU checks: {}",
        refused.len(),
        report["by_kind"]
    );
    Ok(())
}

#[test]
#[ignore = "offline schedule sweep; set CHOPINFER_TUNE_* and run with --ignored"]
fn grouped_moe_schedule_sweep() {
    let config = SweepConfig::from_env().expect("sweep settings");
    if config.phase == SweepPhase::Run {
        assert!(
            kernel_lab::selected_ordinals()
                .expect("GPU selection")
                .is_some(),
            "the run phase needs an idle GPU in CHOPINFER_KERNEL_GPU"
        );
    }
    let instances = config.selection().expect("sweep selection");
    std::fs::create_dir_all(&config.out).expect("create CHOPINFER_TUNE_OUT");
    // Knob sets refused before compiling (illegal, or no plan that fits),
    // then candidates the build, the driver or the bitwise cases refuse.
    let mut refused: Vec<(String, String)> = Vec::new();
    let mut candidates: Vec<Vec<Instance>> = Vec::new();
    let mut plan = Vec::new();
    for instance in &instances {
        let mut legal = Vec::new();
        let mut row_refused = 0;
        for (knobs, candidate) in template::candidate_space(instance) {
            match candidate {
                Ok(candidate) => legal.push(candidate),
                Err(reason) => {
                    row_refused += 1;
                    refused.push((
                        format!("{} {}", instance.baseline.symbol, knobs_json(&knobs)),
                        reason,
                    ));
                }
            }
        }
        plan.push(json!({
            "symbol": instance.baseline.symbol,
            "key": template::key(&instance.baseline).to_string(),
            "current": knobs_json(&instance.knobs),
            "candidates": legal.len(),
            "refused_before_compiling": row_refused,
        }));
        candidates.push(legal);
    }
    write_json(&config.out.join("plan.json"), &json!(plan)).expect("write plan.json");
    eprintln!(
        "{} rows, {} candidates",
        instances.len(),
        candidates.iter().map(Vec::len).sum::<usize>()
    );
    write_refusals(&config.out, &refused).expect("write refused.txt");
    if config.phase == SweepPhase::Plan {
        return;
    }

    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    let mut compiled = Vec::new();
    let mut compile_report = Vec::new();
    for (instance, candidates) in instances.iter().zip(&candidates) {
        let specs = candidates
            .iter()
            .map(Instance::cubin_spec)
            .collect::<Vec<_>>();
        let dir = config.out.join("cubins").join(&instance.baseline.symbol);
        let started = std::time::Instant::now();
        let results = compile_candidates(&toolchain, &dir, &specs);
        let failed = results.values().filter(|r| r.is_err()).count();
        eprintln!(
            "{}: {} candidates compiled, {failed} refused, {:.0} s",
            instance.baseline.symbol,
            results.len() - failed,
            started.elapsed().as_secs_f32()
        );
        for candidate in candidates {
            if let Err(reason) = &results[&candidate.template_cubin] {
                refused.push((candidate.template_symbol.clone(), reason.clone()));
            }
        }
        compile_report.push(json!({
            "symbol": instance.baseline.symbol,
            "candidates": candidates.iter().map(|c| {
                let r = &results[&c.template_cubin];
                json!({
                    "symbol": c.template_symbol,
                    "knobs": knobs_json(&c.knobs),
                    "registers": r.as_ref().ok().map(|c| c.registers),
                    "refused": r.as_ref().err(),
                    "refusal_kind": r.as_ref().err().map(|e| refusal_kind(e)),
                })
            }).collect::<Vec<_>>(),
        }));
        compiled.push(results);
    }
    write_json(
        &config.out.join("compile.json"),
        &json!({"refused": refusal_report(&refused), "rows": compile_report}),
    )
    .expect("write compile.json");
    write_refusals(&config.out, &refused).expect("write refused.txt");
    if config.phase == SweepPhase::Compile {
        return;
    }

    let mut table = Table::new(kernel_lab::catalogs().schedules().target);
    let mut report = Vec::new();
    let mut illegal = Vec::new();
    let mut unstable = Vec::new();
    let mut changed = Vec::new();
    gpu_test(12 << 30, |session| {
        for (index, (profile, _, group)) in banks_of(&instances).into_iter().enumerate() {
            let g = Geometry::of(group[0]);
            let bank = Bank::new(session, g, profile, 0x5eed ^ (index as u64) << 32)?;
            let wide = config.live.is_some()
                && group.iter().any(|i| i.schedule == Schedule::Prefill);
            let extra_banks = if wide {
                (1..WIDE_BANKS)
                    .map(|b| Bank::new(session, g, profile, 0x5eed ^ (index as u64) << 32 ^ (b as u64) << 48))
                    .collect::<LabResult<Vec<_>>>()?
            } else {
                Vec::new()
            };
            let banks = std::iter::once(&bank).chain(&extra_banks).collect::<Vec<_>>();
            for instance in group {
                let at = instances
                    .iter()
                    .position(|i| i.baseline.symbol == instance.baseline.symbol)
                    .unwrap();
                let case = Case::new(session, instance)?;
                let mut rows = Vec::new();
                let mut loaded = Vec::new();
                for candidate in &candidates[at] {
                    let mut row = json!({
                        "symbol": candidate.template_symbol,
                        "knobs": knobs_json(&candidate.knobs),
                    });
                    match &compiled[at][&candidate.template_cubin] {
                        Err(reason) => {
                            row["refused"] = json!(reason);
                            row["refusal_kind"] = json!(refusal_kind(reason));
                        }
                        Ok(c) => {
                            row["registers"] = json!(c.registers);
                            match load_candidate(session, candidate.clone(), &c.path) {
                                Ok(entry) => loaded.push((rows.len(), entry)),
                                Err(e) => {
                                    let reason = format!("load: {e}");
                                    row["refused"] = json!(reason);
                                    row["refusal_kind"] = json!(refusal_kind(&reason));
                                    refused.push((candidate.template_symbol.clone(), reason));
                                }
                            }
                        }
                    }
                    rows.push(row);
                }
                let refs = loaded.iter().map(|(_, c)| c).collect::<Vec<_>>();
                let mut failures =
                    sweep_bitwise(session, &bank, &case, &refs, config.routes.as_ref(), index)?;
                let replayed = sweep_graph_replay(session, &bank, &case, &refs, &failures, index)?;
                for (failure, replay) in failures.iter_mut().zip(replayed) {
                    if failure.is_none() {
                        *failure = replay;
                    }
                }
                let wide_row = config.live.is_some() && instance.schedule == Schedule::Prefill;
                let batch = if instance.schedule == Schedule::Prefill {
                    4
                } else {
                    32
                };
                let live = case.capacity;
                let timed_at = |routing: Routing, seed: u64, live: usize, b: Arm| {
                    case.prepare(session, live, routing, mix(seed ^ index as u64))?;
                    abab(session, config.repetitions, batch, |second| {
                        let arm = if second { b } else { Arm::Template };
                        case.launch(session, &bank, arm, live, false)
                    })
                };
                let timed = |seed: u64, b: Arm| timed_at(Routing::Skewed, seed, live, b);
                // Confirmation inputs: fresh skewed and hot routes at
                // capacity, and fresh skewed routes at half capacity, so a
                // win that depends on one route pattern does not change the
                // row.
                let mut confirmations = vec![
                    (Routing::Skewed, 0xc0f_u64, live),
                    (Routing::Hot, 0xc1f, live),
                ];
                if live / 2 >= 1 {
                    confirmations.push((Routing::Skewed, 0xc2f, live / 2));
                }
                // Null control: the current entry against itself. A band
                // that would accept it as faster or slower means this GPU's
                // timings cannot decide the row.
                let null = if wide_row {
                    None
                } else {
                    Some(timed(0x7133, Arm::Template)?)
                };
                let stable = null.as_ref().is_none_or(|null| {
                    !tuning::accepts(null.comparison, config.margin)
                        && null.comparison.ratio <= 1.0 + null.comparison.noise + config.margin
                });
                let mut screened = Vec::new();
                let mut equal = Vec::new();
                for ((position, candidate), failure) in loaded.iter().zip(&failures) {
                    if let Some(failure) = failure {
                        // Excluded from timing and reported: these knobs are
                        // meant to be rounding-neutral, so a difference is a
                        // template defect for its legality check.
                        let reason = format!("bitwise: {failure}");
                        rows[*position]["bitwise"] = json!(failure);
                        rows[*position]["refusal_kind"] = json!(refusal_kind(&reason));
                        illegal.push(format!(
                            "{}: {failure}",
                            candidate.instance.template_symbol
                        ));
                        refused.push((candidate.instance.template_symbol.clone(), reason));
                        continue;
                    }
                    rows[*position]["bitwise"] = json!("equal");
                    if wide_row {
                        equal.push(candidate);
                        continue;
                    }
                    let timing = timed(0x7133, Arm::Candidate(candidate))?;
                    rows[*position]["timing"] = abab_json(&timing);
                    eprintln!(
                        "{:<80} {:.4} (noise {:.4})",
                        candidate.instance.template_symbol,
                        timing.comparison.ratio,
                        timing.comparison.noise
                    );
                    screened.push((candidate, timing.comparison));
                }
                let row_knobs = screened
                    .iter()
                    .map(|(c, comparison)| (c.instance.knobs.row_knobs(), *comparison))
                    .collect::<Vec<_>>();
                let best = tuning::best(row_knobs.iter().map(|(k, c)| (k, *c)), config.margin)
                    .map(|knobs| {
                        screened
                            .iter()
                            .find(|(c, _)| c.instance.knobs.row_knobs() == *knobs)
                            .expect("best is screened")
                            .0
                    });
                let mut decision = json!({"changed": false});
                if let Some(null) = &null {
                    decision["null_control"] = abab_json(null);
                }
                let mut knobs = instance.knobs;
                if wide_row {
                    let live = config.live.as_deref().unwrap_or_default();
                    let (wide, winners) =
                        wide_timing(session, &banks, &case, &equal, live, &config, index)?;
                    decision["wide"] = wide;
                    // Non-regression controls of each wide winner on the
                    // row's own inputs: capacity and half capacity.
                    let mut controls = Vec::new();
                    if !winners.is_empty() {
                        let null = timed(0x7133, Arm::Template)?;
                        controls.push(json!({"symbol": "null control", "timing": abab_json(&null)}));
                    }
                    for winner in winners {
                        for &(routing, seed, at) in &confirmations {
                            let run = timed_at(routing, seed, at, Arm::Candidate(winner))?;
                            let mut entry = abab_json(&run);
                            entry["routing"] = json!(routing.name());
                            entry["live"] = json!(at);
                            controls.push(json!({
                                "symbol": winner.instance.template_symbol,
                                "timing": entry,
                            }));
                        }
                    }
                    decision["capacity_controls"] = json!(controls);
                } else if let (false, Some(null)) = (stable, &null) {
                    decision["unstable"] = json!(true);
                    unstable.push(format!(
                        "{}: null control ratio {:.4}, noise {:.4}",
                        instance.baseline.symbol, null.comparison.ratio, null.comparison.noise
                    ));
                } else if let Some(best) = best {
                    let mut runs = Vec::new();
                    let mut accepted = true;
                    for &(routing, seed, at) in &confirmations {
                        let run = timed_at(routing, seed, at, Arm::Candidate(best))?;
                        accepted &= tuning::accepts(run.comparison, config.margin);
                        let mut entry = abab_json(&run);
                        entry["routing"] = json!(routing.name());
                        entry["live"] = json!(at);
                        runs.push((entry, run));
                        if !accepted {
                            break;
                        }
                    }
                    // The capacity run on fresh skewed routes is the reported gain.
                    let confirm = &runs[0].1;
                    decision["best"] = json!(best.instance.template_symbol);
                    decision["confirmation"] =
                        json!(runs.iter().map(|(entry, _)| entry.clone()).collect::<Vec<_>>());
                    decision["changed"] = json!(accepted);
                    if accepted {
                        knobs = best.instance.knobs;
                        changed.push(format!(
                            "{}: {} -> {}, {:.2} us -> {:.2} us ({:.1}% faster, median pair ratio {:.4}, noise {:.1}%; all {} confirmations pass)",
                            template::key(&instance.baseline),
                            knobs_json(&instance.knobs),
                            knobs_json(&knobs),
                            confirm.a_us,
                            confirm.b_us,
                            100.0 * (1.0 - confirm.comparison.ratio),
                            confirm.comparison.ratio,
                            100.0 * confirm.comparison.noise,
                            runs.len(),
                        ));
                    }
                }
                table
                    .upsert(tuning::Row {
                        key: template::key(&instance.baseline),
                        schedule: knobs.schedule().name().into(),
                        knobs: knobs.row_knobs(),
                    })
                    .map_err(err)?;
                let entry = json!({
                    "symbol": instance.baseline.symbol,
                    "key": template::key(&instance.baseline).to_string(),
                    "current": knobs_json(&instance.knobs),
                    "decision": decision,
                    "candidates": rows,
                });
                // One line per finished row, so a failed job keeps them.
                let mut progress = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(config.out.join("progress.jsonl"))
                    .map_err(err)?;
                std::io::Write::write_all(&mut progress, format!("{entry}\n").as_bytes())
                    .map_err(err)?;
                report.push(entry);
            }
        }
        Ok(())
    })
    .expect("schedule sweep");
    std::fs::write(config.out.join("rows.json"), table.render()).expect("write rows.json");
    write_refusals(&config.out, &refused).expect("write refused.txt");
    write_json(
        &config.out.join("sweep.json"),
        &json!({
            "model": format!("{:?}", config.model),
            "ranks": config.rank.ranks(),
            "gpu": std::env::var("CHOPINFER_KERNEL_GPU").ok(),
            "margin": config.margin,
            "repetitions": config.repetitions,
            "rule": "median pair ratio below 1 - (widest A-to-A noise + margin) in screening (skewed routes at capacity) and in every confirmation (fresh skewed and hot routes at capacity, fresh skewed routes at half capacity), with the row's null control inside that band; ties by knob order",
            "changed": changed,
            "unstable_rows_kept": unstable,
            "refused": refusal_report(&refused),
            "illegal_candidates": illegal,
            "rows": report,
        }),
    )
    .expect("write sweep.json");
    std::fs::write(config.out.join("changed.txt"), changed.join("\n") + "\n")
        .expect("write changed.txt");
    eprintln!("{} rows changed:\n{}", changed.len(), changed.join("\n"));
    if !unstable.is_empty() {
        eprintln!(
            "{} rows kept because their null control was unstable:\n{}",
            unstable.len(),
            unstable.join("\n")
        );
    }
    if !illegal.is_empty() {
        eprintln!(
            "WARNING: {} candidates differ from the engine and were excluded as illegal; \
             template::legality should refuse them:\n{}",
            illegal.len(),
            illegal.join("\n")
        );
    }
}

/// Folds sweep outputs (`rows.json` in each `CHOPINFER_TUNE_RESULTS`
/// directory, colon separated) into the committed tables of the models
/// whose rows they replace, and writes each model's table to
/// `CHOPINFER_TUNE_TABLE_OUT/<model>.json`, its new `tuning/schedules.json`.
/// Every row must name a served entry's key in a model's table and pass the
/// build's row checks (`template::validate_row`).
#[test]
#[ignore = "merges sweep rows; set CHOPINFER_TUNE_RESULTS and CHOPINFER_TUNE_TABLE_OUT"]
fn grouped_moe_schedule_merge() {
    let results = std::env::var("CHOPINFER_TUNE_RESULTS").expect("CHOPINFER_TUNE_RESULTS");
    let out = std::env::var_os("CHOPINFER_TUNE_TABLE_OUT").expect("CHOPINFER_TUNE_TABLE_OUT");
    let mut tables = kernel_lab::MODELS
        .iter()
        .flat_map(|model| {
            let resolved = model
                .schedule_table
                .map(|resolve| resolve())
                .unwrap_or_else(|| {
                    template::parse_table(model.schedules).expect("a model's schedule table")
                });
            if let Some(measured) = model.geometry_schedules {
                measured()
                    .into_iter()
                    .map(|(id, table)| {
                        assert!(
                            table
                                .rows()
                                .iter()
                                .all(|row| resolved.get(&row.key).is_some()),
                            "measured keys belong to the resolved table"
                        );
                        (model.name, Some(id), table)
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![(model.name, None, resolved)]
            }
        })
        .collect::<Vec<_>>();
    let served = template::lab_instances(kernel_lab::catalogs());
    let mut changes = Vec::new();
    for dir in results.split(':').filter(|d| !d.is_empty()) {
        let text = std::fs::read_to_string(Path::new(dir).join("rows.json")).expect("rows.json");
        let rows = Table::parse(&text).expect("canonical sweep rows");
        for row in rows.rows() {
            let owners = tables
                .iter()
                .filter(|(_, _, table)| table.get(&row.key).is_some())
                .count();
            assert_eq!(
                owners, 1,
                "a sweep key must have one measured geometry owner"
            );
            let (_, _, table) = tables
                .iter_mut()
                .find(|(_, _, table)| table.get(&row.key).is_some())
                .expect("a served key of a model's table");
            assert_eq!(rows.target, table.target, "{dir}: another target");
            let current = table.require(&row.key).expect("a served key");
            let old = ScheduleKnobs::from_row(current).expect("current row");
            // The build's own row checks: arithmetic, legality and a launch
            // plan that fits at capacity.
            let served = served
                .iter()
                .find(|i| template::key(&i.baseline) == row.key)
                .expect("a served entry for the key");
            let new =
                template::validate_row(&served.baseline, row).unwrap_or_else(|e| panic!("{e}"));
            if old != new {
                changes.push(format!(
                    "{}: {} -> {}",
                    row.key,
                    knobs_json(&old),
                    knobs_json(&new)
                ));
            }
            table.upsert(row.clone()).expect("valid row");
        }
    }
    let out = Path::new(&out);
    std::fs::create_dir_all(out).expect("create the table directory");
    let mut documents: std::collections::BTreeMap<&str, std::collections::BTreeMap<&str, &Table>> =
        Default::default();
    for (name, id, table) in &tables {
        if let Some(id) = id {
            documents
                .entry(name)
                .or_default()
                .insert(id.as_str(), table);
        } else {
            std::fs::write(out.join(format!("{name}.json")), table.render())
                .expect("write the merged table");
        }
    }
    for (name, document) in documents {
        let entries = document
            .iter()
            .map(|(id, table)| {
                format!(
                    "  {}: {}",
                    serde_json::to_string(id).unwrap(),
                    table.render().trim_end().replace('\n', "\n  ")
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        std::fs::write(
            out.join(format!("{name}.json")),
            format!("{{\n{entries}\n}}\n"),
        )
        .expect("write geometry-keyed schedules");
    }
    eprintln!("{} rows changed:\n{}", changes.len(), changes.join("\n"));
}

// ---------------------------------------------------------------------------
// Offline sweep of the packed ring rows (`CHOPINFER_TUNE_PACKED_*`).
//
// For each selected packed NVFP4 or MXFP8 decode row, every other packed
// ring or staged schedule `template::packed_candidates` admits is compiled
// outside the lab
// lock, checked bit for bit against the packed engine (full-bucket and
// half-bucket live counts, skewed and crowded routes), and the survivors
// timed A B A B against the row's current entry with the rotating banks
// and route sets of the packed timing: two screening repetitions each,
// then seven confirming repetitions for the three fastest. A candidate
// replaces the row only if its confirmation passes
// `kernel_build::tuning::best` (faster by more than the A-to-A noise plus
// the margin). The sweep writes the replaced rows as a canonical partial
// table (`rows.json`) and every measurement to `report.json`.

/// Screens and confirms one candidate: the current row's entry is A.
fn packed_abab(
    session: &Session,
    current: &runtime_cuda::GraphExec,
    candidate: &runtime_cuda::GraphExec,
    repetitions: usize,
) -> LabResult<(f32, f32, Comparison)> {
    const ROUNDS: usize = 9;
    let arm = |b: bool| time_graph(session, if b { candidate } else { current }, ROUNDS);
    for b in [false, true] {
        arm(b)?;
    }
    let (mut a, mut b, mut pairs, mut noise) = (Vec::new(), Vec::new(), Vec::new(), 0.0_f32);
    for _ in 0..repetitions.max(1) {
        let a1 = arm(false)?;
        let b1 = arm(true)?;
        let a2 = arm(false)?;
        let b2 = arm(true)?;
        noise = noise.max((a1 - a2).abs() / a1.min(a2));
        pairs.extend([b1 / a1, b2 / a2]);
        a.extend([a1, a2]);
        b.extend([b1, b2]);
    }
    Ok((
        median(&mut a),
        median(&mut b),
        Comparison {
            ratio: f64::from(median(&mut pairs)),
            noise: f64::from(noise),
        },
    ))
}

#[test]
#[ignore = "packed ring sweep; set CHOPINFER_TUNE_OUT and run on an exclusive GPU"]
fn grouped_moe_packed_ring_sweep() {
    let out = PathBuf::from(std::env::var_os("CHOPINFER_TUNE_OUT").expect("CHOPINFER_TUNE_OUT"));
    let margin: f64 = std::env::var("CHOPINFER_TUNE_MARGIN")
        .ok()
        .map_or(0.01, |m| {
            m.parse().expect("CHOPINFER_TUNE_MARGIN is a number")
        });
    // Ring symbols containing any listed substring, or every NVFP4 row.
    let filter = env_list("CHOPINFER_TUNE_PACKED_ROWS");
    let instances: Vec<PackedInstance> = model_qwen38_contract::sm120::moe::packed_lab_instances()
        .into_iter()
        .filter(|i| match &filter {
            Some(list) => list.iter().any(|f| i.template.symbol.contains(f.as_str())),
            None => i.template.query.weight == WeightFormat::Nvfp4,
        })
        .collect();
    assert!(
        !instances.is_empty(),
        "CHOPINFER_TUNE_PACKED_ROWS selects no packed row"
    );
    std::fs::create_dir_all(out.join("cubins")).expect("output directory");
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    let mut refused = Vec::new();
    let mut work = Vec::new();
    for instance in &instances {
        let mut candidates = Vec::new();
        for (ring, candidate) in template::packed_candidates(instance) {
            match candidate {
                Ok(operation) => candidates.push((ring, operation)),
                Err(reason) => refused.push((format!("{} {ring:?}", instance.key), reason)),
            }
        }
        let specs = candidates
            .iter()
            .map(|(_, op)| {
                kernels_sm120::moe::cubin_spec(
                    &op.symbol
                        .trim_end_matches("_v1")
                        .replacen(template::TUNE_PREFIX, "tune_", 1),
                    std::slice::from_ref(op),
                )
                .expect("candidate cubin spec")
            })
            .collect::<Vec<_>>();
        let built = compile_candidates(&toolchain, &out.join("cubins"), &specs);
        let mut compiled = Vec::new();
        for ((ring, op), spec) in candidates.into_iter().zip(&specs) {
            match &built[&spec.name] {
                Ok(c) => compiled.push((ring, op, c.path.clone(), c.registers)),
                Err(reason) => refused.push((op.symbol.clone(), reason.clone())),
            }
        }
        work.push((instance, compiled));
    }
    let mut table = Table::new(kernel_lab::catalogs().schedules().target);
    let mut report = Vec::new();
    gpu_test(16 << 30, |session| {
        for (index, (instance, compiled)) in work.iter().enumerate() {
            let g = packed_geometry(instance);
            let seed = mix(0x9ac4 ^ index as u64);
            let banks = (0..TIMING_BANKS)
                .map(|b| Bank::new(session, g, instance.profile, seed ^ (b as u64) << 40))
                .collect::<LabResult<Vec<_>>>()?;
            let current_case = rotating_case(session, instance, seed)?;
            let current = rotating_graph(session, &current_case, &banks, true)?;
            let capacity = current_case.capacity;
            // Bitwise: engine outputs of the current case at each check.
            let checks = [
                (capacity, Routing::Skewed),
                (capacity.div_ceil(2), Routing::Skewed),
                (capacity, Routing::Crowded),
            ];
            let owned = compiled
                .iter()
                .map(|(_, op, _, _)| PackedInstance {
                    template: op.clone(),
                    ..(*instance).clone()
                })
                .collect::<Vec<_>>();
            let mut rows = Vec::new();
            for ((ring, op, path, registers), owned) in compiled.iter().zip(&owned) {
                let shared = PackedCase::plan_of(op, g, capacity)?.dynamic_shared_bytes;
                let module = session.load_cubin(&std::fs::read(path).map_err(err)?)?;
                let function = module.function(&op.symbol).map_err(err)?;
                if shared > 48 * 1024 {
                    function
                        .set_max_dynamic_shared_memory(shared)
                        .map_err(err)?;
                }
                let case =
                    PackedCase::with_ring(session, owned, Entry { function }, TIMING_ROUTES)?;
                let mut bitwise = Ok(());
                for (n, &(live, routing)) in checks.iter().enumerate() {
                    let r = case.prepare(session, live, routing, mix(seed ^ 0xb17 ^ n as u64))?;
                    let engine = case.run(session, &banks[0], false, &r)?;
                    let ring_out = case.run(session, &banks[0], true, &r)?;
                    if engine != ring_out {
                        bitwise = Err(format!("live {live} {} differs", routing.name()));
                        break;
                    }
                }
                write_rotating_routes(session, &case, seed)?;
                let mut row = json!({
                    "symbol": op.symbol,
                    "schedule": template::packed_row(*ring).0,
                    "knobs": format!("{:?}", template::packed_row(*ring).1),
                    "registers": registers,
                    "bitwise": bitwise.is_ok(),
                });
                if let Err(reason) = bitwise {
                    row["bitwise_failure"] = json!(reason);
                    rows.push((row, None, *ring));
                    continue;
                }
                let graph = rotating_graph(session, &case, &banks, true)?;
                let (a, b, c) = packed_abab(session, &current, &graph, 2)?;
                row["screen"] =
                    json!({"current_us": a, "candidate_us": b, "ratio": c.ratio, "noise": c.noise});
                rows.push((row, Some((case, graph, c.ratio)), *ring));
            }
            // Confirm the three fastest screened candidates.
            let mut order = (0..rows.len())
                .filter(|&i| rows[i].1.is_some())
                .collect::<Vec<_>>();
            order.sort_by(|&x, &y| {
                let r = |i: usize| rows[i].1.as_ref().unwrap().2;
                r(x).total_cmp(&r(y))
            });
            let mut confirmed = Vec::new();
            for &i in order.iter().take(3) {
                let (_, graph, _) = rows[i].1.as_ref().unwrap();
                let (a, b, c) = packed_abab(session, &current, graph, 7)?;
                rows[i].0["confirm"] =
                    json!({"current_us": a, "candidate_us": b, "ratio": c.ratio, "noise": c.noise});
                let (schedule, knobs) = template::packed_row(rows[i].2);
                confirmed.push((schedule, knobs, c));
            }
            let chosen = tuning::best(confirmed.iter().map(|(_, k, c)| (k, *c)), margin).cloned();
            if let Some(knobs) = &chosen {
                let schedule = confirmed
                    .iter()
                    .find(|(_, k, _)| k == knobs)
                    .expect("a confirmed candidate")
                    .0;
                table
                    .upsert(kernel_build::tuning::Row {
                        key: instance.key.clone(),
                        schedule: schedule.into(),
                        knobs: knobs.clone(),
                    })
                    .expect("valid packed row");
            }
            eprintln!(
                "{:<56} {} candidates, chosen {:?}",
                instance.template.symbol,
                rows.len(),
                chosen
            );
            report.push(json!({
                "row": instance.key.to_string(),
                "current": format!("{:?}", instance.template.knobs),
                "chosen": chosen.map(|k| format!("{k:?}")),
                "candidates": rows.into_iter().map(|r| r.0).collect::<Vec<_>>(),
            }));
        }
        Ok(())
    })
    .expect("packed ring sweep");
    std::fs::write(out.join("rows.json"), table.render()).expect("write rows.json");
    write_json(
        &out.join("report.json"),
        &json!({"margin": margin, "timing_routes": timing_routes_name(), "rows": report, "refused": refused.len(),
                "refusals": refused.iter().map(|(k, r)| json!([k, r])).collect::<Vec<_>>()}),
    )
    .expect("write report.json");
}
