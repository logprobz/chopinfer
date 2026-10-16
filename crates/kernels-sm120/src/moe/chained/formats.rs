//! A dense chained projection at the served shapes ([`TP4_288_TOP8`]) in a
//! format the served catalog ([`super::catalog`]) does not hold, built into
//! its own cubin `{PREFIX}_{format}` so the served cubins stay unchanged:
//! the NVFP4 dense MLP gate/up at the first dense intermediate, with its
//! gate and up globals, for a checkpoint whose dense MLP layers are NVFP4.
//! Its down projection is a row-split partial written in F32, which the
//! chained dense down (BF16 output) does not write, so it is not here. The
//! routed experts' BF16 projections, for checkpoints whose
//! multi-token-prediction experts are BF16, are [`super::split::catalog`]'s.
//!
//! [`lab_projections`] are what kernel-lab builds beside it and nothing
//! serves: the NVFP4 dense down (the MLP oracle's second half).

use super::{
    GatedMlpProjection, Projection, Refusal, ServedShapes, TP4_288_TOP8, projection_cubin_specs,
};
use kernel_build::CubinSpec;
use kernel_core::contract::WeightFormat;

/// The cubin name prefix of [`catalog`].
pub const PREFIX: &str = "chop_moe_chain_e288l72k8_formats";
/// The cubin name prefix of [`lab_projections`].
pub const LAB_PREFIX: &str = "chop_moe_chain_lab_formats";

/// The decode (row tile 16) and prefill (row tile 64) phases.
fn phases(shapes: &ServedShapes) -> [(u32, u32); 2] {
    [(16, shapes.decode_tokens), (64, shapes.prefill_tokens)]
}

fn projections(
    shapes: &ServedShapes,
    weight: WeightFormat,
    which: &[GatedMlpProjection],
    grouped: bool,
    intermediate: u32,
) -> Result<Vec<Projection>, Refusal> {
    let mut out = Vec::new();
    for &projection in which {
        for (row_tile, tokens) in phases(shapes) {
            let query = shapes.query(weight, projection, grouped, intermediate, tokens);
            out.push(Projection::new(query, row_tile, tokens)?);
        }
    }
    Ok(out)
}

/// The NVFP4 dense gate/up at the decode and prefill capacity.
pub fn catalog() -> Vec<Projection> {
    let shapes = TP4_288_TOP8;
    projections(
        &shapes,
        WeightFormat::Nvfp4,
        &[GatedMlpProjection::GateUp],
        false,
        shapes.dense_intermediates[0],
    )
    .expect("the NVFP4 dense gate/up is supported")
}

/// Lab-only projections: the NVFP4 dense down at the first dense
/// intermediate, at the decode and prefill capacity.
pub fn lab_projections() -> Vec<Projection> {
    let shapes = TP4_288_TOP8;
    projections(
        &shapes,
        WeightFormat::Nvfp4,
        &[GatedMlpProjection::Down],
        false,
        shapes.dense_intermediates[0],
    )
    .expect("the NVFP4 dense down is supported")
}

/// The catalog's cubins, one per format.
pub fn cubin_specs() -> Vec<CubinSpec> {
    projection_cubin_specs(PREFIX, &catalog())
}

/// The lab projections' cubins, one per format.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    projection_cubin_specs(LAB_PREFIX, &lab_projections())
}

/// Admission of NVFP4 local scale codes: UE4M3 has no sign bit and 0x7f is
/// NaN, so every code is at most 0x7e. The engine decodes the byte as E4M3,
/// which would read a code with bit 7 set as a negative scale.
pub fn admit_ue4m3_scales(codes: &[u8]) -> Result<(), Refusal> {
    if codes.iter().any(|&c| c > 0x7e) {
        return Err(Refusal(
            "NVFP4 UE4M3 scale codes must be finite and unsigned (at most 0x7e)",
        ));
    }
    Ok(())
}

/// Admission of an MXFP8 payload: E4M3 codes `[rows, K]` with one UE8M0
/// scale per 32 K, `[rows, K / 32]`, row-major. Every code is finite (no
/// `0x7f` or `0xff`), no scale is byte 255, and every scaled operand the
/// engine forms is a finite BF16 equal to code times scale. Every finite
/// E4M3 code is exact at scale bytes 3 to 246, so only codes under other
/// scales are decoded (`docs/numerics.md`); the engine traps on those it
/// cannot form exactly, and this refuses them before dispatch.
pub fn admit_mxfp8_payload(codes: &[u8], scales: &[u8]) -> Result<(), Refusal> {
    use kernel_core::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64, f32_to_bf16};
    if scales.len().checked_mul(32) != Some(codes.len()) {
        return Err(Refusal("MXFP8 needs one UE8M0 scale per 32 E4M3 codes"));
    }
    if scales.contains(&0xff) {
        return Err(Refusal("MXFP8 UE8M0 scale byte 255 is not finite"));
    }
    for (group, &scale) in codes.chunks_exact(32).zip(scales) {
        if group.iter().any(|&c| c & 0x7f == 0x7f) {
            return Err(Refusal("MXFP8 E4M3 weight codes must be finite"));
        }
        if (3..=246).contains(&scale) {
            continue;
        }
        let factor = decode_ue8m0_f64(scale);
        for &code in group {
            let value = SmallFloat::E4M3.decode_f64(code) * factor;
            let operand = f64::from(bf16_to_f32(f32_to_bf16(value as f32)));
            if !operand.is_finite() || operand != value {
                return Err(Refusal(
                    "MXFP8 scaled weights must be finite and exact in BF16",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{Weights, catalog as served, cubin_specs as served_cubins};
    use super::*;

    #[test]
    fn formats_extend_the_served_catalog_without_changing_it() {
        let projections = catalog();
        // NVFP4 dense gate/up: two phases.
        assert_eq!(projections.len(), 2);
        let served = served();
        let served_symbols = served.iter().map(|p| &p.symbol).collect::<Vec<_>>();
        for p in projections.iter().chain(&lab_projections()) {
            assert!(!served_symbols.contains(&&p.symbol), "{}", p.symbol);
        }
        let names = served_cubins()
            .into_iter()
            .map(|s| s.name)
            .collect::<Vec<_>>();
        for spec in cubin_specs().iter().chain(&lab_cubin_specs()) {
            assert!(!names.contains(&spec.name), "{}", spec.name);
        }
        let symbols = projections
            .iter()
            .map(|p| p.symbol.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            symbols,
            [
                "chop_moe_chain_gateup_nvfp4_dense_n6144_k4096_r16_clamp4120000041200000_v1",
                "chop_moe_chain_gateup_nvfp4_dense_n6144_k4096_r64_clamp4120000041200000_v1",
            ]
        );
        // The dense gate/up reads a gate and an up global.
        let dense = &projections[0];
        assert_eq!(dense.weights(), Weights::Nvfp4);
        assert!(dense.entry().body.contains(
            "Weights::nvfp4,chop::moe::chained::Projection::gate_up,1,1,3072,4096,16,false"
        ));
        assert!(
            dense
                .entry()
                .body
                .contains("SwigluClamp<true,0x41200000u,0x41200000u>")
        );
        assert_eq!(dense.plan(32).unwrap().grid, [2, 48, 1]);
        assert_eq!(projections[1].plan(8192).unwrap().grid, [128, 48, 1]);
        let specs = cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            ["chop_moe_chain_e288l72k8_formats_nvfp4"]
        );
    }

    #[test]
    fn lab_projections_cover_the_dense_down() {
        let lab = lab_projections();
        let symbols = lab.iter().map(|p| p.symbol.as_str()).collect::<Vec<_>>();
        assert_eq!(
            symbols,
            [
                "chop_moe_chain_down_nvfp4_dense_n4096_k3072_r16_v1",
                "chop_moe_chain_down_nvfp4_dense_n4096_k3072_r64_v1",
            ]
        );
        assert_eq!(
            lab_cubin_specs()
                .iter()
                .map(|s| (s.name.as_str(), s.entries.len()))
                .collect::<Vec<_>>(),
            [("chop_moe_chain_lab_formats_nvfp4", 2)]
        );
    }

    #[test]
    fn ue4m3_admission_refuses_signed_and_nan_codes() {
        admit_ue4m3_scales(&(0..=0x7e).collect::<Vec<u8>>()).unwrap();
        for bad in [0x7f, 0x80, 0xb8, 0xff] {
            assert!(admit_ue4m3_scales(&[0x38, bad]).is_err(), "{bad:#x}");
        }
    }

    #[test]
    fn mxfp8_admission_refuses_non_finite_and_inexact_operands() {
        use kernel_core::codec::{SmallFloat, decode_ue8m0_f64};
        // Every finite code under every scale byte from 3 to 246 is admitted.
        let finite = (0..=255_u8)
            .filter(|c| c & 0x7f != 0x7f)
            .collect::<Vec<_>>();
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        for scale in 3..=246 {
            for chunk in finite.chunks(32) {
                let mut group = chunk.to_vec();
                group.resize(32, 0);
                codes.extend(group);
                scales.push(scale);
            }
        }
        admit_mxfp8_payload(&codes, &scales).unwrap();
        // Outside that interval each (code, scale) pair is admitted exactly
        // when its product is zero, or within BF16's largest finite value
        // (255 * 2^120) and a whole multiple of its smallest subnormal
        // (2^-133): E4M3 has four significant bits, BF16 eight.
        let (mut admitted, mut refused) = (0, 0);
        for scale in (0..3).chain(247..255) {
            for &code in &finite {
                let value = SmallFloat::E4M3.decode_f64(code) * decode_ue8m0_f64(scale);
                let exact = value == 0.0
                    || (value.abs() <= 255.0 * 2_f64.powi(120)
                        && (value / 2_f64.powi(-133)).fract() == 0.0);
                let mut group = [0_u8; 32];
                group[5] = code;
                let ok = admit_mxfp8_payload(&group, &[scale]).is_ok();
                assert_eq!(ok, exact, "code {code:#x} scale {scale}");
                if ok {
                    admitted += 1;
                } else {
                    refused += 1;
                }
            }
        }
        assert!(admitted > 0 && refused > 0);
        // The edges: the smallest subnormal code under scale 0 is 2^-136,
        // below BF16's 2^-133; 448 under scale 254 overflows.
        assert!(admit_mxfp8_payload(&[1; 32], &[0]).is_err());
        assert!(admit_mxfp8_payload(&[0x7e; 32], &[254]).is_err());
        assert!(admit_mxfp8_payload(&[0x38; 32], &[254]).is_ok());
        // NaN codes at an ordinary scale, scale byte 255 and a short grid.
        for nan in [0x7f, 0xff] {
            let mut group = [0x38_u8; 32];
            group[31] = nan;
            assert!(admit_mxfp8_payload(&group, &[127]).is_err(), "{nan:#x}");
        }
        assert!(admit_mxfp8_payload(&[0; 32], &[255]).is_err());
        assert!(admit_mxfp8_payload(&[0; 33], &[127]).is_err());
        assert!(admit_mxfp8_payload(&[0; 32], &[127, 127]).is_err());
    }
}
