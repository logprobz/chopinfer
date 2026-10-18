//! Cases of the lagged Sinkhorn hyper-connection ops (the lagged
//! placement's mix-collapse, the F32 expand and the sigmoid-weighted head),
//! with the SM120 selection of the lagged engine's lab entries
//! (`sinkhorn_hc::lagged`).
//!
//! The lagged mix-collapse collapses with the given F32 weights, so its
//! collapsed row is exact bits (`sinkhorn_hyper_connection_oracle::
//! collapse`); its F32 `pre` and `post` compare within enclosures of every
//! kernel result (`lagged_mix_collapse`) and `comb` within its range
//! `[0, 1]`, since twenty Sinkhorn iterations have no useful finite
//! enclosure. The F32 expand and the weighted head are exact functions of
//! their operands and compare as bits.

use super::super::{OpCase, Required, Spec, bf16_bytes, bf16_values};
use crate::LabResult;
use crate::conformance::{Expected, Images, Rows};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::DType;
use kernel_core::sinkhorn_hyper_connection::{
    HyperConnectionPlacement, LaggedMixCollapseBuffers, MixScratch, MixerWeights,
    SinkhornHyperConnectionQuery, StreamExpandBuffers, StreamHeadMode, WeightedStreamHeadBuffers,
};
use kernel_core::sinkhorn_hyper_connection_oracle::{self as hc, MixWeights};
use kernels_sm120::backend::Sm120;
use kernels_sm120::sinkhorn_hc::lagged::{self, LAB};

/// The token rows every lagged Sinkhorn case serves.
pub const CAPACITY: u32 = 16;

fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// F32 values of `count` in `[low, high)` from `seed`, off the BF16 grid:
/// the kit's generator's magnitudes (0.25 to 2) mapped onto the range, plus
/// a small offset that keeps the low mantissa bits set.
fn ranged_f32(seed: u64, count: usize, low: f32, high: f32) -> Vec<f32> {
    bf16_values(seed, count)
        .into_iter()
        .enumerate()
        .map(|(i, b)| {
            let unit = (bf16_to_f32(b).abs() - 0.25) / 1.75;
            low + (high - low) * unit * (1.0 - 2_f32.powi(-12)) + (i % 7) as f32 * 2_f32.powi(-20)
        })
        .collect()
}

/// The case of a lagged mix-collapse over `capacity` rows: BF16 streams
/// from the kit's generator, an F32 mixer `fn` scaled by 2^-7 so the logits
/// stay moderate, F32 `base` and `scale`, F32 collapse weights in `[1/16,
/// 1.25)`; the collapsed row as exact bits, `pre`, `post` and `comb` within
/// the contract's gate.
pub fn mix_collapse_case<'a>(
    op: SinkhornHyperConnectionQuery,
    capacity: u32,
    prepared: Prepared<LaggedMixCollapseBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let SinkhornHyperConnectionQuery::MixCollapse { mix, .. } = op else {
        return Err("a lagged mix-collapse case needs a mix-collapse".into());
    };
    if mix.placement != HyperConnectionPlacement::LaggedSinglePass || mix.fn_dtype != DType::F32 {
        return Err("the kit's lagged mix-collapse case covers a lagged mixer with F32 fn".into());
    }
    if let Some((path, _)) = prepared.info().bytes.first() {
        return Err(format!(
            "the kit's lagged mix-collapse case binds no {path}"
        ));
    }
    let (n, h, mixes) = (
        mix.streams as usize,
        mix.width as usize,
        mix.mixes() as usize,
    );
    let rows = capacity as usize;
    let projection = bf16_values(41, mixes * n * h)
        .into_iter()
        .map(|b| bf16_to_f32(b) / 128.0 + 2_f32.powi(-22))
        .collect::<Vec<_>>();
    let base = (0..mixes)
        .map(|m| (m % 5) as f32 * 0.25 - 0.5)
        .collect::<Vec<_>>();
    let scale = vec![0.5_f32, 0.25, 1.0];
    let roles = LaggedMixCollapseBuffers {
        x: Spec::input(
            DType::Bf16,
            (Rows::Live, n * h),
            bf16_bytes(&bf16_values(42, rows * n * h)),
        ),
        mixer: MixerWeights {
            fn_: Spec::input(
                DType::F32,
                (Rows::Fixed(mixes), n * h),
                f32_bytes(&projection),
            ),
            base: Spec::input(DType::F32, (Rows::Fixed(1), mixes), f32_bytes(&base)),
            scale: Spec::input(DType::F32, (Rows::Fixed(1), 3), f32_bytes(&scale)),
        },
        pre_in: Spec::input(
            DType::F32,
            (Rows::Live, n),
            f32_bytes(&ranged_f32(43, rows * n, 0.0625, 1.25)),
        ),
        collapsed: Spec::output(DType::Bf16, (Rows::Live, h)),
        pre: Spec::output(DType::F32, (Rows::Live, n)),
        post: Spec::output(DType::F32, (Rows::Live, n)),
        comb: Spec::output(DType::F32, (Rows::Live, n * n)),
        scratch: MixScratch {
            partials: None,
            tickets: None,
        },
        live_rows: Spec::live(),
    };
    let query = SinkhornHyperConnectionQuery::MixCollapse { mix, capacity };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let bits = images.u16s("x")?;
        let x = widen(&bits);
        let pre_in = images.f32s("pre_in")?;
        let widen_f32 = |v: Vec<f32>| v.into_iter().map(f64::from).collect::<Vec<_>>();
        let projection = widen_f32(images.f32s("mixer.fn_")?);
        let base = widen_f32(images.f32s("mixer.base")?);
        let scale = widen_f32(images.f32s("mixer.scale")?);
        let weights = MixWeights {
            projection: &projection,
            base: &base,
            scale: &scale,
        };
        let output =
            hc::lagged_mix_collapse(query, &x, &pre_in, weights).map_err(|e| e.0.to_string())?;
        let collapsed = hc::collapse(query, &bits, &pre_in).map_err(|e| e.0.to_string())?;
        Ok(vec![
            ("collapsed".into(), Expected::Bits(bf16_bytes(&collapsed))),
            ("pre".into(), Expected::Values(output.pre)),
            ("post".into(), Expected::Values(output.post)),
            ("comb".into(), Expected::Values(output.comb)),
        ])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The case of an F32 expand over `capacity` rows: BF16 streams and
/// sublayer outputs from the kit's generator, F32 `post` in `[1/16, 2)` and
/// `comb` in `[1/16, 1)` off the BF16 grid, and the contract's F32
/// arithmetic as exact bits.
pub fn expand_case<'a>(
    op: SinkhornHyperConnectionQuery,
    capacity: u32,
    prepared: Prepared<StreamExpandBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let SinkhornHyperConnectionQuery::ExpandF32 { streams, width, .. } = op else {
        return Err("an F32 expand case needs an F32 expand".into());
    };
    let (n, h, rows) = (streams as usize, width as usize, capacity as usize);
    let roles = StreamExpandBuffers {
        x: Spec::input(
            DType::Bf16,
            (Rows::Live, n * h),
            bf16_bytes(&bf16_values(44, rows * n * h)),
        ),
        y: Spec::input(
            DType::Bf16,
            (Rows::Live, h),
            bf16_bytes(&bf16_values(45, rows * h)),
        ),
        post: Spec::input(
            DType::F32,
            (Rows::Live, n),
            f32_bytes(&ranged_f32(46, rows * n, 0.0625, 2.0)),
        ),
        comb: Spec::input(
            DType::F32,
            (Rows::Live, n * n),
            f32_bytes(&ranged_f32(47, rows * n * n, 0.0625, 1.0)),
        ),
        output: Spec::output(DType::Bf16, (Rows::Live, n * h)),
        live_rows: Spec::live(),
    };
    let query = SinkhornHyperConnectionQuery::ExpandF32 {
        streams,
        width,
        capacity,
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let output = hc::expand_f32(
            query,
            &images.u16s("x")?,
            &images.u16s("y")?,
            &images.f32s("post")?,
            &images.f32s("comb")?,
        )
        .map_err(|e| e.0.to_string())?;
        Ok(vec![("output".into(), Expected::Bits(bf16_bytes(&output)))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The case of a sigmoid-weighted head over `capacity` rows: BF16 streams
/// from the kit's generator, F32 weights in `[1/16, 1.25)` off the BF16
/// grid, and the contract's F32 collapse as exact bits.
pub fn head_case<'a>(
    op: SinkhornHyperConnectionQuery,
    capacity: u32,
    prepared: Prepared<WeightedStreamHeadBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let SinkhornHyperConnectionQuery::Head {
        streams,
        width,
        mode,
        ..
    } = op
    else {
        return Err("a weighted head case needs a sigmoid-weighted head".into());
    };
    let dtype = match mode {
        StreamHeadMode::SigmoidWeighted => DType::Bf16,
        StreamHeadMode::SigmoidWeightedF32 => DType::F32,
        StreamHeadMode::MeanOfStreams => return Err("a weighted head case needs weights".into()),
    };
    let (n, h, rows) = (streams as usize, width as usize, capacity as usize);
    let roles = WeightedStreamHeadBuffers {
        x: Spec::input(
            DType::Bf16,
            (Rows::Live, n * h),
            bf16_bytes(&bf16_values(48, rows * n * h)),
        ),
        pre: Spec::input(
            DType::F32,
            (Rows::Live, n),
            f32_bytes(&ranged_f32(49, rows * n, 0.0625, 1.25)),
        ),
        output: Spec::output(dtype, (Rows::Live, h)),
        live_rows: Spec::live(),
    };
    let query = SinkhornHyperConnectionQuery::Head {
        streams,
        width,
        mode,
        capacity,
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let x = images.u16s("x")?;
        let pre = images.f32s("pre")?;
        let output = match mode {
            StreamHeadMode::SigmoidWeighted => {
                bf16_bytes(&hc::weighted_head(query, &x, &pre).map_err(|e| e.0.to_string())?)
            }
            StreamHeadMode::SigmoidWeightedF32 => {
                f32_bytes(&hc::weighted_head_f32(query, &x, &pre).map_err(|e| e.0.to_string())?)
            }
            StreamHeadMode::MeanOfStreams => unreachable!("refused above"),
        };
        Ok(vec![("output".into(), Expected::Bits(output))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// SM120 as the lab builds the lagged engine: its served entries.
pub fn sm120() -> Sm120 {
    Sm120::new().with_sinkhorn_lagged(lagged::catalog())
}

/// The lab mixer's lagged mix-collapse, F32 expand and weighted head at
/// [`CAPACITY`] rows, each required.
pub fn required<'a>() -> Vec<Required<'a>> {
    let need = Need::rows(CAPACITY);
    let mix = LAB.mix;
    let collapse = SinkhornHyperConnectionQuery::MixCollapse {
        mix,
        capacity: LAB.capacity,
    };
    let expand = SinkhornHyperConnectionQuery::ExpandF32 {
        streams: mix.streams,
        width: mix.width,
        capacity: LAB.capacity,
    };
    let head = SinkhornHyperConnectionQuery::Head {
        streams: mix.streams,
        width: mix.width,
        mode: StreamHeadMode::SigmoidWeighted,
        capacity: LAB.capacity,
    };
    let head_f32 = lagged::head_f32_entry(lagged::LAB_PROFILE, LAB).query;
    let shape = format!("{} streams width {}", mix.streams, mix.width);
    vec![
        Required::new(
            format!("sinkhorn lagged mix-collapse {shape}"),
            need,
            move |backend, need| backend.sinkhorn_lagged_mix_collapse(&collapse, need),
            move |prepared| mix_collapse_case(collapse, need.capacity, prepared),
        )
        .required(),
        Required::new(
            format!("sinkhorn F32 expand {shape}"),
            need,
            move |backend, need| backend.sinkhorn_expand(&expand, need),
            move |prepared| expand_case(expand, need.capacity, prepared),
        )
        .required(),
        Required::new(
            format!("sinkhorn weighted head {shape}"),
            need,
            move |backend, need| backend.sinkhorn_weighted_head(&head, need),
            move |prepared| head_case(head, need.capacity, prepared),
        )
        .required(),
        Required::new(
            format!("sinkhorn F32 weighted head {shape}"),
            need,
            move |backend, need| backend.sinkhorn_weighted_head(&head_f32, need),
            move |prepared| head_case(head_f32, need.capacity, prepared),
        )
        .required(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    use kernel_core::codec::f32_to_bf16;

    #[test]
    fn f32_head_case_has_rounding_and_stream_order_witnesses() {
        let query = lagged::head_f32_entry(lagged::LAB_PROFILE, LAB).query;
        let prepared = sm120()
            .sinkhorn_weighted_head(&query, Need::rows(8))
            .unwrap();
        let case = head_case(query, 8, prepared).unwrap();
        case.validate().unwrap();
        // Three live rows selected from a larger capacity retain the same
        // row data used by the device case and its property checks.
        let images = case.select(&case.images(), &[0, 3, 7]).unwrap();
        let (n, h) = (LAB.mix.streams as usize, LAB.mix.width as usize);
        let x = &images.u16s("x").unwrap()[..3 * n * h];
        let pre = &images.f32s("pre").unwrap()[..3 * n];
        let exact = hc::weighted_head_f32(query, x, pre).unwrap();
        let expected = case.oracle_bits(&images, 3).unwrap();
        case.judge_outputs(&images, 3, &expected).unwrap();
        let reversed = (0..exact.len())
            .map(|e| {
                let (row, col) = (e / h, e % h);
                (0..n).rev().fold(0.0_f32, |sum, i| {
                    pre[row * n + i].mul_add(bf16_to_f32(x[(row * n + i) * h + col]), sum)
                })
            })
            .collect::<Vec<_>>();
        let rounded = exact
            .iter()
            .map(|&v| bf16_to_f32(f32_to_bf16(v)))
            .collect::<Vec<_>>();
        for (name, mutant) in [("round_bf16", rounded), ("reverse_streams", reversed)] {
            let differing = mutant
                .iter()
                .zip(&exact)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert!(differing > 0, "{name} escaped the head case");
            let mut outputs = expected.clone();
            outputs.get_mut("output").unwrap()[..mutant.len() * 4]
                .copy_from_slice(&f32_bytes(&mutant));
            let failure = case.judge_outputs(&images, 3, &outputs).unwrap_err();
            assert!(failure.contains("output byte"), "{failure}");
            eprintln!(
                "F32 head {name}: {differing} of {} elements differ",
                exact.len()
            );
        }
        let reference = hc::weighted_head_reference(query, &widen(x), pre).unwrap();
        kernel_core::oracle::check_numeric(
            &exact.iter().copied().map(f64::from).collect::<Vec<_>>(),
            &reference,
            DType::F32,
        )
        .unwrap();
    }

    #[test]
    fn f32_inputs_stay_in_range_and_off_the_bf16_grid() {
        let values = ranged_f32(1, 512, 0.0625, 2.0);
        assert!(values.iter().all(|&v| (0.0625..2.0).contains(&v)));
        assert!(values.iter().any(|&v| bf16_to_f32(f32_to_bf16(v)) != v));
    }
}
