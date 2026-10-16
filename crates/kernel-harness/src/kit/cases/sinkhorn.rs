//! Cases of the Sinkhorn hyper-connection ops a rank runs alone: the
//! mix-collapse before a sublayer, alone or with the sublayer's input norm,
//! the expand of its output into the streams and the stream head, with the
//! SM120 selections the lab's cubins hold (the one-CTA engine, its split
//! mix-collapse, its tiled mix-collapse entries at the row counts SM120's
//! schedule tiles, and the split fused with the input norm's tree).
//!
//! The expand and head are exact functions of their BF16 operands and
//! compare as bits. The mix-collapse compares within the contract's gate
//! (`sinkhorn_hyper_connection_oracle::mix_collapse`): `collapsed` and
//! `post` within enclosures of every kernel result, `comb` only within its
//! range `[0, 1]`, since twenty Sinkhorn iterations have no useful finite
//! enclosure. The mix-collapse with its input norm saturates the collapse
//! weights (`pre` = `1 + eps` or `eps`, whatever the logits), so its
//! collapsed row is exact bits (`sinkhorn_hyper_connection_oracle::
//! collapse`) and so is its tree norm (`input_norm`); `post` and `comb`
//! keep the mix-collapse's gate. The fused comm epilogue
//! (`AllReduceExpandMixNorm`) launches collectively and is not a one-rank
//! case.

use super::super::{OpCase, Required, Spec, bf16_bytes, bf16_values};
use crate::LabResult;
use crate::conformance::{Expected, Images, Rows};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::sinkhorn_hyper_connection::{
    MixCollapseBuffers, MixCollapseNormBuffers, MixScratch, MixerWeights,
    SinkhornHyperConnectionQuery, StreamExpandBuffers, StreamHeadBuffers, StreamHeadMode,
};
use kernel_core::sinkhorn_hyper_connection_oracle::{self as hc, MixWeights};
use kernels_sm120::backend::Sm120;
use kernels_sm120::sinkhorn_hc::{self, SERVED, SERVED_PROFILE, mix_norm_tree, split, tiled};
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// The token rows every Sinkhorn case serves.
pub const CAPACITY: u32 = 16;

fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The mix-collapse oracle of the images' streams and mixer, computed once
/// per distinct images: a case asks for it at every live count and run over
/// the same images, and its cost grows with the rows (some 600 million
/// terms at 1505).
#[derive(Default)]
struct MixOracle(RefCell<Option<(u64, hc::MixCollapseOutput)>>);

impl MixOracle {
    fn get(
        &self,
        query: SinkhornHyperConnectionQuery,
        images: &Images,
    ) -> LabResult<hc::MixCollapseOutput> {
        let names = ["x", "mixer.fn_", "mixer.base", "mixer.scale"];
        let mut hasher = DefaultHasher::new();
        for name in names {
            images.bytes(name)?.hash(&mut hasher);
        }
        let key = hasher.finish();
        if let Some((cached, output)) = &*self.0.borrow()
            && *cached == key
        {
            return Ok(output.clone());
        }
        let x = widen(&images.u16s("x")?);
        let projection = widen(&images.u16s("mixer.fn_")?);
        let base = images
            .f32s("mixer.base")?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        let scale = images
            .f32s("mixer.scale")?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        let weights = MixWeights {
            projection: &projection,
            base: &base,
            scale: &scale,
        };
        let output = hc::mix_collapse(query, &x, weights).map_err(|e| e.0.to_string())?;
        *self.0.borrow_mut() = Some((key, output.clone()));
        Ok(output)
    }
}

/// BF16 values of `count` in `[low, high)` from `seed`: the kit's
/// generator's magnitudes (0.25 to 2) mapped onto the range.
fn ranged(seed: u64, count: usize, low: f32, high: f32) -> Vec<u16> {
    bf16_values(seed, count)
        .into_iter()
        .map(|b| {
            let unit = (bf16_to_f32(b).abs() - 0.25) / 1.75;
            f32_to_bf16(low + (high - low) * unit)
        })
        .collect()
}

/// The case of a mix-collapse over `capacity` rows: BF16 streams from the
/// kit's generator, a BF16 mixer `fn` scaled by 2^-7 so the logits stay
/// moderate, F32 `base` and `scale`, the split scratch the op sizes
/// (partials, and tickets zeroed, which every launch must leave zero), and
/// the contract's gate over `collapsed`, `post` and `comb`.
pub fn mix_collapse_case<'a>(
    op: SinkhornHyperConnectionQuery,
    capacity: u32,
    prepared: Prepared<MixCollapseBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let SinkhornHyperConnectionQuery::MixCollapse { mix, .. } = op else {
        return Err("a mix-collapse case needs a mix-collapse".into());
    };
    if mix.fn_dtype != DType::Bf16 {
        return Err("the kit's mix-collapse case covers a BF16 fn".into());
    }
    let info = prepared.info().clone();
    if let Some((path, _)) = info
        .bytes
        .iter()
        .find(|(path, _)| !matches!(path.as_str(), "scratch.partials" | "scratch.tickets"))
    {
        return Err(format!("the kit's mix-collapse case binds no {path}"));
    }
    let (n, h, mixes) = (
        mix.streams as usize,
        mix.width as usize,
        mix.mixes() as usize,
    );
    let rows = capacity as usize;
    let projection = bf16_values(31, mixes * n * h)
        .into_iter()
        .map(|b| f32_to_bf16(bf16_to_f32(b) / 128.0))
        .collect::<Vec<_>>();
    let base = (0..mixes)
        .map(|m| (m % 5) as f32 * 0.25 - 0.5)
        .collect::<Vec<_>>();
    let scale = vec![0.5_f32, 0.25, 1.0];
    let roles = MixCollapseBuffers {
        x: Spec::input(
            DType::Bf16,
            (Rows::Live, n * h),
            bf16_bytes(&bf16_values(32, rows * n * h)),
        ),
        mixer: MixerWeights {
            fn_: Spec::input(
                DType::Bf16,
                (Rows::Fixed(mixes), n * h),
                bf16_bytes(&projection),
            ),
            base: Spec::input(DType::F32, (Rows::Fixed(1), mixes), f32_bytes(&base)),
            scale: Spec::input(DType::F32, (Rows::Fixed(1), 3), f32_bytes(&scale)),
        },
        collapsed: Spec::output(DType::Bf16, (Rows::Live, h)),
        post: Spec::output(DType::Bf16, (Rows::Live, n)),
        comb: Spec::output(DType::Bf16, (Rows::Live, n * n)),
        scratch: MixScratch {
            partials: info
                .bytes_of("scratch.partials")
                .map(|bytes| Spec::scratch(bytes as usize)),
            tickets: info
                .bytes_of("scratch.tickets")
                .map(|bytes| Spec::zeroed(bytes as usize)),
        },
        live_rows: Spec::live(),
    };
    let query = SinkhornHyperConnectionQuery::MixCollapse { mix, capacity };
    let mix_oracle = MixOracle::default();
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let output = mix_oracle.get(query, images)?;
        Ok(vec![
            ("collapsed".into(), Expected::Values(output.collapsed)),
            ("post".into(), Expected::Values(output.post)),
            ("comb".into(), Expected::Values(output.comb)),
        ])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The case of a mix-collapse and input norm over `capacity` rows: the
/// mix-collapse case's streams, `fn` and `scale` but `scale[0]` zero, so
/// each collapse weight is `sigmoid(base[i]) + eps`, with `base` 64 on even
/// streams (`pre` = `1 + eps`) and -200 on odd ones (`pre` = `eps`); a BF16
/// norm weight from the kit's generator; and the split scratch the op
/// sizes. The collapsed and normed rows compare as bits, `post` and `comb`
/// within the mix-collapse's gate.
pub fn mix_collapse_norm_case<'a>(
    op: SinkhornHyperConnectionQuery,
    capacity: u32,
    prepared: Prepared<MixCollapseNormBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let SinkhornHyperConnectionQuery::MixCollapseNorm {
        mix,
        rmsnorm_eps_bits,
        ..
    } = op
    else {
        return Err("a mix-collapse and norm case needs a mix-collapse and norm".into());
    };
    if mix.fn_dtype != DType::Bf16 {
        return Err("the kit's mix-collapse and norm case covers a BF16 fn".into());
    }
    let info = prepared.info().clone();
    if let Some((path, _)) = info
        .bytes
        .iter()
        .find(|(path, _)| !matches!(path.as_str(), "scratch.partials" | "scratch.tickets"))
    {
        return Err(format!(
            "the kit's mix-collapse and norm case binds no {path}"
        ));
    }
    let (n, h, mixes) = (
        mix.streams as usize,
        mix.width as usize,
        mix.mixes() as usize,
    );
    let rows = capacity as usize;
    let projection = bf16_values(31, mixes * n * h)
        .into_iter()
        .map(|b| f32_to_bf16(bf16_to_f32(b) / 128.0))
        .collect::<Vec<_>>();
    let base = (0..mixes)
        .map(|m| match m {
            m if m < n && m % 2 == 0 => 64.0,
            m if m < n => -200.0,
            m => (m % 5) as f32 * 0.25 - 0.5,
        })
        .collect::<Vec<_>>();
    let scale = vec![0.0_f32, 0.25, 1.0];
    let roles = MixCollapseNormBuffers {
        x: Spec::input(
            DType::Bf16,
            (Rows::Live, n * h),
            bf16_bytes(&bf16_values(32, rows * n * h)),
        ),
        mixer: MixerWeights {
            fn_: Spec::input(
                DType::Bf16,
                (Rows::Fixed(mixes), n * h),
                bf16_bytes(&projection),
            ),
            base: Spec::input(DType::F32, (Rows::Fixed(1), mixes), f32_bytes(&base)),
            scale: Spec::input(DType::F32, (Rows::Fixed(1), 3), f32_bytes(&scale)),
        },
        norm_weight: Spec::input(
            DType::Bf16,
            (Rows::Fixed(1), h),
            bf16_bytes(&bf16_values(38, h)),
        ),
        collapsed: Spec::output(DType::Bf16, (Rows::Live, h)),
        post: Spec::output(DType::Bf16, (Rows::Live, n)),
        comb: Spec::output(DType::Bf16, (Rows::Live, n * n)),
        normed: Spec::output(DType::Bf16, (Rows::Live, h)),
        scratch: MixScratch {
            partials: info
                .bytes_of("scratch.partials")
                .map(|bytes| Spec::scratch(bytes as usize)),
            tickets: info
                .bytes_of("scratch.tickets")
                .map(|bytes| Spec::zeroed(bytes as usize)),
        },
        live_rows: Spec::live(),
    };
    let query = SinkhornHyperConnectionQuery::MixCollapseNorm {
        mix,
        rmsnorm_eps_bits,
        capacity,
    };
    let eps = f32::from_bits(mix.eps_bits);
    let pre = (0..rows * n)
        .map(|i| if i % n % 2 == 0 { 1.0 + eps } else { eps })
        .collect::<Vec<_>>();
    let mix_oracle = MixOracle::default();
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let output = mix_oracle.get(query, images)?;
        let streams = images.u16s("x")?;
        let collapsed = hc::collapse(query, &streams, &pre).map_err(|e| e.0.to_string())?;
        let normed = hc::input_norm(query, &collapsed, &images.u16s("norm_weight")?)
            .map_err(|e| e.0.to_string())?;
        Ok(vec![
            ("collapsed".into(), Expected::Bits(bf16_bytes(&collapsed))),
            ("post".into(), Expected::Values(output.post)),
            ("comb".into(), Expected::Values(output.comb)),
            ("normed".into(), Expected::Bits(bf16_bytes(&normed))),
        ])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The case of an expand over `capacity` rows: BF16 streams and sublayer
/// rows from the kit's generator, `post` in `[1/16, 2)` and `comb` in
/// `[1/16, 1)` as a mix-collapse writes them (no zero, so no sign of zero
/// depends on the order of operations), and the contract's F32 arithmetic
/// as exact bits.
pub fn expand_case<'a>(
    op: SinkhornHyperConnectionQuery,
    capacity: u32,
    prepared: Prepared<StreamExpandBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let SinkhornHyperConnectionQuery::Expand { streams, width, .. } = op else {
        return Err("an expand case needs an expand".into());
    };
    let (n, h, rows) = (streams as usize, width as usize, capacity as usize);
    let roles = StreamExpandBuffers {
        x: Spec::input(
            DType::Bf16,
            (Rows::Live, n * h),
            bf16_bytes(&bf16_values(33, rows * n * h)),
        ),
        y: Spec::input(
            DType::Bf16,
            (Rows::Live, h),
            bf16_bytes(&bf16_values(34, rows * h)),
        ),
        post: Spec::input(
            DType::Bf16,
            (Rows::Live, n),
            bf16_bytes(&ranged(35, rows * n, 0.0625, 2.0)),
        ),
        comb: Spec::input(
            DType::Bf16,
            (Rows::Live, n * n),
            bf16_bytes(&ranged(36, rows * n * n, 0.0625, 1.0)),
        ),
        output: Spec::output(DType::Bf16, (Rows::Live, n * h)),
        live_rows: Spec::live(),
    };
    let query = SinkhornHyperConnectionQuery::Expand {
        streams,
        width,
        capacity,
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let output = hc::expand(
            query,
            &images.u16s("x")?,
            &images.u16s("y")?,
            &images.u16s("post")?,
            &images.u16s("comb")?,
        )
        .map_err(|e| e.0.to_string())?;
        Ok(vec![("output".into(), Expected::Bits(bf16_bytes(&output)))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The case of a mean-of-streams head over `capacity` rows: BF16 streams
/// from the kit's generator and the contract's F32 mean as exact bits.
pub fn head_case<'a>(
    op: SinkhornHyperConnectionQuery,
    capacity: u32,
    prepared: Prepared<StreamHeadBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let SinkhornHyperConnectionQuery::Head {
        streams,
        width,
        mode: StreamHeadMode::MeanOfStreams,
        ..
    } = op
    else {
        return Err("a head case needs a mean-of-streams head".into());
    };
    let (n, h, rows) = (streams as usize, width as usize, capacity as usize);
    let roles = StreamHeadBuffers {
        x: Spec::input(
            DType::Bf16,
            (Rows::Live, n * h),
            bf16_bytes(&bf16_values(37, rows * n * h)),
        ),
        output: Spec::output(DType::Bf16, (Rows::Live, h)),
        live_rows: Spec::live(),
    };
    let query = SinkhornHyperConnectionQuery::Head {
        streams,
        width,
        mode: StreamHeadMode::MeanOfStreams,
        capacity,
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let output = hc::head(query, &images.u16s("x")?).map_err(|e| e.0.to_string())?;
        Ok(vec![("output".into(), Expected::Bits(bf16_bytes(&output)))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// SM120 as the lab builds its Sinkhorn engine: the served one-CTA
/// entries, and with `split` the split mix-collapse, which serves the
/// mix-collapse in their place.
pub fn sm120(split: bool) -> Sm120 {
    let backend = Sm120::new().with_sinkhorn(sinkhorn_hc::catalog());
    if split {
        backend.with_sinkhorn_split(split::entry(SERVED_PROFILE, SERVED))
    } else {
        backend
    }
}

/// SM120 with the split mix-collapse and the served tiles of the tiled
/// entry (`tiled::SERVED_TILES`), which serve the rows SM120's schedule
/// tiles.
pub fn sm120_tiled() -> Sm120 {
    sm120(true).with_sinkhorn_tiled(tiled::SERVED_TILES.map(tiled::served))
}

/// SM120 with the split mix-collapse fused with the input norm's tree at
/// the mixer's norm eps.
pub fn sm120_mix_norm() -> Sm120 {
    sm120(true).with_sinkhorn_mix_norm(mix_norm_tree::served())
}

/// The served shape's mix-collapse at the first row count of each tile
/// SM120's schedule runs (`tiled::TILED_FROM_ROWS`, the 4-row tile, and
/// `tiled::EIGHT_ROWS_FROM`, the 8-row tile), each required.
pub fn required_tiled<'a>() -> Vec<Required<'a>> {
    let collapse = SinkhornHyperConnectionQuery::MixCollapse {
        mix: SERVED.mix,
        capacity: SERVED.capacity,
    };
    [tiled::TILED_FROM_ROWS, tiled::EIGHT_ROWS_FROM]
        .into_iter()
        .map(|rows| {
            let need = Need::rows(rows);
            Required::new(
                format!(
                    "sinkhorn mix-collapse {} streams width {} at {rows} rows",
                    SERVED.mix.streams, SERVED.mix.width
                ),
                need,
                move |backend, need| backend.sinkhorn_mix_collapse(&collapse, need),
                move |prepared| mix_collapse_case(collapse, need.capacity, prepared),
            )
            .required()
        })
        .collect()
}

/// The served shape's mix-collapse and input norm at the mixer's norm eps,
/// at [`CAPACITY`] rows and at 192 (six rows of 32 items), each required.
pub fn required_mix_norm<'a>() -> Vec<Required<'a>> {
    let op = SinkhornHyperConnectionQuery::MixCollapseNorm {
        mix: SERVED.mix,
        rmsnorm_eps_bits: SERVED.mix.norm_eps_bits,
        capacity: SERVED.capacity,
    };
    [CAPACITY, 192]
        .into_iter()
        .map(|rows| {
            let need = Need::rows(rows);
            Required::new(
                format!(
                    "sinkhorn mix-collapse and norm {} streams width {} at {rows} rows",
                    SERVED.mix.streams, SERVED.mix.width
                ),
                need,
                move |backend, need| backend.sinkhorn_mix_collapse_norm(&op, need),
                move |prepared| mix_collapse_norm_case(op, need.capacity, prepared),
            )
            .required()
        })
        .collect()
}

/// The served shape's mix-collapse, expand and head at [`CAPACITY`] rows,
/// each required.
pub fn required<'a>() -> Vec<Required<'a>> {
    let need = Need::rows(CAPACITY);
    let mix = SERVED.mix;
    let collapse = SinkhornHyperConnectionQuery::MixCollapse {
        mix,
        capacity: SERVED.capacity,
    };
    let expand = SinkhornHyperConnectionQuery::Expand {
        streams: mix.streams,
        width: mix.width,
        capacity: SERVED.capacity,
    };
    let head = SinkhornHyperConnectionQuery::Head {
        streams: mix.streams,
        width: mix.width,
        mode: StreamHeadMode::MeanOfStreams,
        capacity: SERVED.capacity,
    };
    let shape = format!("{} streams width {}", mix.streams, mix.width);
    vec![
        Required::new(
            format!("sinkhorn mix-collapse {shape}"),
            need,
            move |backend, need| backend.sinkhorn_mix_collapse(&collapse, need),
            move |prepared| mix_collapse_case(collapse, need.capacity, prepared),
        )
        .required(),
        Required::new(
            format!("sinkhorn expand {shape}"),
            need,
            move |backend, need| backend.sinkhorn_expand(&expand, need),
            move |prepared| expand_case(expand, need.capacity, prepared),
        )
        .required(),
        Required::new(
            format!("sinkhorn head {shape}"),
            need,
            move |backend, need| backend.sinkhorn_head(&head, need),
            move |prepared| head_case(head, need.capacity, prepared),
        )
        .required(),
    ]
}
