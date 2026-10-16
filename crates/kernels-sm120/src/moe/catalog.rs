//! The grouped expert entry builder for the packed and MXFP4 fragment
//! engines, their tuning tables, and the standard profile's entries.

use super::*;
use serde::Deserialize;
use std::collections::BTreeMap;

/// A grouped expert tuning table (`tuning/moe.json`, or a model's own):
/// decode and prefill knobs by token bucket and the MXFP4 fragment engine's
/// prefill rows.
#[derive(Deserialize)]
pub struct Tuning {
    compute_capability: u32,
    sm_count: u32,
    decode: BTreeMap<String, TunedPhase>,
    prefill: BTreeMap<String, TunedPhase>,
    #[serde(default)]
    fragment_prefill: BTreeMap<String, TunedPhase>,
}

impl Tuning {
    /// The table in `json`, for this crate's SM120 target.
    pub fn parse(json: &str) -> Result<Tuning, Refusal> {
        let table: Tuning = serde_json::from_str(json)
            .map_err(|_| Refusal("MoE tuning table is not valid JSON"))?;
        if (table.compute_capability, table.sm_count) != (120, SM_COUNT) {
            return Err(Refusal("MoE tuning table targets another device"));
        }
        Ok(table)
    }
}

#[derive(Deserialize)]
struct TunedPhase {
    #[serde(default)]
    weight_stages: u8,
    row_block: u16,
    n_tile: u16,
    k_tile: u16,
    warps: u8,
    stages: u8,
    split_k: u8,
}

/// The standard profile's tuning table.
pub(super) fn tuning() -> Tuning {
    Tuning::parse(include_str!("../../tuning/moe.json")).expect("MoE tuning table")
}

/// Conservative defaults for the fragment engine, validated by `variant`.
pub fn default_tuning() -> Tuning {
    let mut table = tuning();
    table.fragment_prefill.insert(
        "8192".into(),
        TunedPhase {
            row_block: 64,
            n_tile: 128,
            k_tile: 64,
            warps: 8,
            stages: 2,
            split_k: 1,
            weight_stages: 2,
        },
    );
    table
}

fn knobs(phase: Phase, tokens: u32, table: &Tuning) -> CompileKnobs {
    let tuned = match phase {
        Phase::Decode => &table.decode,
        Phase::Prefill => &table.prefill,
    }
    .get(&tokens.to_string())
    .expect("MoE tuning bucket exists");
    CompileKnobs {
        tile_m: tuned.row_block,
        tile_n: tuned.n_tile,
        tile_k: tuned.k_tile,
        stages: tuned.stages,
        swizzle: 0,
        warps: tuned.warps,
        split_k_capacity: tuned.split_k,
    }
}

/// The grouped expert entry of `profile`'s MoE block `class` in `layout`
/// (the packed engine, or the MXFP4 fragment engine for MXFP4 experts) at
/// a projection, phase, token capacity, weight format and input path, with
/// `table`'s knobs.
#[allow(clippy::too_many_arguments)]
pub fn variant(
    profile: Profile,
    class: &'static str,
    layout: WeightLayout,
    projection: Projection,
    phase: Phase,
    tokens_capacity: u32,
    weight: WeightFormat,
    path: InputPath,
    table: &Tuning,
) -> Result<Variant, Refusal> {
    variant_with(
        profile,
        class,
        layout,
        projection,
        phase,
        tokens_capacity,
        weight,
        path,
        table,
        false,
    )
}

/// The exact entry of an A16 fragment decode bucket (see `Variant::exact`):
/// the per-warp ring kernel with bucket 1's warps and K splits. `None`
/// where the bucket's decode entry already has bucket 1's reduction order.
pub fn exact_variant(
    profile: Profile,
    class: &'static str,
    projection: Projection,
    tokens_capacity: u32,
    table: &Tuning,
) -> Result<Option<Variant>, Refusal> {
    let variant = |tokens, exact| {
        variant_with(
            profile,
            class,
            WeightLayout::Mxfp4Fragment,
            projection,
            Phase::Decode,
            tokens,
            WeightFormat::Mxfp4,
            InputPath::A16Gather,
            table,
            exact,
        )
    };
    let decode = variant(tokens_capacity, false)?;
    let one = variant(1, false)?;
    let order = |v: &Variant| v.knobs.split_k_capacity;
    if order(&decode) == order(&one) {
        return Ok(None);
    }
    let exact = variant(tokens_capacity, true)?;
    debug_assert_eq!(order(&exact), order(&one));
    Ok(Some(exact))
}

#[allow(clippy::too_many_arguments)]
fn variant_with(
    profile: Profile,
    class: &'static str,
    layout: WeightLayout,
    projection: Projection,
    phase: Phase,
    tokens_capacity: u32,
    weight: WeightFormat,
    path: InputPath,
    table: &Tuning,
    exact: bool,
) -> Result<Variant, Refusal> {
    // The MXFP4 fragment engine serves MXFP4 experts; the packed engine
    // every other format.
    let fragment = layout == WeightLayout::Mxfp4Fragment;
    if fragment != (weight == WeightFormat::Mxfp4) || layout == WeightLayout::Fp8Fragment {
        return Err(Refusal(
            "MXFP4 experts take the fragment layout and other formats the packed one",
        ));
    }
    if exact && !(fragment && path == InputPath::A16Gather && phase == Phase::Decode) {
        return Err(Refusal(
            "exact expert entries are A16 decode buckets of fragment experts",
        ));
    }
    if phase == Phase::Decode && tokens_capacity > 32 {
        return Err(Refusal("decode buckets hold at most 32 tokens"));
    }
    let shape = classes(profile)
        .into_iter()
        .find(|&(name, _)| name == class)
        .ok_or(Refusal("the profile's model has no such MoE block class"))?
        .1;
    // The packed and MXFP4 fragment engines weight routes after the down
    // projection and do not clamp SwiGLU inputs.
    let (route_weight, swiglu_clamp) = (RouteWeight::AfterDown, None);
    let (n, k) = shape.projection(projection);
    let (swiglu, weighted) = expert_epilogue(projection, route_weight, swiglu_clamp)?;
    let experts = shape.experts;
    let activation = match path {
        InputPath::A16Gather => ActivationFormat::Bf16,
        InputPath::A8Pregather => match weight {
            WeightFormat::Mxfp4 | WeightFormat::Mxfp8 => ActivationFormat::Mxfp8,
            WeightFormat::Nvfp4 => ActivationFormat::Nvfp4,
            WeightFormat::Fp8Block128x128(_) => ActivationFormat::Fp8Group128(ScaleMode::ExactF32),
            _ => unreachable!(),
        },
    };
    let mut knobs = knobs(phase, tokens_capacity, table);
    let mut weight_stages = 0;
    if fragment && path == InputPath::A16Gather && phase == Phase::Prefill {
        let tuned = table
            .fragment_prefill
            .get(&tokens_capacity.to_string())
            .ok_or(Refusal("fragment prefill bucket is not tuned"))?;
        knobs = CompileKnobs {
            tile_m: tuned.row_block,
            tile_n: tuned.n_tile,
            tile_k: tuned.k_tile,
            stages: tuned.stages,
            swizzle: 0,
            warps: tuned.warps,
            split_k_capacity: tuned.split_k,
        };
        weight_stages = tuned.weight_stages;
        supported_fragment_prefill(knobs, weight_stages)?;
    }
    if fragment && path == InputPath::A16Gather && phase == Phase::Decode {
        // An exact entry takes bucket 1's warps, and so its K splits.
        knobs.warps = fragment_decode_warps(projection, if exact { 1 } else { tokens_capacity });
        if projection == Projection::GateUp {
            knobs.split_k_capacity = knobs.warps;
        }
    }
    if phase == Phase::Prefill || projection == Projection::Down {
        knobs.split_k_capacity = 1;
    } else if matches!(
        weight,
        WeightFormat::Fp8Block128x128(_) | WeightFormat::Mxfp8
    ) {
        knobs.split_k_capacity = knobs.split_k_capacity.max(2);
    }
    let rows = tokens_capacity
        .checked_mul(shape.top_k)
        .ok_or(Refusal("MoE capacity overflow"))?;
    let capacity = padded_capacity(rows, experts, u32::from(knobs.tile_m))?;
    let query = GemmQuery {
        m_capacity: rows,
        n,
        k,
        weight,
        activation,
        output: DType::Bf16,
        epilogues: Epilogues {
            swiglu,
            weighted_moe_combine: weighted,
            ..Epilogues::default()
        },
        grouped: Some(GroupedGemm {
            experts,
            top_k: shape.top_k,
            capacity: RoutedCapacity::TotalRows(capacity),
        }),
    };
    supported(&query, phase, knobs, layout)?;
    let profile_name = class_tag(profile, class);
    let projection_name = match projection {
        Projection::GateUp => "gateup",
        Projection::Down => "down",
    };
    let phase_name = match phase {
        Phase::Decode if exact => "exact",
        Phase::Decode => "decode",
        Phase::Prefill => "prefill",
    };
    let path_name = match path {
        InputPath::A16Gather => "a16",
        InputPath::A8Pregather => "a8",
    };
    let symbol = format!(
        "chop_moe_{profile_name}_{projection_name}_{phase_name}_{}_{}_t{tokens_capacity}_v1",
        format_label(weight),
        path_name
    );
    let variant = Variant {
        profile,
        class,
        layout,
        projection,
        phase,
        tokens_capacity,
        path,
        query,
        knobs,
        weight_stages,
        exact,
        route_weight,
        swiglu_clamp,
        engine: None,
        schedule: None,
        symbol,
        invariance: if exact {
            Invariance::one_row(tokens_capacity)
        } else if phase == Phase::Prefill && fragment && path == InputPath::A16Gather {
            Invariance::PREFILL
        } else {
            Invariance::NONE
        },
    };
    Ok(Variant {
        engine: if fragment {
            mxfp4_fragment::engine_schedule(&variant)
        } else {
            None
        },
        ..variant
    })
}

/// The kernel base's entries, then the models' in `catalogs`.
pub fn catalog(catalogs: &Catalogs) -> Vec<Variant> {
    let mut all = own_catalog();
    for model in catalogs.models() {
        all.extend((model.moe)());
    }
    all
}

/// The standard profile's entries for every MoE block class, format,
/// activation path, projection and bucket.
pub fn own_catalog() -> Vec<Variant> {
    let table = tuning();
    let mut all = Vec::new();
    let profile = Profile::StdDenseMoe;
    for (class, _) in classes(profile) {
        for weight in [
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            WeightFormat::Nvfp4,
            WeightFormat::Mxfp8,
        ] {
            for path in [InputPath::A16Gather, InputPath::A8Pregather] {
                for projection in [Projection::GateUp, Projection::Down] {
                    let at = |phase, tokens| {
                        variant(
                            profile,
                            class,
                            WeightLayout::Packed,
                            projection,
                            phase,
                            tokens,
                            weight,
                            path,
                            &table,
                        )
                    };
                    for (phase, tokens) in DECODE_BUCKETS
                        .map(|tokens| (Phase::Decode, tokens))
                        .into_iter()
                        .chain([(Phase::Prefill, 8192)])
                    {
                        let base = at(phase, tokens).expect("catalog variant");
                        if projection == Projection::GateUp {
                            all.push(interleaved(&base));
                        }
                        all.push(base);
                    }
                }
            }
        }
    }
    all
}

/// `base` with interleaved gate/up columns, which only the packed engine
/// reads.
pub fn interleaved(base: &Variant) -> Variant {
    let mut variant = base.clone();
    variant.query.epilogues.swiglu = Some(SwigluColumns::Interleaved);
    variant.symbol = base.symbol.replacen("_t", "_interleaved_t", 1);
    variant
}

/// `base` with its SwiGLU output quantized to MXFP8 for an A8 down
/// projection.
pub fn quantized(base: &Variant) -> Variant {
    let mut variant = base.clone();
    variant.query.epilogues.output_quantization = Some(OutputQuantization::Mxfp8);
    if variant.phase == Phase::Decode {
        variant.knobs.split_k_capacity = variant.knobs.split_k_capacity.max(2);
    }
    variant.symbol = base.symbol.replacen("_t", "_qmx8_t", 1);
    variant
}
