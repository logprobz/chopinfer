//! MiMo's encoder declarations, whatever kernels serve them: the declaration
//! documents and their geometry-specific forms, the input recipes, MiMo's
//! bespoke variants and the environment its declarations validate in. The
//! variants are registered with `encoder-spec` through `PROVIDER`; shared
//! code never names them. A second model that needs one promotes it to a
//! shared variant under a new id.

use crate::geometry::V26_FLASH as GEOMETRY;

pub use encoder_spec::Recipe;
use encoder_spec::components::rotary::{Axes, Positions, RotaryComponent, Style};
use encoder_spec::components::{
    Boundary, Component, Context, Data, Element, Scratch, Typed, Variant, f32_positive, mul,
};
use encoder_spec::cost::Cost;
use encoder_spec::policy::{self, ParallelClass, Precision};
use encoder_spec::registry::{self, Decoder, Provider, Registration};
use encoder_spec::weights::Role;
use encoder_spec::{Declaration, Environment, Error, parse};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

pub const VISION: &str = include_str!("../../encoders/mimo-vision.json");
pub const AUDIO: &str = include_str!("../../encoders/mimo-audio.json");
/// RVQ codebooks in the audio tokenizer.
const RVQ_STAGES: usize = GEOMETRY.audio_tokenizer().num_quantizers as usize;
/// Speech codes and embedding tables per audio row.
const AUDIO_CHANNELS: usize = GEOMETRY.audio().audio_channels as usize;
// The rvq20 component requires one speech embedding table per RVQ code.
const _: () = assert!(AUDIO_CHANNELS == RVQ_STAGES);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rvq20Params {
    /// Entries of each stage's codebook, in stage order.
    pub codebooks: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Rvq20 {
    /// Residual vector quantizer: 20 stages, each the nearest codebook entry
    /// to the remaining F32 residual, F32 distances over BF16 codebooks.
    /// Code `s` is below `codebooks[s]`. Retired: it stays readable (the
    /// reference evaluates it) but no MiMo kernel runs it.
    #[serde(rename = "mimo26:quantize.rvq20.v1")]
    V1(Rvq20Params),
    /// `V1` over F32 rows and the checkpoint's F32 codebooks, under
    /// `f32-pairs.v1`, on a kernel MiMo's backend owns.
    #[serde(rename = "mimo26:quantize.rvq20-f32.v1")]
    F32(Rvq20Params),
}

impl Rvq20 {
    pub fn params(&self) -> &Rvq20Params {
        match self {
            Rvq20::V1(params) | Rvq20::F32(params) => params,
        }
    }
}

impl Typed for Rvq20 {
    fn policies(&self) -> &'static [Precision] {
        match self {
            Rvq20::V1(_) => policy::LEGACY,
            Rvq20::F32(_) => policy::PAIRS,
        }
    }
}

impl Component for Rvq20 {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::RowLocal
    }

    fn output(&self, _: &Context, input: &Boundary) -> Result<Boundary, String> {
        input.width()?;
        let params = self.params();
        if params.codebooks.len() != RVQ_STAGES || params.codebooks.contains(&0) {
            return Err(format!("rvq20 has {RVQ_STAGES} non-empty codebooks"));
        }
        Ok(input.with(Data::Codes {
            bounds: params.codebooks.clone(),
        }))
    }

    fn roles(&self, _: &Context, input: &Boundary) -> Result<Vec<Role>, String> {
        // One codebook per stage, `[entries, width]`, stacked in stage
        // order: the kernel finds stage `s` at the sum of the earlier sizes.
        // BF16 (narrowed) under `V1`, as stored (F32) under `F32`.
        let params = self.params();
        let width = u64::from(input.width()?);
        let parts = params
            .codebooks
            .iter()
            .map(|&entries| vec![u64::from(entries), width])
            .collect();
        let role = Role::stacked("codebooks", parts)?;
        Ok(vec![match self {
            Rvq20::V1(_) => role,
            Rvq20::F32(_) => role.element(Element::F32),
        }])
    }

    fn scratch(&self, rows: u64, input: &Boundary) -> Result<Vec<Scratch>, String> {
        // The F32 copy of the BF16 rows the residual search reads (`V1`),
        // and the first codebook row of each stage plus the end.
        let params = self.params();
        let offsets = Scratch {
            name: "codebook-offsets",
            bytes: (params.codebooks.len() as u64 + 1) * 4,
        };
        Ok(match self {
            Rvq20::V1(_) => vec![
                Scratch {
                    name: "rvq-input",
                    bytes: mul(mul(rows, u64::from(input.width()?))?, 4)?,
                },
                offsets,
            ],
            Rvq20::F32(_) => vec![offsets],
        })
    }

    fn cost(
        &self,
        rows: [u64; 2],
        input: &Boundary,
        output: &Boundary,
    ) -> Result<Option<Cost>, String> {
        // A distance to every entry of every stage's codebook: per element a
        // subtraction and a fused multiply-add (`rvq`), three FLOPs. The
        // argmin search, the residual update after each stage and the F32
        // cast are not counted. `V1` writes and reads an F32 copy of the
        // rows.
        let params = self.params();
        let width = u64::from(input.width()?);
        let entries: u64 = params.codebooks.iter().copied().map(u64::from).sum();
        let copy = match self {
            Rvq20::V1(_) => 2 * width * 4,
            Rvq20::F32(_) => 0,
        };
        Ok(Some(Cost {
            flops: 3 * rows[0] * width * entries,
            bytes: rows[0] * (input.data.row_bytes() + copy) + rows[1] * output.data.row_bytes(),
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sum20Params {
    /// Rows of each table.
    pub vocab: u32,
    pub width: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Sum20 {
    /// Row `r` sums table `s` row `code[r][s]` over the 20 tables in order,
    /// rounding to BF16 after each add. Readable for the reference only;
    /// no serving kernel implements this retired variant.
    #[serde(rename = "mimo26:embed.sum20.v1")]
    V1(Sum20Params),
    /// `V1` summed in F32 in table order, F32 rows out, under
    /// `f32-pairs.v1`, on a kernel MiMo's backend owns.
    #[serde(rename = "mimo26:embed.sum20-f32.v1")]
    F32(Sum20Params),
}

impl Sum20 {
    pub fn params(&self) -> &Sum20Params {
        match self {
            Sum20::V1(params) | Sum20::F32(params) => params,
        }
    }
}

impl Typed for Sum20 {
    fn policies(&self) -> &'static [Precision] {
        match self {
            Sum20::V1(_) => policy::LEGACY,
            Sum20::F32(_) => policy::PAIRS,
        }
    }
}

impl Component for Sum20 {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::RowLocal
    }

    fn output(&self, _: &Context, input: &Boundary) -> Result<Boundary, String> {
        let params = self.params();
        let Data::Codes { bounds } = &input.data else {
            return Err(format!(
                "expects {AUDIO_CHANNELS} codes per row but receives {}",
                input.data
            ));
        };
        if bounds.len() != AUDIO_CHANNELS {
            return Err(format!(
                "expects {AUDIO_CHANNELS} codes per row but receives {}",
                input.data
            ));
        }
        if params.vocab == 0 || params.width == 0 {
            return Err("`vocab` and `width` are positive".into());
        }
        if let Some((stage, bound)) = bounds
            .iter()
            .enumerate()
            .find(|(_, bound)| **bound > params.vocab)
        {
            return Err(format!(
                "code {stage} reaches {bound} entries but each table has {} rows",
                params.vocab
            ));
        }
        Ok(input.with(Data::Rows {
            width: params.width,
            element: match self {
                Sum20::V1(_) => Element::Bf16,
                Sum20::F32(_) => Element::F32,
            },
        }))
    }

    fn roles(&self, _: &Context, _: &Boundary) -> Result<Vec<Role>, String> {
        // 20 tables of `[vocab, width]`; the kernel reads table `s` at
        // `s * vocab`.
        let params = self.params();
        let table = vec![u64::from(params.vocab), u64::from(params.width)];
        mul(AUDIO_CHANNELS as u64, table[0])?;
        Ok(vec![Role::stacked("tables", vec![table; AUDIO_CHANNELS])?])
    }

    fn cost(
        &self,
        rows: [u64; 2],
        input: &Boundary,
        output: &Boundary,
    ) -> Result<Option<Cost>, String> {
        // One table row gathered and added per code.
        let params = self.params();
        let width = u64::from(params.width);
        Ok(Some(Cost {
            flops: rows[1] * AUDIO_CHANNELS as u64 * width,
            bytes: rows[0] * input.data.row_bytes()
                + rows[1] * (AUDIO_CHANNELS as u64 * width * 2 + output.data.row_bytes()),
        }))
    }
}

fn check_theta(id: &str, theta: f64) -> Result<(), String> {
    let theta = f32_positive(&format!("`{id}`: theta"), theta)?;
    if theta <= 1.0 {
        return Err(format!("`{id}`: theta {theta} is not above 1"));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Axial2dParams {
    pub pairs_per_axis: u32,
    /// Rounded to F32, as the table computes.
    pub theta: f64,
    pub style: Style,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Axial2dF32Powf {
    /// Vision: `1 / theta.powf(2p / dims)` and the angle's `cos`/`sin` in
    /// Rust F32 in the historical table convention, applied to
    /// BF16 q and k.
    #[serde(rename = "mimo26:rotary.axial-2d-f32-powf.v1")]
    V1(Axial2dParams),
}

impl Typed for Axial2dF32Powf {
    fn policies(&self) -> &'static [Precision] {
        policy::LEGACY
    }
}

impl RotaryComponent for Axial2dF32Powf {
    fn rotary_dims(&self) -> Result<u64, String> {
        let Axial2dF32Powf::V1(params) = self;
        mul(4, u64::from(params.pairs_per_axis))
    }

    fn axes(&self) -> Axes {
        Axes::Grid2d
    }

    fn check(&self) -> Result<(), String> {
        let Axial2dF32Powf::V1(params) = self;
        check_theta(&self.id(), params.theta)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rope1dParams {
    pub pairs: u32,
    /// Rounded to F32, as the table computes.
    pub theta: f64,
    pub style: Style,
    pub positions: Positions,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Rope1dBf16Unfused {
    /// Audio: 1D rotary with the F32 `powf` table, emulating unfused PyTorch
    /// Retired BF16 rotary arithmetic, retained for reference decoding only.
    #[serde(rename = "mimo26:rotary.1d-bf16-unfused.v1")]
    V1(Rope1dParams),
}

impl Typed for Rope1dBf16Unfused {
    fn policies(&self) -> &'static [Precision] {
        policy::LEGACY
    }
}

impl RotaryComponent for Rope1dBf16Unfused {
    fn rotary_dims(&self) -> Result<u64, String> {
        let Rope1dBf16Unfused::V1(params) = self;
        mul(2, u64::from(params.pairs))
    }

    fn axes(&self) -> Axes {
        let Rope1dBf16Unfused::V1(params) = self;
        match params.positions {
            Positions::WithinSequence => Axes::Sequence,
        }
    }

    fn check(&self) -> Result<(), String> {
        let Rope1dBf16Unfused::V1(params) = self;
        check_theta(&self.id(), params.theta)
    }
}

/// MiMo's registered variants. An id here never changes meaning.
pub static PROVIDER: Provider = Provider {
    namespace: "mimo26",
    registrations: &[
        Registration {
            id: "mimo26:quantize.rvq20.v1",
            decoder: Decoder::Stage(registry::stage::<Rvq20>),
        },
        Registration {
            id: "mimo26:embed.sum20.v1",
            decoder: Decoder::Stage(registry::stage::<Sum20>),
        },
        Registration {
            id: "mimo26:quantize.rvq20-f32.v1",
            decoder: Decoder::Stage(registry::stage::<Rvq20>),
        },
        Registration {
            id: "mimo26:embed.sum20-f32.v1",
            decoder: Decoder::Stage(registry::stage::<Sum20>),
        },
        Registration {
            id: "mimo26:rotary.axial-2d-f32-powf.v1",
            decoder: Decoder::Rotary(registry::rotary::<Axial2dF32Powf>),
        },
        Registration {
            id: "mimo26:rotary.1d-bf16-unfused.v1",
            decoder: Decoder::Rotary(registry::rotary::<Rope1dBf16Unfused>),
        },
    ],
};

/// MiMo's declarations in the frontend's encoder order: vision, then
/// audio. Parsed and validated once.
pub fn declarations_flash() -> Result<&'static [Declaration; 2], Error> {
    static DECLARATIONS: OnceLock<Result<[Declaration; 2], Error>> = OnceLock::new();
    DECLARATIONS
        .get_or_init(|| {
            let environment = environment()?;
            Ok([parse(VISION, &environment)?, parse(AUDIO, &environment)?])
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Geometry-specific declarations, retaining the shipped template for Flash.
pub fn declarations_for(
    geometry: &'static crate::geometry::Geometry,
) -> Result<&'static [Declaration; 2], Error> {
    type Declarations = Result<[Declaration; 2], Error>;
    static DECLARATIONS: OnceLock<Vec<Declarations>> = OnceLock::new();
    if geometry.id == GEOMETRY.id {
        return declarations_flash();
    }
    let index = crate::geometry::GEOMETRIES
        .iter()
        .position(|g| g.id == geometry.id)
        .ok_or_else(|| Error(format!("unlisted encoder geometry {}", geometry.id)))?;
    DECLARATIONS.get_or_init(|| {
        crate::geometry::GEOMETRIES
            .iter()
            .map(|g| {
                let environment = environment()?;
                let documents = declaration_documents(g)?;
                Ok([
                    parse(&documents[0], &environment)?,
                    parse(&documents[1], &environment)?,
                ])
            })
            .collect()
    })[index]
        .as_ref()
        .map_err(Clone::clone)
}

/// Input recipes share the declaration's bounds and geometry dimensions.
pub fn recipes_for(g: &crate::geometry::Geometry) -> Result<[Vec<encoder_spec::Recipe>; 2], Error> {
    let mut vision: serde_json::Value =
        serde_json::from_str(VISION).map_err(|e| Error(e.to_string()))?;
    let audio: serde_json::Value = serde_json::from_str(AUDIO).map_err(|e| Error(e.to_string()))?;
    for recipe in vision["input"]["recipes"]
        .as_array_mut()
        .ok_or_else(|| Error("vision recipes must be an array".into()))?
    {
        recipe["patch"] = g.vision().patch_size.into();
        recipe["temporal_patch"] = g.vision().temporal_patch_size.into();
        recipe["merge"] = g.vision().spatial_merge_size.into();
    }
    let mut audio = audio["input"]["recipes"].clone();
    audio[0]["mel_bins"] = g.audio_tokenizer().n_mels.into();
    Ok([
        serde_json::from_value(vision["input"]["recipes"].clone())
            .map_err(|e| Error(e.to_string()))?,
        serde_json::from_value(audio).map_err(|e| Error(e.to_string()))?,
    ])
}

fn declaration_documents(g: &crate::geometry::Geometry) -> Result<[String; 2], Error> {
    use serde_json::{Value, json};
    let mut vision: Value = serde_json::from_str(VISION).map_err(|e| Error(e.to_string()))?;
    let mut audio: Value = serde_json::from_str(AUDIO).map_err(|e| Error(e.to_string()))?;
    let config: Value = serde_json::from_str(g.document).map_err(|e| Error(e.to_string()))?;
    let v = g.vision();
    let a = g.audio();
    let t = g.audio_tokenizer();
    vision["checkpoint"]["config"][0]["values"] = config["vision_config"].clone();
    audio["checkpoint"]["config"][0]["values"] = config["audio_config"].clone();
    audio["checkpoint"]["config"][1]["values"] = config["audio_tokenizer_config"].clone();
    for document in [&mut vision, &mut audio] {
        document["output"]["hidden"] = g.hidden().into();
    }
    vision["input"]["columns"] =
        (v.in_chans * v.temporal_patch_size * v.patch_size * v.patch_size).into();
    let [vision_recipes, audio_recipes] = recipes_for(g)?;
    vision["input"]["recipes"] =
        serde_json::to_value(vision_recipes).map_err(|e| Error(e.to_string()))?;
    audio["input"]["recipes"] =
        serde_json::to_value(audio_recipes).map_err(|e| Error(e.to_string()))?;
    let selector = |indices: Vec<usize>| {
        if !indices.is_empty() && indices == (0..v.depth as usize).step_by(2).collect::<Vec<_>>() {
            "even".to_owned()
        } else if !indices.is_empty()
            && indices == (1..v.depth as usize).step_by(2).collect::<Vec<_>>()
        {
            "odd".to_owned()
        } else {
            indices
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        }
    };
    let block_list = |kind| {
        selector(
            g.shape
                .vision
                .vit_window_attn_types
                .iter()
                .enumerate()
                .filter_map(|(i, k)| (*k == kind).then_some(i))
                .collect(),
        )
    };
    for stage in vision["stages"].as_array_mut().unwrap() {
        match stage["op"].as_str().unwrap() {
            "transformer.pre-norm.v1" => {
                stage["blocks"] = v.depth.into();
                stage["width"] = v.hidden_size.into();
                stage["attention"]["q_heads"] = v.num_heads.into();
                stage["attention"]["kv_heads"] = v.num_key_value_heads.into();
                stage["attention"]["head_dim"] = g.vision_head_dim().into();
                stage["attention"]["mask"]["window"] =
                    json!([v.visual_token_window_size, v.visual_token_window_size]);
                stage["mlp"]["hidden"] = v.intermediate_size.into();
                stage["overrides"][0]["blocks"] =
                    block_list(crate::geometry::VisionLayerKind::Full).into();
                stage["reorder"][0]["blocks"] =
                    block_list(crate::geometry::VisionLayerKind::MergeGroupWindow).into();
                let sinks: Vec<_> = g
                    .shape
                    .vision
                    .vit_window_attn_types
                    .iter()
                    .enumerate()
                    .filter_map(|(i, k)| {
                        (*k != crate::geometry::VisionLayerKind::Full).then_some(i)
                    })
                    .collect();
                stage["weights"]["sinks"]["blocks"] = selector(sinks.clone()).into();
                if sinks.is_empty() {
                    stage["attention"]["mask"] = "full".into();
                    stage["attention"]["sink"] = "none".into();
                    stage.as_object_mut().unwrap().remove("overrides");
                    stage["weights"].as_object_mut().unwrap().remove("sinks");
                } else if sinks.len() == v.depth as usize {
                    stage["weights"]["sinks"]
                        .as_object_mut()
                        .unwrap()
                        .remove("blocks");
                }
                for field in ["overrides", "reorder"] {
                    if stage[field][0]["blocks"].as_str() == Some("") {
                        stage.as_object_mut().unwrap().remove(field);
                    }
                }
                if stage["weights"]["sinks"]["blocks"].as_str() == Some("") {
                    stage["weights"].as_object_mut().unwrap().remove("sinks");
                }
            }
            "merge.consecutive.v1" => stage["rows"] = v.spatial_merge_size.pow(2).into(),
            "linear.pairs.v1" => {
                let source = stage["weights"]["weight"].to_string();
                stage["out"] = if source.contains("patch_embed") {
                    v.hidden_size
                } else if source.contains("mlp.0") {
                    v.hidden_size * v.spatial_merge_size.pow(2)
                } else {
                    g.hidden()
                }
                .into();
            }
            _ => {}
        }
    }
    audio["input"]["columns"] = t.n_mels.into();
    audio["input"]["recipes"][0]["mel_bins"] = t.n_mels.into();
    audio["sequences"]["input_rows"] = a.audio_segment_size.into();
    for stage in audio["stages"].as_array_mut().unwrap() {
        let sources = stage["weights"].to_string();
        match stage["op"].as_str().unwrap() {
            "conv1d.im2col.v1" => {
                stage["out"] = t.d_model.into();
                if sources.contains("down_sample_layer") {
                    stage["kernel"] = t.avg_pooler.into();
                    stage["stride"] = t.avg_pooler.into();
                    stage["pad"] = json!([0, t.avg_pooler - 1]);
                } else {
                    stage["kernel"] = t.kernel_size.into();
                    stage["pad"] = json!([t.kernel_size / 2, t.kernel_size / 2]);
                    stage["stride"] = if sources.contains("conv2") {
                        t.stride_size
                    } else {
                        1
                    }
                    .into();
                }
            }
            "transformer.pre-norm.v1" => {
                let tokenizer = sources.contains("encoder.layers.");
                let (blocks, width, heads, head, intermediate, theta) = if tokenizer {
                    (
                        t.encoder_layers,
                        t.d_model,
                        t.encoder_attention_heads,
                        t.d_model / t.encoder_attention_heads,
                        t.encoder_ffn_dim,
                        t.rope_theta,
                    )
                } else {
                    (
                        a.input_local_layers,
                        a.input_local_dim,
                        a.input_local_attn_heads,
                        a.input_local_head_dim,
                        a.input_local_intermediate_size,
                        a.rope_theta,
                    )
                };
                stage["blocks"] = blocks.into();
                stage["width"] = width.into();
                stage["attention"]["q_heads"] = heads.into();
                stage["attention"]["kv_heads"] = heads.into();
                stage["attention"]["head_dim"] = head.into();
                stage["mlp"]["hidden"] = intermediate.into();
                stage["rotary"]["pairs"] = (head / 2).into();
                stage["rotary"]["theta"] = theta.into();
                if tokenizer {
                    stage["weights"]["qkv.bias"]["concat_rows"][1]["zeros"] = width.into();
                    stage["carry"][0]["save_after"] = (t.encoder_skip_layer_id - 1).into();
                    stage["carry"][0]["add_after"] = (blocks - 1).into();
                    stage["overrides"][0]["attention"]["mask"]["causal_window"] =
                        t.encoder_attn_window_size[0].into();
                } else {
                    stage["sequences"]["rows"] = a.group_size.into();
                }
            }
            "mimo26:quantize.rvq20-f32.v1" => {
                stage["codebooks"] = json!(t.codebook_size);
                stage["weights"]["codebooks"]["count"] = t.num_quantizers.into();
            }
            "mimo26:embed.sum20-f32.v1" => {
                stage["vocab"] = a.speech_vocab_size.into();
                stage["width"] = a.input_local_dim.into();
                stage["weights"]["tables"]["count"] = a.audio_channels.into();
            }
            "group.repeat-last.v1" | "merge.consecutive.v1" => stage["rows"] = a.group_size.into(),
            "linear.pairs.v1" => {
                stage["out"] = if sources.contains("mlp.0") {
                    g.hidden() * a.group_size
                } else {
                    g.hidden()
                }
                .into()
            }
            _ => {}
        }
    }
    if t.avg_pooler == 1 {
        audio["stages"]
            .as_array_mut()
            .unwrap()
            .retain(|stage| !stage["weights"].to_string().contains("down_sample"));
    }
    Ok([vision.to_string(), audio.to_string()])
}

/// Whether encoder `encoder`'s declaration admits a media record of
/// `geometry`: some recipe of it can emit that item. The bundle refuses
/// every other record.
pub fn admit_record_flash(encoder: u32, geometry: [u32; 4]) -> Result<(), Error> {
    let declaration = declarations_flash()?
        .get(encoder as usize)
        .ok_or_else(|| Error(format!("MiMo has no encoder {encoder}")))?;
    declaration.admit_record(geometry).map(|_| ())
}

/// The environment MiMo's declarations validate in: its provider.
pub fn environment() -> Result<Environment, Error> {
    let mut environment = Environment::new("mimo26");
    environment.registry.register(&PROVIDER)?;
    Ok(environment)
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    #[test]
    fn payload_limits_admit_hop160_and_nondividing_temporal_patch() {
        for (hop, temporal) in [(240, 2), (160, 2), (240, 3)] {
            let mut geometry = GEOMETRY;
            geometry.id = "payload-policy-test";
            geometry.shape.audio_tokenizer.hop_length = hop;
            geometry.shape.vision.temporal_patch_size = temporal;
            geometry.shape.vision.vit_window_attn_types = &[crate::geometry::VisionLayerKind::Full];
            let mut config: serde_json::Value = serde_json::from_str(geometry.document).unwrap();
            config["vision_config"]["depth"] = 1.into();
            config["vision_config"]["vit_window_attn_types"] = serde_json::json!([-1]);
            config["vision_config"]["fullatt_block_indexes"] = serde_json::json!([0]);
            geometry.document = Box::leak(config.to_string().into_boxed_str());
            let documents = declaration_documents(&geometry).unwrap();
            let environment = environment().unwrap();
            let declarations = documents.map(|document| parse(&document, &environment).unwrap());
            let limits = crate::encoders::payload_limits(&geometry).unwrap();
            for (declaration, limit) in declarations.iter().zip(limits) {
                assert_eq!(
                    limit,
                    declaration
                        .input
                        .recipes
                        .iter()
                        .map(|recipe| {
                            let rows = recipe.rows().unwrap();
                            rows.item * rows.columns * 4
                        })
                        .max()
                        .unwrap()
                );
            }
            let frames = limits[1] / (u64::from(geometry.audio_tokenizer().n_mels) * 4);
            assert!(
                declarations[1]
                    .admit_record([frames as u32, geometry.audio_tokenizer().n_mels, 0, 0])
                    .is_ok()
            );
            assert!(
                declarations[1]
                    .admit_record([frames as u32 + 1, geometry.audio_tokenizer().n_mels, 0, 0])
                    .is_err()
            );
            assert_eq!(
                crate::encoders::maximum_audio_seconds(&geometry).unwrap(),
                if hop == 160 { 400 } else { 600 }
            );
            if temporal == 3 {
                assert_eq!(
                    declarations[0].input.recipes[1].rows().unwrap().item,
                    11_264
                );
            }
        }
        for geometry in crate::geometry::GEOMETRIES {
            let declarations = declarations_for(geometry).unwrap();
            let frames = crate::encoders::payload_limits(geometry).unwrap()[1]
                / (u64::from(geometry.audio_tokenizer().n_mels) * 4);
            assert!(
                declarations[1]
                    .admit_record([frames as u32, geometry.audio_tokenizer().n_mels, 0, 0])
                    .is_ok()
            );
        }
    }

    #[test]
    fn short_vision_and_unpooled_audio_declarations_validate() {
        let mut geometry = GEOMETRY;
        geometry.id = "encoder-test";
        geometry.shape.vision.vit_window_attn_types = &[crate::geometry::VisionLayerKind::Full];
        geometry.shape.audio_tokenizer.avg_pooler = 1;
        let mut config: serde_json::Value = serde_json::from_str(geometry.document).unwrap();
        config["vision_config"]["depth"] = 1.into();
        config["vision_config"]["vit_window_attn_types"] = serde_json::json!([-1]);
        config["vision_config"]["fullatt_block_indexes"] = serde_json::json!([0]);
        config["audio_tokenizer_config"]["avg_pooler"] = 1.into();
        geometry.document = Box::leak(config.to_string().into_boxed_str());
        let documents = declaration_documents(&geometry).unwrap();
        let environment = environment().unwrap();
        for document in &documents {
            parse(document, &environment).unwrap();
        }
        let expected = crate::encoders::expected_for(&geometry);
        assert!(!expected.keys().any(|name| name.contains("down_sample")));
    }
}
