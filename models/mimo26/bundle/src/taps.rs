//! MiMo V2.6's taps (`runtime_bundle_sdk::taps`, declared in
//! `models/mimo26/taps.json`), compiled only into a bundle built with the
//! `diagnostic-taps` feature. With `CHOPINFER_TAPS_DIR` set at create, the
//! instance launches what a serving one does, adding only stream
//! synchronizations and copies: a prefill keeps its schedule, pipelined
//! halves included, and copies each half's rows after its stages; decode
//! and verify run the kernels their graphs hold stage by stage, without
//! graphs. After each stage below it copies these buffers of the step's
//! rows. `tools/qual/tapdiff.py` compares them with the reference runner's
//! taps (`models/mimo26/reference.py`, `taps`).
//!
//! Every layer taps X entering it (`L<nn>.in`), the attention and MLP inputs
//! after their norms (`attn.in`, `mlp.in`), the ranks' partials of the
//! attention and MLP outputs (`attn.out`, `mlp.out`), the routed experts and
//! their weights (`router.ids`, `router.weights`) and X after the layer
//! (`out`); the model taps the token embeddings (`embed.out`) and the final
//! norm (`final.norm`). Layers listed in `CHOPINFER_TAPS_LAYERS` also tap
//! the input of `o_proj` (`attn.core`, each rank's heads). Decode and verify
//! buckets that sum their routed rows inside the MLP all-reduce hold no MLP
//! output, so they do not tap `mlp.out` in MoE layers.
//!
//! A prefill phase also taps the encoder rows each request injects
//! (`enc.vision.out`, `enc.audio.out`, once per inject) on the encoder row
//! axis, read from rank 0's media arena before the phase runs, and the
//! transformer blocks of each media occurrence it begins
//! (`enc.<tower>.B<nn>.out`, and `B<nn>.in` for each transformer's first
//! block; blocks numbered across the tower's transformers), read back by the
//! shared encoder runtime (`Encoder::block_taps`) when the encode runs, each
//! item's rows in their own order. Rows are named by request, media kind and
//! the item's place in its prompt, carried across the prompt's prefill
//! chunks (`bundle_runtime::media::Occurrences`). Each audio inject
//! also taps the RVQ codes its rows merge (`enc.audio.x.codes`), retained
//! with its encode across chunks and shared requests.
//!
//! A serving bundle refuses to create while a taps variable is set
//! (`instance::create_v1`).

#[cfg(test)]
use model_mimo26_contract::geometry::V26_FLASH;

use std::path::Path;

use bundle_runtime::media::{OccurrenceStep, Occurrences};
use runtime_bundle_abi::{BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1};
use runtime_bundle_sdk::taps::{
    EncoderRows, EncoderSegment, TapCompare, TapDType, TapSharding, TapSpec, TapStep, Taps,
};

use crate::inputs::PhaseInputs;
#[cfg(test)]
use crate::program::LAYERS;
use crate::program::Stage;
use crate::scratch::RankScratch;
use crate::selection::Phase;
use crate::topology::Tp;

/// The model name each step's `capture.json` records.
pub const MODEL: &str = "mimo26";

/// The encoder towers whose output rows are tapped, by encoder index.
const TOWERS: [(u32, &str); 2] = [(0, "enc.vision.out"), (1, "enc.audio.out")];

pub(crate) const AUDIO: u32 = 1;
/// The audio tokenizer's RVQ codes, `[codes, STAGES]` u32 per audio item.
const CODES: &str = "enc.audio.x.codes";
pub(crate) const STAGES: u32 = 20;
/// Output row r of an audio encode merges its codes 4r..4r + 3
/// (the declaration's `group.repeat-last.v1` and `merge.consecutive.v1`).
const CODES_PER_ROW: u32 = 4;

fn core_for(tp: Tp, global: bool) -> &'static [[u32; 2]] {
    type GeometryBlocks = (&'static str, [[u32; 2]; 2]);
    static CORE: std::sync::OnceLock<Vec<GeometryBlocks>> = std::sync::OnceLock::new();
    let cores = CORE.get_or_init(|| {
        model_mimo26_contract::geometry::GEOMETRIES
            .iter()
            .map(|g| {
                (
                    g.id,
                    [
                        [g.swa_q_heads(), g.swa_value_head_dim()],
                        [g.attention_q_heads(), g.value_head_dim()],
                    ],
                )
            })
            .collect()
    });
    let (_, blocks) = cores
        .iter()
        .find(|(id, _)| *id == tp.geometry.id)
        .expect("listed geometry");
    &blocks[usize::from(global)..usize::from(global) + 1]
}
pub fn kinds_for(tp: Tp) -> [(u32, &'static str); 3] {
    [
        (tp.geometry.tokens.image, "image"),
        (tp.geometry.tokens.video, "video"),
        (tp.geometry.tokens.audio, "audio"),
    ]
}

/// The taps the environment configures, if any; refuses a filter that
/// selects no tap or detail layers without taps.
pub fn from_env_for(tp: Tp) -> Result<Option<Taps>, String> {
    let Some(taps) = Taps::from_env(None)? else {
        return Ok(None);
    };
    let taps = taps.with_model(MODEL);
    let specs = specs_for(tp);
    taps.check_layers(&specs)?;
    taps.check_sites(&specs)?;
    Ok(Some(taps))
}

/// Every tap the bundle writes, once per name.
fn specs_for(tp: Tp) -> Vec<TapSpec> {
    let mut specs = TOWERS
        .iter()
        .map(|&(_, name)| encoder_for(tp, name))
        .collect::<Vec<_>>();
    specs.extend(crate::encoders::runtime::block_taps_for(tp.geometry));
    specs.push(codes_spec());
    for stage in crate::program::stages_for(tp) {
        specs.extend(
            // Prefill sums its routed rows in the combine.
            taps(stage, tp, false).into_iter().map(|tap| tap.spec),
        );
    }
    specs
}

/// One rank's buffers the taps copy.
pub struct RankBuffers<'a> {
    pub activation: DeviceBufferSpanV1,
    pub normalized: DeviceBufferSpanV1,
    pub scratch: &'a RankScratch,
}

type Buffer = fn(&RankBuffers) -> DeviceBufferSpanV1;

/// One tap and the buffer it copies.
pub struct Tap {
    pub spec: TapSpec,
    pub buffer: Buffer,
}

impl Tap {
    fn new(
        name: String,
        buffer: Buffer,
        cols: u32,
        dtype: TapDType,
        sharding: TapSharding,
    ) -> Self {
        let spec = TapSpec {
            name,
            layer: None,
            detail: false,
            cols,
            dtype,
            sharding,
            compare: TapCompare::Dense,
        };
        Self { spec, buffer }
    }

    /// The layer whose detail taps it is among.
    const fn layer(mut self, layer: u32) -> Self {
        self.spec.layer = Some(layer);
        self
    }

    /// Kept only in the layers `CHOPINFER_TAPS_LAYERS` names.
    const fn detail(mut self) -> Self {
        self.spec.detail = true;
        self
    }

    const fn compare(mut self, compare: TapCompare) -> Self {
        self.spec.compare = compare;
        self
    }
}

/// A BF16 hidden row every rank holds alike.
fn hidden(tp: Tp, name: String, buffer: Buffer) -> Tap {
    Tap::new(
        name,
        buffer,
        tp.geometry.hidden(),
        TapDType::Bf16,
        TapSharding::Replicated,
    )
}

/// A rank's BF16 partial of a hidden row summed over the ranks.
fn partial(tp: Tp, name: String, buffer: Buffer) -> Tap {
    Tap::new(
        name,
        buffer,
        tp.geometry.hidden(),
        TapDType::Bf16,
        TapSharding::RankPartial,
    )
}

/// Eight values per row, every rank's alike.
fn routed(tp: Tp, name: String, buffer: Buffer, dtype: TapDType, compare: TapCompare) -> Tap {
    Tap::new(
        name,
        buffer,
        tp.geometry.top_k(),
        dtype,
        TapSharding::Replicated,
    )
    .compare(compare)
}

fn encoder_for(tp: Tp, name: &str) -> TapSpec {
    hidden(tp, name.into(), |b| b.activation).spec
}

/// The audio codes tap; it reads no rank buffer.
fn codes_spec() -> TapSpec {
    Tap::new(
        CODES.into(),
        |b| b.activation,
        STAGES,
        TapDType::U32,
        TapSharding::Replicated,
    )
    .compare(TapCompare::Codes)
    .spec
}

fn scope(layer: u32) -> String {
    format!("L{layer:02}")
}

/// X entering `layer` and its attention input, which a reduction writes.
fn layer_input(tp: Tp, taps: &mut Vec<Tap>, layer: u32) {
    let name = scope(layer);
    taps.push(hidden(tp, format!("{name}.in"), |b| b.activation).layer(layer));
    taps.push(hidden(tp, format!("{name}.attn.in"), |b| b.normalized).layer(layer));
}

/// The buffers to copy after `stage` at `tp`'s per-rank widths, before the
/// environment's selection. A phase that `folds_combine` sums its
/// routed rows inside the MLP all-reduce (`Ops::folds_combine_in`), so its
/// combine writes no rows to copy.
pub fn taps(stage: Stage, tp: Tp, folds_combine: bool) -> Vec<Tap> {
    let mut taps = Vec::new();
    match stage {
        Stage::Embedding => taps.push(hidden(tp, "embed.out".into(), |b| b.activation)),
        Stage::InputNorm => layer_input(tp, &mut taps, 0),
        Stage::AttentionOutput(layer) => {
            let name = scope(layer);
            let global = tp.geometry.is_global(layer);
            let cols = (tp.q_heads_for(global) * tp.value_dim(global)) as u32;
            let heads = TapSharding::ColShards(core_for(tp, global));
            let core = Tap::new(
                format!("{name}.attn.core"),
                |b| b.scratch.attention,
                cols,
                TapDType::Bf16,
                heads,
            );
            taps.push(core.layer(layer).detail());
            let out = partial(tp, format!("{name}.attn.out"), |b| b.scratch.hidden_partial);
            taps.push(out.layer(layer));
        }
        Stage::AttentionReduceNorm(layer) => {
            let name = format!("{}.mlp.in", scope(layer));
            taps.push(hidden(tp, name, |b| b.normalized).layer(layer));
        }
        Stage::RouterTop8(layer) => {
            let name = format!("{}.router", scope(layer));
            let ids = routed(
                tp,
                format!("{name}.ids"),
                |b| b.scratch.topk_ids,
                TapDType::U32,
                TapCompare::Set,
            );
            taps.push(ids.layer(layer));
            let weights = format!("{name}.weights");
            let weights = routed(
                tp,
                weights,
                |b| b.scratch.topk_weights,
                TapDType::F32,
                TapCompare::Paired,
            );
            taps.push(weights.layer(layer));
        }
        Stage::DenseDown(layer) => {
            let name = format!("{}.mlp.out", scope(layer));
            taps.push(partial(tp, name, |b| b.scratch.hidden_partial).layer(layer));
        }
        Stage::CombineExperts(layer) if !folds_combine => {
            let name = format!("{}.mlp.out", scope(layer));
            taps.push(partial(tp, name, |b| b.scratch.combined).layer(layer));
        }
        Stage::MlpReduceNorm(layer) => {
            let name = format!("{}.out", scope(layer));
            taps.push(hidden(tp, name, |b| b.activation).layer(layer));
            layer_input(tp, &mut taps, layer + 1);
        }
        Stage::FinalReduceNorm => {
            let name = format!("{}.out", scope(tp.geometry.layers() - 1));
            taps.push(hidden(tp, name, |b| b.activation).layer(tp.geometry.layers() - 1));
            taps.push(hidden(tp, "final.norm".into(), |b| b.normalized));
        }
        _ => {}
    }
    taps
}

/// The media kind of each placeholder token (`<|image_pad|>`, `<|video_pad|>`,
/// `<|audio_pad|>` in `compiler/src/frontend.json`).
#[cfg(test)]
pub const KINDS: &[(u32, &str)] = &[
    (V26_FLASH.tokens.image, "image"),
    (V26_FLASH.tokens.video, "video"),
    (V26_FLASH.tokens.audio, "audio"),
];

/// The rows each request injects of the step's encodes of each tower
/// (`OccurrenceStep::encoder_tap`), read from rank 0's `arena` on `stream`.
pub fn encoder_rows_for(
    tp: Tp,
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    arena: DeviceBufferSpanV1,
    step: &OccurrenceStep,
) -> Result<Vec<EncoderRows>, String> {
    let mut rows = Vec::new();
    for (index, name) in TOWERS {
        let (segments, runs) = step.encoder_tap(index);
        if segments.is_empty() {
            continue;
        }
        let row_bytes = tp.geometry.hidden() as u64 * 2;
        let bytes = bundle_runtime::media::read_runs(services, stream, arena, &runs, row_bytes)
            .map_err(|error| error.to_string())?;
        rows.push(EncoderRows {
            spec: encoder_for(tp, name),
            bytes,
            segments,
        });
    }
    Ok(rows)
}

/// The block taps `taps` selects of each occurrence `step` begins: the
/// encoder runtime's block rows of the step's encodes
/// (`Encoder::take_block_rows`) are kept per encode (`Occurrences::keep`),
/// and each occurrence gets its encode's whole rows, in the item's row
/// order, under its request, kind and item, whichever step or request ran
/// the encode: two requests sharing an image, or a request showing it
/// twice, each hold them.
pub fn block_rows_for(
    tp: Tp,
    taps: &Taps,
    occurrences: &mut Occurrences,
    step: &OccurrenceStep,
    rows: Vec<bundle_runtime::encoder::BlockRows>,
) -> Result<Vec<EncoderRows>, String> {
    let mut specs = crate::encoders::runtime::block_taps_for(tp.geometry);
    specs.push(codes_spec());
    let spec = |name: &str| {
        specs
            .iter()
            .find(|spec| spec.name == name)
            .ok_or_else(|| format!("{name} is not a declared block tap"))
    };
    for block in rows {
        let spec = spec(&block.name)?;
        if !taps.selected(spec) {
            continue;
        }
        if block.bytes.len() as u64 % (u64::from(spec.cols) * spec.dtype.bytes()) != 0 {
            return Err(format!("{}: a partial row", block.name));
        }
        let encode = step
            .encodes
            .get(block.record)
            .ok_or_else(|| format!("{}: rows of no encode", block.name))?;
        occurrences.keep(*encode, block.name, block.bytes);
    }
    let mut out: Vec<EncoderRows> = Vec::new();
    for read in step.reads.iter().filter(|read| read.begins) {
        for (name, bytes) in occurrences.kept(read.encode) {
            if name == CODES {
                continue;
            }
            let spec = spec(name)?;
            let rows = bytes.len() as u64 / (u64::from(spec.cols) * spec.dtype.bytes());
            let at = match out.iter().position(|tap| tap.spec.name == *name) {
                Some(at) => at,
                None => {
                    out.push(EncoderRows {
                        spec: spec.clone(),
                        bytes: Vec::new(),
                        segments: Vec::new(),
                    });
                    out.len() - 1
                }
            };
            let tap = &mut out[at];
            let start = tap.rows();
            tap.segments.push(runtime_bundle_sdk::taps::EncoderSegment {
                start,
                index: 0,
                rows: rows as u32,
                ..read.segment
            });
            tap.bytes.extend_from_slice(bytes);
        }
    }
    if taps.selected(&codes_spec()) && step.reads.iter().any(|read| read.encoder == AUDIO) {
        out.push(audio_codes(step, occurrences)?);
    }
    Ok(out)
}

/// The codes each audio inject's rows merge, from the encode's kept codes: per segment of
/// the audio tower, its encode's codes `CODES_PER_ROW * row..`, up to its
/// last real code, as a segment of the same request, kind and item whose
/// index counts codes.
fn audio_codes(step: &OccurrenceStep, occurrences: &Occurrences) -> Result<EncoderRows, String> {
    let row_bytes = STAGES as usize * 4;
    let (mut out, mut bytes) = (Vec::<EncoderSegment>::new(), Vec::new());
    for read in step.reads.iter().filter(|read| read.encoder == AUDIO) {
        let segment = &read.segment;
        let encode = read.encode;
        let row = segment.index;
        let (_, codes) = occurrences
            .kept(encode)
            .iter()
            .find(|(name, _)| name == CODES)
            .ok_or_else(|| format!("audio encode {encode} has no codes read"))?;
        let first = row * CODES_PER_ROW;
        let end = ((row + segment.rows) * CODES_PER_ROW).min((codes.len() / row_bytes) as u32);
        if first >= end {
            return Err(format!("audio encode {encode} rows {row}.. hold no codes"));
        }
        out.push(EncoderSegment {
            start: out.last().map_or(0, |last| last.start + last.rows),
            index: first,
            rows: end - first,
            ..*segment
        });
        bytes.extend_from_slice(&codes[first as usize * row_bytes..end as usize * row_bytes]);
    }
    Ok(EncoderRows {
        spec: codes_spec(),
        bytes,
        segments: out,
    })
}

/// Whether `taps` selects a block tap: the encoder runtime reads blocks
/// back only then.
pub fn selects_blocks_for(tp: Tp, taps: &Taps) -> bool {
    crate::encoders::runtime::block_taps_for(tp.geometry)
        .iter()
        .any(|spec| taps.selected(spec))
}

/// Begins `phase`'s step over `inputs` in `dir`: declares the selected
/// encoder rows and taps of every stage (`folds_combine` as in [`taps`]),
/// and writes the step's metadata and encoder rows.
pub fn begin(
    taps: &Taps,
    dir: &Path,
    phase: Phase,
    tp: Tp,
    folds_combine: bool,
    inputs: &PhaseInputs,
    encoders: Vec<EncoderRows>,
) -> std::io::Result<TapStep> {
    let program = format!("{phase:?}");
    let rows = inputs.live_tokens();
    let mut step = taps.begin(dir, &program, tp.ranks(), rows, &inputs.positions)?;
    for rows in encoders {
        if taps.selected(&rows.spec) {
            step.encoder(rows)?;
        }
    }
    for stage in crate::program::stages_for(tp) {
        for tap in self::taps(stage, tp, folds_combine) {
            if taps.selected(&tap.spec) {
                step.declare(&tap.spec)?;
            }
        }
    }
    step.open(&inputs.tap_items(&program, rows, step.rows(), taps.from()))?;
    Ok(step)
}

#[cfg(test)]
fn specs() -> Vec<TapSpec> {
    specs_for(Tp::TP4)
}
#[cfg(test)]
fn encoder(name: &str) -> TapSpec {
    encoder_for(Tp::TP4, name)
}

#[cfg(test)]
pub fn block_rows(
    taps: &Taps,
    occurrences: &mut Occurrences,
    step: &OccurrenceStep,
    rows: Vec<bundle_runtime::encoder::BlockRows>,
) -> Result<Vec<EncoderRows>, String> {
    block_rows_for(Tp::TP4, taps, occurrences, step, rows)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use runtime_bundle_sdk::taps::{Declaration, TapFilter, TapLayers};

    use super::*;

    fn declaration() -> Declaration {
        Declaration::parse(include_str!("../../taps.json")).unwrap()
    }

    /// Whether an instance's ops fold `phase`'s combine into its MLP
    /// all-reduce (`Ops::folds_combine_in`).
    fn folds(phase: Phase, tp: Tp) -> bool {
        use crate::selection::MimoPhase as _;
        phase != Phase::Prefill
            && phase
                .capacity()
                .is_ok_and(|rows| crate::sm120::folds_combine(tp, rows))
    }

    fn names(taps: &Taps, phase: Phase, tp: Tp) -> Vec<String> {
        crate::program::stages()
            .into_iter()
            .flat_map(|stage| self::taps(stage, tp, folds(phase, tp)))
            .filter(|tap| taps.selected(&tap.spec))
            .map(|tap| tap.spec.name)
            .collect()
    }

    #[test]
    fn taps_use_each_geometry_layers_and_buffer_strides() {
        for tp in crate::topology::all() {
            let scratch = model_mimo26_contract::resources::ScratchLayout::new(tp, 0).unwrap();
            for layer in 0..tp.geometry.layers() {
                let core = self::taps(Stage::AttentionOutput(layer), tp, false)
                    .into_iter()
                    .find(|tap| tap.spec.name.ends_with("attn.core"))
                    .unwrap();
                let global = tp.geometry.is_global(layer);
                assert_eq!(
                    core.spec.cols as u64,
                    tp.q_heads_for(global) * tp.value_dim(global)
                );
                assert!(
                    core.spec.cols as u64 * 2 * crate::contract::PREFILL_TOKENS as u64
                        <= scratch.attention.span_bytes()
                );
                for tap in self::taps(Stage::RouterTop8(layer), tp, false) {
                    assert_eq!(tap.spec.cols, tp.geometry.top_k());
                    assert_eq!(
                        tap.spec.cols as u64 * 4 * crate::contract::PREFILL_TOKENS as u64,
                        scratch.topk_ids.span_bytes()
                    );
                }
            }
            assert!(
                specs_for(tp)
                    .iter()
                    .any(|tap| tap.name == format!("L{:02}.out", tp.geometry.layers() - 1))
            );
        }
    }

    #[test]
    fn every_layer_taps_the_reference_sites_in_program_order() {
        let taps = Taps::new(PathBuf::new(), TapFilter::default());
        let names = names(&taps, Phase::Prefill, Tp::TP4);
        let mut expected = vec!["embed.out".to_owned()];
        for layer in 0..LAYERS {
            let name = scope(layer);
            expected.extend(["in", "attn.in", "attn.out", "mlp.in"].map(|s| format!("{name}.{s}")));
            if layer > 0 {
                expected.extend(["router.ids", "router.weights"].map(|s| format!("{name}.{s}")));
            }
            expected.extend(["mlp.out", "out"].map(|s| format!("{name}.{s}")));
        }
        expected.push("final.norm".into());
        assert_eq!(names, expected);

        // Detail layers add o_proj's input; folded decode buckets have no
        // MoE layer's MLP output.
        let filter = TapFilter {
            layers: Some(TapLayers::List(vec![5])),
            sites: Some(vec!["attn.core".into(), "mlp.out".into()]),
            from: None,
        };
        let taps = Taps::new(PathBuf::new(), filter);
        assert_eq!(
            self::names(&taps, Phase::Decode(1), Tp::TP4),
            ["L00.mlp.out", "L05.attn.core"]
        );
        assert_eq!(
            self::names(&taps, Phase::Prefill, Tp::TP4).len(),
            LAYERS as usize + 1
        );
    }

    #[test]
    fn kinds_name_the_frontends_placeholder_tokens() {
        fn find(value: &serde_json::Value, text: &str) -> Option<u64> {
            match value {
                serde_json::Value::Object(map) if map.get("text") == Some(&text.into()) => {
                    map.get("id").and_then(serde_json::Value::as_u64)
                }
                serde_json::Value::Object(map) => map.values().find_map(|v| find(v, text)),
                serde_json::Value::Array(values) => values.iter().find_map(|v| find(v, text)),
                _ => None,
            }
        }
        let frontend: serde_json::Value =
            serde_json::from_str(include_str!("../../compiler/src/frontend.json")).unwrap();
        for &(token, kind) in KINDS {
            let pad = format!("<|{kind}_pad|>");
            assert_eq!(find(&frontend, &pad), Some(u64::from(token)), "{pad}");
        }
    }

    #[test]
    fn taps_json_declares_every_tap() {
        let declaration = declaration();
        let mut used = vec![false; declaration.entries().len()];
        let all = Taps::new(
            PathBuf::new(),
            TapFilter {
                layers: Some(TapLayers::All),
                ..TapFilter::default()
            },
        );
        for tp in Tp::SUPPORTED {
            for phase in [Phase::Prefill, Phase::Decode(1), Phase::Decode(64)] {
                let specs = crate::program::stages()
                    .into_iter()
                    .flat_map(|stage| taps(stage, tp, folds(phase, tp)))
                    .map(|tap| tap.spec)
                    .chain(TOWERS.map(|(_, name)| encoder(name)))
                    .chain(crate::encoders::runtime::block_taps())
                    .chain([codes_spec()]);
                for spec in specs {
                    assert!(all.selected(&spec));
                    used[declaration.check(&spec, tp.ranks()).unwrap()] = true;
                }
            }
        }
        for (entry, used) in declaration.entries().iter().zip(used) {
            assert!(used, "{} matches no tap", entry["name"]);
        }
    }

    #[test]
    fn filters_that_select_nothing_are_refused() {
        let filter = |sites: &[&str], layers| TapFilter {
            layers,
            sites: Some(sites.iter().map(|site| site.to_string()).collect()),
            from: None,
        };
        let check = |sites, layers| {
            let taps = Taps::new(PathBuf::new(), filter(sites, layers));
            taps.check_layers(&specs())
                .and_then(|()| taps.check_sites(&specs()))
        };
        assert_eq!(check(&["out", "enc.vision.out", "router.*"], None), Ok(()));
        assert_eq!(
            check(&["out", "x.hc.*"], None),
            Err("CHOPINFER_TAPS_SITES selects no tap: x.hc.*".into())
        );
        assert_eq!(
            check(&["attn.core"], None),
            Err("CHOPINFER_TAPS_SITES selects only detail taps of layers \
                 CHOPINFER_TAPS_LAYERS does not keep: attn.core"
                .into())
        );
        assert_eq!(
            check(&["attn.core"], Some(TapLayers::List(vec![47]))),
            Ok(())
        );
        assert_eq!(
            check(&["out"], Some(TapLayers::List(vec![47, 48]))),
            Err("CHOPINFER_TAPS_LAYERS names layers without taps: 48".into())
        );
    }
}
