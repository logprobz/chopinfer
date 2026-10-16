//! Qwen3.8's taps (`runtime_bundle_sdk::taps`, declared in
//! `models/qwen38/taps.json`): with `CHOPINFER_TAPS_DIR` or
//! `CHOPINFER_QWEN38_DUMP_DIR` set at create, every phase runs eagerly and
//! copies these buffers after their stages. `tools/qual/tapdiff.py` and
//! `tools/qual/qwen38_layers.py` compare them with the references.
//!
//! Every layer taps X entering it (`L<nn>.in.streams`), its sublayer inputs
//! (`x.hc.attn`, `x.hc.mlp`), the rank partials of its sublayer outputs
//! (`attn.out`, `mlp.out`), the router logits and X after the layer
//! (`out`). Layers listed in `CHOPINFER_TAPS_LAYERS` (alias
//! `CHOPINFER_QWEN38_DUMP_OPS`) also tap each op's output. Full-attention
//! layers tap each row's selected QSA blocks (`x.indexer.topk`, U32,
//! `TOP_BLOCKS` per row; rows that keep every block list them in their own
//! layout). The other aliases are `CHOPINFER_QWEN38_DUMP_TAPS` (sites),
//! `_FROM`, `_BYTES` and `_SCOPE_JSON`. A prefill phase also taps the
//! merged vision rows each request injects (`enc.vision.out`) on the encoder
//! row axis, read from rank 0's media arena before the phase runs, named by
//! request, media kind and the item's place in its prompt, carried across
//! the prompt's prefill chunks (`bundle_runtime::media::Occurrences`).
//!
//! Only a bundle built with the `diagnostic-taps` feature captures; a
//! serving bundle refuses to create while either directory variable is set.
//! `CHOPINFER_TAPS_SITES` takes patterns of the sites or names above; the
//! older `CHOPINFER_QWEN38_DUMP_TAPS` takes suffixes of the taps' older
//! names (`v1` in `taps.json`, such as `out.x` or `attn.selected`) and
//! selects the taps it selected before. A value that selects no tap, and
//! detail layers that are not a list of the model's layers, refuse the
//! create.

use runtime_bundle_abi::{BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1};
use runtime_bundle_sdk::taps::{
    Declaration, EncoderRows, TapCompare, TapDType, TapEnv, TapSharding, TapSpec, TapStep, Taps,
};

use crate::contract::*;
use crate::inputs::PhaseInputs;
use crate::lower::residual_index_for;
use crate::program::{Phase, Stage, Sublayer};
use crate::scratch::RankScratch;

/// The Qwen names of the taps' environment.
pub const ENV: TapEnv = TapEnv {
    dir: "CHOPINFER_QWEN38_DUMP_DIR",
    layers: "CHOPINFER_QWEN38_DUMP_OPS",
    sites: "CHOPINFER_QWEN38_DUMP_TAPS",
    from: "CHOPINFER_QWEN38_DUMP_FROM",
    bytes: "CHOPINFER_QWEN38_DUMP_BYTES",
    scope_json: "CHOPINFER_QWEN38_DUMP_SCOPE_JSON",
};

/// The taps the environment configures, if any; in a serving bundle, an
/// error naming the variable that asks for them.
pub fn from_env_for(shape: Shape) -> Result<Option<Taps>, String> {
    if !cfg!(feature = "diagnostic-taps")
        && let Some(variable) = Taps::requested(Some(ENV))
    {
        return Err(format!(
            "{variable} is set, but this Qwen3.8 bundle was built without the \
             diagnostic-taps feature; unset it to serve"
        ));
    }
    let Some(taps) = Taps::from_env(Some(ENV))? else {
        return Ok(None);
    };
    let taps = taps.with_model("qwen38");
    let specs = specs_for(shape);
    taps.check_layers(&specs)?;
    if taps.sites_variable() != Some(ENV.sites) {
        taps.check_sites(&specs)?;
        return Ok(Some(taps));
    }
    let sites = older_sites(&taps, &specs)?;
    Ok(Some(taps.with_sites(sites)))
}

/// Every tap the bundle writes, once per name.
fn specs_for(shape: Shape) -> Vec<TapSpec> {
    let mut specs = std::collections::BTreeMap::new();
    specs.insert(ENCODER.into(), encoder_for(shape));
    for phase in [Phase::Prefill, Phase::Decode(1)] {
        for stage in crate::program::stages_for(shape, phase) {
            for tap in taps(stage, shape) {
                specs.entry(tap.spec.name.clone()).or_insert(tap.spec);
            }
        }
    }
    specs.into_values().collect()
}

/// The names of the taps whose older names end with one of the older sites
/// variable's suffixes, as it selected them; refuses suffixes that select
/// no tap, or only detail taps of layers the layers filter does not keep.
fn older_sites(taps: &Taps, specs: &[TapSpec]) -> Result<Vec<String>, String> {
    let suffixes = taps.sites().unwrap_or_default();
    let older = specs
        .iter()
        .filter_map(|spec| v1_name(&spec.name).map(|v1| (spec, v1)))
        .collect::<Vec<_>>();
    let refused = |keep: &dyn Fn(&TapSpec) -> bool| {
        suffixes
            .iter()
            .filter(|suffix| {
                !older
                    .iter()
                    .any(|(spec, v1)| keep(spec) && v1.ends_with(suffix.as_str()))
            })
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let unmatched = refused(&|_| true);
    if !unmatched.is_empty() {
        return Err(format!("{} selects no tap: {unmatched}", ENV.sites));
    }
    let unkept = refused(&|spec| taps.kept(spec));
    if !unkept.is_empty() {
        return Err(format!(
            "{} selects only detail taps of layers {} does not keep: {unkept}",
            ENV.sites,
            taps.layers_variable()
        ));
    }
    Ok(older
        .into_iter()
        .filter(|(_, v1)| suffixes.iter().any(|suffix| v1.ends_with(suffix.as_str())))
        .map(|(spec, _)| spec.name.clone())
        .collect())
}

/// `taps.json`.
fn declaration() -> &'static Declaration {
    static DECLARATION: std::sync::OnceLock<Declaration> = std::sync::OnceLock::new();
    DECLARATION
        .get_or_init(|| Declaration::parse(include_str!("../../taps.json")).expect("taps.json"))
}

/// Tap `name`'s name in older (`chopinfer.qwen38.capture.v1`) dumps.
fn v1_name(name: &str) -> Option<String> {
    let (index, layer) = declaration().entry(name)?;
    let v1 = declaration().entries()[index]["v1"].as_str()?;
    Some(match layer {
        Some(layer) => v1.replace("{nn}", &format!("{layer:02}")),
        None => v1.to_owned(),
    })
}

struct Blocks {
    qkv: [[u32; 2]; 3],
    linear: [[u32; 2]; 1],
    attention: [[u32; 2]; 1],
}

impl Blocks {
    fn of(g: &model_qwen38_contract::geometry::Geometry) -> Self {
        Self {
            qkv: [
                [g.linear_key_heads(), g.linear_head()],
                [g.linear_key_heads(), g.linear_head()],
                [g.linear_value_heads(), g.linear_value_head()],
            ],
            linear: [[g.linear_value_heads(), g.linear_value_head()]],
            attention: [[g.attention_q_heads(), g.head_dim()]],
        }
    }
}

fn blocks(shape: Shape) -> &'static Blocks {
    static BLOCKS: std::sync::OnceLock<Vec<std::sync::OnceLock<Blocks>>> =
        std::sync::OnceLock::new();
    crate::contract::per_geometry(&BLOCKS, shape, Blocks::of)
}

type Buffer = fn(&RankScratch) -> DeviceBufferSpanV1;

/// One tap and the buffer it copies.
#[derive(Clone, Debug)]
pub struct Tap {
    pub spec: TapSpec,
    pub buffer: Buffer,
}

impl Tap {
    /// Kept only in the layers `CHOPINFER_TAPS_LAYERS` names.
    fn detail(mut self) -> Self {
        self.spec.detail = true;
        self
    }

    fn compare(mut self, compare: TapCompare) -> Self {
        self.spec.compare = compare;
        self
    }
}

fn tap(name: String, buffer: Buffer, cols: u32, dtype: TapDType, sharding: TapSharding) -> Tap {
    Tap {
        spec: TapSpec {
            name,
            layer: None,
            detail: false,
            cols,
            dtype,
            sharding,
            compare: TapCompare::Dense,
        },
        buffer,
    }
}

/// A BF16 buffer every rank holds alike.
fn bf16(name: String, buffer: Buffer, cols: u32) -> Tap {
    tap(name, buffer, cols, TapDType::Bf16, TapSharding::Replicated)
}

/// A BF16 buffer of each rank.
fn shard(name: String, buffer: Buffer, cols: u32, sharding: TapSharding) -> Tap {
    tap(name, buffer, cols, TapDType::Bf16, sharding)
}

/// Head-sharded columns: each rank's share of `blocks`, or a per-rank
/// layout of its own when there are none.
fn heads(name: String, buffer: Buffer, cols: u32, blocks: &'static [[u32; 2]]) -> Tap {
    shard(name, buffer, cols, TapSharding::ColShards(blocks))
}

/// A rank's partial of a sum over the ranks.
fn partial(shape: Shape, name: String, buffer: Buffer) -> Tap {
    shard(name, buffer, shape.hidden(), TapSharding::RankPartial)
}

/// Residual buffer `index`.
fn x(index: usize) -> Buffer {
    if index == 0 { |s| s.x[0] } else { |s| s.x[1] }
}

fn scope(layer: u32) -> String {
    format!("L{layer:02}")
}

/// The encoder tap: the vision tower's merged rows.
const ENCODER: &str = "enc.vision.out";

fn encoder_for(shape: Shape) -> TapSpec {
    bf16(ENCODER.into(), |s| s.u, shape.geometry.vision_out()).spec
}

/// The media kind of each placeholder token (`<|image_pad|>` and
/// `<|video_pad|>` in `compiler/src/frontend.json`).
#[cfg(test)]
pub const KINDS: &[(u32, &str)] = &[(248_056, "image"), (248_057, "video")];

pub fn kinds_for(shape: Shape) -> &'static [(u32, &'static str)] {
    type Kinds = [(u32, &'static str); 2];
    static KINDS: std::sync::OnceLock<Vec<std::sync::OnceLock<Kinds>>> = std::sync::OnceLock::new();
    crate::contract::per_geometry::<Kinds>(&KINDS, shape, |g| {
        [
            (g.shape.tokens.image, "image"),
            (g.shape.tokens.video, "video"),
        ]
    })
}

/// The merged rows each request injects of image and video encodes
/// (`OccurrenceStep::encoder_tap`), read from rank 0's `arena` on `stream`;
/// none without an inject.
pub fn encoder_rows(
    shape: Shape,
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    arena: DeviceBufferSpanV1,
    step: &bundle_runtime::media::OccurrenceStep,
) -> Result<Vec<EncoderRows>, String> {
    let (segments, runs) = step.encoder_tap(0);
    if segments.is_empty() {
        return Ok(Vec::new());
    }
    let row_bytes = u64::from(shape.geometry.vision_out()) * 2;
    let bytes = bundle_runtime::media::read_runs(services, stream, arena, &runs, row_bytes)
        .map_err(|error| error.to_string())?;
    Ok(vec![EncoderRows {
        spec: encoder_for(shape),
        bytes,
        segments,
    }])
}

/// Begins `phase`'s step in `dir`: declares the selected encoder rows and
/// taps of every stage, and writes the step's metadata and encoder rows.
pub fn begin(
    taps: &Taps,
    dir: &std::path::Path,
    phase: Phase,
    rows: u32,
    inputs: &PhaseInputs,
    shape: Shape,
    encoders: Vec<EncoderRows>,
) -> std::io::Result<TapStep> {
    let program = format!("{phase:?}");
    let mut step = taps.begin(dir, &program, shape.ranks(), rows, &inputs.positions)?;
    for rows in encoders {
        if taps.selected(&rows.spec) {
            step.encoder(rows)?;
        }
    }
    for stage in crate::program::stages_for(shape, phase) {
        for tap in self::taps(stage, shape) {
            if taps.selected(&tap.spec) {
                step.declare(&tap.spec)?;
            }
        }
    }
    step.open(&inputs.tap_items(&program, rows, step.rows(), taps.from()))?;
    Ok(step)
}

/// The buffers to copy after `stage`, with the per-rank widths of `shape`,
/// before the environment's selection.
pub fn taps(stage: Stage, shape: Shape) -> Vec<Tap> {
    let mut taps = Vec::new();
    match stage {
        Stage::ExpandStreams => taps.push(bf16("embed.out".into(), x(0), shape.wide())),
        Stage::PleResidual => {
            let input = residual_index_for(shape, shape.ple_layer(), Sublayer::Attention) ^ 1;
            let name = scope(shape.ple_layer());
            taps.push(bf16(format!("{name}.in.streams"), x(input), shape.wide()));
            taps.push(bf16(
                format!("{name}.x.ple.out"),
                x(input ^ 1),
                shape.wide(),
            ));
            taps.push(
                bf16(
                    format!("{name}.x.ple.ngram"),
                    |s| s.ngram,
                    shape.ple_embed_dim(),
                )
                .detail(),
            );
            taps.push(bf16(format!("{name}.x.ple.gated"), |s| s.ple_gated, shape.wide()).detail());
            taps.push(bf16(format!("{name}.x.ple.conv"), |s| s.ple_conv, shape.wide()).detail());
        }
        Stage::HcIn(layer, sublayer) => {
            let name = format!("{}.x.hc.{}", scope(layer), sublayer_name(sublayer));
            taps.push(bf16(format!("{name}.xn"), |s| s.xn, shape.wide()).detail());
            taps.push(bf16(format!("{name}.inject"), |s| s.hc_inject, shape.streams()).detail());
        }
        Stage::HcMix(layer, sublayer) => {
            let name = scope(layer);
            if sublayer == Sublayer::Attention {
                // The PLE layer's input is before the PLE add; X here is after it.
                let site = if layer == shape.ple_layer() {
                    "x.hc.attn.streams"
                } else {
                    "in.streams"
                };
                let input = x(residual_index_for(shape, layer, sublayer));
                taps.push(bf16(format!("{name}.{site}"), input, shape.wide()));
            }
            let sub = sublayer_name(sublayer);
            taps.push(bf16(format!("{name}.x.hc.{sub}"), |s| s.u, shape.hidden()));
        }
        Stage::LinearInProj(layer) => {
            let name = format!("{}.x.gdn", scope(layer));
            let qkv = shape.linear_qkv;
            taps.push(
                heads(
                    format!("{name}.qkv"),
                    |s| s.lin_qkv,
                    qkv,
                    &blocks(shape).qkv,
                )
                .detail(),
            );
            let z = shape.linear_value;
            taps.push(heads(format!("{name}.z"), |s| s.lin_z, z, &blocks(shape).linear).detail());
            let ab = shape.linear_ab_stride;
            taps.push(heads(format!("{name}.ab"), |s| s.lin_ab, ab, &[]).detail());
        }
        Stage::LinearConv(layer) => {
            let name = format!("{}.x.gdn.conv", scope(layer));
            taps.push(heads(name, |s| s.lin_conv, shape.linear_qkv, &blocks(shape).qkv).detail());
        }
        Stage::LinearDelta(layer) => {
            let name = format!("{}.x.gdn.core", scope(layer));
            taps.push(
                heads(
                    name,
                    |s| s.lin_core,
                    shape.linear_value,
                    &blocks(shape).linear,
                )
                .detail(),
            );
        }
        Stage::LinearNorm(layer) => {
            let name = format!("{}.attn.core", scope(layer));
            let normed = |s: &RankScratch| s.lin_normed;
            taps.push(heads(name, normed, shape.linear_value, &blocks(shape).linear).detail());
        }
        Stage::AttnInProj(layer) => {
            let name = scope(layer);
            let qkv = format!("{name}.x.attn.qkv");
            taps.push(heads(qkv, |s| s.attn_qkv, shape.attn_row, &[]).detail());
            let index = format!("{name}.x.indexer.qk");
            taps.push(
                shard(
                    index,
                    |s| s.attn_index,
                    shape.index_row(),
                    TapSharding::RankCopies,
                )
                .detail(),
            );
        }
        Stage::AttnQkNormRopeKv(layer) => {
            let name = format!("{}.x.attn.q_rope", scope(layer));
            taps.push(
                heads(
                    name,
                    |s| s.attn_q,
                    shape.attn_value,
                    &blocks(shape).attention,
                )
                .detail(),
            );
        }
        Stage::BlockSelect(layer) => taps.push(
            tap(
                format!("{}.x.indexer.topk", scope(layer)),
                |s| s.selected,
                shape.top_blocks(),
                TapDType::U32,
                TapSharding::RankCopies,
            )
            .compare(TapCompare::Set),
        ),
        Stage::SparseAttention(layer) => {
            let name = format!("{}.x.attn.ungated", scope(layer));
            taps.push(
                heads(
                    name,
                    |s| s.attn_out,
                    shape.attn_value,
                    &blocks(shape).attention,
                )
                .detail(),
            );
        }
        Stage::AttnGate(layer) => {
            let name = format!("{}.attn.core", scope(layer));
            taps.push(
                heads(
                    name,
                    |s| s.attn_gated,
                    shape.attn_value,
                    &blocks(shape).attention,
                )
                .detail(),
            );
        }
        Stage::LinearOut(layer) | Stage::AttnOut(layer) => {
            taps.push(partial(shape, format!("{}.attn.out", scope(layer)), |s| {
                s.partial
            }));
        }
        Stage::Router(layer) => taps.push(tap(
            format!("{}.router.logits", scope(layer)),
            |s| s.router_logits,
            shape.experts(),
            TapDType::F32,
            TapSharding::Replicated,
        )),
        Stage::Route(layer) => {
            let name = format!("{}.router", scope(layer));
            let replicated = TapSharding::Replicated;
            let ids = tap(
                format!("{name}.ids"),
                |s| s.topk_ids,
                shape.top_k(),
                TapDType::U32,
                replicated,
            );
            taps.push(ids.compare(TapCompare::Set).detail());
            let weights = format!("{name}.weights");
            let weights = tap(
                weights,
                |s| s.topk_weights,
                shape.top_k(),
                TapDType::F32,
                replicated,
            );
            taps.push(weights.compare(TapCompare::Paired).detail());
        }
        Stage::ExpertsGateUp(layer) => taps.push(
            shard(
                format!("{}.x.moe.intermediate", scope(layer)),
                |s| s.moe_intermediate,
                shape.top_k() * shape.expert_intermediate(),
                TapSharding::ExpertPartial,
            )
            .detail(),
        ),
        Stage::ExpertsDown(layer) => taps.push(
            shard(
                format!("{}.x.moe.routed", scope(layer)),
                |s| s.moe_routed,
                shape.top_k() * shape.hidden(),
                TapSharding::ExpertPartial,
            )
            .detail(),
        ),
        Stage::SharedGateUp(layer) => taps.push(
            heads(
                format!("{}.x.moe.shared_intermediate", scope(layer)),
                |s| s.shared_intermediate,
                shape.shared_intermediate,
                &[],
            )
            .detail(),
        ),
        Stage::SharedDown(layer) => {
            taps.push(
                partial(shape, format!("{}.moe.shared", scope(layer)), |s| {
                    s.shared_out
                })
                .detail(),
            );
        }
        Stage::Combine(layer) => {
            taps.push(partial(shape, format!("{}.mlp.out", scope(layer)), |s| {
                s.partial
            }));
        }
        Stage::Reduce(layer, Sublayer::Mlp) => taps.push(bf16(
            format!("{}.out", scope(layer)),
            x(residual_index_for(shape, layer, Sublayer::Mlp) ^ 1),
            shape.wide(),
        )),
        Stage::MixerMix => taps.push(bf16("final.x.hc.mixer".into(), |s| s.u, shape.hidden())),
        _ => {}
    }
    for tap in &mut taps {
        tap.spec.layer = stage.layer_for(shape);
    }
    taps
}

const fn sublayer_name(sublayer: Sublayer) -> &'static str {
    match sublayer {
        Sublayer::Attention => "attn",
        Sublayer::Mlp => "mlp",
    }
}

#[cfg(test)]
fn specs() -> Vec<TapSpec> {
    specs_for(Shape::TP4)
}

/// Extra byte captures for the overlap qualification job. These are outside
/// the reference tap declaration and are written only when explicitly asked
/// for on a diagnostic instance. New KV pages are zeroed by the host, so
/// page padding has defined bytes; recurrent captures include only live items.
/// PLE includes its convolution rows and two history words, excluding padding.
pub struct OverlapEvidence<'a> {
    pub services: &'a HostServicesV1,
    pub streams: &'a [BundleStreamHandleV1],
    pub shape: Shape,
    pub inputs: &'a PhaseInputs,
    pub entries: &'a [crate::side::ItemEntries],
    pub scratch: &'a [RankScratch],
    pub kv: &'a [DeviceBufferSpanV1],
    pub rings: &'a [DeviceBufferSpanV1],
    pub ring: Ring,
    pub pool_entries: u64,
    pub page_bytes: u64,
    pub head_rows: u32,
}

impl OverlapEvidence<'_> {
    pub fn write(&self, root: &std::path::Path) -> Result<(), String> {
        use std::io::Write;
        let dir = root.join("overlap-evidence");
        std::fs::create_dir(&dir).map_err(|error| error.to_string())?;
        runtime_bundle_sdk::diagnostic::write_new(
            &dir.join("rows.json"),
            &serde_json::json!({
                "tokens": self.inputs.token_ids, "positions": self.inputs.positions,
                "prefix": self.inputs.prefix_len, "lengths": self.inputs.q_len,
                "logit_rows": self.inputs.logit_rows(), "head_rows": self.head_rows,
                "ranks": self.shape.ranks(), "geometry": self.shape.geometry.id,
            "prompt_rows": self.inputs.prompt_rows, "emits_row": self.inputs.emits_row,
            "anchors": self.entries.iter().map(|entry| entry.anchor.map(|anchor| anchor.end)).collect::<Vec<_>>(),
            "pages": self.inputs.block_tables.iter().map(Vec::len).collect::<Vec<_>>(),
            "page_bytes": self.page_bytes, "vocab_per_rank": self.shape.vocab_per_rank,
            "full_layers": self.shape.full_layers(), "linear_layers": self.shape.linear_layers(),
            "ple_bytes": self.shape.ple_conv_bytes() + 8, "conv_bytes": self.shape.conv_entry_bytes,
            "delta_bytes": self.shape.delta_entry_bytes,
            "ring_key_bytes": self.shape.ring_slot_key_bytes(),
            "ring_position_bytes": self.shape.ring_slot_position_bytes(),
            }),
        )
        .map_err(|error| error.to_string())?;
        let span = |buffer: DeviceBufferSpanV1, offset: u64, length: u64| {
            if offset
                .checked_add(length)
                .is_none_or(|end| end > buffer.length)
            {
                return Err("overlap evidence view exceeds its buffer".to_owned());
            }
            Ok(DeviceBufferSpanV1::new(
                buffer.buffer,
                buffer.offset + offset,
                length,
            ))
        };
        for (rank, &stream) in self.streams.iter().enumerate() {
            self.services
                .synchronize_stream(stream)
                .map_err(|error| error.to_string())?;
            let write = |name: String, source: DeviceBufferSpanV1| -> Result<(), String> {
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(dir.join(format!("{name}.r{rank}.bin")))
                    .map_err(|error| error.to_string())?;
                // Bound each host allocation and stream large logits captures.
                let mut at = 0;
                while at < source.length {
                    let count = (source.length - at).min(16 << 20);
                    let mut bytes = vec![0; count as usize];
                    self.services
                        .copy_device_to_host(stream, &mut bytes, span(source, at, count)?)
                        .map_err(|error| error.to_string())?;
                    self.services
                        .synchronize_stream(stream)
                        .map_err(|error| error.to_string())?;
                    file.write_all(&bytes).map_err(|error| error.to_string())?;
                    at += count;
                }
                Ok(())
            };
            let scratch = &self.scratch[rank];
            write(
                "logits".into(),
                span(
                    scratch.local_scored,
                    0,
                    u64::from(self.head_rows) * u64::from(self.shape.vocab_per_rank) * 4,
                )?,
            )?;
            for (item, entry) in self.entries.iter().enumerate() {
                for (page, &id) in self.inputs.block_tables[item].iter().enumerate() {
                    let offset =
                        u64::try_from(id).map_err(|_| "negative KV page")? * self.page_bytes;
                    write(
                        format!("kv.i{item}.p{page}"),
                        span(self.kv[rank], offset, self.page_bytes)?,
                    )?;
                }
                write(
                    format!("ple.i{item}"),
                    span(
                        scratch.ple_staging[0].output,
                        item as u64 * self.shape.ple_entry_bytes(),
                        self.shape.ple_conv_bytes() + 8,
                    )?,
                )?;
                if let Some(anchor) = entry.anchor {
                    let anchor_slot =
                        u64::try_from(anchor.slot).map_err(|_| "negative anchor slot")?;
                    let anchor_entry =
                        u64::try_from(anchor.entry).map_err(|_| "negative anchor entry")?;
                    write(
                        format!("anchor-ple.i{item}"),
                        span(
                            scratch.anchor_ple,
                            anchor_slot * self.shape.ple_entry_bytes(),
                            self.shape.ple_conv_bytes() + 8,
                        )?,
                    )?;
                    let ring = crate::scratch::anchor_ring(self.shape);
                    for layer in 0..self.shape.full_layers() {
                        let base = layer * ring.layer_bytes();
                        write(
                            format!("anchor-ring-key.l{layer}.i{item}"),
                            span(
                                scratch.anchor_ring,
                                base + anchor_slot * self.shape.ring_slot_key_bytes(),
                                self.shape.ring_slot_key_bytes(),
                            )?,
                        )?;
                        write(
                            format!("anchor-ring-position.l{layer}.i{item}"),
                            span(
                                scratch.anchor_ring,
                                base + ring.key_bytes()
                                    + anchor_slot * self.shape.ring_slot_position_bytes(),
                                self.shape.ring_slot_position_bytes(),
                            )?,
                        )?;
                    }
                    for layer in 0..self.shape.linear_layers() {
                        for (name, pool, bytes) in [
                            ("conv", scratch.conv_pool, self.shape.conv_entry_bytes),
                            ("delta", scratch.delta_pool, self.shape.delta_entry_bytes),
                        ] {
                            write(
                                format!("anchor-{name}.l{layer}.i{item}"),
                                span(
                                    pool,
                                    (layer * self.pool_entries + anchor_entry) * bytes,
                                    bytes,
                                )?,
                            )?;
                        }
                    }
                }
                let slot =
                    u64::try_from(self.inputs.slot_ids[item]).map_err(|_| "negative state slot")?;
                for layer in 0..self.shape.full_layers() {
                    let base = layer * self.ring.layer_bytes();
                    write(
                        format!("ring-key.l{layer}.i{item}"),
                        span(
                            self.rings[rank],
                            base + slot * self.shape.ring_slot_key_bytes(),
                            self.shape.ring_slot_key_bytes(),
                        )?,
                    )?;
                    write(
                        format!("ring-position.l{layer}.i{item}"),
                        span(
                            self.rings[rank],
                            base + self.ring.key_bytes()
                                + slot * self.shape.ring_slot_position_bytes(),
                            self.shape.ring_slot_position_bytes(),
                        )?,
                    )?;
                }
                let working = u64::try_from(entry.working).map_err(|_| "negative working entry")?;
                for layer in 0..self.shape.linear_layers() {
                    for (name, pool, bytes) in [
                        ("conv", scratch.conv_pool, self.shape.conv_entry_bytes),
                        ("delta", scratch.delta_pool, self.shape.delta_entry_bytes),
                    ] {
                        write(
                            format!("{name}.l{layer}.i{item}"),
                            span(pool, (layer * self.pool_entries + working) * bytes, bytes)?,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use runtime_bundle_sdk::taps::{TapFilter, TapLayers};

    use super::*;
    use crate::program::stages;

    fn selected(taps: &Taps, phase: Phase, shape: Shape) -> Vec<String> {
        stages(phase)
            .into_iter()
            .flat_map(|stage| self::taps(stage, shape))
            .filter(|tap| taps.selected(&tap.spec))
            .map(|tap| tap.spec.name)
            .collect()
    }

    #[test]
    fn linear_value_blocks_use_the_value_dimension() {
        let mut geometry = model_qwen38_contract::geometry::FLASH;
        geometry.shape.linear.value_dim = 64;
        let blocks = Blocks::of(&geometry);
        assert_eq!(blocks.qkv[0][1], geometry.linear_head());
        assert_eq!(blocks.qkv[2][1], 64);
        assert_eq!(blocks.linear[0][1], 64);
    }

    #[test]
    fn every_geometry_tap_stride_fits_its_scratch_rows() {
        use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};
        for g in model_qwen38_contract::geometry::GEOMETRIES {
            for ranks in [Ranks::TP2, Ranks::TP4] {
                let shape = Shape::of_geometry(g, ranks).unwrap();
                let served = crate::scratch::Served::new(Items::new(32).unwrap());
                let scratch = RankScratch::allocate(
                    &crate::instance::target(shape, false).comm_bytes(served),
                    0,
                    Default::default(),
                    shape.page_bytes_bf16_mtp(),
                    shape,
                    served,
                    |bytes| {
                        Ok(DeviceBufferSpanV1::new(
                            BundleDeviceBufferHandleV1::from_raw(1),
                            0,
                            bytes,
                        ))
                    },
                )
                .unwrap();
                assert!(!specs_for(shape).is_empty());
                for stage in [
                    Stage::Route(0),
                    Stage::ExpertsGateUp(0),
                    Stage::ExpertsDown(0),
                ] {
                    for tap in self::taps(stage, shape) {
                        let width = match tap.spec.dtype {
                            TapDType::Bf16 => 2,
                            TapDType::U32 | TapDType::F32 => 4,
                        };
                        assert_eq!(
                            (tap.buffer)(&scratch).length,
                            u64::from(PREFILL_TOKENS) * u64::from(tap.spec.cols) * width,
                            "{} {}",
                            g.id,
                            tap.spec.name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_layer_taps_its_inputs_partials_and_output() {
        let filter = TapFilter {
            layers: Some(TapLayers::List(vec![3])),
            ..TapFilter::default()
        };
        let taps = Taps::new(PathBuf::new(), filter);
        let names = selected(&taps, Phase::Prefill, Shape::TP4);
        for layer in 0..LAYERS {
            let name = scope(layer);
            let input = if layer == PLE_LAYER {
                "x.hc.attn.streams"
            } else {
                "in.streams"
            };
            for site in [
                input,
                "x.hc.attn",
                "attn.out",
                "x.hc.mlp",
                "router.logits",
                "mlp.out",
                "out",
            ] {
                assert!(names.contains(&format!("{name}.{site}")), "{name}.{site}");
            }
        }
        assert!(names.contains(&"L03.x.attn.q_rope".into()));
        assert!(!names.contains(&"L04.x.gdn.core".into()));
        assert!(names.contains(&"L01.in.streams".into()));
        assert!(names.contains(&"L01.x.ple.out".into()));
        assert!(!names.contains(&"L01.x.ple.conv".into()));
        assert!(names.contains(&"L03.x.indexer.topk".into()));
        assert!(!names.contains(&"L04.x.indexer.topk".into()));
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "a name is tapped twice");
        assert_eq!(
            names
                .iter()
                .filter(|name| *name == "final.x.hc.mixer")
                .count(),
            1
        );

        let filter = TapFilter {
            layers: Some(TapLayers::List(vec![3])),
            sites: Some(vec!["out".into(), "x.indexer.topk".into()]),
            from: None,
        };
        let kept = selected(
            &Taps::new(PathBuf::new(), filter),
            Phase::Prefill,
            Shape::TP4,
        );
        // embed.out, X after every layer and the key sets of the 12
        // full-attention layers.
        assert_eq!(kept.len(), 1 + LAYERS as usize + LAYERS as usize / 4);
    }

    #[test]
    fn steps_record_first_row_padding_and_completion() {
        let root = std::env::temp_dir().join(format!("qwen-dump-metadata-{}", std::process::id()));
        let filter = TapFilter {
            sites: Some(vec!["out".into()]),
            from: Some(1),
            ..TapFilter::default()
        };
        let mut taps = Taps::new(root.clone(), filter).with_model("qwen38");
        let inputs = PhaseInputs {
            token_ids: vec![10, 11, 12, 20, 21],
            positions: vec![0, 1, 2, 0, 1],
            request_ids: vec![9, 7],
            slot_ids: vec![2, 0],
            q_start: vec![0, 3],
            q_len: vec![3, 2],
            prefix_len: vec![0, 0],
            seq_lens: vec![3, 2],
            token_batch: vec![0, 0, 0, 1, 1],
            item_order: vec![1, 0],
            ..PhaseInputs::default()
        };
        let dir = taps.next_step().unwrap();
        let mut step = begin(
            &taps,
            &dir,
            Phase::Prefill,
            8,
            &inputs,
            Shape::TP4,
            Vec::new(),
        )
        .unwrap();
        let read = |name: &str| -> serde_json::Value {
            serde_json::from_slice(&fs::read(dir.join(name)).unwrap()).unwrap()
        };
        let items = read("items.json");
        assert_eq!(items["dump_rows"], serde_json::json!([1, 5]));
        assert_eq!(items["dump_from"], 1);
        assert_eq!(items["row_map"][3]["position"], 0); // Intervening row is retained.
        assert_eq!(items["row_map"][7]["padding"], true);
        let capture = read("capture.json");
        assert_eq!(capture["schema"], runtime_bundle_sdk::taps::SCHEMA);
        assert_eq!(capture["model"], "qwen38");
        // embed.out and every layer's X after it.
        let expected = capture["expected"].as_array().unwrap();
        assert_eq!(expected.len(), 1 + LAYERS as usize);
        assert_eq!(expected[0]["name"], "embed.out");
        assert_eq!(expected[0]["sharding"], "replicated");
        let tap = self::taps(Stage::ExpandStreams, Shape::TP4).remove(0);
        step.write(&tap.spec, 0, &vec![0; 4 * WIDE as usize * 2])
            .unwrap();
        step.finish().unwrap();
        assert_eq!(read("completion.json")["completion"], "incomplete");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn older_sites_select_what_they_selected() {
        let specs = specs();
        let all = |sites: &[&str]| TapFilter {
            layers: Some(TapLayers::All),
            sites: Some(sites.iter().map(|s| s.to_string()).collect()),
            from: None,
        };
        let older = |sites: &[&str]| older_sites(&Taps::new(PathBuf::new(), all(sites)), &specs);
        // X after every layer: not embed.out, attn.out or mlp.out.
        let out = older(&["out.x"]).unwrap();
        assert_eq!(out.len(), LAYERS as usize);
        assert!(
            out.iter()
                .all(|name| name.starts_with('L') && name.ends_with(".out") && name.len() == 7)
        );
        let selected = older(&["attn.selected"]).unwrap();
        assert_eq!(selected.len(), LAYERS as usize / 4);
        assert!(
            selected
                .iter()
                .all(|name| name.ends_with(".x.indexer.topk"))
        );
        // attn.gated is full attention's attn.core; ple.gated is layer 1's.
        let gated = older(&["gated"]).unwrap();
        assert_eq!(gated.len(), LAYERS as usize / 4 + 1);
        assert!(
            gated.contains(&"L03.attn.core".into()) && !gated.contains(&"L02.attn.core".into())
        );
        assert!(gated.contains(&"L01.x.ple.gated".into()));
        // attn.in, mlp.in and the PLE layer's ple.in, not the other layers' in.x.
        let input = older(&["in"]).unwrap();
        assert_eq!(input.len(), 2 * LAYERS as usize + 1);
        assert!(
            input.contains(&"L01.in.streams".into()) && !input.contains(&"L05.in.streams".into())
        );
        assert_eq!(
            older(&["final.u", "embed"]).unwrap(),
            ["embed.out", "final.x.hc.mixer"]
        );
        assert_eq!(
            older(&["attn.bogus", "out.x", "x.indexer.topk"]),
            Err("CHOPINFER_QWEN38_DUMP_TAPS selects no tap: attn.bogus, x.indexer.topk".into())
        );
        // Without layers, a detail tap's suffix selects nothing.
        let kept = TapFilter {
            layers: None,
            ..all(&["gated"])
        };
        assert_eq!(
            older_sites(&Taps::new(PathBuf::new(), kept), &specs),
            Err(
                "CHOPINFER_QWEN38_DUMP_TAPS selects only detail taps of layers \
                 CHOPINFER_TAPS_LAYERS does not keep: gated"
                    .into()
            )
        );
        // The shared names select by site or whole name; old suffixes select nothing.
        let taps = Taps::new(PathBuf::new(), all(&["out", "L03.attn.core"]));
        assert!(taps.check_sites(&specs).is_ok());
        let taps = Taps::new(PathBuf::new(), all(&["out.x", "x.hc.*"]));
        assert_eq!(
            taps.check_sites(&specs),
            Err("CHOPINFER_TAPS_SITES selects no tap: out.x".into())
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
        let mut used = vec![false; declaration().entries().len()];
        let taps = Taps::new(
            PathBuf::new(),
            TapFilter {
                layers: Some(TapLayers::All),
                ..TapFilter::default()
            },
        );
        for shape in Shape::ALL {
            used[declaration()
                .check(&encoder_for(Shape::TP4), shape.ranks())
                .unwrap()] = true;
            for phase in [Phase::Prefill, Phase::Decode(16)] {
                for tap in stages(phase)
                    .into_iter()
                    .flat_map(|stage| self::taps(stage, shape))
                {
                    assert!(taps.selected(&tap.spec));
                    let index = declaration().check(&tap.spec, shape.ranks()).unwrap();
                    used[index] = true;
                }
            }
        }
        for (entry, used) in declaration().entries().iter().zip(used) {
            assert!(used, "{} matches no tap", entry["name"]);
        }
    }
}
