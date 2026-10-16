//! Exact encoder source geometry, and the encoder image the kernels read.
use crate::geometry::V26_FLASH as GEOMETRY;

pub mod spec;

use encoder_spec::buffers::{self, BufferPlan, Serving};

use runtime_bundle::rank_index::ViewContract;
use std::collections::BTreeMap;

pub const MAIN_FILE: &str = "model_pp0_ep0_shard0.safetensors";
pub const TOKENIZER_FILE: &str = "audio_tokenizer/model.safetensors";

pub type Expected = BTreeMap<String, (&'static str, Vec<u64>, &'static str)>;

/// The encoders' declarations at `geometry` in the frontend's encoder order
/// (vision, then audio): their recipes, row traces and inputs, whatever
/// kernels serve them.
pub fn declarations(
    geometry: &'static crate::geometry::Geometry,
) -> Result<&'static [encoder_spec::Declaration; 2], encoder_spec::Error> {
    spec::declarations_for(geometry)
}

pub fn expected_flash() -> Expected {
    expected_for(&GEOMETRY)
}

pub fn expected_for(geometry: &crate::geometry::Geometry) -> Expected {
    let vision = geometry.vision();
    let audio = geometry.audio();
    let tokenizer = geometry.audio_tokenizer();
    let vh = u64::from(vision.hidden_size);
    let vi = u64::from(vision.intermediate_size);
    let merged = vh * u64::from(vision.spatial_merge_size).pow(2);
    // The pinned vision spec supplies head width; hidden_size is not the attention width.
    let head = geometry.vision_head_dim();
    let vq = u64::from(vision.num_heads) * u64::from(head);
    let vkv = u64::from(vision.num_key_value_heads) * u64::from(head);
    let ah = u64::from(audio.input_local_dim);
    let ai = u64::from(audio.input_local_intermediate_size);
    let th = u64::from(tokenizer.d_model);
    let ti = u64::from(tokenizer.encoder_ffn_dim);
    let mut out = Expected::new();
    let mut add = |name: String, dtype: &'static str, shape: &[u64], file| {
        assert!(out.insert(name, (dtype, shape.to_vec(), file)).is_none());
    };
    let v = MAIN_FILE;
    add(
        "visual.patch_embed.proj.weight".into(),
        "BF16",
        &[
            vh,
            u64::from(vision.in_chans),
            u64::from(vision.temporal_patch_size),
            u64::from(vision.patch_size),
            u64::from(vision.patch_size),
        ],
        v,
    );
    add("visual.merger.ln_q.weight".into(), "BF16", &[vh], v);
    add(
        "visual.merger.mlp.0.weight".into(),
        "BF16",
        &[merged, merged],
        v,
    );
    add(
        "visual.merger.mlp.2.weight".into(),
        "BF16",
        &[u64::from(vision.out_hidden_size), merged],
        v,
    );
    for block in 0..vision.depth {
        let p = format!("visual.blocks.{block}");
        for (suffix, shape) in [
            ("attn.qkv.weight", &[vq + 2 * vkv, vh][..]),
            ("attn.qkv.bias", &[vq + 2 * vkv]),
            ("attn.proj.weight", &[vh, vq]),
            ("attn.proj.bias", &[vh]),
            ("mlp.gate_proj.weight", &[vi, vh]),
            ("mlp.gate_proj.bias", &[vi]),
            ("mlp.up_proj.weight", &[vi, vh]),
            ("mlp.up_proj.bias", &[vi]),
            ("mlp.down_proj.weight", &[vh, vi]),
            ("mlp.down_proj.bias", &[vh]),
            ("norm1.weight", &[vh]),
            ("norm2.weight", &[vh]),
        ] {
            add(format!("{p}.{suffix}"), "BF16", shape, v);
        }
        if vision.vit_window_attn_types[block as usize] != crate::geometry::VisionLayerKind::Full {
            add(
                format!("{p}.attn.sinks"),
                "BF16",
                &[u64::from(vision.num_heads)],
                v,
            );
        }
    }
    for layer in 0..audio.input_local_layers {
        let p = format!("audio_encoder.input_local_transformer.layers.{layer}");
        for (suffix, shape) in [
            ("input_layernorm.weight", &[ah][..]),
            ("post_attention_layernorm.weight", &[ah]),
            ("self_attn.q_proj.weight", &[ah, ah]),
            ("self_attn.q_proj.bias", &[ah]),
            ("self_attn.k_proj.weight", &[ah, ah]),
            ("self_attn.k_proj.bias", &[ah]),
            ("self_attn.v_proj.weight", &[ah, ah]),
            ("self_attn.v_proj.bias", &[ah]),
            ("self_attn.o_proj.weight", &[ah, ah]),
            ("mlp.gate_proj.weight", &[ai, ah]),
            ("mlp.up_proj.weight", &[ai, ah]),
            ("mlp.down_proj.weight", &[ah, ai]),
        ] {
            add(format!("{p}.{suffix}"), "BF16", shape, v);
        }
    }
    add(
        "audio_encoder.input_local_transformer.norm.weight".into(),
        "BF16",
        &[ah],
        v,
    );
    add(
        "audio_encoder.projection.mlp.0.weight".into(),
        "BF16",
        &[
            u64::from(audio.out_hidden_size) * u64::from(audio.group_size),
            ah * u64::from(audio.group_size),
        ],
        v,
    );
    add(
        "audio_encoder.projection.mlp.2.weight".into(),
        "BF16",
        &[
            u64::from(audio.out_hidden_size),
            u64::from(audio.out_hidden_size) * u64::from(audio.group_size),
        ],
        v,
    );
    for stage in 0..audio.audio_channels {
        add(
            format!("speech_embeddings.{stage}.weight"),
            "BF16",
            &[u64::from(audio.speech_vocab_size), ah],
            v,
        );
    }
    let a = TOKENIZER_FILE;
    for (suffix, shape) in [
        (
            "conv1.weight",
            &[
                th,
                u64::from(tokenizer.n_mels),
                u64::from(tokenizer.kernel_size),
            ][..],
        ),
        ("conv1.bias", &[th]),
        ("conv2.weight", &[th, th, u64::from(tokenizer.kernel_size)]),
        ("conv2.bias", &[th]),
        (
            "down_sample_layer.0.weight",
            &[th, th, u64::from(tokenizer.avg_pooler)],
        ),
        ("down_sample_norm.weight", &[th]),
        ("down_sample_norm.bias", &[th]),
        ("layer_norm.weight", &[th]),
        ("layer_norm.bias", &[th]),
    ] {
        if tokenizer.avg_pooler == 1 && suffix.starts_with("down_sample") {
            continue;
        }
        add(format!("encoder.{suffix}"), "BF16", shape, a);
    }
    for layer in 0..tokenizer.encoder_layers {
        let p = format!("encoder.layers.{layer}");
        for (suffix, shape) in [
            ("self_attn_layer_norm.weight", &[th][..]),
            ("self_attn_layer_norm.bias", &[th]),
            ("final_layer_norm.weight", &[th]),
            ("final_layer_norm.bias", &[th]),
            ("self_attn.q_proj.weight", &[th, th]),
            ("self_attn.q_proj.bias", &[th]),
            ("self_attn.k_proj.weight", &[th, th]),
            ("self_attn.v_proj.weight", &[th, th]),
            ("self_attn.v_proj.bias", &[th]),
            ("self_attn.out_proj.weight", &[th, th]),
            ("self_attn.out_proj.bias", &[th]),
            ("fc1.weight", &[ti, th]),
            ("fc1.bias", &[ti]),
            ("fc2.weight", &[th, ti]),
            ("fc2.bias", &[th]),
        ] {
            add(format!("{p}.{suffix}"), "BF16", shape, a);
        }
    }
    for (stage, &entries) in tokenizer.codebook_size.iter().enumerate() {
        add(
            format!("encoder.quantizer.vq.layers.{stage}._codebook.embed"),
            "F32",
            &[u64::from(entries), th],
            a,
        );
    }
    out
}

/// One part of a tensor the encoder kernels read, in order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Part {
    /// A checkpoint tensor as it is stored.
    Copy(String),
    /// A convolution weight, stored `(out, channel, tap)`, with its taps
    /// before its channels, as im2col columns take them.
    Taps(String),
    /// A BF16 tensor widened to F32.
    Widened(String),
    /// Zeros of this many bytes: the bias of a GEMM weight without one.
    Zeros(u64),
}

/// A tensor of the encoder image, laid out as the encoder kernels read it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Prepared {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub parts: Vec<Part>,
}

impl Prepared {
    pub fn layout_id(&self) -> &'static str {
        match self.dtype {
            "F32" => "f32_row_major_v1",
            _ => "bf16_row_major_v1",
        }
    }
}

/// How checkpoint view `name` enters the image: convolution weights put
/// their taps first, sinks widen to F32 and everything else, the F32
/// codebooks included, is stored as it is. Its shape there, and its bytes.
fn part(name: &str, expected: &Expected) -> (Part, Vec<u64>, u64) {
    let (dtype, shape, _) = &expected[name];
    let elements = shape.iter().product::<u64>();
    match (*dtype, shape.len()) {
        ("F32", _) => (Part::Copy(name.into()), shape.clone(), elements * 4),
        (_, 3) => (
            Part::Taps(name.into()),
            vec![shape[0], shape[2], shape[1]],
            elements * 2,
        ),
        _ if name.ends_with(".sinks") => (Part::Widened(name.into()), shape.clone(), elements * 4),
        _ => (Part::Copy(name.into()), shape.clone(), elements * 2),
    }
}

/// Parts stacked along their first axis into one tensor, F32 if any part
/// is.
fn stacked(name: String, parts: Vec<(Part, Vec<u64>, u64)>) -> Prepared {
    let wide = parts
        .iter()
        .any(|(_, shape, bytes)| *bytes == shape.iter().product::<u64>() * 4);
    let mut shape = parts[0].1.clone();
    shape[0] = parts.iter().map(|(_, shape, _)| shape[0]).sum();
    Prepared {
        name,
        dtype: if wide { "F32" } else { "BF16" },
        shape,
        parts: parts.into_iter().map(|(part, _, _)| part).collect(),
    }
}

/// Tensor names in a declared source, preserving concatenation order.
fn source_names(source: &encoder_spec::weights::Source, prefix: &str, block: u32) -> Vec<String> {
    use encoder_spec::weights::Source;
    match source {
        Source::Tensor(tensor) => vec![format!(
            "{prefix}{}",
            tensor.tensor.replace("{i}", &block.to_string())
        )],
        Source::Concat(concat) => concat
            .concat_rows
            .iter()
            .flat_map(|source| source_names(source, prefix, block))
            .collect(),
        Source::ConcatRange(range) => (0..range.count)
            .map(|index| {
                format!(
                    "{prefix}{}",
                    range.concat_range.replace("{j}", &index.to_string())
                )
            })
            .collect(),
        Source::Zeros(_) => Vec::new(),
    }
}

/// The declaration's GEMM order determines the existing image layout:
/// stacked weights and biases first, registered tables next, remaining
/// checkpoint views in name order. Names and layout remain ABI-compatible.
pub fn prepared_flash() -> Vec<Prepared> {
    prepared_for(&GEOMETRY)
}

pub fn prepared_for(geometry: &'static crate::geometry::Geometry) -> Vec<Prepared> {
    use encoder_spec::components::StageOp;
    let expected = expected_for(geometry);
    let mut tensors = Vec::new();
    let mut consumed = std::collections::BTreeSet::new();
    let declarations = spec::declarations_for(geometry).expect("MiMo encoder declarations");
    for declaration in declarations {
        for stage in &declaration.stages {
            let (blocks, roles): (u32, &[&str]) = match &stage.op {
                StageOp::Linear(_) | StageOp::Conv1d(_) => (1, &["weight"]),
                StageOp::Transformer(transformer) => (
                    transformer.params().blocks,
                    if stage.weights.contains_key("gate_up.weight") {
                        &["qkv.weight", "proj.weight", "gate_up.weight", "down.weight"]
                    } else {
                        &["qkv.weight", "proj.weight", "up.weight", "down.weight"]
                    },
                ),
                _ => continue,
            };
            for block in 0..blocks {
                for role in roles {
                    let weights =
                        source_names(&stage.weights[*role], &declaration.checkpoint.prefix, block);
                    let mut matrix = Vec::new();
                    let mut biases = Vec::new();
                    for name in &weights {
                        let weight = part(name, &expected);
                        let rows = weight.1[0];
                        matrix.push(weight);
                        consumed.insert(name.clone());
                        let bias = name.trim_end_matches(".weight").to_owned() + ".bias";
                        if expected.contains_key(&bias) {
                            biases.push(part(&bias, &expected));
                            consumed.insert(bias);
                        } else {
                            biases.push((Part::Zeros(rows * 2), vec![rows], rows * 2));
                        }
                    }
                    let first = &weights[0];
                    tensors.push(stacked(first.clone(), matrix));
                    tensors.push(stacked(
                        first.trim_end_matches(".weight").to_owned() + ".bias",
                        biases,
                    ));
                }
            }
        }
    }
    for declaration in declarations {
        for stage in &declaration.stages {
            for (role, tag) in [("codebooks", "@codebooks"), ("tables", "@embeddings")] {
                let Some(source) = stage.weights.get(role) else {
                    continue;
                };
                let names = source_names(source, &declaration.checkpoint.prefix, 0);
                let parts = names.iter().map(|name| part(name, &expected)).collect();
                consumed.extend(names);
                tensors.push(stacked(tag.into(), parts));
            }
        }
    }
    for name in expected.keys().filter(|name| !consumed.contains(*name)) {
        tensors.push(stacked(name.clone(), vec![part(name, &expected)]));
    }
    tensors
}

/// The media a package's encoders serve, and with them the encoders it
/// plans: the vision tower for images and video, the audio tower with the
/// audio tokenizer for audio. A package's image, buffer plan and media
/// bytes hold only what they serve: the encoders it plans, and of the vision
/// tower's recipes only those of the media it serves (video sizes its
/// payload, `video_sizes_the_vision_payload`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Encoders {
    pub image: bool,
    pub video: bool,
    pub audio: bool,
}

impl Encoders {
    pub const NONE: Self = Self::of(false, false, false);
    pub const ALL: Self = Self::of(true, true, true);
    /// Every set of media a package may serve.
    pub const SETS: [Self; 8] = [
        Self::of(false, false, false),
        Self::of(true, false, false),
        Self::of(false, true, false),
        Self::of(true, true, false),
        Self::of(false, false, true),
        Self::of(true, false, true),
        Self::of(false, true, true),
        Self::of(true, true, true),
    ];
    /// The media kinds, as the frontend and `--modalities` name them.
    pub const MEDIA: [&str; 3] = ["image", "video", "audio"];

    pub const fn of(image: bool, video: bool, audio: bool) -> Self {
        Self {
            image,
            video,
            audio,
        }
    }

    /// Whether the vision tower is planned: for images or video.
    pub const fn vision(self) -> bool {
        self.image || self.video
    }

    pub const fn any(self) -> bool {
        self.vision() || self.audio
    }

    fn served(self) -> [bool; 3] {
        [self.image, self.video, self.audio]
    }

    /// The served media `names` name.
    pub fn named<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<Self, String> {
        let mut encoders = Self::NONE;
        for name in names {
            match name {
                "image" => encoders.image = true,
                "video" => encoders.video = true,
                "audio" => encoders.audio = true,
                _ => return Err(format!("MiMo's encoders serve no {name} input")),
            }
        }
        Ok(encoders)
    }

    /// The served media's names, in [`Self::MEDIA`] order.
    pub fn names(self) -> Vec<&'static str> {
        Self::MEDIA
            .into_iter()
            .zip(self.served())
            .filter_map(|(name, served)| served.then_some(name))
            .collect()
    }

    /// Whether the image holds tensor `name` of [`prepared`]: the vision
    /// tower's (`visual.`) with the vision encoder, every other with the
    /// audio encoder.
    pub fn holds(self, name: &str) -> bool {
        if name.starts_with("visual.") {
            self.vision()
        } else {
            self.audio
        }
    }

    fn index(self) -> usize {
        usize::from(self.image) | usize::from(self.video) << 1 | usize::from(self.audio) << 2
    }

    /// The planned encoders' declarations, in the frontend's encoder order,
    /// `None` for one left out; the vision tower's with the recipes of the
    /// media it serves.
    pub fn declarations_flash(
        self,
    ) -> Result<&'static [Option<encoder_spec::Declaration>; 2], encoder_spec::Error> {
        type Planned = Result<[Option<encoder_spec::Declaration>; 2], encoder_spec::Error>;
        static PLANNED: std::sync::OnceLock<Vec<Planned>> = std::sync::OnceLock::new();
        PLANNED.get_or_init(|| {
            (0..8)
                .map(|index| {
                    let set = Self::of(index & 1 != 0, index & 2 != 0, index & 4 != 0);
                    let [vision, audio] = spec::declarations_flash()?;
                    let vision = set.vision().then(|| {
                        let mut vision = vision.clone();
                        vision.input.recipes.retain(|recipe| match recipe.tag() {
                            tag if tag.starts_with("image.") => set.image,
                            tag if tag.starts_with("video.") => set.video,
                            _ => true,
                        });
                        vision
                    });
                    Ok([vision, set.audio.then(|| audio.clone())])
                })
                .collect()
        })[self.index()]
        .as_ref()
        .map_err(Clone::clone)
    }

    pub fn declarations_for(
        self,
        geometry: &'static crate::geometry::Geometry,
    ) -> Result<Vec<encoder_spec::Declaration>, encoder_spec::Error> {
        let [vision, audio] = spec::declarations_for(geometry)?;
        let mut selected = Vec::new();
        if self.vision() {
            let mut vision = vision.clone();
            vision.input.recipes.retain(|recipe| match recipe.tag() {
                tag if tag.starts_with("image.") => self.image,
                tag if tag.starts_with("video.") => self.video,
                _ => true,
            });
            selected.push(vision);
        }
        if self.audio {
            selected.push(audio.clone());
        }
        Ok(selected)
    }

    /// The image's length on every rank with these encoders
    /// ([`image_bytes`]).
    pub fn image_bytes_flash(self) -> u64 {
        self.image_bytes_for(&GEOMETRY)
    }

    pub fn image_bytes_for(self, geometry: &'static crate::geometry::Geometry) -> u64 {
        image_bytes_of(&views_for(geometry, self))
    }

    /// Device bytes these encoders add on each rank besides their image:
    /// their buffer plan's regions ([`plan_bytes`]), none without encoders.
    pub fn media_bytes_flash(self) -> u64 {
        self.media_bytes_for(&GEOMETRY)
    }

    pub fn media_bytes_for(self, geometry: &'static crate::geometry::Geometry) -> u64 {
        if !self.any() {
            return 0;
        }
        let declarations = self
            .declarations_for(geometry)
            .expect("encoder declarations");
        let selected: Vec<_> = declarations.iter().collect();
        rounded_bytes(&plan_declarations(&selected).expect("encoder buffer plan"))
    }
}

/// The encoder image's views by name: [`prepared`]'s tensors.
pub fn views_flash() -> BTreeMap<String, ViewContract> {
    views_of_flash(Encoders::ALL)
}

/// The views of the image holding `encoders`: [`prepared`]'s tensors they
/// read.
pub fn views_of_flash(encoders: Encoders) -> BTreeMap<String, ViewContract> {
    views_for(&GEOMETRY, encoders)
}

pub fn views_for(
    geometry: &'static crate::geometry::Geometry,
    encoders: Encoders,
) -> BTreeMap<String, ViewContract> {
    prepared_for(geometry)
        .into_iter()
        .filter(|tensor| encoders.holds(&tensor.name))
        .map(|tensor| {
            let contract = super::view(&tensor.shape, tensor.dtype, tensor.layout_id());
            (tensor.name, contract)
        })
        .collect()
}

/// The encoder image's length on every rank: each view at a
/// [`crate::weight_plan::VIEW_ALIGNMENT`] boundary.
pub fn image_bytes_flash() -> u64 {
    image_bytes_of(&views_flash())
}

fn image_bytes_of(views: &BTreeMap<String, ViewContract>) -> u64 {
    views
        .values()
        .map(|view| {
            view.bytes
                .next_multiple_of(crate::weight_plan::VIEW_ALIGNMENT)
        })
        .sum()
}

/// Rows of the media arena the encoders write.
/// Policy: bound encoder arena rows.
pub const ARENA_ROWS: u64 = 32_768;
/// Payload bounds from the encoder declarations' input recipes.
pub fn payload_limits(
    geometry: &crate::geometry::Geometry,
) -> Result<[u64; 2], encoder_spec::Error> {
    let recipes = spec::recipes_for(geometry)?;
    let mut limits = [0; 2];
    for (limit, recipes) in limits.iter_mut().zip(recipes) {
        for recipe in recipes {
            let rows = recipe.rows().map_err(encoder_spec::Error)?;
            let bytes = rows
                .item
                .checked_mul(rows.columns)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| encoder_spec::Error("encoder payload size overflows".into()))?;
            *limit = (*limit).max(bytes);
        }
        if *limit == 0 {
            return Err(encoder_spec::Error(
                "encoder has no nonempty input recipes".into(),
            ));
        }
    }
    Ok(limits)
}

/// Whole seconds whose log-mel frames fit the encoder declaration.
pub fn maximum_audio_seconds(
    geometry: &crate::geometry::Geometry,
) -> Result<u64, encoder_spec::Error> {
    let tokenizer = geometry.audio_tokenizer();
    if tokenizer.n_mels == 0 || tokenizer.sampling_rate == 0 || tokenizer.hop_length == 0 {
        return Err(encoder_spec::Error(
            "audio mel bins, sample rate and hop must be nonzero".into(),
        ));
    }
    let frames = payload_limits(geometry)?[1] / (u64::from(tokenizer.n_mels) * 4);
    Ok((frames - 1) * u64::from(tokenizer.hop_length) / u64::from(tokenizer.sampling_rate))
}

/// Rows of one encoder run, and of one prefill step's arena source rows.
/// Policy: bound encoder step rows.
pub const RUN_ROWS: u64 = 8192;
/// Staging of the encoders' recorded uploads: a record's tables.
/// Policy: bound media staging.
pub const STAGING_BYTES: u64 = 64 << 20;
/// The granule the driver rounds each device allocation up to.
/// Policy: align encoder allocations.
const ALLOCATION_GRANULE: u64 = 2 << 20;

/// The buffer plan `encoders` allocate on every rank: their declarations
/// in runs of `RUN_ROWS`, the arena, the prefill source rows and the
/// staging.
pub fn plan_flash(encoders: Encoders) -> Result<BufferPlan, encoder_spec::Error> {
    let declarations: Vec<_> = encoders.declarations_flash()?.iter().flatten().collect();
    plan_declarations(&declarations)
}

/// The buffer plan of `declarations`, as [`plan_flash`] sizes it.
pub fn plan_declarations(
    declarations: &[&encoder_spec::Declaration],
) -> Result<BufferPlan, encoder_spec::Error> {
    let serving = Serving {
        arena_rows: ARENA_ROWS,
        prefill_rows: RUN_ROWS,
        staging_bytes: STAGING_BYTES,
    };
    buffers::plan(declarations, RUN_ROWS, serving)
}

/// Device bytes the buffer plan of `encoders` takes on every rank: each
/// region rounded up to the driver's allocation granule, and one granule
/// more; nothing without encoders. Derived from the declarations, so the
/// reservation follows the plan.
pub fn plan_bytes_flash(encoders: Encoders) -> u64 {
    static BYTES: std::sync::OnceLock<[u64; 8]> = std::sync::OnceLock::new();
    BYTES.get_or_init(|| {
        Encoders::SETS.map(|encoders| {
            if !encoders.any() {
                return 0;
            }
            rounded_bytes(&plan_flash(encoders).expect("MiMo's encoder declarations plan"))
        })
    })[encoders.index()]
}

/// `plan`'s regions as the driver allocates them, and one granule more.
pub fn rounded_bytes(plan: &BufferPlan) -> u64 {
    plan.regions
        .iter()
        .map(|region| region.bytes.next_multiple_of(ALLOCATION_GRANULE))
        .sum::<u64>()
        + ALLOCATION_GRANULE
}

pub const MEDIA_CAPABILITY: &str = "chop.media-inputs.v1";

/// All stages are bound. Keep serving disabled until preprocessing and full
/// encoder parity satisfy the media contract; kernel and layer checks alone do not.
pub const SERVING_READY: bool = true;

#[cfg(test)]
mod tests {
    use super::*;

    /// The media reservation is the plan as the driver allocates it: each
    /// of its 17 regions rounded up to 2 MiB, and 2 MiB more (1,036 MiB
    /// today; a plan change moves it, and with it the KV pool).
    #[test]
    fn invalid_payload_recipes_are_refused() {
        for (patch, temporal, merge) in [(0, 2, 2), (14, 0, 2), (14, 2, 0)] {
            let mut g = GEOMETRY;
            g.shape.vision.patch_size = patch;
            g.shape.vision.temporal_patch_size = temporal;
            g.shape.vision.spatial_merge_size = merge;
            assert!(payload_limits(&g).is_err());
        }
        let mut g = GEOMETRY;
        g.shape.audio_tokenizer.sampling_rate = 0;
        assert!(maximum_audio_seconds(&g).is_err());
    }

    #[test]
    fn audio_pooling_kernel_uses_pool_factor_instead_of_input_stride() {
        let mut geometry = GEOMETRY;
        geometry.shape.audio_tokenizer.avg_pooler = 4;
        geometry.shape.audio_tokenizer.stride_size = 2;
        let expected = expected_for(&geometry);
        assert_eq!(
            expected["encoder.down_sample_layer.0.weight"].1,
            [1024, 1024, 4]
        );
    }

    #[test]
    fn the_media_reservation_is_the_rounded_plan() {
        let plan = plan_flash(Encoders::ALL).unwrap();
        assert_eq!(plan.regions.len(), 17);
        let rounded: u64 = plan
            .regions
            .iter()
            .map(|region| region.bytes.div_ceil(2 << 20) * (2 << 20))
            .sum();
        assert!(rounded >= plan.bytes());
        assert_eq!(plan_bytes_flash(Encoders::ALL), rounded + (2 << 20));
        assert_eq!(
            Encoders::ALL.media_bytes_flash(),
            plan_bytes_flash(Encoders::ALL)
        );
        assert_eq!(plan_bytes_flash(Encoders::ALL), 1036 << 20);
    }

    /// Each set of served media plans its own: the vision and the audio
    /// images split the full one, and each set's buffer plan holds only its
    /// declarations' regions, so its media bytes fall when it leaves out
    /// audio or video (images size nothing video does not), to none
    /// without media.
    #[test]
    fn each_set_of_media_has_its_own_image_and_media_bytes() {
        let vision = Encoders::of(true, true, false);
        let audio = Encoders::of(false, false, true);
        let (vision_views, audio_views) = (views_of_flash(vision), views_of_flash(audio));
        assert!(!vision_views.is_empty() && !audio_views.is_empty());
        assert!(
            vision_views
                .keys()
                .all(|name| !audio_views.contains_key(name))
        );
        assert_eq!(vision_views.len() + audio_views.len(), views_flash().len());
        assert_eq!(
            vision.image_bytes_flash() + audio.image_bytes_flash(),
            image_bytes_flash()
        );
        assert_eq!(Encoders::ALL.image_bytes_flash(), image_bytes_flash());
        assert_eq!(Encoders::NONE.image_bytes_flash(), 0);
        assert_eq!(Encoders::NONE.media_bytes_flash(), 0);
        for (index, set) in Encoders::SETS.into_iter().enumerate() {
            assert_eq!(set.index(), index);
            assert_eq!(Encoders::named(set.names()), Ok(set));
            assert_eq!(
                set.image_bytes_flash(),
                if set.vision() {
                    vision.image_bytes_flash()
                } else {
                    0
                } + if set.audio {
                    audio.image_bytes_flash()
                } else {
                    0
                }
            );
            // Leaving out audio, or video, takes its regions' share.
            if set.any() {
                assert!(
                    set.media_bytes_flash() <= Encoders::ALL.media_bytes_flash(),
                    "{set:?}"
                );
                assert_eq!(
                    set.media_bytes_flash() < Encoders::ALL.media_bytes_flash(),
                    !set.video || !set.audio
                );
            }
            let [planned_vision, planned_audio] = set.declarations_flash().unwrap();
            assert_eq!(
                (planned_vision.is_some(), planned_audio.is_some()),
                (set.vision(), set.audio)
            );
            eprintln!(
                "{:?}: image {} MiB, media {} MiB",
                set.names(),
                set.image_bytes_flash() >> 20,
                set.media_bytes_flash() >> 20
            );
        }
        let names = ["mimo-vision", "mimo-audio"];
        for (set, name) in [(vision, names[0]), (audio, names[1])] {
            let plan = plan_flash(set).unwrap();
            assert!(plan.regions.iter().all(|region| {
                region
                    .uses
                    .iter()
                    .all(|used| used.encoder.is_empty() || used.encoder == name)
            }));
        }
        assert!(Encoders::named(["speech"]).is_err());
    }

    /// Video sizes the vision tower's payload region (its runs' input rows
    /// are whole videos' patch rows, four times an image's), and nothing
    /// else: a package serving images without video plans the vision tower
    /// without the video recipe and reserves that much less.
    #[test]
    fn video_sizes_the_vision_payload() {
        let images = plan_flash(Encoders::of(true, false, false)).unwrap();
        let both = plan_flash(Encoders::of(true, true, false)).unwrap();
        let video = plan_flash(Encoders::of(false, true, false)).unwrap();
        let regions = |plan: &BufferPlan| {
            plan.regions
                .iter()
                .map(|region| (region.name, region.bytes))
                .collect::<Vec<_>>()
        };
        let differing: Vec<_> = regions(&images)
            .into_iter()
            .zip(regions(&both))
            .filter(|(left, right)| left != right)
            .collect();
        assert_eq!(differing, [(("payload", 24 << 20), ("payload", 96 << 20))]);
        assert_eq!(regions(&video), regions(&both));
        assert_eq!(
            plan_bytes_flash(Encoders::of(true, false, false)) + (72 << 20),
            plan_bytes_flash(Encoders::of(true, true, false))
        );
    }

    /// Every checkpoint view the program reads enters the image exactly once,
    /// every tensor's parts fill its shape, tensor names are unique, and the
    /// image holds 676 tensors: stacked GEMM weights and biases, the codebook
    /// and embedding tables, sinks widened to F32 and convolution weights with
    /// their taps first.
    #[test]
    fn the_image_holds_every_program_view_once_in_its_kernels_layout() {
        let expected = expected_flash();
        let tensors = prepared_flash();
        assert_eq!(tensors.len(), 676);
        let names = tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(names.len(), tensors.len());
        let mut placed = Vec::new();
        for tensor in &tensors {
            let width = if tensor.dtype == "F32" { 4 } else { 2 };
            let mut bytes = 0;
            for stored in &tensor.parts {
                bytes += match stored {
                    Part::Zeros(bytes) => *bytes,
                    Part::Copy(name) | Part::Taps(name) | Part::Widened(name) => {
                        placed.push(name.clone());
                        part(name, &expected).2
                    }
                };
            }
            assert_eq!(
                bytes,
                tensor.shape.iter().product::<u64>() * width,
                "{}",
                tensor.name
            );
        }
        placed.sort();
        assert_eq!(
            placed,
            spec::declarations_flash()
                .unwrap()
                .iter()
                .flat_map(|declaration| declaration.tensors())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
        );
        let by_name = |name: &str| tensors.iter().find(|tensor| tensor.name == name).unwrap();
        let qkv = by_name("encoder.layers.0.self_attn.q_proj.weight");
        assert_eq!(qkv.shape, [3072, 1024]);
        // k_proj has no bias: its rows are zeros between q's and v's.
        assert_eq!(
            by_name("encoder.layers.0.self_attn.q_proj.bias").parts[1],
            Part::Zeros(2048)
        );
        assert_eq!(by_name("encoder.conv2.weight").shape, [1024, 3, 1024]);
        assert!(matches!(
            by_name("encoder.conv2.weight").parts[0],
            Part::Taps(_)
        ));
        let sinks = by_name("visual.blocks.1.attn.sinks");
        assert_eq!(
            (sinks.dtype, sinks.layout_id()),
            ("F32", "f32_row_major_v1")
        );
        assert_eq!(by_name("@codebooks").shape[1], 1024);
        assert_eq!(
            image_bytes_flash(),
            views_flash()
                .values()
                .map(|view| view.bytes.next_multiple_of(4096))
                .sum::<u64>()
        );
    }
}
