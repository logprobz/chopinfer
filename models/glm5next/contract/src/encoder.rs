//! The vision encoder shared by the compiler and the bundle: its
//! declaration (`encoders/glm5next-vision.json`), the replicated encoder
//! image it reads (`program-order-stacked.v1`), the media arena and its
//! working memory. The kernels it generates are the backend's
//! (`crate::sm120::encoder` on SM120).

use std::collections::BTreeMap;
use std::sync::OnceLock;

use encoder_spec::components::{Context, StageOp};
use encoder_spec::weights::Source;
use encoder_spec::{Declaration, Environment, Error};

/// The vision declaration.
pub const VISION: &str = include_str!("../encoders/glm5next-vision.json");

/// The lock family of the encoder cubins in the bundle's `kernels.lock.d`.
pub const FAMILY: &str = "encoder";

/// The media arena (`frontend.json` `multimodal.arena`): rows of
/// `ARENA_HIDDEN` BF16 in pages of `PAGE_ROWS`.
pub const ARENA_ROWS: u64 = 65_536;
pub const PAGE_ROWS: u64 = 64;
pub const ARENA_HIDDEN: u32 = 4096;
/// One image's patch payload at most: 8000 tokens of 2 x 2 patches of 1176
/// F32 columns.
pub const MAX_PAYLOAD_BYTES: u64 = 8000 * 4 * 1176 * 4;
pub const PATCH_COLUMNS: u32 = 1176;

/// The encoder's working memory per row of a run: `b` (the tower's gate
/// and up rows, 8192 F32), `a` (the projector's gate and up rows, 20480 F32
/// per merged row, 5120 per patch row), the residual stream (1024 F32) and
/// the rotary cos and sin (32 F32 each).
pub const BYTES_PER_ROW: u64 = 4 * (8192 + 5120 + 1024 + 64);
/// Bytes of the sequence bounds and live counts.
pub const COUNT_BYTES: u64 = 16;

/// The capability a package requires to receive image inputs: the host
/// then serves the frontend's image modality and sends media steps.
pub const MEDIA_CAPABILITY: &str = "chop.media-inputs.v1";

/// The declaration, parsed and validated once.
pub fn declarations() -> Result<&'static [Declaration; 1], Error> {
    static DECLARATIONS: OnceLock<Result<[Declaration; 1], Error>> = OnceLock::new();
    DECLARATIONS
        .get_or_init(|| Ok([encoder_spec::parse(VISION, &Environment::new("glm5next"))?]))
        .as_ref()
        .map_err(Clone::clone)
}

/// Every encoder allocation of one rank at the declared capacity, the media
/// arena and the prefill's arena source rows included.
pub fn buffer_bytes() -> u64 {
    let declarations = declarations().expect("the vision declaration");
    encoder_spec::buffers::plan(
        &[&declarations[0]],
        u64::from(declarations[0].input.capacity_rows),
        encoder_spec::buffers::Serving {
            arena_rows: ARENA_ROWS,
            prefill_rows: u64::from(crate::contract::PREFILL_TOKENS),
            staging_bytes: 0,
        },
    )
    .expect("encoder buffer plan")
    .bytes()
}

/// The replicated encoder image: one view per tensor the bundle reads, in
/// the declaration's `program-order-stacked.v1` order.
pub mod views {
    use super::*;
    use bundle_runtime::index::encoder::{self, LayoutView};
    use bundle_runtime::index::encoder::{EncoderIndex, Identity};

    pub const IMAGE_ROLE: &[u8] = b"glm5next.encoder-rank-images.v1";
    pub const SCHEMA: &str = "chopinfer.glm5next.encoder-image.v1";
    const ALIGN: u64 = 4096;

    /// The authenticated metadata role of rank `rank`'s encoder index.
    pub fn index_role(rank: u32) -> String {
        bundle_runtime::index::index_role(
            "glm5next",
            bundle_runtime::index::ImageKind::Encoder,
            rank,
        )
    }

    /// What one view holds.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum Content {
        /// A role's tensor after the declaration's transforms (concatenated
        /// parts, zero padding, flattening, permuted columns).
        Role,
        /// Zeros: the bias of a GEMM the checkpoint gives none.
        Zeros,
    }

    /// One stored BF16 view: its name (the checkpoint name of its first
    /// tensor), its shape and offset, and the declaration source it is
    /// written from.
    #[derive(Clone, Debug, PartialEq)]
    pub struct Stored {
        pub name: String,
        pub shape: Vec<u64>,
        pub offset: u64,
        pub content: Content,
        /// The stage and, in a transformer, the block it belongs to.
        pub stage: usize,
        pub block: Option<u32>,
        /// The role it fills (`<role>bias` for a zero bias).
        pub role: String,
        pub source: Option<Source>,
    }

    impl Stored {
        pub fn bytes(&self) -> u64 {
            LayoutView::bytes(self)
        }
    }

    impl LayoutView for Stored {
        fn name(&self) -> &str {
            &self.name
        }
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn offset(&self) -> u64 {
            self.offset
        }
    }

    fn first(source: &Source) -> Option<&encoder_spec::weights::Tensor> {
        match source {
            Source::Tensor(tensor) => Some(tensor),
            Source::Concat(concat) => concat.concat_rows.first().and_then(first),
            Source::ConcatRange(_) | Source::Zeros(_) => None,
        }
    }

    /// The GEMM weight roles of a stage, in launch order: each one's
    /// `<prefix>weight`, whose bias is `<prefix>bias`.
    fn gemms(op: &StageOp) -> &'static [&'static str] {
        match op {
            StageOp::Linear(_) => &[""],
            StageOp::Transformer(_) => &["qkv.", "proj.", "up.", "gate_up.", "down."],
            StageOp::Projector(_) => &["up.", "gate_up.", "down."],
            _ => &[],
        }
    }

    /// The image layout: every GEMM's weight in program order, then every
    /// GEMM's bias in the same order (zeros for a GEMM without one, named
    /// by its weight's stem), then the other roles by name, each view
    /// 4096-aligned. A view is named by the checkpoint name of the first
    /// tensor of its source, as the shared runtime looks it up.
    pub fn layout() -> Result<Vec<Stored>, String> {
        layout_for(&declarations().map_err(|error| error.0)?[0])
    }

    /// `layout` of `declaration`, for tests of smaller towers.
    pub fn layout_for(declaration: &Declaration) -> Result<Vec<Stored>, String> {
        let inputs = declaration.inputs().map_err(|error| error.0)?;
        let context = Context {
            model: &declaration.model,
            precision: declaration.precision,
            patchify: declaration.input.patchify,
            sequences: &declaration.sequences,
            capacity_rows: u64::from(declaration.input.capacity_rows),
        };
        let prefix = &declaration.checkpoint.prefix;
        let (mut weights, mut biases, mut others) = (Vec::new(), Vec::new(), BTreeMap::new());
        for (index, (stage, input)) in declaration.stages.iter().zip(&inputs).enumerate() {
            let component = stage.op.component()?;
            let roles = component.roles(&context, input)?;
            let blocks = component
                .blocks()
                .map_or(vec![None], |count| (0..count).map(Some).collect::<Vec<_>>());
            for block in blocks {
                let named = |role: &str| -> Result<(String, Source), String> {
                    let source = stage
                        .weights
                        .get(role)
                        .ok_or_else(|| format!("stages[{index}] maps no {role}"))?;
                    let tensor = first(source)
                        .ok_or_else(|| format!("stages[{index}] {role} names no tensor"))?;
                    let name = match block {
                        Some(block) => tensor.tensor.replace("{i}", &block.to_string()),
                        None => tensor.tensor.clone(),
                    };
                    Ok((format!("{prefix}{name}"), source.clone()))
                };
                let shape = |role: &str| {
                    roles
                        .iter()
                        .find(|known| known.name == role)
                        .map(|known| known.shape.clone())
                };
                let stored = |role: &str, name: String, source, content| Stored {
                    name,
                    shape: shape(role).unwrap_or_default(),
                    offset: 0,
                    content,
                    stage: index,
                    block,
                    role: role.to_owned(),
                    source,
                };
                let mut taken = Vec::new();
                for gemm in gemms(&stage.op) {
                    let weight = format!("{gemm}weight");
                    let Some(rows) = shape(&weight).map(|shape| shape[0]) else {
                        continue;
                    };
                    let (name, source) = named(&weight)?;
                    let stem = name
                        .strip_suffix(".weight")
                        .ok_or_else(|| format!("{name} is not a weight"))?
                        .to_owned();
                    weights.push(stored(&weight, name, Some(source), Content::Role));
                    let bias = format!("{gemm}bias");
                    biases.push(match shape(&bias) {
                        Some(_) => {
                            let (name, source) = named(&bias)?;
                            if name != format!("{stem}.bias") {
                                return Err(format!("{name} is not the bias of {stem}"));
                            }
                            stored(&bias, name, Some(source), Content::Role)
                        }
                        None => Stored {
                            shape: vec![rows],
                            ..stored(&bias, format!("{stem}.bias"), None, Content::Zeros)
                        },
                    });
                    taken.extend([weight, bias]);
                }
                for role in &roles {
                    let absent = matches!((&role.blocks, block), (Some(set), Some(block)) if !set.contains(&block));
                    if taken.contains(&role.name) || absent {
                        continue;
                    }
                    let (name, source) = named(&role.name)?;
                    others.insert(
                        name.clone(),
                        stored(&role.name, name, Some(source), Content::Role),
                    );
                }
            }
        }
        let mut offset = 0;
        Ok(weights
            .into_iter()
            .chain(biases)
            .chain(others.into_values())
            .map(|mut view| {
                view.offset = offset;
                offset = (offset + view.bytes()).div_ceil(ALIGN) * ALIGN;
                view
            })
            .collect())
    }

    /// Bytes of one rank's encoder image.
    pub fn image_bytes() -> Result<u64, String> {
        Ok(end(&layout()?))
    }

    /// Bytes of an image of `layout`.
    pub fn end(layout: &[Stored]) -> u64 {
        layout.last().map_or(0, |view| {
            (view.offset + view.bytes()).div_ceil(ALIGN) * ALIGN
        })
    }

    /// Parses rank `rank`'s encoder index and checks it against `layout`
    /// exactly: every view in order, its shape, offset, dtype and layout,
    /// and the image length.
    pub fn parse(bytes: &[u8], rank: u32) -> Result<EncoderIndex, String> {
        parse_for(bytes, rank, &layout()?)
    }

    /// `parse` against `expected`, a layout of `layout_for`.
    pub fn parse_for(bytes: &[u8], rank: u32, expected: &[Stored]) -> Result<EncoderIndex, String> {
        encoder::parse(bytes, rank, SCHEMA, end(expected), expected)
    }

    /// The index of rank `rank`'s image, named by `identity`.
    pub fn index(rank: u32, identity: Identity) -> Result<serde_json::Value, String> {
        Ok(index_for(rank, &layout()?, identity))
    }

    /// `index` of an image of `layout`.
    pub fn index_for(rank: u32, layout: &[Stored], identity: Identity) -> serde_json::Value {
        encoder::index(SCHEMA, rank, end(layout), layout, identity)
    }
}

#[cfg(test)]
mod tests {
    use bundle_runtime::index::encoder::Identity;

    use super::views::*;
    use super::*;

    #[test]
    fn working_memory_is_57600_bytes_a_row() {
        let declaration = &declarations().unwrap()[0];
        let plan = |rows| {
            encoder_spec::buffers::plan(
                &[declaration],
                rows,
                encoder_spec::buffers::Serving {
                    arena_rows: 0,
                    prefill_rows: 0,
                    staging_bytes: 0,
                },
            )
            .unwrap()
            .bytes()
        };
        let capacity = u64::from(declaration.input.capacity_rows);
        assert_eq!(capacity, 32_768);
        assert_eq!(plan(capacity), capacity * BYTES_PER_ROW + COUNT_BYTES);
        assert_eq!(plan(capacity), 1_887_436_816);
        // With the arena of BF16 rows and the prefill's i64 source rows.
        let arena = ARENA_ROWS * u64::from(ARENA_HIDDEN) * 2;
        let sources = u64::from(crate::contract::PREFILL_TOKENS) * 8;
        assert_eq!(buffer_bytes(), plan(capacity) + arena + sources);
    }

    #[test]
    fn the_layout_stacks_gemms_then_biases_then_the_rest() {
        let views = layout().unwrap();
        // The weights and biases of 4 GEMMs a block and of the patch,
        // downsample, merger and the projector's two GEMMs; 4 norms a block,
        // the post norm and the projector norm's weight and bias.
        assert_eq!(views.len(), 2 * (24 * 4 + 5) + 24 * 4 + 3);
        let names: Vec<&str> = views.iter().map(|view| view.name.as_str()).collect();
        assert_eq!(names[0], "model.visual.patch_embed.proj.weight");
        assert_eq!(names[1], "model.visual.blocks.0.attn.qkv.weight");
        assert_eq!(names[3], "model.visual.blocks.0.mlp.gate_proj.weight");
        let gemms = 24 * 4 + 5;
        assert_eq!(names[gemms], "model.visual.patch_embed.proj.bias");
        let zero = |name: &str| {
            views
                .iter()
                .find(|view| view.name == name)
                .map(|view| (view.content, view.shape.clone()))
        };
        assert_eq!(
            zero("model.visual.merger.proj.bias"),
            Some((Content::Zeros, vec![4096]))
        );
        assert_eq!(
            zero("model.visual.merger.gate_proj.bias"),
            Some((Content::Zeros, vec![20480]))
        );
        assert_eq!(
            zero("model.visual.blocks.3.mlp.gate_proj.bias"),
            Some((Content::Role, vec![8192]))
        );
        let shape = |name: &str| zero(name).unwrap().1;
        assert_eq!(
            shape("model.visual.patch_embed.proj.weight"),
            vec![1024, 1216]
        );
        assert_eq!(shape("model.visual.downsample.weight"), vec![4096, 4096]);
        assert_eq!(
            shape("model.visual.merger.gate_proj.weight"),
            vec![20480, 4096]
        );
        assert_eq!(shape("model.visual.blocks.23.attn.k_norm.weight"), vec![64]);
        assert!(views.iter().all(|view| view.offset % 4096 == 0));
        assert!(
            views
                .windows(2)
                .all(|pair| pair[0].offset + pair[0].bytes() <= pair[1].offset)
        );
        // About 1.13 GB of checkpoint weights plus the padding, the zero
        // biases and the alignment.
        assert!((1_120_000_000..1_160_000_000).contains(&image_bytes().unwrap()));
    }

    #[test]
    fn the_index_parses_only_the_exact_layout() {
        let index = |rank| views::index(rank, Identity::Sha256(&"0".repeat(64))).unwrap();
        let parsed = parse(&serde_json::to_vec(&index(2)).unwrap(), 2).unwrap();
        assert_eq!(parsed.views.len(), layout().unwrap().len());
        assert!(parse(&serde_json::to_vec(&index(2)).unwrap(), 1).is_err());
        let mut moved = index(0);
        moved["views"][5]["offset"] = serde_json::json!(4096);
        assert!(parse(&serde_json::to_vec(&moved).unwrap(), 0).is_err());
        let mut unpadded = index(0);
        unpadded["views"][0]["shape"] = serde_json::json!([1024, 1176]);
        assert!(parse(&serde_json::to_vec(&unpadded).unwrap(), 0).is_err());
    }
}
