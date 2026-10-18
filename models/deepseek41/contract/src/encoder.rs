//! The vision encoder shared by the compiler, the bundle and the lab: its
//! declaration (`encoders/deepseek41-vision.json`), the replicated encoder
//! image it reads (`view-order.v1`), the media arena and its working memory.
//!
//! The vendor's ViT (`vision.py`) and aligner run in BF16 activations; under
//! `f32-pairs.v1` every row between stages stays F32 and every GEMM runs on
//! BF16 pairs with F32 accumulation, so only the aligner's output rounds to
//! BF16, where the vendor casts the image embeddings to the text model's
//! dtype (`model.py` `embeds.to(h.dtype)`). The aligner's 3 x 3 unfold is a
//! space-to-depth merge whose position-major columns the stored `aligner.w1`
//! takes permuted (`space_to_depth_cols`), and the patch projection's 588
//! columns are padded to 640, a whole number of GEMM K tiles.

use std::sync::OnceLock;

use encoder_spec::components::Context;
use encoder_spec::weights::Source;
use encoder_spec::{Declaration, Environment, Error};

/// The vision declaration.
pub const VISION: &str = include_str!("../encoders/deepseek41-vision.json");

/// The lock family of the encoder cubins in the bundle's `kernels.lock.d`.
pub const FAMILY: &str = "encoder";

/// The media arena: rows of `ARENA_HIDDEN` BF16 in pages of `PAGE_ROWS`,
/// sixteen images of the largest token budget (1024 rows each).
pub const ARENA_ROWS: u64 = 16_384;
pub const PAGE_ROWS: u64 = 64;
pub const ARENA_HIDDEN: u32 = 5120;
/// One patch row: 3 x 14 x 14 F32 values, channel-major.
pub const PATCH_COLUMNS: u32 = 588;
/// One image's patch payload at most: the 1021 cells of a 1024-token
/// budget of 3 x 3 patches each.
pub const MAX_PAYLOAD_BYTES: u64 = 1021 * 9 * PATCH_COLUMNS as u64 * 4;

/// The capability a package requires to receive image inputs: the host
/// then serves the frontend's image modality and sends media steps.
pub const MEDIA_CAPABILITY: &str = "chop.media-inputs.v1";

/// The declaration, parsed and validated once.
pub fn declarations() -> Result<&'static [Declaration; 1], Error> {
    static DECLARATIONS: OnceLock<Result<[Declaration; 1], Error>> = OnceLock::new();
    DECLARATIONS
        .get_or_init(|| {
            Ok([encoder_spec::parse(
                VISION,
                &Environment::new("deepseek41"),
            )?])
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Whether an instance of `geometry` can serve the encoder: its config
/// holds the vision tower the declaration pins.
pub fn serves(geometry: &crate::geometry::Geometry) -> bool {
    let Ok([vision]) = declarations() else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(geometry.document)
        .is_ok_and(|config| vision.check_config("config.json", &config).is_ok())
}

/// The names of the checkpoint tensors the declaration reads: the vision
/// tower, the aligner and the three learned delimiter rows.
pub fn tensors() -> Result<std::collections::BTreeSet<String>, Error> {
    Ok(declarations()?[0].tensors())
}

/// What the tower's diagnostic taps need of the declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TapShape {
    /// Each transformer's width and block count, in stage order.
    pub transformers: Vec<(u32, u32)>,
    /// Whether the residual stream is F32 (`f32-pairs.v1`), else BF16.
    pub f32: bool,
    /// The side of the patch cell one output row merges: the centered
    /// pad's `downsample`, so an image of `[h, w]` patches has
    /// `ceil(h / side)` rows of `ceil(w / side)` cells.
    pub cell_side: u32,
}

/// The declaration's [`TapShape`].
pub fn tap_shape() -> Result<TapShape, Error> {
    use encoder_spec::components::StageOp;
    let declaration = &declarations()?[0];
    let transformers = declaration
        .stages
        .iter()
        .filter_map(|stage| match &stage.op {
            StageOp::Transformer(transformer) => {
                let params = transformer.params();
                Some((params.width, params.blocks))
            }
            _ => None,
        })
        .collect();
    let cell_side = declaration
        .input
        .recipes
        .iter()
        .find_map(|recipe| match recipe {
            encoder_spec::Recipe::ImageCenteredPad(pad) => Some(pad.downsample),
            _ => None,
        })
        .ok_or_else(|| Error("the vision declaration has no centered-pad recipe".into()))?;
    Ok(TapShape {
        transformers,
        f32: declaration.precision == encoder_spec::policy::Precision::F32Pairs,
        cell_side,
    })
}

/// Every encoder allocation of one rank at the declared capacity, the media
/// arena and the prefill's arena source rows included.
pub fn buffer_bytes() -> Result<u64, Error> {
    let declaration = &declarations()?[0];
    Ok(encoder_spec::buffers::plan(
        &[declaration],
        u64::from(declaration.input.capacity_rows),
        encoder_spec::buffers::Serving {
            arena_rows: ARENA_ROWS,
            prefill_rows: u64::from(crate::serving::PREFILL_TOKENS),
            staging_bytes: 0,
        },
    )?
    .bytes())
}

/// The replicated encoder image: one view per tensor the declaration reads,
/// in its `view-order.v1` order.
pub mod views {
    use super::*;
    use bundle_runtime::index::encoder::{self, EncoderIndex, Identity, LayoutView};

    pub const IMAGE_ROLE: &[u8] = b"deepseek41.encoder-rank-images.v1";
    pub const SCHEMA: &str = "chopinfer.deepseek41.encoder-image.v1";
    const ALIGN: u64 = 4096;

    /// The authenticated metadata role of rank `rank`'s encoder index.
    pub fn index_role(rank: u32) -> String {
        bundle_runtime::index::index_role(
            "deepseek41",
            bundle_runtime::index::ImageKind::Encoder,
            rank,
        )
    }

    /// One stored BF16 view: the checkpoint tensor it is written from
    /// (after the declaration's transforms), its shape and offset.
    #[derive(Clone, Debug, PartialEq)]
    pub struct Stored {
        pub name: String,
        pub shape: Vec<u64>,
        pub offset: u64,
        /// The stage, or `None` for a learned delimiter row of the output.
        pub stage: Option<usize>,
        pub source: Source,
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

    /// The image layout of the declaration.
    pub fn layout() -> Result<Vec<Stored>, String> {
        layout_for(&declarations().map_err(|error| error.0)?[0])
    }

    /// Every role each stage reads, in stage, block and role order, then
    /// the output's start, newline and end rows, each view 4096-aligned and
    /// named by its checkpoint tensor as the shared runtime looks it up.
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
        let mut views = Vec::new();
        for (index, (stage, input)) in declaration.stages.iter().zip(&inputs).enumerate() {
            let component = stage.op.component()?;
            let roles = component.roles(&context, input)?;
            let blocks = component
                .blocks()
                .map_or(vec![None], |count| (0..count).map(Some).collect::<Vec<_>>());
            for block in blocks {
                for role in &roles {
                    if let (Some(set), Some(block)) = (&role.blocks, block)
                        && !set.contains(&block)
                    {
                        continue;
                    }
                    let source = stage
                        .weights
                        .get(&role.name)
                        .ok_or_else(|| format!("stages[{index}] maps no {}", role.name))?;
                    let Source::Tensor(tensor) = source else {
                        return Err(format!(
                            "stages[{index}] {}: view-order.v1 binds single tensors",
                            role.name
                        ));
                    };
                    let name = match block {
                        Some(block) => tensor.tensor.replace("{i}", &block.to_string()),
                        None => tensor.tensor.clone(),
                    };
                    views.push(Stored {
                        name: format!("{prefix}{name}"),
                        shape: role.shape.clone(),
                        offset: 0,
                        stage: Some(index),
                        source: source.clone(),
                    });
                }
            }
        }
        let hidden = u64::from(declaration.output.op.params().hidden);
        for role in ["start", "newline", "end"] {
            let source = declaration
                .output
                .weights
                .get(role)
                .ok_or_else(|| format!("the output layout binds no `{role}` row"))?;
            let Source::Tensor(tensor) = source else {
                return Err(format!("the `{role}` row is not a single tensor"));
            };
            views.push(Stored {
                name: format!("{prefix}{}", tensor.tensor),
                shape: vec![hidden],
                offset: 0,
                stage: None,
                source: source.clone(),
            });
        }
        let mut offset = 0;
        for view in &mut views {
            view.offset = offset;
            offset = (offset + view.bytes()).div_ceil(ALIGN) * ALIGN;
        }
        Ok(views)
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

    /// Parses rank `rank`'s encoder index and checks it against the layout
    /// exactly: every view in order, its shape, offset, dtype and layout,
    /// and the image length.
    pub fn parse(bytes: &[u8], rank: u32) -> Result<EncoderIndex, String> {
        let expected = layout()?;
        encoder::parse(bytes, rank, SCHEMA, end(&expected), &expected)
    }

    /// The index of rank `rank`'s image, named by `identity`.
    pub fn index(rank: u32, identity: Identity) -> Result<serde_json::Value, String> {
        let layout = layout()?;
        Ok(encoder::index(
            SCHEMA,
            rank,
            end(&layout),
            &layout,
            identity,
        ))
    }
}

#[cfg(test)]
mod tests {
    use bundle_runtime::index::encoder::Identity;
    use encoder_spec::components::Item;

    use super::views::*;
    use super::*;

    #[test]
    fn the_declaration_pins_the_vendor_vision_config() {
        let declaration = &declarations().unwrap()[0];
        assert_eq!(declaration.name, "deepseek41-vision");
        let config: serde_json::Value = serde_json::from_str(
            r#"{"vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 32,
                "hidden_size": 1024, "num_attention_heads": 16, "intermediate_size": 2816,
                "patch_size": 14, "rope_theta": 10000, "downsample_ratio": 3,
                "max_image_tokens": 1024, "min_pixels": 295936, "max_wh_ratio": null}}"#,
        )
        .unwrap();
        declaration.check_config("config.json", &config).unwrap();
        // A 1021-cell image fills the capacity's patch rows; the arena holds
        // sixteen of the largest spans.
        assert_eq!(declaration.input.capacity_rows, 9216);
        let trace = declaration
            .trace(Item {
                rows: 3 * 3063,
                grid: Some([1, 3, 3063]),
            })
            .unwrap();
        assert_eq!(trace[4].rows, 1021);
        assert_eq!(trace.last().unwrap().rows, 1024);
        assert_eq!(MAX_PAYLOAD_BYTES, 9189 * 588 * 4);
    }

    #[test]
    fn the_flash_geometry_serves_the_encoder() {
        assert!(serves(&crate::geometry::FLASH));
    }

    #[test]
    fn the_merge_rows_are_bounded_by_the_token_budget() {
        let declaration = &declarations().unwrap()[0];
        let plan = |arena_rows| {
            encoder_spec::buffers::plan(
                &[declaration],
                9216,
                encoder_spec::buffers::Serving {
                    arena_rows,
                    prefill_rows: 0,
                    staging_bytes: 0,
                },
            )
            .unwrap()
        };
        let working = plan(0);
        let gather = working.owned("gather-indices", Some(4)).unwrap();
        assert_eq!(gather.bytes, 1021 * 9 * 4);
        // The tower's SwiGLU gate and up rows (5632 F32 a patch row) are the
        // largest region; the merged rows hold 1021 cells, not 9216.
        let largest = working.regions.iter().map(|r| r.bytes).max().unwrap();
        assert_eq!(largest, 9216 * 5632 * 4);
        assert!(working.bytes() < 400 << 20, "{}", working.bytes());
        let arena = ARENA_ROWS * u64::from(ARENA_HIDDEN) * 2;
        let sources = u64::from(crate::serving::PREFILL_TOKENS) * 8;
        assert_eq!(buffer_bytes().unwrap(), working.bytes() + arena + sources);
    }

    #[test]
    fn the_layout_holds_every_tensor_once_at_its_role_shape() {
        let views = layout().unwrap();
        // The patch projection's weight and bias, 8 tensors a block, the
        // final norm, the aligner's two GEMMs and the three delimiter rows.
        assert_eq!(views.len(), 2 + 32 * 8 + 1 + 4 + 3);
        let names: std::collections::BTreeSet<_> =
            views.iter().map(|view| view.name.clone()).collect();
        assert_eq!(names, tensors().unwrap());
        let shape = |name: &str| {
            views
                .iter()
                .find(|view| view.name == name)
                .map(|view| view.shape.clone())
                .unwrap()
        };
        assert_eq!(shape("vision.patch_embed.proj.weight"), [1024, 640]);
        assert_eq!(shape("vision.blocks.31.mlp.w1.weight"), [5632, 1024]);
        assert_eq!(shape("vision.blocks.0.attn.wqkv.bias"), [3072]);
        assert_eq!(shape("aligner.w1.weight"), [5120, 9216]);
        assert_eq!(shape("image_newline"), [5120]);
        assert_eq!(views[0].name, "vision.patch_embed.proj.weight");
        assert_eq!(views.last().unwrap().name, "image_end");
        assert!(views.iter().all(|view| view.offset % 4096 == 0));
        assert!(
            views
                .windows(2)
                .all(|pair| pair[0].offset + pair[0].bytes() <= pair[1].offset)
        );
        // About 0.95 GB of BF16 weights plus the patch padding and the
        // alignment.
        let bytes = image_bytes().unwrap();
        assert!((900_000_000..1_000_000_000).contains(&bytes), "{bytes}");
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
        unpadded["views"][0]["shape"] = serde_json::json!([1024, 588]);
        assert!(parse(&serde_json::to_vec(&unpadded).unwrap(), 0).is_err());
    }
}
