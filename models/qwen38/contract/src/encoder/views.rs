//! The replicated vision encoder image: a separate compiled image beside the
//! text rank images, so adding or rebuilding it never touches them. Every
//! view is row-major BF16 at a fixed, 4096-aligned offset; the layout below
//! is the whole contract between the weight plan and the bundle.

#[cfg(test)]
use std::collections::BTreeSet;

pub use bundle_runtime::index::View;
use bundle_runtime::index::encoder::{self, LayoutView};
pub use bundle_runtime::index::encoder::{BF16_ROWS, EncoderIndex, image_name};

pub const IMAGE_ROLE: &[u8] = b"qwen38.encoder-rank-images.v1";
pub const SCHEMA: &str = "chopinfer.qwen38.encoder-image.v1";
pub const PREFIX: &str = "model.visual";
const ALIGN: u64 = 4096;

/// The authenticated metadata role of rank `rank`'s encoder index.
pub fn index_role(rank: u32) -> String {
    bundle_runtime::index::index_role("qwen38", bundle_runtime::index::ImageKind::Encoder, rank)
}

/// One stored view: its checkpoint tensor name, the stored shape, and the
/// checkpoint shape it is taken from. They differ only where the stored
/// view is zero-padded (`linear_fc1` rows and bias, `linear_fc2` columns
/// to `MLP_PADDED`) or flattened (the patch convolution to `[1152, 1536]`,
/// the same bytes).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stored {
    pub name: String,
    pub shape: Vec<u64>,
    pub source_shape: Vec<u64>,
    pub offset: u64,
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

pub fn layout_flash() -> Vec<Stored> {
    layout_for(&crate::geometry::FLASH)
}

pub fn layout_for(geometry: &crate::geometry::Geometry) -> Vec<Stored> {
    let vit_hidden = geometry.vision_hidden();
    let mlp = geometry.vision_intermediate();
    let mlp_padded = mlp.next_multiple_of(128);
    let vit_blocks = geometry.vision_blocks();
    let vit_out = geometry.vision_out();
    let vit_positions = geometry.vision_positions();
    let hidden = u64::from(vit_hidden);
    let mlp = u64::from(mlp);
    let padded = u64::from(mlp_padded);
    let mut entries: Vec<(String, Vec<u64>, Vec<u64>)> = vec![
        (
            format!("{PREFIX}.patch_embed.proj.weight"),
            vec![
                hidden,
                u64::from(
                    geometry.vision_channels()
                        * geometry.vision_temporal_patch()
                        * geometry.vision_patch().pow(2),
                ),
            ],
            vec![
                hidden,
                u64::from(geometry.vision_channels()),
                u64::from(geometry.vision_temporal_patch()),
                u64::from(geometry.vision_patch()),
                u64::from(geometry.vision_patch()),
            ],
        ),
        (
            format!("{PREFIX}.patch_embed.proj.bias"),
            vec![hidden],
            vec![hidden],
        ),
        (
            format!("{PREFIX}.pos_embed.weight"),
            vec![u64::from(vit_positions), hidden],
            vec![u64::from(vit_positions), hidden],
        ),
    ];
    for block in 0..vit_blocks {
        let stem = format!("{PREFIX}.blocks.{block}");
        let same =
            |suffix: &str, shape: Vec<u64>| (format!("{stem}.{suffix}"), shape.clone(), shape);
        entries.extend([
            same("norm1.weight", vec![hidden]),
            same("norm1.bias", vec![hidden]),
            same("attn.qkv.weight", vec![3 * hidden, hidden]),
            same("attn.qkv.bias", vec![3 * hidden]),
            same("attn.proj.weight", vec![hidden, hidden]),
            same("attn.proj.bias", vec![hidden]),
            same("norm2.weight", vec![hidden]),
            same("norm2.bias", vec![hidden]),
            (
                format!("{stem}.mlp.linear_fc1.weight"),
                vec![padded, hidden],
                vec![mlp, hidden],
            ),
            (
                format!("{stem}.mlp.linear_fc1.bias"),
                vec![padded],
                vec![mlp],
            ),
            (
                format!("{stem}.mlp.linear_fc2.weight"),
                vec![hidden, padded],
                vec![hidden, mlp],
            ),
            same("mlp.linear_fc2.bias", vec![hidden]),
        ]);
    }
    let merged = u64::from(geometry.vision_merge() * geometry.vision_merge()) * hidden;
    let out = u64::from(vit_out);
    for (suffix, shape) in [
        ("norm.weight", vec![hidden]),
        ("norm.bias", vec![hidden]),
        ("linear_fc1.weight", vec![merged, merged]),
        ("linear_fc1.bias", vec![merged]),
        ("linear_fc2.weight", vec![out, merged]),
        ("linear_fc2.bias", vec![out]),
    ] {
        entries.push((format!("{PREFIX}.merger.{suffix}"), shape.clone(), shape));
    }
    let mut offset = 0;
    entries
        .into_iter()
        .map(|(name, shape, source_shape)| {
            let stored = Stored {
                name,
                shape,
                source_shape,
                offset,
            };
            offset = (offset + stored.bytes()).div_ceil(ALIGN) * ALIGN;
            stored
        })
        .collect()
}

/// Bytes of one rank's encoder image.
pub fn image_bytes_flash() -> u64 {
    image_bytes_for(&crate::geometry::FLASH)
}

pub fn image_bytes_for(geometry: &crate::geometry::Geometry) -> u64 {
    layout_for(geometry).last().map_or(0, |view| {
        (view.offset + view.bytes()).div_ceil(ALIGN) * ALIGN
    })
}

/// Parses rank `rank`'s encoder index and checks it against `layout`
/// exactly: every view, shape, offset, dtype and layout, and the image
/// length.
pub fn parse_flash(bytes: &[u8], rank: u32) -> Result<EncoderIndex, String> {
    parse_for(bytes, rank, &crate::geometry::FLASH)
}

pub fn parse_for(
    bytes: &[u8],
    rank: u32,
    geometry: &crate::geometry::Geometry,
) -> Result<EncoderIndex, String> {
    let image_bytes = image_bytes_for(geometry);
    encoder::parse(bytes, rank, SCHEMA, image_bytes, &layout_for(geometry))
}

/// The checkpoint tensors the image is built from.
#[cfg(test)]
pub fn source_tensors() -> BTreeSet<String> {
    layout_flash().into_iter().map(|view| view.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(rank: u32) -> serde_json::Value {
        serde_json::json!({
            "schema": SCHEMA,
            "rank": rank,
            "image": image_name(rank),
            "image_bytes": image_bytes_flash(),
            "sha256": "0".repeat(64),
            "views": layout_flash().iter().map(|view| serde_json::json!({
                "name": view.name,
                "offset": view.offset,
                "bytes": view.bytes(),
                "shape": view.shape,
                "dtype": "BF16",
                "layout_id": BF16_ROWS,
            })).collect::<Vec<_>>(),
        })
    }

    #[test]
    fn layout_covers_the_checkpoint_tower_with_padded_mlp() {
        let views = layout_flash();
        assert_eq!(views.len(), 3 + 27 * 12 + 6);
        assert_eq!(source_tensors().len(), views.len());
        let fc1 = views
            .iter()
            .find(|view| view.name == "model.visual.blocks.26.mlp.linear_fc1.weight")
            .unwrap();
        assert_eq!(
            (fc1.shape.clone(), fc1.source_shape.clone()),
            (vec![4352, 1152], vec![4304, 1152])
        );
        assert!(views.iter().all(|view| view.offset % 4096 == 0));
        assert!(
            views
                .windows(2)
                .all(|pair| pair[0].offset + pair[0].bytes() <= pair[1].offset)
        );
        // About 0.9 GB: the tower's 0.45 B parameters plus the MLP padding.
        assert!((900_000_000..920_000_000).contains(&image_bytes_flash()));
    }

    #[test]
    fn index_parses_only_the_exact_layout_flash() {
        let parsed = parse_flash(&serde_json::to_vec(&index(2)).unwrap(), 2).unwrap();
        assert_eq!(parsed.views.len(), layout_flash().len());
        assert!(parse_flash(&serde_json::to_vec(&index(2)).unwrap(), 1).is_err());
        let mut moved = index(0);
        moved["views"][5]["offset"] = serde_json::json!(4096);
        assert!(parse_flash(&serde_json::to_vec(&moved).unwrap(), 0).is_err());
        let mut unpadded = index(0);
        unpadded["views"][11]["shape"] = serde_json::json!([4304, 1152]);
        assert!(parse_flash(&serde_json::to_vec(&unpadded).unwrap(), 0).is_err());
    }

    #[test]
    fn indexes_carry_an_image_digest_or_a_direct_identity() {
        let parsed = parse_flash(&serde_json::to_vec(&index(1)).unwrap(), 1).unwrap();
        assert_eq!(parsed.image_sha256().unwrap(), "0".repeat(64));
        assert_eq!(parsed.direct_identity, None);
        let mut planned = index(1);
        let fields = planned.as_object_mut().unwrap();
        fields.remove("sha256");
        fields.insert("direct_identity".into(), "b".repeat(64).into());
        let parsed = parse_flash(&serde_json::to_vec(&planned).unwrap(), 1).unwrap();
        assert_eq!(
            parsed.direct_identity.as_deref(),
            Some("b".repeat(64).as_str())
        );
        assert!(parsed.image_sha256().is_err());
        let mut both = planned.clone();
        both["sha256"] = "0".repeat(64).into();
        assert!(parse_flash(&serde_json::to_vec(&both).unwrap(), 1).is_err());
        planned.as_object_mut().unwrap().remove("direct_identity");
        assert!(parse_flash(&serde_json::to_vec(&planned).unwrap(), 1).is_err());
        let mut short = index(1);
        short["sha256"] = "0".repeat(63).into();
        assert!(parse_flash(&serde_json::to_vec(&short).unwrap(), 1).is_err());
    }
}
