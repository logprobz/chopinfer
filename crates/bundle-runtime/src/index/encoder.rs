//! The index of a BF16 encoder image: a separate compiled image beside the
//! text rank images whose views are row-major BF16 at fixed offsets, listed
//! in the model's layout order. A model owns the layout (names, shapes,
//! offsets), its schema and its image role; this module writes and parses
//! the index against that layout.

use std::collections::BTreeMap;

use super::View;

/// Layout id of a row-major BF16 view.
pub const BF16_ROWS: &str = "bf16_row_major_v1";

/// The image file name of rank `rank`'s encoder image.
pub fn image_name(rank: u32) -> String {
    format!("encoder-rank{rank}.img")
}

/// One view of a model's encoder layout, in layout order.
pub trait LayoutView {
    /// The name the runtime looks the view up by.
    fn name(&self) -> &str;
    /// The stored row-major shape.
    fn shape(&self) -> &[u64];
    /// The byte offset in the image.
    fn offset(&self) -> u64;
    /// The stored BF16 bytes.
    fn bytes(&self) -> u64 {
        2 * self.shape().iter().product::<u64>()
    }
}

/// What an index names its image by.
#[derive(Clone, Copy, Debug)]
pub enum Identity<'a> {
    /// An image file's SHA-256.
    Sha256(&'a str),
    /// A planned rank's identity (`chop.weight-plan.v1`).
    Direct(&'a str),
}

#[derive(Clone, Debug)]
pub struct EncoderIndex {
    pub rank: u32,
    pub image_bytes: u64,
    /// The encoder image file's SHA-256.
    pub sha256: Option<String>,
    /// The identity of a directly planned rank, which has no image file.
    pub direct_identity: Option<String>,
    pub views: BTreeMap<String, View>,
}

impl EncoderIndex {
    /// The encoder image's SHA-256, refusing an index that has none.
    pub fn image_sha256(&self) -> Result<&str, String> {
        self.sha256
            .as_deref()
            .ok_or_else(|| format!("encoder rank {} index names no image digest", self.rank))
    }
}

/// The index of rank `rank`'s image of `layout` (`image_bytes` long) under
/// `schema`, named by `identity`.
pub fn index<V: LayoutView>(
    schema: &str,
    rank: u32,
    image_bytes: u64,
    layout: &[V],
    identity: Identity,
) -> serde_json::Value {
    let (key, value) = match identity {
        Identity::Sha256(digest) => ("sha256", digest),
        Identity::Direct(identity) => ("direct_identity", identity),
    };
    serde_json::json!({
        "schema": schema,
        "rank": rank,
        "image": image_name(rank),
        "image_bytes": image_bytes,
        key: value,
        "views": layout.iter().map(|view| serde_json::json!({
            "name": view.name(),
            "offset": view.offset(),
            "bytes": view.bytes(),
            "shape": view.shape(),
            "dtype": "BF16",
            "layout_id": BF16_ROWS,
        })).collect::<Vec<_>>(),
    })
}

/// Parses rank `rank`'s encoder index under `schema` and checks it against
/// `expected` exactly: every view in order, its shape, offset, dtype and
/// layout, and the image length `image_bytes`.
pub fn parse<V: LayoutView>(
    bytes: &[u8],
    rank: u32,
    schema: &str,
    image_bytes: u64,
    expected: &[V],
) -> Result<EncoderIndex, String> {
    let parsed = super::parse_named(bytes, rank, schema, &image_name(rank))?;
    if parsed.image_bytes != image_bytes {
        return Err("encoder index image length mismatch".into());
    }
    if parsed.views.len() != expected.len() {
        return Err("encoder index view count mismatch".into());
    }
    // Preserve the component's ordered-view contract as well as exact offsets.
    let root: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let listed = root["views"].as_array().ok_or("encoder index views")?;
    for (value, stored) in listed.iter().zip(expected) {
        let view = parsed
            .views
            .get(stored.name())
            .ok_or("missing encoder view")?;
        if value["name"].as_str() != Some(stored.name())
            || view.shape != stored.shape()
            || view.offset != stored.offset()
            || view.bytes != stored.bytes()
            || view.dtype != "BF16"
            || view.layout_id != BF16_ROWS
        {
            return Err(format!("invalid encoder view {}", stored.name()));
        }
    }
    Ok(EncoderIndex {
        rank,
        image_bytes,
        sha256: parsed.sha256,
        direct_identity: parsed.direct_identity,
        views: parsed.views,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = "chopinfer.test.encoder-image.v1";

    struct Stored {
        name: &'static str,
        shape: Vec<u64>,
        offset: u64,
    }

    impl LayoutView for Stored {
        fn name(&self) -> &str {
            self.name
        }
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn offset(&self) -> u64 {
            self.offset
        }
    }

    fn layout() -> Vec<Stored> {
        vec![
            Stored {
                name: "tower.b.weight",
                shape: vec![4, 8],
                offset: 0,
            },
            Stored {
                name: "tower.a.bias",
                shape: vec![4],
                offset: 4096,
            },
        ]
    }

    fn bytes(value: &serde_json::Value) -> Vec<u8> {
        serde_json::to_vec(value).unwrap()
    }

    #[test]
    fn an_index_parses_back_against_its_layout() {
        let layout = layout();
        let sha = "a".repeat(64);
        let written = index(SCHEMA, 1, 8192, &layout, Identity::Sha256(&sha));
        let parsed = parse(&bytes(&written), 1, SCHEMA, 8192, &layout).unwrap();
        assert_eq!(parsed.image_sha256().unwrap(), sha);
        assert_eq!(parsed.views["tower.a.bias"].bytes, 8);
        let plan = "b".repeat(64);
        let planned = index(SCHEMA, 1, 8192, &layout, Identity::Direct(&plan));
        let parsed = parse(&bytes(&planned), 1, SCHEMA, 8192, &layout).unwrap();
        assert_eq!(parsed.direct_identity, Some(plan));
        assert!(parsed.image_sha256().is_err());
    }

    #[test]
    fn the_index_must_match_the_layout_exactly() {
        let layout = layout();
        let plan = "b".repeat(64);
        let written = index(SCHEMA, 0, 8192, &layout, Identity::Direct(&plan));
        let refused =
            |value: &serde_json::Value| parse(&bytes(value), 0, SCHEMA, 8192, &layout).unwrap_err();
        assert!(parse(&bytes(&written), 1, SCHEMA, 8192, &layout).is_err());
        assert!(parse(&bytes(&written), 0, "other", 8192, &layout).is_err());
        assert_eq!(
            parse(&bytes(&written), 0, SCHEMA, 4096, &layout).unwrap_err(),
            "encoder index image length mismatch"
        );
        let mut reordered = written.clone();
        reordered["views"].as_array_mut().unwrap().reverse();
        assert_eq!(refused(&reordered), "invalid encoder view tower.b.weight");
        let mut reshaped = written.clone();
        reshaped["views"][1]["shape"] = serde_json::json!([2, 2]);
        assert_eq!(refused(&reshaped), "invalid encoder view tower.a.bias");
        let mut cast = written.clone();
        cast["views"][0]["dtype"] = serde_json::json!("F32");
        cast["views"][0]["bytes"] = serde_json::json!(128);
        assert_eq!(refused(&cast), "invalid encoder view tower.b.weight");
        let mut short = written.clone();
        short["views"].as_array_mut().unwrap().pop();
        assert_eq!(refused(&short), "encoder index view count mismatch");
    }
}
