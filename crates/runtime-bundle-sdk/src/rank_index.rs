//! Checked per-rank image indexes.

use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct View {
    pub offset: u64,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub layout_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankIndex {
    pub rank: u32,
    pub image: String,
    pub image_bytes: u64,
    /// The rank image file's SHA-256.
    pub sha256: Option<String>,
    /// The identity of a directly planned rank, which has no image file.
    pub direct_identity: Option<String>,
    pub views: BTreeMap<String, View>,
}

impl RankIndex {
    /// The rank image's SHA-256, refusing an index that has none.
    pub fn image_sha256(&self) -> Result<&str, String> {
        self.sha256
            .as_deref()
            .ok_or_else(|| format!("rank {} index names no image digest", self.rank))
    }
}

fn integer(value: &serde_json::Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("rank index needs unsigned {key}"))
}

fn string<'a>(value: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("rank index needs string {key}"))
}

/// Parses `rank`'s index of one rank image written under `schema`.
pub fn parse(bytes: &[u8], rank: u32, schema: &str) -> Result<RankIndex, String> {
    parse_named(bytes, rank, schema, &format!("rank{rank}.img"))
}

/// Parses an index with the image name required by its component contract.
pub fn parse_named(
    bytes: &[u8],
    rank: u32,
    schema: &str,
    image_name: &str,
) -> Result<RankIndex, String> {
    let root: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| format!("rank index JSON: {error}"))?;
    if string(&root, "schema")? != schema || integer(&root, "rank")? != u64::from(rank) {
        return Err(format!("rank {rank} index schema or rank mismatch"));
    }
    let image = string(&root, "image")?;
    if image != image_name {
        return Err(format!("rank {rank} index image name mismatch"));
    }
    let image_bytes = integer(&root, "image_bytes")?;
    if image_bytes == 0 || !image_bytes.is_multiple_of(4096) {
        return Err(format!("rank {rank} image length is not 4096 aligned"));
    }
    // An image's index carries its SHA-256; a directly planned rank has no
    // image and carries its plan's identity instead. Exactly one is present.
    let digest = |key: &str| -> Result<Option<String>, String> {
        let Some(value) = root.get(key) else {
            return Ok(None);
        };
        match value.as_str() {
            Some(text) if text.len() == 64 && text.bytes().all(|b| b.is_ascii_hexdigit()) => {
                Ok(Some(text.to_owned()))
            }
            _ => Err(format!("rank {rank} index {key} is invalid")),
        }
    };
    let (sha256, direct_identity) = (digest("sha256")?, digest("direct_identity")?);
    if sha256.is_some() == direct_identity.is_some() {
        return Err(format!(
            "rank {rank} index needs exactly one of an image sha256 and a direct identity"
        ));
    }
    let listed = root
        .get("views")
        .and_then(serde_json::Value::as_array)
        .ok_or("rank index needs views array")?;
    let mut views = BTreeMap::new();
    for value in listed {
        let name = string(value, "name")?;
        let offset = integer(value, "offset")?;
        let length = integer(value, "bytes")?;
        let dtype = string(value, "dtype")?;
        let layout_id = string(value, "layout_id")?;
        let shape = value
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or("rank view needs shape array")?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .filter(|value| *value > 0)
                    .ok_or("rank view has invalid shape")
            })
            .collect::<Result<Vec<_>, _>>()?;
        if name.is_empty()
            || dtype.is_empty()
            || layout_id.is_empty()
            || shape.is_empty()
            || offset % 4096 != 0
            || length == 0
            || offset
                .checked_add(length)
                .is_none_or(|end| end > image_bytes)
        {
            return Err(format!("rank {rank} view {name} is invalid"));
        }
        if views
            .insert(
                name.to_owned(),
                View {
                    offset,
                    bytes: length,
                    dtype: dtype.to_owned(),
                    shape,
                    layout_id: layout_id.to_owned(),
                },
            )
            .is_some()
        {
            return Err(format!("rank {rank} has duplicate view {name}"));
        }
    }
    if views.is_empty() {
        return Err(format!("rank {rank} index has no views"));
    }
    validate_extents(&views, image_bytes)?;
    Ok(RankIndex {
        rank,
        image: image.to_owned(),
        image_bytes,
        sha256,
        direct_identity,
        views,
    })
}

/// Exact model-owned geometry for one materialized view.
#[derive(Clone, Debug)]
pub struct ViewContract {
    pub shape: Vec<u64>,
    pub dtype: &'static str,
    pub layout_id: &'static str,
    pub bytes: u64,
}

pub fn validate_extents(views: &BTreeMap<String, View>, image_bytes: u64) -> Result<(), String> {
    let mut extents = Vec::new();
    for (name, view) in views {
        let end = view
            .offset
            .checked_add(view.bytes)
            .filter(|end| *end <= image_bytes)
            .ok_or_else(|| format!("view {name} exceeds image"))?;
        if view.bytes == 0 || view.offset % 4096 != 0 {
            return Err(format!("invalid extent for {name}"));
        }
        extents.push((view.offset, end, name));
    }
    extents.sort_unstable();
    for pair in extents.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(format!("views {} and {} overlap", pair[0].2, pair[1].2));
        }
    }
    Ok(())
}

pub fn validate_contract(
    views: &BTreeMap<String, View>,
    image_bytes: u64,
    expected: &BTreeMap<String, ViewContract>,
) -> Result<(), String> {
    if views.len() != expected.len() {
        return Err("view count differs from model contract".into());
    }
    for (name, contract) in expected {
        let view = views
            .get(name)
            .ok_or_else(|| format!("missing view {name}"))?;
        if view.shape != contract.shape
            || view.dtype != contract.dtype
            || view.layout_id != contract.layout_id
            || view.bytes != contract.bytes
        {
            return Err(format!(
                "view {name} geometry or layout differs from model contract"
            ));
        }
    }
    validate_extents(views, image_bytes)
}

#[cfg(test)]
mod tests {

    fn direct(fields: &str) -> Vec<u8> {
        format!(r#"{{"schema":"example.rank-image.v1","rank":0,"image":"rank0.img","image_bytes":4096,{fields}"views":[{{"name":"model.norm.weight","offset":0,"bytes":128,"dtype":"BF16","shape":[64],"layout_id":"example.row.v1"}}]}}"#).into_bytes()
    }

    /// A directly planned rank has no image: its index names the plan's
    /// identity instead of an image digest, and exactly one of the two.
    #[test]
    fn indexes_carry_an_image_digest_or_a_direct_identity() {
        let (image, planned) = ("a".repeat(64), "b".repeat(64));
        let index = parse(
            &direct(&format!(r#""sha256":"{image}","#)),
            0,
            "example.rank-image.v1",
        )
        .unwrap();
        assert_eq!(
            (index.sha256.as_deref(), index.direct_identity.as_deref()),
            (Some(image.as_str()), None)
        );
        assert_eq!(index.image_sha256().unwrap(), image);
        let index = parse(
            &direct(&format!(r#""direct_identity":"{planned}","#)),
            0,
            "example.rank-image.v1",
        )
        .unwrap();
        assert_eq!(
            (index.sha256.as_deref(), index.direct_identity.as_deref()),
            (None, Some(planned.as_str()))
        );
        assert!(index.image_sha256().is_err());
        for fields in [
            String::new(),
            format!(r#""sha256":"{image}","direct_identity":"{planned}","#),
            r#""direct_identity":"xyz","#.into(),
        ] {
            assert!(
                parse(&direct(&fields), 0, "example.rank-image.v1").is_err(),
                "{fields}"
            );
        }
    }

    use super::*;

    #[test]
    fn parses_an_index_and_rejects_out_of_bounds_views() {
        let sample = r#"{"schema":"example.rank-image.v1","rank":0,"image":"rank0.img","image_bytes":4096,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","views":[{"name":"model.norm.weight","offset":0,"bytes":128,"dtype":"BF16","shape":[64],"layout_id":"example.row.v1","source_tensor_digests":{}}]}"#;
        let schema = "example.rank-image.v1";
        let index = parse(sample.as_bytes(), 0, schema).expect("rank index");
        assert_eq!(index.views["model.norm.weight"].bytes, 128);
        assert!(parse(sample.as_bytes(), 1, schema).is_err());
        assert!(parse(sample.as_bytes(), 0, "other.rank-image.v1").is_err());
        assert!(
            parse(
                sample.replace("\"bytes\":128", "\"bytes\":8192").as_bytes(),
                0,
                schema
            )
            .is_err()
        );
    }
}
