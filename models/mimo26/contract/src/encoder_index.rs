//! Authenticate the separate encoder image index before resolving device views.

use std::collections::BTreeMap;

use crate::index::View;

pub const IMAGE_ROLE: &[u8] = b"mimo.encoder-rank-images.v1";
/// v2 lays the image out as the encoder kernels read it
/// ([`crate::encoders::prepared_flash`]); v1 stored the checkpoint's views with
/// GEMM weights in MMA fragment order, which the bundle prepared again.
pub const SCHEMA: &str = "chopinfer.mimo26.encoder-image.v2";

#[derive(Clone, Debug)]
pub struct EncoderIndex {
    pub rank: u32,
    pub image_bytes: u64,
    /// The encoder image file's SHA-256.
    pub sha256: Option<String>,
    /// The identity of a directly planned rank, which has no image file.
    pub direct_identity: Option<String>,
    /// Whether the image holds the vision tower's tensors.
    pub vision: bool,
    /// Whether the image holds the audio tower's and tokenizer's tensors.
    pub audio: bool,
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

fn string<'a>(value: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("encoder index needs string {key}"))
}

/// The index of one rank's encoder image, holding the tensors of the
/// encoders its views name (`crate::encoders::Encoders::holds`), each of
/// them whole.
#[cfg(test)]
pub fn parse(bytes: &[u8], rank: u32) -> Result<EncoderIndex, String> {
    parse_for(&crate::geometry::V26_FLASH, bytes, rank)
}

pub fn parse_for(
    geometry: &'static crate::geometry::Geometry,
    bytes: &[u8],
    rank: u32,
) -> Result<EncoderIndex, String> {
    let root: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| format!("encoder index JSON: {error}"))?;
    let entries = root
        .get("views")
        .and_then(serde_json::Value::as_array)
        .ok_or("encoder index needs views array")?;
    // The image's views name the encoders it holds, each whole.
    let mut encoders = crate::encoders::Encoders::NONE;
    for value in entries {
        if string(value, "name")?.starts_with("visual.") {
            (encoders.image, encoders.video) = (true, true);
        } else {
            encoders.audio = true;
        }
    }
    if !encoders.any() {
        return Err("encoder index names no view".into());
    }
    let image_bytes = encoders.image_bytes_for(geometry);
    let parsed = runtime_bundle::rank_index::parse_named(
        bytes,
        rank,
        SCHEMA,
        &format!("encoder-rank{rank}.img"),
    )?;
    if parsed.image_bytes != image_bytes {
        return Err("encoder index image length mismatch".into());
    }
    let contract = crate::encoders::views_for(geometry, encoders);
    runtime_bundle::rank_index::validate_contract(&parsed.views, image_bytes, &contract)?;
    let (sha256, direct_identity, views) = (parsed.sha256, parsed.direct_identity, parsed.views);
    Ok(EncoderIndex {
        rank,
        image_bytes,
        sha256,
        direct_identity,
        vision: encoders.vision(),
        audio: encoders.audio,
        views,
    })
}

#[cfg(test)]
mod geometry_tests {
    fn valid_index() -> serde_json::Value {
        index_of(crate::encoders::Encoders::ALL)
    }

    fn index_of(encoders: crate::encoders::Encoders) -> serde_json::Value {
        let mut offset = 0_u64;
        let views = crate::encoders::views_of_flash(encoders)
            .into_iter()
            .map(|(name, v)| {
                let value = serde_json::json!({"name":name,"offset":offset,"bytes":v.bytes,
                "shape":v.shape,"dtype":v.dtype,"layout_id":v.layout_id});
                offset += v.bytes.next_multiple_of(4096);
                value
            })
            .collect::<Vec<_>>();
        assert_eq!(offset, encoders.image_bytes_flash());
        let root = serde_json::json!({"schema":super::SCHEMA,"rank":0,"image":"encoder-rank0.img",
            "image_bytes":offset,"sha256":"a".repeat(64),"views":views});
        root
    }

    /// An image of one encoder holds that encoder's views, each of them;
    /// its index names the encoders it holds.
    #[test]
    fn an_image_holds_the_views_of_each_encoder_it_names() {
        use crate::encoders::Encoders;
        for encoders in [
            Encoders::of(true, false, false),
            Encoders::of(false, false, true),
            Encoders::ALL,
        ] {
            let index = index_of(encoders);
            let parsed = super::parse(&serde_json::to_vec(&index).unwrap(), 0).unwrap();
            assert_eq!(
                (parsed.vision, parsed.audio),
                (encoders.vision(), encoders.audio)
            );
            assert_eq!(parsed.image_bytes, encoders.image_bytes_flash());
            let mut partial = index;
            partial["views"].as_array_mut().unwrap().pop();
            assert!(super::parse(&serde_json::to_vec(&partial).unwrap(), 0).is_err());
        }
        let mut empty = index_of(Encoders::of(true, true, false));
        empty["views"] = serde_json::json!([]);
        assert!(super::parse(&serde_json::to_vec(&empty).unwrap(), 0).is_err());
    }

    /// A directly planned rank has no image: its index names the plan's
    /// identity instead of an image digest, and exactly one of the two.
    #[test]
    fn indexes_carry_an_image_digest_or_a_direct_identity() {
        let image = valid_index();
        let parsed = super::parse(&serde_json::to_vec(&image).unwrap(), 0).unwrap();
        assert_eq!(parsed.image_sha256().unwrap(), "a".repeat(64));
        assert_eq!(parsed.direct_identity, None);
        let mut planned = image.clone();
        let fields = planned.as_object_mut().unwrap();
        fields.remove("sha256");
        fields.insert("direct_identity".into(), "b".repeat(64).into());
        let parsed = super::parse(&serde_json::to_vec(&planned).unwrap(), 0).unwrap();
        assert_eq!(
            parsed.direct_identity.as_deref(),
            Some("b".repeat(64).as_str())
        );
        assert!(parsed.image_sha256().is_err());
        let mut both = image.clone();
        both["direct_identity"] = "b".repeat(64).into();
        let mut neither = image;
        neither.as_object_mut().unwrap().remove("sha256");
        for bad in [both, neither] {
            assert!(super::parse(&serde_json::to_vec(&bad).unwrap(), 0).is_err());
        }
    }

    #[test]
    fn exact_encoder_contract_rejects_shape_mutations() {
        let root = valid_index();
        assert!(super::parse(&serde_json::to_vec(&root).unwrap(), 0).is_ok());
        let position = root["views"]
            .as_array()
            .unwrap()
            .iter()
            .position(|v| v["name"] == "audio_encoder.projection.mlp.0.weight")
            .unwrap();
        let sinks = root["views"]
            .as_array()
            .unwrap()
            .iter()
            .position(|v| v["name"] == "visual.blocks.1.attn.sinks")
            .unwrap();
        for mutation in 0..6 {
            let mut bad = root.clone();
            let view = &mut bad["views"][position];
            match mutation {
                0 => {
                    view["shape"] = serde_json::json!([1]);
                    view["bytes"] = serde_json::json!(2);
                }
                1 => view["shape"] = serde_json::json!([4096, 16384]),
                2 => view["shape"] = serde_json::json!([1, 16384, 4096]),
                3 => view["shape"] = serde_json::json!([16384 * 4096]),
                4 => view["layout_id"] = serde_json::json!("bf16_mma_fragments_16x16_v1"),
                _ => {
                    // Sinks are widened to F32; a BF16 view of them is not
                    // the image's.
                    let sink = &mut bad["views"][sinks];
                    sink["bytes"] = (sink["bytes"].as_u64().unwrap() / 2).into();
                    sink["dtype"] = serde_json::json!("BF16");
                    sink["layout_id"] = serde_json::json!("bf16_row_major_v1");
                }
            }
            assert!(
                super::parse(&serde_json::to_vec(&bad).unwrap(), 0).is_err(),
                "mutation {mutation}"
            );
        }
    }

    #[test]
    fn refuses_missing_duplicate_overlap_extent_alignment_and_digest_drift() {
        for mutation in 0..7 {
            let mut root = valid_index();
            let expected = match mutation {
                0 => {
                    root["views"].as_array_mut().unwrap().pop();
                    "view count"
                }
                1 => {
                    root["views"][1] = root["views"][0].clone();
                    "duplicate view"
                }
                2 => {
                    root["views"][1]["offset"] = 0.into();
                    "overlap"
                }
                3 => {
                    root["views"][0]["offset"] = crate::encoders::image_bytes_flash().into();
                    "is invalid"
                }
                4 => {
                    root["views"][0]["offset"] = 1.into();
                    "is invalid"
                }
                5 => {
                    root["sha256"] = "z".repeat(64).into();
                    "sha256"
                }
                _ => {
                    root["views"][0]["name"] = "unknown.weight".into();
                    "missing view"
                }
            };
            let error = super::parse(&serde_json::to_vec(&root).unwrap(), 0).unwrap_err();
            assert!(error.contains(expected), "mutation {mutation}: {error}");
        }
    }

    #[test]
    fn rejects_complete_one_element_encoder_views() {
        let views = crate::encoders::views_flash()
            .into_keys()
            .enumerate()
            .map(|(i, name)| {
                serde_json::json!({"name":name,"offset":i*4096,
                "bytes":2,"shape":[1],"dtype":"BF16","layout_id":"bf16_row_major_v1"})
            })
            .collect::<Vec<_>>();
        let bytes = serde_json::to_vec(&serde_json::json!({"schema":super::SCHEMA,
            "rank":0,"image":"encoder-rank0.img","image_bytes":crate::encoders::image_bytes_flash(),
            "sha256":"a".repeat(64),"views":views}))
        .unwrap();
        assert!(super::parse(&bytes, 0).is_err());
    }
}
