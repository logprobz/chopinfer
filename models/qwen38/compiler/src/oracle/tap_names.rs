//! The names of the bundle's taps (`models/qwen38/taps.json`). A dump of
//! the older schema (`chopinfer.qwen38.capture.v1`, or one without
//! `capture.json`) stores each tap under its `v1` name; readers take every
//! tap under its declared name.

use std::path::Path;
use std::sync::OnceLock;

use crate::Result;

const DECLARATION: &str = include_str!("../../../taps.json");
const V1_SCHEMA: &str = "chopinfer.qwen38.capture.v1";

/// One declared tap: its name, its v1 name pattern (none for a tap newer
/// than the v1 schema), the layers it applies to, its sharding when the
/// declaration names one (`replicated`, `rank_partial`, ...), none for a
/// column-block layout, and whether it compares densely (not as a set or
/// paired with another tap).
struct Entry {
    name: String,
    v1: Option<String>,
    layers: Option<Vec<u32>>,
    sharding: Option<String>,
    dense: bool,
}

fn entries() -> &'static [Entry] {
    static ENTRIES: OnceLock<Vec<Entry>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        let value: serde_json::Value =
            serde_json::from_str(DECLARATION).expect("taps.json is valid JSON");
        value["taps"]
            .as_array()
            .expect("taps.json lists its taps")
            .iter()
            .map(|entry| Entry {
                name: entry["name"].as_str().expect("tap name").to_owned(),
                v1: entry["v1"].as_str().map(str::to_owned),
                layers: entry["layers"].as_array().map(|layers| {
                    layers
                        .iter()
                        .map(|layer| layer.as_u64().expect("layer") as u32)
                        .collect()
                }),
                sharding: entry["sharding"].as_str().map(str::to_owned),
                dense: entry["compare"] == "dense",
            })
            .collect()
    })
}

/// The two-digit layer `{nn}` stands for in `name`, `Some(None)` when the
/// pattern has none, or `None` when `name` does not match.
fn matches(pattern: &str, name: &str) -> Option<Option<u32>> {
    let Some((head, tail)) = pattern.split_once("{nn}") else {
        return (pattern == name).then_some(None);
    };
    let digits = name.strip_prefix(head)?.strip_suffix(tail)?;
    (digits.len() == 2 && digits.bytes().all(|b| b.is_ascii_digit())).then(|| digits.parse().ok())
}

/// The declared name of a v1 tap name; none for a tap the declaration does
/// not list (one of an older build).
pub fn from_v1(stored: &str) -> Option<String> {
    entries().iter().find_map(|entry| {
        let layer = matches(entry.v1.as_deref()?, stored)?;
        if let Some(layers) = &entry.layers
            && !layer.is_some_and(|layer| layers.contains(&layer))
        {
            return None;
        }
        Some(match layer {
            Some(layer) => entry.name.replace("{nn}", &format!("{layer:02}")),
            None => entry.name.clone(),
        })
    })
}

/// The declared sharding of a densely compared tap `name` when the
/// declaration names one; none for a column-block layout, a set or paired
/// tap, or an undeclared tap.
pub fn dense_sharding(name: &str) -> Option<&'static str> {
    let entry = entries().iter().find(|entry| {
        matches(&entry.name, name).is_some_and(|layer| {
            entry
                .layers
                .as_ref()
                .is_none_or(|layers| layer.is_some_and(|layer| layers.contains(&layer)))
        })
    })?;
    entry.sharding.as_deref().filter(|_| entry.dense)
}

/// Whether step `dir` names its taps in the older schema.
pub fn is_v1(dir: &Path) -> Result<bool> {
    let path = dir.join("capture.json");
    if !path.exists() {
        return Ok(true);
    }
    let capture: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    Ok(capture["schema"]
        .as_str()
        .is_none_or(|schema| schema == V1_SCHEMA))
}

/// The tap name of `stored` in a step: itself, or its declared name when
/// the step is `v1`; none for an undeclared v1 tap.
pub fn name(stored: &str, v1: bool) -> Option<String> {
    if v1 {
        from_v1(stored)
    } else {
        Some(stored.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_names_take_their_declared_names() {
        for (v1, name) in [
            ("embed", "embed.out"),
            ("layer03.out.x", "L03.out"),
            ("layer01.in.x", "L01.x.hc.attn.streams"),
            ("layer05.in.x", "L05.in.streams"),
            ("layer02.lin.normed", "L02.attn.core"),
            ("layer03.attn.gated", "L03.attn.core"),
            ("layer07.attn.selected", "L07.x.indexer.topk"),
            ("final.u", "final.x.hc.mixer"),
        ] {
            assert_eq!(from_v1(v1).as_deref(), Some(name), "{v1}");
        }
        assert_eq!(from_v1("layer03.attn.renamed"), None);
        assert_eq!(from_v1("layer02.attn.selected"), None);
        assert_eq!(name("L03.out", false).as_deref(), Some("L03.out"));
    }

    #[test]
    fn dense_shardings_follow_the_first_matching_declaration() {
        assert_eq!(dense_sharding("L05.out"), Some("replicated"));
        assert_eq!(dense_sharding("L05.mlp.out"), Some("rank_partial"));
        assert_eq!(dense_sharding("L07.x.indexer.qk"), Some("replicated"));
        assert_eq!(dense_sharding("final.x.hc.mixer"), Some("replicated"));
        // Column blocks, a full-attention tap outside its layers, set and
        // paired taps.
        assert_eq!(dense_sharding("L03.attn.core"), None);
        assert_eq!(dense_sharding("L02.x.indexer.qk"), None);
        assert_eq!(dense_sharding("L03.renamed"), None);
        assert_eq!(dense_sharding("L05.router.ids"), None);
        assert_eq!(dense_sharding("L05.router.weights"), None);
    }
}
