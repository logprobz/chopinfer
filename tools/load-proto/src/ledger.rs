//! Digest ledgers of direct loads: what proves device bytes unchanged across
//! a change without a reference image.
//!
//! One ledger entry describes one image role of one topology: per rank the
//! arena length, the plan's digest, its components' digests and the SHA-256
//! of the bytes the host's executor produces (the bytes a device arena
//! receives). The plan digest covers the encoded plan without its
//! components, and a component's digest leaves out the identity fields a
//! planned rank index carries (`direct_identity`, `sha256`): both name the
//! implementation, which changes with any edit to the host's byte path, while
//! the ledger compares what the bytes are. Two builds whose ledgers are equal
//! for a checkpoint upload equal arenas.
use runtime_bundle_abi::weight_plan::{NamedBytesV1, WeightPlanV1};
use sha2::{Digest, Sha256};

pub const FORMAT: &str = "chopinfer-direct-ledger-v1";

fn hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// SHA-256 of the plan's encoding with its components left out.
pub fn plan_sha256(plan: &WeightPlanV1) -> String {
    hex(&WeightPlanV1 {
        components: Vec::new(),
        ..plan.clone()
    }
    .encode())
}

/// SHA-256 of a component's bytes; a JSON document drops the identity
/// fields a planned rank index names, since those cover the implementation.
pub fn component_sha256(component: &NamedBytesV1) -> String {
    match serde_json::from_slice::<serde_json::Value>(&component.bytes) {
        Ok(serde_json::Value::Object(mut document)) => {
            document.shift_remove("direct_identity");
            document.shift_remove("sha256");
            hex(&serde_json::to_vec(&document).unwrap_or_default())
        }
        _ => hex(&component.bytes),
    }
}

/// The ledger fields of one rank's plan: arena length, plan and component
/// digests. The bytes digest is added once the rank has been streamed.
pub fn rank_entry(plan: &WeightPlanV1) -> serde_json::Value {
    let components: serde_json::Map<String, serde_json::Value> = plan
        .components
        .iter()
        .map(|c| (c.name.clone(), component_sha256(c).into()))
        .collect();
    serde_json::json!({
        "rank": plan.rank,
        "arena_bytes": plan.arena_bytes,
        "records": plan.records.len(),
        "plan_sha256": plan_sha256(plan),
        "components": components,
    })
}

/// The rank entries of `document`: a ledger entry's `per_rank`, or a `bench
/// --digest` report's (`bytes` and `sha256`), so a report from a build
/// before ledgers existed is a reference too (bytes only).
fn ranks(document: &serde_json::Value) -> Result<Vec<serde_json::Value>, String> {
    let per_rank = document["per_rank"]
        .as_array()
        .ok_or("the reference has no per_rank entries")?;
    Ok(per_rank
        .iter()
        .map(|rank| {
            serde_json::json!({
                "rank": rank["rank"],
                "arena_bytes": rank.get("arena_bytes").unwrap_or(&rank["bytes"]),
                "plan_sha256": rank.get("plan_sha256"),
                "components": rank.get("components"),
                "bytes_sha256": rank.get("bytes_sha256").unwrap_or(&rank["sha256"]),
            })
        })
        .collect())
}

/// Differences between `reference` and `current` (ledger entries or bench
/// reports of the same image role and topology); empty when equal. Fields
/// the reference lacks are not compared; the bytes digest always is.
pub fn compare(reference: &serde_json::Value, current: &serde_json::Value) -> Vec<String> {
    let mut differences = Vec::new();
    for key in ["image_role", "ranks", "layers", "checkpoint_identity"] {
        if !reference[key].is_null() && reference[key] != current[key] {
            differences.push(format!(
                "{key}: reference {} current {}",
                reference[key], current[key]
            ));
        }
    }
    let (reference, current) = match (ranks(reference), ranks(current)) {
        (Ok(r), Ok(c)) => (r, c),
        (Err(e), _) | (_, Err(e)) => return vec![e],
    };
    if reference.len() != current.len() {
        differences.push(format!(
            "{} reference ranks, {} current",
            reference.len(),
            current.len()
        ));
        return differences;
    }
    for (expected, actual) in reference.iter().zip(&current) {
        let rank = &expected["rank"];
        if expected["bytes_sha256"].is_null() {
            differences.push(format!("rank {rank}: the reference has no bytes digest"));
        }
        for key in [
            "rank",
            "arena_bytes",
            "bytes_sha256",
            "plan_sha256",
            "components",
        ] {
            if !expected[key].is_null() && expected[key] != actual[key] {
                differences.push(format!(
                    "rank {rank} {key}: reference {} current {}",
                    expected[key], actual[key]
                ));
            }
        }
    }
    differences
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(index: &str) -> WeightPlanV1 {
        WeightPlanV1 {
            ranks: 1,
            rank: 0,
            image_role: "role".into(),
            alignment: 256,
            arena_bytes: 512,
            views: Vec::new(),
            components: vec![NamedBytesV1 {
                name: "index".into(),
                bytes: index.as_bytes().to_vec(),
            }],
            records: Vec::new(),
        }
    }

    #[test]
    fn identities_are_outside_the_digests() {
        let a = rank_entry(&plan(r#"{"image_bytes":512,"direct_identity":"aa"}"#));
        let b = rank_entry(&plan(r#"{"image_bytes":512,"direct_identity":"bb"}"#));
        assert_eq!(a, b);
        let c = rank_entry(&plan(r#"{"image_bytes":256,"direct_identity":"aa"}"#));
        assert_ne!(a["components"], c["components"]);
        assert_eq!(a["plan_sha256"], c["plan_sha256"]);
    }

    #[test]
    fn a_bench_report_compares_by_bytes() {
        let bench = serde_json::json!({
            "image_role": "role", "ranks": 1,
            "per_rank": [{"rank": 0, "bytes": 512, "sha256": "ab"}],
        });
        let mut entry = rank_entry(&plan("{}"));
        entry["bytes_sha256"] = "ab".into();
        let ledger = serde_json::json!({"image_role": "role", "ranks": 1, "per_rank": [entry]});
        assert!(compare(&bench, &ledger).is_empty());
        let mut changed = ledger.clone();
        changed["per_rank"][0]["bytes_sha256"] = "cd".into();
        assert_eq!(compare(&bench, &changed).len(), 1);
        assert_eq!(compare(&ledger, &changed).len(), 1);
        let mut replanned = ledger.clone();
        replanned["per_rank"][0]["plan_sha256"] = "00".into();
        assert_eq!(compare(&ledger, &replanned).len(), 1);
        assert!(compare(&bench, &replanned).is_empty());
    }
}
