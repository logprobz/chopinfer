//! Build-time identities derived from the same views and declarations as admission.
use crate::{contract::Shape, package, prepared, views};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub fn digest(value: &Value) -> String {
    let mut normalized = value.clone();
    normalized.sort_all_objects();
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&normalized).expect("contract JSON"))
    )
}

pub fn weight_contract(shape: Shape) -> String {
    let ngram_head_width = shape.ngram_head_width();
    let ranks: Vec<_> = (0..shape.ranks())
        .map(|rank| {
            views::required(rank, shape)
                .into_iter()
                .chain(views::drafter_required(shape))
                .map(
                    |v| json!({"name":v.name,"dtype":v.dtype,"layout":v.layout_id,"bytes":v.bytes}),
                )
                .collect::<Vec<_>>()
        })
        .collect();
    digest(
        &json!({"version":1,"views":ranks,"alignment":4096,"fp8":"e4m3fn",
        "block_scale":{"block":[128,128],"stored":"f32","meaning":"inverse-multiplier"},
        "qkv":"contiguous-q-k-v","padding":"zero","ngram_head_width":ngram_head_width}),
    )
}

// Runtime overrides are refused during geometry generation, so checkpoint
// option defaults equal the compiled limits used in these profile digests.
pub fn package_contract(shape: Shape) -> String {
    digest(&json!({"abi":runtime_bundle_abi::BUNDLE_ABI_VERSION_V1,
        "create":std::str::from_utf8(bundle_runtime::config::MAGIC).expect("ASCII"),
        "declarations":package::declarations(shape, shape.ranks()==2, crate::geometry::RuntimeValues { indexer_budget: shape.geometry.limits.indexer_budget as usize })}))
}

pub fn profile(shape: Shape) -> Value {
    json!({"profile":profile_name(shape, false),"tensor_parallel_size":shape.ranks(),
        "weight_contract":weight_contract(shape),"package_contract":package_contract(shape)})
}

pub fn profile_name(shape: Shape, prepared: bool) -> String {
    let prefix = if prepared {
        "prepared-text-ar"
    } else {
        "text-ar"
    };
    if shape.geometry.id == crate::geometry::FLASH.id {
        format!("{prefix}-tp{}", shape.ranks())
    } else {
        format!("{prefix}-{}-tp{}", shape.geometry.id, shape.ranks())
    }
}

pub fn document() -> Value {
    let profiles: Vec<_> = crate::geometry::GEOMETRIES
        .iter()
        .flat_map(|g| {
            let shapes = [crate::contract::Ranks::TP2, crate::contract::Ranks::TP4]
                .map(|n| Shape::new(g, n));
            [
                profile(shapes[0]),
                profile(shapes[1]),
                prepared_profile(shapes[0]),
                prepared_profile(shapes[1]),
            ]
        })
        .collect();
    json!({"version":1,"profiles":profiles})
}

/// The prepared profile's weight identity: the fixed views, every view
/// alternative a prepared component can make create require
/// ([`prepared::admitted`]) and the packed layouts they encode.
pub fn prepared_weight_contract(shape: Shape) -> String {
    prepared_weight_digest(
        &weight_contract(shape),
        prepared::admitted(shape).expect("admitted prepared views"),
    )
}

fn prepared_weight_digest(legacy: &str, admitted: &prepared::Admitted) -> String {
    let layouts: Vec<_> = admitted
        .layouts
        .iter()
        .map(|(id, version)| json!({"id":id,"version":version}))
        .collect();
    digest(
        &json!({"version":2,"legacy":legacy,"prepared_views":admitted.ranks,
        "packed_layouts":layouts}),
    )
}

pub fn prepared_profile(shape: Shape) -> Value {
    // Execution dependencies live in the native implementation contract. This
    // identity describes the admitted views and layouts and fixed model ports.
    let weight = prepared_weight_contract(shape);
    let package = digest(&json!({"abi":runtime_bundle_abi::BUNDLE_ABI_VERSION_V1,
        "create":std::str::from_utf8(bundle_runtime::config::MAGIC).expect("ASCII"),
        "declarations":package::prepared_declarations(shape, shape.ranks()==2, crate::geometry::RuntimeValues { indexer_budget: shape.geometry.limits.indexer_budget as usize })}));
    json!({"profile":profile_name(shape, true),"tensor_parallel_size":shape.ranks(),
        "weight_contract":weight,"package_contract":package})
}

/// Standard ELF note header and four-byte aligned owner/description.
pub fn elf_note() -> Vec<u8> {
    elf_note_document(&document())
}

/// Build-time publication can attach the audited implementation contract to
/// the same native consumption note without changing legacy profile identities.
pub fn elf_note_document(document: &Value) -> Vec<u8> {
    let description = serde_json::to_vec(document).expect("note JSON");
    model_compiler_sdk::native_contract::encode_native_note(&description)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prepared::ViewKey;

    /// The identity is derived from typed content, not from source bytes.
    #[test]
    fn identity_sources_embed_no_file_bytes() {
        for file in ["note.rs", "prepared.rs"] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src")
                .join(file);
            let text = std::fs::read_to_string(&path).unwrap();
            for embed in ["include_bytes", "include_str"] {
                assert!(!text.contains(&format!("{embed}!")), "{file} uses {embed}");
            }
        }
    }

    /// Every prepared profile's weight identity is the digest of its
    /// legacy contract and admitted table, and of nothing else.
    #[test]
    fn the_prepared_identity_is_a_function_of_the_admitted_table() {
        for shape in Shape::ALL {
            let legacy = weight_contract(shape);
            let admitted = prepared::admitted(shape).unwrap();
            let expected = prepared_weight_digest(&legacy, admitted);
            assert_eq!(
                prepared_profile(shape)["weight_contract"],
                expected.as_str()
            );
            assert_eq!(prepared_weight_digest(&legacy, &admitted.clone()), expected);
        }
    }

    /// A changed layout id, dtype or byte count of any admitted row (an
    /// `input_scale` row, an F32 norm row and a weight row, on the first
    /// and last rank), a changed layout version or a changed legacy
    /// contract each move the identity.
    #[test]
    fn every_admitted_field_moves_the_identity() {
        let shape = Shape::TP4;
        let legacy = weight_contract(shape);
        let admitted = prepared::admitted(shape).unwrap();
        let identity = prepared_weight_digest(&legacy, admitted);
        let mut altered = Vec::new();
        for rank in [0, admitted.ranks.len() - 1] {
            let rows = &admitted.ranks[rank];
            let position = |matches: &dyn Fn(&ViewKey) -> bool| {
                rows.iter().position(|v| matches(&v.key)).unwrap()
            };
            for row in [
                position(&|k| matches!(k, ViewKey::Port { suffix, .. } if suffix == "input_scale")),
                position(&|k| matches!(k, ViewKey::F32Norm { .. })),
                position(
                    &|k| matches!(k, ViewKey::Port { port, suffix, .. } if port == "ngram" && suffix == "weight"),
                ),
            ] {
                for change in 0..3 {
                    let mut table = admitted.clone();
                    let view = &mut table.ranks[rank][row];
                    match change {
                        0 => view.layout_id = "another_layout_v1",
                        1 => view.dtype = "F16",
                        _ => view.bytes += 1,
                    }
                    altered.push(table);
                }
            }
        }
        let mut table = admitted.clone();
        let (id, version) = table.layouts.pop_first().unwrap();
        table.layouts.insert((id, version + 1));
        altered.push(table);
        for table in &altered {
            assert_ne!(prepared_weight_digest(&legacy, table), identity);
        }
        assert_ne!(prepared_weight_digest(&"0".repeat(64), admitted), identity);
    }
}
