//! Checked rank indexes shared with bundle authors.

pub use runtime_bundle_sdk::rank_index::*;

use std::collections::BTreeMap;

/// Appends one view's contract fields, each length-prefixed so that no two
/// view sets encode alike.
fn encode_fields(
    out: &mut Vec<u8>,
    name: &str,
    dtype: &str,
    shape: &[u64],
    bytes: u64,
    layout_id: &str,
) {
    for text in [name, dtype, layout_id] {
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    out.extend_from_slice(&(shape.len() as u64).to_le_bytes());
    for dim in shape {
        out.extend_from_slice(&dim.to_le_bytes());
    }
    out.extend_from_slice(&bytes.to_le_bytes());
}

const CONTRACT_DIGEST_TAG: &[u8] = b"chopinfer.rank-view-contract.v1\0";

/// SHA-256 of `expected`'s contracts: every view's name, dtype, shape, byte
/// length and layout id, in name order. A model can pin this digest of
/// contracts it derives and check indexes with [`validate_pinned`].
pub fn contract_sha256(expected: &BTreeMap<String, ViewContract>) -> String {
    let mut encoded = CONTRACT_DIGEST_TAG.to_vec();
    for (name, view) in expected {
        encode_fields(
            &mut encoded,
            name,
            view.dtype,
            &view.shape,
            view.bytes,
            view.layout_id,
        );
    }
    runtime_weights::sha256_bytes(&encoded)
}

/// [`contract_sha256`] of the contracts `views` carry. Offsets are left
/// out: they place views in one image and are checked by
/// [`validate_extents`].
pub fn views_sha256(views: &BTreeMap<String, View>) -> String {
    let mut encoded = CONTRACT_DIGEST_TAG.to_vec();
    for (name, view) in views {
        encode_fields(
            &mut encoded,
            name,
            &view.dtype,
            &view.shape,
            view.bytes,
            &view.layout_id,
        );
    }
    runtime_weights::sha256_bytes(&encoded)
}

/// [`validate_contract`] against `pinned`, the [`contract_sha256`] of the
/// expected contracts. Equal digests mean the views carry exactly the
/// expected names, dtypes, shapes, lengths and layouts, so only their
/// extents remain to check. `None` when the digests differ: the caller then
/// checks the expected contracts themselves, which names the difference.
pub fn validate_pinned(
    views: &BTreeMap<String, View>,
    image_bytes: u64,
    pinned: &str,
) -> Option<Result<(), String>> {
    (views_sha256(views) == pinned).then(|| validate_extents(views, image_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contracts() -> BTreeMap<String, ViewContract> {
        [
            ("a.weight", vec![64, 128], "F8_E4M3", "fp8_rows_v1", 8192),
            ("b.weight", vec![64], "BF16", "bf16_row_major_v1", 128),
        ]
        .into_iter()
        .map(|(name, shape, dtype, layout_id, bytes)| {
            (
                name.to_owned(),
                ViewContract {
                    shape,
                    dtype,
                    layout_id,
                    bytes,
                },
            )
        })
        .collect()
    }

    fn views(expected: &BTreeMap<String, ViewContract>) -> BTreeMap<String, View> {
        let mut offset = 0;
        expected
            .iter()
            .map(|(name, contract)| {
                let view = View {
                    offset,
                    bytes: contract.bytes,
                    dtype: contract.dtype.into(),
                    shape: contract.shape.clone(),
                    layout_id: contract.layout_id.into(),
                };
                offset += contract.bytes.next_multiple_of(4096);
                (name.clone(), view)
            })
            .collect()
    }

    /// The digest of an index's views equals the digest of the contracts
    /// they carry, and changing any one contract field of one view, or
    /// adding or renaming a view, changes it.
    #[test]
    fn view_digests_cover_every_contract_field() {
        let expected = contracts();
        let pinned = contract_sha256(&expected);
        let good = views(&expected);
        assert_eq!(views_sha256(&good), pinned);
        assert_eq!(validate_pinned(&good, 1 << 20, &pinned), Some(Ok(())));
        assert!(validate_contract(&good, 1 << 20, &expected).is_ok());
        let edits: [fn(&mut BTreeMap<String, View>); 7] = [
            |v| v.get_mut("a.weight").unwrap().dtype = "BF16".into(),
            |v| v.get_mut("a.weight").unwrap().shape = vec![128, 64],
            |v| v.get_mut("a.weight").unwrap().shape.push(1),
            |v| v.get_mut("a.weight").unwrap().bytes -= 4096,
            |v| v.get_mut("b.weight").unwrap().layout_id = "bf16_mma_v1".into(),
            |v| {
                let view = v.remove("b.weight").unwrap();
                v.insert("c.weight".into(), view);
            },
            |v| {
                let mut view = v["b.weight"].clone();
                view.offset += 4096;
                v.insert("d.weight".into(), view);
            },
        ];
        for (index, edit) in edits.iter().enumerate() {
            let mut changed = good.clone();
            edit(&mut changed);
            assert_ne!(views_sha256(&changed), pinned, "edit {index}");
            assert_eq!(validate_pinned(&changed, 1 << 20, &pinned), None);
            assert!(
                validate_contract(&changed, 1 << 20, &expected).is_err(),
                "edit {index}"
            );
        }
    }

    /// Offsets are not contract fields: moved views keep the digest, and
    /// their extents are still checked.
    #[test]
    fn pinned_validation_checks_extents() {
        let expected = contracts();
        let pinned = contract_sha256(&expected);
        let mut moved = views(&expected);
        moved.get_mut("b.weight").unwrap().offset += 4096;
        assert_eq!(validate_pinned(&moved, 1 << 20, &pinned), Some(Ok(())));
        let mut overlapping = views(&expected);
        overlapping.get_mut("b.weight").unwrap().offset = 0;
        assert!(matches!(
            validate_pinned(&overlapping, 1 << 20, &pinned),
            Some(Err(_))
        ));
        assert!(matches!(
            validate_pinned(&views(&expected), 4096, &pinned),
            Some(Err(_))
        ));
    }
}
