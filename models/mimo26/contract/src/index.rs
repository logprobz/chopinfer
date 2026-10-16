//! MiMo per-rank materialization indexes.

pub use crate::Tp;
pub use runtime_bundle::rank_index::{RankIndex, View};

/// The schema every MiMo rank image index is written under.
pub const SCHEMA: &str = "chopinfer.mimo26.rank-image.v1";

pub fn parse(bytes: &[u8], rank: u32) -> Result<RankIndex, String> {
    runtime_bundle::rank_index::parse(bytes, rank, SCHEMA)
}

/// Checks the complete target geometry at `tp` before resolving or uploading
/// any views. A pinned topology's views are checked against the pinned
/// digest of its contracts; other indexes against the derived contracts.
pub fn validate_text(index: &RankIndex, tp: Tp) -> Result<(), String> {
    use runtime_bundle::rank_index::{validate_contract, validate_pinned};
    if index.rank >= tp.ranks() {
        return Err(format!("text rank outside TP{}", tp.ranks()));
    }
    crate::text_contract_pin(tp)
        .and_then(|pin| validate_pinned(&index.views, index.image_bytes, pin))
        .unwrap_or_else(|| {
            validate_contract(&index.views, index.image_bytes, &crate::text_views(tp))
        })
}

#[cfg(test)]
mod admission_tests {
    #[test]
    fn refuses_overlapping_and_untyped_text_views() {
        let bytes = serde_json::to_vec(&serde_json::json!({
            "schema": super::SCHEMA, "rank": 0, "image": "rank0.img",
            "image_bytes": 8192, "sha256": "a".repeat(64),
            "views": [
                {"name": "model.norm.weight", "offset": 0, "bytes": 8192,
                 "dtype": "WRONG", "shape": [1], "layout_id": "stale_layout_v0"},
                {"name": "model.layers.0.input_layernorm.weight", "offset": 0,
                 "bytes": 8192, "dtype": "BF16", "shape": [4096],
                 "layout_id": "bf16_row_major_v1"}
            ]
        }))
        .unwrap();
        assert!(super::parse(&bytes, 0).is_err());
    }

    /// A pinned topology's index of exactly the derived contracts is
    /// admitted through the pinned digest; one changed field of one view,
    /// a renamed view or an extra view is refused.
    #[test]
    fn pinned_text_indexes_refuse_any_changed_view_field() {
        use runtime_bundle::rank_index::{View, validate_pinned};
        for tp in super::Tp::SUPPORTED {
            let pin = crate::text_contract_pin(tp).expect("the served geometry is pinned");
            let mut offset = 0;
            let views: std::collections::BTreeMap<String, View> = crate::text_views(tp)
                .into_iter()
                .map(|(name, contract)| {
                    let view = View {
                        offset,
                        bytes: contract.bytes,
                        dtype: contract.dtype.into(),
                        shape: contract.shape,
                        layout_id: contract.layout_id.into(),
                    };
                    offset += view.bytes.next_multiple_of(4096);
                    (name, view)
                })
                .collect();
            let index = super::RankIndex {
                rank: 0,
                image: "rank0.img".into(),
                image_bytes: offset,
                sha256: Some("a".repeat(64)),
                direct_identity: None,
                views,
            };
            assert_eq!(
                validate_pinned(&index.views, index.image_bytes, pin),
                Some(Ok(()))
            );
            super::validate_text(&index, tp).unwrap();
            let name = "model.layers.1.self_attn.qkv_proj.weight";
            let edits: [fn(&mut View); 4] = [
                |v| v.dtype = "BF16".into(),
                |v| v.shape[0] += 16,
                |v| v.bytes -= 1,
                |v| v.layout_id = "bf16_row_major_v1".into(),
            ];
            for (number, edit) in edits.iter().enumerate() {
                let mut changed = index.clone();
                edit(changed.views.get_mut(name).unwrap());
                assert!(
                    super::validate_text(&changed, tp).is_err(),
                    "edit {number} at TP{}",
                    tp.ranks()
                );
            }
            let mut renamed = index.clone();
            let view = renamed.views.remove(name).unwrap();
            renamed.views.insert(format!("{name}2"), view);
            assert!(super::validate_text(&renamed, tp).is_err());
            let mut extra = index.clone();
            let mut view = extra.views["model.norm.weight"].clone();
            view.offset = offset;
            extra.image_bytes += 4096;
            extra.views.insert("model.norm.bias".into(), view);
            assert!(super::validate_text(&extra, tp).is_err());
        }
    }
}
