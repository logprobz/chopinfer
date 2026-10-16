//! Norm weights an export may store in F32. HF computes `(1 + w)` in F32, so
//! an F32 weight is bound as stored rather than narrowed to BF16. Admission
//! names the weights whose consumers have an F32 form; the rank index then
//! records each view's stored dtype, and lowering binds each consumer by the
//! dtype of the view it reads ([`NormWeights`]).

use crate::views::{BF16_ROWS, F32_ROWS, RankIndex, Required};
use kernel_core::contract::DType;
use std::collections::BTreeSet;

/// Whether the norm weight `name` has consumers with an F32 form: every
/// hyper-connection norm (the per-stream norm rows and the all-reduce
/// epilogue that applies the next sublayer's), and PLE's key, convolution
/// and query norms. The MTP pre-FC norms run in a fused drafter kernel
/// without an F32 form and stay BF16.
pub fn f32_consumer(name: &str) -> bool {
    if name.ends_with(".hc_norm.weight") {
        return true;
    }
    name.strip_suffix(".weight")
        .and_then(|stem| stem.rsplit_once(".ple."))
        .is_some_and(|(_, norm)| matches!(norm, "norm_key" | "norm_conv" | "norm_query"))
}

/// The F32 row form [`admit`] accepts for `view`, a required BF16 norm with
/// an F32 consumer: its dtype, layout and the bytes of its elements at four
/// bytes each.
pub fn f32_form(view: &Required) -> Option<Required> {
    (view.dtype == "BF16" && view.layout_id == BF16_ROWS && f32_consumer(&view.name)).then(|| {
        Required {
            name: view.name.clone(),
            dtype: "F32",
            layout_id: F32_ROWS,
            bytes: view.bytes * 2,
        }
    })
}

/// Accept the F32 row view the index stores for each required BF16 norm
/// with an F32 consumer ([`f32_form`]). Other requirements, and views the
/// index stores in any other form, are left for [`crate::views::check`] to
/// judge.
pub fn admit(index: &RankIndex, required: &mut [Required]) {
    for view in required.iter_mut() {
        if let Some(wide) = f32_form(view)
            && index.views.get(&view.name).is_some_and(|stored| {
                stored.dtype == wide.dtype && stored.layout_id == wide.layout_id
            })
        {
            *view = wide;
        }
    }
}

/// Check the MTP drafter's views of a prepared rank index, admitting its
/// stored F32 hyper-connection norms as [`admit`] does the text program's.
/// Its pre-FC norms have no F32 consumer and stay BF16.
pub fn validate_drafter(index: &RankIndex, shape: crate::contract::Shape) -> Result<(), String> {
    let mut required = crate::views::drafter_required(shape);
    admit(index, &mut required);
    crate::views::check(index, [shape.mtp_layer()], required, shape)
}

/// The norm weights a rank image stores in F32. Each consumer binds its F32
/// form for these and keeps its BF16 symbol for every other weight.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NormWeights(BTreeSet<String>);

impl NormWeights {
    /// The F32 norm views of an index that [`admit`] accepted.
    pub fn from_index(index: &RankIndex) -> Self {
        Self(
            index
                .views
                .iter()
                .filter(|(name, view)| {
                    f32_consumer(name) && view.dtype == "F32" && view.layout_id == F32_ROWS
                })
                .map(|(name, _)| name.clone())
                .collect(),
        )
    }

    /// The stored dtype of norm weight `name`.
    pub fn dtype(&self, name: &str) -> DType {
        if self.0.contains(name) {
            DType::F32
        } else {
            DType::Bf16
        }
    }

    /// Whether every norm weight is BF16.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::Shape;
    use crate::views::{MODEL, View};

    #[test]
    fn hyper_connection_and_ple_norms_have_f32_consumers() {
        for name in [
            format!("{MODEL}.layers.0.attn_hyper_connection.hc_norm.weight"),
            format!("{MODEL}.layers.47.mlp_hyper_connection.hc_norm.weight"),
            format!("{MODEL}.hyper_connection_mixer.hc_norm.weight"),
            "mtp.layers.0.attn_hyper_connection.hc_norm.weight".into(),
            "mtp.hyper_connection_mixer.hc_norm.weight".into(),
            format!("{MODEL}.layers.1.ple.norm_key.weight"),
            format!("{MODEL}.layers.1.ple.norm_conv.weight"),
            format!("{MODEL}.layers.1.ple.norm_query.weight"),
        ] {
            assert!(f32_consumer(&name), "{name}");
        }
        for name in [
            "mtp.pre_fc_norm_hidden.weight".to_owned(),
            "mtp.pre_fc_norm_embedding.weight".into(),
            format!("{MODEL}.layers.0.linear_attn.norm.weight"),
            format!("{MODEL}.layers.3.self_attn.q_norm.weight"),
            format!("{MODEL}.layers.1.ple.norm_key.bias"),
            format!("{MODEL}.layers.1.ple.conv1d.weight"),
        ] {
            assert!(!f32_consumer(&name), "{name}");
        }
    }

    fn index(required: &[Required], f32: &[&str]) -> RankIndex {
        let mut offset = 0;
        let views = required
            .iter()
            .map(|view| {
                let wide = f32.contains(&view.name.as_str());
                let bytes = if wide { 2 * view.bytes } else { view.bytes };
                let entry = (
                    view.name.clone(),
                    View {
                        offset,
                        bytes,
                        dtype: if wide { "F32" } else { view.dtype }.into(),
                        shape: vec![bytes],
                        layout_id: if wide { F32_ROWS } else { view.layout_id }.into(),
                    },
                );
                offset += bytes.next_multiple_of(4096);
                entry
            })
            .collect();
        RankIndex {
            rank: 0,
            image: "rank0.img".into(),
            image_bytes: offset,
            sha256: Some("0".repeat(64)),
            direct_identity: None,
            views,
        }
    }

    /// An index with F32 norm views validates only once they are admitted,
    /// and the selection survives the index's encoding.
    #[test]
    fn admitted_f32_norm_views_validate_and_round_trip() {
        let shape = Shape::TP4;
        let required = crate::views::required(0, shape);
        let hc = format!("{MODEL}.layers.5.mlp_hyper_connection.hc_norm.weight");
        let query = format!("{MODEL}.layers.1.ple.norm_query.weight");
        let stored = index(&required, &[&hc, &query]);
        let error = crate::views::check(&stored, [], required.clone(), shape).unwrap_err();
        assert!(error.contains(&hc), "{error}");
        let mut admitted = required.clone();
        admit(&stored, &mut admitted);
        crate::views::check(&stored, [], admitted, shape).unwrap();
        let bytes = serde_json::to_vec(&serde_json::json!({
            "schema": crate::views::SCHEMA,
            "rank": 0,
            "image": stored.image,
            "image_bytes": stored.image_bytes,
            "sha256": stored.sha256,
            "views": stored.views.iter().map(|(name, view)| serde_json::json!({
                "name": name, "offset": view.offset, "bytes": view.bytes,
                "dtype": view.dtype, "shape": view.shape, "layout_id": view.layout_id,
            })).collect::<Vec<_>>(),
        }))
        .unwrap();
        let parsed = crate::views::parse(&bytes, 0).unwrap();
        let norms = NormWeights::from_index(&parsed);
        assert_eq!(norms.dtype(&hc), DType::F32);
        assert_eq!(norms.dtype(&query), DType::F32);
        assert_eq!(
            norms.dtype(&format!(
                "{MODEL}.layers.5.attn_hyper_connection.hc_norm.weight"
            )),
            DType::Bf16
        );
        assert!(NormWeights::from_index(&index(&required, &[])).is_empty());
    }

    /// Views without an F32 consumer keep their BF16 requirement.
    #[test]
    fn other_f32_views_are_still_refused() {
        let shape = Shape::TP4;
        let required = crate::views::required(0, shape);
        let norm = format!("{MODEL}.layers.0.linear_attn.norm.weight");
        let stored = index(&required, &[&norm]);
        let mut admitted = required.clone();
        admit(&stored, &mut admitted);
        assert_eq!(admitted, required);
        let error = crate::views::check(&stored, [], admitted, shape).unwrap_err();
        assert!(error.contains(&norm), "{error}");
        assert!(NormWeights::from_index(&stored).is_empty());
    }
}
