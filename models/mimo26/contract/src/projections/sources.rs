//! Checkpoint names materialized from each listed geometry's declaration.
pub use projection_spec::bind::SourceOwner as Source;
use projection_spec::{Declaration, Error, Geometry};
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub struct Sources {
    pub declaration: &'static Declaration,
    pub geometry: &'static Geometry,
    pub names: BTreeMap<String, Source>,
}

/// The sources of `geometry`, built once per process. Names do not depend
/// on the rank count.
pub fn sources(geometry: &'static crate::geometry::Geometry) -> Result<&'static Sources, Error> {
    type Built = (&'static str, Result<Sources, Error>);
    static SOURCES: OnceLock<Vec<Built>> = OnceLock::new();
    SOURCES
        .get_or_init(|| {
            crate::geometry::GEOMETRIES
                .iter()
                .map(|g| {
                    let built = crate::Tp::of_geometry(g, 4)
                        .map_err(Error)
                        .and_then(super::expanded)
                        .and_then(|(declaration, geometry)| {
                            Ok(Sources {
                                declaration,
                                geometry,
                                names: declaration.source_names(geometry)?,
                            })
                        });
                    (g.id, built)
                })
                .collect()
        })
        .iter()
        .find(|(id, _)| *id == geometry.id)
        .ok_or_else(|| Error("unlisted projection geometry".into()))?
        .1
        .as_ref()
        .map_err(Clone::clone)
}

impl Sources {
    /// Resolve companions and fused resident names to their declared source role.
    pub fn source(&self, name: &str) -> Option<Source> {
        if let Some(source) = self.names.get(name) {
            return Some(*source);
        }
        let mut weight = name.to_owned();
        for suffix in ["weight_scale_inv", "weight_scale"] {
            if let Some(stem) = name.strip_suffix(suffix) {
                weight = format!("{stem}weight");
                break;
            }
        }
        weight = weight.replace(".gate_up_proj.", ".gate_proj.");
        if let Some(source) = self.names.get(&weight) {
            return Some(*source);
        }
        // Bank and assembly views reuse declared checkpoint matrices.
        if let Some((stem, tail)) = weight.split_once(".experts.") {
            return self.names.get(&format!("{stem}.experts.0.{tail}")).copied();
        }
        let id = if weight == "context_kv_proj.weight" {
            "dflash.context-kv"
        } else if weight.starts_with("layers.") && weight.ends_with(".self_attn.qkv_proj.weight") {
            "dflash.qkv"
        } else {
            return None;
        };
        self.declaration
            .projections
            .iter()
            .position(|p| p.id == id)
            .map(Source::Projection)
    }
}
