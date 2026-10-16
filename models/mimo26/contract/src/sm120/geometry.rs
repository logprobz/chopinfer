//! Geometry-owned kernel profiles, dimensions and fixture admission.

use crate::geometry::{GEOMETRIES, Geometry, V26_FLASH};
use kernel_core::rank_shape::{
    AttentionConfig, Block, BlockClass, DenseConfig, ExpertSplit, ModelConfig, MoeConfig,
};
use kernels_sm120::{Model, Profile, ProfileNames, RankShape};

#[derive(Clone, Copy, Debug)]
pub struct GeometryModel {
    pub geometry: Geometry,
    pub model: Model,
}

pub const fn moe(g: Geometry) -> MoeConfig {
    MoeConfig {
        experts: g.experts(),
        top_k: g.top_k(),
        intermediate: g.expert_intermediate(),
        split: ExpertSplit::Intermediate,
        shared_intermediate: 0,
        shared_tile: 0,
    }
}
pub const fn dense(g: Geometry) -> DenseConfig {
    DenseConfig {
        intermediate: g.dense_intermediate(),
    }
}
pub const fn ga(g: Geometry) -> AttentionConfig {
    AttentionConfig {
        q_heads: g.attention_q_heads(),
        kv_heads: g.attention_kv_heads(),
        qk_head: g.head_dim(),
        v_head: g.value_head_dim(),
    }
}
pub const fn swa(g: Geometry) -> AttentionConfig {
    AttentionConfig {
        q_heads: g.swa_q_heads(),
        kv_heads: g.swa_kv_heads(),
        qk_head: g.swa_head_dim(),
        v_head: g.swa_value_head_dim(),
    }
}
pub const fn dflash(g: Geometry) -> AttentionConfig {
    AttentionConfig {
        q_heads: g.dflash().num_attention_heads,
        kv_heads: g.dflash().num_key_value_heads,
        qk_head: g.dflash().head_dim,
        v_head: g.dflash().v_head_dim,
    }
}
pub const fn pinned_names(g: Geometry) -> bool {
    g.pinned_names
}

#[cfg(test)]
macro_rules! geometry_model {
    ($id:literal, $tag:literal, $g:expr) => {{
        const G: Geometry = $g;
        GeometryModel {
            geometry: G,
            model: Model::new(
                &[
                    (
                        4,
                        ProfileNames {
                            name: concat!("mimo-", $id),
                            tag: $tag,
                        },
                    ),
                    (
                        2,
                        ProfileNames {
                            name: concat!("mimo-", $id, "-tp2"),
                            tag: concat!($tag, "_tp2"),
                        },
                    ),
                ],
                ModelConfig {
                    hidden: G.hidden(),
                    vocab: G.vocab(),
                    blocks: &[
                        BlockClass {
                            name: "dense",
                            layers: (0, G.dense_layer_end()),
                            block: Block::Dense(dense(G)),
                        },
                        BlockClass {
                            name: "moe",
                            layers: (G.dense_layer_end(), G.layers()),
                            block: Block::Moe(moe(G)),
                        },
                    ],
                    linear: None,
                    ngram_heads: 0,
                },
            ),
        }
    }};
}
// Profile strings and block slices have static storage, derived from the same
// generated list as the compiler's geometry documents.
const NAME_CAPACITY: usize = {
    let mut max = 0;
    let mut i = 0;
    while i < GEOMETRIES.len() {
        let g = GEOMETRIES[i];
        if g.id.len() > max {
            max = g.id.len();
        }
        if g.kernel_tag.len() > max {
            max = g.kernel_tag.len();
        }
        i += 1;
    }
    max + 16
};
const NAMES: [[kernels_sm120::names::Text<NAME_CAPACITY>; 4]; GEOMETRIES.len()] = {
    let mut names = [[kernels_sm120::names::Text::new(&[]); 4]; GEOMETRIES.len()];
    let mut i = 0;
    while i < GEOMETRIES.len() {
        let g = GEOMETRIES[i];
        names[i] = [
            kernels_sm120::names::Text::new(&["mimo-", g.id, "-tp2"]),
            kernels_sm120::names::Text::new(&[g.kernel_tag, "_tp2"]),
            kernels_sm120::names::Text::new(&["mimo-", g.id]),
            kernels_sm120::names::Text::new(&[g.kernel_tag]),
        ];
        i += 1;
    }
    names
};
/// Each geometry's profile names at TP4, its base, and TP2.
const PROFILE_NAMES: &[[(u32, ProfileNames); 2]; GEOMETRIES.len()] = &{
    let mut names = [[(0, ProfileNames { name: "", tag: "" }); 2]; GEOMETRIES.len()];
    let mut i = 0;
    while i < GEOMETRIES.len() {
        names[i] = [
            (
                4,
                ProfileNames {
                    name: NAMES[i][2].as_str(),
                    tag: NAMES[i][3].as_str(),
                },
            ),
            (
                2,
                ProfileNames {
                    name: NAMES[i][0].as_str(),
                    tag: NAMES[i][1].as_str(),
                },
            ),
        ];
        i += 1;
    }
    names
};
const BLOCKS: [[BlockClass; 2]; GEOMETRIES.len()] = {
    let mut blocks = [[
        BlockClass {
            name: "dense",
            layers: (0, 0),
            block: Block::Dense(dense(V26_FLASH)),
        },
        BlockClass {
            name: "moe",
            layers: (0, 0),
            block: Block::Moe(moe(V26_FLASH)),
        },
    ]; GEOMETRIES.len()];
    let mut i = 0;
    while i < GEOMETRIES.len() {
        let g = GEOMETRIES[i];
        blocks[i] = [
            BlockClass {
                name: "dense",
                layers: (0, g.dense_layer_end()),
                block: Block::Dense(dense(g)),
            },
            BlockClass {
                name: "moe",
                layers: (g.dense_layer_end(), g.layers()),
                block: Block::Moe(moe(g)),
            },
        ];
        i += 1;
    }
    blocks
};
pub const MODELS: &[GeometryModel] = &{
    let mut models = [GeometryModel {
        geometry: V26_FLASH,
        model: super::config::MODEL,
    }; GEOMETRIES.len()];
    let mut i = 0;
    while i < GEOMETRIES.len() {
        let g = GEOMETRIES[i];
        models[i] = GeometryModel {
            geometry: g,
            model: Model::new(
                &PROFILE_NAMES[i],
                ModelConfig {
                    hidden: g.hidden(),
                    vocab: g.vocab(),
                    blocks: &BLOCKS[i],
                    linear: None,
                    ngram_heads: 0,
                },
            ),
        };
        i += 1;
    }
    models
};

pub const PROFILES: [Profile; MODELS.len() * 2] = {
    let mut profiles = [super::config::PROFILE; MODELS.len() * 2];
    let mut i = 0;
    while i < MODELS.len() {
        profiles[i * 2] = MODELS[i].model.at(4);
        profiles[i * 2 + 1] = MODELS[i].model.at(2);
        i += 1;
    }
    profiles
};

impl GeometryModel {
    pub fn profile(self, ranks: u32) -> Profile {
        match self.model.try_at(ranks) {
            Some(profile) => profile,
            None => panic!("MiMo serves TP2 and TP4"),
        }
    }
    pub fn rank(self, ranks: u32) -> RankShape {
        self.profile(ranks).rank_shape().expect("MiMo rank shape")
    }
    pub fn profiles(self) -> [Profile; 2] {
        [self.profile(4), self.profile(2)]
    }
}

pub fn owner(profile: Profile) -> Option<&'static GeometryModel> {
    MODELS
        .iter()
        .find(|owner| profile.ranks_of(owner.model).is_some())
}

pub fn require_base_geometries(consumer: &str) -> Result<(), String> {
    let extra: Vec<_> = GEOMETRIES
        .iter()
        .filter(|g| !g.base)
        .map(|g| g.id)
        .collect();
    if extra.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{consumer} refuses fixture geometries: {} (unset CHOPINFER_MIMO26_GEOMETRY_DIRS)",
            extra.join(", ")
        ))
    }
}

pub fn validate_names() -> Result<(), String> {
    for (i, g) in GEOMETRIES.iter().enumerate() {
        for other in &GEOMETRIES[i + 1..] {
            for (a, b) in [(g, other), (other, g)] {
                if let Some(suffix) = a.kernel_tag.strip_prefix(b.kernel_tag)
                    && [
                        "_tp", "_row", "_ring", "_dflash", "_retain", "_verify", "_engine",
                        "_bundle",
                    ]
                    .iter()
                    .any(|reserved| suffix.starts_with(reserved))
                {
                    return Err(format!(
                        "geometry {} tag {} collides with geometry {} derived names",
                        a.id, a.kernel_tag, b.id
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Compiler admission uses the same complete dense queries as catalog generation.
pub fn validate_catalogs() -> Result<(), String> {
    validate_names()?;
    for g in crate::geometry::GEOMETRIES {
        crate::encoders::payload_limits(g).map_err(|e| e.to_string())?;
    }
    for owner in MODELS {
        for ranks in [2, 4] {
            super::gemm::try_shapes_for(*owner, ranks)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const SMALL: Geometry = {
        let mut g = V26_FLASH;
        g.id = "test-small";
        g.base = false;
        g.encoders = false;
        g.pinned_names = false;
        g.kernel_tag = "fp8";
        g.shape.global.heads = 32;
        g.shape.sliding.heads = 32;
        g.shape.sliding.kv_heads = 4;
        g.shape.text.vocab = 4096;
        g.shape.text.dense_intermediate = 8192;
        g.shape.dflash.num_attention_heads = 32;
        g.shape.dflash.num_key_value_heads = 4;
        g.shape.dflash.intermediate_size = 8192;
        g.shape.dflash.target_layer_ids = &[0, 7];
        g.shape.audio.input_local_dim = g.shape.vision.hidden_size;
        g
    };
    const OWNER: GeometryModel = geometry_model!("test-small", "fp8", SMALL);

    #[test]
    fn unlisted_geometry_builds_every_catalog_without_legacy_aliases() {
        assert_eq!(MODELS[0].model, super::super::config::MODEL);
        let dense = super::super::gemm::catalog_for(OWNER);
        for profile in OWNER.profiles() {
            let tag = format!("chop_gemm_{}_", profile.tag());
            let selected: Vec<_> = dense
                .iter()
                .filter(|entry| entry.profile == profile)
                .collect();
            assert!(!selected.is_empty());
            assert!(selected.iter().all(|entry| entry.symbol.starts_with(&tag)));
            let ranks = profile.ranks().unwrap();
            for entry in selected
                .iter()
                .filter(|entry| entry.symbol.starts_with(&format!("{tag}qkv_")))
            {
                let kernel_core::contract::StaticQuery::Gemm(query) = entry.key.query else {
                    panic!("QKV is a GEMM")
                };
                assert_eq!(
                    query.weight,
                    crate::capabilities::Projection::Qkv {
                        global: entry.symbol.contains("qkv_ga")
                    }
                    .weight(if ranks == 4 {
                        crate::tp::Tp::TP4
                    } else {
                        crate::tp::Tp::TP2
                    })
                );
            }
            let shapes = super::super::gemm::shapes_for(OWNER, ranks);
            let k = |role| shapes.iter().find(|p| p.0 == role).unwrap().2;
            assert_ne!(k("o"), k("mtp_eh"));
        }
        assert_eq!(
            super::super::gemm::role_name(V26_FLASH, super::super::gemm::Role::MtpEh),
            "o"
        );
        assert_eq!(
            super::super::gemm::role_name(SMALL, super::super::gemm::Role::MtpEh),
            "mtp_eh"
        );
        let rows = super::super::row::catalog_for(OWNER);
        assert!(rows.iter().all(|row| row.symbol.contains(SMALL.kernel_tag)));
        let legacy: std::collections::BTreeSet<_> = super::super::row::catalog_for(MODELS[0])
            .into_iter()
            .map(|row| row.symbol)
            .collect();
        assert!(
            rows.iter().all(|row| !legacy.contains(&row.symbol)),
            "geometry rows reuse a legacy symbol"
        );
        for variant in super::super::attention::catalog_for(OWNER) {
            variant.validate().unwrap();
        }
        let variants = super::super::moe::catalog_for(OWNER).unwrap();
        assert!(!variants.is_empty());
        let table = super::super::tuning::schedules(&variants).unwrap();
        kernels_sm120::moe::template::scheduled(variants, &table).unwrap();
        let routes = super::super::moe::route_cubins_for(OWNER);
        assert!(routes.iter().all(|c| c.name.contains(SMALL.kernel_tag)));
    }

    #[test]
    fn dense_fallback_checks_all_buckets_and_reports_impossible_shapes() {
        use kernel_core::contract::{DenseKernel, WeightFormat};
        let mut p = (
            "probe",
            4096,
            128,
            WeightFormat::Bf16,
            false,
            8,
            DenseKernel::FragmentCluster { warp_blocks: 2 },
        );
        super::super::tuning::dense(OWNER, 4, &mut p).unwrap();
        assert_eq!(p.6, kernels_sm120::gemm::WHOLE_K_DECODE);
        p.2 = 129;
        let error = super::super::tuning::dense(OWNER, 4, &mut p).unwrap_err();
        assert!(
            error.contains("test-small TP4 role probe N=4096 K=129"),
            "{error}"
        );
    }
}
