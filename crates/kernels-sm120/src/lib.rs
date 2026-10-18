//! SM12x CUDA engines, launch planners, instantiation catalogs and tuning
//! tables. See docs/kernel-library.md.

pub mod arithmetic;
pub mod attention;
pub mod backend;
pub mod catalogs;
pub mod comm;
pub mod context_parallel;
pub mod device;
pub mod encoder;
pub mod f32_projection;
pub mod gated_pooling;
pub mod gemm;
pub mod invariance;
pub mod latent;
pub mod linear;
pub mod moe;
pub mod names;
pub mod owned_writes;
pub mod paged_partials;
pub mod pooled_indexer;
pub mod precision;
pub mod probes;
pub mod row;
pub mod sampling;
pub mod selected_partials;
pub mod sinkhorn_hc;

pub use catalogs::{ByProfile, Catalogs, ModelCatalog, ProjectionInventory};
pub use kernel_core::rank_shape::{ModelConfig, RankShape};

/// Source inputs of this pinned SDK revision, for bundle-owned native builds.
/// Keep this tree inside the consuming spec's source root when mixing headers.
pub fn cuda_source_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda")
}

/// A profile's name and its short tag in cubin and symbol names.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ProfileNames {
    pub name: &'static str,
    pub tag: &'static str,
}

/// A served model whose kernels the catalogs specialize to its shapes,
/// declared as data by the model: its profile names at each rank count it
/// compiles ([`profile_names!`]) and its configuration values. Each model
/// declares its own data in its contract crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Model {
    /// Profile names by rank count. The first profile also holds the
    /// model's kernels whose shapes do not depend on the rank count.
    names: &'static [(u32, ProfileNames)],
    config: ModelConfig,
}

impl Model {
    /// A model of `names` (at least one rank count, each once) and
    /// `config`.
    pub const fn new(names: &'static [(u32, ProfileNames)], config: ModelConfig) -> Model {
        assert!(!names.is_empty(), "a model declares a profile");
        let mut i = 0;
        while i < names.len() {
            let mut j = i + 1;
            while j < names.len() {
                assert!(
                    names[i].0 != names[j].0,
                    "a model declares a rank count once"
                );
                j += 1;
            }
            i += 1;
        }
        Model { names, config }
    }

    /// The configuration values the per-rank shapes split.
    pub const fn config(self) -> ModelConfig {
        self.config
    }

    /// The model's profile at `ranks`, `None` for a rank count it does not
    /// compile.
    pub const fn try_at(self, ranks: u32) -> Option<Profile> {
        let mut i = 0;
        while i < self.names.len() {
            if self.names[i].0 == ranks {
                return Some(Profile {
                    names: self.names[i].1,
                    model: Some((self, ranks)),
                });
            }
            i += 1;
        }
        None
    }

    /// The model's profile at `ranks`, which it must compile.
    pub const fn at(self, ranks: u32) -> Profile {
        match self.try_at(ranks) {
            Some(profile) => profile,
            None => panic!("the model compiles no profile at this rank count"),
        }
    }

    /// The model's first profile, which also holds its rank-invariant
    /// kernels.
    pub const fn base(self) -> Profile {
        self.at(self.names[0].0)
    }

    /// The model's profiles in declaration order, the base first.
    pub fn profiles(self) -> Vec<Profile> {
        self.names
            .iter()
            .map(|&(ranks, _)| self.at(ranks))
            .collect()
    }

    /// The model's profiles that hold only per-rank kernels: all but the
    /// base.
    pub fn other_profiles(self) -> Vec<Profile> {
        self.profiles().split_off(1)
    }
}

/// A model's [`ProfileNames`] by rank count from its base name and tag:
/// the first rank count keeps them, every other appends `-tp{ranks}` and
/// `_tp{ranks}`. `profile_names!("m", "m", 4, 2)` declares `m` at TP4 and
/// `m-tp2` (tag `m_tp2`) at TP2.
#[macro_export]
macro_rules! profile_names {
    ($name:literal, $tag:literal, $base:literal $(, $ranks:literal)* $(,)?) => {
        &[
            (
                $base,
                $crate::ProfileNames {
                    name: $name,
                    tag: $tag,
                },
            ),
            $((
                $ranks,
                $crate::ProfileNames {
                    name: concat!($name, "-tp", $ranks),
                    tag: concat!($tag, "_tp", $ranks),
                },
            ),)*
        ]
    };
}

/// A model profile whose instantiations a catalog lists, declared as data:
/// a model's kernels at one rank count ([`Model::at`]), or a synthetic shape
/// set ([`Profile::synthetic`]).
///
/// Catalog entries declare their profile. Model cubin catalogs filter by
/// that tag; kernel-lab builds all profiles. The shared transport cubin
/// retains its common entry set for every consuming bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Profile {
    names: ProfileNames,
    /// The model and rank count of a model's profile, whose per-rank shapes
    /// ([`Profile::rank_shape`]) its kernels have. The model's base profile
    /// also holds its kernels whose shapes do not depend on the rank count.
    model: Option<(Model, u32)>,
}

#[allow(non_upper_case_globals)]
impl Profile {
    /// Synthetic standard dense GQA and top-k MoE shapes on TP4.
    pub const StdDenseMoe: Profile = Profile::synthetic(ProfileNames {
        name: "std-dense-moe",
        tag: "std",
    });

    /// A profile of synthetic shapes outside any model.
    pub const fn synthetic(names: ProfileNames) -> Profile {
        Profile { names, model: None }
    }

    /// The model, for a served model's profile.
    pub const fn model(self) -> Option<Model> {
        match self.model {
            Some((model, _)) => Some(model),
            None => None,
        }
    }

    /// The rank count of a served model's per-rank shapes.
    pub const fn ranks(self) -> Option<u32> {
        match self.model {
            Some((_, ranks)) => Some(ranks),
            None => None,
        }
    }

    /// Whether this is a served model's base profile ([`Model::base`]),
    /// which also holds the model's rank-invariant kernels.
    pub const fn is_base(self) -> bool {
        match self.model {
            Some((model, ranks)) => model.names[0].0 == ranks,
            None => false,
        }
    }

    /// The per-rank shapes of a served model's profile, derived from its
    /// configuration and rank count.
    pub const fn rank_shape(self) -> Option<RankShape> {
        match self.model {
            Some((model, ranks)) => Some(RankShape::unchecked(model.config, ranks)),
            None => None,
        }
    }

    /// The rank count of `model`'s profile, `None` for another model's.
    pub const fn ranks_of(self, model: Model) -> Option<u32> {
        match self.model {
            Some((owner, ranks)) if same_names(owner.names[0].1, model.names[0].1) => Some(ranks),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        self.names.name
    }

    /// The short name in cubin and symbol names.
    pub fn tag(self) -> &'static str {
        self.names.tag
    }
}

/// Whether two models' base names are the same (a `const` string compare):
/// models are told apart by their profile names.
const fn same_names(a: ProfileNames, b: ProfileNames) -> bool {
    let (a, b) = (a.name.as_bytes(), b.name.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: ModelConfig = ModelConfig {
        hidden: 1024,
        vocab: 2048,
        blocks: &[],
        linear: None,
        ngram_heads: 0,
    };

    /// A model's rank counts are data: the first is its base, and it has
    /// no profile at a rank count it does not declare.
    #[test]
    fn rank_counts_are_declared_data() {
        const MODEL: Model = Model::new(profile_names!("other-flash", "other", 4, 2, 8), CONFIG);
        assert_eq!(
            MODEL
                .profiles()
                .iter()
                .map(|p| p.ranks())
                .collect::<Vec<_>>(),
            [Some(4), Some(2), Some(8)]
        );
        assert_eq!(MODEL.base(), MODEL.at(4));
        assert!(MODEL.at(4).is_base() && !MODEL.at(2).is_base());
        assert_eq!(MODEL.other_profiles(), [MODEL.at(2), MODEL.at(8)]);
        assert_eq!(
            (MODEL.at(8).as_str(), MODEL.at(8).tag()),
            ("other-flash-tp8", "other_tp8")
        );
        assert_eq!(MODEL.at(8).rank_shape().map(|s| s.ranks()), Some(8));
        assert_eq!(MODEL.try_at(16), None);
        assert!(!Profile::StdDenseMoe.is_base());
        assert_eq!(Profile::StdDenseMoe.ranks(), None);
    }

    /// A model outside the kernel base declares its profiles as data.
    #[test]
    fn a_model_declares_its_profiles_as_data() {
        const OTHER: Model = Model::new(profile_names!("other-flash", "other", 4, 2), CONFIG);
        let profile = OTHER.at(2);
        assert_eq!(
            (profile.as_str(), profile.tag()),
            ("other-flash-tp2", "other_tp2")
        );
        assert_eq!(profile.ranks_of(OTHER), Some(2));
        assert_eq!(
            profile.ranks_of(Model::new(
                profile_names!("another", "another", 4, 2),
                OTHER.config()
            )),
            None
        );
        assert!(!Catalogs::base().profiles().contains(&profile));
    }
}
