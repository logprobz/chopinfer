//! The catalogs a consumer reads: the kernel base's own entries and the
//! catalogs of the models the consumer chooses.
//!
//! A model's crate declares its entries on the shared engines as a `static`
//! [`ModelCatalog`]. A consumer builds one immutable [`Catalogs`] value from
//! the models it serves or tests and passes it to every reader: cubin
//! generation, invariance declarations, medium-M selection, template
//! instances and lab selectors. Readers list the kernel base's entries, then
//! each model's in the order given. Construction refuses two models of one
//! name, an entry symbol two catalogs declare, two schedule rows of one key,
//! and any collision in the collection's complete namespaces: two profiles
//! of one cubin tag, two cubins of one name, an entry symbol in two cubins
//! (generated companions and route cubins included) and two different
//! declarations of one symbol. This crate depends on no model.

use crate::invariance::Declared;
use crate::{
    Profile, attention, comm, context_parallel, gemm, invariance, linear, moe, pooled_indexer, row,
    sampling,
};
use kernel_build::CubinSpec;
use kernel_build::tuning::{Key, Table};
use kernel_core::contract::Refusal;
use std::collections::{BTreeMap, BTreeSet};

/// A model's generated projection inventory: its kernel offers, the
/// operation capabilities it retains and its layout copies, each in catalog
/// order.
pub type ProjectionInventory = (
    Vec<crate::precision::KernelOffer>,
    Vec<kernel_core::precision::capability::OperationCapability>,
    Vec<row::precision::LayoutCopy>,
);

/// One model's catalog entries for the shared engines. Each list function
/// returns the entries in catalog order.
pub struct ModelCatalog {
    /// The model's name, unique among a consumer's catalogs.
    pub name: &'static str,
    /// The model's profiles, in the order its per-profile cubins list.
    /// Every non-base profile used by its contributions must be registered here.
    pub profiles: &'static [Profile],
    /// Dense GEMM entries on the shared fragment, cluster and tiled
    /// families.
    pub gemm: fn() -> Vec<gemm::GemmVariant>,
    /// Entries of the fused-weight GEMM kernels (`gemm::fused`).
    pub gemm_fused: fn() -> Vec<gemm::fused::Entry>,
    /// Bind fused GEMM headers to the model-owned CUDA source tree.
    pub gemm_fused_sources: fn(Profile, &mut CubinSpec),
    /// The fragment prefill entries (profile, symbol) that have a medium-M
    /// stand-in (`gemm::mid`).
    pub gemm_mid: fn() -> Vec<(Profile, String)>,
    /// Lab-only K splits of fragment decode entries, by symbol stem
    /// (`gemm::lab_split_catalog`).
    pub gemm_lab_splits: &'static [(&'static str, u8)],
    /// Additional geometry-derived lab split stems.
    pub gemm_lab_splits_for: fn() -> Vec<(String, u8)>,
    /// Grouped expert entries, with their template schedules resolved from
    /// the model's schedule table.
    pub moe: fn() -> Vec<moe::Variant>,
    /// Route, permutation and combine entries of the expert-parallel engine
    /// (`moe::expert_parallel::RouteEntry`).
    pub moe_routes: fn() -> Vec<moe::expert_parallel::RouteEntry>,
    /// The route cubins of a profile (every profile's when `None`), which
    /// compile their arithmetic in.
    pub moe_route_cubins: fn(Option<Profile>) -> Vec<CubinSpec>,
    /// Paged attention variants (`attention::catalog`).
    pub attention: fn() -> Vec<attention::Variant>,
    /// Gated attention rows (`attention::gated::catalog`), compiled in the
    /// profile's attention cubin.
    pub gated: fn() -> Vec<attention::gated::Entry>,
    /// QSA sparse attention and indexer entries (`attention::sparse`).
    pub sparse: fn() -> Vec<attention::sparse::Entry>,
    /// Pooled top-k indexer entries (`pooled_indexer`), compiled per
    /// profile.
    pub pooled_indexer: fn() -> Vec<pooled_indexer::Entry>,
    /// Row entries of the shared row engine (`row::catalog`), compiled in
    /// `row_engine`.
    pub rows: fn() -> Vec<row::RowInstantiation>,
    /// Linear-attention entries, compiled per profile.
    pub linear: fn() -> Vec<linear::Entry>,
    /// The linear-attention cubin identity for a registered profile.
    pub linear_cubin_name: fn(Profile) -> String,
    /// Hyper-connection, n-gram and PLE rows (`row::hyper_connection`),
    /// compiled per profile.
    pub hyper_connection_rows: fn() -> Vec<row::hyper_connection::Entry>,
    /// Bind hyper-connection row headers to the model-owned CUDA source tree.
    pub hyper_connection_sources: fn(Profile, &mut CubinSpec),
    /// The model's declared row shapes of the shared transport at a rank
    /// count (`comm::catalog`).
    pub comm_shapes: fn(u32) -> Vec<comm::Entry>,
    /// The model's own comm cubins: its transports' instances
    /// (`comm::hyper_connection`, `comm::verify_cubin_spec`).
    pub comm_cubins: fn() -> Vec<CubinSpec>,
    /// Declarations of the shared kernels the model launches outside its
    /// catalog entries (`invariance::declarations`), its comm entries among
    /// them.
    pub declared: fn() -> Vec<Declared>,
    /// The model's grouped-template schedule table (`moe::template`), in the
    /// canonical tuning table text.
    pub schedules: &'static str,
    /// Optional resolved table for geometry-owned measured and inherited rows.
    pub schedule_table: Option<fn() -> Table>,
    /// Geometry-keyed measured tables, used when merging sweep results.
    pub geometry_schedules: Option<fn() -> std::collections::BTreeMap<String, Table>>,
    /// Every key the model's schedule table may hold: its template entries'
    /// (`moe::template::schedule_keys`) and any packed ring rows
    /// (`moe::template::packed_keys`). A row outside them is refused.
    pub schedule_keys: fn() -> Vec<Key>,
    /// The model's generated projection inventory over its geometries.
    pub projections: fn() -> Result<ProjectionInventory, String>,
    /// Row stage uses at shapes the model does not serve, such as a
    /// topology it has not admitted yet. Kernel-lab compares their row
    /// stages (the offers' and the copies') in both shape forms.
    pub row_shape_probes: fn() -> Result<ProjectionInventory, String>,
    /// The F32 norm-weight forms its bundle embeds (`comm::norm_weights`).
    pub norm_weights: fn() -> comm::norm_weights::NormWeightForms,
    /// The registered n-gram hash entries its bundle embeds, one form per
    /// rank count (`row::ngram_hash`).
    pub ngram_hashes: fn() -> Vec<row::ngram_hash::NgramHashForm>,
    /// The entries of each context-parallel group size above 1 the model
    /// serves, with their cubins (`context_parallel::GroupEntries`).
    pub context_parallel: fn() -> Vec<context_parallel::GroupEntries>,
}

impl ModelCatalog {
    /// A catalog with no entries, which a model's catalog extends
    /// (`..ModelCatalog::EMPTY`).
    pub const EMPTY: ModelCatalog = ModelCatalog {
        name: "",
        profiles: &[],
        gemm: Vec::new,
        gemm_fused: Vec::new,
        gemm_fused_sources: |_, _| {},
        gemm_mid: Vec::new,
        gemm_lab_splits: &[],
        gemm_lab_splits_for: Vec::new,
        moe: Vec::new,
        moe_routes: Vec::new,
        moe_route_cubins: |_| Vec::new(),
        attention: Vec::new,
        gated: Vec::new,
        sparse: Vec::new,
        pooled_indexer: Vec::new,
        rows: Vec::new,
        linear: Vec::new,
        linear_cubin_name: linear::default_cubin_name,
        hyper_connection_rows: Vec::new,
        hyper_connection_sources: |_, _| {},
        comm_shapes: |_| Vec::new(),
        comm_cubins: Vec::new,
        declared: Vec::new,
        schedules: include_str!("../tuning/schedules.json"),
        schedule_keys: Vec::new,
        schedule_table: None,
        geometry_schedules: None,
        projections: || Ok(ProjectionInventory::default()),
        row_shape_probes: || Ok(ProjectionInventory::default()),
        norm_weights: comm::norm_weights::NormWeightForms::default,
        ngram_hashes: Vec::new,
        context_parallel: Vec::new,
    };
}

/// A catalog family's entries grouped by profile, each group in catalog
/// order: what a topology selects from is its profile's group.
#[derive(Clone, Debug)]
pub struct ByProfile<T> {
    groups: Vec<(Profile, Vec<T>)>,
}

impl<T> ByProfile<T> {
    /// `entries` grouped by `profile`.
    pub fn new(entries: impl IntoIterator<Item = T>, profile: impl Fn(&T) -> Profile) -> Self {
        let mut groups: Vec<(Profile, Vec<T>)> = Vec::new();
        for entry in entries {
            let key = profile(&entry);
            match groups.iter_mut().find(|(group, _)| *group == key) {
                Some((_, group)) => group.push(entry),
                None => groups.push((key, vec![entry])),
            }
        }
        Self { groups }
    }

    /// The entries of `profile`, in catalog order; none where the family
    /// has none of it.
    pub fn of(&self, profile: Profile) -> &[T] {
        self.groups
            .iter()
            .find(|(group, _)| *group == profile)
            .map_or(&[], |(_, entries)| entries)
    }
}

/// An immutable set of catalogs: the kernel base's own and the given
/// models', in order.
pub struct Catalogs {
    models: Vec<&'static ModelCatalog>,
    schedules: Table,
    /// The medium-M entries, derived from the catalogs on first use.
    mid: std::sync::OnceLock<Vec<gemm::mid::Entry>>,
}

/// Refuses a row of `model`'s table `own` outside its schedule keys, a
/// template entry whose key they omit, and an entry whose resolved
/// schedule is not its row's.
fn check_schedules(model: &ModelCatalog, own: &Table) -> Result<(), Refusal> {
    let keys = (model.schedule_keys)().into_iter().collect::<BTreeSet<_>>();
    if own.rows().iter().any(|row| !keys.contains(&row.key)) {
        return Err(Refusal("a schedule row names no entry of its model"));
    }
    for variant in (model.moe)() {
        if moe::template::covers(&variant) && !keys.contains(&moe::template::key(&variant)) {
            return Err(Refusal("a model's schedule keys omit a template entry"));
        }
        let resolved = moe::template::row_knobs(&variant, own)
            .map_err(|_| Refusal("a template entry has no valid schedule row"))?;
        if variant.schedule != resolved {
            return Err(Refusal("a resolved schedule is not its model's row"));
        }
    }
    Ok(())
}

impl Catalogs {
    /// The kernel base's catalogs and `models`'. Refuses two models of one
    /// name, an entry symbol declared by two catalogs, a schedule row key
    /// two tables hold, and any collision in the collection's namespaces
    /// ([`Catalogs::cubins`]): two profiles of one tag, two cubins of one
    /// name, an entry symbol in two cubins, and two declarations of one
    /// symbol with different properties. Every contribution must use the
    /// base standard profile or a profile registered by its owning model.
    pub fn new(models: &[&'static ModelCatalog]) -> Result<Catalogs, Refusal> {
        let mut names = BTreeSet::new();
        if models.iter().any(|model| !names.insert(model.name)) {
            return Err(Refusal("two model catalogs share a name"));
        }
        let mut symbols = BTreeSet::new();
        let mut claim = |list: Vec<String>| {
            list.into_iter()
                .all(|symbol| symbols.insert(symbol))
                .then_some(())
                .ok_or(Refusal("two catalogs declare one entry symbol"))
        };
        claim(
            gemm::own_catalog()
                .into_iter()
                .map(|v| v.symbol.clone())
                .collect(),
        )?;
        claim(moe::own_catalog().into_iter().map(|v| v.symbol).collect())?;
        for model in models {
            claim(
                (model.gemm)()
                    .into_iter()
                    .map(|v| v.symbol.clone())
                    .collect(),
            )?;
            claim((model.gemm_fused)().into_iter().map(|e| e.symbol).collect())?;
            claim((model.moe)().into_iter().map(|v| v.symbol).collect())?;
            claim((model.moe_routes)().into_iter().map(|e| e.symbol).collect())?;
        }
        let mut schedules = moe::template::shared_table();
        for model in models {
            let own = match model.schedule_table {
                Some(table) => table(),
                None => moe::template::parse_table(model.schedules)?,
            };
            check_schedules(model, &own)?;
            for row in own.rows() {
                if schedules.get(&row.key).is_some() {
                    return Err(Refusal("two schedule tables hold one row key"));
                }
                schedules
                    .upsert(row.clone())
                    .map_err(|_| Refusal("invalid schedule row"))?;
            }
        }
        let catalogs = Catalogs {
            models: models.to_vec(),
            schedules,
            mid: std::sync::OnceLock::new(),
        };
        catalogs.check_namespaces()?;
        Ok(catalogs)
    }

    /// Every cubin the collection's catalogs generate, as each family's
    /// served cubins list them: the kernel base's families and the models'
    /// dense, fused, medium-M, grouped, route and context-parallel cubins.
    pub fn cubins(&self) -> Vec<CubinSpec> {
        let mut cubins = gemm::cubin_specs(self, None);
        cubins.extend(gemm::fused::cubin_specs(self, None));
        cubins.extend(gemm::mid::cubin_specs(self, None));
        cubins.extend(moe::cubin_specs(self, None));
        cubins.extend(attention::cubin_specs(self, None));
        cubins.extend(attention::sparse::cubin_specs(self, None));
        cubins.extend(pooled_indexer::cubin_specs(self, None));
        cubins.extend(row::cubin_specs(self, None));
        cubins.extend(row::hyper_connection::cubin_specs(self, None));
        cubins.push(comm::cubin_spec());
        for model in &self.models {
            cubins.extend((model.comm_cubins)());
        }
        cubins.extend(linear::cubin_specs(self, None));
        cubins.extend(sampling::cubin_specs(None));
        cubins.extend(context_parallel::cubin_specs(self, None));
        cubins
    }

    /// Refuses collisions in the complete namespaces, from the raw
    /// contributions rather than a merged inventory.
    fn check_namespaces(&self) -> Result<(), Refusal> {
        // Profiles: the kernel base's and every profile a model's entries
        // name. Cubin and symbol names carry the tag.
        let mut profiles = self.profiles();
        profiles.extend(row::own_catalog().into_iter().map(|entry| entry.profile));
        for model in &self.models {
            let first = profiles.len();
            profiles.extend((model.gemm)().into_iter().map(|v| v.profile));
            profiles.extend((model.gemm_fused)().into_iter().map(|e| e.profile));
            profiles.extend((model.gemm_mid)().into_iter().map(|(profile, _)| profile));
            profiles.extend((model.moe)().into_iter().map(|v| v.profile));
            profiles.extend((model.moe_routes)().into_iter().map(|e| e.profile));
            profiles.extend((model.attention)().into_iter().map(|v| v.profile));
            profiles.extend((model.gated)().into_iter().map(|e| e.profile));
            profiles.extend((model.sparse)().into_iter().map(|e| e.profile));
            profiles.extend((model.pooled_indexer)().into_iter().map(|e| e.profile));
            profiles.extend((model.rows)().into_iter().map(|e| e.profile));
            profiles.extend((model.linear)().into_iter().map(|e| e.profile));
            for ranks in comm::CATALOG_RANKS {
                profiles.extend((model.comm_shapes)(ranks).into_iter().map(|e| e.profile));
            }
            profiles.extend(
                (model.hyper_connection_rows)()
                    .into_iter()
                    .map(|e| e.profile),
            );
            profiles.extend(
                (model.declared)()
                    .into_iter()
                    .filter_map(|entry| entry.profile),
            );
            for group in (model.context_parallel)() {
                group.check()?;
                profiles.push(group.profile);
            }
            if profiles[first..].iter().any(|profile| {
                *profile != Profile::StdDenseMoe && !model.profiles.contains(profile)
            }) {
                return Err(Refusal("a model contribution uses an unregistered profile"));
            }
        }
        let mut tags = BTreeMap::new();
        for profile in profiles {
            if *tags.entry(profile.tag()).or_insert(profile) != profile {
                return Err(Refusal("two profiles share a cubin tag"));
            }
        }
        let mut names = BTreeSet::new();
        let mut symbols = BTreeSet::new();
        for cubin in self.cubins() {
            if !names.insert(cubin.name) {
                return Err(Refusal("two cubins share a name"));
            }
            for entry in cubin.entries {
                if !symbols.insert(entry.symbol) {
                    return Err(Refusal("an entry symbol is in two cubins"));
                }
            }
        }
        invariance::merged(invariance::contributions(self))
            .map(|_| ())
            .map_err(|_| Refusal("two declarations of one symbol differ"))
    }

    /// The medium-M entries ([`gemm::mid::catalog`]), built once: each
    /// selection would otherwise rebuild every dense catalog.
    pub(crate) fn mid_entries(&self) -> &[gemm::mid::Entry] {
        self.mid.get_or_init(|| gemm::mid::entries(self))
    }

    /// The kernel base's catalogs alone.
    pub fn base() -> Catalogs {
        Catalogs::new(&[]).expect("the kernel base's catalogs are consistent")
    }

    /// Bind a fused GEMM recipe through its profile's owning model.
    pub(crate) fn fused_sources(&self, profile: Profile, spec: &mut CubinSpec) {
        let model = self
            .models
            .iter()
            .find(|model| model.profiles.contains(&profile))
            .expect("fused GEMM profile has an owning catalog");
        (model.gemm_fused_sources)(profile, spec);
    }

    /// The models' catalogs, in order.
    pub fn models(&self) -> &[&'static ModelCatalog] {
        &self.models
    }

    /// Every profile: the kernel base's synthetic one, then each model's.
    /// Per-profile cubins list in this order.
    pub fn profiles(&self) -> Vec<Profile> {
        let mut profiles = vec![Profile::StdDenseMoe];
        for model in &self.models {
            profiles.extend(model.profiles.iter().copied());
        }
        profiles
    }

    /// Every grouped-template schedule row: the kernel base's and the
    /// models'.
    pub fn schedules(&self) -> &Table {
        &self.schedules
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;
    use kernel_build::EntrySpec;

    const EMPTY: ModelCatalog = ModelCatalog {
        name: "empty",
        ..ModelCatalog::EMPTY
    };

    /// Each profile's group keeps the catalog order; other profiles have
    /// none.
    #[test]
    fn entries_group_by_profile_in_catalog_order() {
        let (one, two) = (
            Profile::StdDenseMoe,
            Profile::synthetic(ProfileNames {
                name: "two",
                tag: "two",
            }),
        );
        let by = ByProfile::new([(one, 1), (two, 2), (one, 3)], |(profile, _)| *profile);
        assert_eq!(by.of(one), [(one, 1), (one, 3)]);
        assert_eq!(by.of(two), [(two, 2)]);
        let empty = ByProfile::new(Vec::<(Profile, u32)>::new(), |(profile, _)| *profile);
        assert!(empty.of(one).is_empty());
    }

    /// A route cubin `name` holding one entry `symbol`.
    fn route_cubin(name: &str, symbol: &str) -> CubinSpec {
        let mut cubin = comm::cubin_spec();
        cubin.name = name.into();
        let entry = &cubin.entries[0];
        cubin.entries = vec![EntrySpec::new(
            symbol.to_string(),
            entry.params.clone(),
            entry.body.clone(),
            entry.max_threads,
            1,
        )];
        cubin
    }

    /// The kernel base's first dense entry under `symbol` and `profile`.
    fn dense(symbol: String, profile: Profile) -> gemm::GemmVariant {
        let mut variant = gemm::own_catalog().remove(0);
        variant.instantiation.symbol = symbol;
        variant.profile = profile;
        variant
    }

    /// A packed ring row key of a synthetic expert shape.
    fn orphan_key() -> Key {
        let shape = moe::Shape {
            top_k: 8,
            hidden: 1024,
            experts: 64,
            intermediate: 512,
        };
        moe::template::packed_keys(shape).remove(0)
    }

    /// A model whose table holds one row, `orphan_key`'s, and whose
    /// schedule keys are `keys`.
    fn with_one_row(name: &'static str, keys: fn() -> Vec<Key>) -> &'static ModelCatalog {
        let mut table = moe::template::shared_table();
        let knobs = [("depth", 2), ("split_parts", 1), ("strips", 1)]
            .map(|(name, value)| (name.to_string(), kernel_build::tuning::Knob::Count(value)))
            .into();
        table
            .upsert(kernel_build::tuning::Row {
                key: orphan_key(),
                schedule: "packed_ring".into(),
                knobs,
            })
            .unwrap();
        Box::leak(Box::new(ModelCatalog {
            name,
            schedules: Box::leak(table.render().into_boxed_str()),
            schedule_keys: keys,
            ..EMPTY
        }))
    }

    /// A schedule row must name a key of its model's inventory.
    #[test]
    fn refuses_schedule_rows_outside_the_model_keys() {
        assert_eq!(
            Catalogs::new(&[with_one_row("orphan", Vec::new)]).err(),
            Some(Refusal("a schedule row names no entry of its model"))
        );
        assert!(Catalogs::new(&[with_one_row("owned", || vec![orphan_key()])]).is_ok());
    }

    /// Medium-M selection reads the entries built on first use instead of
    /// rebuilding the dense catalogs on every call.
    #[test]
    fn medium_entries_are_built_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static BUILDS: AtomicUsize = AtomicUsize::new(0);
        static COUNTED: ModelCatalog = ModelCatalog {
            name: "counted",
            gemm_mid: || {
                BUILDS.fetch_add(1, Ordering::SeqCst);
                vec![]
            },
            ..EMPTY
        };
        let catalogs = Catalogs::new(&[&COUNTED]).unwrap();
        assert!(gemm::mid::select(&catalogs, "chop_gemm_absent_v1", 64).is_none());
        let built = BUILDS.load(Ordering::SeqCst);
        for _ in 0..3 {
            assert!(gemm::mid::select(&catalogs, "chop_gemm_absent_v1", 64).is_none());
        }
        assert!(gemm::mid::catalog(&catalogs).is_empty());
        assert_eq!(BUILDS.load(Ordering::SeqCst), built);
    }

    #[test]
    fn the_base_and_an_empty_model_compose() {
        assert!(Catalogs::new(&[&EMPTY]).is_ok());
        let cubins = Catalogs::base().cubins();
        assert!(cubins.iter().any(|c| c.name == comm::cubin_spec().name));
    }

    /// A model's dense entry named like a kernel-base comm entry: the two
    /// families' cubins would hold one symbol.
    #[test]
    fn refuses_a_cross_family_symbol() {
        const CROSS_PROFILE: Profile = Profile::synthetic(ProfileNames {
            name: "cross",
            tag: "cross",
        });
        static CROSS: ModelCatalog = ModelCatalog {
            name: "cross",
            profiles: &[CROSS_PROFILE],
            gemm: || {
                let symbol = comm::cubin_spec().entries[0].symbol.clone();
                vec![dense(symbol, CROSS_PROFILE)]
            },
            ..EMPTY
        };
        assert_eq!(
            Catalogs::new(&[&CROSS]).err(),
            Some(Refusal("an entry symbol is in two cubins"))
        );
    }

    /// Two models whose route cubins share a name, with different entries,
    /// and a route cubin entry that another cubin already holds.
    #[test]
    fn refuses_route_cubin_collisions() {
        static FIRST: ModelCatalog = ModelCatalog {
            name: "first",
            moe_route_cubins: |_| vec![route_cubin("chop_moe_test_route", "chop_moe_test_a_v1")],
            ..EMPTY
        };
        static SECOND: ModelCatalog = ModelCatalog {
            name: "second",
            moe_route_cubins: |_| vec![route_cubin("chop_moe_test_route", "chop_moe_test_b_v1")],
            ..EMPTY
        };
        static SHADOW: ModelCatalog = ModelCatalog {
            name: "shadow",
            moe_route_cubins: |_| {
                let symbol = comm::cubin_spec().entries[0].symbol.clone();
                vec![route_cubin("chop_moe_test_shadow", &symbol)]
            },
            ..EMPTY
        };
        assert!(Catalogs::new(&[&FIRST]).is_ok());
        assert!(Catalogs::new(&[&SECOND]).is_ok());
        assert_eq!(
            Catalogs::new(&[&FIRST, &SECOND]).err(),
            Some(Refusal("two cubins share a name"))
        );
        assert_eq!(
            Catalogs::new(&[&SHADOW]).err(),
            Some(Refusal("an entry symbol is in two cubins"))
        );
    }

    /// Two distinct profiles with one tag would name one cubin and one
    /// symbol stem.
    #[test]
    fn refuses_two_profiles_of_one_tag() {
        const ONE: Profile = Profile::synthetic(ProfileNames {
            name: "one",
            tag: "shared_tag",
        });
        const OTHER: Profile = Profile::synthetic(ProfileNames {
            name: "other",
            tag: "shared_tag",
        });
        static A: ModelCatalog = ModelCatalog {
            name: "a",
            profiles: &[ONE],
            gemm: || vec![dense("chop_gemm_shared_tag_a_v1".into(), ONE)],
            ..EMPTY
        };
        static B: ModelCatalog = ModelCatalog {
            name: "b",
            profiles: &[OTHER],
            gemm: || vec![dense("chop_gemm_shared_tag_b_v1".into(), OTHER)],
            ..EMPTY
        };
        const BASE_PROFILE: Profile = Profile::synthetic(ProfileNames {
            name: "not-std",
            tag: "std",
        });
        static BASE_TAG: ModelCatalog = ModelCatalog {
            name: "base-tag",
            profiles: &[BASE_PROFILE],
            gemm: || vec![dense("chop_gemm_not_std_v1".into(), BASE_PROFILE)],
            ..EMPTY
        };
        static COMM_TAG: ModelCatalog = ModelCatalog {
            name: "comm-tag",
            profiles: &[OTHER],
            comm_shapes: |ranks| {
                let mut entry = comm::own_catalog_for(ranks).unwrap().remove(0);
                entry.profile = OTHER;
                vec![entry]
            },
            ..EMPTY
        };
        assert!(Catalogs::new(&[&COMM_TAG]).is_ok());
        assert_eq!(
            Catalogs::new(&[&A, &COMM_TAG]).err(),
            Some(Refusal("two profiles share a cubin tag"))
        );
        assert!(Catalogs::new(&[&A]).is_ok());
        assert_eq!(
            Catalogs::new(&[&A, &B]).err(),
            Some(Refusal("two profiles share a cubin tag"))
        );
        assert_eq!(
            Catalogs::new(&[&BASE_TAG]).err(),
            Some(Refusal("two profiles share a cubin tag"))
        );
    }
}
