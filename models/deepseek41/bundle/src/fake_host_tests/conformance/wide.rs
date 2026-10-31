//! Route coverage of verify steps above the profile's 64-row budget. The
//! tp4-dspark profile served with the package's 192 verify rows
//! (`--speculative-rows 192`) admits eager verify steps of 65 to 192 rows,
//! which the profile's own sweep never reaches. That supported override is
//! swept as its own configuration, `tp4-dspark-rows192`: the profile with
//! `speculation.rows` raised to 192 and nothing else changed, so its graph
//! budget stays at 64 rows and every wider step runs eagerly. The
//! `wide_verify` candidate is swept beside it as `tp4-dspark-rows192-wide`,
//! the same override with the option on. The `f32_row_tiles` candidate is
//! swept as `tp4-dspark-tiles`, the profile itself with the option on.
//! Separate names keep their allowlist entries apart from the default
//! profile, whose full sweep refuses entries it never reaches; entries for
//! the shapes they share name `tp4-dspark*`.
//!
//! Besides the host's own shapes, the envelope adds every eager width with
//! 32 items, and 65 to 96 rows with 16 items, drafts spread as evenly as
//! the per-item depths allow.
use super::*;
use bundle_runtime::fake_host::conformance::route_guard;
use conformance::Startup;
use model_deepseek41_contract::{
    execution_options::{F32_ROW_TILES, WIDE_VERIFY},
    serving::MAX_VERIFY_ROWS,
};
use runtime_bundle_sdk::warmup::VerifyShape;
use std::sync::OnceLock;

/// The swept override's configuration name.
const PROFILE: &str = "tp4-dspark-rows192";
/// The override with the `wide_verify` candidate on.
const WIDE_PROFILE: &str = "tp4-dspark-rows192-wide";
/// The profile with the `f32_row_tiles` candidate on.
const TILES_PROFILE: &str = "tp4-dspark-tiles";
/// The profile the override raises.
const BASE: &str = "tp4-dspark";
/// The package's verify rows: every item at its deepest block.
const ROWS: u32 =
    <model_deepseek41_contract::serving::DeepSeek as bundle_runtime::BundleGeometry>::VERIFY_TOKENS;

struct Wide(Served);

/// A verify shape of `items` items over `rows` rows, the first
/// `rows % items` items one row deeper.
fn eager(rows: u32, items: u32) -> VerifyShape {
    VerifyShape {
        row_bucket: rows as usize,
        item_bucket: items as usize,
        drafts: (0..items)
            .map(|item| (rows / items + u32::from(item < rows % items) - 1) as usize)
            .collect(),
    }
}

/// The eager verify steps above `graphed` rows up to `rows`: every width
/// with 32 items, and with 16 items up to their 96 rows.
fn eager_shapes(graphed: u32, rows: u32) -> Vec<VerifyShape> {
    [(32, rows), (16, rows.min(16 * MAX_VERIFY_ROWS))]
        .into_iter()
        .flat_map(|(items, last)| (graphed + 1..=last).map(move |rows| eager(rows, items)))
        .collect()
}

impl Startup for Wide {
    type Observer = <Served as Startup>::Observer;
    type Limits = Limits;

    /// The package's profiles with the overrides beside them.
    fn profiles() -> &'static str {
        static PROFILES: OnceLock<String> = OnceLock::new();
        PROFILES.get_or_init(|| {
            let mut document: serde_json::Value =
                serde_json::from_str(Served::profiles()).expect("profiles.json");
            let mut wide = document["profiles"][BASE].clone();
            assert!(wide.is_object(), "{BASE} is a profile");
            wide["speculation"]["rows"] = ROWS.into();
            document["profiles"][PROFILE] = wide.clone();
            wide["model_options"][WIDE_VERIFY] = "true".into();
            document["profiles"][WIDE_PROFILE] = wide;
            let mut tiles = document["profiles"][BASE].clone();
            tiles["model_options"][F32_ROW_TILES] = "true".into();
            document["profiles"][TILES_PROFILE] = tiles;
            document.to_string()
        })
    }
    /// The overrides first, so each represents its own quick tier.
    fn servings() -> Vec<conformance::Serving> {
        let mut servings = Served::servings();
        let base = servings
            .iter()
            .find(|serving| serving.profile == BASE)
            .expect("the DSpark profile")
            .clone();
        for (at, profile) in [PROFILE, WIDE_PROFILE, TILES_PROFILE]
            .into_iter()
            .enumerate()
        {
            let mut wide = base.clone();
            wide.profile = profile.into();
            match profile {
                WIDE_PROFILE => {
                    wide.model_options.insert(WIDE_VERIFY.into(), "true".into());
                }
                TILES_PROFILE => {
                    wide.model_options
                        .insert(F32_ROW_TILES.into(), "true".into());
                }
                _ => {}
            }
            servings.insert(at, wide);
        }
        servings
    }
    fn envelope(serving: &conformance::Serving) -> conformance::Envelope {
        if serving.profile == TILES_PROFILE {
            let mut base = serving.clone();
            base.profile = BASE.into();
            return Served::envelope(&base);
        }
        if ![PROFILE, WIDE_PROFILE].contains(&serving.profile.as_str()) {
            return Served::envelope(serving);
        }
        // The profile's own model shapes, at the override's verify rows.
        let mut base = serving.clone();
        base.profile = BASE.into();
        let mut envelope = Served::envelope(&base);
        let graphed = serving.verify.graph_rows.expect("a graph budget");
        assert_eq!((graphed, serving.verify_rows()), (64, ROWS));
        envelope.verify.extend(eager_shapes(graphed, ROWS));
        envelope
    }
    fn check_allocated_route_shapes(
        serving: &conformance::Serving,
        shapes: &[runtime_bundle_sdk::warmup::AllocatedShape],
    ) {
        Served::check_allocated_route_shapes(serving, shapes);
    }
    fn route_tags(&self) -> Vec<kernel_core::route::RouteTag> {
        self.0.route_tags()
    }
    fn route_threads(tier: route_guard::Tier) -> usize {
        Served::route_threads(tier)
    }
    fn route_device(&self) -> &bundle_runtime::fake_host::FakeDevice<Self::Observer> {
        self.0.route_device()
    }
    fn new(serving: &conformance::Serving) -> Self {
        Self(Served::new(serving))
    }
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        self.0.create()
    }
    fn step(&self, items: &[conformance::Item]) -> Vec<u8> {
        self.0.step(items)
    }
    fn execute_route_step(
        &self,
        instance: &mut BundleInstanceV1,
        items: &[conformance::Item],
        prompt_ends: &[u64],
        sink: &mut RecordingSink<Self::Limits>,
    ) -> Result<(), BundleAbiError> {
        self.0
            .execute_route_step(instance, items, prompt_ends, sink)
    }
    fn execute_allocated_step(
        &self,
        instance: &mut BundleInstanceV1,
        items: &[conformance::Item],
        allocation: runtime_bundle_sdk::VerifyAllocationV1,
    ) -> Result<(), BundleAbiError> {
        self.0.execute_allocated_step(instance, items, allocation)
    }
    fn token(&self, position: u64) -> u32 {
        self.0.token(position)
    }
    fn assert_released(&self) {
        self.0.assert_released();
    }
}

/// The overrides are three more configurations beside the package's
/// profiles, and their instances start like served ones: create, prefix
/// description and the host's startup warmup, at 192 verify rows for the
/// raised ones.
#[test]
fn the_override_starts_beside_the_profiles() {
    conformance::startup_profiles::<Wide>(&[
        ("tp4_dspark_rows192", PROFILE),
        ("tp4_dspark_rows192_wide", WIDE_PROFILE),
        ("tp4_dspark_tiles", TILES_PROFILE),
        ("tp4_ar", "tp4-ar"),
        ("tp4_dspark", BASE),
    ]);
    for profile in [PROFILE, WIDE_PROFILE, TILES_PROFILE] {
        let coverage = conformance::serving_startup_of::<Wide>(profile);
        assert_eq!(coverage.profile, profile);
    }
}

#[test]
fn eager_shapes_cover_every_width_above_the_graph_budget() {
    let shapes = eager_shapes(64, ROWS);
    let keys: Vec<_> = shapes
        .iter()
        .map(|shape| (shape.item_bucket, shape.row_bucket))
        .collect();
    assert_eq!(keys.len(), 128 + 32);
    for rows in 65..=192 {
        assert!(keys.contains(&(32, rows)));
        assert_eq!(keys.contains(&(16, rows)), rows <= 96);
    }
    for shape in &shapes {
        assert_eq!(shape.drafts.len(), shape.item_bucket);
        assert_eq!(
            shape.drafts.iter().map(|drafts| drafts + 1).sum::<usize>(),
            shape.row_bucket
        );
        assert!(
            shape
                .drafts
                .iter()
                .all(|&drafts| drafts < MAX_VERIFY_ROWS as usize)
        );
        let (most, least) = (shape.drafts.iter().max(), shape.drafts.iter().min());
        assert!(most.unwrap() - least.unwrap() <= 1);
    }
}

mod wide_route_sweep {
    use super::*;

    #[test]
    fn tp4_dspark_rows192() {
        route_guard::sweep::<Wide>(
            PROFILE,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        );
    }

    #[test]
    fn tp4_dspark_rows192_wide() {
        route_guard::sweep::<Wide>(
            WIDE_PROFILE,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        );
    }

    #[test]
    fn tp4_dspark_tiles() {
        route_guard::sweep::<Wide>(
            TILES_PROFILE,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        );
    }
}
