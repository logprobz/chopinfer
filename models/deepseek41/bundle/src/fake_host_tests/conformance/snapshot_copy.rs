//! Route coverage with the `prefix_snapshot_copy_kernel` candidate on. Each
//! served profile with the candidate on is swept as its own configuration,
//! `tp4-ar-snapshot-copy` and `tp4-dspark-snapshot-copy`, with nothing else
//! changed, so its envelope is its profile's own. The candidate moves
//! prefix snapshots, which no step launches, so the sweeps check that
//! loading the kernel and allocating its tables leave every step's routes
//! as they are. A separate name keeps each one's allowlist entries apart
//! from its profile's.
use super::*;
use bundle_runtime::fake_host::conformance::route_guard;
use conformance::Startup;
use model_deepseek41_contract::execution_options::PREFIX_SNAPSHOT_COPY_KERNEL;
use std::sync::OnceLock;

/// Each swept configuration's name and the profile it turns the candidate
/// on in.
const CANDIDATES: [(&str, &str); 2] = [
    ("tp4-ar-snapshot-copy", "tp4-ar"),
    ("tp4-dspark-snapshot-copy", "tp4-dspark"),
];
const OPTIONS: [&str; 1] = [PREFIX_SNAPSHOT_COPY_KERNEL];

/// The profile `profile` turns the candidate on in, if it is a candidate.
fn base_of(profile: &str) -> Option<&'static str> {
    CANDIDATES
        .iter()
        .find(|(candidate, _)| *candidate == profile)
        .map(|(_, base)| *base)
}

struct SnapshotCopy(Served);

impl Startup for SnapshotCopy {
    type Observer = <Served as Startup>::Observer;
    type Limits = Limits;

    /// The package's profiles with the candidate configurations beside
    /// them.
    fn profiles() -> &'static str {
        static PROFILES: OnceLock<String> = OnceLock::new();
        PROFILES.get_or_init(|| {
            let mut document: serde_json::Value =
                serde_json::from_str(Served::profiles()).expect("profiles.json");
            for (profile, base) in CANDIDATES {
                let mut serving = document["profiles"][base].clone();
                assert!(serving.is_object(), "{base} is a profile");
                for option in OPTIONS {
                    serving["model_options"][option] = "true".into();
                }
                document["profiles"][profile] = serving;
            }
            document.to_string()
        })
    }
    /// The candidate configurations first, so each represents its own
    /// quick tier.
    fn servings() -> Vec<conformance::Serving> {
        let mut servings = Served::servings();
        for (index, (profile, base)) in CANDIDATES.into_iter().enumerate() {
            let mut candidate = servings
                .iter()
                .find(|serving| serving.profile == base)
                .expect("the base profile serves")
                .clone();
            candidate.profile = profile.into();
            for option in OPTIONS {
                candidate.model_options.insert(option.into(), "true".into());
            }
            servings.insert(index, candidate);
        }
        servings
    }
    /// The base profile's own model shapes.
    fn envelope(serving: &conformance::Serving) -> conformance::Envelope {
        let Some(base) = base_of(&serving.profile) else {
            return Served::envelope(serving);
        };
        let mut base_serving = serving.clone();
        base_serving.profile = base.into();
        Served::envelope(&base_serving)
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

/// The candidate configuration starts like a served one beside the
/// package's profiles.
#[test]
fn the_candidates_start_beside_the_profiles() {
    conformance::startup_profiles::<SnapshotCopy>(&[
        ("tp4_ar_snapshot_copy", CANDIDATES[0].0),
        ("tp4_dspark_snapshot_copy", CANDIDATES[1].0),
        ("tp4_ar", "tp4-ar"),
        ("tp4_dspark", "tp4-dspark"),
    ]);
    for (profile, _) in CANDIDATES {
        let coverage = conformance::serving_startup_of::<SnapshotCopy>(profile);
        assert_eq!(coverage.profile, profile);
    }
}

/// Each candidate configuration loads the snapshot copy kernel on each of
/// its four ranks, and its base configuration loads it on none.
#[test]
fn only_the_candidates_load_the_snapshot_copy_kernel() {
    let profiles = CANDIDATES
        .into_iter()
        .flat_map(|(profile, base)| [(profile, 4), (base, 0)]);
    for (profile, ranks) in profiles {
        let serving = SnapshotCopy::servings()
            .into_iter()
            .find(|serving| serving.profile == profile)
            .expect("the profile serves");
        let startup = SnapshotCopy::new(&serving);
        let mut instance = startup.create().unwrap();
        let loaded = startup
            .route_device()
            .module_cubins
            .borrow()
            .values()
            .filter(|cubin| **cubin == crate::sm120::snapshot_copy::CUBIN)
            .count();
        assert_eq!(loaded, ranks, "{profile}");
        instance.destroy().unwrap();
        startup.assert_released();
    }
}

mod snapshot_copy_route_sweep {
    use super::*;

    fn sweep(profile: &str) {
        route_guard::sweep::<SnapshotCopy>(
            profile,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        );
    }

    #[test]
    fn tp4_ar_snapshot_copy() {
        sweep(CANDIDATES[0].0);
    }

    #[test]
    fn tp4_dspark_snapshot_copy() {
        sweep(CANDIDATES[1].0);
    }
}
