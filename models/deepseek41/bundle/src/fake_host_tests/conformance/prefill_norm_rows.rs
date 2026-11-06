//! Route coverage of the `prefill_norm_ceil_rows` candidate. The
//! tp4-dspark profile with the option on is swept as its own configuration,
//! `tp4-dspark-norm-rows`, with nothing else changed, so its envelope is the
//! profile's own and every prefill of 33 rows or more it reaches runs its
//! sublayer input norms fused. A separate name keeps its allowlist entries
//! apart from the default profile's.
use super::*;
use bundle_runtime::fake_host::conformance::route_guard;
use conformance::Startup;
use model_deepseek41_contract::execution_options::PREFILL_NORM_CEIL_ROWS;
use std::sync::OnceLock;

/// The swept configuration's name.
const PROFILE: &str = "tp4-dspark-norm-rows";
/// The profile it turns the candidate on in.
const BASE: &str = "tp4-dspark";

struct NormRows(Served);

impl Startup for NormRows {
    type Observer = <Served as Startup>::Observer;
    type Limits = Limits;

    /// The package's profiles with the candidate configuration beside them.
    fn profiles() -> &'static str {
        static PROFILES: OnceLock<String> = OnceLock::new();
        PROFILES.get_or_init(|| {
            let mut document: serde_json::Value =
                serde_json::from_str(Served::profiles()).expect("profiles.json");
            let mut serving = document["profiles"][BASE].clone();
            assert!(serving.is_object(), "{BASE} is a profile");
            serving["model_options"][PREFILL_NORM_CEIL_ROWS] = "true".into();
            document["profiles"][PROFILE] = serving;
            document.to_string()
        })
    }
    /// The candidate configuration first, so it represents its own quick tier.
    fn servings() -> Vec<conformance::Serving> {
        let mut servings = Served::servings();
        let mut candidate = servings
            .iter()
            .find(|serving| serving.profile == BASE)
            .expect("the DSpark profile")
            .clone();
        candidate.profile = PROFILE.into();
        candidate
            .model_options
            .insert(PREFILL_NORM_CEIL_ROWS.into(), "true".into());
        servings.insert(0, candidate);
        servings
    }
    /// The profile's own model shapes.
    fn envelope(serving: &conformance::Serving) -> conformance::Envelope {
        if serving.profile != PROFILE {
            return Served::envelope(serving);
        }
        let mut base = serving.clone();
        base.profile = BASE.into();
        Served::envelope(&base)
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
fn the_candidate_starts_beside_the_profiles() {
    conformance::startup_profiles::<NormRows>(&[
        ("tp4_dspark_norm_rows", PROFILE),
        ("tp4_ar", "tp4-ar"),
        ("tp4_dspark", BASE),
    ]);
    let coverage = conformance::serving_startup_of::<NormRows>(PROFILE);
    assert_eq!(coverage.profile, PROFILE);
}

mod prefill_norm_rows_route_sweep {
    use super::*;

    #[test]
    fn tp4_dspark_norm_rows() {
        route_guard::sweep::<NormRows>(
            PROFILE,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        );
    }
}
