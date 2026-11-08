//! Route coverage of the `dspark_main_short_rows` candidate. The tp4-dspark
//! profile with the option on is swept as its own configuration,
//! `tp4-dspark-main-short`, with nothing else changed, so its envelope is
//! the profile's own and every main projection pass of 65 to 128 inner rows
//! it reaches runs the compact pair. A separate name keeps its allowlist
//! entries apart from the default profile's. The sweep reaches those rows
//! at commit (`main-inner-128`); the catch-up of restored windows, of short
//! prompts committed whole and of a cold prompt at its prefix capture is
//! `dspark_main_compact`'s to check.
use super::*;
use bundle_runtime::fake_host::conformance::route_guard;
use conformance::Startup;
use model_deepseek41_contract::execution_options::DSPARK_MAIN_SHORT_ROWS;
use std::sync::OnceLock;

/// The swept configuration's name.
const PROFILE: &str = "tp4-dspark-main-short";
/// The profile it turns the candidate on in.
const BASE: &str = "tp4-dspark";

struct MainShort(Served);

impl Startup for MainShort {
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
            serving["model_options"][DSPARK_MAIN_SHORT_ROWS] = "true".into();
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
            .insert(DSPARK_MAIN_SHORT_ROWS.into(), "true".into());
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
    /// The tags of the kernels this serving's options build. The
    /// candidate's rows are registered only with its option on, and the
    /// sweep plans its boundary cases from these tags.
    fn route_tags(&self) -> Vec<kernel_core::route::RouteTag> {
        let enabled = self.0.execution().dspark_main_short_rows;
        self.0
            .route_tags_with(|shape| shape.with_dspark_main_short_rows(enabled))
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
    conformance::startup_profiles::<MainShort>(&[
        ("tp4_dspark_main_short", PROFILE),
        ("tp4_ar", "tp4-ar"),
        ("tp4_dspark", BASE),
    ]);
    let coverage = conformance::serving_startup_of::<MainShort>(PROFILE);
    assert_eq!(coverage.profile, PROFILE);
}

/// The swept main projection tags: the candidate profile's compact pair
/// admits 65 to 128 rows under its own stage-3 admission and m32 compute
/// entries and 129 to 511 under the stage-2 pair, and so does the default
/// profile's, the candidate being the default, so the sweep plans every
/// range edge.
fn check_swept_main_ranges() {
    let main_tags = |profile: &str| {
        let serving = MainShort::servings()
            .into_iter()
            .find(|serving| serving.profile == profile)
            .expect("a declared serving");
        let mut tags: Vec<_> = kernel_core::route::merge(MainShort::new(&serving).route_tags())
            .expect("consistent route registrations")
            .into_iter()
            .filter(|tag| tag.entry.starts_with("dspark_main/compact/"))
            .map(|tag| {
                let entry = tag.symbol.rsplit_once("_compact_").unwrap().1.to_owned();
                assert_eq!(tag.class, kernel_core::route::RouteClass::Fast);
                (entry, tag.admitted)
            })
            .collect();
        tags.sort_by(|a, b| a.0.cmp(&b.0));
        tags
    };
    let upper = [
        ("m64_n128_s2_v1_admit_v1".to_owned(), vec![129..=511]),
        ("m64_n128_s2_v1_packed_v1".to_owned(), vec![129..=511]),
    ];
    let mut short = vec![
        ("m32_n128_s3_v1_packed_v1".to_owned(), vec![65..=128]),
        ("m64_n128_s3_v1_admit_v1".to_owned(), vec![65..=128]),
    ];
    short.extend(upper);
    short.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(main_tags(BASE), short);
    assert_eq!(main_tags(PROFILE), short);
}

#[test]
fn the_swept_tags_admit_the_candidate_rows() {
    check_swept_main_ranges();
}

mod dspark_main_short_route_sweep {
    use super::*;

    /// The sweep plans its range edges from the candidate's tags, so it
    /// first holds them to the candidate table's rows.
    #[test]
    fn tp4_dspark_main_short() {
        check_swept_main_ranges();
        route_guard::sweep::<MainShort>(
            PROFILE,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        );
    }
}
