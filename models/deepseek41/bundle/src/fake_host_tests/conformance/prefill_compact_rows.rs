//! Route coverage of the `prefill_compact_dense_rows` candidate. The
//! tp4-dspark profile with the option on is swept as its own configuration,
//! `tp4-dspark-compact-rows`, with nothing else changed, so its envelope is
//! the profile's own and every prefill projection count the candidate table
//! admits runs the compact entry. A separate name keeps its allowlist
//! entries apart from the default profile's.
use super::*;
use bundle_runtime::fake_host::conformance::route_guard;
use conformance::Startup;
use model_deepseek41_contract::execution_options::PREFILL_COMPACT_DENSE_ROWS;
use std::sync::OnceLock;

/// The swept configuration's name.
const PROFILE: &str = "tp4-dspark-compact-rows";
/// The profile it turns the candidate on in.
const BASE: &str = "tp4-dspark";

struct CompactRows(Served);

impl Startup for CompactRows {
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
            serving["model_options"][PREFILL_COMPACT_DENSE_ROWS] = "true".into();
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
            .insert(PREFILL_COMPACT_DENSE_ROWS.into(), "true".into());
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
    /// candidate's ranges are registered only with its option on, and the
    /// sweep plans its boundary cases from these tags.
    fn route_tags(&self) -> Vec<kernel_core::route::RouteTag> {
        let enabled = self.0.execution().prefill_compact_dense_rows;
        self.0
            .route_tags_with(|shape| shape.with_prefill_compact_dense_rows(enabled))
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
    conformance::startup_profiles::<CompactRows>(&[
        ("tp4_dspark_compact_rows", PROFILE),
        ("tp4_ar", "tp4-ar"),
        ("tp4_dspark", BASE),
    ]);
    let coverage = conformance::serving_startup_of::<CompactRows>(PROFILE);
    assert_eq!(coverage.profile, PROFILE);
}

/// The candidate's ranges as job D1a set them, by fold shape.
const RANGES: [(&str, &[std::ops::RangeInclusive<u32>]); 4] = [
    ("_n512_k5120_g1_", &[1153..=3071, 3073..=3199]),
    ("_n576_k5120_g1_", &[1153..=2943, 2945..=3200]),
    ("_n1280_k5120_g1_", &[1153..=1279, 1281..=2368, 2401..=3200]),
    ("_n1024_k4096_g2_", &[1153..=3200]),
];

/// The swept tags admit each of `RANGES` under both the packed tiles' entry
/// and the admission entry of its projection, and the table admits the same
/// ranges, so the sweep plans every range edge and the rows on either side
/// in each prefill mode. The default profile's tags hold no range.
fn check_swept_ranges() {
    use crate::sm120::mxfp8_fold::{compact, prefill_entries};
    let range_tags = |profile: &str| {
        let serving = CompactRows::servings()
            .into_iter()
            .find(|serving| serving.profile == profile)
            .expect("a declared serving");
        kernel_core::route::merge(CompactRows::new(&serving).route_tags())
            .expect("consistent route registrations")
            .into_iter()
            .filter(|tag| tag.entry.starts_with("mxfp8_prefill/compact_range/"))
            .collect::<Vec<_>>()
    };
    assert!(range_tags(BASE).is_empty());
    let tags = range_tags(PROFILE);
    for (shape, ranges) in RANGES {
        let mut suffixes = Vec::new();
        for tag in tags.iter().filter(|tag| tag.symbol.contains(shape)) {
            assert_eq!(tag.class, kernel_core::route::RouteClass::Fast);
            assert_eq!(tag.admitted, ranges, "{}", tag.entry);
            suffixes.push(tag.symbol.rsplit_once("_v1_").unwrap().1);
        }
        suffixes.sort_unstable();
        assert_eq!(suffixes, ["admit_v1", "packed_v1"], "{shape}");
    }
    assert_eq!(tags.len(), 2 * RANGES.len());
    for base in prefill_entries(&FLASH, 4).unwrap() {
        let q = base.query;
        let shape = format!("_n{}_k{}_g{}_", q.n, q.k, q.groups);
        let table: Vec<_> = compact::ranges(&FLASH, 4, q)
            .unwrap()
            .into_iter()
            .map(|(rows, _)| rows)
            .collect();
        let pinned = RANGES
            .iter()
            .find(|(pinned, _)| *pinned == shape)
            .map_or(&[][..], |(_, ranges)| *ranges);
        assert_eq!(table, pinned, "{shape}");
    }
}

#[test]
fn the_swept_tags_admit_every_candidate_range() {
    check_swept_ranges();
}

mod prefill_compact_rows_route_sweep {
    use super::*;

    /// The sweep plans its range edges from the candidate's tags, so it
    /// first holds them to D1a's ranges: the tp4-dspark profiles' fallback
    /// allowances span every row, so a moved edge would pass the audit.
    #[test]
    fn tp4_dspark_compact_rows() {
        check_swept_ranges();
        route_guard::sweep::<CompactRows>(
            PROFILE,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        );
    }
}
