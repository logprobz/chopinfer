//! The served envelopes with the dense prefill tail candidate enabled.
//! The ordinary profile sweeps retain the option-off coverage and pins.
use super::*;
use bundle_runtime::fake_host::conformance::route_guard;
use conformance::Startup;
use model_deepseek41_contract::execution_options::PREFILL_TAIL_FOLDS;

struct TailFolds(Served);
const PREFIX: &str = "tail-folds-";

fn base(serving: &conformance::Serving) -> conformance::Serving {
    let mut serving = serving.clone();
    serving.profile = serving
        .profile
        .strip_prefix(PREFIX)
        .expect("candidate profile")
        .into();
    serving
        .model_options
        .insert(PREFILL_TAIL_FOLDS.into(), "true".into());
    serving
}

impl Startup for TailFolds {
    type Observer = <Served as Startup>::Observer;
    type Limits = Limits;
    fn profiles() -> &'static str {
        static PROFILES: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        PROFILES.get_or_init(|| {
            let mut document: serde_json::Value = serde_json::from_str(Served::profiles()).unwrap();
            for name in ["tp4-ar", "tp4-dspark"] {
                let mut profile = document["profiles"][name].clone();
                profile["model_options"][PREFILL_TAIL_FOLDS] = "true".into();
                document["profiles"].as_object_mut().unwrap().remove(name);
                document["profiles"][format!("{PREFIX}{name}")] = profile;
            }
            document.to_string()
        })
    }
    fn servings() -> Vec<conformance::Serving> {
        Served::servings()
            .into_iter()
            .map(|mut serving| {
                serving.profile = format!("{PREFIX}{}", serving.profile);
                serving
                    .model_options
                    .insert(PREFILL_TAIL_FOLDS.into(), "true".into());
                serving
            })
            .collect()
    }
    fn envelope(serving: &conformance::Serving) -> conformance::Envelope {
        let mut envelope = Served::envelope(&base(serving));
        envelope
            .probes
            .extend([65, 85, 127, 129, 191, 789, 831, 853, 1557, 1599]);
        envelope
    }
    fn check_allocated_route_shapes(
        serving: &conformance::Serving,
        shapes: &[runtime_bundle_sdk::warmup::AllocatedShape],
    ) {
        Served::check_allocated_route_shapes(&base(serving), shapes);
    }
    fn route_tags(&self) -> Vec<kernel_core::route::RouteTag> {
        let shape = self
            .0
            .host
            .shape
            .with_verify_rows(self.0.serving.verify_rows())
            .with_prefill_tail_folds(true);
        let mut tags = crate::Kernels::route_tags(&shape);
        tags.extend(crate::engram_rows::route_tags());
        tags
    }
    fn route_threads(tier: route_guard::Tier) -> usize {
        Served::route_threads(tier)
    }
    fn route_device(&self) -> &bundle_runtime::fake_host::FakeDevice<Self::Observer> {
        self.0.route_device()
    }
    fn new(serving: &conformance::Serving) -> Self {
        Self(Served::new(&base(serving)))
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

mod tail_route_sweep {
    use super::*;
    fn sweep(profile: &str) {
        route_guard::sweep::<TailFolds>(
            profile,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        );
    }
    #[test]
    fn tp4_ar() {
        sweep("tail-folds-tp4-ar");
    }
    #[test]
    fn tp4_dspark() {
        sweep("tail-folds-tp4-dspark");
    }
}
