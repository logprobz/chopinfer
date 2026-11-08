//! Route coverage with the DSpark successor candidate on
//! (`dspark_successor_items` at 16). The option moves when the drafter's
//! projection and draft run, not what they lower, so the servings and
//! their allowlist entries are the profile's own and the option is the
//! instance's override, as the template sweep's verify mode is. The
//! sweep's fake device writes no acceptance record a successor can plan
//! from, so its steps queue none: this checks that every route stays
//! admitted with the option on, and the fake-host successor tests cover
//! the successors' launches. Every tier sweeps it.
use super::*;
use bundle_runtime::fake_host::conformance::route_guard;
use conformance::Startup;
use model_deepseek41_contract::execution_options::DSPARK_SUCCESSOR_ITEMS;

const OPTIONS: [(&str, &str); 1] = [(DSPARK_SUCCESSOR_ITEMS, "16")];

struct Successor(Served);

impl Startup for Successor {
    type Observer = <Served as Startup>::Observer;
    type Limits = Limits;

    fn profiles() -> &'static str {
        Served::profiles()
    }
    fn servings() -> Vec<conformance::Serving> {
        Served::servings()
    }
    fn envelope(serving: &conformance::Serving) -> conformance::Envelope {
        Served::envelope(serving)
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
        let mut serving = serving.clone();
        for (option, value) in OPTIONS {
            serving.model_options.insert(option.into(), value.into());
        }
        Self(Served::new(&serving))
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

mod successor_route_sweep {
    use super::*;

    #[test]
    fn tp4_dspark() {
        route_guard::sweep::<Successor>(
            "tp4-dspark",
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        );
    }
}
