//! Route coverage with `prefill_streamed_lowering` on: both served
//! profiles, their instances lowering each upper-skip prefill that no kept
//! plan or template serves while it issues. The option moves when the
//! lowering runs, not what it lowers, so the servings and their allowlist
//! entries are the profiles' own and the option is the instance's
//! override, as the template sweep's verify mode is. Every tier sweeps it,
//! and each sweep must stream.
use super::*;
use bundle_runtime::fake_host::conformance::route_guard;
use conformance::Startup;
use model_deepseek41_contract::execution_options::PREFILL_STREAMED_LOWERING;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Streamed launches each profile's sweep made.
static LAUNCHES: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

struct Streamed(Served);

impl Startup for Streamed {
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
        serving
            .model_options
            .insert(PREFILL_STREAMED_LOWERING.into(), "true".into());
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
        // A worker's launches count on its own thread.
        let before = crate::instance::execute::skip::streamed_launches();
        let result = self
            .0
            .execute_route_step(instance, items, prompt_ends, sink);
        *LAUNCHES
            .lock()
            .unwrap()
            .entry(self.0.serving.profile.clone())
            .or_default() += crate::instance::execute::skip::streamed_launches() - before;
        result
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

mod streamed_route_sweep {
    use super::*;

    // CI gives each profile its own process and RSS budget.
    fn sweep(profile: &str) {
        // A sweep in a child process counts its launches there.
        if !route_guard::sweep::<Streamed>(
            profile,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        ) {
            return;
        }
        let launches = LAUNCHES.lock().unwrap().get(profile).copied().unwrap_or(0);
        println!("{profile}: {launches} prefills lowered while they issued");
        assert!(launches > 0, "{profile}: the sweep streamed no prefill");
    }

    #[test]
    fn tp4_ar() {
        sweep("tp4-ar");
    }

    #[test]
    fn tp4_dspark() {
        sweep("tp4-dspark");
    }
}
