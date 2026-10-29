//! Route coverage with upper-skip plan templates, the default-off
//! `prefill_plan_templates` candidate, in its verify mode: both served
//! profiles, their instances keeping sixteen templates and lowering every
//! reused one again, refusing the step unless the fresh plan is the same.
//! Each sweep must reuse templates, so the comparison runs: besides the
//! served envelope, every worker runs one cut structure at least twice.
//! The servings are the profiles' own; the options are the instance's
//! overrides. While the candidate is off by default, only the quick tier
//! sweeps it; the other tiers say so and stop.
use super::*;
use bundle_runtime::fake_host::conformance::route_guard;
use conformance::Startup;
use model_deepseek41_contract::execution_options::{
    PREFILL_PLAN_CACHE_VERIFY, PREFILL_PLAN_TEMPLATES,
};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Templates each profile's sweep reused.
static HITS: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

struct Templated {
    served: Served,
    /// Hits of the shard's instance counted so far.
    seen: Cell<u64>,
}

impl Startup for Templated {
    type Observer = <Served as Startup>::Observer;
    type Limits = Limits;

    fn profiles() -> &'static str {
        Served::profiles()
    }
    fn servings() -> Vec<conformance::Serving> {
        Served::servings()
    }
    /// The served envelope and the same 512-row scheduler cut after eight
    /// prefixes. A sweep deals its cases in turn to its workers
    /// (four in the quick tier), so each worker's instance runs the cut at
    /// least twice and reuses a template whatever else it runs.
    fn envelope(serving: &conformance::Serving) -> conformance::Envelope {
        let mut envelope = Served::envelope(serving);
        envelope
            .cuts
            .extend((2..10).map(|blocks| conformance::CutCase {
                mode: format!("template-repeat-{}", 512 * blocks),
                offset: 512 * blocks,
                lengths: vec![512],
                remaining: vec![8],
            }));
        envelope
    }
    fn check_allocated_route_shapes(
        serving: &conformance::Serving,
        shapes: &[runtime_bundle_sdk::warmup::AllocatedShape],
    ) {
        Served::check_allocated_route_shapes(serving, shapes);
    }
    fn route_tags(&self) -> Vec<kernel_core::route::RouteTag> {
        self.served.route_tags()
    }
    fn route_threads(tier: route_guard::Tier) -> usize {
        Served::route_threads(tier)
    }
    fn route_device(&self) -> &bundle_runtime::fake_host::FakeDevice<Self::Observer> {
        self.served.route_device()
    }
    fn new(serving: &conformance::Serving) -> Self {
        let mut serving = serving.clone();
        for (name, value) in [
            (PREFILL_PLAN_TEMPLATES, "16"),
            (PREFILL_PLAN_CACHE_VERIFY, "true"),
        ] {
            serving.model_options.insert(name.into(), value.into());
        }
        Self {
            served: Served::new(&serving),
            seen: Cell::new(0),
        }
    }
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        self.seen.set(0);
        self.served.create()
    }
    fn step(&self, items: &[conformance::Item]) -> Vec<u8> {
        self.served.step(items)
    }
    fn execute_route_step(
        &self,
        instance: &mut BundleInstanceV1,
        items: &[conformance::Item],
        prompt_ends: &[u64],
        sink: &mut RecordingSink<Self::Limits>,
    ) -> Result<(), BundleAbiError> {
        let result = self
            .served
            .execute_route_step(instance, items, prompt_ends, sink);
        let hits = crate::instance::execute::plan_templates::counts(instance.handle())
            .expect("a templated instance")
            .hits;
        *HITS
            .lock()
            .unwrap()
            .entry(self.served.serving.profile.clone())
            .or_default() += hits - self.seen.replace(hits);
        result
    }
    fn execute_allocated_step(
        &self,
        instance: &mut BundleInstanceV1,
        items: &[conformance::Item],
        allocation: runtime_bundle_sdk::VerifyAllocationV1,
    ) -> Result<(), BundleAbiError> {
        self.served
            .execute_allocated_step(instance, items, allocation)
    }
    fn token(&self, position: u64) -> u32 {
        self.served.token(position)
    }
    fn assert_released(&self) {
        self.served.assert_released();
    }
}

mod template_route_sweep {
    use super::*;

    // CI gives each profile its own process and RSS budget.
    fn sweep(profile: &str) {
        if std::env::var("CHOPINFER_ROUTE_SWEEP").is_ok_and(|tier| tier != "quick") {
            println!("{profile}: route sweep covers the quick tier only");
            return;
        }
        route_guard::sweep::<Templated>(
            profile,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        );
        let hits = HITS.lock().unwrap().get(profile).copied().unwrap_or(0);
        println!("{profile}: {hits} reused templates, each compared with a fresh plan");
        assert!(hits > 0, "{profile}: the sweep reused no template");
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
