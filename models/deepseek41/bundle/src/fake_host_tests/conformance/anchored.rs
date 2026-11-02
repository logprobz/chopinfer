//! Route coverage with prompt-end anchors kept inside items on both served
//! profiles. Each step's first eight eligible prompt-ending items keep their
//! interior grid anchor. The quick tier also compares reused templates with
//! fresh lowering and requires anchored template hits.
use super::*;
use bundle_runtime::fake_host::conformance::route_guard;
use conformance::Startup;
use model_deepseek41_contract::execution_options::PREFILL_PLAN_CACHE_VERIFY;
use std::collections::BTreeMap;
use std::sync::Mutex;

const ANCHOR_ITEMS: u32 = model_deepseek41_contract::execution_options::MAX_ANCHOR_ITEMS;

/// Anchored template hits whose launches were compared with fresh lowering.
static VERIFIED_HITS: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

fn verify_templates() -> bool {
    std::env::var("CHOPINFER_ROUTE_SWEEP").map_or(true, |tier| tier == "quick")
}

struct Anchored(Served);

impl Startup for Anchored {
    type Observer = <Served as Startup>::Observer;
    type Limits = Limits;

    fn profiles() -> &'static str {
        Served::profiles()
    }
    fn servings() -> Vec<conformance::Serving> {
        Served::servings()
    }
    /// The served envelope and a mandatory set of anchored steps, which
    /// every anchored sweep runs: packed
    /// prompts each keeping an anchor, a ring that
    /// crosses the 4096-row key part, restored odd starts, one whose anchor
    /// lies 135 rows in (seeded rings, the last seeded distance) beside a
    /// second anchored item, and a 67-row spare write.
    fn envelope(serving: &conformance::Serving) -> conformance::Envelope {
        let mut envelope = Served::envelope(serving);
        let anchored = |mode: &str, offset, lengths: &[u32]| conformance::CutCase {
            mode: mode.into(),
            offset,
            lengths: lengths.to_vec(),
            remaining: vec![0; lengths.len()],
        };
        envelope.cuts.extend([
            anchored("anchor-packed", 0, &[789, 300, 100]),
            anchored("anchor-part", 0, &[4171]),
            anchored("anchor-restored", 789, &[300]),
            anchored("anchor-restored-packed", 761, &[139, 300]),
            anchored("anchor-short", 829, &[71]),
        ]);
        if verify_templates() {
            // Four quick-tier workers each see this anchored structure at
            // least twice, at different absolute boundaries.
            envelope.cuts.extend((0..8).map(|shift| {
                anchored(
                    &format!("anchor-template-repeat-{shift}"),
                    829 + 128 * shift,
                    &[68],
                )
            }));
        }
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
        let mut serving = serving.clone();
        if verify_templates() {
            serving
                .model_options
                .insert(PREFILL_PLAN_CACHE_VERIFY.into(), "true".into());
        }
        Self(Served::new(&serving))
    }
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        self.0.create_with_prefill(true)
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
        if items.iter().any(|item| item.phase != Phase::Prefill) {
            return self
                .0
                .execute_route_step(instance, items, prompt_ends, sink);
        }
        let mut anchors = 0;
        let mut batch = Vec::new();
        let mut appends = Vec::new();
        let mut slots = Vec::new();
        let mut boundaries = Vec::new();
        for (item, &prompt_tokens) in items.iter().zip(prompt_ends) {
            let rows = item.tokens.len() as u32;
            let end = item.start + u64::from(rows);
            // A setup chunk ending off the granule stands for the earlier
            // request whose prompt ended there, as a restore resumes its
            // snapshot: it ends that prompt, without an anchor.
            let setup = end != prompt_tokens && !end.is_multiple_of(64);
            let prompt_tokens = if setup { end } else { prompt_tokens };
            let rounded = end / 64 * 64;
            let anchor = !setup
                && end == prompt_tokens
                && item.start < rounded
                && rounded < end
                && anchors < ANCHOR_ITEMS;
            anchors += u32::from(anchor);
            batch.push(
                BatchItemV1::new(
                    item.request,
                    false,
                    rows,
                    end == prompt_tokens,
                    item.tokens.clone(),
                )
                .with_anchor(anchor),
            );
            appends.push(BatchAppendV1::new(
                item.start,
                u64::from(rows),
                end,
                (0..end.div_ceil(64))
                    .map(|page| item.slot * 1024 + page)
                    .collect(),
                None,
            ));
            slots.push(item.slot);
            boundaries.push(runtime_bundle_sdk::prefill::Boundary {
                request_id: item.request,
                prompt_tokens,
                chunk_end: end,
                chunk_tokens: 8192,
            });
        }
        let batch = BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
            .with_state_slots(slots)
            .encode()
            .unwrap();
        let encoded = runtime_bundle_sdk::prefill::encode_with_cuts(&batch, &boundaries)
            .expect("anchored prefill boundaries");
        let verified = anchors > 0 && verify_templates();
        let before = if verified {
            crate::instance::execute::plan_templates::counts(instance.handle()).hits
        } else {
            0
        };
        let result = instance.execute_prefill_cuts(&encoded, sink);
        if verified && result.is_ok() {
            let hits = crate::instance::execute::plan_templates::counts(instance.handle()).hits;
            *VERIFIED_HITS
                .lock()
                .unwrap()
                .entry(self.0.serving.profile.clone())
                .or_default() += hits - before;
        }
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

/// The anchored instances serve each profile with eight anchor items on
/// its grid, the legacy test API exposes no allowance, and an anchored step keeps
/// its prompt's rounded end for a capture.
#[test]
fn anchored_instances_keep_eight_anchor_items() {
    for serving in Anchored::servings() {
        for (anchors, declared) in [(false, None), (true, Some(ANCHOR_ITEMS))] {
            let mut instance = if anchors {
                Anchored::new(&serving).create()
            } else {
                Served::new(&serving).create()
            }
            .unwrap();
            assert_eq!(
                instance.prefill_anchor_items(),
                Ok(declared),
                "{}",
                serving.profile
            );
            instance.destroy().unwrap();
        }
        let adapter = Anchored::new(&serving);
        let mut instance = adapter.create().unwrap();
        let items = [conformance::Item {
            request: 7,
            slot: 0,
            start: 0,
            tokens: (1..=789).collect(),
            phase: Phase::Prefill,
        }];
        adapter
            .execute_route_step(&mut instance, &items, &[789], &mut RecordingSink::default())
            .unwrap();
        instance.commit().unwrap();
        let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
        instance
            .capture_prefix_state(7, 768, &pages)
            .unwrap_or_else(|e| panic!("{}: {e:?}", serving.profile));
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        adapter.assert_released();
    }
}

mod anchor_route_sweep {
    use super::*;

    // CI gives each profile its own process and RSS budget.
    fn sweep(profile: &str) {
        VERIFIED_HITS.lock().unwrap().insert(profile.into(), 0);
        route_guard::sweep::<Anchored>(
            profile,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
            env!("CARGO_PKG_NAME"),
        );
        if verify_templates() {
            let hits = VERIFIED_HITS
                .lock()
                .unwrap()
                .get(profile)
                .copied()
                .unwrap_or(0);
            println!(
                "{profile}: {hits} reused anchored templates, each compared with a fresh plan"
            );
            assert!(hits > 0, "{profile}: the sweep reused no anchored template");
        }
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
