//! The serving startup scenario over a probe bundle exported through the ABI
//! macro, whose create record names the step it refuses.

use std::cell::RefCell;
use std::rc::Rc;

use runtime_bundle_abi::{
    BundleKeptRowsV1, BundleStatusV1, ExecuteResultSinkV1, HostServicesTableV1, HostServicesV1,
    PrefixStateRequestV1, ResultSinkLimitsV1,
};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
use runtime_host::fake::FakeHostDeviceV1;
use runtime_host::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};

use super::*;
use crate::entry::{self, Bundle, Failed};
use crate::pipeline::Collectives;
use crate::prefix::PrefixInstance;
use crate::resources::{Held, status};
use crate::table::{InstanceTable, Live};

/// Holds one context and one stream. Its create record names a refusal:
/// `describe` refuses to describe its prefix state, `mixed` a step with
/// decode and verify items, `shallow` a verify item of two rows, `context`
/// a verify item after twelve committed tokens, `drafting` (as an argument
/// check) a step of more than two verify items, and `encoder` a create
/// record that names the vision encoder's components (`+encoder`).
struct Probe {
    services: HostServicesV1,
    held: Vec<Held>,
    collectives: Collectives,
    refuse: Vec<u8>,
}

thread_local! {
    static INSTANCES: RefCell<InstanceTable<Probe>> = const { RefCell::new(InstanceTable::new()) };
}

impl Live for Probe {
    fn collectives(&self) -> &Collectives {
        &self.collectives
    }
}

impl Bundle for Probe {
    fn instances() -> &'static std::thread::LocalKey<RefCell<InstanceTable<Self>>> {
        &INSTANCES
    }

    fn create(services: HostServicesV1, configuration: &[u8]) -> Result<Self, Failed<Self>> {
        if configuration == b"encoder+encoder" {
            return Err(BundleStatusV1::Unsupported.into());
        }
        let mut probe = Self {
            services,
            held: Vec::new(),
            collectives: Collectives::default(),
            refuse: configuration.to_vec(),
        };
        let acquired = probe.services.select_context(0, 0).and_then(|context| {
            probe.held.push(Held::Context(context));
            let stream = probe.services.create_stream(context, 0)?;
            probe.held.push(Held::Stream(stream));
            Ok(())
        });
        match acquired {
            Err(error) => Err(Failed::releasing(probe, status(error))),
            Ok(()) => Ok(probe),
        }
    }

    fn resources(&mut self) -> (&HostServicesV1, &mut Vec<Held>) {
        (&self.services, &mut self.held)
    }

    fn release_graphs(&mut self) -> Result<(), runtime_bundle_abi::BundleAbiError> {
        Ok(())
    }

    fn execute(&mut self, request: &[u8], _sink: &mut ExecuteResultSinkV1) -> BundleStatusV1 {
        let Ok(batch) = BatchRequestV1::decode(request) else {
            return BundleStatusV1::InvalidArgument;
        };
        let items = batch.items();
        let verify = items.iter().filter(|item| item.is_verify()).count();
        let refused = match self.refuse.as_slice() {
            b"mixed" => verify > 0 && items.iter().any(BatchItemV1::is_decode),
            b"shallow" => items
                .iter()
                .any(|item| item.is_verify() && item.phase_tokens() == 2),
            b"context" => items
                .iter()
                .zip(batch.appends())
                .any(|(item, append)| item.is_verify() && append.previous_tokens() >= 12),
            b"drafting" if verify > 2 => return BundleStatusV1::InvalidArgument,
            _ => false,
        };
        if refused {
            return BundleStatusV1::Unsupported;
        }
        BundleStatusV1::Success
    }

    fn commit(&mut self) -> BundleStatusV1 {
        BundleStatusV1::Success
    }

    fn commit_accepted(&mut self, _records: &[BundleKeptRowsV1]) -> BundleStatusV1 {
        BundleStatusV1::Success
    }

    fn cancel(&mut self) -> BundleStatusV1 {
        BundleStatusV1::Success
    }

    fn retire_request(&mut self, _request_id: u64) -> BundleStatusV1 {
        BundleStatusV1::Success
    }
}

impl PrefixInstance for Probe {
    fn state_pages(&self) -> Result<u64, BundleStatusV1> {
        if self.refuse == b"describe" {
            return Err(BundleStatusV1::Unsupported);
        }
        Ok(1)
    }

    fn capture(&mut self, _request: &PrefixStateRequestV1) -> Result<(), BundleStatusV1> {
        Err(BundleStatusV1::Unsupported)
    }

    fn restore(
        &mut self,
        _request: &PrefixStateRequestV1,
        _slot: Option<u64>,
    ) -> Result<(), BundleStatusV1> {
        Err(BundleStatusV1::Unsupported)
    }
}

runtime_bundle_abi::export_bundle_api_v1!(
    with_prefix_cache
    PROBE_API_V1,
    PROBE_EXTENDED_API_V1,
    entry::create::<Probe>,
    entry::execute::<Probe>,
    entry::commit::<Probe>,
    entry::cancel::<Probe>,
    entry::destroy::<Probe>,
    entry::retire_request::<Probe>,
    entry::commit_accepted::<Probe>,
    entry::describe_prefix_state::<Probe>,
    entry::capture_prefix_state::<Probe>,
    entry::restore_prefix_state::<Probe>,
);

const PROFILES: &str = r#"{
  "profiles": {
    "tp1-ar": {
      "tensor_parallel": 1,
      "speculation": {"tokens": 0, "drafter": "none"},
      "max_sequences": 8,
      "prefix_cache": true
    },
    "tp1-mtp": {
      "tensor_parallel": 1,
      "speculation": {"tokens": 3, "drafter": "mtp"},
      "max_sequences": 8,
      "prefix_cache": true,
      "model_options": {"checkpoint.layout": "probe"}
    }
  }
}"#;

const NONE: u8 = 0;
const DESCRIBE: u8 = 1;
const MIXED: u8 = 2;
const SHALLOW: u8 = 3;
const CONTEXT: u8 = 4;
/// Declares at most two drafting items and refuses three.
const LIMITED: u8 = 5;
/// Declares at most two drafting items but accepts three.
const UNDERSTATED: u8 = 6;
/// Declares four sequences where the profiles serve eight.
const MISDECLARED: u8 = 7;
/// The package provides images, so every profile serves them by default.
const IMAGES: u8 = 8;
/// The package provides images, but create refuses the encoder's components.
const NO_ENCODER: u8 = 9;
/// Selects video, which the package does not provide.
const UNPROVIDED: u8 = 10;

struct Limits;
impl SinkLimits for Limits {
    fn limits() -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(64, 1 << 16, 1 << 20)
    }
}

type Backend = HostServicesV1Backend<FakeHostDeviceV1>;

struct Adapter<const REFUSE: u8>(Rc<Backend>, Serving);

impl<const REFUSE: u8> Startup for Adapter<REFUSE> {
    type Limits = Limits;

    fn profiles() -> &'static str {
        PROFILES
    }
    fn servings() -> Vec<Serving> {
        let serving = |profile: &str, draft_tokens: u32| Serving {
            profile: profile.to_owned(),
            tensor_parallel: 1,
            context_parallel: 1,
            sequences: if REFUSE == MISDECLARED { 4 } else { 8 },
            drafter: if draft_tokens > 0 { "mtp" } else { "none" }.to_owned(),
            draft_tokens,
            prefix_cache: true,
            model_options: if draft_tokens > 0 {
                [("checkpoint.layout".to_owned(), "probe".to_owned())].into()
            } else {
                BTreeMap::new()
            },
            provides: match REFUSE {
                IMAGES | NO_ENCODER => vec!["text".to_owned(), "image".to_owned()],
                _ => vec!["text".to_owned()],
            },
            selection: (REFUSE == UNPROVIDED).then(|| vec!["text".to_owned(), "video".to_owned()]),
            package: Some(PackageSpeculation {
                maximum_draft_tokens: 3,
                bundle_drafter: false,
                maximum_verify_tokens: 32,
                row_buckets: vec![2, 4],
                maximum_draft_items: matches!(REFUSE, LIMITED | UNDERSTATED).then_some(2),
                narrow_rows: false,
                exact_graphs: false,
            }),
        };
        vec![serving("tp1-ar", 0), serving("tp1-mtp", 3)]
    }
    fn new(serving: &Serving) -> Self {
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap());
        Self(
            Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config)),
            serving.clone(),
        )
    }
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        let refuse: &[u8] = match REFUSE {
            DESCRIBE => b"describe",
            MIXED => b"mixed",
            SHALLOW => b"shallow",
            CONTEXT => b"context",
            LIMITED => b"drafting",
            NO_ENCODER => b"encoder",
            _ => b"",
        };
        let mut record = refuse.to_vec();
        if self.1.serves_images() {
            record.extend(b"+encoder");
        }
        runtime_bundle_abi::BundleApiV1::new_with_prefix_cache(PROBE_EXTENDED_API_V1)?
            .create(HostServicesTableV1::new(Rc::clone(&self.0)), &record)
    }
    fn step(&self, items: &[Item]) -> Vec<u8> {
        let (items, appends): (Vec<_>, Vec<_>) = items
            .iter()
            .map(|item| {
                let rows = item.tokens.len() as u32;
                let end = item.start + u64::from(rows);
                let batch = match item.phase {
                    Phase::Verify => {
                        BatchItemV1::verify(item.request, rows, false, item.tokens.clone())
                    }
                    phase => BatchItemV1::new(
                        item.request,
                        phase == Phase::Decode,
                        rows,
                        true,
                        item.tokens.clone(),
                    ),
                };
                let append =
                    BatchAppendV1::new(item.start, u64::from(rows), end, vec![item.slot], None);
                (batch, append)
            })
            .unzip();
        BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
            .encode()
            .unwrap()
    }
    fn token(&self, position: u64) -> u32 {
        position as u32 + 1
    }
    fn assert_released(&self) {
        assert_eq!(self.0.live_resource_count(), 0);
    }
}

fn keys(warmup: &VerifyWarmup) -> Vec<(usize, usize)> {
    verify_shapes(8, warmup)
        .iter()
        .map(|shape| (shape.row_bucket, shape.item_bucket))
        .collect()
}

/// Eight sequences take decode waves of 1, 2, 4 and 8, a prefill and a
/// decode each. The drafter then runs a prefill and a drafting step for
/// each of the host's verify keys and each context-ladder prompt.
#[test]
fn every_profile_creates_describes_and_runs_the_hosts_warmup() {
    let coverage = serving_startup::<Adapter<NONE>>();
    let [ar, mtp] = coverage.as_slice() else {
        panic!("{coverage:?}")
    };
    assert_eq!((ar.waves.as_slice(), ar.steps), (&[1, 2, 4, 8][..], 8));
    assert!(ar.verify_keys.is_empty() && ar.contexts.is_empty());
    let warmup = Adapter::<NONE>::servings()[1].verify_warmup().unwrap();
    assert_eq!(
        warmup,
        VerifyWarmup {
            item_rows: vec![2, 4],
            maximum_draft_tokens: 3,
            step_rows: 32,
            drafting_items: None,
            exact_graphs: false,
        }
    );
    assert_eq!(mtp.verify_keys, keys(&warmup));
    // Shallow and mixed steps: one item drafting one, a decode beside it.
    for key in [(2, 1), (4, 2), (32, 8)] {
        assert!(mtp.verify_keys.contains(&key), "{key:?}");
    }
    assert_eq!(mtp.contexts, [1, 3, 6, 12, 24]);
    assert!(!mtp.drafting_limit_refused);
    assert_eq!(mtp.steps, 8 + 2 * (mtp.verify_keys.len() + 5));
}

#[test]
#[should_panic(expected = "tp1-ar: the bundle did not describe its prefix state")]
fn a_bundle_that_cannot_describe_its_prefix_state_fails() {
    serving_startup::<Adapter<DESCRIBE>>();
}

/// The first key, (rows 2, items 1), verifies one draft: the host's
/// shallowest verify item.
#[test]
#[should_panic(expected = "tp1-mtp verify rows 2 items 1: Status(")]
fn a_bundle_that_refuses_a_shallow_verify_item_fails() {
    serving_startup::<Adapter<SHALLOW>>();
}

/// (rows 4, items 2) is the first key whose wave decodes one request beside
/// one that drafts.
#[test]
#[should_panic(expected = "tp1-mtp verify rows 4 items 2: Status(")]
fn a_bundle_that_refuses_a_mixed_warmup_step_fails() {
    serving_startup::<Adapter<MIXED>>();
}

#[test]
#[should_panic(expected = "tp1-mtp context 12: Status(")]
fn a_bundle_that_refuses_a_context_ladder_step_fails() {
    serving_startup::<Adapter<CONTEXT>>();
}

/// A package that limits drafting items: no warmup step exceeds the limit,
/// and the bundle refuses a step that does.
#[test]
fn a_drafting_item_limit_bounds_every_key_and_is_the_bundles_own() {
    let coverage = serving_startup::<Adapter<LIMITED>>();
    let warmup = Adapter::<LIMITED>::servings()[1].verify_warmup().unwrap();
    assert_eq!(warmup.drafting_items, Some(2));
    assert_eq!(coverage[1].verify_keys, keys(&warmup));
    assert!(coverage[1].drafting_limit_refused);
}

#[test]
#[should_panic(
    expected = "the package declares at most 2 drafting items, so the bundle must refuse 3"
)]
fn an_understated_drafting_item_limit_fails() {
    serving_startup::<Adapter<UNDERSTATED>>();
}

#[test]
#[should_panic(expected = "the profile's values")]
fn a_configuration_must_serve_its_profiles_sequences() {
    serving_startup::<Adapter<MISDECLARED>>();
}

/// A package that provides images serves text and image by default, as the
/// host's `auto` does; a selection serves what it lists.
#[test]
fn a_package_serves_the_hosts_default_modalities() {
    let coverage = serving_startup::<Adapter<IMAGES>>();
    for served in &coverage {
        assert_eq!(served.modalities, ["text", "image"], "{}", served.profile);
    }
    assert_eq!(serving_startup::<Adapter<NONE>>()[0].modalities, ["text"]);
    let mut serving = Adapter::<IMAGES>::servings().remove(0);
    serving
        .provides
        .extend(["video".to_owned(), "audio".to_owned()]);
    assert_eq!(serving.modalities(), ["text", "image"]);
    serving.selection = Some(vec!["text".to_owned()]);
    assert_eq!(serving.modalities(), ["text"]);
    assert!(!serving.serves_images());
}

/// The host creates an image-serving profile with the vision encoder's
/// components, so a bundle that cannot create them fails at create.
#[test]
#[should_panic(expected = "tp1-ar: create: Status(")]
fn a_bundle_that_refuses_its_encoders_fails_an_image_profile() {
    serving_startup::<Adapter<NO_ENCODER>>();
}

#[test]
#[should_panic(
    expected = "tp1-ar: modalities [\"text\", \"video\"] must be text first and provided"
)]
fn a_selection_serves_only_what_the_package_provides() {
    serving_startup::<Adapter<UNPROVIDED>>();
}

#[test]
fn verify_rows_follow_the_hosts_rule_for_the_package() {
    let mut serving = Adapter::<NONE>::servings().pop().unwrap();
    assert_eq!(serving.verify_rows(), 32);
    let package = serving.package.as_mut().unwrap();
    package.maximum_verify_tokens = 256;
    assert_eq!(serving.verify_rows(), 256);
    serving.package.as_mut().unwrap().narrow_rows = true;
    assert_eq!(serving.verify_rows(), 32);
    serving.sequences = 33;
    assert_eq!(serving.verify_rows(), 256);
    serving.draft_tokens = 0;
    assert_eq!((serving.verify_rows(), serving.verify_warmup()), (0, None));
}

const STARTUP_TESTS: [(&str, &str); 2] = [("tp1_ar", "tp1-ar"), ("tp1_mtp", "tp1-mtp")];

/// A suite's per-profile tests together run what the whole startup runs.
#[test]
fn each_profile_starts_alone_as_in_the_whole_startup() {
    startup_profiles::<Adapter<NONE>>(&STARTUP_TESTS);
    let alone: Vec<Coverage> = STARTUP_TESTS
        .iter()
        .map(|&(_, profile)| serving_startup_of::<Adapter<NONE>>(profile))
        .collect();
    assert_eq!(alone, serving_startup::<Adapter<NONE>>());
}

#[test]
#[should_panic(expected = "the suite's startup tests: one per profile")]
fn a_suite_must_start_every_profile() {
    startup_profiles::<Adapter<NONE>>(&STARTUP_TESTS[..1]);
}

#[test]
#[should_panic(expected = "the suite's startup tests: one per profile")]
fn a_suite_must_start_each_profile_once() {
    startup_profiles::<Adapter<NONE>>(&[
        ("tp1_ar", "tp1-ar"),
        ("tp1_mtp", "tp1-mtp"),
        ("tp1_ar_again", "tp1-ar"),
    ]);
}

#[test]
#[should_panic(expected = "tp1-mtp: its startup test's name")]
fn a_startup_test_is_named_after_its_profile() {
    startup_profiles::<Adapter<NONE>>(&[("tp1_ar", "tp1-ar"), ("tp1_ar_mtp", "tp1-mtp")]);
}

#[test]
#[should_panic(expected = "the profile's values")]
fn the_profile_list_checks_each_configuration() {
    startup_profiles::<Adapter<MISDECLARED>>(&STARTUP_TESTS);
}

#[test]
#[should_panic(expected = "tp1-moe: no serving configuration")]
fn a_startup_test_needs_its_profiles_configuration() {
    serving_startup_of::<Adapter<NONE>>("tp1-moe");
}

#[test]
#[should_panic(expected = "the profile's values")]
fn a_startup_test_checks_its_configuration() {
    serving_startup_of::<Adapter<MISDECLARED>>("tp1-ar");
}
