//! Serving startup: each profile's configuration creates with the
//! modalities the host serves by default (so with their encoders), describes
//! its prefix state as the host does before it starts the generation worker,
//! and runs the host's startup warmup at the bundle level: its decode waves,
//! one step per verify key, and its context ladder.

use std::collections::BTreeMap;

use runtime_bundle_abi::{BundleAbiError, BundleInstanceV1, BundleKeptRowsV1, BundleStatusV1};
use runtime_bundle_sdk::modalities::auto_served;
use runtime_bundle_sdk::warmup::{
    VerifyWarmup, context_ladder, item_rows, narrows_rows, served_verify_rows, verify_shapes, waves,
};

use super::super::{RecordingSink, SinkLimits};
use super::kept;

/// Prompt tokens of every warmup request but the context ladder's.
const PROMPT_TOKENS: u64 = 5;

/// One serving configuration: a profile of the model's profiles.json, with
/// the package declaration the host reads for it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Serving {
    /// The profile's name in profiles.json.
    pub profile: String,
    pub tensor_parallel: u32,
    /// The profile's `context_parallel_size`, 1 when it declares none.
    pub context_parallel: u32,
    /// Served sequences, the profile's `max_sequences`.
    pub sequences: u32,
    /// The profile's `speculation.drafter`: "none" without speculation.
    pub drafter: String,
    /// Draft tokens of one verify item, the profile's `speculation.tokens`.
    pub draft_tokens: u32,
    /// The profile's `prefix_cache`: the host then describes prefix state.
    pub prefix_cache: bool,
    /// The profile's `model_options`.
    pub model_options: BTreeMap<String, String>,
    /// Modalities the package's checkpoint provides: "text", then those its
    /// encoders serve (the package's media inputs).
    pub provides: Vec<String>,
    /// The `--modalities` list the configuration serves, `None` for the
    /// host's default (`auto`), which every profile serves.
    pub selection: Option<Vec<String>>,
    /// The package's speculation declaration, `None` when it declares none.
    pub package: Option<PackageSpeculation>,
}

/// What a package declares for speculation, from the builders the package
/// itself uses (the manifest's speculation record and capabilities).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackageSpeculation {
    pub maximum_draft_tokens: u32,
    /// The bundle drafts verify rows itself.
    pub bundle_drafter: bool,
    pub maximum_verify_tokens: u32,
    /// Rows one verify item may take; empty when the package declares none.
    pub row_buckets: Vec<u32>,
    /// Most items that draft in one step, when the package limits them.
    pub maximum_draft_items: Option<u32>,
    /// The package declares `chop.speculative-narrow-rows.v1`.
    pub narrow_rows: bool,
}

impl Serving {
    /// The modalities the instance serves, whose encoders its create record
    /// names: the selection's, or those `auto` serves of what the package
    /// provides (`runtime_bundle_sdk::modalities`).
    pub fn modalities(&self) -> Vec<&str> {
        match &self.selection {
            Some(listed) => listed.iter().map(String::as_str).collect(),
            None => auto_served(self.provides.iter().map(String::as_str)),
        }
    }

    /// Whether the instance serves images, so its create record names the
    /// vision encoder's components.
    pub fn serves_images(&self) -> bool {
        self.modalities().contains(&"image")
    }

    /// The verify warmup the host derives from the profile and the package
    /// (runtime-host `speculation_config_v1` and `serve_v1`), `None` when
    /// the profile does not speculate.
    pub fn verify_warmup(&self) -> Option<VerifyWarmup> {
        if self.draft_tokens == 0 {
            return None;
        }
        let profile = &self.profile;
        let package = self
            .package
            .as_ref()
            .unwrap_or_else(|| panic!("{profile}: speculation needs a package declaration"));
        let tokens = self.draft_tokens as usize;
        assert!(
            self.draft_tokens <= package.maximum_draft_tokens,
            "{profile}: draft tokens exceed the package maximum"
        );
        let rows = served_verify_rows(
            package.maximum_verify_tokens as usize,
            self.sequences as usize,
            narrows_rows(package.narrow_rows),
        );
        assert!(
            rows > tokens,
            "{profile}: verify rows must exceed the draft tokens"
        );
        let buckets: Vec<usize> = package
            .row_buckets
            .iter()
            .map(|&rows| rows as usize)
            .collect();
        Some(VerifyWarmup {
            item_rows: item_rows(&buckets, tokens),
            maximum_draft_tokens: tokens,
            step_rows: rows,
            drafting_items: package.maximum_draft_items.map(|items| items as usize),
        })
    }

    /// The verify rows the host reports to the bundle (`query_speculation`),
    /// 0 without speculation.
    pub fn verify_rows(&self) -> u32 {
        self.verify_warmup()
            .map_or(0, |warmup| u32::try_from(warmup.step_rows).unwrap())
    }

    /// Whether each verify item's drafts come from the bundle.
    pub fn bundle_drafts(&self) -> bool {
        self.draft_tokens > 0
            && self
                .package
                .as_ref()
                .is_some_and(|package| package.bundle_drafter)
    }
}

/// The phase of one step item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Prefill,
    Decode,
    /// The bonus token then the item's drafts, one token per row. An
    /// adapter whose bundle drafts sends the bonus token alone.
    Verify,
}

/// One request's item of a step, in its own state slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Item {
    pub request: u64,
    pub slot: u64,
    /// Tokens committed before this item.
    pub start: u64,
    /// One token per row.
    pub tokens: Vec<u32>,
    pub phase: Phase,
}

/// What one serving configuration ran.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Coverage {
    pub profile: String,
    /// The modalities the instance was created with.
    pub modalities: Vec<String>,
    /// Decode wave sizes.
    pub waves: Vec<usize>,
    /// The host's verify keys, (row bucket, item bucket), one step each.
    pub verify_keys: Vec<(usize, usize)>,
    /// The context ladder's prompt lengths, one drafting step each.
    pub contexts: Vec<usize>,
    /// Whether a step of one more drafting item than the package's
    /// maximum ran and was refused.
    pub drafting_limit_refused: bool,
    pub steps: usize,
}

/// A model's serving configurations over its CPU host. Every conformance
/// [`super::Adapter`] names one, and its suite starts each profile once.
pub trait Startup: Sized + 'static {
    type Limits: SinkLimits;

    /// The model's profiles.json.
    fn profiles() -> &'static str;
    /// One configuration per profile.
    fn servings() -> Vec<Serving>;
    /// A host serving `serving`: its served sequences, and the draft tokens
    /// and `Serving::verify_rows` it reports when the profile speculates.
    fn new(serving: &Serving) -> Self;
    /// Creates an instance from the configuration the host passes, which
    /// names the components of the encoders `Serving::modalities` needs.
    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError>;
    /// The encoded step of `items`, with the KV pages each slot needs.
    fn step(&self, items: &[Item]) -> Vec<u8>;
    /// The token at `position` of every warmup request.
    fn token(&self, position: u64) -> u32;
    /// Check all successful acquisitions, including non-buffer resources.
    fn assert_released(&self);
}

/// The profiles of a profiles.json, as servings without the host's values.
fn profiles(json: &str) -> BTreeMap<String, Serving> {
    let document: serde_json::Value = serde_json::from_str(json).expect("profiles.json");
    let number = |value: &serde_json::Value, name: &str| {
        u32::try_from(value.as_u64().unwrap_or_else(|| panic!("{name}"))).unwrap()
    };
    document["profiles"]
        .as_object()
        .expect("profiles")
        .iter()
        .map(|(name, profile)| {
            let speculation = &profile["speculation"];
            let serving = Serving {
                profile: name.clone(),
                tensor_parallel: number(&profile["tensor_parallel"], "tensor_parallel"),
                context_parallel: profile
                    .get("context_parallel_size")
                    .map_or(1, |size| number(size, "context_parallel_size")),
                sequences: number(&profile["max_sequences"], "max_sequences"),
                drafter: speculation["drafter"].as_str().expect("drafter").to_owned(),
                draft_tokens: number(&speculation["tokens"], "tokens"),
                prefix_cache: profile["prefix_cache"].as_bool().expect("prefix_cache"),
                model_options: profile
                    .get("model_options")
                    .and_then(serde_json::Value::as_object)
                    .map(|options| {
                        options
                            .iter()
                            .map(|(key, value)| {
                                (key.clone(), value.as_str().expect("option").to_owned())
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                provides: Vec::new(),
                selection: None,
                package: None,
            };
            (name.clone(), serving)
        })
        .collect()
}

fn execute<S: Startup>(
    adapter: &S,
    instance: &mut BundleInstanceV1,
    items: &[Item],
    label: &str,
) -> RecordingSink<S::Limits> {
    let mut sink = RecordingSink::default();
    instance
        .execute(&adapter.step(items), &mut sink)
        .unwrap_or_else(|error| panic!("{label}: {error:?}"));
    sink
}

/// One instance's warmup requests: fresh request ids in slots from 0.
struct Requests<'a, S> {
    adapter: &'a S,
    next: u64,
    steps: usize,
}

impl<S: Startup> Requests<'_, S> {
    fn tokens(&self, from: u64, count: u64) -> Vec<u32> {
        (from..from + count)
            .map(|position| self.adapter.token(position))
            .collect()
    }

    /// `count` new requests, each a request id and its slot.
    fn admit(&mut self, count: usize) -> Vec<(u64, u64)> {
        let requests = (0..count as u64)
            .map(|slot| (self.next + slot, slot))
            .collect();
        self.next += count as u64;
        requests
    }

    fn item(&self, (request, slot): (u64, u64), start: u64, rows: u64, phase: Phase) -> Item {
        Item {
            request,
            slot,
            start,
            tokens: self.tokens(start, rows),
            phase,
        }
    }

    /// One prefill step of `prompt` tokens for every request, committed.
    fn prefill(
        &mut self,
        instance: &mut BundleInstanceV1,
        requests: &[(u64, u64)],
        prompt: u64,
        label: &str,
    ) {
        let items: Vec<Item> = requests
            .iter()
            .map(|&request| self.item(request, 0, prompt, Phase::Prefill))
            .collect();
        execute(self.adapter, instance, &items, &format!("{label} prefill"));
        instance
            .commit()
            .unwrap_or_else(|error| panic!("{label}: commit prefill: {error:?}"));
        self.steps += 1;
    }

    /// The step after a prefill of `prompt` tokens: each request with zero
    /// drafts decodes, every other verifies its drafts; all rows are kept.
    fn speculate(
        &mut self,
        instance: &mut BundleInstanceV1,
        requests: &[(u64, u64)],
        drafts: &[usize],
        prompt: u64,
        bundle_drafts: bool,
        label: &str,
    ) {
        let items: Vec<Item> = requests
            .iter()
            .zip(drafts)
            .map(|(&request, &drafts)| match drafts {
                0 => self.item(request, prompt, 1, Phase::Decode),
                drafts => self.item(request, prompt, drafts as u64 + 1, Phase::Verify),
            })
            .collect();
        let sink = execute(self.adapter, instance, &items, label);
        let verifying: Vec<&Item> = items
            .iter()
            .filter(|item| item.phase == Phase::Verify)
            .collect();
        if bundle_drafts {
            let drafted: Vec<u64> = sink.draft_tokens().iter().map(|(id, _)| *id).collect();
            let expected: Vec<u64> = verifying.iter().map(|item| item.request).collect();
            assert_eq!(drafted, expected, "{label}: drafts of each verify item");
        }
        let accepted: Vec<BundleKeptRowsV1> = verifying
            .iter()
            .map(|item| kept(item.request, item.tokens.len() as u32))
            .collect();
        instance
            .commit_accepted(&accepted)
            .unwrap_or_else(|error| panic!("{label}: commit verify: {error:?}"));
        self.steps += 1;
    }

    fn retire(instance: &mut BundleInstanceV1, requests: &[(u64, u64)], label: &str) {
        for &(id, _) in requests {
            instance
                .retire_request(id)
                .unwrap_or_else(|error| panic!("{label}: retire {id}: {error:?}"));
        }
    }
}

/// One serving configuration through create, the prefix description and
/// the host's startup warmup.
fn serve<S: Startup>(serving: &Serving) -> Coverage {
    let profile = &serving.profile;
    let adapter = S::new(serving);
    let mut instance = adapter
        .create()
        .unwrap_or_else(|error| panic!("{profile}: create: {error:?}"));
    // The host describes prefix state when it caches prefixes
    // (runtime-host `command.rs`, before the generation worker starts).
    if serving.prefix_cache {
        instance.describe_prefix_state().unwrap_or_else(|error| {
            panic!("{profile}: the bundle did not describe its prefix state: {error}")
        });
    }
    let mut coverage = Coverage {
        profile: profile.clone(),
        modalities: serving
            .modalities()
            .into_iter()
            .map(str::to_owned)
            .collect(),
        ..Coverage::default()
    };
    let mut requests = Requests {
        adapter: &adapter,
        next: 1,
        steps: 0,
    };
    let sequences = serving.sequences as usize;
    // Decode waves: with speculation they pin zero drafts.
    for wave in waves(sequences) {
        let label = format!("{profile} wave {wave}");
        let admitted = requests.admit(wave);
        requests.prefill(&mut instance, &admitted, PROMPT_TOKENS, &label);
        let decode: Vec<Item> = admitted
            .iter()
            .map(|&request| requests.item(request, PROMPT_TOKENS, 1, Phase::Decode))
            .collect();
        execute(&adapter, &mut instance, &decode, &format!("{label} decode"));
        instance
            .commit()
            .unwrap_or_else(|error| panic!("{label}: commit decode: {error:?}"));
        requests.steps += 1;
        Requests::<S>::retire(&mut instance, &admitted, &label);
        coverage.waves.push(wave);
    }
    if let Some(verify) = serving.verify_warmup() {
        let bundle_drafts = serving.bundle_drafts();
        // One wave per verify key: each request pins its drafts, so the
        // step after the prefill reaches the key (runtime-host `verify_waves`).
        for shape in verify_shapes(sequences, &verify) {
            let label = format!(
                "{profile} verify rows {} items {}",
                shape.row_bucket, shape.item_bucket
            );
            let admitted = requests.admit(shape.drafts.len());
            requests.prefill(&mut instance, &admitted, PROMPT_TOKENS, &label);
            requests.speculate(
                &mut instance,
                &admitted,
                &shape.drafts,
                PROMPT_TOKENS,
                bundle_drafts,
                &label,
            );
            Requests::<S>::retire(&mut instance, &admitted, &label);
            coverage
                .verify_keys
                .push((shape.row_bucket, shape.item_bucket));
        }
        // The context ladder: one request whose first drafting step, at the
        // deepest pin, catches the drafter up on its prompt. Every served
        // context exceeds the ladder's longest prompt.
        let pin = verify
            .item_rows
            .iter()
            .filter(|&&rows| rows > 1 && rows - 1 <= verify.maximum_draft_tokens)
            .max()
            .map(|rows| rows - 1);
        if let Some(pin) = pin {
            for prompt in context_ladder(verify.step_rows, usize::MAX) {
                let label = format!("{profile} context {prompt}");
                let admitted = requests.admit(1);
                requests.prefill(&mut instance, &admitted, prompt as u64, &label);
                requests.speculate(
                    &mut instance,
                    &admitted,
                    &[pin],
                    prompt as u64,
                    bundle_drafts,
                    &label,
                );
                Requests::<S>::retire(&mut instance, &admitted, &label);
                coverage.contexts.push(prompt);
            }
        }
        coverage.drafting_limit_refused =
            refuse_excess_drafting(&mut requests, &mut instance, serving, &verify);
    }
    coverage.steps = requests.steps;
    instance
        .destroy()
        .unwrap_or_else(|error| panic!("{profile}: destroy: {error:?}"));
    adapter.assert_released();
    coverage
}

/// When a step could hold more drafting items than the package's maximum,
/// the bundle must refuse one with an item more, so the declared maximum
/// is the bundle's own. Returns whether such a step ran.
fn refuse_excess_drafting<S: Startup>(
    requests: &mut Requests<'_, S>,
    instance: &mut BundleInstanceV1,
    serving: &Serving,
    verify: &VerifyWarmup,
) -> bool {
    let Some(limit) = verify.drafting_items else {
        return false;
    };
    let Some(drafts) = verify
        .item_rows
        .iter()
        .filter(|&&rows| rows > 1 && rows - 1 <= verify.maximum_draft_tokens)
        .min()
        .map(|rows| rows - 1)
    else {
        return false;
    };
    let over = limit + 1;
    if over > serving.sequences as usize || over * (drafts + 1) > verify.step_rows {
        return false;
    }
    let label = format!("{} {over} drafting items", serving.profile);
    let admitted = requests.admit(over);
    requests.prefill(instance, &admitted, PROMPT_TOKENS, &label);
    let items: Vec<Item> = admitted
        .iter()
        .map(|&request| requests.item(request, PROMPT_TOKENS, drafts as u64 + 1, Phase::Verify))
        .collect();
    let mut sink = RecordingSink::<S::Limits>::default();
    let result = instance.execute(&requests.adapter.step(&items), &mut sink);
    assert_eq!(
        result,
        Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument)),
        "{label}: the package declares at most {limit} drafting items, so the bundle must refuse {over}"
    );
    Requests::<S>::retire(instance, &admitted, &label);
    true
}

/// The adapter's serving configuration of `profile`: the only one it
/// declares, with the profile's values, serving text and only modalities
/// the package provides.
fn serving<S: Startup>(profiles: &BTreeMap<String, Serving>, profile: &str) -> Serving {
    let mut servings = S::servings().into_iter().filter(|s| s.profile == profile);
    let serving = servings
        .next()
        .unwrap_or_else(|| panic!("{profile}: no serving configuration"));
    assert!(
        servings.next().is_none(),
        "{profile}: one serving configuration per profile"
    );
    let declared = profiles
        .get(profile)
        .unwrap_or_else(|| panic!("{profile}: not a profile of profiles.json"));
    let expected = Serving {
        provides: serving.provides.clone(),
        selection: serving.selection.clone(),
        package: serving.package.clone(),
        ..declared.clone()
    };
    assert_eq!(serving, expected, "the profile's values");
    assert_eq!(
        serving.provides.first().map(String::as_str),
        Some("text"),
        "{profile}: the package provides text first"
    );
    let modalities = serving.modalities();
    assert!(
        modalities.first() == Some(&"text")
            && modalities
                .iter()
                .all(|modality| serving.provides.iter().any(|p| p == modality)),
        "{profile}: modalities {modalities:?} must be text first and provided by the package {:?}",
        serving.provides
    );
    serving
}

/// Checks that the adapter declares one serving configuration per profile,
/// each with its profile's values. Returns them in the adapter's order.
fn servings<S: Startup>() -> Vec<Serving> {
    let profiles = profiles(S::profiles());
    let servings = S::servings();
    let mut declared: Vec<&str> = servings.iter().map(|s| s.profile.as_str()).collect();
    declared.sort_unstable();
    assert_eq!(
        declared,
        profiles.keys().map(String::as_str).collect::<Vec<_>>(),
        "one serving configuration per profile"
    );
    servings
        .iter()
        .map(|serving| self::serving::<S>(&profiles, &serving.profile))
        .collect()
}

/// The configuration of `serving` through create, the prefix description
/// and the host's startup warmup, reported in one line.
fn start<S: Startup>(serving: &Serving) -> Coverage {
    let served = serve::<S>(serving);
    println!(
        "ServingStartup {}: modalities {}, waves {:?}, {} verify keys {:?}, contexts {:?}, drafting limit refused {}, {} steps",
        served.profile,
        served.modalities.join(","),
        served.waves,
        served.verify_keys.len(),
        served.verify_keys,
        served.contexts,
        served.drafting_limit_refused,
        served.steps
    );
    assert!(served.steps > 0, "{}: no startup steps", served.profile);
    served
}

/// Runs every serving configuration the adapter declares, after checking
/// that they are the model's profiles. Returns what each configuration ran.
pub fn serving_startup<S: Startup>() -> Vec<Coverage> {
    servings::<S>().iter().map(start::<S>).collect()
}

/// Runs the serving configuration of one profile. A bundle's
/// `conformance::suite!` runs each profile once, in its own
/// `serving_startup::<profile>` test, beside [`startup_profiles`].
#[doc(hidden)]
pub fn serving_startup_of<S: Startup>(profile: &str) -> Coverage {
    start::<S>(&serving::<S>(&profiles(S::profiles()), profile))
}

/// Checks that the suite's startup tests, `(test, profile)`, and the
/// adapter's serving configurations are each exactly the model's profiles,
/// and that each test is named after its profile, so one suite run starts
/// every profile once. A bundle's `conformance::suite!` runs it in its
/// `serving_startup::profiles` test.
#[doc(hidden)]
pub fn startup_profiles<S: Startup>(tests: &[(&str, &str)]) {
    let servings = servings::<S>();
    let mut declared: Vec<&str> = servings.iter().map(|s| s.profile.as_str()).collect();
    declared.sort_unstable();
    let mut listed: Vec<&str> = tests.iter().map(|&(_, profile)| profile).collect();
    listed.sort_unstable();
    assert_eq!(
        listed, declared,
        "the suite's startup tests: one per profile"
    );
    for &(test, profile) in tests {
        let name: String = profile
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        assert_eq!(test, name, "{profile}: its startup test's name");
    }
}

#[cfg(test)]
mod tests;
