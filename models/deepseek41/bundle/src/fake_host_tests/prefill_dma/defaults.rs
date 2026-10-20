use super::*;
use crate::ops::{with_bf16_gather, with_dma_setting};
use bundle_runtime::fake_host::RecordingSink;
use model_deepseek41_contract::prefill_profile::{self, CHUNK, Profile, SKIP};
use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1};
use runtime_bundle_sdk::prefill::{Boundary, encode};

fn resolved(profile: Option<&str>) -> BTreeMap<String, serde_json::Value> {
    let profiles: serde_json::Value =
        serde_json::from_str(include_str!("../../../../profiles.json")).unwrap();
    let overrides = profile.map_or_else(BTreeMap::new, |name| {
        serde_json::from_value(profiles["profiles"][name]["model_options"].clone()).unwrap()
    });
    ModelOptionsV1 {
        schema: CAPABILITY.into(),
        options: prefill_profile::declarations().into(),
    }
    .resolve(&overrides)
    .unwrap()
}

fn record(host: &Rc<Host>, prefill: &BTreeMap<String, serde_json::Value>) -> Vec<u8> {
    let mut record = crate::config::tests::record(&host.shape, host.kv, 4, false);
    let at = record
        .windows(MODEL_OPTIONS_TAG_V1.len())
        .position(|window| window == MODEL_OPTIONS_TAG_V1)
        .unwrap()
        + MODEL_OPTIONS_TAG_V1.len();
    let mut options: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&record[at + 4..]).unwrap();
    assert!(!options.contains_key(CHUNK) && !options.contains_key(SKIP));
    options.extend(prefill.clone());
    let options = serde_json::to_vec(&options).unwrap();
    record.truncate(at);
    record.extend_from_slice(&(options.len() as u32).to_le_bytes());
    record.extend_from_slice(&options);
    record
}

/// Include creation and release so an unused inbox, stream or event fails
/// the comparison even when the prefill executor never issues a peer copy.
fn capture(
    setting: Option<&str>,
    bf16: bool,
    prefill: &BTreeMap<String, serde_json::Value>,
    rows: u32,
) -> Vec<String> {
    let host = host();
    let record = record(&host, prefill);
    let profile = Profile::read(prefill).unwrap();
    let mut instance = with_bf16_gather(bf16, || {
        with_dma_setting(setting, || {
            with_mode(0, || {
                api()
                    .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
                    .unwrap()
                    .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            })
        })
    })
    .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
    if profile.chunk == 0 {
        step(
            &mut instance,
            &prompt(7, 0, 0, u64::from(rows), true),
            &mut Views::default(),
        );
    } else {
        assert_eq!(
            instance.prefill_policy().unwrap().chunk_tokens,
            profile.chunk
        );
        let request = encode(
            &prompt(7, 0, 0, u64::from(rows), true),
            &[Boundary {
                request_id: 7,
                prompt_tokens: u64::from(rows),
                chunk_end: u64::from(rows),
                chunk_tokens: profile.chunk,
            }],
        )
        .unwrap();
        instance
            .execute_prefill(
                &request,
                &mut RecordingSink::<super::super::conformance::Limits>::default(),
            )
            .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
        instance.commit().unwrap();
    }
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let journal = host.device.journal.borrow().clone();
    assert!(journal.iter().any(|line| line.starts_with("launch ")));
    journal
}

fn assert_disabled_journal(actual: &[String], disabled: &[String]) {
    assert_eq!(actual.len(), disabled.len());
    for (index, (actual, expected)) in actual.iter().zip(disabled).enumerate() {
        assert_eq!(actual, expected, "lifecycle journal entry {index}");
    }
    assert!(
        !actual
            .iter()
            .any(|line| line.starts_with("peer-copy ") || line.contains(SUM))
    );
}

#[test]
fn unset_dma_legacy_create_record_matches_disabled_journals() {
    let legacy = BTreeMap::new();
    assert_eq!(Profile::read(&legacy).unwrap(), Profile::default());
    for bf16 in [true, false] {
        for rows in [8191, 8192] {
            assert_disabled_journal(
                &capture(None, bf16, &legacy, rows),
                &capture(Some("0"), bf16, &legacy, rows),
            );
        }
    }
}

#[test]
fn package_defaults_do_not_allocate_or_execute_dma() {
    let defaults = resolved(None);
    assert_eq!(
        Profile::read(&defaults).unwrap(),
        Profile {
            chunk: 8192,
            skip: true
        }
    );
    for name in ["tp4-ar", "tp4-dspark"] {
        assert_eq!(resolved(Some(name)), defaults, "{name}");
    }
    assert_disabled_journal(
        &capture(None, true, &defaults, 8192),
        &capture(Some("0"), true, &defaults, 8192),
    );
}

#[test]
fn full_prefill_profile_requires_explicit_dma_selection() {
    let full = resolved(Some("tp4-ar-full-prefill"));
    assert_eq!(Profile::read(&full).unwrap(), Profile::default());
    let disabled = capture(Some("0"), true, &full, 8192);
    assert_disabled_journal(&capture(None, true, &full, 8192), &disabled);
    assert_disabled_journal(&capture(None, true, &BTreeMap::new(), 8192), &disabled);
    let explicit = capture(Some("1"), true, &full, 8192);
    assert_eq!(audit(&explicit).unwrap(), 81 * (24 * 2 + 4 * 3));
    assert_eq!(
        explicit
            .iter()
            .filter(|line| line.starts_with("peer-copy "))
            .count(),
        81 * 24
    );
}

#[test]
fn explicit_dma_keeps_the_existing_transport_controls() {
    for bf16 in [true, false] {
        let journal = capture(Some("1"), bf16, &BTreeMap::new(), 8191);
        assert_eq!(audit(&journal).unwrap(), 81 * (24 * 2 + 4 * 3));
        assert_eq!(
            journal
                .iter()
                .filter(|line| line.starts_with("peer-copy "))
                .count(),
            81 * 24
        );
    }
}
