use super::*;
use bundle_runtime::fake_host::RecordingSink;
use model_deepseek41_contract::prefill_profile::{self, CHUNK, Profile, SKIP};
use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1};
use runtime_bundle_sdk::prefill::{Boundary, encode};

fn resolved(profile: Option<&str>) -> BTreeMap<String, serde_json::Value> {
    let profiles: serde_json::Value =
        serde_json::from_str(include_str!("../../../../profiles.json")).unwrap();
    // The profile's prefill options; its other options have their own
    // declarations.
    let overrides = profile.map_or_else(BTreeMap::new, |name| {
        let all: BTreeMap<String, String> =
            serde_json::from_value(profiles["profiles"][name]["model_options"].clone()).unwrap();
        all.into_iter()
            .filter(|(key, _)| [CHUNK, SKIP].contains(&key.as_str()))
            .collect()
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
fn capture(prefill: &BTreeMap<String, serde_json::Value>, rows: u32) -> Vec<String> {
    capture_with(prefill, rows, false)
}

/// `dspark` creates a drafter instance through the mapped Engram host.
fn capture_with(
    prefill: &BTreeMap<String, serde_json::Value>,
    rows: u32,
    dspark: bool,
) -> Vec<String> {
    let host = host();
    if dspark {
        host.drafter.set(true);
        host.speculation.set(Some((5, 32)));
    }
    // Report residency as the served host does, so the BF16 gather's phased
    // reductions are admitted and the upper-skip executor can replace them.
    host.device.residency.set(Some(BundleRawClusterResidencyV1 {
        multiprocessor_count: 188,
        usable_multiprocessors: 188,
        max_active_clusters: 188,
        flags: BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED
            | BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH,
        ..BundleRawClusterResidencyV1::unreported()
    }));
    let record = record(&host, prefill);
    let profile = Profile::read(prefill).unwrap();
    let mut instance = api()
        .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
        .unwrap()
        .create(
            if dspark {
                HostServicesTableV1::new(Rc::new(
                    super::super::dspark::host::MappedHost::with_peer_copies(Rc::clone(&host)),
                ))
            } else {
                HostServicesTableV1::new(Rc::clone(&host))
            },
            &record,
        )
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
fn package_defaults_select_dma_for_upper_skip_only() {
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
    // The unset default is the explicit transport on the upper-skip
    // executor: the same streams, events, inbox, copies and sums.
    let unset = capture(&defaults, 8192);
    assert_eq!(
        unset
            .iter()
            .filter(|line| line.starts_with("peer-copy "))
            .count(),
        73 * 24 + 12
    );
    assert_eq!(audit(&unset).unwrap(), 73 * (24 * 2 + 4 * 3) + 12 * 2);
    // A DSpark target prefill under the tp4-dspark defaults takes the same
    // transport and allocates the same inbox, streams and events.
    let dspark = capture_with(&defaults, 8192, true);
    assert_eq!(
        dspark
            .iter()
            .filter(|line| line.starts_with("peer-copy "))
            .count(),
        73 * 24 + 12
    );
    assert_eq!(audit(&dspark).unwrap(), 73 * (24 * 2 + 4 * 3) + 12 * 2);
}

#[test]
fn full_prefill_profile_keeps_the_sm_transport() {
    let full = resolved(Some("tp4-ar-full-prefill"));
    assert_eq!(Profile::read(&full).unwrap(), Profile::default());
    let disabled = capture(&full, 8192);
    assert_disabled_journal(&capture(&full, 8192), &disabled);
    assert_disabled_journal(&capture(&BTreeMap::new(), 8192), &disabled);
}

#[test]
fn package_defaults_send_bf16_partials_where_the_prefill_reduces_them() {
    let defaults = resolved(None);
    let launches = |journal: &[String], symbol: &str| {
        journal
            .iter()
            .filter(|line| line.starts_with("launch ") && line.contains(&format!(" {symbol} ")))
            .count()
    };
    for dspark in [false, true] {
        let unset = capture_with(&defaults, 8192, dspark);
        assert_eq!(launches(&unset, BF16_SUM), 73 * 4);
        assert_eq!(launches(&unset, BF16_ROUND), (73 + 36) * 6);
        assert_eq!(launches(&unset, SUM), 0);
        assert_eq!(
            unset
                .iter()
                .filter(|line| line.starts_with("peer-copy "))
                .count(),
            73 * 24 + 12
        );
        assert_eq!(audit(&unset).unwrap(), 73 * (24 * 2 + 4 * 3) + 12 * 2);
    }
}
