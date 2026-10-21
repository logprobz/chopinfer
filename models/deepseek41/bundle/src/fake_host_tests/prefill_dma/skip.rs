use super::*;
use crate::ops::with_bf16_gather;
use runtime_bundle_sdk::prefill::{Boundary, encode};

fn create_sparse(host: &Rc<Host>) -> BundleInstanceV1 {
    create_with_services(host, HostServicesTableV1::new(Rc::clone(host)))
}

fn create_with_services(host: &Rc<Host>, services: HostServicesTableV1) -> BundleInstanceV1 {
    let mut record = crate::config::tests::record(&host.shape, host.kv, 4, false);
    let at = record
        .windows(MODEL_OPTIONS_TAG_V1.len())
        .position(|w| w == MODEL_OPTIONS_TAG_V1)
        .unwrap();
    let mut options: serde_json::Value =
        serde_json::from_slice(&record[at + MODEL_OPTIONS_TAG_V1.len() + 4..]).unwrap();
    options["prefill_chunk_tokens"] = 8192.into();
    options["prefill_upper_skip"] = true.into();
    let options = serde_json::to_vec(&options).unwrap();
    record.truncate(at + MODEL_OPTIONS_TAG_V1.len());
    record.extend_from_slice(&(options.len() as u32).to_le_bytes());
    record.extend_from_slice(&options);
    with_bf16_gather(true, || {
        with_dma(true, || {
            with_mode(0, || {
                api()
                    .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
                    .unwrap()
                    .create(services, &record)
                    .unwrap()
            })
        })
    })
}

fn capture(rows: u64, enabled: Option<bool>) -> Vec<String> {
    let host = host();
    host.device.residency.set(Some(BundleRawClusterResidencyV1 {
        multiprocessor_count: 188,
        usable_multiprocessors: 188,
        max_active_clusters: 188,
        flags: BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED
            | BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH,
        ..BundleRawClusterResidencyV1::unreported()
    }));
    let mut instance = create_sparse(&host);
    assert_eq!(instance.prefill_policy().unwrap().chunk_tokens, 8192);
    host.device.journal.borrow_mut().clear();
    let wire = encode(
        &prompt(7, 0, 0, rows, true),
        &[Boundary {
            request_id: 7,
            prompt_tokens: rows,
            chunk_end: rows,
            chunk_tokens: 8192,
        }],
    )
    .unwrap();
    let mut run = || {
        instance
            .execute_prefill(&wire, &mut Views::default())
            .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
    };
    if let Some(enabled) = enabled {
        crate::instance::execute::skip::with_dma(enabled, run);
    } else {
        run();
    }
    instance.commit().unwrap();
    let journal = host.device.journal.borrow().clone();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    journal
}

fn compute(journal: &[String]) -> Vec<&String> {
    journal
        .iter()
        .filter(|line| {
            line.starts_with("launch ")
                && !line.contains("chop_comm_")
                && !line.contains("chop_row_rounding_round_")
        })
        .collect()
}

#[test]
fn sparse_dma_preserves_compute_and_orders_only_the_required_rows() {
    for rows in [8192, 8191] {
        let reference = capture(rows, Some(false));
        let candidate = capture(rows, None);
        assert_eq!(
            reference
                .iter()
                .filter(|line| line.starts_with("launch ")
                    && line.contains("chop_comm_allreduce_phased_round_f32_v1"))
                .count(),
            73 * 4
        );
        assert!(!reference.iter().any(|line| line.starts_with("peer-copy ")));
        let before = compute(&reference);
        let after = compute(&candidate);
        assert_eq!(before.len(), after.len());
        for (index, (before, after)) in before.iter().zip(&after).enumerate() {
            assert_eq!(before, after, "compute dispatch {index}");
        }
        let copies = candidate
            .iter()
            .filter(|line| line.starts_with("peer-copy "))
            .collect::<Vec<_>>();
        assert_eq!(copies.len(), 73 * 24);
        let upper_rows: u64 = (0..16)
            .map(|layer| 2496 - 128 * layer - (8192 - rows))
            .sum();
        let reduced_rows = 41 * rows + 2 * upper_rows;
        let bytes: u64 = copies
            .iter()
            .map(|line| super::super::shared_overlap::journal_spans(line)[0].2)
            .sum();
        assert_eq!(bytes, reduced_rows * u64::from(FLASH.shape.hidden) * 18);
        assert_eq!(
            candidate
                .iter()
                .filter(|line| line.starts_with("launch ")
                    && line.contains("chop_comm_allreduce_phased_round_f32_v1"))
                .count(),
            0
        );
        assert_eq!(
            candidate
                .iter()
                .filter(|line| line.starts_with("launch ")
                    && line.contains("chop_comm_reduce_scatter_f32_v1"))
                .count(),
            8 * 4
        );
        assert_eq!(audit(&candidate).unwrap(), 73 * (24 * 2 + 4 * 3));
        let missing = candidate
            .into_iter()
            .filter(|line| !line.starts_with("wait "))
            .collect::<Vec<_>>();
        assert!(
            audit(&missing).is_err(),
            "missing peer joins must expose the alias"
        );
    }
}

#[test]
fn sparse_dma_keeps_small_cones_on_the_existing_transport() {
    let journal = capture(511, None);
    assert!(
        !journal
            .iter()
            .any(|line| line.starts_with("peer-copy ") || line.contains(SUM))
    );
}

#[test]
fn sparse_dma_finishes_mapped_uploads_before_compute() {
    let host = host();
    host.drafter.set(true);
    host.device.residency.set(Some(BundleRawClusterResidencyV1 {
        multiprocessor_count: 188,
        usable_multiprocessors: 188,
        max_active_clusters: 188,
        flags: BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED
            | BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH,
        ..BundleRawClusterResidencyV1::unreported()
    }));
    let mut instance = crate::instance::prefill_engram::with_enabled(true, || {
        create_with_services(
            &host,
            HostServicesTableV1::new(Rc::new(dspark::host::MappedHost::new(Rc::clone(&host)))),
        )
    });
    assert_eq!(
        crate::instance::prefill_engram::indices(instance.handle()).len(),
        2
    );
    host.journal.borrow_mut().clear();
    let request = encode(
        &prompt(7, 0, 0, 8192, true),
        &[Boundary {
            request_id: 7,
            prompt_tokens: 8192,
            chunk_end: 8192,
            chunk_tokens: 8192,
        }],
    )
    .unwrap();
    instance
        .execute_prefill(&request, &mut Views::default())
        .unwrap();
    instance.commit().unwrap();
    let journal = host.journal.borrow();
    assert!(
        !journal
            .iter()
            .any(|line| line.starts_with("async mapped ids "))
    );
    let first = journal
        .iter()
        .position(|line| line.starts_with("launch "))
        .unwrap();
    assert_eq!(
        journal[..first]
            .iter()
            .filter(|line| line.starts_with("mapped gather "))
            .count(),
        16
    );
    assert_eq!(
        journal
            .iter()
            .filter(|line| line.starts_with("launch ") && line.contains(SUM))
            .count(),
        73 * 4
    );
    drop(journal);
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn sparse_dma_carries_bf16_partials_where_the_cone_reduces_them() {
    use crate::ops::with_partial_setting;
    let phased = |journal: &[String]| {
        journal
            .iter()
            .filter(|line| {
                line.starts_with("launch ")
                    && (line.contains(" chop_comm_allreduce_phased_bf16_v1 ")
                        || line.contains(" chop_comm_allreduce_phased_turns_bf16_v1 "))
            })
            .count()
    };
    let count = |journal: &[String], symbol: &str| {
        journal
            .iter()
            .filter(|line| line.starts_with("launch ") && line.contains(&format!(" {symbol} ")))
            .count()
    };
    for rows in [8192, 8191] {
        let (reference, candidate) = with_partial_setting(Some("prefill"), || {
            (capture(rows, Some(false)), capture(rows, None))
        });
        assert!(!reference.iter().any(|line| line.starts_with("peer-copy ")));
        assert_eq!(phased(&reference) - phased(&candidate), 73 * 4);
        assert_eq!(count(&candidate, SUM), 0);
        assert_eq!(count(&candidate, BF16_SUM), 73 * 4);
        assert_eq!(count(&candidate, BF16_ROUND), 73 * 6);
        let before = compute(&reference);
        let after = compute(&candidate);
        assert_eq!(before, after);
        let copies = candidate
            .iter()
            .filter(|line| line.starts_with("peer-copy "))
            .collect::<Vec<_>>();
        assert_eq!(copies.len(), 73 * 24);
        let upper_rows: u64 = (0..16)
            .map(|layer| 2496 - 128 * layer - (8192 - rows))
            .sum();
        let reduced_rows = 41 * rows + 2 * upper_rows;
        let bytes: u64 = copies
            .iter()
            .map(|line| super::super::shared_overlap::journal_spans(line)[0].2)
            .sum();
        assert_eq!(bytes, reduced_rows * u64::from(FLASH.shape.hidden) * 12);
        assert_eq!(audit(&candidate).unwrap(), 73 * (24 * 2 + 4 * 3));
        let missing = candidate
            .into_iter()
            .filter(|line| !line.starts_with("wait "))
            .collect::<Vec<_>>();
        assert!(audit(&missing).is_err());
    }
}
