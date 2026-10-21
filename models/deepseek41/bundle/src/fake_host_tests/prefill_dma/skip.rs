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
    capture_with(rows, enabled, false)
}

/// `dspark` creates a drafter instance through the mapped Engram host, so
/// the prefill is a DSpark target chunk with its feature captures.
fn capture_with(rows: u64, enabled: Option<bool>, dspark: bool) -> Vec<String> {
    let host = host();
    if dspark {
        host.drafter.set(true);
        host.speculation.set(Some((5, 32)));
    }
    host.device.residency.set(Some(BundleRawClusterResidencyV1 {
        multiprocessor_count: 188,
        usable_multiprocessors: 188,
        max_active_clusters: 188,
        flags: BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED
            | BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH,
        ..BundleRawClusterResidencyV1::unreported()
    }));
    let mut instance = if dspark {
        let services = dspark::host::MappedHost::with_peer_copies(Rc::clone(&host));
        create_with_services(&host, HostServicesTableV1::new(Rc::new(services)))
    } else {
        create_sparse(&host)
    };
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

/// A DSpark target chunk takes the same ordered transport: every compute
/// launch, feature capture and commit keeps its operands, and only the
/// phased reductions become owner sums and ordered peer copies.
#[test]
fn sparse_dma_serves_dspark_target_prefill() {
    for rows in [8192, 8191] {
        let reference = capture_with(rows, Some(false), true);
        let candidate = capture_with(rows, None, true);
        assert!(
            reference
                .iter()
                .any(|line| line.starts_with("launch ") && line.contains("dspark"))
        );
        let phased = |journal: &[String]| {
            journal
                .iter()
                .filter(|line| {
                    line.starts_with("launch ")
                        && line.contains("chop_comm_allreduce_phased_round_f32_v1")
                })
                .count()
        };
        let reductions = phased(&reference) / 4;
        assert_eq!(phased(&reference), reductions * 4);
        assert_eq!(reductions, 73);
        assert!(!reference.iter().any(|line| line.starts_with("peer-copy ")));
        let before = compute(&reference);
        let after = compute(&candidate);
        assert_eq!(before.len(), after.len());
        for (index, (before, after)) in before.iter().zip(&after).enumerate() {
            assert_eq!(before, after, "compute dispatch {index}");
        }
        assert_eq!(phased(&candidate), 0);
        assert_eq!(
            candidate
                .iter()
                .filter(|line| line.starts_with("peer-copy "))
                .count(),
            reductions * 24
        );
        assert_eq!(audit(&candidate).unwrap(), reductions * (24 * 2 + 4 * 3));
        let missing = candidate
            .into_iter()
            .filter(|line| !line.starts_with("wait "))
            .collect::<Vec<_>>();
        assert!(audit(&missing).is_err());
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

/// The chain's expand, mix and norm-ceil; the MoE-input norm-ceil that also
/// certifies each row's router operands takes the plain one's place.
const CONSUMERS: [&str; 3] = [
    "chop_sinkhorn_hc_deepseek41_expand_f32_v1",
    "chop_sinkhorn_hc_deepseek41_prefill_mix_pipeline_r8_c4_v1",
    "chop_prefill_norm_ceil_deepseek41_cached_v1",
];
const CERTIFIED_NORM: &str = "chop_prefill_norm_ceil_deepseek41_cached_certified_v1";

fn consumer(line: &str) -> bool {
    line.starts_with("launch ")
        && CONSUMERS
            .iter()
            .chain([&CERTIFIED_NORM])
            .any(|s| line.contains(&format!(" {s} ")))
}

fn stream(line: &str) -> &str {
    line.split_whitespace().nth(1).unwrap()
}

fn capture_consumed(rows: u64, consume: bool) -> Vec<String> {
    crate::ops::with_partial_setting(Some("prefill"), || {
        crate::ops::with_dma_consume(consume, || capture(rows, None))
    })
}

/// Walk each stream's consumer launches in submission order: an unsplit
/// launch is unchanged; a split chain is four chains over the owner
/// intervals, the rank's own first. Returns the split chains and how many
/// of them end in the certifying norm-ceil.
fn split_chains(serial: &[String], pipelined: &[String]) -> (usize, usize) {
    let spans = super::super::shared_overlap::journal_spans;
    let streams = serial
        .iter()
        .filter(|l| consumer(l))
        .map(|l| stream(l))
        .collect::<BTreeSet<_>>();
    let (mut chains, mut certified) = (0, 0);
    for st in streams {
        let s = serial
            .iter()
            .filter(|l| consumer(l) && stream(l) == st)
            .collect::<Vec<_>>();
        let p = pipelined
            .iter()
            .filter(|l| consumer(l) && stream(l) == st)
            .collect::<Vec<_>>();
        let (mut i, mut j) = (0, 0);
        while i < s.len() {
            if p.get(j) == Some(&s[i]) {
                i += 1;
                j += 1;
                continue;
            }
            let chain = &s[i..i + 3];
            let parts = &p[j..j + 12];
            let mut offsets = vec![BTreeSet::new(); 3];
            for (k, part) in parts.iter().enumerate() {
                let original = chain[k % 3];
                let norm = if chain[2].contains(&format!(" {CERTIFIED_NORM} ")) {
                    CERTIFIED_NORM
                } else {
                    CONSUMERS[2]
                };
                let kind = [CONSUMERS[0], CONSUMERS[1], norm][k % 3];
                assert!(part.contains(&format!(" {kind} ")), "{part}");
                let (whole, piece) = (spans(original), spans(part));
                assert_eq!(whole.len(), piece.len());
                for (position, (w, q)) in whole.iter().zip(&piece).enumerate() {
                    if w == q {
                        continue;
                    }
                    if w.2 == 4 && q.2 == 4 {
                        // The live count cell.
                        continue;
                    }
                    assert_eq!((q.0, q.2 * 4), (w.0, w.2), "{part}");
                    assert!(q.1 >= w.1 && q.1 + q.2 <= w.1 + w.2);
                    offsets[k % 3].insert((position, (q.1 - w.1) / q.2));
                }
            }
            // Every row argument is cut into the four owner quarters.
            for set in &offsets {
                let mut per = BTreeMap::<usize, BTreeSet<u64>>::new();
                for &(position, quarter) in set {
                    per.entry(position).or_default().insert(quarter);
                }
                assert!(!per.is_empty());
                assert!(per.values().all(|q| q.len() == 4));
            }
            chains += 1;
            certified += usize::from(chain[2].contains(&format!(" {CERTIFIED_NORM} ")));
            i += 3;
            j += 12;
        }
        assert_eq!(j, p.len());
    }
    (chains, certified)
}

#[test]
fn dma_consumes_each_gathered_interval_as_it_lands() {
    let serial = capture_consumed(8192, false);
    let pipelined = capture_consumed(8192, true);
    let others = |j: &[String]| {
        compute(j)
            .into_iter()
            .filter(|l| !consumer(l))
            .cloned()
            .collect::<Vec<_>>()
    };
    assert_eq!(others(&serial), others(&pipelined));
    // The copies and owner sums are the call's in either case.
    let transport = |j: &[String]| {
        j.iter()
            .filter(|l| l.starts_with("peer-copy ") || transport_launch(l))
            .count()
    };
    assert_eq!(transport(&serial), transport(&pipelined));
    // On every rank: the 38 full-row calls followed by the chain (two
    // others go to the dense quantizer) and the 2240- and 2496-row cone
    // calls, whose mix is the r8 c4 pipeline. The 20 full-row chains before
    // MoE end in the norm-ceil that certifies the router's operands.
    assert_eq!(split_chains(&serial, &pipelined), (40 * 4, 20 * 4));
    audit(&pipelined).unwrap();
    // Without the waits before each interval's consumers, an expand reads
    // reduced rows its gather copy may still be writing.
    let mut early = Vec::new();
    for (at, line) in pipelined.iter().enumerate() {
        let next = pipelined[at + 1..].iter().find(|l| !l.starts_with("wait "));
        if line.starts_with("wait ")
            && next.is_some_and(|n| consumer(n) && stream(n) == stream(line))
        {
            continue;
        }
        early.push(line.clone());
    }
    assert!(early.len() < pipelined.len());
    assert!(audit(&early).is_err());
}

#[test]
fn dma_keeps_consumers_after_calls_it_cannot_split() {
    // 8191-row calls do not divide into whole eight-row mix tiles per owner.
    assert_eq!(capture_consumed(8191, true), capture_consumed(8191, false));
}
