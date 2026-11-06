use super::*;
use runtime_bundle_sdk::prefill::{Boundary, encode};

fn create_sparse(host: &Rc<Host>) -> BundleInstanceV1 {
    create_with_services(host, HostServicesTableV1::new(Rc::clone(host)))
}

pub(super) fn create_with_services(
    host: &Rc<Host>,
    services: HostServicesTableV1,
) -> BundleInstanceV1 {
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
    api()
        .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
        .unwrap()
        .create(services, &record)
        .unwrap()
}

fn capture(rows: u64, enabled: Option<bool>) -> Vec<String> {
    capture_with(rows, enabled, false)
}

/// `dspark` creates a drafter instance through the mapped Engram host, so
/// the prefill is a DSpark target chunk with its feature captures.
fn capture_with(rows: u64, enabled: Option<bool>, dspark: bool) -> Vec<String> {
    capture_on(&host(), rows, enabled, dspark)
}

fn capture_on(host: &Rc<Host>, rows: u64, enabled: Option<bool>, dspark: bool) -> Vec<String> {
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
        let services = dspark::host::MappedHost::with_peer_copies(Rc::clone(host));
        create_with_services(host, HostServicesTableV1::new(Rc::new(services)))
    } else {
        create_sparse(host)
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
    let mut journal = host.device.journal.borrow().clone();
    journal.extend(count_metadata(host));
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

fn compute_without_consumers_by_stream(
    journal: &[String],
) -> std::collections::BTreeMap<&str, Vec<&String>> {
    let mut streams = std::collections::BTreeMap::<&str, Vec<&String>>::new();
    for line in compute(journal)
        .into_iter()
        .filter(|line| !consumer(line) && !producer(line))
    {
        streams
            .entry(line.split_whitespace().nth(1).unwrap())
            .or_default()
            .push(line);
    }
    streams
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
                        && (line.contains("chop_comm_allreduce_phased_bf16_v1")
                            || line.contains("chop_comm_allreduce_phased_turns_bf16_v1"))
                })
                .count()
        };
        // Eight short cone calls retain the phased BF16 transport.
        assert_eq!(phased(&reference), 81 * 4);
        assert_eq!(phased(&candidate), 8 * 4);
        let reductions = (phased(&reference) - phased(&candidate)) / 4;
        assert_eq!(reductions, 73);
        assert!(!reference.iter().any(|line| line.starts_with("peer-copy ")));
        // Default asynchronous Engram gathers can change the interleaving
        // of independent streams around the transport joins. Each stream
        // must retain every launch, operand, extent and alias in order.
        // The dependency audit below still checks all cross-stream joins.
        assert_eq!(
            compute_without_consumers_by_stream(&reference),
            compute_without_consumers_by_stream(&candidate)
        );
        assert_eq!(
            split_chains(&reference, &candidate),
            if rows == 8192 {
                (40 * 4, 20 * 4)
            } else {
                (0, 0)
            }
        );
        // The producer splits over the owner intervals at any row count.
        assert_eq!(split_producers(&reference, &candidate).unwrap(), 36 * 4);
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

fn assert_prepared_mapped_reads(journal: &[String]) {
    let first = journal
        .iter()
        .position(|line| line.starts_with("launch "))
        .unwrap();
    let count = |lines: &[String], prefix: &str| {
        lines.iter().filter(|line| line.starts_with(prefix)).count()
    };
    assert_eq!(count(&journal[..first], "async mapped ids "), 2);
    assert_eq!(count(journal, "async mapped wait "), 8);
    assert_eq!(count(journal, "async mapped complete"), 2);
    // The fake service materializes both planes at each stream's join.
    assert_eq!(count(&journal[..first], "mapped gather "), 0);
    assert_eq!(count(&journal[first..], "mapped gather "), 16);
}

#[test]
fn sparse_dma_enqueues_mapped_reads_before_compute() {
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
    let mut instance = create_with_services(
        &host,
        HostServicesTableV1::new(Rc::new(dspark::host::MappedHost::new(Rc::clone(&host)))),
    );
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
    assert_prepared_mapped_reads(&journal);
    assert_eq!(
        journal
            .iter()
            .filter(|line| line.starts_with("launch ") && line.contains(BF16_SUM))
            .count(),
        73 * 4
    );
    drop(journal);
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn dspark_sparse_dma_joins_every_mapped_plane() {
    let journal = capture_with(8192, Some(true), true);
    assert_prepared_mapped_reads(&journal);
    let mut observed: Vec<_> = journal
        .iter()
        .filter(|line| line.starts_with("mapped gather "))
        .cloned()
        .collect();
    let mut expected = Vec::new();
    for rank in 0..4 {
        for layer in [1, 14] {
            for plane in ["weight", "scale"] {
                expected.push(format!(
                    "mapped gather {rank} layers.{layer}.engram.embed.{plane}"
                ));
            }
        }
    }
    observed.sort();
    expected.sort();
    assert_eq!(observed, expected);
}

#[test]
fn sparse_dma_carries_bf16_partials_where_the_cone_reduces_them() {
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
        let (reference, candidate) = { (capture(rows, Some(false)), capture(rows, None)) };
        assert!(!reference.iter().any(|line| line.starts_with("peer-copy ")));
        assert_eq!(phased(&reference) - phased(&candidate), 73 * 4);
        assert_eq!(count(&candidate, SUM), 0);
        assert_eq!(count(&candidate, BF16_SUM), 73 * 4);
        assert_eq!(count(&candidate, BF16_ROUND), 73 * 6 + 36 * 6);
        let before = compute(&reference)
            .into_iter()
            .filter(|line| !consumer(line) && !producer(line))
            .collect::<Vec<_>>();
        let after = compute(&candidate)
            .into_iter()
            .filter(|line| !consumer(line) && !producer(line))
            .collect::<Vec<_>>();
        assert_eq!(before, after);
        assert_eq!(
            split_chains(&reference, &candidate),
            if rows == 8192 {
                (40 * 4, 20 * 4)
            } else {
                (0, 0)
            }
        );
        // The producer splits over the owner intervals at any row count.
        assert_eq!(split_producers(&reference, &candidate).unwrap(), 36 * 4);
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

/// Walk each stream's consumer launches in submission order: an unsplit
/// launch is unchanged; a split chain is four chains over the owner
/// intervals, the rank's own first. Returns the split chains and how many
/// of them end in the certifying norm-ceil.
fn split_chains(serial: &[String], pipelined: &[String]) -> (usize, usize) {
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
            let (rank, counts) = count_table(pipelined, st);
            for (at, part) in parts.iter().enumerate() {
                let quarter = (rank + 4 - (at / 3) as u64) % 4;
                exact_slice(chain[at % 3], part, quarter, counts).unwrap();
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

const PRODUCER: &str = "chop_moe_deepseek41_sum_add_f32_k6_w5120_l384_t8192_v1";

fn producer(line: &str) -> bool {
    line.starts_with("launch ") && line.contains(&format!(" {PRODUCER} "))
}

/// Walk each stream's route-row sums in submission order: an unsplit
/// launch is unchanged; a split one is four launches over the owner
/// intervals in send order, the interval for scatter phase 0's destination
/// first and the rank's own last. Returns the split launches.
fn split_producers(serial: &[String], split: &[String]) -> Result<usize, String> {
    let streams = serial
        .iter()
        .filter(|l| producer(l))
        .map(|l| stream(l))
        .collect::<BTreeSet<_>>();
    let mut count = 0;
    for st in streams {
        let s = serial
            .iter()
            .filter(|l| producer(l) && stream(l) == st)
            .collect::<Vec<_>>();
        let p = split
            .iter()
            .filter(|l| producer(l) && stream(l) == st)
            .collect::<Vec<_>>();
        let (mut i, mut j) = (0, 0);
        while i < s.len() {
            if p.get(j) == Some(&s[i]) {
                i += 1;
                j += 1;
                continue;
            }
            let parts = p
                .get(j..j + 4)
                .ok_or("a split producer has fewer than four parts")?;
            let (rank, counts) = count_table(split, st);
            for (at, part) in parts.iter().enumerate() {
                exact_slice(s[i], part, (rank + at as u64 + 1) % 4, counts)?;
            }
            count += 1;
            i += 1;
            j += 4;
        }
        if j != p.len() {
            return Err("extra producer launches".into());
        }
    }
    Ok(count)
}

/// The served transports with every copy pushed against pulled: the same
/// launches and the same copies, each issued by its source, ordered.
#[test]
fn dma_pushes_every_copy_from_its_source() {
    let served = |rows: u64, pull_only: bool| {
        let host = host();
        host.pull_only.set(pull_only);
        let journal = capture_on(&host, rows, None, false);
        let issued = super::copy_ranks(&host, &journal);
        (journal, issued)
    };
    for rows in [8192, 8191] {
        let (pulled, pulls) = served(rows, true);
        let (pushed, pushes) = served(rows, false);
        let launches = |j: &[String]| {
            j.iter()
                .filter(|l| l.starts_with("launch "))
                .cloned()
                .collect::<Vec<_>>()
        };
        assert_eq!(launches(&pulled), launches(&pushed));
        let copies = |j: &[String]| {
            let mut c = j
                .iter()
                .filter(|l| l.starts_with("peer-copy "))
                .map(|l| l.splitn(3, ' ').nth(2).unwrap().to_owned())
                .collect::<Vec<_>>();
            c.sort();
            c
        };
        assert_eq!(copies(&pulled), copies(&pushed));
        assert_eq!(pulls.len(), 73 * 24);
        assert!(pulls.iter().all(|&(issuer, _, to)| issuer == to));
        assert!(pushes.iter().all(|&(issuer, from, _)| issuer == from));
        assert_eq!(audit(&pushed), audit(&pulled));
        audit_transport(&pulled).unwrap();
        assert_eq!(audit_transport(&pushed), audit_transport(&pulled));
        // Without each destination's waits for the pushes into its inbox,
        // its sum reads rows a peer may still be writing.
        let mut early = Vec::new();
        for (at, line) in pushed.iter().enumerate() {
            let on = |l: &&String| l.split_whitespace().nth(1) == line.split_whitespace().nth(1);
            let next = pushed[at + 1..]
                .iter()
                .find(|l| on(l) && !l.starts_with("wait "));
            if line.starts_with("wait ") && next.is_some_and(|n| transport_launch(n)) {
                continue;
            }
            early.push(line.clone());
        }
        assert!(early.len() < pushed.len());
        assert!(audit_transport(&early).is_err(), "{rows}");
    }
}

#[test]
fn interval_dependencies_require_consumer_and_producer_joins() {
    let pushed = capture(8192, None);
    audit(&pushed).unwrap();
    let at = pushed
        .iter()
        .enumerate()
        .find_map(|(at, line)| {
            let next = pushed[at + 1..]
                .iter()
                .find(|next| stream(next) == stream(line) && !next.starts_with("wait "));
            (line.starts_with("wait ") && next.is_some_and(|next| next.contains(CONSUMERS[0])))
                .then_some(at)
        })
        .expect("a gather join before an interval consumer");
    let mut early = pushed.clone();
    early.remove(at);
    let error = audit(&early).expect_err("consumer ran before its gather completed");
    assert!(error.contains(CONSUMERS[0]), "{error}");

    // A destination-only host retains pulls. Each interval copy must wait
    // for that source's producer and BF16 rounding, independently of sums.
    let host = host();
    host.pull_only.set(true);
    let pulled = capture_on(&host, 8192, None, false);
    audit(&pulled).unwrap();
    let at = pulled
        .iter()
        .enumerate()
        .find_map(|(at, line)| {
            if !line.starts_with("wait ") {
                return None;
            }
            let next = pulled[at + 1..]
                .iter()
                .find(|next| stream(next) == stream(line) && !next.starts_with("wait "))?;
            if !next.starts_with("peer-copy ") {
                return None;
            }
            let event = line.split_whitespace().nth(2)?;
            let record = pulled[..at].iter().rposition(|before| {
                before.starts_with("record ") && before.split_whitespace().nth(2) == Some(event)
            })?;
            let source = stream(&pulled[record]);
            let mut launches = pulled[..record]
                .iter()
                .rev()
                .filter(|before| before.starts_with("launch ") && stream(before) == source);
            let rounded = launches.next()?.contains(BF16_ROUND);
            (rounded && producer(launches.next()?)).then_some(at)
        })
        .expect("a pull join after one interval's producer and rounding");
    let mut early = pulled.clone();
    early.remove(at);
    let error = audit(&early).expect_err("copy ran before its source rounded the interval");
    assert!(error.contains("peer-copy "), "{error}");
}

#[test]
fn interval_audits_reject_changed_launch_configurations() {
    let reference = capture(8192, Some(false));
    let split = capture(8192, None);
    assert_eq!(split_chains(&reference, &split), (160, 80));
    assert_eq!(split_producers(&reference, &split).unwrap(), 144);
    let mut bad = split.clone();
    let at = bad
        .iter()
        .position(|line| line.contains(CONSUMERS[0]))
        .unwrap();
    bad[at] = bad[at].replacen("grid_z: 1", "grid_z: 2", 1);
    assert!(
        std::panic::catch_unwind(|| split_chains(&reference, &bad)).is_err(),
        "an unrelated consumer grid change escaped"
    );
    let mut bad = split.clone();
    let at = bad.iter().position(|line| producer(line)).unwrap();
    bad[at] = bad[at].replacen("grid_z: 1", "grid_z: 2", 1);
    assert!(
        split_producers(&reference, &bad).is_err(),
        "an unrelated producer grid change escaped"
    );
    let argument = |line: &str, at: usize, value: String| {
        let mut args = super::super::query_window::arguments(line);
        args[at] = value;
        format!("{} [{}]", line.split_once(" [").unwrap().0, args.join(", "))
    };
    let expands: Vec<_> = split
        .iter()
        .enumerate()
        .filter(|(_, line)| line.contains(CONSUMERS[0]) && stream(line) == "2")
        .map(|(at, _)| at)
        .collect();
    let mut bad = split.clone();
    let first = super::super::query_window::arguments(&bad[expands[0]])[0].clone();
    let second = super::super::query_window::arguments(&bad[expands[1]])[0].clone();
    bad[expands[0]] = argument(&bad[expands[0]], 0, second);
    bad[expands[1]] = argument(&bad[expands[1]], 0, first);
    assert!(
        std::panic::catch_unwind(|| split_chains(&reference, &bad)).is_err(),
        "mismatched quarters within a consumer escaped"
    );
    let at = split
        .iter()
        .position(|line| line.contains(CERTIFIED_NORM))
        .unwrap();
    let mut bad = split.clone();
    bad[at] = argument(&bad[at], 9, "ScalarF32Bits(0)".into());
    assert!(
        std::panic::catch_unwind(|| split_chains(&reference, &bad)).is_err(),
        "an unrelated norm scalar escaped"
    );
    let at = split.iter().position(|line| producer(line)).unwrap();
    let mut bad = split.clone();
    bad[at] = argument(&bad[at], 3, "ScalarU32(99)".into());
    assert!(
        split_producers(&reference, &bad).is_err(),
        "a changed producer rank escaped"
    );
    let mut bad = split.clone();
    let live = super::super::query_window::arguments(&bad[at])[2].clone();
    let span = super::super::shared_overlap::journal_spans(&live)[0];
    bad[at] = argument(
        &bad[at],
        2,
        live.replacen(
            &format!("offset: {},", span.1),
            &format!("offset: {},", span.1 + 4),
            1,
        ),
    );
    assert!(
        split_producers(&reference, &bad).is_err(),
        "a wrong live-count cell escaped"
    );
}

/// Exact prefill count-table identities, from their create-time uploaded
/// contents and allocation ranks rather than first use in a launch.
fn count_metadata(host: &Host) -> Vec<String> {
    let expected: Vec<u8> = (0..=8192_u32).flat_map(u32::to_le_bytes).collect();
    let allocations = host.device.allocation_ranks.borrow();
    let mut tables = BTreeMap::new();
    for (&(raw, offset), bytes) in host.uploads.borrow().iter() {
        let (rank, length) = allocations[&(raw as u32)];
        if *bytes == expected && offset == 0 && length == expected.len() as u64 {
            assert!(
                tables
                    .insert(rank, (raw, offset, expected.len() as u64))
                    .is_none()
            );
        }
    }
    assert_eq!(tables.len(), 4);
    host.device
        .stream_ranks
        .borrow()
        .iter()
        .map(|(&stream, &rank)| {
            let (raw, offset, length) = tables[&rank];
            format!("prefill-counts {stream} {rank} {raw} {offset} {length}")
        })
        .collect()
}

fn count_table(journal: &[String], stream: &str) -> (u64, Span) {
    let prefix = format!("prefill-counts {stream} ");
    let lines: Vec<_> = journal
        .iter()
        .filter_map(|line| line.strip_prefix(&prefix))
        .collect();
    assert_eq!(lines.len(), 1);
    let values: Vec<u64> = lines[0]
        .split_whitespace()
        .map(|word| word.parse().unwrap())
        .collect();
    assert_eq!(values.len(), 4);
    (values[0], (values[1], values[2], values[3]))
}

/// Compare the entire launch after only the specified owner slice. ABI
/// positions are checked independently of the production row-layout table.
fn exact_slice(original: &str, part: &str, quarter: u64, counts: Span) -> Result<(), String> {
    use super::super::query_window::{arguments, symbol};
    let spans = super::super::shared_overlap::journal_spans;
    let hidden = u64::from(FLASH.shape.hidden);
    let streams = u64::from(FLASH.shape.hc.streams);
    let flat = hidden * streams * 2;
    let mut rows: Vec<(usize, u64)>;
    let (live, row_offset, dimension, step, capacity) = match symbol(original) {
        name if name == CONSUMERS[0] => {
            rows = vec![
                (0, flat),
                (1, hidden * 2),
                (2, streams * 4),
                (3, streams * streams * 4),
                (4, flat),
            ];
            (5, None, "grid_y", 1, false)
        }
        name if name == CONSUMERS[1] => {
            rows = vec![
                (0, flat),
                (4, streams * 4),
                (5, hidden * 2),
                (6, streams * 4),
                (7, streams * 4),
                (8, streams * streams * 4),
            ];
            (9, None, "grid_x", 8, false)
        }
        name if name == CONSUMERS[2] || name == CERTIFIED_NORM => {
            rows = vec![
                (0, hidden * 2),
                (2, hidden * 2),
                (4, hidden),
                (5, hidden / 32),
            ];
            if name == CERTIFIED_NORM {
                rows.push((10, 4));
            }
            (6, Some(8), "grid_x", 1, false)
        }
        PRODUCER => {
            let top_k = u64::from(FLASH.shape.moe.top_k);
            rows = vec![
                (0, top_k * hidden * 4),
                (1, top_k * 4),
                (4, hidden * 4),
                (5, hidden * 4),
            ];
            (2, None, "grid_x", 1, true)
        }
        _ => return Err("not an interval ABI".into()),
    };
    let (old_header, _) = original.split_once(" [").ok_or("missing arguments")?;
    let (new_header, _) = part.split_once(" [").ok_or("missing arguments")?;
    let grid = old_header
        .split_once(&format!("{dimension}: "))
        .ok_or("missing row grid")?
        .1
        .split(',')
        .next()
        .unwrap()
        .parse::<u64>()
        .map_err(|_| "invalid row grid")?;
    let total = grid * step;
    if quarter >= 4 {
        return Err("invalid interval extent".into());
    }
    // The transport's owner intervals: `ceil(total / 4)` rows, clipped.
    let owner = total.div_ceil(4);
    let begin = (owner * quarter).min(total);
    let count = owner.min(total - begin);
    if count == 0 || !count.is_multiple_of(step) {
        return Err("invalid interval extent".into());
    }
    let expected_header = old_header.replacen(
        &format!("{dimension}: {grid},"),
        &format!("{dimension}: {},", count / step),
        1,
    );
    if new_header != expected_header {
        return Err(format!("unrelated launch configuration: {part}"));
    }
    let old = arguments(original);
    let actual = arguments(part);
    if old.len() != actual.len() {
        return Err("argument count changed".into());
    }
    if row_offset.is_some() && old[3] != "NullDevicePointer" {
        rows.push((3, hidden * 4));
    }
    for (at, (before, after)) in old.iter().zip(&actual).enumerate() {
        if let Some(&(_, bytes)) = rows.iter().find(|&&(index, _)| index == at) {
            let (whole, piece) = (spans(before), spans(after));
            if whole.len() != 1 || piece.len() != 1 {
                return Err("row argument is not one span".into());
            }
            let w = whole[0];
            let exact_rows = !capacity || at == 5;
            if w.2 < total * bytes
                || (exact_rows && w.2 != total * bytes)
                || piece[0] != (w.0, w.1 + begin * bytes, count * bytes)
            {
                return Err(format!(
                    "argument {at} is not owner quarter {quarter}: {part}"
                ));
            }
        } else if at == live {
            let original = spans(before);
            if original.len() != 1
                || original[0].2 != 4
                || counts.2 != 8193 * 4
                || spans(after) != vec![(counts.0, counts.1 + count * 4, 4)]
            {
                return Err(format!("live count is not its rank's table cell: {part}"));
            }
        } else if Some(at) == row_offset {
            let first = before
                .strip_prefix("ScalarU64(")
                .and_then(|s| s.strip_suffix(')'))
                .ok_or("row offset is not U64")?
                .parse::<u64>()
                .map_err(|_| "invalid row offset")?;
            if *after != format!("ScalarU64({})", first + begin) {
                return Err("wrong status row offset".into());
            }
        } else if before != after {
            return Err(format!("unrelated argument {at} changed: {part}"));
        }
    }
    Ok(())
}

/// The rows of a launch's row grid.
fn grid_rows(line: &str) -> u64 {
    line.split_once("grid_x: ")
        .and_then(|(_, rest)| rest.split(',').next())
        .and_then(|rows| rows.parse().ok())
        .expect("a row grid")
}

/// The MoE producer before every DMA call (512 rows or more) splits over
/// the transport's owner intervals at any row count and at every layer,
/// each part an exact slice of the unsplit launch, which writes exactly its
/// rows of the partial; every other launch is the unsplit journal's.
#[test]
fn producer_intervals_split_every_moe_call_at_any_row_count() {
    use super::super::query_window::arguments;
    let spans = super::super::shared_overlap::journal_spans;
    let hidden = u64::from(FLASH.shape.hidden);
    let without = |journal: &[String]| {
        compute(journal)
            .into_iter()
            .filter(|line| !consumer(line) && !producer(line))
            .cloned()
            .collect::<Vec<_>>()
    };
    for rows in [512, 1500, 2304, 3000, 8191, 8192] {
        let (unsplit, split) = (capture(rows, Some(false)), capture(rows, None));
        let calls = unsplit
            .iter()
            .filter(|line| producer(line) && grid_rows(line) >= 512)
            .count();
        if rows == 8192 {
            assert_eq!(calls, 36 * 4);
        } else {
            assert!(
                calls >= 20 * 4,
                "rows={rows}: {calls} producers before DMA calls"
            );
        }
        for line in unsplit.iter().filter(|line| producer(line)) {
            let lengths = spans(&arguments(line)[5])
                .into_iter()
                .map(|span| span.2)
                .collect::<Vec<_>>();
            assert_eq!(lengths, vec![grid_rows(line) * hidden * 4], "{line}");
        }
        assert_eq!(
            split_producers(&unsplit, &split).unwrap(),
            calls,
            "rows={rows}"
        );
        assert_eq!(without(&unsplit), without(&split), "rows={rows}");
    }
}
