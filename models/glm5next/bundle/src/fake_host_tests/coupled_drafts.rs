//! Coupled sampled drafts (`runtime.mtp_coupled_drafts`) on an instance
//! that runs the MTP drafter: create loads the draft noise's cubin and
//! allocates each rank's noise records with the option alone; each drafter
//! run uploads its draft rows' records, its request's seed and temperature
//! at the generated index of the run's draft position, and its argmax adds
//! the noise first on every rank. With the option off, or without a
//! drafter, the instance issues what a record without the option issues.

use runtime_bundle_sdk::DraftSamplingV1;

use super::*;
use crate::buffers::Buffer;
use kernel_core::draft_noise::{NOISE_BYTES, NoiseRecord};
use model_glm5next_contract::contract::DRAFT_ITEMS;
use model_glm5next_contract::sm120::engines::DRAFT_NOISE_SYMBOL;

const PARTIAL: &str = "chop_row_argmax_partial_f32_v1";

/// [`drafted`] of request 7 (`rows` rows the drafter drafts, after
/// `previous` tokens) carrying draft sampling `record` when given.
fn sampled(previous: u64, rows: u32, bonus: u32, record: Option<DraftSamplingV1>) -> Vec<u8> {
    drafted_sampling(7, previous, rows, bonus, 0, record)
}

/// A drafting instance on the MTP KV format, its model options naming the
/// coupled drafts option as `coupled` (a JSON value), or not at all.
fn create_drafting(host: &Rc<Host>, coupled: Option<&str>) -> BundleInstanceV1 {
    let options = match coupled {
        Some(value) => crate::config::tests::options_coupled_drafts(2048, value),
        None => options(2048),
    };
    api()
        .create(
            HostServicesTableV1::new(Rc::clone(host)),
            &crate::config::tests::record_with(DECODE_BATCH, &options, host.format),
        )
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
}

fn mtp_host() -> Rc<Host> {
    let format = Shape::TP4.kv_formats()[1];
    assert!(has_mtp_layer(&format));
    FakeHost::with_format((3, 32), format)
}

fn record(seed: u64, generated: u64, temperature: f64) -> DraftSamplingV1 {
    DraftSamplingV1::new(seed, generated, temperature, 1.0, 0.0, 0, 0).unwrap()
}

/// The records a run of one drafting item uploads: its record, then greedy
/// ones to every draft row.
fn uploaded_records(first: NoiseRecord) -> Vec<u8> {
    std::iter::once(first)
        .chain(std::iter::repeat_n(
            NoiseRecord::GREEDY,
            DRAFT_ITEMS as usize - 1,
        ))
        .flat_map(|record| record.encode())
        .collect()
}

/// Every upload to each rank's noise records from journal line `from` on.
fn noise_uploads(host: &FakeHost, handle: BundleHandleV1, from: usize) -> Vec<Vec<Vec<u8>>> {
    (0..4)
        .map(|rank| {
            let span = crate::instance::scratch_span(handle, rank, Buffer::DraftNoise);
            assert_eq!(span.length, u64::from(DRAFT_ITEMS) * NOISE_BYTES as u64);
            host.upload_log
                .borrow()
                .iter()
                .filter(|(line, at, _)| *line >= from && *at == span)
                .map(|(_, _, bytes)| bytes.clone())
                .collect()
        })
        .collect()
}

fn symbol(line: &str) -> &str {
    line.split(' ').nth(2).unwrap()
}

/// A sampled item's three drafts draw at its first verify row's generated
/// index and the two after it: the extend pass's records at 41, each draft
/// step's one later, on every rank. Each rank's argmax adds the noise to
/// its logits shard right before its partial, in the extend pass's graph
/// and the first draft step's (the second replays it). A greedy record
/// and a batch without records upload greedy records.
#[test]
fn sampled_items_draft_from_the_noise_of_each_position() {
    let host = mtp_host();
    let mut instance = create_drafting(&host, Some("true"));
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("runtime.mtp_coupled_drafts=true: the MTP drafter"))
    );
    let resolved = host
        .device
        .journal
        .borrow()
        .iter()
        .filter(|line| line.starts_with("resolve ") && line.ends_with(DRAFT_NOISE_SYMBOL))
        .count();
    assert_eq!(resolved, 4, "every rank resolves the draft noise");
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    instance
        .execute(&prompt_chunk(7, 0, 10, true), &mut DraftSink::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    let seed = 0x0123_4567_89ab_cdef;
    let start = host.device.journal.borrow().len();
    let mut sink = DraftSink::default();
    instance
        .execute(&sampled(10, 4, 21, Some(record(seed, 41, 0.8))), &mut sink)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    assert_eq!(sink.drafts.len(), 1);
    let at = |generated| {
        uploaded_records(NoiseRecord {
            seed,
            generated,
            temperature: 0.8,
        })
    };
    for uploads in noise_uploads(&host, handle, start) {
        assert_eq!(uploads, [at(41), at(42), at(43)]);
    }
    for (stream, launches) in launches_by_stream(&host, start) {
        let noise = launches
            .iter()
            .enumerate()
            .filter(|(_, line)| symbol(line) == DRAFT_NOISE_SYMBOL)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if noise.is_empty() {
            continue;
        }
        assert_eq!(noise.len(), 2, "stream {stream}: the captured passes");
        for index in noise {
            assert_eq!(symbol(&launches[index + 1]), PARTIAL, "stream {stream}");
        }
        let partials = launches
            .iter()
            .filter(|line| symbol(line) == PARTIAL)
            .count();
        assert_eq!(partials, 2, "stream {stream}");
    }
    let noise = |from| count_launches(&host, from, DRAFT_NOISE_SYMBOL);
    assert_eq!(noise(start), 4 * 2);
    instance.commit_accepted(&[kept(7, 2)]).unwrap();
    // A greedy record, and a batch without records, draft greedily.
    for (previous, record) in [(12, Some(DraftSamplingV1::GREEDY)), (14, None)] {
        let start = host.device.journal.borrow().len();
        instance
            .execute(&sampled(previous, 4, 22, record), &mut DraftSink::default())
            .unwrap();
        let greedy = uploaded_records(NoiseRecord::GREEDY);
        for uploads in noise_uploads(&host, handle, start) {
            assert_eq!(uploads, [greedy.clone(), greedy.clone(), greedy.clone()]);
        }
        instance.commit_accepted(&[kept(7, 2)]).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// Launch lines from journal line `from` on of `wanted`.
fn count_launches(host: &FakeHost, from: usize, wanted: &str) -> usize {
    host.device.journal.borrow()[from..]
        .iter()
        .filter(|line| line.starts_with("launch ") && symbol(line) == wanted)
        .count()
}

/// The journal of a drafting instance with the coupled drafts option as
/// `coupled`, through a prompt chunk and two drafted steps of request 7
/// with `record`, and the launches by stream of each step.
fn drafted_journal(
    coupled: Option<&str>,
    record: Option<DraftSamplingV1>,
) -> (Vec<String>, Vec<BTreeMap<u32, Vec<String>>>) {
    let host = mtp_host();
    let mut instance = create_drafting(&host, coupled);
    crate::instance::run_lowered_stages_alone(instance.handle());
    let mut steps = Vec::new();
    let start = host.device.journal.borrow().len();
    instance
        .execute(&prompt_chunk(7, 0, 10, true), &mut DraftSink::default())
        .unwrap();
    instance.commit().unwrap();
    steps.push(launches_by_stream(&host, start));
    for previous in [10, 12] {
        let start = host.device.journal.borrow().len();
        instance
            .execute(&sampled(previous, 4, 21, record), &mut DraftSink::default())
            .unwrap();
        instance.commit_accepted(&[kept(7, 2)]).unwrap();
        steps.push(launches_by_stream(&host, start));
    }
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let journal = host.device.journal.borrow().clone();
    (journal, steps)
}

/// With coupled drafts every stream launches what the greedy drafter
/// launches, under one consistent renaming of buffer handles (create's
/// added module and records shift the handles after them), with each
/// noise launch before an argmax partial added: a sampled record changes
/// what the noise draws, never which launches run.
#[test]
fn coupled_drafters_add_only_the_noise_before_each_argmax() {
    let (_, greedy) = drafted_journal(None, None);
    for record in [None, Some(record(5, 41, 1.0))] {
        let (_, coupled) = drafted_journal(Some("true"), record);
        assert_eq!(greedy.len(), coupled.len());
        for (step, (greedy, coupled)) in greedy.iter().zip(&coupled).enumerate() {
            assert_eq!(
                greedy.keys().collect::<Vec<_>>(),
                coupled.keys().collect::<Vec<_>>(),
                "step {step}"
            );
            let mut noise = 0;
            for (stream, lines) in coupled {
                let kept = lines
                    .iter()
                    .filter(|line| symbol(line) != DRAFT_NOISE_SYMBOL)
                    .cloned()
                    .collect::<Vec<_>>();
                noise += lines.len() - kept.len();
                assert_eq!(
                    super::comm_epilogue::relabel(&kept),
                    super::comm_epilogue::relabel(&greedy[stream]),
                    "step {step}, stream {stream}"
                );
            }
            // The prompt's extend pass launches once per rank; the first
            // drafted step captures its extend pass (one row, a decode
            // bucket) and its first draft step, the second replays the
            // draft steps and captures its extend pass's verify bucket.
            assert_eq!(noise, [4, 8, 4][step], "step {step}");
        }
    }
}

/// An explicit false option creates, prefills, drafts, commits and
/// retires exactly as a record without the option (the journals before the
/// option existed), and so does a true option on an instance that runs no
/// drafter.
#[test]
fn the_option_off_or_without_a_drafter_issues_what_no_option_issues() {
    let sampled = Some(record(5, 41, 1.0));
    let (off, _) = drafted_journal(Some("false"), sampled);
    let (absent, _) = drafted_journal(None, sampled);
    assert_eq!(off, absent);
    let journal = |coupled: Option<&str>| {
        let host = FakeHost::new((3, 32));
        assert!(!has_mtp_layer(&host.format));
        let mut instance = create_drafting(&host, coupled);
        crate::instance::run_lowered_stages_alone(instance.handle());
        instance
            .execute(&prompt_chunk(7, 0, 10, true), &mut NoRows::default())
            .unwrap();
        instance.commit().unwrap();
        instance.destroy().unwrap();
        host.device.journal.borrow().clone()
    };
    assert_eq!(journal(Some("true")), journal(None));
}
