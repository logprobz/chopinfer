//! Coupled sampled drafts on an instance that runs the MTP drafter: create
//! loads the draft noise's cubin and allocates each rank's noise records;
//! each drafter run uploads its draft rows' records, its request's seed and
//! temperature at the generated index of the run's draft position, and its
//! argmax adds the noise first on every rank. An instance without the
//! drafter loads and allocates none of it.

use runtime_bundle_sdk::DraftSamplingV1;

use super::*;
use crate::buffers::Buffer;
use kernel_core::draft_noise::{NOISE_BYTES, NoiseRecord};
use model_glm5next_contract::contract::DRAFT_ITEMS;
use model_glm5next_contract::sm120::engines::{DRAFT_NOISE_CUBIN, DRAFT_NOISE_SYMBOL};

const PARTIAL: &str = "chop_row_argmax_partial_f32_v1";

/// [`drafted`] of request 7 (`rows` rows the drafter drafts, after
/// `previous` tokens) carrying draft sampling `record` when given.
fn sampled(previous: u64, rows: u32, bonus: u32, record: Option<DraftSamplingV1>) -> Vec<u8> {
    drafted_sampling(7, previous, rows, bonus, 0, record)
}

/// A drafting instance on `host`'s KV format.
fn create_drafting(host: &Rc<Host>) -> BundleInstanceV1 {
    api()
        .create(
            HostServicesTableV1::new(Rc::clone(host)),
            &crate::config::tests::record_with(DECODE_BATCH, &options(2048), host.format),
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
    let mut instance = create_drafting(&host);
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

/// An instance on a KV format without the drafter's layer runs no drafter:
/// it neither loads nor resolves the draft noise, and no step launches it
/// (`buffers::tests` shows its records are allocated only with coupled
/// drafts).
#[test]
fn an_instance_without_a_drafter_loads_no_draft_noise() {
    let host = FakeHost::new((3, 32));
    assert!(!has_mtp_layer(&host.format));
    let mut instance = create_drafting(&host);
    crate::instance::run_lowered_stages_alone(instance.handle());
    instance
        .execute(&prompt_chunk(7, 0, 10, true), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    instance.destroy().unwrap();
    assert!(
        host.device.journal.borrow().iter().all(|line| {
            !line.contains(DRAFT_NOISE_SYMBOL) && !line.contains(DRAFT_NOISE_CUBIN)
        })
    );
}
