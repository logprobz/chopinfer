//! Coupled sampled drafting: a V6 batch's draft sampling records reach the
//! drafter as each drafting item's noise record, in item order, at the
//! generated index of the item's first verify row; greedy items, and
//! every item of a batch without records, upload zero records, and the
//! greedy mode launches no noise at all. The create record's
//! `dspark_coupled_drafts` model option selects the mode.
use super::*;
use crate::lower::dspark::DraftMode;
use kernel_core::draft_noise::{NOISE_BYTES, NoiseRecord};
use runtime_bundle_sdk::DraftSamplingV1;

const SEED: u64 = 0x1234_5678_9abc_def0;
const NOISE: &str = crate::sm120::dspark_noise::SYMBOL;

/// One verify step drafting `drafts` for every (request, slot, start,
/// record) item, with the records as V6 draft sampling when `v6`.
fn sampled(items: &[(u64, u64, u64, DraftSamplingV1)], drafts: u32, v6: bool) -> Vec<u8> {
    let rows = u64::from(drafts + 1);
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    for &(request, slot, start, _) in items {
        batch.push(BatchItemV1::verify(request, drafts + 1, true, vec![7]));
        appends.push(BatchAppendV1::new(
            start,
            rows,
            start + rows,
            (0..(start + rows).div_ceil(64))
                .map(|page| slot * 8 + page)
                .collect(),
            None,
        ));
    }
    let request = BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(items.iter().map(|item| item.1).collect());
    if v6 {
        request
            .with_draft_sampling(items.iter().map(|item| item.3).collect())
            .encode()
            .unwrap()
    } else {
        request.encode().unwrap()
    }
}

fn record(generated: u64) -> DraftSamplingV1 {
    DraftSamplingV1::new(SEED, generated, 0.8, 1.0, 0.0, 0, 0).unwrap()
}

/// The bytes the last upload covering `span` left there.
fn uploaded(host: &Host, span: runtime_bundle_abi::DeviceBufferSpanV1) -> Vec<u8> {
    let uploads = host.uploads.borrow();
    let (&(_, offset), bytes) = uploads
        .iter()
        .find(|&(&(buffer, offset), bytes)| {
            buffer == span.buffer.raw
                && offset <= span.offset
                && span.offset + span.length <= offset + bytes.len() as u64
        })
        .expect("an upload covers the noise plane");
    let at = (span.offset - offset) as usize;
    bytes[at..at + span.length as usize].to_vec()
}

/// Each rank's noise plane holds `records` in item order, then zeros.
#[track_caller]
fn assert_noise(host: &Host, instance: &BundleInstanceV1, records: &[NoiseRecord]) {
    let (spans, mode) = crate::instance::dspark_noise(instance.handle()).unwrap();
    assert_eq!(mode, DraftMode::Sampled);
    let mut expected = records
        .iter()
        .flat_map(NoiseRecord::encode)
        .collect::<Vec<_>>();
    expected.resize(records.len() * NOISE_BYTES, 0);
    for span in spans {
        assert_eq!(uploaded(host, span)[..expected.len()], expected);
    }
}

fn noise(generated: u64) -> NoiseRecord {
    NoiseRecord {
        seed: SEED,
        generated,
        temperature: 0.8,
    }
}

/// A drafter instance of four items whose create record couples sampled
/// drafts when `mode` is sampled.
fn create_in(shape: Shape, mode: DraftMode) -> (Rc<Host>, BundleInstanceV1) {
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.drafter.set(true);
    host.speculation.set(Some((5, 32)));
    let record = crate::config::tests::with_coupled_drafts(
        crate::config::tests::record(&shape, host.kv, 4, false),
        mode == DraftMode::Sampled,
    );
    let instance = api()
        .create(
            HostServicesTableV1::new(Rc::new(MappedHost::new(Rc::clone(&host)))),
            &record,
        )
        .unwrap_or_else(|e| panic!("drafter create: {e:?} {:?}", host.logs.borrow()));
    (host, instance)
}

fn noise_launches(host: &Host, from: usize) -> usize {
    noise_launches_in(&host.journal.borrow()[from..])
}

#[test]
fn sampled_items_draft_from_the_noise_of_their_first_verify_row() {
    for ranks in [Ranks::TP1, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let (host, mut instance) = create_in(shape, DraftMode::Sampled);
        for (request, slot) in [(7, 0), (9, 1)] {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(request, slot, 0, 3, true),
            );
            instance.commit().unwrap();
        }
        let from = host.journal.borrow().len();
        // Request 7 samples at temperature 0.8 and has sampled three tokens:
        // its first verify row draws at generated index 3. Request 9 is
        // greedy and keeps a zero record.
        let first = execute(
            &mut instance,
            &host,
            &sampled(
                &[(7, 0, 3, record(3)), (9, 1, 3, DraftSamplingV1::GREEDY)],
                3,
                true,
            ),
        );
        assert_eq!(first.draft_tokens().len(), 2);
        assert_noise(&host, &instance, &[noise(3), NoiseRecord::GREEDY]);
        // The captured drafter program draws once per position and rank.
        assert_eq!(noise_launches(&host, from), 5 * shape.ranks() as usize);
        // Request 7 keeps one draft and the target's correction, so its
        // next cycle restarts two tokens later; the noise follows each item
        // to its new place in the batch.
        instance.commit_accepted(&[kept(7, 2), kept(9, 4)]).unwrap();
        execute(
            &mut instance,
            &host,
            &sampled(
                &[(9, 1, 7, DraftSamplingV1::GREEDY), (7, 0, 5, record(5))],
                3,
                true,
            ),
        );
        assert_noise(&host, &instance, &[NoiseRecord::GREEDY, noise(5)]);
        instance.commit_accepted(&[kept(9, 1), kept(7, 4)]).unwrap();
        // A batch without records drafts every item greedily.
        execute(
            &mut instance,
            &host,
            &sampled(
                &[(7, 0, 9, record(9)), (9, 1, 8, DraftSamplingV1::GREEDY)],
                3,
                false,
            ),
        );
        assert_noise(&host, &instance, &[NoiseRecord::GREEDY; 2]);
        instance.cancel().unwrap();
        for request in [7, 9] {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// The journal from create through two drafted steps of `items`.
fn drafted_journal(mode: DraftMode, items: [DraftSamplingV1; 2]) -> (Vec<String>, Vec<Sink>) {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (host, mut instance) = create_in(shape, mode);
    for (request, slot) in [(7, 0), (9, 1)] {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(request, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    let mut sinks = Vec::new();
    for (start, kept_rows) in [(3, 2), (5, 3)] {
        sinks.push(execute(
            &mut instance,
            &host,
            &sampled(&[(7, 0, start, items[0]), (9, 1, start, items[1])], 3, true),
        ));
        instance
            .commit_accepted(&[kept(7, kept_rows), kept(9, kept_rows)])
            .unwrap();
    }
    for request in [7, 9] {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let journal = host.journal.borrow().clone();
    (journal, sinks)
}

#[test]
fn greedy_items_keep_every_launch_of_the_greedy_drafter() {
    let greedy = [DraftSamplingV1::GREEDY; 2];
    let (coupled, coupled_sinks) = drafted_journal(DraftMode::Sampled, greedy);
    let (plain, plain_sinks) = drafted_journal(DraftMode::Greedy, greedy);
    // Both modes load the module; only the sampled mode launches it, once
    // per position and rank of each captured drafter program.
    assert_eq!(noise_launches_in(&plain), 0);
    assert_eq!(noise_launches_in(&coupled), 5 * 4);
    for (a, b) in coupled_sinks.iter().zip(&plain_sinks) {
        assert_eq!(a.draft_tokens(), b.draft_tokens());
    }
    let (coupled, plain) = (without_noise(&coupled), without_noise(&plain));
    assert!(coupled.iter().any(|line| line.starts_with("launch ")));
    // A captured graph's node count is the one other difference: each
    // rank's drafter graph holds its five noise launches.
    let (coupled_nodes, plain_nodes) = (captured(&coupled), captured(&plain));
    assert_eq!(coupled_nodes.len(), plain_nodes.len());
    let grown = coupled_nodes
        .iter()
        .zip(&plain_nodes)
        .filter(|(a, b)| a != b)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    assert_eq!(grown, [5; 4]);
    let (coupled, plain) = (without_counts(&coupled), without_counts(&plain));
    if let Some(at) = (0..coupled.len().max(plain.len())).find(|&i| coupled.get(i) != plain.get(i))
    {
        panic!(
            "line {at} differs:\n coupled {:?}\n greedy  {:?}",
            &coupled[at.saturating_sub(2)..(at + 3).min(coupled.len())],
            &plain[at.saturating_sub(2)..(at + 3).min(plain.len())]
        );
    }
}

fn is_noise_launch(line: &str) -> bool {
    line.starts_with("launch ") && line.contains(NOISE)
}

fn noise_launches_in(journal: &[String]) -> usize {
    journal.iter().filter(|line| is_noise_launch(line)).count()
}

/// Each captured graph's node count, in journal order.
fn captured(journal: &[String]) -> Vec<usize> {
    journal
        .iter()
        .filter_map(|line| line.strip_prefix("captured "))
        .map(|rest| rest.split_whitespace().nth(2).unwrap().parse().unwrap())
        .collect()
}

/// The journal with captured graphs' node counts removed.
fn without_counts(journal: &[String]) -> Vec<String> {
    journal
        .iter()
        .map(|line| match line.strip_prefix("captured ") {
            Some(rest) => {
                let words = rest.split_whitespace().collect::<Vec<_>>();
                format!("captured {} {}", words[0], words[1])
            }
            None => line.clone(),
        })
        .collect()
}

/// The journal without its noise launches.
fn without_noise(journal: &[String]) -> Vec<String> {
    journal
        .iter()
        .filter(|line| !is_noise_launch(line))
        .cloned()
        .collect()
}

/// The package's declared default reaches the drafter as the sampled mode
/// through the host's resolution, a profile or flag override of
/// `dspark_coupled_drafts` turns it off, and a create record without the
/// option keeps the greedy drafter.
#[test]
fn the_declared_option_selects_the_draft_mode() {
    use model_deepseek41_contract::dspark_coupling::COUPLED_DRAFTS;
    use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1};

    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let declared = ModelOptionsV1 {
        schema: CAPABILITY.into(),
        options: shape
            .geometry
            .model_options(shape.geometry.runtime_defaults()),
    };
    for (overrides, expected) in [
        (vec![], Some(DraftMode::Sampled)),
        (
            vec![(COUPLED_DRAFTS.to_owned(), "false".to_owned())],
            Some(DraftMode::Greedy),
        ),
        (
            vec![(COUPLED_DRAFTS.to_owned(), "true".to_owned())],
            Some(DraftMode::Sampled),
        ),
        (vec![], None),
    ] {
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.drafter.set(true);
        host.speculation.set(Some((5, 32)));
        let record = crate::config::tests::record(&shape, host.kv, 4, false);
        let record = match expected {
            // The test record names no DSpark option, as an older host's.
            None => record,
            Some(_) => {
                let resolved = declared.resolve(&overrides.into_iter().collect()).unwrap();
                crate::config::tests::with_coupled_drafts(
                    record,
                    resolved[COUPLED_DRAFTS].as_bool().unwrap(),
                )
            }
        };
        let mut instance = api()
            .create(
                HostServicesTableV1::new(Rc::new(MappedHost::new(Rc::clone(&host)))),
                &record,
            )
            .unwrap_or_else(|e| panic!("drafter create: {e:?} {:?}", host.logs.borrow()));
        let (_, mode) = crate::instance::dspark_noise(instance.handle()).unwrap();
        assert_eq!(mode, expected.unwrap_or(DraftMode::Greedy));
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
