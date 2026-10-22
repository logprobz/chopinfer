//! AR steps read their row and index status back in one packet
//! (`instance::execute::ar_status`): every launch, replay, upload and memset
//! of a step is the separate reads' one, the step reads back once instead
//! of 16 times without synchronizing a stream, and a flagged word in any of
//! the 16 status parts refuses the step with the separate reads' message.
use super::*;
use crate::instance::execute::ar_status::with_enabled;

/// Index sources of Flash, then two row status planes on each of 4 ranks.
const SOURCES: usize = 8;
const RANKS: usize = 4;
const PARTS: usize = SOURCES + 2 * RANKS;

fn kind(line: &str) -> &str {
    line.split(' ').next().unwrap_or_default()
}

/// The status reads' lines: the packet's joins and copies, and the
/// separate reads' downloads and synchronizations.
fn status_line(line: &str) -> bool {
    matches!(
        kind(line),
        "download" | "synchronize" | "record" | "wait" | "copy" | "peer-copy"
    )
}

/// Each decode step's journal (1, 2, 8 and again 1 live rows, the last
/// replaying the first's graph) of a Flash TP4 instance with the packet on
/// or off.
fn decode_journals(packet: bool) -> Vec<Vec<String>> {
    with_enabled(packet, || {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 8);
        for slot in 0..8 {
            step(
                &mut instance,
                &prompt(slot + 1, slot, 0, 1, true),
                &mut Views::default(),
            );
        }
        let mut ends = [1_u64; 8];
        let mut steps = Vec::new();
        for rows in [1_u32, 2, 8, 1] {
            let from = host.device.journal.borrow().len();
            let items = (0..rows)
                .map(|slot| (u64::from(slot) + 1, u64::from(slot), ends[slot as usize], 7))
                .collect::<Vec<_>>();
            step(&mut instance, &decode(&items), &mut Views::default());
            steps.push(host.device.journal.borrow()[from..].to_vec());
            for end in &mut ends[..rows as usize] {
                *end += 1;
            }
        }
        for id in 1..=8 {
            instance.retire_request(id).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        steps
    })
}

/// Lines with each stream, graph and buffer handle renumbered by first
/// use: the packet's buffer and events take fake handles at create, so
/// later handles' raw numbers shift between runs.
fn canonical(lines: &[String]) -> Vec<String> {
    let mut handles = BTreeMap::<String, usize>::new();
    let mut number = |key: String| {
        let next = handles.len();
        *handles.entry(key).or_insert(next)
    };
    lines
        .iter()
        .map(|line| {
            let mut words = line.splitn(3, ' ');
            let kind = words.next().unwrap_or_default();
            let stream = words.next().unwrap_or_default();
            let mut rest = words.next().unwrap_or_default().to_owned();
            if matches!(kind, "replay" | "captured") {
                let (graph, tail) = rest.split_once(' ').unwrap_or((rest.as_str(), ""));
                rest = format!("g#{} {tail}", number(format!("g{graph}")));
            }
            let mut parts = rest.split("raw: ");
            let stream = number(format!("s{stream}"));
            let mut out = format!("{kind} s#{stream} {}", parts.next().unwrap());
            for part in parts {
                let digits = part.chars().take_while(char::is_ascii_digit).count();
                let raw = number(format!("r{}", &part[..digits]));
                out.push_str(&format!("raw: #{raw}{}", &part[digits..]));
            }
            out
        })
        .collect()
}

#[test]
fn ar_decode_steps_read_their_status_back_once_with_the_same_work() {
    let separate = decode_journals(false);
    let packet = decode_journals(true);
    assert_eq!(separate.len(), packet.len());
    for (step, (old, new)) in separate.iter().zip(&packet).enumerate() {
        let work = |lines: &[String]| {
            canonical(
                &lines
                    .iter()
                    .filter(|l| !status_line(l))
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        };
        let (old_work, new_work) = (work(old), work(new));
        assert!(
            old_work
                .iter()
                .any(|l| kind(l) == "launch" || kind(l) == "replay"),
            "step {step}"
        );
        if let Some(at) =
            (0..old_work.len().max(new_work.len())).find(|&i| old_work.get(i) != new_work.get(i))
        {
            panic!(
                "step {step} differs at {at}:\n{:?}\n{:?}",
                old_work.get(at),
                new_work.get(at)
            );
        }
        let count = |lines: &[String], what: &str| lines.iter().filter(|l| kind(l) == what).count();
        // The separate reads: one download per index source and plane,
        // after one synchronization of rank 0 for the index sources and one
        // of each rank for its planes, and one synchronization after each.
        assert_eq!(
            count(old, "download"),
            count(new, "download") + PARTS - 1,
            "step {step}"
        );
        assert_eq!(
            count(old, "synchronize"),
            count(new, "synchronize") + 1 + SOURCES + RANKS * 3,
            "step {step}"
        );
        // The packet: rank 0 waits on each peer's work, copies rank 0's
        // index sources and planes and pulls each peer's planes.
        assert_eq!(
            count(new, "record"),
            count(old, "record") + RANKS - 1,
            "step {step}"
        );
        assert_eq!(
            count(new, "wait"),
            count(old, "wait") + RANKS - 1,
            "step {step}"
        );
        assert_eq!(
            count(new, "copy"),
            count(old, "copy") + SOURCES + 2,
            "step {step}"
        );
        assert_eq!(
            count(new, "peer-copy"),
            count(old, "peer-copy") + 2 * (RANKS - 1),
            "step {step}"
        );
    }
}

/// The logs of a one-row decode step whose status read back flagged word
/// `part` of the 16 (index sources in order, then each rank's two planes),
/// with the packet on or off.
fn refusal(packet: bool, part: usize) -> Vec<String> {
    with_enabled(packet, || {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 4);
        step(
            &mut instance,
            &prompt(7, 0, 0, 3, true),
            &mut Views::default(),
        );
        if packet {
            let mut words = vec![0_u8; PARTS * 4];
            words[part * 4] = 1;
            host.injected.replace(Some(words));
        } else {
            host.injected.replace(Some(1_u32.to_le_bytes().to_vec()));
            host.injected_after.set(part);
        }
        let from = host.logs.borrow().len();
        let error = instance
            .execute(&decode(&[(7, 0, 3, 1)]), &mut Views::default())
            .unwrap_err();
        assert_eq!(error, BundleAbiError::Status(BundleStatusV1::BundleError));
        assert!(host.injected.borrow().is_none(), "part {part} was not read");
        let logs = host.logs.borrow()[from..].to_vec();
        assert!(instance.commit().is_err());
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        logs
    })
}

#[test]
fn a_flagged_status_part_refuses_the_step_with_the_separate_reads_message() {
    let layers = FLASH.shape.csa.index_sources;
    assert_eq!(layers.len(), SOURCES);
    for part in 0..PARTS {
        let old = refusal(false, part);
        let new = refusal(true, part);
        assert_eq!(old, new, "part {part}");
        let expected = if let Some(layer) = layers.get(part) {
            format!(
                "the index selection of layer {layer} met a non-finite score in row 0 (request Some(7))"
            )
        } else {
            format!(
                "a quantized input on rank {} was invalid in row 0 (request Some(7))",
                (part - SOURCES) / 2
            )
        };
        assert!(
            old.iter().any(|l| l.contains(&expected)),
            "part {part}: {old:?}"
        );
    }
}

/// A prefill of more rows than a packet holds keeps the separate reads:
/// its journal is the packet-free one.
#[test]
fn a_step_wider_than_the_packet_keeps_the_separate_reads() {
    let journal = |packet: bool| {
        with_enabled(packet, || {
            let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
            let host = FakeHost::new(shape, KV_LAYOUTS[0]);
            let mut instance = create(&host, 4);
            let rows = crate::instance::execute::ar_status::ROWS;
            let from = host.device.journal.borrow().len();
            step(
                &mut instance,
                &prompt(7, 0, 0, u64::from(rows) + 8, true),
                &mut Views::default(),
            );
            let journal = canonical(&host.device.journal.borrow()[from..]);
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
            journal
        })
    };
    let separate = journal(false);
    assert!(separate.iter().filter(|l| kind(l) == "download").count() >= PARTS);
    assert_eq!(journal(true), separate);
}
