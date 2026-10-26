//! AR steps read their row and index status back in one packet
//! (`instance::execute::ar_status`): every launch, replay, upload and memset
//! of a step is the separate reads' one, the step reads back once instead
//! of 16 times without synchronizing a stream, and a flagged word in any of
//! the 16 status parts refuses the step with the separate reads' message.
use super::*;

/// Index sources of Flash, then two row status planes on each of 4 ranks.
const SOURCES: usize = 8;
const RANKS: usize = 4;
const PARTS: usize = SOURCES + 2 * RANKS;

fn kind(line: &str) -> &str {
    line.split(' ').next().unwrap_or_default()
}

/// Each decode step's journal (1, 2, 8 and again 1 live rows, the last
/// replaying the first's graph), and its ordered rank streams.
fn decode_journals() -> (Vec<u32>, Vec<Vec<String>>) {
    {
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
        let streams = host
            .device
            .rank_streams
            .borrow()
            .values()
            .copied()
            .collect();
        (streams, steps)
    }
}

/// The logs of a one-row decode step whose status read back flagged word
/// `part` of the 16 (index sources in order, then each rank's two planes),
/// through the packet.
fn refusal(part: usize) -> Vec<String> {
    {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 4);
        step(
            &mut instance,
            &prompt(7, 0, 0, 3, true),
            &mut Views::default(),
        );
        let mut words = vec![0_u8; PARTS * 4];
        words[part * 4] = 1;
        host.injected.replace(Some(words));
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
    }
}

#[test]
fn a_flagged_status_part_refuses_the_step_with_the_separate_reads_message() {
    let layers = FLASH.shape.csa.index_sources;
    assert_eq!(layers.len(), SOURCES);
    for part in 0..PARTS {
        let old = refusal(part);
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
    let journal = || {
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
        let journal = host.device.journal.borrow()[from..].to_vec();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        journal
    };
    let separate = journal();
    assert!(separate.iter().filter(|l| kind(l) == "download").count() >= PARTS);
}

#[test]
fn ar_decode_steps_read_their_status_back_once() {
    let (streams, journals) = decode_journals();
    assert_eq!(streams.len(), RANKS);
    for (rows, journal) in [1_usize, 2, 8, 1].into_iter().zip(journals) {
        let count = |what: &str| journal.iter().filter(|line| kind(line) == what).count();
        assert!(count("launch") + count("replay") > 0);
        assert_eq!(count("download"), 1);
        // Legacy logits views still synchronize their owning rank after
        // the consolidated status readback.
        assert_eq!(count("synchronize"), rows.min(RANKS));
        let download = journal
            .iter()
            .position(|line| kind(line) == "download")
            .unwrap();
        let sync = journal
            .iter()
            .position(|line| kind(line) == "synchronize")
            .unwrap();
        assert!(download < sync);
        assert_eq!(count("copy"), SOURCES + 2);
        assert_eq!(count("peer-copy"), 2 * (RANKS - 1));
        let first_copy = journal
            .iter()
            .position(|line| kind(line) == "copy")
            .unwrap();
        let joins = &journal[first_copy - 2 * (RANKS - 1)..first_copy];
        for (pair, peer) in joins.chunks_exact(2).zip(&streams[1..]) {
            let record: Vec<_> = pair[0].split_whitespace().collect();
            let wait: Vec<_> = pair[1].split_whitespace().collect();
            assert_eq!(record[0], "record");
            assert_eq!(record[1], peer.to_string());
            assert_eq!(wait, ["wait", &streams[0].to_string(), record[2]]);
        }
        assert!(
            journal[first_copy..download]
                .iter()
                .filter(|line| matches!(kind(line), "copy" | "peer-copy"))
                .all(|line| line.split_whitespace().nth(1).unwrap() == streams[0].to_string())
        );
    }
}
