//! The phased all-reduce's quiescent reset: flags carry the call's tagged
//! epoch and stay set, so a tag repeats `engines::PHASED_TAG_PERIOD` calls
//! later. An instance therefore zeroes every rank's phased flags and epoch
//! (the BF16 partials' state and the MoE sublayers' F32 state) while every
//! rank is idle before a run past `engines::PHASED_RESET_RUNS` runs since
//! the last reset; the tests lower the bound.

use super::*;
use model_glm5next_contract::sm120::engines as selected;

/// The journal's lines before the first launch or replay from `from` on
/// that synchronize a stream or zero a span.
fn before_launches(host: &FakeHost, from: usize) -> Vec<String> {
    host.device.journal.borrow()[from..]
        .iter()
        .take_while(|line| !line.starts_with("launch ") && !line.starts_with("replay "))
        .filter(|line| line.starts_with("synchronize ") || line.starts_with("zero "))
        .cloned()
        .collect()
}

/// What a reset journals, from the create journal: every rank's stream
/// waited for, then each rank's zeroing of its phased flags (33864 bytes:
/// the phased entries' words and the turns entry's) and its epoch, then of
/// its F32 state's flags (`engines::rounded_phased_state_bytes`) and epoch, the
/// spans create zeroed (each epoch right after its flags), then every
/// stream waited for again.
fn reset_lines(host: &FakeHost) -> Vec<String> {
    let created = host.device.journal.borrow().clone();
    let zero = |length: u64| {
        let length = format!("length: {length} }}");
        created
            .iter()
            .enumerate()
            .filter(move |(_, line)| line.starts_with("zero ") && line.ends_with(&length))
            .flat_map(|(at, line)| [line.clone(), created[at + 1].clone()])
            .collect::<Vec<_>>()
    };
    let bf16 = zero(33864);
    let f32 = zero(
        selected::rounded_phased_state_bytes(Shape::TP4)
            .unwrap()
            .flags,
    );
    assert_eq!((bf16.len(), f32.len()), (8, 8), "{bf16:?} {f32:?}");
    let state = bf16
        .chunks(2)
        .zip(f32.chunks(2))
        .flat_map(|(bf16, f32)| [bf16, f32].concat())
        .collect::<Vec<_>>();
    assert!(
        state
            .iter()
            .skip(1)
            .step_by(2)
            .all(|line| line.ends_with("length: 4 }"))
    );
    let waits = state
        .iter()
        .step_by(4)
        .map(|line| format!("synchronize {}", line.split(' ').nth(1).unwrap()))
        .collect::<Vec<_>>();
    [waits.clone(), state, waits].concat()
}

/// The resets the instance logged.
fn logged(host: &FakeHost) -> u64 {
    host.logs
        .borrow()
        .iter()
        .filter(|line| line.contains("zeroed every rank's phased all-reduce"))
        .count() as u64
}

/// At a bound of three runs, seven decode steps alternating one item (one
/// row: the low-latency entry) and two (two rows: the phased one-shot),
/// each one run in the scope, captured at its bucket's first use and
/// replayed after: the fourth and the seventh step first run the reset
/// (`reset_lines`) before any launch or replay; no other step does. Each
/// rank's two-row program stays within `engines::PHASED_CALLS_PER_RUN`
/// phased calls of the BF16 partials and the F32 partials together.
#[test]
fn every_rank_zeroes_the_phased_state_while_idle_past_the_bound() {
    const RUNS: u64 = 3;
    let host = FakeHost::new((0, 0));
    let mut instance = create(&host);
    let handle = instance.handle();
    crate::instance::phased_reset_runs(handle, RUNS);
    // The projection stages need the package's stores (`create_one_slot`).
    crate::instance::run_lowered_stages_alone(handle);
    let reset = reset_lines(&host);
    crate::instance::seed(handle, 7, 0, 16);
    crate::instance::seed(handle, 8, 1, 16);
    for n in 1..=7u64 {
        let start = host.device.journal.borrow().len();
        let items: &[(u64, u64, u64, &[u32])] = if n % 2 == 1 {
            &[(7, 0, 15 + n, &[1])]
        } else {
            &[(7, 0, 15 + n, &[1]), (8, 1, 15 + n / 2, &[2])]
        };
        instance
            .execute(&step(items), &mut NoRows::default())
            .unwrap_or_else(|error| panic!("step {n}: {error:?} {:?}", host.logs.borrow()));
        instance
            .commit()
            .unwrap_or_else(|error| panic!("step {n}: {error:?} {:?}", host.logs.borrow()));
        let resets = n % RUNS == 1 && n > 1;
        let expected = if resets { reset.clone() } else { Vec::new() };
        assert_eq!(before_launches(&host, start), expected, "step {n}");
        assert_eq!(logged(&host), (n - 1) / RUNS, "step {n}");
        if n == 2 {
            // The captured two-row programs' phased calls per stream.
            let mut calls = BTreeMap::<String, u64>::new();
            for line in &host.device.journal.borrow()[start..] {
                let mut words = line.split(' ');
                if words.next() == Some("launch") {
                    let stream = words.next().unwrap().to_owned();
                    let symbol = words.next().unwrap();
                    if selected::PHASED_SYMBOLS.contains(&symbol)
                        || selected::PHASED_ROUND_SYMBOLS.contains(&symbol)
                    {
                        *calls.entry(stream).or_default() += 1;
                    }
                }
            }
            assert_eq!(calls.len(), 4, "{calls:?}");
            assert!(
                calls
                    .values()
                    .all(|&count| count > 0 && count <= selected::PHASED_CALLS_PER_RUN)
            );
        }
    }
    instance.destroy().unwrap();
}

/// At a bound of two runs, prefills count, as their steps reduce through
/// the phased entries too: a prompt and its first decode
/// step (two runs), then another request's prompt, which resets first,
/// then its first decode step (no reset).
#[test]
fn prefills_count_toward_the_phased_reset() {
    let host = FakeHost::new((0, 0));
    let mut instance = create_one_slot(&host);
    crate::instance::phased_reset_runs(instance.handle(), 2);
    let reset = reset_lines(&host);
    let run = |instance: &mut BundleInstanceV1, request: &[u8], label: &str| {
        let start = host.device.journal.borrow().len();
        instance
            .execute(request, &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{label}: {error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        before_launches(&host, start)
    };
    let none = Vec::<String>::new();
    assert_eq!(
        run(&mut instance, &prompt_chunk(7, 0, 16, true), "prompt 7"),
        none
    );
    assert_eq!(
        run(&mut instance, &step(&[(7, 0, 16, &[1])]), "step 7"),
        none
    );
    instance.retire_request(7).unwrap();
    assert_eq!(logged(&host), 0);
    assert_eq!(
        run(&mut instance, &prompt_chunk(8, 0, 16, true), "prompt 8"),
        reset
    );
    assert_eq!(logged(&host), 1);
    assert_eq!(
        run(&mut instance, &step(&[(8, 0, 16, &[1])]), "step 8"),
        none
    );
    assert_eq!(logged(&host), 1);
    instance.destroy().unwrap();
}
