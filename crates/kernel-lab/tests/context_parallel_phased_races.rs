//! The phased group gather (`kernels_sm120::context_parallel::phased`) under
//! the races its waits exist for, with nothing on the host between
//! transfers. Each case runs `GENERATIONS` (a mutant `MUTANT_GENERATIONS`)
//! back-to-back transfers on one state (the same flags, epochs and receive
//! buffers), each from its own input words, distinct per generation, source
//! and destination. After
//! every transfer each rank's dependent consumer (`snapshot`, launched with
//! PDL behind the transfer) copies the rank's receive buffer on the device;
//! the host checks the snapshots only after the last generation.
//!
//! - Late sender: topology rank 1 runs a delay before each of its
//!   transfers. A receiver that returned before every source's data arrived
//!   would let its consumer copy the previous generation's words (the
//!   incoming-data wait, `phased_gather.cuh`).
//! - Late consumer: rank 1's consumer holds before it reads. A sender that
//!   wrote the next generation into rank 1's receive buffer before rank 1
//!   entered that transfer would overwrite words the consumer has not read
//!   yet (the reuse acknowledgement wait).
//!
//! Each case runs eagerly, then as one captured graph per rank replayed
//! once, in two groups of two and one group of every rank, for the
//! all-gather of one shard and the exchange of destination-major blocks.
//! The served entry (`comm_group_phased_lab_sm120`) and the lab copy of its
//! body run first, at every size, and must deliver every snapshot word.
//! The lab mutants (`kernel-lab/cuda/probes/phased_races.cuh`), the served
//! body with one wait switched off, then run in their own race and must be
//! caught by observed wrong words, in the eager runs and in the captured
//! replays separately (their sizes and gather and exchange cases counted
//! together). The lab entries' waits are bounded: a mutant without its
//! receive wait can run a call ahead of a late sender and overwrite the
//! acknowledgement that sender waits for, which would hang the process. A
//! wait that gives up is counted, and a case with any is a deadlock outcome
//! of its own, never a detection (its wrong words do not count); the lab
//! copy must have none. A mutant's execution error fails the test, it does
//! not count as detection. `mutants_are_the_served_body_minus_one_wait` checks the
//! mutants' source against the served body.
//!
//! Run with
//! `CHOPINFER_KERNEL_GPUS=0,1,2,3 cargo test -p kernel-lab --release --test context_parallel_phased_races`.

use kernel_core::contract::{Argument, ArgumentList};
use kernel_lab::families::context_parallel_phased::CUBIN as SERVED_CUBIN;
use kernel_lab::families::context_parallel_phased_races::{
    COPY, DELAY, MUTANTS, NO_ACKNOWLEDGEMENT_WAIT, NO_RECEIVE_WAIT, PROBES, SNAPSHOT, TIMEOUTS,
};
use kernel_lab::launch::{self, CollectiveFunction, CollectiveModules};
use kernel_lab::poison;
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi};
use kernels_sm120::context_parallel::group::{GatherState, GroupRank, gather_buffers};
use kernels_sm120::context_parallel::phased;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, LaunchConfig};

/// Transfers per case on one state.
const GENERATIONS: usize = 3;
/// Transfers per mutant case: the second generation is what overwrites a
/// late consumer's first, and a third would only let a mutant without its
/// acknowledgement wait run two calls ahead and overwrite a data flag a
/// peer still waits for (a deadlock outcome, not a detection).
const MUTANT_GENERATIONS: usize = 2;
/// The late sender's delay before each transfer and the late consumer's
/// hold before it reads: several transfers' worth at the cases' sizes.
const DELAY_NS: u64 = 3_000_000;
const HOLD_NS: u64 = 5_000_000;
/// The rank that is late.
const LATE: usize = 1;

/// Words per member: 1,024 rows of the served latent shape's partial blocks
/// (16 heads of 512 + 2 floats) for the exchange and of its absorbed
/// queries (16 heads of 512 BF16) for the all-gather.
const BLOCK_WORDS: usize = 1024 * 16 * 514;
const QUERY_WORDS: usize = 1024 * 16 * 512 / 2;

fn err(e: impl ToString) -> String {
    e.to_string()
}

fn refusal(e: kernel_core::contract::Refusal) -> String {
    e.0.to_string()
}

fn address(buffer: &DeviceBuffer) -> LabResult<u64> {
    buffer.device_address_at(0, buffer.byte_len()).map_err(err)
}

fn zeroed(session: &Session, bytes: usize) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes)?;
    buffer
        .copy_from_bytes(&vec![0; bytes], &session.stream)
        .map_err(err)?;
    Ok(buffer)
}

/// The words topology rank `source` sends group rank `destination` in
/// generation `generation` of run `run`: at every word index, distinct per
/// run, generation, source and destination (the tag fills bits 22 to 31,
/// the index is XORed in), so a stale word never matches.
fn word(run: usize, generation: usize, source: usize, destination: usize, index: usize) -> u32 {
    let tag = (((run * 8 + generation) * 8 + source) * 8 + destination) as u32;
    0x4000_0000 ^ (tag << 22) ^ index as u32
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    /// One shard to every member (input stride 0).
    Gather,
    /// Block d to group rank d (input stride = words).
    Exchange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Race {
    LateSender,
    LateConsumer,
}

/// Which entry a case runs: the served one or a lab entry of the mutants'
/// cubin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Variant {
    Served,
    Lab(&'static str),
}

fn kernel_arguments(list: &ArgumentList) -> Vec<KernelArgument<'_>> {
    list.0
        .iter()
        .map(|argument| match argument {
            Argument::DeviceBuffer(address) | Argument::U64(address) => {
                KernelArgument::U64(*address)
            }
            Argument::U32(value) => KernelArgument::U32(*value),
            Argument::DevicePointerArray(addresses) => {
                KernelArgument::DevicePointerArray(addresses)
            }
            other => unreachable!("a gather takes no {other:?}"),
        })
        .collect()
}

struct Case<'a> {
    sessions: &'a [Session],
    mode: Mode,
    race: Race,
    generations: usize,
    words: usize,
    groups: Vec<GroupRank>,
    /// Per generation, per rank: the input words.
    inputs: Vec<Vec<DeviceBuffer>>,
    /// Per generation, per rank: the consumer's copy of the receive buffer.
    snapshots: Vec<Vec<DeviceBuffer>>,
    receive: Vec<DeviceBuffer>,
    flags: Vec<DeviceBuffer>,
    epochs: Vec<DeviceBuffer>,
    transfer: CollectiveFunction,
    /// The lab entries' timeout count (`take_timeouts`) and each rank's
    /// word it lands in; none for the served entry, whose waits are not
    /// bounded.
    timeouts: Option<(CollectiveFunction, Vec<DeviceBuffer>)>,
    delay: Vec<Function>,
    snapshot: Vec<Function>,
    _modules: CollectiveModules,
}

impl<'a> Case<'a> {
    fn new(
        sessions: &'a [Session],
        variant: Variant,
        mode: Mode,
        race: Race,
        size: usize,
    ) -> LabResult<Self> {
        let ranks = sessions.len();
        let words = match mode {
            Mode::Gather => QUERY_WORDS,
            Mode::Exchange => BLOCK_WORDS,
        };
        let groups = (0..ranks)
            .map(|rank| GroupRank::of(rank as u32, ranks as u32, size as u32).map_err(refusal))
            .collect::<LabResult<Vec<_>>>()?;
        // The bundles' state sizes for these words.
        let buffers = gather_buffers(size as u32, words as u64).map_err(refusal)?;
        phased::check_state(size as u32, buffers.flag_bytes, buffers.epoch_bytes)
            .map_err(refusal)?;
        let blocks = if mode == Mode::Exchange { size } else { 1 };
        let generations = match variant {
            Variant::Lab(symbol) if symbol != COPY => MUTANT_GENERATIONS,
            _ => GENERATIONS,
        };
        let per_rank = |bytes: usize| {
            sessions
                .iter()
                .map(|session| session.allocate(bytes))
                .collect::<LabResult<Vec<_>>>()
        };
        let (cubin, symbol) = match variant {
            Variant::Served => (SERVED_CUBIN, phased::SYMBOL),
            Variant::Lab(symbol) => (MUTANTS, symbol),
        };
        let modules = CollectiveModules::load(sessions, cubin)?;
        let probes = |symbol: &str| {
            sessions
                .iter()
                .map(|session| session.module(PROBES)?.function(symbol).map_err(err))
                .collect::<LabResult<Vec<_>>>()
        };
        Ok(Self {
            sessions,
            mode,
            race,
            generations,
            words,
            groups,
            inputs: (0..generations)
                .map(|_| per_rank(blocks * words * 4))
                .collect::<LabResult<_>>()?,
            snapshots: (0..generations)
                .map(|_| per_rank(size * words * 4))
                .collect::<LabResult<_>>()?,
            receive: per_rank(buffers.output_bytes as usize)?,
            flags: sessions
                .iter()
                .map(|session| zeroed(session, buffers.flag_bytes as usize))
                .collect::<LabResult<_>>()?,
            epochs: sessions
                .iter()
                .map(|session| zeroed(session, buffers.epoch_bytes as usize))
                .collect::<LabResult<_>>()?,
            transfer: modules.function(symbol)?,
            timeouts: match variant {
                Variant::Served => None,
                Variant::Lab(_) => Some((
                    modules.function(TIMEOUTS)?,
                    sessions
                        .iter()
                        .map(|session| zeroed(session, 4))
                        .collect::<LabResult<_>>()?,
                )),
            },
            delay: probes(DELAY)?,
            snapshot: probes(SNAPSHOT)?,
            _modules: modules,
        })
    }

    fn size(&self) -> usize {
        self.groups[0].size as usize
    }

    /// Stages run `run`'s inputs and poisons every receive buffer and
    /// snapshot, with every rank idle.
    fn prepare(&self, run: usize) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            for generation in 0..self.generations {
                let destinations = match self.mode {
                    Mode::Gather => vec![0],
                    Mode::Exchange => (0..self.size()).collect(),
                };
                let staged = destinations
                    .into_iter()
                    .flat_map(|destination| {
                        (0..self.words).map(move |i| word(run, generation, rank, destination, i))
                    })
                    .flat_map(u32::to_ne_bytes)
                    .collect::<Vec<_>>();
                self.inputs[generation][rank]
                    .copy_from_bytes(&staged, &session.stream)
                    .map_err(err)?;
                poison::poison(session, &[&self.snapshots[generation][rank]])?;
            }
            poison::poison(session, &[&self.receive[rank]])?;
        }
        poison::synchronize(self.sessions)
    }

    /// Every generation on rank `only` (or every rank): the late sender's
    /// delay, the transfer, then each rank's consumer.
    fn issue(&self, only: Option<usize>) -> LabResult<()> {
        let states = (0..self.sessions.len())
            .map(|rank| {
                Ok(GatherState {
                    output: address(&self.receive[rank])?,
                    flags: address(&self.flags[rank])?,
                    epoch: address(&self.epochs[rank])?,
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        let ranks = (0..self.sessions.len()).filter(|&rank| only.is_none_or(|only| only == rank));
        let ranks = ranks.collect::<Vec<_>>();
        let delay =
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?;
        let consume =
            launch::simple_config((128, 1, 1), (256, 1, 1), 0, LaunchConfig::NO_CLUSTER, true)?;
        let receive_words = (self.size() * self.words) as u64;
        for generation in 0..self.generations {
            if self.race == Race::LateSender && ranks.contains(&LATE) {
                self.delay[LATE]
                    .launch(
                        &self.sessions[LATE].stream,
                        delay,
                        &[KernelArgument::U64(DELAY_NS)],
                    )
                    .map_err(err)?;
            }
            let arguments = (0..self.sessions.len())
                .map(|rank| {
                    let input = address(&self.inputs[generation][rank])?;
                    let words = self.words as u64;
                    match self.mode {
                        Mode::Gather => {
                            phased::gather_arguments(self.groups[rank], &states, input, words)
                        }
                        Mode::Exchange => {
                            phased::exchange_arguments(self.groups[rank], &states, input, words)
                        }
                    }
                    .map_err(refusal)
                })
                .collect::<LabResult<Vec<_>>>()?;
            launch::submit_collective(
                self.sessions,
                &self.transfer,
                launch::config(phased::plan())?,
                only,
                |rank| Ok(kernel_arguments(&arguments[rank])),
            )?;
            for &rank in &ranks {
                let hold = if self.race == Race::LateConsumer && rank == LATE {
                    HOLD_NS
                } else {
                    0
                };
                self.snapshot[rank]
                    .launch(
                        &self.sessions[rank].stream,
                        consume,
                        &[
                            KernelArgument::Buffer(&self.receive[rank]),
                            KernelArgument::Buffer(&self.snapshots[generation][rank]),
                            KernelArgument::U64(receive_words),
                            KernelArgument::U64(hold),
                        ],
                    )
                    .map_err(err)?;
            }
        }
        Ok(())
    }

    /// Words of every snapshot that differ from what run `run`'s generation
    /// delivers.
    fn wrong(&self, run: usize) -> LabResult<u64> {
        for session in self.sessions {
            session.stream.synchronize().map_err(err)?;
        }
        let mut wrong = 0;
        for (rank, session) in self.sessions.iter().enumerate() {
            let group = self.groups[rank];
            for generation in 0..self.generations {
                let bytes = poison::read(session, &self.snapshots[generation][rank])?;
                for (slot, member) in group.members().enumerate() {
                    let destination = match self.mode {
                        Mode::Gather => 0,
                        Mode::Exchange => group.rank as usize,
                    };
                    for i in 0..self.words {
                        let at = (slot * self.words + i) * 4;
                        let found = u32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
                        if found != word(run, generation, member as usize, destination, i) {
                            wrong += 1;
                        }
                    }
                }
            }
        }
        Ok(wrong)
    }

    /// The bounded waits the lab entries gave up on since the last call,
    /// summed over ranks (cleared on read), with every rank idle.
    fn timeouts(&self) -> LabResult<u64> {
        let Some((take, out)) = &self.timeouts else {
            return Ok(0);
        };
        let config =
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?;
        launch::submit_collective(self.sessions, take, config, None, |rank| {
            Ok(vec![KernelArgument::Buffer(&out[rank])])
        })?;
        let mut total = 0;
        for (rank, session) in self.sessions.iter().enumerate() {
            session.stream.synchronize().map_err(err)?;
            let bytes = poison::read(session, &out[rank])?;
            total += u64::from(u32::from_ne_bytes(bytes[..4].try_into().unwrap()));
        }
        Ok(total)
    }

    /// Wrong snapshot words and timeouts of an eager run, then of a
    /// captured replay.
    fn run(&self) -> LabResult<Outcome> {
        self.prepare(0)?;
        self.timeouts()?;
        self.issue(None)?;
        let eager = (self.wrong(0)?, self.timeouts()?);
        let graphs = RankReplay::new(self.sessions, |rank, _| self.issue(Some(rank)))?;
        self.prepare(1)?;
        graphs.launch(self.sessions)?;
        let replayed = (self.wrong(1)?, self.timeouts()?);
        Ok(Outcome {
            wrong: [eager.0, replayed.0],
            timeouts: [eager.1, replayed.1],
        })
    }
}

/// The group sizes a run covers: two groups of two (at four or more even
/// ranks) and one group of every rank (up to eight).
fn sizes(ranks: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    if ranks >= 4 && ranks.is_multiple_of(2) {
        sizes.push(2);
    }
    if ranks <= 8 {
        sizes.push(ranks);
    }
    sizes
}

/// One case's wrong snapshot words and gave-up waits, eager then replayed.
#[derive(Clone, Copy, Debug)]
struct Outcome {
    wrong: [u64; 2],
    timeouts: [u64; 2],
}

/// The outcome of every (size, mode, race) case of `variant` in `races`,
/// sizes outermost.
fn cases(
    sessions: &[Session],
    variant: Variant,
    races: &[Race],
) -> LabResult<Vec<(String, Outcome)>> {
    let mut out = Vec::new();
    for size in sizes(sessions.len()) {
        for mode in [Mode::Exchange, Mode::Gather] {
            for &race in races {
                let outcome = Case::new(sessions, variant, mode, race, size)?.run()?;
                let label = format!("{variant:?} {mode:?} {race:?} size {size}");
                println!(
                    "{label}: wrong words eager {}, replayed {}; timeouts eager {}, replayed {}",
                    outcome.wrong[0], outcome.wrong[1], outcome.timeouts[0], outcome.timeouts[1]
                );
                out.push((label, outcome));
            }
        }
    }
    Ok(out)
}

/// The served entry and the lab copy at every size first: every snapshot
/// word delivered, no lab wait given up. Then each mutant in its own race:
/// observed wrong words in the eager runs and in the captured replays,
/// each counted only from cases of that mode in which no bounded wait gave
/// up. A case with timeouts is a deadlock outcome, reported on its own and
/// never counted as a detection; an execution error fails the test.
#[test]
fn served_transfers_deliver_and_mutants_are_caught() {
    gpu_test_multi(4 << 30, |sessions| {
        if sessions.len() < 2 || LATE >= sessions.len() {
            return Err("the race cases need at least two ranks".into());
        }
        let races = [Race::LateSender, Race::LateConsumer];
        for variant in [Variant::Served, Variant::Lab(COPY)] {
            for (label, outcome) in cases(sessions, variant, &races)? {
                if outcome.wrong != [0, 0] || outcome.timeouts != [0, 0] {
                    return Err(format!("{label}: {outcome:?}"));
                }
            }
        }
        for (symbol, race) in [
            (NO_RECEIVE_WAIT, Race::LateSender),
            (NO_ACKNOWLEDGEMENT_WAIT, Race::LateConsumer),
        ] {
            let all = cases(sessions, Variant::Lab(symbol), &[race])?;
            // Eager and captured replay each must catch the mutant on their
            // own; sizes and gather/exchange cases may share the count.
            let mut caught = [0_u64; 2];
            let mut deadlocked = [0_usize; 2];
            for (_, outcome) in &all {
                for mode in 0..2 {
                    if outcome.timeouts[mode] == 0 {
                        caught[mode] += outcome.wrong[mode];
                    } else {
                        deadlocked[mode] += 1;
                    }
                }
            }
            println!(
                "{symbol}: wrong words without timeouts in its {race:?} cases: eager {}, \
                 replayed {}; cases with a deadlock outcome: eager {}, replayed {}",
                caught[0], caught[1], deadlocked[0], deadlocked[1]
            );
            for (mode, count) in ["eager", "captured replay"].into_iter().zip(caught) {
                if count == 0 {
                    return Err(format!(
                        "{symbol}: no wrong word observed in its {race:?} cases in {mode} \
                         without a timeout; the race cases do not exercise its wait there"
                    ));
                }
            }
        }
        Ok(())
    })
    .expect("served transfers deliver and mutants are caught");
}

/// The lab mutants are the served gather body line for line apart from
/// the lines marked `// mutation switch`.
#[test]
fn mutants_are_the_served_body_minus_one_wait() {
    fn body(path: &str, skip_switches: bool) -> Vec<String> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let text = std::fs::read_to_string(root.join(path)).unwrap();
        let start = text.find("__device__ void gather(").expect("a gather body");
        let lines = text[start..]
            .lines()
            .take_while(|line| *line != "}")
            .filter(|line| !(skip_switches && line.ends_with("// mutation switch")))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert!(lines.len() > 20, "{path}: {} lines", lines.len());
        lines
    }
    let served = body(
        "kernels-sm120/cuda/context_parallel/phased_gather.cuh",
        false,
    );
    let mutants = body("kernel-lab/cuda/probes/phased_races.cuh", true);
    assert_eq!(mutants, served);
    let switches = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda/probes/phased_races.cuh"),
    )
    .unwrap()
    .lines()
    .filter(|line| line.ends_with("// mutation switch"))
    .count();
    assert_eq!(switches, 2);
    // The body's unqualified waits resolve to the lab header's bounded
    // `wait_system`, declared in the mutants' own namespace.
    assert!(served.iter().any(|line| line.contains(" wait_system(")));
    let header = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda/probes/phased_races.cuh"),
    )
    .unwrap();
    let namespace = header
        .find("namespace chop::probe::phased_races {")
        .unwrap();
    let bounded = header
        .find("__device__ __forceinline__ void wait_system(")
        .unwrap();
    let body = header.find("__device__ void gather(").unwrap();
    assert!(namespace < bounded && bounded < body);
}
