//! Rank-ordered F32 all-reduce of large row sets in one launch
//! (`cuda/comm/allreduce_phased_f32.cuh`): a reduce-scatter then an
//! all-gather, each sent to one destination at a time from `senders` CTAs,
//! with one release flag per block. Every output element is the F32 sum of
//! the ranks' partials in rank order from zero, the bits of the phased
//! pair (`chop_comm_reduce_scatter_f32_v1` then
//! `chop_comm_allgather_scattered_f32_v1`). [`ROUND_SYMBOL`] rounds each sum
//! once to BF16 at its owner (round to nearest even) and gathers the BF16
//! rows instead. The one-shot entries ([`ONE_SHOT_SYMBOL`],
//! [`ONE_SHOT_ROUND_SYMBOL`]) push every rank's whole partial to each peer
//! in turn and sum every element on every rank, for calls of up to
//! [`ONE_SHOT_MAX_ELEMENTS`]: half the phases, for small calls. All four
//! share one state and may follow each other in any order.
//!
//! Every CTA of a launch spins on flags other CTAs or peers release, so the
//! whole grid must be resident at once on every rank.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, LaunchPlan, MAX_COMM_RANKS, Refusal, Validate,
};

/// The entries' header, relative to the CUDA source root.
pub const HEADER: &str = "comm/allreduce_phased_f32.cuh";
/// Threads of every CTA.
pub const THREADS: u32 = 256;
/// Elements of one granule: owner spans and blocks are whole granules, so
/// a call's element count and a live row's elements are multiples of it.
pub const GRANULE: u64 = 1024;
/// Most sending CTAs of a launch.
pub const MAX_SENDERS: u32 = 64;
/// Most blocks of one owner span (`kMaxBlocks`); a call takes about one
/// per 64 granules, and at least one per sender.
pub const MAX_BLOCKS: u64 = 256;
/// Flag words of one rank's state (`kFlagWords`): scatter and gather flags
/// per source and block, reduced flags per block, the sender barrier and
/// the arrival counter.
pub const FLAG_WORDS: u64 = 4 * MAX_COMM_RANKS as u64 * MAX_BLOCKS + MAX_BLOCKS + 2;
/// The one-shot entries' largest call (`kOneShotMaxElements`).
pub const ONE_SHOT_MAX_ELEMENTS: u64 = 256 * GRANULE;
/// Elements of the one-shot's inbox region at the start of every inbox,
/// `[parity][source][ONE_SHOT_MAX_ELEMENTS]` (`kOneShotRegion`).
pub const ONE_SHOT_REGION: u64 = 2 * MAX_COMM_RANKS as u64 * ONE_SHOT_MAX_ELEMENTS;
/// Sending CTAs and workers of [`plan`] until the lab's timings choose
/// them.
pub const DEFAULT_SENDERS: u32 = 16;
pub const DEFAULT_WORKERS: u32 = 32;

/// The F32 sums into every rank's F32 output.
pub const SYMBOL: &str = "chop_comm_allreduce_phased_f32_v1";
/// The F32 sums rounded once to BF16 into every rank's BF16 output.
pub const ROUND_SYMBOL: &str = "chop_comm_allreduce_phased_round_f32_v1";
/// The one-shot protocol's F32 and rounded entries.
pub const ONE_SHOT_SYMBOL: &str = "chop_comm_allreduce_phased_oneshot_f32_v1";
pub const ONE_SHOT_ROUND_SYMBOL: &str = "chop_comm_allreduce_phased_oneshot_round_f32_v1";

/// The lab's mutants (`lab_cubin_specs`): each define, and the tag its
/// cubin and symbols carry.
pub const MUTANTS: [(&str, &str); 5] = [
    (
        "no_scatter_wait",
        "CHOP_COMM_PHASED_F32_TEST_NO_SCATTER_WAIT",
    ),
    ("no_gather_wait", "CHOP_COMM_PHASED_F32_TEST_NO_GATHER_WAIT"),
    ("reversed", "CHOP_COMM_PHASED_F32_TEST_REVERSED"),
    ("late_wait", "CHOP_COMM_PHASED_F32_TEST_LATE_WAIT"),
    ("shared_parity", "CHOP_COMM_PHASED_F32_TEST_SHARED_PARITY"),
];
/// The served cubin's name in the lab.
pub const LAB_CUBIN: &str = "comm_phased_f32_sm120";

/// The lab cubin of the mutant `tag`.
pub fn mutant_cubin(tag: &str) -> String {
    format!("comm_phased_f32_{tag}_sm120")
}

/// `symbol` in the mutant `tag`'s cubin.
pub fn mutant_symbol(symbol: &str, tag: &str) -> String {
    symbol.replacen("chop_comm_", &format!("chop_comm_test_{tag}_"), 1)
}

fn params(output: &str) -> Vec<EntryParam> {
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let array = |name: &str, cxx_type: &str| EntryParam::PointerArray {
        name: name.into(),
        slots: MAX_COMM_RANKS as u8,
        cxx_type: cxx_type.into(),
    };
    let u32 = |name: &str| EntryParam::U32 { name: name.into() };
    vec![
        pointer("input", "const float"),
        pointer("output", output),
        array("inboxes", "float"),
        array("outputs", output),
        array("flags", "unsigned int"),
        pointer("epoch", "unsigned int"),
        EntryParam::U64 {
            name: "elements".into(),
        },
        u32("row_elements"),
        pointer("live_rows", "const uint32_t"),
        u32("rank"),
        u32("ranks"),
        u32("senders"),
    ]
}

/// The four entries.
pub fn entries() -> Vec<EntrySpec> {
    [
        (SYMBOL, "float", false),
        (ROUND_SYMBOL, "uint16_t", false),
        (ONE_SHOT_SYMBOL, "float", true),
        (ONE_SHOT_ROUND_SYMBOL, "uint16_t", true),
    ]
    .into_iter()
    .map(|(symbol, output, one_shot)| {
        let body = format!(
            "chop::comm::phased_f32::allreduce<{output},{one_shot}>(input, output, inboxes, outputs, flags, epoch, elements, row_elements, live_rows, rank, ranks, senders);"
        );
        EntrySpec::new(symbol.into(), params(output), body, THREADS, 1)
    })
    .collect()
}

/// Whether `symbol` is a one-shot entry.
pub fn is_one_shot(symbol: &str) -> bool {
    symbol.contains("_oneshot_")
}

/// A cubin of both entries named `name`.
pub fn cubin_spec(name: String) -> CubinSpec {
    CubinSpec {
        name,
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![("CHOP_COMM_MAX_RANKS".into(), MAX_COMM_RANKS.to_string())],
        entries: entries(),
    }
}

/// The lab's cubins: [`LAB_CUBIN`], and one cubin per [`MUTANTS`] entry
/// whose symbols carry the mutant's tag ([`mutant_symbol`]).
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let mut specs = vec![cubin_spec(LAB_CUBIN.into())];
    for (tag, define) in MUTANTS {
        let mut spec = cubin_spec(mutant_cubin(tag));
        for entry in &mut spec.entries {
            entry.symbol = mutant_symbol(&entry.symbol, tag);
        }
        spec.defines.push((define.into(), "1".into()));
        specs.push(spec);
    }
    specs
}

/// Per-rank buffers of a state that takes calls of up to `capacity`
/// elements at `ranks` ranks: the inbox holds the one-shot region, then one
/// slot of the largest owner span per source; flags and the epoch (one
/// word) start zeroed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Buffers {
    pub inbox_bytes: u64,
    pub flag_bytes: u64,
    pub epoch_bytes: u64,
}

pub fn buffers(ranks: u32, capacity: u64) -> Result<Buffers, Refusal> {
    check_geometry(ranks, capacity)?;
    let granules = capacity / GRANULE;
    let inbox_bytes = granules
        .div_ceil(u64::from(ranks))
        .checked_mul(GRANULE * 4 * u64::from(ranks))
        .and_then(|bytes| bytes.checked_add(ONE_SHOT_REGION * 4))
        .ok_or(Refusal("phased F32 all-reduce inbox overflows u64"))?;
    Ok(Buffers {
        inbox_bytes,
        flag_bytes: FLAG_WORDS * 4,
        epoch_bytes: 4,
    })
}

fn check_geometry(ranks: u32, elements: u64) -> Result<(), Refusal> {
    if ranks == 0 || ranks as usize > MAX_COMM_RANKS {
        return Err(Refusal("phased F32 all-reduce rank count out of range"));
    }
    if elements == 0 || elements % GRANULE != 0 {
        return Err(Refusal(
            "phased F32 all-reduce elements must be a nonzero multiple of 1024",
        ));
    }
    Ok(())
}

/// The launch plan of `senders` sending CTAs and `workers` summing CTAs;
/// with no workers the senders sum their own blocks.
pub fn plan_with(senders: u32, workers: u32) -> Result<LaunchPlan, Refusal> {
    if senders == 0 || senders > MAX_SENDERS {
        return Err(Refusal("phased F32 all-reduce senders out of range"));
    }
    let grid = senders
        .checked_add(workers)
        .ok_or(Refusal("phased F32 all-reduce grid overflows u32"))?;
    let plan = LaunchPlan {
        grid: [grid, 1, 1],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// [`plan_with`] the default senders and workers.
pub fn plan() -> Result<LaunchPlan, Refusal> {
    plan_with(DEFAULT_SENDERS, DEFAULT_WORKERS)
}

/// One rank's call. `inboxes`, `outputs` and `flags` hold every rank's
/// buffer in rank order; `senders` must be the plan's. `live_rows`, when
/// present, limits the call to the first `*live_rows` rows of
/// `row_elements` each.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Call {
    pub input: u64,
    pub output: u64,
    pub inboxes: Vec<u64>,
    pub outputs: Vec<u64>,
    pub flags: Vec<u64>,
    pub epoch: u64,
    pub elements: u64,
    pub row_elements: u32,
    pub live_rows: Option<u64>,
    pub rank: u32,
    pub ranks: u32,
    pub senders: u32,
}

/// The arguments of `call` for the entry `symbol`.
pub fn arguments(symbol: &str, call: Call) -> Result<ArgumentList, Refusal> {
    check_geometry(call.ranks, call.elements)?;
    if is_one_shot(symbol) && call.elements > ONE_SHOT_MAX_ELEMENTS {
        return Err(Refusal(
            "phased F32 one-shot all-reduce exceeds its 262144 elements",
        ));
    }
    let ranks = call.ranks as usize;
    if call.rank >= call.ranks
        || call.inboxes.len() != ranks
        || call.outputs.len() != ranks
        || call.flags.len() != ranks
    {
        return Err(Refusal(
            "phased F32 all-reduce needs one buffer per rank and a rank below the count",
        ));
    }
    if call.live_rows.is_some()
        && (call.row_elements == 0 || u64::from(call.row_elements) % GRANULE != 0)
    {
        return Err(Refusal(
            "phased F32 all-reduce live rows must be whole granules",
        ));
    }
    if call.senders == 0 || call.senders > MAX_SENDERS {
        return Err(Refusal("phased F32 all-reduce senders out of range"));
    }
    let args = ArgumentList(vec![
        Argument::DeviceBuffer(call.input),
        Argument::DeviceBuffer(call.output),
        Argument::DevicePointerArray(call.inboxes),
        Argument::DevicePointerArray(call.outputs),
        Argument::DevicePointerArray(call.flags),
        Argument::DeviceBuffer(call.epoch),
        Argument::U64(call.elements),
        Argument::U32(call.row_elements),
        match call.live_rows {
            Some(handle) => Argument::DeviceBuffer(handle),
            None => Argument::NullDevicePointer,
        },
        Argument::U32(call.rank),
        Argument::U32(call.ranks),
        Argument::U32(call.senders),
    ]);
    args.validate()?;
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call() -> Call {
        Call {
            input: 1,
            output: 2,
            inboxes: vec![10, 11, 12, 13],
            outputs: vec![20, 21, 22, 23],
            flags: vec![30, 31, 32, 33],
            epoch: 3,
            elements: 2048 * 4096,
            row_elements: 4096,
            live_rows: Some(4),
            rank: 1,
            ranks: 4,
            senders: 16,
        }
    }

    #[test]
    fn arguments_follow_the_entry_abi() {
        let entries = entries();
        assert_eq!(entries.len(), 4);
        assert!(entries[2].body.contains("allreduce<float,true>"));
        assert!(entries[1].body.contains("allreduce<uint16_t,false>"));
        assert!(
            arguments(
                ONE_SHOT_SYMBOL,
                Call {
                    elements: ONE_SHOT_MAX_ELEMENTS + GRANULE,
                    ..call()
                }
            )
            .is_err()
        );
        assert!(
            arguments(
                ONE_SHOT_ROUND_SYMBOL,
                Call {
                    elements: ONE_SHOT_MAX_ELEMENTS,
                    ..call()
                }
            )
            .is_ok()
        );
        let args = arguments(SYMBOL, call()).unwrap();
        for entry in &entries {
            assert_eq!(args.0.len(), entry.params.len());
        }
        assert_eq!(args.0[8], Argument::DeviceBuffer(4));
        assert_eq!(args.0[11], Argument::U32(16));
        let unlimited = arguments(
            SYMBOL,
            Call {
                live_rows: None,
                ..call()
            },
        )
        .unwrap();
        assert_eq!(unlimited.0[8], Argument::NullDevicePointer);
    }

    #[test]
    fn calls_and_plans_refuse_bad_geometry() {
        assert!(
            arguments(
                SYMBOL,
                Call {
                    elements: 4096 + 4,
                    ..call()
                }
            )
            .is_err()
        );
        assert!(arguments(SYMBOL, Call { rank: 4, ..call() }).is_err());
        assert!(
            arguments(
                SYMBOL,
                Call {
                    flags: vec![30, 31, 32],
                    ..call()
                }
            )
            .is_err()
        );
        assert!(
            arguments(
                SYMBOL,
                Call {
                    row_elements: 1000,
                    ..call()
                }
            )
            .is_err()
        );
        assert!(
            arguments(
                SYMBOL,
                Call {
                    senders: MAX_SENDERS + 1,
                    ..call()
                }
            )
            .is_err()
        );
        assert!(plan_with(0, 8).is_err());
        assert_eq!(plan_with(8, 0).unwrap().grid, [8, 1, 1]);
        assert!(plan_with(MAX_SENDERS + 1, 8).is_err());
        assert_eq!(
            plan_with(2, u32::MAX).unwrap_err(),
            Refusal("phased F32 all-reduce grid overflows u32")
        );
        let plan = plan().unwrap();
        assert_eq!(plan.grid, [DEFAULT_SENDERS + DEFAULT_WORKERS, 1, 1]);
        assert_eq!(plan.cluster, [1, 1, 1]);
        assert!(plan.pdl);
    }

    #[test]
    fn buffers_hold_the_largest_owner_span_per_source() {
        // 2048 rows of 4096 at four ranks: 8192 granules, 2048 per owner.
        let state = buffers(4, 2048 * 4096).unwrap();
        assert_eq!(
            state.inbox_bytes,
            (2 * 8 * 256 * 1024 + 4 * 2048 * 1024) * 4
        );
        assert_eq!(state.flag_bytes, FLAG_WORDS * 4);
        assert_eq!(FLAG_WORDS, 4 * 8 * 256 + 256 + 2);
        // Three granules at two ranks: owner spans of 1 and 2 granules.
        assert_eq!(
            buffers(2, 3 * GRANULE).unwrap().inbox_bytes,
            (ONE_SHOT_REGION + 2 * 2 * 1024) * 4
        );
        assert!(buffers(0, GRANULE).is_err());
        assert!(buffers(4, GRANULE + 1).is_err());
    }

    #[test]
    fn mutants_rename_their_symbols_and_add_one_define() {
        let specs = lab_cubin_specs();
        assert_eq!(specs.len(), 1 + MUTANTS.len());
        assert_eq!(specs[0].name, LAB_CUBIN);
        assert!(specs[0].sources_root.join(HEADER).is_file());
        for (spec, (tag, define)) in specs[1..].iter().zip(MUTANTS) {
            assert_eq!(spec.name, mutant_cubin(tag));
            assert_eq!(spec.defines.len(), 2);
            assert_eq!(spec.defines[1], (define.to_string(), "1".to_string()));
            for (entry, served) in spec.entries.iter().zip(&specs[0].entries) {
                assert_eq!(entry.symbol, mutant_symbol(&served.symbol, tag));
                assert!(entry.symbol.starts_with("chop_comm_test_"));
                assert_eq!(entry.body, served.body);
            }
        }
    }
}
