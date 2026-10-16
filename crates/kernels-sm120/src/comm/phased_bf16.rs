//! Rank-ordered all-reduce of BF16 partials from F32 inputs in one launch
//! (`cuda/comm/allreduce_phased_bf16.cuh`): every rank rounds its F32
//! partial once to BF16 (round to nearest even), and every output element
//! is the F32 sum of the ranks' BF16 values in rank order from F32 zero,
//! rounded once to BF16. The two-shot entry ([`SYMBOL`]) is a
//! reduce-scatter then an all-gather, each sent to one destination at a
//! time from `senders` CTAs, with one release flag per block; the one-shot
//! entry ([`ONE_SHOT_SYMBOL`]) pushes every rank's whole rounded partial to
//! each peer in turn and sums every element on every rank, for calls of up
//! to [`ONE_SHOT_MAX_ELEMENTS`]. The protocol, plans, block geometry and
//! flags are [`super::phased_f32`]'s, with BF16 payloads: half the bytes.
//! Both entries share one state ([`buffers`]) and may follow each other in
//! any order. A state serves one payload format: its regions are placed by
//! element, so BF16 and F32 calls on one state overlap in bytes, and a
//! state may pass to the phased F32 entries (or back) only after every
//! rank finished every call on it.
//!
//! Every CTA of a launch spins on flags other CTAs or peers release, so the
//! whole grid must be resident at once on every rank.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{Argument, ArgumentList, MAX_COMM_RANKS, Refusal, Validate};

pub use super::phased_f32::{
    FLAG_WORDS, GRANULE, MAX_BLOCKS, MAX_SENDERS, ONE_SHOT_MAX_ELEMENTS, ONE_SHOT_REGION, THREADS,
    plan_with,
};

/// The entries' header, relative to the CUDA source root.
pub const HEADER: &str = "comm/allreduce_phased_bf16.cuh";

/// The two-shot entry: the rounded sums into every rank's BF16 output.
pub const SYMBOL: &str = "chop_comm_allreduce_phased_bf16_v1";
/// The one-shot entry.
pub const ONE_SHOT_SYMBOL: &str = "chop_comm_allreduce_phased_oneshot_bf16_v1";

/// The lab's mutants (`lab_cubin_specs`): each define, and the tag its
/// cubin and symbols carry.
pub const MUTANTS: [(&str, &str); 6] = [
    (
        "no_scatter_wait",
        "CHOP_COMM_PHASED_BF16_TEST_NO_SCATTER_WAIT",
    ),
    (
        "no_gather_wait",
        "CHOP_COMM_PHASED_BF16_TEST_NO_GATHER_WAIT",
    ),
    ("reversed", "CHOP_COMM_PHASED_BF16_TEST_REVERSED"),
    ("late_wait", "CHOP_COMM_PHASED_BF16_TEST_LATE_WAIT"),
    ("shared_parity", "CHOP_COMM_PHASED_BF16_TEST_SHARED_PARITY"),
    ("own_f32", "CHOP_COMM_PHASED_BF16_TEST_OWN_F32"),
];
/// The lab's timing hook (no defect): with this define, every CTA of rank
/// [`SLOW_RANK`] sleeps once before it first reads its inbox or gathers, so
/// peers run ahead of it.
pub const SLOW_RANK_DEFINE: &str = "CHOP_COMM_PHASED_BF16_TEST_SLOW_RANK";
/// The rank [`SLOW_RANK_DEFINE`] slows (present at two and more ranks).
pub const SLOW_RANK: u32 = 1;
/// The mutants whose defect shows only while a peer runs ahead of a slow
/// rank: their cubins carry [`SLOW_RANK_DEFINE`] too.
pub const SLOW_MUTANTS: [&str; 2] = ["no_gather_wait", "shared_parity"];
/// The tag of the served entries built with [`SLOW_RANK_DEFINE`] (the
/// control of [`SLOW_MUTANTS`]: the same timing, no defect).
pub const SLOW_CONTROL: &str = "slow";

/// The served cubin's name in the lab.
pub const LAB_CUBIN: &str = "comm_phased_bf16_sm120";

/// The lab cubin of the mutant `tag`.
pub fn mutant_cubin(tag: &str) -> String {
    format!("comm_phased_bf16_{tag}_sm120")
}

/// `symbol` in the mutant `tag`'s cubin.
pub fn mutant_symbol(symbol: &str, tag: &str) -> String {
    symbol.replacen("chop_comm_", &format!("chop_comm_test_{tag}_"), 1)
}

fn params() -> Vec<EntryParam> {
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
        pointer("output", "uint16_t"),
        array("inboxes", "uint16_t"),
        array("outputs", "uint16_t"),
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

/// Both entries.
pub fn entries() -> Vec<EntrySpec> {
    [(SYMBOL, false), (ONE_SHOT_SYMBOL, true)]
        .into_iter()
        .map(|(symbol, one_shot)| {
            let body = format!(
                "chop::comm::phased_bf16::allreduce<{one_shot}>(input, output, inboxes, outputs, flags, epoch, elements, row_elements, live_rows, rank, ranks, senders);"
            );
            EntrySpec::new(symbol.into(), params(), body, THREADS, 1)
        })
        .collect()
}

/// The index of the `senders` argument in [`arguments`]: the launch's
/// first CTAs, which send; every CTA after them is a worker.
pub const SENDERS_ARGUMENT: usize = 11;

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

/// The lab's cubins: [`LAB_CUBIN`], one cubin per [`MUTANTS`] entry whose
/// symbols carry the mutant's tag ([`mutant_symbol`]), [`SLOW_MUTANTS`]
/// with [`SLOW_RANK_DEFINE`] too, and the [`SLOW_CONTROL`] cubin.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let tagged = |tag: &str, defines: &[&str]| {
        let mut spec = cubin_spec(mutant_cubin(tag));
        for entry in &mut spec.entries {
            entry.symbol = mutant_symbol(&entry.symbol, tag);
        }
        for &define in defines {
            let value = if define == SLOW_RANK_DEFINE {
                SLOW_RANK.to_string()
            } else {
                "1".into()
            };
            spec.defines.push((define.into(), value));
        }
        spec
    };
    let mut specs = vec![cubin_spec(LAB_CUBIN.into())];
    for (tag, define) in MUTANTS {
        specs.push(if SLOW_MUTANTS.contains(&tag) {
            tagged(tag, &[define, SLOW_RANK_DEFINE])
        } else {
            tagged(tag, &[define])
        });
    }
    specs.push(tagged(SLOW_CONTROL, &[SLOW_RANK_DEFINE]));
    specs
}

/// Per-rank buffers of a state that takes calls of up to `capacity`
/// elements at `ranks` ranks: the inbox holds the one-shot region, then one
/// slot of the largest owner span per source, in BF16; flags and the epoch
/// (one word) start zeroed. [`super::phased_f32::buffers`] of half the
/// capacity holds the same calls.
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
        .checked_mul(GRANULE * 2 * u64::from(ranks))
        .and_then(|bytes| bytes.checked_add(ONE_SHOT_REGION * 2))
        .ok_or(Refusal("phased BF16 all-reduce inbox overflows u64"))?;
    Ok(Buffers {
        inbox_bytes,
        flag_bytes: FLAG_WORDS * 4,
        epoch_bytes: 4,
    })
}

fn check_geometry(ranks: u32, elements: u64) -> Result<(), Refusal> {
    if ranks == 0 || ranks as usize > MAX_COMM_RANKS {
        return Err(Refusal("phased BF16 all-reduce rank count out of range"));
    }
    if elements == 0 || elements % GRANULE != 0 {
        return Err(Refusal(
            "phased BF16 all-reduce elements must be a nonzero multiple of 1024",
        ));
    }
    Ok(())
}

/// One rank's call. `input` holds F32, `output` and every rank's `outputs`
/// BF16. `inboxes`, `outputs` and `flags` hold every rank's buffer in rank
/// order; `senders` must be the plan's. `live_rows`, when present, limits
/// the call to the first `*live_rows` rows of `row_elements` each.
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
            "phased BF16 one-shot all-reduce exceeds its 262144 elements",
        ));
    }
    let ranks = call.ranks as usize;
    if call.rank >= call.ranks
        || call.inboxes.len() != ranks
        || call.outputs.len() != ranks
        || call.flags.len() != ranks
    {
        return Err(Refusal(
            "phased BF16 all-reduce needs one buffer per rank and a rank below the count",
        ));
    }
    if call.live_rows.is_some()
        && (call.row_elements == 0 || u64::from(call.row_elements) % GRANULE != 0)
    {
        return Err(Refusal(
            "phased BF16 all-reduce live rows must be whole granules",
        ));
    }
    if call.senders == 0 || call.senders > MAX_SENDERS {
        return Err(Refusal("phased BF16 all-reduce senders out of range"));
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
        assert_eq!(entries.len(), 2);
        assert!(entries[0].body.contains("allreduce<false>"));
        assert!(entries[1].body.contains("allreduce<true>"));
        assert!(is_one_shot(ONE_SHOT_SYMBOL) && !is_one_shot(SYMBOL));
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
                ONE_SHOT_SYMBOL,
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
        assert_eq!(args.0[SENDERS_ARGUMENT], Argument::U32(16));
        assert!(matches!(
            &params()[SENDERS_ARGUMENT],
            EntryParam::U32 { name } if name == "senders"
        ));
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
    fn calls_refuse_bad_geometry() {
        let refused = |call: Call| arguments(SYMBOL, call).is_err();
        assert!(refused(Call {
            elements: 4096 + 4,
            ..call()
        }));
        assert!(refused(Call { rank: 4, ..call() }));
        assert!(refused(Call {
            flags: vec![30, 31, 32],
            ..call()
        }));
        assert!(refused(Call {
            row_elements: 1000,
            ..call()
        }));
        assert!(refused(Call {
            senders: MAX_SENDERS + 1,
            ..call()
        }));
    }

    #[test]
    fn buffers_hold_the_largest_owner_span_per_source_in_bf16() {
        // 8192 rows of 4096 at four ranks: 32768 granules, 8192 per owner.
        let state = buffers(4, 8192 * 4096).unwrap();
        assert_eq!(
            state.inbox_bytes,
            (2 * 8 * 256 * 1024 + 4 * 8192 * 1024) * 2
        );
        assert_eq!(state.flag_bytes, FLAG_WORDS * 4);
        assert_eq!(state.epoch_bytes, 4);
        // The F32 state of half the elements holds the same calls.
        let f32 = super::super::phased_f32::buffers(4, 4096 * 4096).unwrap();
        assert!(f32.inbox_bytes >= buffers(4, 8192 * 4096).unwrap().inbox_bytes);
        assert!(buffers(0, GRANULE).is_err());
        assert!(buffers(4, GRANULE + 1).is_err());
    }

    #[test]
    fn mutants_rename_their_symbols_and_add_one_define() {
        let specs = lab_cubin_specs();
        assert_eq!(specs.len(), 2 + MUTANTS.len());
        assert_eq!(specs[0].name, LAB_CUBIN);
        assert!(specs[0].sources_root.join(HEADER).is_file());
        let slow = (SLOW_RANK_DEFINE.to_string(), SLOW_RANK.to_string());
        let tags = MUTANTS.iter().map(|&(tag, define)| (tag, Some(define)));
        let tags = tags.chain([(SLOW_CONTROL, None)]);
        for (spec, (tag, define)) in specs[1..].iter().zip(tags) {
            assert_eq!(spec.name, mutant_cubin(tag));
            let mut defines = spec.defines[1..].to_vec();
            if let Some(define) = define {
                assert_eq!(defines.remove(0), (define.to_string(), "1".to_string()));
            }
            let slowed = define.is_none() || SLOW_MUTANTS.contains(&tag);
            assert_eq!(defines, if slowed { vec![slow.clone()] } else { vec![] });
            for (entry, served) in spec.entries.iter().zip(&specs[0].entries) {
                assert_eq!(entry.symbol, mutant_symbol(&served.symbol, tag));
                assert!(entry.symbol.starts_with("chop_comm_test_"));
                assert_eq!(entry.body, served.body);
            }
        }
        for tag in SLOW_MUTANTS {
            assert!(MUTANTS.iter().any(|&(mutant, _)| mutant == tag));
        }
    }
}
