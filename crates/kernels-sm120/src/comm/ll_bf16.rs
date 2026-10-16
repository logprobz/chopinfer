//! The low-latency all-reduce of BF16 partials from F32 inputs over CTAs
//! (`cuda/comm/allreduce_ll_bf16.cuh`): every rank rounds its F32 partial
//! once to BF16, and every output element is the F32 sum of the ranks'
//! BF16 values in rank order from F32 zero, rounded once to BF16, bitwise
//! the phased BF16 entries ([`super::phased_bf16`]). Each 4096-element row
//! spreads over CTAs of [`SLICE_ELEMENTS`] elements that exchange their
//! rounded inputs by recursive doubling in push.cuh's low-latency words,
//! two BF16 values per word (no fences, flag words or acknowledgments). Each
//! CTA exchanges with the CTAs of the same index on its peers only and
//! never waits on a CTA of its own grid, but it spins on those
//! counterparts, so a caller admits the grid against the device's
//! residency before serving it. The grid is bounded by [`MAX_CTAS`].
//! Power-of-two rank counts only. The state layout is
//! [`super::ll_f32`]'s.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, LaunchPlan, MAX_COMM_RANKS, Refusal, Validate,
};

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "comm/allreduce_ll_bf16.cuh";
/// The served entry.
pub const SYMBOL: &str = "chop_comm_allreduce_ll_bf16_v1";
/// Threads of the entry, one 16-byte vector of eight BF16 values each.
pub const THREADS: u32 = 256;
/// Elements of one CTA's slice.
pub const SLICE_ELEMENTS: u64 = 8 * THREADS as u64;
/// Most CTAs of a state's capacity and so of one call: 64 rows of 4096.
pub const MAX_CTAS: u32 = 128;
/// Bytes of one CTA's inbox slot of one source: four 8-byte words per
/// thread.
const SLOT_BYTES: u64 = 4 * 8 * THREADS as u64;

/// The CTAs of `elements` elements.
pub fn ctas(elements: u64) -> Result<u32, Refusal> {
    if elements == 0 || !elements.is_multiple_of(SLICE_ELEMENTS) {
        return Err(Refusal(
            "the low-latency BF16 all-reduce takes whole 2048-element slices",
        ));
    }
    let ctas = elements / SLICE_ELEMENTS;
    if ctas > u64::from(MAX_CTAS) {
        return Err(Refusal(
            "the low-latency BF16 all-reduce exceeds its 128 CTAs",
        ));
    }
    Ok(ctas as u32)
}

fn check_ranks(ranks: u32) -> Result<(), Refusal> {
    if ranks == 0 || ranks > MAX_COMM_RANKS as u32 || !ranks.is_power_of_two() {
        return Err(Refusal(
            "the low-latency BF16 all-reduce takes a power-of-two rank count up to 8",
        ));
    }
    Ok(())
}

/// One rank's state over a capacity of `capacity_elements` elements: the
/// inbox peers write and the local epoch cells, both zeroed at allocation.
/// Every call over the state runs up to its capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateBytes {
    pub inbox: u64,
    pub epochs: u64,
}

pub fn state_bytes(ranks: u32, capacity_elements: u64) -> Result<StateBytes, Refusal> {
    check_ranks(ranks)?;
    let capacity = u64::from(ctas(capacity_elements)?);
    Ok(StateBytes {
        inbox: 2 * u64::from(ranks) * capacity * SLOT_BYTES,
        epochs: capacity * 4,
    })
}

/// The plan of a call over `elements` elements: one CTA per slice.
pub fn plan(ranks: u32, elements: u64) -> Result<LaunchPlan, Refusal> {
    check_ranks(ranks)?;
    let plan = LaunchPlan {
        grid: [ctas(elements)?, 1, 1],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// Handles are resolved by the caller before launch. `input` holds F32,
/// `output` BF16. Slot order is rank order; every rank of a state passes
/// the same `capacity_elements`.
#[derive(Clone, Debug)]
pub struct Call {
    pub input: u64,
    pub output: u64,
    pub peer_inboxes: Vec<u64>,
    pub epochs: u64,
    pub elements: u64,
    pub capacity_elements: u64,
    pub rank: u32,
    pub ranks: u32,
}

pub fn arguments(call: &Call) -> Result<ArgumentList, Refusal> {
    check_ranks(call.ranks)?;
    if call.peer_inboxes.len() != call.ranks as usize || call.rank >= call.ranks {
        return Err(Refusal(
            "the low-latency BF16 all-reduce needs one inbox per rank",
        ));
    }
    let ctas = ctas(call.elements)?;
    let capacity = self::ctas(call.capacity_elements)?;
    if ctas > capacity {
        return Err(Refusal(
            "the low-latency BF16 all-reduce exceeds its state's capacity",
        ));
    }
    let args = ArgumentList(vec![
        Argument::DeviceBuffer(call.input),
        Argument::DeviceBuffer(call.output),
        Argument::DevicePointerArray(call.peer_inboxes.clone()),
        Argument::DeviceBuffer(call.epochs),
        Argument::U64(call.elements),
        Argument::U32(capacity),
        Argument::U32(call.rank),
        Argument::U32(call.ranks),
    ]);
    args.validate()?;
    Ok(args)
}

pub fn params() -> Vec<EntryParam> {
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    vec![
        pointer("input", "const float"),
        pointer("output", "uint16_t"),
        EntryParam::PointerArray {
            name: "inboxes".into(),
            slots: MAX_COMM_RANKS as u8,
            cxx_type: "uint64_t".into(),
        },
        pointer("epochs", "unsigned int"),
        EntryParam::U64 {
            name: "elements".into(),
        },
        EntryParam::U32 {
            name: "capacity_ctas".into(),
        },
        EntryParam::U32 {
            name: "rank".into(),
        },
        EntryParam::U32 {
            name: "ranks".into(),
        },
    ]
}

const BODY: &str = "chop::comm::ll_bf16::allreduce(input, output, inboxes, epochs, elements, capacity_ctas, rank, ranks);";

/// A cubin named `name` holding the entry as `symbol`.
fn spec(name: String, symbol: &str) -> CubinSpec {
    CubinSpec {
        name,
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![PathBuf::from("comm"), PathBuf::from("common")],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: super::push_defines(),
        entries: vec![EntrySpec::new(
            symbol.into(),
            params(),
            BODY.into(),
            THREADS,
            1,
        )],
    }
}

/// The served entry's cubin, named `name`.
pub fn cubin_spec(name: String) -> CubinSpec {
    spec(name, SYMBOL)
}

/// The lab's cubin of the served entry.
pub const LAB_CUBIN: &str = "comm_ll_bf16_sm120";
/// The lab's mutants: each tag (its cubin is `comm_ll_bf16_<tag>_sm120`,
/// its symbol `chop_comm_allreduce_ll_bf16_<tag>_v1`), define and value.
/// The wait mutants load their inputs before the dependency wait (1) or
/// never wait (2); the stale-flag mutant's flag ignores the epoch and its
/// receivers keep what they read; the own-F32 mutant adds this rank's own
/// values unrounded.
pub const MUTANTS: [(&str, &str, &str); 4] = [
    ("late_wait", "CHOP_COMM_LL_BF16_TEST_WAIT", "1"),
    ("no_wait", "CHOP_COMM_LL_BF16_TEST_WAIT", "2"),
    ("stale", "CHOP_COMM_LL_BF16_TEST_STALE", "1"),
    ("own_f32", "CHOP_COMM_LL_BF16_TEST_OWN_F32", "1"),
];

/// The lab cubin of the mutant `tag`.
pub fn mutant_cubin(tag: &str) -> String {
    format!("comm_ll_bf16_{tag}_sm120")
}

/// The symbol of the mutant `tag`.
pub fn mutant_symbol(tag: &str) -> String {
    format!("chop_comm_allreduce_ll_bf16_{tag}_v1")
}

/// The lab's cubins: [`LAB_CUBIN`] (the served entry) and one cubin per
/// [`MUTANTS`] entry, which the lab's checks must catch.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let mut specs = vec![cubin_spec(LAB_CUBIN.into())];
    for (tag, define, value) in MUTANTS {
        let mut spec = spec(mutant_cubin(tag), &mutant_symbol(tag));
        spec.defines.push((define.into(), value.into()));
        specs.push(spec);
    }
    specs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_and_states_follow_the_slices() {
        assert_eq!(SLICE_ELEMENTS, 2048);
        for rows in [1_u64, 2, 4, 8, 16, 32, 64] {
            let elements = rows * 4096;
            let plan = plan(4, elements).unwrap();
            assert_eq!(plan.grid, [rows as u32 * 2, 1, 1]);
            assert_eq!(plan.block, [THREADS, 1, 1]);
            assert!(plan.pdl);
        }
        // Two rows: the F32 entry's state of one row.
        let f32 = super::super::ll_f32::state_bytes(4, 4096).unwrap();
        assert_eq!(
            state_bytes(4, 2 * 4096).unwrap(),
            StateBytes {
                inbox: f32.inbox,
                epochs: f32.epochs,
            }
        );
        assert_eq!(
            state_bytes(4, 64 * 4096).unwrap(),
            StateBytes {
                inbox: 2 * 4 * 128 * 8192,
                epochs: 512,
            }
        );
        assert!(plan(4, 65 * 4096).is_err());
        assert!(plan(4, 4096 + 1024).is_err());
        assert!(plan(4, 0).is_err());
        assert!(plan(3, 4096).is_err());
        assert!(plan(16, 4096).is_err());
        assert!(state_bytes(6, 4096).is_err());
    }

    #[test]
    fn arguments_follow_the_entry_abi() {
        let call = Call {
            input: 1,
            output: 2,
            peer_inboxes: vec![10, 11, 12, 13],
            epochs: 3,
            elements: 2 * 4096,
            capacity_elements: 8 * 4096,
            rank: 2,
            ranks: 4,
        };
        let args = arguments(&call).unwrap();
        assert_eq!(args.0.len(), params().len());
        assert_eq!(
            args.0[2],
            Argument::DevicePointerArray(vec![10, 11, 12, 13])
        );
        assert_eq!(args.0[5], Argument::U32(16));
        assert_eq!(&args.0[6..], [Argument::U32(2), Argument::U32(4)]);
        let refused = |change: fn(&mut Call)| {
            let mut call = call.clone();
            change(&mut call);
            arguments(&call).is_err()
        };
        assert!(refused(|call| call.elements = 9 * 4096));
        assert!(refused(|call| {
            call.peer_inboxes.pop();
        }));
        assert!(refused(|call| call.rank = 4));
        assert!(refused(|call| call.capacity_elements = 65 * 4096));
    }

    #[test]
    fn lab_cubins_hold_the_entry_and_its_mutants() {
        let specs = lab_cubin_specs();
        let names: Vec<_> = specs.iter().map(|spec| spec.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "comm_ll_bf16_sm120",
                "comm_ll_bf16_late_wait_sm120",
                "comm_ll_bf16_no_wait_sm120",
                "comm_ll_bf16_stale_sm120",
                "comm_ll_bf16_own_f32_sm120",
            ]
        );
        assert!(specs[0].sources_root.join(HEADER).is_file());
        assert_eq!(specs[0].entries[0].symbol, SYMBOL);
        assert!(
            !specs[0]
                .defines
                .iter()
                .any(|(name, _)| name.starts_with("CHOP_COMM_LL_BF16_TEST"))
        );
        for (spec, (tag, define, value)) in specs[1..].iter().zip(MUTANTS) {
            assert_eq!(spec.entries[0].symbol, mutant_symbol(tag));
            assert_eq!(spec.entries[0].body, specs[0].entries[0].body);
            assert_eq!(
                spec.defines.last(),
                Some(&(define.to_string(), value.to_string()))
            );
        }
    }
}
