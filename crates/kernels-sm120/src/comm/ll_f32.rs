//! The low-latency rank-ordered F32 all-reduce over CTAs
//! (`cuda/comm/allreduce_ll_f32.cuh`): bitwise the F32 one-shot
//! (`chop_comm_allreduce_oneshot_f32_v1`, every element summed from F32 zero
//! in rank order), with each 4096-element tile spread over CTAs of
//! [`SLICE_ELEMENTS`] elements that exchange their inputs by recursive
//! doubling in push.cuh's low-latency words (no fences, flag words or
//! acknowledgments). Each CTA exchanges with the CTAs of the same index on
//! its peers only and never waits on a CTA of its own grid, but it spins on
//! those counterparts, so a caller admits the grid against the device's
//! residency before serving it (a bundle checks its largest grid at
//! create). The grid is bounded by [`MAX_CTAS`]. Power-of-two rank
//! counts only. The round entry ([`ROUND_SYMBOL`]) writes each sum rounded
//! once to BF16: bitwise `comm::phased_f32`'s rounding entries.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, LaunchPlan, MAX_COMM_RANKS, Refusal, Validate,
};

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "comm/allreduce_ll_f32.cuh";
/// The served entry.
pub const SYMBOL: &str = "chop_comm_allreduce_ll_f32_v1";
/// The served round entry: [`SYMBOL`]'s sums rounded once to BF16 into a
/// BF16 output, with [`SYMBOL`]'s arguments, plans and states.
pub const ROUND_SYMBOL: &str = "chop_comm_allreduce_ll_round_f32_v1";
/// Threads of the entry, one 16-byte vector each.
pub const THREADS: u32 = 256;
/// F32 elements of one CTA's slice.
pub const SLICE_ELEMENTS: u64 = 4 * THREADS as u64;
/// Most CTAs of a state's capacity and so of one call: 32 rows of 4096.
pub const MAX_CTAS: u32 = 128;
/// Bytes of one CTA's inbox slot of one source: four 8-byte words per
/// thread.
const SLOT_BYTES: u64 = 4 * 8 * THREADS as u64;

/// The CTAs of `elements` F32 elements.
pub fn ctas(elements: u64) -> Result<u32, Refusal> {
    if elements == 0 || !elements.is_multiple_of(SLICE_ELEMENTS) {
        return Err(Refusal(
            "the low-latency F32 all-reduce takes whole 1024-element slices",
        ));
    }
    let ctas = elements / SLICE_ELEMENTS;
    if ctas > u64::from(MAX_CTAS) {
        return Err(Refusal(
            "the low-latency F32 all-reduce exceeds its 128 CTAs",
        ));
    }
    Ok(ctas as u32)
}

fn check_ranks(ranks: u32) -> Result<(), Refusal> {
    if ranks == 0 || ranks > MAX_COMM_RANKS as u32 || !ranks.is_power_of_two() {
        return Err(Refusal(
            "the low-latency F32 all-reduce takes a power-of-two rank count up to 8",
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

/// Handles are resolved by the caller before launch. Slot order is rank
/// order; every rank of a state passes the same `capacity_elements`.
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
            "the low-latency F32 all-reduce needs one inbox per rank",
        ));
    }
    let ctas = ctas(call.elements)?;
    let capacity = self::ctas(call.capacity_elements)?;
    if ctas > capacity {
        return Err(Refusal(
            "the low-latency F32 all-reduce exceeds its state's capacity",
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

/// The parameters of [`SYMBOL`]; [`ROUND_SYMBOL`]'s output is BF16 bits.
pub fn params() -> Vec<EntryParam> {
    params_with("float")
}

fn params_with(output: &str) -> Vec<EntryParam> {
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    vec![
        pointer("input", "const float"),
        pointer("output", output),
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

const BODY: &str = "chop::comm::ll_f32::allreduce(input, output, inboxes, epochs, elements, capacity_ctas, rank, ranks);";

fn entry(symbol: &str, output: &str) -> EntrySpec {
    EntrySpec::new(symbol.into(), params_with(output), BODY.into(), THREADS, 1)
}

/// A cubin named `name` holding the entry as `symbol`.
fn spec(name: String, symbol: &str) -> CubinSpec {
    CubinSpec {
        name,
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![PathBuf::from("comm"), PathBuf::from("common")],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: super::push_defines(),
        entries: vec![entry(symbol, "float")],
    }
}

/// The served entries' cubin, named `name`: [`SYMBOL`], then
/// [`ROUND_SYMBOL`].
pub fn cubin_spec(name: String) -> CubinSpec {
    let mut spec = spec(name, SYMBOL);
    spec.entries.push(entry(ROUND_SYMBOL, "uint16_t"));
    spec
}

/// The lab's mutant whose dependency wait follows its input loads.
pub const LATE_WAIT_SYMBOL: &str = "chop_comm_allreduce_ll_f32_late_wait_v1";
/// The lab's mutant without a dependency wait.
pub const NO_WAIT_SYMBOL: &str = "chop_comm_allreduce_ll_f32_no_wait_v1";
/// The lab's stale-flag mutant: a flag that ignores the epoch and receivers
/// that keep what they read.
pub const STALE_SYMBOL: &str = "chop_comm_allreduce_ll_f32_stale_v1";

/// The lab's cubin of the served entries.
pub const LAB_CUBIN: &str = "comm_ll_f32_sm120";

/// The lab's cubins: [`LAB_CUBIN`] (the served entries), the two wait
/// mutants (`CHOP_COMM_LL_F32_TEST_WAIT` 1 and 2) and the stale-flag mutant
/// (`CHOP_COMM_LL_F32_TEST_STALE`) of [`SYMBOL`], each in a cubin of its
/// own, which the lab's checks must catch. The round entry shares their
/// transport; only its store differs.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let mutant = |name: &str, symbol: &str, define: &str, value: &str| {
        let mut spec = spec(name.into(), symbol);
        spec.defines.push((define.into(), value.into()));
        spec
    };
    vec![
        cubin_spec(LAB_CUBIN.into()),
        mutant(
            "comm_ll_f32_late_wait_sm120",
            LATE_WAIT_SYMBOL,
            "CHOP_COMM_LL_F32_TEST_WAIT",
            "1",
        ),
        mutant(
            "comm_ll_f32_no_wait_sm120",
            NO_WAIT_SYMBOL,
            "CHOP_COMM_LL_F32_TEST_WAIT",
            "2",
        ),
        mutant(
            "comm_ll_f32_stale_sm120",
            STALE_SYMBOL,
            "CHOP_COMM_LL_F32_TEST_STALE",
            "1",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_and_states_follow_the_slices() {
        assert_eq!(SLICE_ELEMENTS, 1024);
        for rows in [1_u64, 2, 4, 8, 16, 32] {
            let elements = rows * 4096;
            let plan = plan(4, elements).unwrap();
            assert_eq!(plan.grid, [rows as u32 * 4, 1, 1]);
            assert_eq!(plan.block, [THREADS, 1, 1]);
            assert!(plan.pdl);
        }
        assert_eq!(
            state_bytes(4, 32 * 4096).unwrap(),
            StateBytes {
                inbox: 2 * 4 * 128 * 8192,
                epochs: 512,
            }
        );
        assert!(plan(4, 33 * 4096).is_err());
        assert!(plan(4, 4096 + 512).is_err());
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
        assert_eq!(args.0[5], Argument::U32(32));
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
        assert!(refused(|call| call.capacity_elements = 33 * 4096));
    }

    #[test]
    fn lab_cubins_hold_the_entry_and_its_mutants() {
        let specs = lab_cubin_specs();
        let names: Vec<_> = specs.iter().map(|spec| spec.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "comm_ll_f32_sm120",
                "comm_ll_f32_late_wait_sm120",
                "comm_ll_f32_no_wait_sm120",
                "comm_ll_f32_stale_sm120"
            ]
        );
        assert!(specs[0].sources_root.join(HEADER).is_file());
        assert_eq!(specs[0].entries[0].symbol, SYMBOL);
        assert_eq!(specs[0].entries[1].symbol, ROUND_SYMBOL);
        assert_eq!(specs[0].entries[1].body, specs[0].entries[0].body);
        // The round entry differs only in its BF16 output.
        let differing = specs[0].entries[0]
            .params
            .iter()
            .zip(&specs[0].entries[1].params)
            .filter(|(a, b)| a != b)
            .collect::<Vec<_>>();
        assert_eq!(
            differing,
            [(
                &EntryParam::DevicePointer {
                    name: "output".into(),
                    cxx_type: "float".into(),
                },
                &EntryParam::DevicePointer {
                    name: "output".into(),
                    cxx_type: "uint16_t".into(),
                },
            )]
        );
        assert!(
            !specs[0]
                .defines
                .iter()
                .any(|(name, _)| name == "CHOP_COMM_LL_F32_TEST_WAIT")
        );
        for (spec, symbol, define, value) in [
            (
                &specs[1],
                LATE_WAIT_SYMBOL,
                "CHOP_COMM_LL_F32_TEST_WAIT",
                "1",
            ),
            (&specs[2], NO_WAIT_SYMBOL, "CHOP_COMM_LL_F32_TEST_WAIT", "2"),
            (&specs[3], STALE_SYMBOL, "CHOP_COMM_LL_F32_TEST_STALE", "1"),
        ] {
            assert_eq!(spec.entries[0].symbol, symbol);
            assert_eq!(spec.entries[0].body, specs[0].entries[0].body);
            assert_eq!(
                spec.defines.last(),
                Some(&(define.to_string(), value.to_string()))
            );
        }
    }
}
