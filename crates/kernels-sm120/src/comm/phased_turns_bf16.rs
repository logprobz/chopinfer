//! Rank-ordered all-reduce of BF16 partials from F32 inputs whose writers
//! take turns at each destination (`cuda/comm/allreduce_phased_turns_bf16.cuh`):
//! [`super::phased_bf16`]'s two-shot (the same arguments, rounding, inbox
//! slots, block flags and rank-order sums, so every output element is
//! bitwise that entry's), where a rank starts each phase into a destination
//! only after the rank that wrote into it in the previous phase finished
//! issuing its writes. When ranks start a call at different times, this
//! keeps one writer per receiving GPU, as when they start together.
//!
//! The state is [`super::phased_bf16::buffers`]'s with [`TURN_WORDS`] more
//! flag words ([`buffers`]); the phased BF16 entries may share it and follow
//! this entry in any order. Every CTA of a launch spins on flags other CTAs
//! or peers release, so the whole grid must be resident at once on every
//! rank.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{ArgumentList, MAX_COMM_RANKS, Refusal};

pub use super::phased_bf16::{Call, SENDERS_ARGUMENT};
pub use super::phased_f32::{GRANULE, MAX_SENDERS, THREADS, plan_with};

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "comm/allreduce_phased_turns_bf16.cuh";

/// The entry: the rounded sums into every rank's BF16 output.
pub const SYMBOL: &str = "chop_comm_allreduce_phased_turns_bf16_v1";

/// The turn words after the phased flags, one per phase.
pub const TURN_WORDS: u64 = 2 * MAX_COMM_RANKS as u64;
/// Flag words of the state: the phased entries' then the turn words.
pub const FLAG_WORDS: u64 = super::phased_f32::FLAG_WORDS + TURN_WORDS;

/// The lab's mutants (`lab_cubin_specs`): each define, and the tag its
/// cubin and symbols carry. The reversed order and the unrounded own
/// partial act in the sums this entry shares with the phased BF16 header.
pub const MUTANTS: [(&str, &str); 5] = [
    (
        "no_scatter_wait",
        "CHOP_COMM_PHASED_TURNS_BF16_TEST_NO_SCATTER_WAIT",
    ),
    (
        "no_gather_wait",
        "CHOP_COMM_PHASED_TURNS_BF16_TEST_NO_GATHER_WAIT",
    ),
    ("reversed", "CHOP_COMM_PHASED_BF16_TEST_REVERSED"),
    ("late_wait", "CHOP_COMM_PHASED_TURNS_BF16_TEST_LATE_WAIT"),
    ("own_f32", "CHOP_COMM_PHASED_BF16_TEST_OWN_F32"),
];
/// The phased BF16 header's slow-rank hook, which this entry's reductions
/// share: every CTA of rank [`SLOW_RANK`] sleeps once before it first reads
/// its inbox or gathers.
pub use super::phased_bf16::{SLOW_RANK, SLOW_RANK_DEFINE};
/// The mutants whose defect shows only while a peer runs ahead of a slow
/// rank: their cubins carry [`SLOW_RANK_DEFINE`] too.
pub const SLOW_MUTANTS: [&str; 1] = ["no_gather_wait"];
/// The tag of the served entry built with [`SLOW_RANK_DEFINE`].
pub const SLOW_CONTROL: &str = "slow";
/// The lab's late-scatter hook (no defect): with this define, every sending
/// CTA of rank [`LATE_SCATTER_RANK`] sleeps before its last scatter phase,
/// whose data no turn of its destination waits for.
pub const LATE_SCATTER_DEFINE: &str = "CHOP_COMM_PHASED_TURNS_BF16_TEST_LATE_SCATTER";
/// The rank [`LATE_SCATTER_DEFINE`] delays.
pub const LATE_SCATTER_RANK: u32 = 1;
/// The mutants whose defect shows only while a rank's last scatter phase
/// runs late: without workers a rank sums its blocks after its first
/// gather turn, which does not follow that phase. Their cubins carry
/// [`LATE_SCATTER_DEFINE`] too.
pub const LATE_SCATTER_MUTANTS: [&str; 1] = ["no_scatter_wait"];
/// The tag of the served entry built with [`LATE_SCATTER_DEFINE`].
pub const LATE_SCATTER_CONTROL: &str = "late_scatter";
/// Lab timing variants (no defect): each tag and its defines. `unturned`
/// waits for no turn (the phased BF16 two-shot's transfer order);
/// `blocks32` and `blocks128` move 32 or 128 granules per block flag
/// instead of 64.
pub const TIMING_VARIANTS: [(&str, &[(&str, &str)]); 4] = [
    (
        "unturned",
        &[("CHOP_COMM_PHASED_TURNS_BF16_TEST_UNTURNED", "1")],
    ),
    (
        "blocks32",
        &[("CHOP_COMM_PHASED_TURNS_BF16_TEST_BLOCK_GRANULES", "32")],
    ),
    (
        "blocks128",
        &[("CHOP_COMM_PHASED_TURNS_BF16_TEST_BLOCK_GRANULES", "128")],
    ),
    (
        "unturned_blocks128",
        &[
            ("CHOP_COMM_PHASED_TURNS_BF16_TEST_UNTURNED", "1"),
            ("CHOP_COMM_PHASED_TURNS_BF16_TEST_BLOCK_GRANULES", "128"),
        ],
    ),
];

/// The served cubin's name in the lab.
pub const LAB_CUBIN: &str = "comm_phased_turns_bf16_sm120";

/// The lab cubin of the mutant or timing variant `tag`.
pub fn mutant_cubin(tag: &str) -> String {
    format!("comm_phased_turns_bf16_{tag}_sm120")
}

/// `symbol` in the lab cubin `tag`'s.
pub fn mutant_symbol(symbol: &str, tag: &str) -> String {
    symbol.replacen("chop_comm_", &format!("chop_comm_test_{tag}_"), 1)
}

/// The entry: the phased BF16 two-shot's parameters.
pub fn entries() -> Vec<EntrySpec> {
    let mut entry = super::phased_bf16::entries()
        .into_iter()
        .find(|entry| entry.symbol == super::phased_bf16::SYMBOL)
        .expect("the phased BF16 two-shot entry");
    entry.symbol = SYMBOL.into();
    entry.body = "chop::comm::phased_turns_bf16::allreduce(input, output, inboxes, outputs, flags, epoch, elements, row_elements, live_rows, rank, ranks, senders);".into();
    vec![entry]
}

/// A cubin of the entry named `name`.
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
/// symbol carries the mutant's tag ([`mutant_symbol`]), [`SLOW_MUTANTS`]
/// with [`SLOW_RANK_DEFINE`] and [`LATE_SCATTER_MUTANTS`] with
/// [`LATE_SCATTER_DEFINE`] too, the [`SLOW_CONTROL`] and
/// [`LATE_SCATTER_CONTROL`] cubins and one cubin per [`TIMING_VARIANTS`]
/// entry.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let tagged = |tag: &str, defines: Vec<(String, String)>| {
        let mut spec = cubin_spec(mutant_cubin(tag));
        for entry in &mut spec.entries {
            entry.symbol = mutant_symbol(&entry.symbol, tag);
        }
        spec.defines.extend(defines);
        spec
    };
    let slow = || (SLOW_RANK_DEFINE.to_string(), SLOW_RANK.to_string());
    let late = || {
        (
            LATE_SCATTER_DEFINE.to_string(),
            LATE_SCATTER_RANK.to_string(),
        )
    };
    let mut specs = vec![cubin_spec(LAB_CUBIN.into())];
    for (tag, define) in MUTANTS {
        let mut defines = vec![(define.to_string(), "1".to_string())];
        if SLOW_MUTANTS.contains(&tag) {
            defines.push(slow());
        }
        if LATE_SCATTER_MUTANTS.contains(&tag) {
            defines.push(late());
        }
        specs.push(tagged(tag, defines));
    }
    specs.push(tagged(SLOW_CONTROL, vec![slow()]));
    specs.push(tagged(LATE_SCATTER_CONTROL, vec![late()]));
    for (tag, defines) in TIMING_VARIANTS {
        let defines = defines
            .iter()
            .map(|&(name, value)| (name.to_string(), value.to_string()))
            .collect();
        specs.push(tagged(tag, defines));
    }
    specs
}

/// Per-rank buffers of a state that takes calls of up to `capacity`
/// elements at `ranks` ranks: [`super::phased_bf16::buffers`]'s inbox and
/// epoch, and [`FLAG_WORDS`] flag words, all flags starting zeroed.
pub fn buffers(ranks: u32, capacity: u64) -> Result<super::phased_bf16::Buffers, Refusal> {
    let phased = super::phased_bf16::buffers(ranks, capacity)?;
    Ok(super::phased_bf16::Buffers {
        flag_bytes: FLAG_WORDS * 4,
        ..phased
    })
}

/// The arguments of `call` (the phased BF16 two-shot's).
pub fn arguments(call: Call) -> Result<ArgumentList, Refusal> {
    super::phased_bf16::arguments(SYMBOL, call)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_entry_takes_the_phased_two_shot_parameters() {
        let ours = entries();
        let theirs = super::super::phased_bf16::entries();
        assert_eq!(ours.len(), 1);
        assert_eq!(ours[0].symbol, SYMBOL);
        assert_eq!(
            format!("{:?}", ours[0].params),
            format!("{:?}", theirs[0].params)
        );
        assert_eq!(theirs[0].symbol, super::super::phased_bf16::SYMBOL);
        assert!(ours[0].body.contains("phased_turns_bf16::allreduce("));
        assert!(!super::super::phased_bf16::is_one_shot(SYMBOL));
    }

    #[test]
    fn buffers_add_the_turn_words() {
        let ours = buffers(4, 8192 * 4096).unwrap();
        let theirs = super::super::phased_bf16::buffers(4, 8192 * 4096).unwrap();
        assert_eq!(ours.inbox_bytes, theirs.inbox_bytes);
        assert_eq!(ours.epoch_bytes, theirs.epoch_bytes);
        assert_eq!(ours.flag_bytes, theirs.flag_bytes + TURN_WORDS * 4);
        assert!(buffers(0, GRANULE).is_err());
    }

    #[test]
    fn lab_cubins_rename_their_symbols() {
        let specs = lab_cubin_specs();
        assert_eq!(specs.len(), 3 + MUTANTS.len() + TIMING_VARIANTS.len());
        assert_eq!(specs[0].name, LAB_CUBIN);
        assert!(specs[0].sources_root.join(HEADER).is_file());
        for spec in &specs[1..] {
            assert!(spec.entries[0].symbol.starts_with("chop_comm_test_"));
            assert_eq!(spec.entries[0].body, specs[0].entries[0].body);
            assert!(spec.defines.len() >= 2);
        }
        let names: Vec<_> = specs.iter().map(|spec| spec.name.clone()).collect();
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), names.len());
    }
}
