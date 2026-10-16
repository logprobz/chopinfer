//! The phased group gather's race cases (`tests/context_parallel_phased_races.rs`):
//! lab-only mutants of the served gather body, each with one wait switched
//! off and every wait bounded (`kernel-lab/cuda/probes/phased_races.cuh`;
//! every entry of their cubin is named as a comm entry, so
//! `CollectiveModules` loads it), the count of bounded waits that gave up, and the probes the cases
//! run around a transfer: a delay a late sender runs first and a dependent
//! consumer that snapshots a receive buffer on the device.

use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernels_sm120::context_parallel::phased;
use std::path::PathBuf;

pub const FAMILY: &str = "context_parallel_phased_races";

/// The mutants' cubin.
pub const MUTANTS: &str = "comm_group_phased_mutants_lab_sm120";
/// The probes' cubin.
pub const PROBES: &str = "probe_phased_races_sm120";

/// The mutant entries: the served body unchanged, without the incoming-data
/// wait, and without the reuse acknowledgement wait.
pub const COPY: &str = "chop_comm_lab_phased_copy_v1";
pub const NO_RECEIVE_WAIT: &str = "chop_comm_lab_phased_no_receive_wait_v1";
pub const NO_ACKNOWLEDGEMENT_WAIT: &str = "chop_comm_lab_phased_no_acknowledgement_wait_v1";
/// Moves the mutants' module's count of bounded waits that gave up into
/// out[0] and clears it (not a collective; named as the cubin's comm
/// entries are).
pub const TIMEOUTS: &str = "chop_comm_lab_phased_timeouts_v1";

pub const DELAY: &str = "chop_probe_phased_delay_v1";
pub const SNAPSHOT: &str = "chop_probe_phased_snapshot_v1";

fn spec(name: &str, entries: Vec<EntrySpec>) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_dirs: vec![PathBuf::from("kernels-sm120/cuda")],
        headers: vec!["#include \"kernel-lab/cuda/probes/phased_races.cuh\"".into()],
        defines: vec![],
        entries,
    }
}

pub fn cubins() -> Vec<CubinSpec> {
    let mutant = |symbol: &str, receive: bool, acknowledgement: bool| {
        EntrySpec::new(
            symbol.into(),
            phased::params(),
            format!(
                "chop::probe::phased_races::gather<{receive}, {acknowledgement}>(input, \
                 input_stride, outputs, flags, epoch, words, rank, ranks);"
            ),
            256,
            1,
        )
    };
    let pointer = |name: &str, cxx_type: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    crate::declared(
        &[MUTANTS, PROBES],
        vec![
            spec(
                MUTANTS,
                vec![
                    mutant(COPY, true, true),
                    mutant(NO_RECEIVE_WAIT, false, true),
                    mutant(NO_ACKNOWLEDGEMENT_WAIT, true, false),
                    EntrySpec::new(
                        TIMEOUTS.into(),
                        vec![pointer("out", "unsigned int")],
                        "chop::probe::phased_races::take_timeouts(out);".into(),
                        32,
                        1,
                    ),
                ],
            ),
            spec(
                PROBES,
                vec![
                    EntrySpec::new(
                        DELAY.into(),
                        vec![P::U64 { name: "ns".into() }],
                        "chop::probe::phased_races::delay(ns);".into(),
                        32,
                        1,
                    ),
                    EntrySpec::new(
                        SNAPSHOT.into(),
                        vec![
                            pointer("source", "const uint32_t"),
                            pointer("target", "uint32_t"),
                            P::U64 {
                                name: "words".into(),
                            },
                            P::U64 {
                                name: "hold_ns".into(),
                            },
                        ],
                        "chop::probe::phased_races::snapshot(source, target, words, hold_ns);"
                            .into(),
                        256,
                        1,
                    ),
                ],
            ),
        ],
    )
}
