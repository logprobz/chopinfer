//! The low-latency rank-ordered F32 all-reduce over CTAs
//! (`comm::ll_f32::lab_cubin_specs`): the served entry and its mutants, and
//! the PDL producer its tests run behind.

use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use std::path::PathBuf;

pub const FAMILY: &str = "allreduce_ll_f32";

pub fn cubins() -> Vec<CubinSpec> {
    let mut specs = kernels_sm120::comm::ll_f32::lab_cubin_specs();
    specs.push(pdl_late_trigger_spec());
    specs
}

/// The PDL producer that triggers its dependents only after a lead
/// (`kernel-lab/cuda/probes/pdl_late_trigger.cuh`), which leaves room for
/// every rank's all-reduce to be queued before it triggers, and the
/// control's early reader.
fn pdl_late_trigger_spec() -> CubinSpec {
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let u64 = |name: &str| P::U64 { name: name.into() };
    CubinSpec {
        name: "probe_pdl_late_trigger_sm120".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_dirs: vec![],
        headers: vec!["#include \"kernel-lab/cuda/probes/pdl_late_trigger.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            "chop_probe_pdl_late_trigger_producer_v1".into(),
            vec![
                ptr("fresh", "const uint32_t"),
                ptr("output", "uint32_t"),
                u64("words"),
                u64("lead_ns"),
                u64("tail_ns"),
            ],
            "chop::probe::pdl_late_trigger_producer(fresh,output,words,lead_ns,tail_ns);".into(),
            128,
            1,
        ), EntrySpec::new(
            "chop_probe_pdl_early_reader_v1".into(),
            vec![
                ptr("input", "const uint32_t"),
                ptr("seen", "uint32_t"),
                u64("words"),
            ],
            "chop::probe::pdl_early_reader(input,seen,words);".into(),
            128,
            1,
        )],
    }
}
