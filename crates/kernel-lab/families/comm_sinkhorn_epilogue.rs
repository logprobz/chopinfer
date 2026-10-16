//! The Sinkhorn mHC comm epilogue at the Sinkhorn engine's served profile
//! (`comm::sinkhorn_epilogue::lab_cubin_specs`), and the PDL producer its
//! one-shot's late-wait mutation runs behind.

use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use std::path::PathBuf;

pub const FAMILY: &str = "comm_sinkhorn_epilogue";

pub fn cubins() -> Vec<CubinSpec> {
    let mut specs = kernels_sm120::comm::sinkhorn_epilogue::lab_cubin_specs();
    specs.push(pdl_watch_spec());
    specs
}

/// The PDL producer that holds its write until its dependent got ahead
/// (`kernel-lab/cuda/probes/pdl_watch.cuh`): the one-shot epilogue's
/// late-wait mutation runs behind it.
fn pdl_watch_spec() -> CubinSpec {
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    CubinSpec {
        name: "probe_pdl_watch_sm120".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_dirs: vec![],
        headers: vec!["#include \"kernel-lab/cuda/probes/pdl_watch.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            "chop_probe_pdl_watch_producer_v1".into(),
            vec![
                ptr("fresh", "const uint32_t"),
                ptr("output", "uint32_t"),
                P::U64 {
                    name: "words".into(),
                },
                ptr("epoch", "const unsigned int"),
                ptr("flag", "const unsigned int"),
                P::U64 {
                    name: "hold_ns".into(),
                },
                ptr("record", "uint32_t"),
            ],
            "chop::probe::pdl_watch_producer(fresh,output,words,epoch,flag,hold_ns,record);".into(),
            128,
            1,
        )],
    }
}
