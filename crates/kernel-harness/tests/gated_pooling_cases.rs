//! Required preparation and CPU declarations, including verify mappings.

use kernel_core::contract::RecurrentPhase;
use kernel_core::gated_pooling::GatedPoolingQuery;
use kernel_harness::kit::{self, Modules, cases};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::gated_pooling::Entry;

struct CpuModules;

impl Modules for CpuModules {
    fn load<'s>(&'s self, _session: &'s Session) -> LabResult<Box<kit::Lookup<'s>>> {
        Err("CPU declarations do not load GPU modules".into())
    }
}

#[test]
fn all_pooling_shapes_and_verify_mappings_declare_without_refusals() {
    let mut entries = Vec::new();
    for width in [128, 512] {
        for (name, phase) in [
            ("decode", RecurrentPhase::Decode),
            ("verify", RecurrentPhase::Verify { max_rows: 6 }),
        ] {
            entries.push(
                Entry::new(
                    GatedPoolingQuery {
                        group: 2,
                        width,
                        norm_eps_bits: 1e-20f32.to_bits(),
                        phase,
                        capacity: 12,
                    },
                    &format!("chop_pooling_case_{name}_{width}"),
                )
                .unwrap(),
            );
        }
    }
    let mut ops = cases::gated_pooling::family(&entries);
    ops.extend(cases::gated_pooling::steps::family(&entries));
    assert_eq!(ops.len(), 6);
    assert!(ops.iter().all(|op| op.required));
    let backend = Sm120::new().with_gated_pooling(entries);
    let report = kit::conform(&backend, &CpuModules, None, &ops).unwrap();
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    assert_eq!(report.prepared().len(), 6);
}
