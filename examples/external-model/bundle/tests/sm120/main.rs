//! The example's SM120 selection: its locked artifacts, its refusals and the
//! launches of the ops it prepares.

mod cases;
#[path = "../../kernel_cases.rs"]
mod kernel_cases;
#[path = "../../src/sm120/own_ops.rs"]
mod own_ops;
#[path = "../../src/sm120/selection.rs"]
mod selection;

use example_contract::ops::{self, Ops};
use kernel_build::{BUILD_REPORT_FILE, BuildReport};
use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, RmsNormKind};
use kernel_core::rmsnorm::{RmsNormBuffers, RmsNormNumerics};
use std::path::Path;

fn selected() -> kernel_build::KernelSelection {
    selection::selection(
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap(),
        &kernels_sm120::cuda_source_root(),
    )
    .unwrap()
}

fn prepared() -> Ops {
    Ops::prepare(&selection::backend(), &own_ops::Sm120Own).unwrap()
}

fn view(handle: u64, bytes: usize) -> View {
    View {
        handle,
        bytes: bytes as u64,
    }
}

/// The RMSNorm's buffers over handles 1 to 4, as the kernel-lab adapter
/// binds them.
fn norm_buffers() -> RmsNormBuffers<View> {
    let rows = ops::WIDTH * ops::CAPACITY as usize * 2;
    RmsNormBuffers {
        x: view(1, rows),
        weight: view(2, ops::WIDTH * 2),
        residual: None,
        y: view(3, rows),
        live_rows: view(4, 4),
        status: None,
        status_row_offset: 0,
    }
}

#[test]
fn shared_and_local_entries_match_audited_artifacts() {
    let report = BuildReport::read(&Path::new(env!("OUT_DIR")).join(BUILD_REPORT_FILE)).unwrap();
    cases::check_artifacts(&report, &selected());
}

#[test]
fn same_shape_with_incompatible_semantics_is_refused() {
    cases::check_refusals(&selected());
    let hf = kernel_core::rmsnorm::RmsNormRequirement {
        kind: RmsNormKind::HfBf16Intermediate,
        numerics: Some(RmsNormNumerics::HF_ORDERED),
        ..ops::rmsnorm()
    };
    let backend = selection::backend();
    assert!(backend.rmsnorm(&hf, Need::rows(ops::CAPACITY)).is_err());
    assert!(
        backend
            .rmsnorm(&ops::rmsnorm(), Need::rows(ops::CAPACITY + 1))
            .is_err()
    );
}

/// The prepared ops launch what the builders plan and pack: the RMSNorm the
/// selection compiles, gridded at its capacity whatever rows are live, and
/// the probe's one block.
#[test]
fn prepared_ops_launch_the_selected_entries() {
    let ops = prepared();
    let norm = selection::rmsnorm().unwrap();
    let launches = ops
        .rmsnorm
        .launches(&norm_buffers(), View::EMPTY, Counts::rows(ops::CAPACITY))
        .unwrap();
    assert_eq!(launches.len(), 1);
    let launch = &launches[0];
    assert_eq!(launch.symbol, selection::RMS_SYMBOL);
    for live in kernel_cases::LIVE_ROWS {
        assert_eq!(launch.plan, norm.plan(live).unwrap());
    }
    assert_eq!(
        launch.arguments,
        norm.arguments(RmsNormBuffers {
            x: 1,
            weight: 2,
            residual: None,
            y: 3,
            live_rows: 4,
            status: None,
            status_row_offset: 0,
        })
        .unwrap()
    );
    assert_eq!(
        launch.arguments.0[6],
        Argument::U32(selection::requirement().epsilon_bits)
    );
    assert!(
        ops.rmsnorm
            .launches(
                &norm_buffers(),
                View::EMPTY,
                Counts::rows(ops::CAPACITY + 1)
            )
            .is_err()
    );
    for value in kernel_cases::PROBE_VALUES {
        let buffers = ops::ProbeBuffers {
            result: view(1, 4),
            value,
        };
        let launches = ops
            .probe
            .launches(&buffers, View::EMPTY, Counts::rows(1))
            .unwrap();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].symbol, selection::PROBE_SYMBOL);
        assert_eq!(
            launches[0].plan,
            LaunchPlan {
                grid: [1; 3],
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1; 3],
                pdl: false,
            }
        );
        assert_eq!(
            launches[0].arguments,
            ArgumentList(vec![Argument::DeviceBuffer(1), Argument::U32(value)])
        );
    }
}

#[test]
fn model_cases_cover_live_rows_and_declared_oracles() {
    for live in kernel_cases::LIVE_ROWS {
        match kernel_cases::rms_reference(live).unwrap() {
            None => assert_eq!(live, 0),
            Some(reference) => {
                assert_eq!(reference.values.len(), live as usize * ops::WIDTH);
                assert!(
                    reference
                        .values
                        .iter()
                        .all(|v| v.is_finite() && v.abs() < 10.0)
                );
            }
        }
    }
    assert!(kernel_cases::rms_reference(ops::CAPACITY + 1).is_err());
    assert_eq!(
        kernel_cases::PADDING,
        100.0_f32.to_bits().wrapping_shr(16) as u16
    );
    for value in kernel_cases::PROBE_VALUES {
        let reference = ops::sigmoid_probe(value);
        assert!((0.5..1.0).contains(&reference));
        assert!((f64::from(reference as f32) - reference).abs() <= ops::PROBE_ABS_BOUND);
    }
}
