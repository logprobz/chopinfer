//! Compile and audit the model's selected shared kernels before admitting their
//! operation capabilities. Numerical evidence lives in the idle-GPU suite.
#[path = "../src/sm120/precision_selection.rs"]
#[allow(
    dead_code,
    reason = "the publisher also uses the legacy composition helper"
)]
mod precision_selection;

#[test]
fn mixed_projection_selection_compiles_with_audited_resources() {
    use kernel_build::{BuildOptions, Toolchain, report::LockStatus};
    use std::path::PathBuf;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let selection = precision_selection::selection(&root).unwrap();
    let out = std::env::temp_dir().join(format!("qwen-precision-build-{}", std::process::id()));
    std::fs::create_dir(&out).unwrap();
    // A new candidate lock is private test output. The publisher's committed
    // lock is checked by the bundle build, never overwritten by this test.
    let lock = out.join("candidate.lock");
    let toolchain = Toolchain::detect().unwrap();
    let build = |update| {
        selection.build(
            &toolchain,
            &out,
            &lock,
            BuildOptions {
                lock_update: update,
                emit_cargo_directives: false,
                profile_line_info: false,
            },
        )
    };
    let candidate = build(true).unwrap();
    let verified = build(false).unwrap();
    let names = |cubins: &[kernel_build::BuiltCubin]| {
        cubins
            .iter()
            .map(|c| c.name.clone())
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert_eq!(names(&candidate), names(&verified));
    assert_eq!(candidate.len(), selection.specs().len());
    assert_eq!(verified.len(), selection.specs().len());
    // Every geometry's layout copies take their shapes at launch from one
    // shared cubin; its lookups keep compiled shapes in its own rows cubin.
    assert!(names(&verified).contains(precision_selection::SHAPE_FREE));
    for geometry in model_qwen38_contract::geometry::GEOMETRIES {
        let prefix = format!("{}_precision_", geometry.kernel_tag);
        let modules = names(&verified)
            .into_iter()
            .filter(|name| name != precision_selection::SHAPE_FREE)
            .filter_map(|name| name.strip_prefix(&prefix).map(str::to_owned))
            .collect::<std::collections::BTreeSet<_>>();
        // TP4's modules keep their names; TP2's carry a topology suffix and
        // only the entries TP4's lack.
        for topology in ["", "tp2_"] {
            for module in ["dense_v1", "grouped_v1", "rows_v1", "hash_v1"] {
                let required = format!("{topology}{module}");
                assert!(
                    modules.contains(&required),
                    "{} lacks {required}",
                    geometry.id
                );
            }
            assert!(
                !modules.contains(&format!("{topology}layouts_v1")),
                "{} has {topology}layouts_v1",
                geometry.id
            );
        }
        if geometry.id == "flash" {
            assert_eq!(modules.len(), 9);
            assert!(modules.contains("grouped_ring_v1"));
        }
    }
    selection.audited_contract_identity(&verified).unwrap();
    let report = kernel_build::BuildReport::read(&out.join("kernel-build-report.json")).unwrap();
    assert_eq!(report.lock_status, LockStatus::Verified);
    assert!(report.entry_declarations_complete && report.source_ownership_complete);
    report.verify_current().unwrap();
    if let Some(destination) = std::env::var_os("CHOPINFER_QWEN_PRECISION_CANDIDATE") {
        let destination = PathBuf::from(destination);
        assert!(
            !destination.exists(),
            "candidate export must not overwrite a lock"
        );
        std::fs::copy(&lock, destination).unwrap();
    }
    std::fs::remove_dir_all(out).unwrap();
}
