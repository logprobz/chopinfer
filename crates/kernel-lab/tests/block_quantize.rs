use kernel_lab::kit::cases::block_quantize;
use kernel_lab::kit::{self, LabModules, OpResult, Outcome, Report};
use kernels_sm120::row::block_quantize::{LAB_CUBIN, lab_entries};

fn check(report: &Report, gpu: bool) {
    assert_eq!(report.ops.len(), 48);
    assert_eq!(report.prepared().len(), 48);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            panic!("required quantizer refused")
        };
        for name in [
            "declaration",
            "need",
            "conformance",
            kit::PDL,
            "property rows",
            "property chunk",
            "property batch",
        ] {
            if !gpu && !["declaration", "need"].contains(&name) {
                continue;
            }
            let check = checks
                .iter()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("{}: missing {name}", op.name));
            assert!(
                matches!(check.outcome, Outcome::Passed(_)),
                "{}: {name}: {:?}",
                op.name,
                check.outcome
            );
        }
    }
}

#[test]
fn block_quantize_prepares_every_case() {
    assert_eq!(
        kernel_lab::cubin_entries(LAB_CUBIN).unwrap().len(),
        lab_entries().len()
    );
    let report = kit::conform(
        &block_quantize::sm120(),
        &LabModules,
        None,
        &block_quantize::required(),
    )
    .unwrap();
    check(&report, false);
}

#[test]
fn block_quantize_has_pdl_and_no_tf32() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let dir =
        kernel_lab::receipt_root().join(format!("block-quantize-audit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("block_quantize.cubin");
    std::fs::write(&path, kernel_lab::cubin(LAB_CUBIN).unwrap()).unwrap();
    let result = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&path)
        .output()
        .unwrap();
    assert!(result.status.success());
    let sass = String::from_utf8(result.stdout).unwrap();
    assert!(!sass.to_uppercase().contains("TF32"));
    assert!(sass.contains("PREEXIT"));
    assert!(sass.contains("ACQBULK"));
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(dir).unwrap();
}

#[test]
fn block_quantize_bitwise_gpu() {
    kernel_lab::gpu_test(256 << 20, |session| {
        let report = kit::conform(
            &block_quantize::sm120(),
            &LabModules,
            Some(session),
            &block_quantize::required(),
        )?;
        check(&report, true);
        session.receipt(
            "block-quantize",
            std::collections::BTreeMap::from([
                ("report".into(), report.json()),
                ("power_cap_watts".into(), serde_json::json!(250)),
            ]),
        )?;
        Ok(())
    })
    .unwrap();
}
