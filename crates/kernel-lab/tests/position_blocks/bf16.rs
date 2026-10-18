//! Storage equality over the existing 1M-capacity, four-Reindex fixture.
use kernel_core::pooled_indexer::IndexerKeyFormat;
use kernel_lab::families::position_bf16 as family;
use kernels_sm120::backend::Sm120;
use std::collections::BTreeMap;

pub(super) fn backend() -> Sm120 {
    Sm120::new()
        .with_position_indexer(family::dense())
        .with_position_indexer_bf16(family::dense())
        .with_position_block_indexer(family::compact())
        .with_position_block_indexer_bf16(family::compact())
        .with_indexer_candidate_block_ids(family::filters())
        .with_indexer_candidate_blocks(family::dense_filters())
}
#[test]
fn bf16_position_scores_and_selections_match_native_on_a_gpu() {
    kernel_lab::gpu_test(2 << 30, |session| {
        for entry in family::compact() {
            let native =
                super::chain::comparison_chain(session, &entry, 4, IndexerKeyFormat::Fp4, false)?;
            for chunk in [4, 1, 3] {
                let bf16 = super::chain::comparison_chain(
                    session,
                    &entry,
                    chunk,
                    IndexerKeyFormat::Bf16,
                    false,
                )?;
                assert_eq!(
                    native, bf16,
                    "{} chunk {chunk}: BF16/native score bits, IDs and selections",
                    entry.name
                );
            }
        }
        session.receipt(
            "position-bf16-equality",
            BTreeMap::from([
                ("position_capacity".into(), serde_json::json!(1 << 20)),
                ("chunk_rows".into(), serde_json::json!([4, 1, 3])),
                ("reindex_passes".into(), serde_json::json!(4)),
                (
                    "exact".into(),
                    serde_json::json!([
                        "source_scores",
                        "compact_scores",
                        "dense_mask",
                        "ids",
                        "status",
                        "selection"
                    ]),
                ),
            ]),
        )?;
        Ok(())
    })
    .expect("BF16/native indexer equality");
}
#[test]
fn bf16_position_rounding_mutants_fail_on_a_gpu() {
    kernel_lab::gpu_test(2 << 30, |session| {
        for entry in family::compact() {
            let failure =
                super::chain::comparison_chain(session, &entry, 1, IndexerKeyFormat::Bf16, true)
                    .unwrap_err();
            assert!(
                failure.contains("mapped score bits"),
                "wrong rounding-mutant failure: {failure}"
            );
            session.receipt(
                &format!("position-bf16-rounding-{}", entry.name),
                BTreeMap::from([("failure".into(), serde_json::json!(failure))]),
            )?;
        }
        Ok(())
    })
    .expect("BF16 rounding controls");
}
#[test]
fn bf16_position_cubin_has_no_traps_tf32_or_spills() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let root = std::env::temp_dir().join(format!("chopinfer-position-bf16-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("position.cubin");
    std::fs::write(&path, kernel_lab::cubin(family::CUBIN).unwrap()).unwrap();
    let spec = family::cubins().remove(0);
    let resources = kernel_build::audit_cubin(&toolchain, &spec, &path).unwrap();
    assert_eq!(resources.len(), spec.entries.len());
    for r in resources {
        assert_eq!((r.stack_bytes, r.local_bytes), (0, 0), "{}", r.symbol);
    }
    let dump = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&path)
        .output()
        .unwrap();
    assert!(dump.status.success());
    let sass = String::from_utf8(dump.stdout).unwrap();
    for forbidden in ["TRAP", "TF32", "LDL", "STL"] {
        assert!(!sass.contains(forbidden), "{forbidden}");
    }
    std::fs::remove_dir_all(root).unwrap();
}
