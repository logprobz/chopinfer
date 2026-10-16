//! The text rank images' weight plans prepared as a serving host prepares
//! them before create (`runtime_host::direct::PlannedImageV1::new`, with
//! this bundle's exported transforms): every record of every rank is checked
//! against the checkpoint's headers. Only headers are read.
//!
//! The fake host prepares its synthetic checkpoint's plans this way before
//! it serves them, so a record the host would refuse fails every fake-host
//! create of that layout. `real_checkpoints_prepare_every_rank` prepares
//! each checkpoint named by its variable (skipped when unset), and the
//! NVFP4 tests create from those plans.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use model_glm5next_contract::quantization::{
    FP8_BLOCK128, Layout, NVFP4_DENSE_ROUTED, NVFP4_ROUTED,
};
use runtime_bundle_abi::weight_plan::WeightPlanV1;
use runtime_host::direct::{
    DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1, implementation_v1,
};

use super::{RANK_IMAGE_ROLE, plan, test_support};

const RANKS: u32 = 4;

/// Plans every rank of `checkpoint` through the bundle's plan export and
/// prepares the plans as the host does.
fn prepare(checkpoint: DirectCheckpointV1) -> Result<(PlannedImageV1, Vec<WeightPlanV1>), String> {
    let checkpoint = Arc::new(checkpoint);
    let implementation = implementation_v1([7; 32]);
    let plans = (0..RANKS)
        .map(|rank| {
            plan(&checkpoint.request(RANK_IMAGE_ROLE, RANKS, rank, implementation, vec![], vec![]))
                .map(|plan| plan.expect("the rank image role is planned"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let image = PlannedImageV1::new(
        checkpoint,
        plans.clone(),
        Some(crate::chop_bundle_weight_transform_v1),
        implementation,
        DirectLoadOptionsV1 {
            threads_per_rank: 1,
            read_ahead: false,
            direct_reads: false,
            shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
        },
    )?;
    Ok((image, plans))
}

/// A new directory holding the synthetic checkpoint of `layout` with
/// `config` that `test_support` plans from: its one safetensors file is
/// sparse past the header.
fn synthetic(layout: &Layout, config: &[u8]) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let request = test_support::checkpoint_request(layout, config, 0, |_| {});
    let root = std::env::temp_dir().join(format!(
        "glm5next-host-plan-{}-{}-{}",
        layout.id,
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    let [file] = &request.files[..] else {
        panic!("one checkpoint file")
    };
    let mut bytes = (file.header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&file.header);
    std::fs::write(root.join(&file.name), &bytes).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(root.join(&file.name))
        .unwrap()
        .set_len(file.byte_length)
        .unwrap();
    for document in &request.documents {
        std::fs::write(root.join(&document.name), &document.bytes).unwrap();
    }
    root
}

/// Prepares every rank's plan of the synthetic checkpoint of `layout` with
/// `config` as the host does; the host's refusal otherwise.
pub(super) fn prepare_synthetic(layout: &Layout, config: &[u8]) -> Result<(), String> {
    let root = synthetic(layout, config);
    let result = DirectCheckpointV1::open(&root).and_then(prepare);
    std::fs::remove_dir_all(&root).unwrap();
    let (image, plans) = result?;
    for (rank, plan) in plans.iter().enumerate() {
        assert_eq!(image.arena_bytes(rank), plan.arena_bytes, "{}", layout.id);
    }
    Ok(())
}

/// Every rank's planned image of the checkpoint that `variable` names,
/// prepared as the host does; `None` when it is unset.
pub(super) fn real_planned(variable: &str) -> Option<Vec<super::Planned>> {
    let root = std::env::var_os(variable)?;
    let (_, plans) = DirectCheckpointV1::open(Path::new(&root))
        .and_then(prepare)
        .unwrap_or_else(|error| panic!("{variable}: {error}"));
    Some(plans.iter().map(super::Planned::of).collect())
}

/// Each layout's scalar globals (NVFP4 `weight_scale_2`, alone or packed
/// per expert and projection) are records the host gathers as one element.
#[test]
fn each_layout_prepares_every_rank() {
    let layouts: [(&Layout, &[u8]); 3] = [
        (
            &FP8_BLOCK128,
            include_bytes!("../../../contract/fixtures/config.json"),
        ),
        (
            &NVFP4_ROUTED,
            include_bytes!("../../../contract/fixtures/nvfp4-config.json"),
        ),
        (
            &NVFP4_DENSE_ROUTED,
            include_bytes!("../../../contract/fixtures/nvidia-nvfp4-config.json"),
        ),
    ];
    let refused: Vec<String> = layouts
        .into_iter()
        .filter_map(|(layout, config)| {
            prepare_synthetic(layout, config)
                .err()
                .map(|error| format!("{}: {error}", layout.id))
        })
        .collect();
    assert!(refused.is_empty(), "{refused:#?}");
}

/// Each configured checkpoint's headers plan and prepare on every rank, as
/// at a serving host's create, and the host produces every scalar global
/// (one F32 element) as a finite positive scale read from the checkpoint.
#[test]
fn real_checkpoints_prepare_every_rank() {
    for variable in [
        "CHOPINFER_GLM5NEXT_CHECKPOINT",
        "CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT",
        "CHOPINFER_GLM5NEXT_NVIDIA_NVFP4_CHECKPOINT",
    ] {
        let Some(root) = std::env::var_os(variable) else {
            eprintln!("{variable} unset: skipped");
            continue;
        };
        let (image, plans) = DirectCheckpointV1::open(Path::new(&root))
            .and_then(prepare)
            .unwrap_or_else(|error| panic!("{variable}: {error}"));
        let mut start = 0;
        let mut scalars = 0;
        for plan in &plans {
            for record in plan.records.iter().filter(|r| r.stored_shape == [1]) {
                let mut bytes = [0; 4];
                image
                    .read(start + record.arena_offset, &mut bytes)
                    .unwrap_or_else(|error| panic!("{variable}: {error}"));
                let scale = f32::from_le_bytes(bytes);
                assert!(
                    scale.is_finite() && scale > 0.0,
                    "{variable}: {:?} produced {scale}",
                    record.sources
                );
                scalars += 1;
            }
            start += plan.arena_bytes;
        }
        let records: usize = plans.iter().map(|plan| plan.records.len()).sum();
        eprintln!(
            "{variable}: {records} records over {RANKS} ranks prepared, {scalars} scalars produced"
        );
    }
}
