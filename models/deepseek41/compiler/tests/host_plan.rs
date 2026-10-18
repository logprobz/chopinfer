//! The weight plan (`model_deepseek41_contract::weight_plan`) prepared and
//! executed as a serving host prepares and executes it
//! (`runtime_host::direct::PlannedImageV1`).
//!
//! `every_rank_prepares_from_the_headers` prepares every rank of each
//! compiled geometry's synthetic checkpoint (`model_deepseek41::synthetic`)
//! from a sparse file: the host checks every record against the headers.
//!
//! `planned_bytes_are_the_checkpoint_s` reads views of the synthetic
//! checkpoint that `CHOPINFER_DEEPSEEK41_SYNTHETIC_CHECKPOINT` names (the
//! `geometry-checkpoint` example's output; skipped when unset) and checks
//! them against the checkpoint's own bytes: side tensors copied, a
//! vocabulary shard of the embedding, and each dense block-32 FP8 matrix
//! resident as MXFP8 rows, its E4M3 codes unchanged and each 32x32 block
//! scale repeated on its 32 rows, for the replicated, column-split and
//! row-split (K-split) matrices on every rank.
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use model_deepseek41::synthetic;
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, admit, check_ranks};
use model_deepseek41_contract::geometry::GEOMETRIES;
use model_deepseek41_contract::serving::{INDEX_SCHEMA, RANK_IMAGE_ROLE};
use model_deepseek41_contract::weight_plan::plan;
use runtime_bundle_abi::weight_plan::WeightPlanV1;
use runtime_host::direct::{
    DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1, implementation_v1,
};
use serde_json::Value;

mod support;
use support::sparse;

/// Every rank's plan of the checkpoint at `root`, prepared as the host
/// prepares it before create.
fn prepare(root: &Path, ranks: u32) -> Result<(PlannedImageV1, Vec<WeightPlanV1>), String> {
    let checkpoint = Arc::new(DirectCheckpointV1::open(root)?);
    let implementation = implementation_v1([0; 32]);
    let plans = (0..ranks)
        .map(|rank| {
            plan(&checkpoint.request(RANK_IMAGE_ROLE, ranks, rank, implementation, vec![], vec![]))
                .map(|plan| plan.expect("the text image is planned"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let image = PlannedImageV1::new(
        checkpoint,
        plans.clone(),
        None,
        implementation,
        DirectLoadOptionsV1 {
            threads_per_rank: 2,
            read_ahead: false,
            direct_reads: false,
            shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
        },
    )?;
    Ok((image, plans))
}

#[test]
#[ignore = "the grouped dense output LoRA awaits its row-major-groups derivation (C1)"]
fn every_rank_prepares_from_the_headers() {
    for geometry in GEOMETRIES {
        let root = sparse(geometry);
        for &ranks in SUPPORTED_RANKS {
            if check_ranks(geometry, ranks).is_err() {
                continue;
            }
            let started = std::time::Instant::now();
            let (image, plans) = prepare(&root, ranks)
                .unwrap_or_else(|error| panic!("{} tp{ranks}: {error}", geometry.id));
            for (rank, plan) in plans.iter().enumerate() {
                assert_eq!(image.arena_bytes(rank), plan.arena_bytes);
            }
            eprintln!(
                "{} tp{ranks}: every rank prepared in {:?}",
                geometry.id,
                started.elapsed()
            );
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
}

/// One tensor of the checkpoint: its header entry and bytes.
struct Source {
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

/// Equal bytes, or the lengths and the first differing byte.
fn same(got: &[u8], expected: &[u8], what: &str) {
    if got == expected {
        return;
    }
    let first = got.iter().zip(expected).position(|(a, b)| a != b);
    panic!(
        "{what}: {} bytes, expected {}; first difference at {first:?} ({:?} vs {:?})",
        got.len(),
        expected.len(),
        first.map(|i| got[i]),
        first.map(|i| expected[i])
    );
}

fn source(root: &Path, header: &Value, data: u64, name: &str) -> Source {
    let entry = &header[name];
    let [start, end] = [0, 1].map(|i| entry["data_offsets"][i].as_u64().unwrap());
    let mut bytes = vec![0; (end - start) as usize];
    std::fs::File::open(root.join(synthetic::FILE))
        .unwrap()
        .read_exact_at(&mut bytes, data + start)
        .unwrap();
    let shape = entry["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap())
        .collect();
    Source { shape, bytes }
}

/// The 32x32 block-scaled `[n, k]` matrix `stem` resident as MXFP8 rows:
/// rows `rows` and columns `columns` of its E4M3 codes unchanged, and row
/// `r`'s scale for K block `b` the block scale of `(r / 32, b)`.
fn mxfp8_rows(
    codes: &Source,
    scales: &Source,
    rows: std::ops::Range<u64>,
    columns: std::ops::Range<u64>,
) -> (Vec<u8>, Vec<u8>) {
    let k = codes.shape[1];
    let blocks = scales.shape[1];
    let mut expected_codes = Vec::new();
    let mut expected_scales = Vec::new();
    for r in rows {
        let row = (r * k) as usize;
        expected_codes.extend_from_slice(
            &codes.bytes[row + columns.start as usize..row + columns.end as usize],
        );
        let block_row = ((r / 32) * blocks) as usize;
        for b in columns.start / 32..columns.end.div_ceil(32) {
            expected_scales.push(scales.bytes[block_row + b as usize]);
        }
    }
    (expected_codes, expected_scales)
}

#[test]
#[ignore = "the grouped dense output LoRA awaits its row-major-groups derivation (C1)"]
fn planned_bytes_are_the_checkpoint_s() {
    let Some(root) = std::env::var_os("CHOPINFER_DEEPSEEK41_SYNTHETIC_CHECKPOINT") else {
        eprintln!("CHOPINFER_DEEPSEEK41_SYNTHETIC_CHECKPOINT is unset: skipped");
        return;
    };
    let root = PathBuf::from(root);
    let config: Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
    let geometry = admit(&config).unwrap();
    let mut prefix = [0_u8; 8];
    std::fs::File::open(root.join(synthetic::FILE))
        .unwrap()
        .read_exact_at(&mut prefix, 0)
        .unwrap();
    let header_bytes = u64::from_le_bytes(prefix);
    let mut header = vec![0; header_bytes as usize];
    std::fs::File::open(root.join(synthetic::FILE))
        .unwrap()
        .read_exact_at(&mut header, 8)
        .unwrap();
    let header: Value = serde_json::from_slice(&header).unwrap();
    let data = 8 + header_bytes;
    let tensor = |name: &str| source(&root, &header, data, name);
    for &ranks in SUPPORTED_RANKS {
        if check_ranks(geometry, ranks).is_err() {
            continue;
        }
        let (image, plans) = prepare(&root, ranks).unwrap();
        let mut start = 0;
        for (rank, plan) in plans.iter().enumerate() {
            let index =
                bundle_runtime::index::parse(&plan.components[0].bytes, rank as u32, INDEX_SCHEMA)
                    .unwrap();
            let read = |name: &str| -> Vec<u8> {
                let view = index
                    .views
                    .get(name)
                    .unwrap_or_else(|| panic!("tp{ranks} rank {rank}: no view {name}"));
                let mut bytes = vec![0; view.bytes as usize];
                image.read(start + view.offset, &mut bytes).unwrap();
                bytes
            };
            let at = format!("tp{ranks} rank {rank}");
            // Side tensors are copies.
            for name in [
                "norm.weight",
                "layers.0.attn_norm.weight",
                "layers.0.hc_attn_fn",
            ] {
                same(&read(name), &tensor(name).bytes, &format!("{at} {name}"));
            }
            // The embedding's vocabulary shard.
            let embed = tensor("embed.weight");
            let row = (embed.shape[1] * 2) as usize;
            let shard = embed.shape[0] as usize / ranks as usize;
            same(
                &read("embed.weight"),
                &embed.bytes[rank * shard * row..(rank + 1) * shard * row],
                &format!("{at} embed"),
            );
            // Dense block-32 FP8 matrices as MXFP8 rows: replicated, split
            // by rows, split by K.
            let r = rank as u64;
            let n = u64::from(ranks);
            let matrices = |stem: &str| {
                (
                    tensor(&format!("{stem}.weight")),
                    tensor(&format!("{stem}.scale")),
                )
            };
            let (codes, scales) = matrices("layers.0.attn.wq_a");
            let rows = codes.shape[0];
            let k = codes.shape[1];
            let expected = mxfp8_rows(&codes, &scales, 0..rows, 0..k);
            same(
                &read("layers.0.attn.wq_a.weight"),
                &expected.0,
                &format!("{at} wq_a codes"),
            );
            same(
                &read("layers.0.attn.wq_a.scale"),
                &expected.1,
                &format!("{at} wq_a scales"),
            );
            let (codes, scales) = matrices("layers.0.attn.wq_b");
            let (rows, k) = (codes.shape[0] / n, codes.shape[1]);
            let expected = mxfp8_rows(&codes, &scales, r * rows..(r + 1) * rows, 0..k);
            same(
                &read("layers.0.attn.wq_b.weight"),
                &expected.0,
                &format!("{at} wq_b codes"),
            );
            same(
                &read("layers.0.attn.wq_b.scale"),
                &expected.1,
                &format!("{at} wq_b scales"),
            );
            let (codes, scales) = matrices("layers.0.attn.wo_b");
            let (rows, k) = (codes.shape[0], codes.shape[1] / n);
            let expected = mxfp8_rows(&codes, &scales, 0..rows, r * k..(r + 1) * k);
            same(
                &read("layers.0.attn.wo_b.weight"),
                &expected.0,
                &format!("{at} wo_b codes"),
            );
            same(
                &read("layers.0.attn.wo_b.scale"),
                &expected.1,
                &format!("{at} wo_b scales"),
            );
            // The output LoRA's groups, stacked in the checkpoint: a rank
            // holds whole groups, each with its own blocks' scales.
            let (codes, scales) = matrices("layers.0.attn.wo_a");
            let (rows, k) = (codes.shape[0] / n, codes.shape[1]);
            let expected = mxfp8_rows(&codes, &scales, r * rows..(r + 1) * rows, 0..k);
            same(
                &read("layers.0.attn.wo_a.weight"),
                &expected.0,
                &format!("{at} wo_a codes"),
            );
            same(
                &read("layers.0.attn.wo_a.scale"),
                &expected.1,
                &format!("{at} wo_a scales"),
            );
            start += plan.arena_bytes;
        }
        eprintln!("tp{ranks}: planned bytes match the checkpoint");
    }
}
