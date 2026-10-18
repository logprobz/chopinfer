//! The weight plan (`model_deepseek41_contract::weight_plan`) prepared and
//! executed as a serving host prepares and executes it
//! (`runtime_host::direct::PlannedImageV1`).
//!
//! `every_rank_prepares_from_the_headers` prepares every rank of each
//! compiled geometry's synthetic checkpoint (`model_deepseek41::synthetic`)
//! from a sparse file: the host checks every record against the headers.
//!
//! `every_planned_view_is_the_checkpoint_s` writes the synthetic geometry's
//! checkpoint with block scales that vary, prepares every rank at every rank
//! count, and checks every view of every rank index against the rank's share
//! of the checkpoint that the projection declaration selects.
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use model_deepseek41::synthetic;
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
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

/// How a rank selects a tensor: whole, a contiguous share of its rows, or
/// a contiguous share of its K.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Split {
    Whole,
    Rows,
    K,
}

/// Each declared source tensor's split, from the projection declaration
/// itself: `(name suffix after the stem, or the whole name, split)`, for
/// every dense, grouped dense and lookup projection (each fused operand on
/// its own) and every side tensor. Expert banks are matched by their view
/// name instead.
fn declared_splits() -> Vec<(String, Split)> {
    let declaration: Value =
        serde_json::from_str(model_deepseek41_contract::projections::DECLARATION).unwrap();
    let mut splits = Vec::new();
    for projection in declaration["projections"].as_array().unwrap() {
        let Some(template) = projection["source"]["tensor"].as_str() else {
            continue;
        };
        if template.contains("{expert}") {
            continue;
        }
        let split = match projection["split"]["class"].as_str().unwrap() {
            "replicate" => Split::Whole,
            "column" | "expert_owned" | "vocab_sharded" => Split::Rows,
            "row" => Split::K,
            other => panic!("{}: split {other}", projection["id"]),
        };
        let operands: Vec<&str> = projection["operands"]
            .as_array()
            .map(|list| list.iter().map(|o| o.as_str().unwrap()).collect())
            .unwrap_or_default();
        if operands.is_empty() {
            splits.push((template.replace("{stem}", ""), split));
        }
        for operand in operands {
            splits.push((
                template.replace("{stem}", "").replace("{operand}", operand),
                split,
            ));
        }
    }
    for side in declaration["side_tensors"].as_array().unwrap() {
        let split = if side["placement"].is_null() {
            Split::Whole
        } else {
            assert_eq!(side["placement"]["axis"], 0);
            Split::Rows
        };
        let template = side["source"]["tensor"].as_str().unwrap();
        splits.push((template.replace("{stem}", ""), split));
    }
    splits
}

/// The split of checkpoint tensor `name`: the longest declared suffix it
/// ends with, or its whole name.
fn split_of(splits: &[(String, Split)], name: &str) -> Split {
    splits
        .iter()
        .filter(|(suffix, _)| {
            name == suffix.as_str() || (suffix.starts_with('.') && name.ends_with(suffix.as_str()))
        })
        .max_by_key(|(suffix, _)| suffix.len())
        .unwrap_or_else(|| panic!("{name}: no declared source"))
        .1
}

/// Rank `rank`'s share of `0..n` over `ranks`.
fn share(n: u64, rank: u64, ranks: u64) -> std::ops::Range<u64> {
    assert_eq!(n % ranks, 0, "{n} over {ranks}");
    let part = n / ranks;
    rank * part..(rank + 1) * part
}

/// Rows `rows` and byte columns `columns` of a row-major `[_, row_bytes]`
/// tensor.
fn block(
    bytes: &[u8],
    row_bytes: u64,
    rows: std::ops::Range<u64>,
    columns: std::ops::Range<u64>,
) -> Vec<u8> {
    let mut out = Vec::new();
    for r in rows {
        let row = (r * row_bytes) as usize;
        out.extend_from_slice(&bytes[row + columns.start as usize..row + columns.end as usize]);
    }
    out
}

/// A new directory holding `geometry`'s synthetic checkpoint with UE8M0
/// scales that vary per block (so a misplaced block or slice shows), and
/// its config.
fn varied_checkpoint(geometry: &model_deepseek41_contract::geometry::Geometry) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "deepseek41-payload-{}-{}",
        geometry.id,
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let tensors: Vec<_> = synthetic::tensors(geometry)
        .unwrap()
        .into_iter()
        .map(|mut tensor| {
            if let synthetic::Fill::Ue8m0 { exponent } = tensor.fill {
                tensor.fill = synthetic::Fill::Ue8m0Varied {
                    exponent,
                    spread: 3,
                };
            }
            tensor
        })
        .collect();
    synthetic::write(&root, &tensors, 11).unwrap();
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&support::config(geometry)).unwrap(),
    )
    .unwrap();
    root
}

/// Every view of every rank's planned image, as the host prepares and reads
/// it, holds the rank's share of the checkpoint the declaration selects:
/// side tensors and lookup rows copied; block-32 FP8 matrices as MXFP8 rows,
/// each row carrying its own 32x32 block's scales, replicated, split by rows,
/// by K or by groups; BF16 matrices in MMA fragment order; expert banks
/// expert-major, each expert's operands split by rows (gate-up) or K (down),
/// codes in rows and scales in 16x8 tiles. The checkpoint's scales vary per
/// block, so a slice that starts in the wrong block fails. Runs when a
/// synthetic geometry is compiled.
#[test]
fn every_planned_view_is_the_checkpoint_s() {
    use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
    use kernel_core::transform::{Layout, Packed, ScaleTileLayout, Transform};
    let Some(geometry) = GEOMETRIES.iter().find(|g| g.id != "flash") else {
        eprintln!("no synthetic geometry is compiled: skipped");
        return;
    };
    let root = varied_checkpoint(geometry);
    let splits = declared_splits();
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
    let mut groups = std::collections::BTreeSet::new();
    for &ranks in SUPPORTED_RANKS {
        if check_ranks(geometry, ranks).is_err() {
            continue;
        }
        let n = u64::from(ranks);
        let (image, plans) = prepare(&root, ranks).unwrap();
        let mut start = 0;
        for (rank, plan) in plans.iter().enumerate() {
            let r = rank as u64;
            let index =
                bundle_runtime::index::parse(&plan.components[0].bytes, rank as u32, INDEX_SCHEMA)
                    .unwrap();
            for (name, view) in &index.views {
                let mut got = vec![0; view.bytes as usize];
                image.read(start + view.offset, &mut got).unwrap();
                let at = format!("tp{ranks} rank {rank} {name}");
                let generic: Vec<_> = name
                    .split('.')
                    .map(|part| {
                        if part.bytes().all(|b| b.is_ascii_digit()) {
                            "N"
                        } else {
                            part
                        }
                    })
                    .collect();
                groups.insert(generic.join("."));
                let expected = match view.layout_id.as_str() {
                    "mxfp8_e4m3_rows_v1" | "mxfp8_ue8m0_scale_rows_v1" => {
                        let stem = name
                            .strip_suffix(".weight")
                            .or_else(|| name.strip_suffix(".scale"))
                            .unwrap();
                        let codes = tensor(&format!("{stem}.weight"));
                        let scales = tensor(&format!("{stem}.scale"));
                        let (rows, k) = (codes.shape[0], codes.shape[1]);
                        let (rows, columns) = match split_of(&splits, &format!("{stem}.weight")) {
                            Split::Whole => (0..rows, 0..k),
                            Split::Rows => (share(rows, r, n), 0..k),
                            Split::K => (0..rows, share(k, r, n)),
                        };
                        let (codes, scales) = mxfp8_rows(&codes, &scales, rows, columns);
                        if name.ends_with(".weight") {
                            codes
                        } else {
                            scales
                        }
                    }
                    "bf16_row_major_v1" | "f32_row_major_v1" => {
                        let source = tensor(name);
                        let rows = source.shape[0];
                        let row_bytes = source.bytes.len() as u64 / rows;
                        match split_of(&splits, name) {
                            Split::Whole => source.bytes,
                            Split::Rows => {
                                block(&source.bytes, row_bytes, share(rows, r, n), 0..row_bytes)
                            }
                            Split::K => panic!("{at}: a row-major copy split by K"),
                        }
                    }
                    "bf16_mma_fragments_16x16_v1" => {
                        let source = tensor(name);
                        let (rows, k) = (source.shape[0], source.shape[1]);
                        let (rows, columns) = match split_of(&splits, name) {
                            Split::Whole => (0..rows, 0..k),
                            Split::Rows => (share(rows, r, n), 0..k),
                            Split::K => (0..rows, share(k, r, n)),
                        };
                        // The packed operand pads its rows to whole row
                        // tiles; the padding holds zeros.
                        let (local_rows, local_k) = (
                            (rows.end - rows.start) as usize,
                            (columns.end - columns.start) as usize,
                        );
                        let strip_bytes = local_k / 16 * 512;
                        assert_eq!(got.len() % strip_bytes, 0, "{at}: whole row strips");
                        let padded = got.len() / strip_bytes * 16;
                        assert!(padded >= local_rows, "{at}: {padded} rows for {local_rows}");
                        let layout = FragmentWeightLayout {
                            rows: padded,
                            k: local_k,
                            format: FragmentWeightFormat::Bf16,
                        };
                        let mut logical = layout
                            .unpack(&Packed {
                                bytes: got,
                                layout: Layout::FragmentMajor(layout),
                            })
                            .unwrap_or_else(|e| panic!("{at}: {e:?}"));
                        let padding = logical.split_off(local_rows * local_k * 2);
                        assert!(padding.iter().all(|&b| b == 0), "{at}: nonzero row padding");
                        got = logical;
                        block(
                            &source.bytes,
                            k * 2,
                            rows,
                            columns.start * 2..columns.end * 2,
                        )
                    }
                    "mxfp4_e2m1_rows_v1" | "mxfp4_ue8m0_scale_tiles_16x8_v1" => {
                        let (prefix, bank) = name.split_once(".experts.").unwrap();
                        let (bank, part) = bank.split_once('.').unwrap();
                        let operands: &[&str] = if bank == "gate_up" {
                            &["w1", "w3"]
                        } else {
                            &["w2"]
                        };
                        let experts = (0..)
                            .take_while(|e| {
                                header
                                    .get(format!("{prefix}.experts.{e}.w2.weight"))
                                    .is_some()
                            })
                            .count();
                        let mut expected = Vec::new();
                        let mut pieces = Vec::new();
                        for e in 0..experts {
                            for operand in operands {
                                let stem = format!("{prefix}.experts.{e}.{operand}");
                                let codes = tensor(&format!("{stem}.weight"));
                                let scales = tensor(&format!("{stem}.scale"));
                                let (rows, k) = (codes.shape[0], codes.shape[1] * 2);
                                let (rows, columns) = if bank == "gate_up" {
                                    (share(rows, r, n), 0..k)
                                } else {
                                    (0..rows, share(k, r, n))
                                };
                                if part == "weight" {
                                    expected.extend(block(
                                        &codes.bytes,
                                        k / 2,
                                        rows,
                                        columns.start / 2..columns.end / 2,
                                    ));
                                } else {
                                    let blocks = columns.start / 32..columns.end / 32;
                                    pieces.push((
                                        (rows.end - rows.start) as usize,
                                        (blocks.end - blocks.start) as usize,
                                    ));
                                    expected.extend(block(&scales.bytes, k / 32, rows, blocks));
                                }
                            }
                        }
                        if part == "scale" {
                            // Each expert's operand is one piece, tiled on its own.
                            let mut logical = Vec::new();
                            let mut offset = 0;
                            for (rows, k_blocks) in pieces {
                                let layout = ScaleTileLayout {
                                    rows,
                                    k_blocks,
                                    tile_rows: 16,
                                    k_blocks_per_tile: 8,
                                };
                                let bytes = rows.div_ceil(16) * k_blocks.div_ceil(8) * 128;
                                logical.extend(
                                    layout
                                        .unpack(&Packed {
                                            bytes: got[offset..offset + bytes].to_vec(),
                                            layout: Layout::ScaleTile(layout),
                                        })
                                        .unwrap_or_else(|e| panic!("{at}: {e:?}")),
                                );
                                offset += bytes;
                            }
                            assert_eq!(offset, got.len(), "{at}: tiled scale bytes");
                            got = logical;
                        }
                        expected
                    }
                    other => panic!("{at}: no expectation for layout {other}"),
                };
                same(&got, &expected, &at);
            }
            start += plan.arena_bytes;
        }
        eprintln!("tp{ranks}: every planned view matches the checkpoint");
    }
    eprintln!("{} view groups: {groups:?}", groups.len());
    std::fs::remove_dir_all(&root).unwrap();
}
