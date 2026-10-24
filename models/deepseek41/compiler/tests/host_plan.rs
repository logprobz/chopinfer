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
//! of the checkpoint that the projection declaration selects, a mapped view
//! as the host maps it.
//!
//! `a_fake_host_gathers_rows_of_every_planned_mapped_view` gathers rows of
//! each Engram plane the plan maps through the shared fake host
//! (`bundle_runtime::fake_host`), its mapped bytes those the host maps.
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use model_deepseek41::synthetic;
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::GEOMETRIES;
use model_deepseek41_contract::serving::{INDEX_SCHEMA, RANK_IMAGE_ROLE};
use model_deepseek41_contract::weight_plan::{RoutedLayout, plan, plan_with};
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
    prepare_with(root, ranks, None)
}

/// [`prepare`] with the backbone's routed gate/up bank in `routed`, or as
/// the served [`plan`] stores it.
fn prepare_with(
    root: &Path,
    ranks: u32,
    routed: Option<RoutedLayout>,
) -> Result<(PlannedImageV1, Vec<WeightPlanV1>), String> {
    let checkpoint = Arc::new(DirectCheckpointV1::open(root)?);
    let implementation = implementation_v1([0; 32]);
    let plans = (0..ranks)
        .map(|rank| {
            let request =
                checkpoint.request(RANK_IMAGE_ROLE, ranks, rank, implementation, vec![], vec![]);
            match routed {
                Some(routed) => plan_with(&request, routed),
                None => plan(&request),
            }
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

/// The backbone's routed gate/up bank in row tiles, or in quad-lane row
/// tiles, changes nothing but its own records' parameters (the packing
/// layout), its views' index layouts and so every rank's identity: the
/// arena, every view's place and size and every other record stay as the
/// row-major plan's. The served plan stores quad-lane tiles at TP4, where
/// every routed consumer reads them, and rows at TP1 and TP2. The host prepares
/// each plan from the headers at every supported rank count.
#[test]
fn row_tiles_change_only_the_backbone_gate_up_records_and_identity() {
    let geometry = &model_deepseek41_contract::geometry::FLASH;
    let root = sparse(geometry);
    for &ranks in SUPPORTED_RANKS {
        let layout = if ranks == 4 {
            RoutedLayout::Lanes
        } else {
            RoutedLayout::Rows
        };
        assert_eq!(RoutedLayout::served(geometry, ranks), layout);
        let (served, served_plans) = prepare(&root, ranks).unwrap();
        let (rows, rows_plans) = prepare_with(&root, ranks, Some(RoutedLayout::Rows)).unwrap();
        let (tiles, tiles_plans) = prepare_with(&root, ranks, Some(RoutedLayout::Tiles)).unwrap();
        let (lanes, lanes_plans) = prepare_with(&root, ranks, Some(RoutedLayout::Lanes)).unwrap();
        let (layout_prepared, layout_plans) = match layout {
            RoutedLayout::Rows => (&rows, &rows_plans),
            RoutedLayout::Tiles => (&tiles, &tiles_plans),
            RoutedLayout::Lanes => (&lanes, &lanes_plans),
        };
        assert_eq!(&served_plans, layout_plans);
        let rows = (&rows, rows_plans.as_slice());
        check_tiled_plans(
            geometry,
            ranks,
            rows,
            (&tiles, &tiles_plans),
            RoutedLayout::Tiles,
        );
        check_tiled_plans(
            geometry,
            ranks,
            rows,
            (&lanes, &lanes_plans),
            RoutedLayout::Lanes,
        );
        for rank in 0..rows_plans.len() {
            let at = format!("tp{ranks} rank {rank}");
            assert_eq!(
                served.identity(rank),
                layout_prepared.identity(rank),
                "{at}"
            );
            assert_ne!(tiles.identity(rank), lanes.identity(rank), "{at}");
        }
    }
    std::fs::remove_dir_all(&root).unwrap();
}

/// A plan of `ranks` with the bank in `tiled` (`tiles`) against the
/// row-major one (`rows`): only the backbone gate/up bank's records and
/// view layouts, and so each rank's identity, differ.
fn check_tiled_plans(
    geometry: &model_deepseek41_contract::geometry::Geometry,
    ranks: u32,
    (rows, rows_plans): (&PlannedImageV1, &[WeightPlanV1]),
    (tiles, tiles_plans): (&PlannedImageV1, &[WeightPlanV1]),
    tiled: RoutedLayout,
) {
    use kernel_core::precision::layout::{ROW_TILES_64X128, ROW_TILES_64X128_LANES};
    use kernel_core::weight_transform::WeightTransform;
    let [codes, scales] = tiled.view_layouts().unwrap();
    let id = match tiled {
        RoutedLayout::Rows => panic!("rows are not tiled"),
        RoutedLayout::Tiles => ROW_TILES_64X128,
        RoutedLayout::Lanes => ROW_TILES_64X128_LANES,
    };
    for (rank, (row, tile)) in rows_plans.iter().zip(tiles_plans).enumerate() {
        let at = format!("tp{ranks} {id} rank {rank}");
        assert_ne!(rows.identity(rank), tiles.identity(rank), "{at}");
        assert_eq!(
            (row.arena_bytes, &row.views, row.records.len()),
            (tile.arena_bytes, &tile.views, tile.records.len()),
            "{at}"
        );
        let index = |plan: &WeightPlanV1| {
            bundle_runtime::index::parse(&plan.components[0].bytes, rank as u32, INDEX_SCHEMA)
                .unwrap()
        };
        let (row_index, tile_index) = (index(row), index(tile));
        assert_ne!(row_index.direct_identity, tile_index.direct_identity);
        // The byte ranges of the tiled views.
        let mut banks = Vec::new();
        for (name, view) in &tile_index.views {
            let mut expected = row_index.views[name].clone();
            if let Some(bank) = name
                .strip_suffix(".ffn.experts.gate_up.weight")
                .or_else(|| name.strip_suffix(".ffn.experts.gate_up.scale"))
                && bank.starts_with("layers.")
            {
                expected.layout_id = if name.ends_with(".weight") {
                    codes
                } else {
                    scales
                }
                .into();
                banks.push(view.offset..view.offset + view.bytes);
            }
            assert_eq!(*view, expected, "{at} {name}");
        }
        assert_eq!(banks.len(), 2 * geometry.shape.layers as usize, "{at}");
        let mut tiled = 0;
        for (row, tile) in row.records.iter().zip(&tile.records) {
            let banked = banks.iter().any(|bank| bank.contains(&row.arena_offset));
            if !banked {
                assert_eq!(row, tile, "{at} record at {}", row.arena_offset);
                continue;
            }
            tiled += 1;
            let decode = |record: &runtime_bundle_abi::weight_plan::WeightRecordV1| {
                match WeightTransform::decode(record.transform, &record.params).unwrap() {
                    WeightTransform::PackedOperand(operand) => operand,
                    other => panic!("{at}: a bank record is {other:?}"),
                }
            };
            let (mut expected, operand) = (decode(row), decode(tile));
            assert_eq!(expected.plan.layout.id, "row-major-groups");
            expected.plan.layout.id = id.into();
            assert_eq!(operand, expected, "{at}");
            assert_eq!(
                (&row.sources, row.byte_length),
                (&tile.sources, tile.byte_length)
            );
        }
        // Two sections of each expert's two operands, every layer.
        let experts = u64::from(geometry.shape.moe.experts);
        let layers = u64::from(geometry.shape.layers);
        assert_eq!(tiled, layers * experts * 4, "{at}");
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

/// The safetensors header of the synthetic checkpoint at `root` and the
/// file offset of its data.
fn header(root: &Path) -> (Value, u64) {
    let file = std::fs::File::open(root.join(synthetic::FILE)).unwrap();
    let mut prefix = [0_u8; 8];
    file.read_exact_at(&mut prefix, 0).unwrap();
    let header_bytes = u64::from_le_bytes(prefix);
    let mut header = vec![0; header_bytes as usize];
    file.read_exact_at(&mut header, 8).unwrap();
    (serde_json::from_slice(&header).unwrap(), 8 + header_bytes)
}

/// A new directory, named for the test `test`, holding `geometry`'s
/// synthetic checkpoint with UE8M0 scales that vary per block (so a
/// misplaced block or slice shows), and its config.
fn varied_checkpoint(
    geometry: &model_deepseek41_contract::geometry::Geometry,
    test: &str,
) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "deepseek41-{test}-{}-{}",
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
/// codes and scales in rows, the backbone's gate/up bank also planned in
/// row tiles and in quad-lane row tiles. The checkpoint's scales vary per block, so a slice that
/// starts in the wrong block fails. Runs when a synthetic geometry is
/// compiled.
#[test]
fn every_planned_view_is_the_checkpoint_s() {
    use kernel_core::precision::layout::{
        ROW_TILE_CODES_VIEW, ROW_TILE_LANE_CODES_VIEW, ROW_TILE_SCALES_VIEW,
    };
    use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
    use kernel_core::transform::{Layout, Packed, QuadLaneLayout, RowTileLayout, Transform};
    use kernel_core::weight_transform::mxfp8_fold::{LAYOUT_ID as FOLD_LAYOUT, Mxfp8FoldLayout};
    let Some(geometry) = GEOMETRIES.iter().find(|g| g.id != "flash") else {
        eprintln!("no synthetic geometry is compiled: skipped");
        return;
    };
    let root = varied_checkpoint(geometry, "payload");
    let splits = declared_splits();
    let (header, data) = header(&root);
    let tensor = |name: &str| source(&root, &header, data, name);
    let mut groups = std::collections::BTreeSet::new();
    let routed = [None, Some(RoutedLayout::Tiles), Some(RoutedLayout::Lanes)];
    for (&ranks, routed) in SUPPORTED_RANKS.iter().flat_map(|r| routed.map(|l| (r, l))) {
        if check_ranks(geometry, ranks).is_err() {
            continue;
        }
        let n = u64::from(ranks);
        let (image, plans) = prepare_with(&root, ranks, routed).unwrap();
        let mut start = 0;
        for (rank, plan) in plans.iter().enumerate() {
            let r = rank as u64;
            let index =
                bundle_runtime::index::parse(&plan.components[0].bytes, rank as u32, INDEX_SCHEMA)
                    .unwrap();
            for (name, view) in &index.views {
                let at = format!("tp{ranks} rank {rank} {name}");
                // A mapped view is its whole checkpoint tensor, which the
                // host maps instead of producing.
                if let Some(mapped) = image.mapped_view(rank as u32, name.as_bytes()) {
                    groups.insert(format!("{name} (mapped)"));
                    same(mapped.bytes(), &tensor(name).bytes, &at);
                    continue;
                }
                let mut got = vec![0; view.bytes as usize];
                image.read(start + view.offset, &mut got).unwrap();
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
                    // Dense codes are stored in the packed fold layout,
                    // their rows' row-major codes permuted.
                    "mxfp8_e4m3_rows_v1" | "mxfp8_ue8m0_scale_rows_v1" | FOLD_LAYOUT => {
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
                        let fold = Mxfp8FoldLayout {
                            groups: 1,
                            n: (rows.end - rows.start) as usize,
                            k: (columns.end - columns.start) as usize,
                        };
                        let (codes, scales) = mxfp8_rows(&codes, &scales, rows, columns);
                        if view.layout_id == FOLD_LAYOUT {
                            fold.pack(&codes).unwrap_or_else(|e| panic!("{at}: {e}"))
                        } else if name.ends_with(".weight") {
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
                    "mxfp4_e2m1_rows_v1"
                    | "mxfp4_ue8m0_scale_rows_v1"
                    | ROW_TILE_CODES_VIEW
                    | ROW_TILE_LANE_CODES_VIEW
                    | ROW_TILE_SCALES_VIEW => {
                        let tiled = view.layout_id.starts_with("mxfp4_e2m1_row_tiles")
                            || view.layout_id.starts_with("mxfp4_ue8m0_scale_row_tiles");
                        let layout = routed.unwrap_or(RoutedLayout::served(geometry, ranks));
                        let backbone_gate_up =
                            name.starts_with("layers.") && name.contains(".experts.gate_up.");
                        assert_eq!(tiled, layout.tiled() && backbone_gate_up, "{at}");
                        let lanes = view.layout_id == ROW_TILE_LANE_CODES_VIEW;
                        assert_eq!(
                            lanes,
                            layout == RoutedLayout::Lanes
                                && backbone_gate_up
                                && name.ends_with(".weight"),
                            "{at}"
                        );
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
                                let height = (rows.end - rows.start) as usize;
                                let (matrix, row_bytes, tile_bytes) = if part == "weight" {
                                    let row_bytes = (columns.end - columns.start) / 2;
                                    let codes = block(
                                        &codes.bytes,
                                        k / 2,
                                        rows,
                                        columns.start / 2..columns.end / 2,
                                    );
                                    (codes, row_bytes, 64)
                                } else {
                                    let blocks = columns.start / 32..columns.end / 32;
                                    let row_bytes = blocks.end - blocks.start;
                                    (block(&scales.bytes, k / 32, rows, blocks), row_bytes, 4)
                                };
                                if tiled {
                                    let tiles = RowTileLayout {
                                        rows: height,
                                        row_bytes: row_bytes as usize,
                                        tile_rows: 64,
                                        tile_bytes,
                                    };
                                    let mut packed = tiles.pack(&matrix).unwrap().bytes;
                                    if lanes {
                                        QuadLaneLayout {
                                            bytes: packed.len(),
                                        }
                                        .pack_in_place(&mut packed)
                                        .unwrap();
                                    }
                                    expected.extend(packed);
                                } else {
                                    expected.extend(matrix);
                                }
                            }
                        }
                        expected
                    }
                    other => panic!("{at}: no expectation for layout {other}"),
                };
                same(&got, &expected, &at);
            }
            start += plan.arena_bytes;
        }
        eprintln!("tp{ranks} {routed:?}: every planned view matches the checkpoint");
    }
    eprintln!("{} view groups: {groups:?}", groups.len());
    std::fs::remove_dir_all(&root).unwrap();
}

/// A fake host package whose mapped views are the planned image's: the
/// bytes the host maps for each, which the shared backend gathers rows of.
/// It records every upload's bytes and serves nothing else.
struct Mapped {
    device: bundle_runtime::fake_host::FakeDevice,
    image: PlannedImageV1,
    uploads: std::cell::RefCell<Vec<Vec<u8>>>,
}

mod mapped_package {
    use super::Mapped;
    use bundle_runtime::fake_host::{FakeDevice, FakePackage, NoObserver};
    use runtime_bundle_abi::*;

    impl FakePackage for Mapped {
        type Observer = NoObserver;

        fn device(&self) -> &FakeDevice {
            &self.device
        }
        fn cubins() -> impl IntoIterator<Item = (&'static str, &'static [u8])> {
            []
        }
        fn open_compiled_image(
            &self,
            _role: &[u8],
        ) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn upload_rank_arena(
            &self,
            _context: BundleContextHandleV1,
            _pool: BundleMemoryPoolHandleV1,
            _stream: BundleStreamHandleV1,
            _image: BundleCompiledImageHandleV1,
            _rank: u32,
        ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn resolve_compiled_image_view(
            &self,
            _image: BundleCompiledImageHandleV1,
            _arena: BundleRankArenaHandleV1,
            _rank: u32,
            _view_id: &[u8],
        ) -> HostServiceResultV1<CompiledImageSpanV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn open_authenticated_component(
            &self,
            _role: &[u8],
        ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
            Err(BundleStatusV1::NotFound)
        }
        fn component(&self, slot: u32) -> (String, Vec<u8>) {
            unreachable!("no component {slot} was opened")
        }
        fn acquire_serving_kv(&self, _layout_id: &[u8]) -> HostServiceResultV1<u64> {
            Err(BundleStatusV1::Unsupported)
        }
        fn resolve_serving_kv_pool(&self, _rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn query_speculation(&self) -> HostServiceResultV1<BundleRawSpeculationV1> {
            Ok(BundleRawSpeculationV1::disabled())
        }
        fn mapped_view(
            &self,
            _image: BundleCompiledImageHandleV1,
            rank: u32,
            view_id: &[u8],
        ) -> HostServiceResultV1<Vec<u8>> {
            self.image
                .mapped_view(rank, view_id)
                .map(|rows| rows.bytes().to_vec())
                .ok_or(BundleStatusV1::NotFound)
        }
        fn allocate_state(
            &self,
            _request: &StateRequestV1,
        ) -> HostServiceResultV1<BundleStateHandleV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn resolve_state_view(
            &self,
            _state: BundleStateHandleV1,
            _rank: u32,
        ) -> HostServiceResultV1<DeviceBufferSpanV1> {
            Err(BundleStatusV1::Unsupported)
        }
        fn read_back(
            &self,
            _destination: &mut [u8],
            _source: DeviceBufferSpanV1,
        ) -> HostServiceResultV1<()> {
            Err(BundleStatusV1::Unsupported)
        }
        fn on_upload(
            &self,
            _stream: BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
            source: &[u8],
        ) -> HostServiceResultV1<()> {
            self.uploads.borrow_mut().push(source.to_vec());
            Ok(())
        }
    }
}

/// The shared fake host gathers rows of every mapped view the plan of each
/// rank count declares, bit-exactly the checkpoint's rows of the plane:
/// the last row, the first and one between, in that order. A view the plan
/// does not map is not found. Runs when a synthetic geometry is compiled.
#[test]
fn a_fake_host_gathers_rows_of_every_planned_mapped_view() {
    use bundle_runtime::fake_host::Backend;
    use runtime_bundle_abi::{
        BundleCompiledImageHandleV1, BundleStatusV1, DeviceBufferSpanV1, HostServicesBackendV1,
        MappedRowGatherV1,
    };
    let Some(geometry) = GEOMETRIES.iter().find(|g| g.id != "flash") else {
        eprintln!("no synthetic geometry is compiled: skipped");
        return;
    };
    let root = varied_checkpoint(geometry, "gather");
    let (header, data) = header(&root);
    let file = std::fs::File::open(root.join(synthetic::FILE)).unwrap();
    let mut gathered = 0;
    for &ranks in SUPPORTED_RANKS {
        if check_ranks(geometry, ranks).is_err() {
            continue;
        }
        let (image, plans) = prepare(&root, ranks).unwrap();
        let host = Backend::new(Mapped {
            device: Default::default(),
            image,
            uploads: Default::default(),
        });
        let context = host.select_context(0, 0).unwrap();
        let pool = host.select_memory_pool(context).unwrap();
        let stream = host.create_stream(context, 0).unwrap();
        let image = BundleCompiledImageHandleV1::from_parts(1, 1);
        for plan in &plans {
            assert!(
                !plan.views.is_empty(),
                "tp{ranks}: the plan maps the Engram planes"
            );
            for view in &plan.views {
                let at = format!("tp{ranks} rank {} {}", plan.rank, view.id);
                let entry = &header[&view.id];
                let start = entry["data_offsets"][0].as_u64().unwrap();
                let rows = entry["shape"][0].as_u64().unwrap();
                let row_bytes = view.byte_length / rows;
                let picked = [rows - 1, 0, rows / 2];
                let length = picked.len() as u64 * row_bytes;
                let buffer = host
                    .allocate_device_buffer(pool, stream, length, 256)
                    .unwrap();
                let span = DeviceBufferSpanV1::new(buffer, 0, length);
                let picked_rows = picked.map(|row| row as u32);
                host.gather_mapped_rows(
                    stream,
                    span,
                    MappedRowGatherV1 {
                        image,
                        rank: plan.rank,
                        view_id: view.id.as_bytes(),
                        row_bytes,
                        rows: &picked_rows,
                    },
                )
                .unwrap_or_else(|e| panic!("{at}: {e:?}"));
                let mut expected = Vec::new();
                for row in picked {
                    let mut bytes = vec![0; row_bytes as usize];
                    file.read_exact_at(&mut bytes, data + start + row * row_bytes)
                        .unwrap();
                    expected.extend(bytes);
                }
                let uploaded = host.uploads.borrow_mut().pop().unwrap();
                same(&uploaded, &expected, &at);
                gathered += 1;
            }
            let buffer = host.allocate_device_buffer(pool, stream, 256, 256).unwrap();
            assert_eq!(
                host.gather_mapped_rows(
                    stream,
                    DeviceBufferSpanV1::new(buffer, 0, 256),
                    MappedRowGatherV1 {
                        image,
                        rank: plan.rank,
                        view_id: b"embed.weight",
                        row_bytes: 256,
                        rows: &[0],
                    },
                ),
                Err(BundleStatusV1::NotFound)
            );
        }
    }
    eprintln!("{gathered} mapped views gathered bit-exactly");
    std::fs::remove_dir_all(&root).unwrap();
}

#[path = "host_plan/encoded.rs"]
mod encoded;
#[path = "host_plan/routed.rs"]
mod routed;
