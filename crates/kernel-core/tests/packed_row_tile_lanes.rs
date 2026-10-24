//! Quad-lane row tiles (`row-tiles-64x128-lanes`): every materializer stores
//! the row tiles' sections with each 64-byte row block of the codes in quad
//! lane order, the bytes a quad lane's m16n8k16 A fragments convert side by
//! side. The scales are the row tiles', and the inverse restores the
//! row-major bytes.
#![allow(clippy::single_range_in_vec_init)]
use kernel_core::contract::{DType, SwigluColumns, WeightFormat};
use kernel_core::precision::layout::{PackingPlan, ROW_TILES_64X128, ROW_TILES_64X128_LANES};
use kernel_core::precision::{DecodeOrder, NibbleOrder, WeightLayout};
use kernel_core::transform::{Layout, Packed, QuadLaneLayout, RowTileLayout, Transform};
use kernel_core::weight_transform::{
    PackedMatrixSource, PackedOperand, PackedSection, ScaleEncoding, SourceTensor, WeightRecord,
    WeightTransform,
};

const SOURCE_ROWS: u64 = 160;
const SOURCE_K: u64 = 640;

fn matrix(codes: u32) -> PackedMatrixSource {
    PackedMatrixSource {
        expert: 0,
        projection: 0,
        format: WeightFormat::Mxfp4,
        rows: SOURCE_ROWS,
        k: SOURCE_K,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes,
        local: Some(codes + 1),
        local_dtype: Some(DType::Ue8M0),
        global: None,
        source_expert: 0,
        source_experts: 1,
        rows_kept: vec![0..64],
        k_kept: 0..128,
    }
}

fn operand(layout: &str, experts: u32, projections: u32, rows: u32, k: u32) -> PackedOperand {
    PackedOperand {
        plan: PackingPlan {
            layout: WeightLayout {
                id: layout.into(),
                version: 1,
            },
            format: WeightFormat::Mxfp4,
            experts,
            projections,
            rows,
            k,
            columns: SwigluColumns::ConcatenatedHalves,
            local_scale_dtype: Some(DType::Ue8M0),
        },
        section: PackedSection::Codes,
        decode_order: DecodeOrder::ExactScaledOperand,
        matrices: vec![matrix(0)],
        local_encoding: None,
    }
}

/// One record reading `matrices` pairs of codes and scales.
fn record(operand: &PackedOperand, matrices: usize) -> WeightRecord {
    let pair = [
        SourceTensor {
            dtype: "U8".into(),
            shape: vec![SOURCE_ROWS, SOURCE_K / 2],
            byte_length: SOURCE_ROWS * SOURCE_K / 2,
        },
        SourceTensor {
            dtype: "F8_E8M0".into(),
            shape: vec![SOURCE_ROWS, SOURCE_K / 32],
            byte_length: SOURCE_ROWS * SOURCE_K / 32,
        },
    ];
    WeightRecord {
        transform: WeightTransform::PackedOperand(Box::new(operand.clone())),
        sources: (0..matrices).flat_map(|_| pair.clone()).collect(),
        axis: None,
        ranges: vec![],
        stored_shape: vec![],
    }
}

struct Seed(u64);

impl Seed {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// One codes tensor and its finite scales.
    fn sources(&mut self) -> [Vec<u8>; 2] {
        [
            (0..SOURCE_ROWS * SOURCE_K / 2)
                .map(|_| self.next() as u8)
                .collect(),
            (0..SOURCE_ROWS * SOURCE_K / 32)
                .map(|_| (self.next() % 252) as u8)
                .collect(),
        ]
    }
}

/// Single-matrix records with sliced, offset and padded rows and K, as a
/// rank's expert piece reads its share of a checkpoint matrix.
fn seeded() -> Vec<(PackedOperand, WeightRecord, [Vec<u8>; 2])> {
    let mut seed = Seed(0x7175_6164_5f6c_616e);
    (0..48)
        .map(|_| {
            let rows = 64 * (1 + seed.next() % 2) as u32;
            let k = 128 * (1 + seed.next() % 3) as u32;
            let mut operand = operand(ROW_TILES_64X128_LANES, 1, 1, rows, k);
            let matrix = &mut operand.matrices[0];
            let kept = 1 + seed.next() % u64::from(rows);
            let start = seed.next() % (SOURCE_ROWS - kept + 1);
            let split = start + seed.next() % kept;
            matrix.rows_kept = if split == start {
                vec![start..start + kept]
            } else {
                vec![split..start + kept, start..split]
            };
            matrix.row_origin = seed.next() % 17;
            matrix.k_origin = (seed.next() % 4) * 32;
            let width = 1 + seed.next() % u64::from(k);
            let start = (seed.next() % ((SOURCE_K - width) / 32 + 1)) * 32;
            matrix.k_kept = start..start + width;
            matrix.nibble_order = if seed.next() % 2 == 0 {
                NibbleOrder::EvenLow
            } else {
                NibbleOrder::EvenHigh
            };
            let record = record(&operand, 1);
            (operand, record, seed.sources())
        })
        .collect()
}

fn with_layout(mut operand: PackedOperand, layout: &str) -> PackedOperand {
    operand.plan.layout.id = layout.into();
    operand
}

/// The row tiles' sections with their codes in quad lane order.
fn lanes(mut sections: [Vec<u8>; 3]) -> [Vec<u8>; 3] {
    let codes = &mut sections[0];
    QuadLaneLayout { bytes: codes.len() }
        .pack_in_place(codes)
        .unwrap();
    sections
}

/// The row-major sections of `lanes`-ordered tiles of `rows` rows of `k`
/// MXFP4 elements: the inverse a prepared bank's reader would apply.
fn inverse(sections: &[Vec<u8>; 3], rows: usize, k: usize) -> [Vec<u8>; 3] {
    let tiles = |row_bytes, tile_bytes| RowTileLayout {
        rows,
        row_bytes,
        tile_rows: 64,
        tile_bytes,
    };
    let codes = QuadLaneLayout {
        bytes: sections[0].len(),
    };
    let unlaned = codes
        .unpack(&Packed {
            bytes: sections[0].clone(),
            layout: Layout::QuadLanes(codes),
        })
        .unwrap();
    let untile = |bytes: Vec<u8>, layout: RowTileLayout| {
        layout
            .unpack(&Packed {
                bytes,
                layout: Layout::RowTiles(layout),
            })
            .unwrap()
    };
    [
        untile(unlaned, tiles(k / 2, 64)),
        untile(sections[1].clone(), tiles(k / 32, 4)),
        sections[2].clone(),
    ]
}

#[test]
fn every_materializer_stores_the_row_tiles_in_quad_lane_order() {
    for (mut operand, mut record, [codes, scales]) in seeded() {
        let sources = [codes.as_slice(), scales.as_slice()];
        let tiled = with_layout(operand.clone(), ROW_TILES_64X128)
            .execute_sections(&record, &sources)
            .unwrap();
        let expected = lanes(tiled.clone());
        assert_eq!(
            operand.execute_sections(&record, &sources).unwrap(),
            expected
        );
        assert_eq!(expected[1], tiled[1], "the scales are the row tiles'");
        for section in [PackedSection::Codes, PackedSection::LocalScales] {
            operand.section = section;
            record.transform = WeightTransform::PackedOperand(Box::new(operand.clone()));
            let index = section.index();
            for wanted in [[false; 3], [true; 3]] {
                let actual = operand
                    .execute_needed_sections_reusing(
                        &record,
                        &sources,
                        std::array::from_fn(|_| vec![0xa5; 70_000]),
                        wanted,
                    )
                    .unwrap();
                for part in 0..3 {
                    if part == index || wanted[part] {
                        assert_eq!(actual[part], expected[part], "{section:?} part {part}");
                    } else {
                        assert!(actual[part].is_empty());
                    }
                }
            }
        }
        assert_eq!(operand.independent_scale_source(&record), Some(1));
        assert_eq!(
            operand
                .independent_scales(&record, &scales, vec![0xa5; 70_000])
                .unwrap()
                .unwrap(),
            expected[1]
        );
    }
}

/// Undoing the quad lane order and then the row tiles reproduces the
/// row-major sections byte for byte.
#[test]
fn the_inverse_reproduces_the_row_major_bytes() {
    for (operand, record, [codes, scales]) in seeded() {
        let sources = [codes.as_slice(), scales.as_slice()];
        let (rows, k) = (operand.plan.rows as usize, operand.plan.k as usize);
        let row_major = with_layout(operand.clone(), "row-major-groups")
            .execute_sections(&record, &sources)
            .unwrap();
        let stored = operand.execute_sections(&record, &sources).unwrap();
        assert_ne!(stored[0], row_major[0]);
        assert_eq!(inverse(&stored, rows, k), row_major);
    }
}

/// The address a quad-lane kernel computes. Lane `t` of the quad on row `r`
/// of tile `strip * blocks + block` loads 16 code bytes at
/// `tile * 4096 + r * 64 + 16 * t`, whose word `g` byte `q` is the row's
/// packed byte `16 * g + 4 * q + t` of the block, and one scale word at
/// `tile * 256 + r * 4`, whose byte `g` is group `g`'s scale.
#[test]
fn lanes_are_the_kernels_fragment_addresses() {
    for (operand, record, [codes, scales]) in seeded() {
        let sources = [codes.as_slice(), scales.as_slice()];
        let (rows, k) = (operand.plan.rows as usize, operand.plan.k as usize);
        let row_major = with_layout(operand.clone(), "row-major-groups")
            .execute_sections(&record, &sources)
            .unwrap();
        let stored = operand.execute_sections(&record, &sources).unwrap();
        let blocks = k / 128;
        for row in 0..rows {
            let (strip, r) = (row / 64, row % 64);
            for block in 0..blocks {
                let tile = strip * blocks + block;
                for lane in 0..4 {
                    for group in 0..4 {
                        for q in 0..4 {
                            assert_eq!(
                                stored[0][tile * 4096 + r * 64 + 16 * lane + 4 * group + q],
                                row_major[0][row * k / 2 + block * 64 + 16 * group + 4 * q + lane]
                            );
                        }
                        assert_eq!(
                            stored[1][tile * 256 + r * 4 + group],
                            row_major[1][row * k / 32 + block * 4 + group]
                        );
                    }
                }
            }
        }
    }
}

/// The whole-bank materializer (fused projections, several experts) stores
/// the tiled bank in quad lane order.
#[test]
fn a_fused_bank_is_stored_in_quad_lane_order() {
    let mut seed = Seed(0x6261_6e6b_5f6c_616e);
    let mut operand = operand(ROW_TILES_64X128_LANES, 2, 2, 128, 256);
    operand.matrices = (0..4)
        .map(|index| {
            let mut matrix = matrix(2 * index);
            matrix.expert = index / 2;
            matrix.projection = index % 2;
            matrix.rows_kept = vec![u64::from(index) * 8..u64::from(index) * 8 + 128];
            matrix.k_kept = 64..320;
            matrix
        })
        .collect();
    let record = record(&operand, 4);
    let sources = (0..4).flat_map(|_| seed.sources()).collect::<Vec<_>>();
    let views = sources.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let tiled = with_layout(operand.clone(), ROW_TILES_64X128)
        .execute_sections(&record, &views)
        .unwrap();
    let stored = operand.execute_sections(&record, &views).unwrap();
    assert_eq!(stored, lanes(tiled));
    let row_major = with_layout(operand, "row-major-groups")
        .execute_sections(&record, &views)
        .unwrap();
    assert_eq!(inverse(&stored, 512, 256), row_major);
}

/// Encoded local scales (`scale-row-offset-v1`) are encoded from the
/// row-major scales and left untiled, as for the row tiles; only the codes
/// take the quad lane order.
#[test]
fn encoded_scales_are_the_row_tiles_encoding() {
    use kernel_core::precision::scale_offset::{ScaleCode, WIDTHS, census};
    use kernel_core::weight_transform::ScaleFormat;
    for (operand, mut record, [codes, scales]) in seeded().into_iter().take(8) {
        let sources = [codes.as_slice(), scales.as_slice()];
        let row_major = with_layout(operand.clone(), "row-major-groups")
            .execute_sections(&record, &sources)
            .unwrap();
        let (rows, groups) = operand.plan.scale_shape();
        let counts = census(&row_major[1], rows as u64, groups as u64).unwrap();
        for format in [ScaleFormat::RowOffsetV1, ScaleFormat::RowTilesOffsetV1] {
            for (slot, width) in WIDTHS.into_iter().enumerate() {
                let mut encoded = operand.clone();
                encoded.local_encoding = Some(ScaleEncoding {
                    format,
                    code: ScaleCode::Ue8m0,
                    width,
                    exceptions: counts[slot],
                });
                record.transform = WeightTransform::PackedOperand(Box::new(encoded.clone()));
                let tiled = with_layout(encoded.clone(), ROW_TILES_64X128)
                    .execute_sections(&record, &sources)
                    .unwrap();
                let stored = encoded.execute_sections(&record, &sources).unwrap();
                assert_eq!(stored[1], tiled[1]);
                assert_eq!(stored[0], lanes(tiled)[0]);
            }
        }
    }
}

#[test]
fn quad_lanes_keep_section_sizes_and_refuse_what_row_tiles_refuse() {
    let lanes = operand(ROW_TILES_64X128_LANES, 3, 2, 576, 5120);
    lanes.plan.validate().unwrap();
    let tiled = with_layout(lanes.clone(), ROW_TILES_64X128);
    assert_eq!(
        lanes.plan.section_bytes().unwrap(),
        tiled.plan.section_bytes().unwrap()
    );
    assert_eq!(lanes.plan.row_tiles(), tiled.plan.row_tiles());
    assert_eq!(
        lanes.plan.quad_lanes(),
        Some(QuadLaneLayout {
            bytes: 3 * 2 * 576 * 5120 / 2
        })
    );
    assert_eq!(tiled.plan.quad_lanes(), None);
    for (rows, k, format) in [
        (96, 128, WeightFormat::Mxfp4),
        (64, 192, WeightFormat::Mxfp4),
        (64, 128, WeightFormat::Mxfp8),
    ] {
        let mut operand = operand(ROW_TILES_64X128_LANES, 1, 1, rows, k);
        operand.plan.format = format;
        assert!(operand.plan.validate().is_err(), "{rows} {k} {format:?}");
        let record = record(&operand, 1);
        assert!(operand.output_bytes(&record).is_err());
    }
}

/// The layout is part of the record's parameters, so a plan's identity
/// differs from the row tiles'.
#[cfg(feature = "precision-serde")]
#[test]
fn the_lane_layout_travels_in_the_record_parameters() {
    let lanes =
        WeightTransform::PackedOperand(Box::new(operand(ROW_TILES_64X128_LANES, 1, 1, 64, 128)));
    let tiles = WeightTransform::PackedOperand(Box::new(operand(ROW_TILES_64X128, 1, 1, 64, 128)));
    let params = lanes.encode_params().unwrap();
    assert_ne!(params, tiles.encode_params().unwrap());
    assert_eq!(lanes.id(), tiles.id());
    assert_eq!(WeightTransform::decode(lanes.id(), &params).unwrap(), lanes);
}
