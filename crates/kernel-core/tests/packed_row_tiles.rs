//! Row-tiled packed operands (`row-tiles-64x128-groups`): every
//! materializer stores the row-major sections, byte for byte, in tiles of
//! 64 rows by 128 K, the order the tiled routed kernels address.
#![allow(clippy::single_range_in_vec_init)]
use kernel_core::contract::{DType, SwigluColumns, WeightFormat};
use kernel_core::precision::layout::{PackingPlan, ROW_TILES_64X128};
use kernel_core::precision::{DecodeOrder, NibbleOrder, WeightLayout};
use kernel_core::transform::{RowTileLayout, Transform};
use kernel_core::weight_transform::{
    PackedMatrixSource, PackedOperand, PackedSection, SourceTensor, WeightRecord, WeightTransform,
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

/// The code and scale tiles of `rows` rows of `k` MXFP4 elements.
fn tiles(rows: usize, k: usize) -> [RowTileLayout; 2] {
    [(k / 2, 64), (k / 32, 4)].map(|(row_bytes, tile_bytes)| RowTileLayout {
        rows,
        row_bytes,
        tile_rows: 64,
        tile_bytes,
    })
}

/// Row-major sections stored in their tiles, by the reversible transform.
fn tiled(sections: &[Vec<u8>; 3], rows: usize, k: usize) -> [Vec<u8>; 3] {
    let [codes, scales] = tiles(rows, k);
    [
        codes.pack(&sections[0]).unwrap().bytes,
        scales.pack(&sections[1]).unwrap().bytes,
        sections[2].clone(),
    ]
}

/// Single-matrix records with sliced, offset and padded rows and K, as a
/// rank's expert piece reads its share of a checkpoint matrix.
fn seeded() -> Vec<(PackedOperand, WeightRecord, [Vec<u8>; 2])> {
    let mut seed = Seed(0x726f_775f_7469_6c65);
    (0..48)
        .map(|_| {
            let rows = 64 * (1 + seed.next() % 2) as u32;
            let k = 128 * (1 + seed.next() % 3) as u32;
            let mut operand = operand(ROW_TILES_64X128, 1, 1, rows, k);
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

fn rows_of(mut operand: PackedOperand) -> PackedOperand {
    operand.plan.layout.id = "row-major-groups".into();
    operand
}

#[test]
fn every_materializer_stores_the_row_major_sections_in_tiles() {
    for (mut operand, mut record, [codes, scales]) in seeded() {
        let sources = [codes.as_slice(), scales.as_slice()];
        let (rows, k) = (operand.plan.rows as usize, operand.plan.k as usize);
        let row_major = rows_of(operand.clone())
            .execute_sections(&record, &sources)
            .unwrap();
        let expected = tiled(&row_major, rows, k);
        assert_eq!(
            operand.execute_sections(&record, &sources).unwrap(),
            expected
        );
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
        assert_eq!(
            operand.independent_scale_source(&record),
            Some(1),
            "a tiled scale section keeps the independent path"
        );
        assert_eq!(
            operand
                .independent_scales(&record, &scales, vec![0xa5; 70_000])
                .unwrap()
                .unwrap(),
            expected[1]
        );
    }
}

/// The address a tiled kernel lane computes: tile `strip * blocks + block`
/// of 4096 code bytes (64 rows of 64) and 256 scale bytes (64 rows of 4).
#[test]
fn tiles_are_the_kernels_strip_and_k_block_addresses() {
    for (operand, record, [codes, scales]) in seeded() {
        let sources = [codes.as_slice(), scales.as_slice()];
        let (rows, k) = (operand.plan.rows as usize, operand.plan.k as usize);
        let row_major = rows_of(operand.clone())
            .execute_sections(&record, &sources)
            .unwrap();
        let tiled = operand.execute_sections(&record, &sources).unwrap();
        let blocks = k / 128;
        for row in 0..rows {
            let (strip, r) = (row / 64, row % 64);
            for block in 0..blocks {
                let tile = strip * blocks + block;
                for byte in 0..64 {
                    assert_eq!(
                        tiled[0][tile * 4096 + r * 64 + byte],
                        row_major[0][row * k / 2 + block * 64 + byte]
                    );
                }
                for scale in 0..4 {
                    assert_eq!(
                        tiled[1][tile * 256 + r * 4 + scale],
                        row_major[1][row * k / 32 + block * 4 + scale]
                    );
                }
            }
        }
    }
}

/// The whole-bank materializer (fused projections, several experts) tiles
/// the bank's rows; 64-row strips never straddle two matrices.
#[test]
fn a_fused_bank_is_tiled_across_its_matrices() {
    let mut seed = Seed(0x6261_6e6b_5f74_696c);
    let mut operand = operand(ROW_TILES_64X128, 2, 2, 128, 256);
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
    let row_major = rows_of(operand.clone())
        .execute_sections(&record, &views)
        .unwrap();
    assert_eq!(
        operand.execute_sections(&record, &views).unwrap(),
        tiled(&row_major, 512, 256)
    );
}

#[test]
fn row_tiles_keep_section_sizes_and_refuse_partial_tiles_or_other_formats() {
    let tiled = operand(ROW_TILES_64X128, 3, 2, 576, 5120);
    tiled.plan.validate().unwrap();
    assert_eq!(
        tiled.plan.section_bytes().unwrap(),
        rows_of(tiled.clone()).plan.section_bytes().unwrap()
    );
    assert_eq!(tiled.plan.row_tiles(), Some(tiles(3 * 2 * 576, 5120)));
    assert_eq!(rows_of(tiled).plan.row_tiles(), None);
    for (rows, k, format) in [
        (96, 128, WeightFormat::Mxfp4),
        (64, 192, WeightFormat::Mxfp4),
        (64, 128, WeightFormat::Mxfp8),
    ] {
        let mut operand = operand(ROW_TILES_64X128, 1, 1, rows, k);
        operand.plan.format = format;
        assert!(operand.plan.validate().is_err(), "{rows} {k} {format:?}");
        let record = record(&operand, 1);
        assert!(operand.output_bytes(&record).is_err());
    }
}

/// The layout is part of the record's parameters, so a plan's identity
/// (which digests every record's parameters) differs with it.
#[cfg(feature = "precision-serde")]
#[test]
fn the_layout_travels_in_the_record_parameters() {
    let tiled = WeightTransform::PackedOperand(Box::new(operand(ROW_TILES_64X128, 1, 1, 64, 128)));
    let rows = WeightTransform::PackedOperand(Box::new(operand("row-major-groups", 1, 1, 64, 128)));
    let params = tiled.encode_params().unwrap();
    assert_ne!(params, rows.encode_params().unwrap());
    assert_eq!(tiled.id(), rows.id());
    assert_eq!(WeightTransform::decode(tiled.id(), &params).unwrap(), tiled);
}
