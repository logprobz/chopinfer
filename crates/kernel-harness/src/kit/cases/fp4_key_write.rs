//! Exact cache writes against the shared trained FP4 quantizer. Alternating
//! index values give an exactly representable F32 mean square. Norm and
//! RoPE keep F32 values until the quantizer's single BF16 input rounding.

use crate::LabResult;
use crate::conformance::{Expected, Holes, Images, Rows, Write};
use crate::kit::{OpCase, Required, Spec, bf16_bytes};
use crate::poison::WriteMap;
use kernel_core::attention::BlockTables;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::compressed_kv::{
    CompressedKvPages, CompressedKvStorage, CompressedKvWriteBuffers, CompressedKvWriteQuery,
};
use kernel_core::contract::{DType, Refusal, RmsNormKind};
use kernel_core::pooled_indexer::{
    IndexerCandidates, IndexerKeyFormat, PositionIndexerQuery, PositionKeyPages,
    PositionKeyWriteBuffers,
};
use kernel_core::precision::quantizer::{
    BlockQuantization, BlockQuantizeQuery, NONFINITE_INPUT, QuantizeOutput,
};
use kernel_core::quantize::Scales;

pub const CAPACITY: u32 = 6;
pub const PAGE: u32 = 64;
const PAGES: usize = 10;
const TABLE_ROWS: u32 = 132;
const ITEMS: [i32; 6] = [0, 0, 1, 1, 2, 2];
const TABLES: [i32; 9] = [1, 3, -1, 0, 4, -1, 2, 5, -1];

#[derive(Clone, Copy, Debug)]
pub enum Query {
    Index(PositionIndexerQuery),
    IndexBf16(PositionIndexerQuery),
    Compressed(CompressedKvWriteQuery),
}

impl Query {
    fn page(self) -> usize {
        match self {
            Self::Index(q) | Self::IndexBf16(q) => q.page_size as usize,
            Self::Compressed(q) => q.page_size as usize,
        }
    }
    pub fn width(self) -> u32 {
        match self {
            Self::Index(q) | Self::IndexBf16(q) => q.head_dim,
            Self::Compressed(q) => q.width,
        }
    }
    fn ratio(self) -> u32 {
        match self {
            Self::Index(q) | Self::IndexBf16(q) => q.compression_ratio,
            Self::Compressed(q) => q.compression_ratio,
        }
    }
    fn packed(self) -> bool {
        !matches!(self, Self::IndexBf16(_))
            && !matches!(self, Self::Compressed(q) if q.storage == CompressedKvStorage::Bf16)
    }
    fn format(self) -> BlockQuantization {
        match self {
            Self::Index(_) | Self::IndexBf16(_) => BlockQuantization::E2M1Ceil32,
            Self::Compressed(_) => BlockQuantization::E2M1E4M3Scale16,
        }
    }
    fn row_bytes(self) -> usize {
        self.width() as usize * if self.packed() { 1 } else { 4 } / 2
    }
    fn scale_bytes(self) -> usize {
        if self.packed() {
            (self.width() / self.format().group_size()) as usize
        } else {
            0
        }
    }
    fn strides(self) -> (usize, usize) {
        (
            self.page() * self.row_bytes() + 64,
            self.page() * self.scale_bytes() + 32,
        )
    }
}

pub fn queries() -> Vec<Query> {
    [1, 2]
        .into_iter()
        .flat_map(|ratio| {
            [
                Query::Index(PositionIndexerQuery {
                    heads: 32,
                    head_dim: 128,
                    compression_ratio: ratio,
                    tail_rope_dims: 64,
                    key_norm_eps_bits: 1e-20f32.to_bits(),
                    max_top_k: 512,
                    page_size: PAGE,
                    candidates: IndexerCandidates::All,
                    capacity: CAPACITY,
                }),
                Query::Compressed(CompressedKvWriteQuery {
                    width: 512,
                    compression_ratio: ratio,
                    tail_rope_dims: 64,
                    page_size: PAGE,
                    storage: CompressedKvStorage::Packed,
                    capacity: CAPACITY,
                }),
                Query::Compressed(CompressedKvWriteQuery {
                    width: 512,
                    compression_ratio: ratio,
                    tail_rope_dims: 64,
                    page_size: PAGE,
                    storage: CompressedKvStorage::Bf16,
                    capacity: CAPACITY,
                }),
            ]
        })
        .collect()
}

kernel_core::scratch_record! {
    #[derive(Clone)]
    pub struct ProbeBuffers {
        pub key: value,
        pub status: value,
        pub status_row_offset: scalar(u64),
        pub norm_weight: option,
        pub cos_sin: value,
        pub positions: value,
        pub row_batch: value,
        pub live_rows: value,
        pub codes: value,
        pub scales: option,
        pub tables: record(BlockTables),
        pub code_stride: scalar(u64),
        pub scale_stride: scalar(u64),
    }
}

enum Probe {
    Index(Prepared<PositionKeyWriteBuffers<View>>),
    Compressed(Prepared<CompressedKvWriteBuffers<View>>),
}

impl PreparedOp<ProbeBuffers<View>> for Probe {
    fn info(&self) -> &OpInfo {
        match self {
            Self::Index(op) => op.info(),
            Self::Compressed(op) => op.info(),
        }
    }
    fn launches(
        &self,
        b: &ProbeBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        match self {
            Self::Index(op) => op.launches(
                &PositionKeyWriteBuffers {
                    status: b.status,
                    status_row_offset: b.status_row_offset,
                    key: b.key,
                    norm_weight: b
                        .norm_weight
                        .ok_or(Refusal("index case needs norm weight"))?,
                    cos_sin: b.cos_sin,
                    positions: b.positions,
                    row_batch: b.row_batch,
                    live_rows: b.live_rows,
                    keys: PositionKeyPages {
                        codes: b.codes,
                        scales: b.scales.unwrap_or(View::EMPTY),
                        pages: PAGES as u32,
                        code_page_stride: b.code_stride,
                        scale_page_stride: b.scale_stride,
                        page_tables: b.tables,
                    },
                },
                workspace,
                counts,
            ),
            Self::Compressed(op) => op.launches(
                &CompressedKvWriteBuffers {
                    status: b.status,
                    status_row_offset: b.status_row_offset,
                    latent: b.key,
                    cos_sin: b.cos_sin,
                    positions: b.positions,
                    row_batch: b.row_batch,
                    live_rows: b.live_rows,
                    cache: CompressedKvPages {
                        values: b.codes,
                        scales: b.scales,
                        pages: PAGES as u32,
                        value_page_stride: b.code_stride,
                        scale_page_stride: b.scale_stride,
                        page_tables: b.tables,
                    },
                },
                workspace,
                counts,
            ),
        }
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn slots(
    query: Query,
    images: &Images,
    live: u32,
    scales: bool,
) -> LabResult<Vec<(usize, std::ops::Range<usize>)>> {
    let positions = images.i32s("positions")?;
    let items = images.i32s("row_batch")?;
    let tables = images.i32s("tables.tables")?;
    let (code_stride, scale_stride) = query.strides();
    let (stride, bytes) = if scales {
        (scale_stride, query.scale_bytes())
    } else {
        (code_stride, query.row_bytes())
    };
    let mut out = Vec::new();
    for row in 0..live as usize {
        if positions[row] < 0 || items[row] < 0 || items[row] >= 3 {
            continue;
        }
        let position = positions[row] as usize;
        if position * query.ratio() as usize >= TABLE_ROWS as usize || position / query.page() >= 3
        {
            continue;
        }
        let page = tables[items[row] as usize * 3 + position / query.page()];
        if page < 0 || page as usize >= PAGES {
            continue;
        }
        let start = page as usize * stride + (position % query.page()) * bytes;
        out.push((row, start..start + bytes));
    }
    Ok(out)
}

fn old_pages(bytes: usize, salt: u8) -> Vec<u8> {
    (0..bytes)
        .map(|i| ((i / 31 + i / 257) as u8).wrapping_add(salt))
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatusCase {
    Normal,
    Saturation,
    InputNonfinite,
    RoundedOverflow,
    RotaryNonfinite,
    RotaryOverflow,
    NormNonfinite,
    DecodeOverflow,
    Holes,
    PriorFlags,
}

fn case(query: Query, prepared: Prepared<ProbeBuffers<View>>) -> LabResult<OpCase<'static>> {
    status_case(query, prepared, StatusCase::Normal)
}

fn status_case(
    query: Query,
    prepared: Prepared<ProbeBuffers<View>>,
    variant: StatusCase,
) -> LabResult<OpCase<'static>> {
    let (page, rotary, capacity) = match query {
        Query::Index(q) | Query::IndexBf16(q) => (q.page_size, q.tail_rope_dims, q.capacity),
        Query::Compressed(q) => (q.page_size, q.tail_rope_dims, q.capacity),
    };
    if !matches!(page, 32 | 64)
        || rotary != 64
        || capacity < CAPACITY
        || !matches!(query.width(), 128 | 512)
    {
        return Err(
            "FP4 write fixtures need 32/64-position pages, tail 64, width 128/512 and six rows"
                .into(),
        );
    }
    let width = query.width() as usize;
    let packed = query.packed();
    let normalize = matches!(query, Query::Index(_) | Query::IndexBf16(_));
    let (code_stride, scale_stride) = query.strides();
    let sample = |row: usize, d: usize| -> f32 {
        if normalize {
            return if d % 2 == 0 {
                8.0 + 1.0 / 64.0
            } else {
                -8.0 + 1.0 / 64.0
            };
        }
        let scale = match (d / 16 + row) % 6 {
            0 => 0.0,
            1 => 2f32.powi(-9),
            2 => f32::MIN_POSITIVE,
            3 => 1.0,
            4 => 0.25,
            _ => 448.0,
        };
        let values: [f32; 16] = [
            6.0, -0.0, 0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75,
            -3.5, -5.0,
        ];
        let value = if d >= width - 64 && values[d % 16].abs() == 5.0 {
            values[d % 16].signum() * 4.5
        } else {
            values[d % 16]
        };
        let perturb = if matches!(scale, 1.0 | 0.25) && value != 0.0 {
            scale * ((d % 3) as f32 - 1.0) / 100.0
        } else {
            0.0
        };
        if perturb == 0.0 {
            value * scale
        } else {
            value * scale + perturb
        }
    };
    let mut key: Vec<f32> = (0..CAPACITY as usize)
        .flat_map(|row| (0..width).map(move |d| sample(row, d)))
        .collect();
    let mut weights: Vec<u16> = (0..width)
        .map(|d| f32_to_bf16([0.0, 0.25, 0.75, 1.25, 2.5, 3.5, 5.0, 6.03125][d % 8]))
        .collect();
    let mut table: Vec<f32> = (0..TABLE_ROWS)
        .flat_map(|position| {
            (0..32).flat_map(move |pair| {
                let angle = f64::from(position * 13 + pair * 7) / 17.0;
                [angle.cos() as f32, angle.sin() as f32]
            })
        })
        .collect();
    let mut initial_status = vec![0_u32; CAPACITY as usize];
    let mut positions = vec![page as i32 - 1, page as i32, 5, -1, 2, page as i32 + 1];
    let mut items = ITEMS.to_vec();
    let mut tables = TABLES.to_vec();
    match variant {
        StatusCase::Normal => {}
        StatusCase::Saturation => {
            for (row, value) in [2704., 2688., -2704.].into_iter().enumerate() {
                key[row * width] = value;
            }
        }
        StatusCase::InputNonfinite => {
            for (row, value) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
                .into_iter()
                .enumerate()
            {
                key[row * width] = value;
            }
        }
        StatusCase::RoundedOverflow => {
            key[0] = f32::MAX;
            key[width] = -f32::MAX;
        }
        StatusCase::RotaryNonfinite => {
            for (row, value) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
                .into_iter()
                .enumerate()
            {
                table[positions[row] as usize * query.ratio() as usize * 64] = value;
            }
        }
        StatusCase::RotaryOverflow => {
            key[width - 64] = 2.;
            table[positions[0] as usize * query.ratio() as usize * 64] = f32::MAX;
        }
        StatusCase::NormNonfinite => {
            weights[0] = f32_to_bf16(f32::NAN);
            weights[1] = f32_to_bf16(f32::INFINITY);
            weights[2] = f32_to_bf16(f32::NEG_INFINITY);
        }
        StatusCase::DecodeOverflow => {
            for (i, value) in key.iter_mut().enumerate() {
                *value = if i % 2 == 0 { 1. } else { -1. };
            }
            weights.fill(0x7f7f);
        }
        StatusCase::Holes => {
            // Keep the first row valid so every positive live prefix has
            // an observable write; later rows exercise each hole guard.
            positions[5] = -1;
            positions[1] = i32::MAX;
            items[2] = -1;
            positions[3] = 5;
            items[3] = 3;
            tables[6] = PAGES as i32;
        }
        StatusCase::PriorFlags => {
            initial_status = [0x80, 2, 1, 0x40, 0, 4].to_vec();
        }
    }
    let mut pools = vec![("codes", code_stride)];
    if packed {
        pools.push(("scales", scale_stride));
    }
    let roles = ProbeBuffers {
        status: super::block_quantize::status_role(&initial_status),
        status_row_offset: 0,
        key: Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&key)),
        norm_weight: normalize
            .then(|| Spec::input(DType::Bf16, (Rows::Fixed(1), width), bf16_bytes(&weights))),
        cos_sin: Spec::input(
            DType::F32,
            (Rows::Fixed(TABLE_ROWS as usize), 64),
            f32_bytes(&table),
        ),
        positions: if variant == StatusCase::Holes {
            // These invalid positions are tested directly, outside the
            // valid-ID mutation generator's domain.
            Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&positions))
        } else {
            Spec::ids((Rows::Live, 1), positions, &[], Holes::LastLive)
        },
        row_batch: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&items)),
        live_rows: Spec::live(),
        codes: Spec::state(
            DType::U8,
            (Rows::Fixed(PAGES), code_stride),
            old_pages(PAGES * code_stride, 17),
            Write::Map(Box::new(move |images, live| {
                WriteMap::of(DType::U8).byte_ranges(
                    slots(query, images, live, false)?
                        .into_iter()
                        .map(|(_, range)| range),
                )
            })),
        ),
        scales: packed.then(|| {
            Spec::state(
                DType::U8,
                (Rows::Fixed(PAGES), scale_stride),
                old_pages(PAGES * scale_stride, 83),
                Write::Map(Box::new(move |images, live| {
                    WriteMap::of(DType::U8).byte_ranges(
                        slots(query, images, live, true)?
                            .into_iter()
                            .map(|(_, range)| range),
                    )
                })),
            )
        }),
        tables: BlockTables {
            tables: if variant == StatusCase::Holes {
                // Invalid page IDs are intentional data in this regression,
                // outside the valid-ID permutation generator's domain.
                Spec::input(DType::I32, (Rows::Fixed(3), 3), i32_bytes(&tables))
            } else {
                Spec::ids((Rows::Fixed(3), 3), tables, &pools, Holes::Row(2))
            },
            sequences: 3,
            max_pages: 3,
        },
        code_stride: code_stride as u64,
        scale_stride: if packed { scale_stride as u64 } else { 0 },
    };
    Ok(OpCase::with_counts(prepared, CAPACITY, roles, |rows| {
        Counts::rows(rows).with_positions(TABLE_ROWS)
    })?
    .oracle(move |images, live| oracle(query, images, live))
    .live_counts(vec![0, 1, 3, 6]))
}

fn oracle(query: Query, images: &Images, live: u32) -> LabResult<Vec<(String, Expected)>> {
    oracle_rounding(query, images, live, false)
}

fn oracle_rounding(
    query: Query,
    images: &Images,
    live: u32,
    early_bf16: bool,
) -> LabResult<Vec<(String, Expected)>> {
    let width = query.width() as usize;
    let key = images.f32s("key")?;
    let positions = images.i32s("positions")?;
    let table = images.f32s("cos_sin")?;
    let mut codes = images.bytes("codes")?.to_vec();
    let mut scales = if query.packed() {
        images.bytes("scales")?.to_vec()
    } else {
        vec![]
    };
    let mut status = images
        .bytes("status")?
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    let scale_slots = slots(query, images, live, true)?;
    for (row, destination) in slots(query, images, live, false)? {
        let mut values = key[row * width..(row + 1) * width].to_vec();
        let mut flags = if values.iter().any(|v| !v.is_finite()) {
            NONFINITE_INPUT
        } else {
            0
        };
        if let Query::Index(q) | Query::IndexBf16(q) = query {
            let weights: Vec<f64> = images
                .u16s("norm_weight")?
                .into_iter()
                .map(|b| f64::from(bf16_to_f32(b)))
                .collect();
            if weights.iter().any(|v| !v.is_finite()) {
                flags |= NONFINITE_INPUT;
            }
            if flags == 0 {
                if values.iter().enumerate().any(|(i, v)| *v != values[i % 2]) {
                    return Err("exact index fixture needs alternating F32 values".into());
                }
                let normalized = kernel_core::oracle::rmsnorm(
                    &values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                    width,
                    &weights,
                    f64::from(f32::from_bits(q.key_norm_eps_bits)),
                    RmsNormKind::Plain,
                    None,
                )
                .map_err(|e| e.0)?;
                let mean = (values[0] * values[0] + values[1] * values[1]) * 0.5;
                let inv =
                    (1.0 / f64::from(mean + f32::from_bits(q.key_norm_eps_bits)).sqrt()) as f32;
                for (v, w) in values.iter_mut().zip(weights) {
                    *v = (*v * inv) * w as f32;
                }
                kernel_core::oracle::check_numeric(
                    &values.iter().copied().map(f64::from).collect::<Vec<_>>(),
                    &normalized,
                    DType::F32,
                )
                .map_err(|e| format!("index norm fixture violates its f64 bound: {e:?}"))?;
            }
        }
        if flags != 0 {
            values.fill(0.);
        }
        if early_bf16 {
            for value in &mut values {
                *value = bf16_to_f32(f32_to_bf16(*value));
            }
        }
        let position = positions[row] as usize * query.ratio() as usize;
        for d in (width - 64..width).step_by(2) {
            let cs = &table[position * 64 + d - (width - 64)..];
            if !cs[0].is_finite() || !cs[1].is_finite() {
                flags |= NONFINITE_INPUT;
            }
            let (x, y) = (values[d], values[d + 1]);
            values[d] = x * cs[0] - y * cs[1];
            values[d + 1] = x * cs[1] + y * cs[0];
        }
        let q = BlockQuantizeQuery {
            width: width as u32,
            input: DType::F32,
            format: query.format(),
            output: if query.packed() {
                QuantizeOutput::Codes
            } else {
                QuantizeOutput::Bf16
            },
        };
        status[row] = q
            .row_status(&values, &[status[row] | flags])
            .map_err(|e| e.0)?[0];
        let (bytes, scale_bytes) = if status[row] != 0 {
            (
                vec![0; query.row_bytes()],
                vec![q.invalid_scale_code(); query.scale_bytes()],
            )
        } else if query.packed() {
            let quantized = query.format().apply(&values, 1, width).map_err(|e| e.0)?;
            let scales = match quantized.scales {
                Scales::Ue8M0(v) | Scales::Ue4M3(v) => v,
                _ => return Err("FP4 scales are bytes".into()),
            };
            (quantized.codes, scales)
        } else {
            (
                bf16_bytes(
                    &query
                        .format()
                        .fake_quant(&values, 1, width)
                        .map_err(|e| e.0)?,
                ),
                Vec::new(),
            )
        };
        codes[destination].copy_from_slice(&bytes);
        if query.packed() {
            let target = scale_slots
                .iter()
                .find(|(r, _)| *r == row)
                .ok_or("missing scale slot")?
                .1
                .clone();
            scales[target].copy_from_slice(&scale_bytes);
        }
    }
    let mut results = vec![
        ("codes".into(), Expected::Bits(codes)),
        (
            "status".into(),
            Expected::Bits(status.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ),
    ];
    if query.packed() {
        results.push(("scales".into(), Expected::Bits(scales)));
    }
    Ok(results)
}

pub fn status_family(variant: StatusCase) -> Vec<Required<'static>> {
    status_family_for(variant, queries())
}

pub fn status_family_for(
    variant: StatusCase,
    queries: impl IntoIterator<Item = Query>,
) -> Vec<Required<'static>> {
    queries
        .into_iter()
        .filter(|query| match variant {
            StatusCase::NormNonfinite | StatusCase::DecodeOverflow => {
                matches!(query, Query::Index(_) | Query::IndexBf16(_))
            }
            StatusCase::Saturation | StatusCase::RoundedOverflow | StatusCase::RotaryOverflow => {
                matches!(query, Query::Compressed(_))
            }
            _ => true,
        })
        .map(|query| {
            Required::new(
                format!("FP4 status {variant:?} {query:?}"),
                Need::rows(CAPACITY),
                move |backend, need| match query {
                    Query::IndexBf16(q) => backend
                        .position_indexer_key_write_with_storage(&q, IndexerKeyFormat::Bf16, need)
                        .map(|op| Box::new(Probe::Index(op)) as Prepared<ProbeBuffers<View>>),
                    Query::Index(q) => backend
                        .position_indexer_key_write(&q, need)
                        .map(|op| Box::new(Probe::Index(op)) as Prepared<ProbeBuffers<View>>),
                    Query::Compressed(q) => backend
                        .compressed_kv_write(&q, need)
                        .map(|op| Box::new(Probe::Compressed(op)) as Prepared<ProbeBuffers<View>>),
                },
                move |op| status_case(query, op, variant),
            )
            .required()
        })
        .collect()
}

pub const STATUS_CASES: [StatusCase; 9] = [
    StatusCase::Saturation,
    StatusCase::InputNonfinite,
    StatusCase::RoundedOverflow,
    StatusCase::RotaryNonfinite,
    StatusCase::RotaryOverflow,
    StatusCase::NormNonfinite,
    StatusCase::DecodeOverflow,
    StatusCase::Holes,
    StatusCase::PriorFlags,
];

/// A 64-token page holds 64/ratio compressed entries in each stream.
pub fn comparison_queries() -> Vec<Query> {
    queries()
        .into_iter()
        .filter_map(|query| match query {
            Query::Index(mut q) => {
                q.page_size = 64 / q.compression_ratio;
                Some(Query::Index(q))
            }
            Query::Compressed(mut q) if q.storage == CompressedKvStorage::Packed => {
                q.page_size = 64 / q.compression_ratio;
                Some(Query::Compressed(q))
            }
            _ => None,
        })
        .collect()
}

pub fn bf16_storage(query: Query) -> Query {
    match query {
        Query::Index(q) | Query::IndexBf16(q) => Query::IndexBf16(q),
        Query::Compressed(q) => Query::Compressed(CompressedKvWriteQuery {
            storage: CompressedKvStorage::Bf16,
            ..q
        }),
    }
}

pub fn bf16_family() -> Vec<Required<'static>> {
    let queries = comparison_queries()
        .into_iter()
        .map(bf16_storage)
        .collect::<Vec<_>>();
    let mut cases = family_for(queries.clone());
    for variant in STATUS_CASES {
        cases.extend(status_family_for(variant, queries.clone()));
    }
    cases
}

pub fn family() -> Vec<Required<'static>> {
    family_for(queries())
}

/// The same required cases over a consumer's geometry-derived queries.
/// These fixtures use 32/64-position pages, widths 128/512 and a 64-channel
/// rotary tail. The query still owns its ratio and normalization epsilon.
pub fn family_for(queries: impl IntoIterator<Item = Query>) -> Vec<Required<'static>> {
    queries
        .into_iter()
        .map(|query| {
            Required::new(
                format!("FP4 key write {query:?}"),
                Need::rows(CAPACITY),
                move |backend, need| match query {
                    Query::IndexBf16(q) => backend
                        .position_indexer_key_write_with_storage(&q, IndexerKeyFormat::Bf16, need)
                        .map(|op| Box::new(Probe::Index(op)) as Prepared<ProbeBuffers<View>>),
                    Query::Index(q) => backend
                        .position_indexer_key_write(&q, need)
                        .map(|op| Box::new(Probe::Index(op)) as Prepared<ProbeBuffers<View>>),
                    Query::Compressed(q) => backend
                        .compressed_kv_write(&q, need)
                        .map(|op| Box::new(Probe::Compressed(op)) as Prepared<ProbeBuffers<View>>),
                },
                move |prepared| case(query, prepared),
            )
            .required()
        })
        .collect()
}

/// Decode the observed native write and compare every addressed BF16 value.
/// Each launch also checks its own exact oracle, guards, status and poison.
pub fn compare_storage_on_device(
    backend: &dyn kernel_core::backend::Backend,
    modules: &dyn crate::kit::Modules,
    session: &crate::Session,
) -> LabResult<serde_json::Value> {
    use kernel_core::codec::{SmallFloat, decode_ue8m0_f64, f64_to_bf16};
    let functions = modules.load(session)?;
    let mut pairs = 0;
    let mut values = 0;
    for query in comparison_queries() {
        let comparison = bf16_storage(query);
        for variant in std::iter::once(StatusCase::Normal).chain(STATUS_CASES) {
            let Some(native_required) = status_family_for(variant, [query]).pop() else {
                continue;
            };
            let bf16_required = status_family_for(variant, [comparison]).pop().unwrap();
            let native = super::testing::prepared(backend, &native_required);
            let bf16 = super::testing::prepared(backend, &bf16_required);
            let a = native.images();
            let b = bf16.images();
            for live in [1, 3, CAPACITY] {
                let left = native.launch(session, functions.as_ref(), &a, live)?;
                let right = bf16.launch(session, functions.as_ref(), &b, live)?;
                native.judge_outputs(&a, live, &left)?;
                bf16.judge_outputs(&b, live, &right)?;
                if left["status"] != right["status"] {
                    return Err("comparison row status differs".into());
                }
                let scales = slots(query, &a, live, true)?;
                let targets = slots(comparison, &b, live, false)?;
                for (row, source) in slots(query, &a, live, false)? {
                    let scale = &scales.iter().find(|(r, _)| *r == row).unwrap().1;
                    let target = &targets.iter().find(|(r, _)| *r == row).unwrap().1;
                    for channel in 0..query.width() as usize {
                        let byte = left["codes"][source.start + channel / 2];
                        let code = (byte >> (4 * (channel % 2))) & 15;
                        let scale = left["scales"]
                            [scale.start + channel / query.format().group_size() as usize];
                        let scale = match query {
                            Query::Index(_) | Query::IndexBf16(_) => decode_ue8m0_f64(scale),
                            Query::Compressed(_) => SmallFloat::Ue4M3.decode_f64(scale),
                        };
                        let product = SmallFloat::E2M1.decode_f64(code) * scale;
                        let expected = f64_to_bf16(product);
                        if !product.is_finite()
                            || f64::from(bf16_to_f32(expected)).to_bits() != product.to_bits()
                        {
                            return Err("native writer emitted an inexact BF16 product".into());
                        }
                        let at = target.start + channel * 2;
                        if right["codes"][at..at + 2] != expected.to_le_bytes() {
                            return Err(format!(
                                "native/BF16 product differs at row {row} channel {channel}"
                            ));
                        }
                        values += 1;
                    }
                }
                pairs += 1;
            }
        }
    }
    Ok(serde_json::json!({"launch_pairs": pairs, "exact_products": values}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    use kernels_sm120::backend::Sm120;
    use kernels_sm120::gated_pooling::key_write::Entry;

    fn prepared_case(query: Query) -> OpCase<'static> {
        let entry = match query {
            Query::Index(q) => Entry::index(&q, "chop_fp4_test_v1").unwrap(),
            Query::IndexBf16(q) => {
                Entry::index_with_storage(&q, IndexerKeyFormat::Bf16, "chop_fp4_test_v1").unwrap()
            }
            Query::Compressed(q) => Entry::compressed(&q, "chop_fp4_test_v1").unwrap(),
        };
        let backend = Sm120::new().with_fp4_key_writes([entry]);
        let prepared = match query {
            Query::IndexBf16(q) => Probe::Index(
                backend
                    .position_indexer_key_write_with_storage(
                        &q,
                        IndexerKeyFormat::Bf16,
                        Need::rows(CAPACITY),
                    )
                    .unwrap(),
            ),
            Query::Index(q) => Probe::Index(
                backend
                    .position_indexer_key_write(&q, Need::rows(CAPACITY))
                    .unwrap(),
            ),
            Query::Compressed(q) => Probe::Compressed(
                backend
                    .compressed_kv_write(&q, Need::rows(CAPACITY))
                    .unwrap(),
            ),
        };
        case(query, Box::new(prepared)).unwrap()
    }

    #[test]
    fn status_fixtures_pin_saturation_trap_sites_and_holes() {
        for variant in STATUS_CASES {
            for query in queries().into_iter().filter(|q| match variant {
                StatusCase::NormNonfinite | StatusCase::DecodeOverflow => {
                    matches!(q, Query::Index(_) | Query::IndexBf16(_))
                }
                StatusCase::Saturation
                | StatusCase::RoundedOverflow
                | StatusCase::RotaryOverflow => matches!(q, Query::Compressed(_)),
                _ => true,
            }) {
                let entry = match query {
                    Query::Index(q) => Entry::index(&q, "status_fixture_v1"),
                    Query::IndexBf16(q) => {
                        Entry::index_with_storage(&q, IndexerKeyFormat::Bf16, "status_fixture_v1")
                    }
                    Query::Compressed(q) => Entry::compressed(&q, "status_fixture_v1"),
                }
                .unwrap();
                let backend = Sm120::new().with_fp4_key_writes([entry]);
                let prepared = match query {
                    Query::IndexBf16(q) => Probe::Index(
                        backend
                            .position_indexer_key_write_with_storage(
                                &q,
                                IndexerKeyFormat::Bf16,
                                Need::rows(CAPACITY),
                            )
                            .unwrap(),
                    ),
                    Query::Index(q) => Probe::Index(
                        backend
                            .position_indexer_key_write(&q, Need::rows(CAPACITY))
                            .unwrap(),
                    ),
                    Query::Compressed(q) => Probe::Compressed(
                        backend
                            .compressed_kv_write(&q, Need::rows(CAPACITY))
                            .unwrap(),
                    ),
                };
                let case = status_case(query, Box::new(prepared), variant).unwrap();
                let images = case.images();
                let outputs = oracle(query, &images, CAPACITY).unwrap();
                let status = outputs.iter().find(|(name, _)| name == "status").unwrap();
                let Expected::Bits(bytes) = &status.1 else {
                    panic!("exact status")
                };
                let words = bytes
                    .chunks_exact(4)
                    .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>();
                match variant {
                    StatusCase::Saturation => {
                        assert!(words.iter().all(|&v| v == 0));
                        let destination =
                            slots(query, &images, CAPACITY, false).unwrap()[0].1.clone();
                        let Expected::Bits(codes) = &outputs[0].1 else {
                            panic!("exact codes")
                        };
                        if query.packed() {
                            assert_eq!(codes[destination.start] & 15, 7);
                            let scale_slot =
                                slots(query, &images, CAPACITY, true).unwrap()[0].1.start;
                            let Expected::Bits(scales) =
                                &outputs.iter().find(|(n, _)| n == "scales").unwrap().1
                            else {
                                panic!("exact scales")
                            };
                            assert_eq!(scales[scale_slot], 0x7e);
                        } else {
                            assert_eq!(
                                &codes[destination.start..destination.start + 2],
                                &f32_to_bf16(2688.).to_le_bytes()
                            );
                        }
                    }
                    StatusCase::DecodeOverflow => assert_ne!(
                        words[0] & kernel_core::precision::quantizer::UNREPRESENTABLE_OUTPUT,
                        0
                    ),
                    StatusCase::Holes => {
                        assert!(words.iter().all(|&v| v == 0));
                        let valid_rows = slots(query, &images, CAPACITY, false)
                            .unwrap()
                            .into_iter()
                            .map(|(row, _)| row)
                            .collect::<Vec<_>>();
                        // The device control must pass live=1, then first
                        // differ at row 1's status word (byte 4) at live=3.
                        assert_eq!(valid_rows, [0]);
                        assert_eq!(images.i32s("positions").unwrap()[1], i32::MAX);
                    }
                    StatusCase::PriorFlags => assert_eq!(bytes, images.bytes("status").unwrap()),
                    _ => assert_ne!(words[0] & NONFINITE_INPUT, 0, "{variant:?} {query:?}"),
                }
            }
        }
    }

    #[test]
    fn exact_cache_bits_reject_early_projection_and_pre_rope_rounding() {
        let bits = |expected: Vec<(String, Expected)>| {
            expected
                .into_iter()
                .filter_map(|(_, value)| match value {
                    Expected::Bits(b) => Some(b),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let mut projection_changes = 0;
        let mut norm_changes = 0;
        let mut latent_changes = 0;
        for query in queries() {
            let case = prepared_case(query);
            let images = case.images();
            let correct = bits(oracle(query, &images, CAPACITY).unwrap());
            let rounded: Vec<_> = images
                .f32s("key")
                .unwrap()
                .into_iter()
                .map(|v| bf16_to_f32(f32_to_bf16(v)))
                .collect();
            let early_projection = images.clone().with("key", f32_bytes(&rounded));
            if matches!(query, Query::Index(_) | Query::IndexBf16(_))
                && correct != bits(oracle(query, &early_projection, CAPACITY).unwrap())
            {
                projection_changes += 1;
            }
            if correct != bits(oracle_rounding(query, &images, CAPACITY, true).unwrap()) {
                match query {
                    Query::Index(_) | Query::IndexBf16(_) => norm_changes += 1,
                    Query::Compressed(_) => latent_changes += 1,
                }
            }
        }
        assert!(
            projection_changes > 0,
            "fixture did not detect rounding the projected key"
        );
        assert!(
            norm_changes > 0,
            "fixture did not detect rounding the normalized key"
        );
        assert!(
            latent_changes > 0,
            "fixture did not detect rounding the latent before RoPE"
        );
    }

    #[test]
    fn exact_fp4_cases_validate_both_ratios_formats_and_page_witnesses() {
        let mut checked = 0;
        for query in queries() {
            let entry = match query {
                Query::Index(q) => Entry::index(&q, "chop_fp4_test_v1").unwrap(),
                Query::IndexBf16(q) => {
                    Entry::index_with_storage(&q, IndexerKeyFormat::Bf16, "chop_fp4_test_v1")
                        .unwrap()
                }
                Query::Compressed(q) => Entry::compressed(&q, "chop_fp4_test_v1").unwrap(),
            };
            let backend = Sm120::new().with_fp4_key_writes([entry]);
            let prepared = match query {
                Query::IndexBf16(q) => Probe::Index(
                    backend
                        .position_indexer_key_write_with_storage(
                            &q,
                            IndexerKeyFormat::Bf16,
                            Need::rows(CAPACITY),
                        )
                        .unwrap(),
                ),
                Query::Index(q) => Probe::Index(
                    backend
                        .position_indexer_key_write(&q, Need::rows(CAPACITY))
                        .unwrap(),
                ),
                Query::Compressed(q) => Probe::Compressed(
                    backend
                        .compressed_kv_write(&q, Need::rows(CAPACITY))
                        .unwrap(),
                ),
            };
            case(query, Box::new(prepared)).unwrap().validate().unwrap();
            checked += 1;
        }
        assert_eq!(checked, 6);
    }
}
