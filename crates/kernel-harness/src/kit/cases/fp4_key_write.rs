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
    IndexerCandidates, PositionIndexerQuery, PositionKeyPages, PositionKeyWriteBuffers,
};
use kernel_core::precision::quantizer::BlockQuantization;
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
    Compressed(CompressedKvWriteQuery),
}

impl Query {
    fn page(self) -> usize {
        match self {
            Self::Index(q) => q.page_size as usize,
            Self::Compressed(q) => q.page_size as usize,
        }
    }
    pub fn width(self) -> u32 {
        match self {
            Self::Index(q) => q.head_dim,
            Self::Compressed(q) => q.width,
        }
    }
    fn ratio(self) -> u32 {
        match self {
            Self::Index(q) => q.compression_ratio,
            Self::Compressed(q) => q.compression_ratio,
        }
    }
    fn packed(self) -> bool {
        !matches!(self, Self::Compressed(q) if q.storage == CompressedKvStorage::Bf16)
    }
    fn format(self) -> BlockQuantization {
        match self {
            Self::Index(_) => BlockQuantization::E2M1Ceil32,
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
                        scales: b.scales.ok_or(Refusal("index case needs scales"))?,
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
        if positions[row] < 0 {
            continue;
        }
        let position = positions[row] as usize;
        let page = tables[items[row] as usize * 3 + position / query.page()];
        if page < 0 {
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

fn case(query: Query, prepared: Prepared<ProbeBuffers<View>>) -> LabResult<OpCase<'static>> {
    let (page, rotary, capacity) = match query {
        Query::Index(q) => (q.page_size, q.tail_rope_dims, q.capacity),
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
    let normalize = matches!(query, Query::Index(_));
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
    let key: Vec<f32> = (0..CAPACITY as usize)
        .flat_map(|row| (0..width).map(move |d| sample(row, d)))
        .collect();
    let weights: Vec<u16> = (0..width)
        .map(|d| f32_to_bf16([0.0, 0.25, 0.75, 1.25, 2.5, 3.5, 5.0, 6.03125][d % 8]))
        .collect();
    let table: Vec<f32> = (0..TABLE_ROWS)
        .flat_map(|position| {
            (0..32).flat_map(move |pair| {
                let angle = f64::from(position * 13 + pair * 7) / 17.0;
                [angle.cos() as f32, angle.sin() as f32]
            })
        })
        .collect();
    let mut pools = vec![("codes", code_stride)];
    if packed {
        pools.push(("scales", scale_stride));
    }
    let roles = ProbeBuffers {
        key: Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&key)),
        norm_weight: normalize
            .then(|| Spec::input(DType::Bf16, (Rows::Fixed(1), width), bf16_bytes(&weights))),
        cos_sin: Spec::input(
            DType::F32,
            (Rows::Fixed(TABLE_ROWS as usize), 64),
            f32_bytes(&table),
        ),
        positions: Spec::ids(
            (Rows::Live, 1),
            vec![page as i32 - 1, page as i32, 5, -1, 2, page as i32 + 1],
            &[],
            Holes::LastLive,
        ),
        row_batch: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&ITEMS)),
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
            tables: Spec::ids((Rows::Fixed(3), 3), TABLES.to_vec(), &pools, Holes::Row(2)),
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
    let scale_slots = slots(query, images, live, true)?;
    for (row, destination) in slots(query, images, live, false)? {
        let mut values = key[row * width..(row + 1) * width].to_vec();
        if let Query::Index(q) = query {
            if values.iter().enumerate().any(|(i, v)| *v != values[i % 2]) {
                return Err("exact index fixture needs alternating F32 values".into());
            }
            let weights: Vec<f64> = images
                .u16s("norm_weight")?
                .into_iter()
                .map(|b| f64::from(bf16_to_f32(b)))
                .collect();
            let normalized = kernel_core::oracle::rmsnorm(
                &values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                width,
                &weights,
                f64::from(f32::from_bits(q.key_norm_eps_bits)),
                RmsNormKind::Plain,
                None,
            )
            .map_err(|e| e.0)?;
            // Every pair has the same exactly representable mean square,
            // so the tree reduction and the analytic pair mean agree.
            let mean = (values[0] * values[0] + values[1] * values[1]) * 0.5;
            let inv = (1.0 / f64::from(mean + f32::from_bits(q.key_norm_eps_bits)).sqrt()) as f32;
            for (value, weight) in values.iter_mut().zip(&weights) {
                *value = (*value * inv) * *weight as f32;
            }
            kernel_core::oracle::check_numeric(
                &values.iter().copied().map(f64::from).collect::<Vec<_>>(),
                &normalized,
                DType::F32,
            )
            .map_err(|e| format!("index norm fixture violates its f64 bound: {e:?}"))?;
        }
        if early_bf16 {
            for value in &mut values {
                *value = bf16_to_f32(f32_to_bf16(*value));
            }
        }
        let position = positions[row] as usize * query.ratio() as usize;
        for d in (width - 64..width).step_by(2) {
            let cs = &table[position * 64 + d - (width - 64)..];
            let (x, y) = (values[d], values[d + 1]);
            values[d] = x * cs[0] - y * cs[1];
            values[d + 1] = x * cs[1] + y * cs[0];
        }
        let quantized = query.format().apply(&values, 1, width).map_err(|e| e.0)?;
        let decoded = query
            .format()
            .fake_quant(&values, 1, width)
            .map_err(|e| e.0)?;
        if query.packed() {
            codes[destination].copy_from_slice(&quantized.codes);
            let bytes = match quantized.scales {
                Scales::Ue8M0(v) | Scales::Ue4M3(v) => v,
                _ => return Err("FP4 scales are bytes".into()),
            };
            let target = scale_slots
                .iter()
                .find(|(r, _)| *r == row)
                .ok_or("missing scale slot")?
                .1
                .clone();
            scales[target].copy_from_slice(&bytes);
        } else {
            codes[destination].copy_from_slice(&bf16_bytes(&decoded));
        }
    }
    let mut results = vec![("codes".into(), Expected::Bits(codes))];
    if query.packed() {
        results.push(("scales".into(), Expected::Bits(scales)));
    }
    Ok(results)
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

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    use kernels_sm120::backend::Sm120;
    use kernels_sm120::gated_pooling::key_write::Entry;

    fn prepared_case(query: Query) -> OpCase<'static> {
        let entry = match query {
            Query::Index(q) => Entry::index(&q, "chop_fp4_test_v1").unwrap(),
            Query::Compressed(q) => Entry::compressed(&q, "chop_fp4_test_v1").unwrap(),
        };
        let backend = Sm120::new().with_fp4_key_writes([entry]);
        let prepared = match query {
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
            if matches!(query, Query::Index(_))
                && correct != bits(oracle(query, &early_projection, CAPACITY).unwrap())
            {
                projection_changes += 1;
            }
            if correct != bits(oracle_rounding(query, &images, CAPACITY, true).unwrap()) {
                match query {
                    Query::Index(_) => norm_changes += 1,
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
                Query::Compressed(q) => Entry::compressed(&q, "chop_fp4_test_v1").unwrap(),
            };
            let backend = Sm120::new().with_fp4_key_writes([entry]);
            let prepared = match query {
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
