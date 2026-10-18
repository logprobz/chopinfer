//! Position-indexer cases: f64-bounded scores and exact selection over the
//! actual GPU scores. Separate one-term cases give exact score bits for
//! the high-page-id and missing-page witnesses.

use crate::{
    LabResult,
    conformance::{Expected, Holes, Images, Rows},
    kit::{OpCase, Required, Spec},
};
use kernel_core::{
    attention::BlockTables,
    backend::{Need, Prepared, View},
    codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64, f32_to_bf16},
    contract::DType,
    oracle::{OracleOutput, check_numeric},
    pooled_indexer::{
        CandidateBlocksBuffers, CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery,
        PositionKeyPages, PositionSelectBuffers,
    },
    position_indexer_oracle as oracle,
};
use kernels_sm120::pooled_indexer::position::{CandidateEntry, Entry};
use std::{cell::RefCell, rc::Rc};

const CAPACITY: u32 = 4;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

/// Required cases over the caller's registered entries. Every score form
/// gets an exact page-remap case and a bounded multi-head case. Filters
/// cross the 16,384-position boundary, including zero, full and partial
/// newest blocks.
pub fn family(entries: &[Entry], filters: &[CandidateEntry]) -> Vec<Required<'static>> {
    let mut ops = Vec::new();
    for entry in entries {
        for exact in [true, false] {
            let query = entry.query;
            let top_k = query.max_top_k.min(if exact { 512 } else { 7 });
            let width = if exact { top_k + 17 } else { 257 };
            ops.push(
                Required::new(
                    format!("position {} exact {exact}", entry.name),
                    Need::rows(CAPACITY),
                    move |backend, need| backend.position_indexer(&query, top_k, width, need),
                    move |op| selection_case(query, top_k, width, exact, op),
                )
                .required(),
            );
        }
    }
    for entry in filters {
        let query = entry.query;
        ops.push(
            Required::new(
                format!("candidate {} boundary", entry.name),
                Need::rows(CAPACITY),
                move |backend, need| backend.indexer_candidate_blocks(&query, 16392, need),
                move |op| candidate_case(query, 16392, op),
            )
            .required(),
        );
    }
    ops
}

pub fn selection_case(
    query: PositionIndexerQuery,
    top_k: u32,
    width: u32,
    exact: bool,
    prepared: Prepared<PositionSelectBuffers<View>>,
) -> LabResult<OpCase<'static>> {
    let dim = query.head_dim as usize;
    let heads = query.heads as usize;
    let page = query.page_size as usize;
    let max_pages = width.div_ceil(query.page_size) as usize;
    let pages = 2 * max_pages + 17;
    let (code_stride, scale_stride) = (page * dim / 2, page * dim / 32);
    let codes = (0..pages * code_stride)
        .map(|i| {
            let low = ((i * 13 + i / code_stride) % 16) as u8;
            let high = ((i * 7 + i / (dim / 2) + 3) % 16) as u8;
            low | (high << 4)
        })
        .collect::<Vec<_>>();
    let scales = (0..pages * scale_stride)
        .map(|i| 124 + (i % 7) as u8)
        .collect::<Vec<_>>();
    let mut q = vec![0u16; CAPACITY as usize * heads * dim];
    let mut weights = vec![0.0_f32; CAPACITY as usize * heads];
    for row in 0..CAPACITY as usize {
        if exact {
            q[row * heads * dim] = f32_to_bf16(1.0);
            weights[row * heads] = 1.001;
        } else {
            for (i, value) in q[row * heads * dim..(row + 1) * heads * dim]
                .iter_mut()
                .enumerate()
            {
                *value = f32_to_bf16(
                    SmallFloat::E2M1.decode_f32(((i * 11 + row * 3) % 16) as u8)
                        * 2.0_f32.powi((i / 32 % 5) as i32 - 2),
                );
            }
            for head in 0..heads {
                weights[row * heads + head] = [0.50013, -1.25027, 2.00031][(head + row) % 3];
            }
        }
    }
    let tables = (0..2 * max_pages)
        .map(|i| (2 * (i % max_pages) + i / max_pages) as i32)
        .collect::<Vec<_>>();
    let table_spec = if exact {
        Spec::ids(
            (Rows::Fixed(2), max_pages),
            tables,
            &[("keys.codes", code_stride), ("keys.scales", scale_stride)],
            Holes::Row(0),
        )
    } else {
        Spec::input(DType::I32, (Rows::Fixed(2), max_pages), i32_bytes(&tables))
    };
    let row_values = |dtype, bytes| Spec::input(dtype, (Rows::Live, 1), bytes);
    let positions = [
        width as i32 * query.compression_ratio as i32 - 1,
        12 * query.compression_ratio as i32 - 1,
        -1,
        width as i32 * query.compression_ratio as i32 - 2,
    ];
    let mask = (0..CAPACITY as usize * width as usize)
        .map(|i| u8::from(i % 5 != 0))
        .collect::<Vec<_>>();
    let roles = PositionSelectBuffers {
        status: Spec::output(DType::I32, (Rows::Live, 1)),
        q: Spec::input(
            DType::Bf16,
            (Rows::Live, heads * dim),
            q.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        head_weights: Spec::input(
            DType::F32,
            (Rows::Live, heads),
            weights.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        keys: PositionKeyPages {
            codes: Spec::input(DType::U8, (Rows::Fixed(pages), code_stride), codes),
            scales: Spec::input(DType::U8, (Rows::Fixed(pages), scale_stride), scales),
            pages: pages as u32,
            code_page_stride: code_stride as u64,
            scale_page_stride: scale_stride as u64,
            page_tables: BlockTables {
                tables: table_spec,
                sequences: 2,
                max_pages: max_pages as u32,
            },
        },
        positions: row_values(DType::I32, i32_bytes(&positions)),
        row_batch: row_values(DType::I32, i32_bytes(&[0, 1, 0, 1])),
        live_rows: Spec::live(),
        position_capacity: width,
        candidate_mask: (query.candidates == IndexerCandidates::Mask)
            .then(|| Spec::input(DType::U8, (Rows::Live, width as usize), mask)),
        scores: Spec::output(DType::F32, (Rows::Live, width as usize)),
        output: Spec::output(DType::I32, (Rows::Live, query.max_top_k as usize))
            .filled_with(&[0xa5]),
    };
    Ok(
        OpCase::new(prepared, CAPACITY, roles)?.oracle(move |images, live| {
            selection_expected(query, top_k, width as usize, exact, images, live)
        }),
    )
}

struct RowReference {
    scores: OracleOutput,
    mask: Vec<bool>,
    position: i32,
}

fn selection_expected(
    query: PositionIndexerQuery,
    top_k: u32,
    width: usize,
    exact: bool,
    images: &Images,
    live: u32,
) -> LabResult<Vec<(String, Expected)>> {
    let dim = query.head_dim as usize;
    let heads = query.heads as usize;
    let page = query.page_size as usize;
    let tables = images.i32s("keys.page_tables.tables")?;
    let max_pages = width.div_ceil(page);
    let batches = images.i32s("row_batch")?;
    let positions = images.i32s("positions")?;
    let q = images.u16s("q")?;
    let weights = images.f32s("head_weights")?;
    let codes = images.bytes("keys.codes")?;
    let scales = images.bytes("keys.scales")?;
    let mask = (query.candidates == IndexerCandidates::Mask)
        .then(|| images.bytes("candidate_mask"))
        .transpose()?;
    let mut references = Vec::new();
    let mut exact_scores = vec![f32::NEG_INFINITY; CAPACITY as usize * width];
    for row in 0..live as usize {
        let mut keys = vec![0.0; width * dim];
        let mut allowed = vec![false; width];
        let visible = if positions[row] < 0 {
            0
        } else {
            query.visible_positions(positions[row] as u64).unwrap() as usize
        };
        for index in 0..width {
            let physical = tables[batches[row] as usize * max_pages + index / page];
            allowed[index] = index < visible
                && physical >= 0
                && mask.is_none_or(|mask| mask[row * width + index] != 0);
            if physical < 0 {
                continue;
            }
            let key = physical as usize * page + index % page;
            for channel in 0..dim {
                let byte = codes[key * dim / 2 + channel / 2];
                let code = (byte >> ((channel % 2) * 4)) & 15;
                keys[index * dim + channel] = SmallFloat::E2M1.decode_f64(code)
                    * decode_ue8m0_f64(scales[key * dim / 32 + channel / 32]);
            }
        }
        let q = q[row * heads * dim..(row + 1) * heads * dim]
            .iter()
            .map(|&v| bf16_to_f32(v))
            .collect::<Vec<_>>();
        let w = weights[row * heads..(row + 1) * heads].to_vec();
        if exact {
            let scored = oracle::scores_f32(
                query.selection(),
                &q,
                &w,
                &keys.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            )
            .map_err(|e| e.0)?;
            for i in 0..width {
                if allowed[i] {
                    exact_scores[row * width + i] = scored[i];
                }
            }
        }
        let reference = oracle::scores(
            query.selection(),
            &q.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &w.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &keys,
        )
        .map_err(|e| e.0)?;
        references.push(RowReference {
            scores: reference,
            mask: allowed,
            position: positions[row],
        });
    }
    if exact {
        let output = selected(query, top_k, width, &references, &exact_scores)?;
        return Ok(vec![
            ("scores".into(), Expected::Bits(f32_bytes(&exact_scores))),
            ("output".into(), Expected::Bits(i32_bytes(&output))),
            (
                "status".into(),
                Expected::Bits(vec![0; CAPACITY as usize * 4]),
            ),
        ]);
    }
    let references = Rc::new(references);
    let actual_scores = Rc::new(RefCell::new(None));
    let score_refs = references.clone();
    let saved_scores = actual_scores.clone();
    let score_judge = move |bytes: &[u8]| -> LabResult<()> {
        let actual = decode_f32(bytes);
        if actual.len() != CAPACITY as usize * width {
            return Err("score output size mismatch".into());
        }
        for (row, reference) in score_refs.iter().enumerate() {
            for i in 0..width {
                let value = actual[row * width + i];
                if !reference.mask[i] {
                    if value.to_bits() != f32::NEG_INFINITY.to_bits() {
                        return Err(format!("excluded score {row}:{i} was read or not masked"));
                    }
                    continue;
                }
                let element = OracleOutput {
                    values: vec![reference.scores.values[i]],
                    sum_abs_products: vec![reference.scores.sum_abs_products[i]],
                    reduction_lengths: vec![reference.scores.reduction_lengths[i]],
                };
                check_numeric(&[f64::from(value)], &element, DType::F32)
                    .map_err(|e| format!("score {row}:{i}: {e:?}"))?;
            }
        }
        *saved_scores.borrow_mut() = Some(actual);
        Ok(())
    };
    let select_judge = move |bytes: &[u8]| -> LabResult<()> {
        let scores = actual_scores.borrow();
        let scores = scores
            .as_ref()
            .ok_or("selection judge needs this run's checked F32 scores")?;
        let expected = i32_bytes(&selected(query, top_k, width, &references, scores)?);
        let checked = live as usize * query.max_top_k as usize * 4;
        if bytes.len() != expected.len() || bytes[..checked] != expected[..checked] {
            return Err("selection differs from exact top-k over this run's F32 scores".into());
        }
        Ok(())
    };
    // Case compares in returned order. The first judge both bounds and
    // captures this run's scores; no emulated score feeds the second.
    Ok(vec![
        ("scores".into(), Expected::Judge(Box::new(score_judge))),
        ("output".into(), Expected::Judge(Box::new(select_judge))),
        (
            "status".into(),
            Expected::Bits(vec![0; CAPACITY as usize * 4]),
        ),
    ])
}

fn selected(
    query: PositionIndexerQuery,
    top_k: u32,
    width: usize,
    refs: &[RowReference],
    scores: &[f32],
) -> LabResult<Vec<i32>> {
    let mut output = vec![-1; CAPACITY as usize * query.max_top_k as usize];
    for (row, reference) in refs.iter().enumerate() {
        if reference.position < 0 {
            continue;
        }
        let masked = PositionIndexerQuery {
            candidates: IndexerCandidates::Mask,
            ..query
        };
        let selected = oracle::select(
            masked,
            top_k,
            reference.position as u64,
            &scores[row * width..(row + 1) * width],
            Some(&reference.mask),
        )
        .map_err(|e| e.0)?;
        output[row * query.max_top_k as usize..(row + 1) * query.max_top_k as usize]
            .copy_from_slice(&selected);
    }
    Ok(output)
}

pub fn candidate_case(
    query: CandidateBlocksQuery,
    width: u32,
    prepared: Prepared<CandidateBlocksBuffers<View>>,
) -> LabResult<OpCase<'static>> {
    let scores = (0..CAPACITY as usize * width as usize)
        .map(|i| {
            let column = i % width as usize;
            if column >= 16384 {
                -10.0
            } else {
                ((column / 8) % 13) as f32
            }
        })
        .collect::<Vec<_>>();
    let roles = CandidateBlocksBuffers {
        scores: Spec::input(DType::F32, (Rows::Live, width as usize), f32_bytes(&scores)),
        // Live-one chain omission must need the block-score producer.
        // Keep the empty row later so every live prefix has that witness.
        visible_positions: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&[16385, 8, 16, 0])),
        position_capacity: width,
        live_rows: Spec::live(),
        output: Spec::output(DType::U8, (Rows::Live, width as usize)),
    };
    Ok(
        OpCase::new(prepared, CAPACITY, roles)?.oracle(move |images, live| {
            let scores = images.f32s("scores")?;
            let visible = images.i32s("visible_positions")?;
            let mut output = vec![0u8; CAPACITY as usize * width as usize];
            for (row, &visible) in visible.iter().enumerate().take(live as usize) {
                let start = row * width as usize;
                let mask = oracle::select_candidate_blocks(
                    query,
                    &scores[start..start + width as usize],
                    visible as usize,
                )
                .map_err(|e| e.0)?;
                for (out, keep) in output[start..start + width as usize].iter_mut().zip(mask) {
                    *out = u8::from(keep);
                }
            }
            Ok(vec![("output".into(), Expected::Bits(output))])
        }),
    )
}

/// Small explicit device witnesses, separate from the broad numeric cases.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Witness {
    Ties,
    Overflow,
    NanWeight,
    NanDot,
}

/// Eight equal finite candidates must keep [0, 1]. Only head zero has a
/// weight, and the low/high key nibbles decode to 1/2, so the same fixture
/// independently witnesses head-weight and nibble faults at the score oracle.
/// Failure cases use exactly representable decoded MXFP4 values of 2^80.
pub fn witness_case(
    query: PositionIndexerQuery,
    witness: Witness,
    prepared: Prepared<PositionSelectBuffers<View>>,
) -> LabResult<OpCase<'static>> {
    let (heads, dim, page) = (
        query.heads as usize,
        query.head_dim as usize,
        query.page_size as usize,
    );
    if query.candidates != IndexerCandidates::All
        || heads < 2
        || dim < 160
        || page < 8
        || query.max_top_k < 2
    {
        return Err(
            "position witnesses require an unrestricted multi-head 160-channel entry".into(),
        );
    }
    let width = 8;
    let mut codes = vec![0_u8; page * dim / 2];
    let mut scales = vec![127_u8; page * dim / 32];
    let mut q = vec![0_u16; CAPACITY as usize * heads * dim];
    let mut weights = vec![0_f32; CAPACITY as usize * heads];
    for row in 0..CAPACITY as usize {
        q[row * heads * dim] = f32_to_bf16(1.0);
        weights[row * heads] = 1.0;
        match witness {
            Witness::Overflow | Witness::NanDot => {
                q[row * heads * dim] = f32_to_bf16(2.0_f32.powi(80));
                if witness == Witness::NanDot {
                    q[row * heads * dim + 128] = f32_to_bf16(-2.0_f32.powi(80));
                }
            }
            Witness::NanWeight => weights[row * heads] = f32::NAN,
            Witness::Ties => {}
        }
    }
    for i in 0..width as usize {
        codes[i * dim / 2] = 0x64;
        scales[i * dim / 32] = 126;
        if matches!(witness, Witness::Overflow | Witness::NanDot) {
            codes[i * dim / 2] = 6;
            scales[i * dim / 32] = 205;
            if witness == Witness::NanDot {
                codes[i * dim / 2 + 64] = 6;
                scales[i * dim / 32 + 4] = 205;
            }
        }
    }
    let roles = PositionSelectBuffers {
        q: Spec::input(
            DType::Bf16,
            (Rows::Live, heads * dim),
            q.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        head_weights: Spec::input(DType::F32, (Rows::Live, heads), f32_bytes(&weights)),
        keys: PositionKeyPages {
            codes: Spec::input(DType::U8, (Rows::Fixed(1), page * dim / 2), codes),
            scales: Spec::input(DType::U8, (Rows::Fixed(1), page * dim / 32), scales),
            pages: 1,
            code_page_stride: (page * dim / 2) as u64,
            scale_page_stride: (page * dim / 32) as u64,
            page_tables: BlockTables {
                tables: Spec::input(DType::I32, (Rows::Fixed(1), 1), i32_bytes(&[0])),
                sequences: 1,
                max_pages: 1,
            },
        },
        positions: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(&[(width * query.compression_ratio - 1) as i32; CAPACITY as usize]),
        ),
        row_batch: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(&[0; CAPACITY as usize]),
        ),
        live_rows: Spec::live(),
        candidate_mask: None,
        position_capacity: width,
        scores: Spec::output(DType::F32, (Rows::Live, width as usize)).filled_with(&[0xa5]),
        status: Spec::output(DType::I32, (Rows::Live, 1)),
        output: Spec::output(DType::I32, (Rows::Live, query.max_top_k as usize))
            .filled_with(&[0xa5]),
    };
    Ok(OpCase::new(prepared, CAPACITY, roles)?.oracle(move |_, _| {
        let failed = witness != Witness::Ties;
        let score = if failed {
            f32::NEG_INFINITY
        } else {
            oracle::weight_scale(query.selection()).map_err(|e| e.0)?
        };
        let mut selected = vec![-1; CAPACITY as usize * query.max_top_k as usize];
        if !failed {
            for row in selected.chunks_exact_mut(query.max_top_k as usize) {
                row[..2].copy_from_slice(&[0, 1]);
            }
        }
        Ok(vec![
            (
                "scores".into(),
                Expected::Bits(f32_bytes(&vec![score; CAPACITY as usize * width as usize])),
            ),
            (
                "status".into(),
                Expected::Bits(i32_bytes(&[i32::from(failed); CAPACITY as usize])),
            ),
            ("output".into(), Expected::Bits(i32_bytes(&selected))),
        ])
    }))
}
