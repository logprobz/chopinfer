//! Compact Position v3 exact witnesses, including stale and malformed rows.

use crate::{
    LabResult,
    conformance::{Expected, Rows},
    kit::{OpCase, Required, Spec},
};
use kernel_core::{
    attention::BlockTables,
    backend::{Need, Prepared, View},
    codec::f32_to_bf16,
    contract::DType,
    pooled_indexer::PositionKeyPages,
    position_blocks::{CandidateBlockIds, PositionBlockIndexerQuery, PositionBlockSelectBuffers},
    position_blocks_oracle as oracle, position_indexer_oracle,
};
use kernels_sm120::pooled_indexer::position_blocks::Entry;

const ROWS: u32 = 4;
const WIDTH: u32 = 40;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Witness {
    Valid,
    Unsorted,
    Duplicate,
    Stale,
    DroppedPin,
    NanWeight,
    NanDot,
    Overflow,
}

fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn tags(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

pub fn family(entries: &[Entry]) -> Vec<Required<'static>> {
    entries
        .iter()
        .filter(|e| e.query.top_k_blocks == 2)
        .map(|entry| {
            let q = entry.query;
            Required::new(
                format!("compact {}", entry.name),
                Need::rows(ROWS),
                move |backend, need| backend.position_block_indexer(&q, 7, WIDTH, need),
                move |op| case(q, Witness::Valid, op),
            )
            .required()
        })
        .collect()
}

pub fn witness(entry: &Entry, witness: Witness) -> Required<'static> {
    let q = entry.query;
    Required::new(
        format!("compact {witness:?}"),
        Need::rows(ROWS),
        move |backend, need| backend.position_block_indexer(&q, 7, WIDTH, need),
        move |op| case(q, witness, op),
    )
    .required()
}

pub fn case(
    q: PositionBlockIndexerQuery,
    witness: Witness,
    op: Prepared<PositionBlockSelectBuffers<View>>,
) -> LabResult<OpCase<'static>> {
    if q.top_k_blocks != 2 || q.indexer.head_dim < 160 {
        return Err("compact witness requires two blocks and a crossed chain".into());
    }
    let dim = q.indexer.head_dim as usize;
    let heads = q.indexer.heads as usize;
    let page = q.indexer.page_size as usize;
    let slots = q.score_slots().map_err(|e| e.0)? as usize;
    let mut ids = vec![0, 4, 0, 4, -1, -1, 0, 2];
    let visible = [40, 33, 0, 17];
    let current_tags = [0x100000001u64, 0x100000002, 0x100000003, 0x100000004];
    let mut stored_tags = current_tags;
    // Keep row zero valid: even the one-live-row conformance must need the
    // score launch. A malformed first row can reject without reading scores.
    match witness {
        Witness::Unsorted => ids[2..4].copy_from_slice(&[4, 0]),
        Witness::Duplicate => ids[2..4].copy_from_slice(&[4, 4]),
        Witness::Stale => {
            // A previous request chunk had another kept block but the same
            // extent and newest block. Only provenance distinguishes it.
            ids[2] = 1;
            stored_tags[1] -= 1u64 << 32;
        }
        Witness::DroppedPin => ids[3] = -1,
        _ => {}
    }
    let mut queries = vec![0u16; ROWS as usize * heads * dim];
    let mut weights = vec![0.0_f32; ROWS as usize * heads];
    for row in 0..ROWS as usize {
        queries[row * heads * dim] = f32_to_bf16(1.0);
        weights[row * heads] = 1.001;
    }
    let mut codes = vec![0x64; page * dim / 2];
    let mut scales = vec![126; page * dim / 32];
    match witness {
        Witness::NanWeight => weights[0] = f32::NAN,
        Witness::NanDot | Witness::Overflow => {
            queries[0] = f32_to_bf16(2.0_f32.powi(80));
            if witness == Witness::NanDot {
                queries[128] = f32_to_bf16(-2.0_f32.powi(80));
            }
            codes.fill(0x66);
            scales.fill(205);
        }
        _ => {}
    }
    let input = |dtype, width, bytes| Spec::input(dtype, (Rows::Live, width), bytes);
    let positions = visible.map(|v| v * q.indexer.compression_ratio as i32 - 1);
    let roles = PositionBlockSelectBuffers {
        q: input(
            DType::Bf16,
            heads * dim,
            queries.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        head_weights: input(DType::F32, heads, bytes_f32(&weights)),
        keys: PositionKeyPages {
            codes: Spec::input(DType::U8, (Rows::Fixed(1), codes.len()), codes),
            scales: Spec::input(DType::U8, (Rows::Fixed(1), scales.len()), scales),
            pages: 1,
            code_page_stride: (page * dim / 2) as u64,
            scale_page_stride: (page * dim / 32) as u64,
            page_tables: BlockTables {
                tables: Spec::input(DType::I32, (Rows::Fixed(1), 1), bytes_i32(&[0])),
                sequences: 1,
                max_pages: 1,
            },
        },
        positions: input(DType::I32, 1, bytes_i32(&positions)),
        row_batch: input(DType::I32, 1, vec![0; 16]),
        row_tags: input(
            DType::I64,
            1,
            current_tags.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        live_rows: Spec::live(),
        candidates: CandidateBlockIds {
            ids: input(DType::I32, 2, bytes_i32(&ids)),
            row_tags: input(
                DType::I64,
                1,
                stored_tags.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            visible_positions: input(DType::I32, 1, bytes_i32(&visible)),
            status: input(DType::I32, 1, vec![0; 16]),
            position_capacity: WIDTH,
        },
        scores: Spec::output(DType::F32, (Rows::Live, slots)).filled_with(&[0xa5]),
        status: Spec::output(DType::I32, (Rows::Live, 1)).filled_with(&[0xa5]),
        output: Spec::output(DType::I32, (Rows::Live, q.indexer.max_top_k as usize))
            .filled_with(&[0xa5]),
    };
    Ok(OpCase::new(op, ROWS, roles)?.oracle(move |images, live| {
        let ids = images.i32s("candidates.ids")?;
        let visible = images.i32s("candidates.visible_positions")?;
        let current = tags(images.bytes("row_tags")?);
        let stored = tags(images.bytes("candidates.row_tags")?);
        let source = images.i32s("candidates.status")?;
        let positions = images.i32s("positions")?;
        let qvalues = images.u16s("q")?;
        let w = images.f32s("head_weights")?;
        let codes = images.bytes("keys.codes")?;
        let scales = images.bytes("keys.scales")?;
        let mut scores = vec![f32::NEG_INFINITY; ROWS as usize * slots];
        let mut selected = vec![-1; ROWS as usize * q.indexer.max_top_k as usize];
        let mut status = vec![0; ROWS as usize];
        for row in 0..live as usize {
            let count = if positions[row] < 0 {
                0
            } else {
                q.indexer.visible_positions(positions[row] as u64).unwrap() as u32
            };
            let row_ids = &ids[row * 2..(row + 1) * 2];
            if oracle::validate_ids(
                q,
                WIDTH,
                count,
                row_ids,
                current[row],
                stored[row],
                visible[row] as u32,
                source[row],
            )
            .is_err()
            {
                status[row] = 1;
                continue;
            }
            let query = qvalues[row * heads * dim..(row + 1) * heads * dim]
                .iter()
                .map(|&v| kernel_core::codec::bf16_to_f32(v))
                .collect::<Vec<_>>();
            let weights = &w[row * heads..(row + 1) * heads];
            let mut allowed = vec![false; slots];
            for slot in 0..slots {
                let id = row_ids[slot / 8];
                if id < 0 {
                    continue;
                }
                let index = id as usize * 8 + slot % 8;
                if index >= count as usize {
                    continue;
                }
                allowed[slot] = true;
                let key = (0..dim)
                    .map(|c| {
                        let code = (codes[index * dim / 2 + c / 2] >> ((c % 2) * 4)) & 15;
                        kernel_core::codec::SmallFloat::E2M1.decode_f32(code)
                            * kernel_core::codec::decode_ue8m0_f64(
                                scales[index * dim / 32 + c / 32],
                            ) as f32
                    })
                    .collect::<Vec<_>>();
                match position_indexer_oracle::scores_f32(
                    q.indexer.selection(),
                    &query,
                    weights,
                    &key,
                ) {
                    Ok(value) => scores[row * slots + slot] = value[0],
                    Err(_) => status[row] = 1,
                }
            }
            if status[row] == 0 {
                let result = oracle::select(
                    q,
                    7,
                    WIDTH,
                    count,
                    row_ids,
                    &scores[row * slots..(row + 1) * slots],
                    &allowed,
                )
                .map_err(|e| e.0)?;
                selected
                    [row * q.indexer.max_top_k as usize..(row + 1) * q.indexer.max_top_k as usize]
                    .copy_from_slice(&result);
            }
        }
        Ok(vec![
            ("status".into(), Expected::Bits(bytes_i32(&status))),
            ("scores".into(), Expected::Bits(bytes_f32(&scores))),
            ("output".into(), Expected::Bits(bytes_i32(&selected))),
        ])
    }))
}
