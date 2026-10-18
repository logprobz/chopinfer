//! Compare every compact score slot and original-position selection against
//! dense Position v3 across row splits, incomplete groups and four reindexes.
use super::{backend, family};
use kernel_core::{
    attention::BlockTables,
    backend::{Backend, Counts, Launch, Need, View},
    codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64, f32_to_bf16},
    contract::DType,
    oracle::check_numeric,
    pooled_indexer::{
        CandidateBlocksBuffers, IndexerCandidates, PositionKeyPages, PositionSelectBuffers,
    },
    position_blocks::{CandidateBlockIds, CandidateBlockIdsBuffers, PositionBlockSelectBuffers},
    position_blocks_oracle, position_indexer_oracle,
};
use kernel_harness::{
    LabResult, Session,
    launch::{Bindings, launch},
    poison::{Replay, WriteMap},
};
use runtime_cuda::DeviceBuffer;
use std::collections::BTreeMap;

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn indices(v: &[u8]) -> Vec<i32> {
    v.chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn floats(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
struct Arena<'a> {
    session: &'a Session,
    buffers: Vec<DeviceBuffer>,
    outputs: Vec<(String, View, DType)>,
}
impl<'a> Arena<'a> {
    fn new(session: &'a Session) -> Self {
        Self {
            session,
            buffers: Vec::new(),
            outputs: Vec::new(),
        }
    }
    fn input(&mut self, bytes: &[u8]) -> LabResult<View> {
        let buffer = self.session.allocate(bytes.len())?;
        buffer
            .copy_from_bytes(bytes, &self.session.stream)
            .map_err(|e| e.to_string())?;
        self.buffers.push(buffer);
        Ok(View {
            handle: self.buffers.len() as u64,
            bytes: bytes.len() as u64,
        })
    }
    fn output(&mut self, name: &str, bytes: usize, dtype: DType) -> LabResult<View> {
        let view = self.input(&vec![0xa5; bytes])?;
        self.outputs.push((name.into(), view, dtype));
        Ok(view)
    }
    fn read(&self, view: View) -> LabResult<Vec<u8>> {
        kernel_harness::poison::read(self.session, &self.buffers[view.handle as usize - 1])
    }
    fn write(&self, view: View, bytes: &[u8]) -> LabResult<()> {
        self.buffers[view.handle as usize - 1]
            .copy_from_bytes(bytes, &self.session.stream)
            .map_err(|e| e.to_string())
    }
    fn run(&self, launches: &[Launch], poison: bool) -> LabResult<()> {
        let bindings = Bindings {
            buffers: self
                .buffers
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u64 + 1, b))
                .collect(),
        };
        let module = self.session.module(family::CUBIN)?;
        let functions = launches
            .iter()
            .map(|l| module.function(&l.symbol).map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        let run = |stream: &runtime_cuda::Stream| {
            for (l, f) in launches.iter().zip(&functions) {
                launch(
                    &self.session.context,
                    f,
                    stream,
                    l.plan,
                    &l.arguments,
                    &bindings,
                )?;
            }
            Ok(())
        };
        if poison {
            let graph = self.session.capture(run)?;
            let mut replay = Replay::new(self.session);
            for (name, view, dtype) in &self.outputs {
                // -1 is valid ID padding, so all-ones poison would collide.
                replay = replay.output_with(
                    name,
                    &self.buffers[view.handle as usize - 1],
                    WriteMap::of(*dtype).byte_prefix(view.bytes as usize)?,
                    &[0xa5],
                );
            }
            replay.eager(|| run(&self.session.stream))?;
            replay.relaunch(|| run(&self.session.stream))?;
            replay.graph(&graph)?;
        } else {
            run(&self.session.stream)?;
        }
        self.session.stream.synchronize().map_err(|e| e.to_string())
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ResultRow {
    ids: Vec<i32>,
    selections: Vec<Vec<i32>>,
    score_bits: Vec<Vec<u32>>,
}

#[test]
fn compact_dense_bits_match_across_chunks_and_reindex_on_a_gpu() {
    kernel_lab::gpu_test(2 << 30, |session| {
        for entry in family::entries() {
            let whole = chain(session, &entry, 4)?;
            for chunk in [1, 3] {
                assert_eq!(
                    whole,
                    chain(session, &entry, chunk)?,
                    "{} chunk {chunk}",
                    entry.name
                );
            }
        }
        session.receipt(
            "compact-dense-chain",
            BTreeMap::from([
                ("position_capacity".into(), serde_json::json!(1 << 20)),
                ("chunk_rows".into(), serde_json::json!([4, 1, 3])),
                ("reindex_passes".into(), serde_json::json!(4)),
                (
                    "exact".into(),
                    serde_json::json!([
                        "membership",
                        "mapped_score_bits",
                        "status",
                        "selection",
                        "row_splits"
                    ]),
                ),
            ]),
        )?;
        Ok(())
    })
    .expect("compact dense bit equality");
}

fn chain(
    session: &Session,
    entry: &kernels_sm120::pooled_indexer::position_blocks::Entry,
    chunk: usize,
) -> LabResult<Vec<ResultRow>> {
    let backend = backend();
    let cq = entry.query;
    let q = cq.indexer;
    let width = 1u32 << 20;
    let dim = q.head_dim as usize;
    let heads = q.heads as usize;
    let slots = cq.score_slots().map_err(|e| e.0)? as usize;
    let kept = cq.top_k_blocks as usize;
    let page = q.page_size as usize;
    let physical_pages = 17;
    let codes = (0..physical_pages * page * dim / 2)
        .map(|i| ((i * 13 + i / (dim / 2)) % 256) as u8)
        .collect::<Vec<_>>();
    let scales = (0..physical_pages * page * dim / 32)
        .map(|i| 124 + (i % 5) as u8)
        .collect::<Vec<_>>();
    let tables = (0..width as usize / page)
        .map(|i| {
            if i % 11 == 3 {
                -1
            } else {
                (i % physical_pages) as i32
            }
        })
        .collect::<Vec<_>>();
    // Adjacent positions straddle complete/partial blocks and compression
    // groups. The later chunk shares the prior newest block then advances it.
    let token_positions = [0, 7, 8, 16383, 16384, 16385, 32767, 32768, 32769];
    let mut result = Vec::new();
    for (chunk_index, positions) in token_positions.chunks(chunk).enumerate() {
        let rows = positions.len();
        let start = chunk_index * chunk;
        let visible = positions
            .iter()
            .map(|&p| ((p as u32 + 1) / q.compression_ratio) as i32)
            .collect::<Vec<_>>();
        let mut arena = Arena::new(session);
        let positions_gpu = arena.input(&i32s(positions))?;
        let batches_gpu = arena.input(&vec![0; rows * 4])?;
        let live = arena.input(&(rows as u32).to_le_bytes())?;
        let tags = (start..start + rows)
            .map(|i| 0x100000000u64 + i as u64)
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<_>>();
        let tags_gpu = arena.input(&tags)?;
        let visible_gpu = arena.input(&i32s(&visible))?;
        let keys = PositionKeyPages {
            codes: arena.input(&codes)?,
            scales: arena.input(&scales)?,
            pages: physical_pages as u32,
            code_page_stride: (page * dim / 2) as u64,
            scale_page_stride: (page * dim / 32) as u64,
            page_tables: BlockTables {
                tables: arena.input(&i32s(&tables))?,
                sequences: 1,
                max_pages: width / q.page_size,
            },
        };
        let filter = backend
            .indexer_candidate_blocks(&cq.filter(), width, Need::rows(rows as u32))
            .map_err(|e| e.to_string())?;
        let id_filter = backend
            .indexer_candidate_block_ids(&cq.filter(), width, Need::rows(rows as u32))
            .map_err(|e| e.to_string())?;
        let full = backend
            .position_indexer(&q, q.max_top_k, width, Need::rows(rows as u32))
            .map_err(|e| e.to_string())?;
        let masked_q = kernel_core::pooled_indexer::PositionIndexerQuery {
            candidates: IndexerCandidates::Mask,
            ..q
        };
        let dense = backend
            .position_indexer(&masked_q, q.max_top_k, width, Need::rows(rows as u32))
            .map_err(|e| e.to_string())?;
        let compact = backend
            .position_block_indexer(&cq, q.max_top_k, width, Need::rows(rows as u32))
            .map_err(|e| e.to_string())?;
        let record = CandidateBlockIds {
            ids: arena.output("ids", rows * kept * 4, DType::I32)?,
            row_tags: arena.output("stored tags", rows * 8, DType::I64)?,
            visible_positions: arena.output("stored visibility", rows * 4, DType::I32)?,
            status: arena.output("source status", rows * 4, DType::I32)?,
            position_capacity: width,
        };
        let mask = arena.output("dense mask", rows * width as usize, DType::U8)?;
        let workspace =
            arena.output("dense blocks", filter.info().workspace as usize, DType::F32)?;
        let id_workspace = arena.output(
            "compact blocks",
            id_filter.info().workspace as usize,
            DType::F32,
        )?;
        let mut launches = Vec::new();
        let mut dense_buffers = Vec::new();
        let mut compact_buffers = Vec::new();
        let mut operands = Vec::new();
        for pass in 0..=4 {
            let mut query = Vec::new();
            let mut weights = Vec::new();
            for row in start..start + rows {
                for i in 0..heads * dim {
                    query.push(f32_to_bf16(
                        SmallFloat::E2M1.decode_f32(((i * 7 + row * 3 + pass * 5) % 16) as u8)
                            * 2.0_f32.powi((i / 32 % 3) as i32 - 1),
                    ));
                }
                for head in 0..heads {
                    weights.push([0.50013f32, -1.25027, 2.00031][(head + pass + row) % 3]);
                }
            }
            let q_gpu = arena.input(
                &query
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let w_gpu = arena.input(
                &weights
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let b = PositionSelectBuffers {
                q: q_gpu,
                head_weights: w_gpu,
                keys,
                positions: positions_gpu,
                row_batch: batches_gpu,
                live_rows: live,
                candidate_mask: (pass > 0).then_some(mask),
                position_capacity: width,
                scores: arena.output(
                    &format!("dense scores {pass}"),
                    rows * width as usize * 4,
                    DType::F32,
                )?,
                status: arena.output(&format!("dense status {pass}"), rows * 4, DType::I32)?,
                output: arena.output(
                    &format!("dense selection {pass}"),
                    rows * q.max_top_k as usize * 4,
                    DType::I32,
                )?,
            };
            launches.extend(
                if pass == 0 { &full } else { &dense }
                    .launches(&b, View::EMPTY, Counts::rows(rows as u32))
                    .map_err(|e| e.0)?,
            );
            if pass == 0 {
                launches.extend(
                    filter
                        .launches(
                            &CandidateBlocksBuffers {
                                scores: b.scores,
                                visible_positions: visible_gpu,
                                live_rows: live,
                                position_capacity: width,
                                output: mask,
                            },
                            workspace,
                            Counts::rows(rows as u32),
                        )
                        .map_err(|e| e.0)?,
                );
                launches.extend(
                    id_filter
                        .launches(
                            &CandidateBlockIdsBuffers {
                                scores: b.scores,
                                visible_positions: visible_gpu,
                                row_tags: tags_gpu,
                                source_status: b.status,
                                live_rows: live,
                                output: record,
                            },
                            id_workspace,
                            Counts::rows(rows as u32),
                        )
                        .map_err(|e| e.0)?,
                );
            } else {
                let cb = PositionBlockSelectBuffers {
                    q: q_gpu,
                    head_weights: w_gpu,
                    keys,
                    positions: positions_gpu,
                    row_batch: batches_gpu,
                    row_tags: tags_gpu,
                    live_rows: live,
                    candidates: record,
                    scores: arena.output(
                        &format!("compact scores {pass}"),
                        rows * slots * 4,
                        DType::F32,
                    )?,
                    status: arena.output(
                        &format!("compact status {pass}"),
                        rows * 4,
                        DType::I32,
                    )?,
                    output: arena.output(
                        &format!("compact selection {pass}"),
                        rows * q.max_top_k as usize * 4,
                        DType::I32,
                    )?,
                };
                launches.extend(
                    compact
                        .launches(&cb, View::EMPTY, Counts::rows(rows as u32))
                        .map_err(|e| e.0)?,
                );
                compact_buffers.push(cb);
            }
            operands.push((query, weights));
            dense_buffers.push(b);
        }
        // Poison every produced plane and both block workspaces; replay the
        // complete Full -> candidates -> four Reindex graph on the same storage.
        arena.run(&launches, chunk == 4)?;
        let ids = indices(&arena.read(record.ids)?);
        let dense_mask = arena.read(mask)?;
        assert_eq!(arena.read(record.row_tags)?, tags);
        assert_eq!(indices(&arena.read(record.visible_positions)?), visible);
        assert_eq!(indices(&arena.read(record.status)?), vec![0; rows]);
        let source_scores = floats(&arena.read(dense_buffers[0].scores)?);
        for row in 0..rows {
            let row_ids = ids[row * kept..(row + 1) * kept].to_vec();
            let expected = position_blocks_oracle::candidate_ids(
                cq.filter(),
                &source_scores[row * width as usize..(row + 1) * width as usize],
                visible[row] as usize,
            )
            .map_err(|e| e.0)?;
            assert_eq!(row_ids, expected, "source own-F32 block IDs");
            let mut membership = vec![false; width as usize / 8];
            for &id in &row_ids {
                if id >= 0 {
                    membership[id as usize] = true;
                }
            }
            for (block, bytes) in dense_mask[row * width as usize..(row + 1) * width as usize]
                .chunks(8)
                .enumerate()
            {
                assert!(
                    bytes.iter().all(|&b| (b != 0) == membership[block]),
                    "dense membership row {row} block {block}"
                );
            }
            result.push(ResultRow {
                ids: row_ids,
                selections: Vec::new(),
                score_bits: Vec::new(),
            });
        }
        for pass in 1..=4 {
            let b = &dense_buffers[pass];
            let cb = &compact_buffers[pass - 1];
            let dense_scores = floats(&arena.read(b.scores)?);
            let compact_scores = floats(&arena.read(cb.scores)?);
            let dense_out = indices(&arena.read(b.output)?);
            assert_eq!(
                arena.read(b.output)?,
                arena.read(cb.output)?,
                "dense/compact pass {pass}"
            );
            assert_eq!(arena.read(b.status)?, arena.read(cb.status)?);
            assert_eq!(indices(&arena.read(cb.status)?), vec![0; rows]);
            for row in 0..rows {
                let row_ids = &ids[row * kept..(row + 1) * kept];
                let mut allowed = vec![false; slots];
                for slot in 0..slots {
                    let id = row_ids[slot / 8];
                    let index = if id < 0 {
                        width as usize
                    } else {
                        id as usize * 8 + slot % 8
                    };
                    let expected = if index < width as usize {
                        dense_scores[row * width as usize + index]
                    } else {
                        f32::NEG_INFINITY
                    };
                    assert_eq!(
                        compact_scores[row * slots + slot].to_bits(),
                        expected.to_bits(),
                        "mapped score row {row} pass {pass} slot {slot}"
                    );
                    allowed[slot] = index < visible[row] as usize && tables[index / page] >= 0;
                    // Bound every score in the 128-chain fixture, and samples
                    // distributed across the list for the wide-head fixture.
                    if allowed[slot] && (slots <= 16 || slot % (slots / 32) == 0) {
                        let physical = tables[index / page] as usize * page + index % page;
                        let key = (0..dim)
                            .map(|c| {
                                SmallFloat::E2M1.decode_f64(
                                    (codes[physical * dim / 2 + c / 2] >> ((c % 2) * 4)) & 15,
                                ) * decode_ue8m0_f64(scales[physical * dim / 32 + c / 32])
                            })
                            .collect::<Vec<_>>();
                        let (query, weights) = &operands[pass];
                        let decoded = query[row * heads * dim..(row + 1) * heads * dim]
                            .iter()
                            .map(|&v| f64::from(bf16_to_f32(v)))
                            .collect::<Vec<_>>();
                        let weights = weights[row * heads..(row + 1) * heads]
                            .iter()
                            .map(|&v| f64::from(v))
                            .collect::<Vec<_>>();
                        let reference = position_indexer_oracle::scores(
                            q.selection(),
                            &decoded,
                            &weights,
                            &key,
                        )
                        .map_err(|e| e.0)?;
                        check_numeric(
                            &[f64::from(compact_scores[row * slots + slot])],
                            &reference,
                            DType::F32,
                        )
                        .map_err(|e| format!("compact score f64 bound: {e:?}"))?;
                    }
                }
                let expected = position_blocks_oracle::select(
                    cq,
                    q.max_top_k,
                    width,
                    visible[row] as u32,
                    row_ids,
                    &compact_scores[row * slots..(row + 1) * slots],
                    &allowed,
                )
                .map_err(|e| e.0)?;
                assert_eq!(
                    &dense_out[row * q.max_top_k as usize..(row + 1) * q.max_top_k as usize],
                    expected
                );
                result[start + row].selections.push(expected);
                result[start + row].score_bits.push(
                    compact_scores[row * slots..(row + 1) * slots]
                        .iter()
                        .map(|s| s.to_bits())
                        .collect(),
                );
            }
        }
        if chunk == 4 && chunk_index == 0 {
            // A failed Full row must publish an empty retained record and
            // reject every later Reindex, without a block-max trap.
            let original = arena.read(dense_buffers[0].head_weights)?;
            let mut bad = original.clone();
            for row in 0..rows {
                bad[row * heads * 4..row * heads * 4 + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            }
            arena.write(dense_buffers[0].head_weights, &bad)?;
            arena.run(&launches, false)?;
            let expected = visible
                .iter()
                .map(|&n| i32::from(n > 0))
                .collect::<Vec<_>>();
            assert_eq!(indices(&arena.read(record.status)?), expected);
            assert!(indices(&arena.read(record.ids)?).iter().all(|&i| i == -1));
            for cb in &compact_buffers {
                assert_eq!(indices(&arena.read(cb.status)?), expected);
                assert!(indices(&arena.read(cb.output)?).iter().all(|&i| i == -1));
            }
            arena.write(dense_buffers[0].head_weights, &original)?;
            arena.run(&launches, false)?;
        }
        // Reuse the same storage with a tag from a previous chunk. The good
        // consumer must reject it, then overwrite status on a valid replay.
        let cb = compact_buffers[0];
        let stale = vec![0u8; rows * 8];
        arena.write(record.row_tags, &stale)?;
        let launches = compact
            .launches(&cb, View::EMPTY, Counts::rows(rows as u32))
            .map_err(|e| e.0)?;
        arena.run(&launches, false)?;
        assert_eq!(indices(&arena.read(cb.status)?), vec![1; rows]);
        assert!(indices(&arena.read(cb.output)?).iter().all(|&i| i == -1));
        arena.write(record.row_tags, &tags)?;
        arena.run(&launches, false)?;
        assert_eq!(indices(&arena.read(cb.status)?), vec![0; rows]);
    }
    Ok(result)
}

#[test]
fn compact_candidate_data_errors_report_status_on_a_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let query = family::candidates()[0].query;
        let backend = backend();
        let width = 40usize;
        let rows = 4usize;
        let prepared = backend
            .indexer_candidate_block_ids(&query, width as u32, Need::rows(rows as u32))
            .map_err(|e| e.to_string())?;
        let mut arena = Arena::new(session);
        let valid_scores = vec![1.0_f32; rows * width];
        let score_bytes = |scores: &[f32]| {
            scores
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let scores = arena.input(&score_bytes(&valid_scores))?;
        let visible = arena.input(&i32s(&[17; 4]))?;
        let source_status = arena.input(&i32s(&[0; 4]))?;
        let tags = [1u64, 2, 3, 4]
            .into_iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<_>>();
        let row_tags = arena.input(&tags)?;
        let live_rows = arena.input(&(rows as u32).to_le_bytes())?;
        let output = CandidateBlockIds {
            ids: arena.output(
                "filter IDs",
                rows * query.top_k_blocks as usize * 4,
                DType::I32,
            )?,
            row_tags: arena.output("filter tags", rows * 8, DType::I64)?,
            visible_positions: arena.output("filter counts", rows * 4, DType::I32)?,
            status: arena.output("filter status", rows * 4, DType::I32)?,
            position_capacity: width as u32,
        };
        let workspace = arena.output(
            "filter maxima",
            prepared.info().workspace as usize,
            DType::F32,
        )?;
        let buffers = CandidateBlockIdsBuffers {
            scores,
            visible_positions: visible,
            row_tags,
            source_status,
            live_rows,
            output,
        };
        let launches = prepared
            .launches(&buffers, workspace, Counts::rows(rows as u32))
            .map_err(|e| e.0)?;
        let valid_ids = vec![0, 2, 0, 2, 0, 2, 0, 2];
        for scenario in 0..3 {
            let mut values = valid_scores.clone();
            let mut counts = [17; 4];
            let mut source = [0; 4];
            let (expected_status, expected_ids) = match scenario {
                0 => {
                    values[width] = f32::NAN;
                    values[2 * width + 16] = f32::NAN; // Pinning must not hide this error.
                    values[3 * width] = f32::INFINITY;
                    ([0, 1, 1, 1], vec![0, 2, -1, -1, -1, -1, -1, -1])
                }
                1 => {
                    counts = [-1, width as i32 + 1, 0, 17];
                    values[2 * width..3 * width].fill(f32::NAN); // All future, never read.
                    values[3 * width + 17] = f32::INFINITY; // Just outside visibility.
                    ([1, 1, 0, 0], vec![-1, -1, -1, -1, -1, -1, 0, 2])
                }
                _ => {
                    source[0] = 1;
                    values[width..2 * width].fill(f32::NEG_INFINITY);
                    ([1, 0, 0, 0], vec![-1, -1, 2, -1, 0, 2, 0, 2])
                }
            };
            arena.write(scores, &score_bytes(&values))?;
            arena.write(visible, &i32s(&counts))?;
            arena.write(source_status, &i32s(&source))?;
            arena.run(&launches, true)?;
            assert_eq!(
                indices(&arena.read(output.status)?),
                expected_status,
                "filter scenario {scenario}"
            );
            assert_eq!(indices(&arena.read(output.ids)?), expected_ids);
            assert_eq!(arena.read(output.row_tags)?, tags);
            assert_eq!(
                indices(&arena.read(output.visible_positions)?),
                counts.map(|n| n.clamp(0, width as i32))
            );
            // Reuse all planes and prove that a prior failure is overwritten.
            arena.write(scores, &score_bytes(&valid_scores))?;
            arena.write(visible, &i32s(&[17; 4]))?;
            arena.write(source_status, &i32s(&[0; 4]))?;
            arena.run(&launches, false)?;
            assert_eq!(indices(&arena.read(output.status)?), [0; 4]);
            assert_eq!(indices(&arena.read(output.ids)?), valid_ids);
        }
        session.receipt(
            "compact-filter-data-errors",
            BTreeMap::from([(
                "cases".into(),
                serde_json::json!([
                    "nan",
                    "nan_in_pinned_block",
                    "positive_infinity",
                    "negative_visible",
                    "oversized_visible",
                    "future_nonfinite",
                    "failed_source",
                    "all_negative_infinity",
                    "status_reset"
                ]),
            )]),
        )?;
        Ok(())
    })
    .expect("compact candidate errors return status without a trap");
}
