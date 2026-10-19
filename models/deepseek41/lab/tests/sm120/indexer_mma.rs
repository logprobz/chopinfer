#[path = "indexer_mma_guard.rs"]
mod guard;
// Serving bindings and compact mapping against the sealed real operands.
use kernel_core::{
    attention::BlockTables,
    backend::{Backend, Counts, Launch, Need, Prepared, View},
    contract::Argument,
    pooled_indexer::{PositionKeyPages, PositionSelectBuffers},
    position_blocks::{CandidateBlockIds, PositionBlockSelectBuffers},
};
use kernel_harness::{LabResult, Session};
use kernels_sm120::{backend::Sm120, pooled_indexer::position_mma::Workspace};
use model_deepseek41_contract::{geometry::FLASH, indexer};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

fn backend(native: bool) -> Sm120 {
    let old = crate::indexer_entries::register(Sm120::new(), &FLASH);
    if native {
        old.with_native_position_scores()
    } else {
        old
    }
}
fn v(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}
fn cpu_buffers(rows: u32, ratio: u32, width: u32) -> PositionSelectBuffers<View> {
    let page = 64 / ratio;
    let n = u64::from(rows);
    PositionSelectBuffers {
        q: v(1, n * 8192),
        head_weights: v(2, n * 128),
        keys: PositionKeyPages {
            codes: v(3, u64::from(width) * 64),
            scales: v(4, u64::from(width) * 4),
            pages: width / page,
            code_page_stride: u64::from(page) * 64,
            scale_page_stride: u64::from(page) * 4,
            page_tables: BlockTables {
                tables: v(5, u64::from(width / page) * 4),
                sequences: 1,
                max_pages: width / page,
            },
        },
        positions: v(6, n * 4),
        row_batch: v(7, n * 4),
        live_rows: v(8, 4),
        candidate_mask: None,
        position_capacity: width,
        scores: v(9, n * u64::from(width) * 4),
        status: v(10, n * 4),
        output: v(11, n * 512 * 4),
    }
}
fn cpu_compare<B>(old: Prepared<B>, new: Prepared<B>, buffers: &B) {
    let layout = Workspace::new(old.info().workspace, 4096).unwrap();
    assert_eq!(new.info().workspace, layout.bytes);
    assert!(layout.codes >= old.info().workspace && layout.codes % 256 == 0);
    assert_eq!(layout.scales - layout.codes, 4096 * 2048);
    assert_eq!(layout.status - layout.scales, 4096 * 128);
    for rows in [1, 2, 4, 6, 8, 16, 32, 64, 4096] {
        let scratch = v(99, new.info().workspace);
        let a = old.launches(buffers, scratch, Counts::rows(rows)).unwrap();
        let b = new.launches(buffers, scratch, Counts::rows(rows)).unwrap();
        assert_eq!(b.len(), a.len() + 1);
        assert!(b[0].symbol.ends_with("_pack_fp4_v1"));
        assert!(b[1].symbol.ends_with("_mma_fp4_v1"));
        assert_eq!(b[0].plan.grid, [rows, 1, 1]);
        assert_eq!(b[0].plan.block, [256, 1, 1]);
        assert!(b.iter().all(|call| call.plan.pdl));
        assert_eq!(b[1].plan, a[0].plan);
        assert_eq!(b[2..], a[1..], "selection must remain unchanged");
        assert_eq!(
            b[0].arguments.0[..2],
            [Argument::DeviceBuffer(1), Argument::DeviceBuffer(8)]
        );
        assert_eq!(b[1].arguments.0[..a[0].arguments.0.len()], a[0].arguments.0);
        assert!(
            new.launches(buffers, v(99, new.info().workspace - 1), Counts::rows(rows))
                .is_err()
        );
        assert!(
            new.launches(buffers, v(1, new.info().workspace), Counts::rows(rows))
                .is_err()
        );
    }
}
#[test]
fn cpu_native_packing_keeps_selection_and_scratch_disjoint() {
    for layer in [2, 20, 24] {
        let width = 8192;
        let dense = cpu_buffers(4096, FLASH.compress_ratio(layer), width);
        if let Some(query) = indexer::block_selection(&FLASH, layer, 4096) {
            let buffers = PositionBlockSelectBuffers {
                q: dense.q,
                head_weights: dense.head_weights,
                keys: dense.keys,
                positions: dense.positions,
                row_batch: dense.row_batch,
                live_rows: dense.live_rows,
                row_tags: v(12, 4096 * 8),
                candidates: CandidateBlockIds {
                    ids: v(13, 4096 * 2048 * 4),
                    row_tags: v(14, 4096 * 8),
                    visible_positions: v(15, 4096 * 4),
                    status: v(16, 4096 * 4),
                    position_capacity: width,
                },
                scores: v(9, 4096 * 16384 * 4),
                status: dense.status,
                output: dense.output,
            };
            for k in [1, 128, 512] {
                cpu_compare(
                    backend(false)
                        .position_block_indexer(&query, k, width, Need::rows(4096))
                        .unwrap(),
                    backend(true)
                        .position_block_indexer(&query, k, width, Need::rows(4096))
                        .unwrap(),
                    &buffers,
                );
            }
        } else {
            let query = indexer::selection(&FLASH, layer, 4096).unwrap();
            for k in [1, 128, 512] {
                cpu_compare(
                    backend(false)
                        .position_indexer(&query, k, width, Need::rows(4096))
                        .unwrap(),
                    backend(true)
                        .position_indexer(&query, k, width, Need::rows(4096))
                        .unwrap(),
                    &dense,
                );
            }
        }
    }
    assert!(Workspace::new(u64::MAX, 1).is_err());
}

struct Arena<'a> {
    session: &'a Session,
    buffers: Vec<runtime_cuda::DeviceBuffer>,
}
impl Arena<'_> {
    fn input(&mut self, bytes: &[u8]) -> LabResult<View> {
        let buffer = self.session.allocate(bytes.len())?;
        buffer
            .copy_from_bytes(bytes, &self.session.stream)
            .map_err(|e| e.to_string())?;
        self.buffers.push(buffer);
        Ok(v(self.buffers.len() as u64, bytes.len() as u64))
    }
    fn output(&mut self, bytes: usize) -> LabResult<View> {
        self.input(&vec![0xa5; bytes])
    }
    fn write(&self, view: View, bytes: &[u8]) -> LabResult<()> {
        self.buffers[view.handle as usize - 1]
            .copy_from_bytes(bytes, &self.session.stream)
            .map_err(|e| e.to_string())
    }
    fn read(&self, view: View) -> LabResult<Vec<u8>> {
        kernel_harness::poison::read(self.session, &self.buffers[view.handle as usize - 1])
    }
    fn poison(&self, views: &[View]) -> LabResult<()> {
        for &view in views {
            self.write(view, &vec![0xa5; view.bytes as usize])?;
        }
        Ok(())
    }
}
fn bytes(path: &Path) -> LabResult<Vec<u8>> {
    fs::read(path).map_err(|e| e.to_string())
}
fn read_json(path: &Path) -> LabResult<serde_json::Value> {
    serde_json::from_slice(&bytes(path)?).map_err(|e| e.to_string())
}
fn ints(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values.into_iter().flat_map(i32::to_le_bytes).collect()
}
fn values(meta: &serde_json::Value, name: &str) -> Vec<i32> {
    meta[name]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect()
}
fn fire(
    session: &Session,
    arena: &Arena<'_>,
    modules: &BTreeMap<String, runtime_cuda::Module>,
    calls: &[Launch],
) -> LabResult<()> {
    let bindings = kernel_harness::launch::Bindings {
        buffers: arena
            .buffers
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u64 + 1, b))
            .collect(),
    };
    for call in calls {
        let (name, _) = bundle_deepseek41::sm120::fold_cubin_for_symbol(&call.symbol)
            .ok_or("missing served cubin")?;
        let function = modules[name]
            .function(&call.symbol)
            .map_err(|e| e.to_string())?;
        kernel_harness::launch::launch(
            &session.context,
            &function,
            &session.stream,
            call.plan,
            &call.arguments,
            &bindings,
        )?;
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn compare<B>(
    session: &Session,
    arena: &mut Arena<'_>,
    buffers: &B,
    planes: [View; 3],
    ops: [Prepared<B>; 2],
    rows: u32,
    active: u32,
    modules: &mut BTreeMap<String, runtime_cuda::Module>,
) -> LabResult<[Vec<u8>; 3]> {
    let mut result = Vec::new();
    for op in ops {
        let scratch = arena.output(op.info().workspace as usize)?;
        let calls = op
            .launches(buffers, scratch, Counts::rows(rows))
            .map_err(|e| e.to_string())?;
        for call in &calls {
            let (name, data) = bundle_deepseek41::sm120::fold_cubin_for_symbol(&call.symbol)
                .ok_or("missing served entry")?;
            if !modules.contains_key(name) {
                modules.insert(name.to_owned(), session.load_cubin(data)?);
            }
        }
        let graph = session.capture(|_| fire(session, arena, modules, &calls))?;
        let mut previous = None;
        for _ in 0..2 {
            arena.poison(&[planes[0], planes[1], planes[2], scratch])?;
            graph.launch(&session.stream).map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let output = [
                arena.read(planes[0])?,
                arena.read(planes[1])?,
                arena.read(planes[2])?,
            ];
            for (view, bytes) in planes.iter().zip(&output) {
                let first = (view.bytes / u64::from(rows) * u64::from(active)) as usize;
                assert!(
                    bytes[first..].iter().all(|&b| b == 0xa5),
                    "inactive rows changed"
                );
            }
            if let Some(ref old) = previous {
                assert_eq!(old, &output, "graph replay changed bytes");
            }
            previous = Some(output);
        }
        result.push(previous.unwrap());
    }
    assert_eq!(
        result[0], result[1],
        "served native score/select differs from SIMT"
    );
    Ok(result.remove(0))
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_INDEX_CAPTURE_MANIFEST and idle GPUs"]
fn gpu_serving_scores_and_failure_paths_match_simt() {
    captured_comparisons(backend);
}
fn captured_comparisons(backend: fn(bool) -> Sm120) {
    let ran = kernel_harness::gpu_test(2 << 30, |session| {
        let manifest = PathBuf::from(
            std::env::var_os("CHOPINFER_INDEX_CAPTURE_MANIFEST")
                .ok_or("capture manifest required")?,
        );
        let data = read_json(&manifest)?;
        let shard: usize = std::env::var("CHOPINFER_L4_SHARD")
            .map_err(|e| e.to_string())?
            .parse()
            .map_err(|_| "bad shard")?;
        assert!(shard < 2);
        let mut comparisons = 0;
        let mut modules = BTreeMap::new();
        for (i, capture) in data["captures"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == shard)
        {
            let dir = Path::new(capture["path"].as_str().unwrap());
            let meta = read_json(&dir.join("operands.json"))?;
            let width = meta["width"].as_u64().unwrap() as u32;
            let layer = meta["layer"].as_u64().unwrap() as u32;
            let page = meta["page_size"].as_u64().unwrap() as u32;
            let compact = meta["reindex"].as_bool().unwrap();
            let rows = 32_u32;
            let source_q = bytes(&dir.join("query.bf16"))?;
            let source_w = bytes(&dir.join("weights.f32"))?;
            for case in 0..10 {
                let mut arena = Arena {
                    session,
                    buffers: vec![],
                };
                let mut q = source_q.clone();
                let mut weights = source_w.clone();
                let mut positions = values(&meta, "positions");
                let mut live = rows;
                let mut scales = bytes(&dir.join("scales.ue8m0"))?;
                let top_k = if case == 7 { 128 } else { 512 };
                match case {
                    1 => {
                        live = 3;
                    }
                    2 => {
                        live = 0;
                    }
                    3 => {
                        q[(31 * 4096) * 2..(31 * 4096) * 2 + 2]
                            .copy_from_slice(&0x3f81_u16.to_le_bytes());
                    }
                    4 => {
                        q[(31 * 4096) * 2..(31 * 4096) * 2 + 2]
                            .copy_from_slice(&0x7fc1_u16.to_le_bytes());
                    }
                    5 => {
                        weights[(31 * 32) * 4..(31 * 32) * 4 + 4]
                            .copy_from_slice(&f32::INFINITY.to_le_bytes());
                    }
                    6 => {
                        scales[0] = 255;
                    }
                    8 if !compact => positions.fill(-1),
                    9 if !compact => positions.fill(0),
                    _ => (),
                }
                let dense = PositionSelectBuffers {
                    q: arena.input(&q)?,
                    head_weights: arena.input(&weights)?,
                    keys: PositionKeyPages {
                        codes: arena.input(&bytes(&dir.join("keys.e2m1"))?)?,
                        scales: arena.input(&scales)?,
                        pages: meta["physical_pages"].as_array().unwrap().len() as u32,
                        code_page_stride: u64::from(page) * 64,
                        scale_page_stride: u64::from(page) * 4,
                        page_tables: BlockTables {
                            tables: arena.input(&ints(values(&meta, "page_table")))?,
                            sequences: 1,
                            max_pages: width / page,
                        },
                    },
                    positions: arena.input(&ints(positions))?,
                    row_batch: arena.input(&ints(vec![0; 32]))?,
                    live_rows: arena.input(&live.to_le_bytes())?,
                    candidate_mask: None,
                    position_capacity: width,
                    scores: arena.output(32 * if compact { 16384 } else { width as usize } * 4)?,
                    status: arena.output(32 * 4)?,
                    output: arena.output(32 * 512 * 4)?,
                };
                let planes = [dense.scores, dense.status, dense.output];
                let output = if compact {
                    let mut ids = bytes(&dir.join("candidates.i32"))?;
                    let mut tags = bytes(&dir.join("candidate_tags.u64"))?;
                    if case == 8 {
                        tags[0] ^= 1;
                    }
                    if case == 9 {
                        ids[(31 * 2048 + 2047) * 4..(31 * 2048 + 2048) * 4]
                            .copy_from_slice(&i32::MAX.to_le_bytes());
                    }
                    let buffers = PositionBlockSelectBuffers {
                        q: dense.q,
                        head_weights: dense.head_weights,
                        keys: dense.keys,
                        positions: dense.positions,
                        row_batch: dense.row_batch,
                        live_rows: dense.live_rows,
                        row_tags: arena.input(&bytes(&dir.join("row_tags.u64"))?)?,
                        candidates: CandidateBlockIds {
                            ids: arena.input(&ids)?,
                            row_tags: arena.input(&tags)?,
                            visible_positions: arena
                                .input(&bytes(&dir.join("candidate_visible.i32"))?)?,
                            status: arena.input(&bytes(&dir.join("candidate_status.u32"))?)?,
                            position_capacity: width,
                        },
                        scores: dense.scores,
                        status: dense.status,
                        output: dense.output,
                    };
                    let query = indexer::block_selection(&FLASH, layer, rows).unwrap();
                    let ops = [false, true].map(|native| {
                        backend(native)
                            .position_block_indexer(&query, top_k, width, Need::rows(rows))
                            .unwrap()
                    });
                    compare(
                        session,
                        &mut arena,
                        &buffers,
                        planes,
                        ops,
                        rows,
                        live,
                        &mut modules,
                    )?
                } else {
                    let query = indexer::selection(&FLASH, layer, rows).unwrap();
                    let ops = [false, true].map(|native| {
                        backend(native)
                            .position_indexer(&query, top_k, width, Need::rows(rows))
                            .unwrap()
                    });
                    compare(
                        session,
                        &mut arena,
                        &dense,
                        planes,
                        ops,
                        rows,
                        live,
                        &mut modules,
                    )?
                };
                if case == 0 {
                    assert_eq!(
                        output[2],
                        bytes(&dir.join("selected.i32"))?,
                        "capture selection changed"
                    );
                }
                comparisons += 1;
            }
            println!("SERVING_MMA_CAPTURE index={i} cases=10");
        }
        println!("SERVING_MMA_COMPLETE shard={shard} comparisons={comparisons}");
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some(), "GPU test skipped");
}
