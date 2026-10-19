//! Paired old/new bitwise checks with live-context graph replay and timings.
use kernel_core::{
    attention::BlockTables,
    backend::{Backend, Counts, Launch, Need, View},
    pooled_indexer::{IndexerKeyFormat, PositionKeyPages, PositionSelectBuffers},
    position_blocks::{CandidateBlockIds, CandidateBlockIdsBuffers, PositionBlockSelectBuffers},
};
use kernel_harness::{
    LabResult, Session,
    launch::{Bindings, launch},
};
use kernel_lab_families::position_live as family;
use kernels_sm120::{backend::Sm120, pooled_indexer::position_live as live};
use runtime_cuda::{DeviceBuffer, Module};
use std::path::PathBuf;

fn backend(new: bool) -> Sm120 {
    let b = Sm120::new()
        .with_position_indexer([family::dense(1), family::dense(2)])
        .with_position_indexer_bf16([family::dense(1), family::dense(2)])
        .with_position_block_indexer([family::compact()])
        .with_position_block_indexer_bf16([family::compact()])
        .with_indexer_candidate_block_ids([family::candidates()])
        .with_indexer_candidate_block_ids_runtime([family::candidates()]);
    if new {
        b.with_live_position_indexer()
    } else {
        b
    }
}

fn cases() -> Vec<(u32, u32)> {
    let mut cases = Vec::new();
    for context in [1024, 8192, 32768, 131072, 1048576] {
        for rows in 1..=32 {
            cases.push((rows, context));
        }
    }
    cases.push((64, 8192));
    cases
}

#[test]
fn cpu_matrix_and_fixed_grid() {
    if let Some(root) = std::env::var_os("CHOPINFER_POSITION_LIVE_CUBINS") {
        kernel_build::report::BuildReport::read(
            &PathBuf::from(root).join("kernel-build-report.json"),
        )
        .and_then(|report| report.verify_current())
        .unwrap();
    }
    assert_eq!(cases().len(), 161);
    for rows in [1, 32, 64] {
        for width in [1024, 1 << 20] {
            let q = family::dense(1).query;
            let old = backend(false)
                .position_indexer(&q, 512, width, Need::rows(rows))
                .unwrap();
            let new = backend(true)
                .position_indexer(&q, 512, width, Need::rows(rows))
                .unwrap();
            assert_eq!(old.info().implementation, new.info().implementation);
            assert_eq!(old.info().guarantees, new.info().guarantees);
            assert_eq!(new.info().workspace, live::workspace_bytes(rows, 512));
            assert_eq!(new.info().symbols.len(), 3);
            let bytes = u64::from(rows);
            let v = |handle, bytes| View { handle, bytes };
            let b = PositionSelectBuffers {
                q: v(1, bytes * 32 * 128 * 2),
                head_weights: v(2, bytes * 32 * 4),
                keys: PositionKeyPages {
                    codes: v(3, u64::from(width) * 64),
                    scales: v(4, u64::from(width) * 4),
                    pages: width / 64,
                    code_page_stride: 4096,
                    scale_page_stride: 256,
                    page_tables: BlockTables {
                        tables: v(5, u64::from(width / 64) * 4),
                        sequences: 1,
                        max_pages: width / 64,
                    },
                },
                positions: v(6, bytes * 4),
                row_batch: v(7, bytes * 4),
                live_rows: v(8, 4),
                candidate_mask: None,
                position_capacity: width,
                scores: v(9, bytes * u64::from(width) * 4),
                status: v(10, bytes * 4),
                output: v(11, bytes * 512 * 4),
            };
            let scratch = v(12, new.info().workspace);
            let calls = new.launches(&b, scratch, Counts::rows(rows)).unwrap();
            assert_eq!(calls[0].plan.grid, [live::SCORE_CTAS, rows, 1]);
            assert_eq!(calls[1].plan.grid, [live::SHARDS, rows, 1]);
            assert_eq!(calls[2].plan.grid, [1, rows, 1]);
            assert!(calls.iter().all(|c| c.plan.pdl));
            let compiled = family::cubins().remove(0);
            for call in &calls {
                let entry = compiled
                    .entries
                    .iter()
                    .find(|e| e.symbol == call.symbol)
                    .unwrap();
                assert_eq!(entry.params.len(), call.arguments.0.len());
            }
            assert!(
                new.launches(
                    &b,
                    View {
                        bytes: scratch.bytes - 1,
                        ..scratch
                    },
                    Counts::rows(rows)
                )
                .is_err()
            );
            assert!(
                new.launches(
                    &b,
                    View {
                        handle: b.scores.handle,
                        ..scratch
                    },
                    Counts::rows(rows)
                )
                .is_err()
            );
            assert_eq!(
                old.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap()[0]
                    .plan
                    .grid[0],
                width / 4
            );
        }
    }
    for kept in [1, 65, 1024, 2048] {
        let q = family::candidates().query;
        let old = backend(false)
            .indexer_candidate_block_ids_runtime(&q, kept, 1 << 20, Need::rows(32))
            .unwrap();
        let new = backend(true)
            .indexer_candidate_block_ids_runtime(&q, kept, 1 << 20, Need::rows(32))
            .unwrap();
        assert_eq!(old.info().bytes, new.info().bytes);
        assert_eq!(
            new.info().workspace,
            old.info().workspace + live::workspace_bytes(32, 2048)
        );
    }
    println!(
        "CPU dry run: 160 decode shapes (rows 1..32, five contexts), one 64-row prefill; native/BF16; ratios 1/2; runtime candidate counts; graph shrink/grow"
    );
}

struct Arena<'a> {
    session: &'a Session,
    buffers: Vec<DeviceBuffer>,
}
impl Arena<'_> {
    fn input(&mut self, bytes: &[u8]) -> LabResult<View> {
        let b = self.session.allocate(bytes.len())?;
        b.copy_from_bytes(bytes, &self.session.stream)
            .map_err(|e| e.to_string())?;
        self.buffers.push(b);
        Ok(View {
            handle: self.buffers.len() as u64,
            bytes: bytes.len() as u64,
        })
    }
    fn output(&mut self, bytes: usize) -> LabResult<View> {
        self.input(&vec![0xa5; bytes])
    }
    fn write(&self, v: View, bytes: &[u8]) -> LabResult<()> {
        self.buffers[v.handle as usize - 1]
            .copy_from_bytes(bytes, &self.session.stream)
            .map_err(|e| e.to_string())
    }
    fn read(&self, v: View) -> LabResult<Vec<u8>> {
        kernel_harness::poison::read(self.session, &self.buffers[v.handle as usize - 1])
    }
    fn poison(&self, views: &[View]) -> LabResult<()> {
        for &v in views {
            self.write(v, &vec![0xa5; v.bytes as usize])?;
        }
        Ok(())
    }
    fn bindings(&self) -> Bindings<'_> {
        Bindings {
            buffers: self
                .buffers
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u64 + 1, b))
                .collect(),
        }
    }
}
fn i32s(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values.into_iter().flat_map(i32::to_le_bytes).collect()
}
fn fire(session: &Session, module: &Module, arena: &Arena<'_>, calls: &[Launch]) -> LabResult<()> {
    for c in calls {
        let f = module.function(&c.symbol).map_err(|e| e.to_string())?;
        launch(
            &session.context,
            &f,
            &session.stream,
            c.plan,
            &c.arguments,
            &arena.bindings(),
        )?;
    }
    Ok(())
}

fn paired(
    session: &Session,
    module: &Module,
    rows: u32,
    context: u32,
    bf16: bool,
    ratio: u32,
) -> LabResult<()> {
    let width = (1 << 20) / ratio;
    let n = rows as usize;
    let q = family::dense(ratio).query;
    let mut a = Arena {
        session,
        buffers: Vec::new(),
    };
    let operands = arithmetic::Operands::new(
        n,
        width as usize,
        bf16,
        arithmetic::SEED ^ (u64::from(rows) << 32) ^ u64::from(context) ^ u64::from(ratio),
    );
    let queries = a.input(&operands.query_bytes(false))?;
    let weights = a.input(&operands.weight_bytes(false))?;
    let codes = a.input(&operands.codes)?;
    let scales = if bf16 {
        View::EMPTY
    } else {
        a.input(&operands.scales)?
    };
    drop(operands);
    let pages = width / q.page_size;
    let tables = a.input(&i32s(
        (0..pages).map(|i| if i == 2 { -1 } else { (pages - 1 - i) as i32 }),
    ))?;
    let positions = a.input(&i32s(vec![0; n]))?;
    let visible = a.input(&i32s(vec![0; n]))?;
    let batches = a.input(&i32s(vec![0; n]))?;
    let live_rows = a.input(&rows.to_le_bytes())?;
    let tags = a.input(
        &(0..n as u64)
            .flat_map(|i| (0x12340000 + i).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let scores = a.output(n * width as usize * 4)?;
    let status = a.output(n * 4)?;
    let output = a.output(n * 512 * 4)?;
    let ids = CandidateBlockIds {
        ids: a.output(n * 2048 * 4)?,
        row_tags: a.output(n * 8)?,
        visible_positions: a.output(n * 4)?,
        status: a.output(n * 4)?,
        position_capacity: width,
    };
    let compact_scores = a.output(n * 16384 * 4)?;
    let compact_status = a.output(n * 4)?;
    let compact_output = a.output(n * 512 * 4)?;
    let keys = PositionKeyPages {
        codes,
        scales,
        pages,
        code_page_stride: u64::from(q.page_size) * if bf16 { 256 } else { 64 },
        scale_page_stride: if bf16 { 0 } else { u64::from(q.page_size) * 4 },
        page_tables: BlockTables {
            tables,
            sequences: 1,
            max_pages: pages,
        },
    };
    let storage = if bf16 {
        IndexerKeyFormat::Bf16
    } else {
        IndexerKeyFormat::Fp4
    };
    let dense = PositionSelectBuffers {
        q: queries,
        head_weights: weights,
        keys,
        positions,
        row_batch: batches,
        live_rows,
        candidate_mask: None,
        position_capacity: width,
        scores,
        status,
        output,
    };
    let producer = CandidateBlockIdsBuffers {
        scores,
        visible_positions: visible,
        row_tags: tags,
        source_status: status,
        live_rows,
        output: ids,
    };
    let compact = PositionBlockSelectBuffers {
        q: queries,
        head_weights: weights,
        keys,
        positions,
        row_batch: batches,
        live_rows,
        row_tags: tags,
        candidates: ids,
        scores: compact_scores,
        status: compact_status,
        output: compact_output,
    };
    let mut calls = Vec::new();
    let mut scratches = Vec::new();
    for new in [false, true] {
        let b = backend(new);
        let op = b
            .position_indexer_with_storage(&q, storage, 512, width, Need::rows(rows))
            .map_err(|e| e.to_string())?;
        let cp = b
            .indexer_candidate_block_ids_runtime(
                &family::candidates().query,
                if rows % 2 == 0 { 2048 } else { 65 },
                width,
                Need::rows(rows),
            )
            .map_err(|e| e.to_string())?;
        let reindex = if ratio == 1 {
            Some(
                b.position_block_indexer_with_storage(
                    &family::compact().query,
                    storage,
                    512,
                    width,
                    Need::rows(rows),
                )
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let size = op
            .info()
            .workspace
            .max(cp.info().workspace)
            .max(reindex.as_ref().map_or(0, |x| x.info().workspace));
        let scratch = a.output(size as usize)?;
        scratches.push(scratch);
        let mut seq = op
            .launches(&dense, scratch, Counts::rows(rows))
            .map_err(|e| e.to_string())?;
        seq.extend(
            cp.launches(&producer, scratch, Counts::rows(rows))
                .map_err(|e| e.to_string())?,
        );
        if let Some(reindex) = reindex {
            seq.extend(
                reindex
                    .launches(&compact, scratch, Counts::rows(rows))
                    .map_err(|e| e.to_string())?,
            );
        }
        calls.push(seq);
    }
    let mut planes = vec![
        scores,
        status,
        output,
        ids.ids,
        ids.row_tags,
        ids.visible_positions,
        ids.status,
    ];
    if ratio == 1 {
        planes.extend([compact_scores, compact_status, compact_output]);
    }
    let graphs = calls
        .iter()
        .map(|seq| session.capture(|_| fire(session, module, &a, seq)))
        .collect::<LabResult<Vec<_>>>()?;
    // Reuse each capture while context and live row count shrink and grow.
    for (replay, length) in [context, context.saturating_sub(9), context]
        .into_iter()
        .enumerate()
    {
        let active_rows = if replay == 1 {
            rows.saturating_sub(1)
        } else {
            rows
        };
        a.write(live_rows, &active_rows.to_le_bytes())?;
        let pos: Vec<_> = (0..n)
            .map(|r| length.saturating_sub(if rows == 64 { (n - r) as u32 } else { 1 }) as i32)
            .collect();
        a.write(positions, &i32s(pos.iter().copied()))?;
        a.write(visible, &i32s(pos.iter().map(|p| (p + 1) / ratio as i32)))?;
        let mut expected = Vec::new();
        for arm in 0..2 {
            a.poison(&planes)?;
            a.poison(&[scratches[arm]])?;
            if replay == 0 {
                fire(session, module, &a, &calls[arm])?;
            } else {
                graphs[arm]
                    .launch(&session.stream)
                    .map_err(|e| e.to_string())?;
            }
            session.stream.synchronize().map_err(|e| e.to_string())?;
            if arm == 0 {
                expected = planes
                    .iter()
                    .map(|&v| a.read(v))
                    .collect::<LabResult<Vec<_>>>()?;
            } else {
                for (i, &v) in planes.iter().enumerate() {
                    let actual = a.read(v)?;
                    compare_bytes(
                        &expected[i],
                        &actual,
                        &format!(
                            "rows={rows} context={length} bf16={bf16} ratio={ratio} replay={replay} plane={i}"
                        ),
                    )?;
                }
            }
        }
    }
    for (arm, seq) in calls.iter().enumerate() {
        // Timings are GPU events around each kernel after an untimed warmup.
        fire(session, module, &a, seq)?;
        for c in seq {
            let start = session.context.timed_event().map_err(|e| e.to_string())?;
            let end = session.context.timed_event().map_err(|e| e.to_string())?;
            start.record(&session.stream).map_err(|e| e.to_string())?;
            fire(session, module, &a, std::slice::from_ref(c))?;
            end.record(&session.stream).map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let ms = start.elapsed_ms(&end).map_err(|e| e.to_string())?;
            println!(
                "{}",
                serde_json::json!({"rows":rows,"context":context,"bf16":bf16,"ratio":ratio,"arm":arm,"kernel":c.symbol,"milliseconds":ms})
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires two allocated idle GPUs via the staged job"]
fn gpu_bitwise_matrix() {
    let result = kernel_harness::gpu_test(2 << 30, |session| {
        let root = PathBuf::from(std::env::var_os("CHOPINFER_POSITION_LIVE_CUBINS").ok_or("prebuilt live cubins required")?);
        kernel_build::report::BuildReport::read(&root.join("kernel-build-report.json")).and_then(|r|r.verify_current()).map_err(|e|e.to_string())?;
        let module = session.load_cubin(&std::fs::read(root.join(format!("{}.cubin",family::CUBIN))).map_err(|e|e.to_string())?)?;
        let bf16 = std::env::var("CHOPINFER_POSITION_LIVE_STORAGE").map_err(|e|e.to_string())? == "bf16";
        let mut count = 0;
        for (rows,context) in cases() {
            for ratio in [1,2] { paired(session,&module,rows,context,bf16,ratio)?; count += 1; }
        }
        assert_eq!(count,322);
        println!("completed {count} paired shapes; three poisoned runs each, including captured context and row-count changes");
        Ok(())
    }).unwrap();
    assert!(result.is_some(), "required GPU comparison skipped");
}

#[path = "position_live/edges.rs"]
mod edges;

fn compare_bytes(expected: &[u8], actual: &[u8], label: &str) -> LabResult<()> {
    if actual != expected {
        return Err(format!(
            "bit mismatch {label}: lengths {}/{}, byte {:?}",
            expected.len(),
            actual.len(),
            actual.iter().zip(expected).position(|(x, y)| x != y)
        ));
    }
    Ok(())
}

#[path = "position_live/arithmetic.rs"]
mod arithmetic;
