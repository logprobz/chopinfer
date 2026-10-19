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
use kernel_lab_families::position_valid as family;
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
        b.with_valid_position_candidates()
    } else {
        b
    }
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
#[path = "position_live/failures.rs"]
mod failures;

#[path = "position_prefill/edges.rs"]
mod edges;

const CHUNK_ROWS: u32 = 8192;
const SHARD_ROWS: u32 = CHUNK_ROWS / 2;
const FULL_ROWS: u32 = 64;
const OFFSETS: [u32; 4] = [0, 8192, 24576, 122880];

fn positions(offset: u32, first: u32, rows: u32) -> Vec<i32> {
    (first..first + rows)
        .map(|row| (offset + row) as i32)
        .collect()
}
fn visible(positions: &[i32], ratio: u32) -> Vec<i32> {
    positions.iter().map(|p| (p + 1) / ratio as i32).collect()
}
fn storage(bf16: bool) -> IndexerKeyFormat {
    if bf16 {
        IndexerKeyFormat::Bf16
    } else {
        IndexerKeyFormat::Fp4
    }
}
fn neg_inf(bytes: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(bytes);
    result.extend(f32::NEG_INFINITY.to_le_bytes());
    while result.len() < bytes {
        let take = result.len().min(bytes - result.len());
        result.extend_from_within(..take);
    }
    result
}

#[test]
fn cpu_prefill_causal_blocks_and_abi() {
    if let Some(root) = std::env::var_os("CHOPINFER_POSITION_LIVE_CUBINS") {
        kernel_build::report::BuildReport::read(
            &PathBuf::from(root).join("kernel-build-report.json"),
        )
        .and_then(|report| report.verify_current())
        .unwrap();
    }
    for offset in OFFSETS {
        for ratio in [1, 2] {
            let mut visited = Vec::new();
            for shard in [0, 1] {
                for first in
                    (shard * SHARD_ROWS..(shard + 1) * SHARD_ROWS).step_by(FULL_ROWS as usize)
                {
                    let p = positions(offset, first, FULL_ROWS);
                    let counts = visible(&p, ratio);
                    assert_eq!(counts[0], ((offset + first + 1) / ratio) as i32);
                    assert_eq!(counts[63], ((offset + first + 64) / ratio) as i32);
                    visited.extend(first..first + FULL_ROWS);
                }
            }
            assert_eq!(visited, (0..CHUNK_ROWS).collect::<Vec<_>>());
            let q = family::dense(ratio).query;
            let op = backend(true)
                .position_indexer(&q, 512, (1 << 20) / ratio, Need::rows(FULL_ROWS))
                .unwrap();
            assert_eq!(op.info().symbols.len(), 3);
            assert!(op.info().symbols[0].0.ends_with("_live_v1"));
        }
    }
    let q = family::compact().query;
    for bf16 in [false, true] {
        let op = backend(true)
            .position_block_indexer_with_storage(
                &q,
                storage(bf16),
                512,
                1 << 20,
                Need::rows(SHARD_ROWS),
            )
            .unwrap();
        assert!(op.info().symbols.iter().all(|(s, _)| s.contains("_valid_")));
        assert_eq!(op.info().workspace, live::workspace_bytes(SHARD_ROWS, 512));
    }
    for kept in [2048, 1024] {
        let query = family::candidates().query;
        let op = if kept == 2048 {
            backend(true).indexer_candidate_block_ids(&query, 1 << 20, Need::rows(FULL_ROWS))
        } else {
            backend(true).indexer_candidate_block_ids_runtime(
                &query,
                kept,
                1 << 20,
                Need::rows(FULL_ROWS),
            )
        }
        .unwrap();
        assert!(op.info().symbols.iter().all(|(s, _)| s.contains("_valid_")));
        let v = |handle, bytes| View { handle, bytes };
        let b = CandidateBlockIdsBuffers {
            scores: v(1, u64::from(FULL_ROWS) * (1 << 20) * 4),
            visible_positions: v(2, u64::from(FULL_ROWS) * 4),
            row_tags: v(3, u64::from(FULL_ROWS) * 8),
            source_status: v(4, u64::from(FULL_ROWS) * 4),
            live_rows: v(5, 4),
            output: CandidateBlockIds {
                ids: v(6, u64::from(FULL_ROWS) * 2048 * 4),
                row_tags: v(7, u64::from(FULL_ROWS) * 8),
                visible_positions: v(8, u64::from(FULL_ROWS) * 4),
                status: v(9, u64::from(FULL_ROWS) * 4),
                position_capacity: 1 << 20,
            },
        };
        let calls = op
            .launches(&b, v(10, op.info().workspace), Counts::rows(FULL_ROWS))
            .unwrap();
        assert_eq!(
            calls[0].plan.grid,
            [
                kernels_sm120::pooled_indexer::position_valid::CANDIDATE_CTAS,
                FULL_ROWS,
                1
            ]
        );
        assert_eq!(calls[1].plan.grid, [live::SHARDS, FULL_ROWS, 1]);
        assert_eq!(calls[2].plan.grid, [1, FULL_ROWS, 1]);
    }
    let compiled = family::cubins().remove(0);
    assert_eq!(compiled.entries.len(), 40);
    println!(
        "PREFILL_CPU rows=8192 offsets=0,8192,24576,122880 dense_rows=64 compact_rows=4096 formats=2 ratios=2 candidates=2048,1024"
    );
}

struct Stage {
    calls: [Vec<Launch>; 2],
    workspaces: [View; 2],
    outputs: Vec<View>,
    graphs: Vec<runtime_cuda::GraphExec>,
    parts: Vec<Vec<runtime_cuda::GraphExec>>,
}
impl Stage {
    fn new<B>(
        a: &mut Arena<'_>,
        module: &Module,
        ops: [kernel_core::backend::Prepared<B>; 2],
        buffers: &B,
        rows: u32,
        outputs: Vec<View>,
    ) -> LabResult<Self> {
        let mut calls = [Vec::new(), Vec::new()];
        let mut workspaces = [View::EMPTY; 2];
        for (arm, op) in ops.into_iter().enumerate() {
            workspaces[arm] = a.output((op.info().workspace as usize).max(8))?;
            calls[arm] = op
                .launches(buffers, workspaces[arm], Counts::rows(rows))
                .map_err(|e| e.to_string())?;
        }
        let graphs = calls
            .iter()
            .map(|seq| a.session.capture(|_| fire(a.session, module, a, seq)))
            .collect::<LabResult<Vec<_>>>()?;
        let parts = calls
            .iter()
            .map(|seq| {
                seq.iter()
                    .map(|call| {
                        a.session
                            .capture(|_| fire(a.session, module, a, std::slice::from_ref(call)))
                    })
                    .collect::<LabResult<Vec<_>>>()
            })
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Self {
            parts,
            calls,
            workspaces,
            outputs,
            graphs,
        })
    }
    fn run(&self, a: &Arena<'_>, arm: usize) -> LabResult<(Vec<Vec<u8>>, Vec<f64>)> {
        a.poison(&self.outputs)?;
        // NaN-filled candidate scratch proves that gather cannot read its
        // untouched capacity tail. Every consumed prefix is rewritten.
        a.write(
            self.workspaces[arm],
            &vec![0xff; self.workspaces[arm].bytes as usize],
        )?;
        let mut events = Vec::new();
        for graph in &self.parts[arm] {
            let start = a.session.context.timed_event().map_err(|e| e.to_string())?;
            let end = a.session.context.timed_event().map_err(|e| e.to_string())?;
            start.record(&a.session.stream).map_err(|e| e.to_string())?;
            graph.launch(&a.session.stream).map_err(|e| e.to_string())?;
            end.record(&a.session.stream).map_err(|e| e.to_string())?;
            events.push((start, end));
        }
        a.session.stream.synchronize().map_err(|e| e.to_string())?;
        let ms = events
            .into_iter()
            .map(|(start, end)| {
                start
                    .elapsed_ms(&end)
                    .map(f64::from)
                    .map_err(|e| e.to_string())
            })
            .collect::<LabResult<Vec<_>>>()?;
        let result = self
            .outputs
            .iter()
            .map(|&v| a.read(v))
            .collect::<LabResult<Vec<_>>>()?;
        Ok((result, ms))
    }
    fn compare(
        &self,
        a: &Arena<'_>,
        reverse: bool,
        label: &str,
        total: &mut [f64; 2],
        kernel_ms: &mut std::collections::BTreeMap<String, f64>,
    ) -> LabResult<Vec<Vec<u8>>> {
        let mut values = [Vec::new(), Vec::new()];
        for arm in if reverse { [1, 0] } else { [0, 1] } {
            let (data, ms) = self.run(a, arm)?;
            total[arm] += ms.iter().sum::<f64>();
            for (call, elapsed) in self.calls[arm].iter().zip(ms) {
                *kernel_ms.entry(call.symbol.clone()).or_default() += elapsed;
            }
            values[arm] = data;
        }
        for (i, (old, new)) in values[0].iter().zip(&values[1]).enumerate() {
            compare_bytes(old, new, &format!("{label} plane={i}"))?;
        }
        Ok(std::mem::take(&mut values[1]))
    }
    fn replay(&self, a: &Arena<'_>, expected: &[Vec<u8>], label: &str) -> LabResult<()> {
        a.poison(&self.outputs)?;
        a.write(
            self.workspaces[1],
            &vec![0xff; self.workspaces[1].bytes as usize],
        )?;
        self.graphs[1]
            .launch(&a.session.stream)
            .map_err(|e| e.to_string())?;
        a.session.stream.synchronize().map_err(|e| e.to_string())?;
        let values = self
            .outputs
            .iter()
            .map(|&v| a.read(v))
            .collect::<LabResult<Vec<_>>>()?;
        for (i, (old, new)) in expected.iter().zip(&values).enumerate() {
            compare_bytes(old, new, &format!("{label} replay plane={i}"))?;
        }
        Ok(())
    }
}
fn prepare<T>(value: Result<T, kernel_core::backend::Refused>) -> LabResult<T> {
    value.map_err(|e| e.to_string())
}
fn check_dense(data: &[Vec<u8>], p: &[i32], ratio: u32, width: u32) -> LabResult<()> {
    let padding = neg_inf(width as usize * 4);
    let page = 64 / ratio;
    for (row, count) in visible(p, ratio).into_iter().enumerate() {
        let values = &data[0][row * width as usize * 4..(row + 1) * width as usize * 4];
        let first = count as usize * 4;
        compare_bytes(
            &padding[..values.len() - first],
            &values[first..],
            "invisible dense score tail",
        )?;
        let hole = (2 * page) as usize * 4;
        if first > hole {
            let end = first.min((3 * page) as usize * 4);
            compare_bytes(
                &padding[..end - hole],
                &values[hole..end],
                "missing page scores",
            )?;
        }
        assert_eq!(&data[1][row * 4..row * 4 + 4], 0_i32.to_le_bytes());
    }
    Ok(())
}
struct Records {
    ids: Vec<u8>,
    tags: Vec<u8>,
    visible: Vec<u8>,
    status: Vec<u8>,
}
impl Records {
    fn new() -> Self {
        Self {
            ids: Vec::new(),
            tags: Vec::new(),
            visible: Vec::new(),
            status: Vec::new(),
        }
    }
    fn append(&mut self, data: &[Vec<u8>]) {
        self.ids.extend_from_slice(&data[0]);
        self.tags.extend_from_slice(&data[1]);
        self.visible.extend_from_slice(&data[2]);
        self.status.extend_from_slice(&data[3]);
    }
}
fn keys(
    a: &mut Arena<'_>,
    width: u32,
    bf16: bool,
    ratio: u32,
    operands: &arithmetic::Operands,
) -> LabResult<PositionKeyPages<View>> {
    let page = 64 / ratio;
    Ok(PositionKeyPages {
        codes: a.input(&operands.codes)?,
        scales: if bf16 {
            View::EMPTY
        } else {
            a.input(&operands.scales)?
        },
        pages: width / page,
        code_page_stride: u64::from(page) * if bf16 { 256 } else { 64 },
        scale_page_stride: if bf16 { 0 } else { u64::from(page) * 4 },
        page_tables: BlockTables {
            tables: a.input(&i32s((0..width / page).map(|p| {
                if p == 2 {
                    -1
                } else {
                    (width / page - 1 - p) as i32
                }
            })))?,
            sequences: 1,
            max_pages: width / page,
        },
    })
}

fn prefill_case(
    session: &Session,
    module: &Module,
    offset: u32,
    shard: u32,
    bf16: bool,
    ratio: u32,
) -> LabResult<()> {
    let host_start = std::time::Instant::now();
    let width = (1 << 20) / ratio;
    let first_row = shard * SHARD_ROWS;
    let mut a = Arena {
        session,
        buffers: Vec::new(),
    };
    let operands = arithmetic::Operands::new(
        FULL_ROWS as usize,
        width as usize,
        bf16,
        arithmetic::SEED ^ u64::from(offset) ^ u64::from(ratio),
    );
    let keys = keys(&mut a, width, bf16, ratio, &operands)?;
    let dense = PositionSelectBuffers {
        q: a.input(&operands.query_bytes(false))?,
        head_weights: a.input(&operands.weight_bytes(false))?,
        keys,
        positions: a.output(FULL_ROWS as usize * 4)?,
        row_batch: a.input(&i32s(vec![0; FULL_ROWS as usize]))?,
        live_rows: a.input(&FULL_ROWS.to_le_bytes())?,
        candidate_mask: None,
        position_capacity: width,
        scores: a.output(FULL_ROWS as usize * width as usize * 4)?,
        status: a.output(FULL_ROWS as usize * 4)?,
        output: a.output(FULL_ROWS as usize * 512 * 4)?,
    };
    drop(operands);
    let q = family::dense(ratio).query;
    let dense_stage = Stage::new(
        &mut a,
        module,
        [
            prepare(backend(false).position_indexer_with_storage(
                &q,
                storage(bf16),
                512,
                width,
                Need::rows(FULL_ROWS),
            ))?,
            prepare(backend(true).position_indexer_with_storage(
                &q,
                storage(bf16),
                512,
                width,
                Need::rows(FULL_ROWS),
            ))?,
        ],
        &dense,
        FULL_ROWS,
        vec![dense.scores, dense.status, dense.output],
    )?;
    let retained = CandidateBlockIds {
        ids: a.output(FULL_ROWS as usize * 2048 * 4)?,
        row_tags: a.output(FULL_ROWS as usize * 8)?,
        visible_positions: a.output(FULL_ROWS as usize * 4)?,
        status: a.output(FULL_ROWS as usize * 4)?,
        position_capacity: width,
    };
    let producer = CandidateBlockIdsBuffers {
        scores: dense.scores,
        visible_positions: a.output(FULL_ROWS as usize * 4)?,
        row_tags: a.output(FULL_ROWS as usize * 8)?,
        source_status: dense.status,
        live_rows: dense.live_rows,
        output: retained,
    };
    let mut producers = Vec::new();
    if ratio == 1 {
        for kept in [2048, 1024] {
            let ops = [false, true].map(|new| {
                if kept == 2048 {
                    backend(new).indexer_candidate_block_ids(
                        &family::candidates().query,
                        width,
                        Need::rows(FULL_ROWS),
                    )
                } else {
                    backend(new).indexer_candidate_block_ids_runtime(
                        &family::candidates().query,
                        kept,
                        width,
                        Need::rows(FULL_ROWS),
                    )
                }
            });
            let [old, new] = ops;
            producers.push(Stage::new(
                &mut a,
                module,
                [prepare(old)?, prepare(new)?],
                &producer,
                FULL_ROWS,
                vec![
                    retained.ids,
                    retained.row_tags,
                    retained.visible_positions,
                    retained.status,
                ],
            )?);
        }
    }
    let mut records = [Records::new(), Records::new()];
    let mut kernel_ms = std::collections::BTreeMap::new();
    let mut dense_ms = [0.; 2];
    let mut candidate_ms = [[0.; 2]; 2];
    let mut negative_controls = 0;
    for first in (first_row..first_row + SHARD_ROWS).step_by(FULL_ROWS as usize) {
        let p = positions(offset, first, FULL_ROWS);
        a.write(dense.positions, &i32s(p.iter().copied()))?;
        a.write(producer.visible_positions, &i32s(visible(&p, ratio)))?;
        a.write(
            producer.row_tags,
            &(first..first + FULL_ROWS)
                .flat_map(|r| ((u64::from(offset) << 32) | u64::from(r)).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let label = format!("prefill offset={offset} first={first} bf16={bf16} ratio={ratio}");
        let data = dense_stage.compare(
            &a,
            (first / FULL_ROWS) % 2 != 0,
            &label,
            &mut dense_ms,
            &mut kernel_ms,
        )?;
        check_dense(&data, &p, ratio, width)?;
        if first == first_row {
            let mut wrong = p.clone();
            wrong[0] += ratio as i32;
            a.write(dense.positions, &i32s(wrong))?;
            let (control, _) = dense_stage.run(&a, 1)?;
            assert!(compare_bytes(&data[0], &control[0], "causal row negative control").is_err());
            assert!(check_dense(&control, &p, ratio, width).is_err());
            negative_controls += 1;
            a.write(dense.positions, &i32s(p.iter().copied()))?;
            dense_stage.replay(&a, &data, &label)?;
        }
        drop(data);
        for (budget, stage) in producers.iter().enumerate() {
            let data = stage.compare(
                &a,
                (first / FULL_ROWS) % 2 != 0,
                &label,
                &mut candidate_ms[budget],
                &mut kernel_ms,
            )?;
            assert!(data[3].chunks_exact(4).all(|v| v == 0_i32.to_le_bytes()));
            if first == first_row {
                // Consume one untouched NaN block beyond row zero's live
                // range without recomputing maxima. The control must fail.
                let mut wrong = visible(&p, ratio);
                wrong[0] += 8;
                a.write(producer.visible_positions, &i32s(wrong))?;
                a.poison(&stage.outputs)?;
                fire(session, module, &a, &stage.calls[1][1..])?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                assert_eq!(&a.read(retained.status)?[..4], 1_i32.to_le_bytes());
                assert!(
                    compare_bytes(
                        &data[0],
                        &a.read(retained.ids)?,
                        "candidate tail negative control"
                    )
                    .is_err()
                );
                negative_controls += 1;
                a.write(producer.visible_positions, &i32s(visible(&p, ratio)))?;
                stage.replay(&a, &data, &label)?;
            }
            records[budget].append(&data);
        }
        println!(
            "PREFILL_BLOCK {}",
            serde_json::json!({"offset":offset,"shard":shard,"bf16":bf16,"ratio":ratio,"first":first,"rows":FULL_ROWS,"bitwise":true})
        );
    }
    let mut compact_ms = [[0.; 2]; 2];
    if ratio == 1 {
        let operands =
            arithmetic::Operands::new(SHARD_ROWS as usize, 1, bf16, arithmetic::SEED ^ 0xabc123);
        let p = positions(offset, first_row, SHARD_ROWS);
        let c = CandidateBlockIds {
            ids: a.output(SHARD_ROWS as usize * 2048 * 4)?,
            row_tags: a.output(SHARD_ROWS as usize * 8)?,
            visible_positions: a.output(SHARD_ROWS as usize * 4)?,
            status: a.output(SHARD_ROWS as usize * 4)?,
            position_capacity: width,
        };
        let compact = PositionBlockSelectBuffers {
            q: a.input(&operands.query_bytes(false))?,
            head_weights: a.input(&operands.weight_bytes(false))?,
            keys,
            positions: a.input(&i32s(p))?,
            row_batch: a.input(&i32s(vec![0; SHARD_ROWS as usize]))?,
            live_rows: a.input(&SHARD_ROWS.to_le_bytes())?,
            row_tags: a.input(
                &(first_row..first_row + SHARD_ROWS)
                    .flat_map(|r| ((u64::from(offset) << 32) | u64::from(r)).to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            candidates: c,
            scores: a.output(SHARD_ROWS as usize * 16384 * 4)?,
            status: a.output(SHARD_ROWS as usize * 4)?,
            output: a.output(SHARD_ROWS as usize * 512 * 4)?,
        };
        drop(operands);
        let q = family::compact().query;
        let stage = Stage::new(
            &mut a,
            module,
            [
                prepare(backend(false).position_block_indexer_with_storage(
                    &q,
                    storage(bf16),
                    512,
                    width,
                    Need::rows(SHARD_ROWS),
                ))?,
                prepare(backend(true).position_block_indexer_with_storage(
                    &q,
                    storage(bf16),
                    512,
                    width,
                    Need::rows(SHARD_ROWS),
                ))?,
            ],
            &compact,
            SHARD_ROWS,
            vec![compact.scores, compact.status, compact.output],
        )?;
        let padding = neg_inf(16384 * 4);
        for (budget, record) in records.into_iter().enumerate() {
            assert_eq!(record.ids.len(), SHARD_ROWS as usize * 2048 * 4);
            a.write(c.ids, &record.ids)?;
            a.write(c.row_tags, &record.tags)?;
            a.write(c.visible_positions, &record.visible)?;
            a.write(c.status, &record.status)?;
            let data = stage.compare(
                &a,
                budget % 2 != 0,
                "prefill reindex",
                &mut compact_ms[budget],
                &mut kernel_ms,
            )?;
            assert!(data[1].chunks_exact(4).all(|v| v == 0_i32.to_le_bytes()));
            for row in 0..SHARD_ROWS as usize {
                let ids = &record.ids[row * 2048 * 4..(row + 1) * 2048 * 4];
                let prefix = ids
                    .chunks_exact(4)
                    .position(|v| v == (-1_i32).to_le_bytes())
                    .unwrap_or(2048);
                let tail = row * 16384 * 4 + prefix * 8 * 4;
                compare_bytes(
                    &padding[..(2048 - prefix) * 8 * 4],
                    &data[0][tail..(row + 1) * 16384 * 4],
                    "compact padded score tail",
                )?;
            }
            stage.replay(&a, &data, "prefill compact")?;
        }
    }
    assert_eq!(negative_controls, if ratio == 1 { 3 } else { 1 });
    println!(
        "PREFILL_CASE {}",
        serde_json::json!({"offset":offset,"shard":shard,"bf16":bf16,"ratio":ratio,
        "first":first_row,"rows":SHARD_ROWS,"chunk_rows":CHUNK_ROWS,"dense_blocks":SHARD_ROWS/FULL_ROWS,
        "candidate_counts":if ratio==1 {vec![2048,1024]} else {vec![]},"compact_rows":if ratio==1 {SHARD_ROWS} else {0},
        "negative_controls":negative_controls,"bitwise":true,"kernel_ms":kernel_ms,"dense_score_select_ms":dense_ms,
        "candidate_ms":candidate_ms,"compact_score_select_ms":compact_ms,"wall_ms":host_start.elapsed().as_secs_f64()*1000.})
    );
    Ok(())
}

#[test]
#[ignore = "requires the staged two-GPU prefill job"]
fn gpu_prefill_chunks() {
    let result = kernel_harness::gpu_test(4 << 30, |session| {
        let root = PathBuf::from(
            std::env::var_os("CHOPINFER_POSITION_LIVE_CUBINS")
                .ok_or("prebuilt prefill cubins required")?,
        );
        kernel_build::report::BuildReport::read(&root.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        let module = session.load_cubin(
            &std::fs::read(root.join(format!("{}.cubin", family::CUBIN)))
                .map_err(|e| e.to_string())?,
        )?;
        let shard: u32 = std::env::var("CHOPINFER_PREFILL_SHARD")
            .map_err(|e| e.to_string())?
            .parse()
            .map_err(|_| "invalid prefill shard")?;
        assert!(shard < 2);
        let mut cases = 0;
        for bf16 in [false, true] {
            for offset in OFFSETS {
                for ratio in [1, 2] {
                    prefill_case(session, &module, offset, shard, bf16, ratio)?;
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 16);
        println!("PREFILL_COMPLETE shard={shard} cases={cases} rows_per_case={SHARD_ROWS}");
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "required prefill comparison skipped");
}
