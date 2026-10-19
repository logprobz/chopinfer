//! Bounded-stride eager batches against the guarded 64-row serving schedule.
//! Serving keeps its existing schedule until this pair has GPU evidence.
use super::*;
use kernel_core::position_blocks::{CandidateBlockIds, CandidateBlockIdsBuffers};
use kernel_harness::{Cubins, eviction::EvictionMode};
use model_deepseek41_contract::indexer::prefill::{Batch, SCORE_BYTES, batches};
use runtime_cuda::{DeviceBuffer, KernelArgument, Module};

const POISON: u32 = 0xa5a5a5a5;
fn backend() -> Sm120 {
    super::backend(false).with_guarded_position_scores()
}
fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_INDEX_PREFILL_ARTIFACTS")
        .expect("staged prefill artifacts")
        .into()
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    layer: u32,
    context: u32,
    rows: u32,
    case: u32,
}
fn shapes() -> Vec<Shape> {
    let mut out = Vec::new();
    for layer in [2, 20] {
        for context in [8192, 32768, 131072] {
            // 2496 is the actual layer-20 upper-skip suffix of the cold 8K
            // profile. 1984/448 are later suffix sizes; 8191/65 are partials.
            for rows in [8192, 8191, 4096, 2496, 1984, 448, 65] {
                out.push(Shape {
                    layer,
                    context,
                    rows,
                    case: 0,
                });
            }
        }
        for case in 1..=6 {
            out.push(Shape {
                layer,
                context: 8192,
                rows: 65,
                case,
            });
        }
    }
    out
}
fn plan(s: Shape, positions: &[i32]) -> Vec<Batch> {
    batches(
        positions,
        FLASH.compress_ratio(s.layer),
        indexer::position_capacity(&FLASH, s.layer).unwrap(),
        SCORE_BYTES,
        if s.layer == 20 { 512 } else { 4096 },
    )
    .unwrap()
}

#[test]
fn cpu_prefill_plans_fit_existing_allocations() {
    let compact = indexer::block_selection(&FLASH, 24, 4096).unwrap();
    let available = backend()
        .position_block_indexer(&compact, 512, 1 << 20, Need::rows(4096))
        .unwrap()
        .info()
        .workspace;
    assert_eq!(shapes().len(), 54);
    for s in shapes() {
        let positions: Vec<_> = (s.context - s.rows..s.context).map(|x| x as i32).collect();
        for batch in plan(s, &positions) {
            let n = batch.rows.end - batch.rows.start;
            let query = indexer::selection(&FLASH, s.layer, n).unwrap();
            let b = cpu_buffers(n, FLASH.compress_ratio(s.layer), batch.width);
            let op = backend()
                .position_indexer(&query, 512, batch.width, Need::rows(n))
                .unwrap();
            assert!(op.info().workspace <= available);
            assert!(b.scores.bytes <= SCORE_BYTES);
            let calls = op.launches(&b, v(99, available), Counts::rows(n)).unwrap();
            assert_eq!(calls.len(), 4);
            assert_eq!(calls[1].plan.grid, [192, n, 1]);
            assert_eq!(calls[3].plan.grid, [1, n, 1]);
            assert!(calls[1].symbol.ends_with("_mma_fp4_guard_v1"));
            if s.layer == 20 {
                let producer = backend()
                    .indexer_candidate_block_ids_runtime(
                        &indexer::candidates(&FLASH, n),
                        if s.case == 4 {
                            31
                        } else if s.case == 5 {
                            1
                        } else {
                            2048
                        },
                        batch.width,
                        Need::rows(n),
                    )
                    .unwrap();
                assert!(producer.info().workspace <= available);
                let c = CandidateBlockIdsBuffers {
                    scores: b.scores,
                    visible_positions: v(20, u64::from(n) * 4),
                    row_tags: v(21, u64::from(n) * 8),
                    source_status: b.status,
                    live_rows: b.live_rows,
                    output: CandidateBlockIds {
                        ids: v(22, u64::from(n) * 8192),
                        row_tags: v(23, u64::from(n) * 8),
                        visible_positions: v(24, u64::from(n) * 4),
                        status: v(25, u64::from(n) * 4),
                        position_capacity: batch.width,
                    },
                };
                assert_eq!(
                    producer
                        .launches(&c, v(99, available), Counts::rows(n))
                        .unwrap()
                        .len(),
                    3
                );
            }
        }
    }
}

struct Memory<'a> {
    session: &'a Session,
    buffers: Vec<DeviceBuffer>,
    views: Vec<(usize, usize, usize)>,
}
impl Memory<'_> {
    fn alloc(&mut self, bytes: usize) -> LabResult<View> {
        self.buffers.push(self.session.allocate(bytes)?);
        self.views.push((self.buffers.len() - 1, 0, bytes));
        Ok(v(self.views.len() as u64, bytes as u64))
    }
    fn input(&mut self, data: &[u8]) -> LabResult<View> {
        let view = self.alloc(data.len())?;
        self.buffers
            .last()
            .unwrap()
            .copy_from_bytes(data, &self.session.stream)
            .map_err(|e| e.to_string())?;
        Ok(view)
    }
    fn slice(&mut self, view: View, first: u64, bytes: u64) -> View {
        assert!(first + bytes <= view.bytes && bytes > 0);
        let (buffer, offset, _) = self.views[view.handle as usize - 1];
        self.views
            .push((buffer, offset + first as usize, bytes as usize));
        v(self.views.len() as u64, bytes)
    }
    fn row(&mut self, view: View, row_bytes: u64, range: std::ops::Range<u32>) -> View {
        self.slice(
            view,
            u64::from(range.start) * row_bytes,
            u64::from(range.end - range.start) * row_bytes,
        )
    }
    fn argument(&self, view: View) -> KernelArgument<'_> {
        let (buffer, offset, len) = self.views[view.handle as usize - 1];
        KernelArgument::Range {
            buffer: &self.buffers[buffer],
            offset,
            byte_len: len,
        }
    }
    fn read(&self, view: View) -> LabResult<Vec<u8>> {
        let (buffer, offset, len) = self.views[view.handle as usize - 1];
        let mut data = vec![0; len];
        self.buffers[buffer]
            .copy_to_bytes_at_async(offset, &mut data, &self.session.stream)
            .map_err(|e| e.to_string())?;
        self.session
            .stream
            .synchronize()
            .map_err(|e| e.to_string())?;
        Ok(data)
    }
    fn fill(&self, module: &Module, view: View, value: u32) -> LabResult<()> {
        assert!(view.bytes.is_multiple_of(4));
        let f = module
            .function("chop_index_prefill_fill_v1")
            .map_err(|e| e.to_string())?;
        f.launch(
            &self.session.stream,
            kernel_harness::launch::simple_config((192, 1, 1), (256, 1, 1), 0, (1, 1, 1), false)?,
            &[
                self.argument(view),
                KernelArgument::U64(view.bytes / 4),
                KernelArgument::U32(value),
            ],
        )
        .map_err(|e| e.to_string())
    }
    fn fire(&self, modules: &BTreeMap<String, Module>, calls: &[Launch]) -> LabResult<()> {
        for call in calls {
            let (name, _) = bundle_deepseek41::sm120::fold_cubin_for_symbol(&call.symbol)
                .ok_or("missing serving symbol")?;
            let f = modules[name]
                .function(&call.symbol)
                .map_err(|e| e.to_string())?;
            let args = call
                .arguments
                .0
                .iter()
                .map(|a| match a {
                    Argument::DeviceBuffer(handle) => self.argument(v(*handle, 0)),
                    Argument::U32(x) => KernelArgument::U32(*x),
                    Argument::U64(x) => KernelArgument::U64(*x),
                    Argument::I32(x) => KernelArgument::I32(*x),
                    Argument::NullDevicePointer => KernelArgument::NullBuffer,
                    _ => panic!("unexpected indexer argument"),
                })
                .collect::<Vec<_>>();
            kernel_harness::launch::launch_with(
                &f,
                &self.session.stream,
                kernel_harness::launch::config(call.plan)?,
                &args,
            )?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn compare(
        &self,
        probe: &Module,
        old: View,
        now: View,
        counts: View,
        rows: u32,
        live: u32,
        old_width: u32,
        width: u32,
        first: u32,
        counter: u32,
    ) -> LabResult<()> {
        assert!(old.bytes >= u64::from(rows) * u64::from(old_width) * 4);
        assert!(now.bytes >= u64::from(first + rows) * u64::from(width) * 4);
        let f = probe
            .function("chop_index_prefill_compare_v1")
            .map_err(|e| e.to_string())?;
        f.launch(
            &self.session.stream,
            kernel_harness::launch::simple_config((32, rows, 1), (256, 1, 1), 0, (1, 1, 1), false)?,
            &[
                self.argument(old),
                self.argument(now),
                self.argument(counts),
                KernelArgument::U32(rows),
                KernelArgument::U32(live),
                KernelArgument::U32(old_width),
                KernelArgument::U32(width),
                KernelArgument::U32(first),
                KernelArgument::U32(counter),
            ],
        )
        .map_err(|e| e.to_string())
    }
}

struct Outputs {
    scores: View,
    status: View,
    ids: View,
    candidates: CandidateBlockIds<View>,
}
impl Outputs {
    fn allocate(m: &mut Memory<'_>, rows: u32, score_bytes: u64, capacity: u32) -> LabResult<Self> {
        let n = u64::from(rows);
        Ok(Self {
            scores: m.alloc(score_bytes as usize)?,
            status: m.alloc((n * 4) as usize)?,
            ids: m.alloc((n * 512 * 4) as usize)?,
            candidates: CandidateBlockIds {
                ids: m.alloc((n * 2048 * 4) as usize)?,
                row_tags: m.alloc((n * 8) as usize)?,
                visible_positions: m.alloc((n * 4) as usize)?,
                status: m.alloc((n * 4) as usize)?,
                position_capacity: capacity,
            },
        })
    }
    fn poison(&self, m: &Memory<'_>, probe: &Module) -> LabResult<()> {
        for v in [
            self.scores,
            self.status,
            self.ids,
            self.candidates.ids,
            self.candidates.row_tags,
            self.candidates.visible_positions,
            self.candidates.status,
        ] {
            m.fill(probe, v, POISON)?;
        }
        Ok(())
    }
}

struct Inputs {
    b: PositionSelectBuffers<View>,
    tags: View,
    visible: View,
}
fn fixture(m: &mut Memory<'_>, s: Shape) -> LabResult<(Inputs, Vec<i32>)> {
    let ratio = FLASH.compress_ratio(s.layer);
    let page = 64 / ratio;
    let capacity = indexer::position_capacity(&FLASH, s.layer).unwrap();
    let key_width = s.context.div_ceil(64) * 64 / ratio;
    let mut seed = 0x5e79a123_u32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let mut q = Vec::with_capacity(s.rows as usize * 8192);
    for row in 0..s.rows {
        for _ in 0..32 {
            for group in 0..4 {
                for lane in 0..32 {
                    let r = rng();
                    let mut x = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(r & 7) as usize]
                        * if r & 8 == 0 { 1. } else { -1. };
                    if lane == 31 {
                        x = 6.;
                    }
                    // Regular rows stay in the exact MMA domain; case 2 exercises repair.
                    let exponent = if s.case == 2 {
                        [0, 24, 0, -24][group]
                    } else {
                        group as i32 - 2
                    };
                    x *= 2_f32.powi(exponent);
                    if row % 19 == 0 {
                        x = 0.;
                    } // Exact ties, ordered by original position.
                    q.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
                }
            }
        }
    }
    let mut weights: Vec<_> = (0..s.rows * 32)
        .map(|_| ((rng() % 65) as f32 - 32.) / 32.)
        .collect();
    if s.case == 3 {
        weights[32] = f32::NAN;
        weights[64] = f32::INFINITY;
        weights[96] = f32::MAX;
    }
    let codes: Vec<_> = (0..key_width * 64).map(|_| rng() as u8).collect();
    let mut scales: Vec<_> = (0..key_width * 4).map(|i| 125 + (i % 4) as u8).collect();
    if s.case == 3 {
        scales[0] = 255;
        scales[4] = 252;
    }
    let mut positions: Vec<_> = (s.context - s.rows..s.context).map(|p| p as i32).collect();
    if s.case == 1 {
        positions.reverse();
        positions[0] = -1;
        positions[1] = 0;
        positions[2] = 1;
        positions[3] = 63;
        positions[4] = 64;
    }
    if s.case == 6 {
        positions.fill(-1);
        positions[1] = 0;
        positions[2] = 1;
    }
    let mut tables = vec![-1; (capacity / page) as usize * 2];
    for batch in 0..2 {
        for i in 0..key_width / page {
            if i % 17 != 5 {
                tables[(batch * capacity / page + i) as usize] = if batch == 0 {
                    i as i32
                } else {
                    (key_width / page - 1 - i) as i32
                };
            }
        }
    }
    let b = PositionSelectBuffers {
        q: m.input(&q)?,
        head_weights: m.input(
            &weights
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?,
        keys: PositionKeyPages {
            codes: m.input(&codes)?,
            scales: m.input(&scales)?,
            pages: key_width / page,
            code_page_stride: u64::from(page) * 64,
            scale_page_stride: u64::from(page) * 4,
            page_tables: BlockTables {
                tables: m.input(&ints(tables))?,
                sequences: 2,
                max_pages: capacity / page,
            },
        },
        positions: m.input(&ints(positions.clone()))?,
        row_batch: m.input(&ints((0..s.rows).map(|i| (i % 2) as i32)))?,
        live_rows: View::EMPTY,
        candidate_mask: None,
        position_capacity: capacity,
        scores: View::EMPTY,
        status: View::EMPTY,
        output: View::EMPTY,
    };
    let tags = m.input(
        &(0..s.rows)
            .flat_map(|r| ((0x1234_u64 << 32) + u64::from(s.context - s.rows + r)).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let visible = m.input(&ints(positions.iter().map(|&p| {
        if p < 0 {
            0
        } else {
            ((p as u64 + 1) / u64::from(ratio)) as i32
        }
    })))?;
    Ok((Inputs { b, tags, visible }, positions))
}

struct Schedule {
    calls: Vec<Launch>,
    score_end: usize,
    select_end: usize,
    live: u32,
    rows: std::ops::Range<u32>,
}
#[allow(clippy::too_many_arguments)]
fn schedule(
    m: &mut Memory<'_>,
    s: Shape,
    input: &Inputs,
    out: &Outputs,
    batch: &Batch,
    scratch: View,
    live: u32,
    output_first: u32,
) -> LabResult<Schedule> {
    let range = batch.rows.clone();
    let n = range.end - range.start;
    let out_range = output_first..output_first + n;
    let mut b = input.b;
    b.q = m.row(b.q, 8192, range.clone());
    b.head_weights = m.row(b.head_weights, 128, range.clone());
    b.positions = m.row(b.positions, 4, range.clone());
    b.row_batch = m.row(b.row_batch, 4, range.clone());
    let nlive = live.saturating_sub(range.start).min(n);
    b.live_rows = m.input(&nlive.to_le_bytes())?;
    b.position_capacity = batch.width;
    b.scores = out.scores;
    b.status = m.row(out.status, 4, out_range.clone());
    b.output = m.row(out.ids, 2048, out_range.clone());
    let query = indexer::selection(&FLASH, s.layer, n).unwrap();
    let top = if s.case == 4 {
        31
    } else if matches!(s.case, 5 | 6) {
        1
    } else {
        512
    };
    let op = backend()
        .position_indexer(&query, top, batch.width, Need::rows(n))
        .unwrap();
    assert!(op.info().workspace <= scratch.bytes);
    let mut calls = op
        .launches(&b, scratch, Counts::rows(n))
        .map_err(|e| e.to_string())?;
    let score_end = 2;
    let select_end = calls.len();
    if s.layer == 20 {
        let candidate_k = if s.case == 4 {
            31
        } else if matches!(s.case, 5 | 6) {
            1
        } else {
            2048
        };
        let query = indexer::candidates(&FLASH, n);
        let producer = if candidate_k == query.top_k_blocks {
            backend().indexer_candidate_block_ids(&query, batch.width, Need::rows(n))
        } else {
            backend().indexer_candidate_block_ids_runtime(
                &query,
                candidate_k,
                batch.width,
                Need::rows(n),
            )
        }
        .unwrap();
        assert!(producer.info().workspace <= scratch.bytes);
        let c = out.candidates;
        let output = CandidateBlockIds {
            ids: m.row(c.ids, 8192, out_range.clone()),
            row_tags: m.row(c.row_tags, 8, out_range.clone()),
            visible_positions: m.row(c.visible_positions, 4, out_range.clone()),
            status: m.row(c.status, 4, out_range),
            position_capacity: batch.width,
        };
        calls.extend(
            producer
                .launches(
                    &CandidateBlockIdsBuffers {
                        scores: b.scores,
                        visible_positions: m.row(input.visible, 4, range.clone()),
                        row_tags: m.row(input.tags, 8, range.clone()),
                        source_status: b.status,
                        live_rows: b.live_rows,
                        output,
                    },
                    scratch,
                    Counts::rows(n),
                )
                .map_err(|e| e.to_string())?,
        );
    }
    Ok(Schedule {
        calls,
        score_end,
        select_end,
        live: nlive,
        rows: range,
    })
}

fn test_cubins() -> Cubins {
    let data = Box::leak(
        bytes(&root().join("probe_sm120.cubin"))
            .unwrap()
            .into_boxed_slice(),
    );
    Cubins::new(
        Box::leak(Box::new([("probe_sm120", &data[..])])),
        &[(
            "probe_sm120",
            &[
                "chop_probe_gate_v1",
                "chop_probe_flush_v1",
                "chop_probe_read_v1",
            ],
        )],
    )
}

#[test]
#[ignore = "requires CHOPINFER_INDEX_PREFILL_ARTIFACTS and an idle GPU pair"]
fn gpu_batched_prefill_bitwise_and_timing() {
    let ran=kernel_harness::gpu_test_with(test_cubins(),8<<30,|session| {
        let shard:usize=std::env::var("CHOPINFER_L4_SHARD").unwrap().parse().unwrap();assert!(shard<2);
        let probe=session.load_cubin(&bytes(&root().join("indexer_prefill_probe.cubin"))?)?;
        let mut modules=BTreeMap::new();let mut completed=0;
        for (id,s) in shapes().into_iter().enumerate().filter(|(i,_)|i%2==shard) {
            let mut m=Memory{session,buffers:vec![],views:vec![]};
            let (input,positions)=fixture(&mut m,s)?;let batches=plan(s,&positions);
            let capacity=input.b.position_capacity;
            let compact=indexer::block_selection(&FLASH,24,4096).unwrap();
            let scratch_bytes=backend().position_block_indexer(&compact,512,1<<20,Need::rows(4096)).unwrap().info().workspace;
            let old_scratch=m.alloc(scratch_bytes as usize)?;let new_scratch=m.alloc(scratch_bytes as usize)?;
            let old=Outputs::allocate(&mut m,64,64*u64::from(capacity)*4,capacity)?;
            let max_rows=batches.iter().map(|b|b.rows.end-b.rows.start).max().unwrap();
            let width=batches[0].width;
            let now=Outputs::allocate(&mut m,max_rows,u64::from(max_rows)*u64::from(width)*4,width)?;
            let counters=m.input(&[0;32])?;
            // Inactive launch tails remain poisoned in both schedules.
            let live=if s.case==5 {0}else if s.case==4{s.rows-1}else{s.rows};
            let mut old_calls=Vec::new();let mut new_calls=Vec::new();
            for batch in &batches {
                let next=schedule(&mut m,s,&input,&now,batch,new_scratch,live,0)?;
                for call in &next.calls { let (name,data)=bundle_deepseek41::sm120::fold_cubin_for_symbol(&call.symbol).ok_or("missing served cubin")?;
                    if !modules.contains_key(name) {modules.insert(name.into(),session.load_cubin(data)?);} }
                now.poison(&m,&probe)?;m.fill(&probe,new_scratch,POISON)?;
                let next_graph=session.capture(|_|m.fire(&modules,&next.calls))?;
                next_graph.launch(&session.stream).map_err(|e|e.to_string())?;
                for first in (batch.rows.start..batch.rows.end).step_by(64) {
                    let previous=schedule(&mut m,s,&input,&old,&Batch{rows:first..(first+64).min(batch.rows.end),width:capacity},old_scratch,live,0)?;
                    old.poison(&m,&probe)?;m.fill(&probe,old_scratch,POISON)?;
                    m.fire(&modules,&previous.calls)?;
                    let n=previous.rows.end-first;let target=first-batch.rows.start;
                    m.compare(&probe,old.scores,now.scores,counters,n,previous.live,capacity,width,target,0)?;
                    for (a,b,w,channel) in [(old.ids,now.ids,512,1),(old.status,now.status,1,2)] {
                        m.compare(&probe,a,b,counters,n,previous.live,w,w,target,channel)?;
                    }
                    if s.layer==20 {
                        for (a,b,w,channel) in [(old.candidates.ids,now.candidates.ids,2048,3),(old.candidates.row_tags,now.candidates.row_tags,2,4),
                            (old.candidates.visible_positions,now.candidates.visible_positions,1,5),(old.candidates.status,now.candidates.status,1,6)] {
                            m.compare(&probe,a,b,counters,n,previous.live,w,w,target,channel)?;
                        }
                    }
                    old_calls.push(previous);
                }
                new_calls.push(next);
            }
            let counts=m.read(counters)?;
            assert_eq!(counts,[0;32],"batch mismatch {s:?}: {counts:?}");
            // A one-bit corruption of an ordered ID must fail the same device
            // comparator. Use the last old block and its last new batch.
            let last=old_calls.last().unwrap();let last_new=new_calls.last().unwrap();
            let row=last.rows.start-last_new.rows.start;
            let target=m.slice(now.ids,u64::from(row)*2048,4);
            let old_first=m.slice(old.ids,0,4);
            let a=m.read(old_first)?;
            let wrong=u32::from_le_bytes(a.try_into().unwrap())^1;
            m.fill(&probe,target,wrong)?;m.fill(&probe,counters,0)?;
            m.compare(&probe,old.ids,now.ids,counters,1,1,512,512,row,1)?;
            let negative=m.read(counters)?;
            assert_eq!(u32::from_le_bytes(negative[4..8].try_into().unwrap()),1,"negative control was not detected");
            m.fire(&modules,&last_new.calls)?;
            println!("PREFILL_BATCH_CASE {}",serde_json::json!({"id":id,"layer":s.layer,"context":s.context,"rows":s.rows,"case":s.case,
                "live":live,"width":width,"old_batches":old_calls.len(),"new_batches":new_calls.len(),"mismatches":0,"negative_control_rejected":true,
                "candidate_planes":s.layer==20,"score_bytes":now.scores.bytes,"workspace_bytes":scratch_bytes}));
            if s.case==0 {
                // Complete schedules, not a sum of overlapping kernel times.
                // Prefix intervals always include the producers of their
                // scratch. A selector alone would read the last batch's scores
                // for earlier rows and would not measure the claimed work.
                for stage in ["all","pack_score","pack_score_select"] {
                    if stage=="pack_score_select" && s.layer!=20 {continue;}
                    let arms=[&old_calls,&new_calls];
                    let graphs=arms.iter().map(|list|session.capture(|_| {
                        for op in *list {
                            let range=match stage {"pack_score"=>0..op.score_end,"pack_score_select"=>0..op.select_end,_=>0..op.calls.len()};
                            m.fire(&modules,&op.calls[range])?;
                        }
                        Ok(())
                    })).collect::<LabResult<Vec<_>>>()?;
                    let timing=kernel_harness::with_timing_label(&format!("prefill-batch-{id}-{stage}"),||kernel_harness::time_ab(session,1,3,[EvictionMode::Write;2],|arm,stream|graphs[arm].launch(stream).map_err(|e|e.to_string())))?;
                    println!("PREFILL_BATCH_TIMING {}",serde_json::json!({"id":id,"stage":stage,"median_ms":timing.arms.map(|a|a.median_ms),"paired_ratio":timing.paired_ratio,"exclusive":true}));
                }
            }
            session.receipt("indexer-prefill-batch",BTreeMap::from([
                ("id".into(),serde_json::json!(id)),("mismatches".into(),serde_json::json!(0)),
                ("execution_mode".into(),serde_json::json!("guarded-baseline-eager_candidate-graph")),
            ]))?;
            completed+=1;
        }
        println!("PREFILL_BATCH_COMPLETE shard={shard} shapes={completed} total={}",shapes().len());
        Ok(())
    }).unwrap();
    assert!(ran.is_some());
}
