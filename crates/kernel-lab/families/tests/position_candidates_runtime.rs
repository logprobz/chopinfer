//! Exact candidate membership, compiled-count equality and poisoned replay.
use kernel_core::{
    backend::{Backend, Counts, Need, View},
    position_blocks::{CandidateBlockIds, CandidateBlockIdsBuffers},
    position_blocks_oracle::candidate_ids,
};
use kernel_harness::{
    LabResult, Session,
    launch::{Bindings, launch},
};
use kernel_lab_families::position_candidates_runtime as family;
use kernels_sm120::backend::Sm120;
use runtime_cuda::{DeviceBuffer, Module};
use std::{collections::BTreeMap, path::PathBuf};

fn backend() -> Sm120 {
    Sm120::new()
        .with_indexer_candidate_block_ids([family::compiled()])
        .with_indexer_candidate_block_ids_runtime([family::compiled()])
}

#[test]
fn runtime_candidates_prepare_with_a_fixed_output_stride() {
    let backend = backend();
    let q = family::compiled().query;
    let runtime =
        kernels_sm120::pooled_indexer::position_candidates_runtime::entry(&family::compiled());
    for kept in [1, 2, 65, 1024, 2048] {
        let op = backend
            .indexer_candidate_block_ids_runtime(&q, kept, 16385, Need::rows(8))
            .unwrap();
        assert_eq!(
            op.info().symbols,
            vec![(runtime.score_symbol(), 0), (runtime.gather_symbol(), 0)]
        );
        assert!(
            op.info()
                .bytes
                .contains(&("output.ids".into(), 8 * 2048 * 4))
        );
    }
    for kept in [0, 2049, u32::MAX] {
        assert!(
            backend
                .indexer_candidate_block_ids_runtime(&q, kept, 16385, Need::rows(8))
                .is_err()
        );
    }
    let compiled = backend
        .indexer_candidate_block_ids(&q, 16385, Need::rows(8))
        .unwrap();
    assert_eq!(
        compiled.info().symbols,
        vec![
            (family::compiled().score_symbol(), 0),
            (family::compiled().gather_symbol(), 0)
        ]
    );
}

struct Arena<'a> {
    session: &'a Session,
    buffers: Vec<DeviceBuffer>,
}
impl<'a> Arena<'a> {
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
    fn read(&self, view: View) -> LabResult<Vec<u8>> {
        kernel_harness::poison::read(self.session, &self.buffers[view.handle as usize - 1])
    }
    fn poison(&self, views: &[View]) -> LabResult<()> {
        for view in views {
            self.buffers[view.handle as usize - 1]
                .copy_from_bytes(&vec![0xa5; view.bytes as usize], &self.session.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    session: &Session,
    module: &Module,
    width: u32,
    kept: u32,
    live: u32,
    pattern: u32,
    runtime: bool,
) -> LabResult<Vec<Vec<u8>>> {
    let q = family::compiled().query;
    let backend = backend();
    let op = if runtime {
        backend.indexer_candidate_block_ids_runtime(&q, kept, width, Need::rows(8))
    } else {
        backend.indexer_candidate_block_ids(&q, width, Need::rows(8))
    }
    .map_err(|e| e.to_string())?;
    let mut arena = Arena {
        session,
        buffers: Vec::new(),
    };
    let visible: [i32; 8] = [
        0,
        1,
        width.min(8) as i32,
        width.min(17) as i32,
        width as i32,
        width as i32,
        -1,
        width as i32 + 1,
    ];
    let tags: [u64; 8] = [1, 9, 13, 1024, 1 << 32, u64::MAX, 71, 81];
    let source: [i32; 8] = [0, 0, 0, 0, 0, 1, 0, 0];
    let mut scores = (0..8 * width as usize)
        .map(|i| match pattern {
            0 => {
                if i.is_multiple_of(2) {
                    0.0_f32
                } else {
                    -0.0
                }
            }
            2 => f32::NEG_INFINITY,
            _ => ((i * 17 % 31) as f32 - 15.0) / 8.0,
        })
        .collect::<Vec<_>>();
    scores[3 * width as usize] = if pattern == 2 {
        f32::INFINITY
    } else {
        f32::NAN
    };
    if width > 1 {
        scores[width as usize + 1] = f32::INFINITY;
    }
    let output = CandidateBlockIds {
        ids: arena.input(&vec![0xa5; 8 * 2048 * 4])?,
        row_tags: arena.input(&[0xa5; 8 * 8])?,
        visible_positions: arena.input(&[0xa5; 8 * 4])?,
        status: arena.input(&[0xa5; 8 * 4])?,
        position_capacity: width,
    };
    let buffers = CandidateBlockIdsBuffers {
        scores: arena.input(
            &scores
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        visible_positions: arena.input(
            &visible
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        row_tags: arena.input(
            &tags
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        source_status: arena.input(
            &source
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        live_rows: arena.input(&live.to_le_bytes())?,
        output,
    };
    let workspace = arena.input(&vec![0xa5; op.info().workspace as usize])?;
    let launches = op
        .launches(&buffers, workspace, Counts::rows(8))
        .map_err(|e| e.to_string())?;
    let bindings = Bindings {
        buffers: arena
            .buffers
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u64 + 1, b))
            .collect(),
    };
    let functions = launches
        .iter()
        .map(|l| module.function(&l.symbol).map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    let fire = |stream: &runtime_cuda::Stream| {
        for (l, f) in launches.iter().zip(&functions) {
            launch(&session.context, f, stream, l.plan, &l.arguments, &bindings)?;
        }
        Ok(())
    };
    let planes = [
        output.ids,
        output.row_tags,
        output.visible_positions,
        output.status,
    ];
    let graph = session.capture(fire)?;
    let mut expected = vec![
        vec![0xa5; 8 * 2048 * 4],
        vec![0xa5; 8 * 8],
        vec![0xa5; 8 * 4],
        vec![0xa5; 8 * 4],
    ];
    for row in 0..live as usize {
        let values = &scores[row * width as usize..(row + 1) * width as usize];
        let oracle = if source[row] != 0 || visible[row] < 0 || visible[row] > width as i32 {
            None
        } else {
            candidate_ids(
                kernel_core::pooled_indexer::CandidateBlocksQuery {
                    top_k_blocks: kept,
                    ..q
                },
                values,
                visible[row] as usize,
            )
            .ok()
        };
        let failed = oracle.is_none();
        let mut ids = oracle.unwrap_or_default();
        ids.resize(2048, -1);
        expected[0][row * 2048 * 4..(row + 1) * 2048 * 4]
            .copy_from_slice(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        expected[1][row * 8..(row + 1) * 8].copy_from_slice(&tags[row].to_le_bytes());
        expected[2][row * 4..(row + 1) * 4]
            .copy_from_slice(&visible[row].clamp(0, width as i32).to_le_bytes());
        expected[3][row * 4..(row + 1) * 4].copy_from_slice(&i32::from(failed).to_le_bytes());
    }
    for mode in [0, 1, 2] {
        arena.poison(&planes)?;
        if mode == 2 {
            graph.launch(&session.stream).map_err(|e| e.to_string())?;
        } else {
            fire(&session.stream)?;
        }
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let actual = planes
            .iter()
            .map(|&v| arena.read(v))
            .collect::<LabResult<Vec<_>>>()?;
        assert_eq!(
            actual, expected,
            "width {width}, kept {kept}, live {live}, pattern {pattern}, runtime {runtime}, mode {mode}"
        );
    }
    planes.iter().map(|&v| arena.read(v)).collect()
}

#[test]
fn runtime_candidates_match_compiled_and_oracle_on_gpu() {
    let result = kernel_harness::gpu_test(256 << 20, |session| {
        let root = PathBuf::from(
            std::env::var_os("CHOPINFER_POSITION_CANDIDATES_RUNTIME_CUBINS")
                .ok_or("prebuilt runtime candidate cubins required")?,
        );
        kernel_build::report::BuildReport::read(&root.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        let module = session.load_cubin(
            &std::fs::read(root.join(format!("{}.cubin", family::CUBIN)))
                .map_err(|e| e.to_string())?,
        )?;
        let mut cases = 0;
        let mut equal_at_max = 0;
        for width in [1, 9, 16385, 1 << 20] {
            for live in [7, 8] {
                for pattern in [0, 1, 2] {
                    for kept in [1, 2, 65, 1024, 2048] {
                        let observed = run(session, &module, width, kept, live, pattern, true)?;
                        cases += 1;
                        if kept == 2048 {
                            assert_eq!(
                                observed,
                                run(session, &module, width, kept, live, pattern, false)?
                            );
                            equal_at_max += 1;
                        }
                    }
                }
            }
        }
        assert_eq!((cases, equal_at_max), (120, 24));
        session.receipt(
            "runtime-candidates",
            BTreeMap::from([
                ("oracle_cases".into(), serde_json::json!(cases)),
                (
                    "compiled_equal_at_2048".into(),
                    serde_json::json!(equal_at_max),
                ),
                ("replays_per_case".into(), serde_json::json!(3)),
            ]),
        )?;
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(
            result.is_some(),
            "required runtime candidate conformance skipped"
        );
    }
}
