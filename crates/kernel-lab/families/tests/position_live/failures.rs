//! Poisoned dense and compact comparisons for visibility and error handling.
use super::*;
use kernel_core::position_blocks_oracle::validate_ids;

const WIDTH: u32 = 1024;
const ROWS: u32 = 2;
const KEPT: usize = 2048;
const SLOTS: usize = KEPT * 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fault {
    None,
    StaleTag,
    StaleVisible,
    SourceStatus,
    Duplicate,
    Unsorted,
    Negative,
    OutOfRange,
    TailId,
    MissingPin,
    QueryNan,
    QueryInf,
    WeightNan,
    WeightInf,
    DotOverflow,
    HeadOverflow,
    ScoreNan,
    ScoreInf,
    ScoreNegInf,
    HiddenNan,
}
impl Fault {
    fn malformed(self) -> bool {
        matches!(
            self,
            Self::StaleTag
                | Self::StaleVisible
                | Self::SourceStatus
                | Self::Duplicate
                | Self::Unsorted
                | Self::Negative
                | Self::OutOfRange
                | Self::TailId
                | Self::MissingPin
        )
    }
    fn numeric(self) -> bool {
        matches!(
            self,
            Self::QueryNan
                | Self::QueryInf
                | Self::WeightNan
                | Self::WeightInf
                | Self::DotOverflow
                | Self::HeadOverflow
        )
    }
    fn injected(self) -> Option<f32> {
        match self {
            Self::ScoreNan | Self::HiddenNan => Some(f32::from_bits(0x7fc01234)),
            Self::ScoreInf => Some(f32::INFINITY),
            Self::ScoreNegInf => Some(f32::NEG_INFINITY),
            _ => None,
        }
    }
    fn failed(self) -> bool {
        self != Self::None && self != Self::HiddenNan
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    visible: u32,
    top_k: u32,
    fault: Fault,
}
fn cases() -> Vec<Case> {
    let mut result = Vec::new();
    for visible in [0, 1, 9, WIDTH] {
        for top_k in [1, 7, 255, 511, 512] {
            result.push(Case {
                visible,
                top_k,
                fault: Fault::None,
            });
        }
    }
    for fault in [
        Fault::StaleTag,
        Fault::StaleVisible,
        Fault::SourceStatus,
        Fault::Duplicate,
        Fault::Unsorted,
        Fault::Negative,
        Fault::OutOfRange,
        Fault::TailId,
        Fault::MissingPin,
        Fault::QueryNan,
        Fault::QueryInf,
        Fault::WeightNan,
        Fault::WeightInf,
        Fault::DotOverflow,
        Fault::HeadOverflow,
        Fault::ScoreNan,
        Fault::ScoreInf,
        Fault::ScoreNegInf,
        Fault::HiddenNan,
    ] {
        for top_k in [1, 7, 511] {
            result.push(Case {
                visible: 9,
                top_k,
                fault,
            });
        }
    }
    // A record in the far padded tail must fail even when no slot is active.
    result.push(Case {
        visible: 0,
        top_k: 1,
        fault: Fault::TailId,
    });
    result.push(Case {
        visible: 1,
        top_k: 1,
        fault: Fault::TailId,
    });
    result
}

fn ids(case: Case) -> Vec<i32> {
    let mut ids = vec![-1; KEPT];
    for (block, id) in ids
        .iter_mut()
        .take(case.visible.div_ceil(8) as usize)
        .enumerate()
    {
        *id = block as i32;
    }
    match case.fault {
        Fault::Duplicate => ids[0] = 1,
        Fault::Unsorted => ids.swap(0, 1),
        Fault::Negative => ids[0] = -2,
        Fault::OutOfRange => ids[0] = WIDTH as i32 / 8,
        Fault::TailId => ids[KEPT - 1] = 0,
        Fault::MissingPin => ids[1] = -1,
        _ => {}
    }
    ids
}
fn tags(case: Case) -> (u64, u32, i32) {
    (
        if case.fault == Fault::StaleTag { 2 } else { 1 },
        case.visible + u32::from(case.fault == Fault::StaleVisible),
        i32::from(case.fault == Fault::SourceStatus),
    )
}
fn weights(case: Case) -> Vec<f32> {
    let mut weights = vec![1.; ROWS as usize * 32];
    match case.fault {
        Fault::WeightNan => weights[0] = f32::NAN,
        Fault::WeightInf => weights[0] = f32::INFINITY,
        Fault::HeadOverflow => weights[..32].fill(f32::MAX),
        _ => {}
    }
    weights
}
fn queries(case: Case) -> Vec<u16> {
    let mut queries = vec![0x3f80; ROWS as usize * 32 * 128];
    match case.fault {
        Fault::QueryNan => queries[0] = 0x7fc1,
        Fault::QueryInf => queries[0] = 0x7f80,
        Fault::DotOverflow => queries[..32 * 128].fill(0x7f7f),
        _ => {}
    }
    queries
}
fn injected_scores(case: Case, stride: usize) -> Vec<u8> {
    let mut values = vec![f32::NEG_INFINITY; stride];
    values[..case.visible as usize].fill(0.0);
    let slot = if case.fault == Fault::HiddenNan {
        case.visible as usize
    } else {
        0
    };
    values[slot] = case.fault.injected().unwrap();
    values
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .chain(std::iter::repeat_n(0xa5, stride * 4))
        .collect()
}

#[test]
fn cpu_failure_cases_cover_the_contract() {
    let cases = cases();
    assert_eq!(cases.len(), 79);
    for case in &cases {
        let (stored_tag, stored_visible, source_status) = tags(*case);
        let valid = validate_ids(
            family::compact().query,
            WIDTH,
            case.visible,
            &ids(*case),
            1,
            stored_tag,
            stored_visible,
            source_status,
        );
        assert_eq!(valid.is_err(), case.fault.malformed(), "{case:?}");
        assert!(case.top_k <= 512);
        if case.fault == Fault::TailId {
            assert_eq!(ids(*case)[KEPT - 1], 0);
            assert!(case.visible.div_ceil(8) < KEPT as u32 - 1);
        }
    }
    assert!(cases.iter().any(|c| c.visible == 0 && !c.fault.failed()));
    assert!(cases.iter().any(|c| c.visible == 1 && !c.fault.failed()));
    for fault in [Fault::DotOverflow, Fault::HeadOverflow] {
        let case = Case {
            visible: 9,
            top_k: 7,
            fault,
        };
        let q: Vec<_> = queries(case)[..32 * 128]
            .iter()
            .map(|&v| f32::from_bits(u32::from(v) << 16))
            .collect();
        let w = weights(case);
        assert!(q.iter().chain(&w).all(|v| v.is_finite()));
        assert!(
            kernel_core::position_indexer_oracle::scores_f32(
                family::dense(1).query.selection(),
                &q,
                &w[..32],
                &[1.; 128]
            )
            .is_err()
        );
    }
    println!(
        "failure matrix: 79 compact cases and 50 dense cases; empty/one context, small top-k, malformed tail IDs, nonfinite scores and finite-input overflow"
    );
}

fn key_codes(bf16: bool) -> Vec<u8> {
    if bf16 {
        vec![0x3f80_u16; WIDTH as usize * 128]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect()
    } else {
        vec![0x22; WIDTH as usize * 64]
    }
}

#[test]
fn cpu_failure_key_fixtures_fit_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(|| {
            for bf16 in [false, true] {
                let codes = key_codes(bf16);
                assert_eq!(codes.len(), WIDTH as usize * if bf16 { 256 } else { 64 });
                if bf16 {
                    assert!(codes.chunks_exact(2).all(|v| v == 0x3f80_u16.to_le_bytes()));
                } else {
                    assert!(codes.iter().all(|&v| v == 0x22));
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

fn run(session: &Session, module: &Module, case: Case, bf16: bool, compact: bool) -> LabResult<()> {
    let mut a = Arena {
        session,
        buffers: Vec::new(),
    };
    let n = ROWS as usize;
    let q = a.input(
        &queries(case)
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    let head_weights = a.input(
        &weights(case)
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    let codes = a.input(&key_codes(bf16))?;
    let scales = if bf16 {
        View::EMPTY
    } else {
        a.input(&vec![127; WIDTH as usize * 4])?
    };
    let keys = PositionKeyPages {
        codes,
        scales,
        pages: WIDTH / 64,
        code_page_stride: if bf16 { 16384 } else { 4096 },
        scale_page_stride: if bf16 { 0 } else { 256 },
        page_tables: BlockTables {
            tables: a.input(&i32s(0..(WIDTH / 64) as i32))?,
            sequences: 1,
            max_pages: WIDTH / 64,
        },
    };
    let positions = a.input(&i32s([case.visible as i32 - 1, WIDTH as i32 - 1]))?;
    let row_batch = a.input(&i32s([0, 0]))?;
    let live_rows = a.input(&1_u32.to_le_bytes())?;
    let stride = if compact { SLOTS } else { WIDTH as usize };
    let scores = a.output(n * stride * 4)?;
    let status = a.output(n * 4)?;
    let output = a.output(n * 512 * 4)?;
    let row_tags = a.input(
        &[1_u64, 3]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    let (stored_tag, stored_visible, source_status) = tags(case);
    let candidates = CandidateBlockIds {
        ids: a.input(&i32s(
            ids(case).into_iter().chain(std::iter::repeat_n(-1, KEPT)),
        ))?,
        row_tags: a.input(
            &[stored_tag, 3]
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>(),
        )?,
        visible_positions: a.input(&i32s([stored_visible as i32, WIDTH as i32]))?,
        status: a.input(&i32s([source_status, 0]))?,
        position_capacity: WIDTH,
    };
    let dense = PositionSelectBuffers {
        q,
        head_weights,
        keys,
        positions,
        row_batch,
        live_rows,
        candidate_mask: None,
        position_capacity: WIDTH,
        scores,
        status,
        output,
    };
    let blocks = PositionBlockSelectBuffers {
        q,
        head_weights,
        keys,
        positions,
        row_batch,
        live_rows,
        row_tags,
        candidates,
        scores,
        status,
        output,
    };
    let storage = if bf16 {
        IndexerKeyFormat::Bf16
    } else {
        IndexerKeyFormat::Fp4
    };
    let mut calls = Vec::new();
    let mut workspaces = Vec::new();
    for new in [false, true] {
        let b = backend(new);
        let (seq, scratch) = if compact {
            let op = b
                .position_block_indexer_with_storage(
                    &family::compact().query,
                    storage,
                    case.top_k,
                    WIDTH,
                    Need::rows(ROWS),
                )
                .map_err(|e| e.to_string())?;
            let scratch = a.output((op.info().workspace as usize).max(8))?;
            (op.launches(&blocks, scratch, Counts::rows(ROWS)), scratch)
        } else {
            let op = b
                .position_indexer_with_storage(
                    &family::dense(1).query,
                    storage,
                    case.top_k,
                    WIDTH,
                    Need::rows(ROWS),
                )
                .map_err(|e| e.to_string())?;
            let scratch = a.output((op.info().workspace as usize).max(8))?;
            (op.launches(&dense, scratch, Counts::rows(ROWS)), scratch)
        };
        let mut seq = seq.map_err(|e| e.to_string())?;
        if case.fault.injected().is_some() {
            seq.remove(0);
        }
        calls.push(seq);
        workspaces.push(scratch);
    }
    let graphs = calls
        .iter()
        .map(|seq| session.capture(|_| fire(session, module, &a, seq)))
        .collect::<LabResult<Vec<_>>>()?;
    let mut reference = Vec::new();
    for replay in 0..3 {
        for arm in 0..2 {
            a.poison(&[scores, status, output, workspaces[arm]])?;
            if case.fault.injected().is_some() {
                a.write(scores, &injected_scores(case, stride))?;
            }
            if replay == 0 {
                if case.fault.numeric() {
                    fire(session, module, &a, &calls[arm][..1])?;
                    session.stream.synchronize().map_err(|e| e.to_string())?;
                    assert_eq!(
                        &a.read(scores)?[..4],
                        &0x7fc00000_u32.to_le_bytes(),
                        "score did not flag {case:?}"
                    );
                    fire(session, module, &a, &calls[arm][1..])?;
                } else {
                    fire(session, module, &a, &calls[arm])?;
                }
            } else {
                graphs[arm]
                    .launch(&session.stream)
                    .map_err(|e| e.to_string())?;
            }
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let actual = [scores, status, output]
                .into_iter()
                .map(|v| a.read(v))
                .collect::<LabResult<Vec<_>>>()?;
            assert_eq!(
                &actual[1][..4],
                &i32::from(case.fault.failed()).to_le_bytes(),
                "{case:?} compact={compact}"
            );
            assert_eq!(&actual[1][4..], &[0xa5; 4]);
            assert!(actual[0][stride * 4..].iter().all(|&v| v == 0xa5));
            assert!(actual[2][512 * 4..].iter().all(|&v| v == 0xa5));
            let selected = if case.fault.failed() {
                0
            } else {
                case.visible.min(case.top_k)
            };
            let expected: Vec<_> = (0..512)
                .map(|i| if i < selected { i as i32 } else { -1 })
                .collect();
            assert_eq!(
                &actual[2][..512 * 4],
                i32s(expected),
                "selection/padding {case:?}"
            );
            if case.fault.failed() {
                let sanitized = if case.fault.malformed() {
                    stride
                } else if case.fault.numeric() {
                    case.visible as usize
                } else {
                    1
                };
                for bytes in actual[0][..sanitized * 4].chunks_exact(4) {
                    assert_eq!(bytes, f32::NEG_INFINITY.to_le_bytes());
                }
            }
            if replay == 0 && arm == 0 {
                reference = actual;
            } else {
                for (plane, bytes) in actual.iter().enumerate() {
                    compare_bytes(
                        &reference[plane],
                        bytes,
                        &format!(
                            "{case:?} compact={compact} bf16={bf16} replay={replay} arm={arm} plane={plane}"
                        ),
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated idle GPU via the staged job"]
fn gpu_score_select_failures() {
    let result = kernel_harness::gpu_test(64 << 20, |session| {
        let root = PathBuf::from(
            std::env::var_os("CHOPINFER_POSITION_LIVE_CUBINS").ok_or("prebuilt cubins required")?,
        );
        let module = session.load_cubin(
            &std::fs::read(root.join(format!("{}.cubin", family::CUBIN)))
                .map_err(|e| e.to_string())?,
        )?;
        let bf16 =
            std::env::var("CHOPINFER_POSITION_LIVE_STORAGE").map_err(|e| e.to_string())? == "bf16";
        let mut count = 0;
        for case in cases() {
            for compact in [false, true] {
                if !compact && case.fault.malformed() {
                    continue;
                }
                run(session, &module, case, bf16, compact)?;
                count += 1;
            }
        }
        assert_eq!(count, 129);
        println!("completed {count} poisoned score/select failure comparisons");
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "required failure comparisons skipped");
}
