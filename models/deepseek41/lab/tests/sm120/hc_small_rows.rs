//! Qualify existing chain producers against the production fallback at rows 1-8.
//! Each logical rank has a distinct finite input. Special values, inactive
//! rows and the partial workspace are checked in eager and captured replays.
use super::*;
use bundle_deepseek41::sm120::{fold_cubin_for_symbol, hc_round_chain};
use runtime_cuda::Module;

const COUNTS: std::ops::RangeInclusive<u32> = 1..=8;
const WARPS: [u32; 2] = [4, 8];

struct Arm {
    functions: Vec<Function>,
    plans: [LaunchPlan; 5],
    _modules: Vec<Module>,
}

impl Arm {
    fn new(s: &Session, rows: u32, warps: Option<u32>) -> LabResult<Self> {
        let mut names = symbols();
        let mut plans = plans(rows, rows)?;
        if let Some(warps) = warps {
            let entry = hc_round_chain::entry(&FLASH);
            names[3] = entry.symbols()[usize::from(warps == 8)].clone();
            plans[3] = entry.plan(rows, rows, warps).map_err(|e| e.to_string())?;
        }
        let mut modules = Vec::new();
        let mut functions = Vec::new();
        for name in names {
            let (_, bytes) = fold_cubin_for_symbol(&name).ok_or("missing locked HC module")?;
            let module = s.load_cubin(bytes)?;
            functions.push(module.function(&name).map_err(|e| e.to_string())?);
            modules.push(module);
        }
        Ok(Self {
            functions,
            plans,
            _modules: modules,
        })
    }

    fn launch(&self, i: &Input, o: &Output, aliased: bool, stream: &Stream) -> LabResult<()> {
        run(&self.functions, true, aliased, &self.plans, i, o, stream)
    }
}

fn inputs(s: &Session, rows: u32) -> LabResult<Input> {
    let r = rows as usize;
    Ok(Input {
        x: s.allocate(r * N * H * 2)?,
        y: s.allocate(r * H * 4)?,
        fn_: s.allocate(MIXES * N * H * 4)?,
        base: s.allocate(MIXES * 4)?,
        scale: s.allocate(12)?,
        pre: s.allocate(r * N * 4)?,
        post_in: s.allocate(r * N * 4)?,
        comb_in: s.allocate(r * N * N * 4)?,
        live: s.allocate(4)?,
    })
}

fn rank_inputs(s: &Session, rows: u32, rank: u32, pattern: &str, i: &Input) -> LabResult<()> {
    set_inputs(s, rows, pattern, i)?;
    if pattern == "finite" {
        upload(
            &i.x,
            &artifacts::bf16_bytes(&artifacts::random(
                rows as usize * N * H,
                123 + u64::from(rank) * 7919,
                4.0,
            )),
            &s.stream,
        )?;
        upload(
            &i.y,
            &artifacts::f32_bytes(&artifacts::random(
                rows as usize * H,
                71 + u64::from(rank) * 7907,
                2.0,
            )),
            &s.stream,
        )?;
    }
    Ok(())
}

fn values(s: &Session, o: &Output) -> LabResult<Vec<Vec<u8>>> {
    let mut result = o.read(s)?;
    result.push(download(&o.scratch, o.scratch_len, &s.stream)?);
    Ok(result)
}

fn compare(a: &[Vec<u8>], b: &[Vec<u8>]) -> LabResult<()> {
    let names = [
        "expanded",
        "collapsed",
        "pre",
        "post",
        "comb",
        "rounded",
        "partials",
    ];
    compare_roles(&names, a, b)
}

#[test]
fn plans_cover_every_small_row_without_changing_the_workspace() {
    let entry = hc_round_chain::entry(&FLASH);
    for rows in COUNTS {
        for warps in WARPS {
            for live in live_counts(rows) {
                let plan = entry.plan(rows, live, warps).unwrap();
                assert_eq!(plan.grid, [160, rows, 1]);
                assert_eq!(plan.block, [warps * 32, 1, 1]);
                assert_eq!(
                    plan.workspace_bytes,
                    plans(rows, live).unwrap()[3].workspace_bytes
                );
            }
            assert!(entry.plan(rows, rows + 1, warps).is_err());
        }
    }
    assert!(entry.plan(0, 0, 4).is_err());
    assert!(entry.plan(8, 8, 3).is_err());
}

fn qualify(s: &Session, patterns: &[&str]) -> LabResult<()> {
    let mut cases = 0;
    let mut negatives = 0;
    for rows in COUNTS {
        let i = inputs(s, rows)?;
        let reference = Arm::new(s, rows, None)?;
        for warps in WARPS {
            let candidate = Arm::new(s, rows, Some(warps))?;
            let outputs = [Output::new(s, rows, true)?, Output::new(s, rows, true)?];
            let arms = [&reference, &candidate];
            let graphs = arms
                .iter()
                .zip(&outputs)
                .map(|(arm, out)| s.capture(|stream| arm.launch(&i, out, true, stream)))
                .collect::<LabResult<Vec<_>>>()?;
            for rank in 0..4 {
                for &pattern in patterns {
                    rank_inputs(s, rows, rank, pattern, &i)?;
                    for live in live_counts(rows) {
                        upload(&i.live, &live.to_ne_bytes(), &s.stream)?;
                        let mut eager: Option<Vec<Vec<u8>>> = None;
                        for replay in 0..3 {
                            let initial = outputs
                                .iter()
                                .map(|o| o.reset(s, rows, pattern))
                                .collect::<LabResult<Vec<_>>>()?;
                            for (n, arm) in arms.iter().enumerate() {
                                if replay == 0 {
                                    arm.launch(&i, &outputs[n], true, &s.stream)?;
                                } else {
                                    graphs[n].launch(&s.stream).map_err(|e| e.to_string())?;
                                }
                            }
                            s.stream.synchronize().map_err(|e| e.to_string())?;
                            let actual = outputs
                                .iter()
                                .map(|o| values(s, o))
                                .collect::<LabResult<Vec<_>>>()?;
                            let context = format!(
                                "rows={rows} warps={warps} rank={rank} pattern={pattern} live={live} replay={replay}"
                            );
                            compare(&actual[0], &actual[1])
                                .map_err(|e| format!("{context}: {e}"))?;
                            if let Some(previous) = &eager {
                                compare(previous, &actual[1])
                                    .map_err(|e| format!("{context} eager: {e}"))?;
                            } else {
                                eager = Some(actual[0].clone());
                            }
                            for (n, data) in actual.iter().enumerate() {
                                for (role, bytes) in data[..ROLES.len()].iter().enumerate() {
                                    let start = bytes.len() / rows as usize * live as usize;
                                    if bytes[start..] != initial[n][role][start..] {
                                        return Err(format!(
                                            "{context}: inactive {} changed",
                                            ROLES[role]
                                        ));
                                    }
                                }
                                let scratch = data.last().unwrap();
                                let start = scratch.len() / rows as usize * live as usize;
                                if scratch[start..].iter().any(|&v| v != 0xa5) {
                                    return Err(format!("{context}: inactive workspace changed"));
                                }
                            }
                        }
                    }
                    cases += 1;
                }
                rank_inputs(s, rows, rank, "finite", &i)?;
                upload(&i.live, &rows.to_ne_bytes(), &s.stream)?;
                for out in &outputs {
                    out.reset(s, rows, "finite")?;
                }
                reference.launch(&i, &outputs[0], true, &s.stream)?;
                candidate.launch(&i, &outputs[1], true, &s.stream)?;
                s.stream.synchronize().map_err(|e| e.to_string())?;
                let expected = values(s, &outputs[0])?;
                compare(&expected, &values(s, &outputs[1])?)?;
                // A missing producer must leave poisoned writes, including its partials.
                outputs[1].reset(s, rows, "finite")?;
                if compare(&expected, &values(s, &outputs[1])?).is_ok() {
                    return Err("omitted producer escaped the comparator".into());
                }
                negatives += 1;
                upload(
                    &i.pre,
                    &artifacts::f32_bytes(&vec![32.0; rows as usize * N]),
                    &s.stream,
                )?;
                graphs[1].launch(&s.stream).map_err(|e| e.to_string())?;
                s.stream.synchronize().map_err(|e| e.to_string())?;
                if compare(&expected, &values(s, &outputs[1])?).is_ok() {
                    return Err("changed collapse weights escaped the captured replay".into());
                }
                negatives += 1;
                eprintln!(
                    "HC_SMALL_ROW_PASS rows={rows} warps={warps} rank={rank} patterns={} negatives=2",
                    patterns.len()
                );
            }
        }
    }
    assert_eq!(cases, 8 * 2 * 4 * patterns.len());
    assert_eq!(negatives, 8 * 2 * 4 * 2);
    eprintln!("HC_SMALL_COMPLETE cases={cases} negatives={negatives} mismatches=0");
    Ok(())
}

#[test]
fn chain_matches_fallback_on_gpu() {
    kernel_harness::gpu_test(64 << 20, |s| qualify(s, &PATTERNS)).unwrap();
}

#[test]
#[ignore = "run explicitly under compute-sanitizer on an idle GPU"]
fn chain_memcheck_on_gpu() {
    let result = kernel_harness::gpu_test(64 << 20, |s| {
        qualify(s, &["finite", "sum_nan", "weight_nan"])
    })
    .unwrap();
    assert!(result.is_some());
}

#[test]
#[ignore = "requires staged CHOPINFER_ROUND_ARTIFACTS and a quiet timing window"]
fn paired_chain_timing_on_gpu() {
    let result = kernel_harness::gpu_test_with(artifacts::timing_cubins().unwrap(), 1 << 30, |s| {
        let mut measured = 0;
        for rows in COUNTS {
            let i = inputs(s, rows)?;
            rank_inputs(s, rows, 0, "finite", &i)?;
            upload(&i.live, &rows.to_ne_bytes(), &s.stream)?;
            // Separate baseline arms provide the A/A control for this row count.
            for warps in [None, Some(4), Some(8)] {
                let arms = [Arm::new(s, rows, None)?, Arm::new(s, rows, warps)?];
                let outputs = [Output::new(s, rows, true)?, Output::new(s, rows, true)?];
                let graphs = arms
                    .iter()
                    .zip(&outputs)
                    .map(|(arm, out)| {
                        s.capture(|stream| {
                            for _ in 0..32 {
                                arm.launch(&i, out, false, stream)?;
                            }
                            Ok(())
                        })
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                for out in &outputs {
                    out.reset(s, rows, "finite")?;
                }
                for graph in &graphs {
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                }
                s.stream.synchronize().map_err(|e| e.to_string())?;
                compare(&values(s, &outputs[0])?, &values(s, &outputs[1])?)?;
                let label = format!("HC_SMALL rows={rows} warps={}", warps.unwrap_or(0));
                let timing = kernel_harness::with_timing_label(&label, || {
                    kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                        graphs[arm].launch(stream).map_err(|e| e.to_string())
                    })
                })?;
                eprintln!(
                    "HC_SMALL_TIMING {}",
                    json!({
                        "rows":rows, "warps":warps, "aa":warps.is_none(),
                        "old_us":timing.arms[0].median_ms * 1000. / 32.,
                        "new_us":timing.arms[1].median_ms * 1000. / 32.,
                        "ratio":timing.paired_ratio, "rounds":timing.arms[0].rounds,
                    })
                );
                measured += 1;
            }
        }
        assert_eq!(measured, 24);
        eprintln!("HC_SMALL_TIMING_COMPLETE cases={measured}");
        Ok(())
    })
    .unwrap();
    assert!(result.is_some());
}
