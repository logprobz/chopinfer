//! Explicit TP4 range candidates using existing compact kernels and knobs.
use super::*;

const CASE_ENV: &str = "CHOPINFER_DEEPSEEK41_COMPACT_RANGE";
const SHAPES: &[(u32, u32, u32)] = &[
    (1280, 5120, 1),
    (8192, 1280, 1),
    (512, 5120, 1),
    (1024, 4096, 2),
    (5120, 2048, 1),
    (576, 5120, 1),
    (5120, 576, 1),
    (4096, 1280, 1),
    (25600, 6144, 1),
    (5120, 15360, 1),
];

struct Case {
    entry: Entry,
    ranks: u32,
    rows: Vec<u32>,
    first: u32,
    last: u32,
}

fn configured(value: Value) -> LabResult<Case> {
    let object = value.as_object().ok_or("range case must be an object")?;
    let expected = [
        "n",
        "k",
        "groups",
        "stages",
        "ranks",
        "first",
        "last",
        "neighbors",
    ];
    if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
        return Err("range case has missing or unknown fields".into());
    }
    let count = |key: &str| {
        value[key]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("range case requires a u32 {key}"))
    };
    let (n, k, groups, stages, ranks) = (
        count("n")?,
        count("k")?,
        count("groups")?,
        count("stages")?,
        count("ranks")?,
    );
    let (first, last) = (count("first")?, count("last")?);
    if ranks != 4
        || !SHAPES.contains(&(n, k, groups))
        || !matches!(stages, 2 | 3)
        || first > last
        || !((65..=127).contains(&first) && last <= 127
            || (129..=511).contains(&first) && last <= 511)
    {
        return Err("range case is outside the TP4 candidate bands or matched shapes".into());
    }
    if (n, k) == (5120, 15360) && first < 129 {
        return Err("DSpark main projection currently qualifies only the upper band".into());
    }
    let neighbors = value["neighbors"]
        .as_bool()
        .ok_or("neighbors must be a boolean")?;
    let entry = family::entries()
        .into_iter()
        .find(|e| {
            let q = e.base.query;
            (q.n, q.k, q.groups) == (n, k, groups) && e.stages == stages
        })
        .ok_or("missing existing compact entry")?;
    if !topology(&entry).contains(&ranks) {
        return Err("range entry is absent at TP4".into());
    }
    let rows = if neighbors {
        (first - 1..=last + 1).collect()
    } else {
        (first..=last).collect()
    };
    Ok(Case {
        entry,
        ranks,
        rows,
        first,
        last,
    })
}

fn from_env() -> LabResult<Case> {
    let text = std::env::var(CASE_ENV).map_err(|e| e.to_string())?;
    if text.len() > 1024 {
        return Err("range case exceeds its byte limit".into());
    }
    configured(serde_json::from_str(&text).map_err(|e| e.to_string())?)
}

fn packed_candidate(
    arena: &mut Arena<'_>,
    e: &Entry,
    rows: u32,
    buffers: &Buffers<View>,
) -> LabResult<Vec<Launch>> {
    let mut b = *buffers;
    b.weights = if arena.session.is_some() {
        let q = e.base.query;
        let codes = arena.read(b.weights)?;
        let packed = kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout {
            groups: q.groups as usize,
            n: q.n as usize,
            k: q.k as usize,
        }
        .pack(&codes[..b.weights.bytes as usize])
        .map_err(|e| e.to_string())?;
        arena.bytes(&packed)?
    } else {
        arena.filled(b.weights.bytes, 0)?
    };
    Sm120::new()
        .with_mxfp8_prefill(e.base.clone())
        .with_mxfp8_prefill_compact(e.clone(), rows)
        .with_packed_fold_weights()
        .mxfp8_prefill_gemm(&e.base.query, Need::rows(rows))
        .map_err(|e| e.to_string())?
        .launches(&b, View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.to_string())
}

fn calls(
    arena: &mut Arena<'_>,
    case: &Case,
    rows: u32,
    buffers: &[Buffers<View>; 3],
) -> LabResult<[Vec<Launch>; 3]> {
    let baseline = if (case.entry.base.query.n, case.entry.base.query.k) == (5120, 15360) {
        // DSpark Prepared::main uses the generic fold above its small-row
        // capacity. Comparing against the prefill wrapper would miss that gap.
        let mut b = buffers[0];
        let q = case.entry.base.query;
        b.weights = if arena.session.is_some() {
            let codes = arena.read(b.weights)?;
            let packed = kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout {
                groups: q.groups as usize,
                n: q.n as usize,
                k: q.k as usize,
            }
            .pack(&codes[..b.weights.bytes as usize])
            .map_err(|e| e.to_string())?;
            arena.bytes(&packed)?
        } else {
            arena.filled(b.weights.bytes, 0)?
        };
        registration::register(Sm120::new(), &FLASH, case.ranks)?
            .with_packed_fold_weights()
            .mxfp8_fold_gemm(&q, Need::rows(rows))
            .map_err(|e| e.to_string())?
            .launches(&b, View::EMPTY, Counts::rows(rows))
            .map_err(|e| e.to_string())?
    } else {
        prepare_at_rank(arena, &case.entry, rows, &buffers[0], false, case.ranks)?
    };
    Ok([
        baseline,
        prepare_at_rank(arena, &case.entry, rows, &buffers[1], true, case.ranks)?,
        packed_candidate(arena, &case.entry, rows, &buffers[2])?,
    ])
}

fn banks(
    arena: &mut Arena<'_>,
    e: &Entry,
    rows: u32,
    weights: [View; 2],
) -> LabResult<[Buffers<View>; 3]> {
    let first = buffers(arena, e, rows, weights)?;
    let mut other = || -> LabResult<Buffers<View>> {
        Ok(Buffers {
            output: arena.filled(first.output.bytes, 255)?,
            status: arena.filled(first.status.bytes, 0)?,
            ..first
        })
    };
    Ok([first, other()?, other()?])
}

#[test]
fn range_configuration_covers_bands_and_neighbours() {
    let value = |first, last| {
        json!({"n":1280,"k":5120,"groups":1,"stages":2,"ranks":4,
               "first":first,"last":last,"neighbors":true})
    };
    for (first, last, expected) in [(65, 127, 65), (129, 511, 385)] {
        let case = configured(value(first, last)).unwrap();
        assert_eq!(case.rows.len(), expected);
        assert_eq!(case.rows, (first - 1..=last + 1).collect::<Vec<_>>());
    }
    for (first, last) in [(0, 65), (64, 127), (65, 128), (129, 512), (130, 129)] {
        assert!(configured(value(first, last)).is_err());
    }
    let mut wrong = value(65, 127);
    wrong["ranks"] = json!(1);
    assert!(configured(wrong).is_err());

    let case = configured(json!({"n":5120,"k":15360,"groups":1,"stages":2,"ranks":4,
                                "first":192,"last":192,"neighbors":true}))
    .unwrap();
    assert_eq!(case.rows, vec![191, 192, 193]);
    let mut arena = Arena::new(None);
    let size = case.entry.base.bytes(1);
    let weights = [
        arena.filled(size.weights, 0).unwrap(),
        arena.filled(size.weight_scales, 0).unwrap(),
    ];
    let banks = banks(&mut arena, &case.entry, 192, weights).unwrap();
    let arms = calls(&mut arena, &case, 192, &banks).unwrap();
    assert!(
        arms[0].iter().any(|call| call.symbol
            == "chop_deepseek41_mxfp8_fold_m8192_n5120_k15360_g1_f32_v2_packed_v1")
    );
    assert!(
        arms[2]
            .iter()
            .any(|call| call.symbol.contains("_compact_") && call.symbol.ends_with("_packed_v1"))
    );
}

#[test]
#[ignore = "requires CHOPINFER_DEEPSEEK41_COMPACT_RANGE"]
fn range_inventory() {
    let case = from_env().unwrap();
    let mut records = Vec::new();
    for &rows in &case.rows {
        let mut arena = Arena::new(None);
        let size = case.entry.base.bytes(1);
        let weights = [
            arena.filled(size.weights, 0).unwrap(),
            arena.filled(size.weight_scales, 0).unwrap(),
        ];
        let banks = banks(&mut arena, &case.entry, rows, weights).unwrap();
        let arms = calls(&mut arena, &case, rows, &banks).unwrap();
        for call in arms.iter().flatten() {
            call.plan.validate_geometry().unwrap();
            assert_eq!(call.plan.workspace_bytes, 0);
        }
        assert!(
            arms[2]
                .iter()
                .any(|call| call.symbol.ends_with("_packed_v1"))
        );
        records.push(json!({"rows":rows,"symbols":arms.iter().map(|arm|arm.iter().map(|c|&c.symbol).collect::<Vec<_>>()).collect::<Vec<_>>()}));
    }
    println!(
        "compact_range_inventory {}",
        json!({"ranks":case.ranks,"first":case.first,"last":case.last,"cases":records,"serving_enabled":false})
    );
}

#[test]
#[ignore = "requires CHOPINFER_DEEPSEEK41_COMPACT_RANGE and staged compact cubins"]
fn range_bitwise_gpu() {
    let case = from_env().unwrap();
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 8 << 30, |s| {
        let mut arena = Arena::new(Some(s));
        let e = &case.entry;
        let size = e.base.bytes(1);
        let weights = [
            arena.bytes(&seed(size.weights as usize, false, 901))?,
            arena.bytes(&seed(size.weight_scales as usize, true, 1901))?,
        ];
        let keep = arena.next;
        let started = std::time::Instant::now();
        let mut completed = 0;
        for &rows in &case.rows {
            if started.elapsed().as_secs() >= 240 {
                return Err("compact range exceeded its budget; coverage is incomplete".into());
            }
            let banks = banks(&mut arena, e, rows, weights)?;
            let codes = seed(banks[0].activations.bytes as usize, false, 1771);
            let scales = seed(banks[0].activation_scales.bytes as usize, true, 2771);
            arena.write(banks[0].activations, &codes)?;
            arena.write(banks[0].activation_scales, &scales)?;
            let arms = calls(&mut arena, &case, rows, &banks)?;
            let functions = functions(s, &arms.concat())?;
            let graphs = arms
                .iter()
                .map(|arm| s.capture(|stream| issue(s, &arena, &functions, arm, stream)))
                .collect::<LabResult<Vec<_>>>()?;
            let mut live_rows = vec![rows, rows / 2];
            live_rows.extend([127, 128, 129].into_iter().filter(|&n| n < rows));
            live_rows.extend([0, rows]);
            let mut comparisons = 0;
            for bad in [false, true] {
                let mut current_codes = codes.clone();
                let mut current_scales = scales.clone();
                if bad {
                    current_codes[e.base.query.activation_row_stride as usize] = 127;
                    current_scales[2 * e.base.query.activation_row_stride as usize / 32] = 255;
                }
                arena.write(banks[0].activations, &current_codes)?;
                arena.write(banks[0].activation_scales, &current_scales)?;
                for &live in &live_rows {
                    for captured in [false, true] {
                        for (index, bank) in banks.iter().enumerate() {
                            reset(&arena, bank, rows, live, bad)?;
                            if captured {
                                graphs[index].launch(&s.stream).map_err(|e| e.to_string())?;
                            } else {
                                issue(s, &arena, &functions, &arms[index], &s.stream)?;
                            }
                        }
                        for bank in &banks[1..] {
                            equality(&arena, &banks[0], bank, rows, live, bad)?;
                            comparisons += 1;
                        }
                    }
                }
            }
            println!("compact_range_bitwise {}", json!({"n":e.base.query.n,"k":e.base.query.k,"groups":e.base.query.groups,"stages":e.stages,"ranks":case.ranks,"rows":rows,"live_rows":live_rows,"comparisons":comparisons,"bitwise":true,"packed_twin":true,"serving_enabled":false,"symbols":arms.iter().map(|arm|arm.iter().map(|c|&c.symbol).collect::<Vec<_>>()).collect::<Vec<_>>()}));
            completed += 1;
            arena.buffers.retain(|&handle, _| handle <= keep);
            arena.next = keep;
        }
        if completed != case.rows.len() || completed == 0 {
            return Err("incomplete compact range".into());
        }
        println!("compact_range_complete {}", json!({"cases":completed,"ranks":case.ranks,"first":case.first,"last":case.last,"serving_enabled":false}));
        Ok(())
    })
    .unwrap();
}

#[test]
#[ignore = "requires CHOPINFER_DEEPSEEK41_COMPACT_RANGE and staged compact cubins"]
fn range_timing_gpu() {
    let case = from_env().unwrap();
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 8 << 30, |s| {
        let mut arena = Arena::new(Some(s));
        let e = &case.entry;
        let size = e.base.bytes(1);
        let weights = [arena.bytes(&seed(size.weights as usize, false, 901))?,
                       arena.bytes(&seed(size.weight_scales as usize, true, 1901))?];
        let keep = arena.next;
        let started = std::time::Instant::now();
        let mut completed = 0;
        for &rows in &case.rows {
            if started.elapsed().as_secs() >= 240 {
                return Err("compact timing exceeded its budget; coverage is incomplete".into());
            }
            let banks = banks(&mut arena, e, rows, weights)?;
            arena.write(banks[0].activations, &seed(banks[0].activations.bytes as usize, false, 1771))?;
            arena.write(banks[0].activation_scales, &seed(banks[0].activation_scales.bytes as usize, true, 2771))?;
            let arms = calls(&mut arena, &case, rows, &banks)?;
            let functions = functions(s, &arms.concat())?;
            let graphs = [0, 2].map(|arm| s.capture(|stream| issue(s, &arena, &functions, &arms[arm], stream)));
            let [baseline, candidate] = graphs;
            let graphs = [baseline?, candidate?];
            for (graph, bank) in graphs.iter().zip([&banks[0], &banks[2]]) {
                reset(&arena, bank, rows, rows, false)?;
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
            }
            equality(&arena, &banks[0], &banks[2], rows, rows, false)?;
            let aa_before = timing::pair(s, |_, stream| graphs[0].launch(stream).map_err(|e| e.to_string()))?;
            let ab = timing::pair(s, |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()))?;
            let aa_after = timing::pair(s, |_, stream| graphs[0].launch(stream).map_err(|e| e.to_string()))?;
            equality(&arena, &banks[0], &banks[2], rows, rows, false)?;
            println!("compact_range_timing {}", json!({"n":e.base.query.n,"k":e.base.query.k,"groups":e.base.query.groups,
                "stages":e.stages,"ranks":case.ranks,"rows":rows,"bitwise":true,"packed_twin":true,"serving_enabled":false,
                "baseline_symbols":arms[0].iter().map(|c|&c.symbol).collect::<Vec<_>>(),
                "candidate_symbols":arms[2].iter().map(|c|&c.symbol).collect::<Vec<_>>(),
                "aa_before":aa_before,"ab":ab,"aa_after":aa_after}));
            completed += 1;
            arena.buffers.retain(|&handle, _| handle <= keep);
            arena.next = keep;
        }
        if completed != case.rows.len() || completed == 0 {
            return Err("incomplete compact timing range".into());
        }
        println!("compact_range_timing_complete {}", json!({"cases":completed,"serving_enabled":false}));
        Ok(())
    }).unwrap();
}
