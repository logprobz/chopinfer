//! Compare existing packed matrix tiles with the served small-row Engram
//! fold, and the wide verify tile with the fold it replaces above 64 rows.
use super::*;

const ROWS: [u32; 7] = [32, 33, 40, 48, 56, 63, 64];

/// Wide verify steps the served 64-row, two-stage tile computes at their
/// exact rows (`sm120::engram_verify`): both sides of the 128-row bucket
/// and of the tile's 64-row boundaries, up to the widest 192-row step.
const WIDE_ROWS: [u32; 8] = [65, 96, 127, 128, 129, 160, 191, 192];

/// The packed matrix tiles compared at `rows`: every tile up to 64 rows
/// at the decode route's rows, the served wide verify tile above them.
fn entries(rows: u32) -> Vec<prefill::Entry> {
    let wide = rows > 64;
    let entries: Vec<_> = sm120::prefill_matrix::catalog()
        .unwrap()
        .into_iter()
        .filter(|e| {
            (e.base.query.n, e.base.query.k, e.base.query.groups) == (25600, 6144, 1)
                && if wide {
                    (e.tile_rows, e.stages) == (64, 2)
                } else {
                    e.tile_rows <= 64
                }
        })
        .collect();
    assert_eq!(entries.len(), if wide { 1 } else { 3 });
    entries
}

fn captured() -> LabResult<Vec<data::Input>> {
    let manifest = data::manifest()?;
    if manifest["schema"] != "chopinfer.engram-fold-subgraph.v1" {
        return Err("expected authenticated Engram subgraph inputs".into());
    }
    let captures = manifest["captures"].as_array().ok_or("capture matrix")?;
    if captures.len() != 2 {
        return Err("expected both Engram sites".into());
    }
    captures
        .iter()
        .enumerate()
        .map(|(index, c)| {
            if c["layer"] != [1, 14][index]
                || c["rows"] != 8192
                || c["n"] != 25600
                || c["k"] != 6144
                || c["groups"] != 1
            {
                return Err("unexpected Engram capture geometry".into());
            }
            let input = data::Input {
                id: c["id"].as_str().ok_or("capture id")?.into(),
                rows: 8192,
                n: 25600,
                k: 6144,
                groups: 1,
                codes: data::blob(&c["codes"])?,
                scales: data::blob(&c["scales"])?,
                weights: data::blob(&c["weights"])?,
                weight_scales: data::blob(&c["weight_scales"])?,
                expected: None,
            };
            if input.codes.len() != 8192 * 6144
                || input.scales.len() != 8192 * 192
                || input.weights.len() != 25600 * 6144
                || input.weight_scales.len() != 25600 * 192
            {
                return Err("unexpected Engram capture extent".into());
            }
            Ok(input)
        })
        .collect()
}

/// The first `rows` rows of the capture's first or last 64 rows, or of its
/// first or last 192 for a wide verify step.
fn slice(input: &data::Input, rows: u32, tail: bool) -> data::Input {
    let window = if rows <= 64 { 64 } else { 192 };
    assert!(rows <= window && input.rows >= window);
    let start = if tail {
        (input.rows - window) as usize
    } else {
        0
    };
    let width = input.k as usize;
    data::Input {
        id: format!(
            "{}-{}{window}",
            input.id,
            if tail { "tail" } else { "head" }
        ),
        rows,
        n: input.n,
        k: input.k,
        groups: input.groups,
        codes: input.codes[start * width..(start + rows as usize) * width].to_vec(),
        scales: input.scales[start * (width / 32)..(start + rows as usize) * (width / 32)].to_vec(),
        weights: input.weights.clone(),
        weight_scales: input.weight_scales.clone(),
        expected: None,
    }
}

fn matrix(entry: &prefill::Entry, b: &Buffers<View>, rows: u32) -> LabResult<Vec<Launch>> {
    kernels_sm120::backend::packed_op(
        entry.prepare(rows).map_err(|e| e.to_string())?,
        [entry.symbol(entry.tile_rows)],
    )
    .map_err(|e| e.to_string())?
    .launches(b, View::EMPTY, Counts::rows(rows))
    .map_err(|e| e.to_string())
}

/// What a verify step above 64 rows serves without the `wide_verify`
/// option (`DenseOps::project_with`): the general fold prepared for the
/// prefill capacity.
fn verify_reference(
    a: &mut Arena<'_>,
    base: &mxfp8_fold::Entry,
    b: &Buffers<View>,
    rows: u32,
) -> LabResult<Calls> {
    let b = &packed_views(a, base, b)?;
    let mut q = base.query;
    q.m_capacity = 8192;
    let op = backend()
        .mxfp8_fold_gemm(&q, Need::rows(rows))
        .map_err(|e| e.to_string())?;
    let work = a.filled(op.info().workspace.max(4), 0)?;
    let run = op
        .launches(b, work, Counts::rows(rows))
        .map_err(|e| e.to_string())?;
    // The general fold's own entry, not the matrix tile under comparison.
    if run.iter().any(|c| c.symbol.contains("_matrix_")) {
        return Err("the verify reference must be the general fold".into());
    }
    Ok(Calls {
        init: vec![],
        run,
        state: None,
    })
}

/// Each tile of `entries(rows)` against the served baseline at the
/// input's rows: the decode route up to 64 rows, `verify_reference` above.
/// `timed` also times every arm warm and cold.
fn check(s: &Session, input: &data::Input, timed: bool) -> LabResult<()> {
    let rows = input.rows;
    let entries = entries(rows);
    let base = &entries[0].base;
    let arms = entries.len() + 1;
    let mut a = Arena::new(Some(s));
    let b = (0..arms)
        .map(|_| common(&mut a, base, rows, input))
        .collect::<LabResult<Vec<_>>>()?;
    let baseline = if rows > 64 {
        verify_reference(&mut a, base, &b[0], rows)?
    } else {
        reference(&mut a, base, &b[0], rows)?
    };
    assert!(baseline.init.is_empty());
    if matches!(rows, 32 | 64) {
        assert!(baseline.run.iter().any(|c| c.symbol.contains("direct")));
    }
    let mut calls = vec![baseline.run];
    let mut controls = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let packed = packed_views(&mut a, base, &b[index + 1])?;
        calls.push(matrix(entry, &packed, rows)?);
        // The locked omit-first control reads row-major codes.
        let mut omit = entry
            .prepare(rows)
            .map_err(|e| e.to_string())?
            .launches(&b[index + 1], View::EMPTY, Counts::rows(rows))
            .map_err(|e| e.to_string())?;
        omit.last_mut().unwrap().symbol.push_str("_omit_first_v1");
        controls.push(omit);
    }
    let all: Vec<_> = calls.iter().chain(&controls).flatten().cloned().collect();
    let f = fused::load(s, &all)?;
    let graphs = calls
        .iter()
        .map(|v| s.capture(|stream| issue(s, &a, &f, v, stream)))
        .collect::<LabResult<Vec<_>>>()?;
    let mut comparisons = 0;
    for (live, bad) in [
        (rows, false),
        (rows - 1, true),
        (0, false),
        (rows, true),
        (rows, false),
    ] {
        for replay in 0..3 {
            for arm in 0..arms {
                reset(&a, &b[arm], input, rows, live, bad)?;
                if replay == 0 {
                    issue(s, &a, &f, &fused::eager(&calls[arm]), &s.stream)?;
                } else {
                    graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                }
                let actual = a.read(b[arm].status)?;
                let expected = status(rows, live, bad);
                if actual[..12] != expected[..12]
                    || actual[(rows as usize + 3) * 4..] != expected[(rows as usize + 3) * 4..]
                {
                    return Err("status guard changed".into());
                }
            }
            for arm in 1..arms {
                compare(&a, &[b[0], b[arm]], input, rows, live, bad, None)?;
                comparisons += 1;
            }
        }
    }
    for (index, control) in controls.iter().enumerate() {
        reset(&a, &b[index + 1], input, rows, rows, false)?;
        issue(s, &a, &f, &fused::eager(control), &s.stream)?;
        if a.read(b[0].output)? == a.read(b[index + 1].output)? {
            return Err("omit-first control was ineffective".into());
        }
    }
    let minimum_free = std::cell::Cell::new(s.context.memory_info().map_err(|e| e.to_string())?.0);
    let mut record = json!({
        "rows": rows, "source":input.id, "comparisons":comparisons,
        "bitwise":true, "negative_controls":controls.len(), "graph_replays":2,
        "reference":calls[0].iter().map(|c| &c.symbol).collect::<Vec<_>>(),
        "variants":entries.iter().map(|e| format!("m{}s{}",e.tile_rows,e.stages)).collect::<Vec<_>>(),
        "modules":fused::served_modules(&all)
    });
    if timed {
        for buffer in &b {
            reset(&a, buffer, input, rows, rows, false)?;
        }
        for (name, cold) in [("warm", false), ("cold", true)] {
            let sampled = std::cell::Cell::new(false);
            record[name] = timing::arms(s, cold, arms, |arm, stream| {
                // The first callback is untimed warmup, with eviction storage live.
                if !sampled.replace(true) {
                    let free = s.context.memory_info().map_err(|e| e.to_string())?.0;
                    minimum_free.set(minimum_free.get().min(free));
                }
                graphs[arm].launch(stream).map_err(|e| e.to_string())
            })?;
        }
    }
    record["extra_device_bytes"] = json!(
        s.metadata["memory_free_bytes"]
            .as_u64()
            .unwrap()
            .saturating_sub(minimum_free.get())
    );
    println!("ENGRAM_WIDE_CASE {record}");
    Ok(())
}

#[test]
fn packed_matrix_candidates_preserve_admission_and_exact_rows() {
    for rows in ROWS.into_iter().chain(WIDE_ROWS) {
        for entry in entries(rows) {
            let mut a = Arena::new(None);
            let b = common(&mut a, &entry.base, rows, &data::empty(&entry.base, rows)).unwrap();
            let calls = matrix(&entry, &b, rows).unwrap();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].symbol, entry.admission_symbol());
            assert!(calls[1].symbol.ends_with("_packed_v1"));
            assert_eq!(calls[1].plan.grid, [200, rows.div_ceil(entry.tile_rows), 1]);
            assert!(
                entry
                    .prepare(rows)
                    .unwrap()
                    .launches(&b, View::EMPTY, Counts::rows(rows + 1))
                    .is_err()
            );
        }
    }
}

#[test]
#[ignore = "requires sealed Engram subgraph inputs and assigned GPUs"]
fn gpu_pair() {
    assert!(
        is_decode(),
        "the baseline must use the served decode selector"
    );
    let bytes = Box::leak(
        std::fs::read(root().join("cubins/probe_sm120.cubin"))
            .unwrap()
            .into_boxed_slice(),
    );
    let cubins = kernel_harness::Cubins::new(
        Box::leak(Box::new([("probe_sm120", &bytes[..])])),
        &[(
            "probe_sm120",
            &["chop_probe_read_v1", "chop_probe_flush_v1"],
        )],
    );
    let ran = kernel_harness::gpu_test_with(cubins, 6 << 30, |s| {
        let mode = std::env::var("CHOPINFER_ENGRAM_WIDE_MODE").map_err(|e| e.to_string())?;
        assert!(matches!(mode.as_str(), "time" | "memcheck"));
        let mem = mode == "memcheck";
        let rows = if mem { vec![33, 64] } else { ROWS.to_vec() };
        let captures = captured()?;
        let mut cases = 0;
        for rows in rows {
            check(s, &data::synthetic(&entries(rows)[0].base, rows), !mem)?;
            cases += 1;
            for capture in &captures {
                for tail in if mem { vec![true] } else { vec![false, true] } {
                    check(s, &slice(capture, rows, tail), !mem)?;
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, if mem { 6 } else { 35 });
        println!("ENGRAM_WIDE_DONE {}", json!({"cases":cases,"mode":mode}));
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some(), "an idle GPU is required");
}

/// The served wide verify tile against the fold it replaces, bitwise, at
/// every row count of `WIDE_ROWS`: live and rejected rows, eager and two
/// graph replays, and the omit-first control. Synthetic inputs always;
/// the captured Engram inputs' first and last 192 rows too when
/// `CHOPINFER_FOLD_CORE_ROOT` holds them. Untimed.
#[test]
#[ignore = "requires assigned GPUs"]
fn gpu_wide_verify() {
    let ran = kernel_harness::gpu_test(6 << 30, |s| {
        let captures = match std::env::var_os("CHOPINFER_FOLD_CORE_ROOT") {
            Some(_) => captured()?,
            None => Vec::new(),
        };
        let mut cases = 0;
        for rows in WIDE_ROWS {
            check(s, &data::synthetic(&entries(rows)[0].base, rows), false)?;
            cases += 1;
            for capture in &captures {
                for tail in [false, true] {
                    check(s, &slice(capture, rows, tail), false)?;
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, WIDE_ROWS.len() * (1 + 2 * captures.len()));
        println!(
            "ENGRAM_WIDE_VERIFY_DONE {}",
            json!({"cases":cases,"captured":captures.len()})
        );
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some(), "an idle GPU is required");
}
