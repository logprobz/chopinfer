//! Wide cached stages against split outputs and legacy cached chain partials.
use super::*;
use kernels_sm120::f32_projection::cached_spans::{Buffers, Candidate};

fn buffers(view: impl Fn(u64) -> View, output: u64) -> ProjectionBuffers<View> {
    ProjectionBuffers {
        activation: view(1),
        weights: QuantizedWeights {
            codes: view(2),
            local_scales: None,
            global_scales: None,
        },
        output: view(output),
        live_rows: view(5),
        residual: None,
        activation_global: None,
        routing: None,
        scratch: ProjectionScratch {
            weight_domain: Some(view(6)),
            partial: View::EMPTY,
            semaphores: View::EMPTY,
            unused: View::EMPTY,
            temporary: None,
            gathered: None,
            codes: None,
            scales: None,
            activation_global: None,
            row_amax: None,
            output_codes: None,
            output_scales: None,
            quantizer_status: None,
            quantizer_status_row_offset: 0,
        },
        row_offset: 0,
        shard_rows: 0,
    }
}

fn partial_reference(
    s: &Session,
    old: &Entry,
    weights: &ComparisonWeights,
    rows: u32,
    a: &[f32],
    actual: &[u8],
) -> LabResult<()> {
    let q = old.query();
    let legacy = old.entry_for_rows(1);
    for first in (0..rows).step_by(32) {
        let count = (rows - first).min(32);
        let x = s.upload_f32(&a[(first * q.k) as usize..((first + count) * q.k) as usize])?;
        let live = s.allocate(4)?;
        live.copy_from_bytes(&count.to_le_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
        let out = s.allocate((count * q.n * 4) as usize)?;
        let work = s.allocate(legacy.runtime_workspace_bytes(count) as usize)?;
        let bindings = launch::Bindings {
            buffers: BTreeMap::from([
                (1, &x),
                (2, &weights.weight),
                (3, &out),
                (5, &live),
                (6, &weights.domain),
                (7, &work),
            ]),
        };
        let view = |handle| View {
            handle,
            bytes: bindings.buffers[&handle].byte_len() as u64,
        };
        let op = prepared(&legacy, count);
        let calls = op
            .launches(&buffers(view, 3), view(7), Counts::rows(count))
            .map_err(|e| e.to_string())?;
        run_launches(
            s,
            &weights.modules.iter().collect::<Vec<_>>(),
            &bindings,
            &calls,
            &s.stream,
        )?;
        let reference = poison::read(s, &work)?;
        for chain in 0..q.k.div_ceil(128) {
            let start = 128 + (chain * count * q.n * 4) as usize;
            let offset = ((chain * rows + first) * q.n * 4) as usize;
            let len = (count * q.n * 4) as usize;
            if reference[start..start + len] != actual[offset..offset + len] {
                return Err(format!(
                    "cached chain differs n={} rows={rows} first={first} chain={chain}",
                    q.n
                ));
            }
        }
    }
    Ok(())
}

fn compare_spans(
    s: &Session,
    old: &Entry,
    c: &Candidate,
    weights: &ComparisonWeights,
    rows: u32,
    a: &[f32],
    witness: Witness,
) -> LabResult<()> {
    let q = old.query();
    let pattern = format!("{witness:?}");
    let measure = witness == Witness::General;
    let x = s.upload_f32(a)?;
    let live = s.allocate(4)?;
    live.copy_from_bytes(&rows.to_le_bytes(), &s.stream)
        .map_err(|e| e.to_string())?;
    let output = [
        s.allocate((rows * q.n * 4) as usize)?,
        s.allocate((rows * q.n * 4) as usize)?,
    ];
    let work = s.allocate(old.runtime_workspace_bytes(rows) as usize)?;
    let row_domain = s.allocate(512)?;
    let bytes = c.partial_bytes(rows).map_err(|e| e.to_string())? as usize;
    let partials = s.allocate(bytes + 256)?;
    let bindings = launch::Bindings {
        buffers: BTreeMap::from([
            (1, &x),
            (2, &weights.weight),
            (3, &output[0]),
            (4, &output[1]),
            (5, &live),
            (6, &weights.domain),
            (7, &work),
            (9, &row_domain),
            (10, &partials),
        ]),
    };
    let view = |handle| View {
        handle,
        bytes: bindings.buffers[&handle].byte_len() as u64,
    };
    let op = prepared(old, rows);
    let module = s.load_cubin(
        fold_cubin_for_symbol(&c.symbols()[0])
            .ok_or("wide cached cubin absent")?
            .1,
    )?;
    let mut modules = weights.modules.iter().collect::<Vec<_>>();
    modules.push(&module);
    if !weights.initialized.get() {
        run_launches(
            s,
            &modules,
            &bindings,
            &op.initialize(&buffers(view, 3))
                .map_err(|e| e.to_string())?,
            &s.stream,
        )?;
        let flags = poison::read(s, &weights.domain)?;
        for &(tile, expected) in &weights.expected_flags {
            if u32::from_le_bytes(flags[tile * 4..tile * 4 + 4].try_into().unwrap()) != expected {
                return Err("weight-domain classification differs".into());
            }
        }
        weights.initialized.set(true);
    }
    let calls = [
        op.launches(&buffers(view, 3), view(7), Counts::rows(rows))
            .map_err(|e| e.to_string())?,
        c.stages(
            Buffers {
                activation: view(1),
                weight: view(2),
                row_domain: view(9),
                partials: view(10),
                live_rows: view(5),
                weight_domain: view(6),
                output: view(4),
            },
            rows,
        )
        .map_err(|e| e.to_string())?
        .into_launches()
        .into(),
    ];
    let graphs = calls
        .iter()
        .map(|calls| s.capture(|stream| run_launches(s, &modules, &bindings, calls, stream)))
        .collect::<LabResult<Vec<_>>>()?;
    let mut chain_checked = false;
    for count in [rows, rows - 1, 0, rows] {
        live.copy_from_bytes(&count.to_le_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
        for captured in [false, true, true] {
            poison::poison(s, &[&output[0], &output[1], &work, &row_domain, &partials])?;
            for (arm, graph) in graphs.iter().enumerate() {
                if captured {
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                } else {
                    run_launches(s, &modules, &bindings, &calls[arm], &s.stream)?;
                }
            }
            let expected = poison::read(s, &output[0])?;
            let actual = poison::read(s, &output[1])?;
            if actual != expected {
                return Err(format!(
                    "wide F32 output differs n={} rows={rows} live={count} pattern={pattern} graph={captured}",
                    q.n
                ));
            }
            if actual[(count * q.n * 4) as usize..]
                .iter()
                .any(|&b| b != poison::POISON)
            {
                return Err("inactive output was written".into());
            }
            let flags = poison::read(s, &row_domain)?;
            for row in 0..rows {
                let expected = u32::from(
                    row >= count
                        || a[(row * q.k) as usize..((row + 1) * q.k) as usize]
                            .iter()
                            .all(|&v| cases::tensor_operand(v)),
                );
                if u32::from_le_bytes(
                    flags[row as usize * 4..row as usize * 4 + 4]
                        .try_into()
                        .unwrap(),
                ) != expected
                {
                    return Err("row-domain classification differs".into());
                }
            }
            if flags[rows as usize * 4..]
                .iter()
                .any(|&b| b != poison::POISON)
            {
                return Err("row-domain guard changed".into());
            }
            let observed = poison::read(s, &partials)?;
            if observed[bytes..].iter().any(|&b| b != poison::POISON) {
                return Err("partial guard changed".into());
            }
            for chain in 0..q.k.div_ceil(128) {
                let first = ((chain * rows + count) * q.n * 4) as usize;
                let end = ((chain + 1) * rows * q.n * 4) as usize;
                if observed[first..end].iter().any(|&b| b != poison::POISON) {
                    return Err("inactive chain was written".into());
                }
            }
            if count == rows && !chain_checked {
                partial_reference(s, old, weights, rows, a, &observed)?;
                chain_checked = true;
            }
            let mut changed = actual.clone();
            changed[0] ^= 1;
            if changed == expected {
                return Err("output byte negative control survived".into());
            }
        }
    }
    if measure {
        let result = kernel_harness::time_ab(s, 2, 7, [EvictionMode::Write; 2], |arm, stream| {
            graphs[arm].launch(stream).map_err(|e| e.to_string())
        })?;
        println!(
            "F32_SPANS_TIMING n={} k={} rows={rows} old_us={} new_us={} ratio={}",
            q.n,
            q.k,
            result.arms[0].median_ms * 1000.,
            result.arms[1].median_ms * 1000.,
            result.paired_ratio
        );
    }
    println!(
        "F32_SPANS_BITWISE n={} k={} rows={rows} pattern={pattern} partials=true PASS",
        q.n, q.k
    );
    Ok(())
}

#[test]
#[ignore = "requires frozen F32 timing probes and an allocated GPU pair"]
fn gpu_wide_cached_spans() {
    use bundle_deepseek41::sm120::f32_verify;
    use model_deepseek41_contract::geometry::FLASH;
    let ran = kernel_harness::gpu_test_with(timing_cubins(), 4 << 30, |s| {
        let mut count = 0;
        let mut timings = 0;
        for c in f32_verify::entries().map_err(|e| e.to_string())? {
            let q = c.entry().query();
            let old = f32_split::entries(&FLASH, 4)
                .unwrap()
                .into_iter()
                .find(|e| e.query().n == q.n && e.query().k == q.k)
                .ok_or("baseline shape missing")?;
            for witness in EDGE_WITNESSES {
                let (a, b) = cases::operands(c.entry(), witness);
                let weights = ComparisonWeights::routed(s, &old, old.clone(), &b)?;
                for rows in [33, 40, 48, 56, 63, 64] {
                    let measure = witness == Witness::General;
                    compare_spans(
                        s,
                        &old,
                        &c,
                        &weights,
                        rows,
                        &a[..(rows * q.k) as usize],
                        witness,
                    )?;
                    count += 1;
                    timings += usize::from(measure);
                }
            }
        }
        assert_eq!(count, 330);
        assert_eq!(timings, 30);
        println!("F32_SPANS_COMPLETE cases={count} timings={timings}");
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some(), "an idle GPU is required");
}
