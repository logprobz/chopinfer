use super::*;
use kernel_harness::poison::{LiveProducer, Rewrites};

#[allow(clippy::too_many_arguments)]
fn compare(
    s: &Session,
    a: &Arena<'_>,
    f: &BTreeMap<String, Function>,
    ops: &[Vec<Launch>; 2],
    b: &[Buffers<View>; 2],
    rows: u32,
    live: u32,
    bad: bool,
) -> LabResult<Vec<u8>> {
    for (calls, b) in ops.iter().zip(b) {
        reset(a, b, rows, live, bad)?;
        issue(s, a, f, calls, &s.stream)?;
    }
    equality(a, &b[0], &b[1], rows, live, bad)
}
fn chunks(
    s: &Session,
    a: &mut Arena<'_>,
    e: &Entry,
    weights: [View; 2],
    codes: &[u8],
    scales: &[u8],
    full: &[u8],
) -> LabResult<u32> {
    let width = e.base.query.activation_row_stride as usize;
    let output = (e.base.query.groups * e.base.query.n * 4) as usize;
    let keep = a.next;
    let mut count = 0;
    for partition in [
        vec![2048, 2048, 2048, 2048],
        vec![2048, 2048, 2048, 1983, 65],
        vec![2048, 2048, 2048, 1791, 257],
    ] {
        let mut first = 0;
        for rows in partition {
            let old = buffers(a, e, rows, weights)?;
            a.write(
                old.activations,
                &codes[first * width..(first + rows as usize) * width],
            )?;
            a.write(
                old.activation_scales,
                &scales[first * width / 32..(first + rows as usize) * width / 32],
            )?;
            let new = Buffers {
                output: a.filled(old.output.bytes, 255)?,
                status: a.filled(old.status.bytes, 0)?,
                ..old
            };
            let b = [old, new];
            let ops = [
                prepare(a, e, rows, &b[0], false)?,
                prepare(a, e, rows, &b[1], true)?,
            ];
            let f = functions(s, &ops.concat())?;
            let got = compare(s, a, &f, &ops, &b, rows, rows, false)?;
            if got != full[first * output..(first + rows as usize) * output] {
                return Err(format!("chunk bits differ at {first}+{rows}"));
            }
            first += rows as usize;
            count += 1;
            a.buffers.retain(|&h, _| h <= keep);
            a.next = keep;
        }
        assert_eq!(first, 8192);
    }
    Ok(count)
}
#[allow(clippy::too_many_arguments)]
fn pdl(
    s: &Session,
    a: &mut Arena<'_>,
    f: &BTreeMap<String, Function>,
    ops: &[Launch],
    b: &[Buffers<View>; 2],
    rows: u32,
    codes: &[u8],
    scales: &[u8],
) -> LabResult<()> {
    let staged = [a.bytes(codes)?, a.bytes(scales)?];
    let rewrites = Rewrites::new(
        s,
        &[
            (
                &a.buffers[&staged[0].handle],
                &a.buffers[&b[1].activations.handle],
            ),
            (
                &a.buffers[&staged[1].handle],
                &a.buffers[&b[1].activation_scales.handle],
            ),
        ],
    )?;
    let module = s.load_cubin(bundle_deepseek41::sm120::dependency_probes().0)?;
    let producer = LiveProducer::with_function(
        s,
        module
            .function(kernels_sm120::probes::LIVE)
            .map_err(|e| e.to_string())?,
    )?;
    reset(a, &b[1], rows, rows, false)?;
    producer.launch_rewriting(s, &a.buffers[&b[1].live_rows.handle], &rewrites)?;
    issue(s, a, f, ops, &s.stream)?;
    equality(a, &b[0], &b[1], rows, rows, false)?;
    Ok(())
}
pub fn run(s: &Session, timed: bool) -> LabResult<()> {
    let selected = kernel_harness::selected_ordinals()?.ok_or("GPU ordinal required")?;
    if selected.len() != 1 {
        return Err("each dense worker requires one GPU ordinal".into());
    }
    let rows_list = if timed { TIMED } else { ROWS };
    let mut count = 0;
    for e in family::entries() {
        let mut a = Arena::new(Some(s));
        let size = e.base.bytes(1);
        let weights = [
            a.bytes(&seed(size.weights as usize, false, 901))?,
            a.bytes(&seed(size.weight_scales as usize, true, 1901))?,
        ];
        let keep = a.next;
        for &rows in rows_list {
            let old = buffers(&mut a, &e, rows, weights)?;
            let codes = seed(old.activations.bytes as usize, false, 1771);
            let scales = seed(old.activation_scales.bytes as usize, true, 2771);
            a.write(old.activations, &codes)?;
            a.write(old.activation_scales, &scales)?;
            let new = Buffers {
                output: a.filled(old.output.bytes, 255)?,
                status: a.filled(old.status.bytes, 0)?,
                ..old
            };
            let b = [old, new];
            let ops = [
                prepare(&mut a, &e, rows, &b[0], false)?,
                prepare(&mut a, &e, rows, &b[1], true)?,
            ];
            let mut all = ops.concat();
            let mut control = ops[1].clone();
            control[1].symbol.push_str("_omit_first_v1");
            if !timed && [16, 65, 129].contains(&rows) {
                all.extend(control.clone());
            }
            let f = functions(s, &all)?;
            let mut record = json!({"n":e.base.query.n,"k":e.base.query.k,"groups":e.base.query.groups,"stages":e.stages,"rows":rows,"topologies":topology(&e),"bitwise":true,"serving_enabled":false,
                "baseline_symbols":ops[0].iter().map(|o|&o.symbol).collect::<Vec<_>>(),"candidate_symbols":ops[1].iter().map(|o|&o.symbol).collect::<Vec<_>>()});
            if timed {
                for buf in &b {
                    reset(&a, buf, rows, rows, false)?;
                }
                let graphs = ops
                    .iter()
                    .map(|op| s.capture(|stream| issue(s, &a, &f, op, stream)))
                    .collect::<LabResult<Vec<_>>>()?;
                for graph in &graphs {
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                }
                equality(&a, &b[0], &b[1], rows, rows, false)?;
                record["timing"] = timing::pair(s, |arm, stream| {
                    graphs[arm].launch(stream).map_err(|e| e.to_string())
                })?;
                record["graph_replays_per_arm"] = json!(19);
                record["compulsory_weight_bytes"] = json!(size.weights + size.weight_scales);
                println!("compact_dense_timing {record}");
            } else {
                let full = compare(s, &a, &f, &ops, &b, rows, rows, false)?;
                let graph = s.capture(|stream| issue(s, &a, &f, &ops[1], stream))?;
                for mode in 0..2 {
                    reset(&a, &b[1], rows, rows, false)?;
                    if mode == 0 {
                        issue(s, &a, &f, &ops[1], &s.stream)?;
                    } else {
                        graph.launch(&s.stream).map_err(|e| e.to_string())?;
                    }
                    equality(&a, &b[0], &b[1], rows, rows, false)?;
                }
                record["poisoned_eager_and_graph"] = json!(true);
                if [16, 65, 129].contains(&rows) {
                    pdl(s, &mut a, &f, &ops[1], &b, rows, &codes, &scales)?;
                    reset(&a, &b[1], rows, rows, false)?;
                    issue(s, &a, &f, &control, &s.stream)?;
                    if a.read(b[1].output)? == full {
                        return Err("omit-first control was not detected".into());
                    }
                    record["pdl_input_and_live"] = json!(true);
                    record["negative_control_caught"] = json!(true);
                }
                if rows == 257 {
                    let mut bad_codes = codes.clone();
                    bad_codes[e.base.query.activation_row_stride as usize] = 127;
                    let mut bad_scales = scales.clone();
                    bad_scales[2 * e.base.query.activation_row_stride as usize / 32] = 255;
                    a.write(b[0].activations, &bad_codes)?;
                    a.write(b[0].activation_scales, &bad_scales)?;
                    for live in [rows, 129, 0, rows] {
                        compare(s, &a, &f, &ops, &b, rows, live, true)?;
                        reset(&a, &b[1], rows, live, true)?;
                        graph.launch(&s.stream).map_err(|e| e.to_string())?;
                        equality(&a, &b[0], &b[1], rows, live, true)?;
                    }
                    record["refusal_live_sequence"] = json!([rows, 129, 0, rows]);
                }
                if rows == 8192 {
                    record["chunk_checks"] =
                        json!(chunks(s, &mut a, &e, weights, &codes, &scales, &full)?);
                }
                println!("compact_dense_bitwise {record}");
            }
            count += 1;
            a.buffers.retain(|&h, _| h <= keep);
            a.next = keep;
        }
    }
    if count != family::entries().len() * rows_list.len() {
        return Err("incomplete dense matrix".into());
    }
    println!(
        "compact_dense_complete {}",
        json!({"kind":if timed{"timing"}else{"bitwise"},"cases":count,"serving_enabled":false})
    );
    Ok(())
}
