use super::*;
pub(super) fn source(
    a: &mut Arena<'_>,
    case: Case,
    live: View,
) -> LabResult<(View, View, Vec<Launch>)> {
    use kernel_core::precision::quantizer::BlockQuantizeBuffers;
    let input = a.filled(u64::from(case.rows) * 5120, 0)?;
    let scales = a.filled(u64::from(case.rows) * 160, 0)?;
    let status = a.filled(u64::from(case.rows) * 4, 0)?;
    let values = if a.session.is_some() {
        let mut seed = 0xc837_6251_aa19_46d3u64;
        let mut uniform = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 32) as u32 as f64 + 0.5) / (u32::MAX as f64 + 1.)
        };
        (0..u64::from(case.rows) * 5120)
            .flat_map(|i| {
                let normal =
                    (-2. * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
                let value = normal
                    * (0.5 + (i / 32 % 5) as f64 * 0.25)
                    * if i % 997 == 0 { 6. } else { 1. };
                kernel_core::codec::f32_to_bf16(value as f32).to_le_bytes()
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let bf16 = if a.session.is_some() {
        a.bytes(&values)?
    } else {
        a.filled(u64::from(case.rows) * 5120 * 2, 0)?
    };
    let backend = registration::quantizers(&FLASH, 4)?
        .into_iter()
        .fold(Sm120::new(), Sm120::with_block_quantize);
    let q = backend
        .block_quantize(&moe::quantizer(5120), Need::rows(case.rows))
        .map_err(|e| e.to_string())?;
    let ops = q
        .launches(
            &BlockQuantizeBuffers {
                input: bf16,
                output: input,
                scales: Some(scales),
                live_rows: live,
                status,
                status_row_offset: 0,
            },
            View::EMPTY,
            Counts::rows(case.rows),
        )
        .map_err(|e| e.to_string())?;
    Ok((input, scales, ops))
}
const ENV: &str = "CHOPINFER_DEEPSEEK41_EXACT_PREFILL_CUBINS";
pub(super) fn equal(a: &Arena<'_>, case: Case, old: &Arm, new: &Arm, live: u32) -> LabResult<()> {
    for (i, (&left, &right)) in old.outputs.iter().zip(&new.outputs).enumerate() {
        let x = a.read(left)?;
        let y = a.read(right)?;
        if x != y {
            return Err(format!(
                "{} live={live} output={i}: bitwise mismatch at byte {:?}",
                case.label(),
                x.iter().zip(&y).position(|(x, y)| x != y)
            ));
        }
        let bytes = left.bytes as usize * live as usize / case.rows as usize;
        let data = &x[..bytes];
        let finite = match i {
            1 | 5 | 6 => data
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            2 => data
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80),
            3 => data.iter().all(|b| b & 0x7f != 0x7f),
            4 => !data.contains(&255),
            _ => true,
        };
        if !finite {
            return Err(format!("nonfinite or unwritten live output {i}"));
        }
    }
    for arm in [old, new] {
        if a.read(arm.status)?.iter().any(|&x| x != 0) {
            return Err("nonzero sticky quantizer status".into());
        }
    }
    Ok(())
}
pub(super) fn poison(a: &Arena<'_>, arm: &Arm) -> LabResult<()> {
    let s = &a.session.unwrap().stream;
    for v in &arm.poison {
        a.buffers[&v.handle]
            .copy_from_bytes(&vec![255; v.bytes as usize], s)
            .map_err(|e| e.to_string())?;
    }
    a.buffers[&arm.status.handle]
        .copy_from_bytes(&vec![0; arm.status.bytes as usize], s)
        .map_err(|e| e.to_string())
}
pub(super) fn clock_table(a: &Arena<'_>, arm: &Arm) -> LabResult<serde_json::Value> {
    let mut parts = serde_json::Map::new();
    for (part, view) in ["gateup", "down"].into_iter().zip(&arm.clocks) {
        let values = a
            .read(*view)?
            .chunks_exact(8)
            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        let queues=values.chunks_exact(family::CTAS as usize*8).enumerate().map(|(q,ctas)| {
   let mut cycles=[0_u64;7];let mut items=0;let mut active=0;let mut max=0;
   for row in ctas.chunks_exact(8) {if row[7]>0{active+=1;}items+=row[7];
    max=max.max(row[..7].iter().sum::<u64>());for i in 0..7{cycles[i]+=row[i];}}
   serde_json::json!({"rows":([8,32,64,128][q]),"cta_phase_cycles":cycles,"items":items,"active_ctas":active,"max_cta_cycles":max})
  }).collect::<Vec<_>>();
        parts.insert(part.into(), queues.into());
    }
    Ok(
        serde_json::json!({"phase_order":["setup_prologue","wait_barrier","decode_barrier","mma","fold_barrier","enqueue","epilogue_barrier"],"scope":"lane zero in every launched CTA; diagnostic kernels only; summed phase cycles are not exclusive wall time","parts":parts}),
    )
}
pub(super) fn utilization(
    a: &Arena<'_>,
    old: &Arm,
    v: family::Variant,
) -> LabResult<serde_json::Value> {
    let counts = a
        .read(old.route_counts)?
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut tiles = [0_u64; 4];
    let mut actual = 0;
    for n in counts {
        actual += u64::from(n);
        let full = n / v.rows;
        let tail = n % v.rows;
        tiles[if v.rows == 64 { 2 } else { 3 }] += u64::from(full);
        if tail > 0 {
            tiles[if tail <= 8 {
                0
            } else if tail <= 32 {
                1
            } else if tail <= 64 {
                2
            } else {
                3
            }] += 1;
        }
    }
    let allocated = tiles
        .iter()
        .zip([8, 32, 64, 128])
        .map(|(&n, r)| n * r)
        .sum::<u64>();
    Ok(
        serde_json::json!({"tiles":tiles,"live_routes":actual,"allocated_rows":allocated,"row_utilization":actual as f64/allocated as f64}),
    )
}
pub(super) fn run(session: &Session) -> LabResult<()> {
    let directory = std::path::PathBuf::from(std::env::var(ENV).map_err(|e| e.to_string())?);
    let bytes = std::fs::read(directory.join(format!("{}.cubin", family::CUBIN)))
        .map_err(|e| e.to_string())?;
    let candidate_module = session.load_cubin(&bytes)?;
    let mut embedded = BTreeMap::new();
    let mut functions = BTreeMap::<String, Function>::new();
    for set in moe::expert_sets(&FLASH) {
        let mut a = Arena {
            session: Some(session),
            next: 0,
            buffers: BTreeMap::new(),
        };
        let weights = [
            weights(&mut a, set.count as usize, 1152, 5120)?,
            weights(&mut a, set.count as usize, 5120, 576)?,
        ];
        let weights_keep = a.next;
        for case in cases().into_iter().filter(|c| c.set == set) {
            let live = a.bytes(&case.rows.to_le_bytes())?;
            let (input, scales, prefix) = source(&mut a, case, live)?;
            let inputs = Inputs {
                input,
                scales,
                live,
                weights,
                logits: a.bytes(&case.logits())?,
            };
            let old = arm(&mut a, case, inputs)?;
            let baseline_keep = a.next;
            for v in family::VARIANTS {
                let mut new = arm(&mut a, case, inputs)?;
                candidate(&mut a, case, &mut new, v)?;
                let arms = [&old, &new];
                let mut ops = prefix.clone();
                ops.extend(component(&old, "full"));
                ops.extend(component(&new, "full"));
                ops.extend(new.clock_launches.clone());
                let mut negative = component(&new, "full");
                if v.name == "pipe64" {
                    for op in &mut negative {
                        if op.symbol.contains("before_down_") && op.symbol.contains("_r") {
                            op.symbol = op.symbol.replace("_v1", "_omit_v1");
                        }
                    }
                    ops.extend(negative.clone());
                }
                for op in &ops {
                    if functions.contains_key(&op.symbol) {
                        continue;
                    }
                    let function = if op.symbol.starts_with("chop_exact_prefill") {
                        candidate_module
                            .function(&op.symbol)
                            .map_err(|e| e.to_string())?
                    } else {
                        let (name, bytes) =
                            bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol)
                                .ok_or_else(|| format!("missing {}", op.symbol))?;
                        if !embedded.contains_key(name) {
                            embedded.insert(name, session.load_cubin(bytes)?);
                        }
                        embedded[name]
                            .function(&op.symbol)
                            .map_err(|e| e.to_string())?
                    };
                    if op.plan.dynamic_shared_bytes > 0 {
                        function
                            .set_max_dynamic_shared_memory(op.plan.dynamic_shared_bytes)
                            .map_err(|e| e.to_string())?;
                    }
                    functions.insert(op.symbol.clone(), function);
                }
                let bindings = Bindings {
                    buffers: a.buffers.iter().map(|(&h, b)| (h, b)).collect(),
                };
                let issue = |ops: &[Launch], stream: &Stream| -> LabResult<()> {
                    for op in ops {
                        launch(
                            &session.context,
                            &functions[&op.symbol],
                            stream,
                            op.plan,
                            &op.arguments,
                            &bindings,
                        )?;
                    }
                    Ok(())
                };
                issue(&prefix, &session.stream)?;
                let graphs = arms
                    .iter()
                    .map(|arm| session.capture(|s| issue(&component(arm, "full"), s)))
                    .collect::<LabResult<Vec<_>>>()?;
                let mut replay_count = 0;
                // Reuse captured graphs with poisoned state, including empty and tail replays.
                for count in [case.rows, case.rows - 7, 0, case.rows] {
                    a.buffers[&live.handle]
                        .copy_from_bytes(&count.to_le_bytes(), &session.stream)
                        .map_err(|e| e.to_string())?;
                    for i in 0..2 {
                        poison(&a, arms[i])?;
                        graphs[i]
                            .launch(&session.stream)
                            .map_err(|e| e.to_string())?;
                    }
                    session.stream.synchronize().map_err(|e| e.to_string())?;
                    equal(&a, case, &old, &new, count)?;
                    replay_count += 1;
                }
                let mut times = serde_json::Map::new();
                for part in ["gateup", "down", "projections", "full"] {
                    let stages = arms
                        .iter()
                        .map(|arm| session.capture(|s| issue(&component(arm, part), s)))
                        .collect::<LabResult<Vec<_>>>()?;
                    times.insert(
                        part.into(),
                        timing::pair(session, |i, s| {
                            stages[i].launch(s).map_err(|e| e.to_string())
                        })?,
                    );
                }
                // Clock diagnostics run after performance timing and must also preserve bits.
                issue(&new.clock_launches, &session.stream)?;
                issue(&new.sum, &session.stream)?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                equal(&a, case, &old, &new, case.rows)?;
                let clocks = clock_table(&a, &new)?;
                let occupancy = utilization(&a, &old, v)?;
                let negative_caught = if v.name == "pipe64" {
                    let before = a.read(new.outputs[6])?;
                    issue(&negative, &session.stream)?;
                    session.stream.synchronize().map_err(|e| e.to_string())?;
                    if before == a.read(new.outputs[6])? {
                        return Err(
                            "omit-first-chain control escaped the final-output comparison".into(),
                        );
                    }
                    true
                } else {
                    false
                };
                println!(
                    "exact_prefill_result {}",
                    serde_json::json!({"case":case.label(),"experts":set.count,"top_k":set.top_k,"input_rows":case.rows,"routed_rows":case.rows*set.top_k,"routing":if case.skewed{"skewed"}else{"balanced"},"variant":v.name,"bitwise":true,"clock_bitwise":true,"boundary_replays":replay_count,"negative_control_caught":negative_caught,"serving_enabled":false,"timing":times,"clock":clocks,"tile_utilization":occupancy,"baseline":{"gateup":component(&old,"gateup").iter().map(|o|&o.symbol).collect::<Vec<_>>(),"down":component(&old,"down").iter().map(|o|&o.symbol).collect::<Vec<_>>()}})
                );
                drop(graphs);
                drop(bindings);
                a.buffers.retain(|&h, _| h <= baseline_keep);
            }
            a.buffers.retain(|&h, _| h <= weights_keep);
        }
    }
    Ok(())
}
