//! Used-depth head packing against the full five-row projection. This lab
//! checks one TP4 shard; served parity covers the full collective program.
use bundle_deepseek41::sm120::{self, fold_cubin_for_symbol};
use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::DType;
use kernel_core::precision::buffers::{ProjectionBuffers, ProjectionScratch, QuantizedWeights};
use kernel_core::row_copy::{GatherByIndex, GatherByIndexBuffers};
use kernel_harness::eviction::EvictionMode;
use kernel_harness::kit::cases::f32_projection;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{Cubins, LabResult, Session};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{DeviceBuffer, Module, Stream};
use std::collections::BTreeMap;

#[path = "dspark_depth/prefix.rs"]
mod prefix;

const POISON: u32 = 0xa5a5_a5a5;

fn upload(s: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let b = s.allocate(bytes.len())?;
    b.copy_from_bytes(bytes, &s.stream)
        .map_err(|e| e.to_string())?;
    Ok(b)
}
fn words(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}
fn read(s: &Session, b: &DeviceBuffer) -> LabResult<Vec<u32>> {
    Ok(s.download_f32(b, b.byte_len() / 4)?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}
fn modules(s: &Session, plans: &[Launch]) -> LabResult<Vec<Module>> {
    let mut sources = BTreeMap::new();
    for p in plans {
        let (name, bytes) = fold_cubin_for_symbol(&p.symbol)
            .ok_or_else(|| format!("missing embedded depth-lab symbol {}", p.symbol))?;
        sources.insert(name, bytes);
    }
    sources.into_values().map(|b| s.load_cubin(b)).collect()
}
fn run(
    s: &Session,
    modules: &[Module],
    plans: &[Launch],
    bindings: &Bindings<'_>,
    stream: &Stream,
) -> LabResult<()> {
    for p in plans {
        let f = modules
            .iter()
            .find_map(|m| m.function(&p.symbol).ok())
            .ok_or_else(|| format!("missing depth-lab function {}", p.symbol))?;
        launch(&s.context, &f, stream, p.plan, &p.arguments, bindings)?;
    }
    Ok(())
}

fn probes() -> LabResult<Cubins> {
    let Some(path) = std::env::var_os("CHOPINFER_DSPARK_DEPTH_PROBE") else {
        return Ok(Cubins::NONE);
    };
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let lock: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../crates/kernel-lab/kernels.lock"
    ))
    .map_err(|e| e.to_string())?;
    let record = lock
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "probe_sm120")
        .ok_or("missing timing probe lock")?;
    if record["cubin_sha256"] != kernel_build::digest(&bytes) {
        return Err("depth timing probe differs from its lock".into());
    }
    let bytes = Box::leak(bytes.into_boxed_slice());
    Ok(Cubins::new(
        Box::leak(Box::new([("probe_sm120", &bytes[..])])),
        &[(
            "probe_sm120",
            &[
                "chop_probe_gate_v1",
                "chop_probe_gate_release_v1",
                "chop_probe_flush_v1",
                "chop_probe_read_v1",
            ],
        )],
    ))
}

#[test]
fn gpu_used_depth_head_is_bitwise_and_keeps_unused_rows_poisoned() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("the used-depth head comparison needs an idle GPU").unwrap();
        return;
    }
    let timing = std::env::var_os("CHOPINFER_DSPARK_DEPTH_PROBE").is_some();
    let result = kernel_harness::gpu_test_with(probes().unwrap(), 2 << 30, |s| {
        let k = FLASH.shape.hidden as usize;
        let n = (FLASH.shape.vocab / 4) as usize;
        let backend = sm120::register_dspark_f32_projections(
            Sm120::new().with_row_gather(kernels_sm120::row::gather::SYMBOL), &FLASH, 4)?;
        let entry = sm120::f32_head::entry(&FLASH, 4)?;
        let projection = backend.projection(&f32_projection::operation(&entry), entry.usage(), Need::rows(40))
            .map_err(|e| e.to_string())?.handle;
        let gather = backend.gather_by_index(&GatherByIndex {
            dtype: DType::F32, width: k as u32, alignment_bytes: 16,
        }, Need::rows(24)).map_err(|e| e.to_string())?;
        // A repeating bank keeps fixture creation bounded without sampling columns.
        let bank: Vec<_> = (0..17*k).map(|i| f32_to_bf16(((i*31)%251) as f32 / 131.7 - 0.94)).collect();
        let mut weight_bytes = Vec::with_capacity(n*k*2);
        for col in 0..n { weight_bytes.extend(bank[col%17*k..][..k].iter().flat_map(|w| w.to_le_bytes())); }
        let weight = upload(s, &weight_bytes)?;
        drop(weight_bytes);
        let mut cases = 0;
        let mut prefixes = 0;
        for items in 3..=8_u32 {
            let rows = items * 5;
            let packed_rows = items * 3;
            let input = s.allocate((rows as usize + 1) * k * 4)?;
            let compact = s.allocate((packed_rows as usize + 1) * k * 4)?;
            let outputs = [s.allocate((rows as usize + 1) * n * 4)?, s.allocate((packed_rows as usize + 1) * n * 4)?];
            let live = [upload(s, &rows.to_le_bytes())?, upload(s, &packed_rows.to_le_bytes())?];
            let indices: Vec<_> = (0..3).flat_map(|p| (0..items).map(move |i| 5*i+p)).collect();
            let indices_gpu = upload(s, &words(indices.iter().copied()))?;
            let domain = s.allocate(projection.info().bytes_of("scratch.weight_domain").unwrap() as usize)?;
            let workspace = s.allocate(projection.info().workspace as usize)?;
            let bindings = Bindings { buffers: BTreeMap::from([
                (1, &input), (2, &weight), (3, &outputs[0]), (4, &compact), (5, &outputs[1]),
                (6, &live[0]), (7, &live[1]), (8, &indices_gpu), (9, &domain), (10, &workspace),
            ]) };
            let view = |handle| View { handle, bytes: bindings.buffers[&handle].byte_len() as u64 };
            let roles = |packed| ProjectionBuffers {
                activation: view(if packed {4} else {1}), live_rows: view(if packed {7} else {6}),
                output: view(if packed {5} else {3}), residual: None,
                weights: QuantizedWeights { codes: view(2), local_scales: None, global_scales: None },
                activation_global: None, routing: None,
                scratch: ProjectionScratch {
                    weight_domain: Some(view(9)), partial: View::EMPTY, semaphores: View::EMPTY,
                    unused: View::EMPTY, temporary: None, gathered: None, codes: None, scales: None,
                    activation_global: None, row_amax: None, output_codes: None, output_scales: None,
                    quantizer_status: None, quantizer_status_row_offset: 0,
                }, row_offset: 0, shard_rows: 0,
            };
            let reference = projection.launches(&roles(false), view(10), Counts::rows(rows)).map_err(|e| e.to_string())?;
            let mut packed = gather.launches(&GatherByIndexBuffers {
                source: view(1), row_indices: view(8), count: view(7), destination: view(4),
            }, View::EMPTY, Counts::rows(packed_rows)).map_err(|e| e.to_string())?;
            packed.extend(projection.launches(&roles(true), view(10), Counts::rows(packed_rows)).map_err(|e| e.to_string())?);
            let init = projection.initialize(&roles(false)).map_err(|e| e.to_string())?;
            let all: Vec<_> = reference.iter().chain(&packed).chain(&init).cloned().collect();
            let modules = modules(s, &all)?;
            run(s, &modules, &init, &bindings, &s.stream)?;
            let plans = [reference, packed];
            let graphs = plans.iter().map(|p| s.capture(|stream| run(s, &modules, p, &bindings, stream))).collect::<LabResult<Vec<_>>>()?;
            let mut clean_prefix = None;
            for pattern in 0..3 {
                let mut x: Vec<u32> = (0..(rows as usize+1)*k).map(|i| {
                    let v = ((i*13 + i/k*127)%997) as f32 / 251.13 - 1.73;
                    if pattern == 1 && i%k == k-1 { 1 } else { v.to_bits() }
                }).collect();
                if pattern == 2 {
                    for i in 0..items as usize { x[(i*5+3)*k..(i*5+5)*k].fill(0x7fc0_1234); }
                }
                input.copy_from_bytes(&words(x.iter().copied()), &s.stream).map_err(|e| e.to_string())?;
                for graph in [false, true] {
                    compact.copy_from_bytes(&words(std::iter::repeat_n(POISON, compact.byte_len()/4)), &s.stream).map_err(|e| e.to_string())?;
                    for output in &outputs { output.copy_from_bytes(&words(std::iter::repeat_n(POISON, output.byte_len()/4)), &s.stream).map_err(|e| e.to_string())?; }
                    for arm in 0..2 {
                        workspace.copy_from_bytes(&vec![0xa5; workspace.byte_len()], &s.stream).map_err(|e| e.to_string())?;
                        if graph { graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?; }
                        else { run(s, &modules, &plans[arm], &bindings, &s.stream)?; }
                    }
                    s.stream.synchronize().map_err(|e| e.to_string())?;
                    let gathered = read(s, &compact)?;
                    for (to, &from) in indices.iter().enumerate() {
                        if gathered[to*k..][..k] != x[from as usize*k..][..k] { return Err("head row gather changed bits".into()); }
                    }
                    if gathered[packed_rows as usize*k..].iter().any(|&w| w != POISON) { return Err("head gather wrote beyond used rows".into()); }
                    let full = read(s, &outputs[0])?;
                    let short = read(s, &outputs[1])?;
                    for (to, &from) in indices.iter().enumerate() {
                        if short[to*n..][..n] != full[from as usize*n..][..n] {
                            let col = short[to*n..][..n].iter().zip(&full[from as usize*n..][..n]).position(|(a,b)| a != b).unwrap();
                            return Err(format!("items={items} pattern={pattern} graph={graph} packed row={to} col={col} differs"));
                        }
                    }
                    for (out, live_rows) in [(&full, rows), (&short, packed_rows)] {
                        if out[live_rows as usize*n..].iter().any(|&w| w != POISON) { return Err("head projection wrote beyond live rows".into()); }
                    }
                    if pattern == 0 { clean_prefix = Some(short[..packed_rows as usize*n].to_vec()); }
                    if pattern == 2 && clean_prefix.as_deref() != Some(&short[..packed_rows as usize*n]) {
                        return Err("unused backbone rows affected used head logits".into());
                    }
                    if graph {
                        prefixes += prefix::compare(s, items, &x, &full, &short)?;
                    }
                    cases += 1;
                }
            }
            // Restore identical finite operands before timing either arm.
            let x: Vec<_> = (0..(rows as usize+1)*k).map(|i| (((i*13)%997) as f32 / 251.13 - 1.73).to_bits()).collect();
            input.copy_from_bytes(&words(x), &s.stream).map_err(|e| e.to_string())?;
            if timing && [4,8].contains(&items) {
                let aa = kernel_harness::with_timing_label(&format!("depth-head-aa-items-{items}"), || kernel_harness::time_ab(s, 2, 11, [EvictionMode::Read; 2], |_, stream| graphs[0].launch(stream).map_err(|e| e.to_string())))?;
                let ab = kernel_harness::with_timing_label(&format!("depth-head-ab-items-{items}"), || kernel_harness::time_ab(s, 2, 11, [EvictionMode::Read; 2], |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string())))?;
                println!("DSPARK_DEPTH_TIMING {}", serde_json::json!({"items":items,"aa_ratio":aa.paired_ratio,"aa_us":[aa.arms[0].median_ms*1000.,aa.arms[1].median_ms*1000.],"ratio":ab.paired_ratio,"us":[ab.arms[0].median_ms*1000.,ab.arms[1].median_ms*1000.]}));
            }
        }
        assert_eq!(cases, 36);
        s.receipt("deepseek41-dspark-used-depth", BTreeMap::from([
            ("bitwise".into(), serde_json::json!(true)),
            ("cases".into(), serde_json::json!(cases)),
            ("prefix_cases".into(), serde_json::json!(prefixes)),
            ("timing".into(), serde_json::json!(timing)),
        ]))?;
        println!("DSPARK_DEPTH_COMPLETE cases={cases} prefix_cases={prefixes} timing={timing}");
        Ok(())
    }).unwrap();
    assert!(result.is_some(), "the depth comparison ran");
}
