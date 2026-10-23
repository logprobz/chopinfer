//! Complete shared-expert dense sequence with and without down-weight look-ahead.
use super::{ahead::Case, *};
use bundle_deepseek41::sm120::{ceil_parallel, fold_cubin_for_symbol};
use kernel_core::backend::{Counts, View};
use kernel_core::gated_mlp::quantized::SiluMulQuantizeBuffers;
use kernel_harness::launch::Bindings;

pub(super) fn check_module() -> LabResult<serde_json::Value> {
    let symbols = SUPPORTED_RANKS
        .iter()
        .map(|&ranks| ceil_parallel::activation(&FLASH, ranks).map(|e| e.symbol))
        .collect::<Result<Vec<_>, _>>()?;
    let (name, bytes) = fold_cubin_for_symbol(&symbols[0]).ok_or("missing activation module")?;
    let path = modules::directory()?.join(format!("{name}.cubin"));
    let staged = std::fs::read(path).map_err(|e| e.to_string())?;
    if staged != bytes
        || symbols
            .iter()
            .any(|s| fold_cubin_for_symbol(s).map(|m| m.0) != Some(name))
    {
        return Err("shared-expert activation module differs from serving".into());
    }
    Ok(json!({"sequence_module":name,"sha256":kernel_build::digest(bytes),"symbols":symbols}))
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_FOLD_CUBINS"]
fn stream_sublayer_gpu() {
    check_module().unwrap();
    worker::run(worker::SUBLAYER,modules::timing_cubins().unwrap(), |session| {
        let entries = catalog::catalog(&FLASH)?;
        for &ranks in SUPPORTED_RANKS {
            let width = FLASH.shape.moe.intermediate / ranks;
            let find = |n, k, paired| {
                entries.iter().find(|e| (e.ticket.query.n,e.ticket.query.k,e.ticket.query.groups,e.ticket.paired)==(n,k,1,paired))
                    .ok_or_else(|| "missing shared-expert fold".to_string())
            };
            let gate_entry = find(width, FLASH.shape.hidden, true)?;
            let down_entry = find(FLASH.shape.hidden, width, false)?;
            let activation = ceil_parallel::activation(&FLASH, ranks)?;
            for rows in ROWS {
                let gate = Case::new(session, gate_entry, rows, 23)?;
                let down = Case::new(session, down_entry, rows, 29)?;
                let op = Sm120::new().with_silu_mul_quantize(activation.clone())
                    .silu_mul_quantize(&activation.query, Need::rows(rows)).map_err(|e|e.to_string())?;
                // The producer writes exactly the down fold's code/scale allocations.
                // Every fold owns separate workspace and persistent ticket state.
                let buffers = [&gate.buffers[4], &gate.buffers[12], &down.buffers[0],
                    &down.buffers[1], &down.buffers[5], &down.buffers[6]];
                let view = |i:usize| View{handle:i as u64+1, bytes:buffers[i].byte_len() as u64};
                let input = SiluMulQuantizeBuffers{gate:view(0),up:view(1),codes:view(2),
                    scales:view(3),live_rows:view(4),status:view(5),status_row_offset:0};
                let bindings = Bindings{buffers:buffers.iter().enumerate().map(|(i,b)|(i as u64+1,*b)).collect()};
                let calls = op.launches(&input,View::EMPTY,Counts::rows(rows)).map_err(|e|e.to_string())?;
                let functions = previous::functions(session,&calls)?;
                let prefetch = [kernels_sm120::gemm::mxfp8_fold_stream::prefetch_launch([
                    View{handle:3,bytes:down.buffers[2].byte_len() as u64},
                    View{handle:4,bytes:down.buffers[3].byte_len() as u64},View::EMPTY,View::EMPTY]).map_err(|e|e.to_string())?];
                let prefetch_functions = previous::functions(session,&prefetch)?;
                let run = |stream:&runtime_cuda::Stream| {
                    gate.run(session,stream)?;
                    previous::run(session,stream,&calls,&functions,&bindings)?;
                    down.run(session,stream)
                };
                let baseline = session.capture(run)?;
                let ahead = session.capture(|stream| {
                    previous::run(session,stream,&prefetch,&prefetch_functions,&down.bindings())?;
                    run(stream)
                })?;
                let graphs = [baseline,ahead];
                graphs[0].launch(&session.stream).map_err(|e|e.to_string())?;
                let expected = [down.read(session,0)?,down.read(session,1)?,down.read(session,4)?,down.read(session,6)?];
                graphs[1].launch(&session.stream).map_err(|e|e.to_string())?;
                for (index,want) in [0,1,4,6].into_iter().zip(expected) {
                    assert_eq!(down.read(session,index)?,want,"shared sequence role {index}");
                }
                assert!(gate.read(session,6)?.iter().chain(down.read(session,6)?.iter()).all(|&v|v==0));
                let mode = kernel_harness::eviction::EvictionMode::selected()?;
                let timing = kernel_harness::time_ab(session,2,5,[mode,mode],|arm,stream|graphs[arm].launch(stream).map_err(|e|e.to_string()))?;
                assert!(gate.read(session,9)?.iter().chain(down.read(session,9)?.iter()).all(|&v|v==0));
                let baseline_symbols = vec![gate.calls[0].symbol.clone(),calls[0].symbol.clone(),down.calls[0].symbol.clone()];
                let mut candidate_symbols = baseline_symbols.clone();candidate_symbols.insert(0,prefetch[0].symbol.clone());
                println!("{}",json!({"phase":"ahead-comparison","sequence":"shared-expert","device":session.ordinal,
                    "n":down_entry.ticket.query.n,"k":width,"groups":1,"rows":rows,"ranks":ranks,
                    "bitwise":true,"baseline_us":timing.arms[0].median_ms*1000.0,"ahead_us":timing.arms[1].median_ms*1000.0,
                    "paired_ratio":timing.paired_ratio,"graph_replays":8,"baseline_symbols":baseline_symbols,"candidate_symbols":candidate_symbols,
                    "dram_traffic_proven":false,"look_ahead_qualified":false}));
            }
        }
        Ok(())
    }).unwrap();
}
