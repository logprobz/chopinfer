//! A fold-sized lead task followed by a distinct target weight bank.
use super::*;
use kernel_core::backend::{Counts, Launch, View};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldStreamBuffers};
use kernel_harness::{Session, launch::Bindings};
use runtime_cuda::{DeviceBuffer, Function};

pub(super) struct Case {
    pub(super) buffers: Vec<DeviceBuffer>,
    pub(super) calls: Vec<Launch>,
    functions: Vec<Function>,
}
impl Case {
    pub(super) fn new(
        session: &Session,
        entry: &kernels_sm120::gemm::mxfp8_fold_stream::Entry,
        rows: u32,
        seed: usize,
    ) -> LabResult<Self> {
        let op = Sm120::new()
            .with_mxfp8_fold_stream(entry.clone())
            .mxfp8_fold_gemm_stream(&entry.ticket.query, entry.ticket.paired, Need::rows(rows))
            .map_err(|e| e.to_string())?;
        let b = entry
            .ticket
            .legacy()
            .map_err(|e| e.to_string())?
            .bytes(rows);
        let mut images = vec![
            previous::pattern(b.activations, seed),
            previous::scales(b.activation_scales, seed),
            previous::pattern(b.weights, seed + 1),
            previous::scales(b.weight_scales, seed + 1),
            vec![0xa5; b.output as usize],
            rows.to_ne_bytes().to_vec(),
            vec![0; rows as usize * 4],
            vec![0xcd; op.info().workspace as usize],
            vec![0xcd; entry.packed_bytes() as usize],
            vec![0xcd; entry.state_bytes(rows).map_err(|e| e.to_string())? as usize],
        ];
        if entry.ticket.paired {
            images.extend([
                previous::pattern(b.weights, seed + 2),
                previous::scales(b.weight_scales, seed + 2),
                vec![0xa5; b.output as usize],
                vec![0xcd; entry.packed_bytes() as usize],
            ]);
        }
        let buffers = images
            .iter()
            .map(|bytes| {
                let buffer = session.allocate(bytes.len())?;
                buffer
                    .copy_from_bytes(bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
                Ok(buffer)
            })
            .collect::<LabResult<Vec<_>>>()?;
        let v = |i: usize| View {
            handle: i as u64 + 1,
            bytes: buffers[i].byte_len() as u64,
        };
        let first = Mxfp8FoldGemmBuffers {
            activations: v(0),
            activation_scales: v(1),
            weights: v(2),
            weight_scales: v(3),
            output: v(4),
            live_rows: v(5),
            status: v(6),
            status_row_offset: 0,
        };
        let inputs = Mxfp8FoldStreamBuffers {
            first,
            second: entry.ticket.paired.then(|| Mxfp8FoldGemmBuffers {
                weights: v(10),
                weight_scales: v(11),
                output: v(12),
                ..first
            }),
            packed_weights: [
                v(8),
                if entry.ticket.paired {
                    v(13)
                } else {
                    View::EMPTY
                },
            ],
            state: v(9),
        };
        let init = op.initialize(&inputs).map_err(|e| e.to_string())?;
        let init_functions = previous::functions(session, &init)?;
        let bindings = Bindings {
            buffers: buffers
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u64 + 1, b))
                .collect(),
        };
        previous::run(session, &session.stream, &init, &init_functions, &bindings)?;
        let calls = op
            .launches(&inputs, v(7), Counts::rows(rows))
            .map_err(|e| e.to_string())?;
        let functions = previous::functions(session, &calls)?;
        Ok(Self {
            buffers,
            calls,
            functions,
        })
    }
    pub(super) fn bindings(&self) -> Bindings<'_> {
        Bindings {
            buffers: self
                .buffers
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u64 + 1, b))
                .collect(),
        }
    }
    pub(super) fn run(&self, session: &Session, stream: &runtime_cuda::Stream) -> LabResult<()> {
        previous::run(
            session,
            stream,
            &self.calls,
            &self.functions,
            &self.bindings(),
        )
    }
    pub(super) fn read(&self, session: &Session, index: usize) -> LabResult<Vec<u8>> {
        let mut data = vec![0; self.buffers[index].byte_len()];
        self.buffers[index]
            .copy_to_bytes(&mut data, &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(data)
    }
}

#[test]
fn stream_ahead_gpu() {
    let cubins = modules::timing_cubins().unwrap();
    worker::run(worker::AHEAD,cubins,|session| {
        let entries=catalog::catalog(&FLASH)?;
        let lead_entry=entries.iter().find(|e|(e.ticket.query.n,e.ticket.query.k,e.ticket.query.groups,e.ticket.paired)==(1280,5120,1,false)).ok_or("missing ahead lead")?;
        let traffic_arm=std::env::var("CHOPINFER_FOLD_AHEAD_ARM").ok().map(|s|s.parse::<usize>().map_err(|e|e.to_string())).transpose()?;
        let target_filter=std::env::var("CHOPINFER_FOLD_AHEAD_N").ok().map(|s|s.parse::<u32>().map_err(|e|e.to_string())).transpose()?;
        if traffic_arm.is_some_and(|a|a>1) {return Err("ahead arm must be zero or one".into());}
        for &(n,k,groups) in &[(512,5120,1),(1280,5120,1),(1024,4096,2),(5120,15360,1)] {
            if target_filter.is_some_and(|filter|filter!=n){continue;}
            let target_entry=entries.iter().find(|e|(e.ticket.query.n,e.ticket.query.k,e.ticket.query.groups,e.ticket.paired)==(n,k,groups,false)).ok_or("missing ahead target")?;
            let lead=Case::new(session,lead_entry,1,11)?;
            let target=Case::new(session,target_entry,1,17)?;
            let prefetch=kernels_sm120::gemm::mxfp8_fold_stream::prefetch_launch([
                View{handle:9,bytes:target.buffers[8].byte_len() as u64},View{handle:4,bytes:target.buffers[3].byte_len() as u64},View::EMPTY,View::EMPTY]).map_err(|e|e.to_string())?;
            let prefetch_calls=[prefetch];
            let prefetch_functions=previous::functions(session,&prefetch_calls)?;
            let baseline=session.capture(|stream|{lead.run(session,stream)?;target.run(session,stream)})?;
            let ahead=session.capture(|stream|{
                previous::run(session,stream,&prefetch_calls,&prefetch_functions,&target.bindings())?;
                lead.run(session,stream)?;target.run(session,stream)
            })?;
            let graphs=[baseline,ahead];
            if let Some(arm)=traffic_arm {
                // Nsight Compute profiles this complete graph, preserving
                // dependencies and cache contents between its nodes.
                let mode=kernel_harness::eviction::EvictionMode::selected()?;
                kernel_harness::time_prepared(session,1,3,mode,|_|Ok(()),|stream|graphs[arm].launch(stream).map_err(|e|e.to_string()))?;
                println!("{}",json!({"phase":"ahead-traffic","device":session.ordinal,"n":n,"k":k,"groups":groups,"arm":arm,"graph_replays":4}));
            } else {
                graphs[0].launch(&session.stream).map_err(|e|e.to_string())?;
                let expected=target.read(session,4)?;
                graphs[1].launch(&session.stream).map_err(|e|e.to_string())?;
                assert_eq!(target.read(session,4)?,expected,"ahead target output");
                assert!(target.read(session,9)?.iter().all(|&v|v==0));
                assert!(lead.read(session,9)?.iter().all(|&v|v==0));
                let mode=kernel_harness::eviction::EvictionMode::selected()?;
                let timing=kernel_harness::time_ab(session,2,5,[mode,mode],|arm,stream|graphs[arm].launch(stream).map_err(|e|e.to_string()))?;
                println!("{}",json!({"phase":"ahead-comparison","sequence":"two-folds","device":session.ordinal,"n":n,"k":k,"groups":groups,"rows":1,
                    "baseline_symbols":[lead.calls[0].symbol.clone(),target.calls[0].symbol.clone()],
                    "candidate_symbols":[prefetch_calls[0].symbol.clone(),lead.calls[0].symbol.clone(),target.calls[0].symbol.clone()],
                    "lead_n":1280,"lead_k":5120,"bitwise":true,"baseline_us":timing.arms[0].median_ms*1000.0,
                    "ahead_us":timing.arms[1].median_ms*1000.0,"paired_ratio":timing.paired_ratio,"graph_replays":8,
                    "weight_bytes":lead_entry.packed_bytes()+target_entry.packed_bytes()+lead.buffers[3].byte_len() as u64+target.buffers[3].byte_len() as u64}));
            }
        }
        Ok(())
    }).unwrap();
}
