//! Indexed raw retention versus the previous ordered device-copy list.
use bundle_deepseek41::{
    dspark::{
        state::{CommitItem, Config, Kind, Rows},
        transfers,
    },
    sm120::{retain_rows, retain_rows_entries},
};
use kernel_core::backend::{Counts, Need, View};
use kernel_harness::{LabResult, Session};
use model_deepseek41_contract::own_ops::retain_rows::{
    RetainRows, RetainRowsBuffers, Segment, oracle,
};
use serde_json::json;

struct Case {
    items: u32,
    depth: u32,
    pattern: u32,
    op: RetainRows,
    segments: Vec<Segment>,
}

fn case(items: u32, depth: u32, pattern: u32, row_bytes: u32, first: Option<u64>) -> Case {
    let config = Config {
        slots: items,
        stages: 3,
        window: 128,
        max_prefill_rows: 8192,
        context_tokens: 8192,
        position_capacity: 8192,
    };
    let mut segments = Vec::new();
    let mut source = if depth == 5 { 0 } else { 5 };
    for item in 0..items {
        let slot = (item * 5 + 1) % items;
        let start =
            first.unwrap_or([133, 134, 267, 268, 127, 128, 0][(item + pattern) as usize % 7]);
        let kept = match pattern {
            1 => 1,
            2 if item % 2 == 1 => 1,
            _ => depth + 1,
        };
        let copies = transfers::retain_copies(
            config,
            CommitItem {
                rows: Rows {
                    request: u64::from(item) + 1,
                    slot,
                    start,
                    kind: Kind::Decode { drafts: depth },
                },
                kept,
                project_from_staging: false,
                retain_offset: 0,
            },
            source,
        )
        .unwrap();
        // Independent per-row mapping checks the previous copy-list fixture.
        let actual: Vec<_> = copies
            .iter()
            .flat_map(|copy| {
                (0..copy.rows).map(move |row| {
                    (
                        copy.staged_row + u64::from(row),
                        copy.history_row + u64::from(row),
                    )
                })
            })
            .collect();
        let expected: Vec<_> = (0..kept)
            .map(|row| {
                (
                    source + u64::from(row),
                    u64::from(slot) * 134 + (start + u64::from(row)) % 134,
                )
            })
            .collect();
        assert_eq!(actual, expected);
        segments.extend(copies.into_iter().map(|copy| Segment {
            source: copy.staged_row as u32,
            destination: copy.history_row as u32,
            rows: copy.rows,
        }));
        source += u64::from(depth + 1);
    }
    Case {
        items,
        depth,
        pattern,
        op: RetainRows {
            row_bytes,
            source_rows: 192,
            destination_rows: items * 134,
        },
        segments,
    }
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for items in [1, 8, 16, 32] {
        for depth in [0, 1, 3] {
            for pattern in 0..if depth == 0 { 1 } else { 3 } {
                cases.push(case(items, depth, pattern, 30720, None));
            }
        }
    }
    cases.push(case(32, 5, 0, 30720, None));
    cases.push(case(1, 3, 0, 30720, Some(268)));
    cases.push(case(8, 3, 2, 7, None));
    cases.push(case(8, 3, 2, 19, None));
    assert_eq!(cases.len(), 32);
    cases
}

fn pattern(bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|i| (i.wrapping_mul(37) ^ (i >> 8) ^ (i >> 17)) as u8)
        .collect()
}

#[test]
fn retention_case_inventory_is_bounded_and_covers_full_physical_histories() {
    let mut wraps = 0;
    for case in cases() {
        let encoded = case.op.encode(&case.segments).unwrap();
        assert_eq!(encoded.len(), case.segments.len() * 12);
        assert!(case.segments.len() <= 64);
        wraps += usize::from(case.segments.len() > case.items as usize);
        let source = pattern(case.op.source_bytes() as usize);
        let mut destination = vec![0xff; case.op.destination_bytes() as usize + 64];
        oracle(case.op, &source, &case.segments, &mut destination).unwrap();
        let copied: u32 = case.segments.iter().map(|segment| segment.rows).sum();
        assert!(copied >= case.items && copied <= case.items * (case.depth + 1));
    }
    assert!(wraps >= 8);
    println!(
        "{}",
        json!({"cases":32,"wrap_cases":wraps,"raw_rows_per_slot":134,"cpu_only":true})
    );
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<runtime_cuda::DeviceBuffer> {
    let buffer = session.allocate(bytes.len())?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn read(session: &Session, buffer: &runtime_cuda::DeviceBuffer) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

#[test]
fn retention_bytes_match_scalar_copies_with_poison_and_consumers() {
    kernel_harness::gpu_test(1 << 30, |session| {
        use kernel_core::contract::DType;
        use kernel_harness::launch::{Bindings,launch};
        use kernel_harness::poison::{LiveProducer,Replay,Rewrites,WriteMap};
        use std::collections::BTreeMap;
        let (name, cubin) = bundle_deepseek41::sm120::fold_cubin_for_symbol(retain_rows_entries::SYMBOL)
            .ok_or("retention cubin is absent")?;
        assert_eq!(name,retain_rows_entries::CUBIN);
        let module = session.load_cubin(cubin)?;
        let function = module.function(retain_rows_entries::SYMBOL).map_err(|e| e.to_string())?;
        let probes = session.load_cubin(bundle_deepseek41::sm120::dependency_probes().0)?;
        let producer = LiveProducer::with_function(session,
            probes.function(kernels_sm120::probes::LIVE).map_err(|e| e.to_string())?)?;
        let initial_free = session.context.memory_info().map_err(|e| e.to_string())?.0;
        let mut peak = 0;
        let mut controls = 0;
        for (index,case) in cases().into_iter().enumerate() {
            let source = pattern(case.op.source_bytes() as usize);
            let encoded = case.op.encode(&case.segments).map_err(|e| e.to_string())?;
            let mut expected = vec![0xff;case.op.destination_bytes() as usize+64];
            oracle(case.op,&source,&case.segments,&mut expected).map_err(|e| e.to_string())?;
            let src = upload(session,&source)?;
            let src_image = upload(session,&source)?;
            let mapping = upload(session,&encoded)?;
            let mapping_image = upload(session,&encoded)?;
            let output = upload(session,&vec![0xff;expected.len()])?;
            let scalar = upload(session,&vec![0xff;expected.len()])?;
            let observed = upload(session,&vec![0xff;expected.len()])?;
            let live = upload(session,&1_u32.to_le_bytes())?;
            let views = RetainRowsBuffers {
                source: View {handle:1,bytes:case.op.source_bytes()},
                segments: View {handle:2,bytes:encoded.len() as u64},
                destination: View {handle:3,bytes:case.op.destination_bytes()},
            };
            let workspace = View {handle:0,bytes:0};
            let call = retain_rows::prepare(case.op,Need::rows(64)).map_err(|e| e.to_string())?
                .launches(&views,workspace,Counts::rows(case.segments.len() as u32)).map_err(|e| e.to_string())?.remove(0);
            let consumer_op = RetainRows {source_rows:case.op.destination_rows,..case.op};
            let consumer_segments: Vec<_> = case.segments.iter().map(|segment| Segment {
                source:segment.destination,..*segment }).collect();
            let consumer_mapping = upload(session,&consumer_op.encode(&consumer_segments).map_err(|e| e.to_string())?)?;
            let consumer = retain_rows::prepare(consumer_op,Need::rows(64)).map_err(|e| e.to_string())?
                .launches(&RetainRowsBuffers {source:View{bytes:consumer_op.source_bytes(),..views.source},..views},
                    workspace,Counts::rows(case.segments.len() as u32)).map_err(|e| e.to_string())?.remove(0);
            let bindings = Bindings { buffers:BTreeMap::from([(1,&src),(2,&mapping),(3,&output)]) };
            let consumer_bindings = Bindings { buffers:BTreeMap::from([(1,&output),(2,&consumer_mapping),(3,&observed)]) };
            let invoke = |stream:&runtime_cuda::Stream| {
                launch(&session.context,&function,stream,call.plan,&call.arguments,&bindings)?;
                launch(&session.context,&function,stream,consumer.plan,&consumer.arguments,&consumer_bindings)
            };
            for segment in &case.segments {
                scalar.copy_range_from_async(segment.destination as usize*case.op.row_bytes as usize,
                    &src,segment.source as usize*case.op.row_bytes as usize,
                    segment.rows as usize*case.op.row_bytes as usize,&session.stream).map_err(|e|e.to_string())?;
            }
            if read(session,&scalar)? != expected { return Err(format!("scalar case {index} differs from byte oracle")); }
            let graph = session.capture(invoke)?;
            let rewrites = Rewrites::new(session,&[(&src_image,&src),(&mapping_image,&mapping)])?;
            let mut ranges: Vec<_> = case.segments.iter().map(|segment| {
                let start=segment.destination as usize*case.op.row_bytes as usize;
                start..start+segment.rows as usize*case.op.row_bytes as usize
            }).collect();
            ranges.sort_by_key(|range| range.start);
            let mut replay = Replay::new(session)
                .output_complement("retained",&output,WriteMap::of(DType::U8).ranges(ranges.clone())?,expected.clone())
                .output_complement("consumer",&observed,WriteMap::of(DType::U8).ranges(ranges)?,expected.clone());
            replay.eager(||invoke(&session.stream))?;
            replay.relaunch(||invoke(&session.stream))?;
            for _ in 0..3 { replay.graph(&graph)?; }
            replay.pdl(||producer.launch_rewriting(session,&live,&rewrites),
                ||invoke(&session.stream),call.plan.pdl && consumer.plan.pdl)?;
            if read(session,&output)? != expected || read(session,&observed)? != expected {
                return Err(format!("retention or consumer case {index} differs"));
            }
            for bad in [Segment {source:case.op.source_rows,destination:0,rows:1},
                Segment {source:0,destination:case.op.destination_rows,rows:1},
                Segment {source:0,destination:0,rows:0},case.segments[0]] {
                let mut invalid=case.segments.clone();invalid.push(bad);
                if case.op.encode(&invalid).is_ok() {return Err("invalid metadata was admitted".into());}
                controls+=1;
            }
            if read(session,&output)? != expected {return Err("refused metadata changed retained state".into());}
            let free=session.context.memory_info().map_err(|e|e.to_string())?.0;
            peak=peak.max(initial_free.saturating_sub(free));
            let mut receipt=BTreeMap::from([("case".into(),json!(index)),("items".into(),json!(case.items)),
                ("depth".into(),json!(case.depth)),("pattern".into(),json!(case.pattern)),
                ("row_bytes".into(),json!(case.op.row_bytes)),("segments".into(),json!(case.segments.len())),
                ("bitwise".into(),json!(true)),("scalar_reference".into(),json!(true)),
                ("full_physical_rows".into(),json!(134)),("extra_device_bytes".into(),json!(initial_free.saturating_sub(free)))]);
            replay.record(&mut receipt);
            session.receipt("deepseek41-retention-pair",receipt)?;
            println!("RETENTION_PAIR_BITWISE case={index} items={} depth={} bytes={} PASS",
                case.items, case.depth, case.op.row_bytes);
        }
        assert_eq!(controls,128);
        println!("RETENTION_PAIR_COMPLETE cases=32 negatives=128 PASS");
        let final_free=session.context.memory_info().map_err(|e|e.to_string())?.0;
        println!("{}",json!({"cases":32,"negative_controls":controls,"bitwise":true,
            "peak_extra_device_bytes":peak,"final_extra_device_bytes":initial_free.saturating_sub(final_free)}));
        Ok(())
    }).expect("retained bytes and dependent consumers match the previous copy list");
}
