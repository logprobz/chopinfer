//! Export actual prepared MoE programs for external complete-graph pairs.
use super::*;
use kernel_core::allocation::Traversal;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::Write;

fn span_json(span: DeviceBufferSpanV1) -> Value {
    json!({"buffer":span.buffer.raw,"offset":span.offset,"bytes":span.length})
}

fn calls_json(
    calls: &[Lowered],
    allocations: &mut BTreeMap<u64, u64>,
    row_bounds: &BTreeMap<u64, u64>,
) -> Vec<Value> {
    calls
        .iter()
        .map(|op| {
            let call = &op.launch;
            let arguments = call
                .arguments
                .0
                .iter()
                .map(|argument| match argument {
                    Argument::DeviceBuffer(handle) => {
                        let mut span = call.bindings.get(*handle).unwrap();
                        if let Some(&end) = row_bounds.get(&span.buffer.raw) {
                            assert!(span.offset < end);
                            span.length = span.length.min(end - span.offset);
                        }
                        let end = span.offset.checked_add(span.length).unwrap();
                        allocations
                            .entry(span.buffer.raw)
                            .and_modify(|bytes| *bytes = (*bytes).max(end))
                            .or_insert(end);
                        json!({"pointer":span_json(span)})
                    }
                    Argument::U32(value) => json!({"u32":value}),
                    Argument::I32(value) => json!({"i32":value}),
                    Argument::U64(value) => json!({"u64":value}),
                    Argument::NullDevicePointer => json!({"null":true}),
                    other => panic!("MoE export needs explicit support for {other:?}"),
                })
                .collect::<Vec<_>>();
            json!({"symbol":call.symbol,"rows":call.rows,"grid":call.plan.grid,
            "block":call.plan.block,"cluster":call.plan.cluster,
            "shared_bytes":call.plan.dynamic_shared_bytes,"pdl":call.plan.pdl,
            "arguments":arguments})
        })
        .collect()
}

fn export(shape: &Shape, ops: &Ops, rank: u32, rows: u32, width: u32) -> Value {
    let mut b = crate::buffers::unaliased(|| rank_buffers(shape, ops, rank));
    // The ordinary lowering fixture overallocates weight views. Replay
    // needs the actual bank extents even when a binder retains the full view.
    let hidden = u64::from(shape.dims().hidden);
    let intermediate = u64::from(FLASH.shape.moe.intermediate / 4);
    let experts = u64::from(FLASH.shape.moe.experts);
    for (name, view) in &mut b.views {
        let bytes = if name.ends_with("ffn.gate.weight") {
            Some(experts * hidden * 2)
        } else if name.ends_with("ffn.gate.weight.f32_domain") {
            Some(experts.div_ceil(16) * 4)
        } else if name.ends_with("ffn.gate.bias") || name.ends_with("ffn.gate.bias_vl") {
            Some(experts * 4)
        } else if name.ends_with("ffn.experts.gate_up.weight") {
            Some(experts * intermediate * hidden)
        } else if name.ends_with("ffn.experts.down.weight") {
            Some(experts * hidden * intermediate / 2)
        } else if name.contains("ffn.shared_experts.") && name.ends_with(".weight") {
            Some(hidden * intermediate)
        } else if name.contains("ffn.shared_experts.") && name.ends_with(".scale") {
            Some(hidden * intermediate / 32)
        } else {
            None
        };
        if let Some(bytes) = bytes {
            view.length = bytes;
        }
    }
    if width != 0 {
        b.scratch.expert_scales = Some(span(
            EXPANDED,
            rank,
            crate::buffers::expert_scale_bytes(shape).unwrap(),
        ));
        for gate in [true, false] {
            let name = crate::lower::moe::scale_bank_name(0, gate);
            let format = kernel_core::weight_transform::ScaleFormat::RowTilesOffsetV1;
            b.encoded_scales.insert(
                name.clone(),
                crate::buffers::ScaleBankEncoding { format, width },
            );
            b.views.get_mut(&name).unwrap().length = crate::Kernels::encoded_scale_minimum(
                ops.moe.as_ref().unwrap().scale_bank(gate),
                width,
                format,
            )
            .unwrap();
        }
    }
    let live = span(999, rank, 4);
    let inputs = DeviceInputs {
        live_tokens: live,
        ..DeviceInputs::within(
            PhaseArena {
                arena: live,
                block_tables: live,
            },
            &InputLayout::decode_within(Items::new(1).unwrap()),
        )
    };
    let peers = [b.clone()];
    let phase = Phase::Verify {
        rows,
        items: rows.div_ceil(6),
    };
    let lowering = Lowering {
        prefill_row_origin: None,
        shape,
        ops,
        buffers: &b,
        peers: &peers,
        inputs: &inputs,
        rank,
        kv: KvExtent {
            storage: KvStorage::Native,
            pages: 1,
            extent: Extent {
                served: 1,
                bucket: rows.min(32),
                verify_rows: rows,
            },
            sequences: 1,
            max_pages: 1,
        },
        head_rows: 1,
        prefill_hc_fusion: true,
    };
    // Both arms start from the same normed BF16 rows. This setup quantizer
    // completes before the fork and is excluded from the MoE timing scopes.
    let prefix = lowering.quantize_input(phase).unwrap().unwrap();
    assert!(!prefix.is_empty());
    let lowered = crate::program::moe(0)
        .into_iter()
        .map(|stage| (stage, lowering.stage(phase, stage).unwrap().unwrap()))
        .collect::<Vec<_>>();
    let stages = lowered
        .iter()
        .flat_map(|(stage, calls)| std::iter::repeat_n(*stage, calls.len()))
        .collect::<Vec<_>>();
    // Some generic binders retain an entire prefill activation view.
    // Bound the replay to its planned row prefix, preserving each address
    // and row stride. Guards then detect writes beyond the planned rows.
    let scratch = &b.scratch;
    let row_bounds = [
        scratch.normed,
        scratch.normed_wide,
        scratch.normed_codes,
        scratch.normed_scales,
        scratch.router_logits,
        scratch.router_scores,
        scratch.expert_ids,
        scratch.expert_weights,
        scratch.bias_select,
        scratch.expert_intermediate,
        scratch.intermediate_codes,
        scratch.intermediate_scales,
        scratch.routed,
        scratch.shared_gate,
        scratch.shared_up,
        scratch.shared_codes,
        scratch.shared_scales,
        scratch.shared_partial,
        scratch.partial,
        scratch.row_status,
        scratch.shared_row_status,
    ]
    .into_iter()
    .map(|span| {
        assert!(
            span.length
                .is_multiple_of(u64::from(crate::buffers::scratch_rows()))
        );
        let prefix = crate::buffers::rows(span, rows);
        (span.buffer.raw, prefix.offset + prefix.length)
    })
    .collect::<BTreeMap<_, _>>();
    let mut allocations = BTreeMap::new();
    let prefix = calls_json(&prefix, &mut allocations, &row_bounds);
    let mut scopes = Vec::new();
    let mut calls = Vec::new();
    for (stage, lowered) in &lowered {
        let start = calls.len();
        calls.extend(calls_json(lowered, &mut allocations, &row_bounds));
        scopes.push(json!({"stage":format!("{stage:?}"),"start":start,"end":calls.len()}));
    }
    let launches = lowered
        .into_iter()
        .flat_map(|(_, calls)| calls.into_iter().map(|c| c.launch))
        .collect::<Vec<_>>();
    let units = crate::shared_overlap::checked_units(
        &stages,
        &launches,
        &[b.scratch.normed_codes, b.scratch.normed_scales, live],
        false,
    )
    .unwrap();
    let units = units
        .into_iter()
        .flatten()
        .map(|unit| {
            use crate::shared_overlap::Op;
            match unit {
                Op::Compute(range) => json!({"kind":"compute","start":range.start,"end":range.end}),
                Op::Side(range) => json!({"kind":"side","start":range.start,"end":range.end}),
                Op::Fork => json!({"kind":"fork"}),
                Op::Join => json!({"kind":"join"}),
            }
        })
        .collect::<Vec<_>>();
    let mut roles = BTreeMap::new();
    let mut add = |name: String, mut span: DeviceBufferSpanV1| {
        if let Some(&bytes) = allocations.get(&span.buffer.raw) {
            if span.offset >= bytes {
                return;
            }
            span.length = span.length.min(bytes - span.offset);
            roles.insert(name, span_json(span));
        }
    };
    b.scratch
        .visit_named(&mut |name, span| add(format!("scratch.{name}"), *span));
    for (name, span) in &b.views {
        add(name.clone(), *span);
    }
    for (name, span) in [
        ("workspace", b.workspace),
        ("shared_workspace", b.shared_workspace),
        ("sorted_rows", b.sorted_rows),
        ("live", live),
    ] {
        add(name.into(), span);
    }
    assert!(roles.contains_key("scratch.shared_row_status"));
    assert!(roles.contains_key("scratch.partial"));
    json!({"schema":"chopinfer.moe-wide-graph.v1","rows":rows,"rank":rank,"scale_width":width,"bounded_row_views":true,
        "prefix":prefix,"calls":calls,"units":units,"scopes":scopes,"roles":roles,"allocations":allocations})
}

#[test]
#[ignore = "writes frozen lab programs to an explicitly selected external directory"]
fn export_wide_moe_graphs() {
    let root = std::path::PathBuf::from(
        std::env::var_os("CHOPINFER_MOE_GRAPH_EXPORT").expect("set external graph output"),
    );
    std::fs::create_dir_all(&root).unwrap();
    let root = root.canonicalize().unwrap();
    assert!(
        !root.starts_with(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .canonicalize()
                .unwrap()
        )
    );
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap().with_verify_rows(64);
    let kernels = crate::Kernels::new(&shape).unwrap();
    let ops = Ops::prepare_set(
        &kernels,
        &shape,
        KvStorage::Native,
        moe::expert_sets(&FLASH)[0],
    );
    let mut cases = Vec::new();
    for rank in 0..4 {
        for rows in [16, 32].into_iter().chain(33..=64) {
            for width in [0, 1, 2, 4] {
                let value = export(&shape, &ops, rank, rows, width);
                let name = format!("r{rows}-rank{rank}-w{width}.json");
                let mut out = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(root.join(&name))
                    .unwrap();
                out.write_all(&serde_json::to_vec(&value).unwrap()).unwrap();
                cases.push(name);
            }
        }
    }
    assert_eq!(cases.len(), 544);
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join("graphs.json"))
        .unwrap();
    out.write_all(
        &serde_json::to_vec(&json!({"schema":"chopinfer.moe-wide-inventory.v1","cases":cases}))
            .unwrap(),
    )
    .unwrap();
    println!("MOE_WIDE_EXPORT cases=544 PASS");
}
