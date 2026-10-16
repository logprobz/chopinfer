//! Real-weight MiMo layer gate and shared decode launch fixture.

use kernel_build::EntryParam;
use kernel_core::contract::{Argument, AttentionPhase, CommQuery, DType, RuntimeCounts};
use kernels_sm120::{attention, gemm, moe, row};

#[path = "common/mimo_planned.rs"]
pub mod mimo_planned;

fn find_row(symbol: &str) -> row::RowInstantiation {
    row::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .unwrap_or_else(|| panic!("missing row entry {symbol}"))
}

fn find_gemm(symbol: &str) -> gemm::GemmVariant {
    gemm::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .unwrap_or_else(|| panic!("missing GEMM entry {symbol}"))
}

/// The expert phase and bucket a step of `rows` runs: decode buckets above
/// 16 rows (`serving::DECODE_EXPERT_ROWS`) route as prefill does and run
/// the prefill entries, as served (`workspace::moe::prefill_route`).
fn moe_entry(prefill: bool, rows: u32) -> (moe::Phase, u32) {
    if prefill || rows > 16 {
        (moe::Phase::Prefill, 8192)
    } else {
        (moe::Phase::Decode, rows)
    }
}

fn find_moe(phase: moe::Phase, projection: moe::Projection, tokens: u32) -> moe::Variant {
    moe::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| {
            entry.profile == model_mimo26_contract::sm120::config::PROFILE
                && entry.phase == phase
                && entry.projection == projection
                && entry.tokens_capacity == tokens
                && entry.path == moe::InputPath::A16Gather
        })
        .expect("missing MiMo MoE entry")
}

struct LayerPath {
    layer: usize,
    rows: u32,
    batch: u32,
    phase: AttentionPhase,
    row: Vec<row::RowInstantiation>,
    gemm: Vec<gemm::GemmVariant>,
    moe: [moe::Variant; 2],
    attention: attention::Variant,
}

impl LayerPath {
    fn new(layer: usize, phase: AttentionPhase, batch: u32) -> Self {
        assert!([1, 5].contains(&layer));
        let global = layer == 5;
        let prefill = phase == AttentionPhase::Prefill;
        let rows = if prefill { 64 } else { batch };
        let gemm_suffix = if prefill {
            "prefill".into()
        } else {
            format!("m{batch}")
        };
        let qkv = if global { "ga" } else { "swa" };
        let names = [
            format!("chop_gemm_mimo_qkv_{qkv}_{gemm_suffix}_v1"),
            format!("chop_gemm_mimo_o_{gemm_suffix}_v1"),
            format!("chop_gemm_mimo_router_{gemm_suffix}_v1"),
        ];
        let mut row_names = vec![
            "chop_row_rms_mimo_hf4096_v1".to_owned(),
            format!(
                "chop_row_rope_mimo_{}_computed_p{}_pos32_v1",
                if global {
                    "ga_bf16"
                } else if prefill {
                    "swa_prefill"
                } else {
                    "swa_decode"
                },
                if global { 64 } else { 32 }
            ),
            "chop_row_rms_mimo_hf4096_residual_v1".to_owned(),
            "chop_row_router_mimo_sigmoid256_v1".to_owned(),
        ];
        if !global {
            row_names.push("chop_row_ring_write_mimo_bf16_pos32_v1".into());
        }
        let block = if prefill { "mimo_b64" } else { "mimo_b16" };
        for operation in ["permute_count", "permute_scan", "permute_scatter"] {
            row_names.push(format!("chop_row_{operation}_{block}_v1"));
        }
        row_names.push("chop_row_combine_mimo4096_v1".into());
        let attention_name = match (global, prefill) {
            (true, true) => "mimo_ga_bf16_prefill",
            (true, false) => "mimo_ga_bf16_decode",
            (false, true) => "mimo_swa_prefill",
            (false, false) => "mimo_swa_decode",
        };
        let attention = attention::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|variant| variant.name == attention_name)
            .expect("missing MiMo attention entry");
        let (moe_phase, moe_capacity) = moe_entry(prefill, batch);
        Self {
            layer,
            rows,
            batch,
            phase,
            row: row_names.iter().map(|name| find_row(name)).collect(),
            gemm: names.iter().map(|name| find_gemm(name)).collect(),
            moe: [
                find_moe(moe_phase, moe::Projection::GateUp, moe_capacity),
                find_moe(moe_phase, moe::Projection::Down, moe_capacity),
            ],
            attention,
        }
    }

    fn validate_planners(&self) {
        let counts = RuntimeCounts {
            m: self.rows,
            batch: self.batch,
            context: 64,
            routed_rows: self.rows * 8,
            split_k: 1,
        };
        for entry in &self.row {
            let plan = entry.plan(self.rows, counts).unwrap();
            assert!(plan.pdl, "{}", entry.symbol);
            let args = entry
                .params
                .iter()
                .map(|param| match param {
                    EntryParam::DevicePointer { .. } => Argument::DeviceBuffer(1),
                    EntryParam::U32 { .. } => Argument::U32(1),
                    EntryParam::U64 { .. } => Argument::U64(1),
                    EntryParam::I32 { .. } => Argument::I32(1),
                    EntryParam::TensorMap { .. } | EntryParam::PointerArray { .. } => {
                        panic!("unexpected row ABI in {}", entry.symbol)
                    }
                })
                .collect();
            entry.pack_arguments(args).unwrap();
        }
        for entry in &self.gemm {
            let mut counts = counts;
            counts.split_k = gemm::choose_split_k(&entry.key).unwrap();
            let plan = gemm::plan(&entry.key, counts).unwrap();
            assert!(plan.pdl, "{}", entry.symbol);
        }
        for entry in &self.moe {
            let mut counts = counts;
            counts.split_k = entry.knobs.split_k_capacity;
            let plan = entry.plan(counts).unwrap();
            assert!(plan.pdl, "{}", entry.symbol);
            if let Some(reduction) = entry.reduction_plan(counts).unwrap() {
                assert!(reduction.pdl);
            }
            entry
                .arguments([1; 15], entry.weight_tensor_map(1).unwrap())
                .unwrap();
        }
        let mut variant = self.attention;
        if self.phase == AttentionPhase::Prefill {
            variant.query_capacity = 64;
        }
        let (k_map, v_map) = variant.cache_maps(1, 2, 3).unwrap();
        let buffers = attention::Buffers {
            q: 1,
            k_cache: 2,
            v_cache: 3,
            k_page_stride_bytes: tight_page_stride(&variant, variant.query.qk_dim),
            v_page_stride_bytes: tight_page_stride(&variant, variant.query.v_dim),
            k_map: Some(k_map),
            v_map: Some(v_map),
            block_tables: Some(4),
            slot_ids: Some(5),
            seq_lens: Some(6),
            q_start: Some(7),
            q_len: Some(8),
            prefix_len: Some(9),
            live_batch: 10,
            k_new: Some(11),
            v_new: Some(12),
            sink: (self.layer == 1).then_some(13),
            output: 14,
            workspace: 15,
            k_scale: 1.0,
            v_scale: 1.0,
        };
        let planned = variant
            .plan_for_bucket(self.batch, counts, buffers)
            .unwrap();
        assert!(planned.partial.pdl && planned.merge.pdl);
        let collective = CommQuery::AllReduce {
            ranks: 4,
            elements: u64::from(self.rows) * 4096,
            dtype: DType::Bf16,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        assert!(
            kernels_sm120::comm::plan(&collective, self.rows)
                .unwrap()
                .pdl
        );
        kernels_sm120::comm::arguments(kernels_sm120::comm::OneShotCall {
            input: 1,
            output: 2,
            peer_inboxes: vec![3, 4, 5, 6],
            peer_flags: vec![7, 8, 9, 10],
            epoch: 11,
            elements: u64::from(self.rows) * 4096,
            rank: 0,
            ranks: 4,
        })
        .unwrap();
    }
}

#[test]
fn layer_paths_have_public_pdl_plans() {
    for layer in [1, 5] {
        LayerPath::new(layer, AttentionPhase::Prefill, 1).validate_planners();
        for batch in [1, 8] {
            LayerPath::new(layer, AttentionPhase::Decode, batch).validate_planners();
        }
    }
}

#[test]
fn real_weight_layers() {
    if module_path!() != "mimo_layers" {
        return;
    }
    let Ok(checkpoint) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
        kernel_lab::skip("real MiMo layers: set CHOPINFER_MIMO_CHECKPOINT").unwrap();
        return;
    };
    let Some(ordinals) = kernel_lab::selected_ordinals().expect("GPU gate") else {
        kernel_lab::skip("real MiMo layers: set CHOPINFER_KERNEL_GPUS to idle GPUs").unwrap();
        return;
    };
    assert_eq!(ordinals.len(), 4, "MiMo G3 requires four GPU ordinals");
    let checkpoint = std::path::Path::new(&checkpoint);
    let images = execution::images(checkpoint).expect("TP4 layer images");
    let diagnostic_root =
        std::env::var_os("CHOPINFER_MIMO_DIAGNOSTIC_DIR").map(std::path::PathBuf::from);
    let selected_identity = diagnostic_root.as_ref().map(|_| {
        model_mimo26::checkpoint::identity(
            &std::env::var("CHOPINFER_MIMO_CHECKPOINT_ID").unwrap_or_else(|_| "mopd".into()),
        )
        .expect("registered checkpoint")
    });
    if let Some(root) = &diagnostic_root {
        std::fs::create_dir(root).expect("fresh external diagnostic directory");
    }
    let mut references = Vec::new();
    let selected_layer = std::env::var("CHOPINFER_MIMO_LAYER").ok().map(|value| {
        value
            .parse::<usize>()
            .expect("CHOPINFER_MIMO_LAYER is an integer")
    });
    assert!(selected_layer.is_none_or(|layer| [1, 5].contains(&layer)));
    for layer in [1, 5]
        .into_iter()
        .filter(|layer| selected_layer.is_none_or(|selected| selected == *layer))
    {
        let mut cases = Vec::new();
        cases.push((
            true,
            0_usize,
            execution::seeded_hidden(64, 0x6371_0026 ^ layer as u64),
        ));
        for step in 0..4 {
            cases.push((
                false,
                64 + step as usize,
                execution::seeded_hidden(1, 0x6371_2000 ^ ((layer as u64) << 8) ^ step),
            ));
        }
        references.push((layer, cases));
    }
    kernel_lab::gpu_test_multi(4 * 1024 * 1024 * 1024, |sessions| {
        let mut failures = Vec::new();
        for (layer, cases) in &references {
            let mut oracle = if diagnostic_root.is_some() {
                model_mimo26::oracle::LayerOracle::load_verified(checkpoint,*layer,selected_identity.expect("diagnostic checkpoint"))
            } else { model_mimo26::oracle::LayerOracle::load(checkpoint,*layer) }.map_err(|e|e.to_string())?;
            let mut lineage = Vec::new();
            let mut history = model_mimo26::oracle::KvHistory::default();
            let ranks = execution::ranks(sessions, &images, *layer, 1, true)?;
            for (prefill, position, input) in cases {
                use model_mimo26::oracle::diagnostic as diagnostic;
                let before = diagnostic_root.as_ref().map(|_|diagnostic::history_identity(&history));
                let mut captures = if let Some(root) = &diagnostic_root {
                    let scope = diagnostic::scope(*layer,oracle.global,input,*position,&lineage,false);
                    let op_scope = diagnostic::scope(*layer,oracle.global,input,*position,&lineage,true);
                    let case = root.join(format!("layer{layer:02}-position{position}"));
                    std::fs::create_dir(&case).map_err(|e|e.to_string())?;
                    Some((diagnostic::Capture::begin(&case.join("reference"),scope.clone(),"cpu-f64-bf16-boundaries"),
                        diagnostic::Capture::begin(&case.join("candidate"),scope,"gpu-eager-bf16"),
                        diagnostic::Capture::begin(&case.join("op-reference"),op_scope.clone(),"cpu-f64-matched-operands"),
                        diagnostic::Capture::begin(&case.join("op-candidate"),op_scope,"gpu-eager-bf16")))
                        .map(|(a,b,c,d)| Ok::<_,String>((a.map_err(|e|e.to_string())?,b.map_err(|e|e.to_string())?,c.map_err(|e|e.to_string())?,d.map_err(|e|e.to_string())?))).transpose()?
                } else {None};
                if let Some((reference,candidate,op_reference,op_candidate))=&mut captures {
                    for capture in [&mut *reference,candidate,op_reference,op_candidate] {
                        capture.bf16("input.hidden",input.len()/4096,input).map_err(|e|e.to_string())?;
                    }
                    if !history.keys.is_empty() {
                        reference.f64("history.keys",history.keys.len(),&history.keys.concat()).map_err(|e|e.to_string())?;
                        reference.f64("history.values",history.values.len(),&history.values.concat()).map_err(|e|e.to_string())?;
                    }
                }
                for (rank, session) in ranks.iter().zip(sessions) {
                    rank.set_inputs(session, input, *position, *prefill)?;
                }
                execution::launch_all(sessions, &ranks, *prefill, true)?;
                execution::synchronize(sessions)?;
                let choices = execution::router_choices(&sessions[0], &ranks[0], if *prefill { 64 } else { 1 })?;
                let (bits, traces) = oracle.run_traced_with_router_ids(input, &mut history, Some(&choices))
                    .map_err(|e| e.to_string())?;
                let expected = bits.iter().map(|&bits| f64::from(kernel_core::codec::bf16_to_f32(bits))).collect::<Vec<_>>();
                let qkv_direct = if *prefill {
                    (0..4).map(|rank| oracle.qkv_rank_direct(&traces[0].input_norm, rank))
                        .collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?
                } else { Vec::new() };
                let mut decoded = serde_json::Map::new();
                for (rank_index, (rank, session)) in ranks.iter().zip(sessions).enumerate() {
                    if let Some((reference,candidate,op_reference,op_candidate)) = &mut captures {
                        let rows = input.len()/4096;
                        for (index,name) in diagnostic::BOUNDARIES.iter().enumerate() {
                            let actual = execution::intermediate(session,rank,name,rows)?;
                            let width = diagnostic::widths(oracle.global)[index];
                            let expected = traces.iter().map(|trace|diagnostic::boundary(trace,name,rank_index,width))
                                .collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string())?.concat();
                            let name = format!("layer{layer:02}.{name}.r{rank_index}");
                            reference.f64(&name,rows,&expected).map_err(|e|e.to_string())?;
                            candidate.bf16(&name,rows,&actual).map_err(|e|e.to_string())?;
                        }
                        let norm = execution::intermediate(session,rank,"input_norm",rows)?;
                        let (weight,scales)=execution::diagnostic_qkv_operands(session,rank)?;
                        let operand=oracle.verify_qkv_operand(rank_index,&weight,&scales).map_err(|e|e.to_string())?;
                        decoded.insert(format!("layer{layer:02}.qkv.weight.r{rank_index}"),operand);
                        let values=norm.iter().map(|&b|f64::from(kernel_core::codec::bf16_to_f32(b))).collect::<Vec<_>>();
                        let numeric=oracle.qkv_rank_numeric(&values,rank_index).map_err(|e|e.to_string())?;
                        let name=format!("layer{layer:02}.qkv.r{rank_index}");
                        op_reference.f64(&name,rows,&numeric.values).map_err(|e|e.to_string())?;
                        let actual=execution::intermediate(session,rank,"qkv",rows)?;
                        op_candidate.bf16(&name,rows,&actual).map_err(|e|e.to_string())?;
                        // S and K preserve the existing elementwise bound, not a cosine surrogate.
                        op_reference.f64(&format!("{name}.sum_abs_products"),rows,&numeric.sum_abs_products).map_err(|e|e.to_string())?;
                        for capture in [op_reference,op_candidate] {
                            capture.bf16(&format!("layer{layer:02}.input_norm.r{rank_index}"),rows,&norm).map_err(|e|e.to_string())?;
                        }
                    }
                    let actual = execution::output(session, rank, if *prefill { 64 } else { 1 })?;
                    let max_abs = actual
                        .iter()
                        .zip(&expected)
                        .map(|(&a, &b)| {
                            (f64::from(kernel_core::codec::bf16_to_f32(a)) - b).abs()
                        })
                        .fold(0.0_f64, f64::max);
                    let gate = model_mimo26::oracle::compare_layer_tokens(&actual, &expected);
                    let mut receipt = std::collections::BTreeMap::new();
                    receipt.insert("layer".into(), serde_json::json!(layer));
                    receipt.insert("rank".into(), serde_json::json!(rank_index));
                    receipt.insert("phase".into(), serde_json::json!(if *prefill { "prefill64" } else { "decode" }));
                    receipt.insert("position".into(), serde_json::json!(position));
                    receipt.insert("tokens".into(), serde_json::json!(if *prefill { 64 } else { 1 }));
                    receipt.insert("cosine_threshold".into(), serde_json::json!(0.9999));
                    receipt.insert("max_abs_error".into(), serde_json::json!(max_abs));
                    receipt.insert("passed".into(), serde_json::json!(gate.is_ok()));
                    receipt.insert("router_near_tie_flips".into(), serde_json::json!(traces.iter().map(|trace| trace.router_flips).sum::<usize>()));
                    if *prefill {
                        let width = qkv_direct[rank_index].len();
                        let qkv = execution::intermediate(session, rank, "qkv", 64)?;
                        let (dot, aa, bb, max_abs) = qkv[..width]
                            .iter()
                            .zip(&qkv_direct[rank_index])
                            .fold((0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64),
                                |(dot, aa, bb, err), (&bits, &reference)| {
                                    let actual = f64::from(kernel_core::codec::bf16_to_f32(bits));
                                    (dot + actual * reference, aa + actual * actual,
                                     bb + reference * reference, err.max((actual - reference).abs()))
                                });
                        let cosine = dot / (aa.sqrt() * bb.sqrt()).max(1e-30);
                        receipt.insert("qkv_direct_f64_cosine".into(), serde_json::json!(cosine));
                        receipt.insert("qkv_direct_f64_max_abs".into(), serde_json::json!(max_abs));
                        if cosine < 0.9999 {
                            failures.push(format!(
                                "layer {layer} rank {rank_index}: QKV chunk vs direct f64 cosine {cosine:.7}, max abs {max_abs:.6}"
                            ));
                        }
                    }
                    let localization = if gate.is_err()
                        && rank_index == 0
                        && (*prefill || *position == 64)
                    {
                        Some(execution::localize(session, rank, &traces)?)
                    } else {
                        None
                    };
                    if *layer == 1 && *prefill && rank_index == 0 && gate.is_err() {
                        execution::save_moe_repro(session, rank, &images, &traces[0])?;
                    }
                    if let Err(error) = &gate {
                        receipt.insert("failure".into(), serde_json::json!(format!("{error:?}")));
                        receipt.insert("localization".into(), serde_json::json!(localization));
                    }
                    session.receipt("layers", receipt)?;
                    if let Err(error) = gate {
                        failures.push(format!(
                            "layer {layer} rank {rank_index} position {position}: {error:?}; max abs {max_abs}; {localization:?}"
                        ));
                    }
                }
                if let Some((reference,candidate,op_reference,op_candidate))=captures.take() {
                    let mut identity=oracle.diagnostic_identity().map_err(|e|e.to_string())?;
                    identity["history_before"]=before.expect("diagnostic history");
                    identity["history_after"]=diagnostic::history_identity(&history);
                    identity["router_substitutions"]=serde_json::json!({"checked_ids":choices,"near_tie_flips":traces.iter().map(|t|t.router_flips).sum::<usize>()});
                    reference.finish(identity.clone()).map_err(|e|e.to_string())?;
                    identity["decoded_operands"]=serde_json::Value::Object(decoded);
                    op_reference.finish(identity.clone()).map_err(|e|e.to_string())?;
                    // The candidate's weights are the image planned from the
                    // checkpoint root the oracle authenticated: the plan
                    // identities, linked to the oracle's source lock. No
                    // view is checked against the oracle's source ranges.
                    images.verify_unchanged()?;
                    let mut prepared=images.record();
                    prepared["source_lock_sha256"]=identity["checkpoint"]["source_lock_sha256"].clone();
                    identity["prepared_weights"]=prepared;
                    identity["reference_history_before"]=identity.as_object_mut().unwrap().remove("history_before").unwrap();
                    identity["reference_history_after"]=identity.as_object_mut().unwrap().remove("history_after").unwrap();
                    candidate.finish(identity.clone()).map_err(|e|e.to_string())?;
                    op_candidate.finish(identity).map_err(|e|e.to_string())?;
                }
                if diagnostic_root.is_some() {lineage.push(diagnostic::bits_hash(input));}
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{} real-layer comparisons failed; first: {}",
                failures.len(),
                failures[0]
            ))
        }
    })
    .expect("real MiMo layer gate");
}

pub mod execution {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::ops::Range;
    use std::path::Path;

    pub use super::mimo_planned::PlannedImages as Images;
    use kernel_build::EntryParam;
    use kernel_core::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};
    use kernel_core::contract::{
        Argument, ArgumentList, AttentionPhase, CommQuery, DType, LaunchPlan, RoutedCapacity,
        RuntimeCounts, Validate,
    };
    use kernel_core::oracle::SiluGlu;
    use kernel_core::quantize::{QuantFormat, Quantized, ScaleFormat, Scales};
    use kernel_core::transform::{Layout, Packed, ScaleTileLayout, Transform};
    use kernel_lab::launch::{self, Bindings};
    use kernel_lab::poison::{self, Fill, POISON, WriteMap};
    use kernel_lab::{LabResult, Session};
    use kernels_sm120::{attention, comm, gemm, moe, row};

    use model_mimo26_contract::weight_plan::TEXT_IMAGE_ROLE;
    use runtime_cuda::{DeviceBuffer, Function, Module};

    const H: usize = 4096;
    const W_IN: u64 = 1;
    const W_POST: u64 = 2;
    const W_QKV: u64 = 3;
    const W_QKVS: u64 = 4;
    const W_O: u64 = 5;
    const W_ROUTER: u64 = 6;
    const W_BIAS: u64 = 7;
    const W_GATE: u64 = 8;
    const W_GATES: u64 = 9;
    const W_DOWN: u64 = 10;
    const W_DOWNS: u64 = 11;
    const W_SINK: u64 = 12;
    const X: u64 = 20;
    const NORM: u64 = 21;
    const QKV: u64 = 22;
    const Q: u64 = 23;
    const KNEW: u64 = 24;
    const VNEW: u64 = 25;
    const KPOOL: u64 = 26;
    const VPOOL: u64 = 27;
    const ATTN: u64 = 28;
    const ATTN_WORK: u64 = 29;
    const O_PART: u64 = 30;
    const O_TOTAL: u64 = 31;
    const RESID: u64 = 32;
    const POSTNORM: u64 = 33;
    const LOGITS: u64 = 34;
    const IDS: u64 = 35;
    const TOPK: u64 = 36;
    const COUNTS: u64 = 37;
    const OFFSETS: u64 = 38;
    const SORTED: u64 = 39;
    const PADDED: u64 = 40;
    const GATEOUT: u64 = 41;
    const DOWNOUT: u64 = 42;
    const COMBINE: u64 = 43;
    const MOE_TOTAL: u64 = 44;
    const FINAL: u64 = 45;
    const NORM_SCRATCH: u64 = 46;
    const TILE_COUNTS: u64 = 47;
    const TILE_OFFSETS: u64 = 48;
    const LIVE: u64 = 50;
    const POS: u64 = 51;
    const TOKEN_BATCH: u64 = 52;
    const LAST_POS: u64 = 53;
    const TABLE: u64 = 54;
    const SLOT: u64 = 55;
    const SEQ: u64 = 56;
    const START: u64 = 57;
    const LEN: u64 = 58;
    const PREFIX: u64 = 59;
    const LIVE_BATCH: u64 = 60;
    const ACTIVE: u64 = 61;
    const DUMMY: u64 = 62;
    const ONE: u64 = 63;
    const GEMM_PART: [u64; 3] = [70, 72, 74];
    const GEMM_SEM: [u64; 3] = [71, 73, 75];
    const MOE_PART: [u64; 2] = [80, 81];
    const COMM_INBOX: [u64; 2] = [90, 93];
    const COMM_FLAGS: [u64; 2] = [91, 94];
    const COMM_EPOCH: [u64; 2] = [92, 95];
    const DENSE: u64 = 96;
    // Embedding and head fixtures, used by the model audit only.
    #[allow(dead_code)]
    const W_EMBED: u64 = 97;
    #[allow(dead_code)]
    const W_HEAD: u64 = 98;
    #[allow(dead_code)]
    const TOKEN_IDS: u64 = 99;
    #[allow(dead_code)]
    const HEAD_OUT: u64 = 100;

    /// Slots of the sliding-window KV ring per sequence.
    const RING_SIZE: u32 = 128;

    /// KV heads and KV cache page size of a layer's RoPE and attention.
    fn kv_layout(global: bool) -> (u32, u32) {
        if global { (1, 64) } else { (2, 32) }
    }

    /// Buffers a decode pass writes in full before reading them, which
    /// `poison_decode` fills whole.
    #[allow(dead_code)] // Used by decode_chain; this module is shared.
    const DECODE_OUTPUTS: [(&str, u64); 23] = [
        ("norm", NORM),
        ("qkv", QKV),
        ("q", Q),
        ("k_new", KNEW),
        ("v_new", VNEW),
        ("attention_workspace", ATTN_WORK),
        ("attention", ATTN),
        ("o_partial", O_PART),
        ("attention_residual", RESID),
        ("post_norm", POSTNORM),
        ("router_logits", LOGITS),
        ("route_ids", IDS),
        ("route_weights", TOPK),
        ("expert_counts", COUNTS),
        ("expert_offsets", OFFSETS),
        ("sorted_routes", SORTED),
        ("padded_rows", PADDED),
        ("active_experts", ACTIVE),
        ("gate_up", GATEOUT),
        ("down", DOWNOUT),
        ("moe_combine", COMBINE),
        ("norm_scratch", NORM_SCRATCH),
        ("final", FINAL),
    ];

    /// Global-attention layers; the others use the sliding window.
    pub const GLOBAL_LAYERS: [usize; 9] = [0, 5, 11, 17, 23, 29, 35, 41, 47];
    pub const LAYERS: usize = 48;
    #[allow(dead_code)]
    pub const VOCAB_PER_RANK: usize = 38_144;

    /// The TP4 text image of layers 1 and 5, planned from `checkpoint`.
    pub fn images(checkpoint: &Path) -> LabResult<Images> {
        layer_images(checkpoint, Some(&BTreeSet::from([1, 5])))
    }

    /// The TP4 text image of `layers` (`None`: every layer), with the
    /// embedding, final norm and head, planned from `checkpoint`.
    pub fn layer_images(checkpoint: &Path, layers: Option<&BTreeSet<usize>>) -> LabResult<Images> {
        Images::plan(checkpoint, TEXT_IMAGE_ROLE, 4, layers, None)
    }

    fn bytes_u16(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    fn bytes_i32(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    fn bytes_i64(values: &[i64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    pub fn seeded_hidden(rows: usize, seed: u64) -> Vec<u16> {
        let mut state = seed;
        (0..rows * H)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let centered = ((state >> 32) % 1025) as i32 - 512;
                f32_to_bf16(centered as f32 * (0.7 / 512.0))
            })
            .collect()
    }

    pub struct Stage {
        pub name: String,
        pub plan: LaunchPlan,
        pub args: ArgumentList,
        pub function: Function,
    }

    pub struct Rank {
        pub buffers: BTreeMap<u64, DeviceBuffer>,
        pub prefill: Vec<Stage>,
        pub decode: Vec<Stage>,
        _modules: Vec<Module>,
        pub layer: usize,
        pub rank: usize,
        pub batch: usize,
        pub rows_capacity: usize,
        pub global: bool,
        /// Follow bundle-mimo26's launch choices (dense split-K, fused
        /// prefill all-reduce and norm) instead of the fixture defaults.
        pub bundle_mode: bool,
        last_prefill: std::cell::Cell<Option<bool>>,
    }

    fn gemm_variant(symbol: &str) -> LabResult<gemm::GemmVariant> {
        gemm::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|v| v.symbol == symbol)
            .ok_or_else(|| format!("missing GEMM {symbol}"))
    }

    fn row_entry(symbol: &str) -> LabResult<row::RowInstantiation> {
        row::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|entry| entry.symbol == symbol)
            .ok_or_else(|| format!("missing row {symbol}"))
    }

    fn moe_variant(
        phase: moe::Phase,
        projection: moe::Projection,
        tokens: u32,
    ) -> LabResult<moe::Variant> {
        moe::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|v| {
                v.profile == model_mimo26_contract::sm120::config::PROFILE
                    && v.phase == phase
                    && v.projection == projection
                    && v.tokens_capacity == tokens
                    && v.path == moe::InputPath::A16Gather
            })
            .ok_or_else(|| format!("missing MiMo MoE {phase:?} {projection:?} t{tokens}"))
    }

    impl Rank {
        fn alloc(&mut self, session: &Session, id: u64, bytes: usize) -> LabResult<()> {
            if self
                .buffers
                .get(&id)
                .is_some_and(|buffer| buffer.byte_len() >= bytes.max(1))
            {
                return Ok(());
            }
            self.buffers.insert(id, session.allocate(bytes.max(1))?);
            Ok(())
        }

        fn upload(&mut self, session: &Session, id: u64, bytes: &[u8]) -> LabResult<()> {
            let buffer = session.allocate(bytes.len().max(1))?;
            if !bytes.is_empty() {
                buffer
                    .copy_from_bytes(bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
            }
            self.buffers.insert(id, buffer);
            Ok(())
        }

        fn zero(&mut self, session: &Session, id: u64, bytes: usize) -> LabResult<()> {
            if self
                .buffers
                .get(&id)
                .is_some_and(|buffer| buffer.byte_len() >= bytes.max(1))
            {
                return Ok(());
            }
            self.upload(session, id, &vec![0; bytes.max(1)])
        }

        fn buffer(&self, id: u64) -> &DeviceBuffer {
            &self.buffers[&id]
        }

        #[allow(clippy::too_many_arguments)]
        fn push_row(
            &self,
            stages: &mut Vec<Stage>,
            module: &Module,
            symbol: &str,
            rows: u32,
            pointers: &[(&str, u64)],
            u32s: &[(&str, u32)],
            u64s: &[(&str, u64)],
        ) -> LabResult<()> {
            let entry = row_entry(symbol)?;
            let args = entry
                .params
                .iter()
                .map(|param| match param {
                    EntryParam::DevicePointer { name, .. } => Ok(pointers
                        .iter()
                        .find(|(key, _)| *key == name)
                        .map_or(Argument::NullDevicePointer, |(_, id)| {
                            Argument::DeviceBuffer(*id)
                        })),
                    EntryParam::U32 { name } => u32s
                        .iter()
                        .find(|(key, _)| *key == name)
                        .map(|(_, value)| Argument::U32(*value))
                        .ok_or_else(|| format!("{symbol}: missing {name}")),
                    EntryParam::U64 { name } => u64s
                        .iter()
                        .find(|(key, _)| *key == name)
                        .map(|(_, value)| Argument::U64(*value))
                        .ok_or_else(|| format!("{symbol}: missing {name}")),
                    other => Err(format!("{symbol}: unsupported row parameter {other:?}")),
                })
                .collect::<LabResult<Vec<_>>>()?;
            let counts = RuntimeCounts {
                m: rows,
                batch: self.batch as u32,
                context: 68,
                routed_rows: rows * 8,
                split_k: 1,
            };
            stages.push(Stage {
                name: symbol.into(),
                plan: entry.plan(rows, counts).map_err(|e| e.to_string())?,
                args: entry.pack_arguments(args).map_err(|e| e.to_string())?,
                function: module.function(symbol).map_err(|e| e.to_string())?,
            });
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn push_gemm(
            &mut self,
            session: &Session,
            stages: &mut Vec<Stage>,
            module: &Module,
            symbol: &str,
            index: usize,
            rows: u32,
            activation: u64,
            weight: u64,
            scales: Option<u64>,
            output: u64,
        ) -> LabResult<()> {
            let variant = gemm_variant(symbol)?;
            let counts = RuntimeCounts {
                m: rows,
                batch: self.batch as u32,
                context: 68,
                routed_rows: rows * 8,
                split_k: if self.bundle_mode {
                    gemm::required_split_k(&variant.key).unwrap_or(1)
                } else {
                    gemm::choose_split_k(&variant.key).map_err(|e| e.to_string())?
                },
            };
            let plan = gemm::plan(&variant.key, counts).map_err(|e| e.to_string())?;
            self.alloc(session, GEMM_PART[index], plan.workspace_bytes as usize)?;
            self.zero(session, GEMM_SEM[index], plan.workspace_bytes as usize)?;
            // The engine builds maps with its own tile and swizzle rules.
            let maps =
                gemm::tensor_maps(&variant.key, weight, activation).map_err(|e| e.to_string())?;
            let args = gemm::arguments(
                &variant.key,
                counts,
                maps,
                gemm::GemmBuffers {
                    output,
                    partial: GEMM_PART[index],
                    semaphores: GEMM_SEM[index],
                    weight_scales: scales,
                    activation_scales: None,
                    residual: None,
                    weight_global_scale: None,
                    activation_global_scale: None,
                },
            )
            .map_err(|e| e.to_string())?;
            let function = module.function(symbol).map_err(|e| e.to_string())?;
            if plan.dynamic_shared_bytes >= 48 * 1024 {
                function
                    .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                    .map_err(|e| e.to_string())?;
            }
            stages.push(Stage {
                name: symbol.into(),
                plan,
                args,
                function,
            });
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn push_moe(
            &mut self,
            session: &Session,
            stages: &mut Vec<Stage>,
            module: &Module,
            variant: &moe::Variant,
            rows: u32,
            activation: u64,
            weight: u64,
            scales: u64,
            output: u64,
            partial_index: usize,
        ) -> LabResult<()> {
            let counts = RuntimeCounts {
                m: rows,
                batch: self.batch as u32,
                context: 68,
                routed_rows: rows * 8,
                split_k: variant.knobs.split_k_capacity,
            };
            let plan = variant.plan(counts).map_err(|e| e.to_string())?;
            let partial = MOE_PART[partial_index];
            self.alloc(session, partial, plan.workspace_bytes.max(1) as usize)?;
            let handles = [
                activation, DUMMY, ONE, scales, ONE, output, TOPK, OFFSETS, SORTED, PADDED, LIVE,
                ACTIVE, DUMMY, DUMMY, partial,
            ];
            let args = variant
                .arguments(
                    handles,
                    variant
                        .weight_tensor_map(weight)
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
            let function = module
                .function(&variant.symbol)
                .map_err(|e| e.to_string())?;
            if plan.dynamic_shared_bytes > 48 * 1024 {
                function
                    .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                    .map_err(|e| e.to_string())?;
            }
            stages.push(Stage {
                name: variant.symbol.clone(),
                plan,
                args,
                function,
            });
            if let Some(plan) = variant.reduction_plan(counts).map_err(|e| e.to_string())? {
                let symbol = variant
                    .reduction_symbol()
                    .ok_or("missing MoE reduction symbol")?;
                let args = variant
                    .reduction_arguments(handles)
                    .map_err(|e| e.to_string())?
                    .ok_or("missing MoE reduction arguments")?;
                stages.push(Stage {
                    name: symbol.clone(),
                    plan,
                    args,
                    function: module.function(&symbol).map_err(|e| e.to_string())?,
                });
            }
            Ok(())
        }

        fn push_comm(
            &self,
            stages: &mut Vec<Stage>,
            module: &Module,
            rows: u32,
            input: u64,
            output: u64,
            index: usize,
        ) -> LabResult<()> {
            let elements = u64::from(rows) * H as u64;
            let query = CommQuery::AllReduce {
                ranks: 4,
                elements,
                dtype: DType::Bf16,
                fused_residual: false,
                fused_rmsnorm: None,
            };
            let plan = comm::plan(&query, rows).map_err(|e| e.to_string())?;
            let base = if index == 0 { 1000 } else { 2000 };
            let inboxes = (0..4).map(|rank| base + rank).collect::<Vec<_>>();
            let flags = (0..4).map(|rank| base + 100 + rank).collect::<Vec<_>>();
            let scatter = comm::arguments(comm::OneShotCall {
                input,
                output,
                peer_inboxes: inboxes,
                peer_flags: flags.clone(),
                epoch: COMM_EPOCH[index],
                elements,
                rank: self.rank as u32,
                ranks: 4,
            })
            .map_err(|e| e.to_string())?;
            let variant = comm::choose(&query).map_err(|e| e.to_string())?;
            let symbol = match variant {
                comm::Variant::OneShot => "chop_comm_allreduce_oneshot_bf16_v1",
                comm::Variant::TwoShot => "chop_comm_reduce_scatter_bf16_v1",
                other => return Err(format!("unsupported MiMo comm variant {other:?}")),
            };
            stages.push(Stage {
                name: format!("{symbol}:{}", index),
                plan,
                args: scatter,
                function: module.function(symbol).map_err(|e| e.to_string())?,
            });
            if variant == comm::Variant::TwoShot {
                // The comm engine has no gather argument builder yet.
                let gather = ArgumentList(vec![
                    Argument::DeviceBuffer(output),
                    Argument::DevicePointerArray((0..4).map(|rank| base + 200 + rank).collect()),
                    Argument::DevicePointerArray(flags),
                    Argument::DeviceBuffer(COMM_EPOCH[index]),
                    Argument::U64(elements),
                    Argument::U32(self.rank as u32),
                    Argument::U32(4),
                ]);
                gather.validate().map_err(|e| e.to_string())?;
                let symbol = "chop_comm_allgather_scattered_bf16_v1";
                stages.push(Stage {
                    name: format!("{symbol}:{}", index),
                    plan,
                    args: gather,
                    function: module.function(symbol).map_err(|e| e.to_string())?,
                });
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn push_fused_comm(
            &self,
            stages: &mut Vec<Stage>,
            module: &Module,
            rows: u32,
            input: u64,
            output: u64,
            residual: u64,
            residual_output: u64,
            weights: u64,
            index: usize,
            combine: bool,
        ) -> LabResult<()> {
            let elements = u64::from(rows) * H as u64;
            let query = CommQuery::AllReduce {
                ranks: 4,
                elements,
                dtype: DType::Bf16,
                fused_residual: true,
                fused_rmsnorm: Some(kernel_core::contract::RmsNormEpilogue {
                    kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                    width: 4096,
                    eps: 1e-6,
                }),
            };
            if comm::choose(&query).map_err(|e| e.to_string())? != comm::Variant::FusedOneShot {
                return Err("decode fixture requires fused one-shot comm".into());
            }
            let base = if index == 0 { 1000 } else { 2000 };
            let call = comm::FusedWithResidualOutputCall {
                fused: comm::FusedOneShotCall {
                    base: comm::OneShotCall {
                        input,
                        output,
                        peer_inboxes: (0..4).map(|rank| base + rank).collect(),
                        peer_flags: (0..4).map(|rank| base + 100 + rank).collect(),
                        epoch: COMM_EPOCH[index],
                        elements,
                        rank: self.rank as u32,
                        ranks: 4,
                    },
                    residual: Some(residual),
                    weights,
                    epsilon: 1e-6,
                },
                residual_output,
            };
            let (args, symbol) = if combine {
                (
                    comm::fused_combine_arguments(call, Some(LIVE)),
                    "chop_comm_allreduce_combine_norm_bf16_v1",
                )
            } else {
                (
                    comm::fused_arguments_v2(call),
                    "chop_comm_allreduce_oneshot_norm_bf16_v2",
                )
            };
            let args = args.map_err(|e| e.to_string())?;
            stages.push(Stage {
                name: format!("{symbol}:{index}"),
                plan: comm::plan(&query, rows).map_err(|e| e.to_string())?,
                args,
                function: module.function(symbol).map_err(|e| e.to_string())?,
            });
            Ok(())
        }

        /// bundle-mimo26's prefill hidden reduction: a reduce-scatter that
        /// adds the residual before its one rounding, then a gather that
        /// applies HF RMSNorm. `buffers` are input, normalized output,
        /// residual, residual output, norm weight.
        fn push_fused_prefill_comm(
            &self,
            stages: &mut Vec<Stage>,
            module: &Module,
            rows: u32,
            buffers: [u64; 5],
            index: usize,
        ) -> LabResult<()> {
            let [input, output, residual, residual_output, weights] = buffers;
            let elements = u64::from(rows) * H as u64;
            let query = CommQuery::AllReduce {
                ranks: 4,
                elements,
                dtype: DType::Bf16,
                fused_residual: true,
                fused_rmsnorm: Some(kernel_core::contract::RmsNormEpilogue {
                    kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                    width: 4096,
                    eps: 1e-6,
                }),
            };
            if comm::choose(&query).map_err(|e| e.to_string())? != comm::Variant::FusedTwoShot {
                return Err("bundle prefill requires the fused two-shot comm".into());
            }
            let plan = comm::plan(&query, rows).map_err(|e| e.to_string())?;
            let base = if index == 0 { 1000 } else { 2000 };
            let flags = (0..4).map(|rank| base + 100 + rank).collect::<Vec<_>>();
            let call = comm::OneShotCall {
                input,
                output,
                peer_inboxes: (0..4).map(|rank| base + rank).collect(),
                peer_flags: flags,
                epoch: COMM_EPOCH[index],
                elements,
                rank: self.rank as u32,
                ranks: 4,
            };
            let gather = comm::round_once_gather_arguments(
                &call,
                (0..4).map(|rank| base + 200 + rank).collect(),
                weights,
                1e-6,
                residual_output,
            )
            .map_err(|e| e.to_string())?;
            let scatter =
                comm::round_once_scatter_arguments(call, residual).map_err(|e| e.to_string())?;
            for (symbol, args) in [
                (comm::PREFILL_ROUND_ONCE_SYMBOL, scatter),
                (comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL, gather),
            ] {
                stages.push(Stage {
                    name: format!("{symbol}:{index}"),
                    plan,
                    args,
                    function: module.function(symbol).map_err(|e| e.to_string())?,
                });
            }
            Ok(())
        }

        /// Buffer a peer reads in the gather stage of hidden reduction `index`.
        fn gather_target(&self, index: usize) -> u64 {
            match (self.bundle_mode, index) {
                (false, 0) => O_TOTAL,
                (false, _) => MOE_TOTAL,
                (true, 0) => POSTNORM,
                (true, _) => NORM_SCRATCH,
            }
        }

        fn push_decode_route(
            &self,
            stages: &mut Vec<Stage>,
            module: &Module,
            rows: u32,
        ) -> LabResult<()> {
            let symbol = model_mimo26_contract::sm120::moe::DECODE_ROUTE_SYMBOL;
            stages.push(Stage {
                name: symbol.into(),
                plan: moe::decode_route_plan(rows, rows).map_err(|e| e.to_string())?,
                args: moe::decode_route_arguments(
                    [
                        LOGITS, W_BIAS, IDS, TOPK, COUNTS, OFFSETS, SORTED, PADDED, LIVE, ACTIVE,
                    ],
                    1.0,
                )
                .map_err(|e| e.to_string())?,
                function: module.function(symbol).map_err(|e| e.to_string())?,
            });
            Ok(())
        }

        fn push_attention(
            &mut self,
            session: &Session,
            stages: &mut Vec<Stage>,
            module: &Module,
            phase: AttentionPhase,
            rows: u32,
        ) -> LabResult<()> {
            let global = self.global;
            let name = match (global, phase) {
                (true, AttentionPhase::Prefill) => "mimo_ga_bf16_prefill",
                (true, AttentionPhase::Decode) => "mimo_ga_bf16_decode",
                (false, AttentionPhase::Prefill) => "mimo_swa_prefill",
                (false, AttentionPhase::Decode) => "mimo_swa_decode",
                (true, AttentionPhase::Verify) => "mimo_ga_bf16_verify",
                (false, AttentionPhase::Verify) => "mimo_swa_verify",
            };
            let mut variant = attention::catalog(kernel_lab::catalogs())
                .into_iter()
                .find(|v| v.name == name)
                .ok_or_else(|| format!("missing attention {name}"))?;
            if phase == AttentionPhase::Prefill {
                variant.query_capacity = rows;
            }
            let slots = if global {
                self.batch * self.rows_capacity.div_ceil(64).max(2)
            } else {
                self.batch
            };
            let maps = if phase == AttentionPhase::Prefill {
                let (k, v) = variant
                    .cache_maps(slots as u32, KPOOL, VPOOL)
                    .map_err(|e| e.to_string())?;
                (Some(k), Some(v))
            } else {
                (None, None)
            };
            // Prefill plans the variant's largest context, as the served
            // workspace does: the kernel splits the keys of rows past 256
            // positions whatever the plan says, so a smaller context would
            // skip their merge and undersize the workspace. The plan is
            // fixed before the inputs, whose prefix may be nonzero.
            let counts = RuntimeCounts {
                m: rows,
                batch: self.batch as u32,
                context: if phase == AttentionPhase::Prefill {
                    variant.query.max_context_capacity
                } else {
                    68
                },
                routed_rows: rows * 8,
                split_k: 1,
            };
            let buffers = attention::Buffers {
                q: Q,
                k_cache: KPOOL,
                v_cache: VPOOL,
                k_page_stride_bytes: super::tight_page_stride(&variant, variant.query.qk_dim),
                v_page_stride_bytes: super::tight_page_stride(&variant, variant.query.v_dim),
                k_map: maps.0,
                v_map: maps.1,
                block_tables: global.then_some(TABLE),
                slot_ids: (!global).then_some(SLOT),
                seq_lens: Some(SEQ),
                q_start: (phase == AttentionPhase::Prefill).then_some(START),
                q_len: (phase == AttentionPhase::Prefill).then_some(LEN),
                prefix_len: (phase == AttentionPhase::Prefill).then_some(PREFIX),
                live_batch: LIVE_BATCH,
                k_new: (!global && phase == AttentionPhase::Prefill).then_some(KNEW),
                v_new: (!global && phase == AttentionPhase::Prefill).then_some(VNEW),
                sink: (!global).then_some(W_SINK),
                output: ATTN,
                workspace: ATTN_WORK,
                k_scale: 1.0,
                v_scale: 1.0,
            };
            let planned = variant
                .plan_for_bucket(self.batch as u32, counts, buffers)
                .map_err(|e| e.to_string())?;
            self.alloc(session, ATTN_WORK, planned.partial.workspace_bytes as usize)?;
            for (symbol, plan, args) in [
                (
                    planned.partial_symbol,
                    planned.partial,
                    planned.partial_args,
                ),
                (
                    format!("chop_attn_{name}_merge_v1"),
                    planned.merge,
                    planned.merge_args,
                ),
            ] {
                let function = module.function(&symbol).map_err(|e| e.to_string())?;
                if plan.dynamic_shared_bytes > 48 * 1024 {
                    function
                        .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                        .map_err(|e| e.to_string())?;
                }
                stages.push(Stage {
                    name: symbol.clone(),
                    plan,
                    args,
                    function,
                });
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn build_phase(
            &mut self,
            session: &Session,
            phase: AttentionPhase,
            row_module: &Module,
            gemm_module: &Module,
            attention_module: &Module,
            comm_module: &Module,
            route_module: &Module,
            gate_module: &Module,
            down_module: &Module,
        ) -> LabResult<Vec<Stage>> {
            let prefill = phase == AttentionPhase::Prefill;
            let rows = if prefill {
                self.rows_capacity as u32
            } else {
                self.batch as u32
            };
            let global = self.global;
            let (kv_heads, page_size) = kv_layout(global);
            let max_pages = 1_048_576 / page_size;
            let k_stride = u64::from(page_size * kv_heads * 192 * 2);
            let v_stride = u64::from(page_size * kv_heads * 128 * 2);
            let suffix = if prefill {
                "prefill".into()
            } else {
                format!("m{rows}")
            };
            let mut stages = Vec::new();
            self.push_row(
                &mut stages,
                row_module,
                "chop_row_rms_mimo_hf4096_v1",
                rows,
                &[("x", X), ("weight", W_IN), ("y", NORM), ("live_rows", LIVE)],
                &[("eps_bits", 1e-6_f32.to_bits())],
                &[],
            )?;
            self.push_gemm(
                session,
                &mut stages,
                gemm_module,
                &format!(
                    "chop_gemm_mimo_qkv_{}_{}_v1",
                    if global { "ga" } else { "swa" },
                    suffix
                ),
                0,
                rows,
                NORM,
                W_QKV,
                Some(W_QKVS),
                QKV,
            )?;
            let rope_name = format!(
                "chop_row_rope_mimo_{}_computed_p{}_pos32_v1",
                if global {
                    "ga_bf16"
                } else if prefill {
                    "swa_prefill"
                } else {
                    "swa_decode"
                },
                page_size
            );
            self.push_row(
                &mut stages,
                row_module,
                &rope_name,
                rows,
                &[
                    ("qkv", QKV),
                    ("q", Q),
                    ("k_new", KNEW),
                    ("v_new", VNEW),
                    ("k_pool", KPOOL),
                    ("v_pool", VPOOL),
                    ("block_tables", TABLE),
                    ("slot_ids", SLOT),
                    ("token_batch", TOKEN_BATCH),
                    ("positions", POS),
                    ("live_tokens", LIVE),
                ],
                &[
                    ("max_pages", max_pages),
                    ("ring_size", RING_SIZE),
                    (
                        "theta_bits",
                        if global { 1e7_f32 } else { 1e4_f32 }.to_bits(),
                    ),
                    ("v_multiplier_bits", 0.707_f32.to_bits()),
                    ("k_scale_bits", 1.0_f32.to_bits()),
                    ("v_scale_bits", 1.0_f32.to_bits()),
                ],
                &[
                    ("k_page_stride_bytes", k_stride),
                    ("v_page_stride_bytes", v_stride),
                ],
            )?;
            self.push_attention(session, &mut stages, attention_module, phase, rows)?;
            // Ring prefill attention reads the cached prefix from the ring
            // and the chunk from k_new/v_new; as served, the chunk's tail
            // replaces ring cells only after attention has read them.
            if !global && prefill {
                self.push_row(
                    &mut stages,
                    row_module,
                    if rows >= 2048 && std::env::var_os("CHOPINFER_PREFILL_LEGACY_RING").is_none() {
                        "chop_row_ring_write_mimo_bf16_single_sequence_pos32_v1"
                    } else {
                        "chop_row_ring_write_mimo_bf16_pos32_v1"
                    },
                    rows,
                    &[
                        ("k_new", KNEW),
                        ("v_new", VNEW),
                        ("k_ring", KPOOL),
                        ("v_ring", VPOOL),
                        ("slot_ids", SLOT),
                        ("token_batch", TOKEN_BATCH),
                        ("positions", POS),
                        ("last_positions", LAST_POS),
                        ("live_tokens", LIVE),
                    ],
                    &[
                        ("ring_size", RING_SIZE),
                        ("k_scale_bits", 1.0_f32.to_bits()),
                        ("v_scale_bits", 1.0_f32.to_bits()),
                    ],
                    &[],
                )?;
            }
            self.push_gemm(
                session,
                &mut stages,
                gemm_module,
                &format!("chop_gemm_mimo_o_{suffix}_v1"),
                1,
                rows,
                ATTN,
                W_O,
                None,
                O_PART,
            )?;
            if prefill && self.bundle_mode {
                self.push_fused_prefill_comm(
                    &mut stages,
                    comm_module,
                    rows,
                    [O_PART, POSTNORM, X, RESID, W_POST],
                    0,
                )?;
            } else if prefill {
                self.push_comm(&mut stages, comm_module, rows, O_PART, O_TOTAL, 0)?;
                self.push_row(
                    &mut stages,
                    row_module,
                    "chop_row_rms_mimo_hf4096_residual_v1",
                    rows,
                    &[
                        ("x", O_TOTAL),
                        ("weight", W_POST),
                        ("residual", X),
                        ("residual_out", RESID),
                        ("y", POSTNORM),
                        ("live_rows", LIVE),
                    ],
                    &[("eps_bits", 1e-6_f32.to_bits())],
                    &[],
                )?;
            } else {
                self.push_fused_comm(
                    &mut stages,
                    comm_module,
                    rows,
                    O_PART,
                    POSTNORM,
                    X,
                    RESID,
                    W_POST,
                    0,
                    false,
                )?;
            }
            // Opt-in: the decode all-reduce combines the routed rows itself.
            let fused_combine = !prefill
                && self.layer != 0
                && std::env::var("CHOPINFER_MIMO_FUSED_COMBINE").as_deref() == Ok("1");
            if self.layer == 0 {
                // Dense MLP: fused gate/up with a SwiGLU epilogue, then down.
                self.push_gemm(
                    session,
                    &mut stages,
                    gemm_module,
                    &format!("chop_gemm_mimo_mlp_gate_up_{suffix}_v1"),
                    2,
                    rows,
                    POSTNORM,
                    W_GATE,
                    Some(W_GATES),
                    DENSE,
                )?;
                self.push_gemm(
                    session,
                    &mut stages,
                    gemm_module,
                    &format!("chop_gemm_mimo_mlp_down_{suffix}_v1"),
                    0,
                    rows,
                    DENSE,
                    W_DOWN,
                    Some(W_DOWNS),
                    COMBINE,
                )?;
            } else {
                self.push_gemm(
                    session,
                    &mut stages,
                    gemm_module,
                    &format!("chop_gemm_mimo_router_{suffix}_v1"),
                    2,
                    rows,
                    POSTNORM,
                    W_ROUTER,
                    None,
                    LOGITS,
                )?;
                let (moe_phase, moe_capacity) = super::moe_entry(prefill, rows);
                if moe_phase == moe::Phase::Prefill {
                    self.push_row(
                        &mut stages,
                        row_module,
                        "chop_row_router_mimo_sigmoid256_v1",
                        rows,
                        &[
                            ("logits", LOGITS),
                            ("bias", W_BIAS),
                            ("ids", IDS),
                            ("weights", TOPK),
                            ("live_rows", LIVE),
                        ],
                        &[("scale_bits", 1.0_f32.to_bits())],
                        &[],
                    )?;
                    // Decode buckets permute by tiles, as served.
                    if !prefill
                        || rows >= 2048
                            && std::env::var_os("CHOPINFER_PREFILL_LEGACY_ROUTE").is_none()
                    {
                        for (name, pointers) in [
                            (
                                "chop_row_permute_tile_count_mimo_b64_v1",
                                vec![
                                    ("ids", IDS),
                                    ("tile_counts", TILE_COUNTS),
                                    ("live_rows", LIVE),
                                ],
                            ),
                            (
                                "chop_row_permute_tile_prefix_mimo_b64_v1",
                                vec![
                                    ("tile_counts", TILE_COUNTS),
                                    ("counts", COUNTS),
                                    ("tile_offsets", TILE_OFFSETS),
                                    ("live_rows", LIVE),
                                ],
                            ),
                            (
                                "chop_row_permute_tile_scan_mimo_b64_v1",
                                vec![
                                    ("counts", COUNTS),
                                    ("offsets", OFFSETS),
                                    ("sorted_rows", SORTED),
                                    ("padded_rows", PADDED),
                                ],
                            ),
                            (
                                "chop_row_permute_tile_scatter_mimo_b64_v1",
                                vec![
                                    ("ids", IDS),
                                    ("tile_offsets", TILE_OFFSETS),
                                    ("offsets", OFFSETS),
                                    ("sorted_rows", SORTED),
                                    ("live_rows", LIVE),
                                ],
                            ),
                        ] {
                            self.push_row(
                                &mut stages,
                                row_module,
                                name,
                                rows,
                                &pointers,
                                &[],
                                &[],
                            )?;
                        }
                    } else {
                        let permute = "mimo_b64";
                        self.push_row(
                            &mut stages,
                            row_module,
                            &format!("chop_row_permute_count_{permute}_v1"),
                            rows,
                            &[("ids", IDS), ("counts", COUNTS), ("live_rows", LIVE)],
                            &[],
                            &[],
                        )?;
                        self.push_row(
                            &mut stages,
                            row_module,
                            &format!("chop_row_permute_scan_{permute}_v1"),
                            rows,
                            &[
                                ("counts", COUNTS),
                                ("offsets", OFFSETS),
                                ("padded_rows", PADDED),
                            ],
                            &[],
                            &[],
                        )?;
                        self.push_row(
                            &mut stages,
                            row_module,
                            &format!("chop_row_permute_scatter_{permute}_v1"),
                            rows,
                            &[
                                ("ids", IDS),
                                ("offsets", OFFSETS),
                                ("sorted_rows", SORTED),
                                ("live_rows", LIVE),
                            ],
                            &[],
                            &[],
                        )?;
                    }
                } else {
                    self.push_decode_route(&mut stages, route_module, rows)?;
                }
                let gate = moe_variant(moe_phase, moe::Projection::GateUp, moe_capacity)?;
                let down = moe_variant(moe_phase, moe::Projection::Down, moe_capacity)?;
                self.push_moe(
                    session,
                    &mut stages,
                    gate_module,
                    &gate,
                    rows,
                    POSTNORM,
                    W_GATE,
                    W_GATES,
                    GATEOUT,
                    0,
                )?;
                self.push_moe(
                    session,
                    &mut stages,
                    down_module,
                    &down,
                    rows,
                    GATEOUT,
                    W_DOWN,
                    W_DOWNS,
                    DOWNOUT,
                    1,
                )?;
                if !fused_combine {
                    self.push_row(
                        &mut stages,
                        row_module,
                        "chop_row_combine_mimo4096_v1",
                        rows,
                        &[
                            ("routed", DOWNOUT),
                            ("weights", TOPK),
                            ("output", COMBINE),
                            ("live_rows", LIVE),
                        ],
                        &[],
                        &[],
                    )?;
                }
            }
            if prefill && self.bundle_mode {
                self.push_fused_prefill_comm(
                    &mut stages,
                    comm_module,
                    rows,
                    [COMBINE, NORM_SCRATCH, RESID, FINAL, W_POST],
                    1,
                )?;
            } else if prefill {
                self.push_comm(&mut stages, comm_module, rows, COMBINE, MOE_TOTAL, 1)?;
                self.push_row(
                    &mut stages,
                    row_module,
                    "chop_row_rms_mimo_hf4096_residual_v1",
                    rows,
                    &[
                        ("x", MOE_TOTAL),
                        ("weight", W_POST),
                        ("residual", RESID),
                        ("residual_out", FINAL),
                        ("y", NORM_SCRATCH),
                        ("live_rows", LIVE),
                    ],
                    &[("eps_bits", 1e-6_f32.to_bits())],
                    &[],
                )?;
            } else {
                self.push_fused_comm(
                    &mut stages,
                    comm_module,
                    rows,
                    if fused_combine { DOWNOUT } else { COMBINE },
                    NORM_SCRATCH,
                    RESID,
                    FINAL,
                    W_POST,
                    1,
                    fused_combine,
                )?;
            }
            Ok(stages)
        }

        pub fn new(
            session: &Session,
            images: &Images,
            rank: usize,
            layer: usize,
            batch: usize,
            with_prefill: bool,
            prefill_rows: usize,
        ) -> LabResult<Self> {
            Self::build(
                session,
                images,
                rank,
                layer,
                batch,
                with_prefill,
                prefill_rows,
                false,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn build(
            session: &Session,
            images: &Images,
            rank: usize,
            layer: usize,
            batch: usize,
            with_prefill: bool,
            prefill_rows: usize,
            bundle_mode: bool,
        ) -> LabResult<Self> {
            if layer >= LAYERS || rank >= 4 || ![1, 8, 16, 32].contains(&batch) {
                return Err("unsupported MiMo layer fixture shape".into());
            }
            let global = GLOBAL_LAYERS.contains(&layer);
            let mut this = Self {
                buffers: BTreeMap::new(),
                prefill: Vec::new(),
                decode: Vec::new(),
                _modules: Vec::new(),
                layer,
                rank,
                batch,
                rows_capacity: if with_prefill { prefill_rows } else { batch },
                global,
                bundle_mode,
                last_prefill: std::cell::Cell::new(None),
            };
            let prefix = format!("model.layers.{layer}");
            let moe_names = [
                (W_IN, format!("{prefix}.input_layernorm.weight")),
                (W_POST, format!("{prefix}.post_attention_layernorm.weight")),
                (W_QKV, format!("{prefix}.self_attn.qkv_proj.weight")),
                (
                    W_QKVS,
                    format!("{prefix}.self_attn.qkv_proj.weight_scale_inv"),
                ),
                (W_O, format!("{prefix}.self_attn.o_proj.weight")),
                (W_ROUTER, format!("{prefix}.mlp.gate.weight")),
                (W_BIAS, format!("{prefix}.mlp.gate.e_score_correction_bias")),
                (W_GATE, format!("{prefix}.mlp.experts.gate_up_proj.weight")),
                (
                    W_GATES,
                    format!("{prefix}.mlp.experts.gate_up_proj.weight_scale"),
                ),
                (W_DOWN, format!("{prefix}.mlp.experts.down_proj.weight")),
                (
                    W_DOWNS,
                    format!("{prefix}.mlp.experts.down_proj.weight_scale"),
                ),
                (W_SINK, format!("{prefix}.self_attn.attention_sink_bias")),
            ];
            let dense_names = [
                (W_GATE, format!("{prefix}.mlp.gate_up_proj.weight")),
                (
                    W_GATES,
                    format!("{prefix}.mlp.gate_up_proj.weight_scale_inv"),
                ),
                (W_DOWN, format!("{prefix}.mlp.down_proj.weight")),
                (W_DOWNS, format!("{prefix}.mlp.down_proj.weight_scale_inv")),
            ];
            let names = if layer == 0 {
                moe_names
                    .into_iter()
                    .filter(|(id, _)| {
                        ![W_ROUTER, W_BIAS, W_GATE, W_GATES, W_DOWN, W_DOWNS].contains(id)
                    })
                    .chain(dense_names)
                    .collect::<Vec<_>>()
            } else {
                moe_names.to_vec()
            };
            for (id, name) in names {
                if global && id == W_SINK {
                    continue;
                }
                // The planned views are already fragment-major for dense and
                // expert weights.
                let bytes = images.read_named(rank, &name)?;
                this.upload(session, id, &bytes)?;
            }
            let max_rows = this.rows_capacity;
            let kv_heads = if global { 1 } else { 2 };
            let qkv_width = if global { 3392 } else { 3712 };
            let gemm_capacity = if with_prefill { 8192 } else { batch };
            for (id, bytes) in [
                (X, max_rows * H * 2),
                (NORM, gemm_capacity * H * 2),
                (QKV, max_rows * qkv_width * 2),
                (Q, max_rows * 16 * 192 * 2),
                (KNEW, max_rows * kv_heads * 192 * 2),
                (VNEW, max_rows * kv_heads * 128 * 2),
                (
                    KPOOL,
                    batch * max_rows.div_ceil(64).max(2) * 64 * kv_heads * 192 * 2,
                ),
                (
                    VPOOL,
                    batch * max_rows.div_ceil(64).max(2) * 64 * kv_heads * 128 * 2,
                ),
                (ATTN, gemm_capacity * 16 * 128 * 2),
                (O_PART, max_rows * H * 2),
                (O_TOTAL, max_rows * H * 2),
                (RESID, max_rows * H * 2),
                (POSTNORM, gemm_capacity * H * 2),
                (LOGITS, max_rows * 256 * 4),
                (IDS, max_rows * 8 * 4),
                (TOPK, max_rows * 8 * 4),
                (COUNTS, 256 * 4),
                (OFFSETS, 257 * 4),
                (GATEOUT, max_rows * 8 * 512 * 2),
                (DOWNOUT, max_rows * 8 * H * 2),
                (COMBINE, max_rows * H * 2),
                (MOE_TOTAL, max_rows * H * 2),
                (FINAL, max_rows * H * 2),
                (NORM_SCRATCH, max_rows * H * 2),
                (DENSE, if layer == 0 { gemm_capacity * H * 2 } else { 0 }),
            ] {
                this.alloc(session, id, bytes)?;
            }
            let (moe_phase, moe_tokens) = super::moe_entry(with_prefill, batch as u32);
            let gate = moe_variant(moe_phase, moe::Projection::GateUp, moe_tokens)?;
            let padded = match gate
                .query
                .grouped
                .ok_or("missing grouped capacity")?
                .capacity
            {
                RoutedCapacity::TotalRows(rows) => rows as usize,
                _ => return Err("MiMo MoE requires total routed capacity".into()),
            };
            this.alloc(session, SORTED, padded * 4)?;
            this.alloc(session, PADDED, 4)?;
            if with_prefill && prefill_rows >= 2048
                || super::moe_entry(false, batch as u32).0 == moe::Phase::Prefill
            {
                this.alloc(session, TILE_COUNTS, 256 * 256 * 4)?;
                this.alloc(session, TILE_OFFSETS, 256 * 256 * 4)?;
            }
            // The decode route writes the active expert count to word 0 and
            // up to 256 expert IDs after it (route.cuh). Prefill reads word
            // 0 as its limit: a prefill can select any of the 256 experts.
            let mut active = vec![0_i32; 257];
            active[0] = 256;
            this.upload(session, ACTIVE, &bytes_i32(&active))?;
            this.upload(session, DUMMY, &[0])?;
            this.upload(session, ONE, &1.0_f32.to_ne_bytes())?;
            let k_bytes = this.buffer(KPOOL).byte_len();
            let v_bytes = this.buffer(VPOOL).byte_len();
            this.upload(session, KPOOL, &vec![0; k_bytes])?;
            this.upload(session, VPOOL, &vec![0; v_bytes])?;
            let max_pages = if global { 1_048_576 / 64 } else { 1 };
            let mut table = vec![-1_i32; batch * max_pages];
            if global {
                for b in 0..batch {
                    for page in 0..max_rows.div_ceil(64).max(2) {
                        table[b * max_pages + page] =
                            (b * max_rows.div_ceil(64).max(2) + page) as i32;
                    }
                }
            }
            this.upload(session, TABLE, &bytes_i32(&table))?;
            this.upload(
                session,
                SLOT,
                &bytes_i32(&(0..batch as i32).collect::<Vec<_>>()),
            )?;
            this.alloc(session, LIVE, 4)?;
            this.alloc(session, POS, max_rows * 4)?;
            this.alloc(session, TOKEN_BATCH, max_rows * 4)?;
            this.alloc(session, LAST_POS, batch * 8)?;
            this.alloc(session, SEQ, batch * 4)?;
            this.alloc(session, START, batch * 4)?;
            this.alloc(session, LEN, batch * 4)?;
            this.alloc(session, PREFIX, batch * 4)?;
            this.upload(session, LIVE_BATCH, &(batch as i32).to_ne_bytes())?;
            let push = comm::push_buffers(4, batch as u32).map_err(|e| e.to_string())?;
            for index in 0..2 {
                let elements = max_rows * H;
                let tiles = elements.div_ceil(4096);
                let phased = kernels_sm120::comm::phased_buffers(4, elements as u64, DType::Bf16)
                    .map_err(|e| e.to_string())?;
                let inbox_bytes = (4 * elements * 2)
                    .max(push.inbox_bytes as usize)
                    .max(phased.inbox_bytes as usize);
                let fresh = this
                    .buffers
                    .get(&COMM_INBOX[index])
                    .is_none_or(|buffer| buffer.byte_len() < inbox_bytes);
                this.alloc(session, COMM_INBOX[index], inbox_bytes)?;
                // The low-latency region must start zeroed.
                if fresh {
                    this.buffer(COMM_INBOX[index])
                        .zero_at_async(0, push.zeroed_inbox_bytes as usize, &session.stream)
                        .map_err(|e| e.to_string())?;
                }
                this.zero(
                    session,
                    COMM_FLAGS[index],
                    (4 * 4 * tiles * 4).max(push.flag_bytes as usize),
                )?;
                this.zero(
                    session,
                    COMM_EPOCH[index],
                    (tiles * 4).max(push.epoch_bytes as usize),
                )?;
            }
            let row_module = session
                .context
                .load_module(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/row_engine.cubin"
                )))
                .map_err(|e| e.to_string())?;
            let gemm_module = session.module("gemm_mimo_sm120")?;
            let attention_module = session
                .context
                .load_module(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/attention_mimo_sm120.cubin"
                )))
                .map_err(|e| e.to_string())?;
            let comm_module = session.module("comm_sm120")?;
            let route_module = session
                .context
                .load_module(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/chop_moe_mimo_decode_route_b16.cubin"
                )))
                .map_err(|e| e.to_string())?;
            let mut modules = vec![
                row_module,
                gemm_module,
                attention_module,
                comm_module,
                route_module,
            ];
            let phases = if with_prefill {
                vec![AttentionPhase::Prefill, AttentionPhase::Decode]
            } else {
                vec![AttentionPhase::Decode]
            };
            for phase in phases {
                let prefill = phase == AttentionPhase::Prefill;
                let (moe_phase, tokens) = super::moe_entry(prefill, batch as u32);
                let gate = moe_variant(moe_phase, moe::Projection::GateUp, tokens)?;
                let down = moe_variant(moe_phase, moe::Projection::Down, tokens)?;
                let load_moe = |variant: &moe::Variant| -> LabResult<Module> {
                    let spec = moe::cubin_specs(
                        kernel_lab::catalogs(),
                        Some(model_mimo26_contract::sm120::config::PROFILE),
                    )
                    .into_iter()
                    .find(|spec| {
                        spec.entries
                            .iter()
                            .any(|entry| entry.symbol == variant.symbol)
                    })
                    .ok_or_else(|| format!("missing MoE cubin {}", variant.symbol))?;
                    session
                        .context
                        .load_module(kernel_lab::cubin(&spec.name)?)
                        .map_err(|e| e.to_string())
                };
                let gate_module = load_moe(&gate)?;
                let down_module = load_moe(&down)?;
                let built = this.build_phase(
                    session,
                    phase,
                    &modules[0],
                    &modules[1],
                    &modules[2],
                    &modules[3],
                    &modules[4],
                    &gate_module,
                    &down_module,
                )?;
                if prefill {
                    this.prefill = built;
                } else {
                    this.decode = built;
                }
                modules.push(gate_module);
                modules.push(down_module);
            }
            this._modules = modules;
            Ok(this)
        }

        /// Load `input` rows (at most the phase capacity) at `position`. A
        /// prefill chunk starts a sequence at `position`, so its earlier
        /// context is empty unless `position` is nonzero in decode.
        pub fn set_inputs(
            &self,
            session: &Session,
            input: &[u16],
            position: usize,
            prefill: bool,
        ) -> LabResult<()> {
            let capacity = if prefill {
                self.rows_capacity
            } else {
                self.batch
            };
            let rows = input.len() / H;
            if input.len() % H != 0
                || rows == 0
                || rows > capacity
                || (!prefill && rows != self.batch)
            {
                return Err("MiMo input shape mismatch".into());
            }
            let copy = |id: u64, bytes: &[u8]| {
                self.buffer(id)
                    .copy_from_bytes(bytes, &session.stream)
                    .map_err(|e| e.to_string())
            };
            let mut padded = input.to_vec();
            padded.resize(capacity * H, 0);
            copy(X, &bytes_u16(&padded))?;
            copy(LIVE, &(rows as i32).to_ne_bytes())?;
            let positions = if prefill {
                (0..capacity as i32)
                    .map(|row| position as i32 + row)
                    .collect::<Vec<_>>()
            } else {
                vec![position as i32; rows]
            };
            copy(POS, &bytes_i32(&positions))?;
            let batches = if prefill {
                vec![0; capacity]
            } else {
                (0..self.batch as i32).collect()
            };
            copy(TOKEN_BATCH, &bytes_i32(&batches))?;
            let last_position = position + if prefill { rows - 1 } else { 0 };
            copy(
                LAST_POS,
                &bytes_i64(&vec![last_position as i64; self.batch]),
            )?;
            copy(
                SEQ,
                &bytes_i32(&vec![
                    (position + if prefill { rows } else { 1 }) as i32;
                    self.batch
                ]),
            )?;
            copy(START, &bytes_i32(&vec![0; self.batch]))?;
            copy(
                LEN,
                &bytes_i32(&vec![if prefill { rows as i32 } else { 1 }; self.batch]),
            )?;
            copy(PREFIX, &bytes_i32(&vec![position as i32; self.batch]))?;
            if self
                .last_prefill
                .replace(Some(prefill))
                .is_some_and(|last| last != prefill)
            {
                // Prefill and decode use different communication tile counts,
                // so their flag indexes overlap with different peer slots.
                for id in COMM_FLAGS.into_iter().chain(COMM_EPOCH) {
                    copy(id, &vec![0; self.buffer(id).byte_len()])?;
                }
            }
            Ok(())
        }
    }

    // Shared with decode_chain and prefill_chain, which do not use the ends.
    #[allow(dead_code)]
    impl Rank {
        fn ends(rank: usize, rows_capacity: usize) -> Self {
            Self {
                buffers: BTreeMap::new(),
                prefill: Vec::new(),
                decode: Vec::new(),
                _modules: Vec::new(),
                layer: LAYERS,
                rank,
                batch: 1,
                rows_capacity,
                global: false,
                bundle_mode: true,
                last_prefill: std::cell::Cell::new(None),
            }
        }

        fn upload_view(
            &mut self,
            session: &Session,
            images: &Images,
            id: u64,
            name: &str,
        ) -> LabResult<()> {
            let bytes = images.read_named(self.rank, name)?;
            self.upload(session, id, &bytes)
        }

        /// Token embedding for up to `rows_capacity` prefill rows or one
        /// decode row: `TOKEN_IDS` in, `FINAL` out, as bundle-mimo26 runs it.
        pub fn embedding(
            session: &Session,
            images: &Images,
            rank: usize,
            rows_capacity: usize,
        ) -> LabResult<Self> {
            let mut this = Self::ends(rank, rows_capacity);
            this.upload_view(session, images, W_EMBED, "model.embed_tokens.weight")?;
            this.alloc(session, TOKEN_IDS, rows_capacity * 4)?;
            this.alloc(session, FINAL, rows_capacity * H * 2)?;
            this.alloc(session, LIVE, 4)?;
            let module = session
                .context
                .load_module(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/row_engine.cubin"
                )))
                .map_err(|e| e.to_string())?;
            for (rows, prefill) in [(rows_capacity, true), (1, false)] {
                let mut stages = Vec::new();
                this.push_row(
                    &mut stages,
                    &module,
                    "chop_row_embedding_full_v1",
                    rows as u32,
                    &[
                        ("table", W_EMBED),
                        ("ids", TOKEN_IDS),
                        ("output", FINAL),
                        ("live_rows", LIVE),
                    ],
                    &[
                        ("vocab_offset", 0),
                        ("shard_vocab", (4 * VOCAB_PER_RANK) as u32),
                    ],
                    &[],
                )?;
                if prefill {
                    this.prefill = stages;
                } else {
                    this.decode = stages;
                }
            }
            this._modules = vec![module];
            Ok(this)
        }

        /// Final RMSNorm and this rank's vocabulary slice: `X` in, `HEAD_OUT`
        /// out (F32 for prefill as for scored prompts, BF16 for decode).
        pub fn head(
            session: &Session,
            images: &Images,
            rank: usize,
            rows_capacity: usize,
        ) -> LabResult<Self> {
            let mut this = Self::ends(rank, rows_capacity);
            this.upload_view(session, images, W_POST, "model.norm.weight")?;
            this.upload_view(session, images, W_HEAD, "lm_head.weight")?;
            this.alloc(session, X, rows_capacity * H * 2)?;
            this.alloc(session, NORM, 8192 * H * 2)?;
            this.alloc(session, HEAD_OUT, 8192 * VOCAB_PER_RANK * 4)?;
            this.alloc(session, LIVE, 4)?;
            let row_module = session
                .context
                .load_module(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/row_engine.cubin"
                )))
                .map_err(|e| e.to_string())?;
            let gemm_module = session.module("gemm_mimo_sm120")?;
            for (rows, symbol) in [
                (rows_capacity, "chop_gemm_mimo_head_f32_prefill_v1"),
                (1, "chop_gemm_mimo_head_m1_v1"),
            ] {
                let mut stages = Vec::new();
                this.push_row(
                    &mut stages,
                    &row_module,
                    "chop_row_rms_mimo_hf4096_v1",
                    rows as u32,
                    &[
                        ("x", X),
                        ("weight", W_POST),
                        ("y", NORM),
                        ("live_rows", LIVE),
                    ],
                    &[("eps_bits", 1e-6_f32.to_bits())],
                    &[],
                )?;
                this.push_gemm(
                    session,
                    &mut stages,
                    &gemm_module,
                    symbol,
                    0,
                    rows as u32,
                    NORM,
                    W_HEAD,
                    None,
                    HEAD_OUT,
                )?;
                if rows == 1 {
                    this.decode = stages;
                } else {
                    this.prefill = stages;
                }
            }
            this._modules = vec![row_module, gemm_module];
            Ok(this)
        }

        /// Load token IDs (embedding) or hidden rows (head) and the live count.
        pub fn set_end_inputs(
            &self,
            session: &Session,
            ids: Option<&[u32]>,
            hidden: Option<&[u16]>,
        ) -> LabResult<()> {
            let rows = ids.map_or_else(|| hidden.map_or(0, |h| h.len() / H), <[u32]>::len);
            if rows == 0 || rows > self.rows_capacity {
                return Err("MiMo end fixture row count".into());
            }
            let copy = |id: u64, bytes: &[u8]| {
                self.buffer(id)
                    .copy_from_bytes(bytes, &session.stream)
                    .map_err(|e| e.to_string())
            };
            if let Some(ids) = ids {
                copy(
                    TOKEN_IDS,
                    &bytes_i32(&ids.iter().map(|&id| id as i32).collect::<Vec<_>>()),
                )?;
            }
            if let Some(hidden) = hidden {
                copy(X, &bytes_u16(hidden))?;
            }
            copy(LIVE, &(rows as i32).to_ne_bytes())
        }

        /// This rank's vocabulary slice of the head output as F32.
        pub fn head_logits(
            &self,
            session: &Session,
            rows: usize,
            prefill: bool,
        ) -> LabResult<Vec<f32>> {
            if prefill {
                session.download_f32(self.buffer(HEAD_OUT), rows * VOCAB_PER_RANK)
            } else {
                Ok(session
                    .download_bf16(self.buffer(HEAD_OUT), rows * VOCAB_PER_RANK)?
                    .into_iter()
                    .map(bf16_to_f32)
                    .collect())
            }
        }
    }

    // Decode poisoning and checks, used by decode_chain; this module is
    // shared with tests that do not call them.
    #[allow(dead_code)]
    impl Rank {
        /// `count` i32 words of buffer `id` from word `first`.
        fn words(
            &self,
            session: &Session,
            id: u64,
            first: usize,
            count: usize,
        ) -> LabResult<Vec<i32>> {
            if count == 0 {
                return Ok(Vec::new());
            }
            let mut bytes = vec![0_u8; count * 4];
            self.buffer(id)
                .copy_to_bytes_at_async(first * 4, &mut bytes, &session.stream)
                .map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            Ok(bytes
                .chunks_exact(4)
                .map(|word| i32::from_ne_bytes(word.try_into().expect("four bytes")))
                .collect())
        }

        /// Byte ranges of `KPOOL` and `VPOOL` the decode RoPE stage writes:
        /// each live token's K and V rows of every KV head at the cache slot
        /// rope_kv.cuh addresses from the token's position, through the ring
        /// (sliding window: slot `SLOT[b] * RING_SIZE + position %
        /// RING_SIZE`) or the page table (global: token `position % page` of
        /// page `TABLE[b][position / page]`, pages of tight stride, as
        /// `build_phase` passes them). Earlier positions are context the
        /// pass only reads.
        fn decode_kv_rows(&self, session: &Session) -> LabResult<[Vec<Range<usize>>; 2]> {
            let (kv_heads, page_size) = kv_layout(self.global);
            let (kv_heads, page_size) = (kv_heads as usize, page_size as usize);
            let (k_row, v_row) = (kv_heads * 192 * 2, kv_heads * 128 * 2);
            let index = |value: i32, what: &str| {
                usize::try_from(value).map_err(|_| format!("negative {what} {value}"))
            };
            let live = index(self.words(session, LIVE, 0, 1)?[0], "live count")?;
            let positions = self.words(session, POS, 0, live)?;
            let batches = self.words(session, TOKEN_BATCH, 0, live)?;
            let max_pages = 1_048_576 / page_size;
            let mut rows = [Vec::new(), Vec::new()];
            for (&position, &batch) in positions.iter().zip(&batches) {
                let position = index(position, "position")?;
                let batch = index(batch, "batch row")?;
                let slot = if self.global {
                    let page =
                        self.words(session, TABLE, batch * max_pages + position / page_size, 1)?[0];
                    index(page, "page")? * page_size + position % page_size
                } else {
                    let ring = RING_SIZE as usize;
                    index(self.words(session, SLOT, batch, 1)?[0], "ring slot")? * ring
                        + position % ring
                };
                rows[0].push(slot * k_row..(slot + 1) * k_row);
                rows[1].push(slot * v_row..(slot + 1) * v_row);
            }
            Ok(rows)
        }

        /// Poisons what this rank's decode pass writes before reading it:
        /// the stage outputs and intermediates, the new token's K and V rows,
        /// GEMM split partials and the MoE partials a later read depends on.
        /// It never touches the input `X`, weights, GEMM semaphores (self
        /// resetting), the stream-decode flag prefix or the comm state.
        fn poison_decode(&self, session: &Session) -> LabResult<()> {
            for (name, id) in DECODE_OUTPUTS {
                let buffer = self.buffers.get(&id).ok_or_else(|| {
                    format!(
                        "layer {} rank {} has no {name} buffer",
                        self.layer, self.rank
                    )
                })?;
                poison::poison(session, &[buffer])?;
            }
            let [k_rows, v_rows] = self.decode_kv_rows(session)?;
            poison::fill(session, self.buffer(KPOOL), &k_rows, &[POISON])?;
            poison::fill(session, self.buffer(VPOOL), &v_rows, &[POISON])?;
            // Every CTA of the fragment GEMMs writes its partials before the
            // cluster or the last CTA reads them.
            for id in GEMM_PART {
                if let Some(buffer) = self.buffers.get(&id) {
                    poison::poison(session, &[buffer])?;
                }
            }
            if self.layer == 0 {
                return Ok(());
            }
            let batch = self.batch as u32;
            for (index, projection) in [(0, moe::Projection::GateUp), (1, moe::Projection::Down)] {
                let (phase, tokens) = super::moe_entry(false, batch);
                let variant = moe_variant(phase, projection, tokens)?;
                let counts = RuntimeCounts {
                    m: batch,
                    batch,
                    context: 68,
                    routed_rows: batch * 8,
                    split_k: variant.knobs.split_k_capacity,
                };
                let buffer = self.buffer(MOE_PART[index]);
                let bytes = buffer.byte_len();
                let partials = if variant
                    .reduction_plan(counts)
                    .map_err(|e| e.to_string())?
                    .is_some()
                {
                    // The reduction kernel reads every split partial.
                    0..bytes
                } else if variant.fragment_decode() || variant.fragment_prefill() {
                    // Fragment decode and prefill take no partial workspace.
                    0..0
                } else {
                    return Err(format!(
                        "unexpected MiMo MoE decode variant {}",
                        variant.symbol
                    ));
                };
                poison::fill(session, buffer, &[partials], &[POISON])?;
            }
            Ok(())
        }
    }

    /// Poisons what the decode pass of every rank writes before reading it
    /// (`Rank::poison_decode`), once every rank is idle.
    #[allow(dead_code)] // Used by decode_chain; this module is shared.
    pub fn poison_decode(sessions: &[Session], ranks: &[Rank]) -> LabResult<()> {
        synchronize(sessions)?;
        for (session, rank) in sessions.iter().zip(ranks) {
            rank.poison_decode(session)?;
        }
        Ok(())
    }

    /// What `poison_decode` fills, for receipts.
    #[allow(dead_code)] // Used by decode_chain; this module is shared.
    pub fn decode_poisoned() -> Vec<String> {
        DECODE_OUTPUTS
            .iter()
            .map(|(name, _)| (*name).to_owned())
            .chain(
                [
                    "k_pool_new_token_rows",
                    "v_pool_new_token_rows",
                    "gemm_partials",
                    "moe_partials",
                ]
                .map(str::to_owned),
            )
            .collect()
    }

    /// FINAL after a decode pass on poisoned buffers. The second fused
    /// all-reduce writes every batch row of its residual output (comm README
    /// "Fused decode all-reduce"), so no element of them may keep its
    /// poison, and no byte past them may change.
    #[allow(dead_code)] // Used by decode_chain; this module is shared.
    pub fn decode_final(session: &Session, rank: &Rank, label: &str) -> LabResult<Vec<u16>> {
        let bytes = poison::read(session, rank.buffer(FINAL))?;
        poison::check(
            &format!("{label} final"),
            &bytes,
            &WriteMap::of(DType::Bf16).prefix(rank.batch * H)?,
            Fill::Pattern(&[POISON]),
        )?;
        Ok(bytes[..rank.batch * H * 2]
            .chunks_exact(2)
            .map(|pair| u16::from_ne_bytes([pair[0], pair[1]]))
            .collect())
    }

    /// The route's active-expert list after a decode pass on poisoned
    /// buffers, checked against the decode route's contract (route.cuh
    /// `route_decode_mimo_body`): word 0 holds the number of distinct valid
    /// experts (0 to 255) among the live rows' top-8 route IDs, words 1 to
    /// that count hold them in ascending order, and every later word keeps
    /// its poison.
    #[allow(dead_code)] // Used by decode_chain; this module is shared.
    pub fn decode_active(session: &Session, rank: &Rank, label: &str) -> LabResult<Vec<i32>> {
        let label = format!("{label} active experts");
        let live = usize::try_from(rank.words(session, LIVE, 0, 1)?[0])
            .map_err(|_| format!("{label}: negative live count"))?;
        let experts = rank
            .words(session, IDS, 0, live * 8)?
            .into_iter()
            .filter(|id| (0..256).contains(id))
            .collect::<BTreeSet<_>>();
        let expected = std::iter::once(experts.len() as i32)
            .chain(experts)
            .collect::<Vec<_>>();
        let bytes = poison::read(session, rank.buffer(ACTIVE))?;
        poison::check(
            &label,
            &bytes,
            &WriteMap::of(DType::F32).prefix(expected.len())?,
            Fill::Pattern(&[POISON]),
        )?;
        let words = bytes
            .chunks_exact(4)
            .map(|word| i32::from_ne_bytes(word.try_into().expect("four bytes")))
            .collect::<Vec<_>>();
        if words[..expected.len()] != expected[..] {
            return Err(format!(
                "{label}: {:?}, the route contract gives {expected:?}",
                &words[..expected.len()]
            ));
        }
        Ok(words)
    }

    pub fn ranks(
        sessions: &[Session],
        images: &Images,
        layer: usize,
        batch: usize,
        with_prefill: bool,
    ) -> LabResult<Vec<Rank>> {
        if sessions.len() != 4 {
            return Err("MiMo layer runner requires four sessions".into());
        }
        sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| Rank::new(session, images, rank, layer, batch, with_prefill, 64))
            .collect()
    }

    // This shared module is also compiled by decode_chain, which does not call this helper.
    #[allow(dead_code)]
    pub fn ranks_with_prefill_rows(
        sessions: &[Session],
        images: &Images,
        layer: usize,
        rows: usize,
    ) -> LabResult<Vec<Rank>> {
        if ![2048, 8192].contains(&rows) || sessions.len() != 4 {
            return Err("unsupported MiMo prefill profile shape".into());
        }
        sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| Rank::new(session, images, rank, layer, 1, true, rows))
            .collect()
    }

    pub fn stage_count(ranks: &[Rank], prefill: bool) -> LabResult<usize> {
        if ranks.len() != 4 {
            return Err("MiMo layer runner requires four ranks".into());
        }
        let count = if prefill {
            ranks[0].prefill.len()
        } else {
            ranks[0].decode.len()
        };
        if ranks.iter().all(|rank| {
            (if prefill {
                rank.prefill.len()
            } else {
                rank.decode.len()
            }) == count
        }) {
            Ok(count)
        } else {
            Err("rank launch sequences differ".into())
        }
    }

    pub fn launch_one(
        sessions: &[Session],
        ranks: &[Rank],
        prefill: bool,
        stage_index: usize,
        rank_index: usize,
        pdl: bool,
    ) -> LabResult<()> {
        {
            let session = &sessions[rank_index];
            let rank = &ranks[rank_index];
            let stages = if prefill { &rank.prefill } else { &rank.decode };
            let stage = &stages[stage_index];
            let mut buffers = rank
                .buffers
                .iter()
                .map(|(&id, buffer)| (id, buffer))
                .collect::<BTreeMap<_, _>>();
            for (peer, peer_rank) in ranks.iter().enumerate() {
                for index in 0..2 {
                    let base = if index == 0 { 1000 } else { 2000 };
                    // Embedding and head fixtures have no hidden reductions.
                    let (Some(inbox), Some(flags), Some(target)) = (
                        peer_rank.buffers.get(&COMM_INBOX[index]),
                        peer_rank.buffers.get(&COMM_FLAGS[index]),
                        peer_rank.buffers.get(&peer_rank.gather_target(index)),
                    ) else {
                        continue;
                    };
                    buffers.insert(base + peer as u64, inbox);
                    buffers.insert(base + 100 + peer as u64, flags);
                    buffers.insert(base + 200 + peer as u64, target);
                }
            }
            let mut plan = stage.plan;
            plan.pdl = pdl;
            launch::launch(
                &session.context,
                &stage.function,
                &session.stream,
                plan,
                &stage.args,
                &Bindings { buffers },
            )
            .map_err(|e| format!("layer {} rank {rank_index} {}: {e}", rank.layer, stage.name))?;
        }
        Ok(())
    }

    pub fn launch_stage(
        sessions: &[Session],
        ranks: &[Rank],
        prefill: bool,
        stage_index: usize,
        pdl: bool,
    ) -> LabResult<()> {
        for rank in 0..ranks.len() {
            launch_one(sessions, ranks, prefill, stage_index, rank, pdl)?;
        }
        Ok(())
    }

    /// Frozen captured tensors for a single-rank prefill control. The real
    /// attention prefix still runs, but its local partial cannot replace the
    /// captured TP4 residual. Down uses fixed weights because audit dumps
    /// contain expert IDs, not router weights.
    #[allow(dead_code)]
    pub struct PrefillReplayBindings {
        residual: DeviceBuffer,
        zero: DeviceBuffer,
        ids: DeviceBuffer,
        weights: DeviceBuffer,
        gate: Option<DeviceBuffer>,
        scales: Option<DeviceBuffer>,
    }

    #[allow(dead_code)]
    impl PrefillReplayBindings {
        pub fn new(
            session: &Session,
            rank: &Rank,
            residual: &[u16],
            ids: &[i32],
            relocate_weights: bool,
        ) -> LabResult<Self> {
            let rows = residual.len() / H;
            if residual.len() != rows * H
                || rows == 0
                || rows > rank.rows_capacity
                || ids.len() != rows * 8
                || ids.iter().any(|&id| !(0..256).contains(&id))
            {
                return Err("captured prefill binding shape or expert ID mismatch".into());
            }
            let upload_ids = session.allocate(ids.len() * 4)?;
            upload_ids
                .copy_from_bytes(&bytes_i32(ids), &session.stream)
                .map_err(|e| e.to_string())?;
            let duplicate = |id| -> LabResult<DeviceBuffer> {
                let source = rank.buffer(id);
                let buffer = session.allocate(source.byte_len())?;
                buffer
                    .copy_range_from_async(0, source, 0, source.byte_len(), &session.stream)
                    .map_err(|e| e.to_string())?;
                Ok(buffer)
            };
            Ok(Self {
                residual: session.upload_bf16(residual)?,
                zero: session.upload_bf16(&vec![0; residual.len()])?,
                ids: upload_ids,
                weights: session.upload_f32(&vec![0.125; ids.len()])?,
                gate: relocate_weights.then(|| duplicate(W_GATE)).transpose()?,
                scales: relocate_weights.then(|| duplicate(W_GATES)).transpose()?,
            })
        }

        pub fn weight_addresses(&self, rank: &Rank) -> LabResult<[u64; 2]> {
            let address = |buffer: &DeviceBuffer| {
                buffer
                    .device_address_at(0, buffer.byte_len())
                    .map_err(|e| e.to_string())
            };
            Ok([
                address(self.gate.as_ref().unwrap_or_else(|| rank.buffer(W_GATE)))?,
                address(self.scales.as_ref().unwrap_or_else(|| rank.buffer(W_GATES)))?,
            ])
        }
    }
    /// One-rank prefill diagnostic. Replace each collective's result with
    /// this rank's partial, then run the existing separate residual/norm row.
    /// This preserves the local compute sequence, not TP4 numerics or traffic.
    #[allow(dead_code)] // Used by prefill_chain; this module is also shared with decode tests.
    pub fn launch_local_prefill(
        session: &Session,
        rank: &Rank,
        stage_index: usize,
        live_rows: usize,
        ctas: u32,
        pdl: bool,
    ) -> LabResult<()> {
        launch_local_prefill_replay(session, rank, stage_index, live_rows, ctas, pdl, None)
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn launch_local_prefill_replay(
        session: &Session,
        rank: &Rank,
        stage_index: usize,
        live_rows: usize,
        ctas: u32,
        pdl: bool,
        replay: Option<&PrefillReplayBindings>,
    ) -> LabResult<()> {
        let stage = &rank.prefill[stage_index];
        if rank.bundle_mode || stage.name.starts_with("chop_comm_") || !(1..=188).contains(&ctas) {
            return Err(
                "local prefill requires unfused norms, a local stage and a valid grid".into(),
            );
        }
        let mut buffers = rank
            .buffers
            .iter()
            .map(|(&id, buffer)| (id, buffer))
            .collect::<BTreeMap<_, _>>();
        buffers.insert(O_TOTAL, rank.buffer(O_PART));
        buffers.insert(MOE_TOTAL, rank.buffer(COMBINE));
        if let Some(replay) = replay {
            if stage.name == "chop_row_rms_mimo_hf4096_residual_v1" {
                buffers.insert(O_TOTAL, &replay.residual);
                buffers.insert(X, &replay.zero);
            }
            // Keep router GEMM/top-k as preceding kernels; only consumers use
            // the captured IDs. Never let the router overwrite frozen inputs.
            if stage.name.starts_with("chop_row_permute_")
                || stage.name.starts_with("chop_moe_")
                || stage.name.starts_with("chop_row_combine_")
            {
                buffers.insert(IDS, &replay.ids);
                buffers.insert(TOPK, &replay.weights);
            }
            if stage.name.starts_with("chop_moe_mimo_gateup_prefill_") {
                if let Some(gate) = &replay.gate {
                    buffers.insert(W_GATE, gate);
                }
                if let Some(scales) = &replay.scales {
                    buffers.insert(W_GATES, scales);
                }
            }
        }
        if live_rows == 0 || live_rows > rank.rows_capacity {
            return Err("local prefill live rows exceed capacity".into());
        }
        let mut args = stage.args.clone();
        if stage.name.starts_with("chop_gemm_") {
            // gemm::params and gemm::arguments place live_m after the two
            // input maps/pointers and three output/workspace pointers.
            let live_m = args.0.get_mut(5).ok_or("missing GEMM live_m")?;
            if *live_m != Argument::U32(rank.rows_capacity as u32) {
                return Err("unexpected local prefill GEMM live_m ABI".into());
            }
            *live_m = Argument::U32(live_rows as u32);
        }
        let mut plan = stage.plan;
        plan.pdl = pdl;
        if plan.grid == [188, 1, 1]
            && (stage.name.starts_with("chop_gemm_") || stage.name.starts_with("chop_moe_"))
        {
            plan.grid[0] = ctas;
        }
        launch::launch(
            &session.context,
            &stage.function,
            &session.stream,
            plan,
            &args,
            &Bindings { buffers },
        )
    }

    pub fn launch_all(
        sessions: &[Session],
        ranks: &[Rank],
        prefill: bool,
        pdl: bool,
    ) -> LabResult<()> {
        for stage in 0..stage_count(ranks, prefill)? {
            launch_stage(sessions, ranks, prefill, stage, pdl)?;
        }
        Ok(())
    }

    pub fn synchronize(sessions: &[Session]) -> LabResult<()> {
        for session in sessions {
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn output(session: &Session, rank: &Rank, rows: usize) -> LabResult<Vec<u16>> {
        session.download_bf16(rank.buffer(FINAL), rows * H)
    }

    pub fn router_choices(
        session: &Session,
        rank: &Rank,
        rows: usize,
    ) -> LabResult<Vec<Vec<usize>>> {
        let mut bytes = vec![0; rows * 8 * 4];
        rank.buffer(IDS)
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        bytes
            .chunks_exact(8 * 4)
            .map(|row| {
                row.chunks_exact(4)
                    .map(|word| {
                        let id = i32::from_ne_bytes(word.try_into().expect("four bytes"));
                        usize::try_from(id).map_err(|_| format!("negative router expert ID {id}"))
                    })
                    .collect()
            })
            .collect()
    }

    pub fn diagnostic_qkv_operands(
        session: &Session,
        rank: &Rank,
    ) -> LabResult<(Vec<u8>, Vec<u8>)> {
        use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
        let rows = if rank.global { 3392 } else { 3712 };
        let layout = FragmentWeightLayout {
            rows,
            k: H,
            format: FragmentWeightFormat::Fp8E4M3,
        };
        let mut weight = vec![0; layout.packed_bytes().map_err(|e| e.to_string())?];
        let mut scales = vec![0; rows.div_ceil(128) * (H / 128) * 4];
        rank.buffer(W_QKV)
            .copy_to_bytes(&mut weight, &session.stream)
            .map_err(|e| e.to_string())?;
        rank.buffer(W_QKVS)
            .copy_to_bytes(&mut scales, &session.stream)
            .map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        Ok((weight, scales))
    }

    pub fn intermediate(
        session: &Session,
        rank: &Rank,
        name: &str,
        rows: usize,
    ) -> LabResult<Vec<u16>> {
        let handle = match name {
            "input_norm" => NORM,
            "qkv" => QKV,
            "rope_q" => Q,
            "rope_k" => KNEW,
            "scaled_v" => VNEW,
            "attention" => ATTN,
            "o_partial" => O_PART,
            "o_total" => O_TOTAL,
            "attention_residual" => RESID,
            "post_norm" => POSTNORM,
            "gate_up" => GATEOUT,
            "down" => DOWNOUT,
            "moe_combine" => COMBINE,
            "moe_total" => MOE_TOTAL,
            "final" => FINAL,
            _ => return Err(format!("unknown intermediate {name}")),
        };
        let width = match name {
            "qkv" => {
                if rank.global {
                    3392
                } else {
                    3712
                }
            }
            "rope_q" => 16 * 192,
            "rope_k" => {
                if rank.global {
                    192
                } else {
                    2 * 192
                }
            }
            "scaled_v" => {
                if rank.global {
                    128
                } else {
                    2 * 128
                }
            }
            "attention" => 2048,
            "gate_up" => 8 * 512,
            "down" => 8 * H,
            _ => H,
        };
        session.download_bf16(rank.buffer(handle), rows * width)
    }

    /// Save one real expert and one routed token for a standalone MoE debug case.
    pub fn save_moe_repro(
        session: &Session,
        rank: &Rank,
        images: &Images,
        trace: &model_mimo26::oracle::LayerTrace,
    ) -> LabResult<()> {
        let directory = std::env::temp_dir().join("moe-repro/layer1-rank0-token0");
        fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        if directory.join("metadata.json").exists() {
            return Ok(());
        }
        let expert = *trace
            .router_ids
            .as_ref()
            .ok_or("missing MoE route IDs")?
            .first()
            .ok_or("empty MoE routes")?;
        let actual = intermediate(session, rank, "gate_up", 1)?;
        let activation = intermediate(session, rank, "post_norm", 1)?;
        let image_index = &images.indexes[rank.rank];
        let mut weight_bytes = Vec::new();
        let mut scale_bytes = Vec::new();
        for (name, file_name, expected_bytes) in [
            ("gate_up_proj.weight", "gate_up_weights.mxfp4", 1024 * 2048),
            (
                "gate_up_proj.weight_scale",
                "gate_up_scales.e8m0",
                1024 * 128,
            ),
        ] {
            let view_name = format!("model.layers.1.mlp.experts.{name}");
            let view = image_index.view(&view_name)?;
            let expert_bytes = view.bytes / 256;
            if expert_bytes != expected_bytes {
                return Err(format!("unexpected expert view size for {view_name}"));
            }
            let bytes = images.read(
                rank.rank,
                view.offset + expert as u64 * expert_bytes,
                expert_bytes,
            )?;
            fs::write(directory.join(file_name), &bytes).map_err(|e| e.to_string())?;
            if name.ends_with("weight") {
                weight_bytes = bytes;
            } else {
                scale_bytes = bytes;
            }
        }
        // The image stores the scales in the grouped MMA's 16-row by
        // 8-block tiles.
        let tiles = ScaleTileLayout {
            rows: 1024,
            k_blocks: 128,
            tile_rows: 16,
            k_blocks_per_tile: 8,
        };
        let packed = Packed {
            bytes: scale_bytes,
            layout: Layout::ScaleTile(tiles),
        };
        let scales = Scales::Ue8M0(tiles.unpack(&packed).map_err(|e| e.0)?);
        let mxfp4 = QuantFormat {
            element: SmallFloat::E2M1,
            block: (1, 32),
            scale: ScaleFormat::Ue8M0,
            chunk_rows: None,
        };
        let weight = Quantized::from_parts(1024, 4096, mxfp4, weight_bytes, scales, 1.0)
            .map_err(|e| e.0)?
            .decoded();
        let round = |value: f64| f64::from(bf16_to_f32(f32_to_bf16(value as f32)));
        let dot = |row: usize| {
            activation
                .iter()
                .zip(&weight[row * 4096..(row + 1) * 4096])
                .map(|(&bits, &w)| f64::from(bf16_to_f32(bits)) * w)
                .sum::<f64>()
        };
        let gate = (0..512).map(|row| round(dot(row))).collect::<Vec<_>>();
        let up = (512..1024).map(|row| round(dot(row))).collect::<Vec<_>>();
        // HF materializes SiLU in BF16 before multiplying by up.
        let materialized = SiluGlu {
            silu_output: Some(DType::Bf16),
            ..SiluGlu::UNCLAMPED
        };
        let expected = materialized
            .apply(&gate, &up)
            .map_err(|e| e.0)?
            .iter()
            .map(|&v| f32_to_bf16(v as f32))
            .collect::<Vec<_>>();
        fs::write(directory.join("activation.bf16"), bytes_u16(&activation))
            .map_err(|e| e.to_string())?;
        fs::write(
            directory.join("expected_gate_up.bf16"),
            bytes_u16(&expected),
        )
        .map_err(|e| e.to_string())?;
        fs::write(
            directory.join("actual_gate_up.bf16"),
            bytes_u16(&actual[..512]),
        )
        .map_err(|e| e.to_string())?;
        let first_actual = actual[..16]
            .iter()
            .map(|&bits| bf16_to_f32(bits))
            .collect::<Vec<_>>();
        let first_expected = expected[..16]
            .iter()
            .map(|&bits| bf16_to_f32(bits))
            .collect::<Vec<_>>();
        let (dot, aa, bb, max_abs) = actual[..512].iter().zip(&expected).fold(
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64),
            |(dot, aa, bb, error), (&actual, &expected)| {
                let a = f64::from(bf16_to_f32(actual));
                let b = f64::from(bf16_to_f32(expected));
                (
                    dot + a * b,
                    aa + a * a,
                    bb + b * b,
                    error.max((a - b).abs()),
                )
            },
        );
        let metadata = serde_json::json!({
            "layer": 1,
            "rank": rank.rank,
            "token": 0,
            "route_slot": 0,
            "expert_id": expert,
            "activation_shape": [1, 4096],
            "gate_up_weight_shape": [1024, 4096],
            "gate_up_output_shape": [1, 512],
            "weight_layout": "mxfp4_gate_up_low_nibble_row_major_v1",
            "scale_layout": "mxfp4_gate_up_e8m0_mma_scale_16x8_v1",
            "first_16_actual": first_actual,
            "first_16_expected": first_expected,
            "cosine": dot / (aa.sqrt() * bb.sqrt()).max(1e-30),
            "max_abs_error": max_abs,
            "notes": "One real routed token. Weight and scale files contain one rank-local expert in gate rows then up rows. The expected BF16 output is computed by direct f64 MXFP4 matmul and HF rounding of the saved activation. All BF16 files are little endian."
        });
        fs::write(
            directory.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn localize(
        session: &Session,
        rank: &Rank,
        traces: &[model_mimo26::oracle::LayerTrace],
    ) -> LabResult<String> {
        let rows = traces.len();
        for name in [
            "input_norm",
            "qkv",
            "rope_q",
            "rope_k",
            "scaled_v",
            "attention",
            "attention_residual",
            "post_norm",
            "moe_total",
            "final",
        ] {
            let actual = intermediate(session, rank, name, rows)?;
            let width = actual.len() / rows;
            let reference = traces
                .iter()
                .flat_map(|trace| match name {
                    "input_norm" => trace.input_norm.clone(),
                    "qkv" => trace.qkv[rank.rank * width..(rank.rank + 1) * width].to_vec(),
                    "rope_q" => trace.q[rank.rank * width..(rank.rank + 1) * width].to_vec(),
                    "rope_k" => trace.k[rank.rank * width..(rank.rank + 1) * width].to_vec(),
                    "scaled_v" => trace.v[rank.rank * width..(rank.rank + 1) * width].to_vec(),
                    "attention" => {
                        trace.attention[rank.rank * width..(rank.rank + 1) * width].to_vec()
                    }
                    "attention_residual" => trace.attention_residual.clone(),
                    "post_norm" => trace.post_norm.clone(),
                    "moe_total" => trace.moe.clone(),
                    "final" => trace.final_output.clone(),
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            let mut worst_cosine = 1.0_f64;
            let mut max_abs = 0.0_f64;
            for (a, b) in actual
                .chunks_exact(width)
                .zip(reference.chunks_exact(width))
            {
                let mut dot = 0.0;
                let mut aa = 0.0;
                let mut bb = 0.0;
                for (&bits, &ref_value) in a.iter().zip(b) {
                    let value = f64::from(kernel_core::codec::bf16_to_f32(bits));
                    dot += value * ref_value;
                    aa += value * value;
                    bb += ref_value * ref_value;
                    max_abs = max_abs.max((value - ref_value).abs());
                }
                let cosine = dot / (aa.sqrt() * bb.sqrt()).max(1e-30);
                worst_cosine = worst_cosine.min(cosine);
            }
            if worst_cosine < 0.9999 {
                if name == "moe_total" {
                    let logits = session.download_f32(rank.buffer(LOGITS), rows * 256)?;
                    for (row, trace) in traces.iter().enumerate() {
                        if let Some(reference) = &trace.router_logits {
                            let slice = &logits[row * 256..(row + 1) * 256];
                            let worst = slice
                                .iter()
                                .zip(reference)
                                .map(|(&actual, &expected)| (f64::from(actual) - expected).abs())
                                .fold(0.0_f64, f64::max);
                            if worst > 0.01 {
                                return Ok(format!(
                                    "router logits diverge at token {row}: max abs {worst:.6}"
                                ));
                            }
                        }
                    }
                    let mut ids_bytes = vec![0; rows * 8 * 4];
                    rank.buffer(IDS)
                        .copy_to_bytes(&mut ids_bytes, &session.stream)
                        .map_err(|e| e.to_string())?;
                    let ids = ids_bytes
                        .chunks_exact(4)
                        .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("four bytes")))
                        .collect::<Vec<_>>();
                    let weights = session.download_f32(rank.buffer(TOPK), rows * 8)?;
                    for (row, trace) in traces.iter().enumerate() {
                        if let (Some(expected_ids), Some(expected_weights)) =
                            (&trace.router_ids, &trace.router_weights)
                        {
                            for slot in 0..8 {
                                let index = row * 8 + slot;
                                if ids[index] != expected_ids[slot] as i32 {
                                    return Ok(format!(
                                        "router top-k differs at token {row} slot {slot}: {} != {}",
                                        ids[index], expected_ids[slot]
                                    ));
                                }
                                if (f64::from(weights[index]) - expected_weights[slot]).abs() > 1e-3
                                {
                                    return Ok(format!(
                                        "router weight differs at token {row} slot {slot}: {} != {}",
                                        weights[index], expected_weights[slot]
                                    ));
                                }
                            }
                        }
                    }
                    let gate = intermediate(session, rank, "gate_up", rows)?;
                    let mut dot = 0.0_f64;
                    let mut aa = 0.0_f64;
                    let mut bb = 0.0_f64;
                    let mut gate_max_abs = 0.0_f64;
                    for (row, trace) in traces.iter().enumerate() {
                        if let Some(routes) = &trace.gate_up_routes {
                            for (slot, route) in routes.iter().enumerate() {
                                let actual =
                                    &gate[(row * 8 + slot) * 512..(row * 8 + slot + 1) * 512];
                                let expected = &route[rank.rank * 512..(rank.rank + 1) * 512];
                                for (&bits, &value) in actual.iter().zip(expected) {
                                    let sample = f64::from(kernel_core::codec::bf16_to_f32(bits));
                                    dot += sample * value;
                                    aa += sample * sample;
                                    bb += value * value;
                                    gate_max_abs = gate_max_abs.max((sample - value).abs());
                                }
                            }
                        }
                    }
                    let gate_cosine = dot / (aa.sqrt() * bb.sqrt()).max(1e-30);
                    if gate_cosine < 0.9999 {
                        let mut offset_bytes = vec![0_u8; 257 * 4];
                        rank.buffer(OFFSETS)
                            .copy_to_bytes(&mut offset_bytes, &session.stream)
                            .map_err(|e| e.to_string())?;
                        let offsets = offset_bytes
                            .chunks_exact(4)
                            .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("four bytes")))
                            .collect::<Vec<_>>();
                        let mut sorted_bytes = vec![0_u8; rank.buffer(SORTED).byte_len()];
                        rank.buffer(SORTED)
                            .copy_to_bytes(&mut sorted_bytes, &session.stream)
                            .map_err(|e| e.to_string())?;
                        let sorted = sorted_bytes
                            .chunks_exact(4)
                            .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("four bytes")))
                            .collect::<Vec<_>>();
                        let mut misplaced = Vec::new();
                        for expert in 0..256 {
                            for &route in
                                &sorted[offsets[expert] as usize..offsets[expert + 1] as usize]
                            {
                                if route >= 0 && ids[route as usize] != expert as i32 {
                                    misplaced.push((expert, route, ids[route as usize]));
                                }
                            }
                        }
                        let first = traces[0].gate_up_routes.as_ref().expect("MoE route trace");
                        let route_stats = first
                            .iter()
                            .enumerate()
                            .map(|(slot, route)| {
                                let actual = &gate[slot * 512..(slot + 1) * 512];
                                let expected = &route[rank.rank * 512..(rank.rank + 1) * 512];
                                let (dot, aa, bb) = actual.iter().zip(expected).fold(
                                    (0.0_f64, 0.0_f64, 0.0_f64),
                                    |(dot, aa, bb), (&bits, &b)| {
                                        let a = f64::from(kernel_core::codec::bf16_to_f32(bits));
                                        (dot + a * b, aa + a * a, bb + b * b)
                                    },
                                );
                                format!(
                                    "{slot}:cos={:.4},norm={:.4}/{:.4}",
                                    dot / (aa.sqrt() * bb.sqrt()).max(1e-30),
                                    aa.sqrt(),
                                    bb.sqrt()
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        return Ok(format!(
                            "MoE gate/up diverges after router: cosine {gate_cosine:.7}, max abs {gate_max_abs:.6}; misplaced routes {:?}; routes [{route_stats}]",
                            &misplaced[..misplaced.len().min(8)]
                        ));
                    }
                }
                return Ok(format!(
                    "first divergent boundary {name}: cosine {worst_cosine:.7}, max abs {max_abs:.6}{}",
                    if name == "moe_total" {
                        "; router passed"
                    } else {
                        ""
                    }
                ));
            }
        }
        Ok("all sampled boundaries pass cosine 0.9999; inspect intra-op rounding".into())
    }
}

/// Page stride of a tightly packed single-layer K or V block for `variant`.
fn tight_page_stride(variant: &attention::Variant, dim: u32) -> u64 {
    let bytes = if variant.query.kv_dtype == kernel_core::contract::DType::Bf16 {
        2
    } else {
        1
    };
    u64::from(variant.query.page_size) * u64::from(variant.query.kv_heads) * u64::from(dim) * bytes
}
