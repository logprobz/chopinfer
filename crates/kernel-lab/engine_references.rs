// The per-model grouped-MoE engine references, included by build.rs (which
// compiles them into kernel-lab) and by tests/moe_template.rs (which checks
// their specs on the CPU).

// The per-model grouped-MoE engines the template replaced, frozen in
// cuda/baselines as lab-only references: each served A16 expert entry gets a
// chop_moe_eng_ copy with the entry ABI, the engine plan's threads and the
// engine's body. tests/moe_template.rs compares the served template with it
// bit for bit on generated and checkpoint inputs.
pub fn moe_engine_reference_specs() -> Vec<kernel_build::CubinSpec> {
    moe_engine_reference_specs_for(&kernels_sm120::moe::template::lab_instances(catalogs()))
}

/// The references of `instances`' variants.
pub fn moe_engine_reference_specs_for(
    instances: &[kernels_sm120::moe::template::Instance],
) -> Vec<kernel_build::CubinSpec> {
    use kernels_sm120::moe::{Phase, Projection};
    let mut groups: Vec<kernel_build::CubinSpec> = Vec::new();
    for instance in instances {
        let v = &instance.baseline;
        let gate_up = u8::from(v.projection == Projection::GateUp);
        let experts = v.query.grouped.expect("grouped").experts;
        let (header, defines) = if v.fp8_fragment_weights() {
            (
                "#include \"crates/kernel-lab/cuda/baselines/qwen_moe_engine_83c99d4c8.cuh\"",
                catalogs()
                    .models()
                    .iter()
                    .find(|model| model.profiles.contains(&v.profile))
                    .and_then(|model| (model.moe_route_cubins)(Some(v.profile)).into_iter().next())
                    .expect("fragment reference has an owning route cubin")
                    .defines,
            )
        } else {
            (
                "#include \"crates/kernel-lab/cuda/baselines/mimo_moe_engine_83c99d4c8.cuh\"",
                Vec::new(),
            )
        };
        let body = if v.fragment_prefill() {
            format!(
                "chop::moe::prefill::run_fragment<{gate_up}, {experts}, {}, {}, {}, {}>(activation, weight_bytes, weight_scales, output, topk_weights, expert_offsets, sorted_rows, num_padded_rows, live_tokens, active_experts);",
                v.query.n, v.query.k, v.weight_stages, v.knobs.stages,
            )
        } else if v.fragment_decode() {
            format!(
                "chop::moe::run_fragment_decode<{gate_up}, {experts}, {}, {}, {}, {}, {}>(activation, weight_bytes, weight_scales, output, topk_weights, expert_offsets, sorted_rows, num_padded_rows, live_tokens, active_experts);",
                v.query.n, v.query.k, v.knobs.split_k_capacity, v.tokens_capacity, v.knobs.warps,
            )
        } else {
            match v.phase {
                Phase::Decode => format!(
                    "chop::moe::qwen::run_decode<{gate_up}, {}, {}, {}, {}, {}, {}>(activation, weight_bytes, weight_f32_scales, output, topk_weights, expert_offsets, sorted_rows, num_padded_rows, live_tokens, active_experts);",
                    v.query.n,
                    v.query.k,
                    v.knobs.split_k_capacity,
                    v.tokens_capacity,
                    v.knobs.warps,
                    v.knobs.stages,
                ),
                Phase::Prefill => format!(
                    "chop::moe::qwen::run_prefill<{gate_up}, {}, {}, {}, {}>(activation, weight_bytes, weight_f32_scales, output, topk_weights, expert_offsets, sorted_rows, num_padded_rows, live_tokens);",
                    v.query.n, v.query.k, v.weight_stages, v.knobs.stages,
                ),
            }
        };
        // Built from the engine alone: the entry ABI, the engine body and the
        // engine plan's block as its bound. The served entry's thread count
        // follows its schedule-table row and must not reach the reference.
        let block = v
            .engine_plan(kernel_core::contract::RuntimeCounts {
                m: v.tokens_capacity,
                batch: 0,
                context: 0,
                routed_rows: v.query.m_capacity,
                split_k: v.knobs.split_k_capacity,
            })
            .expect("engine reference plan")
            .block[0];
        let entry = kernel_build::EntrySpec::new(
            instance.engine_symbol.clone(),
            kernels_sm120::moe::parameters(),
            body,
            block,
            1,
        );
        match groups.iter_mut().find(|g| g.name == instance.engine_cubin) {
            Some(group) => group.entries.push(entry),
            None => groups.push(kernel_build::CubinSpec {
                name: instance.engine_cubin.clone(),
                sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
                include_dirs: vec![
                    std::path::PathBuf::from("crates/kernels-sm120/cuda"),
                    std::path::PathBuf::from("models/qwen38/bundle/cuda"),
                    std::path::PathBuf::from("crates/kernels-sm120/cuda/common"),
                    std::path::PathBuf::from("crates/kernels-sm120/cuda/moe"),
                ],
                headers: vec![header.into()],
                defines,
                entries: vec![entry],
            }),
        }
    }
    groups
}
