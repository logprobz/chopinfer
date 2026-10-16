// The frozen cluster decode kernels the shared template replaced, included by
// build.rs (which compiles them into kernel-lab) and by
// tests/dense_template.rs (which checks their specs on the CPU).

// Qwen3.8's cluster decode as served before its entries ran the shared
// template, frozen in cuda/baselines as a lab-only reference: each served
// entry of a model that serves the template gets a chop_gemm_eng_ copy with
// the entry ABI, threads and minimum blocks and the old kernel's body.
// tests/dense_template.rs compares the served template with it bit for bit
// on generated and checkpoint inputs.
pub fn gemm_cluster_reference_specs() -> Vec<kernel_build::CubinSpec> {
    gemm_cluster_reference_specs_for(&kernels_sm120::gemm::cluster::lab_instances(catalogs()))
}

/// The old `chop::gemm::qwen38::run_cluster_decode` body of a Qwen3.8
/// cluster decode entry (the HC precompute producers with the bottleneck
/// output), as `gemm::fused::Entry::body` generated it.
pub fn qwen_cluster_reference_body(entry: &kernels_sm120::gemm::fused::Entry, precompute: bool) -> String {
    let kernels_sm120::gemm::fused::Kind::ClusterDecode {
        strips,
        warp_blocks,
        passes,
    } = entry.kind
    else {
        panic!("{} is not a cluster decode entry", entry.symbol);
    };
    let count = entry.segments.len();
    let list = |f: &dyn Fn(usize) -> String| (0..count).map(f).collect::<Vec<_>>().join(", ");
    let body = format!(
        "const chop::gemm::qwen38::DecodeSegments<{count}> segments{{{{{}}}, {{{}}}, {{{}}}, {{{}}}}}; chop::gemm::qwen38::run_cluster_decode<{}, {}, {strips}, {warp_blocks}, {passes}, {count}, {}>(segments, activation, partial, live_m, output_f32);",
        list(&|s| format!("weight{s}")),
        list(&|s| format!("output{s}")),
        list(&|s| format!("{}u", entry.segments[s])),
        list(&|s| format!("stride{s}")),
        entry.query.m_capacity.next_power_of_two().max(8),
        entry.query.k / 128,
        entry.query.epilogues.swiglu.is_some()
    );
    if precompute {
        body.replacen("false>(segments", "false, true>(segments", 1)
            .replacen("live_m, output_f32);", "live_m, output_f32, bottleneck);", 1)
    } else {
        body
    }
}

/// The references of `instances` whose model serves the template.
pub fn gemm_cluster_reference_specs_for(
    instances: &[kernels_sm120::gemm::cluster::Instance],
) -> Vec<kernel_build::CubinSpec> {
    use kernels_sm120::gemm::cluster::Served;
    let mut groups: Vec<kernel_build::CubinSpec> = Vec::new();
    for instance in instances.iter().filter(|i| i.serves_template()) {
        let body = match &instance.served {
            Served::Fused(entry) => qwen_cluster_reference_body(entry, false),
            Served::FusedHcPrecompute(entry) => qwen_cluster_reference_body(entry, true),
            Served::Shared(_) => panic!("{}: MiMo keeps its own kernel", instance.served_symbol),
        };
        // The served entry's ABI, threads and minimum blocks.
        let mut entry = instance.template.clone();
        entry.symbol = instance.reference_symbol.clone();
        entry.body = body;
        match groups.iter_mut().find(|g| g.name == instance.reference_cubin) {
            Some(group) => group.entries.push(entry),
            None => groups.push(kernel_build::CubinSpec {
                name: instance.reference_cubin.clone(),
                sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".."),
                include_dirs: vec![std::path::PathBuf::from("kernels-sm120/cuda")],
                headers: vec![
                    "#include \"kernel-lab/cuda/baselines/qwen_cluster_decode_97d9ac95e.cuh\""
                        .into(),
                ],
                defines: vec![],
                entries: vec![entry],
            }),
        }
    }
    groups
}
