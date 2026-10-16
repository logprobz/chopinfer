use std::collections::BTreeSet;
use std::path::Path;

#[test]
fn comm_cubin_compiles_and_is_audited() {
    let toolchain = kernel_build::Toolchain::detect().expect("CUDA toolchain");
    if toolchain.skipped {
        eprintln!("skipping comm cubin compile: CUDA toolchain unavailable");
        return;
    }
    let spec = kernels_sm120::comm::cubin_spec();
    let generated = kernel_build::generate(&spec).expect("generate comm wrapper");
    assert!(
        generated
            .translation_unit
            .contains("chop_comm_allreduce_oneshot_f32_v1")
    );
    let scratch = std::env::temp_dir().join(format!("chop-comm-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("create compile scratch");
    let lock = scratch.join("kernels.lock");
    let options = kernel_build::BuildOptions {
        lock_update: true,
        emit_cargo_directives: false,
        profile_line_info: false,
    };
    let built = kernel_build::build(
        &toolchain,
        &[spec, kernels_sm120::comm::decode_cubin_spec_v2()],
        Path::new(&scratch),
        &lock,
        options,
    )
    .expect("build audited comm cubin");
    assert_eq!(built.len(), 2);
    let exports: BTreeSet<_> = built
        .iter()
        .flat_map(|cubin| &cubin.entries)
        .map(|entry| entry.symbol.as_str())
        .collect();
    for entry in built.iter().flat_map(|cubin| &cubin.entries) {
        assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        assert_eq!(entry.stack_bytes, 0, "{} uses a stack", entry.symbol);
        if entry.symbol == kernels_sm120::comm::DECODE_SYMBOL_V2 {
            for ranks in [2, 4, 8] {
                let plan = kernels_sm120::comm::decode_plan(ranks, 2).unwrap();
                assert_eq!(entry.max_threads, plan.block[0]);
                assert!(
                    kernels_sm120::comm::sm120_ctas_per_sm(
                        plan.block[0],
                        entry.registers,
                        entry.shared_static_bytes + plan.dynamic_shared_bytes,
                    ) >= 1,
                    "{} cannot fit its LL block",
                    entry.symbol
                );
            }
        }
    }
    // The wide entries' plans count on this many resident CTAs per SM.
    for symbol in [
        kernels_sm120::comm::WIDE_SYMBOL,
        kernels_sm120::comm::WIDE_COMBINE_SYMBOL,
    ] {
        let entry = built
            .iter()
            .flat_map(|cubin| &cubin.entries)
            .find(|entry| entry.symbol == symbol)
            .expect("wide entry built");
        let resident = kernels_sm120::comm::sm120_ctas_per_sm(
            entry.max_threads,
            entry.registers,
            entry.shared_static_bytes,
        );
        assert!(
            entry.min_blocks == kernels_sm120::comm::WIDE_CTAS_PER_SM
                && resident >= kernels_sm120::comm::WIDE_CTAS_PER_SM,
            "{symbol}: {resident} CTAs per SM ({} registers, {} shared bytes)",
            entry.registers,
            entry.shared_static_bytes
        );
    }
    for symbol in [
        "chop_comm_exchange_rows_f32_v1",
        "chop_comm_exchange_rows_bf16_v1",
    ] {
        assert!(exports.contains(symbol), "{symbol} is absent");
    }
    for item in kernels_sm120::comm::catalog(&kernels_sm120::Catalogs::new(&[]).unwrap()) {
        assert!(
            exports.contains(item.instantiation.symbol.as_str()),
            "catalog symbol {} is absent",
            item.instantiation.symbol
        );
    }
    std::fs::remove_dir_all(scratch).expect("remove compile scratch");
}
