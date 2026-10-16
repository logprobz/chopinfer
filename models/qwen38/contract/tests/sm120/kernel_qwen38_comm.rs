use kernel_build::{BuildOptions, Toolchain};
use kernel_core::contract::SigmoidMeanHyperConnectionQuery;
use model_qwen38_contract::sm120::comm as qwen38;
#[test]
fn qwen_comm_compiles_and_refuses_other_geometries() {
    let query =
        |ranks, streams, width, capacity| SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
            ranks,
            streams,
            width,
            capacity,
            eps_bits: 1e-6_f32.to_bits(),
        };
    for q in [
        query(1, 4, 2560, 32),
        query(3, 4, 2560, 32),
        query(16, 4, 2560, 32),
        query(4, 2, 2560, 32),
        query(4, 4, 4096, 32),
        query(4, 4, 2560, 8193),
    ] {
        assert!(qwen38::plan(q, 1).is_err());
    }
    assert!(
        qwen38::cubin_specs(Some(
            kernels_sm120::Model::new(
                kernels_sm120::profile_names!("other-model", "other", 4, 2),
                model_qwen38_contract::sm120::config::CONFIG
            )
            .at(4)
        ))
        .is_empty()
    );
    for capacity in [1, 32, 512, 8192] {
        let q = query(4, 4, 2560, capacity);
        assert_eq!(qwen38::plan(q, capacity).unwrap().cluster, [8, 1, 1]);
        assert!(qwen38::plan(q, capacity + 1).is_err());
        let size = qwen38::buffers(q).unwrap();
        assert_eq!(
            size.inbox_bytes,
            (8 * 32 + 4 * u64::from(capacity.div_ceil(4))) * 2560 * 2
                + u64::from(capacity) * 2560 * 4
        );
        assert_eq!((size.flags_bytes, size.epoch_bytes), (1408, 16));
        assert_eq!(
            qwen38::one_shot_capacity(q).unwrap(),
            capacity.min(qwen38::ONE_SHOT_ROWS)
        );
        assert!(qwen38::prefill_plan(q, capacity + 1).is_err());
    }
    // One sending cluster and a worker per row, 8 to 128 workers.
    let q = query(4, 4, 2560, 8192);
    for (rows, grid) in [
        (0, 16),
        (8, 16),
        (9, 24),
        (33, 48),
        (120, 128),
        (121, 136),
        (512, 136),
        (8192, 136),
    ] {
        let plan = qwen38::prefill_plan(q, rows).unwrap();
        assert_eq!(
            (plan.grid[0], plan.cluster, plan.pdl),
            (grid, [8, 1, 1], true),
            "{rows}"
        );
    }
    // The protocol choice by rows, and the one-shot capacity on a shared
    // state.
    let call = |capacity| qwen38::Call {
        input: 1,
        residual: 2,
        logits: 3,
        weight: 4,
        residual_out: 5,
        normed: 6,
        inboxes: vec![7; 4],
        flags: vec![8; 4],
        epoch: 9,
        live_rows: 10,
        capacity,
        rank: 1,
    };
    let symbols = |rows| -> Vec<_> {
        qwen38::launches(q, rows, call(8192))
            .unwrap()
            .iter()
            .map(|l| (l.symbol, l.plan.grid[0]))
            .collect()
    };
    assert_eq!(symbols(32), [(qwen38::SYMBOL, 8)]);
    assert_eq!(
        symbols(33),
        [(qwen38::SCATTER_SYMBOL, 48), (qwen38::GATHER_SYMBOL, 48)]
    );
    for rows in [0, 1, 32, 33, 65, 129, 512, 8192] {
        let launches = qwen38::launches(q, rows, call(8192)).unwrap();
        assert!(launches.iter().all(|launch| launch.plan.grid[0] <= 136));
        let expected = if rows <= 32 {
            vec![qwen38::SYMBOL]
        } else {
            vec![qwen38::SCATTER_SYMBOL, qwen38::GATHER_SYMBOL]
        };
        assert_eq!(
            launches
                .iter()
                .map(|launch| launch.symbol)
                .collect::<Vec<_>>(),
            expected
        );
    }
    // Capacity, rank and rank count close the arguments.
    let one_shot = &qwen38::launches(q, 1, call(8192)).unwrap()[0].arguments.0;
    assert!(matches!(
        one_shot[one_shot.len() - 3..],
        [
            kernel_core::contract::Argument::U32(32),
            kernel_core::contract::Argument::U32(1),
            kernel_core::contract::Argument::U32(4)
        ]
    ));
    assert!(qwen38::launches(q, 33, call(4096)).is_err());
    assert!(qwen38::launches(q, 8193, call(8192)).is_err());
    assert!(qwen38::arguments(call(32)).is_ok());
    assert!(qwen38::arguments(call(33)).is_err());
    assert!(qwen38::standalone_arguments(call(8192)).is_ok());
    assert!(!qwen38::phased(qwen38::ONE_SHOT_ROWS));
    assert!(qwen38::phased(qwen38::ONE_SHOT_ROWS + 1));
    let root = std::env::temp_dir().join(format!("qwen38-comm-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let reports = kernel_build::build(
        &Toolchain::detect().unwrap(),
        &qwen38::cubin_specs(None),
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    let symbols: Vec<_> = reports[0]
        .entries
        .iter()
        .map(|e| e.symbol.as_str())
        .collect();
    assert_eq!(
        symbols,
        [
            qwen38::SYMBOL,
            qwen38::SCATTER_SYMBOL,
            qwen38::GATHER_SYMBOL
        ]
    );
    assert!(reports[0].entries.iter().all(|e| e.local_bytes == 0));
    std::fs::remove_dir_all(root).unwrap();
}

/// Read the device's simple offset arithmetic rather than copying its
/// constants into the host check. A changed expression must still agree
/// with every supported rank count's flag allocation.
#[test]
fn prefill_gather_flags_match_device_offsets() {
    let header = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../crates/kernels-sm120/cuda/comm/hyper_connection.cuh"
    ));
    let expression = |prefix: &str| {
        assert_eq!(header.matches(prefix).count(), 1, "{prefix}");
        header
            .split_once(prefix)
            .unwrap()
            .1
            .split(';')
            .next()
            .unwrap()
    };
    let blocks: u64 = expression("constexpr uint32_t Blocks = ")
        .trim()
        .parse()
        .unwrap();
    let first = expression("uint32_t first_slot() const { return ");
    let gather = expression("const uint32_t gather_blocks = ");
    let body = header.split_once("void allgather_hc_v2(").unwrap().1;
    let offset_expression = |prefix: &str| {
        assert_eq!(body.matches(prefix).count(), 1, "{prefix}");
        body.split_once(prefix)
            .unwrap()
            .1
            .split(", mark)")
            .next()
            .unwrap()
    };
    let writer = offset_expression("publish(slots(peer(flags, dest), layout) + ");
    let reader = offset_expression("wait_for(local + ");
    let evaluate = |expression: &str, ranks: u64, source: u64, block: u64, gather: u64| -> u64 {
        expression
            .split('+')
            .map(|term| {
                term.split('*')
                    .map(|factor| match factor.trim() {
                        "ranks" => ranks,
                        "rank" | "source" => source,
                        "block" => block,
                        "Blocks" => blocks,
                        "gather_blocks" => gather,
                        literal => literal
                            .parse::<u64>()
                            .expect("device layout uses literal arithmetic"),
                    })
                    .product::<u64>()
            })
            .sum()
    };
    assert_eq!(blocks, 32);
    for ranks in [2_u32, 4, 8] {
        let state = qwen38::buffers(qwen38::query(ranks)).unwrap();
        let ranks = u64::from(ranks);
        let first_slot = evaluate(first, ranks, 0, 0, 0);
        let gather = evaluate(gather, ranks, 0, 0, 0);
        let begin = 8 * (first_slot + gather);
        assert_eq!(
            begin,
            8 * (3 * ranks + blocks) + 32,
            "the first gather flag follows the scatter's flags, counter and spare bytes"
        );
        assert_eq!(state.flags_bytes, 8 * (3 * ranks + 36 + 32 * ranks));
        for source in 0..ranks {
            for block in 0..blocks {
                let expected = begin + 8 * (source * blocks + block);
                for expression in [writer, reader] {
                    let offset =
                        8 * (first_slot + evaluate(expression, ranks, source, block, gather));
                    assert_eq!(offset, expected, "device source/block slot");
                    assert!(offset + 8 <= state.flags_bytes);
                }
            }
        }
        let end = begin + 8 * ranks * blocks;
        assert_eq!(end, state.flags_bytes, "last device flag fits exactly");
    }
}
