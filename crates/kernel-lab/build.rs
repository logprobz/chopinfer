#[path = "norm_weights.rs"]
mod norm_weights;

#[path = "shapes.rs"]
mod lab_shapes;

use kernel_build::{BuildOptions, CubinSpec, EntryParam as P, EntrySpec, Toolchain};
use std::path::PathBuf;

/// The defines of a probe build: the MiMo ring's at TP4, then the Qwen3.8
/// bundle rows' at `qwen38`'s rank count.
fn probe_defines(qwen38: model_qwen38_contract::sm120::config::RankShape) -> Vec<(String, String)> {
    let mimo = model_mimo26_contract::sm120::config::PROFILE
        .rank_shape()
        .expect("MiMo's TP4 shapes");
    let mut defines = model_mimo26_contract::sm120::config::ring_defines(mimo);
    defines.extend(qwen38.speculation_defines());
    defines
}

fn probe_spec() -> CubinSpec {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut mma_entries = Vec::new();
    for (format, name) in [
        "bf16",
        "e4m3",
        "e2m1",
        "mxf8_e4m3",
        "mxf8_e2m1",
        "nvfp4_2x",
        "nvfp4_4x",
        "mxf4",
    ]
    .iter()
    .enumerate()
    {
        mma_entries.push(EntrySpec::new(
            format!("chop_probe_mma_{name}_v1"),
            vec![
                P::DevicePointer {
                    name: "output".into(),
                    cxx_type: "float".into(),
                },
                P::DevicePointer {
                    name: "input".into(),
                    cxx_type: "const uint32_t".into(),
                },
                P::U32 {
                    name: "iterations".into(),
                },
            ],
            format!("chop::probe::mma_throughput<{format}>(output, input, iterations);"),
            256,
            1,
        ));
    }
    let mut spec = CubinSpec {
        name: "probe_sm120".into(),
        sources_root: root.join("../.."),
        include_dirs: vec![
            PathBuf::from("crates/kernel-lab/cuda/probes"),
            PathBuf::from("models/qwen38/bundle/cuda"),
            PathBuf::from("crates/kernels-sm120/cuda/common"),
            PathBuf::from("crates/kernels-sm120/cuda"),
        ],
        headers: vec![
            "#include \"crates/kernel-lab/cuda/probes/probe.cuh\"".into(),
            "#include \"models/mimo26/bundle/cuda/mimo_ring.cuh\"".into(),
            "#include \"models/qwen38/bundle/cuda/qwen38_bundle.cuh\"".into(),
            "#include \"models/qwen38/bundle/cuda/qwen38_speculation.cuh\"".into(),
        ],
        defines: probe_defines(
            model_qwen38_contract::sm120::config::profile_shape(
                model_qwen38_contract::sm120::config::PROFILE,
            )
            .expect("Qwen3.8's TP4 shapes"),
        ),
        entries: vec![
            EntrySpec::new(
                "chop_probe_sm120_v1".into(),
                vec![P::DevicePointer {
                    name: "output".into(),
                    cxx_type: "unsigned int".into(),
                }],
                "chop::probe::sm120(output);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_probe_gate_v1".into(),
                vec![P::DevicePointer {
                    name: "flag".into(),
                    cxx_type: "const unsigned int".into(),
                }],
                "chop::probe::gate(flag);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_probe_gate_release_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "flag".into(),
                        cxx_type: "const unsigned int".into(),
                    },
                    P::PointerArray {
                        name: "peers".into(),
                        slots: 8,
                        cxx_type: "unsigned int".into(),
                    },
                    P::U32 {
                        name: "count".into(),
                    },
                ],
                "chop::probe::gate_release(flag, peers, count);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_probe_flush_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "buffer".into(),
                        cxx_type: "unsigned int".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                ],
                "chop::probe::flush(buffer, words);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_cluster_dsm_v1".into(),
                vec![P::DevicePointer {
                    name: "output".into(),
                    cxx_type: "unsigned int".into(),
                }],
                "chop::probe::cluster_dsm(output);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_probe_cluster_control_v1".into(),
                vec![P::DevicePointer {
                    name: "output".into(),
                    cxx_type: "unsigned int".into(),
                }],
                "chop::probe::cluster_control(output);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_probe_gather4_v1".into(),
                vec![
                    P::TensorMap { name: "map".into() },
                    P::DevicePointer {
                        name: "output".into(),
                        cxx_type: "unsigned int".into(),
                    },
                ],
                "chop::probe::gather4(&map, output);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_probe_p2p_pull_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "output".into(),
                        cxx_type: "unsigned int".into(),
                    },
                    P::PointerArray {
                        name: "peers".into(),
                        slots: 2,
                        cxx_type: "const unsigned int".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                ],
                "chop::probe::p2p_pull(output, peers, words);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_p2p_push_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "input".into(),
                        cxx_type: "const unsigned int".into(),
                    },
                    P::PointerArray {
                        name: "peers".into(),
                        slots: 2,
                        cxx_type: "unsigned int".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                ],
                "chop::probe::p2p_push(input, peers, words);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_p2p_push_fanout_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "input".into(),
                        cxx_type: "const unsigned int".into(),
                    },
                    P::PointerArray {
                        name: "peers".into(),
                        slots: 8,
                        cxx_type: "unsigned int".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                ],
                "chop::probe::p2p_push_fanout(input, peers, words);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_p2p_push_pull_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "input".into(),
                        cxx_type: "const unsigned int".into(),
                    },
                    P::PointerArray {
                        name: "peers".into(),
                        slots: 8,
                        cxx_type: "unsigned int".into(),
                    },
                    P::DevicePointer {
                        name: "local".into(),
                        cxx_type: "unsigned int".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                    P::U32 {
                        name: "first_row".into(),
                    },
                ],
                "chop::probe::p2p_push_pull(input, peers, local, words, first_row);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_p2p_ping_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "local_flag".into(),
                        cxx_type: "unsigned int".into(),
                    },
                    P::PointerArray {
                        name: "peers".into(),
                        slots: 2,
                        cxx_type: "unsigned int".into(),
                    },
                    P::DevicePointer {
                        name: "stamps".into(),
                        cxx_type: "uint64_t".into(),
                    },
                    P::U32 {
                        name: "iterations".into(),
                    },
                ],
                "chop::probe::p2p_ping(local_flag, peers, stamps, iterations);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_probe_p2p_pong_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "local_flag".into(),
                        cxx_type: "unsigned int".into(),
                    },
                    P::PointerArray {
                        name: "peers".into(),
                        slots: 2,
                        cxx_type: "unsigned int".into(),
                    },
                    P::U32 {
                        name: "iterations".into(),
                    },
                ],
                "chop::probe::p2p_pong(local_flag, peers, iterations);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_probe_shared_limit_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "output".into(),
                        cxx_type: "unsigned int".into(),
                    },
                    P::U32 {
                        name: "bytes".into(),
                    },
                ],
                "chop::probe::shared_limit(output, bytes);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_probe_pdl_first_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "stamps".into(),
                        cxx_type: "uint64_t".into(),
                    },
                    P::U64 {
                        name: "cycles".into(),
                    },
                ],
                "chop::probe::pdl_first(stamps, cycles);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_probe_pdl_second_v1".into(),
                vec![P::DevicePointer {
                    name: "stamps".into(),
                    cxx_type: "uint64_t".into(),
                }],
                "chop::probe::pdl_second(stamps);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_probe_read_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "input".into(),
                        cxx_type: "const unsigned int".into(),
                    },
                    P::DevicePointer {
                        name: "checksum".into(),
                        cxx_type: "unsigned int".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                    P::U32 {
                        name: "repeats".into(),
                    },
                ],
                "chop::probe::bandwidth_read(input, checksum, words, repeats);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_write_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "output".into(),
                        cxx_type: "unsigned int".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                ],
                "chop::probe::bandwidth_write(output, words);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_copy_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "input".into(),
                        cxx_type: "const unsigned int".into(),
                    },
                    P::DevicePointer {
                        name: "output".into(),
                        cxx_type: "unsigned int".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                ],
                "chop::probe::bandwidth_copy(input, output, words);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_read_vector_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "input".into(),
                        cxx_type: "const uint4".into(),
                    },
                    P::DevicePointer {
                        name: "checksum".into(),
                        cxx_type: "unsigned int".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                    P::U32 {
                        name: "repeats".into(),
                    },
                ],
                "chop::probe::bandwidth_read_vector(input, checksum, words, repeats);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_write_vector_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "output".into(),
                        cxx_type: "uint4".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                ],
                "chop::probe::bandwidth_write_vector(output, words);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_copy_vector_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "input".into(),
                        cxx_type: "const uint4".into(),
                    },
                    P::DevicePointer {
                        name: "output".into(),
                        cxx_type: "uint4".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                ],
                "chop::probe::bandwidth_copy_vector(input, output, words);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_read_tma_v1".into(),
                vec![
                    P::DevicePointer {
                        name: "input".into(),
                        cxx_type: "const unsigned int".into(),
                    },
                    P::DevicePointer {
                        name: "checksum".into(),
                        cxx_type: "unsigned int".into(),
                    },
                    P::U64 {
                        name: "words".into(),
                    },
                ],
                "chop::probe::bandwidth_read_tma(input, checksum, words);".into(),
                256,
                1,
            ),
        ],
    };
    spec.entries.extend(mma_entries);
    // The MiMo bundle's SWA ring backup, checked against its host bookkeeping.
    let pointer = |name: &str, cxx_type: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    spec.entries.push(EntrySpec::new(
        "chop_probe_mimo_ring_backup_v1".into(),
        vec![
            pointer("state", "uint8_t"),
            pointer("slot_ids", "const int32_t"),
            pointer("q_len", "const int32_t"),
            pointer("prefix_len", "const int32_t"),
            pointer("live_batch", "const uint32_t"),
            P::U32 {
                name: "span".into(),
            },
        ],
        "chop::mimo_ring::backup(state, slot_ids, q_len, prefix_len, live_batch, span);".into(),
        96,
        1,
    ));
    // Its restore of a verify step's rejected tail.
    spec.entries.push(EntrySpec::new(
        "chop_probe_mimo_ring_restore_v1".into(),
        vec![
            pointer("state", "uint8_t"),
            pointer("slot_ids", "const int32_t"),
            pointer("starts", "const int32_t"),
            pointer("counts", "const int32_t"),
            pointer("live_items", "const uint32_t"),
            P::U32 {
                name: "span".into(),
            },
        ],
        "chop::mimo_ring::restore(state, slot_ids, starts, counts, live_items, span);".into(),
        96,
        1,
    ));
    // Its prefix-cache snapshot copy between a ring slot and pool pages.
    spec.entries.push(EntrySpec::new(
        "chop_probe_mimo_ring_prefix_copy_v1".into(),
        vec![
            pointer("state", "uint8_t"),
            pointer("drafter", "uint8_t"),
            P::U64 {
                name: "drafter_bytes".into(),
            },
            pointer("pool", "uint8_t"),
            pointer("pages", "const uint64_t"),
            P::U32 {
                name: "page_count".into(),
            },
            P::U64 {
                name: "page_bytes".into(),
            },
            P::U32 {
                name: "slot".into(),
            },
            P::U32 {
                name: "to_pages".into(),
            },
        ],
        "chop::mimo_ring::prefix_copy(state, drafter, drafter_bytes, pool, pages, page_count, \
         page_bytes, slot, to_pages);"
            .into(),
        256,
        1,
    ));
    // The Qwen3.8 bundle's own rows, checked against CPU references.
    let u32_param = |name: &str| P::U32 { name: name.into() };
    let u64_param = |name: &str| P::U64 { name: name.into() };
    for (symbol, params, body) in [
        (
            "chop_probe_qwen38_ple_residual_v1",
            vec![
                pointer("x", "const uint16_t"),
                pointer("gated", "const uint16_t"),
                pointer("conv", "const uint16_t"),
                pointer("out", "uint16_t"),
                pointer("live_rows", "const uint32_t"),
            ],
            "ple_residual(x, gated, conv, out, live_rows)",
        ),
        (
            "chop_probe_qwen38_ple_stage_v1",
            vec![
                pointer("pool", "const uint8_t"),
                pointer("staging", "uint8_t"),
                pointer("slot_ids", "const int32_t"),
                pointer("prefix_len", "const int32_t"),
                pointer("live_items", "const uint32_t"),
            ],
            "ple_stage(pool, staging, slot_ids, prefix_len, live_items)",
        ),
        (
            "chop_probe_qwen38_copy_entries_v1",
            vec![
                pointer("src", "const uint8_t"),
                pointer("dst", "uint8_t"),
                pointer("src_ids", "const int32_t"),
                pointer("dst_ids", "const int32_t"),
                pointer("live_items", "const uint32_t"),
                u64_param("entry_bytes"),
                u64_param("plane_stride"),
            ],
            "copy_entries(src, dst, src_ids, dst_ids, live_items, entry_bytes, plane_stride)",
        ),
        (
            "chop_probe_qwen38_ngram_verify_v1",
            vec![
                pointer("tokens", "const uint32_t"),
                pointer("history", "const uint32_t"),
                pointer("multipliers", "const uint64_t"),
                pointer("sizes", "const uint64_t"),
                pointer("offsets", "const uint64_t"),
                pointer("table", "const uint8_t"),
                pointer("scale", "const uint16_t"),
                pointer("output", "uint16_t"),
                pointer("row_ids", "uint64_t"),
                pointer("history_out", "uint32_t"),
                pointer("cu_rows", "const int32_t"),
                pointer("q_len", "const int32_t"),
                pointer("live_items", "const uint32_t"),
                P::U32 {
                    name: "rank".into(),
                },
                u64_param("table_start"),
                u64_param("table_rows"),
                u64_param("entry_words"),
            ],
            "ngram_verify(tokens, history, multipliers, sizes, offsets, table, scale, output, \
             row_ids, history_out, cu_rows, q_len, live_items, rank, table_start, table_rows, \
             entry_words)",
        ),
        (
            "chop_probe_qwen38_ple_conv_verify_v1",
            vec![
                pointer("input", "const uint16_t"),
                pointer("weight", "const uint16_t"),
                pointer("state", "const uint16_t"),
                pointer("output", "uint16_t"),
                pointer("state_out", "uint16_t"),
                pointer("cu_rows", "const int32_t"),
                pointer("q_len", "const int32_t"),
                pointer("live_items", "const uint32_t"),
                u64_param("entry"),
            ],
            "ple_conv_verify(input, weight, state, output, state_out, cu_rows, q_len, \
             live_items, entry)",
        ),
        (
            "chop_probe_qwen38_ple_checkpoints_v1",
            vec![
                pointer("input", "const uint16_t"),
                pointer("tokens", "const uint32_t"),
                pointer("staged", "const uint8_t"),
                pointer("checkpoints", "uint8_t"),
                pointer("cu_rows", "const int32_t"),
                pointer("q_len", "const int32_t"),
                pointer("live_items", "const uint32_t"),
            ],
            "ple_checkpoints(input, tokens, staged, checkpoints, cu_rows, q_len, live_items)",
        ),
        (
            "chop_probe_qwen38_mtp_gather_v1",
            vec![
                pointer("last", "const uint16_t"),
                pointer("retained", "const uint16_t"),
                pointer("sources", "const int32_t"),
                pointer("token_sources", "const int32_t"),
                pointer("drafts", "const int32_t"),
                pointer("x", "uint16_t"),
                pointer("token_ids", "int32_t"),
                pointer("live_rows", "const uint32_t"),
            ],
            "mtp_gather(last, retained, sources, token_sources, drafts, x, token_ids, live_rows)",
        ),
        (
            "chop_probe_qwen38_mtp_norms_v1",
            vec![
                pointer("x", "const uint16_t"),
                pointer("hidden_weight", "const uint16_t"),
                pointer("embed", "const uint16_t"),
                pointer("embed_weight", "const uint16_t"),
                pointer("x_normed", "uint16_t"),
                pointer("embed_normed", "uint16_t"),
                pointer("live_rows", "const uint32_t"),
            ],
            "mtp_norms(x, hidden_weight, embed, embed_weight, x_normed, embed_normed, live_rows)",
        ),
        (
            "chop_probe_qwen38_mtp_combine_v1",
            vec![
                pointer("hidden", "const uint16_t"),
                pointer("token", "const uint16_t"),
                pointer("x", "uint16_t"),
                pointer("live_rows", "const uint32_t"),
            ],
            "mtp_combine(hidden, token, x, live_rows)",
        ),
        (
            "chop_probe_qwen38_draft_argmax_partial_v1",
            vec![
                pointer("logits", "const uint16_t"),
                pointer("best", "uint32_t"),
                pointer("live_rows", "const uint32_t"),
                P::U32 {
                    name: "rank".into(),
                },
            ],
            "draft_argmax_partial(logits, best, live_rows, rank)",
        ),
        (
            "chop_probe_qwen38_draft_argmax_final_v1",
            vec![
                pointer("bests", "const uint32_t"),
                pointer("slots", "const int32_t"),
                pointer("targets", "const int32_t"),
                pointer("drafts", "int32_t"),
                pointer("token_ids", "int32_t"),
                pointer("live_rows", "const uint32_t"),
            ],
            "draft_argmax_final(bests, slots, targets, drafts, token_ids, live_rows)",
        ),
        (
            "chop_probe_qwen38_prefix_copy_v1",
            vec![
                pointer("conv_pool", "uint8_t"),
                pointer("delta_pool", "uint8_t"),
                pointer("ple_pool", "uint8_t"),
                pointer("ring", "uint8_t"),
                pointer("retained", "uint8_t"),
                pointer("pool", "uint8_t"),
                pointer("pages", "const uint64_t"),
                u32_param("ring_layers"),
                u64_param("page_bytes"),
                u32_param("slot"),
                u32_param("index"),
                u32_param("to_pages"),
                u32_param("pool_entries"),
                u32_param("ring_slots"),
            ],
            "prefix_copy(conv_pool, delta_pool, ple_pool, ring, retained, pool, pages, \
             ring_layers, page_bytes, slot, index, to_pages, pool_entries, ring_slots)",
        ),
    ] {
        spec.entries.push(EntrySpec::new(
            symbol.into(),
            params,
            format!("chop::qwen38_bundle::{body};"),
            256,
            1,
        ));
    }
    spec
}

/// The comm cubin with the test-only lag of `comm.cuh` (`test_skew`) and a
/// slow consumer entry, for the cross-rank synchronization stress tests.
/// Symbols are unique per build, so its entries are `chop_comm_skew_*`.
/// The probe's Qwen3.8 bundle rows whose sizes depend on the rank count,
/// built with a TP2 rank's bundle defines (as bundle-qwen38's build
/// does), in `probe_{tag}` as `chop_probe_{tag}_*`.
fn probe_qwen38_rank_spec(ranks: u32) -> CubinSpec {
    let shape = model_qwen38_contract::sm120::config::RankShape { ranks };
    let tag = model_qwen38_contract::sm120::config::profile(ranks)
        .expect("Qwen3.8 rank count")
        .tag();
    let mut spec = probe_spec();
    spec.name = format!("probe_{tag}");
    spec.defines = probe_defines(shape);
    spec.entries.retain(|entry| {
        [
            "chop_probe_qwen38_ngram_verify_v1",
            "chop_probe_qwen38_draft_argmax_partial_v1",
            "chop_probe_qwen38_draft_argmax_final_v1",
            "chop_probe_qwen38_prefix_copy_v1",
        ]
        .contains(&entry.symbol.as_str())
    });
    assert_eq!(spec.entries.len(), 4, "probe Qwen3.8 rank rows");
    for entry in &mut spec.entries {
        entry.symbol = entry.symbol.replacen("qwen38", tag, 1);
    }
    spec
}

fn comm_skew_spec() -> CubinSpec {
    let mut spec = kernels_sm120::comm::cubin_spec();
    spec.name = "comm_sm120_skew".into();
    for entry in &mut spec.entries {
        entry.symbol = entry.symbol.replacen("chop_comm_", "chop_comm_skew_", 1);
    }
    spec.defines
        .push(("CHOP_COMM_TEST_SKEW_NS".into(), "20000".into()));
    spec.entries.push(EntrySpec::new(
        "chop_comm_skew_spin_v1".into(),
        Vec::new(),
        "chop::comm::test_spin();".into(),
        32,
        1,
    ));
    spec
}

fn decode_skew_spec_v2() -> CubinSpec {
    let mut spec = kernels_sm120::comm::decode_cubin_spec_v2();
    spec.name = "comm_decode_v2_skew_sm120".into();
    for entry in &mut spec.entries {
        entry.symbol = entry.symbol.replacen("chop_comm_", "chop_comm_skew_", 1);
    }
    spec.defines
        .push(("CHOP_COMM_TEST_SKEW_NS".into(), "20000".into()));
    spec
}

fn verify_skew_spec() -> CubinSpec {
    let mut spec = kernels_sm120::comm::verify_cubin_spec("comm_verify_skew_sm120");
    for entry in &mut spec.entries {
        entry.symbol = entry.symbol.replacen("chop_comm_", "chop_comm_skew_", 1);
    }
    spec.defines
        .push(("CHOP_COMM_TEST_SKEW_NS".into(), "20000".into()));
    spec
}

fn flat_sum_spec() -> CubinSpec {
    let mut spec = kernels_sm120::comm::decode_cubin_spec_v2();
    spec.name = "comm_decode_flat_sm120".into();
    for entry in &mut spec.entries {
        entry.symbol = entry.symbol.replacen("chop_comm_", "chop_comm_flat_", 1);
    }
    spec.defines
        .push(("CHOP_COMM_TEST_FLAT_SUM".into(), "1".into()));
    spec
}

// Force the two-shot even at two ranks for protocol-transition checks.
// Its norm order is the wide entry's, which supplies the bitwise reference.
fn transition_two_shot_spec() -> CubinSpec {
    let mut spec = threshold_spec(
        "transition",
        "CHOP_COMM_ONESHOT_ROWS_BY_RANKS",
        &[2, 4, 8],
        2,
    );
    spec.entries
        .retain(|entry| entry.symbol.contains("_decode_"));
    let (_, minimum) = spec
        .defines
        .iter_mut()
        .find(|(name, _)| name == "CHOP_COMM_TWO_SHOT_MIN_RANKS")
        .unwrap();
    *minimum = "2".into();
    spec.defines
        .push(("CHOP_COMM_TEST_SKEW_NS".into(), "20000".into()));
    spec
}

/// The Qwen3.8 comm cubin with the same test-only lag at the phased pair's
/// scatter block wait and gather wait; entries are `chop_comm_qwen38_skew_*`.
fn comm_qwen38_skew_spec() -> CubinSpec {
    let mut spec = model_qwen38_contract::sm120::comm::cubin_specs(None).remove(0);
    spec.name = "comm_qwen38_skew".into();
    for entry in &mut spec.entries {
        entry.symbol = entry
            .symbol
            .replacen("chop_comm_qwen38_", "chop_comm_qwen38_skew_", 1);
    }
    spec.defines
        .push(("CHOP_COMM_TEST_SKEW_NS".into(), "20000".into()));
    spec
}

/// The Qwen3.8 decode all-reduce with the same test-only lag, entries
/// `chop_comm_qwen38_skew_allreduce_hc_decode_r{2,4,8}_v{2,3}`.
fn qwen38_decode_skew_spec() -> CubinSpec {
    let mut spec = model_qwen38_contract::sm120::comm::decode::cubin_specs(None).remove(0);
    spec.name = "comm_qwen38_decode_skew".into();
    for entry in &mut spec.entries {
        entry.symbol = entry
            .symbol
            .replacen("chop_comm_qwen38_", "chop_comm_qwen38_skew_", 1);
    }
    spec.defines
        .push(("CHOP_COMM_TEST_SKEW_NS".into(), "20000".into()));
    spec
}

/// The Qwen3.8 wide decode all-reduce with the same test-only lag, entry
/// `chop_comm_qwen38_skew_allreduce_hc_wide_r{2,4,8}_v1`.
fn qwen38_decode_wide_skew_spec() -> CubinSpec {
    let mut spec = model_qwen38_contract::sm120::comm::decode::wide_cubin_specs(None).remove(0);
    spec.name = "comm_qwen38_decode_wide_skew".into();
    for entry in &mut spec.entries {
        entry.symbol = entry
            .symbol
            .replacen("chop_comm_qwen38_", "chop_comm_qwen38_skew_", 1);
    }
    spec.defines
        .push(("CHOP_COMM_TEST_SKEW_NS".into(), "20000".into()));
    spec
}

fn pdl_prefetch_spec() -> CubinSpec {
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    CubinSpec {
        name: "probe_pdl_prefetch_sm120".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".."),
        include_dirs: vec![],
        headers: vec!["#include \"kernel-lab/cuda/probes/pdl_prefetch.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            "chop_probe_pdl_prefetch_producer_v1".into(),
            vec![
                ptr("eviction", "const uint4"),
                ptr("checksum", "uint32_t"),
                P::U64 { name: "vectors".into() },
                ptr("operands", "const chop::probe::PdlOperand"),
                P::U32 { name: "operand_count".into() },
                ptr("live", "uint32_t"),
                P::U32 { name: "rows".into() },
                P::U64 { name: "tail_ns".into() },
            ],
            "chop::probe::pdl_prefetch_producer(eviction,checksum,vectors,operands,operand_count,live,rows,tail_ns);".into(),
            128,
            1,
        )],
    }
}

/// BF16 mma.sync accumulation schemes for `tests/mma_accumulation.rs`.
fn mma_accumulation_spec() -> CubinSpec {
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let u32 = |name: &str| P::U32 { name: name.into() };
    CubinSpec {
        name: "probe_mma_accumulation_sm120".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".."),
        include_dirs: vec![],
        headers: vec!["#include \"kernel-lab/cuda/probes/mma_accumulation.cuh\"".into()],
        defines: vec![],
        entries: vec![
            EntrySpec::new(
                "chop_probe_mma_chain_v1".into(),
                vec![
                    ptr("weight", "const uint16_t"),
                    ptr("activation", "const uint16_t"),
                    ptr("output", "float"),
                    u32("n"),
                    u32("m"),
                    u32("k"),
                    u32("chain"),
                    u32("compensated"),
                ],
                "chop::probe::mma_chain(weight, activation, output, n, m, k, chain, compensated);"
                    .into(),
                128,
                1,
            ),
            EntrySpec::new(
                "chop_probe_fma_chain_v1".into(),
                vec![
                    ptr("weight", "const uint16_t"),
                    ptr("activation", "const uint16_t"),
                    ptr("output", "float"),
                    u32("n"),
                    u32("m"),
                    u32("k"),
                ],
                "chop::probe::fma_chain(weight, activation, output, n, m, k);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_probe_mma_once_v1".into(),
                vec![
                    ptr("a", "const uint16_t"),
                    ptr("b", "const uint16_t"),
                    ptr("c", "const float"),
                    ptr("d", "float"),
                ],
                "chop::probe::mma_once(a, b, c, d);".into(),
                32,
                1,
            ),
        ],
    }
}

// Separate lab cubin: pair and online-softmax accumulation measurements.
fn non_dense_accumulation_spec() -> CubinSpec {
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let u32 = |name: &str| P::U32 { name: name.into() };
    let mut entries = Vec::new();
    for (name, fma) in [
        ("encoder", false),
        ("encoder_fma", true),
        ("pv", false),
        ("pv_fma", true),
    ] {
        let pv = name.starts_with("pv");
        let mut params = vec![
            ptr("weight", "const uint16_t"),
            ptr("activation", "const float"),
        ];
        let mut arguments = String::from("weight, activation");
        if pv {
            params.push(ptr("correction", "const float"));
            arguments.push_str(", correction");
        }
        params.extend([ptr("output", "float"), u32("n"), u32("m"), u32("k")]);
        arguments.push_str(", output, n, m, k");
        if pv {
            params.push(u32("terms"));
            arguments.push_str(", terms");
        }
        if !fma {
            params.push(u32("mode"));
            arguments.push_str(", mode");
        }
        entries.push(EntrySpec::new(
            format!("chop_probe_{name}_accumulation_v1"),
            params,
            format!("chop::probe::non_dense::{name}({arguments});"),
            if fma { 256 } else { 32 },
            1,
        ));
    }
    entries.push(EntrySpec::new(
        "chop_probe_encoder_served_accumulation_v1".into(),
        vec![
            ptr("weight", "const uint16_t"),
            ptr("activation", "const float"),
            ptr("output", "float"),
            ptr("live", "const uint32_t"),
            u32("k"),
        ],
        "chop::probe::non_dense::encoder_served(weight, activation, output, live, k);".into(),
        384,
        1,
    ));
    CubinSpec {
        name: "probe_non_dense_accumulation_sm120".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".."),
        include_dirs: vec![],
        headers: vec!["#include \"kernel-lab/cuda/probes/non_dense_accumulation.cuh\"".into()],
        defines: vec![],
        entries,
    }
}

// Scale-order-faithful expert reduction probes, separate from served cubins.
fn grouped_accumulation_spec() -> CubinSpec {
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let u32 = |name: &str| P::U32 { name: name.into() };
    let mut entries = Vec::new();
    for (name, threads) in [("chain", 32), ("fma", 256)] {
        entries.push(EntrySpec::new(format!("chop_probe_grouped_{name}_v1"),
            vec![ptr("weight", "const uint16_t"), ptr("activation", "const uint16_t"),
                 ptr("scales", "const float"), ptr("offsets", "const uint32_t"), ptr("output", "float"),
                 u32("n"), u32("m"), u32("k"), u32("parts"), u32("post_scale"), u32("run_k"), u32("global_bits")],
            format!("chop::probe::grouped::{name}(weight, activation, scales, offsets, output, n, m, k, parts, post_scale, run_k, __uint_as_float(global_bits));"), threads, 1));
    }
    entries.push(EntrySpec::new(
        "chop_probe_grouped_swiglu_v1".into(),
        vec![
            ptr("gate", "const float"),
            ptr("up", "const float"),
            ptr("output", "uint16_t"),
            u32("count"),
        ],
        "chop::probe::grouped::swiglu(gate, up, output, count);".into(),
        256,
        1,
    ));
    entries.push(EntrySpec::new(
        "chop_probe_tma_chunk_control_v1".into(),
        vec![
            ptr("weight", "const uint16_t"),
            ptr("activation", "const uint16_t"),
            ptr("output", "float"),
            u32("k"),
        ],
        "chop::probe::grouped::tma_chunk(weight, activation, output, k);".into(),
        32,
        1,
    ));
    CubinSpec {
        name: "probe_grouped_accumulation_sm120".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".."),
        include_dirs: vec![],
        headers: vec!["#include \"kernel-lab/cuda/probes/grouped_accumulation.cuh\"".into()],
        defines: vec![],
        entries,
    }
}

/// The fused push entries of `comm_sm120` and plain decode v2 with the
/// per-rank-count row thresholds of `define` (one byte per rank count from
/// ranks 1 in the low byte) set to `rows` at `ranks`, entries renamed from
/// `chop_comm_` to `chop_comm_{tag}_`: protocols the tuning does not pick,
/// for `tests/comm_latency.rs`.
fn threshold_spec(tag: &str, define: &str, ranks: &[u32], rows: u64) -> CubinSpec {
    let mut spec = kernels_sm120::comm::cubin_spec();
    spec.entries
        .extend(kernels_sm120::comm::decode_cubin_spec_v2().entries);
    spec.name = format!("comm_sm120_{tag}");
    spec.entries.retain(|entry| {
        entry.symbol.contains("_decode_") || entry.symbol.contains("_norm_bf16_v2")
    });
    for entry in &mut spec.entries {
        entry.symbol = entry
            .symbol
            .replacen("chop_comm_", &format!("chop_comm_{tag}_"), 1);
    }
    let (_, value) = spec
        .defines
        .iter_mut()
        .find(|(name, _)| name == define)
        .expect("threshold define");
    let packed = u64::from_str_radix(value.trim_start_matches("0x").trim_end_matches("ull"), 16)
        .expect("packed thresholds");
    let packed = ranks.iter().fold(packed, |packed, ranks| {
        let shift = 8 * (ranks - 1);
        (packed & !(0xff << shift)) | (rows << shift)
    });
    *value = format!("{packed:#018x}ull");
    spec
}

// Preserve the original score traversal for a bitwise comparison at the
// original fixed grid. This baseline is only linked by kernel-lab.
fn qsa_fixed_grid_spec() -> CubinSpec {
    let mut spec = kernels_sm120::attention::sparse::cubin_specs(
        catalogs(),
        Some(model_qwen38_contract::sm120::config::PROFILE),
    )
    .remove(0);
    spec.name = "qsa_fixed_grid".into();
    spec.entries.retain(|entry| {
        entry.symbol.contains("block_scores_decode") || entry.symbol.contains("block_scores_verify")
    });
    for entry in &mut spec.entries {
        entry.symbol = entry.symbol.replace("chop_qsa_", "chop_qsa_fixed_");
        entry.body = entry.body.replace(",true>(q", ",false>(q");
    }
    spec
}

// Exercise grouped scoring through the shared engine against the existing
// row-router implementation, without adding production instantiations.
fn grouped_route_spec() -> CubinSpec {
    use kernels_sm120::moe::parallel::{ParallelRoute, RouteScoring};
    let profile = model_qwen38_contract::sm120::config::PROFILE;
    let mut spec =
        model_qwen38_contract::sm120::moe::parallel_route_cubin_spec_for(profile).unwrap();
    spec.name = "route_grouped_lab".into();
    spec.headers.push("#include \"row/router.cuh\"".into());
    let geometry = ParallelRoute {
        experts: 256,
        local_experts: 256,
        top_k: 8,
        padding: 16,
        capacity: 64,
        groups: 8,
        take_groups: 4,
        scoring: RouteScoring::GroupedSoftmax,
    };
    let mut candidate = spec.entries.pop().unwrap();
    candidate.symbol = "chop_route_grouped_parallel_lab_v1".into();
    candidate.max_threads = geometry.plan(64).unwrap().block[0];
    candidate.body = geometry.body("rank", "nullptr", "1.f").unwrap();
    let mut baseline = candidate.clone();
    baseline.symbol = "chop_route_grouped_reference_lab_v1".into();
    baseline.max_threads = 256;
    baseline.body = "chop::row::router<256,8,2,true,8,4>(logits,nullptr,ids,weights,reinterpret_cast<const uint32_t*>(live_tokens),1.f);".into();
    spec.entries = vec![candidate, baseline];
    spec
}

// The frozen per-model grouped-MoE engine references (engine_references.rs,
// shared with tests/moe_template.rs).
include!("engine_references.rs");

// The frozen cluster decode kernels the shared template replaced
// (gemm_references.rs, shared with tests/dense_template.rs).
include!("gemm_references.rs");

// Packed NVFP4 and MXFP8 decode at Qwen3.8 TP4's expert shape
// (`template::packed_lab_instances`): the packed engine, which serves these
// entries outside the template, as the chop_moe_eng_ reference, and the
// template's packed ring at its tuned schedule. The engine is
// built from moe.cuh, which still serves every other packed entry, so its
// reference cubins are the served code. tests/moe_template.rs compares the
// ring with the engine bit for bit.
fn moe_packed_specs() -> Vec<CubinSpec> {
    use kernels_sm120::moe;
    let instances = model_qwen38_contract::sm120::moe::packed_lab_instances();
    let mut cubins: Vec<&str> = instances
        .iter()
        .map(|i| i.template_cubin.as_str())
        .collect();
    // One entry per cubin, in first-seen order (the lock's record order).
    let mut seen = std::collections::BTreeSet::new();
    cubins.retain(|cubin| seen.insert(*cubin));
    let mut specs = Vec::new();
    for cubin in cubins {
        let group: Vec<_> = instances
            .iter()
            .filter(|i| i.template_cubin == cubin)
            .collect();
        let engines: Vec<_> = group.iter().map(|i| i.engine.clone()).collect();
        let templates: Vec<_> = group.iter().map(|i| i.template.clone()).collect();
        specs.push(moe::cubin_spec(&group[0].engine_cubin, &engines).expect("packed engine cubin"));
        specs.push(moe::cubin_spec(cubin, &templates).expect("packed ring cubin"));
    }
    specs
}

/// Whether a lab test launches a MoE catalog cubin. The standard profile's
/// prefill entries run only in the full-path test (`ops::moe::FullCase`),
/// with FP8 block weights, A16 gathered inputs and split gate/up columns;
/// the lab builds no other standard prefill cubin.
fn launched_moe_cubin(spec: &CubinSpec) -> bool {
    !spec.name.starts_with("chop_moe_std_")
        || !spec.name.contains("_prefill_")
        || (spec.name.contains("_fp8block_a16_") && !spec.name.contains("_interleaved_"))
}

/// The catalogs the lab builds: the kernel base's, then every model's of
/// models.rs, as `kernel_lab::catalogs`.
#[path = "models.rs"]
pub mod models;

fn catalogs() -> &'static kernels_sm120::Catalogs {
    static CATALOGS: std::sync::OnceLock<kernels_sm120::Catalogs> = std::sync::OnceLock::new();
    CATALOGS.get_or_init(|| {
        kernels_sm120::Catalogs::new(models::CATALOGS)
            .expect("the lab's model catalogs are consistent")
    })
}

fn main() {
    model_mimo26_contract::sm120::geometry::require_base_geometries("kernel-lab")
        .unwrap_or_else(|error| panic!("{error}"));
    if let Err(error) =
        model_qwen38_contract::sm120::geometry::require_base_geometries("kernel-lab")
            .and_then(|()| model_qwen38_contract::sm120::geometry::validate_catalogs())
    {
        eprintln!("{error}");
        std::process::exit(1);
    }
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    let specs: Vec<_> = [probe_spec(), kernels_sm120::comm::cubin_spec()]
        .into_iter()
        .chain(std::iter::once(
            model_mimo26_contract::sm120::comm::verify_cubin_spec(),
        ))
        .chain([norm_weights::cubin_spec(), norm_weights::wrong_tree_spec()])
        .chain(std::iter::once(pdl_prefetch_spec()))
        .chain(std::iter::once(mma_accumulation_spec()))
        .chain(std::iter::once(non_dense_accumulation_spec()))
        .chain(std::iter::once(grouped_accumulation_spec()))
        .chain(std::iter::once(comm_skew_spec()))
        .chain([
            kernels_sm120::comm::decode_cubin_spec_v2(),
            decode_skew_spec_v2(),
            verify_skew_spec(),
            flat_sum_spec(),
            transition_two_shot_spec(),
        ])
        .chain({
            let spec = comm_qwen38_skew_spec();
            [norm_weights::legacy_epilogue_spec(spec.clone()), spec]
        })
        .chain(kernels_sm120::attention::cubin_specs(catalogs(), None))
        .chain(kernels_sm120::attention::sparse::cubin_specs(
            catalogs(),
            None,
        ))
        .chain(std::iter::once(qsa_fixed_grid_spec()))
        .chain(std::iter::once(grouped_route_spec()))
        .chain(kernels_sm120::gemm::cubin_specs(catalogs(), None))
        .chain(kernels_sm120::gemm::fused::cubin_specs(catalogs(), None))
        .chain(std::iter::once(
            model_mimo26_contract::sm120::lab::rows64_probe_cubin_spec(),
        ))
        .chain(kernels_sm120::gemm::mid::cubin_specs(catalogs(), None))
        .chain(kernels_sm120::gemm::cluster::lab_cubin_specs(catalogs()))
        .chain(gemm_cluster_reference_specs())
        .chain(kernels_sm120::row::cubin_specs(catalogs(), None))
        .chain(kernels_sm120::row::hyper_connection::cubin_specs(
            catalogs(),
            None,
        ))
        .chain(model_qwen38_contract::sm120::comm::cubin_specs(None))
        .chain(model_qwen38_contract::sm120::comm::decode::cubin_specs(
            None,
        ))
        .chain(std::iter::once(qwen38_decode_skew_spec()))
        .chain([probe_qwen38_rank_spec(2)])
        .chain(model_qwen38_contract::sm120::comm::decode::wide_cubin_specs(None))
        .chain(std::iter::once(qwen38_decode_wide_skew_spec()))
        .chain(
            kernels_sm120::moe::cubin_specs(catalogs(), None)
                .into_iter()
                .filter(launched_moe_cubin),
        )
        .chain(moe_engine_reference_specs())
        .chain(moe_packed_specs())
        .chain(kernels_sm120::sampling::cubin_specs(None))
        .chain(kernels_sm120::linear::cubin_specs(catalogs(), None))
        .chain(kernels_sm120::context_parallel::cubin_specs(
            catalogs(),
            None,
        ))
        .collect();
    let (hc_specs, specs): (Vec<_>, Vec<_>) = specs.into_iter().partition(|spec| {
        spec.name == norm_weights::LEGACY_CUBIN
            || spec.entries.iter().any(|entry| {
                entry.body.contains("::allgather_hc_v2<") || entry.body.contains("::allreduce_hc<")
            })
    });
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    let lock = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels.lock");
    let context = kernel_build::BuildContext::new(
        concat!(env!("CARGO_PKG_NAME"), "@", env!("CARGO_PKG_VERSION")),
        "chopinfer",
        "kernel-lab.catalogs-and-probes.v1",
    )
    .source_root(
        "chopinfer",
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
    );
    let mut built = kernel_build::build_with_context(
        &toolchain,
        &specs,
        &out,
        &lock,
        BuildOptions::from_env(),
        &context,
    )
    .expect("build audited kernel-lab cubins");
    // Model-owned lab records stay beside their owned CUDA source root.
    let hc = hc_specs.first().expect("hyper-connection lab specs");
    let hc_lock = hc
        .sources_root
        .join(hc.include_dirs.last().expect("owned CUDA root"))
        .join("comm/lab/kernels.lock");
    let hc_out = out.join("hc-prefill");
    built.extend(
        kernel_build::build_with_context(
            &toolchain,
            &hc_specs,
            &hc_out,
            &hc_lock,
            BuildOptions::from_env(),
            &context,
        )
        .expect("build audited hyper-connection lab cubins"),
    );
    std::fs::copy(&hc_lock, hc_out.join("kernels.lock"))
        .expect("embed hyper-connection lab identities");
    // New engines' cubins gate on their own lock families, one per file of
    // `families/` (`kernel_lab_families`), in one build into `families/` with
    // one lock directory listing every family, so the root and
    // hyper-connection builds keep their own reports.
    let families = kernel_lab_families::all();
    let lock = families.iter().fold(
        kernel_build::LockDir::new(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels.lock.d"),
        ),
        |lock, (family, specs)| lock.family(*family, specs.iter().map(|spec| &spec.name)),
    );
    let family_specs: Vec<_> = families.into_iter().flat_map(|(_, specs)| specs).collect();
    built.extend(
        kernel_build::build_in_families(
            &toolchain,
            &family_specs,
            &out.join("families"),
            &lock,
            BuildOptions::from_env(),
            Some(&context),
        )
        .expect("build audited kernel-lab family cubins"),
    );
    // Every built cubin by name, for `Session::module`.
    let mut table = String::from("static CUBINS: &[(&str, &[u8])] = &[\n");
    for cubin in built {
        let file = cubin.path.strip_prefix(&out).expect("cubin under OUT_DIR");
        table.push_str(&format!(
            "({:?}, include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{}\"))),\n",
            cubin.name,
            file.to_str().expect("UTF-8 cubin filename")
        ));
    }
    table.push_str("];\n");
    // Every built cubin's entry symbols by name, for `Session::module`.
    table.push_str("static CUBIN_ENTRIES: &[(&str, &[&str])] = &[\n");
    for spec in specs.iter().chain(&hc_specs).chain(&family_specs) {
        table.push_str(&format!("({:?}, &[", spec.name));
        for entry in &spec.entries {
            table.push_str(&format!("{:?}, ", entry.symbol));
        }
        table.push_str("]),\n");
    }
    table.push_str("];\n");
    std::fs::write(out.join("cubins.rs"), table).expect("write the cubin table");
}
