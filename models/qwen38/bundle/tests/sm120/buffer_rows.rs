//! Check geometry-owned allocation bytes and positional scratch consumption.
use bundle_qwen38::sm120::catalog;
use bundle_qwen38::{contract::*, prefix, scratch};
use model_qwen38_contract::buffers;

#[test]
fn shared_rows_preserve_every_geometry_allocation() {
    for shape in model_qwen38_contract::geometry::GEOMETRIES
        .iter()
        .flat_map(|g| {
            [Ranks::TP2, Ranks::TP4].map(|r| Shape::of_geometry(g, r).expect("served geometry"))
        })
    {
        for sequences in 1..=DECODE_BATCH {
            let served = scratch::Served::new(Items::new(sequences).unwrap());
            let comm = comm(shape, served);
            let page_bytes = shape.page_bytes_bf16_mtp();
            let overhead = overhead(123_456, &comm, page_bytes, shape);
            for (verify_rows, drafter) in
                [(0, false), (2, false), (4, false), (8, false), (8, true)]
            {
                let expected = scratch::sizes(
                    &comm,
                    overhead.workspace,
                    scratch::Speculation {
                        verify_rows,
                        drafter,
                    },
                    page_bytes,
                    shape,
                    served,
                );
                let actual = m0_rows(
                    shape,
                    buffers::Served {
                        slots: served.slots,
                        items: served.items,
                        rows: served.rows,
                    },
                    buffers::Speculation {
                        verify_rows,
                        drafter,
                    },
                    overhead,
                );
                let mut allocated = Vec::new();
                let scratch = scratch::RankScratch::allocate(
                    &comm,
                    overhead.workspace,
                    scratch::Speculation {
                        verify_rows,
                        drafter,
                    },
                    page_bytes,
                    shape,
                    served,
                    |bytes| {
                        let index = allocated.len() as u64;
                        let offset = 257 + index * 4099;
                        allocated.push(bytes);
                        Ok(runtime_bundle_abi::DeviceBufferSpanV1::new(
                            runtime_bundle_abi::BundleDeviceBufferHandleV1::from_raw(1001 + index),
                            offset,
                            bytes,
                        ))
                    },
                )
                .unwrap();
                assert_eq!(
                    allocated,
                    actual.iter().map(|(_, b)| *b).collect::<Vec<_>>()
                );
                assert_bindings(&scratch, &actual, false, shape);
                assert_eq!(
                    actual,
                    row_refs(&expected),
                    "TP{} C{sequences} verify={verify_rows} drafter={drafter}",
                    shape.ranks()
                );
            }
        }
    }
}

/// The comm state's bytes, from the SM120 comm catalog: the decode
/// all-reduce's grows for the wide entry above 32 served rows.
fn comm(shape: Shape, served: scratch::Served) -> scratch::CommBytes {
    use model_qwen38_contract::sm120::comm::decode;
    let owner = catalog::geometry_owner(shape);
    let hc =
        kernels_sm120::comm::hyper_connection::buffers(&owner.transport, catalog::hc_query(shape))
            .unwrap();
    let ll = decode::buffers_with(owner.decode.shape, shape.ranks()).unwrap();
    let decode = if served.rows > DECODE_ENTRY_ROWS {
        decode::wide_buffers_with(owner.decode.shape, shape.ranks()).unwrap()
    } else {
        ll
    };
    scratch::CommBytes {
        hc_inbox: hc.inbox_bytes,
        hc_flags: hc.flags_bytes,
        hc_epoch: hc.epoch_bytes,
        decode_inbox: decode.inbox_bytes,
        decode_flags: decode.flags_bytes,
        decode_epochs: decode.epoch_bytes,
        decode_zeroed_inbox: ll.zeroed_inbox_bytes,
        gather_flags: kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES,
        gather_epoch: kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES,
    }
}

/// The overhead rows of `comm` and a `workspace`-byte workspace.
fn overhead(
    workspace: u64,
    comm: &scratch::CommBytes,
    page_bytes: u64,
    shape: Shape,
) -> buffers::Overhead {
    buffers::Overhead {
        workspace,
        hc_inbox: comm.hc_inbox,
        hc_flags: comm.hc_flags,
        hc_epoch: comm.hc_epoch,
        hc_decode_inbox: comm.decode_inbox,
        hc_decode_flags: comm.decode_flags,
        hc_decode_epochs: comm.decode_epochs,
        gathers: shape.gather_sets() as u64 * (comm.gather_flags + comm.gather_epoch),
        prefix_pages: prefix::page_table_entries(page_bytes, shape) * 8,
    }
}

fn assert_bindings(
    scratch: &scratch::RankScratch,
    rows: &[(&str, u64)],
    overlap: bool,
    shape: Shape,
) {
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};
    let expected = |name: &str| {
        let index = rows.iter().position(|(n, _)| *n == name).unwrap();
        let physical = rows[index].1
            * if overlap && matches!(name, "x0" | "x1" | "xn" | "hc_inject" | "u" | "partial") {
                2
            } else {
                1
            };
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_raw(1001 + index as u64),
            257 + index as u64 * 4099,
            physical,
        )
    };
    let mut fields = vec![
        ("x0", scratch.x[0]),
        ("x1", scratch.x[1]),
        ("xn", scratch.xn),
        ("hc_down", scratch.hc_down),
        ("hc_bottleneck", scratch.hc_bottleneck),
        ("hc_inject", scratch.hc_inject),
        ("hc_up", scratch.hc_up),
        ("u", scratch.u),
        ("partial", scratch.partial),
        ("embed", scratch.embed),
        ("lin_qkv", scratch.lin_qkv),
        ("lin_z", scratch.lin_z),
        ("lin_ab", scratch.lin_ab),
        ("lin_conv", scratch.lin_conv),
        ("lin_core", scratch.lin_core),
        ("lin_normed", scratch.lin_normed),
        ("attn_qkv", scratch.attn_qkv),
        ("attn_index", scratch.attn_index),
        ("attn_q", scratch.attn_q),
        ("index_q", scratch.index_q),
        ("attn_out", scratch.attn_out),
        ("attn_gated", scratch.attn_gated),
        ("scores", scratch.scores),
        ("selected", scratch.selected),
        ("router_logits", scratch.router_logits),
        ("topk_ids", scratch.topk_ids),
        ("topk_weights", scratch.topk_weights),
        ("expert_counts", scratch.expert_counts),
        ("expert_offsets", scratch.expert_offsets),
        ("sorted_rows", scratch.sorted_rows),
        ("padded_rows", scratch.padded_rows),
        ("active_experts", scratch.active_experts),
        ("route_tiles", scratch.route_tiles),
        ("moe_intermediate", scratch.moe_intermediate),
        ("moe_routed", scratch.moe_routed),
        ("shared_intermediate", scratch.shared_intermediate),
        ("shared_out", scratch.shared_out),
        ("unused", scratch.unused),
        ("ngram_local", scratch.ngram_local),
        ("ngram_rows", scratch.ngram_rows),
        ("ngram", scratch.ngram),
        ("ple_value", scratch.ple_value),
        ("ple_key_normed", scratch.ple_key_normed),
        ("ple_gated", scratch.ple_gated),
        ("ple_conv", scratch.ple_conv),
        ("ple_prefill_in", scratch.ple_staging[0].input),
        ("ple_prefill_out", scratch.ple_staging[0].output),
        ("ple_decode_in", scratch.ple_staging[1].input),
        ("ple_decode_out", scratch.ple_staging[1].output),
        ("head_rows", scratch.head_rows),
        ("local_logits", scratch.local_logits),
        ("local_scored", scratch.local_scored),
        ("gathered_scored", scratch.gathered_scored),
        ("gathered_decode", scratch.gathered_decode),
        ("gemm_semaphores", scratch.gemm_semaphores),
        ("workspace", scratch.workspace),
        ("hc_inbox", scratch.hc_inbox),
        ("hc_flags", scratch.hc_flags),
        ("hc_epoch", scratch.hc_epoch),
        ("hc_decode_inbox", scratch.hc_decode_inbox),
        ("hc_decode_flags", scratch.hc_decode_flags),
        ("hc_decode_epochs", scratch.hc_decode_epochs),
        ("cos_sin", scratch.cos_sin),
        ("conv_pool", scratch.conv_pool),
        ("delta_pool", scratch.delta_pool),
        ("ple_pool", scratch.ple_pool),
        ("anchor_ple", scratch.anchor_ple),
        ("anchor_ring", scratch.anchor_ring),
        ("prefix_pages", scratch.prefix_pages),
        ("route_ticket", scratch.route_ticket),
    ];
    if let Some(draft) = scratch.drafter {
        fields.extend([
            ("retained", draft.retained),
            ("staged_prefill", draft.staged[0]),
            ("staged_decode", draft.staged[1]),
            ("drafts", draft.drafts),
            ("anchor_retained", draft.anchor_retained),
            ("best", draft.best),
            ("bests", draft.bests),
        ]);
    }
    assert_eq!(
        fields.len() + 1,
        rows.len(),
        "gathers is the only subdivided allocation"
    );
    for (name, span) in fields {
        let wanted = expected(name);
        assert_eq!(
            (span.buffer, span.offset, span.length),
            (wanted.buffer, wanted.offset, wanted.length),
            "{name}"
        );
    }
    assert!(
        scratch.precision_temporary.is_none(),
        "create allocates precision separately"
    );
    let gathers = expected("gathers");
    let flags = kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES;
    let epoch = kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES;
    assert_eq!(scratch.gathers.len(), shape.gather_sets());
    for (index, gather) in scratch.gathers.iter().enumerate() {
        let offset = gathers.offset + index as u64 * (flags + epoch);
        assert_eq!(
            (
                gather.flags.buffer,
                gather.flags.offset,
                gather.flags.length
            ),
            (gathers.buffer, offset, flags)
        );
        assert_eq!(
            (
                gather.epoch.buffer,
                gather.epoch.offset,
                gather.epoch.length
            ),
            (gathers.buffer, offset + flags, epoch)
        );
    }
    assert_eq!(gathers.length, shape.gather_sets() as u64 * (flags + epoch));
    let ll = model_qwen38_contract::sm120::comm::decode::buffers_with(
        catalog::geometry_owner(shape).decode.shape,
        shape.ranks(),
    )
    .unwrap()
    .zeroed_inbox_bytes;
    let mut zeroed = [
        "route_ticket",
        "gemm_semaphores",
        "hc_flags",
        "hc_epoch",
        "hc_decode_inbox",
        "hc_decode_flags",
        "hc_decode_epochs",
        "ple_pool",
    ]
    .map(expected)
    .to_vec();
    zeroed[4].length = ll;
    for index in 0..shape.gather_sets() {
        let offset = gathers.offset + index as u64 * (flags + epoch);
        zeroed.push(DeviceBufferSpanV1::new(gathers.buffer, offset, flags));
        zeroed.push(DeviceBufferSpanV1::new(
            gathers.buffer,
            offset + flags,
            epoch,
        ));
    }
    let span_rows = |spans: Vec<DeviceBufferSpanV1>| {
        spans
            .into_iter()
            .map(|span| (span.buffer, span.offset, span.length))
            .collect::<Vec<_>>()
    };
    let comm = comm(shape, scratch::Served::new(Items::new(1).unwrap()));
    assert_eq!(span_rows(scratch.zeroed(&comm)), span_rows(zeroed));
}

#[test]
fn overlap_backing_and_aliases_preserve_nonzero_base_offsets() {
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};
    for shape in [Shape::TP2, Shape::TP4] {
        for drafter in [false, true] {
            let served = scratch::Served::new(Items::new(33).unwrap());
            let speculation = scratch::Speculation {
                verify_rows: 8,
                drafter,
            };
            let rows = m0_sizes(
                123_456,
                speculation,
                shape.page_bytes_bf16_mtp(),
                shape,
                served,
            );
            let mut allocated = Vec::new();
            let scratch = scratch::RankScratch::allocate_with_overlap(
                &comm(shape, served),
                123_456,
                speculation,
                shape.page_bytes_bf16_mtp(),
                shape,
                served,
                true,
                1,
                |bytes| {
                    let index = allocated.len() as u64;
                    allocated.push(bytes);
                    Ok(DeviceBufferSpanV1::new(
                        BundleDeviceBufferHandleV1::from_raw(1001 + index),
                        257 + index * 4099,
                        bytes,
                    ))
                },
            )
            .unwrap();
            for (index, &(name, bytes)) in rows.iter().enumerate() {
                let doubled = matches!(name, "x0" | "x1" | "xn" | "hc_inject" | "u" | "partial");
                assert_eq!(
                    allocated[index],
                    bytes * if doubled { 2 } else { 1 },
                    "{name}"
                );
            }
            assert_bindings(&scratch, &rows, true, shape);
            let part = scratch.overlap_part(128, 1, None, shape);
            for (whole, alias) in scratch.x.into_iter().zip(part.x).chain([
                (scratch.xn, part.xn),
                (scratch.hc_inject, part.hc_inject),
                (scratch.u, part.u),
                (scratch.partial, part.partial),
            ]) {
                let skip = 128 * (whole.length / (2 * PREFILL_TOKENS as u64));
                assert_eq!(
                    (alias.buffer, alias.offset, alias.length),
                    (whole.buffer, whole.offset + skip, whole.length - skip)
                );
            }
            for (whole, alias) in [
                (scratch.ple_staging[0].input, part.ple_staging[0].input),
                (scratch.ple_staging[0].output, part.ple_staging[0].output),
            ] {
                let skip = shape.ple_entry_bytes();
                assert_eq!(
                    (alias.buffer, alias.offset, alias.length),
                    (whole.buffer, whole.offset + skip, whole.length - skip)
                );
            }
            let history =
                DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(999), 31, 100);
            let part = scratch.overlap_part(128, 1, Some(history), shape);
            assert_eq!(part.ple_staging[0].input.buffer, history.buffer);
            assert_eq!(part.ple_staging[0].input.offset, history.offset);
            assert_eq!(part.ple_staging[0].input.length, history.length);
            assert_eq!(part.workspace.buffer, scratch.workspace.buffer);
            assert_eq!(part.workspace.offset, scratch.workspace.offset);
            assert_eq!(part.workspace.length, scratch.workspace.length);
        }
    }
}

// Independent ordered allocation expectations retained from M0.
/// Every named allocation, in the bundle's allocation order.
fn m0_rows(
    shape: Shape,
    served: buffers::Served,
    speculation: buffers::Speculation,
    overhead: buffers::Overhead,
) -> Vec<(&'static str, u64)> {
    let p = u64::from(PREFILL_TOKENS);
    let b = u64::from(DRAFT_ITEMS);
    let wide = p * shape.wide() as u64 * BF16_BYTES;
    let hidden = p * shape.hidden() as u64 * BF16_BYTES;
    let routed = p * u64::from(shape.top_k());
    let sorted = u64::from(
        kernels_sm120::moe::padded_capacity(routed as u32, shape.local_experts, 16)
            .expect("MoE capacity"),
    );
    let local_experts = u64::from(shape.local_experts);
    let ple = u64::from(served.items) * shape.ple_entry_bytes();
    let rows = u64::from(served.rows);
    let mut buffers = vec![
        ("x0", wide),
        ("x1", wide),
        ("xn", wide),
        ("hc_down", p * shape.hc_rank() as u64 * BF16_BYTES),
        ("hc_bottleneck", p * shape.hc_rank() as u64 * BF16_BYTES),
        ("hc_inject", p * shape.streams() as u64 * BF16_BYTES),
        ("hc_up", wide),
        ("u", hidden),
        ("partial", hidden),
        ("embed", hidden),
        ("lin_qkv", p * shape.linear_qkv as u64 * BF16_BYTES),
        ("lin_z", p * shape.linear_value as u64 * BF16_BYTES),
        ("lin_ab", p * shape.linear_ab_stride as u64 * BF16_BYTES),
        ("lin_conv", p * shape.linear_qkv as u64 * BF16_BYTES),
        ("lin_core", p * shape.linear_value as u64 * BF16_BYTES),
        ("lin_normed", p * shape.linear_value as u64 * BF16_BYTES),
        ("attn_qkv", p * shape.attn_row as u64 * BF16_BYTES),
        ("attn_index", p * shape.index_row() as u64 * BF16_BYTES),
        ("attn_q", p * shape.attn_value as u64 * BF16_BYTES),
        (
            "index_q",
            p * (shape.index_heads() * shape.index_dim()) as u64 * BF16_BYTES,
        ),
        ("attn_out", p * shape.attn_value as u64 * BF16_BYTES),
        ("attn_gated", p * shape.attn_value as u64 * BF16_BYTES),
        (
            "scores",
            u64::from(SCORE_ROWS)
                * (shape.context_tokens() / u64::from(shape.qsa_block())).next_multiple_of(64)
                * 4,
        ),
        ("selected", p * shape.top_blocks() as u64 * 4),
        ("router_logits", p * shape.experts() as u64 * 4),
        ("topk_ids", routed * 4),
        ("topk_weights", routed * 4),
        ("expert_counts", local_experts * 4),
        ("expert_offsets", (local_experts + 1) * 4),
        ("sorted_rows", sorted * 4),
        ("padded_rows", 4),
        ("active_experts", (local_experts + 1) * 4),
        ("route_tiles", routed.div_ceil(256) * local_experts * 4 * 2),
        (
            "moe_intermediate",
            routed * shape.expert_intermediate() as u64 * BF16_BYTES,
        ),
        ("moe_routed", routed * shape.hidden() as u64 * BF16_BYTES),
        (
            "shared_intermediate",
            p * shape.shared_intermediate as u64 * BF16_BYTES,
        ),
        ("shared_out", hidden),
        ("unused", 256),
        ("ngram_local", p * shape.ngram_local as u64 * BF16_BYTES),
        ("ngram_rows", p * shape.ngram_local_heads as u64 * 8),
        (
            "ngram",
            p * u64::from(shape.hidden().max(shape.ple_embed_dim())) * BF16_BYTES,
        ),
        ("ple_value", hidden),
        ("ple_key_normed", wide),
        ("ple_gated", wide),
        ("ple_conv", wide),
        ("ple_prefill_in", ple),
        ("ple_prefill_out", ple),
        ("ple_decode_in", ple),
        ("ple_decode_out", ple),
        ("head_rows", hidden),
        // Decode rows past the decode heads' buckets, like prefill rows,
        // take `local_scored`; verify and drafter rows are at most 32.
        ("local_logits", b * shape.vocab_per_rank as u64 * 4),
        (
            "local_scored",
            SCORED_ROWS as u64 * shape.vocab_per_rank as u64 * 4,
        ),
        (
            "gathered_scored",
            SCORED_ROWS as u64 * shape.vocab() as u64 * 4,
        ),
        ("gathered_decode", rows * shape.vocab() as u64 * 4),
        ("gemm_semaphores", 4096),
        ("workspace", overhead.workspace),
        ("hc_inbox", overhead.hc_inbox),
        ("hc_flags", overhead.hc_flags),
        ("hc_epoch", overhead.hc_epoch),
        ("hc_decode_inbox", overhead.hc_decode_inbox),
        ("hc_decode_flags", overhead.hc_decode_flags),
        ("hc_decode_epochs", overhead.hc_decode_epochs),
        ("gathers", overhead.gathers),
        (
            "cos_sin",
            shape.context_tokens() * u64::from(shape.rotary() / 2) * 2 * 4,
        ),
    ];
    let pool = u64::from(served.slots)
        + if speculation.verify_rows > 1 {
            u64::from(served.slots.min(DRAFT_ITEMS)) * u64::from(speculation.verify_rows)
        } else {
            0
        }
        + u64::from(ANCHOR_ITEMS);
    let mut sizes = vec![
        (
            "conv_pool",
            shape.linear_layers() * pool * shape.conv_entry_bytes,
        ),
        (
            "delta_pool",
            shape.linear_layers() * pool * shape.delta_entry_bytes,
        ),
        (
            "ple_pool",
            (u64::from(served.slots)
                + if speculation.verify_rows > 0 {
                    u64::from(served.items) * u64::from(MAX_VERIFY_ROWS)
                } else {
                    0
                })
                * shape.ple_entry_bytes(),
        ),
        (
            "anchor_ple",
            u64::from(ANCHOR_ITEMS) * shape.ple_entry_bytes(),
        ),
        (
            "anchor_ring",
            shape
                .ring(shape.full_layers_mtp(), u64::from(ANCHOR_ITEMS))
                .bytes(),
        ),
    ];
    if speculation.drafter {
        sizes.extend([
            (
                "retained",
                u64::from(served.slots)
                    * u64::from(MAX_VERIFY_ROWS)
                    * (u64::from(shape.wide()) * BF16_BYTES),
            ),
            (
                "staged_prefill",
                u64::from(served.items) * (u64::from(shape.wide()) * BF16_BYTES),
            ),
            (
                "staged_decode",
                u64::from(served.rows) * (u64::from(shape.wide()) * BF16_BYTES),
            ),
            ("drafts", b * 4),
            (
                "anchor_retained",
                u64::from(ANCHOR_ITEMS)
                    * u64::from(MAX_VERIFY_ROWS)
                    * (u64::from(shape.wide()) * BF16_BYTES),
            ),
            ("best", b * 16),
            ("bests", b * 16 * u64::from(shape.ranks())),
        ]);
    }
    sizes.push(("prefix_pages", overhead.prefix_pages));
    sizes.push((
        "route_ticket",
        kernels_sm120::moe::parallel::PARALLEL_ROUTE_TICKET_BYTES,
    ));
    buffers.extend(sizes);
    buffers
}

fn m0_sizes(
    workspace: u64,
    speculation: scratch::Speculation,
    page_bytes: u64,
    shape: Shape,
    served: scratch::Served,
) -> Vec<(&'static str, u64)> {
    m0_rows(
        shape,
        buffers::Served {
            slots: served.slots,
            items: served.items,
            rows: served.rows,
        },
        buffers::Speculation {
            verify_rows: speculation.verify_rows,
            drafter: speculation.drafter,
        },
        overhead(workspace, &comm(shape, served), page_bytes, shape),
    )
}

#[test]
fn m0_rows_pin_production_declaration_order_and_bytes() {
    for shape in [Shape::TP2, Shape::TP4] {
        for sequences in [1, 32, 33, 128, 256] {
            let served = scratch::Served::new(Items::new(sequences).unwrap());
            for (verify_rows, drafter) in
                [(0, false), (2, false), (4, false), (8, false), (8, true)]
            {
                let speculation = scratch::Speculation {
                    verify_rows,
                    drafter,
                };
                let expected = m0_sizes(
                    123_456,
                    speculation,
                    shape.page_bytes_bf16_mtp(),
                    shape,
                    served,
                );
                let actual = scratch::sizes(
                    &comm(shape, served),
                    123_456,
                    speculation,
                    shape.page_bytes_bf16_mtp(),
                    shape,
                    served,
                );
                assert_eq!(row_refs(&actual), expected);
            }
        }
    }
}

fn row_refs<N: AsRef<str>>(rows: &[(N, u64)]) -> Vec<(&str, u64)> {
    rows.iter()
        .map(|(name, bytes)| (name.as_ref(), *bytes))
        .collect()
}
