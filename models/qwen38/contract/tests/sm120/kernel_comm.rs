#[cfg(test)]
mod prefill {
    use kernel_core::contract::{Argument, SigmoidMeanHyperConnectionQuery};
    use kernels_sm120::comm::{Epilogue, Input, input};
    use model_qwen38_contract::sm120::comm::*;

    fn call(ranks: usize, flags: usize, rank: u32, capacity: u32) -> Call {
        Call {
            input: 1,
            residual: 2,
            logits: 3,
            weight: 4,
            residual_out: 5,
            normed: 6,
            inboxes: vec![7; ranks],
            flags: vec![8; flags],
            epoch: 9,
            live_rows: 10,
            capacity,
            rank,
        }
    }

    #[test]
    fn buffers_hold_a_slot_per_rank_and_keep_four_ranks_sizes() {
        let tp4 = buffers(query(4)).unwrap();
        // The sizes before the entries took a rank count.
        let one_shot = 2 * 4 * 32 * 2560 * 2;
        let scatter = 4 * 2048 * 2560 * 2;
        let sums = 8192 * 2560 * 4;
        assert_eq!(tp4.inbox_bytes, one_shot + scatter + sums);
        assert_eq!((tp4.flags_bytes, tp4.epoch_bytes), (1408, 16));
        for (ranks, flags) in [(2, 848), (8, 2528)] {
            let sized = buffers(query(ranks)).unwrap();
            let ranks64 = u64::from(ranks);
            let scatter = ranks64 * 8192_u64.div_ceil(ranks64) * 2560 * 2;
            assert_eq!(
                sized.inbox_bytes,
                2 * ranks64 * 32 * 2560 * 2 + scatter + sums
            );
            // The arrival counter, word 2 (ranks + 2 ranks + 32), then 28
            // spare bytes and the gather's 32 slots per source.
            assert!((2 * (3 * ranks64 + 32) + 1) * 4 <= flags);
            assert_eq!(8 * (3 * ranks64 + 36 + 32 * ranks64), flags);
            assert_eq!(sized.flags_bytes, flags);
        }
        assert_eq!(standalone_inbox_bytes(8, 32), 2 * 8 * 32 * 2560 * 2);
        for ranks in [0, 1, 3, 16] {
            assert!(contract(query(ranks)).is_err(), "{ranks}");
        }
        assert_eq!(
            catalog()
                .iter()
                .map(|e| match e.query {
                    SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm { ranks, .. } => ranks,
                    _ => panic!("expected an all-reduce query"),
                })
                .collect::<Vec<_>>(),
            RANKS
        );
    }

    #[test]
    fn launches_pass_the_rank_count_after_the_rank() {
        for ranks in RANKS {
            let q = query(ranks);
            for (rows, symbols) in [
                (32, vec![SYMBOL]),
                (33, vec![SCATTER_SYMBOL, GATHER_SYMBOL]),
            ] {
                let launched =
                    launches(q, rows, call(ranks as usize, ranks as usize, 1, 8192)).unwrap();
                assert_eq!(
                    launched.iter().map(|l| l.symbol).collect::<Vec<_>>(),
                    symbols
                );
                for launch in launched {
                    let args = &launch.arguments.0;
                    assert_eq!(args[args.len() - 2], Argument::U32(1));
                    assert_eq!(args[args.len() - 1], Argument::U32(ranks));
                }
            }
        }
        // Slots that disagree with each other or with the query are refused.
        assert!(launches(query(4), 1, call(2, 2, 0, 8192)).is_err());
        assert!(launches(query(4), 1, call(4, 2, 0, 8192)).is_err());
        assert!(launches(query(2), 1, call(2, 2, 2, 8192)).is_err());
        assert!(standalone_arguments(call(3, 3, 0, 32)).is_err());
    }

    /// Shapes the transports refuse at compile time are refused here, and
    /// an F32 input doubles exactly the regions that hold input elements.
    #[test]
    fn shapes_size_their_input_regions() {
        let f32 = HcShape {
            input: input::F32,
            ..SHAPE
        };
        let wide = |width| HcShape {
            epilogue: Epilogue { width, ..EPILOGUE },
            ..SHAPE
        };
        assert!(
            SHAPE.validate().is_ok() && f32.validate().is_ok() && wide(5120).validate().is_ok()
        );
        assert!(wide(0).validate().is_err() && wide(2560 + 8).validate().is_err());
        let no_streams = HcShape {
            epilogue: Epilogue {
                streams: 0,
                ..EPILOGUE
            },
            ..SHAPE
        };
        assert!(no_streams.validate().is_err());
        for element_bytes in [1, 8, 16] {
            let other = HcShape {
                input: Input {
                    element_bytes,
                    ..input::F32
                },
                ..SHAPE
            };
            assert!(other.validate().is_err(), "{element_bytes}");
        }
        assert_eq!((SHAPE.per_vector(), f32.per_vector()), (8, 4));
        assert_eq!((SHAPE.vectors(), f32.vectors()), (320, 640));
        for ranks in RANKS {
            let bf16 = buffers_with(SHAPE, ranks, 8192);
            let doubled = buffers_with(f32, ranks, 8192);
            let ranks = u64::from(ranks);
            let input = 2 * ranks * 32 * 2560 * 2 + ranks * 8192_u64.div_ceil(ranks) * 2560 * 2;
            assert_eq!(doubled.inbox_bytes, bf16.inbox_bytes + input);
            assert_eq!(
                (doubled.flags_bytes, doubled.epoch_bytes),
                (bf16.flags_bytes, bf16.epoch_bytes)
            );
        }
    }
}

#[cfg(test)]
mod decode {
    use kernel_build::EntryParam;
    use kernel_core::contract::Argument;
    use model_qwen38_contract::sm120::comm::decode::*;
    use model_qwen38_contract::sm120::comm::{EPILOGUE, HcShape, RANKS, SHAPE};
    const SENDERS: u32 = 8;

    #[test]
    fn protocols_follow_the_prepared_rows() {
        let ll = plan(2).unwrap();
        assert_eq!((ll.grid, ll.cluster), ([8, 1, 1], [4, 1, 1]));
        let one = plan(4).unwrap();
        assert_eq!((one.grid, one.cluster), ([24, 1, 1], [8, 1, 1]));
        assert_eq!(plan(32).unwrap().grid, [136, 1, 1]);
        assert!(plan(3).is_err() && plan(33).is_err() && plan(0).is_err());
        // Four ranks keep their former sizes; a region slot per rank.
        let sizes = buffers(4).unwrap();
        assert_eq!(sizes.inbox_bytes, 2_621_440 + 1_310_720);
        assert_eq!(sizes.zeroed_inbox_bytes, 2_621_440);
        assert_eq!(sizes.flags_bytes, 32);
        for ranks in RANKS {
            let sized = buffers(ranks).unwrap();
            assert_eq!(sized.inbox_bytes * 4, sizes.inbox_bytes * u64::from(ranks));
            assert_eq!(
                sized.zeroed_inbox_bytes * 4,
                sizes.zeroed_inbox_bytes * u64::from(ranks)
            );
            assert_eq!(sized.flags_bytes, 8 * u64::from(ranks));
            assert_eq!(sized.epoch_bytes, sizes.epoch_bytes);
        }
        assert!(buffers(3).is_err() && buffers(1).is_err() && buffers(16).is_err());
    }

    #[test]
    fn each_rank_count_has_an_entry_with_that_many_pointer_slots() {
        let spec = cubin_specs(None).remove(0);
        assert_eq!(spec.entries.len(), 2 * RANKS.len());
        for (index, entry) in spec.entries.iter().enumerate() {
            let ranks = RANKS[index % RANKS.len()];
            let (symbol, flags) = if index < RANKS.len() {
                (symbol_v2(ranks).unwrap(), "")
            } else {
                (symbol_v3(ranks).unwrap(), ", true")
            };
            assert_eq!(entry.symbol, symbol);
            assert!(entry.body.contains(&format!(
                "allreduce_hc<{}, {}, {ranks}{flags}>(",
                EPILOGUE.functor(),
                SHAPE.input.policy
            )));
            let slots: Vec<_> = entry
                .params
                .iter()
                .filter_map(|p| match p {
                    EntryParam::PointerArray { slots, .. } => Some(u32::from(*slots)),
                    _ => None,
                })
                .collect();
            assert_eq!(slots, [ranks, ranks]);
        }
        for ranks in RANKS {
            assert_eq!(contract(ranks).unwrap().symbol, symbol_v2(ranks).unwrap());
        }
        assert_eq!(
            symbol_v2(4).unwrap(),
            "chop_comm_qwen38_allreduce_hc_decode_r4_v2"
        );
        assert!(symbol_v2(3).is_err() && symbol_v2(16).is_err());
        let tp4 = cubin_specs_for(None, &[4]).remove(0);
        assert_eq!(
            tp4.entries
                .iter()
                .map(|e| e.symbol.as_str())
                .collect::<Vec<_>>(),
            [symbol_v2(4).unwrap(), symbol_v3(4).unwrap()]
        );
    }

    /// The decode entries, which exact K3 verify steps launch, claim
    /// one-row rows up to `EXACT_VERIFY_ROWS`, the prepared rows of such a
    /// step plan (low latency up to two, the one-shot at four).
    #[test]
    fn decode_entries_declare_exact_verify_rows() {
        use kernels_sm120::invariance::{Invariance, declaration};
        let rows = model_qwen38_contract::sm120::config::EXACT_VERIFY_ROWS;
        for symbol in SYMBOLS_V2.into_iter().chain(SYMBOLS_V3) {
            assert_eq!(
                declaration(model_qwen38_contract::sm120::catalogs(), symbol).map(|d| d.invariance),
                Some(Invariance::one_row(rows)),
                "{symbol}"
            );
        }
        // A step of 1 to 4 rows takes the bucket of 1, 2 or 4 prepared rows.
        for live in 1..=rows {
            let prepared = live.next_power_of_two();
            assert!(prepared <= rows);
            let grid = plan(prepared).unwrap().grid[0];
            let ll = prepared <= LL_MAX_ROWS;
            assert_eq!(grid, if ll { prepared * 4 } else { 8 + prepared * 4 });
        }
    }

    #[test]
    fn arguments_carry_the_rank_count_of_the_pointer_arrays() {
        let call = |slots: usize, flags: usize, rank| Call {
            input: 1,
            residual: 2,
            logits: 3,
            weight: 4,
            residual_out: 5,
            normed: 6,
            inboxes: vec![7; slots],
            flags: vec![8; flags],
            epochs: 9,
            live_rows: 10,
            rows: 4,
            rank,
        };
        for ranks in RANKS {
            let args = arguments(call(ranks as usize, ranks as usize, ranks - 1)).unwrap();
            assert_eq!(args.0.len(), 13);
            assert_eq!(args.0[11], Argument::U32(ranks - 1));
            assert_eq!(args.0[12], Argument::U32(ranks));
        }
        assert!(arguments(call(3, 3, 0)).is_err());
        assert!(arguments(call(4, 2, 0)).is_err());
        assert!(arguments(call(2, 2, 2)).is_err());
    }
    #[test]
    fn wide_calls_extend_the_decode_state() {
        // The sending cluster and a worker per (row, stream) up to 128.
        for (rows, grid) in [(1, 16), (2, 16), (33, 136), (64, 136), (256, 136)] {
            let plan = wide_plan(rows).unwrap();
            assert_eq!((plan.grid, plan.cluster), ([grid, 1, 1], [8, 1, 1]));
            assert!(plan.pdl);
        }
        assert!(wide_plan(0).is_err() && wide_plan(257).is_err());
        // One grid per GPU is resident at every row count.
        for rows in 1..=WIDE_MAX_ROWS {
            kernels_sm120::comm::wide_fits_target_layout(&wide_plan(rows).unwrap()).unwrap();
        }
        for (ranks, rows, protocol) in [
            (2, 33, WideProtocol::OneShot),
            (2, 256, WideProtocol::OneShot),
            (4, 32, WideProtocol::OneShot),
            (4, 33, WideProtocol::TwoShot),
            (4, 256, WideProtocol::TwoShot),
            (8, 33, WideProtocol::TwoShot),
        ] {
            assert_eq!(wide_protocol(ranks, rows).unwrap(), protocol);
        }
        assert!(wide_protocol(3, 33).is_err() && wide_protocol(4, 257).is_err());
        // Graphs take the wide entry above 32 rows up to 256 at two ranks
        // and 128 at four and eight; the phased pair above.
        for (ranks, rows, wide) in [
            (2, 32, false),
            (2, 33, true),
            (2, 256, true),
            (4, 33, true),
            (4, 128, true),
            (4, 129, false),
            (8, 128, true),
            (8, 256, false),
        ] {
            assert_eq!(
                graph_takes_wide(ranks, rows).unwrap(),
                wide,
                "{ranks} {rows}"
            );
        }
        assert!(graph_takes_wide(3, 64).is_err());
        for ranks in RANKS {
            let decode = buffers(ranks).unwrap();
            let wide = wide_buffers(ranks).unwrap();
            let ranks = u64::from(ranks);
            // The decode regions, then 2 ranks banks of 256 BF16 rows, 256
            // BF16 rows of scatter slots and 256 F32 rows of gather slots.
            assert_eq!(
                wide.inbox_bytes,
                decode.inbox_bytes + (4 * ranks + 2 + 4) * 256 * 2560
            );
            assert_eq!(wide.zeroed_inbox_bytes, decode.zeroed_inbox_bytes);
            // kWideFlagWords in qwen38_decode_wide.cuh.
            assert_eq!(wide.flags_bytes, 153 * 4);
            assert_eq!(wide.epoch_bytes, decode.epoch_bytes);
        }
        assert_eq!(wide_buffers(4).unwrap().inbox_bytes, 18_350_080);
        let spec = wide_cubin_specs(None).remove(0);
        assert_eq!(spec.name, "comm_qwen38_decode_wide");
        for (entry, ranks) in spec.entries.iter().zip(RANKS) {
            assert_eq!(entry.symbol, wide_symbol(ranks).unwrap());
            assert!(entry.body.contains(&format!(
                "allreduce_hc_wide<{}, {}, {ranks}>(",
                EPILOGUE.functor(),
                SHAPE.input.policy
            )));
            assert_eq!(wide_contract(ranks).unwrap().symbol, entry.symbol);
        }
        assert!(
            spec.defines
                .contains(&("CHOP_QWEN38_WIDE_TWO_SHOT_ROWS_R2".into(), "257".into()))
        );
    }

    /// Another shape's decode and wide states: input regions follow its
    /// element size (the LL region too, kept when its layout does not fit),
    /// the all-gather sums stay F32, and its plans follow its streams.
    #[test]
    fn shapes_size_their_decode_states() {
        use kernels_sm120::comm::{Epilogue, input};
        let f32 = HcShape {
            input: input::F32,
            ..SHAPE
        };
        let f32_wide = HcShape {
            epilogue: Epilogue {
                width: 5120,
                ..EPILOGUE
            },
            input: input::F32,
        };
        assert!(SHAPE.low_latency_fits() && f32.low_latency_fits());
        assert!(!f32_wide.low_latency_fits());
        assert_eq!(low_latency_rows(f32_wide), 0);
        assert!(defines_with(f32_wide).contains(&("CHOP_QWEN38_LL_MAX_ROWS".into(), "0".into())));
        assert_eq!(defines_with(SHAPE), defines());
        assert_eq!(plan_with(f32, 2).unwrap().cluster, [4, 1, 1]);
        assert_eq!(plan_with(f32_wide, 2).unwrap().grid, [16, 1, 1]);
        // Without LL every row count takes the padded one-shot; with LL the
        // one-shot still needs whole clusters of groups.
        for rows in 1..=MAX_ROWS {
            let plan = plan_with(f32_wide, rows).unwrap();
            assert_eq!(plan.grid[0], SENDERS + (rows * 4).next_multiple_of(SENDERS));
        }
        assert!(plan_with(SHAPE, 3).is_err() && plan(3).is_err());
        assert_eq!(plan_with(SHAPE, 4).unwrap(), plan(4).unwrap());
        for ranks in RANKS {
            let (bf16, doubled) = (buffers(ranks).unwrap(), buffers_with(f32, ranks).unwrap());
            assert_eq!(doubled.inbox_bytes, 2 * bf16.inbox_bytes);
            assert_eq!(doubled.zeroed_inbox_bytes, 2 * bf16.zeroed_inbox_bytes);
            let (wide, wide_f32) = (
                wide_buffers(ranks).unwrap(),
                wide_buffers_with(f32, ranks).unwrap(),
            );
            let ranks = u64::from(ranks);
            // The one-shot banks and scatter slots double; the gather does not.
            assert_eq!(
                wide_f32.inbox_bytes - doubled.inbox_bytes,
                wide.inbox_bytes - bf16.inbox_bytes + (2 * ranks + 1) * 256 * 2560 * 2
            );
        }
    }
}

#[test]
fn hc_contracts_refuse_arithmetic_the_shipped_epilogue_does_not_implement() {
    use kernels_sm120::comm::{hyper_connection as hc, hyper_connection_decode as decode};
    use model_qwen38_contract::sm120::comm;

    for eps_bits in [1e-3_f32.to_bits(), 1e-5_f32.to_bits()] {
        let prefill = hc::HcTransport {
            eps_bits,
            ..comm::TRANSPORT
        };
        assert!(hc::contract(&prefill, prefill.query(4)).is_err());
        let decode = decode::HcDecodeTransport {
            eps_bits,
            ..comm::decode::TRANSPORT
        };
        assert!(decode::contract(&decode, 4).is_err());
        assert!(decode::wide_contract(&decode, 4).is_err());
    }
    for streams in [1, 2, 8] {
        let shape = hc::HcShape {
            epilogue: kernels_sm120::comm::Epilogue {
                streams,
                ..comm::EPILOGUE
            },
            ..comm::SHAPE
        };
        let prefill = hc::HcTransport {
            shape,
            ..comm::TRANSPORT
        };
        assert!(hc::contract(&prefill, prefill.query(4)).is_err());
        let decode = decode::HcDecodeTransport {
            shape,
            ..comm::decode::TRANSPORT
        };
        assert!(decode::contract(&decode, 4).is_err());
        assert!(decode::wide_contract(&decode, 4).is_err());
    }
}

#[test]
fn shipped_hc_contracts_keep_their_queries_symbols_and_compile_knobs() {
    use kernel_core::contract::{CompileKnobs, StaticQuery, Target};
    use model_qwen38_contract::sm120::comm;

    for ranks in comm::RANKS {
        for (spec, capacity, symbol) in [
            (
                comm::contract(comm::query(ranks)).unwrap(),
                8192,
                comm::SYMBOL,
            ),
            (
                comm::decode::contract(ranks).unwrap(),
                32,
                comm::decode::symbol_v2(ranks).unwrap(),
            ),
            (
                comm::decode::wide_contract(ranks).unwrap(),
                256,
                comm::decode::wide_symbol(ranks).unwrap(),
            ),
        ] {
            assert_eq!(spec.symbol, symbol);
            assert_eq!(spec.key.target, Target::Sm120f);
            assert_eq!(spec.key.dense, None);
            assert_eq!(
                spec.key.query,
                StaticQuery::HyperConnection(
                    kernel_core::contract::SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
                        ranks,
                        streams: 4,
                        width: 2560,
                        eps_bits: 1e-6_f32.to_bits(),
                        capacity,
                    }
                )
            );
            assert_eq!(
                spec.key.knobs,
                CompileKnobs {
                    tile_m: 1,
                    tile_n: 256,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: 8,
                    split_k_capacity: 1,
                }
            );
        }
    }
}

#[test]
fn custom_hc_epilogues_keep_their_own_arithmetic_requirements() {
    use kernels_sm120::comm::{
        Epilogue, hyper_connection as hc, hyper_connection_decode as decode,
    };
    use model_qwen38_contract::sm120::comm;

    let shape = hc::HcShape {
        epilogue: Epilogue {
            header: "tests/comm_epilogue_stub.cuh",
            template: "stub::Threads_<256>::Epilogue",
            width: 2560,
            streams: 2,
        },
        ..comm::SHAPE
    };
    let eps_bits = 1e-3_f32.to_bits();
    let prefill = hc::HcTransport {
        shape,
        eps_bits,
        ..comm::TRANSPORT
    };
    assert!(hc::contract(&prefill, prefill.query(4)).is_ok());
    let decode = decode::HcDecodeTransport {
        shape,
        eps_bits,
        ..comm::decode::TRANSPORT
    };
    assert!(decode::contract(&decode, 4).is_ok());
    assert!(decode::wide_contract(&decode, 4).is_ok());
}
