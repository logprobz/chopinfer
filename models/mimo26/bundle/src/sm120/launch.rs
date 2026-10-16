//! The catalog entries of each SM120 engine family, and the launch
//! adapters the reference builders plan with (test code).
#[cfg(test)]
pub use model_mimo26_contract::sm120::workspace::tight_page_stride;
pub use model_mimo26_contract::sm120::workspace::workspace_bytes;

pub mod attention {
    #[cfg(test)]
    use kernel_core::contract::{Refusal, RuntimeCounts};
    pub use kernels_sm120::attention::Variant;
    #[cfg(test)]
    pub use kernels_sm120::attention::{Buffers, PlannedAttention, RopeInputs};

    pub use model_mimo26_contract::sm120::workspace::attention::mimo_variants;

    #[cfg(test)]
    pub fn plan(
        variant: Variant,
        counts: RuntimeCounts,
        buffers: Buffers,
    ) -> Result<PlannedAttention, Refusal> {
        if !model_mimo26_contract::sm120::PROFILES.contains(&variant.profile) {
            return Err(Refusal("attention entry is not a MiMo instantiation"));
        }
        variant.plan(counts, buffers)
    }

    #[cfg(test)]
    pub fn plan_for_bucket(
        variant: Variant,
        prepared_batch: u32,
        counts: RuntimeCounts,
        buffers: Buffers,
    ) -> Result<PlannedAttention, Refusal> {
        if !model_mimo26_contract::sm120::PROFILES.contains(&variant.profile) {
            return Err(Refusal("attention entry is not a MiMo instantiation"));
        }
        variant.plan_for_bucket(prepared_batch, counts, buffers)
    }

    #[cfg(test)]
    pub fn plan_folded(
        variant: Variant,
        prepared_batch: u32,
        counts: RuntimeCounts,
        buffers: Buffers,
        rope: RopeInputs,
    ) -> Result<PlannedAttention, Refusal> {
        if !model_mimo26_contract::sm120::PROFILES.contains(&variant.profile) {
            return Err(Refusal("attention entry is not a MiMo instantiation"));
        }
        variant.plan_folded(prepared_batch, counts, buffers, rope)
    }

    /// The partial or merge launch of a planned attention. A direct-output
    /// partial writes the final rows; its merge has nothing to fold, so it
    /// is not launched.
    #[cfg(test)]
    pub fn stage(
        planned: PlannedAttention,
        bindings: crate::dispatch::BufferBindings,
        rows: u32,
        partial: bool,
    ) -> Option<crate::pipeline::PreparedLaunch> {
        let (symbol, plan, arguments) = if partial {
            (
                planned.partial_symbol,
                planned.partial,
                planned.partial_args,
            )
        } else if !planned.direct_output {
            (planned.merge_symbol, planned.merge, planned.merge_args)
        } else {
            return None;
        };
        Some(crate::pipeline::PreparedLaunch {
            symbol,
            rows,
            plan,
            arguments,
            bindings,
        })
    }
}

pub mod row {
    #[cfg(test)]
    use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, RuntimeCounts};
    pub use kernels_sm120::row::RowInstantiation;

    pub use model_mimo26_contract::sm120::workspace::row::entries_for;
    #[cfg(test)]
    pub use model_mimo26_contract::sm120::workspace::row::mimo_entries;

    #[cfg(test)]
    pub fn plan(
        entry: &RowInstantiation,
        capacity: u32,
        counts: RuntimeCounts,
        arguments: Vec<Argument>,
    ) -> Result<(LaunchPlan, ArgumentList), Refusal> {
        if !mimo_entries()
            .iter()
            .any(|item| item.symbol == entry.symbol)
        {
            return Err(Refusal("row entry is not a MiMo instantiation"));
        }
        Ok((
            entry.plan(capacity, counts)?,
            entry.pack_arguments(arguments)?,
        ))
    }
}

pub mod dense {
    #[cfg(test)]
    pub use model_mimo26_contract::sm120::workspace::dense::entries_for;
    pub use model_mimo26_contract::sm120::workspace::dense::mimo_entries;
}

pub mod moe {
    #[cfg(test)]
    use kernel_core::contract::{ArgumentList, LaunchPlan, Refusal, RuntimeCounts};
    #[cfg(test)]
    pub use kernels_sm120::Profile;
    #[cfg(test)]
    pub use kernels_sm120::moe::{InputPath, Phase, Projection, Variant};

    pub use model_mimo26_contract::sm120::workspace::moe::mimo_variants;

    #[cfg(test)]
    pub fn plan(
        variant: &Variant,
        counts: RuntimeCounts,
        handles: [u64; 15],
        weight_buffer: u64,
    ) -> Result<Vec<(String, LaunchPlan, ArgumentList)>, Refusal> {
        if !model_mimo26_contract::sm120::PROFILES.contains(&variant.profile) {
            return Err(Refusal("MoE entry is not a MiMo instantiation"));
        }
        let weight_map = variant.weight_tensor_map(weight_buffer)?;
        let mut launches = vec![(
            variant.symbol.clone(),
            variant.plan(counts)?,
            variant.arguments(handles, weight_map)?,
        )];
        if let Some(reduction) = variant.reduction_plan(counts)? {
            launches.push((
                variant.reduction_symbol().expect("split reduction symbol"),
                reduction,
                variant
                    .reduction_arguments(handles)?
                    .expect("split reduction arguments"),
            ));
        }
        Ok(launches)
    }
}

/// MiMo's collectives, which its SM120 selection owns.
pub use crate::sm120::comm;

#[cfg(test)]
mod tests {
    use kernel_build::{EntryParam, EntrySpec};
    use kernel_core::contract::{
        ActivationFormat, Argument, RuntimeCounts, StaticQuery, TensorMap, TensorMapElement,
        TensorMapInterleave, TensorMapL2Promotion, TensorMapOobFill, TensorMapSwizzle,
        WeightFormat,
    };

    use super::{attention, comm, dense, moe, row};

    fn planned(direct_output: bool) -> attention::PlannedAttention {
        let plan = |grid| kernel_core::contract::LaunchPlan {
            grid: [grid, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: false,
        };
        attention::PlannedAttention {
            partial: plan(2),
            merge: plan(3),
            partial_args: kernel_core::contract::ArgumentList(vec![Argument::U32(2)]),
            merge_args: kernel_core::contract::ArgumentList(vec![Argument::U32(3)]),
            split_capacity: 1,
            partial_symbol: "partial".into(),
            merge_symbol: "merge".into(),
            direct_output,
            writes_cache: false,
        }
    }

    #[test]
    fn attention_stage_selects_the_stage_and_skips_a_direct_output_merge() {
        use crate::dispatch::BufferBindings;
        for direct_output in [false, true] {
            let launch =
                attention::stage(planned(direct_output), BufferBindings::default(), 5, true)
                    .unwrap();
            assert_eq!(
                (launch.symbol.as_str(), launch.rows, launch.plan.grid[0]),
                ("partial", 5, 2)
            );
            assert_eq!(launch.arguments.0, [Argument::U32(2)]);
        }
        let launch = attention::stage(planned(false), BufferBindings::default(), 7, false).unwrap();
        assert_eq!(
            (launch.symbol.as_str(), launch.rows, launch.plan.grid[0]),
            ("merge", 7, 3)
        );
        assert_eq!(launch.arguments.0, [Argument::U32(3)]);
        assert!(attention::stage(planned(true), BufferBindings::default(), 7, false).is_none());
    }

    #[test]
    fn bf16_attention_cache_map_translates_to_host_stride_layout() {
        let variant = attention::mimo_variants(crate::topology::Tp::TP4)
            .iter()
            .find(|variant| {
                variant.layout == kernel_core::contract::AttentionLayout::Paged
                    && variant.query.kv_dtype == kernel_core::contract::DType::Bf16
            })
            .unwrap();
        let handle = runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1);
        let mut bindings = crate::dispatch::BufferBindings::default();
        let id = bindings.bind(runtime_bundle_abi::DeviceBufferSpanV1::new(
            handle,
            256,
            32768 * 192 * 64 * 2,
        ));
        let (mut k_map, _) = variant.cache_maps(32768, id, id).unwrap();
        k_map.base_buffer = id;
        let args = crate::dispatch::arguments(
            &kernel_core::contract::ArgumentList(vec![Argument::TensorMap(k_map)]),
            &bindings,
        )
        .expect("BF16 attention cache map");
        let runtime_bundle_abi::KernelArgumentV1::TensorMap(map) = args[0] else {
            panic!("tensor map")
        };
        assert_eq!(map.global_dims[0], 192);
        assert_eq!(map.global_strides_bytes[0], 384);
        assert_eq!(map.global_strides_bytes[1], 384);
    }

    #[test]
    fn rank_workspace_covers_all_engine_planners() {
        for tp in crate::topology::Tp::SUPPORTED {
            let narrow =
                super::workspace_bytes(tp, crate::contract::ITEMS_32).expect("planner workspace");
            assert!(narrow > 0);
            // Wider buckets plan wider attention; never less than 32 items.
            let wide =
                super::workspace_bytes(tp, bundle_runtime::geometry::Items::new(256).unwrap())
                    .expect("wide planner workspace");
            assert!(wide >= narrow);
        }
    }

    fn map(
        buffer: u64,
        k: u32,
        rows: u32,
        tile_k: u16,
        tile_rows: u16,
        element: TensorMapElement,
    ) -> TensorMap {
        TensorMap {
            base_buffer: buffer,
            dims: vec![u64::from(k), u64::from(rows)],
            strides_bytes: vec![1, u64::from(k)],
            box_dims: vec![u32::from(tile_k), u32::from(tile_rows)],
            element_strides: vec![1, 1],
            swizzle: TensorMapSwizzle::B128,
            interleave: TensorMapInterleave::None,
            l2_promotion: TensorMapL2Promotion::B256,
            oob_fill: TensorMapOobFill::Zero,
            element,
        }
    }

    fn assert_entry_layout(entry: &EntrySpec, args: &[Argument]) {
        assert_eq!(entry.params.len(), args.len(), "{}", entry.symbol);
        for (param, arg) in entry.params.iter().zip(args) {
            assert!(
                matches!(
                    (param, arg),
                    (
                        EntryParam::DevicePointer { .. },
                        Argument::DeviceBuffer(_) | Argument::NullDevicePointer
                    ) | (EntryParam::TensorMap { .. }, Argument::TensorMap(_))
                        | (EntryParam::U32 { .. }, Argument::U32(_))
                        | (EntryParam::U64 { .. }, Argument::U64(_))
                        | (EntryParam::I32 { .. }, Argument::I32(_))
                        | (
                            EntryParam::PointerArray { .. },
                            Argument::DevicePointerArray(_)
                        )
                ),
                "{}",
                entry.symbol
            );
        }
    }

    #[test]
    fn mimo_row_launches_pack_the_declared_abi_kinds() {
        let entries = row::mimo_entries();
        assert!(entries.len() > 60);
        for entry in entries {
            let values = entry
                .params
                .iter()
                .map(|param| match param {
                    EntryParam::DevicePointer { .. } => Argument::DeviceBuffer(1),
                    EntryParam::U64 { .. } => Argument::U64(1),
                    EntryParam::U32 { .. } => Argument::U32(1),
                    EntryParam::I32 { .. } => Argument::I32(1),
                    EntryParam::TensorMap { .. } | EntryParam::PointerArray { .. } => {
                        panic!("MiMo row entry uses an unhandled ABI kind")
                    }
                })
                .collect();
            let (plan, args) = row::plan(
                entry,
                32,
                RuntimeCounts {
                    m: 1,
                    batch: 1,
                    context: 1,
                    routed_rows: 8,
                    split_k: 1,
                },
                values,
            )
            .expect("row launch and arguments");
            assert_eq!(args.0.len(), entry.params.len(), "{}", entry.symbol);
            assert!(plan.pdl, "{}", entry.symbol);
        }
    }

    #[test]
    fn mimo_attention_arguments_match_every_partial_and_merge_entry() {
        for variant in crate::topology::Tp::SUPPORTED
            .into_iter()
            .flat_map(|tp| attention::mimo_variants(tp).iter())
        {
            let slots = if variant.layout == kernel_core::contract::AttentionLayout::Paged {
                32_768
            } else {
                32
            };
            let (k_map, v_map) = variant.cache_maps(slots, 1, 2).expect("cache maps");
            let buffers = attention::Buffers {
                q: 3,
                k_cache: 1,
                v_cache: 2,
                k_page_stride_bytes: super::tight_page_stride(variant, variant.query.qk_dim),
                v_page_stride_bytes: super::tight_page_stride(variant, variant.query.v_dim),
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
                sink: Some(13),
                output: 14,
                workspace: 15,
                k_scale: 1.0,
                v_scale: 1.0,
            };
            let counts = RuntimeCounts {
                m: if variant.query.phase == kernel_core::contract::AttentionPhase::Prefill {
                    variant.query_capacity
                } else {
                    1
                },
                batch: 1,
                context: 1,
                routed_rows: 8,
                split_k: 1,
            };
            let folded = variant.rope_fold.map(|_| {
                attention::plan_folded(
                    *variant,
                    32,
                    counts,
                    buffers.clone(),
                    attention::RopeInputs {
                        qkv: 16,
                        theta: crate::row_plan::SWA_ROPE_THETA,
                        v_multiplier: crate::row_plan::V_MULTIPLIER,
                    },
                )
                .expect("folded attention launch")
            });
            let planned = attention::plan(*variant, counts, buffers).expect("attention launch");
            let cubin = variant
                .cubin(std::path::Path::new("cuda"))
                .expect("attention cubin");
            let partial = cubin
                .entries
                .iter()
                .find(|entry| entry.symbol == planned.partial_symbol)
                .expect("planned attention partial entry");
            assert_entry_layout(partial, &planned.partial_args.0);
            assert_entry_layout(&cubin.entries[1], &planned.merge_args.0);
            assert!(planned.partial.pdl && planned.merge.pdl);
            if let Some(folded) = folded {
                let entry = cubin
                    .entries
                    .iter()
                    .find(|entry| entry.symbol == folded.partial_symbol)
                    .expect("folded attention entry");
                assert_entry_layout(entry, &folded.partial_args.0);
                assert!(folded.partial.pdl && folded.writes_cache);
                assert!(
                    crate::CUBINS
                        .iter()
                        .any(|cubin| cubin.symbols.contains(&folded.partial_symbol.as_str()))
                );
            }
        }
    }

    #[test]
    fn mimo_attention_catalog_covers_both_phases_and_kv_formats() {
        for tp in crate::topology::Tp::SUPPORTED {
            let entries = attention::mimo_variants(tp);
            assert_eq!(entries.len(), 10);
            assert!(
                entries
                    .iter()
                    .all(|entry| entry.profile == model_mimo26_contract::sm120::profile(tp))
            );
        }
        for entry in crate::topology::Tp::SUPPORTED
            .into_iter()
            .flat_map(|tp| attention::mimo_variants(tp).iter())
        {
            entry.validate().expect("MiMo attention contract");
            let cubin = entry
                .cubin(std::path::Path::new("cuda"))
                .expect("cubin spec");
            // Prefill and windowed decode write their output directly; verify
            // partials are merged.
            let partial = if entry.query.phase == kernel_core::contract::AttentionPhase::Prefill
                || entry.query.phase == kernel_core::contract::AttentionPhase::Decode
                    && entry.layout == kernel_core::contract::AttentionLayout::Ring
            {
                "_direct"
            } else {
                "_partial"
            };
            for part in [partial, "_merge"] {
                assert!(
                    cubin.entries.iter().any(|e| e.symbol.contains(part)),
                    "missing {part} entry"
                );
            }
        }
    }

    #[test]
    fn mimo_dense_arguments_match_every_compiled_entry() {
        let entries = dense::mimo_entries().collect::<Vec<_>>();
        assert!(entries.len() >= 40);
        assert_eq!(
            entries
                .iter()
                .filter(|entry| entry.symbol.contains("mimo_mlp_gate_up"))
                .inspect(|entry| {
                    let StaticQuery::Gemm(query) = entry.key.query else {
                        panic!("GEMM query")
                    };
                    assert_eq!(
                        query.epilogues.swiglu,
                        Some(kernel_core::contract::SwigluColumns::ConcatenatedHalves)
                    );
                })
                .count(),
            7,
            "six decode buckets and prefill must use fused gate/up SwiGLU"
        );
        let specs = entries
            .iter()
            .map(|&entry| entry.clone())
            .collect::<Vec<_>>();
        let cubin = kernels_sm120::gemm::cubin_spec(&specs).expect("dense cubin spec");
        for entry in entries {
            let StaticQuery::Gemm(query) = entry.key.query else {
                panic!("GEMM query")
            };
            let decode = query.m_capacity <= 32;
            let weight_element = if query.weight == WeightFormat::Bf16 {
                TensorMapElement::Bf16
            } else {
                TensorMapElement::Fp8E4M3
            };
            let activation_element = if query.activation == ActivationFormat::Bf16 {
                TensorMapElement::Bf16
            } else {
                TensorMapElement::Fp8E4M3
            };
            let mut weight = map(
                1,
                query.k,
                query.n,
                entry.key.knobs.tile_k,
                if decode {
                    entry.key.knobs.tile_m
                } else {
                    entry.key.knobs.tile_n
                },
                weight_element,
            );
            if kernels_sm120::gemm::fragment_weight_enabled(&entry.key)
                && weight_element == TensorMapElement::Bf16
            {
                weight.swizzle = TensorMapSwizzle::None;
            }
            let mut activation = map(
                2,
                query.k,
                query.m_capacity,
                entry.key.knobs.tile_k,
                if decode {
                    entry.key.knobs.tile_n
                } else {
                    entry.key.knobs.tile_m
                },
                activation_element,
            );
            if (query.activation == ActivationFormat::Bf16 && entry.key.knobs.tile_k > 64)
                || (matches!(
                    query.activation,
                    ActivationFormat::Fp8Group128(_) | ActivationFormat::Mxfp8
                ) && entry.key.knobs.tile_k > 128)
            {
                activation.swizzle = TensorMapSwizzle::None;
            }
            let maps = if decode {
                [weight, activation]
            } else {
                [activation, weight]
            };
            let counts = RuntimeCounts {
                m: query.m_capacity,
                batch: 1,
                context: 1,
                routed_rows: 8,
                split_k: kernels_sm120::gemm::required_split_k(&entry.key).unwrap_or(1),
            };
            let buffers = kernels_sm120::gemm::GemmBuffers {
                output: 3,
                partial: 4,
                semaphores: 5,
                weight_scales: (query.weight != WeightFormat::Bf16).then_some(6),
                activation_scales: (query.activation != ActivationFormat::Bf16).then_some(7),
                residual: None,
                weight_global_scale: None,
                activation_global_scale: None,
            };
            let plan = kernels_sm120::gemm::plan(&entry.key, counts).expect("MiMo dense plan");
            let args = kernels_sm120::gemm::arguments(&entry.key, counts, maps, buffers)
                .expect("MiMo dense arguments");
            assert!(plan.pdl, "{}", entry.symbol);
            if kernels_sm120::gemm::fragment_weight_enabled(&entry.key) {
                assert!(matches!(
                    args.0[if decode { 0 } else { 1 }],
                    Argument::DeviceBuffer(_)
                ));
                if decode {
                    assert!(matches!(args.0[1], Argument::DeviceBuffer(_)));
                } else {
                    assert!(matches!(args.0[0], Argument::TensorMap(_)));
                }
            }
            let compiled = cubin
                .entries
                .iter()
                .find(|item| item.symbol == entry.symbol)
                .expect("compiled symbol");
            assert_entry_layout(compiled, &args.0);
        }
    }

    #[test]
    fn mimo_moe_arguments_match_compute_and_reduction_entries() {
        let variants = moe::mimo_variants(crate::topology::Tp::TP4);
        assert!(variants.len() >= 28);
        let cubins = model_mimo26_contract::sm120::PROFILES
            .into_iter()
            .filter(|profile| profile.ranks().is_some())
            .flat_map(|profile| {
                kernels_sm120::moe::cubin_specs(
                    model_mimo26_contract::sm120::catalogs(),
                    Some(profile),
                )
            })
            .collect::<Vec<_>>();
        {
            let tp = crate::topology::Tp::TP2;
            // Five decode buckets and prefill per projection, and the exact
            // entries of gate/up at 2, 4 and 8 rows.
            assert_eq!(moe::mimo_variants(tp).len(), 15);
        }
        for variant in crate::topology::Tp::SUPPORTED
            .into_iter()
            .flat_map(|tp| moe::mimo_variants(tp).iter())
        {
            let counts = RuntimeCounts {
                m: 1,
                batch: 1,
                context: 1,
                routed_rows: 8,
                split_k: variant.knobs.split_k_capacity,
            };
            let launches = moe::plan(variant, counts, [1; 15], 16).expect("MiMo MoE launches");
            for (symbol, plan, args) in launches {
                assert!(plan.pdl, "{symbol}");
                let compiled = cubins
                    .iter()
                    .flat_map(|cubin| &cubin.entries)
                    .find(|item| item.symbol == symbol)
                    .expect("compiled symbol");
                assert_entry_layout(compiled, &args.0);
            }
        }
    }

    #[test]
    fn decode_v2_residency_covers_both_launch_shapes() {
        for tp in crate::topology::Tp::SUPPORTED {
            let ranks = tp.ranks();
            let entries = comm::spinning_entries_for(tp, 32, None).unwrap();
            assert!(entries.iter().all(|entry| entry.plan.cluster[0] > 1));
            for rows in 1..=kernels_sm120::comm::DECODE_MAX_ROWS {
                assert_eq!(
                    comm::push_symbols(ranks, rows),
                    (
                        kernels_sm120::comm::DECODE_SYMBOL_V2,
                        kernels_sm120::comm::DECODE_COMBINE_SYMBOL,
                    )
                );
            }
            assert_eq!(
                comm::push_symbols(ranks, 13),
                (
                    "chop_comm_allreduce_oneshot_norm_bf16_v2",
                    "chop_comm_allreduce_combine_norm_bf16_v1",
                )
            );
            let entries: Vec<_> = entries
                .iter()
                .filter(|entry| entry.symbol == kernels_sm120::comm::DECODE_SYMBOL_V2)
                .collect();
            assert_eq!(entries.len(), 2);
            for entry in entries {
                let rows = if entry.plan.cluster[0] == 2 { 2 } else { 12 };
                let expected = kernels_sm120::comm::decode_plan(ranks, rows).unwrap();
                // Equal grids may come from fewer rows and epoch cells.
                // Residency depends on the launch shape, not workspace bytes.
                assert_eq!(entry.plan.grid, expected.grid);
                assert_eq!(entry.plan.block, expected.block);
                assert_eq!(entry.plan.cluster, expected.cluster);
                assert_eq!(
                    entry.plan.dynamic_shared_bytes,
                    expected.dynamic_shared_bytes
                );
                assert!(!entry.wide);
            }
        }
    }

    #[test]
    fn production_hidden_collectives_refuse_mismatched_ranks() {
        let call = || comm::OneShotCall {
            input: 1,
            output: 2,
            peer_inboxes: vec![3; 2],
            peer_flags: vec![4; 2],
            epoch: 5,
            elements: 4096,
            rank: 0,
            ranks: 2,
        };
        let residual = || comm::ResidualBuffers {
            input: 6,
            output: 7,
        };
        assert!(
            comm::fused_hidden_for(
                crate::topology::Tp::TP4,
                1,
                0,
                comm::HiddenRoute::Prefill,
                call(),
                vec![8; 2],
                residual(),
                9,
                1e-6
            )
            .is_err()
        );
        assert!(
            comm::fused_combine_hidden_for(
                crate::topology::Tp::TP4,
                1,
                0,
                comm::HiddenRoute::ExactRows,
                call(),
                residual(),
                9,
                1e-6,
                10
            )
            .is_err()
        );
    }

    #[test]
    fn legacy_hidden_collectives_keep_eight_rank_transport() {
        comm::exact_symbols(8, 32).unwrap();
        let launches = comm::fused_hidden(
            64,
            0,
            comm::HiddenRoute::Prefill,
            comm::OneShotCall {
                input: 1,
                output: 2,
                peer_inboxes: (3..11).collect(),
                peer_flags: (11..19).collect(),
                epoch: 19,
                elements: 64 * 4096,
                rank: 0,
                ranks: 8,
            },
            (20..28).collect(),
            comm::ResidualBuffers {
                input: 28,
                output: 29,
            },
            30,
            1e-6,
        )
        .unwrap();
        assert_eq!(
            launches
                .iter()
                .map(|(symbol, _, _)| symbol.as_str())
                .collect::<Vec<_>>(),
            comm::PHASED_SYMBOLS
        );
    }

    #[test]
    fn mimo_comm_stages_match_every_selected_cubin_abi() {
        let cubin = kernels_sm120::comm::cubin_spec();
        let decode_cubin = kernels_sm120::comm::decode_cubin_spec_v2();
        let verify_cubin = model_mimo26_contract::sm120::comm::verify_cubin_spec();
        for rows in [1, 2, 4, 8, 16, 32, 64, 256, 8192] {
            let elements = u64::from(rows) * 4096;
            for graph in [false, true] {
                let launches = comm::fused_hidden(
                    rows,
                    0,
                    if graph {
                        comm::HiddenRoute::Graph { wide: true }
                    } else {
                        comm::HiddenRoute::Prefill
                    },
                    comm::OneShotCall {
                        input: 1,
                        output: 2,
                        peer_inboxes: vec![3; 4],
                        peer_flags: vec![4; 4],
                        epoch: 5,
                        elements,
                        rank: 0,
                        ranks: 4,
                    },
                    vec![6; 4],
                    comm::ResidualBuffers {
                        input: 7,
                        output: 8,
                    },
                    9,
                    1e-6,
                )
                .expect("fused hidden stages");
                for (symbol, plan, _) in &launches {
                    if symbol == kernels_sm120::comm::DECODE_SYMBOL_V2 {
                        assert_eq!(*plan, kernels_sm120::comm::decode_plan(4, rows).unwrap());
                    }
                }
                let wide = graph && rows > 32 && rows <= 256;
                assert_eq!(launches.len(), if rows <= 32 || wide { 1 } else { 2 });
                if launches.len() == 2 {
                    // The scatter reads the residual; its gather takes none.
                    use kernel_core::contract::Argument;
                    let symbols = [launches[0].0.as_str(), launches[1].0.as_str()];
                    assert_eq!(symbols, comm::PHASED_SYMBOLS);
                    assert!(launches[0].2.0.contains(&Argument::DeviceBuffer(7)));
                    assert!(!launches[1].2.0.contains(&Argument::DeviceBuffer(7)));
                }
                // Prefill rows of the push path take the exact entries.
                if !graph && rows <= 32 {
                    assert_eq!(
                        launches[0].0,
                        comm::exact_symbols(4, rows).unwrap().0,
                        "{rows}"
                    );
                }
                for (symbol, plan, args) in launches {
                    assert!(plan.pdl, "{symbol}");
                    let entry = cubin
                        .entries
                        .iter()
                        .chain(&verify_cubin.entries)
                        .chain(&decode_cubin.entries)
                        .find(|entry| entry.symbol == symbol)
                        .expect("comm symbol");
                    assert_entry_layout(entry, &args.0);
                }
            }
            // Graphs fold the combine only up to 32 rows.
            assert_eq!(
                comm::graph_folds_combine(crate::contract::RANKS, rows),
                rows <= 32,
                "{rows}"
            );
            let combine = |route| {
                comm::fused_combine_hidden(
                    rows,
                    0,
                    route,
                    comm::OneShotCall {
                        input: 1,
                        output: 2,
                        peer_inboxes: vec![3; 4],
                        peer_flags: vec![4; 4],
                        epoch: 5,
                        elements,
                        rank: 0,
                        ranks: 4,
                    },
                    comm::ResidualBuffers {
                        input: 7,
                        output: 8,
                    },
                    9,
                    1e-6,
                    10,
                )
            };
            if let Ok((symbol, plan, args)) = combine(comm::HiddenRoute::Graph { wide: true }) {
                assert!(plan.pdl, "{symbol}");
                assert_ne!(symbol, kernels_sm120::comm::DECODE_SYMBOL_V2);
                let entry = cubin
                    .entries
                    .iter()
                    .chain(&decode_cubin.entries)
                    .find(|entry| entry.symbol == symbol)
                    .expect("comm combine symbol");
                assert_entry_layout(entry, &args.0);
            } else {
                assert!(rows > 32, "{rows}");
            }
            // Exact rows take the verify combine entry up to its 32 rows at
            // TP4, and the decode step's entry at low-latency rows.
            match combine(comm::HiddenRoute::ExactRows) {
                Ok((symbol, plan, args)) => {
                    assert!(rows <= 32 && plan.pdl, "{rows}");
                    assert_eq!(
                        symbol,
                        if rows <= 2 {
                            comm::push_symbols(4, rows).1
                        } else {
                            kernels_sm120::comm::VERIFY_COMBINE_SYMBOL
                        }
                    );
                    let entry = cubin
                        .entries
                        .iter()
                        .chain(&verify_cubin.entries)
                        .chain(&decode_cubin.entries)
                        .find(|entry| entry.symbol == symbol)
                        .expect("comm verify combine symbol");
                    assert_entry_layout(entry, &args.0);
                }
                Err(_) => assert!(rows > 16, "{rows}"),
            }
            for dtype in [
                kernel_core::contract::DType::Bf16,
                kernel_core::contract::DType::F32,
            ] {
                let (symbol, plan, args) = comm::gather_logits_rows(
                    dtype,
                    kernels_sm120::comm::GatherRowsCall {
                        input: 1,
                        outputs: vec![2; 4],
                        flags: vec![3; 4],
                        epoch: 4,
                        row_indices: Some(5),
                        rows,
                        shard: u64::from(crate::contract::VOCAB_PER_RANK),
                        rank: 0,
                        ranks: crate::contract::RANKS,
                        target: 0,
                    },
                )
                .expect("batched vocabulary row gather");
                assert_eq!(plan.grid, [8, 1, 1]);
                assert!(plan.pdl);
                let entry = cubin
                    .entries
                    .iter()
                    .find(|entry| entry.symbol == symbol)
                    .expect("batched gather symbol");
                assert_entry_layout(entry, &args.0);
            }
        }
    }
}
