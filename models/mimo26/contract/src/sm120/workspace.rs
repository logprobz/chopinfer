//! Shared workspace planning and model catalog selection.

use crate::serving::{DECODE_ENTRY_ROWS, PREFILL_TOKENS};

use kernel_core::contract::{AttentionPhase, Refusal, RmsNormEpilogue, RuntimeCounts};

/// The page stride of a tightly packed single-layer K or V block. Ring layouts
/// ignore it; paged serving pools pass their interleaved page stride instead.
pub fn tight_page_stride(variant: &attention::Variant, dim: u32) -> u64 {
    let bytes = if variant.query.kv_dtype == kernel_core::contract::DType::Bf16 {
        2
    } else {
        1
    };
    u64::from(variant.query.page_size) * u64::from(variant.query.kv_heads) * u64::from(dim) * bytes
}

/// Maximum reusable scratch allocation needed by a prepared MiMo rank at
/// `tp` serving `items`: over its own attention and expert entries and the
/// dense and row entries it can select (TP4's and the rank-invariant ones,
/// then its own). Above 32 items the attention entries also plan every
/// wider bucket up to the largest.
pub fn workspace_bytes(
    tp: crate::topology::Tp,
    items: bundle_runtime::geometry::Items,
) -> Result<u64, Refusal> {
    let mut maximum = 0;
    for variant in attention::mimo_variants(tp) {
        let slots = if variant.layout == kernels_sm120::attention::CacheLayout::Paged {
            32_768
        } else {
            crate::serving::DECODE_ENTRY_ROWS
        };
        let (k_map, v_map) = variant.cache_maps(slots, 1, 2)?;
        let buffers = attention::Buffers {
            q: 3,
            k_cache: 1,
            v_cache: 2,
            k_page_stride_bytes: tight_page_stride(variant, variant.query.qk_dim),
            v_page_stride_bytes: tight_page_stride(variant, variant.query.v_dim),
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
            m: if variant.query.phase == AttentionPhase::Prefill {
                variant.query_capacity
            } else {
                1
            },
            batch: 1,
            context: if variant.query.phase == AttentionPhase::Prefill {
                variant.query.max_context_capacity
            } else {
                1
            },
            routed_rows: 8,
            split_k: 1,
        };
        maximum = maximum.max(
            variant
                .plan(counts, buffers.clone())?
                .partial
                .workspace_bytes,
        );
        // Exact verify rows: one decode step's workspace per token row, over
        // every verify token row when wide buckets run them
        // (`serving::VERIFY_ATTENTION`).
        if variant.has_rows() {
            let rows = if crate::serving::verify_attention_rows() {
                crate::serving::VERIFY_TOKENS
            } else {
                crate::serving::EXACT_VERIFY_ROWS
            };
            maximum = maximum.max(
                variant
                    .plan_rows(rows, counts, buffers.clone())?
                    .partial
                    .workspace_bytes,
            );
        }
        if items.capacity() > variant.query.max_batch_capacity {
            let wide = variant.with_batch_capacity(items.capacity())?;
            for bucket in items
                .buckets()
                .filter(|&bucket| bucket > variant.query.max_batch_capacity)
            {
                let counts = RuntimeCounts {
                    m: if wide.query.phase == AttentionPhase::Verify {
                        (bucket * wide.query_capacity)
                            .min(kernels_sm120::attention::MAX_VERIFY_TOKENS)
                    } else {
                        counts.m
                    },
                    batch: bucket,
                    ..counts
                };
                maximum = maximum.max(
                    wide.plan_for_bucket(bucket, counts, buffers.clone())?
                        .partial
                        .workspace_bytes,
                );
            }
        }
    }
    for entry in dense::entries_for(tp) {
        maximum = maximum.max(kernels_sm120::gemm::capacity_workspace_bytes(&entry.key)?);
    }
    for variant in moe::mimo_variants(tp) {
        let counts = RuntimeCounts {
            m: variant.tokens_capacity,
            batch: 1,
            context: 1,
            routed_rows: variant.query.m_capacity,
            split_k: variant.knobs.split_k_capacity,
        };
        maximum = maximum.max(variant.plan(counts)?.workspace_bytes);
    }
    for entry in row::entries_for(tp) {
        let counts = RuntimeCounts {
            m: PREFILL_TOKENS,
            batch: 1,
            context: 1,
            routed_rows: PREFILL_TOKENS * tp.geometry.top_k(),
            split_k: 1,
        };
        maximum = maximum.max(entry.plan(PREFILL_TOKENS, counts)?.workspace_bytes);
    }
    for rows in [DECODE_ENTRY_ROWS, PREFILL_TOKENS] {
        maximum = maximum.max(
            kernels_sm120::comm::plan(
                &kernel_core::contract::CommQuery::AllReduce {
                    ranks: tp.ranks(),
                    elements: u64::from(rows) * tp.geometry.hidden() as u64,
                    dtype: kernel_core::contract::DType::Bf16,
                    fused_residual: true,
                    fused_rmsnorm: Some(RmsNormEpilogue {
                        kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                        width: tp.geometry.hidden(),
                        eps: tp.geometry.eps() as f32,
                    }),
                },
                rows,
            )?
            .workspace_bytes,
        );
    }
    maximum = maximum.max(
        kernels_sm120::comm::plan(
            &kernel_core::contract::CommQuery::AllGather {
                ranks: tp.ranks(),
                elements_per_rank: u64::from(crate::serving::DECODE_ENTRY_ROWS)
                    * u64::from(crate::topology::vocab_per_rank(tp)),
                dtype: kernel_core::contract::DType::Bf16,
            },
            crate::serving::DECODE_ENTRY_ROWS,
        )?
        .workspace_bytes,
    );
    Ok(maximum)
}

/// The dense and row entries a topology selects, which MiMo's SM120
/// module owns.
pub use crate::sm120::entries::{dense, row};

pub mod attention {
    pub use kernels_sm120::attention::{Buffers, Variant};
    /// The compiled MiMo attention entries of `tp`'s per-rank heads, built
    /// once per process.
    pub fn mimo_variants(tp: crate::topology::Tp) -> &'static [Variant] {
        static VARIANTS: std::sync::OnceLock<kernels_sm120::ByProfile<Variant>> =
            std::sync::OnceLock::new();
        VARIANTS
            .get_or_init(|| {
                kernels_sm120::ByProfile::new(crate::sm120::attention::catalog(), |variant| {
                    variant.profile
                })
            })
            .of(crate::sm120::profile(tp))
    }
}

pub mod moe {
    use crate::serving::prefill_route;
    use kernel_core::contract::Refusal;
    use kernel_core::precision::ExecutionPhase;
    use kernels_sm120::moe::{InputPath, Phase, Projection, Variant};

    /// Which grouped expert entries an instance runs: exact entries for
    /// verify buckets of exact rows (`serving::VERIFY_EXACT`).
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct Selection {
        pub exact: bool,
    }

    impl Selection {
        /// The process's selection, from its environment.
        pub fn configured() -> Self {
            Self {
                exact: crate::serving::verify_exact_enabled(),
            }
        }
    }

    /// The grouped expert entry of `projection` a `tp` instance runs for a
    /// `phase` step of bucket `capacity` under `selection`: the prefill
    /// entry where the step routes as prefill does (`serving::prefill_route`),
    /// otherwise the bucket's decode entry, or for verify buckets of exact
    /// rows their exact entry where the bucket has one (a bucket without one
    /// already has the one-token reduction order).
    pub fn entry(
        tp: crate::topology::Tp,
        projection: Projection,
        phase: ExecutionPhase,
        capacity: u32,
        selection: Selection,
    ) -> Result<&'static Variant, Refusal> {
        use crate::serving::{EXACT_VERIFY_ROWS, PREFILL_TOKENS};
        let (moe_phase, tokens) = if prefill_route(phase, capacity) {
            (Phase::Prefill, PREFILL_TOKENS)
        } else {
            (Phase::Decode, capacity)
        };
        let find = |exact: bool| {
            mimo_variants(tp).iter().find(|variant| {
                variant.projection == projection
                    && variant.phase == moe_phase
                    && variant.tokens_capacity == tokens
                    && variant.path == InputPath::A16Gather
                    && variant.exact == exact
            })
        };
        let exact_rows =
            selection.exact && phase == ExecutionPhase::Verify && capacity <= EXACT_VERIFY_ROWS;
        exact_rows
            .then(|| find(true))
            .flatten()
            .or_else(|| find(false))
            .ok_or(Refusal("MiMo grouped expert entry missing"))
    }

    /// The compiled MiMo grouped-expert entries of `tp`'s intermediate,
    /// built once per process.
    pub fn mimo_variants(tp: crate::topology::Tp) -> &'static [Variant] {
        static VARIANTS: std::sync::OnceLock<kernels_sm120::ByProfile<Variant>> =
            std::sync::OnceLock::new();
        VARIANTS
            .get_or_init(|| {
                kernels_sm120::ByProfile::new(crate::sm120::moe::catalog(), |variant| {
                    variant.profile
                })
            })
            .of(crate::sm120::profile(tp))
    }
}
