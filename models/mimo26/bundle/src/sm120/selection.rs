//! The catalog's entry symbols by family and stem, and the entries MiMo's
//! stages select that are not prepared ops: the row engine entries of the
//! embedding and head-row gathers, RoPE and the ring write, and the
//! attention variants, by stage, phase and global KV format.

use kernel_core::contract::{DType, Refusal};

use crate::program::Stage;
use crate::selection::{MimoPhase as _, Phase};
use crate::sm120::launch;
use crate::switches::Switches;
use crate::topology::{Tp, row_infix};

/// Construct a catalog symbol from its family and complete stem tokens.
pub fn entry_symbol(tp: Tp, family: &str, stem: &str) -> String {
    let g = tp.geometry;
    let pinned = model_mimo26_contract::sm120::geometry::pinned_names(*g);
    if family == "comm" {
        return if pinned {
            format!("chop_comm_{stem}_v1")
        } else {
            format!("chop_comm_{}_{stem}_v1", g.kernel_tag)
        };
    }
    let stem = match stem {
        "rms_hidden" => format!("rms_{}_hf{}", g.kernel_tag, g.hidden()),
        "router" => format!("router_{}_sigmoid{}", g.kernel_tag, g.experts()),
        "combine_hidden" => format!("combine_{}{}", g.kernel_tag, g.hidden()),
        "embedding_full" if !pinned => format!("{}_embedding_full", g.kernel_tag),
        _ if stem.starts_with("dflash_") => {
            let suffix = stem.strip_prefix("dflash_").unwrap();
            let stem = format!("dflash{}_{suffix}", row_infix(tp));
            if pinned {
                stem
            } else {
                format!("{}_{stem}", g.kernel_tag)
            }
        }
        _ => stem
            .split('_')
            .map(|token| match token {
                "mimo" => g.kernel_tag.to_owned(),
                _ => token.to_owned(),
            })
            .collect::<Vec<_>>()
            .join("_"),
    };
    format!("chop_{family}_{stem}_v1")
}

/// The row entry `name` of a drafter.
pub fn drafter_row_symbol(tp: Tp, name: &str) -> String {
    entry_symbol(tp, "row", name)
}

pub enum Selected {
    MediaInject,
    Row(launch::row::RowInstantiation),
    Attention(launch::attention::Variant),
    LogitsGather,
    RingBackup,
}

pub(crate) fn row_for(tp: Tp, name: &str) -> Result<Selected, Refusal> {
    let symbol = entry_symbol(tp, "row", name);
    launch::row::entries_for(tp)
        .find(|entry| entry.symbol == symbol)
        .cloned()
        .map(Selected::Row)
        .ok_or(Refusal("MiMo row entry missing"))
}

/// The entry of a stage that is not a declared projection; projections run
/// the authenticated precision component's bindings.
pub fn select(
    stage: Stage,
    phase: Phase,
    ga_kv: DType,
    tp: Tp,
    switches: Switches,
) -> Result<Selected, Refusal> {
    let row = |name: &str| row_for(tp, name);
    let infix = row_infix(tp);
    if stage
        .layer()
        .is_some_and(|layer| layer >= tp.geometry.layers())
    {
        return Err(Refusal("MiMo layer index out of range"));
    }
    phase.capacity()?;
    if !matches!(ga_kv, DType::Bf16 | DType::Fp8E4M3) {
        return Err(Refusal("unsupported MiMo global KV format"));
    }
    let route = match stage {
        Stage::MediaInject => Ok(Selected::MediaInject),
        Stage::RingBackup => Ok(Selected::RingBackup),
        Stage::GatherHeadRows => row("embedding_full"),
        // The backend serves the initial norm, the hidden all-reduces, the
        // MoE routes and the combine (`crate::ops`).
        Stage::InputNorm
        | Stage::AttentionReduceNorm(_)
        | Stage::MlpReduceNorm(_)
        | Stage::FinalReduceNorm
        | Stage::RouterTop8(_)
        | Stage::PermuteCount(_)
        | Stage::PermuteScan(_)
        | Stage::PermuteScatter(_)
        | Stage::CombineExperts(_) => Err(Refusal("MiMo prepares this stage's op through its ops")),
        // Row stages, folded decode and DFlash share the context-sized HF tables.
        Stage::RopeAndPagedWrite(layer) if tp.geometry.is_global(layer) => row(&format!(
            "rope_mimo{infix}_ga_{}_table_p64_pos32",
            if ga_kv == DType::Bf16 { "bf16" } else { "fp8" }
        )),
        Stage::RopeForSlidingWindow(layer) if !tp.geometry.is_global(layer) => row(&format!(
            "rope_mimo{infix}_swa_{}_table_p32_pos32",
            phase.label()
        )),
        Stage::SlidingWindowWrite(layer) if !tp.geometry.is_global(layer) => {
            row(&format!("ring_write_mimo{infix}_bf16_pos32"))
        }
        Stage::AttentionPartial(layer) | Stage::AttentionMerge(layer) => {
            let layout = if tp.geometry.is_global(layer) {
                kernel_core::contract::AttentionLayout::Paged
            } else {
                kernel_core::contract::AttentionLayout::Ring
            };
            launch::attention::mimo_variants(tp)
                .iter()
                .find(|variant| {
                    variant.layout == layout
                        && variant.query.phase == phase.attention(switches)
                        && variant.query.kv_dtype
                            == if layout == kernel_core::contract::AttentionLayout::Paged {
                                ga_kv
                            } else {
                                DType::Bf16
                            }
                })
                .cloned()
                .map(Selected::Attention)
                .ok_or(Refusal("MiMo attention entry missing"))
        }
        Stage::GatherLogits => Ok(Selected::LogitsGather),
        _ => Err(Refusal("stage does not match MiMo layer geometry")),
    }?;
    Ok(route)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_rope_selects_tables_for_target_rows() {
        for phase in [Phase::Decode(16), Phase::Verify(128), Phase::Prefill] {
            for dtype in [DType::Bf16, DType::Fp8E4M3] {
                for stage in [Stage::RopeAndPagedWrite(0), Stage::RopeForSlidingWindow(1)] {
                    let Selected::Row(row) =
                        select(stage, phase, dtype, Tp::TP4, Switches::configured()).unwrap()
                    else {
                        panic!("row");
                    };
                    assert!(row.symbol.contains("_table_"));
                }
            }
        }
    }

    #[test]
    fn every_stage_selects_a_compiled_entry_for_both_phases_and_ga_formats() {
        for phase in [
            Phase::Decode(1),
            Phase::Decode(2),
            Phase::Decode(4),
            Phase::Decode(8),
            Phase::Decode(16),
            Phase::Decode(32),
            Phase::Verify(1),
            Phase::Verify(3),
            Phase::Verify(8),
            Phase::Verify(13),
            Phase::Verify(32),
            Phase::Prefill,
        ] {
            for (format, tp) in [DType::Bf16, DType::Fp8E4M3]
                .into_iter()
                .flat_map(|format| Tp::SUPPORTED.map(|tp| (format, tp)))
            {
                for stage in crate::program::stages() {
                    // Declared projections run the precision component's
                    // bindings, whose stages fake_host_tests::precision checks,
                    // and the initial norm and hidden all-reduces their
                    // prepared ops, whose entries `ops::tests` checks.
                    if stage.projection() || stage.prepared() {
                        assert!(select(stage, phase, format, tp, Switches::configured()).is_err());
                        continue;
                    }
                    match select(stage, phase, format, tp, Switches::configured()).unwrap() {
                        Selected::MediaInject => {
                            assert_eq!(stage, Stage::MediaInject);
                        }
                        Selected::Row(entry) => assert!(
                            crate::CUBINS
                                .iter()
                                .any(|cubin| { cubin.symbols.contains(&entry.symbol.as_str()) })
                        ),
                        Selected::Attention(entry) => {
                            let partial = entry.instantiation().unwrap().symbol;
                            let merge = format!("chop_attn_{}_merge_v1", entry.name);
                            assert!(crate::CUBINS.iter().any(|cubin| {
                                cubin.symbols.contains(&partial.as_str())
                                    && cubin.symbols.contains(&merge.as_str())
                            }));
                        }
                        Selected::RingBackup => assert!(crate::CUBINS.iter().any(|cubin| {
                            cubin.symbols.contains(
                                &crate::sm120::ring_entries::ring_backup_symbol(tp, 32).as_str(),
                            )
                        })),
                        Selected::LogitsGather => assert!(crate::CUBINS.iter().any(|cubin| {
                            cubin.symbols.contains(&"chop_comm_allgather_f32_v1")
                                && cubin.symbols.contains(&"chop_comm_allgather_bf16_v1")
                        })),
                    }
                }
            }
        }
        assert!(
            select(
                Stage::AttentionPartial(48),
                Phase::Decode(1),
                DType::Bf16,
                Tp::TP4,
                Switches::configured()
            )
            .is_err()
        );
    }

    /// Each topology's attention comes from its own profile at its own
    /// per-rank shapes.
    #[test]
    fn other_topologies_select_their_own_shapes() {
        let tp = Tp::TP2;
        let Selected::Attention(attention) = select(
            Stage::AttentionPartial(5),
            Phase::Decode(4),
            DType::Bf16,
            tp,
            Switches::configured(),
        )
        .unwrap() else {
            panic!("attention");
        };
        assert_eq!(attention.profile, model_mimo26_contract::sm120::profile(tp));
        assert_eq!(u64::from(attention.query.q_heads), tp.q_heads());
    }

    /// A step routes its experts as prefill does exactly where the
    /// contract's expert selection takes the prefill entries.
    #[test]
    fn prefill_routes_follow_the_expert_selection() {
        use kernel_core::precision::ExecutionPhase;
        let buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256];
        let phases = buckets
            .iter()
            .flat_map(|&rows| [Phase::Decode(rows), Phase::Verify(rows)])
            .chain([Phase::Prefill]);
        for phase in phases {
            let Ok(capacity) = phase.capacity() else {
                continue;
            };
            let execution = match phase {
                Phase::Prefill => ExecutionPhase::Prefill,
                _ => ExecutionPhase::Decode,
            };
            assert_eq!(
                phase.prefill_route(),
                model_mimo26_contract::serving::prefill_route(execution, capacity),
                "{phase:?}"
            );
        }
    }
}
