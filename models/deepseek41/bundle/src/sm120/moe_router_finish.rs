//! Bind the measured exact router epilogue beside the cached partial producer.
use super::moe_direct_kernels::R2_VARIANT;
use crate::ops::moe::direct::{Call, Point, Pointer, Source};
use kernel_core::backend::{Launch, Refused};
use kernel_core::contract::{Argument, ArgumentList};
use kernel_core::precision::ExecutionPhase;
use kernels_sm120::moe::chained::{floor::Entry, router_finish as kernel};
use model_deepseek41_contract::moe::Experts;

// TP4, 250 W, 2026-10-19, complete bitwise graphs and clean eager/graph
// memcheck. Backbone full-MoE gains across three routing patterns are
// 9.3..11.6% at Decode-1, 2.4..9.9% at Decode-8, 0.4..3.9% at Decode-32,
// and 3.0..8.7% at Verify-6. Drafter Decode-1 gains 6.6..8.6%.
// Backbone Verify-8/32 also win, but require existing direct admission;
// this selector never enables a previously unqualified direct bucket.
// Drafter 8/32 has mixed results; Verify-6's smallest gain is only 0.2%.
// Retain those paths. No unmeasured row counts are inferred.
// TP4, 250 W, 2026-10-21, captured full MoE graphs (row prefixes of C8 K3
// verify captures): at backbone Verify-4 (DSpark C1 K3) the fused router
// alone saves 0.24 ms per 40-layer step, 20 of 20 layers, bitwise.
pub(super) fn qualified(set: Experts, phase: ExecutionPhase, rows: u32) -> bool {
    #[cfg(test)]
    if DISABLED.get() {
        return false;
    }
    match (set.count, set.top_k, phase) {
        (384, 6, ExecutionPhase::Decode) => [1, 8, 32].contains(&rows),
        (384, 6, ExecutionPhase::Verify) => [4, 6, 8, 32].contains(&rows),
        (128, 3, ExecutionPhase::Decode) => rows == 1,
        _ => false,
    }
}

// TP4, 250 W, 2026-10-21, captured full MoE graphs, every case bitwise
// (outputs and all route metadata, eager and graphs, memcheck clean). The
// r2 finish ends 7-9 us sooner, before the shared expert, so it binds only
// with the gate joined after the shared expert
// (`shared_overlap::joins_before_gate`). Together they save 0.32 ms per
// 40-layer step at Decode-8, 0.25 at Decode-32 and 0.19 at Verify-32 (C8),
// faster in 20 of 20 layers each with no candidate sample above the served
// range; the gate never starts beside the shared expert. Alone the r2
// finish loses at these rows (the gate starts beside the shared expert's
// down). Row prefixes of the same captures (c1-router-r1): 0.39 ms per
// step at Decode-1 (AR C1) and 0.59 at Verify-4 (DSpark C1 K3, against the
// unfused router), 20 of 20 layers each, no sample above the served range.
// Verify-6/8 and the drafter bank are unmeasured.
// Served at Decode-8 and Decode-32 with a programmatic gate the pair lost
// (TP4 AR, 2026-10-21, GPU-0 exclusive time over 20 full-concurrency decode
// cycles): 0.35 ms per step at C8 and 1.0 ms at C32, because the routed gate
// started beside the shared down kernel in most layers. With the gate after
// the join launched as a full dependency (shared_overlap::Lanes::issue) no
// gate starts early, and the pair gains 0.25 ms per C8 cycle and 0.29 ms per
// C32 cycle (the finish saves 0.34 ms at C32; same profile, 2026-10-21).
pub(super) fn r2(set: Experts, phase: ExecutionPhase, rows: u32) -> bool {
    qualified(set, phase, rows)
        && matches!(
            (set.count, set.top_k, phase, rows),
            (384, 6, ExecutionPhase::Decode, 1 | 8 | 32) | (384, 6, ExecutionPhase::Verify, 4 | 32)
        )
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub partial_bytes: u64,
    pub arrival: u64,
    pub routed: u64,
}

pub(crate) fn layout(entry: &Entry, rows: u32) -> Layout {
    let partial_bytes = kernel::router_workspace_bytes(entry, rows);
    let arrival = partial_bytes.next_multiple_of(256);
    Layout {
        partial_bytes,
        arrival,
        routed: arrival + 256,
    }
}

pub(super) fn calls(e: &Entry, rows: u32, scale_bits: u32, r2: bool) -> Result<Vec<Call>, Refused> {
    let layout = layout(e, rows);
    let tickets = layout.routed + e.workspace_bytes(rows)?.next_multiple_of(256);
    kernel::STAGES
        .into_iter()
        .map(|stage| {
            let spec = if r2 && stage == kernel::Stage::Finish {
                kernel::r2_entry(e, R2_VARIANT, false)
            } else {
                kernel::entry(e, stage)
            };
            let plan = kernel::plan(e, stage, rows)?;
            let pointers = spec
                .params
                .iter()
                .enumerate()
                .filter_map(|(index, param)| {
                    let kernel_build::EntryParam::DevicePointer { name, .. } = param else {
                        return None;
                    };
                    let (source, bytes, alignment) = match name.as_str() {
                        "workspace" => (Source::WorkspaceAt(0), layout.partial_bytes, 256),
                        "arrived" => (Source::WorkspaceAt(layout.arrival), 4, 4),
                        "scratch" => (
                            Source::WorkspaceAt(layout.routed),
                            e.workspace_bytes(rows).unwrap(),
                            256,
                        ),
                        "gate_tickets" => (Source::WorkspaceAt(tickets), 256, 4),
                        "input" => (
                            Source::View("wide".into()),
                            u64::from(rows) * u64::from(e.hidden()) * 4,
                            4,
                        ),
                        role => (
                            Source::View(role.into()),
                            match role {
                                "live" | "padded" => 4,
                                "bias" | "alternate_bias" | "counts" => u64::from(e.experts()) * 4,
                                "offsets" | "active" => (u64::from(e.experts()) + 1) * 4,
                                "logits" | "scores" => u64::from(rows) * u64::from(e.experts()) * 4,
                                "bias_select" => u64::from(rows),
                                "ids" | "weights" => u64::from(rows) * u64::from(e.top_k()) * 4,
                                "sorted" => u64::from(e.experts()) * 64 * 4,
                                _ => unreachable!("unknown router finish role {role}"),
                            },
                            1,
                        ),
                    };
                    Some(Pointer {
                        index,
                        source,
                        bytes,
                        alignment,
                    })
                })
                .collect();
            let arguments = ArgumentList(
                spec.params
                    .iter()
                    .map(|param| match param {
                        kernel_build::EntryParam::U32 { name } if name == "rows" => {
                            Argument::U32(rows)
                        }
                        kernel_build::EntryParam::U32 { name } if name == "scale_bits" => {
                            Argument::U32(scale_bits)
                        }
                        _ => Argument::NullDevicePointer,
                    })
                    .collect(),
            );
            Ok(Call {
                point: if stage == kernel::Stage::Prepare {
                    Point::RouterPrepare
                } else {
                    Point::Route
                },
                pointers,
                launch: Launch {
                    symbol: spec.symbol,
                    rows,
                    plan,
                    arguments,
                    collective: false,
                    persistent: false,
                },
            })
        })
        .collect()
}

#[cfg(test)]
thread_local! { static DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

#[cfg(test)]
pub(crate) fn without_fusion<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DISABLED.set(self.0);
        }
    }
    let _restore = Restore(DISABLED.replace(true));
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::{geometry::FLASH, moe};
    #[test]
    fn router_finish_admits_only_measured_winners_inside_direct_buckets() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = crate::shape::Shape::new(&FLASH, ranks).unwrap();
            for set in moe::expert_sets(&FLASH) {
                let ops = super::super::moe_direct::prepare_set(&shape, set).unwrap();
                for (phase, rows, op) in ops {
                    let wanted = match (set.count, phase) {
                        (384, ExecutionPhase::Decode) => [1, 8, 32].contains(&rows),
                        (384, ExecutionPhase::Verify) => [4, 6, 8, 32].contains(&rows),
                        (128, ExecutionPhase::Decode) => rows == 1,
                        _ => false,
                    };
                    assert_eq!(op.fuses_router(), wanted);
                    // The r2 finish only where it was measured with the
                    // gate joined after the shared expert.
                    let r2 = set.count == 384
                        && matches!(
                            (phase, rows),
                            (ExecutionPhase::Decode, 1 | 8 | 32) | (ExecutionPhase::Verify, 4 | 32)
                        );
                    let finishes = op
                        .info
                        .symbols
                        .iter()
                        .filter(|(s, _)| s.contains("_router_finish_") && !s.contains("_prepare_"))
                        .map(|(s, _)| s.as_str())
                        .collect::<Vec<_>>();
                    let expected = if r2 {
                        vec!["chop_deepseek41_moe_router_finish_e384_k6_finish_r2v7_v1"]
                    } else if wanted {
                        vec![if set.count == 384 {
                            "chop_deepseek41_moe_router_finish_e384_k6_finish_v1"
                        } else {
                            "chop_deepseek41_moe_router_finish_e128_k3_finish_v1"
                        }]
                    } else {
                        Vec::new()
                    };
                    assert_eq!(finishes, expected, "{phase:?} {rows}");
                    assert!(
                        finishes
                            .iter()
                            .all(|s| crate::shared_overlap::joins_before_gate(s) == r2)
                    );
                }
                assert_eq!(qualified(set, ExecutionPhase::Verify, 4), set.count == 384);
                assert!(!qualified(set, ExecutionPhase::Decode, 4));
                assert!(!qualified(set, ExecutionPhase::Prefill, 4));
                for rows in [0, 2, 3, 5, 7, 12, 16, 24, 48, 64, 192, 512, 8192] {
                    for phase in [
                        ExecutionPhase::Decode,
                        ExecutionPhase::Verify,
                        ExecutionPhase::Prefill,
                    ] {
                        assert!(!qualified(set, phase, rows));
                    }
                }
            }
            for cubin in [
                "router_finish_moe_deepseek41_sm120",
                "router_finish_moe_deepseek41_r2_sm120",
            ] {
                assert_eq!(crate::Kernels::loads(cubin, &shape), ranks == Ranks::TP4);
            }
        }
    }
}
