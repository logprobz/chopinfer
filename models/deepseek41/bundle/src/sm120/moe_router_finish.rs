//! Bind the measured exact router epilogue beside the cached partial producer.
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
pub(super) fn qualified(set: Experts, phase: ExecutionPhase, rows: u32) -> bool {
    #[cfg(test)]
    if DISABLED.get() {
        return false;
    }
    match (set.count, set.top_k, phase) {
        (384, 6, ExecutionPhase::Decode) => [1, 8, 32].contains(&rows),
        (384, 6, ExecutionPhase::Verify) => [6, 8, 32].contains(&rows),
        (128, 3, ExecutionPhase::Decode) => rows == 1,
        _ => false,
    }
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

pub(super) fn calls(e: &Entry, rows: u32, scale_bits: u32) -> Result<Vec<Call>, Refused> {
    let layout = layout(e, rows);
    let tickets = layout.routed + e.workspace_bytes(rows)?.next_multiple_of(256);
    kernel::STAGES
        .into_iter()
        .map(|stage| {
            let spec = kernel::entry(e, stage);
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
                        (384, ExecutionPhase::Verify) => [6, 8, 32].contains(&rows),
                        (128, ExecutionPhase::Decode) => rows == 1,
                        _ => false,
                    };
                    assert_eq!(op.fuses_router(), wanted);
                }
                for rows in [0, 2, 3, 4, 5, 7, 12, 16, 24, 48, 64, 192, 512, 8192] {
                    for phase in [
                        ExecutionPhase::Decode,
                        ExecutionPhase::Verify,
                        ExecutionPhase::Prefill,
                    ] {
                        assert!(!qualified(set, phase, rows));
                    }
                }
            }
            assert_eq!(
                crate::Kernels::loads("router_finish_moe_deepseek41_sm120", &shape),
                ranks == Ranks::TP4
            );
        }
    }
}
