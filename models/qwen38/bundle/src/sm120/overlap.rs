//! The pipelined prefill on SM120: the GPU's SMs, the pipelined
//! all-reduce kernels' cluster, the lanes' settings Qwen measured on SM120
//! by rank count, and the grids a lane moves ([`adjust`]).
//!
//! A launch an op made carries the op's facts (`ops::Lowered`): its
//! persistence as the op's `Launch` states it, and the phased pair, the two
//! collective launches (scatter, then gather) of the general all-reduce. A
//! launch Qwen built without an op (a registered adapter's, a stored
//! projection's, the media inject) is judged by its symbol ([`built`]).

use bundle_runtime::lanes::LaunchFacts as Facts;
use bundle_runtime::lanes::{Config, Lane, adjust_grid};

use super::catalog;
use crate::contract::Shape;
use crate::pipeline::PreparedLaunch;

/// SMs of the GPU, the grid of a persistent GEMM or MoE launch.
pub const SMS: u32 = kernels_sm120::gemm::SM_COUNT;

/// CTAs per cluster of the pipelined all-reduce kernels.
pub const COMM_CLUSTER: u32 = 8;

/// The most CTAs of the phased all-reduce kernels' 256 threads an SM120 GPU
/// holds at once: six per SM, its 1536 resident threads. Create refuses a
/// larger comm grid (`catalog::spinning_entries`), which could never be
/// resident, even where the device cannot report its residency
/// (`bundle_runtime::residency::UNCHECKED`).
pub const MAX_COMM_CTAS: u32 = SMS * (1536 / 256);

/// The facts of a launch by its symbol: persistent fragment prefill GEMMs
/// and prefill MoE variants, and the phased prefill all-reduce pair, the
/// only collectives among them.
pub fn built(shape: Shape, launch: &PreparedLaunch) -> Facts {
    use std::collections::BTreeSet;
    let symbol = launch.symbol.as_str();
    static PERSISTENT: std::sync::OnceLock<BTreeSet<String>> = std::sync::OnceLock::new();
    let persistent = PERSISTENT
        .get_or_init(|| {
            catalog::dense::profiled()
                .iter()
                .filter(|(_, entry)| {
                    matches!(
                        entry.key.dense,
                        Some(kernel_core::contract::DenseKernel::FragmentPrefill { .. })
                    )
                })
                .map(|(_, entry)| entry.symbol.clone())
                .chain(
                    catalog::moe::variants()
                        .iter()
                        .filter(|entry| entry.phase == catalog::moe::Phase::Prefill)
                        .map(|entry| entry.symbol.clone()),
                )
                .collect()
        })
        .contains(symbol);
    let phased_comm = catalog::phased_comm(shape, symbol);
    Facts {
        persistent,
        phased_comm,
        collective: phased_comm,
    }
}

/// Leaves SMs for the comm lane (`lanes::adjust_grid`). On the compute lane
/// a launch whose facts state it persistent, gridded at every SM, shrinks by
/// `reserved_sms`: the launches SM120 marks persistent are its persistent
/// GEMM and prefill MoE entries, whose CTAs stride over their tiles until
/// none is left, so any grid computes the same rows in the same order per
/// tile. On the comm lane the two launches of a phased all-reduce take
/// `comm_ctas` CTAs, a multiple of their cluster that their kernels stride
/// over likewise. Other grids stay.
pub fn adjust(launch: &mut PreparedLaunch, facts: Facts, lane: Lane, config: &Config) {
    adjust_grid(&mut launch.plan.grid, lane, config, facts);
}

#[cfg(test)]
thread_local! {
    /// This thread's pipelined launches whose grids the facts and the
    /// symbol rule set alike: all of them, those whose grid shrank, the
    /// launches of phased pairs, and every launch on the comm lane.
    pub static COMPARED: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
}

/// Checks that `facts` move `launch`'s grid on `lane` as the symbol rule
/// did before launches carried their op's facts ([`built`]).
#[cfg(test)]
pub fn compare_with_symbols(
    shape: Shape,
    launch: &PreparedLaunch,
    facts: Facts,
    lane: Lane,
    config: &Config,
) {
    let grid = |facts| {
        let mut grid = launch.plan.grid;
        adjust_grid(&mut grid, lane, config, facts);
        grid
    };
    assert_eq!(
        grid(facts),
        grid(built(shape, launch)),
        "{} on the {lane:?} lane",
        launch.symbol
    );
    COMPARED.with(|compared| {
        let [all, shrunk, phased, comm] = compared.get();
        compared.set([
            all + 1,
            shrunk + usize::from(lane == Lane::Compute && grid(facts) != launch.plan.grid),
            phased + usize::from(facts.phased_comm),
            comm + usize::from(lane == Lane::Comm),
        ])
    });
}

/// The lanes of an instance of `ranks` ranks, if the pipelined prefill is
/// on: the ranks Qwen measured on SM120 default on, others need the switch
/// and take TP4's settings.
pub(crate) fn config_with(ranks: u32, value: impl FnMut(&str) -> Option<String>) -> Option<Config> {
    use bundle_runtime::lanes::Settings;
    let defaults = match ranks {
        2 => Some((24, 0, 4096)),
        4 => Some((16, 80, 1024)),
        _ => None,
    };
    // Untuned ranks require explicit opt-in and use the TP4 policy.
    let (reserved_sms, comm_ctas, min_tokens) = defaults.unwrap_or((16, 80, 1024));
    Settings {
        enable: "CHOPINFER_QWEN38_PREFILL_OVERLAP",
        reserved_sms: "CHOPINFER_QWEN38_OVERLAP_RESERVED_SMS",
        comm_ctas: "CHOPINFER_QWEN38_OVERLAP_COMM_CTAS",
        min_tokens: "CHOPINFER_QWEN38_OVERLAP_MIN_TOKENS",
        default_reserved_sms: reserved_sms,
        default_comm_ctas: comm_ctas,
        default_min_tokens: min_tokens,
        sms: SMS,
        comm_cluster: COMM_CLUSTER,
    }
    .read_switched(defaults.is_some(), value)
}

/// The lanes of an instance of `ranks` ranks, from the environment.
pub(crate) fn config(ranks: u32) -> Option<Config> {
    let config = config_with(ranks, |name| {
        #[cfg(test)]
        if name == "CHOPINFER_QWEN38_PREFILL_OVERLAP"
            && let Some(enabled) = crate::instance::FORCE_OVERLAP.with(|value| value.get())
        {
            return Some(if enabled { "1" } else { "0" }.into());
        }
        std::env::var(name).ok()
    })?;
    #[cfg(test)]
    let config = Config {
        min_tokens: crate::instance::FORCE_OVERLAP_MIN
            .with(|value| value.get())
            .unwrap_or(config.min_tokens),
        ..config
    };
    Some(config)
}

#[cfg(test)]
mod tests {
    use super::config_with as overlap_config_with;

    #[test]
    fn tuned_ranks_default_on_and_untuned_ranks_require_opt_in() {
        for (ranks, reserved, ctas, minimum) in [(2, 24, None, 4096), (4, 16, Some(80), 1024)] {
            let config = overlap_config_with(ranks, |_| None).expect("tuned default on");
            assert_eq!(
                (config.reserved_sms, config.comm_ctas, config.min_tokens),
                (reserved, ctas, minimum)
            );
        }
        assert!(overlap_config_with(8, |_| None).is_none());
        for ranks in [2, 4, 8] {
            assert!(
                overlap_config_with(ranks, |name| (name == "CHOPINFER_QWEN38_PREFILL_OVERLAP")
                    .then(|| "0".into()))
                .is_none()
            );
        }
        let forced = overlap_config_with(8, |name| {
            (name == "CHOPINFER_QWEN38_PREFILL_OVERLAP").then(|| "1".into())
        })
        .unwrap();
        assert_eq!(
            (forced.reserved_sms, forced.comm_ctas, forced.min_tokens),
            (16, Some(80), 1024)
        );
    }

    #[test]
    fn explicit_resource_overrides_keep_zero_as_planned_grid() {
        for ranks in [2, 4] {
            let config = overlap_config_with(ranks, |name| match name {
                "CHOPINFER_QWEN38_OVERLAP_RESERVED_SMS" => Some("0".into()),
                "CHOPINFER_QWEN38_OVERLAP_COMM_CTAS" => Some("0".into()),
                "CHOPINFER_QWEN38_OVERLAP_MIN_TOKENS" => Some("128".into()),
                _ => None,
            })
            .unwrap();
            assert_eq!(
                (config.reserved_sms, config.comm_ctas, config.min_tokens),
                (0, None, 128)
            );
            let capped = overlap_config_with(ranks, |name| {
                (name == "CHOPINFER_QWEN38_OVERLAP_COMM_CTAS").then(|| "64".into())
            })
            .unwrap();
            assert_eq!(capped.comm_ctas, Some(64));
        }
    }
}
