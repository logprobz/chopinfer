//! The pipelined prefill (`crate::overlap`) on SM120: the GPU's SMs, the
//! pipelined all-reduce kernels' cluster, the lanes' settings MiMo measured
//! on SM120, and the grids a lane moves ([`adjust`]).

use bundle_runtime::lanes::{Config, Lane};

use crate::overlap::Facts;
use crate::pipeline::PreparedLaunch;

/// SMs of the GPU, the grid of a persistent GEMM or MoE launch.
pub const SMS: u32 = kernels_sm120::gemm::SM_COUNT;

/// CTAs per cluster of the pipelined all-reduce kernels.
pub const COMM_CLUSTER: u32 = 8;

/// Exactly `0` turns the pipelined prefill off and exactly `1` on at any
/// rank count. Unset or any other value (empty, `true`, `2`, padded with
/// whitespace) keeps the rank default: on at TP4 only.
pub const ENABLE: &str = "CHOPINFER_MIMO_PREFILL_OVERLAP";
/// SMs the persistent prefill GEMM and MoE grids leave free, default 16.
pub const RESERVED_SMS: &str = "CHOPINFER_MIMO_OVERLAP_RESERVED_SMS";
/// Grid of the pipelined all-reduce kernels, a multiple of their 8-CTA
/// cluster, default 24; `0` keeps the planned grid (64 for any half).
pub const COMM_CTAS: &str = "CHOPINFER_MIMO_OVERLAP_COMM_CTAS";
/// The most CTAs of the pipelined all-reduce kernels' 256 threads an SM120
/// GPU holds at once: six per SM, its 1536 resident threads. Create refuses
/// a larger [`COMM_CTAS`] grid (`comm::spinning_entries_for`), which could
/// never be resident and whose gather's warp count wraps to zero from 2^29
/// workers; smaller ones meet the device's residency check.
pub const MAX_COMM_CTAS: u32 = SMS * (1536 / 256);
/// Chunks with fewer tokens run unpipelined, default 1024.
pub const MIN_TOKENS: &str = "CHOPINFER_MIMO_OVERLAP_MIN_TOKENS";

// Engine prefill tok/s with overlap over without, two paired server starts
// per point (2026-10-04). An 8K-token C1 sweep of {0, 8, 16} reserved SMs x
// {24, 64 (planned), 80, 96} comm CTAs found nothing better than TP4's 16 SMs
// with 24 CTAs (1.08 and 1.10) and every TP2 setting slower than no overlap
// (0.91 to 0.97). At C1 and C8, the TP4 default then measured 1.03 to 1.09
// at 2K and 32K, and TP2's 0.85 to 0.94 at 2K, 8K and 32K with worse TTFT.
// So only TP4 pipelines by default, and other rank counts use the TP4
// settings when `ENABLE` is 1.
const DEFAULT_RESERVED_SMS: u32 = 16;
const DEFAULT_COMM_CTAS: u32 = 24;
const DEFAULT_MIN_TOKENS: u32 = 1024;
const DEFAULT_RANKS: u32 = 4;

/// The lanes of a `tp` instance from the process environment, if the
/// pipelined prefill is on.
pub fn config(tp: crate::topology::Tp) -> Option<Config> {
    config_with(tp.ranks(), |name| std::env::var(name).ok())
}

fn config_with(ranks: u32, value: impl FnMut(&str) -> Option<String>) -> Option<Config> {
    bundle_runtime::lanes::Settings {
        enable: ENABLE,
        reserved_sms: RESERVED_SMS,
        comm_ctas: COMM_CTAS,
        min_tokens: MIN_TOKENS,
        default_reserved_sms: DEFAULT_RESERVED_SMS,
        default_comm_ctas: DEFAULT_COMM_CTAS,
        default_min_tokens: DEFAULT_MIN_TOKENS,
        sms: SMS,
        comm_cluster: COMM_CLUSTER,
    }
    .read_switched(ranks == DEFAULT_RANKS, value)
}

/// Leaves SMs for the all-reduces (`lanes::adjust_grid`). On the compute
/// lane a launch whose op states it persistent, gridded at every SM, shrinks
/// by `reserved_sms`: the ops SM120's backend marks persistent are its
/// persistent GEMM and MoE entries, whose CTAs stride over their tiles until
/// none is left, so any grid computes the same rows in the same order per
/// tile. On the comm lane the two launches of a phased all-reduce take
/// `comm_ctas` CTAs, a multiple of their cluster that their kernels stride
/// over likewise. Other grids stay.
pub fn adjust(launch: &mut PreparedLaunch, facts: Facts, lane: Lane, config: &Config) {
    bundle_runtime::lanes::adjust_grid(&mut launch.plan.grid, lane, config, facts);
}

#[cfg(test)]
thread_local! {
    /// This thread's pipelined launches: all of them, those whose grid their
    /// facts shrank, the launches of phased pairs, and every launch on the
    /// comm lane.
    pub static ADJUSTED: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
}

/// Counts how `facts` move `launch`'s grid on `lane` ([`ADJUSTED`]).
#[cfg(test)]
pub fn count(launch: &PreparedLaunch, facts: Facts, lane: Lane, config: &Config) {
    let mut grid = launch.plan.grid;
    bundle_runtime::lanes::adjust_grid(&mut grid, lane, config, facts);
    ADJUSTED.with(|adjusted| {
        let [all, shrunk, phased, comm] = adjusted.get();
        adjusted.set([
            all + 1,
            shrunk + usize::from(lane == Lane::Compute && grid != launch.plan.grid),
            phased + usize::from(facts.phased_comm),
            comm + usize::from(lane == Lane::Comm),
        ])
    });
}

#[cfg(test)]
mod tests {
    use kernel_core::contract::{ArgumentList, LaunchPlan};

    use super::*;

    #[test]
    fn only_tp4_pipelines_by_default_and_the_switch_overrides_it() {
        let unset = |_: &str| None;
        let tp4 = config_with(4, unset).expect("TP4 default on");
        assert_eq!(
            (tp4.reserved_sms, tp4.comm_ctas, tp4.min_tokens),
            (16, Some(24), 1024)
        );
        for ranks in [1, 2, 8] {
            assert_eq!(config_with(ranks, unset), None, "TP{ranks}");
            let forced = config_with(ranks, |name| (name == ENABLE).then(|| "1".into()));
            assert_eq!(forced, Some(tp4), "TP{ranks}");
        }
        for ranks in [2, 4] {
            assert_eq!(
                config_with(ranks, |name| (name == ENABLE).then(|| "0".into())),
                None
            );
        }
        // Only the literal switch values override the rank default.
        for value in ["", "true", "2", " 1", "1 ", " 0", "0 ", "on", "off"] {
            for (ranks, expected) in [(2, None), (4, Some(tp4))] {
                let config = config_with(ranks, |name| (name == ENABLE).then(|| value.into()));
                assert_eq!(config, expected, "TP{ranks} {value:?}");
            }
        }
        let planned = config_with(4, |name| match name {
            RESERVED_SMS => Some("0".into()),
            COMM_CTAS => Some("0".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!((planned.reserved_sms, planned.comm_ctas), (0, None));
    }

    fn plan(grid: [u32; 3]) -> LaunchPlan {
        LaunchPlan {
            grid,
            block: [384, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        }
    }

    fn launch(symbol: &str, grid: [u32; 3]) -> PreparedLaunch {
        PreparedLaunch {
            symbol: symbol.into(),
            // `adjust` reads only the grid.
            rows: 1,
            plan: plan(grid),
            arguments: ArgumentList(Vec::new()),
            bindings: crate::dispatch::BufferBindings::default(),
        }
    }

    #[test]
    fn only_persistent_compute_and_phased_comm_grids_change() {
        let config = Config {
            sms: SMS,
            reserved_sms: 24,
            comm_ctas: Some(16),
            min_tokens: 512,
        };
        let adjusted = |grid, facts, lane, config: &Config| {
            let mut launch = launch("any", grid);
            adjust(&mut launch, facts, lane, config);
            launch.plan.grid
        };
        let persistent = Facts {
            persistent: true,
            phased_comm: false,
            collective: false,
        };
        let plain = Facts::default();
        assert_eq!(
            adjusted([SMS, 1, 1], persistent, Lane::Compute, &config),
            [164, 1, 1]
        );
        assert_eq!(
            adjusted([SMS, 1, 1], plain, Lane::Compute, &config),
            [SMS, 1, 1]
        );
        assert_eq!(
            adjusted([752, 1, 1], persistent, Lane::Compute, &config),
            [752, 1, 1]
        );
        let phased = Facts {
            persistent: false,
            phased_comm: true,
            collective: true,
        };
        for comm_ctas in [Some(16), Some(24), None] {
            let config = Config {
                comm_ctas,
                ..config
            };
            assert_eq!(
                adjusted([64, 1, 1], phased, Lane::Comm, &config),
                [comm_ctas.unwrap_or(64), 1, 1]
            );
            assert_eq!(adjusted([64, 1, 1], plain, Lane::Comm, &config), [64, 1, 1]);
        }
    }

    /// Create checks the phased pair at the comm lane's grid as well as the
    /// planned one, and refuses a grid past what the GPU can hold at once:
    /// the review's 536870920 (2^29 workers, whose gather's warp count
    /// wraps to zero) among them. The default and the measured grids keep
    /// their checks.
    #[test]
    fn create_checks_and_bounds_the_comm_lanes_grid() {
        use super::super::comm::{PHASED_SYMBOLS, spinning_entries_for};
        let tp = crate::topology::Tp::SUPPORTED
            .into_iter()
            .find(|tp| tp.ranks() == DEFAULT_RANKS)
            .unwrap();
        let read = |ctas: &str| {
            config_with(DEFAULT_RANKS, |name| {
                (name == COMM_CTAS).then(|| ctas.into())
            })
            .unwrap()
        };
        let phased_grids = |config: Option<&Config>| {
            spinning_entries_for(tp, 32, config).map(|entries| {
                entries
                    .iter()
                    .filter(|entry| PHASED_SYMBOLS.contains(&entry.symbol))
                    .map(|entry| entry.plan.grid[0])
                    .collect::<Vec<_>>()
            })
        };
        let planned = phased_grids(None).unwrap();
        assert_eq!(planned, [64, 64]);
        // The default (24) and every grid at most the planned one leave the
        // planned grid the largest checked; wider grids are checked at
        // their own size.
        for (ctas, checked) in [("24", 64), ("16", 64), ("64", 64), ("96", 96)] {
            assert_eq!(
                phased_grids(Some(&read(ctas))).unwrap(),
                [checked, checked],
                "{ctas}"
            );
        }
        let most = MAX_COMM_CTAS.to_string();
        assert_eq!(MAX_COMM_CTAS, 1128);
        assert_eq!(
            phased_grids(Some(&read(&most))).unwrap(),
            [MAX_COMM_CTAS; 2]
        );
        for ctas in [MAX_COMM_CTAS + 8, 536_870_920, u32::MAX - 7] {
            let config = read(&ctas.to_string());
            assert_eq!(config.comm_ctas, Some(ctas));
            let refusal = phased_grids(Some(&config)).unwrap_err();
            assert!(refusal.0.contains("resident"), "{ctas}: {refusal}");
        }
        // Without a comm grid the planned one stays.
        assert_eq!(phased_grids(Some(&read("0"))).unwrap(), planned);
    }
}
