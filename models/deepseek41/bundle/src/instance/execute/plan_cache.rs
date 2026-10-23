//! Upper-skip plans of short single-item text prompts, kept by their exact
//! lowering inputs and reused instead of lowering every rank again.
//!
//! `skip::Plan::prepare` reads the phase's rows, item extents, prefix
//! lengths, positions, logits rows, input layout and cache range, and
//! otherwise only state the instance fixes at create (buffers, arenas, ops,
//! KV extent, prefill counts, and a DSpark target's tap, feature and Engram
//! staging spans). An eligible run starts its one item at row zero with no
//! prefix and positions `0..rows`, has no prefill cache range, media or
//! position axes, and scores only its last row, so its rows and layout fix
//! every input, a DSpark target's retained feature rows included; they are
//! the key. Token IDs, block tables, slots, window and tail inputs, index
//! tags and Engram rows reach the kernels through the per-step uploads and
//! are never part of a plan.
//!
//! `CHOPINFER_DEEPSEEK41_PREFILL_PLAN_CACHE`: unset or `1` keeps plans,
//! `0` lowers every step, `verify` keeps plans and lowers every hit again,
//! refusing the step if the fresh plan differs from the kept one.

use super::{PREFILL, Run, skip};
use crate::instance::Instance;
use crate::program::Phase;
use bundle_runtime::dispatch::{arguments, launch_config};
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::upload::InputLayout;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

const SWITCH: &str = "CHOPINFER_DEEPSEEK41_PREFILL_PLAN_CACHE";
/// The longest prompt a plan is kept for. At most one plan of each length
/// is kept; one TP4 Flash plan holds about ten thousand launches in about
/// 20 MiB of host memory.
pub(crate) const MAX_ROWS: u32 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Mode {
    Off,
    On,
    Verify,
}

impl Mode {
    fn read(setting: Option<&str>) -> Result<Self, String> {
        match setting {
            None | Some("1") => Ok(Self::On),
            Some("0") => Ok(Self::Off),
            Some("verify") => Ok(Self::Verify),
            Some(other) => Err(format!("{SWITCH}={other:?} is not 0, 1 or verify")),
        }
    }
}

#[cfg(test)]
thread_local! {
    static IN_TESTS: Cell<Option<Mode>> = const { Cell::new(None) };
}

/// Runs `run` with instances created inside it in `mode`.
#[cfg(test)]
pub(crate) fn with_mode<T>(mode: Mode, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<Mode>);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_TESTS.set(self.0);
        }
    }
    let _restore = Restore(IN_TESTS.replace(Some(mode)));
    run()
}

#[derive(Debug, Eq, PartialEq)]
struct Key {
    rows: u32,
    layout: InputLayout,
}

type Kept = Option<(Key, Rc<skip::Plan>)>;

pub(crate) struct Cache {
    mode: Mode,
    /// A plan of each length, `rows - 1` its index.
    plans: RefCell<Vec<Kept>>,
    hits: Cell<u64>,
    misses: Cell<u64>,
}

impl Cache {
    pub(crate) fn from_env() -> Result<Self, String> {
        #[cfg(test)]
        if let Some(mode) = IN_TESTS.get() {
            return Ok(Self::new(mode));
        }
        match std::env::var(SWITCH) {
            Ok(value) => Mode::read(Some(&value)),
            Err(std::env::VarError::NotPresent) => Mode::read(None),
            Err(std::env::VarError::NotUnicode(_)) => Err(format!("{SWITCH} is not unicode")),
        }
        .map(Self::new)
    }

    fn new(mode: Mode) -> Self {
        Self {
            mode,
            plans: RefCell::new((0..MAX_ROWS).map(|_| None).collect()),
            hits: Cell::new(0),
            misses: Cell::new(0),
        }
    }

    /// Plans served from the cache and plans lowered for it.
    #[cfg(test)]
    fn counts(&self) -> (u64, u64) {
        (self.hits.get(), self.misses.get())
    }

    /// `run`'s upper-skip plan: the kept plan of its key, or a fresh one,
    /// kept when the run is eligible.
    pub(super) fn plan(&self, instance: &Instance, run: &Run) -> Result<Rc<skip::Plan>, String> {
        let key = match self.mode {
            Mode::Off => None,
            Mode::On | Mode::Verify => key(run),
        };
        let Some(key) = key else {
            return skip::Plan::prepare(instance, run).map(Rc::new);
        };
        let at = key.rows as usize - 1;
        let kept = match &self.plans.borrow()[at] {
            Some((k, plan)) if *k == key => Some(Rc::clone(plan)),
            _ => None,
        };
        if let Some(plan) = kept {
            self.hits.set(self.hits.get() + 1);
            if self.mode == Mode::Verify && !plan.same(&skip::Plan::prepare(instance, run)?)? {
                return Err("a kept prefill plan differs from a fresh lowering".into());
            }
            return Ok(plan);
        }
        let plan = Rc::new(skip::Plan::prepare(instance, run)?);
        // A DMA plan carries transport state of its own; keep only plain ones.
        if plan.uses_dma() {
            return Ok(plan);
        }
        self.misses.set(self.misses.get() + 1);
        let mut plans = self.plans.borrow_mut();
        plans[at] = Some((key, Rc::clone(&plan)));
        instance.log(&format!(
            "prefill plan cache: kept {} rows; {} plans, {} hits, {} misses",
            at + 1,
            plans.iter().flatten().count(),
            self.hits.get(),
            self.misses.get()
        ));
        Ok(plan)
    }
}

/// `run`'s key, or `None` if its plan may depend on more than the key.
fn key(run: &Run) -> Option<Key> {
    let inputs = &run.inputs;
    let rows = inputs.live_tokens();
    let eligible = run.arena == PREFILL
        && run.phase == Phase::Prefill { rows }
        && (1..=MAX_ROWS).contains(&rows)
        && run.media.is_none()
        && run.cache.is_none()
        && inputs.request_ids.len() == 1
        && inputs.q_start == [0]
        && inputs.q_len == [rows as i32]
        && inputs.prefix_len == [0]
        && inputs.position_axes.is_empty()
        && inputs.positions.iter().copied().eq(0..rows as i32)
        && inputs.logit_rows() == [rows - 1]
        && run.head_rows == 1;
    eligible.then_some(Key {
        rows,
        layout: run.layout,
    })
}

/// Whether two programs issue the same launches with the same arguments as
/// the host resolves them, the same collectives and the same stages.
pub(super) fn same_programs(a: &super::Programs, b: &super::Programs) -> Result<bool, String> {
    if a.1 != b.1 || a.2 != b.2 || a.0.len() != b.0.len() {
        return Ok(false);
    }
    for (a, b) in a.0.iter().zip(&b.0) {
        if a.len() != b.len() {
            return Ok(false);
        }
        for (a, b) in a.iter().zip(b) {
            if !same_launch(a, b)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn same_launch(a: &PreparedLaunch, b: &PreparedLaunch) -> Result<bool, String> {
    let resolve = |launch: &PreparedLaunch| {
        Ok::<_, String>((
            launch_config(launch.plan).map_err(|e| format!("{}: {e:?}", launch.symbol))?,
            arguments(&launch.arguments, &launch.bindings)
                .map_err(|e| format!("{}: {e:?}", launch.symbol))?,
        ))
    };
    Ok(a.symbol == b.symbol && a.rows == b.rows && a.plan == b.plan && resolve(a)? == resolve(b)?)
}

/// The (hits, misses) of the instance `handle`.
#[cfg(test)]
pub(crate) fn counts(handle: runtime_bundle_abi::BundleHandleV1) -> (u64, u64) {
    use bundle_runtime::entry::Bundle;
    let mut counts = None;
    Instance::instances().with(|table| {
        bundle_runtime::table::with_instance(table, handle, |instance| {
            counts = Some(instance.plans.counts());
            runtime_bundle_abi::BundleStatusV1::Success
        })
    });
    counts.expect("a live instance")
}

#[cfg(test)]
mod tests {
    use super::{Mode, same_programs};
    use crate::program::Stage;
    use bundle_runtime::dispatch::BufferBindings;
    use bundle_runtime::pipeline::PreparedLaunch;
    use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    /// One launch of `symbol` reading `offset` of a buffer and `scalar`,
    /// bound after `unused` other views, so planner IDs differ.
    fn launch(symbol: &str, offset: u64, scalar: u32, unused: u64) -> PreparedLaunch {
        let mut bindings = BufferBindings::default();
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        for at in 0..unused {
            bindings.bind(DeviceBufferSpanV1::new(buffer, at * 512, 256));
        }
        let view = bindings.bind(DeviceBufferSpanV1::new(buffer, offset, 256));
        PreparedLaunch {
            symbol: symbol.into(),
            rows: 1,
            plan: LaunchPlan {
                grid: [1, 1, 1],
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: false,
            },
            arguments: ArgumentList(vec![Argument::DeviceBuffer(view), Argument::U32(scalar)]),
            bindings,
        }
    }

    fn program(launch: PreparedLaunch, waits: bool) -> super::super::Programs {
        (
            vec![vec![launch]],
            vec![vec![waits]],
            vec![vec![Stage::HeadCollapse]],
        )
    }

    #[test]
    fn programs_compare_by_what_the_host_resolves() {
        let same = |a, b| same_programs(&a, &b).unwrap();
        let base = || program(launch("k", 256, 7, 0), false);
        assert!(same(base(), base()));
        // Other planner IDs of the same views resolve to the same launch.
        assert!(same(base(), program(launch("k", 256, 7, 3), false)));
        assert!(!same(base(), program(launch("j", 256, 7, 0), false)));
        assert!(!same(base(), program(launch("k", 768, 7, 0), false)));
        assert!(!same(base(), program(launch("k", 256, 8, 0), false)));
        assert!(!same(base(), program(launch("k", 256, 7, 0), true)));
    }

    #[test]
    fn the_switch_defaults_on_and_refuses_other_values() {
        assert_eq!(Mode::read(None), Ok(Mode::On));
        assert_eq!(Mode::read(Some("1")), Ok(Mode::On));
        assert_eq!(Mode::read(Some("0")), Ok(Mode::Off));
        assert_eq!(Mode::read(Some("verify")), Ok(Mode::Verify));
        assert!(Mode::read(Some("on")).is_err());
        assert!(Mode::read(Some("")).is_err());
    }
}
