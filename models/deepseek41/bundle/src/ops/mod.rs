//! The ops an instance prepares once through `kernel_core::backend` and
//! the model's `OwnOps`, and their launches over a step's buffers by role.
//!
//! DeepSeek states each op as a kernel-core contract (or one of its own)
//! and the exactness it needs; the target's kernels (`crate::Kernels`)
//! prepare it when the instance is created. A step then binds its spans to
//! the op's roles and calls `PreparedOp::launches`: no catalog search or
//! entry choice happens per step. One file per family holds its ops.

pub(crate) mod exact;
mod rows;

pub(crate) use rows::{EmbeddingViews, NormViews, StreamViews, verify_rows};

use std::cell::RefCell;

use bundle_runtime::dispatch::BufferBindings;
use bundle_runtime::pipeline::PreparedLaunch;
use kernel_core::backend::{Counts, Launch, Need, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::properties::Invariance;
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::Kernels;
use crate::shape::Shape;

/// The model's name in refusals.
pub(crate) const NAME: &str = "DeepSeek V4.1";

/// Every op an instance prepares at create. A family the kernels refuse is
/// absent and named in `missing`: create refuses such an instance, except
/// the fake-host tests that run the lowered stages alone.
pub struct Ops {
    pub(crate) rows: Option<rows::RowOps>,
    /// Each family the kernels refused, with the refusal.
    pub(crate) missing: Vec<String>,
    /// What each op asked at create and was granted, in prepare order.
    #[cfg_attr(not(test), allow(dead_code, reason = "the asks journal reads them"))]
    pub(crate) asked: Vec<Asked>,
    /// The kernels' names: the shared contracts' backend and the model's
    /// own ops'.
    pub(crate) backends: [String; 2],
}

impl Ops {
    /// Prepares every op of an instance of `shape` on `kernels`, recording
    /// each family they refuse.
    pub(crate) fn prepare(kernels: &Kernels, shape: &Shape) -> Ops {
        let asks = Asks::default();
        let mut missing = Vec::new();
        let backend = kernels.backend.name();
        let mut family = |name: &str, prepared: Result<_, Refused>| match prepared {
            Ok(ops) => Some(ops),
            Err(refusal) => {
                missing.push(format!("{backend} refuses {NAME}'s {name} ops: {refusal}"));
                None
            }
        };
        let rows = family(
            "row",
            rows::RowOps::prepare(kernels.backend.as_ref(), shape, &asks),
        );
        Ops {
            rows,
            missing,
            asked: asks.asked(),
            backends: [backend.to_owned(), kernels.own.name().to_owned()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    use model_deepseek41_contract::own_ops::OwnOps;

    struct Nothing;

    impl Backend for Nothing {
        fn name(&self) -> &str {
            "nothing"
        }
    }

    impl OwnOps for Nothing {
        fn name(&self) -> &str {
            "nothing own"
        }
    }

    #[test]
    fn kernels_without_the_ops_prepare_nothing_and_name_each_family() {
        let shape = Shape::new(
            &model_deepseek41_contract::geometry::FLASH,
            bundle_runtime::geometry::Ranks::TP4,
        )
        .unwrap();
        let kernels = Kernels {
            backend: Box::new(Nothing),
            own: Box::new(Nothing),
        };
        let ops = Ops::prepare(&kernels, &shape);
        assert!(ops.rows.is_none());
        assert!(ops.asked.is_empty());
        assert_eq!(ops.backends, ["nothing", "nothing own"]);
        assert!(
            ops.missing[0].starts_with("nothing refuses DeepSeek V4.1's row ops"),
            "{:?}",
            ops.missing
        );
    }
}

/// One op's ask at create: what DeepSeek needs of it, and what the backend
/// granted (`OpInfo::guarantees`, which provide the need, or the backend
/// refused the op and create with it).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Asked {
    /// The op: its family and role, with its rank where it has one.
    pub what: String,
    pub need: Need,
    pub granted: Invariance,
    /// The backend's versioned implementation (`OpInfo::implementation`).
    pub implementation: String,
    /// The entries its launches name.
    pub symbols: Vec<String>,
}

/// The asks of one create, recorded as each op prepares.
#[derive(Default)]
pub(crate) struct Asks(RefCell<Vec<Asked>>);

impl Asks {
    /// The asks recorded, in the order the ops prepared.
    pub(crate) fn asked(self) -> Vec<Asked> {
        self.0.into_inner()
    }

    /// The op `prepare` makes for `need`, its ask recorded as `what`.
    pub(crate) fn ask<B>(
        &self,
        what: impl Into<String>,
        need: Need,
        prepare: impl FnOnce(Need) -> Result<Prepared<B>, Refused>,
    ) -> Result<Prepared<B>, Refused> {
        let prepared = prepare(need)?;
        let info = prepared.info();
        self.0.borrow_mut().push(Asked {
            what: what.into(),
            need,
            granted: info.guarantees,
            implementation: info.implementation.clone(),
            symbols: info.symbols.iter().map(|(s, _)| s.clone()).collect(),
        });
        Ok(prepared)
    }
}

/// A launch with whether it waits for its peer ranks, as the op that made
/// it states (`Launch::collective`).
pub struct Lowered {
    pub launch: PreparedLaunch,
    pub collective: bool,
}

/// Spans bound for one stage's launches, as role views.
#[derive(Default)]
pub(crate) struct Bind(BufferBindings);

impl Bind {
    pub(crate) fn view(&mut self, span: DeviceBufferSpanV1) -> View {
        View {
            handle: self.0.bind(span),
            bytes: span.length,
        }
    }

    /// `op`'s launches over `buffers` as the pipeline's launches, every one
    /// with these bindings and the facts the op states of it. DeepSeek binds
    /// no backend workspace.
    pub(crate) fn launch<B>(
        self,
        op: &Prepared<B>,
        buffers: &B,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        self.lowered_op(op.as_ref(), buffers, counts)
    }

    fn lowered_op<B>(
        self,
        op: &dyn PreparedOp<B>,
        buffers: &B,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        if op.info().workspace != 0 {
            return Err(Refusal("DeepSeek V4.1 binds no backend workspace"));
        }
        let launches = op.launches(buffers, View::EMPTY, counts)?;
        Ok(launches
            .into_iter()
            .map(|launch| self.lowered(launch))
            .collect())
    }

    fn lowered(&self, launch: Launch) -> Lowered {
        Lowered {
            collective: launch.collective,
            launch: PreparedLaunch {
                symbol: launch.symbol,
                rows: launch.rows,
                plan: launch.plan,
                arguments: launch.arguments,
                bindings: self.0.clone(),
            },
        }
    }
}
