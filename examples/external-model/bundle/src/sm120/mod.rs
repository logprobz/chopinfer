//! The example's SM120 module: its kernel selections, the backends that
//! prepare its ops from them and its own kernels' implementation. Generic
//! code names nothing here: it prepares ops through `&dyn Backend` and
//! `&dyn OwnOps`, which this module supplies (`Kernels::new`,
//! `PreparedExercise::new`).

#[cfg(feature = "cuda-probe")]
use example_contract::ops;
#[cfg(feature = "cuda-probe")]
mod own_ops;
// The build compiles the selections; the bundle registers their entries.
#[cfg(feature = "cuda-probe")]
#[allow(dead_code)]
mod selection;

#[cfg(feature = "precision-exercise")]
use super::precision_contract;
#[cfg(feature = "precision-exercise")]
#[allow(dead_code)]
mod precision_selection;

#[cfg(feature = "cuda-probe")]
impl crate::Kernels {
    /// The SM120 backend over the RMSNorm entry the build compiles, and the
    /// example's own kernels.
    pub(crate) fn new() -> Self {
        Self {
            backend: Box::new(selection::backend()),
            own: Box::new(own_ops::Sm120Own),
        }
    }
}

#[cfg(feature = "precision-exercise")]
mod precision {
    use super::precision_selection;
    use crate::precision::PreparedExercise;
    use kernel_core::precision::execution::ActivationPolicy;

    impl PreparedExercise {
        /// The caller supplies an authenticated component. The prepared ops
        /// launch only entries embedded in this native module.
        pub fn new(bytes: &[u8], policy: ActivationPolicy) -> Result<Self, String> {
            let available = crate::precision::symbols().map(str::to_owned).collect();
            let backend = precision_selection::backend(
                precision_selection::offers()?,
                available,
                include_str!(concat!(env!("OUT_DIR"), "/precision-dependency.txt")).into(),
            )?;
            Self::bind(bytes, policy, &backend)
        }
    }
}
