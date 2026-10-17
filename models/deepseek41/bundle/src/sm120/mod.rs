//! The bundle's SM120 module: the entries the build compiles and their
//! registration on `Sm120`, and the `OwnOps` implementation over the model's
//! own cubins. Each engine adds one registration file and its module line
//! here. Generic code names nothing in this module beyond constructing the
//! backend and the own ops.

use kernels_sm120::backend::Sm120;

// The model's own kernels: their entries and the `OwnOps` implementation.
#[allow(
    dead_code,
    reason = "instance creation prepares the own ops once the program lands"
)]
pub(crate) mod own_entries;
#[allow(
    dead_code,
    reason = "instance creation prepares the own ops once the program lands"
)]
pub(crate) mod own_ops;

impl crate::Kernels {
    /// The SM120 backend over the entries the build compiles for `shape`,
    /// and the model's own kernels.
    #[allow(dead_code, reason = "the instance prepares its ops on it")]
    pub(crate) fn new(shape: &crate::shape::Shape) -> Result<Self, String> {
        let _ = shape;
        Ok(Self {
            backend: Box::new(Sm120::new()),
            own: Box::new(own_ops::Sm120Own::new()),
        })
    }
}
