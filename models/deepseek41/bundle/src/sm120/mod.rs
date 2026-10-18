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
// The build compiles each registration file's cubins; the bundle registers
// their entries.
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
pub(crate) mod encoder;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod rows;

impl crate::Kernels {
    /// The SM120 backend over the entries the build compiles for `shape`,
    /// and the model's own kernels.
    pub(crate) fn new(shape: &crate::shape::Shape) -> Result<Self, String> {
        let sm120 = rows::register(Sm120::new(), shape.geometry, shape.ranks());
        // The vision encoder's shared entries (`crate::encoders`).
        let sm120 = encoder::register(sm120)?;
        Ok(Self {
            backend: Box::new(sm120),
            own: Box::new(own_ops::Sm120Own::new()),
        })
    }
}
