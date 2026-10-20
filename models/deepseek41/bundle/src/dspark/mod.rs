//! DSpark's committed feature history, stage cursors and block lifecycle.
//!
//! The target program stages raw attention-input features in target-layer
//! order (Flash layers 37, 38 and 39, after averaging the hyper streams).
//! It keeps these separate from the retained committed features. Each
//! drafter stage projects the same raw features through its own KV path.

pub mod acceptance;
pub mod calibration;
pub(crate) mod cycle;
pub(crate) mod device;
pub mod execution;
pub mod features;
pub(crate) mod prefetch;
pub(crate) mod results;
pub mod staging;
pub mod state;
pub mod transfers;
pub(crate) mod upload;

pub(crate) mod status;
