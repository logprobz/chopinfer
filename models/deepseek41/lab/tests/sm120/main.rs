//! Kernel GPU correctness tests of the model's catalog instances, own ops
//! and encoder, on `kernel_harness`. Each engine adds its test file and its
//! module line here.

// The own ops: the bundle's entries, cubin specification and `OwnOps`
// implementation, and their tests.
mod own;
#[path = "../../../bundle/src/sm120/own_cubin.rs"]
mod own_cubin;
#[path = "../../../bundle/src/sm120/own_entries.rs"]
mod own_entries;
#[path = "../../../bundle/src/sm120/own_ops.rs"]
mod own_ops;
