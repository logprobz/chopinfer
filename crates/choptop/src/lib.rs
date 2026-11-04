//! choptop: a terminal dashboard of running `chopinfer serve`
//! instances and of this host: its CPUs, memory, disks, sensors and GPUs,
//! with a model loader that starts serve processes and stops the ones they
//! replace.
//!
//! It reads only `/health` and `/metrics` from servers. Instances come from
//! explicit targets, from this user's serve processes found in `/proc`, from
//! Docker containers whose process list shows `chopinfer serve`, and from
//! choptop's own launches. It starts and stops only serve processes of this
//! user, each after a confirm, and never a container or a server elsewhere;
//! `--read-only` starts and stops nothing. The host comes from `/proc` and
//! `/sys`, and GPUs from NVML, loaded at run time by `gpu::nvml`, the one
//! module allowed unsafe code under the workspace's `unsafe_code = "deny"`.

pub mod app;
pub mod catalog;
pub mod cli;
pub mod discover;
pub mod gpu;
pub mod health;
pub mod host;
pub mod http;
pub mod instances;
pub mod launch;
pub mod listing;
pub mod loader;
pub mod poll;
pub mod prometheus;
#[cfg(test)]
mod scratch;
pub mod stats;
pub mod targets;
pub mod tui;
pub mod ui;
