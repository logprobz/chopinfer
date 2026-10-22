//! chopinfer-top: a terminal dashboard of running `chopinfer serve`
//! instances and of this host: its CPUs, memory, disks, sensors and GPUs.
//!
//! It reads only `/health` and `/metrics` from servers. Instances come from
//! explicit targets, from this user's serve processes found in `/proc`, and
//! from Docker containers whose process list shows `chopinfer serve`. The
//! host comes from `/proc` and `/sys`, and GPUs from NVML, loaded at run time
//! by `gpu::nvml`, the one module allowed unsafe code under the workspace's
//! `unsafe_code = "deny"`.

pub mod app;
pub mod cli;
pub mod discover;
pub mod gpu;
pub mod health;
pub mod host;
pub mod http;
pub mod instances;
pub mod listing;
pub mod poll;
pub mod prometheus;
pub mod stats;
pub mod targets;
pub mod tui;
pub mod ui;
