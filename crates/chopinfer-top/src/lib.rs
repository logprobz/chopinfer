#![forbid(unsafe_code)]
//! chopinfer-top: a terminal dashboard of running `chopinfer serve`
//! instances.
//!
//! It reads only `/health` and `/metrics`. Instances come from explicit
//! targets, from this user's serve processes found in `/proc`, and from
//! Docker containers whose process list shows `chopinfer serve`.

pub mod app;
pub mod cli;
pub mod discover;
pub mod health;
pub mod http;
pub mod instances;
pub mod listing;
pub mod poll;
pub mod prometheus;
pub mod stats;
pub mod targets;
pub mod tui;
pub mod ui;
