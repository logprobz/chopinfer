#![forbid(unsafe_code)]

//! The program an operator runs: verify a bundle, install one, or serve one.
//!
//! Every command lives in the library, so verification here is the
//! verification a serving process performs at load time, and an install here
//! is the same no-replace publication. This program adds nothing to them but
//! its own name.
//!
//! It links no model and no bundle. Which bundle it serves is a runtime
//! selection, so the same unchanged program serves a different one by being
//! pointed at a different install root or directory.

use std::env;
use std::process::ExitCode;

use runtime_host::{parse_host_command_v1, run_command_line_v1};

fn main() -> ExitCode {
    run_command_line_v1("chopinfer", env::args_os().skip(1), parse_host_command_v1)
}
