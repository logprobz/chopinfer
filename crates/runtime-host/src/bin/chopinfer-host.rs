#![forbid(unsafe_code)]

//! The conformance host: one selected bundle, driven through one batch.
//!
//! This program runs the same commands `chopinfer` does, from the same library.
//! It differs in one thing: an invocation that names no command is its
//! conformance run, because driving one bounded batch through one loaded
//! bundle is what this program exists for. Serving is still reached by naming
//! `serve`, since the two are different commands.
//!
//! It links no model and no bundle, so the same unchanged binary runs every
//! bundle a conformance suite installs.

use std::env;
use std::process::ExitCode;

use runtime_host::{parse_conformance_host_command_v1, run_command_line_v1};

fn main() -> ExitCode {
    run_command_line_v1(
        "chopinfer-host",
        env::args_os().skip(1),
        parse_conformance_host_command_v1,
    )
}
