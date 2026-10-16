//! `chopinfer eval`: qualifies a running server against a model's
//! declaration (`models/<m>/qual.json`) with the shared check kinds of
//! `tools/qual/kinds.py`, to be the one scorer implementation the Python
//! tooling calls. docs/cli.md describes the commands, the report and the
//! exit codes.
//!
//! It names no model: every model fact comes from the declaration, the
//! references and the fixtures it is pointed at.

mod applicability;
pub mod capture;
pub mod cli;
pub mod declaration;
pub mod exact;
pub mod files;
pub mod http;
pub mod json;
pub mod kind;
pub mod kinds;
pub mod kv_offload;
pub mod launch;
pub mod limits;
pub mod locate;
pub mod lock;
pub mod pyjson;
pub mod references;
pub mod report;
pub mod repr;
pub mod rescore;
mod resolver;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

pub use report::{Exit, Tool};

/// The name `<m>` of the model (`models/<m>`) a checkpoint directory
/// belongs to, or why none is.
pub type CheckpointModel = dyn Fn(&Path) -> Result<String, String>;

/// What the program running `eval` lends it: its own binary, whose install
/// holds the models a package or checkpoint belongs to and which `run
/// --launch` starts as `serve`, the route `serve --model` takes from a
/// checkpoint to its model, and where `serve` listens by default.
pub struct Host {
    /// The running `chopinfer`.
    pub executable: PathBuf,
    pub checkpoint_model: Box<CheckpointModel>,
    /// The address `serve` listens on without `--listen`.
    pub serve_listen: Option<String>,
}

impl Host {
    /// This process's binary, which selects no model for a checkpoint and
    /// has no default listen address.
    pub fn current() -> Self {
        Self {
            executable: std::env::current_exe().unwrap_or_default(),
            checkpoint_model: Box::new(|_| {
                Err("this program selects no model for a checkpoint".into())
            }),
            serve_listen: None,
        }
    }
}

/// Runs `eval ARGUMENTS...` for the program `tool` hosted by `host` and
/// returns its exit code (docs/cli.md).
pub fn command_line(tool: &Tool, host: &Host, arguments: &[String]) -> ExitCode {
    cli::run_hosted(tool, host, arguments).code()
}
