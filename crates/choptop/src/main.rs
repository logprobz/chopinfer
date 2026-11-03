#![forbid(unsafe_code)]

//! choptop: a live view of running chopinfer serve instances.

use std::env;
use std::io::IsTerminal;
use std::process::ExitCode;

use choptop::catalog::{self, Checkpoints};
use choptop::cli::{self, Options, USAGE};
use choptop::discover::Discovery;
use choptop::discover::docker::DockerScanner;
use choptop::discover::proc::ProcScanner;
use choptop::instances::Tone;
use choptop::launch::records;
use choptop::ui::widgets::sanitize;
use choptop::{listing, targets, tui};

fn main() -> ExitCode {
    let options = match cli::parse(env::args().skip(1)) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("choptop: {error}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    if options.help {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if options.version {
        println!("choptop {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    let file = match &options.targets_file {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => Some(text),
            Err(error) => {
                eprintln!("choptop: cannot read {}: {error}", path.display());
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    let environment = env::var("CHOPINFER_TOP_TARGETS").ok();
    let targets = match targets::collect(&options.targets, file.as_deref(), environment.as_deref())
    {
        Ok(targets) => targets,
        Err(error) => {
            eprintln!("choptop: {error}");
            return ExitCode::from(2);
        }
    };
    let bearer = options.api_key.clone().or_else(|| {
        env::var("CHOPINFER_API_KEY")
            .ok()
            .filter(|key| !key.is_empty())
    });
    let (discovery, notes, sources) = discovery(&options, !targets.is_empty());
    if options.list {
        let (registry, gpus, now) = listing::collect(
            targets,
            discovery,
            notes,
            options.interval,
            bearer.as_deref(),
            options.gpus,
        );
        if options.json {
            println!("{}", listing::document(&registry, &gpus, now));
        } else {
            print!("{}", listing::table(&registry, &gpus, now));
            for event in &registry.events {
                if matches!(event.tone, Tone::Warning | Tone::Bad) {
                    eprintln!(
                        "choptop: {}: {}",
                        sanitize(&event.who),
                        sanitize(&event.what)
                    );
                }
            }
        }
        return ExitCode::SUCCESS;
    }
    if !std::io::stdout().is_terminal() {
        eprintln!("choptop: stdout is not a terminal; use --list, or --list --json");
        return ExitCode::from(2);
    }
    let setup = tui::Setup {
        targets,
        discovery,
        notes,
        interval: options.interval,
        rescan: options.rescan,
        bearer,
        sources,
        gpus: options.gpus,
        gpu_counters: options.gpu_counters,
        chopinfer: catalog::find_binary(
            options.chopinfer.as_deref(),
            env::var_os("PATH").as_deref(),
        ),
        checkpoints: Checkpoints::load(
            catalog::checkpoints_path(env::var_os("XDG_CONFIG_HOME"), env::var_os("HOME")),
            catalog::checkpoint_variables(env::vars()),
        ),
        environment: env::vars()
            .filter(|(name, _)| name.starts_with("CHOPINFER_"))
            .collect(),
        logs: logs_shown(),
        state: records::state_directory(env::var_os("XDG_STATE_HOME"), env::var_os("HOME")),
        read_only: options.read_only,
    };
    match tui::run(setup) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("choptop: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Where launch logs go, `~` for the home directory: under
/// `$XDG_STATE_HOME/choptop`, else `~/.local/state/choptop`.
fn logs_shown() -> String {
    let home = env::var("HOME").ok().filter(|home| home.starts_with('/'));
    let state = env::var("XDG_STATE_HOME")
        .ok()
        .filter(|state| state.starts_with('/'));
    let path = match (&state, &home) {
        (Some(state), _) => format!("{state}/choptop/logs"),
        (None, Some(home)) => format!("{home}/.local/state/choptop/logs"),
        (None, None) => return "~/.local/state/choptop/logs".to_owned(),
    };
    match &home {
        Some(home) if path.starts_with(&format!("{home}/")) => format!("~{}", &path[home.len()..]),
        _ => path,
    }
}

/// The scanners the options ask for, notes about any that cannot run, and
/// the sources shown in the header.
fn discovery(
    options: &Options,
    targets: bool,
) -> (Option<Discovery>, Vec<String>, Vec<&'static str>) {
    let mut notes = Vec::new();
    let mut sources = Vec::new();
    let mut discovery = None;
    if options.discover {
        let processes = ProcScanner::new("/proc");
        sources.push("proc");
        let containers = if options.docker {
            match DockerScanner::from_environment(
                env::var("DOCKER_HOST").ok().as_deref(),
                env::var("XDG_RUNTIME_DIR").ok().as_deref(),
            ) {
                Ok(scanner) => {
                    sources.push("docker");
                    Some(scanner)
                }
                Err(note) => {
                    notes.push(note);
                    None
                }
            }
        } else {
            None
        };
        discovery = Some(Discovery::new(Some(processes), containers));
    }
    if targets {
        sources.push("--target");
    }
    if !options.read_only {
        sources.push("launch");
    }
    (discovery, notes, sources)
}
