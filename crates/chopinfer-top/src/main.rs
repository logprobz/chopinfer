#![forbid(unsafe_code)]

//! chopinfer-top: a live view of running chopinfer serve instances.

use std::env;
use std::io::IsTerminal;
use std::process::ExitCode;

use chopinfer_top::cli::{self, Options, USAGE};
use chopinfer_top::discover::Discovery;
use chopinfer_top::discover::docker::DockerScanner;
use chopinfer_top::discover::proc::ProcScanner;
use chopinfer_top::instances::Tone;
use chopinfer_top::ui::widgets::sanitize;
use chopinfer_top::{listing, targets, tui};

fn main() -> ExitCode {
    let options = match cli::parse(env::args().skip(1)) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("chopinfer-top: {error}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    if options.help {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if options.version {
        println!("chopinfer-top {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    let file = match &options.targets_file {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => Some(text),
            Err(error) => {
                eprintln!("chopinfer-top: cannot read {}: {error}", path.display());
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
            eprintln!("chopinfer-top: {error}");
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
        let (registry, now) = listing::collect(
            targets,
            discovery,
            notes,
            options.interval,
            bearer.as_deref(),
        );
        if options.json {
            println!("{}", listing::document(&registry, now));
        } else {
            print!("{}", listing::table(&registry, now));
            for event in &registry.events {
                if matches!(event.tone, Tone::Warning | Tone::Bad) {
                    eprintln!(
                        "chopinfer-top: {}: {}",
                        sanitize(&event.who),
                        sanitize(&event.what)
                    );
                }
            }
        }
        return ExitCode::SUCCESS;
    }
    if !std::io::stdout().is_terminal() {
        eprintln!("chopinfer-top: stdout is not a terminal; use --list, or --list --json");
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
    };
    match tui::run(setup) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("chopinfer-top: {error}");
            ExitCode::FAILURE
        }
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
    (discovery, notes, sources)
}
