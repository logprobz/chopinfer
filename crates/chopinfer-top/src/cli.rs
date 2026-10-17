//! Command-line options.

use std::path::PathBuf;

pub const USAGE: &str = "\
usage: chopinfer-top [--target HOST:PORT]... [--targets-file FILE]
                     [--no-discover] [--no-docker] [--interval SECONDS]
                     [--rescan SECONDS] [--api-key KEY] [--no-gpus]
                     [--no-gpu-counters] [--list [--json]]

A live view of running chopinfer serve instances and this host's GPUs. It
reads only /health and /metrics from servers, and the GPUs through NVML.

  --target HOST:PORT   Watch this server; repeatable. Also http://HOST:PORT,
                       or :PORT for 127.0.0.1. CHOPINFER_TOP_TARGETS adds a
                       comma-separated list.
  --targets-file FILE  One target per line; # starts a comment.
  --no-discover        Do not look for serve processes or containers.
  --no-docker          Do not ask Docker for containers running serve.
  --interval SECONDS   Seconds between polls of each instance, 0.5 to 60
                       (default 1).
  --rescan SECONDS     Seconds between discovery scans, 2 to 300 (default 15).
  --api-key KEY        Bearer token for explicit targets behind an
                       authenticating proxy (default CHOPINFER_API_KEY).
  --no-gpus            Do not read this host's GPUs.
  --no-gpu-counters    Do not sample GPM counters (SM, tensor, DRAM) in the
                       GPU view.
  --list               Print the instances once and exit; with --json, as a
                       chopinfer.top.list.v1 document.
  -h, --help           Print this help.
  --version            Print the version.

Keys: arrows or j/k select, Enter opens, Esc returns, Tab switches, g shows
the GPUs, w changes the window, r rescans, p pauses, ? shows help, q quits.
";

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub targets: Vec<String>,
    pub targets_file: Option<PathBuf>,
    pub discover: bool,
    pub docker: bool,
    pub interval: f64,
    pub rescan: f64,
    pub api_key: Option<String>,
    pub gpus: bool,
    pub gpu_counters: bool,
    pub list: bool,
    pub json: bool,
    pub help: bool,
    pub version: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            targets_file: None,
            discover: true,
            docker: true,
            interval: 1.0,
            rescan: 15.0,
            api_key: None,
            gpus: true,
            gpu_counters: true,
            list: false,
            json: false,
            help: false,
            version: false,
        }
    }
}

/// Parses arguments after the program name.
pub fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut options = Options::default();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let (flag, inline) = match argument.split_once('=') {
            Some((flag, value)) if flag.starts_with("--") => {
                (flag.to_owned(), Some(value.to_owned()))
            }
            _ => (argument.clone(), None),
        };
        let mut value = |name: &str| {
            inline
                .clone()
                .or_else(|| arguments.next())
                .ok_or_else(|| format!("{name} needs a value"))
        };
        let seconds = |name: &str, text: String, range: (f64, f64)| {
            text.parse::<f64>()
                .ok()
                .filter(|seconds| (range.0..=range.1).contains(seconds))
                .ok_or_else(|| {
                    format!(
                        "{name} must be a number of seconds from {} to {}",
                        range.0, range.1
                    )
                })
        };
        match flag.as_str() {
            "--target" => options.targets.push(value("--target")?),
            "--targets-file" => {
                options.targets_file = Some(PathBuf::from(value("--targets-file")?))
            }
            "--no-discover" => options.discover = false,
            "--no-docker" => options.docker = false,
            "--interval" => {
                options.interval = seconds("--interval", value("--interval")?, (0.5, 60.0))?
            }
            "--rescan" => options.rescan = seconds("--rescan", value("--rescan")?, (2.0, 300.0))?,
            "--api-key" => options.api_key = Some(value("--api-key")?),
            "--no-gpus" => options.gpus = false,
            "--no-gpu-counters" => options.gpu_counters = false,
            "--list" => options.list = true,
            "--json" => options.json = true,
            "-h" | "--help" => options.help = true,
            "--version" => options.version = true,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if options.json && !options.list {
        return Err("--json goes with --list".to_owned());
    }
    Ok(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(text: &str) -> Result<Options, String> {
        parse(text.split_whitespace().map(str::to_owned))
    }

    #[test]
    fn reads_every_option() {
        let options = parsed(
            "--target :8311 --target=gpu:8310 --targets-file t.txt --no-discover --no-docker \
             --interval 0.5 --rescan=30 --api-key k --no-gpus --no-gpu-counters --list --json",
        )
        .unwrap();
        assert_eq!(options.targets, [":8311", "gpu:8310"]);
        assert_eq!(options.targets_file, Some(PathBuf::from("t.txt")));
        assert!(!options.discover && !options.docker && options.list && options.json);
        assert!(!options.gpus && !options.gpu_counters);
        assert_eq!((options.interval, options.rescan), (0.5, 30.0));
        assert_eq!(options.api_key.as_deref(), Some("k"));
        assert_eq!(parsed("").unwrap(), Options::default());
    }

    #[test]
    fn refuses_bad_values_and_unknown_flags() {
        for (text, message) in [
            ("--interval 0.1", "from 0.5 to 60"),
            ("--rescan 1", "from 2 to 300"),
            ("--interval fast", "from 0.5 to 60"),
            ("--target", "needs a value"),
            ("--json", "goes with --list"),
            ("--verbose", "unknown argument"),
        ] {
            let error = parsed(text).unwrap_err();
            assert!(error.contains(message), "{text}: {error}");
        }
    }
}
