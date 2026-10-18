//! `chopinfer eval list | run | capture | verify-references | files | declaration |
//! rescore`.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Value, json};

use crate::Host;
use crate::declaration::{self, Check, Loaded};
use crate::http::{BindError, Client, HttpError};
use crate::kind::{self, Capture, KindInfo, Status};
use crate::references::{self, ReferenceError, Verified};
use crate::report::{CheckReport, Exit, ReferenceUse, Report, Tool, utc_stamp};
use crate::{files, launch, locate};

pub const USAGE: &str = "\
eval list (--model DIR | --package DIR) [--variant NAME [--variant-dir DIR]] [--json]
eval run (--model DIR | --package DIR) [--variant NAME [--variant-dir DIR]] [--served-model NAME]
  [--suite quick|accuracy | --check ID... [--cohort quick|full]] [--references DIR]
  [--report DIR] [--json]
  [--timeout SECONDS] [--server-log LOG] (--url URL | --launch [SERVE FLAGS...])
eval capture --kind determinism|greedy_pin (--model DIR | --package DIR)
  [--variant NAME [--variant-dir DIR]] [--served-model NAME] [--references DIR]
  [--timeout SECONDS] [--server-log LOG] --url URL --report DIR RECORD ARGUMENTS...
  (one record a runner compares, not a declared check, written to REPORT/STAMP-CHECK.json:
  determinism --arm ARM --run-id RUN [--image] [--image-reason TEXT];
  greedy_pin --arm ARM [--k N] [--cold-cache] [--concurrency 1] [--prompt-tokens N]
  [--max-tokens N], over the texts of the reference pinned at the declared exact input)
eval verify-references (--model DIR | --package DIR) [--variant NAME [--variant-dir DIR]]
  [--references DIR] [--json]
eval files (--model DIR | --package DIR) [--variant NAME]
  (the files under ROOT eval reads, one per line)
eval declaration --model DIR [--variant NAME [--variant-dir DIR]]
  | eval declaration --root ROOT < DECLARATION.json
  (validates a declaration and prints it; --root reads ROOT/tools/qual/modules.json)
eval rescore < REQUEST.json   (offline verify, score, judge and limit calls; docs/cli.md)
--model DIR is a model directory ROOT/models/<model> holding qual.json (declared paths resolve
under ROOT) or a checkpoint directory, resolved as serve --model resolves it. --package DIR is a
package directory or an install root with one model. --variant NAME reads the model's
qual.NAME.json, which qualifies another checkpoint or serving configuration, in place of
qual.json. --variant-dir DIR (default $CHOPINFER_QUAL_VARIANTS) reads a variant the tree does not
declare from DIR/<model>/NAME.json, an external variant beside its lock. --launch runs this chopinfer's serve with the flags after it for the run, logging to
REPORT/server.log, and stops it after. --server-log LOG also binds the server to its startup log:
LOG must be the log of the process listening at the URL, naming the checkpoint and startup nonce
its /health reports. A check takes its declared arguments, then those its cohort adds: the suite's
cohort, or --cohort (default quick) with --check. A teacher check whose arguments hold
--capture-out writes its compact capture to REPORT/captures/ID.json.
exit: 0 pass, 1 a gate failed, 2 usage or declaration error, 3 insufficient evidence, 4 environment error";

/// A refusal before any check, with its exit.
#[derive(Debug)]
struct Refusal(Exit, String);

fn usage(message: impl Into<String>) -> Refusal {
    Refusal(Exit::Usage, format!("{}\n{USAGE}", message.into()))
}

/// Parsed options.
#[derive(Debug, Default)]
struct Options {
    model: Option<PathBuf>,
    package: Option<PathBuf>,
    variant: Option<String>,
    /// The external variant root (`--variant-dir`).
    variant_dir: Option<PathBuf>,
    url: Option<String>,
    served_model: Option<String>,
    suite: Option<String>,
    checks: Vec<String>,
    /// The cohort whose declared arguments checks named with `--check` take.
    cohort: Option<String>,
    references: Option<PathBuf>,
    report: Option<PathBuf>,
    /// The startup log the server is bound to as well.
    server_log: Option<PathBuf>,
    json: bool,
    timeout: Option<Duration>,
    /// The serve flags of `--launch`.
    launch: Option<Vec<String>>,
}

fn parse(command: &str, arguments: &[String]) -> Result<Options, Refusal> {
    let allowed: &[&str] = match command {
        "list" => &[
            "--model",
            "--package",
            "--variant",
            "--variant-dir",
            "--json",
        ],
        "run" => &[
            "--model",
            "--package",
            "--variant",
            "--variant-dir",
            "--url",
            "--served-model",
            "--suite",
            "--check",
            "--cohort",
            "--references",
            "--report",
            "--json",
            "--timeout",
            "--server-log",
            "--launch",
        ],
        "verify-references" => &[
            "--model",
            "--package",
            "--variant",
            "--variant-dir",
            "--references",
            "--json",
        ],
        "files" => &["--model", "--package", "--variant"],
        "capture" => &[
            "--model",
            "--package",
            "--variant",
            "--variant-dir",
            "--url",
            "--served-model",
            "--references",
            "--report",
            "--timeout",
            "--server-log",
        ],
        _ => return Err(usage(format!("unsupported eval command '{command}'"))),
    };
    let mut options = Options::default();
    let mut words = arguments.iter();
    while let Some(word) = words.next() {
        if !allowed.contains(&word.as_str()) {
            return Err(usage(format!("eval {command} does not take {word}")));
        }
        if word == "--json" {
            if options.json {
                return Err(usage("--json is given twice"));
            }
            options.json = true;
            continue;
        }
        if word == "--launch" {
            options.launch = Some(words.by_ref().cloned().collect());
            break;
        }
        let value = words
            .next()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| usage(format!("{word} needs a value")))?
            .clone();
        let twice = || usage(format!("{word} is given twice"));
        match word.as_str() {
            "--check" => options.checks.push(value),
            "--model" if options.model.is_none() => options.model = Some(value.into()),
            "--package" if options.package.is_none() => options.package = Some(value.into()),
            "--variant" if options.variant.is_none() => options.variant = Some(value),
            "--variant-dir" if options.variant_dir.is_none() => {
                options.variant_dir = Some(value.into())
            }
            "--url" if options.url.is_none() => options.url = Some(value),
            "--served-model" if options.served_model.is_none() => {
                options.served_model = Some(value)
            }
            "--suite" if options.suite.is_none() => {
                if value != "quick" && value != "accuracy" {
                    return Err(usage("--suite takes quick or accuracy"));
                }
                options.suite = Some(value);
            }
            "--cohort" if options.cohort.is_none() => {
                if value != "quick" && value != "full" {
                    return Err(usage("--cohort takes quick or full"));
                }
                options.cohort = Some(value);
            }
            "--references" if options.references.is_none() => {
                options.references = Some(value.into())
            }
            "--report" if options.report.is_none() => options.report = Some(value.into()),
            "--server-log" if options.server_log.is_none() => {
                options.server_log = Some(value.into())
            }
            "--timeout" if options.timeout.is_none() => {
                let timeout = value
                    .parse::<f64>()
                    .ok()
                    .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
                    .filter(|timeout| {
                        !timeout.is_zero() && Instant::now().checked_add(*timeout).is_some()
                    })
                    .ok_or_else(|| {
                        usage("--timeout takes a positive number of seconds in range")
                    })?;
                options.timeout = Some(timeout);
            }
            _ => return Err(twice()),
        }
    }
    match (&options.model, &options.package) {
        (None, None) => {
            return Err(usage(format!(
                "eval {command} requires --model DIR or --package DIR"
            )));
        }
        (Some(_), Some(_)) => {
            return Err(usage(
                "--model and --package name the model two ways; give one",
            ));
        }
        _ => {}
    }
    if options.variant_dir.is_some() && options.variant.is_none() {
        return Err(usage("--variant-dir selects an external --variant NAME"));
    }
    if command == "run" && options.url.is_some() == options.launch.is_some() {
        return Err(usage("eval run requires one of --url URL and --launch"));
    }
    if command == "capture" && (options.url.is_none() || options.report.is_none()) {
        return Err(usage("eval capture requires --url URL and --report DIR"));
    }
    if options.suite.is_some() && !options.checks.is_empty() {
        return Err(usage(
            "--suite and --check select checks two ways; give one",
        ));
    }
    if options.cohort.is_some() && options.checks.is_empty() {
        return Err(usage(
            "--cohort names the cohort of checks named with --check",
        ));
    }
    Ok(options)
}

fn load(options: &Options, host: &Host) -> Result<Loaded, Refusal> {
    let model = match (&options.model, &options.package) {
        (Some(model), _) => {
            locate::model(model, host).map_err(|error| Refusal(Exit::Usage, error))?
        }
        (None, Some(package)) => {
            locate::package(package, host).map_err(|error| Refusal(Exit::Usage, error))?
        }
        (None, None) => unreachable!("parse requires --model or --package"),
    };
    let variants = variant_root(options.variant_dir.as_deref());
    declaration::load_variant_from(&model, options.variant.as_deref(), variants.as_deref())
        .map_err(|error| Refusal(Exit::Usage, error.to_string()))
}

/// The external variant root: `--variant-dir`, else `$CHOPINFER_QUAL_VARIANTS`.
fn variant_root(given: Option<&Path>) -> Option<PathBuf> {
    given
        .map(Path::to_path_buf)
        .or_else(|| environment(declaration::VARIANTS_ENV).map(PathBuf::from))
}

fn environment(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// Runs `eval ARGUMENTS...` hosted by this process: writes its output and
/// returns its exit.
pub fn run(tool: &Tool, arguments: &[String]) -> Exit {
    run_hosted(tool, &Host::current(), arguments)
}

/// Runs `eval ARGUMENTS...` hosted by `host`: writes its output and returns
/// its exit.
pub fn run_hosted(tool: &Tool, host: &Host, arguments: &[String]) -> Exit {
    let Some((command, rest)) = arguments.split_first() else {
        eprintln!("eval requires a command\n{USAGE}");
        return Exit::Usage;
    };
    if matches!(command.as_str(), "--help" | "-h") {
        println!("{USAGE}");
        return Exit::Pass;
    }
    if command == "rescore" {
        return rescore(rest);
    }
    if command == "declaration" {
        return declaration_command(rest, host);
    }
    if command == "capture" {
        return match record_arguments(rest).and_then(|(kind, record, rest)| {
            let options = parse(command, &rest)?;
            capture_record(host, &options, kind, record)
        }) {
            Ok(exit) => exit,
            Err(Refusal(exit, message)) => {
                eprintln!("eval capture: {message}");
                exit
            }
        };
    }
    let outcome = parse(command, rest).and_then(|options| match command.as_str() {
        "list" => list(&options, host),
        "verify-references" => verify_references(&options, host),
        "files" => list_files(&options, host),
        _ => run_checks(tool, host, &options),
    });
    match outcome {
        Ok(exit) => exit,
        Err(Refusal(exit, message)) => {
            eprintln!("eval: {message}");
            exit
        }
    }
}

/// `eval declaration`: a model's declaration (`--model DIR [--variant
/// NAME]`), or the one on stdin validated with ROOT's module metadata
/// (`--root ROOT`), printed as validated; exit 2 with the refusal.
fn declaration_command(arguments: &[String], host: &Host) -> Exit {
    use std::io::Read as _;
    let mut options: [Option<&String>; 4] = [None; 4];
    let mut words = arguments.iter();
    while let Some(word) = words.next() {
        let slot = match word.as_str() {
            "--model" => 0,
            "--variant" => 1,
            "--root" => 2,
            "--variant-dir" => 3,
            _ => {
                eprintln!("eval: eval declaration does not take {word}\n{USAGE}");
                return Exit::Usage;
            }
        };
        match words.next().filter(|value| !value.is_empty()) {
            Some(value) if options[slot].is_none() => options[slot] = Some(value),
            _ => {
                eprintln!("eval: {word} needs one value\n{USAGE}");
                return Exit::Usage;
            }
        }
    }
    let value = match options {
        [Some(model), variant, None, variants] if variant.is_some() || variants.is_none() => {
            let variants = variant_root(variants.map(Path::new));
            locate::model(Path::new(model), host)
                .map_err(|error| error.to_string())
                .and_then(|model| {
                    declaration::load_variant_from(
                        &model,
                        variant.map(String::as_str),
                        variants.as_deref(),
                    )
                    .map_err(|error| error.to_string())
                })
                .map(|loaded| loaded.value)
        }
        [None, None, Some(root), None] => {
            let mut bytes = Vec::new();
            std::io::stdin()
                .read_to_end(&mut bytes)
                .map_err(|error| format!("stdin: {error}"))
                .and_then(|_| {
                    crate::json::parse(&bytes).map_err(|error| format!("the declaration: {error}"))
                })
                .and_then(|value| {
                    declaration::Modules::read(Path::new(root))
                        .and_then(|modules| declaration::validate(&value, &modules))
                        .map(|_| value)
                        .map_err(|error| error.to_string())
                })
        }
        _ => {
            eprintln!(
                "eval: eval declaration takes --model DIR [--variant NAME [--variant-dir DIR]] or --root ROOT\n{USAGE}"
            );
            return Exit::Usage;
        }
    };
    match value {
        Ok(value) => {
            println!("{}", String::from_utf8_lossy(&crate::json::compact(&value)));
            Exit::Pass
        }
        Err(message) => {
            eprintln!("eval declaration: {message}");
            Exit::Usage
        }
    }
}

/// `eval rescore`: one JSON request on stdin, its response on stdout.
fn rescore(arguments: &[String]) -> Exit {
    use std::io::Read as _;
    if let Some(word) = arguments.first() {
        eprintln!("eval: eval rescore does not take {word}\n{USAGE}");
        return Exit::Usage;
    }
    let mut bytes = Vec::new();
    let response = std::io::stdin()
        .read_to_end(&mut bytes)
        .map_err(|error| format!("stdin: {error}"))
        .and_then(|_| crate::json::parse(&bytes).map_err(|error| format!("the request: {error}")))
        .and_then(|request| crate::rescore::respond(&request));
    match response {
        Ok(response) => {
            println!(
                "{}",
                String::from_utf8_lossy(&crate::json::compact(&response))
            );
            Exit::Pass
        }
        Err(message) => {
            eprintln!("eval rescore: {message}");
            Exit::Usage
        }
    }
}

/// The reference status rows of every pinned reference.
fn reference_rows(loaded: &Loaded, references: Option<&Path>) -> (Vec<Value>, bool) {
    let mut verified_all = true;
    let mut rows = Vec::new();
    let lock = references::checkpoint_lock(loaded);
    if let Err(error) = &lock {
        verified_all = false;
        rows.push(
            json!({"name": "checkpoint lock", "path": loaded.declaration.checkpoint.lock,
                         "status": status_of(error), "error": error.to_string()}),
        );
    }
    for pin in &loaded.declaration.references {
        let mut row = json!({"name": pin.name, "path": loaded.value["references"][&pin.name]["path"], "sha256": pin.sha256,
                             "schema": pin.schema, "variants": pin.variants});
        match references::pinned(loaded, pin, references) {
            Ok(verified) => {
                row["status"] = json!("verified");
                row["binding"] = verified.binding.to_json();
            }
            Err(error) => {
                verified_all = false;
                row["status"] = json!(status_of(&error));
                row["error"] = json!(error.to_string());
            }
        }
        rows.push(row);
    }
    (rows, verified_all)
}

fn status_of(error: &ReferenceError) -> &'static str {
    match error {
        ReferenceError::Missing(_) => "missing",
        ReferenceError::Unverified(_) => "unverified",
        ReferenceError::Stale(..) => "stale",
    }
}

fn print_reference_rows(rows: &[Value]) {
    for row in rows {
        let binding = row["binding"]["status"].as_str().unwrap_or("");
        print!(
            "  {:<16} {:<10} {}",
            row["name"].as_str().unwrap_or_default(),
            row["status"].as_str().unwrap_or_default(),
            row["path"].as_str().unwrap_or_default()
        );
        if !binding.is_empty() {
            print!(" ({binding})");
        }
        if let Some(error) = row["error"].as_str() {
            print!(": {error}");
        }
        println!();
    }
}

fn kind_of(check: &Check) -> Option<&'static KindInfo> {
    kind::shared_kind(&check.module)
}

fn list(options: &Options, host: &Host) -> Result<Exit, Refusal> {
    let loaded = load(options, host)?;
    let declaration = &loaded.declaration;
    let (references, _) = reference_rows(&loaded, None);
    let checks: Vec<Value> = declaration
        .checks
        .iter()
        .map(|check| {
            let kind = kind_of(check);
            let cohorts: Vec<&str> = declaration
                .cohorts
                .iter()
                .filter(|(_, ids)| ids.contains(&check.id))
                .map(|(name, _)| name.as_str())
                .collect();
            json!({"id": check.id, "module": check.module, "kind": kind.map(|kind| kind.name),
                   "version": kind.map(|kind| kind.version), "cohorts": cohorts,
                   "reference": kind.and_then(|kind| kind.reference),
                   "informational_reason": check.informational_reason})
        })
        .collect();
    if options.json {
        let value = json!({"model": declaration.model, "declaration": loaded.path,
                           "declaration_sha256": loaded.sha256, "checks": checks,
                           "references": references});
        println!(
            "{}",
            serde_json::to_string_pretty(&value).expect("JSON values serialize")
        );
        return Ok(Exit::Pass);
    }
    println!("{} ({})", declaration.model, loaded.path.display());
    println!("checks:");
    for check in &checks {
        println!(
            "  {:<20} {:<16} cohorts {}{}",
            check["id"].as_str().unwrap_or_default(),
            check["kind"].as_str().unwrap_or("tools/qual only"),
            check["cohorts"]
                .as_array()
                .map(|cohorts| cohorts
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(","))
                .unwrap_or_default(),
            if check["informational_reason"].is_null() {
                ""
            } else {
                " (informational)"
            }
        );
    }
    println!("references:");
    print_reference_rows(&references);
    Ok(Exit::Pass)
}

fn verify_references(options: &Options, host: &Host) -> Result<Exit, Refusal> {
    let loaded = load(options, host)?;
    let (rows, verified) = reference_rows(&loaded, options.references.as_deref());
    let exit = if verified {
        Exit::Pass
    } else {
        Exit::Insufficient
    };
    if options.json {
        let value = json!({"model": loaded.declaration.model, "references": rows,
                           "verified": verified});
        println!(
            "{}",
            serde_json::to_string_pretty(&value).expect("JSON values serialize")
        );
    } else {
        println!(
            "{} references: {}",
            loaded.declaration.model,
            if verified { "verified" } else { "NOT verified" }
        );
        print_reference_rows(&rows);
    }
    Ok(exit)
}

/// `eval files`: what an install ships for the model, relative to its root.
fn list_files(options: &Options, host: &Host) -> Result<Exit, Refusal> {
    let loaded = load(options, host)?;
    let files = files::files(&loaded).map_err(|error| Refusal(Exit::Insufficient, error))?;
    for file in files {
        println!("{file}");
    }
    Ok(Exit::Pass)
}

/// The check as its cohort runs it (`release_gate.run_check`): its declared
/// arguments, then those the cohort adds (`cohort_args`).
fn in_cohort(check: &Check, cohort: &str) -> Check {
    let mut check = check.clone();
    let added = check.declared["cohort_args"][cohort].as_array().cloned();
    if let (Some(added), Some(arguments)) = (added, check.declared["args"].as_array_mut()) {
        arguments.extend(added);
    }
    check
}

/// The checks a run selects, each with its cohort's arguments: those named
/// with `--check` (in `--cohort`, default quick), else the shared kinds'
/// checks of the suite's cohort (quick: `quick`, accuracy: `full`).
fn selected(loaded: &Loaded, options: &Options) -> Result<(String, Vec<Check>), Refusal> {
    let (suite, checks) = selected_declared(loaded, options)?;
    let cohort = match (suite.as_str(), &options.cohort) {
        ("accuracy", _) => "full",
        (_, Some(cohort)) => cohort.as_str(),
        _ => "quick",
    };
    let checks = checks
        .into_iter()
        .map(|check| in_cohort(check, cohort))
        .collect();
    Ok((suite, checks))
}

fn selected_declared<'a>(
    loaded: &'a Loaded,
    options: &Options,
) -> Result<(String, Vec<&'a Check>), Refusal> {
    let declaration = &loaded.declaration;
    if !options.checks.is_empty() {
        let mut checks: Vec<&Check> = Vec::new();
        for id in &options.checks {
            let check = declaration
                .check(id)
                .ok_or_else(|| usage(format!("{} declares no check {id}", declaration.model)))?;
            if kind_of(check).is_none() {
                return Err(usage(format!(
                    "check {id} ({}) is not a shared kind; run it with tools/qual",
                    check.module
                )));
            }
            if !checks.iter().any(|seen| seen.id == check.id) {
                checks.push(check);
            }
        }
        return Ok(("checks".into(), checks));
    }
    let suite = options.suite.clone().unwrap_or_else(|| "quick".into());
    let cohort = if suite == "quick" { "quick" } else { "full" };
    let ids = declaration
        .cohort(cohort)
        .ok_or_else(|| usage(format!("{} declares no {cohort} cohort", declaration.model)))?;
    let checks = ids
        .iter()
        .filter_map(|id| declaration.check(id))
        .filter(|check| kind_of(check).is_some())
        .collect();
    Ok((suite, checks))
}

/// The report directory: `--report DIR`, else a new directory under the
/// declared receipts root.
fn report_directory(
    loaded: &Loaded,
    options: &Options,
    suite: &str,
    at: &str,
) -> Result<PathBuf, Refusal> {
    if let Some(directory) = &options.report {
        return Ok(directory.clone());
    }
    let root = references::input_path(loaded, "receipts", &environment).ok_or_else(|| {
        usage("the declared receipts directory needs its environment variable; pass --report DIR")
    })?;
    Ok(root.join(format!("{at}-eval-{}-{suite}", loaded.declaration.model)))
}

fn run_checks(tool: &Tool, host: &Host, options: &Options) -> Result<Exit, Refusal> {
    let started = Instant::now();
    let loaded = load(options, host)?;
    let (suite, checks) = selected(&loaded, options)?;
    let at = utc_stamp(SystemTime::now());
    let directory = report_directory(&loaded, options, &suite, &at)?;
    if directory.exists() {
        return Err(usage(format!("{} already exists", directory.display())));
    }
    // Every deadline must be representable before any request.
    for check in &checks {
        if declared_timeout(check).is_none() {
            return Err(Refusal(
                Exit::Usage,
                format!("checks.{}.timeout_seconds is out of range", check.id),
            ));
        }
    }
    let timeout = options.timeout.unwrap_or(Duration::from_secs(900));
    let url = match (&options.url, &options.launch) {
        (Some(url), _) => url.clone(),
        (None, Some(flags)) => launch::url(flags, host.serve_listen.as_deref()).map_err(usage)?,
        (None, None) => unreachable!("parse requires --url or --launch"),
    };
    let url = url.as_str();
    let mut client = Client::new(url, timeout, 3).map_err(usage)?;
    let mut report = Report {
        tool: tool.clone(),
        model: loaded.declaration.model.clone(),
        declaration_sha256: loaded.sha256.clone(),
        variant_source: crate::lock::variant_source(&loaded)
            .map_err(|error| Refusal(Exit::Usage, error))?,
        suite,
        at,
        server: json!({"url": url}),
        checks: Vec::new(),
        seconds: 0.0,
        failure: None,
    };
    if checks.is_empty() {
        report.fail_run(
            Exit::Insufficient,
            "the selection holds no shared-kind check".into(),
        );
    }

    // Every reference a selected check reads, verified before any request.
    let implemented = kind::implemented();
    let reference_name = |check: &Check| -> Option<String> {
        match implemented
            .iter()
            .find(|kind| kind_of(check).is_some_and(|info| info.name == kind.info().name))
        {
            Some(kind) => kind.reference_name(check),
            None => kind_of(check).and_then(|info| info.reference.map(String::from)),
        }
    };
    let mut verified: Vec<(String, Result<Verified, ReferenceError>)> = Vec::new();
    for check in &checks {
        let Some(name) = reference_name(check) else {
            continue;
        };
        if verified.iter().any(|(seen, _)| *seen == name) {
            continue;
        }
        let result = match loaded.declaration.reference(&name) {
            Some(pin) => references::pinned(&loaded, pin, options.references.as_deref()),
            None => Err(ReferenceError::Missing(format!(
                "{} pins no {name} reference",
                loaded.declaration.model
            ))),
        };
        verified.push((name, result));
    }

    // A launched server, ready before it is bound.
    let mut server = None;
    let mut launch_failure = None;
    if let Some(flags) = &options.launch {
        let program = std::env::var_os(launch::SERVE_BINARY)
            .map_or_else(|| host.executable.clone(), PathBuf::from);
        let log = directory.join("server.log");
        let started = std::fs::create_dir_all(&directory)
            .map_err(|error| format!("{}: {error}", directory.display()))
            .and_then(|()| launch::Server::start(&program, flags, &log));
        match started {
            Ok(mut started) => {
                launch_failure = started.wait_ready(url, timeout).err();
                server = Some(started);
            }
            Err(error) => {
                report.server["launch"] = json!({"command": launch::command(&program, flags),
                                                 "log": log});
                launch_failure = Some(error);
            }
        }
    }

    // The server, bound to the pinned checkpoint. A failure here is the
    // run's, whatever the checks' statuses.
    let identity = match launch_failure {
        Some(error) => Err(HttpError::Transport(error)),
        None => client.identity(options.served_model.as_deref()),
    };
    let bound = match identity {
        Ok(identity) => {
            report.server = json!({"url": url, "identity": identity.to_json()});
            let bound = client
                .bind(&identity, &loaded.declaration.checkpoint)
                .and_then(|bound| match &options.server_log {
                    Some(log) => bound.with_log(log, url),
                    None => Ok(bound),
                });
            match bound {
                Ok(bound) => {
                    report.server["checkpoint"] = bound.to_json();
                    Ok((identity, bound))
                }
                Err(error) => {
                    report.fail_run(bind_exit(&error), error.to_string());
                    Err(error)
                }
            }
        }
        Err(error) => {
            let (exit, error) = match error {
                HttpError::Transport(_) => (
                    Exit::Environment,
                    BindError::Unidentified(error.to_string()),
                ),
                HttpError::Unavailable(_) => (Exit::Fail, BindError::Mismatch(error.to_string())),
                _ => (
                    Exit::Insufficient,
                    BindError::Unidentified(format!(
                        "the server's identity could not be read: {error}"
                    )),
                ),
            };
            report.fail_run(exit, error.to_string());
            Err(error)
        }
    };
    // The compact captures the checks composed, by their report entry.
    let mut composed = Vec::new();
    for check in &checks {
        let info = kind_of(check).expect("selected checks are shared kinds");
        let without = |status: Status, reason: &str| {
            CheckReport::without_verdict(&check.id, info.name, info.version, status, reason)
        };
        if let Some(reason) = &check.informational_reason {
            report.checks.push(without(Status::Informational, reason));
            continue;
        }
        let reference = match reference_name(check) {
            Some(name) => match verified
                .iter()
                .find(|(seen, _)| *seen == name)
                .map(|(_, result)| result)
            {
                Some(Ok(reference)) => Some(reference),
                Some(Err(error)) => {
                    report.checks.push(without(
                        Status::Insufficient,
                        &format!("reference {name}: {error}"),
                    ));
                    continue;
                }
                None => None,
            },
            None => None,
        };
        let (identity, bound) = match &bound {
            Ok(pair) => pair,
            Err(error) => {
                let status = match report.failure.as_ref().map(|failure| failure.exit) {
                    Some(Exit::Fail) => Status::Fail,
                    Some(Exit::Environment) => Status::NotRun,
                    _ => Status::Insufficient,
                };
                report.checks.push(without(status, &error.to_string()));
                continue;
            }
        };
        let Some(kind) = implemented
            .iter()
            .find(|kind| kind.info().name == info.name)
        else {
            report.checks.push(without(
                Status::NotRun,
                &format!("this build does not implement the {} kind", info.name),
            ));
            continue;
        };
        let check_started = Instant::now();
        let Some(deadline) =
            declared_timeout(check).and_then(|timeout| check_started.checked_add(timeout))
        else {
            report.checks.push(without(
                Status::Fail,
                "the declared timeout is out of range",
            ));
            continue;
        };
        let mut capture = Capture {
            client: &mut client,
            declaration: &loaded,
            check,
            server: identity,
            bound,
            reference,
            deadline,
            host_load_start: crate::capture::host_load::sample(),
            mechanics: loaded.declaration.mechanics_only,
            capture_out: crate::capture::flag(check, "--capture-out").then(|| {
                kind::CompactCapture {
                    path: directory
                        .join("captures")
                        .join(format!("{}.json", check.id)),
                    bytes: None,
                }
            }),
        };
        let captured = kind.capture(&mut capture).and_then(|receipt| {
            crate::capture::remaining(&mut capture)?;
            Ok(receipt)
        });
        capture.client.set_deadline(None);
        // Only a receipt's own capture is published, and only later.
        if let Some(out) = capture.capture_out.take()
            && let (Ok(_), Some(bytes)) = (&captured, out.bytes)
        {
            composed.push((report.checks.len(), out.path, bytes));
        }
        let mut entry = match captured {
            Ok(receipt) => {
                let rescored = match kind.reference(check, reference) {
                    Ok(read) => kind::rescore(
                        kind.as_ref(),
                        &receipt,
                        read.as_ref(),
                        check,
                        loaded.declaration.mechanics_only,
                    ),
                    Err(error) => kind::Rescored {
                        reason: None,
                        status: Status::Insufficient,
                        metrics: None,
                        predicates: None,
                        recorded_fields: Vec::new(),
                        error: Some(format!("reference: {error}")),
                    },
                };
                let mut entry = without(rescored.status, rescored.error.as_deref().unwrap_or(""));
                entry.error = rescored.error;
                entry.record = Some(receipt);
                if let Some(metrics) = &rescored.metrics {
                    entry.limits = metrics.get("limits").cloned();
                    entry.spread = metrics.get("reference_spread").cloned();
                }
                entry.metrics = rescored.metrics;
                entry.recorded_fields = rescored.recorded_fields;
                entry.predicates = rescored
                    .predicates
                    .unwrap_or_default()
                    .into_iter()
                    .map(|judged| judged.to_json())
                    .collect();
                entry
            }
            Err(kind::KindError::Environment(message)) => {
                report.fail_run(Exit::Environment, message.clone());
                without(Status::NotRun, &message)
            }
            Err(kind::KindError::Insufficient(message)) => without(Status::Insufficient, &message),
            Err(kind::KindError::Invalid(message)) => without(Status::Fail, &message),
            Err(kind::KindError::Unsupported(message)) => without(Status::NotRun, &message),
        };
        entry.seconds = check_started.elapsed().as_secs_f64();
        if let Some(reference) = reference {
            entry.references.push(ReferenceUse {
                name: reference.name.clone(),
                sha256: reference.sha256.clone(),
                binding: reference.binding.to_json(),
            });
        }
        report.checks.push(entry);
    }
    // A run bound to the server stands only if the same process wrote every
    // response and still serves: a changed checkpoint or another process
    // fails the run, and no check of it passes.
    if let Ok((identity, bound)) = &bound
        && let Err(error) = client.confirm_live(&identity.model, bound)
    {
        report.fail_run(bind_exit(&error), format!("refused: {error}"));
        for check in &mut report.checks {
            if matches!(check.status, Status::Pass) {
                check.status = Status::Insufficient;
                check.error = Some(format!("refused: {error}"));
            }
        }
    } else {
        // Confirmed live: the compact captures stand and are published,
        // each a new file. A refused run never reaches this, so it leaves
        // no capture a later comparison of arms could admit.
        for (index, path, bytes) in composed {
            if let Err(error) = publish(&path, &bytes) {
                let message = format!("{}: {error}", path.display());
                report.fail_run(Exit::Environment, message.clone());
                let check = &mut report.checks[index];
                check.status = Status::NotRun;
                check.error = Some(message);
            }
        }
    }
    if let Some(server) = server {
        report.server["launch"] = server.stop();
    }
    report.seconds = started.elapsed().as_secs_f64();
    let (path, sha256) = report
        .write(&directory)
        .map_err(|error| Refusal(Exit::Environment, error))?;
    if options.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report.to_json()).expect("JSON values serialize")
        );
    } else {
        print!("{}", report.summary());
        println!("report {} sha256 {sha256}", path.display());
    }
    Ok(report.exit())
}

/// A record kind of `eval capture`: its name and the record arguments it
/// takes, with a value and without.
struct RecordKind {
    name: &'static str,
    options: &'static [&'static str],
    flags: &'static [&'static str],
}

/// The records a runner compares that are not declared checks: an arm's
/// determinism record and greedy record.
const RECORD_KINDS: [RecordKind; 2] = [
    RecordKind {
        name: "determinism",
        options: &["--arm", "--run-id", "--image-reason"],
        flags: &["--image"],
    },
    RecordKind {
        name: "greedy_pin",
        options: &[
            "--arm",
            "--k",
            "--concurrency",
            "--prompt-tokens",
            "--max-tokens",
        ],
        flags: &["--cold-cache"],
    },
];

/// `eval capture`'s words split into the record kind, the record's own
/// arguments and the rest, eval's options.
fn record_arguments(
    arguments: &[String],
) -> Result<(&'static RecordKind, Vec<String>, Vec<String>), Refusal> {
    let named = arguments
        .iter()
        .position(|word| word == "--kind")
        .and_then(|index| arguments.get(index + 1))
        .ok_or_else(|| usage("eval capture requires --kind determinism|greedy_pin"))?;
    let kind = RECORD_KINDS
        .iter()
        .find(|kind| kind.name == named)
        .ok_or_else(|| {
            usage(format!(
                "eval capture records determinism or greedy_pin, not {named}; run declared checks \
                 with eval run --check"
            ))
        })?;
    let (mut record, mut rest) = (Vec::new(), Vec::new());
    let mut seen_kind = false;
    let mut words = arguments.iter();
    while let Some(word) = words.next() {
        if word == "--kind" {
            if seen_kind {
                return Err(usage("--kind is given twice"));
            }
            seen_kind = true;
            words.next();
            continue;
        }
        let takes_value = kind.options.contains(&word.as_str());
        if !takes_value && !kind.flags.contains(&word.as_str()) {
            rest.push(word.clone());
            if let Some(value) = words.next() {
                rest.push(value.clone());
            }
            continue;
        }
        if record.contains(word) {
            return Err(usage(format!("{word} is given twice")));
        }
        record.push(word.clone());
        if takes_value {
            let value = words
                .next()
                .ok_or_else(|| usage(format!("{word} needs a value")))?;
            record.push(value.clone());
        }
    }
    Ok((kind, record, rest))
}

/// `eval capture`: one record against the server at `--url`, bound to the
/// pinned checkpoint (and to `--server-log`), confirmed live after its last
/// request and written to the report directory as the Python record writes
/// its receipt (`common.receipt`).
fn capture_record(
    host: &Host,
    options: &Options,
    record: &RecordKind,
    arguments: Vec<String>,
) -> Result<Exit, Refusal> {
    let loaded = load(options, host)?;
    let info = kind::SHARED_KINDS
        .iter()
        .find(|info| info.name == record.name)
        .expect("record kinds are shared kinds");
    let implemented = kind::implemented();
    let kind = implemented
        .iter()
        .find(|kind| kind.info().name == record.name)
        .expect("this build captures its record kinds");
    let check = Check {
        id: format!("{}-record", record.name),
        module: info.modules[0].into(),
        timeout_seconds: RECORD_SECONDS,
        rules: Vec::new(),
        informational_reason: None,
        declared: json!({"module": info.modules[0], "args": arguments}),
    };
    // A greedy record runs the texts of the reference pinned where the
    // declared exact input points, verified before any request.
    let reference = if record.name == "greedy_pin" {
        let pin = references::pinned_input(&loaded, "exact").ok_or_else(|| {
            Refusal(
                Exit::Insufficient,
                format!(
                    "{} pins no reference at its exact input",
                    loaded.declaration.model
                ),
            )
        })?;
        let verified =
            references::pinned(&loaded, pin, options.references.as_deref()).map_err(|error| {
                Refusal(
                    Exit::Insufficient,
                    format!("reference {}: {error}", pin.name),
                )
            })?;
        Some(verified)
    } else {
        None
    };
    let url = options.url.as_deref().expect("parse requires --url");
    let directory = options.report.as_deref().expect("parse requires --report");
    let timeout = options.timeout.unwrap_or(Duration::from_secs(900));
    let mut client = Client::new(url, timeout, 3).map_err(usage)?;
    let identity =
        client
            .identity(options.served_model.as_deref())
            .map_err(|error| match error {
                HttpError::Transport(_) => Refusal(Exit::Environment, error.to_string()),
                HttpError::Unavailable(_) => Refusal(Exit::Fail, error.to_string()),
                _ => Refusal(
                    Exit::Insufficient,
                    format!("the server's identity could not be read: {error}"),
                ),
            })?;
    let bound = client
        .bind(&identity, &loaded.declaration.checkpoint)
        .and_then(|bound| match &options.server_log {
            Some(log) => bound.with_log(log, url),
            None => Ok(bound),
        })
        .map_err(|error| Refusal(bind_exit(&error), error.to_string()))?;
    let mut capture = Capture {
        client: &mut client,
        declaration: &loaded,
        check: &check,
        server: &identity,
        bound: &bound,
        reference: reference.as_ref(),
        deadline: Instant::now() + Duration::from_secs_f64(RECORD_SECONDS),
        host_load_start: crate::capture::host_load::sample(),
        mechanics: loaded.declaration.mechanics_only,
        capture_out: None,
    };
    let receipt = kind.capture(&mut capture).map_err(|error| match error {
        kind::KindError::Environment(message) => Refusal(Exit::Environment, message),
        kind::KindError::Insufficient(message) | kind::KindError::Unsupported(message) => {
            Refusal(Exit::Insufficient, message)
        }
        kind::KindError::Invalid(message) => Refusal(Exit::Fail, message),
    })?;
    // The record stands only if the bound process wrote every response.
    client
        .confirm_live(&identity.model, &bound)
        .map_err(|error| Refusal(bind_exit(&error), format!("refused: {error}")))?;
    let name = format!(
        "{}-{}.json",
        receipt["at"].as_str().unwrap_or_default(),
        receipt["check"].as_str().unwrap_or_default()
    );
    let path = directory.join(name);
    let bytes = crate::json::receipt_text(&receipt);
    let written = std::fs::create_dir_all(directory).and_then(|()| {
        use std::io::Write as _;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?
            .write_all(&bytes)
    });
    written.map_err(|error| Refusal(Exit::Environment, format!("{}: {error}", path.display())))?;
    println!(
        "record {} sha256 {}",
        path.display(),
        crate::json::sha256_hex(&bytes)
    );
    Ok(Exit::Pass)
}

/// Writes `bytes` to `path`, a new file.
fn publish(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)
}

/// The time a record may take; the runner that asks for it bounds it too.
const RECORD_SECONDS: f64 = 3600.0;

/// The exit a failed server binding gives the run.
fn bind_exit(error: &BindError) -> Exit {
    match error {
        BindError::Mismatch(_) => Exit::Fail,
        BindError::Unidentified(_) => Exit::Insufficient,
    }
}

/// A check's declared timeout, `None` when it is out of range.
fn declared_timeout(check: &Check) -> Option<Duration> {
    Duration::try_from_secs_f64(check.timeout_seconds)
        .ok()
        .filter(|timeout| Instant::now().checked_add(*timeout).is_some())
}
