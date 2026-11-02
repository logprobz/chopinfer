//! What `chopinfer` can serve, as the loader asks it: every serve option,
//! the installed models with their profiles, what `serve --model` picks for
//! a checkpoint, the final command line for a choice, and the options a
//! package declares. choptop links no workspace crate, so it runs
//! `chopinfer` and reads the JSON document each command prints.
//!
//! Commands run on background threads, as the pollers do: one lane for the
//! catalog, one for the command line and one for package options, so a slow
//! package preparation holds back nothing else. The command and package
//! lanes run only the latest request waiting for them.

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::Read;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Map, Value};

use crate::ui::widgets::sanitize;

/// How long a catalog or command request may run.
const CATALOG_TIMEOUT: Duration = Duration::from_secs(120);
/// How long reading package options may run: the first read prepares the
/// package in the model cache.
const PACKAGE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
/// Most bytes read from a command's output.
const OUTPUT_LIMIT: u64 = 16 * 1024 * 1024;
/// Longest error message kept from a command's stderr.
const MESSAGE_LIMIT: usize = 400;
/// The schema of the remembered checkpoints file.
pub const CHECKPOINTS_SCHEMA: &str = "choptop.checkpoints.v1";

type Object = Map<String, Value>;

/// The shape of an option's value, as `--describe-options` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// `N`: a whole number.
    Count,
    /// `F`: a number.
    Number,
    Text,
    Directory,
    File,
    Json,
    /// `none`: a switch that takes no value.
    Switch,
}

impl Shape {
    fn parse(text: &str) -> Result<Self, String> {
        Ok(match text {
            "N" => Self::Count,
            "F" => Self::Number,
            "TEXT" => Self::Text,
            "DIR" => Self::Directory,
            "FILE" => Self::File,
            "JSON" => Self::Json,
            "none" => Self::Switch,
            other => return Err(format!("unknown value shape {other:?}")),
        })
    }
}

/// One serve option.
#[derive(Clone, Debug, PartialEq)]
pub struct ServeOption {
    /// `--max-sequences`.
    pub flag: String,
    pub shape: Shape,
    pub choices: Option<Vec<String>>,
    /// `model`, `scheduling`, `memory`, `speculation`, `prefill`, `cache`,
    /// `server`, `media`, `logging` or `debug`.
    pub group: String,
    pub help: String,
    /// Whether it applies to `serve --model`.
    pub model_launch: bool,
    pub repeatable: bool,
}

impl ServeOption {
    /// The flag without its dashes, as effective settings name it.
    pub fn name(&self) -> &str {
        self.flag.trim_start_matches('-')
    }
}

/// An environment variable serve reads that an operator sets.
#[derive(Clone, Debug, PartialEq)]
pub struct Variable {
    pub name: String,
    pub shape: Shape,
    /// A secret is never shown.
    pub secret: bool,
    pub help: String,
}

/// `chopinfer serve --describe-options`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ServeOptions {
    pub options: Vec<ServeOption>,
    pub environment: Vec<Variable>,
}

impl ServeOptions {
    pub fn parse(text: &str) -> Result<Self, String> {
        let document = document(text, "chopinfer.serve-options.v1")?;
        let options = objects(&document, "options")?
            .into_iter()
            .map(|option| {
                let flag = string(option, "flag")?;
                if !flag.starts_with("--") {
                    return Err(format!("option {flag:?} is not a flag"));
                }
                Ok(ServeOption {
                    shape: Shape::parse(&string(option, "value")?)?,
                    choices: match option.get("choices") {
                        None | Some(Value::Null) => None,
                        Some(_) => Some(strings(option, "choices")?),
                    },
                    group: string(option, "group")?,
                    help: string(option, "help")?,
                    model_launch: boolean(option, "model_launch")?,
                    repeatable: boolean(option, "repeatable")?,
                    flag,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let environment = objects(&document, "environment")?
            .into_iter()
            .map(|variable| {
                Ok(Variable {
                    name: string(variable, "name")?,
                    shape: Shape::parse(&string(variable, "value")?)?,
                    secret: boolean(variable, "secret")?,
                    help: string(variable, "help")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            options,
            environment,
        })
    }

    /// The option `--name`.
    pub fn option(&self, name: &str) -> Option<&ServeOption> {
        self.options.iter().find(|option| option.name() == name)
    }

    pub fn variable(&self, name: &str) -> Option<&Variable> {
        self.environment
            .iter()
            .find(|variable| variable.name == name)
    }
}

/// One serving profile of an installed model.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub name: String,
    /// `qualified` or `experimental`.
    pub status: String,
    pub tensor_parallel: u32,
    /// `none` for an autoregressive profile.
    pub drafter: String,
    pub speculative_tokens: u32,
    pub max_sequences: u64,
    /// `None` for an experimental profile.
    pub qualified_max_sequences: Option<u64>,
    pub qualification: String,
    /// Effective serve settings, keyed by flag name without dashes.
    pub settings: BTreeMap<String, String>,
    pub model_options: BTreeMap<String, String>,
}

/// One installed model.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// `qwen38`.
    pub model: String,
    pub helper: String,
    pub served_model_name: String,
    pub profiles: Vec<Profile>,
}

impl Model {
    pub fn profile(&self, name: &str) -> Option<&Profile> {
        self.profiles.iter().find(|profile| profile.name == name)
    }
}

/// `chopinfer models --json`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Models {
    pub models: Vec<Model>,
    /// Helpers that could not describe their profiles, and why.
    pub errors: Vec<(String, String)>,
}

impl Models {
    pub fn parse(text: &str) -> Result<Self, String> {
        let document = document(text, "chopinfer.models.v1")?;
        let models = objects(&document, "models")?
            .into_iter()
            .map(|model| {
                let profiles = objects(model, "profiles")?
                    .into_iter()
                    .map(|profile| {
                        Ok(Profile {
                            name: string(profile, "name")?,
                            status: string(profile, "status")?,
                            tensor_parallel: count(profile, "tensor_parallel")?,
                            drafter: string(profile, "drafter")?,
                            speculative_tokens: count(profile, "speculative_tokens")?,
                            max_sequences: number(profile, "max_sequences")?,
                            qualified_max_sequences: match profile.get("qualified_max_sequences") {
                                None | Some(Value::Null) => None,
                                Some(_) => Some(number(profile, "qualified_max_sequences")?),
                            },
                            qualification: string(profile, "qualification")?,
                            settings: scalars(profile, "settings")?,
                            model_options: scalars(profile, "model_options")?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(Model {
                    model: string(model, "model")?,
                    helper: string(model, "helper")?,
                    served_model_name: string(model, "served_model_name")?,
                    profiles,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let errors = objects(&document, "errors")?
            .into_iter()
            .map(|error| Ok((string(error, "helper")?, string(error, "message")?)))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { models, errors })
    }

    pub fn model(&self, model: &str) -> Option<&Model> {
        self.models.iter().find(|seen| seen.model == model)
    }
}

/// Whether a profile can run on a checkpoint as `serve --model` plus flags.
#[derive(Clone, Debug, PartialEq)]
pub enum Availability {
    /// It runs as the pick with these flags, and these model options of
    /// the pick reset to their package defaults.
    Available {
        arguments: Vec<String>,
        inherited_model_options: Vec<String>,
    },
    /// It cannot, and why.
    Unavailable(String),
}

/// One profile at one tensor parallel size of a pick.
#[derive(Clone, Debug, PartialEq)]
pub struct PickProfile {
    pub name: String,
    pub availability: Availability,
}

/// What `serve --model` does at one tensor parallel size.
#[derive(Clone, Debug, PartialEq)]
pub struct Size {
    pub tensor_parallel: u32,
    /// The profile it picks, or `None` with the helper's refusal.
    pub pick: Option<String>,
    pub refusal: Option<String>,
    pub profiles: Vec<PickProfile>,
}

impl Size {
    pub fn profile(&self, name: &str) -> Option<&PickProfile> {
        self.profiles.iter().find(|profile| profile.name == name)
    }
}

/// `chopinfer models pick --model DIR --json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Pick {
    /// The model whose helper takes the checkpoint, or `None` with why
    /// none does.
    pub model: Option<String>,
    pub refusal: Option<String>,
    pub sizes: Vec<Size>,
}

impl Pick {
    pub fn parse(text: &str) -> Result<Self, String> {
        let document = document(text, "chopinfer.models-pick.v1")?;
        let sizes = objects(&document, "sizes")?
            .into_iter()
            .map(|size| {
                let profiles = objects(size, "profiles")?
                    .into_iter()
                    .map(|profile| {
                        let availability = if boolean(profile, "available")? {
                            Availability::Available {
                                arguments: strings(profile, "arguments")?,
                                inherited_model_options: strings(
                                    profile,
                                    "inherited_model_options",
                                )?,
                            }
                        } else {
                            Availability::Unavailable(string(profile, "reason")?)
                        };
                        Ok(PickProfile {
                            name: string(profile, "name")?,
                            availability,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(Size {
                    tensor_parallel: count(size, "tensor_parallel")?,
                    pick: optional_string(size, "pick")?,
                    refusal: optional_string(size, "refusal")?,
                    profiles,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            model: optional_string(&document, "model")?,
            refusal: optional_string(&document, "refusal")?,
            sizes,
        })
    }

    pub fn size(&self, tensor_parallel: u32) -> Option<&Size> {
        self.sizes
            .iter()
            .find(|size| size.tensor_parallel == tensor_parallel)
    }
}

/// `chopinfer models command`: the `serve --model` argument list, without
/// `--listen`.
pub fn parse_command(text: &str) -> Result<Vec<String>, String> {
    let document = document(text, "chopinfer.models-command.v1")?;
    let arguments = strings(&document, "arguments")?;
    if arguments.first().map(String::as_str) != Some("serve") {
        return Err("the arguments do not start with serve".to_owned());
    }
    if arguments
        .iter()
        .any(|argument| argument == "--listen" || argument.starts_with("--listen="))
    {
        return Err("the arguments hold --listen".to_owned());
    }
    Ok(arguments)
}

/// A model option as `serve --check` resolves it.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedOption {
    pub name: String,
    pub value: String,
    pub default: String,
    pub default_is_qualified: bool,
}

/// `chopinfer serve ARGS --check`.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub profile: String,
    pub settings: BTreeMap<String, String>,
    pub model_options: Vec<CheckedOption>,
    pub warnings: Vec<String>,
}

impl Check {
    pub fn parse(text: &str) -> Result<Self, String> {
        let document = document(text, "chopinfer.serve-check.v1")?;
        let model_options = objects(&document, "model_options")?
            .into_iter()
            .map(|option| {
                Ok(CheckedOption {
                    name: string(option, "name")?,
                    value: scalar(option, "value")?,
                    default: scalar(option, "default")?,
                    default_is_qualified: boolean(option, "default_is_qualified")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            profile: string(&document, "profile")?,
            settings: scalars(&document, "settings")?,
            model_options,
            warnings: strings(&document, "warnings")?,
        })
    }
}

/// The type and limits of a package option.
#[derive(Clone, Debug, PartialEq)]
pub enum PackageKind {
    Integer {
        default: i64,
        minimum: i64,
        maximum: i64,
    },
    /// Kept as written, so a default reads back as the package wrote it.
    Number {
        default: String,
        minimum: f64,
        maximum: f64,
    },
    Boolean {
        default: bool,
    },
    String {
        default: String,
        allowed: Vec<String>,
    },
}

/// An option a package declares, set with `--model-option NAME=VALUE`.
#[derive(Clone, Debug, PartialEq)]
pub struct PackageOption {
    pub name: String,
    pub kind: PackageKind,
    pub default_is_qualified: bool,
}

impl PackageOption {
    /// The default as `--model-option` writes it.
    pub fn default_text(&self) -> String {
        match &self.kind {
            PackageKind::Integer { default, .. } => default.to_string(),
            PackageKind::Number { default, .. } => default.clone(),
            PackageKind::Boolean { default } => default.to_string(),
            PackageKind::String { default, .. } => default.clone(),
        }
    }

    /// Why `text` is not a value of this option; `None` when it is.
    pub fn problem(&self, text: &str) -> Option<String> {
        match &self.kind {
            PackageKind::Integer {
                minimum, maximum, ..
            } => text
                .parse::<i64>()
                .ok()
                .filter(|value| (*minimum..=*maximum).contains(value))
                .is_none()
                .then(|| format!("from {minimum} to {maximum}")),
            PackageKind::Number {
                minimum, maximum, ..
            } => text
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite() && (*minimum..=*maximum).contains(value))
                .is_none()
                .then(|| format!("from {minimum} to {maximum}")),
            PackageKind::Boolean { .. } => {
                (!matches!(text, "true" | "false")).then(|| "true or false".to_owned())
            }
            PackageKind::String { allowed, .. } => (!allowed.iter().any(|value| value == text))
                .then(|| format!("one of {}", allowed.join(", "))),
        }
    }

    /// The values ←→ cycles through: the choices of a string, both values
    /// of a switch; none for numbers, which are typed.
    pub fn choices(&self) -> Vec<String> {
        match &self.kind {
            PackageKind::Boolean { .. } => vec!["true".to_owned(), "false".to_owned()],
            PackageKind::String { allowed, .. } => allowed.clone(),
            _ => Vec::new(),
        }
    }
}

/// `chopinfer serve ... --list-model-options`: the package's options, none
/// when it declares none.
pub fn parse_package_options(text: &str) -> Result<Vec<PackageOption>, String> {
    let value: Value =
        serde_json::from_str(text).map_err(|error| format!("not a JSON document: {error}"))?;
    if value.is_null() {
        return Ok(Vec::new());
    }
    let document = value
        .as_object()
        .ok_or_else(|| "not a JSON object".to_owned())?;
    let schema = string(document, "schema")?;
    if schema != "chop.model-options.v1" {
        return Err(format!("unexpected schema {schema:?}"));
    }
    objects(document, "options")?
        .into_iter()
        .map(|option| {
            let definition = option
                .get("definition")
                .and_then(Value::as_object)
                .ok_or_else(|| "a model option has no definition".to_owned())?;
            let float = |name: &str| {
                definition
                    .get(name)
                    .and_then(Value::as_f64)
                    .ok_or_else(|| format!("definition.{name} is not a number"))
            };
            let integer = |name: &str| {
                definition
                    .get(name)
                    .and_then(Value::as_i64)
                    .ok_or_else(|| format!("definition.{name} is not a whole number"))
            };
            let kind = match string(definition, "type")?.as_str() {
                "integer" => PackageKind::Integer {
                    default: integer("default")?,
                    minimum: integer("minimum")?,
                    maximum: integer("maximum")?,
                },
                "number" => PackageKind::Number {
                    default: definition
                        .get("default")
                        .filter(|value| value.is_number())
                        .map(Value::to_string)
                        .ok_or_else(|| "definition.default is not a number".to_owned())?,
                    minimum: float("minimum")?,
                    maximum: float("maximum")?,
                },
                "boolean" => PackageKind::Boolean {
                    default: boolean(definition, "default")?,
                },
                "string" => PackageKind::String {
                    default: string(definition, "default")?,
                    allowed: strings(definition, "allowed_values")?,
                },
                other => return Err(format!("unknown model option type {other:?}")),
            };
            Ok(PackageOption {
                name: string(option, "name")?,
                kind,
                default_is_qualified: boolean(option, "default_is_qualified")?,
            })
        })
        .collect()
}

fn document(text: &str, schema: &str) -> Result<Object, String> {
    let value: Value =
        serde_json::from_str(text).map_err(|error| format!("not a JSON document: {error}"))?;
    let Value::Object(document) = value else {
        return Err("not a JSON object".to_owned());
    };
    let seen = string(&document, "schema")?;
    if seen != schema {
        return Err(format!("schema {seen:?}, not {schema}"));
    }
    Ok(document)
}

fn string(object: &Object, name: &str) -> Result<String, String> {
    object
        .get(name)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("{name} is missing or not text"))
}

fn optional_string(object: &Object, name: &str) -> Result<Option<String>, String> {
    match object.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(format!("{name} is not text")),
    }
}

fn boolean(object: &Object, name: &str) -> Result<bool, String> {
    object
        .get(name)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{name} is missing or not true or false"))
}

fn number(object: &Object, name: &str) -> Result<u64, String> {
    object
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{name} is missing or not a whole number"))
}

fn count(object: &Object, name: &str) -> Result<u32, String> {
    u32::try_from(number(object, name)?).map_err(|_| format!("{name} is too large"))
}

fn strings(object: &Object, name: &str) -> Result<Vec<String>, String> {
    object
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{name} is missing or not a list"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{name} holds something that is not text"))
        })
        .collect()
}

fn objects<'a>(object: &'a Object, name: &str) -> Result<Vec<&'a Object>, String> {
    object
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{name} is missing or not a list"))?
        .iter()
        .map(|value| {
            value
                .as_object()
                .ok_or_else(|| format!("{name} holds something that is not an object"))
        })
        .collect()
}

/// Text, a number or a switch as text: `"8192"`, `8192` and `true` read as
/// `8192` and `true`.
fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn scalar(object: &Object, name: &str) -> Result<String, String> {
    object
        .get(name)
        .and_then(scalar_text)
        .ok_or_else(|| format!("{name} is missing or not a value"))
}

fn scalars(object: &Object, name: &str) -> Result<BTreeMap<String, String>, String> {
    object
        .get(name)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{name} is missing or not an object"))?
        .iter()
        .map(|(key, value)| {
            scalar_text(value)
                .map(|text| (key.clone(), text))
                .ok_or_else(|| format!("{name}.{key} is not a value"))
        })
        .collect()
}

/// What the loader asks `chopinfer`.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// `serve --describe-options`.
    Options,
    /// `models --json`.
    Models,
    /// `models pick --model DIR --json`.
    Pick(PathBuf),
    /// `models command`, answered with the same generation.
    Command {
        generation: u64,
        checkpoint: PathBuf,
        tensor_parallel: u32,
        profile: String,
        options: Vec<String>,
    },
    /// `serve ARGUMENTS --list-model-options` with these variables set.
    PackageOptions {
        generation: u64,
        arguments: Vec<String>,
        environment: Vec<(String, String)>,
    },
}

impl Request {
    /// The `chopinfer` arguments, the variables to set, and how long it may
    /// run.
    fn command(&self) -> (Vec<String>, Vec<(String, String)>, Duration) {
        let owned = |arguments: &[&str]| arguments.iter().map(|text| (*text).to_owned()).collect();
        match self {
            Self::Options => (
                owned(&["serve", "--describe-options"]),
                Vec::new(),
                CATALOG_TIMEOUT,
            ),
            Self::Models => (owned(&["models", "--json"]), Vec::new(), CATALOG_TIMEOUT),
            Self::Pick(directory) => {
                let mut arguments: Vec<String> = owned(&["models", "pick", "--model"]);
                arguments.push(directory.display().to_string());
                arguments.push("--json".to_owned());
                (arguments, Vec::new(), CATALOG_TIMEOUT)
            }
            Self::Command {
                checkpoint,
                tensor_parallel,
                profile,
                options,
                ..
            } => {
                let mut arguments: Vec<String> = owned(&["models", "command", "--model"]);
                arguments.push(checkpoint.display().to_string());
                arguments.push("--tensor-parallel-size".to_owned());
                arguments.push(tensor_parallel.to_string());
                arguments.push("--profile".to_owned());
                arguments.push(profile.clone());
                if !options.is_empty() {
                    arguments.push("--".to_owned());
                    arguments.extend(options.iter().cloned());
                }
                (arguments, Vec::new(), CATALOG_TIMEOUT)
            }
            Self::PackageOptions {
                arguments,
                environment,
                ..
            } => {
                let mut all = vec!["serve".to_owned()];
                all.extend(arguments.iter().cloned());
                all.push("--list-model-options".to_owned());
                (all, environment.clone(), PACKAGE_TIMEOUT)
            }
        }
    }
}

/// What came back for a [`Request`].
#[derive(Clone, Debug, PartialEq)]
pub enum Reply {
    Options(Result<ServeOptions, String>),
    Models(Result<Models, String>),
    Pick(PathBuf, Result<Pick, String>),
    Command(u64, Result<Vec<String>, String>),
    PackageOptions(u64, Result<Vec<PackageOption>, String>),
}

/// Runs `request` with `binary` and reads its document.
pub fn answer(binary: &Path, request: Request) -> Reply {
    let (arguments, environment, timeout) = request.command();
    let shown = format!(
        "chopinfer {}",
        arguments[..2.min(arguments.len())].join(" ")
    );
    let output = run(binary, &arguments, &environment, timeout);
    let failed = |message: String| format!("{shown}: {message}");
    match request {
        Request::Options => Reply::Options(
            output
                .and_then(|text| ServeOptions::parse(&text))
                .map_err(failed),
        ),
        Request::Models => {
            Reply::Models(output.and_then(|text| Models::parse(&text)).map_err(failed))
        }
        Request::Pick(directory) => Reply::Pick(
            directory,
            output.and_then(|text| Pick::parse(&text)).map_err(failed),
        ),
        Request::Command { generation, .. } => Reply::Command(
            generation,
            output.and_then(|text| parse_command(&text)).map_err(failed),
        ),
        Request::PackageOptions { generation, .. } => Reply::PackageOptions(
            generation,
            output
                .and_then(|text| parse_package_options(&text))
                .map_err(failed),
        ),
    }
}

/// Runs `binary` with `arguments` and `environment` added to choptop's
/// own, stdin closed, for at most `timeout`. Its stdout when it succeeds,
/// else the end of its stderr.
pub fn run(
    binary: &Path,
    arguments: &[String],
    environment: &[(String, String)],
    timeout: Duration,
) -> Result<String, String> {
    let mut child = Command::new(binary)
        .args(arguments)
        .envs(environment.iter().map(|(name, value)| (name, value)))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot run {}: {error}", binary.display()))?;
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(pipe) = pipe {
                let _ = pipe.take(OUTPUT_LIMIT).read_to_end(&mut bytes);
            }
            String::from_utf8_lossy(&bytes).into_owned()
        })
    };
    let stdout = drain(
        child
            .stdout
            .take()
            .map(|pipe| Box::new(pipe) as Box<dyn Read + Send>),
    );
    let stderr = drain(
        child
            .stderr
            .take()
            .map(|pipe| Box::new(pipe) as Box<dyn Read + Send>),
    );
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                // A helper it started may still hold the pipes, so their
                // readers are left to end on their own.
                return Err(format!("no answer within {}s", timeout.as_secs()));
            }
            Ok(None) => thread::sleep(Duration::from_millis(20)),
            Err(error) => return Err(format!("cannot wait for it: {error}")),
        }
    };
    let stdout = stdout.join().unwrap_or_default();
    let stderr = stderr.join().unwrap_or_default();
    if status.success() {
        return Ok(stdout);
    }
    let last = stderr
        .lines()
        .map(str::trim)
        .rfind(|line| !line.is_empty())
        .map(sanitize);
    Err(match last {
        Some(line) if line.chars().count() > MESSAGE_LIMIT => {
            line.chars().take(MESSAGE_LIMIT).collect::<String>() + "..."
        }
        Some(line) => line,
        None => match status.code() {
            Some(code) => format!("exited with status {code}"),
            None => "ended by a signal".to_owned(),
        },
    })
}

/// The threads that run requests. Dropping it ends them once their current
/// command is done.
#[derive(Debug)]
pub struct Worker {
    catalog: Sender<Request>,
    command: Sender<Request>,
    package: Sender<Request>,
}

impl Worker {
    /// Runs requests with `binary` and sends each reply to `replies`.
    pub fn spawn(binary: PathBuf, replies: Sender<Reply>) -> Self {
        let lane = |latest: bool| {
            let (sender, receiver) = mpsc::channel();
            let binary = binary.clone();
            let replies = replies.clone();
            thread::spawn(move || run_lane(&binary, &receiver, &replies, latest));
            sender
        };
        Self {
            catalog: lane(false),
            command: lane(true),
            package: lane(true),
        }
    }

    pub fn send(&self, request: Request) {
        let lane = match request {
            Request::Options | Request::Models | Request::Pick(_) => &self.catalog,
            Request::Command { .. } => &self.command,
            Request::PackageOptions { .. } => &self.package,
        };
        let _ = lane.send(request);
    }
}

/// Answers requests in turn; with `latest`, only the newest of those
/// waiting, since each replaces the ones before it.
fn run_lane(binary: &Path, requests: &Receiver<Request>, replies: &Sender<Reply>, latest: bool) {
    while let Ok(mut request) = requests.recv() {
        if latest {
            while let Ok(newer) = requests.try_recv() {
                request = newer;
            }
        }
        if replies.send(answer(binary, request)).is_err() {
            return;
        }
    }
}

/// The `chopinfer` to run: `--chopinfer PATH`, else `chopinfer` on `path`.
pub fn find_binary(explicit: Option<&Path>, path: Option<&OsStr>) -> Result<PathBuf, String> {
    let runnable = |candidate: &Path| {
        fs::metadata(candidate)
            .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
    };
    if let Some(explicit) = explicit {
        return if runnable(explicit) {
            Ok(explicit.to_owned())
        } else {
            Err(format!(
                "--chopinfer {} is not a program choptop can run",
                sanitize(&explicit.display().to_string())
            ))
        };
    }
    path.into_iter()
        .flat_map(std::env::split_paths)
        .filter(|directory| directory.is_absolute())
        .map(|directory| directory.join("chopinfer"))
        .find(|candidate| runnable(candidate))
        .ok_or_else(|| "chopinfer is not on PATH; --chopinfer names it".to_owned())
}

/// Where remembered checkpoints are kept: `$XDG_CONFIG_HOME/choptop`,
/// else `~/.config/choptop`.
pub fn checkpoints_path(config_home: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    let base = config_home
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| {
            home.map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .map(|home| home.join(".config"))
        })?;
    Some(base.join("choptop/checkpoints.json"))
}

/// The `CHOPINFER_*_CHECKPOINT` variables of `variables` that name a
/// directory, in name order.
pub fn checkpoint_variables(
    variables: impl IntoIterator<Item = (String, String)>,
) -> Vec<(String, PathBuf)> {
    let mut found = variables
        .into_iter()
        .filter(|(name, value)| {
            name.strip_prefix("CHOPINFER_")
                .and_then(|rest| rest.strip_suffix("_CHECKPOINT"))
                .is_some_and(|model| !model.is_empty())
                && !value.is_empty()
                && Path::new(value).is_dir()
        })
        .map(|(name, value)| (name, PathBuf::from(value)))
        .collect::<Vec<_>>();
    found.sort();
    found
}

/// Checkpoint directories: one remembered per model, and those the
/// environment names.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Checkpoints {
    /// The file they are kept in; `None` keeps them only while choptop runs.
    path: Option<PathBuf>,
    remembered: BTreeMap<String, PathBuf>,
    environment: Vec<(String, PathBuf)>,
    /// Why the file could not be read.
    pub problem: Option<String>,
}

impl Checkpoints {
    /// Reads the file at `path`, if any, beside the `environment`'s.
    pub fn load(path: Option<PathBuf>, environment: Vec<(String, PathBuf)>) -> Self {
        let mut checkpoints = Self {
            path,
            remembered: BTreeMap::new(),
            environment,
            problem: None,
        };
        let Some(path) = &checkpoints.path else {
            return checkpoints;
        };
        match fs::read_to_string(path) {
            Ok(text) => match Self::parse(&text) {
                Ok(remembered) => checkpoints.remembered = remembered,
                Err(error) => {
                    checkpoints.problem = Some(format!("{} is unreadable: {error}", path.display()))
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                checkpoints.problem = Some(format!("cannot read {}: {error}", path.display()))
            }
        }
        checkpoints
    }

    fn parse(text: &str) -> Result<BTreeMap<String, PathBuf>, String> {
        let document = document(text, CHECKPOINTS_SCHEMA)?;
        Ok(scalars(&document, "checkpoints")?
            .into_iter()
            .map(|(model, directory)| (model, PathBuf::from(directory)))
            .collect())
    }

    pub fn remembered(&self, model: &str) -> Option<&Path> {
        self.remembered.get(model).map(PathBuf::as_path)
    }

    pub fn environment(&self) -> &[(String, PathBuf)] {
        &self.environment
    }

    /// Every directory to ask `models pick` about, once each.
    pub fn directories(&self) -> Vec<PathBuf> {
        let mut directories: Vec<PathBuf> = Vec::new();
        for directory in self
            .remembered
            .values()
            .chain(self.environment.iter().map(|(_, directory)| directory))
        {
            if !directories.contains(directory) {
                directories.push(directory.clone());
            }
        }
        directories
    }

    /// Keeps `directory` as the checkpoint of `model`, in the file too.
    pub fn remember(&mut self, model: &str, directory: &Path) -> Result<(), String> {
        self.remembered
            .insert(model.to_owned(), directory.to_owned());
        let Some(path) = &self.path else {
            return Ok(());
        };
        let checkpoints = self
            .remembered
            .iter()
            .map(|(model, directory)| {
                (
                    model.clone(),
                    Value::String(directory.display().to_string()),
                )
            })
            .collect::<Map<_, _>>();
        let mut document = Map::new();
        document.insert("schema".to_owned(), CHECKPOINTS_SCHEMA.into());
        document.insert("checkpoints".to_owned(), Value::Object(checkpoints));
        let text = serde_json::to_string_pretty(&Value::Object(document))
            .map_err(|error| error.to_string())?
            + "\n";
        let write = || -> std::io::Result<()> {
            if let Some(parent) = path.parent() {
                fs::DirBuilder::new()
                    .recursive(true)
                    .mode(0o700)
                    .create(parent)?;
            }
            let partial = path.with_extension("json.partial");
            fs::write(&partial, text)?;
            fs::rename(&partial, path)
        };
        write().map_err(|error| format!("cannot write {}: {error}", path.display()))
    }
}

/// One fetch the loader waits on.
#[derive(Clone, Debug, PartialEq)]
pub enum Fetch<T> {
    Waiting,
    Ready(T),
    Failed(String),
}

impl<T> From<Result<T, String>> for Fetch<T> {
    fn from(result: Result<T, String>) -> Self {
        match result {
            Ok(value) => Self::Ready(value),
            Err(error) => Self::Failed(error),
        }
    }
}

impl<T> Fetch<T> {
    pub fn ready(&self) -> Option<&T> {
        match self {
            Self::Ready(value) => Some(value),
            _ => None,
        }
    }
}

/// Where a model's checkpoint comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// Remembered in the checkpoints file.
    Saved,
    /// Named by this variable.
    Environment(String),
}

/// What is known about a model's checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub enum Checkpoint {
    Found {
        directory: PathBuf,
        source: Source,
    },
    /// A known directory is still being asked about.
    Checking,
    Missing,
}

impl Checkpoint {
    pub fn directory(&self) -> Option<&Path> {
        match self {
            Self::Found { directory, .. } => Some(directory),
            _ => None,
        }
    }
}

/// Everything fetched from `chopinfer`, and what to ask it next.
#[derive(Debug)]
pub struct Catalog {
    /// The `chopinfer` to run, or why there is none.
    pub binary: Result<PathBuf, String>,
    pub options: Option<Fetch<ServeOptions>>,
    pub models: Option<Fetch<Models>>,
    /// `models pick` for each directory asked about.
    pub picks: BTreeMap<PathBuf, Fetch<Pick>>,
    pub checkpoints: Checkpoints,
    /// Requests for the worker, taken by the main loop.
    pub outbox: Vec<Request>,
}

impl Catalog {
    pub fn new(binary: Result<PathBuf, String>, checkpoints: Checkpoints) -> Self {
        Self {
            binary,
            options: None,
            models: None,
            picks: BTreeMap::new(),
            checkpoints,
            outbox: Vec::new(),
        }
    }

    /// Without `chopinfer`, for a choptop that never loads models.
    pub fn none() -> Self {
        Self::new(
            Err("chopinfer was not looked for".to_owned()),
            Checkpoints::default(),
        )
    }

    /// Asks once for the serve options, the models, and a pick for every
    /// known checkpoint directory.
    pub fn open(&mut self) {
        if self.binary.is_err() || self.options.is_some() {
            return;
        }
        self.options = Some(Fetch::Waiting);
        self.models = Some(Fetch::Waiting);
        self.outbox.push(Request::Options);
        self.outbox.push(Request::Models);
        for directory in self.checkpoints.directories() {
            self.pick(directory);
        }
    }

    /// Asks which model `directory` is, unless that was asked already.
    pub fn pick(&mut self, directory: PathBuf) {
        if self.binary.is_err() || self.picks.contains_key(&directory) {
            return;
        }
        self.picks.insert(directory.clone(), Fetch::Waiting);
        self.outbox.push(Request::Pick(directory));
    }

    /// Asks a request the loader owns, such as a command line.
    pub fn send(&mut self, request: Request) {
        if self.binary.is_ok() {
            self.outbox.push(request);
        }
    }

    /// Takes a catalog reply; command and package replies are given back
    /// for the loader.
    pub fn apply(&mut self, reply: Reply) -> Option<Reply> {
        match reply {
            Reply::Options(result) => self.options = Some(result.into()),
            Reply::Models(result) => self.models = Some(result.into()),
            Reply::Pick(directory, result) => {
                self.picks.insert(directory, result.into());
            }
            other => return Some(other),
        }
        None
    }

    pub fn models(&self) -> &[Model] {
        self.models
            .as_ref()
            .and_then(Fetch::ready)
            .map_or(&[], |models| models.models.as_slice())
    }

    pub fn model(&self, model: &str) -> Option<&Model> {
        self.models().iter().find(|seen| seen.model == model)
    }

    pub fn options(&self) -> Option<&ServeOptions> {
        self.options.as_ref().and_then(Fetch::ready)
    }

    /// The pick of `directory`, once read.
    pub fn picked(&self, directory: &Path) -> Option<&Pick> {
        self.picks.get(directory).and_then(Fetch::ready)
    }

    /// The checkpoint of `model`: the remembered one, else the first
    /// variable's directory that `models pick` says is this model.
    pub fn checkpoint(&self, model: &str) -> Checkpoint {
        let mut checking = false;
        let candidates = self
            .checkpoints
            .remembered(model)
            .map(|directory| (directory.to_owned(), Source::Saved))
            .into_iter()
            .chain(
                self.checkpoints
                    .environment()
                    .iter()
                    .map(|(name, directory)| {
                        (directory.clone(), Source::Environment(name.clone()))
                    }),
            );
        for (directory, source) in candidates {
            match self.picks.get(&directory) {
                Some(Fetch::Ready(pick)) if pick.model.as_deref() == Some(model) => {
                    return Checkpoint::Found { directory, source };
                }
                // The remembered directory comes first, so the variables
                // wait for its answer.
                Some(Fetch::Waiting) | None if source == Source::Saved => {
                    return Checkpoint::Checking;
                }
                Some(Fetch::Waiting) => checking = true,
                _ => {}
            }
        }
        if checking {
            Checkpoint::Checking
        } else {
            Checkpoint::Missing
        }
    }

    /// What went wrong reading the catalog, for the Models pane: helpers
    /// that failed, and checkpoints no helper takes.
    pub fn troubles(&self) -> Vec<String> {
        let mut troubles = Vec::new();
        if let Err(error) = &self.binary {
            troubles.push(error.clone());
        }
        for fetch in [
            self.options.as_ref().map(|fetch| match fetch {
                Fetch::Failed(error) => Some(error),
                _ => None,
            }),
            self.models.as_ref().map(|fetch| match fetch {
                Fetch::Failed(error) => Some(error),
                _ => None,
            }),
        ]
        .into_iter()
        .flatten()
        .flatten()
        {
            troubles.push(fetch.clone());
        }
        if let Some(Fetch::Ready(models)) = &self.models {
            for (helper, message) in &models.errors {
                troubles.push(format!("{helper}: {message}"));
            }
        }
        for (directory, fetch) in &self.picks {
            let shown = self
                .checkpoints
                .environment()
                .iter()
                .find(|(_, seen)| seen == directory)
                .map_or_else(|| directory.display().to_string(), |(name, _)| name.clone());
            match fetch {
                Fetch::Failed(error) => troubles.push(format!("{shown}: {error}")),
                Fetch::Ready(Pick {
                    model: None,
                    refusal,
                    ..
                }) => troubles.push(format!(
                    "{shown}: {}",
                    refusal.as_deref().unwrap_or("no installed model takes it")
                )),
                _ => {}
            }
        }
        troubles.iter().map(|trouble| sanitize(trouble)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SERVE_OPTIONS: &str = include_str!("../tests/fixtures/contracts/serve-options.json");
    const MODELS: &str = include_str!("../tests/fixtures/contracts/models.json");
    const PICK: &str = include_str!("../tests/fixtures/contracts/models-pick.json");
    const COMMAND: &str = include_str!("../tests/fixtures/contracts/models-command.json");
    const CHECK: &str = include_str!("../tests/fixtures/contracts/serve-check.json");
    const PACKAGE: &str = include_str!("../tests/fixtures/contracts/list-model-options.json");

    fn scratch(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("choptop-catalog-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn reads_the_serve_options() {
        let options = ServeOptions::parse(SERVE_OPTIONS).unwrap();
        let sequences = options.option("max-sequences").unwrap();
        assert_eq!(sequences.flag, "--max-sequences");
        assert_eq!(sequences.shape, Shape::Count);
        assert_eq!(sequences.group, "scheduling");
        assert!(sequences.model_launch && !sequences.repeatable);
        let cache = options.option("prefix-cache").unwrap();
        assert_eq!(
            cache.choices.as_deref(),
            Some(&["on".into(), "off".into()][..])
        );
        assert_eq!(
            options.option("allow-credentials").unwrap().shape,
            Shape::Switch
        );
        assert!(!options.option("bundle").unwrap().model_launch);
        assert!(
            options
                .option("allowed-local-media-path")
                .unwrap()
                .repeatable
        );
        let key = options.variable("CHOPINFER_API_KEY").unwrap();
        assert!(key.secret);
        assert_eq!(key.shape, Shape::Text);
        assert_eq!(options.environment.len(), 2);
    }

    #[test]
    fn reads_the_models_and_their_profiles() {
        let models = Models::parse(MODELS).unwrap();
        assert_eq!(models.models.len(), 2);
        let qwen = models.model("qwen38").unwrap();
        assert_eq!(qwen.helper, "chopinfer-model-qwen38");
        assert_eq!(qwen.served_model_name, "Qwen3.8-Flash");
        let mtp = qwen.profile("tp4-mtp").unwrap();
        assert_eq!(
            (
                mtp.tensor_parallel,
                mtp.speculative_tokens,
                mtp.max_sequences
            ),
            (4, 3, 32)
        );
        assert_eq!(mtp.qualified_max_sequences, Some(32));
        assert_eq!(mtp.settings["prefix-cache"], "on");
        let ar = qwen.profile("tp2-ar").unwrap();
        assert_eq!(
            (ar.status.as_str(), ar.qualified_max_sequences),
            ("experimental", None)
        );
        assert_eq!(ar.drafter, "none");
        let dspark = models
            .model("deepseek41")
            .unwrap()
            .profile("tp4-dspark")
            .unwrap();
        // A switch written as JSON reads as the text --model-option takes.
        assert_eq!(dspark.model_options["prefill_upper_skip"], "true");
        assert_eq!(dspark.model_options["dspark_draft_chunk_items"], "12");
        assert_eq!(
            models.errors,
            [(
                "chopinfer-model-broken".to_owned(),
                "serving-profiles exited with status 2".to_owned()
            )]
        );
    }

    #[test]
    fn reads_what_serve_model_picks() {
        let pick = Pick::parse(PICK).unwrap();
        assert_eq!(pick.model.as_deref(), Some("deepseek41"));
        assert_eq!(pick.refusal, None);
        let four = pick.size(4).unwrap();
        assert_eq!(four.pick.as_deref(), Some("tp4-dspark"));
        assert_eq!(
            four.profile("tp4-ar").unwrap().availability,
            Availability::Available {
                arguments: vec![
                    "--speculative-tokens".into(),
                    "0".into(),
                    "--prefill-budget".into(),
                    "fixed".into()
                ],
                inherited_model_options: vec!["dspark_draft_chunk_items".into()],
            }
        );
        assert!(matches!(
            &four.profile("tp4-mtp").unwrap().availability,
            Availability::Unavailable(reason) if reason.contains("no MTP head")
        ));
        let two = pick.size(2).unwrap();
        assert_eq!(two.pick, None);
        assert!(two.refusal.as_deref().unwrap().contains("does not fit"));
        assert!(pick.size(8).is_none());
    }

    #[test]
    fn reads_a_command_line_and_a_check() {
        let arguments = parse_command(COMMAND).unwrap();
        assert_eq!(&arguments[..3], ["serve", "--model", "/models/deepseek"]);
        assert!(arguments.contains(&"dspark_draft_chunk_items=12".to_owned()));
        let check = Check::parse(CHECK).unwrap();
        assert_eq!(check.profile, "tp4-dspark");
        assert_eq!(check.settings["max-sequences"], "48");
        assert_eq!(
            check.model_options[1],
            CheckedOption {
                name: "prefill_upper_skip".into(),
                value: "false".into(),
                default: "true".into(),
                default_is_qualified: true,
            }
        );
        assert!(!check.model_options[2].default_is_qualified);
        assert_eq!(check.warnings.len(), 1);
    }

    #[test]
    fn reads_package_options_or_none() {
        let options = parse_package_options(PACKAGE).unwrap();
        let names = options
            .iter()
            .map(|option| option.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "dspark_draft_chunk_items",
                "prefill_chunk_tokens",
                "prefill_upper_skip",
                "text_config.rope_scale"
            ]
        );
        let items = &options[0];
        assert_eq!(items.default_text(), "12");
        assert_eq!(items.problem("16"), None);
        assert_eq!(items.problem("33").as_deref(), Some("from 1 to 32"));
        assert_eq!(items.problem("x").as_deref(), Some("from 1 to 32"));
        assert_eq!(
            options[1].problem("4096").as_deref(),
            Some("one of 0, 2048, 8192")
        );
        assert_eq!(options[1].choices(), ["0", "2048", "8192"]);
        assert_eq!(options[2].default_text(), "true");
        assert_eq!(options[2].problem("yes").as_deref(), Some("true or false"));
        assert_eq!(options[3].default_text(), "1.5");
        assert_eq!(options[3].problem("2.25"), None);
        assert!(!options[3].default_is_qualified);
        assert!(parse_package_options("null\n").unwrap().is_empty());
    }

    #[test]
    fn refuses_another_schema_or_a_missing_field() {
        let error = Models::parse(SERVE_OPTIONS).unwrap_err();
        assert!(error.contains("chopinfer.serve-options.v1"), "{error}");
        let broken = MODELS.replacen("\"served_model_name\"", "\"served\"", 1);
        assert!(
            Models::parse(&broken)
                .unwrap_err()
                .contains("served_model_name")
        );
        assert!(Pick::parse("[]").unwrap_err().contains("not a JSON object"));
        assert!(
            ServeOptions::parse("{")
                .unwrap_err()
                .contains("not a JSON document")
        );
        let listen = COMMAND.replacen("\"--max-sequences\"", "\"--listen\"", 1);
        assert!(parse_command(&listen).unwrap_err().contains("--listen"));
        let shape = SERVE_OPTIONS.replacen("\"N\"", "\"COUNT\"", 1);
        assert!(ServeOptions::parse(&shape).unwrap_err().contains("COUNT"));
    }

    /// A stand-in `chopinfer` that prints a fixture for each command and
    /// fails `models pick` for one directory.
    fn fake_chopinfer(root: &Path) -> PathBuf {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/contracts");
        let script = root.join("chopinfer");
        let text = format!(
            "#!/bin/sh\n\
             fixtures='{}'\n\
             case \"$1 $2\" in\n\
             'serve --describe-options') cat \"$fixtures/serve-options.json\" ;;\n\
             'models --json') cat \"$fixtures/models.json\" ;;\n\
             'models pick')\n\
               if [ \"$4\" = /models/broken ]; then echo 'warming up' >&2; echo 'error: no helper takes /models/broken' >&2; exit 2; fi\n\
               cat \"$fixtures/models-pick.json\" ;;\n\
             'models command') echo \"$*\" > \"$(dirname \"$0\")/last-command\"; cat \"$fixtures/models-command.json\" ;;\n\
             *) if [ \"$CHOPINFER_LOG_LEVEL\" = debug ]; then cat \"$fixtures/list-model-options.json\"; else echo null; fi ;;\n\
             esac\n",
            fixtures.display()
        );
        fs::write(&script, text).unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        script
    }

    #[test]
    fn runs_chopinfer_and_reads_what_it_prints() {
        let root = scratch("answer");
        let binary = fake_chopinfer(&root);
        assert!(matches!(
            answer(&binary, Request::Options),
            Reply::Options(Ok(_))
        ));
        let Reply::Pick(directory, Err(error)) =
            answer(&binary, Request::Pick("/models/broken".into()))
        else {
            panic!("a refused pick")
        };
        assert_eq!(directory, Path::new("/models/broken"));
        // The last line of stderr says why.
        assert_eq!(
            error,
            "chopinfer models pick: error: no helper takes /models/broken"
        );
        let reply = answer(
            &binary,
            Request::Command {
                generation: 7,
                checkpoint: "/models/deepseek".into(),
                tensor_parallel: 4,
                profile: "tp4-ar".into(),
                options: vec!["--max-sequences".into(), "48".into()],
            },
        );
        assert!(matches!(reply, Reply::Command(7, Ok(_))), "{reply:?}");
        assert_eq!(
            fs::read_to_string(root.join("last-command"))
                .unwrap()
                .trim(),
            "models command --model /models/deepseek --tensor-parallel-size 4 --profile tp4-ar -- --max-sequences 48"
        );
        let package = |environment: Vec<(String, String)>| {
            answer(
                &binary,
                Request::PackageOptions {
                    generation: 3,
                    arguments: vec!["--model".into(), "/models/deepseek".into()],
                    environment,
                },
            )
        };
        assert_eq!(
            package(Vec::new()),
            Reply::PackageOptions(3, Ok(Vec::new()))
        );
        let Reply::PackageOptions(3, Ok(options)) =
            package(vec![("CHOPINFER_LOG_LEVEL".into(), "debug".into())])
        else {
            panic!("package options")
        };
        assert_eq!(options.len(), 4);
        let missing = answer(&root.join("absent"), Request::Models);
        assert!(
            matches!(&missing, Reply::Models(Err(error)) if error.contains("cannot run")),
            "{missing:?}"
        );
    }

    #[test]
    fn a_command_that_hangs_is_ended() {
        let root = scratch("hang");
        let script = root.join("chopinfer");
        fs::write(&script, "#!/bin/sh\nexec sleep 30\n").unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        let started = Instant::now();
        let error = run(&script, &[], &[], Duration::from_millis(200)).unwrap_err();
        assert!(error.contains("no answer within"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn the_worker_answers_every_catalog_request_and_the_latest_command() {
        let root = scratch("worker");
        let (replies, received) = mpsc::channel();
        let worker = Worker::spawn(fake_chopinfer(&root), replies);
        worker.send(Request::Options);
        worker.send(Request::Models);
        worker.send(Request::Pick("/models/deepseek".into()));
        for generation in 1..=5 {
            worker.send(Request::Command {
                generation,
                checkpoint: "/models/deepseek".into(),
                tensor_parallel: 4,
                profile: "tp4-ar".into(),
                options: Vec::new(),
            });
        }
        let mut catalog = 0;
        let mut commands = Vec::new();
        while catalog < 3 || commands.last() != Some(&5) {
            match received.recv_timeout(Duration::from_secs(10)).unwrap() {
                Reply::Command(generation, result) => {
                    assert!(result.is_ok());
                    commands.push(generation);
                }
                _ => catalog += 1,
            }
        }
        // The first may have started before the rest arrived; the ones
        // between were replaced.
        assert!(commands.len() <= 2, "{commands:?}");
    }

    #[test]
    fn finds_chopinfer_on_path_or_where_told() {
        let root = scratch("binary");
        let bin = root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        let program = bin.join("chopinfer");
        fs::write(&program, "#!/bin/sh\n").unwrap();
        fs::write(root.join("chopinfer"), "not runnable").unwrap();
        let path =
            std::env::join_paths([root.clone(), PathBuf::from("relative"), bin.clone()]).unwrap();
        assert!(find_binary(None, Some(&path)).is_err());
        fs::set_permissions(&program, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(find_binary(None, Some(&path)).unwrap(), program);
        assert!(find_binary(None, None).unwrap_err().contains("--chopinfer"));
        assert_eq!(find_binary(Some(&program), None).unwrap(), program);
        let error = find_binary(Some(&root.join("chopinfer")), Some(&path)).unwrap_err();
        assert!(error.contains("not a program"), "{error}");
    }

    #[test]
    fn checkpoints_come_from_the_environment_and_the_file() {
        let root = scratch("checkpoints");
        let qwen = root.join("qwen");
        fs::create_dir_all(&qwen).unwrap();
        let variables = checkpoint_variables([
            (
                "CHOPINFER_QWEN38_CHECKPOINT".to_owned(),
                qwen.display().to_string(),
            ),
            (
                "CHOPINFER_GONE_CHECKPOINT".to_owned(),
                root.join("gone").display().to_string(),
            ),
            ("CHOPINFER_EMPTY_CHECKPOINT".to_owned(), String::new()),
            (
                "CHOPINFER_CHECKPOINT".to_owned(),
                qwen.display().to_string(),
            ),
            ("CHOPINFER_DATA".to_owned(), qwen.display().to_string()),
        ]);
        assert_eq!(
            variables,
            [("CHOPINFER_QWEN38_CHECKPOINT".to_owned(), qwen.clone())]
        );
        assert_eq!(
            checkpoints_path(Some("/config".into()), Some("/home".into())),
            Some(PathBuf::from("/config/choptop/checkpoints.json"))
        );
        assert_eq!(
            checkpoints_path(Some("relative".into()), Some("/users/a".into())),
            Some(PathBuf::from("/users/a/.config/choptop/checkpoints.json"))
        );
        assert_eq!(checkpoints_path(None, None), None);
        let path = root.join("config/choptop/checkpoints.json");
        let mut checkpoints = Checkpoints::load(Some(path.clone()), variables.clone());
        assert_eq!(checkpoints.problem, None);
        assert_eq!(checkpoints.directories(), std::slice::from_ref(&qwen));
        checkpoints
            .remember("deepseek41", Path::new("/models/deepseek"))
            .unwrap();
        checkpoints.remember("qwen38", &qwen).unwrap();
        let mode = fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700);
        let again = Checkpoints::load(Some(path.clone()), variables);
        assert_eq!(
            again.remembered("deepseek41"),
            Some(Path::new("/models/deepseek"))
        );
        assert_eq!(
            again.directories(),
            [PathBuf::from("/models/deepseek"), qwen.clone()]
        );
        fs::write(&path, "{\"schema\": \"other\"}").unwrap();
        let broken = Checkpoints::load(Some(path), Vec::new());
        assert!(broken.problem.unwrap().contains("unreadable"));
    }

    #[test]
    fn the_catalog_asks_once_and_names_each_models_checkpoint() {
        let environment = vec![
            (
                "CHOPINFER_DEEPSEEK_CHECKPOINT".to_owned(),
                PathBuf::from("/models/deepseek"),
            ),
            (
                "CHOPINFER_OTHER_CHECKPOINT".to_owned(),
                PathBuf::from("/models/other"),
            ),
        ];
        let mut catalog = Catalog::new(
            Ok("/opt/chopinfer/bin/chopinfer".into()),
            Checkpoints::load(None, environment),
        );
        catalog.open();
        catalog.open();
        assert_eq!(
            catalog.outbox,
            [
                Request::Options,
                Request::Models,
                Request::Pick("/models/deepseek".into()),
                Request::Pick("/models/other".into()),
            ]
        );
        assert_eq!(catalog.checkpoint("deepseek41"), Checkpoint::Checking);
        let pick = Pick::parse(PICK).unwrap();
        assert_eq!(
            catalog.apply(Reply::Pick("/models/deepseek".into(), Ok(pick))),
            None
        );
        let refused = Pick {
            model: None,
            refusal: Some("no helper takes it".into()),
            sizes: Vec::new(),
        };
        catalog.apply(Reply::Pick("/models/other".into(), Ok(refused)));
        assert_eq!(
            catalog.checkpoint("deepseek41"),
            Checkpoint::Found {
                directory: "/models/deepseek".into(),
                source: Source::Environment("CHOPINFER_DEEPSEEK_CHECKPOINT".into()),
            }
        );
        assert_eq!(catalog.checkpoint("qwen38"), Checkpoint::Missing);
        catalog.apply(Reply::Models(Models::parse(MODELS)));
        assert_eq!(catalog.models().len(), 2);
        assert_eq!(
            catalog.troubles(),
            [
                "chopinfer-model-broken: serving-profiles exited with status 2",
                "CHOPINFER_OTHER_CHECKPOINT: no helper takes it",
            ]
        );
        // A remembered directory wins over the environment's.
        catalog
            .checkpoints
            .remember("deepseek41", Path::new("/models/ds2"))
            .unwrap();
        assert_eq!(catalog.checkpoint("deepseek41"), Checkpoint::Checking);
        // Replies the loader owns are handed back.
        let command = Reply::Command(1, Ok(vec!["serve".into()]));
        assert_eq!(catalog.apply(command.clone()), Some(command));
        // Without chopinfer nothing is asked.
        let mut none = Catalog::none();
        none.open();
        none.pick("/models/x".into());
        assert!(none.outbox.is_empty());
        assert_eq!(none.troubles(), ["chopinfer was not looked for"]);
    }
}
