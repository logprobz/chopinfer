//! The command line of a launch. The form's choices become explicit serve
//! options, which `chopinfer models command` merges over the chosen
//! profile into the final argument list; choptop adds `--listen` for each
//! instance and the environment it runs with.
//!
//! The GPUs are placed only through `CUDA_VISIBLE_DEVICES`, set from the
//! UUIDs of the GPUs picked, so a GPU that was not checked is never used.

use std::collections::BTreeMap;

use crate::catalog::{PackageOption, ServeOptions, Shape};

/// Flags choptop sets itself, which a custom row may not, and why.
pub const OWNED: [(&str, &str); 7] = [
    ("--model", "the checkpoint row sets it"),
    (
        "--tensor-parallel-size",
        "the profile and the GPUs picked set it",
    ),
    ("--devices", "CUDA_VISIBLE_DEVICES places the GPUs picked"),
    ("--bundle", "serve --model prepares the package"),
    ("--checkpoint", "the checkpoint row sets --model"),
    ("--profile", "the Profiles pane chooses it"),
    ("--topology", "serve --model picks it"),
];
/// Variables that say which GPUs a process uses. choptop sets the first
/// from the GPUs it checked; a custom row sets none of them.
pub const PLACEMENT_VARIABLES: [&str; 3] = [
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
];
/// The API key travels in the environment, never on the command line.
pub const API_KEY: &str = "CHOPINFER_API_KEY";

/// What a form value sets.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Key {
    /// A serve flag, by name without dashes.
    Flag(String),
    /// A variable of the launch's environment.
    Variable(String),
    /// A package option, set with `--model-option NAME=VALUE`.
    ModelOption(String),
}

impl Key {
    /// `--max-sequences`, `CHOPINFER_API_KEY` or `--model-option NAME`.
    pub fn shown(&self) -> String {
        match self {
            Self::Flag(name) => format!("--{name}"),
            Self::Variable(name) => name.clone(),
            Self::ModelOption(name) => format!("--model-option {name}"),
        }
    }
}

/// Where a piece of the command line comes from, which sets its color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The environment.
    Environment,
    /// `chopinfer serve --model DIR --tensor-parallel-size N --listen`.
    Base,
    /// The chosen profile: flags that make the pick serve it, and resets
    /// of model options the pick sets.
    Profile,
    /// A value set in the form.
    Set,
    /// A custom row, as written.
    Custom,
}

/// Splits a custom row into words at spaces, keeping what double quotes
/// enclose together; the quotes themselves are dropped.
pub fn split(row: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut quoted = false;
    let mut started = false;
    for character in row.chars() {
        match character {
            '"' => {
                quoted = !quoted;
                started = true;
            }
            ' ' | '\t' if !quoted => {
                if started {
                    words.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            other => {
                word.push(other);
                started = true;
            }
        }
    }
    if quoted {
        return Err("a double quote is not closed".to_owned());
    }
    if started {
        words.push(word);
    }
    Ok(words)
}

/// Writes `words` back as a row [`split`] reads the same.
pub fn join(words: &[String]) -> String {
    words
        .iter()
        .map(|word| {
            if word.is_empty() || word.contains([' ', '\t']) {
                format!("\"{word}\"")
            } else {
                word.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// `word` as a POSIX shell reads it back as one word: as it is when every
/// character is a letter, digit or one of `_@%+=:,./-`, else in single
/// quotes, a quote inside as `'\''`. Only for showing a command line; the
/// arguments a process gets are never quoted.
pub fn quoted(word: &str) -> String {
    let plain =
        |character: char| character.is_ascii_alphanumeric() || "_@%+=:,./-".contains(character);
    if !word.is_empty() && word.chars().all(plain) {
        return word.to_owned();
    }
    format!("'{}'", word.replace('\'', "'\\''"))
}

/// `words` as a command line a POSIX shell reads back as them.
pub fn line(words: &[String]) -> String {
    words
        .iter()
        .map(|word| quoted(word))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `NAME=value` as shown: the value quoted as [`quoted`] does, or `****`
/// for a secret.
pub fn shown_variable(name: &str, value: &str, secret: bool) -> String {
    if secret {
        format!("{name}=****")
    } else {
        format!("{name}={}", quoted(value))
    }
}

/// `NAME=value` with a variable's name: capitals, digits and underscores.
fn assignment(word: &str) -> Option<(&str, &str)> {
    let (name, value) = word.split_once('=')?;
    let mut characters = name.chars();
    let first = characters.next()?;
    ((first.is_ascii_uppercase() || first == '_')
        && characters.all(|character| {
            character.is_ascii_uppercase() || character.is_ascii_digit() || character == '_'
        }))
    .then_some((name, value))
}

/// What a custom row holds once read against the catalog.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Classified {
    /// Values of options the form holds, which the row moves onto them.
    pub folds: Vec<(Key, String)>,
    /// Serve arguments passed as written.
    pub arguments: Vec<String>,
    /// Variables set in the launch's environment, after the form's.
    pub variables: Vec<(String, String)>,
}

impl Classified {
    /// What stays a custom row once the folds are moved: the arguments and
    /// variables, as a row.
    pub fn rest(&self) -> Option<String> {
        let mut words = self
            .variables
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>();
        words.extend(self.arguments.iter().cloned());
        (!words.is_empty()).then(|| join(&words))
    }
}

/// Reads a custom row: a flag or variable the catalog holds is moved onto
/// its option, `--listen` onto the listen row and `--api-key` onto the
/// key's variable. A flag choptop owns or a placement variable is refused
/// with the reason.
pub fn classify(row: &str, catalog: Option<&ServeOptions>) -> Result<Classified, String> {
    let words = split(row)?;
    let mut classified = Classified::default();
    let mut index = 0;
    while index < words.len() {
        let word = &words[index];
        index += 1;
        if let Some((name, value)) = assignment(word) {
            if PLACEMENT_VARIABLES.contains(&name) {
                return Err(format!(
                    "{name} is set from the GPUs picked in Placement, so only checked GPUs are used"
                ));
            }
            if name == API_KEY || catalog.is_some_and(|catalog| catalog.variable(name).is_some()) {
                classified
                    .folds
                    .push((Key::Variable(name.to_owned()), value.to_owned()));
            } else {
                classified
                    .variables
                    .push((name.to_owned(), value.to_owned()));
            }
            continue;
        }
        let Some(flag) = word.strip_prefix("--").filter(|flag| !flag.is_empty()) else {
            classified.arguments.push(word.clone());
            continue;
        };
        let (name, inline) = match flag.split_once('=') {
            Some((name, value)) => (name, Some(value.to_owned())),
            None => (flag, None),
        };
        if let Some((owned, why)) = OWNED.iter().find(|(owned, _)| owned[2..] == *name) {
            return Err(format!("{owned} is choptop's to set: {why}"));
        }
        // The words up to the next flag or variable are its values.
        let mut following = Vec::new();
        while index < words.len()
            && !words[index].starts_with("--")
            && assignment(&words[index]).is_none()
        {
            following.push(words[index].clone());
            index += 1;
        }
        let values = inline
            .into_iter()
            .chain(following.clone())
            .collect::<Vec<_>>();
        let option = catalog.and_then(|catalog| catalog.option(name));
        let single = |values: &[String]| match values {
            [value] => Ok(value.clone()),
            [] => Err(format!("--{name} needs a value")),
            _ => Err(format!("--{name} takes one value; quote one with spaces")),
        };
        match name {
            "listen" => classified
                .folds
                .push((Key::Flag("listen".to_owned()), single(&values)?)),
            "api-key" => classified
                .folds
                .push((Key::Variable(API_KEY.to_owned()), single(&values)?)),
            "model-option" => {
                let value = single(&values)?;
                let (option, setting) = value
                    .split_once('=')
                    .ok_or_else(|| "--model-option takes NAME=VALUE".to_owned())?;
                classified
                    .folds
                    .push((Key::ModelOption(option.to_owned()), setting.to_owned()));
            }
            _ => match option {
                Some(option) if option.model_launch && !option.repeatable => {
                    let value = if option.shape == Shape::Switch {
                        if !values.is_empty() {
                            return Err(format!("--{name} takes no value"));
                        }
                        "on".to_owned()
                    } else {
                        single(&values)?
                    };
                    classified.folds.push((Key::Flag(name.to_owned()), value));
                }
                _ => {
                    classified.arguments.push(word.clone());
                    classified.arguments.extend(following);
                }
            },
        }
    }
    Ok(classified)
}

/// The form's choices as explicit serve options and variables.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Explicit {
    /// Groups of serve arguments, each a flag and its values, with where
    /// it comes from.
    pub options: Vec<(Vec<String>, Origin)>,
    /// Variables to set after `CUDA_VISIBLE_DEVICES`: the form's, then the
    /// custom rows'.
    pub environment: Vec<(String, String, Origin)>,
    /// The listen address the form sets.
    pub listen: Option<String>,
    /// What is wrong with the form.
    pub blocked: Vec<String>,
    /// What the command waits for.
    pub waiting: Vec<String>,
}

impl Explicit {
    /// The options as `chopinfer models command` takes them after `--`.
    pub fn arguments(&self) -> Vec<String> {
        self.options
            .iter()
            .flat_map(|(group, _)| group.iter().cloned())
            .collect()
    }
}

/// The form: values set, by what they set, and custom rows.
pub struct Form<'a> {
    pub edits: &'a BTreeMap<Key, String>,
    pub custom: &'a [String],
}

/// Turns `form` into explicit options. Each model option the pick sets and
/// the chosen profile leaves out (`inherited`) is reset to its package
/// default, so the command stays blocked until `package` options are read.
pub fn explicit(
    form: &Form,
    catalog: Option<&ServeOptions>,
    inherited: &[String],
    package: Option<&[PackageOption]>,
) -> Explicit {
    let mut explicit = Explicit::default();
    let switch = |name: &str| {
        catalog
            .and_then(|catalog| catalog.option(name))
            .is_some_and(|option| option.shape == Shape::Switch)
    };
    let flag = |explicit: &mut Explicit, name: &str, value: &str, origin| {
        if switch(name) {
            if value == "on" {
                explicit.options.push((vec![format!("--{name}")], origin));
            }
        } else {
            explicit
                .options
                .push((vec![format!("--{name}"), value.to_owned()], origin));
        }
    };
    let model_option =
        |name: &str, value: &str| vec!["--model-option".to_owned(), format!("{name}={value}")];
    for (key, value) in form.edits {
        match key {
            // The checkpoint is part of the base command.
            Key::Flag(name) if name == "model" => {}
            Key::Flag(name) if name == "listen" => explicit.listen = Some(value.clone()),
            Key::Flag(name) => flag(&mut explicit, name, value, Origin::Set),
            Key::Variable(name) => {
                explicit
                    .environment
                    .push((name.clone(), value.clone(), Origin::Set));
            }
            Key::ModelOption(name) => explicit
                .options
                .push((model_option(name, value), Origin::Set)),
        }
    }
    for row in form.custom {
        match classify(row, catalog) {
            Err(reason) => explicit.blocked.push(reason),
            Ok(classified) => {
                for (key, value) in classified.folds {
                    match key {
                        Key::Flag(name) if name == "listen" => explicit.listen = Some(value),
                        Key::Flag(name) => flag(&mut explicit, &name, &value, Origin::Custom),
                        Key::Variable(name) => {
                            explicit.environment.push((name, value, Origin::Custom));
                        }
                        Key::ModelOption(name) => explicit
                            .options
                            .push((model_option(&name, &value), Origin::Custom)),
                    }
                }
                if !classified.arguments.is_empty() {
                    explicit
                        .options
                        .push((classified.arguments, Origin::Custom));
                }
                for (name, value) in classified.variables {
                    explicit.environment.push((name, value, Origin::Custom));
                }
            }
        }
    }
    for name in inherited {
        if form.edits.contains_key(&Key::ModelOption(name.clone())) {
            continue;
        }
        match package {
            None => explicit
                .waiting
                .push(format!("the package's options, to reset {name}")),
            Some(options) => match options.iter().find(|option| option.name == *name) {
                Some(option) => explicit
                    .options
                    .push((model_option(name, &option.default_text()), Origin::Profile)),
                None => explicit
                    .blocked
                    .push(format!("the package declares no model option {name}")),
            },
        }
    }
    explicit
}

/// The environment one instance adds to choptop's own: the UUIDs of its
/// GPUs in `CUDA_VISIBLE_DEVICES`, overriding any inherited value, then the
/// form's variables and the custom rows'.
pub fn environment(explicit: &Explicit, uuids: &[String]) -> Vec<(String, String)> {
    let mut environment = vec![("CUDA_VISIBLE_DEVICES".to_owned(), uuids.join(","))];
    environment.extend(
        explicit
            .environment
            .iter()
            .filter(|(name, _, _)| !PLACEMENT_VARIABLES.contains(&name.as_str()))
            .map(|(name, value, _)| (name.clone(), value.clone())),
    );
    environment
}

/// The argument list one instance runs: the final list from `models
/// command` and its listen address.
pub fn arguments(command: &[String], listen: &str) -> Vec<String> {
    let mut arguments = command.to_vec();
    arguments.push("--listen".to_owned());
    arguments.push(listen.to_owned());
    arguments
}

/// The command box's tokens for one instance: the environment, with
/// `CUDA_VISIBLE_DEVICES` as the GPU indices it stands for and secrets
/// hidden, then `program` and the arguments, each flag with its values,
/// every word quoted as [`quoted`] does.
pub fn tokens(
    program: &str,
    command: &[String],
    explicit: &Explicit,
    indices: &[u32],
    listen: &str,
    secrets: &[&str],
) -> Vec<(String, Origin)> {
    let shown_indices = indices
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut tokens = vec![(
        format!("CUDA_VISIBLE_DEVICES={shown_indices}"),
        Origin::Environment,
    )];
    for (name, value, _) in &explicit.environment {
        tokens.push((
            shown_variable(name, value, secrets.contains(&name.as_str())),
            Origin::Environment,
        ));
    }
    let mut words = command.iter().peekable();
    let mut first = vec![program.to_owned()];
    if words.peek().map(|word| word.as_str()) == Some("serve") {
        first.extend(words.next().cloned());
    }
    tokens.push((line(&first), Origin::Base));
    let mut groups: Vec<Vec<String>> = Vec::new();
    for word in words {
        match groups.last_mut() {
            Some(group) if !word.starts_with("--") => group.push(word.clone()),
            _ => groups.push(vec![word.clone()]),
        }
    }
    for group in groups {
        let origin = match group[0].as_str() {
            "--model" | "--tensor-parallel-size" => Origin::Base,
            _ => explicit
                .options
                .iter()
                .find(|(words, _)| *words == group)
                .map_or(Origin::Profile, |(_, origin)| *origin),
        };
        tokens.push((line(&group), origin));
    }
    tokens.push((
        format!("--listen {}", quoted(listen)),
        if explicit.listen.is_some() {
            Origin::Set
        } else {
            Origin::Base
        },
    ));
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{PackageKind, parse_package_options};

    const SERVE_OPTIONS: &str = include_str!("../../tests/fixtures/contracts/serve-options.json");
    const PACKAGE: &str = include_str!("../../tests/fixtures/contracts/list-model-options.json");

    fn catalog() -> ServeOptions {
        ServeOptions::parse(SERVE_OPTIONS).unwrap()
    }

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn rows_split_at_spaces_and_quotes_group() {
        assert_eq!(
            split(r#"--served-model-name "Qwen Flash"#).unwrap_err(),
            "a double quote is not closed"
        );
        assert_eq!(
            split(r#"--served-model-name "Qwen Flash" --x=" y z" "" w"#).unwrap(),
            ["--served-model-name", "Qwen Flash", "--x= y z", "", "w"]
        );
        assert_eq!(split("  a\tb  ").unwrap(), ["a", "b"]);
        assert!(split("").unwrap().is_empty());
        let row = vec!["--name".to_owned(), "two words".to_owned(), String::new()];
        assert_eq!(split(&join(&row)).unwrap(), row);
    }

    #[test]
    fn a_custom_row_naming_a_catalog_option_moves_onto_it() {
        let catalog = catalog();
        let classified = classify(
            "--max-sequences 48 --prefix-cache=off --allow-credentials CHOPINFER_API_KEY=k \
             --listen 0.0.0.0:9000 --model-option dspark_draft_chunk_items=16 \
             --kv-offload-prefix off --x=1 y TRACE=1",
            Some(&catalog),
        )
        .unwrap();
        assert_eq!(
            classified.folds,
            [
                (Key::Flag("max-sequences".into()), "48".into()),
                (Key::Flag("prefix-cache".into()), "off".into()),
                (Key::Flag("allow-credentials".into()), "on".into()),
                (Key::Variable("CHOPINFER_API_KEY".into()), "k".into()),
                (Key::Flag("listen".into()), "0.0.0.0:9000".into()),
                (
                    Key::ModelOption("dspark_draft_chunk_items".into()),
                    "16".into()
                ),
            ]
        );
        // A flag the catalog does not hold is passed as written.
        assert_eq!(
            classified.arguments,
            words("--kv-offload-prefix off --x=1 y")
        );
        assert_eq!(classified.variables, [("TRACE".to_owned(), "1".to_owned())]);
        assert_eq!(
            classified.rest().as_deref(),
            Some("TRACE=1 --kv-offload-prefix off --x=1 y")
        );
        // --api-key is the key's variable; a repeatable flag stays a row.
        let key = classify("--api-key secret", Some(&catalog)).unwrap();
        assert_eq!(
            key.folds,
            [(Key::Variable(API_KEY.into()), "secret".into())]
        );
        let media = classify("--allowed-local-media-path /media", Some(&catalog)).unwrap();
        assert_eq!(media.arguments, words("--allowed-local-media-path /media"));
        assert!(media.folds.is_empty());
        // Before the catalog is read nothing folds but listen and the key.
        let early = classify("--max-sequences 48", None).unwrap();
        assert_eq!(early.arguments, words("--max-sequences 48"));
    }

    #[test]
    fn owned_flags_and_placement_variables_are_refused() {
        let catalog = catalog();
        for (row, reason) in [
            ("--model /models/other", "--model is choptop's to set"),
            (
                "--tensor-parallel-size 8",
                "--tensor-parallel-size is choptop's",
            ),
            ("--devices 0,1", "CUDA_VISIBLE_DEVICES places the GPUs"),
            ("--bundle=/pkg", "--bundle is choptop's"),
            ("--checkpoint /c", "--checkpoint is choptop's"),
            ("--profile tp4-ar", "--profile is choptop's"),
            ("--topology x", "--topology is choptop's"),
            ("CUDA_VISIBLE_DEVICES=0,1", "only checked GPUs are used"),
            (
                "CUDA_DEVICE_ORDER=PCI_BUS_ID",
                "CUDA_DEVICE_ORDER is set from",
            ),
            ("--x 1 NVIDIA_VISIBLE_DEVICES=all", "NVIDIA_VISIBLE_DEVICES"),
            ("--max-sequences", "--max-sequences needs a value"),
            ("--max-sequences 1 2", "takes one value"),
            ("--allow-credentials yes", "takes no value"),
            ("--model-option x", "NAME=VALUE"),
        ] {
            let error = classify(row, Some(&catalog)).unwrap_err();
            assert!(error.contains(reason), "{row}: {error}");
        }
    }

    fn package() -> Vec<PackageOption> {
        parse_package_options(PACKAGE).unwrap()
    }

    #[test]
    fn the_form_becomes_explicit_options_and_inherited_resets() {
        let catalog = catalog();
        let mut edits = BTreeMap::new();
        edits.insert(Key::Flag("model".into()), "/models/deepseek".into());
        edits.insert(Key::Flag("max-sequences".into()), "48".into());
        edits.insert(Key::Flag("allow-credentials".into()), "on".into());
        edits.insert(Key::Flag("listen".into()), "0.0.0.0:9000".into());
        edits.insert(Key::Variable(API_KEY.into()), "secret".into());
        edits.insert(
            Key::ModelOption("prefill_upper_skip".into()),
            "false".into(),
        );
        let custom = vec!["--kv-offload-prefix off".to_owned(), "TRACE=1".to_owned()];
        let form = Form {
            edits: &edits,
            custom: &custom,
        };
        let inherited = vec![
            "dspark_draft_chunk_items".to_owned(),
            "prefill_upper_skip".to_owned(),
        ];
        // Until the package's options are read the defaults are unknown.
        let waiting = explicit(&form, Some(&catalog), &inherited, None);
        assert_eq!(
            waiting.waiting,
            ["the package's options, to reset dspark_draft_chunk_items"]
        );
        assert!(waiting.blocked.is_empty());
        let package = package();
        let explicit = explicit(&form, Some(&catalog), &inherited, Some(&package));
        assert!(explicit.blocked.is_empty(), "{:?}", explicit.blocked);
        assert_eq!(
            explicit.arguments(),
            words(
                "--allow-credentials --max-sequences 48 \
                 --model-option prefill_upper_skip=false --kv-offload-prefix off \
                 --model-option dspark_draft_chunk_items=12"
            )
        );
        assert_eq!(explicit.options[4].1, Origin::Profile);
        assert_eq!(explicit.options[3].1, Origin::Custom);
        assert_eq!(explicit.listen.as_deref(), Some("0.0.0.0:9000"));
        // The key is in the environment, never on the command line.
        assert!(
            !explicit
                .arguments()
                .iter()
                .any(|word| word.contains("secret"))
        );
        let uuids = vec!["GPU-a".to_owned(), "GPU-b".to_owned()];
        assert_eq!(
            environment(&explicit, &uuids),
            [
                ("CUDA_VISIBLE_DEVICES".to_owned(), "GPU-a,GPU-b".to_owned()),
                (API_KEY.to_owned(), "secret".to_owned()),
                ("TRACE".to_owned(), "1".to_owned()),
            ]
        );
        // A package without the option cannot reset it.
        let none = super::explicit(&form, Some(&catalog), &inherited, Some(&[]));
        assert_eq!(
            none.blocked,
            ["the package declares no model option dspark_draft_chunk_items"]
        );
        // A switch set off is left out.
        let mut off = edits.clone();
        off.insert(Key::Flag("allow-credentials".into()), "off".into());
        let form = Form {
            edits: &off,
            custom: &[],
        };
        let without = super::explicit(&form, Some(&catalog), &[], Some(&package));
        assert!(
            !without
                .arguments()
                .contains(&"--allow-credentials".to_owned())
        );
        assert!(matches!(package[0].kind, PackageKind::Integer { .. }));
    }

    #[test]
    fn the_command_box_shows_indices_and_colors_each_flag_by_origin() {
        let mut edits = BTreeMap::new();
        edits.insert(Key::Flag("max-sequences".into()), "48".into());
        edits.insert(Key::Variable(API_KEY.into()), "secret".into());
        let custom = vec!["--kv-offload-prefix off".to_owned()];
        let explicit = explicit(
            &Form {
                edits: &edits,
                custom: &custom,
            },
            Some(&catalog()),
            &[],
            Some(&[]),
        );
        let command = words(
            "serve --model /models/qwen --tensor-parallel-size 2 --speculative-tokens 0 \
             --max-sequences 48 --kv-offload-prefix off",
        );
        let tokens = tokens(
            "chopinfer",
            &command,
            &explicit,
            &[0, 1],
            "127.0.0.1:8310",
            &[API_KEY],
        );
        assert_eq!(
            tokens,
            [
                ("CUDA_VISIBLE_DEVICES=0,1".to_owned(), Origin::Environment),
                ("CHOPINFER_API_KEY=****".to_owned(), Origin::Environment),
                ("chopinfer serve".to_owned(), Origin::Base),
                ("--model /models/qwen".to_owned(), Origin::Base),
                ("--tensor-parallel-size 2".to_owned(), Origin::Base),
                ("--speculative-tokens 0".to_owned(), Origin::Profile),
                ("--max-sequences 48".to_owned(), Origin::Set),
                ("--kv-offload-prefix off".to_owned(), Origin::Custom),
                ("--listen 127.0.0.1:8310".to_owned(), Origin::Base),
            ]
        );
        assert_eq!(
            arguments(&command, "127.0.0.1:8310")[command.len()..],
            ["--listen", "127.0.0.1:8310"]
        );
    }

    #[test]
    fn words_with_spaces_or_shell_characters_are_shown_quoted() {
        assert_eq!(
            quoted("/models/qwen-3.8_a:b,c@d%e+f=g"),
            "/models/qwen-3.8_a:b,c@d%e+f=g"
        );
        assert_eq!(
            quoted("/models/x/checkpoint with spaces"),
            "'/models/x/checkpoint with spaces'"
        );
        assert_eq!(quoted("it's"), r"'it'\''s'");
        assert_eq!(quoted(""), "''");
        assert_eq!(quoted("$HOME;*"), "'$HOME;*'");
        let mut edits = BTreeMap::new();
        edits.insert(Key::Variable(API_KEY.into()), "a secret".into());
        let custom = vec!["CHOPINFER_LOG_LEVEL=\"info debug\"".to_owned()];
        let explicit = explicit(
            &Form {
                edits: &edits,
                custom: &custom,
            },
            Some(&catalog()),
            &[],
            Some(&[]),
        );
        let command = vec![
            "serve".to_owned(),
            "--model".to_owned(),
            "/models/x/checkpoint with spaces".to_owned(),
            "--tensor-parallel-size".to_owned(),
            "2".to_owned(),
        ];
        let shown = tokens(
            "chopinfer",
            &command,
            &explicit,
            &[0, 1],
            "[::1]:8310",
            &[API_KEY],
        )
        .into_iter()
        .map(|(token, _)| token)
        .collect::<Vec<_>>();
        assert_eq!(
            shown,
            [
                "CUDA_VISIBLE_DEVICES=0,1",
                "CHOPINFER_API_KEY=****",
                "CHOPINFER_LOG_LEVEL='info debug'",
                "chopinfer serve",
                "--model '/models/x/checkpoint with spaces'",
                "--tensor-parallel-size 2",
                "--listen '[::1]:8310'",
            ]
        );
        // The process gets the words themselves.
        assert_eq!(
            arguments(&command, "[::1]:8310")[2],
            "/models/x/checkpoint with spaces"
        );
    }
}
