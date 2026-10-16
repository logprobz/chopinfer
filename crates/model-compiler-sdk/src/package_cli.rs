//! Common package CLI option parsing.
use std::collections::BTreeMap;
use std::path::PathBuf;

pub struct Options {
    pub values: BTreeMap<String, String>,
}

/// Installation flags of earlier packagers: a package is served from the
/// directory it is emitted to.
const INSTALL_FLAGS: [&str; 3] = ["--no-install", "--install-root", "--chopinfer"];

/// Parse valued options. Removed model flags and the installation flags
/// retain a diagnostic instead of consuming a value.
pub fn parse(
    args: impl IntoIterator<Item = String>,
    removed: &[(&str, &str)],
) -> Result<Options, String> {
    let mut values = BTreeMap::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if let Some((_, message)) = removed.iter().find(|(flag, _)| *flag == arg) {
            return Err((*message).into());
        }
        if INSTALL_FLAGS.contains(&arg.as_str()) {
            return Err(format!(
                "{arg} is gone: packages are not installed; serve the --output directory with \
                 chopinfer serve --bundle DIR --checkpoint DIR"
            ));
        }
        if !arg.starts_with("--") {
            return Err(format!("unexpected argument {arg}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        if values.insert(arg[2..].to_owned(), value).is_some() {
            return Err(format!("{arg} given twice"));
        }
    }
    Ok(Options { values })
}

pub fn path(
    options: &mut BTreeMap<String, String>,
    name: &str,
    env: Option<&str>,
) -> Result<PathBuf, String> {
    options
        .remove(name)
        .map(PathBuf::from)
        .or_else(|| env.and_then(std::env::var_os).map(PathBuf::from))
        .ok_or_else(|| format!("missing --{name}"))
}

/// An `on` or `off` option, `off` when absent.
pub fn switch(options: &mut BTreeMap<String, String>, name: &str) -> Result<bool, String> {
    match options.remove(name).as_deref() {
        None | Some("off") => Ok(false),
        Some("on") => Ok(true),
        Some(other) => Err(format!("--{name} must be on or off, not {other}")),
    }
}

/// Refuses the first option no reader took.
pub fn refuse_unknown(options: &BTreeMap<String, String>) -> Result<(), String> {
    match options.first_key_value() {
        Some((name, _)) => Err(format!("unknown option --{name}")),
        None => Ok(()),
    }
}

/// A packager's `main`: `--help` or `-h` prints `usage`, and a failing `run`
/// prints its error after `name` and fails.
pub fn main<E: std::fmt::Display>(
    name: &str,
    usage: &str,
    run: impl FnOnce() -> Result<(), E>,
) -> std::process::ExitCode {
    if std::env::args()
        .skip(1)
        .any(|arg| arg == "--help" || arg == "-h")
    {
        println!("{usage}");
        return std::process::ExitCode::SUCCESS;
    }
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{name}: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_preserves_values_and_refuses_duplicate_missing_and_positional_arguments() {
        let parse = |args: &[&str]| super::parse(args.iter().map(|arg| (*arg).to_owned()), &[]);
        let parsed = parse(&["--module", "a b"]).unwrap();
        assert_eq!(parsed.values["module"], "a b");
        assert!(parse(&["--module"]).is_err());
        assert!(parse(&["--module", "a", "--module", "b"]).is_err());
        assert!(parse(&["positional"]).is_err());
        for flag in super::INSTALL_FLAGS {
            let error = parse(&[flag, "value"]).err().unwrap();
            assert!(error.contains("packages are not installed"), "{error}");
        }
    }
}
