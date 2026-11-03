//! Compiler configuration evidence shared by local reuse, the wrapper and reports.

use crate::{BuildError, BuildOptions, GeneratedUnit, Toolchain, digest, err, io, run};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const MODELLED_SETTINGS: &[&str] = &[
    "PATH",
    "LD_LIBRARY_PATH",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "NVCC_CCBIN",
    "SOURCE_DATE_EPOCH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "GCC_EXEC_PREFIX",
    "COMPILER_PATH",
];
const LOADER_VARIABLES: &[&str] = &["LD_PRELOAD", "LD_AUDIT"];
const SEARCH_SETTINGS: &[&str] = &[
    "PATH",
    "LD_LIBRARY_PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "GCC_EXEC_PREFIX",
    "COMPILER_PATH",
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilerInput {
    pub lookup: PathBuf,
    pub location: PathBuf,
    pub sha256: String,
}

/// Additive freshness evidence. It is not part of a locked cubin identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilerFreshness {
    pub version: u32,
    pub toolchain_version: String,
    pub flags: Vec<String>,
    pub environment: BTreeMap<String, Option<String>>,
    pub files: BTreeMap<String, CompilerInput>,
    pub search_directories: BTreeMap<PathBuf, Option<Vec<String>>>,
    pub sha256: String,
}

impl CompilerFreshness {
    fn identity(&self) -> Result<String, BuildError> {
        let files: BTreeMap<_, _> = self.files.iter().map(|(k, v)| (k, &v.sha256)).collect();
        let mut environment: BTreeMap<String, serde_json::Value> = self
            .environment
            .iter()
            .map(|(key, value)| (key.clone(), serde_json::json!(value)))
            .collect();
        for name in ["PATH", "LD_LIBRARY_PATH"] {
            let candidates: Vec<_> = self
                .environment
                .get(name)
                .and_then(Option::as_deref)
                .filter(|value| !value.is_empty())
                .into_iter()
                .flat_map(|value| value.split(':'))
                .filter_map(|path| {
                    self.search_directories
                        .get(Path::new(path))
                        .and_then(Option::as_ref)
                })
                .flatten()
                .collect();
            environment.insert(name.into(), serde_json::json!(candidates));
        }
        let mut searches: Vec<_> = self
            .search_directories
            .values()
            .filter_map(Option::as_ref)
            .filter(|entries| !entries.is_empty())
            .collect();
        searches.sort();
        let material = (
            &self.version,
            &self.toolchain_version,
            &self.flags,
            environment,
            files,
            searches,
        );
        Ok(digest(
            &serde_json::to_vec(&material).map_err(|e| err(e.to_string()))?,
        ))
    }

    pub(crate) fn verify_current(&self) -> Result<(), BuildError> {
        if self.version != 1
            || self.identity()? != self.sha256
            || !self.files.contains_key("nvcc")
            || !self.files.contains_key("nvcc.profile")
            || MODELLED_SETTINGS
                .iter()
                .chain(LOADER_VARIABLES)
                .any(|name| !self.environment.contains_key(*name))
        {
            return Err(err("invalid compiler freshness descriptor"));
        }
        if env::vars_os().any(|(name, _)| {
            name.to_str().is_some_and(|name| {
                name.starts_with("NVCC_") && !self.environment.contains_key(name)
            })
        }) {
            return Err(err("unmodelled nvcc environment"));
        }
        for (name, expected) in &self.environment {
            let current = env::var(name).ok();
            let matches =
                if current != *expected && matches!(name.as_str(), "PATH" | "LD_LIBRARY_PATH") {
                    // Cargo adds build/tool directories to these variables. Empty
                    // search directories cannot change the selected compiler bytes.
                    let names = self.files.keys().cloned().collect();
                    search_candidates(current.as_deref(), &names)?
                        == search_candidates(expected.as_deref(), &names)?
                } else {
                    current == *expected
                };
            if !matches {
                return Err(err(format!("stale compiler environment: {name}")));
            }
        }
        for input in self.files.values() {
            if fs::canonicalize(&input.lookup).map_err(|e| io("resolve compiler input", e))?
                != input.location
                || digest(&fs::read(&input.location).map_err(|e| io("read compiler input", e))?)
                    != input.sha256
            {
                return Err(err("stale compiler input"));
            }
        }
        for (path, expected) in &self.search_directories {
            if directory_entries(path, &self.files.keys().cloned().collect())? != *expected {
                return Err(err("stale compiler search directory"));
            }
        }
        Ok(())
    }

    pub(crate) fn variables(
        &self,
        closure: &str,
        options: BuildOptions,
    ) -> [(&'static str, String); 3] {
        variables(&self.sha256, closure, options)
    }

    pub(crate) fn emit_cargo(&self) {
        for name in self.environment.keys() {
            println!("cargo:rerun-if-env-changed={name}");
        }
        for input in self.files.values() {
            println!("cargo:rerun-if-changed={}", input.location.display());
            if input.lookup != input.location {
                println!("cargo:rerun-if-changed={}", input.lookup.display());
            }
        }
        // Cargo scans watched directories recursively. Its library search path
        // includes the profile directory and therefore this script's own output.
        // Watch existing candidates as files; each driver invocation and report
        // validation still checks the complete search inventory.
        for (directory, entries) in &self.search_directories {
            for entry in entries.iter().flatten() {
                if let Some((name, _)) = entry.rsplit_once(':') {
                    println!("cargo:rerun-if-changed={}", directory.join(name).display());
                }
            }
        }
    }
}

pub(crate) fn compiler_freshness(
    toolchain: &Toolchain,
    flags: &[&str],
) -> Option<CompilerFreshness> {
    // No path-only memoization: a caller may change inputs and build again in this process.
    match compiler_inputs(toolchain, flags) {
        Ok(mut value) => {
            value.sha256 = value.identity().ok()?;
            Some(value)
        }
        Err(error) => {
            eprintln!("compiler freshness unavailable, compiling without reuse: {error}");
            None
        }
    }
}

pub(crate) fn emit_environment() {
    for name in MODELLED_SETTINGS.iter().chain(LOADER_VARIABLES) {
        println!("cargo:rerun-if-env-changed={name}");
    }
}

fn variables(compiler: &str, closure: &str, options: BuildOptions) -> [(&'static str, String); 3] {
    [
        ("CHOPINFER_NVCC_CACHE_CLOSURE", closure.into()),
        ("CHOPINFER_NVCC_CACHE_TOOLCHAIN", compiler.into()),
        (
            "CHOPINFER_NVCC_CACHE_LOOKUP",
            if !options.lock_update && !options.profile_line_info {
                "1"
            } else {
                "0"
            }
            .into(),
        ),
    ]
}

fn names_build_time(bytes: &[u8]) -> bool {
    [b"__DATE__".as_slice(), b"__TIME__", b"__TIMESTAMP__"]
        .iter()
        .any(|name| bytes.windows(name.len()).any(|window| window == *name))
}

pub(crate) fn sources_reusable(
    files: &[PathBuf],
    unit: &GeneratedUnit,
    defines: &[(String, String)],
) -> bool {
    !names_build_time(unit.translation_unit.as_bytes())
        && !names_build_time(unit.header.as_bytes())
        && !defines.iter().any(|(name, value)| {
            names_build_time(name.as_bytes()) || names_build_time(value.as_bytes())
        })
        && !files
            .iter()
            .any(|file| fs::read(file).map_or(true, |bytes| names_build_time(&bytes)))
}

fn safe_option(option: &str) -> bool {
    if let Some(options) = option.strip_prefix("-Xptxas=") {
        return options.split(',').all(|v| {
            matches!(
                v,
                "-O0" | "-O1" | "-O2" | "-O3" | "-v" | "-lineinfo" | "-fmad=true" | "-fmad=false"
            ) || v.strip_prefix("-warn-").is_some_and(|s| {
                !s.is_empty() && s.bytes().all(|c| c.is_ascii_lowercase() || c == b'-')
            }) || v
                .strip_prefix("-maxrregcount=")
                .is_some_and(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()))
        });
    }
    matches!(
        option,
        "-cubin"
            | "-lineinfo"
            | "-O0"
            | "-O1"
            | "-O2"
            | "-O3"
            | "--fmad=true"
            | "--fmad=false"
            | "--use_fast_math"
    ) || option
        .strip_prefix("-std=c++")
        .is_some_and(|s| s.len() == 2 && s.bytes().all(|c| c.is_ascii_digit()))
        || option.strip_prefix("-arch=").is_some_and(|s| {
            !s.is_empty() && s.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        })
        || option.strip_prefix("-I").is_some_and(|s| !s.is_empty())
        || option
            .strip_prefix("-D")
            .is_some_and(|s| crate::identifier(s.split('=').next().unwrap_or_default()))
}

fn directory_entries(
    path: &Path,
    names: &BTreeSet<String>,
) -> Result<Option<Vec<String>>, BuildError> {
    match fs::read_dir(path) {
        Ok(entries) => {
            let mut values = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|e| io("read compiler search directory", e))?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if names.contains(&name) {
                    let bytes = fs::read(entry.path())
                        .map_err(|e| io("read compiler search candidate", e))?;
                    values.push(format!("{name}:{}", digest(&bytes)));
                }
            }
            values.sort();
            Ok(Some(values))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(io("read compiler search directory", error)),
    }
}

fn search_candidates(
    value: Option<&str>,
    names: &BTreeSet<String>,
) -> Result<Vec<String>, BuildError> {
    let mut candidates = Vec::new();
    for path in value
        .filter(|value| !value.is_empty())
        .into_iter()
        .flat_map(|value| value.split(':'))
    {
        let path = Path::new(path);
        if !path.is_absolute() {
            return Err(err("relative compiler search path"));
        }
        candidates.extend(directory_entries(path, names)?.unwrap_or_default());
    }
    Ok(candidates)
}

fn compiler_inputs(toolchain: &Toolchain, flags: &[&str]) -> Result<CompilerFreshness, BuildError> {
    if env::vars_os().any(|(name, _)| {
        name.to_str()
            .is_some_and(|name| name.starts_with("NVCC_") && !MODELLED_SETTINGS.contains(&name))
    }) {
        return Err(err("unmodelled nvcc environment"));
    }
    let nvcc = toolchain.tool("nvcc");
    let output = Command::new(&nvcc)
        .current_dir(&toolchain.home)
        .arg("-dryrun")
        .args(flags)
        .args(["identity.cu", "-o", "identity.cubin"])
        .output()
        .map_err(|e| io("nvcc -dryrun", e))?;
    if !output.status.success() {
        return Err(err("nvcc -dryrun failed"));
    }
    let text = String::from_utf8(output.stderr)
        .map_err(|e| err(format!("nvcc -dryrun is not UTF-8: {e}")))?;
    let mut files = BTreeMap::<String, CompilerInput>::new();
    let mut add = |path: &Path| -> Result<(), BuildError> {
        // Retain the lookup path as well as its bytes, so replacing a symlink's target is seen.
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            toolchain.home.join(path)
        };
        let name = path
            .file_name()
            .ok_or_else(|| err("unnamed compiler input"))?
            .to_string_lossy()
            .into_owned();
        let bytes = fs::read(&path).map_err(|e| io("read compiler input", e))?;
        let input = CompilerInput {
            lookup: path.clone(),
            location: fs::canonicalize(&path).map_err(|e| io("resolve compiler input", e))?,
            sha256: digest(&bytes),
        };
        if let Some(previous) = files.insert(name, input.clone())
            && previous.sha256 != input.sha256
        {
            return Err(err("ambiguous compiler input name"));
        }
        Ok(())
    };
    add(&nvcc)?;
    add(&nvcc.with_file_name("nvcc.profile"))?;
    let mut variables = BTreeMap::new();
    let mut programs = Vec::new();
    for line in text.lines().filter_map(|line| line.strip_prefix("#$ ")) {
        if let Some((name, value)) = line.split_once('=')
            && crate::identifier(name)
        {
            variables.insert(name.to_string(), value.trim().to_string());
            continue;
        }
        let Some(program) = line.split_whitespace().next() else {
            continue;
        };
        let program = resolve(&expand(program.trim_matches('"'), &variables)?, &variables)?;
        if line.split_whitespace().any(|arg| arg == "-E") {
            let mut command = Command::new(&program);
            command.arg("-print-prog-name=cc1plus");
            let cc1plus = PathBuf::from(run(command, "host compiler cc1plus")?.trim());
            if !cc1plus.is_absolute() {
                return Err(err("unresolved host compiler component"));
            }
            add(&cc1plus)?;
            programs.push(cc1plus);
        }
        add(&program)?;
        programs.push(program);
    }
    let names: BTreeSet<_> = variables
        .keys()
        .map(String::as_str)
        .chain(MODELLED_SETTINGS.iter().copied())
        .chain(LOADER_VARIABLES.iter().copied())
        .collect();
    let mut environment = BTreeMap::new();
    for name in names {
        let value = env::var(name).map(Some).or_else(|error| match error {
            env::VarError::NotPresent => Ok(None),
            other => Err(err(other.to_string())),
        })?;
        if value.is_some() && !MODELLED_SETTINGS.contains(&name) {
            return Err(err(format!("unmodelled compiler environment: {name}")));
        }
        environment.insert(name.to_string(), value);
    }
    for name in ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"] {
        if let Some(value) = environment.get(name).and_then(Option::as_ref)
            && (value.contains(['\\', '\'', '"'])
                || names_build_time(value.as_bytes())
                || !value.split_whitespace().all(safe_option))
        {
            return Err(err("unsupported nvcc options or build-time macro"));
        }
    }
    for program in programs.iter().chain(std::iter::once(&nvcc)) {
        for library in shared_libraries(program)? {
            add(&library)?;
        }
    }
    let loader_cache = Path::new("/etc/ld.so.cache");
    if loader_cache.is_file() {
        add(loader_cache)?;
    }
    let libdevice = variables
        .get("NVVMIR_LIBRARY_DIR")
        .ok_or_else(|| err("nvcc named no libdevice directory"))?;
    let mut library_files = fs::read_dir(libdevice)
        .map_err(|e| io("read libdevice", e))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| io("read libdevice", e))?;
    library_files.sort();
    for file in library_files {
        add(&file)?;
    }
    let search_names = files.keys().cloned().collect();
    let mut search_directories = BTreeMap::new();
    for name in SEARCH_SETTINGS {
        if let Some(value) = environment.get(*name).and_then(Option::as_ref) {
            if value.is_empty() {
                continue;
            }
            for path in value.split(':').map(Path::new) {
                if !path.is_absolute() {
                    return Err(err("relative compiler search path"));
                }
                search_directories
                    .insert(path.to_path_buf(), directory_entries(path, &search_names)?);
            }
        }
    }
    for path in [toolchain.home.join("bin"), PathBuf::from(libdevice)] {
        let path =
            fs::canonicalize(path).map_err(|e| io("resolve compiler search directory", e))?;
        search_directories.insert(path.clone(), directory_entries(&path, &search_names)?);
    }
    Ok(CompilerFreshness {
        version: 1,
        toolchain_version: toolchain.version.clone(),
        flags: flags.iter().map(|s| (*s).into()).collect(),
        environment,
        files,
        search_directories,
        sha256: String::new(),
    })
}

fn shared_libraries(program: &Path) -> Result<Vec<PathBuf>, BuildError> {
    let mut command = Command::new("ldd");
    command.arg(program);
    let listing = run(command, &format!("ldd {}", program.display()))?;
    Ok(listing
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let path = line.split_once("=> ").map_or(line, |(_, path)| path);
            path.split_whitespace()
                .next()
                .filter(|path| path.starts_with('/'))
        })
        .map(PathBuf::from)
        .collect())
}

/// Substitute `$NAME`, `$(NAME)` and `${NAME}` from the dry run's settings.
fn expand(text: &str, variables: &BTreeMap<String, String>) -> Result<String, BuildError> {
    let mut text = text.to_string();
    let mut names: Vec<&String> = variables.keys().collect();
    names.sort_by_key(|name| std::cmp::Reverse(name.len()));
    for name in names {
        let value = variables[name].trim_matches('"');
        for form in [
            format!("$({name})"),
            format!("${{{name}}}"),
            format!("${name}"),
        ] {
            text = text.replace(&form, value);
        }
    }
    if text.contains('$') {
        return Err(err(format!("cannot expand {text} from nvcc -dryrun")));
    }
    Ok(text)
}

/// A program path, or a bare name searched in the dry run's PATH.
fn resolve(program: &str, variables: &BTreeMap<String, String>) -> Result<PathBuf, BuildError> {
    if program.contains('/') {
        return Ok(PathBuf::from(program));
    }
    variables
        .get("PATH")
        .into_iter()
        .flat_map(|path| path.split(':'))
        .map(|dir| Path::new(dir).join(program))
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| err(format!("{program} is not on nvcc's PATH")))
}

#[cfg(test)]
mod tests {
    use super::{BuildOptions, compiler_inputs, expand, names_build_time, variables};
    use crate::{NVCC_FLAGS, Toolchain};
    use std::collections::BTreeMap;

    #[test]
    fn source_time_scan_observes_in_process_mutation() {
        let path = std::env::temp_dir().join(format!("kernel-time-scan-{}.h", std::process::id()));
        let unit = crate::GeneratedUnit {
            translation_unit: String::new(),
            header: String::new(),
        };
        std::fs::write(&path, "int unchanged;\n").unwrap();
        assert!(super::sources_reusable(
            std::slice::from_ref(&path),
            &unit,
            &[]
        ));
        std::fs::write(&path, "const char *clock = __TIME__;\n").unwrap();
        assert!(!super::sources_reusable(
            std::slice::from_ref(&path),
            &unit,
            &[]
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn compiler_descriptor_observes_flags_and_profile_in_process() {
        let home = std::env::var_os("CHOPINFER_CUDA_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "/usr/local/cuda".into());
        if std::env::consts::OS != "linux" || !home.join("bin/nvcc").is_file() {
            eprintln!("SKIP: the compiler descriptor requires Linux and nvcc");
            return;
        }
        let real = Toolchain::detect().unwrap();
        let root =
            std::env::temp_dir().join(format!("kernel-compiler-descriptor-{}", std::process::id()));
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::copy(real.tool("nvcc"), bin.join("nvcc")).unwrap();
        std::fs::copy(real.tool("ptxas"), bin.join("ptxas")).unwrap();
        let profile = std::fs::read_to_string(real.tool("nvcc.profile"))
            .unwrap()
            .replace("$(_HERE_)/..", &real.home.display().to_string());
        std::fs::write(bin.join("nvcc.profile"), &profile).unwrap();
        let private = Toolchain {
            home: root.clone(),
            ..real
        };
        let first = super::compiler_freshness(&private, NVCC_FLAGS).unwrap();
        let mut flags = NVCC_FLAGS.to_vec();
        flags.push("--fmad=false");
        assert_ne!(
            first.sha256,
            super::compiler_freshness(&private, &flags).unwrap().sha256
        );
        std::fs::write(
            bin.join("nvcc.profile"),
            format!("{profile}\n# compiler freshness probe\n"),
        )
        .unwrap();
        assert!(
            first
                .verify_current()
                .unwrap_err()
                .to_string()
                .contains("compiler")
        );
        assert_ne!(
            first.sha256,
            super::compiler_freshness(&private, NVCC_FLAGS)
                .unwrap()
                .sha256
        );
        let component = super::compiler_freshness(&private, NVCC_FLAGS).unwrap();
        assert_eq!(component.files["ptxas"].location, bin.join("ptxas"));
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(bin.join("ptxas"))
            .unwrap()
            .write_all(b"\0compiler freshness probe")
            .unwrap();
        assert!(component.verify_current().is_err());
        assert_ne!(
            component.sha256,
            super::compiler_freshness(&private, NVCC_FLAGS)
                .unwrap()
                .sha256
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_compiler_identity_covers_the_cubin_compilers() {
        if std::env::consts::OS != "linux" || !std::path::Path::new("/usr/local/cuda").exists() {
            eprintln!("SKIP: the compiler identity requires Linux and nvcc");
            return;
        }
        let inputs = compiler_inputs(&Toolchain::detect().unwrap(), NVCC_FLAGS).unwrap();
        for name in [
            "nvcc",
            "nvcc.profile",
            "cicc",
            "ptxas",
            "cc1plus",
            "libc.so.6",
        ] {
            assert!(inputs.files.contains_key(name), "{name}: {inputs:?}");
        }
        assert!(
            inputs
                .files
                .keys()
                .any(|name| name.starts_with("libdevice")),
            "{inputs:?}"
        );
    }

    #[test]
    fn only_lock_verified_builds_look_up_cached_cubins() {
        let lookup = |options| variables("compiler", "closure", options)[2].1.clone();
        assert_eq!(lookup(BuildOptions::default()), "1");
        for options in [
            BuildOptions {
                lock_update: true,
                ..BuildOptions::default()
            },
            BuildOptions {
                profile_line_info: true,
                ..BuildOptions::default()
            },
        ] {
            assert_eq!(lookup(options), "0");
        }
        let variables = variables("compiler", "closure", BuildOptions::default());
        assert_eq!(
            variables[..2],
            [
                ("CHOPINFER_NVCC_CACHE_CLOSURE", "closure".into()),
                ("CHOPINFER_NVCC_CACHE_TOOLCHAIN", "compiler".into())
            ]
        );
    }

    #[test]
    fn build_time_macros_are_found() {
        for text in ["x = __DATE__;", "__TIME__", "s(__TIMESTAMP__)"] {
            assert!(names_build_time(text.as_bytes()), "{text}");
        }
        assert!(!names_build_time(b"__DATE_ __TIMES __CUDA_ARCH__"));
    }

    #[test]
    fn dry_run_settings_expand() {
        let variables = BTreeMap::from([
            ("CICC_PATH".to_string(), "/cuda/nvvm/bin".to_string()),
            ("TOP".to_string(), "\"/cuda\"".to_string()),
        ]);
        assert_eq!(
            expand("$CICC_PATH/cicc", &variables).unwrap(),
            "/cuda/nvvm/bin/cicc"
        );
        assert_eq!(expand("$(TOP)/bin", &variables).unwrap(), "/cuda/bin");
        assert_eq!(expand("${TOP}/bin", &variables).unwrap(), "/cuda/bin");
        assert!(expand("$OTHER/bin", &variables).is_err());
    }
}
