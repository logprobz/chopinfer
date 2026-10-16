//! What a caching `CHOPINFER_NVCC_WRAPPER` (tools/nvcc_cache) is told about
//! one cubin compilation. The wrapper keys a cubin on these variables and its
//! arguments, so they hold everything outside the source closure that changes
//! the cubin; a compilation they cannot describe gets no key and the wrapper
//! runs nvcc unchanged.

use crate::{BuildError, BuildOptions, GeneratedUnit, Toolchain, digest, err, io, run};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// The dry-run settings the key models: the programs found on PATH and the
/// shared libraries they load are hashed into the compiler identity, and the
/// wrapper keys the NVCC_* text, caching only options it fully describes.
/// When the environment sets any other setting `nvcc -dryrun` reports
/// (PTXAS_FLAGS, INCLUDES, NVVMIR_LIBRARY_DIR and the rest), or one of
/// LOADER_VARIABLES, the compilation gets no key.
const MODELLED_SETTINGS: &[&str] = &[
    "PATH",
    "LD_LIBRARY_PATH",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "NVCC_CCBIN",
];
/// Loader variables that change the compilers without changing their files.
const LOADER_VARIABLES: &[&str] = &["LD_PRELOAD", "LD_AUDIT"];
/// nvcc options from the environment, scanned for build-time macros.
const NVCC_OPTION_VARIABLES: &[&str] = &["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"];

/// The key variables for one compilation, or None when its sources or
/// defines name the build time or the compiler configuration cannot be
/// identified. A cached cubin is offered only when the lock comparison
/// verifies the result.
pub(crate) fn cache_variables(
    toolchain: &Toolchain,
    flags: &[&str],
    files: &[PathBuf],
    unit: &GeneratedUnit,
    defines: &[(String, String)],
    source_closure_sha256: &str,
    options: BuildOptions,
) -> Option<[(&'static str, String); 3]> {
    let defined_time = defines.iter().any(|(name, value)| {
        names_build_time(name.as_bytes()) || names_build_time(value.as_bytes())
    }) || NVCC_OPTION_VARIABLES.iter().any(|name| {
        env::var_os(name).is_some_and(|value| names_build_time(value.as_encoded_bytes()))
    });
    if defined_time || uses_build_time(files, unit) {
        return None;
    }
    let compiler = compiler_identity(toolchain, flags)?;
    Some(variables(&compiler, source_closure_sha256, options))
}

fn variables(
    compiler: &str,
    source_closure_sha256: &str,
    options: BuildOptions,
) -> [(&'static str, String); 3] {
    let lock_verified = !options.lock_update && !options.profile_line_info;
    [
        ("CHOPINFER_NVCC_CACHE_CLOSURE", source_closure_sha256.into()),
        ("CHOPINFER_NVCC_CACHE_TOOLCHAIN", compiler.into()),
        (
            "CHOPINFER_NVCC_CACHE_LOOKUP",
            if lock_verified { "1" } else { "0" }.into(),
        ),
    ]
}

fn names_build_time(bytes: &[u8]) -> bool {
    [b"__DATE__".as_slice(), b"__TIME__", b"__TIMESTAMP__"]
        .iter()
        .any(|name| bytes.windows(name.len()).any(|window| window == *name))
}

/// Whether the generated unit or a closure file names the compilation date or
/// time, which no key holds. Each file is read once per process.
fn uses_build_time(files: &[PathBuf], unit: &GeneratedUnit) -> bool {
    static SCANNED: Mutex<BTreeMap<PathBuf, bool>> = Mutex::new(BTreeMap::new());
    if names_build_time(unit.translation_unit.as_bytes())
        || names_build_time(unit.header.as_bytes())
    {
        return true;
    }
    let mut scanned = SCANNED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    files.iter().any(|file| {
        *scanned
            .entry(file.clone())
            .or_insert_with(|| fs::read(file).map_or(true, |bytes| names_build_time(&bytes)))
    })
}

/// A digest of the nvcc version and binary, nvcc.profile, and every program
/// and libdevice file `nvcc -dryrun` shows a cubin compilation using, with
/// the host preprocessor's cc1plus. File contents, not paths, enter it.
/// Computed once per toolkit and process.
fn compiler_identity(toolchain: &Toolchain, flags: &[&str]) -> Option<String> {
    static IDENTITIES: Mutex<BTreeMap<PathBuf, Option<String>>> = Mutex::new(BTreeMap::new());
    let mut identities = IDENTITIES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    identities
        .entry(toolchain.home.clone())
        .or_insert_with(|| match compiler_inputs(toolchain, flags) {
            Ok(inputs) => {
                let mut text = format!("{}\n", toolchain.version);
                for (name, sha256) in inputs {
                    text.push_str(&format!("{name} {sha256}\n"));
                }
                Some(digest(text.as_bytes()))
            }
            Err(error) => {
                eprintln!("nvcc cache key unavailable, compiling without the cache: {error}");
                None
            }
        })
        .clone()
}

fn compiler_inputs(
    toolchain: &Toolchain,
    flags: &[&str],
) -> Result<BTreeMap<String, String>, BuildError> {
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
        .map_err(|e| err(format!("nvcc -dryrun output is not UTF-8: {e}")))?;
    let mut inputs = BTreeMap::from([("nvcc".to_string(), toolchain.nvcc_sha256.clone())]);
    let mut add = |path: &Path| -> Result<(), BuildError> {
        let path = fs::canonicalize(path).map_err(|e| io(&path.display().to_string(), e))?;
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let bytes = fs::read(&path).map_err(|e| io(&path.display().to_string(), e))?;
        inputs.insert(name, digest(&bytes));
        Ok(())
    };
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
            // The host compiler driver preprocesses through cc1plus.
            let mut command = Command::new(&program);
            command.arg("-print-prog-name=cc1plus");
            let cc1plus = PathBuf::from(run(command, "host compiler cc1plus")?.trim());
            if cc1plus.is_absolute() {
                add(&cc1plus)?;
                programs.push(cc1plus);
            }
        }
        add(&program)?;
        programs.push(program);
    }
    let unmodelled = variables
        .keys()
        .map(String::as_str)
        .filter(|name| !MODELLED_SETTINGS.contains(name))
        .chain(LOADER_VARIABLES.iter().copied())
        .find(|name| env::var_os(name).is_some());
    if let Some(name) = unmodelled {
        return Err(err(format!("the environment sets {name}")));
    }
    for program in &programs {
        for library in shared_libraries(program)? {
            add(&library)?;
        }
    }
    let libdevice = variables
        .get("NVVMIR_LIBRARY_DIR")
        .ok_or_else(|| err("nvcc -dryrun named no NVVMIR_LIBRARY_DIR"))?;
    let mut files = fs::read_dir(libdevice)
        .map_err(|e| io("read libdevice", e))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| io("read libdevice", e))?;
    files.sort();
    for file in files {
        add(&file)?;
    }
    Ok(inputs)
}

/// The shared libraries the loader resolves for a program under the current
/// LD_LIBRARY_PATH, the loader included.
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
            assert!(inputs.contains_key(name), "{name}: {inputs:?}");
        }
        assert!(
            inputs.keys().any(|name| name.starts_with("libdevice")),
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
