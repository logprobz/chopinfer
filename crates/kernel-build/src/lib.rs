//! Build-script driver for locked, audited SM120 native cubins.

mod cache_key;
pub mod report;
pub mod selection;
pub mod tuning;
pub use report::{BUILD_REPORT_FILE, BuildReport};
pub use selection::{BuildContext, EntryDeclaration, KernelSelection, SelectedCubin, SourceRoot};

use object::{Object, ObjectSymbol};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output};

/// The fixed nvcc flags included in every cubin identity. The ptxas spill
/// warnings go to nvcc's stderr, which `run` shows only on failure; the cubin
/// audit's local and stack byte limit is what refuses a spilling entry.
pub const NVCC_FLAGS: &[&str] = &[
    "-arch=sm_120f",
    "-cubin",
    "-std=c++20",
    "-O3",
    "-Xptxas=-warn-spills,-warn-lmem-usage",
];
const PROFILE_LINE_INFO_FLAG: &str = "-lineinfo";
const MAX_SHARED_BYTES: u32 = 101_376;

/// A failed validation, tool invocation, audit, or lock comparison.
#[derive(Debug)]
pub struct BuildError(pub String);

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for BuildError {}

fn err(message: impl Into<String>) -> BuildError {
    BuildError(message.into())
}
fn io(context: &str, error: std::io::Error) -> BuildError {
    err(format!("{context}: {error}"))
}
/// Canonical lowercase SHA-256 for publisher-generated companion contracts.
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// A pinned CUDA 13.2 compiler and its identity. Off Linux, `skipped` is true.
#[derive(Debug, Clone)]
pub struct Toolchain {
    /// CUDA toolkit root.
    pub home: PathBuf,
    /// Full `nvcc --version` output.
    pub version: String,
    /// SHA-256 of the nvcc binary.
    pub nvcc_sha256: String,
    /// Whether the target platform skips CUDA builds.
    pub skipped: bool,
}

impl Toolchain {
    /// Detect CUDA from `CHOPINFER_CUDA_HOME` or `/usr/local/cuda` and require release 13.2.
    pub fn detect() -> Result<Self, BuildError> {
        let target = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| env::consts::OS.into());
        if target != "linux" {
            return Ok(Self {
                home: PathBuf::new(),
                version: String::new(),
                nvcc_sha256: String::new(),
                skipped: true,
            });
        }
        let home = env::var_os("CHOPINFER_CUDA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
        let home = fs::canonicalize(&home).map_err(|e| io("canonicalize CUDA toolkit", e))?;
        let nvcc = home.join("bin/nvcc");
        let output = Command::new(&nvcc)
            .arg("--version")
            .output()
            .map_err(|e| io(&format!("execute {}", nvcc.display()), e))?;
        if !output.status.success() {
            return Err(err(format!("{} --version failed", nvcc.display())));
        }
        let version = String::from_utf8(output.stdout)
            .map_err(|e| err(format!("nvcc version is not UTF-8: {e}")))?;
        if !version.contains("release 13.2,") {
            return Err(err(format!(
                "nvcc release 13.2 required; {} reported {}",
                nvcc.display(),
                version.trim()
            )));
        }
        let nvcc_sha256 = digest(&fs::read(&nvcc).map_err(|e| io("read nvcc", e))?);
        Ok(Self {
            home,
            version: version.trim_end().into(),
            nvcc_sha256,
            skipped: false,
        })
    }

    fn tool(&self, name: &str) -> PathBuf {
        self.home.join("bin").join(name)
    }
}

fn nvcc_wrapper() -> Option<std::ffi::OsString> {
    env::var_os("CHOPINFER_NVCC_WRAPPER").filter(|value| !value.is_empty())
}

/// Construct a pinned nvcc invocation, optionally through a compilation cache.
///
/// `CHOPINFER_NVCC_WRAPPER` names one executable, like `RUSTC_WRAPPER`.
/// Source-closure discovery, cubin audits and lock verification remain direct.
/// An unset or empty value preserves the production build command.
pub fn nvcc_command(compiler: &Path) -> Command {
    match nvcc_wrapper() {
        Some(wrapper) => {
            let mut command = Command::new(wrapper);
            command.arg(compiler);
            command
        }
        None => Command::new(compiler),
    }
}

/// Environment decisions passed explicitly to [`build`].
#[derive(Debug, Clone, Copy, Default)]
pub struct BuildOptions {
    /// Rewrite `kernels.lock`, or the produced families of a [`LockDir`], when true.
    pub lock_update: bool,
    /// Print Cargo rerun directives when called from a build script.
    pub emit_cargo_directives: bool,
    /// Add line information for local profiling and bypass the release lock.
    pub profile_line_info: bool,
}
impl BuildOptions {
    /// Read `KERNELS_LOCK_UPDATE` and `CHOPINFER_KERNEL_LINEINFO`.
    pub fn from_env() -> Self {
        Self {
            lock_update: env::var("KERNELS_LOCK_UPDATE").as_deref() == Ok("1"),
            emit_cargo_directives: true,
            profile_line_info: env::var("CHOPINFER_KERNEL_LINEINFO").as_deref() == Ok("1"),
        }
    }
}

/// The candidate directory `tools.model_sdk relock` passes to model builds.
pub const MODEL_RELOCK_ENV: &str = "CHOPINFER_MODEL_RELOCK_DIR";
/// The file relock creates in its candidate directory. A directory without it
/// is not a relock, so a stray or configured value fails instead of skipping
/// lock verification.
pub const MODEL_RELOCK_MARKER: &str = ".chopinfer-relock-candidates";

/// A lock the model owns: a lock file, or the directory of a [`LockDir`].
///
/// An ordinary build uses the lock in place with the options
/// [`BuildOptions::from_env`] reads: it verifies the lock, rewrites it under
/// `KERNELS_LOCK_UPDATE=1` or bypasses it under `CHOPINFER_KERNEL_LINEINFO=1`,
/// so the in-tree update and profiling flows keep working. The SDK's `build`
/// and `relock` force those variables empty through Cargo configuration, so
/// an SDK build only verifies. Under `tools.model_sdk
/// relock`, which sets [`MODEL_RELOCK_ENV`] to an absolute directory holding
/// [`MODEL_RELOCK_MARKER`], the build writes its records
/// to a candidate that mirrors the lock's absolute path inside that directory
/// and never writes the model's lock; relock reviews and applies the
/// candidates. Shared crates keep [`BuildOptions::from_env`].
#[derive(Debug, Clone)]
pub struct ModelLock {
    lock: PathBuf,
    candidate: Option<PathBuf>,
    /// The options of a build that is not relocking.
    environment: BuildOptions,
}
impl ModelLock {
    /// Read [`MODEL_RELOCK_ENV`] and [`BuildOptions::from_env`] for the lock
    /// at `lock` and print its Cargo rerun directive, so a build script
    /// reruns for each relock.
    pub fn from_env(lock: impl Into<PathBuf>) -> Result<Self, BuildError> {
        println!("cargo:rerun-if-env-changed={MODEL_RELOCK_ENV}");
        Self::new(
            lock,
            env::var_os(MODEL_RELOCK_ENV).filter(|v| !v.is_empty()),
            BuildOptions::from_env(),
        )
    }

    fn new(
        lock: impl Into<PathBuf>,
        relock: Option<std::ffi::OsString>,
        environment: BuildOptions,
    ) -> Result<Self, BuildError> {
        let lock = lock.into();
        let Some(relock) = relock.map(PathBuf::from) else {
            return Ok(Self {
                lock,
                candidate: None,
                environment,
            });
        };
        if !relock.is_absolute() || !relock.is_dir() {
            return Err(err(format!(
                "{MODEL_RELOCK_ENV} must name an existing absolute directory"
            )));
        }
        if !relock.join(MODEL_RELOCK_MARKER).is_file() {
            return Err(err(format!(
                "{MODEL_RELOCK_ENV} names {} without {MODEL_RELOCK_MARKER}; only tools.model_sdk relock sets it",
                relock.display()
            )));
        }
        // The parent resolves even when relock creates the lock.
        let (Some(parent), Some(name)) = (lock.parent(), lock.file_name()) else {
            return Err(err(format!("invalid model lock path {}", lock.display())));
        };
        let parent = fs::canonicalize(parent)
            .map_err(|e| io(&format!("resolve model lock {}", lock.display()), e))?;
        let relative = parent.join(name);
        let relative = relative.strip_prefix("/").unwrap_or(&relative);
        let candidate = relock.join(relative);
        if let Some(parent) = candidate.parent() {
            fs::create_dir_all(parent).map_err(|e| io("create relock candidate directory", e))?;
        }
        Ok(Self {
            lock,
            candidate: Some(candidate),
            environment,
        })
    }

    /// The path this build compares or writes: the model's lock, or its relock candidate.
    pub fn path(&self) -> &Path {
        self.candidate.as_deref().unwrap_or(&self.lock)
    }

    /// Whether this build writes a relock candidate.
    pub fn relocking(&self) -> bool {
        self.candidate.is_some()
    }

    /// Build-script options: write the relock candidate, or else those of
    /// [`BuildOptions::from_env`].
    pub fn options(&self) -> BuildOptions {
        if self.relocking() {
            BuildOptions {
                lock_update: true,
                emit_cargo_directives: true,
                profile_line_info: false,
            }
        } else {
            self.environment
        }
    }
}

fn flags(options: BuildOptions) -> Vec<&'static str> {
    let mut flags = NVCC_FLAGS.to_vec();
    if options.profile_line_info {
        flags.push(PROFILE_LINE_INFO_FLAG);
    }
    flags
}

/// One generated cubin and its entry wrappers.
#[derive(Debug, Clone)]
pub struct CubinSpec {
    /// Stable cubin basename, without an extension.
    pub name: String,
    /// Root of all project CUDA sources.
    pub sources_root: PathBuf,
    /// Include directories relative to `sources_root`.
    pub include_dirs: Vec<PathBuf>,
    /// C++ `#include` lines placed in the generated translation unit.
    pub headers: Vec<String>,
    /// Preprocessor defines supplied to nvcc.
    pub defines: Vec<(String, String)>,
    /// Exported kernel entries.
    pub entries: Vec<EntrySpec>,
}

/// One versioned C entry wrapper.
#[derive(Debug, Clone)]
pub struct EntrySpec {
    /// Unmangled C identifier ending in `_v<N>`.
    pub symbol: String,
    /// Launch ABI parameters in declaration order.
    pub params: Vec<EntryParam>,
    /// CUDA C++ statements inserted into the entry body.
    pub body: String,
    /// Launch bound threads, a multiple of 32 in 1..=1024.
    pub max_threads: u32,
    /// Minimum blocks per SM, at least one.
    pub min_blocks: u32,
    /// Allowed local memory bytes; zero prohibits spills.
    pub allowed_local_bytes: u32,
}
impl EntrySpec {
    /// Create an entry with the default zero-byte local-memory allowance.
    pub fn new(
        symbol: String,
        params: Vec<EntryParam>,
        body: String,
        max_threads: u32,
        min_blocks: u32,
    ) -> Self {
        Self {
            symbol,
            params,
            body,
            max_threads,
            min_blocks,
            allowed_local_bytes: 0,
        }
    }
}

/// Most slots of a pointer array parameter: one per rank, up to
/// `kernel_core::contract::MAX_COMM_RANKS`, which kernels-sm120 asserts
/// equal (this crate links no kernel contract).
pub const MAX_POINTER_ARRAY_SLOTS: u8 = 8;

/// A bundle ABI argument kind. Opaque bytes have no launch encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum EntryParam {
    /// Typed device pointer.
    DevicePointer { name: String, cxx_type: String },
    /// Unsigned 64-bit scalar.
    U64 { name: String },
    /// Unsigned 32-bit scalar.
    U32 { name: String },
    /// Signed 32-bit scalar.
    I32 { name: String },
    /// By-value CUDA tensor map.
    TensorMap { name: String },
    /// By-value struct containing 1 to [`MAX_POINTER_ARRAY_SLOTS`] typed
    /// pointers.
    PointerArray {
        name: String,
        slots: u8,
        cxx_type: String,
    },
}
impl EntryParam {
    fn name(&self) -> &str {
        match self {
            Self::DevicePointer { name, .. }
            | Self::U64 { name }
            | Self::U32 { name }
            | Self::I32 { name }
            | Self::TensorMap { name }
            | Self::PointerArray { name, .. } => name,
        }
    }
}

/// Host-side resources reported for one exported CUDA entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryReport {
    /// Exported symbol.
    pub symbol: String,
    /// Registers per thread.
    pub registers: u32,
    /// Static shared memory bytes.
    pub shared_static_bytes: u32,
    /// Local memory bytes.
    pub local_bytes: u32,
    /// Stack bytes.
    pub stack_bytes: u32,
    /// Declared thread bound.
    pub max_threads: u32,
    /// Declared minimum block bound.
    pub min_blocks: u32,
}

/// An audited native cubin and its source identity.
#[derive(Debug, Clone)]
pub struct BuiltCubin {
    /// Spec name.
    pub name: String,
    /// Cubin path in `out_dir`.
    pub path: PathBuf,
    /// Sidecar JSON identity path.
    pub identity_path: PathBuf,
    /// True for a profiling cubin that bypassed `kernels.lock`.
    pub line_info: bool,
    /// Cubin SHA-256.
    pub sha256: String,
    /// SHA-256 of the path-keyed dependency closure.
    pub source_closure_sha256: String,
    /// Per-entry resource report.
    pub entries: Vec<EntryReport>,
}

/// Stable generated files for one spec.
#[derive(Debug, Clone)]
pub struct GeneratedUnit {
    /// CUDA C++ translation unit.
    pub translation_unit: String,
    /// Header containing pointer array structs.
    pub header: String,
}

fn identifier(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}
fn versioned(s: &str) -> bool {
    let Some((base, version)) = s.rsplit_once("_v") else {
        return false;
    };
    identifier(s)
        && !base.is_empty()
        && !version.is_empty()
        && version.bytes().all(|b| b.is_ascii_digit())
}
fn relative_dir(path: &Path) -> bool {
    path.components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
}
fn validate(specs: &[CubinSpec]) -> Result<(), BuildError> {
    let mut names = BTreeSet::new();
    let mut symbols = BTreeSet::new();
    for spec in specs {
        if !identifier(&spec.name) || !names.insert(&spec.name) {
            return Err(err(format!(
                "invalid or duplicate cubin name {}",
                spec.name
            )));
        }
        if spec.entries.is_empty() {
            return Err(err(format!("{}: no entries", spec.name)));
        }
        for dir in &spec.include_dirs {
            if !relative_dir(dir) {
                return Err(err(format!(
                    "{}: include directory must be relative: {}",
                    spec.name,
                    dir.display()
                )));
            }
        }
        for (name, _) in &spec.defines {
            if !identifier(name) {
                return Err(err(format!("{}: invalid define {name}", spec.name)));
            }
        }
        for entry in &spec.entries {
            let label = format!("{}:{}", spec.name, entry.symbol);
            if !versioned(&entry.symbol) || !symbols.insert(&entry.symbol) {
                return Err(err(format!(
                    "{label}: invalid or duplicate versioned symbol"
                )));
            }
            if entry.body.trim().is_empty() {
                return Err(err(format!("{label}: empty body")));
            }
            if !(1..=1024).contains(&entry.max_threads)
                || entry.max_threads % 32 != 0
                || entry.min_blocks == 0
            {
                return Err(err(format!("{label}: invalid launch bounds")));
            }
            let mut params = BTreeSet::new();
            for param in &entry.params {
                if !identifier(param.name()) || !params.insert(param.name()) {
                    return Err(err(format!(
                        "{label}: invalid or duplicate parameter {}",
                        param.name()
                    )));
                }
                if let EntryParam::PointerArray { slots, .. } = param
                    && !(1..=MAX_POINTER_ARRAY_SLOTS).contains(slots)
                {
                    return Err(err(format!(
                        "{label}: pointer array slots must be 1..={MAX_POINTER_ARRAY_SLOTS}"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Generate deterministic C++ wrapper and pointer-array header text.
pub fn generate(spec: &CubinSpec) -> Result<GeneratedUnit, BuildError> {
    validate(std::slice::from_ref(spec))?;
    let mut header = String::from("#pragma once\n");
    let mut unit = format!(
        "#include <stdint.h>\n#include <cuda.h>\n#include \"{}_params.h\"\n",
        spec.name
    );
    for line in &spec.headers {
        if !line.trim_start().starts_with("#include ") || line.contains('\n') {
            return Err(err(format!(
                "{}: header must be one #include line",
                spec.name
            )));
        }
        unit.push_str(line);
        unit.push('\n');
    }
    for entry in &spec.entries {
        let mut declarations = Vec::new();
        for param in &entry.params {
            let declaration = match param {
                EntryParam::DevicePointer { name, cxx_type } => format!("{cxx_type}* {name}"),
                EntryParam::U64 { name } => format!("uint64_t {name}"),
                EntryParam::U32 { name } => format!("uint32_t {name}"),
                EntryParam::I32 { name } => format!("int32_t {name}"),
                EntryParam::TensorMap { name } => {
                    format!("const __grid_constant__ CUtensorMap {name}")
                }
                EntryParam::PointerArray {
                    name,
                    slots,
                    cxx_type,
                } => {
                    let struct_name = format!("{}_{}_params", entry.symbol, name);
                    header.push_str(&format!("struct {struct_name} {{\n"));
                    for index in 0..*slots {
                        header.push_str(&format!("    {cxx_type}* slot_{index};\n"));
                    }
                    header.push_str("};\n");
                    format!("{struct_name} {name}")
                }
            };
            declarations.push(declaration);
        }
        unit.push_str(&format!(
            "extern \"C\" __global__ void __launch_bounds__({}, {}) {}({}) {{\n{}\n}}\n",
            entry.max_threads,
            entry.min_blocks,
            entry.symbol,
            declarations.join(", "),
            entry.body
        ));
    }
    Ok(GeneratedUnit {
        translation_unit: unit,
        header,
    })
}

fn run(mut command: Command, label: &str) -> Result<String, BuildError> {
    let Output {
        status,
        stdout,
        stderr,
    } = command.output().map_err(|e| io(label, e))?;
    if !status.success() {
        return Err(err(format!(
            "{label} failed: {}",
            String::from_utf8_lossy(&stderr).trim()
        )));
    }
    String::from_utf8(stdout).map_err(|e| err(format!("{label} output is not UTF-8: {e}")))
}
fn dump(toolchain: &Toolchain, path: &Path, flag: &str) -> Result<String, BuildError> {
    let mut command = Command::new(toolchain.tool("cuobjdump"));
    command.arg(flag).arg(path);
    run(command, &format!("cuobjdump {flag} {}", path.display()))
}
fn field(line: &str, key: &str) -> Result<u32, BuildError> {
    let prefix = format!("{key}:");
    let value = line
        .split_whitespace()
        .find_map(|part| part.strip_prefix(&prefix))
        .ok_or_else(|| err(format!("resource report lacks {key}: {line}")))?;
    value
        .parse()
        .map_err(|e| err(format!("invalid {key} resource {value}: {e}")))
}

/// Audit an existing cubin against one spec without executing any GPU code.
pub fn audit_cubin(
    toolchain: &Toolchain,
    spec: &CubinSpec,
    path: &Path,
) -> Result<Vec<EntryReport>, BuildError> {
    audit_cubin_with_line_info(toolchain, spec, path, false)
}

fn audit_cubin_with_line_info(
    toolchain: &Toolchain,
    spec: &CubinSpec,
    path: &Path,
    profile_line_info: bool,
) -> Result<Vec<EntryReport>, BuildError> {
    validate(std::slice::from_ref(spec))?;
    let label = &spec.name;
    let elf = dump(toolchain, path, "-elf")?;
    // CUDA 13.2 reports sm_120f as sm=120, while sm_120a is sm=120a.
    if !elf.lines().any(|line| {
        line.starts_with("64-bit ELF:")
            && line
                .split(|c: char| c == ',' || c.is_whitespace())
                .any(|part| part == "sm=120")
    }) {
        return Err(err(format!(
            "{label}: architecture audit: expected sm_120f (cuobjdump sm=120)"
        )));
    }
    if !profile_line_info
        && elf
            .lines()
            .any(|line| line.contains(".nv_debug_ptx_txt") || line.contains(".ptx"))
    {
        return Err(err(format!("{label}: PTX audit: embedded PTX section")));
    }
    if !dump(toolchain, path, "--dump-ptx")?.trim().is_empty() {
        return Err(err(format!("{label}: PTX audit: embedded PTX image")));
    }
    let symbols = dump(toolchain, path, "--dump-elf-symbols")?;
    let actual: BTreeSet<String> = symbols
        .lines()
        .filter(|line| {
            line.contains("STT_FUNC") && line.contains("STB_GLOBAL") && line.contains("STO_ENTRY")
        })
        .filter_map(|line| line.split_whitespace().last().map(str::to_owned))
        .collect();
    let expected: BTreeSet<String> = spec
        .entries
        .iter()
        .map(|entry| entry.symbol.clone())
        .collect();
    if actual != expected {
        return Err(err(format!(
            "{label}: entry symbol audit: expected {expected:?}, found {actual:?}"
        )));
    }
    let bytes = fs::read(path).map_err(|e| io("read cubin", e))?;
    let object = object::File::parse(bytes.as_slice())
        .map_err(|e| err(format!("{label}: ELF parse: {e}")))?;
    for symbol in object.symbols() {
        if symbol.is_undefined() && !symbol.is_weak() && symbol.kind() == object::SymbolKind::Text {
            return Err(err(format!(
                "{label}: unresolved device function {}",
                symbol.name().unwrap_or("<unknown>")
            )));
        }
    }
    let resources = dump(toolchain, path, "-res-usage")?;
    let mut by_symbol = BTreeMap::new();
    let mut lines = resources.lines().peekable();
    while let Some(line) = lines.next() {
        if let Some(symbol) = line
            .trim()
            .strip_prefix("Function ")
            .and_then(|part| part.strip_suffix(':'))
        {
            let usage = lines
                .next()
                .ok_or_else(|| err(format!("{label}:{symbol}: missing resource line")))?;
            by_symbol.insert(
                symbol.to_owned(),
                (
                    field(usage, "REG")?,
                    field(usage, "STACK")?,
                    field(usage, "SHARED")?,
                    field(usage, "LOCAL")?,
                ),
            );
        }
    }
    spec.entries
        .iter()
        .map(|entry| {
            let symbol = &entry.symbol;
            let (registers, stack_bytes, shared_static_bytes, local_bytes) = by_symbol
                .get(symbol)
                .copied()
                .ok_or_else(|| err(format!("{label}:{symbol}: missing resource usage")))?;
            if local_bytes > entry.allowed_local_bytes {
                return Err(err(format!(
                    "{label}:{symbol}: LOCAL {local_bytes} exceeds allowed {}",
                    entry.allowed_local_bytes
                )));
            }
            if stack_bytes > entry.allowed_local_bytes {
                return Err(err(format!(
                    "{label}:{symbol}: STACK {stack_bytes} exceeds allowed local bytes {}",
                    entry.allowed_local_bytes
                )));
            }
            if shared_static_bytes > MAX_SHARED_BYTES {
                return Err(err(format!(
                    "{label}:{symbol}: SHARED {shared_static_bytes} exceeds {MAX_SHARED_BYTES}"
                )));
            }
            Ok(EntryReport {
                symbol: symbol.clone(),
                registers,
                shared_static_bytes,
                local_bytes,
                stack_bytes,
                max_threads: entry.max_threads,
                min_blocks: entry.min_blocks,
            })
        })
        .collect()
}

fn closure(
    toolchain: &Toolchain,
    spec: &CubinSpec,
    stage: &Path,
    unit: &GeneratedUnit,
    flags: &[&str],
) -> Result<(String, Vec<PathBuf>, Vec<report::ClosureInput>), BuildError> {
    let root =
        fs::canonicalize(&spec.sources_root).map_err(|e| io("canonicalize sources_root", e))?;
    let generated = stage.join(format!("{}.cu", spec.name));
    let mut command = Command::new(toolchain.tool("nvcc"));
    command
        .current_dir(stage)
        .arg("-M")
        .arg(format!("-I{}", root.display()));
    for flag in flags.iter().copied().filter(|flag| *flag != "-cubin") {
        command.arg(flag);
    }
    for dir in &spec.include_dirs {
        command.arg(format!("-I{}", root.join(dir).display()));
    }
    for (key, value) in &spec.defines {
        command.arg(format!("-D{key}={value}"));
    }
    command.arg(
        generated
            .file_name()
            .ok_or_else(|| err("invalid generated filename"))?,
    );
    let deps = run(command, &format!("{}: nvcc -M", spec.name))?;
    let deps = deps.replace("\\\n", " ");
    let (_, paths) = deps
        .split_once(':')
        .ok_or_else(|| err(format!("{}: invalid nvcc -M output", spec.name)))?;
    let mut inputs = BTreeMap::new();
    inputs.insert(
        format!("generated/{}.cu", spec.name),
        unit.translation_unit.as_bytes().to_vec(),
    );
    inputs.insert(
        format!("generated/{}_params.h", spec.name),
        unit.header.as_bytes().to_vec(),
    );
    let mut locations = BTreeMap::from([
        (format!("generated/{}.cu", spec.name), generated.clone()),
        (
            format!("generated/{}_params.h", spec.name),
            stage.join(format!("{}_params.h", spec.name)),
        ),
    ]);
    let mut files = Vec::new();
    for token in paths.split_whitespace() {
        let path = PathBuf::from(token);
        let path = if path.is_absolute() {
            path
        } else {
            stage.join(path)
        };
        let path = fs::canonicalize(&path)
            .map_err(|e| io(&format!("{}: dependency {}", spec.name, path.display()), e))?;
        if path.starts_with(stage) {
            continue;
        }
        let key = if let Ok(relative) = path.strip_prefix(&root) {
            format!("source/{}", relative.display())
        } else if let Ok(relative) = path.strip_prefix(&toolchain.home) {
            format!("toolkit/{}", relative.display())
        } else {
            format!(
                "system/{}",
                path.strip_prefix("/").unwrap_or(&path).display()
            )
        };
        files.push(path.clone());
        locations.insert(key.clone(), path.clone());
        inputs.insert(key, fs::read(&path).map_err(|e| io("read dependency", e))?);
    }
    let mut hash = Sha256::new();
    let mut membership = Vec::new();
    for (key, contents) in inputs {
        hash.update((key.len() as u64).to_le_bytes());
        hash.update(key.as_bytes());
        hash.update((contents.len() as u64).to_le_bytes());
        hash.update(&contents);
        membership.push(report::ClosureInput {
            path: locations.remove(&key).expect("closure input location"),
            sha256: digest(&contents),
            bytes: contents.len() as u64,
            key,
        });
    }
    Ok((format!("{:x}", hash.finalize()), files, membership))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct IdentityEntry {
    symbol: String,
    params: Vec<EntryParam>,
    max_threads: u32,
    min_blocks: u32,
    allowed_local_bytes: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CubinIdentity {
    name: String,
    line_info: bool,
    toolchain_version: String,
    nvcc_sha256: String,
    flags: Vec<String>,
    defines: Vec<(String, String)>,
    entries: Vec<IdentityEntry>,
    source_closure_sha256: String,
    cubin_sha256: String,
    resources: Vec<EntryReport>,
}
fn compare_lock(
    lock_path: &Path,
    records: &[CubinIdentity],
    update: bool,
) -> Result<(), BuildError> {
    let mut json = serde_json::to_string_pretty(records)
        .map_err(|e| err(format!("serialize kernels.lock: {e}")))?;
    json.push('\n');
    if update {
        fs::write(lock_path, json).map_err(|e| io("write kernels.lock", e))?;
        return Ok(());
    }
    let existing = fs::read_to_string(lock_path).map_err(|e| {
        err(format!(
            "{}: {e}; rerun with KERNELS_LOCK_UPDATE=1 to accept",
            lock_path.display()
        ))
    })?;
    let locked: Vec<CubinIdentity> = serde_json::from_str(&existing).map_err(|e| {
        err(format!(
            "{}: invalid lock: {e}; rerun with KERNELS_LOCK_UPDATE=1 to accept",
            lock_path.display()
        ))
    })?;
    if locked == records {
        return Ok(());
    }
    let expected = serde_json::to_value(records).map_err(|e| err(e.to_string()))?;
    let actual = serde_json::to_value(&locked).map_err(|e| err(e.to_string()))?;
    let fields = if let (Some(a), Some(b)) = (actual.as_array(), expected.as_array()) {
        let mut differences = Vec::new();
        if a.len() != b.len() {
            differences.push("record count".to_string());
        }
        for index in 0..a.len().min(b.len()) {
            if a[index]["name"] != b[index]["name"] {
                differences.push(format!("record {index} name"));
            }
            if let (Some(x), Some(y)) = (a[index].as_object(), b[index].as_object()) {
                for key in x.keys().chain(y.keys()) {
                    if x.get(key) != y.get(key)
                        && !differences.contains(&format!("record {index} {key}"))
                    {
                        differences.push(format!("record {index} {key}"));
                    }
                }
            }
        }
        differences.join(", ")
    } else {
        "records".into()
    };
    Err(err(format!(
        "{}: kernels.lock mismatch in {fields}; rerun with KERNELS_LOCK_UPDATE=1 to accept",
        lock_path.display()
    )))
}

/// A directory-form kernel lock, the alternative to one `kernels.lock` file.
///
/// A *source family* is a named group of cubins declared on the build call
/// with [`LockDir::family`]. Each family owns two files in the directory:
/// `<family>.lock` holds the gating records and `<family>.provenance.json`
/// the source-closure digests. A record gates on name, toolchain version,
/// nvcc hash, flags, defines, entry ABI, `cubin_sha256` and the resource
/// report audited from that cubin; the source closure is provenance that is
/// reported but never fails a build. Each family is compared on its own, and
/// a build reads or (under `KERNELS_LOCK_UPDATE=1`) rewrites only the families
/// of the cubins it built. Other files in the directory are never opened, so
/// branches that add different families do not conflict. One build call must
/// name every cubin it builds and produce every cubin of each family it names.
#[derive(Debug, Clone)]
pub struct LockDir {
    dir: PathBuf,
    families: Vec<(String, Vec<String>)>,
}

/// File suffix of a family's gating records in a [`LockDir`].
pub const FAMILY_LOCK_SUFFIX: &str = ".lock";
/// File suffix of a family's report-only provenance in a [`LockDir`].
pub const FAMILY_PROVENANCE_SUFFIX: &str = ".provenance.json";

impl LockDir {
    /// A lock directory with no families yet.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            families: Vec::new(),
        }
    }

    /// Assign cubins, by spec name, to the family `name`. A family name is
    /// ASCII letters, digits, `_` and `-`, starting with a letter or digit.
    pub fn family<S: AsRef<str>>(
        mut self,
        name: impl Into<String>,
        cubins: impl IntoIterator<Item = S>,
    ) -> Self {
        self.families.push((
            name.into(),
            cubins.into_iter().map(|c| c.as_ref().to_owned()).collect(),
        ));
        self
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The gating lock file of one family.
    pub fn lock_file(&self, family: &str) -> PathBuf {
        self.dir.join(format!("{family}{FAMILY_LOCK_SUFFIX}"))
    }

    /// The report-only provenance file of one family.
    pub fn provenance_file(&self, family: &str) -> PathBuf {
        self.dir.join(format!("{family}{FAMILY_PROVENANCE_SUFFIX}"))
    }

    /// Family name by cubin name, after checking that the families cover
    /// `specs` exactly once.
    fn assign(&self, specs: &[CubinSpec]) -> Result<BTreeMap<String, String>, BuildError> {
        let mut families = BTreeSet::new();
        let mut by_cubin = BTreeMap::new();
        for (family, cubins) in &self.families {
            let mut chars = family.chars();
            if !matches!(chars.next(), Some(c) if c.is_ascii_alphanumeric())
                || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                return Err(err(format!("invalid lock family name {family:?}")));
            }
            if !families.insert(family) || cubins.is_empty() {
                return Err(err(format!(
                    "lock family {family} is declared twice or has no cubins"
                )));
            }
            for cubin in cubins {
                if by_cubin.insert(cubin.clone(), family.clone()).is_some() {
                    return Err(err(format!(
                        "cubin {cubin} is assigned to more than one lock family"
                    )));
                }
            }
        }
        let built: BTreeSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        if let Some(spec) = specs.iter().find(|s| !by_cubin.contains_key(&s.name)) {
            return Err(err(format!("cubin {} has no lock family", spec.name)));
        }
        if let Some((cubin, family)) = by_cubin.iter().find(|(c, _)| !built.contains(c.as_str())) {
            return Err(err(format!(
                "lock family {family} names cubin {cubin}, which this build does not produce"
            )));
        }
        Ok(by_cubin)
    }
}

/// A [`LockDir`] record: the legacy identity without its source closure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GatedIdentity {
    name: String,
    line_info: bool,
    toolchain_version: String,
    nvcc_sha256: String,
    flags: Vec<String>,
    defines: Vec<(String, String)>,
    entries: Vec<IdentityEntry>,
    cubin_sha256: String,
    resources: Vec<EntryReport>,
}
impl From<&CubinIdentity> for GatedIdentity {
    fn from(identity: &CubinIdentity) -> Self {
        Self {
            name: identity.name.clone(),
            line_info: identity.line_info,
            toolchain_version: identity.toolchain_version.clone(),
            nvcc_sha256: identity.nvcc_sha256.clone(),
            flags: identity.flags.clone(),
            defines: identity.defines.clone(),
            entries: identity.entries.clone(),
            cubin_sha256: identity.cubin_sha256.clone(),
            resources: identity.resources.clone(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    name: String,
    source_closure_sha256: String,
}

fn lock_json<T: Serialize>(records: &[T], path: &Path) -> Result<String, BuildError> {
    let mut json = serde_json::to_string_pretty(records)
        .map_err(|e| err(format!("serialize {}: {e}", path.display())))?;
    json.push('\n');
    Ok(json)
}

/// Write only changed bytes, so an update does not touch unchanged files.
fn write_if_changed(path: &Path, json: &str) -> Result<(), BuildError> {
    if fs::read(path).ok().as_deref() == Some(json.as_bytes()) {
        return Ok(());
    }
    fs::write(path, json).map_err(|e| io(&format!("write {}", path.display()), e))
}

/// Field-level differences between locked and built gating records, by
/// cubin name, ABI or resource symbol and field.
fn gated_differences(locked: &[GatedIdentity], built: &[GatedIdentity]) -> Vec<String> {
    let value = |r: &GatedIdentity| serde_json::to_value(r).expect("serialize gated record");
    let locked: BTreeMap<&str, serde_json::Value> =
        locked.iter().map(|r| (r.name.as_str(), value(r))).collect();
    let built: BTreeMap<&str, serde_json::Value> =
        built.iter().map(|r| (r.name.as_str(), value(r))).collect();
    let mut differences = Vec::new();
    for (name, record) in &built {
        let Some(old) = locked.get(name) else {
            differences.push(format!("{name} (not locked)"));
            continue;
        };
        let (Some(new), Some(old)) = (record.as_object(), old.as_object()) else {
            continue;
        };
        for (key, new) in new {
            let old = &old[key];
            if new == old {
                continue;
            }
            let by_symbol = |v: &serde_json::Value| -> Option<BTreeMap<String, serde_json::Value>> {
                v.as_array()?
                    .iter()
                    .map(|e| Some((e["symbol"].as_str()?.to_owned(), e.clone())))
                    .collect()
            };
            match (by_symbol(new), by_symbol(old)) {
                (Some(new), Some(old)) if new.keys().eq(old.keys()) => {
                    for (symbol, entry) in &new {
                        for (field, v) in entry.as_object().into_iter().flatten() {
                            if old[symbol][field] != *v {
                                differences.push(format!("{name}.{key}[{symbol}].{field}"));
                            }
                        }
                    }
                }
                _ => differences.push(format!("{name}.{key}")),
            }
        }
    }
    for name in locked.keys().filter(|name| !built.contains_key(*name)) {
        differences.push(format!("{name} (locked but not built)"));
    }
    if differences.is_empty() {
        differences.push("record order".into());
    }
    differences
}

/// Compare or rewrite each produced family on its own. Gating differences
/// fail; provenance differences are returned and reported.
fn compare_families(
    lock: &LockDir,
    assignment: &BTreeMap<String, String>,
    records: &[CubinIdentity],
    options: BuildOptions,
) -> Result<Vec<report::FamilyLock>, BuildError> {
    let mut families: BTreeMap<&str, Vec<&CubinIdentity>> = BTreeMap::new();
    for record in records {
        families
            .entry(assignment[&record.name].as_str())
            .or_default()
            .push(record);
    }
    if options.lock_update {
        fs::create_dir_all(&lock.dir).map_err(|e| io("create lock directory", e))?;
    }
    // Reports name absolute locations, so they verify from any directory.
    let dir = fs::canonicalize(&lock.dir)
        .or_else(|_| std::path::absolute(&lock.dir))
        .map_err(|e| io("resolve lock directory", e))?;
    let located = |path: PathBuf| {
        let path = dir.join(path.file_name().expect("family file name"));
        fs::canonicalize(&path).unwrap_or(path)
    };
    let mut failures = Vec::new();
    let mut outcomes = Vec::new();
    for (family, records) in families {
        let lock_path = located(lock.lock_file(family));
        let provenance_path = located(lock.provenance_file(family));
        let gated: Vec<GatedIdentity> = records.iter().map(|r| GatedIdentity::from(*r)).collect();
        let provenance: Vec<Provenance> = records
            .iter()
            .map(|r| Provenance {
                name: r.name.clone(),
                source_closure_sha256: r.source_closure_sha256.clone(),
            })
            .collect();
        let mut provenance_changed = Vec::new();
        if options.lock_update {
            write_if_changed(&lock_path, &lock_json(&gated, &lock_path)?)?;
            write_if_changed(&provenance_path, &lock_json(&provenance, &provenance_path)?)?;
        } else {
            let locked = fs::read_to_string(&lock_path)
                .map_err(|e| e.to_string())
                .and_then(|text| {
                    serde_json::from_str::<Vec<GatedIdentity>>(&text)
                        .map_err(|e| format!("invalid lock: {e}"))
                });
            let failure = match locked {
                Err(e) => format!("{}: family {family}: {e}", lock_path.display()),
                Ok(locked) if locked != gated => format!(
                    "{}: kernel lock mismatch in family {family}: {}",
                    lock_path.display(),
                    gated_differences(&locked, &gated).join(", ")
                ),
                Ok(_) => String::new(),
            };
            // Provenance is reported only for a family whose records gate clean.
            if !failure.is_empty() {
                failures.push(failure);
                continue;
            }
            let recorded: BTreeMap<String, String> = fs::read_to_string(&provenance_path)
                .ok()
                .and_then(|text| serde_json::from_str::<Vec<Provenance>>(&text).ok())
                .into_iter()
                .flatten()
                .map(|p| (p.name, p.source_closure_sha256))
                .collect();
            provenance_changed = provenance
                .iter()
                .filter(|p| recorded.get(&p.name) != Some(&p.source_closure_sha256))
                .map(|p| p.name.clone())
                .collect();
            if !provenance_changed.is_empty() {
                let message = format!(
                    "kernel lock family {family}: source provenance differs for {} with identical locked cubins; KERNELS_LOCK_UPDATE=1 refreshes {}",
                    provenance_changed.join(", "),
                    provenance_path.display()
                );
                if options.emit_cargo_directives {
                    println!("cargo:warning={message}");
                } else {
                    eprintln!("{message}");
                }
            }
        }
        outcomes.push(report::FamilyLock {
            family: family.into(),
            lock_sha256: fs::read(&lock_path).ok().map(|bytes| digest(&bytes)),
            lock_path,
            provenance_path,
            cubins: records.iter().map(|r| r.name.clone()).collect(),
            provenance_changed,
        });
    }
    if failures.is_empty() {
        Ok(outcomes)
    } else {
        Err(err(format!(
            "{}; rerun with KERNELS_LOCK_UPDATE=1 to accept",
            failures.join("; ")
        )))
    }
}

/// Where a build's lock records live.
#[derive(Clone, Copy)]
enum LockTarget<'a> {
    File(&'a Path),
    Families(&'a LockDir),
}
impl LockTarget<'_> {
    fn path(&self) -> &Path {
        match self {
            Self::File(path) => path,
            Self::Families(lock) => &lock.dir,
        }
    }
}

/// Generate, compile, audit, digest, and verify all specs against one lock file.
///
/// Toolkit headers are keyed relative to the CUDA toolkit root, and system
/// headers are keyed by absolute path. Both are hashed with project headers.
/// Generated files are keyed under `generated/`.
pub fn build(
    toolchain: &Toolchain,
    specs: &[CubinSpec],
    out_dir: &Path,
    lock_path: &Path,
    options: BuildOptions,
) -> Result<Vec<BuiltCubin>, BuildError> {
    build_declared(
        toolchain,
        specs,
        out_dir,
        LockTarget::File(lock_path),
        options,
        None,
        None,
    )
}

/// Retain legacy catalogs while identifying their consumer and source roots.
/// Entries without declarations are explicitly reported as incomplete.
pub fn build_with_context(
    toolchain: &Toolchain,
    specs: &[CubinSpec],
    out_dir: &Path,
    lock_path: &Path,
    options: BuildOptions,
    context: &BuildContext,
) -> Result<Vec<BuiltCubin>, BuildError> {
    build_declared(
        toolchain,
        specs,
        out_dir,
        LockTarget::File(lock_path),
        options,
        Some(context),
        None,
    )
}

/// Build like [`build_with_context`], comparing each source family of `lock`
/// with its own files. See [`LockDir`] for the identity and family rules.
pub fn build_in_families(
    toolchain: &Toolchain,
    specs: &[CubinSpec],
    out_dir: &Path,
    lock: &LockDir,
    options: BuildOptions,
    context: Option<&BuildContext>,
) -> Result<Vec<BuiltCubin>, BuildError> {
    build_declared(
        toolchain,
        specs,
        out_dir,
        LockTarget::Families(lock),
        options,
        context,
        None,
    )
}

fn build_declared(
    toolchain: &Toolchain,
    specs: &[CubinSpec],
    out_dir: &Path,
    lock: LockTarget<'_>,
    options: BuildOptions,
    context: Option<&BuildContext>,
    declarations: Option<&BTreeMap<String, Vec<EntryDeclaration>>>,
) -> Result<Vec<BuiltCubin>, BuildError> {
    fs::create_dir_all(out_dir).map_err(|e| io("create cubin out_dir", e))?;
    let report_path = report::path(out_dir, options);
    if let Some(context) = context
        && let Ok(previous) = BuildReport::read(&report_path)
        && let Some(owner) = previous.context
        && owner.consumer != context.consumer
    {
        return Err(err("cubin output directory belongs to another consumer"));
    }
    // A failed build must never leave an old report claiming current coverage.
    match fs::remove_file(&report_path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(io("invalidate kernel build report", e)),
    }
    if let Some(context) = context {
        context.validate()?;
    }
    if options.emit_cargo_directives {
        println!("cargo:rerun-if-env-changed=KERNELS_LOCK_UPDATE");
        println!("cargo:rerun-if-env-changed=CHOPINFER_CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CHOPINFER_KERNEL_LINEINFO");
        println!("cargo:rerun-if-env-changed=CHOPINFER_NVCC_WRAPPER");
        match lock {
            LockTarget::File(path) => println!("cargo:rerun-if-changed={}", path.display()),
            LockTarget::Families(lock) => {
                for (family, _) in &lock.families {
                    println!(
                        "cargo:rerun-if-changed={}",
                        lock.lock_file(family).display()
                    );
                    println!(
                        "cargo:rerun-if-changed={}",
                        lock.provenance_file(family).display()
                    );
                }
            }
        }
    }
    if toolchain.skipped {
        report::skipped(out_dir, lock.path(), options, context)?;
        return Ok(Vec::new());
    }
    validate(specs)?;
    let assignment = match lock {
        LockTarget::Families(lock) => Some(lock.assign(specs)?),
        LockTarget::File(_) => None,
    };
    let out_dir = fs::canonicalize(out_dir).map_err(|e| io("canonicalize cubin out_dir", e))?;
    let mut sorted: Vec<&CubinSpec> = specs.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut built = Vec::new();
    let mut records = Vec::new();
    let mut memberships = Vec::new();
    let flags = flags(options);
    let workers = build_workers(
        std::thread::available_parallelism().map_or(8, |count| count.get()),
        std::env::var("CHOPINFER_KERNEL_BUILD_JOBS").ok().as_deref(),
    );
    for batch in sorted.chunks(workers) {
        let out_dir = out_dir.as_path();
        let flags = flags.as_slice();
        let results = std::thread::scope(|scope| {
            let workers = batch
                .iter()
                .map(|&spec| {
                    scope.spawn(move || build_one(toolchain, spec, out_dir, options, flags))
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .map_err(|_| err("cubin build worker panicked"))?
                })
                .collect::<Result<Vec<_>, BuildError>>()
        })?;
        for (cubin, identity, membership) in results {
            built.push(cubin);
            records.push(identity);
            memberships.push(membership);
        }
    }
    let evidence = match (lock, assignment) {
        _ if options.profile_line_info => report::LockEvidence::Bypassed(lock.path()),
        (LockTarget::File(path), _) => {
            compare_lock(path, &records, options.lock_update)?;
            report::LockEvidence::File(path)
        }
        (LockTarget::Families(dir), Some(assignment)) => report::LockEvidence::Families(
            &dir.dir,
            compare_families(dir, &assignment, &records, options)?,
        ),
        (LockTarget::Families(_), None) => unreachable!("families are assigned above"),
    };
    report::write(
        toolchain,
        specs,
        &built,
        &records,
        &memberships,
        &out_dir,
        evidence,
        options,
        context,
        declarations,
    )?;
    Ok(built)
}

/// Concurrent nvcc builds: at most 16 and the available cores, lowered by
/// CHOPINFER_KERNEL_BUILD_JOBS when CI shares the cores between builds. The
/// count only bounds concurrency; every cubin and lock record is the same.
fn build_workers(available: usize, requested: Option<&str>) -> usize {
    let limit = available.clamp(1, 16);
    requested
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&count| count > 0)
        .map_or(limit, |count| count.min(limit))
}

fn build_one(
    toolchain: &Toolchain,
    spec: &CubinSpec,
    out_dir: &Path,
    options: BuildOptions,
    flags: &[&str],
) -> Result<(BuiltCubin, CubinIdentity, Vec<report::ClosureInput>), BuildError> {
    let generated = generate(spec)?;
    let stage = out_dir.join(format!("{}.stage", spec.name));
    if stage.exists() {
        fs::remove_dir_all(&stage).map_err(|e| io("clear cubin stage", e))?;
    }
    fs::create_dir_all(&stage).map_err(|e| io("create cubin stage", e))?;
    fs::write(
        stage.join(format!("{}.cu", spec.name)),
        &generated.translation_unit,
    )
    .map_err(|e| io("write generated TU", e))?;
    fs::write(
        stage.join(format!("{}_params.h", spec.name)),
        &generated.header,
    )
    .map_err(|e| io("write generated header", e))?;
    let (source_closure_sha256, files, membership) =
        closure(toolchain, spec, &stage, &generated, flags)?;
    if options.emit_cargo_directives {
        for path in &files {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    let suffix = if options.profile_line_info {
        ".profile"
    } else {
        ""
    };
    let path = out_dir.join(format!("{}{suffix}.cubin", spec.name));
    let identity_path = out_dir.join(format!("{}{suffix}.identity.json", spec.name));
    let expected_entries = spec
        .entries
        .iter()
        .map(|e| IdentityEntry {
            symbol: e.symbol.clone(),
            params: e.params.clone(),
            max_threads: e.max_threads,
            min_blocks: e.min_blocks,
            allowed_local_bytes: e.allowed_local_bytes,
        })
        .collect::<Vec<_>>();
    if let (Ok(bytes), Ok(cubin_bytes)) = (fs::read(&identity_path), fs::read(&path))
        && let Ok(cached) = serde_json::from_slice::<CubinIdentity>(&bytes)
        && cached.name == spec.name
        && cached.line_info == options.profile_line_info
        && cached.toolchain_version == toolchain.version
        && cached.nvcc_sha256 == toolchain.nvcc_sha256
        && cached.flags == flags
        && cached.defines == spec.defines
        && cached.entries == expected_entries
        && cached.source_closure_sha256 == source_closure_sha256
        && cached.cubin_sha256 == digest(&cubin_bytes)
    {
        return Ok((
            BuiltCubin {
                name: spec.name.clone(),
                path: path.clone(),
                identity_path: identity_path.clone(),
                line_info: options.profile_line_info,
                sha256: cached.cubin_sha256.clone(),
                source_closure_sha256: source_closure_sha256.clone(),
                entries: cached.resources.clone(),
            },
            cached,
            membership,
        ));
    }
    let root =
        fs::canonicalize(&spec.sources_root).map_err(|e| io("canonicalize sources_root", e))?;
    for file in &files {
        let Ok(relative) = file.strip_prefix(&root) else {
            continue;
        };
        let destination = stage.join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|e| io("create staged source directory", e))?;
        }
        fs::copy(file, destination).map_err(|e| io("stage source file", e))?;
    }
    let mut command = nvcc_command(&toolchain.tool("nvcc"));
    command.current_dir(&stage);
    if nvcc_wrapper().is_some()
        && let Some(variables) = cache_key::cache_variables(
            toolchain,
            flags,
            &files,
            &generated,
            &spec.defines,
            &source_closure_sha256,
            options,
        )
    {
        command.envs(variables);
    }
    command.args(flags).arg("-I.");
    for dir in &spec.include_dirs {
        command.arg(format!("-I{}", dir.display()));
    }
    for (key, value) in &spec.defines {
        command.arg(format!("-D{key}={value}"));
    }
    command
        .arg(format!("{}.cu", spec.name))
        .arg("-o")
        .arg(&path);
    run(command, &format!("{}: nvcc cubin", spec.name))?;
    let entries = audit_cubin_with_line_info(toolchain, spec, &path, options.profile_line_info)?;
    let sha256 = digest(&fs::read(&path).map_err(|e| io("read built cubin", e))?);
    let identity = CubinIdentity {
        name: spec.name.clone(),
        line_info: options.profile_line_info,
        toolchain_version: toolchain.version.clone(),
        nvcc_sha256: toolchain.nvcc_sha256.clone(),
        flags: flags.iter().map(|flag| (*flag).to_owned()).collect(),
        defines: spec.defines.clone(),
        entries: expected_entries,
        source_closure_sha256: source_closure_sha256.clone(),
        cubin_sha256: sha256.clone(),
        resources: entries.clone(),
    };
    let mut identity_json = serde_json::to_string_pretty(&identity)
        .map_err(|e| err(format!("serialize {} identity: {e}", spec.name)))?;
    identity_json.push('\n');
    fs::write(&identity_path, identity_json).map_err(|e| io("write cubin identity", e))?;
    Ok((
        BuiltCubin {
            name: spec.name.clone(),
            path,
            identity_path,
            line_info: options.profile_line_info,
            sha256,
            source_closure_sha256,
            entries,
        },
        identity,
        membership,
    ))
}

/// Emit the runtime cubin record and static table together, without a runtime
/// dependency on the build driver. `ranks` returns zero for every topology.
pub fn write_embedded(
    out: &Path,
    cubins: &[BuiltCubin],
    ranks: impl Fn(&str) -> u32,
) -> Result<(), BuildError> {
    write_embedded_with_prefix(out, Path::new(""), cubins, ranks)
}

/// Embed a selection built in a child of OUT_DIR. Separate selections retain
/// separate reports and locks while sharing one native bundle.
pub fn write_embedded_with_prefix(
    out: &Path,
    prefix: &Path,
    cubins: &[BuiltCubin],
    ranks: impl Fn(&str) -> u32,
) -> Result<(), BuildError> {
    use std::fmt::Write as _;
    if prefix
        .components()
        .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(err(
            "embedded selection prefix must be a confined relative path",
        ));
    }
    let mut source = String::from(
        "pub(crate) struct EmbeddedCubin {\n\
         #[allow(dead_code, reason = \"names support bundle lookup and test diagnostics\")]\n\
         name: &'static str,\n\
         bytes: &'static [u8],\n\
         #[allow(dead_code, reason = \"only bundles with multiple topologies filter by rank count\")]\n\
         ranks: u32,\n\
         symbols: &'static [&'static str],\n\
         #[allow(dead_code, reason = \"bundles that check launch blocks read it\")]\n\
         max_threads: &'static [u32],\n}\n\
         pub(crate) static CUBINS: &[EmbeddedCubin] = &[\n",
    );
    for cubin in cubins {
        let file = cubin
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| err("cubin filename is not UTF-8"))?;
        write!(source,
            "EmbeddedCubin {{ name: {:?}, bytes: include_bytes!(concat!(env!(\"OUT_DIR\"), {:?})), ranks: {}, symbols: &[",
            cubin.name, format!("/{}", prefix.join(file).display()), ranks(&cubin.name),
        ).expect("generated source");
        for entry in &cubin.entries {
            write!(source, "{:?},", entry.symbol).expect("generated source");
        }
        // Each symbol's launch-bound threads, in `symbols` order.
        source.push_str("], max_threads: &[");
        for entry in &cubin.entries {
            write!(source, "{},", entry.max_threads).expect("generated source");
        }
        source.push_str("] },\n");
    }
    source.push_str("];\n");
    fs::write(out.join("embedded.rs"), source)
        .map_err(|error| io("write embedded cubin list", error))
}

#[cfg(test)]
mod tests {
    use super::{BuildOptions, MODEL_RELOCK_MARKER, ModelLock, build_workers};
    use std::path::Path;

    #[test]
    fn model_lock_verifies_in_place_unless_relocking() {
        let model = tempfile_dir("model");
        let lock = model.join("bundle/kernels.lock");
        std::fs::create_dir_all(lock.parent().unwrap()).unwrap();
        let verify = BuildOptions {
            emit_cargo_directives: true,
            ..BuildOptions::default()
        };
        let plain = ModelLock::new(&lock, None, verify).unwrap();
        assert_eq!(plain.path(), lock);
        assert!(!plain.options().lock_update && !plain.relocking());
        // Outside a relock the environment's options apply, as
        // BuildOptions::from_env reads them: the in-tree update flow.
        let update = BuildOptions {
            lock_update: true,
            profile_line_info: true,
            ..verify
        };
        let updating = ModelLock::new(&lock, None, update).unwrap();
        assert_eq!(updating.path(), lock);
        assert!(!updating.relocking());
        let options = updating.options();
        assert!(options.lock_update && options.profile_line_info && options.emit_cargo_directives);

        let relock = tempfile_dir("relock");
        // A directory relock did not create, such as a configured workspace
        // path, refuses rather than skipping verification.
        let error = ModelLock::new(&lock, Some(relock.clone().into()), verify).unwrap_err();
        assert!(error.to_string().contains(MODEL_RELOCK_MARKER), "{error}");
        std::fs::write(relock.join(MODEL_RELOCK_MARKER), "").unwrap();
        // A relock writes its candidate whatever the environment says, and
        // never with line information.
        let candidate = ModelLock::new(&lock, Some(relock.clone().into()), update).unwrap();
        let resolved = std::fs::canonicalize(lock.parent().unwrap())
            .unwrap()
            .join("kernels.lock");
        assert_eq!(
            candidate.path(),
            relock.join(resolved.strip_prefix("/").unwrap())
        );
        assert!(candidate.options().lock_update && !candidate.options().profile_line_info);
        assert!(candidate.path().parent().unwrap().is_dir());

        for bad in ["relative/dir", "/nonexistent-relock-candidates"] {
            assert!(ModelLock::new(&lock, Some(bad.into()), verify).is_err());
        }
        assert!(ModelLock::new(Path::new("/"), Some(relock.clone().into()), verify).is_err());
        for dir in [model, relock] {
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    fn tempfile_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kernel-build-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn requested_build_jobs_only_lower_the_worker_count() {
        assert_eq!(build_workers(48, None), 16);
        assert_eq!(build_workers(4, None), 4);
        assert_eq!(build_workers(48, Some("6")), 6);
        assert_eq!(build_workers(4, Some("12")), 4);
        for ignored in ["0", "", "many", "-3"] {
            assert_eq!(build_workers(48, Some(ignored)), 16);
        }
        assert_eq!(build_workers(0, None), 1);
    }
}
