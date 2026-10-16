//! Explicit, offline AOT compilation and inspection of native CUDA cubins.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use runtime_weights::sha256_bytes;
use serde::{Deserialize, Serialize};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Deserialize)]
pub struct KernelManifestV1 {
    #[serde(default)]
    pub cache_dependency_roots: Vec<PathBuf>,
    #[serde(default)]
    pub cache_tool_paths: Vec<PathBuf>,
    pub source_paths: Vec<PathBuf>,
    #[serde(default)]
    pub header_paths: Vec<PathBuf>,
    pub nvcc_path: PathBuf,
    pub nvcc_version: String,
    pub cuobjdump_path: PathBuf,
    pub cuobjdump_version: String,
    pub readelf_path: PathBuf,
    pub readelf_version: String,
    pub target_architecture: String,
    pub flags: Vec<String>,
    pub expected_exported_symbols: Vec<String>,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct KernelReceiptV1 {
    schema: u32,
    status: String,
    manifest_sha256: String,
    source_paths: Vec<String>,
    source_sha256: Vec<String>,
    header_paths: Vec<String>,
    header_sha256: Vec<String>,
    nvcc_path: String,
    nvcc_version: String,
    nvcc_sha256: String,
    cuobjdump_path: String,
    cuobjdump_version: String,
    cuobjdump_sha256: String,
    readelf_path: String,
    readelf_version: String,
    readelf_sha256: String,
    flags: Vec<String>,
    flags_sha256: String,
    target_architecture: String,
    cubin_path: String,
    cubin_sha256: String,
    sass_sha256: String,
    exported_symbols: Vec<String>,
}

struct Context {
    manifest: KernelManifestV1,
    manifest_sha256: String,
    source_paths: Vec<PathBuf>,
    header_paths: Vec<PathBuf>,
    output: PathBuf,
    tools: Tools,
}

struct Tools {
    nvcc: Tool,
    cuobjdump: Tool,
    readelf: Tool,
}

struct Tool {
    path: PathBuf,
    version: String,
    sha256: String,
}

struct Audit {
    sass_sha256: String,
    symbols: Vec<String>,
}

/// Actual nvcc dispatches made by one cached lookup. A dependency scan reads the
/// include closure with `-M`; a compilation emits a cubin. They are counted apart
/// so a warm lookup can be told from a rebuild.
#[derive(Default)]
struct Dispatches {
    compilations: std::cell::Cell<u64>,
    dependency_scans: std::cell::Cell<u64>,
}
impl Dispatches {
    fn record(counter: &std::cell::Cell<u64>) {
        counter.set(counter.get() + 1);
    }
}

pub fn build(manifest_path: &Path) -> Result<String, String> {
    let context = load(manifest_path, true)?;
    reject_incomplete_pair(&context.output)?;
    let initial_sources = digests(&context.source_paths)?;
    let initial_headers = digests(&context.header_paths)?;
    verify_tools(&context)?;
    let temporary = reserve_temp(&context.output)?;
    let _temporary_guard = TempGuard::new(temporary.clone());
    let compile_result = compile(&context, &temporary);
    if let Err(error) = compile_result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    let bytes = match fs::read(&temporary) {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(format!("cannot read compiler output: {error}"));
        }
    };
    if digests(&context.source_paths)? != initial_sources
        || digests(&context.header_paths)? != initial_headers
    {
        let _ = fs::remove_file(&temporary);
        return Err("declared source or header changed during compilation".to_owned());
    }
    verify_tool(&context.tools.nvcc, &context.manifest.nvcc_version)?;
    let audit = match audit(&context, &temporary, &bytes) {
        Ok(audit) => audit,
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
    };
    let receipt = make_receipt(&context, &initial_sources, &initial_headers, &bytes, audit);
    fsync_file(&temporary)?;
    publish_no_replace(&temporary, &context.output)?;
    let cubin_published = true;
    if let Err(error) = sync_directory(&context.output) {
        let _ = fs::remove_file(&context.output);
        return Err(error);
    }
    let receipt_path = context.output.with_extension("receipt.json");
    let receipt_temp = match reserve_temp(&receipt_path) {
        Ok(path) => path,
        Err(error) => {
            if cubin_published {
                let _ = fs::remove_file(&context.output);
            }
            return Err(error);
        }
    };
    let _receipt_guard = TempGuard::new(receipt_temp.clone());
    let receipt_bytes =
        serde_json::to_vec_pretty(&receipt).map_err(|e| format!("cannot encode receipt: {e}"))?;
    if let Err(error) = write_fsynced(&receipt_temp, &receipt_bytes)
        .and_then(|_| publish_no_replace(&receipt_temp, &receipt_path))
    {
        let _ = fs::remove_file(&receipt_temp);
        let _ = fs::remove_file(&context.output);
        return Err(format!("cannot publish receipt: {error}"));
    }
    sync_directory(&context.output)?;
    Ok(format!(
        "built native cubin {} and receipt {}",
        context.output.display(),
        receipt_path.display()
    ))
}

pub fn inspect(manifest_path: &Path) -> Result<String, String> {
    let context = load(manifest_path, false)?;
    let sources = digests(&context.source_paths)?;
    let headers = digests(&context.header_paths)?;
    verify_tools(&context)?;
    let bytes = fs::read(&context.output)
        .map_err(|e| format!("cannot read cubin {}: {e}", context.output.display()))?;
    let audit = audit(&context, &context.output, &bytes)?;
    let expected = make_receipt(&context, &sources, &headers, &bytes, audit);
    let receipt_path = context.output.with_extension("receipt.json");
    let receipt_bytes = fs::read(&receipt_path)
        .map_err(|e| format!("cannot read receipt {}: {e}", receipt_path.display()))?;
    let actual: KernelReceiptV1 = serde_json::from_slice(&receipt_bytes)
        .map_err(|e| format!("invalid kernel receipt: {e}"))?;
    if actual != expected {
        return Err(
            "kernel receipt does not match current manifest, inputs, tools, or cubin".to_owned(),
        );
    }
    Ok(format!(
        "inspected native cubin {}: architecture {}, sha256 {}",
        context.output.display(),
        context.manifest.target_architecture,
        sha256_bytes(&bytes)
    ))
}

fn load(path: &Path, create_output_parent: bool) -> Result<Context, String> {
    load_context(path, create_output_parent, false)
}

fn load_context(path: &Path, create_output_parent: bool, cache: bool) -> Result<Context, String> {
    let manifest_bytes =
        fs::read(path).map_err(|e| format!("cannot read manifest {}: {e}", path.display()))?;
    let manifest: KernelManifestV1 = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("invalid kernel manifest: {e}"))?;
    if manifest.source_paths.len() != 1 {
        return Err("one cubin manifest must declare exactly one source path".to_owned());
    }
    if manifest.target_architecture.is_empty() || manifest.expected_exported_symbols.is_empty() {
        return Err("manifest must declare architecture and exported symbols".to_owned());
    }
    if manifest.flags.iter().any(|flag| {
        let lower = flag.to_ascii_lowercase();
        (lower == "-ptx" || lower == "--ptx" || lower == "--device-c")
            || lower.contains("nvrtc")
            || lower.contains("triton")
            || lower.contains("tilelang")
            || flag.contains(';')
            || flag.contains('|')
            || flag.contains('&')
            || flag == "-o"
            || flag == "--output-file"
    }) {
        return Err("manifest flags contain a forbidden JIT, shell, or output override".to_owned());
    }
    let base = path.parent().unwrap_or(Path::new("."));
    let source_paths = canonical_inputs(&manifest.source_paths, base)?;
    let header_paths = canonical_inputs(&manifest.header_paths, base)?;
    let output = if cache {
        PathBuf::new()
    } else {
        resolve_output(base, &manifest.output_path, create_output_parent)?
    };
    let tools = Tools {
        nvcc: pinned_tool(&manifest.nvcc_path, &manifest.nvcc_version, base)?,
        cuobjdump: pinned_tool(&manifest.cuobjdump_path, &manifest.cuobjdump_version, base)?,
        readelf: pinned_tool(&manifest.readelf_path, &manifest.readelf_version, base)?,
    };
    Ok(Context {
        manifest,
        manifest_sha256: sha256_bytes(&manifest_bytes),
        source_paths,
        header_paths,
        output,
        tools,
    })
}

fn canonical_inputs(paths: &[PathBuf], base: &Path) -> Result<Vec<PathBuf>, String> {
    paths
        .iter()
        .map(|path| {
            fs::canonicalize(resolve(base, path))
                .map_err(|e| format!("cannot resolve declared input {}: {e}", path.display()))
        })
        .collect()
}
fn resolve_output(base: &Path, path: &Path, create_parent: bool) -> Result<PathBuf, String> {
    let path = resolve(base, path);
    let parent = path
        .parent()
        .ok_or_else(|| "output path has no parent".to_owned())?;
    if create_parent {
        fs::create_dir_all(parent).map_err(|e| format!("cannot create output directory: {e}"))?;
    }
    let parent =
        fs::canonicalize(parent).map_err(|e| format!("cannot resolve output directory: {e}"))?;
    Ok(parent.join(
        path.file_name()
            .ok_or_else(|| "output path has no file name".to_owned())?,
    ))
}
fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}
fn pinned_tool(path: &Path, version: &str, base: &Path) -> Result<Tool, String> {
    let path = fs::canonicalize(resolve(base, path))
        .map_err(|e| format!("cannot resolve pinned tool {}: {e}", path.display()))?;
    let sha256 = sha256_bytes(
        &fs::read(&path).map_err(|e| format!("cannot read pinned tool {}: {e}", path.display()))?,
    );
    Ok(Tool {
        path,
        version: version.to_owned(),
        sha256,
    })
}
fn verify_tools(context: &Context) -> Result<(), String> {
    verify_tool(&context.tools.nvcc, &context.manifest.nvcc_version)?;
    verify_tool(
        &context.tools.cuobjdump,
        &context.manifest.cuobjdump_version,
    )?;
    verify_tool(&context.tools.readelf, &context.manifest.readelf_version)
}
fn verify_tool(tool: &Tool, expected: &str) -> Result<(), String> {
    let actual = command_output(&tool.path, &["--version"])?;
    if actual != expected
        || sha256_bytes(&fs::read(&tool.path).map_err(|e| format!("cannot read pinned tool: {e}"))?)
            != tool.sha256
    {
        return Err(format!(
            "pinned tool identity changed: {}",
            tool.path.display()
        ));
    }
    Ok(())
}
fn command_output(path: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new(path)
        .args(args)
        .output()
        .map_err(|e| format!("cannot execute {}: {e}", path.display()))?;
    if !output.status.success() {
        return Err(format!("{} {:?} failed", path.display(), args));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}
fn compile(context: &Context, temporary: &Path) -> Result<(), String> {
    let mut command = Command::new(&context.tools.nvcc.path);
    command
        .args(&context.manifest.flags)
        .arg("-cubin")
        .arg("-o")
        .arg(temporary)
        .arg(&context.source_paths[0]);
    for header in &context.header_paths {
        command.arg(format!("-I{}", header.parent().unwrap().display()));
    }
    let output = command
        .output()
        .map_err(|e| format!("cannot execute nvcc: {e}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "nvcc failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}
fn audit(context: &Context, cubin: &Path, bytes: &[u8]) -> Result<Audit, String> {
    if bytes.is_empty() {
        return Err("cubin is empty".to_owned());
    }
    let elf = command_output(
        &context.tools.cuobjdump.path,
        &["--dump-elf", &cubin.to_string_lossy()],
    )?;
    let architectures = elf
        .lines()
        .filter(|line| line.contains("64-bit ELF:"))
        .filter_map(|line| {
            line.split_once("sm=").map(|(_, rest)| {
                rest.split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .trim_matches(|c: char| c == ',' || c == ')' || c == ']')
            })
        })
        .filter(|field| !field.is_empty())
        .collect::<Vec<_>>();
    let expected_arch = context
        .manifest
        .target_architecture
        .trim_start_matches("sm_");
    let expected_elf_arch = if expected_arch == "120f" {
        "120"
    } else {
        expected_arch
    };
    if architectures.len() != 1
        || architectures[0] != expected_elf_arch
        || (expected_arch == "120f" && !elf.contains("Tool Command Line Arguments: -arch sm_120f "))
    {
        return Err(format!(
            "expected exactly one ELF sm field {}, got {:?}",
            context.manifest.target_architecture, architectures
        ));
    }
    let symbols_text = command_output(
        &context.tools.readelf.path,
        &["-Ws", &cubin.to_string_lossy()],
    )?;
    let symbols_text_dump = command_output(
        &context.tools.cuobjdump.path,
        &["--dump-elf-symbols", &cubin.to_string_lossy()],
    )?;
    let symbols = entry_symbols(&symbols_text_dump);
    let expected: BTreeSet<_> = context
        .manifest
        .expected_exported_symbols
        .iter()
        .cloned()
        .collect();
    if symbols.iter().cloned().collect::<BTreeSet<_>>() != expected
        || symbols.len() != expected.len()
    {
        return Err(format!(
            "entry symbols differ: expected {:?}, got {:?}",
            expected, symbols
        ));
    }
    let undefined = symbols_text
        .lines()
        .filter(|line| {
            line.contains(" FUNC ") && (line.contains(" UND ") || line.contains("SHN_UNDEF"))
        })
        .count();
    if undefined != 0 {
        return Err(format!("cubin has {undefined} undefined device functions"));
    }
    let sections = command_output(
        &context.tools.readelf.path,
        &["-SW", &cubin.to_string_lossy()],
    )?;
    if sections.lines().any(|line| {
        line.split_whitespace()
            .any(|field| field.to_ascii_lowercase().contains("ptx"))
    }) {
        return Err("cubin ELF contains a PTX section".to_owned());
    }
    if !command_output(
        &context.tools.cuobjdump.path,
        &["--dump-ptx", &cubin.to_string_lossy()],
    )?
    .trim()
    .is_empty()
    {
        return Err("cubin contains PTX extraction".to_owned());
    }
    let dynamic = command_output(
        &context.tools.readelf.path,
        &["-d", &cubin.to_string_lossy()],
    )?;
    if dynamic
        .lines()
        .any(|line| line.contains("NEEDED") || line.contains(".dynamic"))
    {
        return Err("cubin has dynamic dependencies".to_owned());
    }
    let sass = command_output(
        &context.tools.cuobjdump.path,
        &["--dump-sass", &cubin.to_string_lossy()],
    )?;
    if sass.trim().is_empty() {
        return Err("cubin has no SASS".to_owned());
    }
    Ok(Audit {
        sass_sha256: sha256_bytes(sass.as_bytes()),
        symbols,
    })
}
fn entry_symbols(text: &str) -> Vec<String> {
    let mut symbols = text
        .lines()
        .filter(|line| line.contains("STO_ENTRY"))
        .filter_map(|line| line.split_whitespace().last().map(str::to_owned))
        .collect::<Vec<_>>();
    symbols.sort();
    symbols.dedup();
    symbols
}
fn digests(paths: &[PathBuf]) -> Result<Vec<String>, String> {
    paths
        .iter()
        .map(|path| {
            fs::read(path)
                .map(|bytes| sha256_bytes(&bytes))
                .map_err(|e| format!("cannot read {}: {e}", path.display()))
        })
        .collect()
}
fn make_receipt(
    context: &Context,
    sources: &[String],
    headers: &[String],
    bytes: &[u8],
    audit: Audit,
) -> KernelReceiptV1 {
    KernelReceiptV1 {
        schema: 1,
        status: "passed".to_owned(),
        manifest_sha256: context.manifest_sha256.clone(),
        source_paths: context
            .source_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect(),
        source_sha256: sources.to_vec(),
        header_paths: context
            .header_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect(),
        header_sha256: headers.to_vec(),
        nvcc_path: context.tools.nvcc.path.display().to_string(),
        nvcc_version: context.tools.nvcc.version.clone(),
        nvcc_sha256: context.tools.nvcc.sha256.clone(),
        cuobjdump_path: context.tools.cuobjdump.path.display().to_string(),
        cuobjdump_version: context.tools.cuobjdump.version.clone(),
        cuobjdump_sha256: context.tools.cuobjdump.sha256.clone(),
        readelf_path: context.tools.readelf.path.display().to_string(),
        readelf_version: context.tools.readelf.version.clone(),
        readelf_sha256: context.tools.readelf.sha256.clone(),
        flags: context.manifest.flags.clone(),
        flags_sha256: sha256_bytes(context.manifest.flags.join("\0").as_bytes()),
        target_architecture: context.manifest.target_architecture.clone(),
        cubin_path: context.output.display().to_string(),
        cubin_sha256: sha256_bytes(bytes),
        sass_sha256: audit.sass_sha256,
        exported_symbols: audit.symbols,
    }
}
struct TempGuard {
    path: Option<PathBuf>,
}
impl TempGuard {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }
}
impl Drop for TempGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_file(path);
        }
    }
}
fn reject_incomplete_pair(output: &Path) -> Result<(), String> {
    let receipt = output.with_extension("receipt.json");
    match (output.exists(), receipt.exists()) {
        (true, false) => Err(format!(
            "refusing incomplete kernel pair, cubin exists without receipt: {}",
            output.display()
        )),
        (false, true) => Err(format!(
            "refusing incomplete kernel pair, receipt exists without cubin: {}",
            receipt.display()
        )),
        _ => Ok(()),
    }
}
fn random_token() -> Result<String, String> {
    let mut bytes = [0_u8; 16];
    File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|e| format!("cannot obtain random temporary identity: {e}"))?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("clock error: {e}"))?
        .as_nanos();
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    Ok(format!(
        "{:x}-{}-{}",
        u128::from_le_bytes(bytes),
        now,
        sequence
    ))
}
fn reserve_temp(destination: &Path) -> Result<PathBuf, String> {
    let parent = destination
        .parent()
        .ok_or_else(|| "temporary destination has no parent".to_owned())?;
    for _ in 0..8 {
        let candidate = parent.join(format!(
            ".{}.tmp-{}",
            destination.file_name().unwrap().to_string_lossy(),
            random_token()?
        ));
        match OpenOptions::new()
            .write(true)
            .read(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(_) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("cannot reserve temporary file: {error}")),
        }
    }
    Err("cannot reserve unique temporary file".to_owned())
}
fn fsync_file(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|e| format!("cannot fsync {}: {e}", path.display()))
}
fn write_fsynced(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|e| format!("cannot open temporary receipt: {e}"))?;
    file.write_all(bytes)
        .map_err(|e| format!("cannot write temporary receipt: {e}"))?;
    file.sync_all()
        .map_err(|e| format!("cannot fsync temporary receipt: {e}"))
}
fn publish_no_replace(temp: &Path, destination: &Path) -> Result<(), String> {
    fs::hard_link(temp, destination).map_err(|e| {
        format!(
            "cannot publish {} without replacement: {e}",
            destination.display()
        )
    })?;
    fs::remove_file(temp).map_err(|e| format!("cannot remove published temporary file: {e}"))
}
fn sync_directory(path: &Path) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "published path has no parent".to_owned())?;
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|e| format!("cannot fsync directory {}: {e}", parent.display()))
}

/// Returns a JSON development receipt. Cached mode never writes manifest.output_path.
/// Build permits a miss; inspect requires an existing verified entry.
pub fn cached(manifest_path: &Path, root: &Path, allow_build: bool) -> Result<String, String> {
    let context = load_context(manifest_path, false, true)?;
    verify_tools(&context)?;
    let dispatches = Dispatches::default();
    let identity = cache_identity(&context, manifest_path, &dispatches)?;
    let key = sha256_bytes(&serde_json::to_vec(&identity).map_err(|e| e.to_string())?);
    fs::create_dir_all(root).map_err(|e| format!("cannot create cache: {e}"))?;
    let root = fs::canonicalize(root).map_err(|e| e.to_string())?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join(format!("{key}.lock")))
        .map_err(|e| e.to_string())?;
    lock.lock()
        .map_err(|e| format!("cannot lock cache key: {e}"))?;
    let destination = root.join(&key);
    let hit = destination.try_exists().map_err(|e| e.to_string())?;
    if !hit {
        if !allow_build {
            return Err(format!("cache entry absent: {key}"));
        }
        let stage = root.join(format!(".{key}.tmp-{}", random_token()?));
        fs::create_dir(&stage).map_err(|e| e.to_string())?;
        let _stage = CacheStage(stage.clone());
        let cubin = stage.join("module.cubin");
        let mut command = cache_compiler(&context);
        command
            .arg("-cubin")
            .arg("-o")
            .arg(&cubin)
            .arg(&context.source_paths[0]);
        Dispatches::record(&dispatches.compilations);
        let output = command
            .output()
            .map_err(|e| format!("cannot execute nvcc: {e}"))?;
        if !output.status.success() {
            return Err(format!(
                "nvcc failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let bytes = small_artifact(&cubin)?;
        let audit = audit(&context, &cubin, &bytes)?;
        let sass = command_output(
            &context.tools.cuobjdump.path,
            &["--dump-sass", &cubin.to_string_lossy()],
        )?;
        if sha256_bytes(sass.as_bytes()) != audit.sass_sha256 {
            return Err("SASS changed during audit".to_owned());
        }
        let receipt = CacheReceipt {
            schema: "module-cache-v1".to_owned(),
            qualified: false,
            key: key.clone(),
            identity: identity.clone(),
            cubin_sha256: sha256_bytes(&bytes),
            sass_sha256: audit.sass_sha256,
            exported_symbols: audit.symbols,
        };
        fs::write(stage.join("module.sass"), sass).map_err(|e| e.to_string())?;
        fs::write(
            stage.join("receipt.json"),
            serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if cache_identity(&context, manifest_path, &dispatches)? != identity {
            return Err("cache inputs changed during compilation".to_owned());
        }
        verify_tools(&context)?;
        for name in ["module.cubin", "module.sass", "receipt.json"] {
            fsync_file(&stage.join(name))?;
        }
        fsync_file(&stage)?;
        // The held key lock serializes every publication to this directory.
        fs::rename(&stage, &destination).map_err(|e| format!("cannot publish cache entry: {e}"))?;
        sync_directory(&destination)?;
    }
    let receipt = verify_cache_entry(&context, &destination, &key, &identity)?;
    if cache_identity(&context, manifest_path, &dispatches)? != identity {
        return Err("cache inputs changed during verification".to_owned());
    }
    serde_json::to_string(&serde_json::json!({
        "schema": "module-cache-result-v1", "qualified": false,
        "cache_hit": hit, "key": key, "cubin_path": destination.join("module.cubin"),
        "sass_path": destination.join("module.sass"), "receipt_path": destination.join("receipt.json"),
        "cubin_sha256": receipt.cubin_sha256,
        "nvcc_compilations": dispatches.compilations.get(),
        "nvcc_dependency_scans": dispatches.dependency_scans.get(),
    })).map_err(|e| e.to_string())
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheReceipt {
    schema: String,
    qualified: bool,
    key: String,
    identity: serde_json::Value,
    cubin_sha256: String,
    sass_sha256: String,
    exported_symbols: Vec<String>,
}

struct CacheStage(PathBuf);
impl Drop for CacheStage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn small_artifact(path: &Path) -> Result<Vec<u8>, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|e| format!("cache artifact missing: {e}"))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > 64 * 1024 * 1024 {
        return Err(format!("invalid cache artifact: {}", path.display()));
    }
    fs::read(path).map_err(|e| e.to_string())
}

fn verify_cache_entry(
    context: &Context,
    dir: &Path,
    key: &str,
    identity: &serde_json::Value,
) -> Result<CacheReceipt, String> {
    if !fs::symlink_metadata(dir)
        .map_err(|e| e.to_string())?
        .is_dir()
    {
        return Err("cache entry is not a directory".to_owned());
    }
    let names = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<Result<BTreeSet<_>, _>>()
        .map_err(|e| e.to_string())?;
    if names
        != ["module.cubin", "module.sass", "receipt.json"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    {
        return Err("cache entry has missing or extra artifacts".to_owned());
    }
    let receipt: CacheReceipt = serde_json::from_slice(&small_artifact(&dir.join("receipt.json"))?)
        .map_err(|e| format!("invalid cache receipt: {e}"))?;
    let bytes = small_artifact(&dir.join("module.cubin"))?;
    let sass = small_artifact(&dir.join("module.sass"))?;
    let audit = audit(context, &dir.join("module.cubin"), &bytes)?;
    if receipt.schema != "module-cache-v1"
        || receipt.qualified
        || receipt.key != key
        || &receipt.identity != identity
        || receipt.cubin_sha256 != sha256_bytes(&bytes)
        || receipt.sass_sha256 != sha256_bytes(&sass)
        || receipt.sass_sha256 != audit.sass_sha256
        || receipt.exported_symbols != audit.symbols
    {
        return Err("corrupted cache entry: receipt, inputs, cubin, or SASS differ".to_owned());
    }
    Ok(receipt)
}

fn cache_compiler(context: &Context) -> Command {
    let mut command = Command::new(&context.tools.nvcc.path);
    // No caller environment may inject unkeyed compiler flags or include paths.
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("LC_ALL", "C")
        .current_dir(context.source_paths[0].parent().unwrap())
        .args(&context.manifest.flags)
        .arg(format!("-arch={}", context.manifest.target_architecture));
    for header in &context.header_paths {
        command.arg(format!("-I{}", header.parent().unwrap().display()));
    }
    command
}

fn cache_identity(
    context: &Context,
    manifest_path: &Path,
    dispatches: &Dispatches,
) -> Result<serde_json::Value, String> {
    let base = manifest_path.parent().unwrap_or(Path::new("."));
    // Restrict cached flags so response files, plugins, alternate tools, and
    // implicit output modes cannot introduce untracked compilation inputs.
    for flag in &context.manifest.flags {
        if !(flag.starts_with("-std=")
            || flag.starts_with("--std=")
            || matches!(
                flag.as_str(),
                "-O0" | "-O1" | "-O2" | "-O3" | "-lineinfo" | "--use_fast_math"
            )
            || flag.starts_with("-D")
            || (flag.starts_with("-I") && Path::new(&flag[2..]).is_absolute()))
        {
            return Err(format!("unsupported cached compiler flag: {flag}"));
        }
    }
    let dependencies = compiler_dependencies(context, dispatches)?;
    let mut paths = BTreeSet::new();
    paths.extend(context.source_paths.iter().cloned());
    paths.extend(context.header_paths.iter().cloned());
    let mut roots = canonical_inputs(&context.manifest.cache_dependency_roots, base)?;
    roots.extend(
        context
            .header_paths
            .iter()
            .map(|p| p.parent().unwrap().to_path_buf()),
    );
    roots.push(context.source_paths[0].parent().unwrap().to_path_buf());
    roots.extend(
        context
            .manifest
            .flags
            .iter()
            .filter_map(|f| f.strip_prefix("-I").map(PathBuf::from)),
    );
    let toolkit = context.tools.nvcc.path.parent().unwrap().parent().unwrap();
    if toolkit.join("include").is_dir() {
        roots.push(toolkit.join("include"));
    }
    let mut visited = BTreeSet::new();
    let mut symlinks = BTreeMap::new();
    for root in roots {
        collect_cache_files(&root, &mut paths, &mut visited, &mut symlinks, true)?;
    }
    for dependency in &dependencies {
        if !paths.contains(dependency) {
            return Err(format!(
                "uncovered compiler dependency {}; declare header_paths or cache_dependency_roots",
                dependency.display()
            ));
        }
    }
    // Extensionless compiler outputs are not inputs unless consumed or declared.
    // Scan on every lookup to detect newly selected extensionless includes.
    paths.retain(|path| {
        path.extension().is_some()
            || dependencies.contains(path)
            || context.source_paths.contains(path)
            || context.header_paths.contains(path)
    });
    let mut inputs = BTreeMap::new();
    for path in paths {
        inputs.insert(
            path.display().to_string(),
            sha256_bytes(&fs::read(&path).map_err(|e| e.to_string())?),
        );
    }
    let mut tool_paths: BTreeSet<_> = canonical_inputs(&context.manifest.cache_tool_paths, base)?
        .into_iter()
        .collect();
    tool_paths.extend([
        context.tools.nvcc.path.clone(),
        context.tools.cuobjdump.path.clone(),
        context.tools.readelf.path.clone(),
    ]);
    // nvcc delegates to toolkit executables and the system host compiler.
    for root in [
        toolkit.join("bin"),
        toolkit.join("nvvm/bin"),
        toolkit.join("nvvm/libdevice"),
    ] {
        if root.is_dir() {
            collect_cache_files(
                &root,
                &mut tool_paths,
                &mut BTreeSet::new(),
                &mut symlinks,
                false,
            )?;
        }
    }
    for name in ["/usr/bin/gcc", "/usr/bin/g++", "/usr/bin/as", "/usr/bin/ld"] {
        let path = PathBuf::from(name);
        if path.is_file() {
            tool_paths.insert(fs::canonicalize(path).map_err(|e| e.to_string())?);
        }
    }
    if Path::new("/usr/bin/g++").is_file() {
        let cc1 = command_output(Path::new("/usr/bin/g++"), &["-print-prog-name=cc1plus"])?;
        tool_paths.insert(fs::canonicalize(cc1.trim()).map_err(|e| e.to_string())?);
    }
    let mut tools = BTreeMap::new();
    for path in tool_paths {
        tools.insert(
            path.display().to_string(),
            sha256_bytes(&fs::read(path).map_err(|e| e.to_string())?),
        );
    }
    Ok(serde_json::json!({"schema": "module-inputs-v1",
        "dependency_policy": "compiler-closure-extensionless-v1",
        "compiler_dependencies": dependencies, "inputs": inputs, "tools": tools, "symlinks": symlinks,
        "nvcc_version": context.tools.nvcc.version, "cuobjdump_version": context.tools.cuobjdump.version,
        "readelf_version": context.tools.readelf.version, "flags": context.manifest.flags,
        "architecture": context.manifest.target_architecture, "exports": context.manifest.expected_exported_symbols,
        "source_paths": context.source_paths, "header_paths": context.header_paths,
        "environment": {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}}))
}

fn collect_cache_files(
    root: &Path,
    paths: &mut BTreeSet<PathBuf>,
    visited: &mut BTreeSet<PathBuf>,
    symlinks: &mut BTreeMap<String, String>,
    headers: bool,
) -> Result<(), String> {
    if fs::symlink_metadata(root)
        .map_err(|e| e.to_string())?
        .is_symlink()
    {
        symlinks.insert(
            root.display().to_string(),
            fs::read_link(root)
                .map_err(|e| e.to_string())?
                .display()
                .to_string(),
        );
    }
    let root = fs::canonicalize(root)
        .map_err(|e| format!("cannot resolve cache input {}: {e}", root.display()))?;
    if !visited.insert(root.clone()) {
        return Ok(());
    }
    if root.is_dir() {
        for entry in fs::read_dir(&root).map_err(|e| e.to_string())? {
            collect_cache_files(
                &entry.map_err(|e| e.to_string())?.path(),
                paths,
                visited,
                symlinks,
                headers,
            )?;
        }
    } else if root.is_file()
        && (!headers
            || root.extension().is_none_or(|e| {
                matches!(
                    e.to_str(),
                    Some("h" | "hpp" | "hxx" | "cuh" | "inc" | "inl" | "tcc" | "def")
                )
            }))
    {
        paths.insert(root);
    }
    Ok(())
}

fn compiler_dependencies(
    context: &Context,
    dispatches: &Dispatches,
) -> Result<BTreeSet<PathBuf>, String> {
    Dispatches::record(&dispatches.dependency_scans);
    let output = cache_compiler(context)
        .arg("-M")
        .arg("-MT")
        .arg("module")
        .arg(&context.source_paths[0])
        .output()
        .map_err(|e| format!("cannot scan CUDA dependencies: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "CUDA dependency scan failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|e| e.to_string())?
        .replace("\\\n", " ");
    let (_, dependencies) = text
        .split_once(':')
        .ok_or("invalid CUDA dependency output")?;
    let mut paths = BTreeSet::new();
    // Escaped filenames are refused rather than ambiguously parsed.
    if dependencies.contains('\\') || dependencies.contains('$') {
        return Err("unsupported dependency filename escape".to_owned());
    }
    for dependency in dependencies.split_whitespace() {
        let path = fs::canonicalize(resolve(
            context.source_paths[0].parent().unwrap(),
            Path::new(dependency),
        ))
        .map_err(|e| format!("cannot resolve compiler dependency {dependency}: {e}"))?;
        paths.insert(path);
    }
    if paths.is_empty() {
        return Err("empty compiler dependency list".to_owned());
    }
    if !paths.contains(&context.source_paths[0]) {
        return Err("compiler dependency list omits source".to_owned());
    }
    Ok(paths)
}
