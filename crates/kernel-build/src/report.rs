//! Build evidence from the same inputs used for the locked source digest.

use crate::selection::{BuildContext, EntryDeclaration};
use crate::{
    BuildError, BuildOptions, BuiltCubin, CubinIdentity, CubinSpec, Toolchain, digest, err, io,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const BUILD_REPORT_FILE: &str = "kernel-build-report.json";
const SCHEMA: &str = "chopinfer-kernel-build-v1";

pub(crate) struct ClosureInput {
    pub key: String,
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    Project,
    Generated,
    Toolkit,
    System,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub kind: SourceKind,
    pub owner: Option<String>,
    pub path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceFile {
    pub identity: SourceIdentity,
    pub sha256: String,
    pub bytes: u64,
    /// Local lookup information, separate from portable identity.
    pub location: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Membership {
    /// The exact key used by the legacy closure hash.
    pub closure_key: String,
    /// Reference into the report's deduplicated file table.
    pub file: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildRecipe {
    pub toolchain_version: String,
    pub nvcc_sha256: String,
    pub flags: Vec<String>,
    pub defines: Vec<(String, String)>,
    pub headers: Vec<String>,
    pub include_dirs: Vec<PathBuf>,
    pub generated_sha256: BTreeMap<String, String>,
    pub source_closure_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReportedCubin {
    pub name: String,
    pub path: PathBuf,
    pub identity_path: PathBuf,
    /// The unchanged locked identity, including entry ABI and audited resources.
    pub identity: serde_json::Value,
    pub recipe: BuildRecipe,
    pub recipe_sha256: String,
    pub membership: Vec<Membership>,
    pub declarations: Vec<EntryDeclaration>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockStatus {
    Verified,
    Updated,
    Profiling,
    Skipped,
}

/// One source family of a directory-form lock, as compared by this build.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FamilyLock {
    pub family: String,
    pub lock_path: PathBuf,
    pub lock_sha256: Option<String>,
    pub provenance_path: PathBuf,
    /// The family's cubins, in lock order.
    pub cubins: Vec<String>,
    /// Cubins whose recorded source closure differs from this build. This is
    /// a report only; the gating records matched.
    pub provenance_changed: Vec<String>,
}

/// The lock a build compared or wrote, for its report.
pub(crate) enum LockEvidence<'a> {
    File(&'a Path),
    Families(&'a Path, Vec<FamilyLock>),
    /// Profiling builds bypass the lock.
    Bypassed(&'a Path),
}

/// Membership describes this build's consumers only. A validation tool must
/// still retain consumers with absent or stale reports. Declaration coverage
/// is not numerical correctness or deployment qualification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildReport {
    pub schema: String,
    pub context: Option<BuildContext>,
    pub source_membership_complete: bool,
    pub source_ownership_complete: bool,
    pub entry_declarations_complete: bool,
    pub lock_status: LockStatus,
    /// The lock file, or the directory of a directory-form lock.
    pub lock_path: PathBuf,
    /// The lock file digest; `None` for a directory-form lock.
    pub lock_sha256: Option<String>,
    /// The compared families of a directory-form lock; empty for a lock file.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lock_families: Vec<FamilyLock>,
    pub compiler_path: PathBuf,
    pub files: BTreeMap<String, SourceFile>,
    pub cubins: Vec<ReportedCubin>,
}

fn file_id(file: &SourceFile) -> Result<String, BuildError> {
    let bytes = serde_json::to_vec(&(&file.identity, &file.sha256))
        .map_err(|e| err(format!("serialize source identity: {e}")))?;
    Ok(digest(&bytes))
}

impl BuildReport {
    pub fn read(path: &Path) -> Result<Self, BuildError> {
        let bytes = fs::read(path).map_err(|e| io("read kernel build report", e))?;
        serde_json::from_slice(&bytes).map_err(|e| err(format!("invalid kernel build report: {e}")))
    }

    /// Check current local inputs, compiler, generated units, cubins and lock.
    /// This checks evidence consistency, not the truth of native declarations.
    pub fn verify_current(&self) -> Result<(), BuildError> {
        if self.schema != SCHEMA
            || !self.source_membership_complete
            || self.lock_status != LockStatus::Verified
            || self.cubins.is_empty()
        {
            return Err(err(
                "kernel build report is incomplete or not lock-verified",
            ));
        }
        if self.lock_families.is_empty() {
            let identities = self.cubins.iter().map(|c| c.identity.clone()).collect();
            verify_lock(&self.lock_path, self.lock_sha256.as_deref(), identities)?;
        } else {
            let mut covered = Vec::new();
            for family in &self.lock_families {
                let mut identities = Vec::new();
                for name in &family.cubins {
                    let cubin = self
                        .cubins
                        .iter()
                        .find(|c| c.name == *name)
                        .ok_or_else(|| err("lock family names an unreported cubin"))?;
                    // The provenance file never gates, so it is not verified.
                    let mut identity = cubin.identity.clone();
                    identity
                        .as_object_mut()
                        .ok_or_else(|| err("reported cubin identity is not an object"))?
                        .remove("source_closure_sha256");
                    identities.push(identity);
                    covered.push(name);
                }
                verify_lock(&family.lock_path, family.lock_sha256.as_deref(), identities)?;
            }
            covered.sort();
            if !covered
                .iter()
                .copied()
                .eq(self.cubins.iter().map(|c| &c.name))
            {
                return Err(err("lock families do not cover the reported cubins once"));
            }
        }
        let compiler =
            digest(&fs::read(&self.compiler_path).map_err(|e| io("read reported compiler", e))?);
        let mut contents = BTreeMap::new();
        for (id, file) in &self.files {
            let bytes = fs::read(&file.location).map_err(|e| io("read reported source", e))?;
            if bytes.len() as u64 != file.bytes
                || digest(&bytes) != file.sha256
                || *id != file_id(file)?
            {
                return Err(err(format!(
                    "stale source membership: {}",
                    file.identity.path
                )));
            }
            contents.insert(id, bytes);
        }
        for cubin in &self.cubins {
            let identity: serde_json::Value = serde_json::from_slice(
                &fs::read(&cubin.identity_path)
                    .map_err(|e| io("read reported cubin identity", e))?,
            )
            .map_err(|e| err(format!("parse cubin identity: {e}")))?;
            if identity != cubin.identity || identity["nvcc_sha256"] != compiler {
                return Err(err(format!(
                    "{}: stale compiler or cubin identity",
                    cubin.name
                )));
            }
            let bytes = fs::read(&cubin.path).map_err(|e| io("read reported cubin", e))?;
            if identity["cubin_sha256"] != digest(&bytes) {
                return Err(err(format!("{}: cubin digest differs", cubin.name)));
            }
            let recipe = serde_json::to_vec(&cubin.recipe).map_err(|e| err(e.to_string()))?;
            if digest(&recipe) != cubin.recipe_sha256 {
                return Err(err(format!("{}: recipe digest differs", cubin.name)));
            }
            let mut keys = BTreeMap::new();
            for member in &cubin.membership {
                let bytes = contents
                    .get(&member.file)
                    .ok_or_else(|| err("source membership references a missing file"))?;
                if keys.insert(&member.closure_key, bytes).is_some() {
                    return Err(err("duplicate source closure key"));
                }
            }
            let mut hash = Sha256::new();
            for (key, bytes) in keys {
                hash.update((key.len() as u64).to_le_bytes());
                hash.update(key.as_bytes());
                hash.update((bytes.len() as u64).to_le_bytes());
                hash.update(bytes);
            }
            if identity["source_closure_sha256"] != format!("{:x}", hash.finalize()) {
                return Err(err(format!("{}: source closure differs", cubin.name)));
            }
        }
        Ok(())
    }
}

fn verify_lock(
    path: &Path,
    sha256: Option<&str>,
    identities: Vec<serde_json::Value>,
) -> Result<(), BuildError> {
    let lock = fs::read(path).map_err(|e| io("read reported lock", e))?;
    if sha256 != Some(digest(&lock).as_str()) {
        return Err(err("kernel build report has a stale lock"));
    }
    let locked: Vec<serde_json::Value> =
        serde_json::from_slice(&lock).map_err(|e| err(format!("parse reported lock: {e}")))?;
    if locked != identities {
        return Err(err("kernel build report identities differ from the lock"));
    }
    Ok(())
}

pub(crate) fn path(out: &Path, options: BuildOptions) -> PathBuf {
    out.join(if options.profile_line_info {
        "kernel-build-report.profile.json"
    } else {
        BUILD_REPORT_FILE
    })
}

fn save(out: &Path, options: BuildOptions, report: &BuildReport) -> Result<(), BuildError> {
    let mut bytes = serde_json::to_vec_pretty(report).map_err(|e| err(e.to_string()))?;
    bytes.push(b'\n');
    fs::write(path(out, options), bytes).map_err(|e| io("write kernel build report", e))
}

pub(crate) fn skipped(
    out: &Path,
    lock_path: &Path,
    options: BuildOptions,
    context: Option<&BuildContext>,
) -> Result<(), BuildError> {
    save(
        out,
        options,
        &BuildReport {
            schema: SCHEMA.into(),
            context: context.cloned(),
            source_membership_complete: false,
            source_ownership_complete: false,
            entry_declarations_complete: false,
            lock_status: LockStatus::Skipped,
            lock_path: lock_path.into(),
            lock_sha256: None,
            lock_families: vec![],
            compiler_path: PathBuf::new(),
            files: BTreeMap::new(),
            cubins: vec![],
        },
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "report the existing build inputs without changing identities"
)]
pub(crate) fn write(
    toolchain: &Toolchain,
    specs: &[CubinSpec],
    built: &[BuiltCubin],
    identities: &[CubinIdentity],
    inputs: &[Vec<ClosureInput>],
    out: &Path,
    lock: LockEvidence<'_>,
    options: BuildOptions,
    context: Option<&BuildContext>,
    declarations: Option<&BTreeMap<String, Vec<EntryDeclaration>>>,
) -> Result<(), BuildError> {
    let mut roots = Vec::new();
    if let Some(context) = context {
        for source in &context.sources {
            roots.push((
                &source.id,
                fs::canonicalize(&source.path).map_err(|e| io("resolve report source root", e))?,
            ));
        }
        roots.sort_by_key(|(_, path)| std::cmp::Reverse(path.components().count()));
    }
    let (lock, lock_sha256, lock_families) = match lock {
        LockEvidence::File(path) | LockEvidence::Bypassed(path) => (
            path,
            fs::read(path).ok().map(|bytes| digest(&bytes)),
            vec![],
        ),
        LockEvidence::Families(dir, families) => (dir, None, families),
    };
    let mut report = BuildReport {
        schema: SCHEMA.into(),
        context: context.cloned(),
        source_membership_complete: !built.is_empty(),
        source_ownership_complete: context.is_some(),
        entry_declarations_complete: declarations.is_some() && !built.is_empty(),
        lock_status: if options.profile_line_info {
            LockStatus::Profiling
        } else if options.lock_update {
            LockStatus::Updated
        } else {
            LockStatus::Verified
        },
        lock_path: fs::canonicalize(lock).unwrap_or_else(|_| lock.into()),
        lock_sha256,
        lock_families,
        compiler_path: toolchain.tool("nvcc"),
        files: BTreeMap::new(),
        cubins: vec![],
    };
    for ((cubin, identity), inputs) in built.iter().zip(identities).zip(inputs) {
        let spec = specs
            .iter()
            .find(|s| s.name == cubin.name)
            .expect("built spec");
        let mut membership = Vec::new();
        let mut generated_sha256 = BTreeMap::new();
        for input in inputs {
            let source_identity = if let Some(relative) = input.key.strip_prefix("generated/") {
                generated_sha256.insert(relative.into(), input.sha256.clone());
                SourceIdentity {
                    kind: SourceKind::Generated,
                    owner: None,
                    path: relative.into(),
                }
            } else if let Some((id, root)) =
                roots.iter().find(|(_, root)| input.path.starts_with(root))
            {
                SourceIdentity {
                    kind: SourceKind::Project,
                    owner: Some((*id).clone()),
                    path: input
                        .path
                        .strip_prefix(root)
                        .expect("matched root")
                        .to_string_lossy()
                        .into(),
                }
            } else if let Some(relative) = input.key.strip_prefix("toolkit/") {
                SourceIdentity {
                    kind: SourceKind::Toolkit,
                    owner: None,
                    path: relative.into(),
                }
            } else if let Some(relative) = input.key.strip_prefix("source/") {
                report.source_ownership_complete = false;
                SourceIdentity {
                    kind: SourceKind::Project,
                    owner: None,
                    path: relative.into(),
                }
            } else {
                SourceIdentity {
                    kind: SourceKind::System,
                    owner: None,
                    path: input.key.clone(),
                }
            };
            let file = SourceFile {
                identity: source_identity,
                sha256: input.sha256.clone(),
                bytes: input.bytes,
                location: input.path.clone(),
            };
            let id = file_id(&file)?;
            report.files.insert(id.clone(), file);
            membership.push(Membership {
                closure_key: input.key.clone(),
                file: id,
            });
        }
        let recipe = BuildRecipe {
            toolchain_version: identity.toolchain_version.clone(),
            nvcc_sha256: identity.nvcc_sha256.clone(),
            flags: identity.flags.clone(),
            defines: spec.defines.clone(),
            headers: spec.headers.clone(),
            include_dirs: spec.include_dirs.clone(),
            generated_sha256,
            source_closure_sha256: identity.source_closure_sha256.clone(),
        };
        let recipe_sha256 = digest(&serde_json::to_vec(&recipe).map_err(|e| err(e.to_string()))?);
        report.cubins.push(ReportedCubin {
            name: cubin.name.clone(),
            path: cubin.path.clone(),
            identity_path: cubin.identity_path.clone(),
            identity: serde_json::to_value(identity).map_err(|e| err(e.to_string()))?,
            recipe,
            recipe_sha256,
            membership,
            declarations: declarations
                .and_then(|d| d.get(&cubin.name))
                .cloned()
                .unwrap_or_default(),
        });
    }
    save(out, options, &report)
}
