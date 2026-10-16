//! Model-owned construction lists and declarations, without operation enums.

use crate::{BuildError, BuildOptions, BuiltCubin, CubinSpec, LockDir, LockTarget, Toolchain, err};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::path::{Path, PathBuf};

/// Logical source ownership. The path is local lookup information and is
/// excluded from portable recipe and source identities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceRoot {
    pub id: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Consumer {
    /// Caller-supplied Cargo package identity, including its version.
    pub package: String,
    /// Stable source namespace, such as the SDK or model publisher's ID.
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildContext {
    pub consumer: Consumer,
    pub selection: String,
    pub sources: Vec<SourceRoot>,
    /// Report-only tuning provenance by entry symbol. This is not qualification.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tuning: BTreeMap<String, String>,
}

impl BuildContext {
    pub fn new(
        package: impl Into<String>,
        source: impl Into<String>,
        selection: impl Into<String>,
    ) -> Self {
        Self {
            consumer: Consumer {
                package: package.into(),
                source: source.into(),
            },
            selection: selection.into(),
            sources: vec![],
            tuning: BTreeMap::new(),
        }
    }

    pub fn source_root(mut self, id: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        self.sources.push(SourceRoot {
            id: id.into(),
            path: path.into(),
        });
        self
    }

    /// Add the owning selection and operation to a shared builder's refusal.
    pub fn require<T, E: Display>(
        &self,
        operation: &str,
        result: Result<T, E>,
    ) -> Result<T, BuildError> {
        result.map_err(|e| err(format!("selection {} / {operation}: {e}", self.selection)))
    }

    pub(crate) fn validate(&self) -> Result<(), BuildError> {
        if self.consumer.package.is_empty()
            || self.consumer.source.is_empty()
            || self.selection.is_empty()
        {
            return Err(err(
                "build ownership requires package, source and selection identities",
            ));
        }
        let mut ids = BTreeSet::new();
        let mut paths = BTreeSet::new();
        for source in &self.sources {
            if source.id.is_empty() || !ids.insert(&source.id) {
                return Err(err("source-root identities must be nonempty and unique"));
            }
            let path = std::fs::canonicalize(&source.path)
                .map_err(|e| crate::io("resolve ownership source root", e))?;
            if !path.is_dir() || !paths.insert(path) {
                return Err(err("ownership source roots must be distinct directories"));
            }
        }
        Ok(())
    }
}

/// A declaration describes an entry, not a deployment qualification. A shared
/// builder derives its fields from a validated typed requirement. A native
/// model-local entry declares its own contract and correctness case; the build
/// driver cannot prove arbitrary CUDA arithmetic from this metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryDeclaration {
    pub symbol: String,
    pub operation: String,
    pub contract: String,
    pub implementation: String,
    pub requirements: BTreeMap<String, String>,
    pub correctness_case: Option<String>,
}

pub struct SelectedCubin {
    spec: CubinSpec,
    declarations: Vec<EntryDeclaration>,
}

impl SelectedCubin {
    /// New selections use explicit versioned cubin names.
    pub fn new(spec: CubinSpec, declarations: Vec<EntryDeclaration>) -> Result<Self, BuildError> {
        if !crate::versioned(&spec.name) {
            return Err(err(format!(
                "new cubin {} needs a versioned name",
                spec.name
            )));
        }
        Self::existing(spec, declarations)
    }

    /// Attach declarations without renaming an existing locked cubin.
    pub fn existing(
        spec: CubinSpec,
        declarations: Vec<EntryDeclaration>,
    ) -> Result<Self, BuildError> {
        let symbols: BTreeSet<_> = spec.entries.iter().map(|e| e.symbol.as_str()).collect();
        let mut declared = BTreeSet::new();
        for declaration in &declarations {
            if !symbols.contains(declaration.symbol.as_str())
                || !declared.insert(declaration.symbol.as_str())
            {
                return Err(err(format!(
                    "{}: duplicate or unknown declared entry {}",
                    spec.name, declaration.symbol
                )));
            }
            if declaration.operation.is_empty()
                || declaration.contract.is_empty()
                || declaration.implementation.is_empty()
                || declaration.requirements.is_empty()
                || declaration
                    .correctness_case
                    .as_ref()
                    .is_some_and(String::is_empty)
            {
                return Err(err(format!(
                    "{}: incomplete declaration for {}",
                    spec.name, declaration.symbol
                )));
            }
        }
        if declared != symbols {
            return Err(err(format!(
                "{}: every entry needs exactly one declaration",
                spec.name
            )));
        }
        Ok(Self { spec, declarations })
    }
}

/// A bundle's complete construction list. No central model registration or
/// execution program is needed to compose shared and model-local entries.
pub struct KernelSelection {
    context: BuildContext,
    specs: Vec<CubinSpec>,
    declarations: BTreeMap<String, Vec<EntryDeclaration>>,
}

impl KernelSelection {
    pub fn new(context: BuildContext, cubins: Vec<SelectedCubin>) -> Result<Self, BuildError> {
        context.validate()?;
        if cubins.is_empty() {
            return Err(err("kernel selection must contain at least one cubin"));
        }
        let mut specs = Vec::new();
        let mut declarations = BTreeMap::new();
        for cubin in cubins {
            declarations.insert(cubin.spec.name.clone(), cubin.declarations);
            specs.push(cubin.spec);
        }
        context.require("construction", crate::validate(&specs))?;
        Ok(Self {
            context,
            specs,
            declarations,
        })
    }

    pub fn context(&self) -> &BuildContext {
        &self.context
    }

    pub fn specs(&self) -> &[CubinSpec] {
        &self.specs
    }

    pub fn declarations(&self) -> &BTreeMap<String, Vec<EntryDeclaration>> {
        &self.declarations
    }

    /// Portable identity of selected entry wrappers and their complete typed
    /// declarations. Source paths and consumer ownership are not arithmetic.
    /// A native module records this only after `audited_contract_identity`.
    pub fn contract_identity(&self) -> Result<String, BuildError> {
        let mut records = BTreeMap::new();
        for spec in &self.specs {
            let unit = crate::generate(spec)?;
            let mut declarations = self.declarations[&spec.name].clone();
            declarations.sort_by(|a, b| a.symbol.cmp(&b.symbol));
            records.insert(
                &spec.name,
                (unit.translation_unit, unit.header, declarations),
            );
        }
        let bytes = serde_json::to_vec(&("selected-implementation-contract-v1", records))
            .map_err(|e| err(format!("encode selected implementation contract: {e}")))?;
        Ok(crate::digest(&bytes))
    }

    /// Require exact audited release coverage before publishing the contract
    /// in a native consumption note, alongside `audited_dependency_identity`.
    pub fn audited_contract_identity(&self, built: &[BuiltCubin]) -> Result<String, BuildError> {
        if built.len() != self.specs.len() {
            return Err(err("audited implementation set differs from its selection"));
        }
        let mut seen = BTreeSet::new();
        for cubin in built {
            let spec = self
                .specs
                .iter()
                .find(|s| s.name == cubin.name)
                .ok_or_else(|| err("audited cubin is outside its selection"))?;
            if !seen.insert(&cubin.name)
                || cubin.line_info
                || cubin.entries.len() != spec.entries.len()
            {
                return Err(err(
                    "implementation contract requires each audited release cubin once",
                ));
            }
            let mut entries = BTreeSet::new();
            for entry in &cubin.entries {
                let expected = spec
                    .entries
                    .iter()
                    .find(|e| e.symbol == entry.symbol)
                    .ok_or_else(|| err("audited entry is outside its selection"))?;
                if !entries.insert(&entry.symbol)
                    || entry.max_threads != expected.max_threads
                    || entry.min_blocks != expected.min_blocks
                    || entry.local_bytes > expected.allowed_local_bytes
                {
                    return Err(err(
                        "audited entry differs from its selected resource contract",
                    ));
                }
            }
        }
        self.contract_identity()
    }

    /// Source and compiler inputs remain separate from the operation contract.
    /// Embed both identities in the native note so dependency-only changes
    /// invalidate prepared execution without forcing compatible weight repacks.
    pub fn audited_dependency_identity(&self, built: &[BuiltCubin]) -> Result<String, BuildError> {
        let contract = self.audited_contract_identity(built)?;
        let mut records = BTreeMap::new();
        for cubin in built {
            let sidecar = std::fs::read(&cubin.identity_path)
                .map_err(|e| crate::io("read audited implementation identity", e))?;
            records.insert(
                &cubin.name,
                (
                    &cubin.sha256,
                    &cubin.source_closure_sha256,
                    &cubin.entries,
                    crate::digest(&sidecar),
                ),
            );
        }
        let bytes = serde_json::to_vec(&(contract, records))
            .map_err(|e| err(format!("encode audited implementation dependencies: {e}")))?;
        Ok(crate::digest(&bytes))
    }

    pub fn build(
        &self,
        toolchain: &Toolchain,
        out_dir: &Path,
        lock_path: &Path,
        options: BuildOptions,
    ) -> Result<Vec<BuiltCubin>, BuildError> {
        self.context.require(
            "build",
            crate::build_declared(
                toolchain,
                &self.specs,
                out_dir,
                LockTarget::File(lock_path),
                options,
                Some(&self.context),
                Some(&self.declarations),
            ),
        )
    }

    /// Build against a directory-form lock; see [`LockDir`].
    pub fn build_in_families(
        &self,
        toolchain: &Toolchain,
        out_dir: &Path,
        lock: &LockDir,
        options: BuildOptions,
    ) -> Result<Vec<BuiltCubin>, BuildError> {
        self.context.require(
            "build",
            crate::build_declared(
                toolchain,
                &self.specs,
                out_dir,
                LockTarget::Families(lock),
                options,
                Some(&self.context),
                Some(&self.declarations),
            ),
        )
    }
}
