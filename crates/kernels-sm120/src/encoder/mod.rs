//! Encoder kernel catalogs generated from encoder declarations
//! (`encoder-spec`). `catalog` returns the entries one declaration runs,
//! with symbols derived from (variant id, parameters) (`symbol`), and
//! `cubin_specs` builds a model's two encoder cubins from the union of its
//! declarations' entries, each including only the headers its entries'
//! variants need. A new variant contributes entries only to declarations
//! that use it, so no other model's entries, cubins or kernels.lock records
//! change.
//!
//! The shared variants' kernels are listed in `lower`. A model's registered
//! variants answer through `Registered`, which the model implements beside
//! its provider; shared code never names them.

mod attention;
pub mod kernel;
mod lower;
pub mod stability;
pub mod symbol;

use std::path::PathBuf;

use encoder_spec::components::rotary::RotaryComponent;
use encoder_spec::components::{Attention, Boundary, Component};
use encoder_spec::{Declaration, Error};
use kernel_build::{CubinSpec, EntryParam};
use kernel_core::contract::{
    Argument, ArgumentList, InstantiationSpec, LaunchPlan, MultimodalQuery, Refusal,
};

/// Rows of the language model's prefill one embedding override launch
/// covers.
pub const OVERRIDE_ROWS: u32 = 8192;

/// The cubin of a model's row and GEMM entries.
pub const ROWS_GEMM_CUBIN: &str = "multimodal_rows_gemm_sm120";
/// The cubin of a model's contiguous attention entries.
pub const ATTENTION_CUBIN: &str = "multimodal_attention_sm120";

/// A kernel a variant runs, before its symbol is derived.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Kernel {
    /// The id of the variant that runs it.
    pub variant: String,
    /// Which of the variant's kernels, for a variant that runs several.
    pub role: Option<&'static str>,
    pub query: MultimodalQuery,
}

impl Kernel {
    pub fn new(
        variant: impl Into<String>,
        role: Option<&'static str>,
        query: MultimodalQuery,
    ) -> Self {
        Self {
            variant: variant.into(),
            role,
            query,
        }
    }

    /// The entry: the derived symbol and the query's fixed ABI.
    pub fn entry(&self) -> Result<Entry, String> {
        let entry = Entry {
            symbol: symbol::derive(&self.variant, self.role, self.query)?,
            query: self.query,
            params: kernel::params(self.query),
        };
        entry
            .contract()
            .map_err(|refusal| format!("{}: {}", entry.symbol, refusal.0))?;
        Ok(entry)
    }
}

/// Where a kernel runs in a declaration: its stage (the stage count for the
/// output layout), the part of the stage it runs, and for a per-block kernel
/// its block. Parts: `linear`, `position`, `norm`, `pad`, `gather` (a
/// space-to-depth merge), `cast`, `layout`; a registered
/// stage's kernel's role, else `registered`;
/// a conv1d's `unfold`, `linear` and `gelu_rows`; a transformer's `norm`
/// (both of its norms), `qkv`, `qk_norm`, `split`, `rotary`, `attention`
/// (per block), `proj`, `residual_add`, `reorder`, `up`, `swiglu`, `down`
/// and `carry`; a projector's `norm`, `gelu`, `up`, `swiglu` and `down`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Slot {
    pub stage: usize,
    pub part: &'static str,
    pub block: Option<u32>,
}

/// One generated catalog entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub symbol: String,
    pub query: MultimodalQuery,
    pub params: Vec<EntryParam>,
}

impl Entry {
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        kernel::contract(&self.symbol, self.query)
    }

    /// The prepared row capacity (tokens for attention, M for GEMM).
    pub fn capacity(&self) -> u32 {
        kernel::capacity(self.query)
    }

    pub fn plan(&self, live_rows: u32) -> Result<LaunchPlan, Refusal> {
        kernel::plan(&self.symbol, self.query, live_rows)
    }

    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        kernel::pack_arguments(&self.params, values)
    }

    /// Dynamic shared memory to configure at load time, if any.
    pub fn dynamic_shared_bytes(&self) -> Option<u32> {
        kernel::dynamic_shared_bytes(self.query)
    }

    pub fn is_attention(&self) -> bool {
        kernel::is_attention(self.query)
    }

    /// The symbol without its identity digest (`symbol`).
    pub fn stem(&self) -> String {
        match self.symbol.rsplitn(3, '_').collect::<Vec<_>>()[..] {
            [version, _digest, head] => format!("{head}_{version}"),
            _ => self.symbol.clone(),
        }
    }
}

/// The kernels of a model's registered variants. The model implements this
/// beside the provider that registers them.
pub trait Registered {
    /// The kernels of registered stage `component` reading `input`, planned
    /// for `capacity` rows.
    fn stage(
        &self,
        component: &dyn Component,
        input: &Boundary,
        capacity: u32,
    ) -> Result<Vec<Kernel>, String>;

    /// The kernels registered rotary variant `rotary` runs on `attention`'s
    /// q and k.
    fn rotary(
        &self,
        rotary: &dyn RotaryComponent,
        attention: &Attention,
        capacity: u32,
    ) -> Result<Vec<Kernel>, String>;
}

/// No registered variants: a declaration of shared variants only.
pub struct Shared;

impl Registered for Shared {
    fn stage(
        &self,
        component: &dyn Component,
        _: &Boundary,
        _: u32,
    ) -> Result<Vec<Kernel>, String> {
        Err(format!("`{}` is not a shared variant", component.id()))
    }

    fn rotary(
        &self,
        rotary: &dyn RotaryComponent,
        _: &Attention,
        _: u32,
    ) -> Result<Vec<Kernel>, String> {
        Err(format!("`{}` is not a shared variant", rotary.id()))
    }
}

/// Every kernel `declaration` runs with the slot it fills and the variant
/// that runs it, in stage order: a kernel two slots run appears once per
/// slot, an attention kernel once per block. Every kernel is planned for the
/// declaration's `capacity_rows`, except the embedding override, which runs
/// in the language model's prefill.
pub fn kernels(
    declaration: &Declaration,
    registered: &dyn Registered,
) -> Result<Vec<(Slot, Kernel)>, Error> {
    let fail = |error: String| Error(format!("{}: {error}", declaration.name));
    let inputs = declaration.inputs()?;
    let mut lowering = lower::Lowering {
        precision: declaration.precision,
        capacity: declaration.input.capacity_rows,
        registered,
        kernels: Vec::new(),
        slot: Slot::default(),
    };
    for (index, (stage, input)) in declaration.stages.iter().zip(&inputs).enumerate() {
        lowering.slot.stage = index;
        lowering
            .stage(&stage.op, input)
            .map_err(|error| fail(format!("stages[{index}] ({}): {error}", stage.op.id())))?;
    }
    lowering.slot.stage = declaration.stages.len();
    lowering
        .layout(&declaration.output.op)
        .map_err(|error| fail(format!("output: {error}")))?;
    Ok(lowering.kernels)
}

/// Every entry `declaration` runs with the slot it fills (`kernels`).
pub fn slots(
    declaration: &Declaration,
    registered: &dyn Registered,
) -> Result<Vec<(Slot, Entry)>, Error> {
    let fail = |error: String| Error(format!("{}: {error}", declaration.name));
    kernels(declaration, registered)?
        .into_iter()
        .map(|(slot, kernel)| Ok((slot, kernel.entry().map_err(fail)?)))
        .collect()
}

/// The entries `declaration` runs, each once, in stage order (`slots`).
pub fn catalog(
    declaration: &Declaration,
    registered: &dyn Registered,
) -> Result<Vec<Entry>, Error> {
    union(
        slots(declaration, registered)?
            .into_iter()
            .map(|(_, entry)| entry),
    )
}

/// The entry of `entries` whose symbol reads `stem` (the symbol without its
/// identity digest), refusing a stem two entries share. For hand-written
/// launch sequences; generated lowering names entries by their symbols.
pub fn by_stem<'a>(entries: &'a [Entry], stem: &str) -> Result<&'a Entry, String> {
    let mut found = entries.iter().filter(|entry| entry.stem() == stem);
    match (found.next(), found.next()) {
        (Some(entry), None) => Ok(entry),
        (None, _) => Err(format!("no encoder entry {stem}")),
        (Some(_), Some(_)) => Err(format!("two encoder entries read {stem}")),
    }
}

/// The entries of several declarations (one model's), each symbol once, in
/// order of first appearance. One symbol never stands for two bodies.
/// Entries of one symbol that differ only in runtime values merge: the
/// entry grids for the largest capacity (every kernel reads its live rows
/// on the device), and keeps the first entry's eps, which launches pass as
/// an argument.
pub fn union(entries: impl IntoIterator<Item = Entry>) -> Result<Vec<Entry>, Error> {
    let mut union: Vec<Entry> = Vec::new();
    for entry in entries {
        let Some(known) = union.iter_mut().find(|known| known.symbol == entry.symbol) else {
            union.push(entry);
            continue;
        };
        let code = |entry: &Entry| {
            (
                kernel::header(entry.query),
                kernel::body(entry.query).ok(),
                kernel::threads(entry.query),
                entry.params.clone(),
            )
        };
        if code(known) != code(&entry) {
            return Err(Error(format!(
                "{} stands for {:?} and {:?}",
                entry.symbol, known.query, entry.query
            )));
        }
        if entry.capacity() > known.capacity() {
            known.query = kernel::with_capacity(known.query, entry.capacity());
        }
    }
    Ok(union)
}

/// A model's catalog: its declarations' entries.
pub fn model_catalog<'a>(
    declarations: impl IntoIterator<Item = &'a Declaration>,
    registered: &dyn Registered,
) -> Result<Vec<Entry>, Error> {
    let mut entries = Vec::new();
    for declaration in declarations {
        entries.extend(catalog(declaration, registered)?);
    }
    union(entries)
}

/// A model's two encoder cubins: rows and GEMMs, then contiguous attention
/// (a different engine).
pub fn cubin_specs(entries: &[Entry]) -> Result<[CubinSpec; 2], Refusal> {
    let pick = |attention: bool| {
        entries
            .iter()
            .filter(move |entry| entry.is_attention() == attention)
            .map(|entry| (entry.symbol.as_str(), &entry.query, entry.params.as_slice()))
    };
    Ok([
        kernel::cubin_spec(ROWS_GEMM_CUBIN, vec![], pick(false))?,
        kernel::cubin_spec(ATTENTION_CUBIN, vec![PathBuf::from(".")], pick(true))?,
    ])
}
