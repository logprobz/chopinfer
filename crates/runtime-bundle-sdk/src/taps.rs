//! Per-layer taps for numeric bring-up. With `CHOPINFER_TAPS_DIR` set at
//! create, a bundle runs every phase eagerly and, after the stages that
//! carry taps, copies the live rows of the tapped buffers to
//! `<dir>/step<NNNN>/<name>.r<rank>.bin`. Each step directory holds:
//!
//! - `capture.json`: the model (whose `models/<model>/taps.json` declares
//!   its taps), the program, rows and every declared tap and rank
//!   (`expected`, with its `sharding` and `compare` kind), written before
//!   the first copy;
//!   With a producer (`Taps::with_producer`, the serving process's startup
//!   nonce from the create record) it also holds `producer.startup_nonce`;
//! - `items.json`: the model's row map (each item's engine request ID,
//!   positions, tokens);
//! - `index.jsonl`: `{"name", "rank", "first_row", "rows", "cols", "dtype"}`
//!   per written tap, in capture order;
//! - `completion.json`: `full` once every declared tap was written.
//!
//! An encoder tap ([`EncoderRows`], such as `enc.vision.out`) holds the
//! output rows of the step's encodes of one tower instead of phase rows:
//! its index entry has `"axis": "encoder"` and items.json `encoder` maps
//! its rows to each request's media rows (`{"request", "start", "kind",
//! "item", "index", "rows"}`: rows `index..` of the request's occurrence
//! `item` of media `kind`, counted per kind in prompt order). It is read
//! before the phase and written first.
//!
//! Tap names follow the reference runner's grammar (`tools/refrun/taps.py`):
//! `<scope>.<site>`, scope `embed`, `L<nn>`, `final`, `enc.<tower>`,
//! `enc.<tower>.B<nn>` or `mtp.L<n>`, and a canonical site or a
//! model-specific `x.<...>` one.
//! `tools/qual/tapdiff.py` reads this format against the runner's taps,
//! reducing rank shards as each entry's `sharding` states. A model declares
//! its taps once in `models/<m>/taps.json`.
//!
//! The environment:
//!
//! - `CHOPINFER_TAPS_LAYERS`: layers whose detail (per-op) taps are kept,
//!   comma separated or `all`; other taps are kept in every layer. Other
//!   text is refused, and a bundle refuses layers without detail taps
//!   ([`Taps::check_layers`]);
//! - `CHOPINFER_TAPS_SITES`: comma separated patterns (`*` matches any
//!   text) of a tap's site or whole name, such as `out,x.hc.*,L03.attn.out`;
//!   only taps one matches are kept. An empty list is refused, and a bundle
//!   refuses a pattern that matches none of its taps, or only detail taps of
//!   layers `CHOPINFER_TAPS_LAYERS` does not keep ([`Taps::check_sites`]);
//! - `CHOPINFER_TAPS_FROM=P`: keep only rows at position P (a number from 0)
//!   or later, as the contiguous span from the first to the last such row;
//! - `CHOPINFER_TAPS_BYTES`: the tensor byte budget of one step, from 1
//!   (2 GiB, at most [`diagnostic::MAX_BYTES`]);
//! - `CHOPINFER_TAPS_SCOPE_JSON`: a comparison scope recorded in
//!   `capture.json`, validated at create (see [`diagnostic::validate_scope`]).
//!
//! A model may keep older names as aliases ([`TapEnv`]); the generic name
//! wins when both are set. [`Taps::from_env`] refuses a value it cannot read,
//! naming the variable, so a bundle refuses it at create. The completion inventory is not a content seal:
//! an offline assembler binds a step to its reference.
//!
//! Capturing synchronizes the streams after every tapped stage. It is a
//! debugging mode, never a serving one: a bundle captures only when built
//! as a diagnostic bundle (a cargo feature such as `diagnostic-taps`), and a
//! serving bundle refuses to create while [`Taps::requested`] names a
//! variable. A model that serves through device graphs must also add an
//! eager, synchronized capture path that replays its stages one at a time,
//! since a graph launch cannot be tapped between stages.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use runtime_bundle_abi::{
    BundleAbiError, BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1,
};
use serde_json::{Value, json};

use crate::diagnostic;

/// The schema of a step's `capture.json`.
pub const SCHEMA: &str = "chopinfer.taps.v1";

/// The default tensor byte budget of one step.
pub const DEFAULT_BUDGET: u64 = 2 << 30;

/// The environment variables that configure taps.
#[derive(Clone, Copy, Debug)]
pub struct TapEnv {
    pub dir: &'static str,
    pub layers: &'static str,
    pub sites: &'static str,
    pub from: &'static str,
    pub bytes: &'static str,
    pub scope_json: &'static str,
}

/// The tag of the create record's optional last field: the startup nonce of
/// the serving process, which a host sends only when the generic taps
/// directory (`TAPS_ENV.dir`) is set, so a diagnostic bundle's steps name the
/// process that wrote them (`Taps::with_producer`). Other bundles ignore it.
pub const PRODUCER_TAG: &[u8] = b"chop.tap-producer.v1";

/// The generic names.
pub const TAPS_ENV: TapEnv = TapEnv {
    dir: "CHOPINFER_TAPS_DIR",
    layers: "CHOPINFER_TAPS_LAYERS",
    sites: "CHOPINFER_TAPS_SITES",
    from: "CHOPINFER_TAPS_FROM",
    bytes: "CHOPINFER_TAPS_BYTES",
    scope_json: "CHOPINFER_TAPS_SCOPE_JSON",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TapDType {
    Bf16,
    F32,
    U32,
}

impl TapDType {
    pub const fn bytes(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 | Self::U32 => 4,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::U32 => "u32",
        }
    }
}

/// How the ranks hold a tap, as `taps.json` declares it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TapSharding {
    /// Every rank holds the same value; rank 0 is written.
    Replicated,
    /// Every rank computes the same value; each rank's copy is written.
    RankCopies,
    /// The value is the rank-ordered F32 sum of the ranks' partials.
    RankPartial,
    /// Each rank holds its share of every `[heads, dim]` block of a row, the
    /// blocks' heads split evenly over the ranks; no blocks when the rank's
    /// layout is its own (padded or packed), which tapdiff does not reduce.
    ColShards(&'static [[u32; 2]]),
    /// Each rank holds the rows' slots of the experts it owns.
    ExpertPartial,
}

impl TapSharding {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Replicated | Self::RankCopies => "replicated",
            Self::RankPartial => "rank_partial",
            Self::ColShards(_) => "col_shards",
            Self::ExpertPartial => "expert_partial",
        }
    }

    /// The `sharding` field of an expected entry, as `taps.json` writes it.
    pub fn json(self) -> Value {
        match self {
            Self::ColShards(blocks) if !blocks.is_empty() => {
                json!({"col_shards": {"axis": "heads", "blocks": blocks}})
            }
            _ => json!(self.name()),
        }
    }

    /// The ranks written of `ranks`.
    pub const fn ranks(self, ranks: u32) -> u32 {
        match self {
            Self::Replicated => 1,
            _ => ranks,
        }
    }

    /// The `combine` field of an expected entry.
    const fn combine(self) -> &'static str {
        match self {
            Self::Replicated => "replicated-rank0",
            Self::RankPartial => "rank-ordered-f32-sum",
            _ => "model-shards",
        }
    }
}

/// How tapdiff compares a tap with the reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TapCompare {
    /// Relative L2 per row.
    Dense,
    /// Top-k ids, compared as sets.
    Set,
    /// Weights aligned by the ids of the sibling `.ids` tap.
    Paired,
    /// Integer codes, compared one by one (an RVQ quantizer's
    /// `[rows, stages]` codes).
    Codes,
    /// Reported, not gated.
    Discrete,
}

impl TapCompare {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Set => "set",
            Self::Paired => "paired",
            Self::Codes => "codes",
            Self::Discrete => "discrete",
        }
    }
}

/// One tap: its name, the layer of the stage that writes it, whether it is
/// a detail tap (kept only in `_LAYERS`), its columns per rank and dtype,
/// its sharding and comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TapSpec {
    pub name: String,
    pub layer: Option<u32>,
    pub detail: bool,
    pub cols: u32,
    pub dtype: TapDType,
    pub sharding: TapSharding,
    pub compare: TapCompare,
}

impl TapSpec {
    /// Rows `first..first + rows` of `buffer`, clamped to it.
    pub fn span(&self, buffer: DeviceBufferSpanV1, first: u32, rows: u32) -> DeviceBufferSpanV1 {
        let row = u64::from(self.cols) * self.dtype.bytes();
        let start = (u64::from(first) * row).min(buffer.length);
        let bytes = (u64::from(rows) * row).min(buffer.length - start);
        DeviceBufferSpanV1::new(buffer.buffer, buffer.offset + start, bytes)
    }
}

/// The layers whose detail taps are kept.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TapLayers {
    All,
    List(Vec<u32>),
}

/// The taps and rows to keep.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TapFilter {
    /// Layers of the detail taps; none when unset.
    pub layers: Option<TapLayers>,
    /// Patterns of the sites or names of the taps to keep; all when unset.
    pub sites: Option<Vec<String>>,
    /// The first position whose rows are kept; all when unset.
    pub from: Option<i32>,
}

/// The configured taps of one bundle instance, and its step counter.
pub struct Taps {
    /// The variables the sites and layers filters come from.
    sites_variable: Option<&'static str>,
    layers_variable: Option<&'static str>,
    /// The byte budget of one step; the default when unset.
    budget: Option<u64>,
    /// The comparison scope each step records.
    scope: Option<Value>,
    model: Option<&'static str>,
    /// The serving process's startup nonce each step records.
    producer: Option<String>,
    root: PathBuf,
    filter: TapFilter,
    step: u64,
}

impl Taps {
    pub fn new(root: PathBuf, filter: TapFilter) -> Self {
        Self {
            sites_variable: None,
            layers_variable: None,
            budget: None,
            scope: None,
            model: None,
            producer: None,
            root,
            filter,
            step: 0,
        }
    }

    /// The variable that asks for taps (the directory), if one is set.
    pub fn requested(aliases: Option<TapEnv>) -> Option<&'static str> {
        variable(aliases, |env| env.dir)
    }

    /// The taps the environment configures, if any; refuses detail layers
    /// that are not `all` or a list of layer numbers, an empty sites list,
    /// and a first position or byte budget that is not a number it can use.
    pub fn from_env(aliases: Option<TapEnv>) -> Result<Option<Self>, String> {
        let Some(root) = var_os(aliases, |env| env.dir) else {
            return Ok(None);
        };
        let (layers, layers_variable) = parsed(
            aliases,
            |env| env.layers,
            parse_layers,
            "is not all or comma-separated layer numbers",
        )?;
        let (sites, sites_variable) = parsed(
            aliases,
            |env| env.sites,
            |text| {
                let sites = text
                    .split(',')
                    .map(|site| site.trim().to_owned())
                    .filter(|site| !site.is_empty())
                    .collect::<Vec<_>>();
                (!sites.is_empty()).then_some(sites)
            },
            "is empty",
        )?;
        let (from, _) = parsed(
            aliases,
            |env| env.from,
            |text| text.trim().parse().ok().filter(|&from: &i32| from >= 0),
            "is not a position",
        )?;
        let (budget, _) = parsed(
            aliases,
            |env| env.bytes,
            |text| text.trim().parse().ok().filter(|&bytes: &u64| bytes > 0),
            "is not a byte count from 1",
        )?;
        let mut taps = Self::new(
            PathBuf::from(root),
            TapFilter {
                layers,
                sites,
                from,
            },
        );
        taps.sites_variable = sites_variable;
        taps.layers_variable = layers_variable;
        taps.budget = budget;
        taps.scope = scope(aliases)?;
        Ok(Some(taps))
    }

    /// The taps, writing `model` into each step's `capture.json`, so readers
    /// find its declaration (`models/<model>/taps.json`).
    pub fn with_model(mut self, model: &'static str) -> Self {
        self.model = Some(model);
        self
    }

    /// Records `startup_nonce`, the serving process's (the create record's
    /// `PRODUCER_TAG` field), in every step's capture.json, beside each
    /// item's engine request ID in items.json: a step then names the process
    /// and the request that wrote it.
    pub fn with_producer(mut self, startup_nonce: Option<String>) -> Self {
        self.producer = startup_nonce;
        self
    }

    /// Refuses detail layers in which `specs`, a model's taps, have no
    /// detail tap, naming the variable and those layers.
    pub fn check_layers(&self, specs: &[TapSpec]) -> Result<(), String> {
        let Some(TapLayers::List(listed)) = &self.filter.layers else {
            return Ok(());
        };
        let unknown = listed
            .iter()
            .filter(|&&layer| {
                !specs
                    .iter()
                    .any(|spec| spec.detail && spec.layer == Some(layer))
            })
            .map(u32::to_string)
            .collect::<Vec<_>>();
        if unknown.is_empty() {
            return Ok(());
        }
        Err(format!(
            "{} names layers without taps: {}",
            self.layers_variable(),
            unknown.join(", ")
        ))
    }

    /// The variable the layers filter came from, else the generic one.
    pub fn layers_variable(&self) -> &'static str {
        self.layers_variable.unwrap_or(TAPS_ENV.layers)
    }

    /// The variable the sites filter came from: the generic one, or the
    /// alias a model may read in its own older terms.
    pub const fn sites_variable(&self) -> Option<&'static str> {
        self.sites_variable
    }

    pub fn sites(&self) -> Option<&[String]> {
        self.filter.sites.as_deref()
    }

    /// The taps with `sites` as their filter.
    pub fn with_sites(mut self, sites: Vec<String>) -> Self {
        self.filter.sites = Some(sites);
        self
    }

    /// Refuses a sites filter with patterns that match none of `specs`, a
    /// model's taps, or only detail taps of layers the layers filter does
    /// not keep, naming its variable and those patterns.
    pub fn check_sites(&self, specs: &[TapSpec]) -> Result<(), String> {
        let variable = self.sites_variable.unwrap_or(TAPS_ENV.sites);
        let patterns = self.sites().unwrap_or_default();
        let refused = |keep: &dyn Fn(&TapSpec) -> bool| {
            patterns
                .iter()
                .filter(|pattern| {
                    !specs
                        .iter()
                        .any(|spec| keep(spec) && matches(pattern, &spec.name))
                })
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        };
        let unmatched = refused(&|_| true);
        if !unmatched.is_empty() {
            return Err(format!("{variable} selects no tap: {unmatched}"));
        }
        let unkept = refused(&|spec| self.kept(spec));
        if !unkept.is_empty() {
            return Err(format!(
                "{variable} selects only detail taps of layers {} does not keep: {unkept}",
                self.layers_variable()
            ));
        }
        Ok(())
    }

    /// Whether the layers filter keeps `spec`: every tap but the detail
    /// taps of layers it does not name.
    pub fn kept(&self, spec: &TapSpec) -> bool {
        !spec.detail
            || spec.layer.is_some_and(|layer| match &self.filter.layers {
                Some(TapLayers::All) => true,
                Some(TapLayers::List(layers)) => layers.contains(&layer),
                None => false,
            })
    }

    pub const fn from(&self) -> Option<i32> {
        self.filter.from
    }

    /// Whether `spec` is kept.
    pub fn selected(&self, spec: &TapSpec) -> bool {
        self.kept(spec)
            && self
                .filter
                .sites
                .as_ref()
                .is_none_or(|sites| sites.iter().any(|pattern| matches(pattern, &spec.name)))
    }

    /// The rows `[first, end)` of a phase of `rows` rows to copy: all, or
    /// the span of the live rows at `from` or later (empty when none are).
    pub fn rows(&self, rows: u32, positions: &[i32]) -> (u32, u32) {
        let Some(from) = self.filter.from else {
            return (0, rows);
        };
        let live = positions.len().min(rows as usize);
        let mut kept = (0..live).filter(|&row| positions[row] >= from);
        let Some(first) = kept.next() else {
            return (0, 0);
        };
        let last = kept.next_back().unwrap_or(first);
        (first as u32, last as u32 + 1)
    }

    /// Starts the next phase's directory.
    pub fn next_step(&mut self) -> std::io::Result<PathBuf> {
        let dir = self.root.join(format!("step{:04}", self.step));
        self.step += 1;
        fs::create_dir_all(&self.root)?;
        fs::create_dir(&dir)?;
        Ok(dir)
    }

    /// Begins a step of `program` over `rows` rows on `ranks` ranks; the
    /// caller declares its taps and then opens it.
    pub fn begin(
        &self,
        dir: &Path,
        program: &str,
        ranks: u32,
        rows: u32,
        positions: &[i32],
    ) -> std::io::Result<TapStep> {
        let (first, end) = self.rows(rows, positions);
        Ok(TapStep {
            dir: dir.into(),
            program: program.into(),
            ranks,
            rows,
            first,
            end,
            budget: self
                .budget
                .unwrap_or(DEFAULT_BUDGET)
                .min(diagnostic::MAX_BYTES),
            model: self.model,
            producer: self.producer.clone(),
            step: self.step,
            scope: self.scope.clone(),
            expected: Vec::new(),
            encoders: Vec::new(),
            parts: Default::default(),
            bytes: 0,
            index: None,
            observed: 0,
        })
    }
}

/// A model's tap declaration (`models/<m>/taps.json`): entries in program
/// order whose `name` is a tap name or a pattern in which `{nn}` stands for
/// a two-digit layer, limited to `layers` when given. The first entry that
/// matches a tap declares it.
pub struct Declaration {
    entries: Vec<Value>,
}

impl Declaration {
    pub fn parse(text: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(text).map_err(|error| error.to_string())?;
        let entries = value["taps"]
            .as_array()
            .cloned()
            .ok_or("a tap declaration lists its taps")?;
        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[Value] {
        &self.entries
    }

    /// The index of the entry that declares tap `name`, and the layer its
    /// `{nn}` stands for.
    pub fn entry(&self, name: &str) -> Option<(usize, Option<u32>)> {
        self.entries.iter().enumerate().find_map(|(index, entry)| {
            let layer = pattern_layer(entry["name"].as_str()?, name)?;
            let layers = entry["layers"].as_array();
            if layers
                .is_some_and(|layers| !layer.is_some_and(|layer| layers.contains(&layer.into())))
            {
                return None;
            }
            Some((index, layer))
        })
    }

    /// Refuses `spec`, a tap a bundle writes at `ranks` ranks, unless its
    /// entry declares its dtype, detail, sharding, comparison, ranks and
    /// columns (the full width for head shards); returns the entry's index.
    pub fn check(&self, spec: &TapSpec, ranks: u32) -> Result<usize, String> {
        let name = &spec.name;
        let (index, _) = self
            .entry(name)
            .ok_or_else(|| format!("{name} is not declared"))?;
        let entry = &self.entries[index];
        let every = entry["ranks"] == "every";
        let cols = entry["cols"].as_u64().map(|cols| {
            if entry["sharding"].is_object() {
                cols / u64::from(ranks)
            } else {
                cols
            }
        });
        let differs = [
            ("dtype", entry["dtype"] != spec.dtype.name()),
            (
                "detail",
                entry["detail"].as_bool().unwrap_or(false) != spec.detail,
            ),
            ("sharding", entry["sharding"] != spec.sharding.json()),
            ("compare", entry["compare"] != spec.compare.name()),
            ("ranks", every != (spec.sharding == TapSharding::RankCopies)),
            (
                "cols",
                match cols {
                    Some(cols) => cols != u64::from(spec.cols),
                    None => entry["sharding"] != "col_shards",
                },
            ),
        ];
        match differs.iter().find(|(_, differs)| *differs) {
            Some((field, _)) => Err(format!("{name} at {ranks} ranks: its {field} differs")),
            None => Ok(index),
        }
    }
}

/// The layer `{nn}` stands for in `name`, `Some(None)` when `pattern` has
/// none, or `None` when `name` does not match it.
fn pattern_layer(pattern: &str, name: &str) -> Option<Option<u32>> {
    let Some((head, tail)) = pattern.split_once("{nn}") else {
        return (pattern == name).then_some(None);
    };
    let digits = name.strip_prefix(head)?.strip_suffix(tail)?;
    (digits.len() == 2 && digits.bytes().all(|b| b.is_ascii_digit())).then(|| digits.parse().ok())
}

/// The site of a tap name: the name after its scope (`embed`, `L<nn>`,
/// `final`, `enc.<tower>`, `enc.<tower>.B<nn>` or `mtp.L<n>`).
pub fn site(name: &str) -> &str {
    let mut parts = name.split('.');
    let scope = match parts.next() {
        Some("enc") => {
            let block = parts.nth(1).is_some_and(|part| {
                part.len() > 1
                    && part.starts_with('B')
                    && part[1..].bytes().all(|b| b.is_ascii_digit())
            });
            if block { 3 } else { 2 }
        }
        Some("mtp") => 2,
        _ => 1,
    };
    name.splitn(scope + 1, '.').nth(scope).unwrap_or("")
}

/// Whether `pattern` matches the site or the whole of tap name `name`.
fn matches(pattern: &str, name: &str) -> bool {
    glob(pattern, site(name)) || glob(pattern, name)
}

/// Whether `text` matches `pattern`, where `*` matches any text.
fn glob(pattern: &str, text: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == text,
        Some((head, tail)) => {
            text.starts_with(head)
                && (head.len()..=text.len())
                    .any(|start| text.is_char_boundary(start) && glob(tail, &text[start..]))
        }
    }
}

/// Detail layers: `all` or comma separated layer numbers; none when the
/// text is neither.
fn parse_layers(text: &str) -> Option<TapLayers> {
    if text.trim() == "all" {
        return Some(TapLayers::All);
    }
    text.split(',')
        .map(|layer| layer.trim().parse().ok())
        .collect::<Option<Vec<u32>>>()
        .map(TapLayers::List)
}

/// The value of the set variable `name` (the generic one, else its alias)
/// and its name; refuses text `parse` cannot read, as `refusal` says.
fn parsed<T>(
    aliases: Option<TapEnv>,
    name: fn(&TapEnv) -> &'static str,
    parse: impl Fn(&str) -> Option<T>,
    refusal: &str,
) -> Result<(Option<T>, Option<&'static str>), String> {
    let Some(variable) = variable(aliases, name) else {
        return Ok((None, None));
    };
    let text = std::env::var_os(variable).unwrap_or_default();
    let value = text
        .to_str()
        .and_then(parse)
        .ok_or_else(|| format!("{variable} {refusal}: {}", text.to_string_lossy()))?;
    Ok((Some(value), Some(variable)))
}

/// The comparison scope the set scope variable holds, validated; refuses
/// text that is not one, naming the variable.
fn scope(aliases: Option<TapEnv>) -> Result<Option<Value>, String> {
    let Some(variable) = variable(aliases, |env| env.scope_json) else {
        return Ok(None);
    };
    let text = std::env::var_os(variable).unwrap_or_default();
    let text = text
        .to_str()
        .ok_or_else(|| format!("{variable} is not text"))?;
    if text.len() > 16 << 20 {
        return Err(format!("{variable} exceeds the 16 MiB metadata budget"));
    }
    let scope = serde_json::from_str(text)
        .map_err(|error| error.to_string())
        .and_then(|scope| diagnostic::validate_scope(&scope).map(|()| scope))
        .map_err(|error| format!("{variable} is not a comparison scope: {error}"))?;
    Ok(Some(scope))
}

/// The name of the set variable `name`: the generic one, else its alias.
fn variable(aliases: Option<TapEnv>, name: fn(&TapEnv) -> &'static str) -> Option<&'static str> {
    [Some(TAPS_ENV), aliases]
        .into_iter()
        .flatten()
        .map(|env| name(&env))
        .find(|name| std::env::var_os(name).is_some())
}

/// The generic variable `name`, else its alias.
fn var_os(aliases: Option<TapEnv>, name: fn(&TapEnv) -> &'static str) -> Option<OsString> {
    std::env::var_os(name(&TAPS_ENV))
        .or_else(|| aliases.and_then(|env| std::env::var_os(name(&env))))
}

/// A refused capture: a host service, or the files.
#[derive(Debug)]
pub enum TapError {
    Host(BundleAbiError),
    Io(std::io::Error),
}

impl From<std::io::Error> for TapError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// The rows `start..start + rows` of an encoder tap: rows `index..` of
/// occurrence `item` of media `kind` in `request`, counted in prompt order
/// per kind (none when the step cannot count them).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncoderSegment {
    pub request: Option<u64>,
    pub kind: Option<&'static str>,
    pub item: u32,
    pub start: u32,
    pub index: u32,
    pub rows: u32,
}

/// An encoder tap of one step: the rows of the step's encodes of one tower
/// that its requests inject, once per inject, read before the phase runs. Its rows are encoder rows, not phase rows
/// (`"axis": "encoder"` in the index), mapped to requests by items.json
/// `encoder` segments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncoderRows {
    pub spec: TapSpec,
    pub bytes: Vec<u8>,
    pub segments: Vec<EncoderSegment>,
}

impl EncoderRows {
    pub fn rows(&self) -> u32 {
        self.segments.iter().map(|segment| segment.rows).sum()
    }
}

/// One fresh eager phase. Until it is opened it collects the declared taps;
/// the index stays open until completion.
pub struct TapStep {
    dir: PathBuf,
    program: String,
    ranks: u32,
    rows: u32,
    first: u32,
    end: u32,
    budget: u64,
    model: Option<&'static str>,
    producer: Option<String>,
    step: u64,
    scope: Option<Value>,
    expected: Vec<Value>,
    /// The encoder taps, declared first and written when the step opens.
    encoders: Vec<EncoderRows>,
    /// Per tap and rank captured in parts, the phase rows covered so far
    /// and the kept rows among them.
    parts: std::collections::BTreeMap<(String, u32), (u32, Vec<u8>)>,
    bytes: u64,
    index: Option<File>,
    observed: usize,
}

impl TapStep {
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The rows `[first, end)` this step copies.
    pub const fn rows(&self) -> (u32, u32) {
        (self.first, self.end)
    }

    /// Declares `spec`'s ranks, in capture order.
    pub fn declare(&mut self, spec: &TapSpec) -> std::io::Result<()> {
        self.declare_rows(spec, self.first, self.end - self.first)
            .map(drop)
    }

    /// Declares an encoder tap, rank 0 only, ahead of the phase's taps; it is
    /// written when the step opens.
    pub fn encoder(&mut self, tap: EncoderRows) -> std::io::Result<()> {
        if self.expected.len() != self.encoders.len() {
            return Err(std::io::Error::other(
                "encoder taps precede the phase's taps",
            ));
        }
        if tap.spec.sharding != TapSharding::Replicated {
            return Err(std::io::Error::other("encoder taps are replicated"));
        }
        let entry = self.declare_rows(&tap.spec, 0, tap.rows())?;
        entry["axis"] = "encoder".into();
        self.encoders.push(tap);
        Ok(())
    }

    fn declare_rows(
        &mut self,
        spec: &TapSpec,
        first: u32,
        rows: u32,
    ) -> std::io::Result<&mut Value> {
        let start = self.expected.len();
        for rank in 0..spec.sharding.ranks(self.ranks) {
            let length = u64::from(rows) * u64::from(spec.cols) * spec.dtype.bytes();
            self.bytes = self
                .bytes
                .checked_add(length)
                .ok_or_else(|| std::io::Error::other("dump byte budget overflow"))?;
            self.expected
                .push(json!({"name":spec.name,"rank":rank,"first_row":first,
                "rows":rows,"cols":spec.cols,"dtype":spec.dtype.name(),"bytes":length,
                "shape":[rows,spec.cols],"ordinal":self.expected.len(),
                "combine":spec.sharding.combine(),"sharding":spec.sharding.json(),
                "compare":spec.compare.name()}));
        }
        self.expected
            .get_mut(start)
            .ok_or_else(|| std::io::Error::other("tap has no rank"))
    }

    /// Writes `capture.json` and `items.json` (with the encoder taps'
    /// segments) and opens the index, then writes the encoder taps; refuses
    /// a step whose declared taps exceed the budget.
    pub fn open(&mut self, items: &Value) -> std::io::Result<()> {
        let budget = self.budget;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(std::io::Error::other)?;
        let mut value = json!({"schema":SCHEMA,"model":self.model,"ranks":self.ranks,
            "capture_id":format!("{}-{}-{}",std::process::id(),self.step,now.as_nanos()),
            "program":self.program,"rows":self.rows,"first_row":self.first,"captured_rows":self.end-self.first,
            "comparison_scope":self.scope,"expected":self.expected,"tensor_bytes":self.bytes,"budget_bytes":budget,"completion":"pending",
            "execution":"eager-synchronized","limitations":["Cannot clear graph-only failures","Not ordinary-serving performance evidence"],
            "state_origin":"requires parent initialization/state binding"});
        if let Some(nonce) = &self.producer {
            value["producer"] = json!({"startup_nonce": nonce});
        }
        diagnostic::write_new(&self.dir.join("capture.json"), &value)?;
        if self.bytes > budget || budget == 0 {
            diagnostic::write_new(
                &self.dir.join("completion.json"),
                &json!({"completion":"incomplete","reason":"tensor budget exhausted before capture"}),
            )?;
            return Err(std::io::Error::other("dump tensor budget exhausted"));
        }
        if self.encoders.is_empty() {
            diagnostic::write_new(&self.dir.join("items.json"), items)?;
        } else {
            let mut items = items.clone();
            items["encoder"] = self
                .encoders
                .iter()
                .map(|tap| {
                    let segments = tap.segments.iter().map(|segment| {
                        json!({"request":segment.request,"start":segment.start,
                            "kind":segment.kind,"item":segment.item,
                            "index":segment.index,"rows":segment.rows})
                    });
                    (tap.spec.name.clone(), segments.collect())
                })
                .collect::<serde_json::Map<_, _>>()
                .into();
            diagnostic::write_new(&self.dir.join("items.json"), &items)?;
        }
        self.index = Some(File::create_new(self.dir.join("index.jsonl"))?);
        for tap in std::mem::take(&mut self.encoders) {
            self.write(&tap.spec, 0, &tap.bytes)?;
        }
        Ok(())
    }

    /// Copies the step's rows of `buffer`, `spec`'s buffer on `rank`, after
    /// the work queued on `stream`; nothing when the step keeps no rows.
    pub fn capture(
        &mut self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        spec: &TapSpec,
        rank: u32,
        buffer: DeviceBufferSpanV1,
    ) -> Result<(), TapError> {
        let (first, end) = (self.first, self.end);
        if end == first {
            return Ok(());
        }
        services
            .synchronize_stream(stream)
            .map_err(TapError::Host)?;
        let span = spec.span(buffer, first, end - first);
        if span.length != u64::from(end - first) * u64::from(spec.cols) * spec.dtype.bytes() {
            return Err(std::io::Error::other("tap buffer is shorter than its rows").into());
        }
        let mut bytes = vec![0_u8; span.length as usize];
        services
            .copy_device_to_host(stream, &mut bytes, span)
            .map_err(TapError::Host)?;
        services
            .synchronize_stream(stream)
            .map_err(TapError::Host)?;
        Ok(self.write(spec, rank, &bytes)?)
    }

    /// Copies the kept rows of a part of the phase (such as one half of a
    /// pipelined prefill): `rows` phase rows from row `start`, which
    /// `buffer`, `spec`'s buffer on `rank`, holds from its first row, after
    /// the work queued on `stream`. Parts arrive in row order; the tap is
    /// written once they cover the phase.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_part(
        &mut self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        spec: &TapSpec,
        rank: u32,
        buffer: DeviceBufferSpanV1,
        start: u32,
        rows: u32,
    ) -> Result<(), TapError> {
        self.add_part(spec, rank, start, rows, |first, count| {
            services
                .synchronize_stream(stream)
                .map_err(TapError::Host)?;
            let span = spec.span(buffer, first, count);
            if span.length != u64::from(count) * u64::from(spec.cols) * spec.dtype.bytes() {
                return Err(std::io::Error::other("tap buffer is shorter than its rows").into());
            }
            let mut bytes = vec![0_u8; span.length as usize];
            services
                .copy_device_to_host(stream, &mut bytes, span)
                .map_err(TapError::Host)?;
            services
                .synchronize_stream(stream)
                .map_err(TapError::Host)?;
            Ok(bytes)
        })
    }

    /// Adds the kept rows of a part, `read(first, count)` reading `count` of
    /// its rows from its row `first`.
    fn add_part(
        &mut self,
        spec: &TapSpec,
        rank: u32,
        start: u32,
        rows: u32,
        read: impl FnOnce(u32, u32) -> Result<Vec<u8>, TapError>,
    ) -> Result<(), TapError> {
        let (first, end) = (self.first, self.end);
        if end == first {
            return Ok(());
        }
        let key = (spec.name.clone(), rank);
        let (covered, mut bytes) = self.parts.remove(&key).unwrap_or_default();
        if covered != start || start + rows > self.rows {
            return Err(std::io::Error::other("tap parts do not follow each other").into());
        }
        let (from, to) = (first.max(start), end.min(start + rows));
        if from < to {
            bytes.extend(read(from - start, to - from)?);
        }
        if start + rows < self.rows {
            self.parts.insert(key, (start + rows, bytes));
            return Ok(());
        }
        Ok(self.write(spec, rank, &bytes)?)
    }

    /// Writes one tap's rows of one rank, which must be the next declared
    /// entry.
    pub fn write(&mut self, spec: &TapSpec, rank: u32, bytes: &[u8]) -> std::io::Result<()> {
        let index = self
            .index
            .as_mut()
            .ok_or_else(|| std::io::Error::other("tap step is not open"))?;
        let expected = self
            .expected
            .get(self.observed)
            .ok_or_else(|| std::io::Error::other("unexpected dump entry"))?;
        let encoder = expected.get("axis").is_some();
        let (first, rows) = if encoder {
            (0, expected["rows"].as_u64().unwrap_or_default() as u32)
        } else {
            (self.first, self.end - self.first)
        };
        if expected["name"] != spec.name
            || expected["rank"] != rank
            || expected["first_row"] != first
            || expected["cols"] != spec.cols
            || expected["dtype"] != spec.dtype.name()
            || expected["rows"] != rows
            || expected["bytes"].as_u64() != Some(bytes.len() as u64)
        {
            return Err(std::io::Error::other(
                "dump entry shape/rank/row/span differs from declared scope",
            ));
        }
        let mut file = File::create_new(self.dir.join(format!("{}.r{rank}.bin", spec.name)))?;
        file.write_all(bytes)?;
        file.sync_all()?;
        let mut line = json!({
            "name": spec.name,
            "rank": rank,
            "first_row": first,
            "rows": rows,
            "cols": spec.cols,
            "dtype": spec.dtype.name(),
        });
        if encoder {
            line["axis"] = "encoder".into();
        }
        writeln!(index, "{line}")?;
        self.observed += 1;
        Ok(())
    }

    /// Removes a cancelled transaction's capture so its retry has no duplicate rows.
    pub fn cancel(mut self) -> std::io::Result<()> {
        self.index.take();
        fs::remove_dir_all(&self.dir)
    }

    /// Writes `completion.json`: `full` when every declared entry was
    /// written.
    pub fn finish(mut self) -> std::io::Result<()> {
        if let Some(index) = &mut self.index {
            index.flush()?;
            index.sync_all()?;
        }
        diagnostic::write_new(
            &self.dir.join("completion.json"),
            &json!({
                "completion": if self.observed == self.expected.len() && self.observed > 0 {"full"} else {"incomplete"},
                "expected":self.expected.len(),"observed":self.observed,
                "reason":if self.observed == 0 {Some("no selected live tensors")} else {None}
            }),
        )
    }
}

impl Drop for TapStep {
    fn drop(&mut self) {
        if self.index.is_some() && !self.dir.join("completion.json").exists() {
            let _ = diagnostic::write_new(
                &self.dir.join("completion.json"),
                &json!({
                "completion":"incomplete","expected":self.expected.len(),"observed":self.observed,
                "reason":"capture ended before successful completion"}),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, layer: Option<u32>, detail: bool) -> TapSpec {
        TapSpec {
            name: name.into(),
            layer,
            detail,
            cols: 3,
            dtype: TapDType::Bf16,
            sharding: TapSharding::Replicated,
            compare: TapCompare::Dense,
        }
    }

    #[test]
    fn filters_keep_detail_layers_sites_and_late_rows() {
        let mut taps = Taps::new(
            PathBuf::new(),
            TapFilter {
                layers: Some(TapLayers::List(vec![3])),
                ..TapFilter::default()
            },
        );
        assert!(taps.selected(&spec("L04.out", Some(4), false)));
        assert!(taps.selected(&spec("L03.x.q", Some(3), true)));
        assert!(!taps.selected(&spec("L04.x.q", Some(4), true)));
        assert!(!taps.selected(&spec("embed", None, true)));
        taps.filter.layers = Some(TapLayers::All);
        assert!(taps.selected(&spec("L04.x.q", Some(4), true)));
        taps.filter.sites = Some(vec!["out".into(), "x.hc.*".into()]);
        assert!(!taps.selected(&spec("L04.x.q", Some(4), true)));
        assert!(taps.selected(&spec("L04.out", Some(4), false)));
        assert!(!taps.selected(&spec("L04.attn.out", Some(4), false)));
        assert!(taps.selected(&spec("L04.x.hc.attn", Some(4), false)));
        assert!(taps.selected(&spec("enc.vision.B03.out", None, false)));
        assert!(taps.selected(&spec("enc.vision.out", None, false)));
        assert!(!taps.selected(&spec("final.x.hc", None, false)));
        assert_eq!(site("mtp.L0.x.hc.mlp"), "x.hc.mlp");
        assert_eq!(site("enc.audio.out"), "out");
        assert_eq!(site("enc.vision.B03.x.attn"), "x.attn");
        // A whole name selects that tap alone.
        let taps = taps.with_sites(vec!["L03.attn.out".into()]);
        assert!(taps.selected(&spec("L03.attn.out", Some(3), false)));
        assert!(!taps.selected(&spec("L04.attn.out", Some(4), false)));
        let specs = [
            spec("L03.attn.out", Some(3), false),
            spec("L03.x.hc.attn", Some(3), false),
            spec("L04.x.q", Some(4), true),
        ];
        assert_eq!(taps.check_sites(&specs), Ok(()));
        let taps = taps.with_sites(vec!["x.hc.*".into(), "out".into(), "L09.*".into()]);
        assert_eq!(
            taps.check_sites(&specs),
            Err("CHOPINFER_TAPS_SITES selects no tap: out, L09.*".into())
        );
        // A detail tap's site selects only in the layers kept.
        let mut taps = taps.with_sites(vec!["x.q".into()]);
        taps.filter.layers = Some(TapLayers::List(vec![3]));
        assert_eq!(
            taps.check_sites(&specs),
            Err(
                "CHOPINFER_TAPS_SITES selects only detail taps of layers CHOPINFER_TAPS_LAYERS \
                 does not keep: x.q"
                    .into()
            )
        );
        taps.filter.layers = Some(TapLayers::List(vec![4]));
        assert_eq!(taps.check_sites(&specs), Ok(()));
        assert_eq!(parse_layers(" all"), Some(TapLayers::All));
        assert_eq!(parse_layers("0, 1,3"), Some(TapLayers::List(vec![0, 1, 3])));
        for text in ["0;1", "3,,4", "", "-1", "all,1"] {
            assert_eq!(parse_layers(text), None, "{text}");
        }
        taps.filter.layers = Some(TapLayers::List(vec![3, 4, 99]));
        assert_eq!(
            taps.check_layers(&specs),
            Err("CHOPINFER_TAPS_LAYERS names layers without taps: 3, 99".into())
        );
        taps.filter.layers = Some(TapLayers::All);
        assert_eq!(taps.check_layers(&[]), Ok(()));

        // Rows at position 5 or later: the span from the first to the last.
        let positions = [7, 8, 0, 1, 5, 2, 9];
        assert_eq!(taps.rows(8, &positions), (0, 8));
        taps.filter.from = Some(5);
        assert_eq!(taps.rows(8, &positions), (0, 7));
        assert_eq!(taps.rows(3, &positions), (0, 2));
        taps.filter.from = Some(10);
        assert_eq!(taps.rows(8, &positions), (0, 0));
    }

    #[test]
    fn declarations_name_taps_by_layer_pattern() {
        let declaration = Declaration::parse(
            r#"{"taps": [
                {"name": "L{nn}.attn.core", "dtype": "bf16", "cols": 8, "compare": "dense",
                 "sharding": {"col_shards": {"axis": "heads", "blocks": [[2, 4]]}}, "layers": [3]},
                {"name": "L{nn}.attn.core", "dtype": "bf16", "cols": 3, "compare": "dense",
                 "sharding": "replicated", "detail": true},
                {"name": "enc.vision.out", "dtype": "bf16", "cols": 3, "compare": "dense",
                 "sharding": "replicated"}]}"#,
        )
        .unwrap();
        assert_eq!(declaration.entry("L03.attn.core"), Some((0, Some(3))));
        assert_eq!(declaration.entry("L04.attn.core"), Some((1, Some(4))));
        assert_eq!(declaration.entry("L4.attn.core"), None);
        assert_eq!(declaration.entry("enc.vision.out"), Some((2, None)));
        let core = TapSpec {
            cols: 4,
            sharding: TapSharding::ColShards(&[[2, 4]]),
            ..spec("L03.attn.core", Some(3), false)
        };
        assert_eq!(declaration.check(&core, 2), Ok(0));
        assert_eq!(
            declaration.check(&core, 4),
            Err("L03.attn.core at 4 ranks: its cols differs".into())
        );
        assert_eq!(
            declaration.check(&spec("L05.attn.core", Some(5), true), 4),
            Ok(1)
        );
        assert_eq!(
            declaration.check(&spec("L05.attn.core", Some(5), false), 4),
            Err("L05.attn.core at 4 ranks: its detail differs".into())
        );
        assert_eq!(
            declaration.check(&spec("L05.attn.out", Some(5), false), 4),
            Err("L05.attn.out is not declared".into())
        );
    }

    #[test]
    fn cancelled_steps_remove_their_rows_and_completion() {
        let root = std::env::temp_dir().join(format!("sdk-taps-cancel-{}", std::process::id()));
        let mut taps = Taps::new(root.clone(), TapFilter::default());
        let dir = taps.next_step().unwrap();
        let mut step = taps.begin(&dir, "Prefill", 1, 1, &[0]).unwrap();
        let tap = spec("out", None, false);
        step.declare(&tap).unwrap();
        step.open(&json!({})).unwrap();
        step.write(&tap, 0, &[0; 6]).unwrap();
        assert!(!dir.join("completion.json").exists());
        step.cancel().unwrap();
        assert!(!dir.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn steps_name_the_producing_process() {
        let root = std::env::temp_dir().join(format!("sdk-taps-producer-{}", std::process::id()));
        let mut taps =
            Taps::new(root.clone(), TapFilter::default()).with_producer(Some("ab".repeat(16)));
        let dir = taps.next_step().unwrap();
        let mut step = taps.begin(&dir, "Prefill", 1, 2, &[0, 1]).unwrap();
        step.open(&json!({"rows": 2})).unwrap();
        let capture: Value =
            serde_json::from_slice(&fs::read(dir.join("capture.json")).unwrap()).unwrap();
        assert_eq!(
            capture["producer"],
            json!({"startup_nonce": "ab".repeat(16)})
        );
        drop(step);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn steps_write_declared_entries_in_order_and_record_completion() {
        let root = std::env::temp_dir().join(format!("sdk-taps-{}", std::process::id()));
        let filter = TapFilter {
            from: Some(1),
            ..TapFilter::default()
        };
        let mut taps = Taps::new(root.clone(), filter);
        let positions = [0, 1, 2, 0, 1];
        let tap = TapSpec {
            sharding: TapSharding::RankPartial,
            ..spec("L00.attn.out", Some(0), false)
        };
        let dir = taps.next_step().unwrap();
        let mut step = taps.begin(&dir, "Prefill", 2, 8, &positions).unwrap();
        assert_eq!(step.rows(), (1, 5));
        step.declare(&tap).unwrap();
        assert!(step.write(&tap, 0, &[0; 24]).is_err(), "not open");
        step.open(&json!({"rows": 8})).unwrap();
        let capture: Value =
            serde_json::from_slice(&fs::read(dir.join("capture.json")).unwrap()).unwrap();
        assert_eq!(capture["schema"], SCHEMA);
        assert!(capture.get("producer").is_none(), "no producer named");
        assert_eq!(capture["expected"][1]["rank"], 1);
        assert_eq!(capture["expected"][1]["combine"], "rank-ordered-f32-sum");
        assert_eq!(capture["expected"][1]["sharding"], "rank_partial");
        assert_eq!(capture["expected"][1]["compare"], "dense");
        assert_eq!(
            TapSharding::ColShards(&[[16, 128], [48, 128]]).json(),
            json!({"col_shards": {"axis": "heads", "blocks": [[16, 128], [48, 128]]}})
        );
        assert_eq!(TapSharding::ColShards(&[]).json(), "col_shards");
        assert_eq!(capture["tensor_bytes"], 48);
        assert!(step.write(&tap, 1, &[0; 24]).is_err(), "rank out of order");
        assert!(step.write(&tap, 0, &[0; 18]).is_err(), "short rows");
        step.write(&tap, 0, &[0; 24]).unwrap();
        step.write(&tap, 1, &[0; 24]).unwrap();
        assert!(step.write(&tap, 1, &[0; 24]).is_err(), "undeclared");
        step.finish().unwrap();
        let completion: Value =
            serde_json::from_slice(&fs::read(dir.join("completion.json")).unwrap()).unwrap();
        assert_eq!(completion["completion"], "full");
        let index = fs::read_to_string(dir.join("index.jsonl")).unwrap();
        assert_eq!(index.lines().count(), 2);

        // An open step dropped before finishing is incomplete.
        let dir = taps.next_step().unwrap();
        let mut step = taps.begin(&dir, "Prefill", 2, 8, &positions).unwrap();
        step.declare(&tap).unwrap();
        step.open(&json!({})).unwrap();
        drop(step);
        let completion: Value =
            serde_json::from_slice(&fs::read(dir.join("completion.json")).unwrap()).unwrap();
        assert_eq!(completion["completion"], "incomplete");

        // A declared scope over the budget is refused before any copy.
        let dir = taps.next_step().unwrap();
        let mut step = taps.begin(&dir, "Prefill", 2, 8, &positions).unwrap();
        step.budget = 40;
        step.declare(&tap).unwrap();
        assert!(step.open(&json!({})).is_err());
        assert!(!dir.join("index.jsonl").exists());

        // Encoder rows are declared first and written when the step opens,
        // with their own rows and segments.
        let dir = taps.next_step().unwrap();
        let mut step = taps.begin(&dir, "Prefill", 2, 8, &positions).unwrap();
        let encoder = EncoderRows {
            spec: spec("enc.vision.out", None, false),
            bytes: vec![7; 3 * 3 * 2],
            segments: vec![
                EncoderSegment {
                    request: Some(9),
                    kind: Some("image"),
                    item: 1,
                    start: 0,
                    index: 0,
                    rows: 2,
                },
                EncoderSegment {
                    request: None,
                    kind: Some("video"),
                    item: 0,
                    start: 2,
                    index: 0,
                    rows: 1,
                },
            ],
        };
        step.encoder(encoder.clone()).unwrap();
        step.declare(&tap).unwrap();
        let partial = TapSpec {
            sharding: TapSharding::RankPartial,
            ..encoder.spec.clone()
        };
        assert!(
            step.encoder(EncoderRows {
                spec: partial,
                ..encoder.clone()
            })
            .is_err(),
            "after the phase's taps"
        );
        step.open(&json!({"rows": 8})).unwrap();
        step.write(&tap, 0, &[0; 24]).unwrap();
        step.write(&tap, 1, &[0; 24]).unwrap();
        step.finish().unwrap();
        let items: Value =
            serde_json::from_slice(&fs::read(dir.join("items.json")).unwrap()).unwrap();
        assert_eq!(
            items["encoder"]["enc.vision.out"],
            json!([{"request": 9, "start": 0, "kind": "image", "item": 1, "index": 0, "rows": 2},
                {"request": null, "start": 2, "kind": "video", "item": 0, "index": 0, "rows": 1}])
        );
        let index = fs::read_to_string(dir.join("index.jsonl")).unwrap();
        let first: Value = serde_json::from_str(index.lines().next().unwrap()).unwrap();
        assert_eq!(
            first,
            json!({"name": "enc.vision.out", "rank": 0, "first_row": 0, "rows": 3,
                "cols": 3, "dtype": "bf16", "axis": "encoder"})
        );
        let capture: Value =
            serde_json::from_slice(&fs::read(dir.join("capture.json")).unwrap()).unwrap();
        assert_eq!(capture["expected"][0]["axis"], "encoder");
        assert_eq!(capture["expected"][1]["rows"], 4);
        assert_eq!(capture["tensor_bytes"], 18 + 48);
        assert_eq!(
            fs::read(dir.join("enc.vision.out.r0.bin")).unwrap(),
            [7; 18]
        );
        let completion: Value =
            serde_json::from_slice(&fs::read(dir.join("completion.json")).unwrap()).unwrap();
        assert_eq!(completion["completion"], "full");

        // Parts of the phase are kept until they cover it, then written.
        let dir = taps.next_step().unwrap();
        let mut step = taps.begin(&dir, "Prefill", 2, 5, &positions).unwrap();
        assert_eq!(step.rows(), (1, 5));
        step.declare(&tap).unwrap();
        step.open(&json!({})).unwrap();
        let row = |byte: u8, count: u32| Ok(vec![byte; count as usize * 6]);
        let part = |step: &mut TapStep, rank, byte, start, rows| {
            step.add_part(&tap, rank, start, rows, |first, count| {
                assert_eq!(first, if start == 0 { 1 } else { 0 });
                row(byte, count)
            })
        };
        assert!(part(&mut step, 0, 1, 2, 3).is_err(), "out of order");
        part(&mut step, 0, 1, 0, 2).unwrap();
        assert!(!dir.join("L00.attn.out.r0.bin").exists());
        part(&mut step, 0, 2, 2, 3).unwrap();
        // Row 1 of the first part, then rows 2..5 of the second.
        let written = fs::read(dir.join("L00.attn.out.r0.bin")).unwrap();
        assert_eq!(written, [[1; 6], [2; 6], [2; 6], [2; 6]].concat());
        part(&mut step, 1, 3, 0, 5).unwrap();
        step.finish().unwrap();
        let completion: Value =
            serde_json::from_slice(&fs::read(dir.join("completion.json")).unwrap()).unwrap();
        assert_eq!(completion["completion"], "full");

        taps.step = 0;
        assert!(taps.next_step().is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
