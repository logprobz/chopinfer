//! Declarative conformance runs of one lab entry, or of a chain of entries.
//!
//! A test declares a [`Case`]: the entry ([`LabEntry`]), its buffers by
//! parameter name with their dtype, rows and role, its scalars, the device
//! live count, and an oracle. A chain ([`Case::then`]), such as split
//! partials and their merge, runs as one launch. [`Case::run`] then
//! performs the checks every engine owes, so a new engine gets them by
//! declaring its buffers:
//!
//! - eager, relaunch, graph and PDL launches on poisoned buffers
//!   ([`Replay`]) at every live count ([`poison::live_counts`] by default;
//!   a custom list keeps 0, a partial count and the capacity), through one
//!   graph captured at the capacity;
//! - PDL phases behind a producer that holds the device live count at zero
//!   until its tail ([`LiveProducer`]) and, for a produced input, also
//!   behind one that rewrites that input ([`CopyProducer`]), with the
//!   live-count producer then holding the input too. These probe for reads
//!   before the PDL wait; they catch one only when it lands inside the
//!   producer's window;
//! - typed write maps ([`WriteMap::of`]) of the buffer's dtype that, under
//!   any rule, write nothing at live 0 and nothing past the live rows of a
//!   buffer of live rows, so those bytes must keep their poison or initial
//!   bytes;
//! - every output and state writes at every positive live count unless the
//!   case declares that count no-write with a reason, and a launch that
//!   writes nothing must fail the checks at each count where something
//!   writes;
//! - an oracle, when declared, covers every output and state that writes,
//!   whole, at every live count;
//! - ID inputs that index a pool rerun at the largest live count with
//!   their IDs and the pools' rows moved to the top of the pool, and must
//!   match the first run bit for bit, with the pools moved back;
//! - bounded scratch ([`Replay::scratch_within`]);
//! - ID inputs that admit holes rerun with a row the entry reads all -1,
//!   against the oracle.
//!
//! A rerun counts only when it can fail a launch that ignores what it
//! changes, judged on exact results alone (`Case::exact_results`): the
//! bytes the checks accept for an output or state whose oracle result is
//! [`Expected::Bits`], or that writes nothing. Required write maps shape
//! them; a state's update ranges, which only permit writes, do not; and a
//! numeric reference never counts, since two tolerance gates can accept one
//! result. The high-id rerun's results must differ from those of the old
//! IDs over the moved pools, the stale read a launch that ignores the moved
//! IDs makes; the hole rerun's from the first run's. A case that cannot
//! show this declares no pools or [`Holes::Never`], and the evidence lists
//! that rerun as not exercised.
//!
//! [`Case::validate`] checks the declaration on the CPU, including the
//! maps and oracle results of both reruns; `run` calls it first.
//!
//! A case's launches come from its stage source: lab entries bound by
//! parameter name, or a prepared op's launches over views of the case's
//! buffers (`crate::kit`), whose device handles name the buffers. Every
//! check above runs alike over either.

use crate::lab_entry::LabEntry;
use crate::poison::{self, CopyProducer, LiveProducer, POISON, Replay, Rewrites, WriteMap};
use crate::{LabResult, Session, check_output, launch};
use kernel_build::EntryParam;
use kernel_core::backend::Launch;
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan};
use kernel_core::oracle::OracleOutput;
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

/// Dynamic shared memory a function takes without opting in.
pub(crate) const DEFAULT_SHARED_BYTES: u32 = 48 * 1024;

/// The rows of a buffer: the case's capacity, or a fixed count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    Live,
    Fixed(usize),
}

/// Host images of a case's inputs, ID inputs and in-place states, by name,
/// as the launch under check sees them.
#[derive(Clone, Debug, Default)]
pub struct Images(BTreeMap<String, Vec<u8>>);

impl Images {
    pub fn bytes(&self, name: &str) -> LabResult<&[u8]> {
        self.0
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("no image named {name}"))
    }

    pub fn i32s(&self, name: &str) -> LabResult<Vec<i32>> {
        Ok(self
            .bytes(name)?
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes(word.try_into().expect("4 bytes")))
            .collect())
    }

    pub fn u16s(&self, name: &str) -> LabResult<Vec<u16>> {
        Ok(self
            .bytes(name)?
            .chunks_exact(2)
            .map(|half| u16::from_le_bytes([half[0], half[1]]))
            .collect())
    }

    /// These images with `name` holding `bytes`.
    pub fn with(&self, name: &str, bytes: Vec<u8>) -> Images {
        let mut images = self.clone();
        images.0.insert(name.into(), bytes);
        images
    }

    pub fn f32s(&self, name: &str) -> LabResult<Vec<f32>> {
        Ok(self
            .bytes(name)?
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("4 bytes")))
            .collect())
    }
}

/// A function of the images and the live count.
pub type Rule<'a, T> = Box<dyn Fn(&Images, u32) -> LabResult<T> + 'a>;

/// The elements a launch must write in an output or in-place state.
pub enum Write<'a> {
    /// Every element of the rows below the live count.
    LiveRows,
    /// Every element at a positive live count, none at live 0: for a
    /// buffer of fixed rows written whole whenever any row is live.
    All,
    /// A map from the images and the live count.
    Map(Rule<'a, WriteMap>),
    /// A map from the images and the live count that may write at live 0
    /// too: metadata a launch writes at every live count, zero included,
    /// such as the route counts and offsets an expert op reads when no
    /// token is live. The oracle must give its live-0 values, which the
    /// run compares like any other count's, so a stray write still fails.
    EveryCount(Rule<'a, WriteMap>),
}

/// An oracle's result for one output or state.
pub enum Expected {
    /// The whole buffer's bits. Only bytes in an output's write map are
    /// compared (the map check covers the rest); a state compares every
    /// byte.
    Bits(Vec<u8>),
    /// One value per element of a BF16, F32 or FP8 E4M3 buffer. The
    /// elements in an output's write map, or every element of a state, are
    /// compared at their own indices. An F32 element passes within the
    /// oracle's 2^-22 K S of the reference ([`check_output`]). A BF16 or
    /// E4M3 element passes when it is a value correct rounding gives some
    /// F32 within that bound of the reference: from the rounding of the
    /// interval's lowest F32 to that of its highest, ties to even and for
    /// E4M3 saturating at 448 ([`admitted_range`]). With no bound that is
    /// the rounding of the reference's nearest F32 alone. E4M3 codes decode to
    /// values in the code's units (no tensor scale), the units of the
    /// oracle's values for such a buffer.
    /// Tolerance gates can overlap, so these never show that a high-id or
    /// hole rerun reads what it changes; only bits do.
    Values(OracleOutput),
    /// The oracle's own judge of the whole buffer's bytes, for buffers no
    /// uniform bound covers, such as records that mix exact fields with
    /// tolerances. The write map and poison checks still apply; like
    /// values, a judge never tells two runs apart.
    Judge(Judge),
}

/// A judge of a buffer's bytes ([`Expected::Judge`]).
pub type Judge = Box<dyn Fn(&[u8]) -> LabResult<()>>;

/// The row of an ID input that reruns with every ID a -1 hole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holes {
    /// No hole rerun.
    Never,
    /// The last live row at the largest live count, for an input of live
    /// rows ([`Rows::Live`]).
    LastLive,
    /// This row, for an input whose rows are not the live rows, such as a
    /// per-item page table: one the entry reads at the largest live count.
    Row(usize),
}

enum Role<'a> {
    Input,
    /// I32 IDs, -1 for a hole, indexing rows of `stride` bytes of each pool.
    Ids {
        pools: Vec<(String, usize)>,
        holes: Holes,
    },
    Output {
        write: Write<'a>,
        pattern: Vec<u8>,
    },
    /// Bytes a launch may write.
    Scratch(Rule<'a, Vec<Range<usize>>>),
    State {
        write_only: Write<'a>,
        updated: Rule<'a, Vec<Range<usize>>>,
    },
}

struct Buffer<'a> {
    name: String,
    dtype: DType,
    rows: Rows,
    row_elements: usize,
    role: Role<'a>,
    /// The initial bytes of an input, ID input or state.
    image: Vec<u8>,
}

impl Buffer<'_> {
    fn rows(&self, capacity: u32) -> usize {
        match self.rows {
            Rows::Live => capacity as usize,
            Rows::Fixed(rows) => rows,
        }
    }

    fn elements(&self, capacity: u32) -> usize {
        self.rows(capacity) * self.row_elements
    }

    fn bytes(&self, capacity: u32) -> LabResult<usize> {
        Ok(self.elements(capacity) * WriteMap::of(self.dtype).bytes()?)
    }

    fn has_image(&self) -> bool {
        matches!(
            self.role,
            Role::Input | Role::Ids { .. } | Role::State { .. }
        )
    }

    fn uploaded(&self) -> bool {
        matches!(self.role, Role::Input | Role::Ids { .. })
    }

    fn checked(&self) -> bool {
        matches!(self.role, Role::Output { .. } | Role::State { .. })
    }
}

/// What a buffer holds before a launch ([`Case::fills`]).
pub(crate) enum Fill {
    Image(Vec<u8>),
    /// A pattern repeated over the buffer.
    Pattern(Vec<u8>),
}

/// An oracle over the images and the live count.
type Oracle<'a> = Rule<'a, Vec<(String, Expected)>>;

/// Whole-buffer bytes of outputs and states, by name.
pub(crate) type Results = BTreeMap<String, Vec<u8>>;

/// The producer of one PDL phase ([`Case::pdl_producers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PdlProducer {
    /// [`CopyProducer`] rewriting the produced input.
    Copy,
    /// [`LiveProducer`] holding the device live count unless it is a step
    /// input (`count`), and the produced input when there is one, until its
    /// tail.
    Live { produced: bool, count: bool },
}

impl PdlProducer {
    fn describe(self) -> &'static str {
        match self {
            PdlProducer::Copy => "PDL copy producer",
            PdlProducer::Live {
                produced: false, ..
            } => "PDL live-count producer",
            PdlProducer::Live {
                produced: true,
                count: true,
            } => "PDL live-count and input producer",
            PdlProducer::Live {
                produced: true,
                count: false,
            } => "PDL input producer",
        }
    }
}

/// Whether a conformance run's `error` comes from one of its PDL phases,
/// rather than from a phase before them or after.
pub(crate) fn pdl_phase_error(error: &str) -> bool {
    [
        PdlProducer::Copy,
        PdlProducer::Live {
            produced: false,
            count: true,
        },
        PdlProducer::Live {
            produced: true,
            count: true,
        },
        PdlProducer::Live {
            produced: true,
            count: false,
        },
    ]
    .iter()
    .any(|producer| error.contains(&format!(": {}: ", producer.describe())))
}

/// The PDL producer probes a conformance run launches ahead of the stages
/// under check ([`Case::pdl_producers`]): the live-count producer and, for
/// a produced input, the copy producer. They are kernels of the device's
/// architecture: [`Probes::sm120`], or another backend's own kernels through
/// `LiveProducer::with_function` and `CopyProducer::with_function`. A run
/// without them skips its PDL phases.
pub struct Probes {
    pub live: LiveProducer,
    pub copy: Option<CopyProducer>,
}

impl Probes {
    /// SM120's probes: the live-count producer of `probe_pdl_prefetch_sm120`
    /// and, given a module `lab` that holds it, the copy producer
    /// ([`poison::SM120_COPY_PRODUCER`]).
    pub fn sm120(session: &Session, lab: Option<&Module>) -> LabResult<Self> {
        Ok(Self {
            live: LiveProducer::new(session)?,
            copy: lab.map(CopyProducer::new).transpose()?,
        })
    }

    /// SM120's probes from the session's cubins ([`crate::Cubins`]): the
    /// copy producer from the cubin that holds it, if any.
    pub fn of_session(session: &Session) -> LabResult<Self> {
        let copy = match session.cubins().holding(poison::SM120_COPY_PRODUCER) {
            Some(name) => Some(session.module(name)?),
            None => None,
        };
        Self::sm120(session, copy.as_ref())
    }
}

/// A prepared op's launches at a row count, over views whose handles the
/// first argument gives by buffer name.
pub(crate) type Plan<'a> = Box<dyn Fn(&dyn Fn(&str) -> u64, u32) -> LabResult<Vec<Launch>> + 'a>;

/// Where a case's launches come from.
enum Source<'a> {
    /// Lab entries launched in order, one or a chain, each planned at the
    /// row count and bound by parameter name.
    Entries(Vec<&'a dyn LabEntry>),
    /// A prepared op's launches, which may launch any of `symbols`.
    Prepared {
        symbols: Vec<String>,
        plan: Plan<'a>,
    },
}

/// One launch of a case at a row count: its symbol, plan and packed
/// arguments, and the buffer each device handle in the arguments names.
struct Stage {
    symbol: String,
    plan: LaunchPlan,
    arguments: ArgumentList,
    names: BTreeMap<u64, String>,
}

/// A stage loaded for a run: its symbol, function, plan, arguments and the
/// device buffers its handles bind.
type Loaded<'d> = (
    String,
    Function,
    LaunchPlan,
    ArgumentList,
    launch::Bindings<'d>,
);

/// One entry's declared conformance run, a chain of entries run as one
/// launch ([`Case::then`]), or a prepared op's launches.
pub struct Case<'a> {
    source: Source<'a>,
    /// The stages plan at each live count's rows, not at the capacity
    /// ([`Case::host_rows`]).
    host_rows: bool,
    capacity: u32,
    live: String,
    live_counts: Vec<u32>,
    buffers: Vec<Buffer<'a>>,
    scalars: BTreeMap<String, Argument>,
    produced: Option<String>,
    oracle: Option<Oracle<'a>>,
    /// (output or state, live count) pairs that write nothing, with the
    /// reason.
    no_write: BTreeMap<(String, u32), String>,
    /// Stages the chain computes its results without, with the reason
    /// ([`Case::redundant`]).
    redundant: BTreeMap<String, String>,
    /// A producer's output to check before an omission test runs its
    /// consumers. Poisoned indexing metadata is not a safe kernel input.
    omission_guards: BTreeMap<String, String>,
    /// Why the device live count is a step input a launch may read before
    /// its PDL wait ([`Case::live_count_before_wait`]), if it is.
    step_live: Option<String>,
}

impl<'a> Case<'a> {
    /// A case at `capacity` prepared rows, whose live count is the u32
    /// parameter `live_rows`, run at [`poison::live_counts`].
    pub fn new(entry: &'a dyn LabEntry, capacity: u32) -> Self {
        Self::with_source(Source::Entries(vec![entry]), capacity)
    }

    /// A case over a prepared op's launches (`crate::kit`), which may launch
    /// any of `symbols`. `plan` gives the launches at a row count over
    /// views whose handles it takes by buffer name; the case binds the
    /// live count under the name [`Case::live`] gives.
    pub(crate) fn prepared(symbols: Vec<String>, capacity: u32, plan: Plan<'a>) -> Self {
        Self::with_source(Source::Prepared { symbols, plan }, capacity)
    }

    fn with_source(source: Source<'a>, capacity: u32) -> Self {
        Self {
            source,
            host_rows: false,
            capacity,
            live: "live_rows".into(),
            live_counts: poison::live_counts(capacity),
            buffers: Vec::new(),
            scalars: BTreeMap::new(),
            produced: None,
            oracle: None,
            no_write: BTreeMap::new(),
            redundant: BTreeMap::new(),
            omission_guards: BTreeMap::new(),
            step_live: None,
        }
    }

    /// Plans a prepared op's launches at each positive live count's rows,
    /// with the device live count equal, instead of at the capacity: for
    /// an op some launch of which does not read the device live count, so
    /// that its rows follow [`kernel_core::backend::Counts::rows`] alone.
    /// The graph phase captures each count's launches; at live 0 nothing
    /// launches.
    pub(crate) fn host_rows(mut self) -> Self {
        self.host_rows = true;
        self
    }

    /// The parameter holding the device live count.
    pub fn live(mut self, name: &str) -> Self {
        self.live = name.into();
        self
    }

    /// Live counts in place of [`poison::live_counts`]. They must still
    /// hold 0, a partial count and the capacity.
    pub fn live_counts(mut self, counts: Vec<u32>) -> Self {
        self.live_counts = counts;
        self
    }

    /// Launches `entry` after the entries before it, on the same stream, so
    /// the case checks the chain as one launch: split partials then their
    /// merge, for example. Each entry plans at the capacity, opts in to its
    /// shared memory and binds the buffers, the live count and the scalars
    /// its parameters name; every buffer and scalar must be a parameter of
    /// some entry. The links are checked on the declared parameters, not
    /// on what the kernels read: a pointer parameter not declared `const`
    /// is a write and must name an output, scratch or state, and every
    /// entry after the first must have a `const` pointer parameter naming
    /// at least one buffer an earlier entry writes. Other reads, such as
    /// of scratch no entry writes, are not checked. Such an intermediate is
    /// declared once: as scratch, which the run poisons before each launch
    /// and bounds by its `may_write` ranges, or as an output or state the
    /// oracle covers, whose bytes the run then checks like any other. The
    /// phases prepare the buffers before the chain and check them after
    /// it, so they see what the chain leaves, not which entry wrote it; the
    /// null launch also drops each entry in turn and must fail. Graph and
    /// PDL phases launch every entry with PDL behind its predecessor, the
    /// first behind the producer, with no synchronization between entries:
    /// an entry that reads an intermediate before its PDL wait is caught
    /// only when the read lands on poison and changes a checked result.
    pub fn then(mut self, entry: &'a dyn LabEntry) -> Self {
        match &mut self.source {
            Source::Entries(stages) => stages.push(entry),
            Source::Prepared { .. } => unreachable!("a prepared op's case plans its own chain"),
        }
        self
    }

    /// The entries' symbols, or the prepared op's, joined in launch order.
    fn symbol(&self) -> String {
        match &self.source {
            Source::Entries(stages) => stages
                .iter()
                .map(|stage| stage.symbol())
                .collect::<Vec<_>>()
                .join(" then "),
            Source::Prepared { symbols, .. } => symbols.join(" then "),
        }
    }

    /// The launches at `rows`. Entries plan at `rows`, bind their
    /// parameters by name and number their device buffers from 1 in each
    /// entry; a prepared op plans over views whose handles `handle` gives,
    /// and every handle in its arguments must name a buffer of the case or
    /// its live count. A case runs on one device, so a prepared op's
    /// collective launches are refused.
    fn stages(&self, rows: u32, handle: &dyn Fn(&str) -> u64) -> LabResult<Vec<Stage>> {
        let mut names = BTreeSet::from([self.live.as_str()]);
        names.extend(self.buffers.iter().map(|buffer| buffer.name.as_str()));
        match &self.source {
            Source::Entries(stages) => {
                let bound = bind_chain(stages, &names, &self.scalars)?;
                stages
                    .iter()
                    .zip(bound)
                    .map(|(stage, (arguments, bound))| {
                        let symbol = stage.symbol();
                        let at = |e: kernel_core::contract::Refusal| format!("{symbol}: {e}");
                        let plan = stage.plan(rows).map_err(at)?;
                        let arguments = stage.pack_arguments(arguments).map_err(at)?;
                        let names = (1..).zip(bound).collect();
                        Ok(Stage {
                            symbol,
                            plan,
                            arguments,
                            names,
                        })
                    })
                    .collect()
            }
            Source::Prepared { plan, .. } => {
                if let Some(name) = self.scalars.keys().next() {
                    return Err(format!(
                        "scalar {name}: a prepared op's launches carry their own scalars"
                    ));
                }
                let by_handle = names
                    .iter()
                    .map(|&name| (handle(name), name.to_owned()))
                    .collect::<BTreeMap<_, _>>();
                if by_handle.len() != names.len() {
                    return Err("two buffers of the case share a handle".into());
                }
                plan(handle, rows)?
                    .into_iter()
                    .map(|launch| {
                        let symbol = launch.symbol;
                        if launch.collective {
                            return Err(format!("{symbol} is collective; a case runs one rank"));
                        }
                        let mut names = BTreeMap::new();
                        for (at, handle) in handles(&launch.arguments) {
                            let name = by_handle.get(&handle).ok_or_else(|| {
                                format!(
                                    "{symbol} argument {at} holds handle {handle:#x}, \
                                     which names no buffer of the case"
                                )
                            })?;
                            names.insert(handle, name.clone());
                        }
                        Ok(Stage {
                            symbol,
                            plan: launch.plan,
                            arguments: launch.arguments,
                            names,
                        })
                    })
                    .collect()
            }
        }
    }

    /// Checks a chain's declared links: every buffer an entry writes is an
    /// output, scratch or state, and every entry after the first has a
    /// `const` pointer parameter naming one that an earlier entry writes.
    /// Its other reads are not checked.
    fn validate_links(&self) -> LabResult<()> {
        let Source::Entries(stages) = &self.source else {
            return Ok(());
        };
        if stages.len() < 2 {
            return Ok(());
        }
        let mut written = BTreeSet::new();
        for (at, stage) in stages.iter().enumerate() {
            let mut reads = BTreeSet::new();
            let mut writes = BTreeSet::new();
            for param in stage.params() {
                if let EntryParam::DevicePointer { name, cxx_type } = param {
                    if cxx_type.trim_start().starts_with("const ") {
                        reads.insert(name);
                    } else {
                        writes.insert(name);
                    }
                }
            }
            for name in &writes {
                let buffer = self.find(name)?;
                if !matches!(
                    buffer.role,
                    Role::Output { .. } | Role::Scratch(_) | Role::State { .. }
                ) {
                    return Err(format!(
                        "{} writes {name}, which is declared an input; declare it an output, \
                         scratch or state",
                        stage.symbol()
                    ));
                }
            }
            if at > 0 && reads.intersection(&written).next().is_none() {
                return Err(format!(
                    "{} reads nothing an earlier entry of the chain writes",
                    stage.symbol()
                ));
            }
            written.extend(writes);
        }
        Ok(())
    }

    fn buffer(
        mut self,
        name: &str,
        dtype: DType,
        (rows, row_elements): (Rows, usize),
        role: Role<'a>,
        image: Vec<u8>,
    ) -> Self {
        self.buffers.push(Buffer {
            name: name.into(),
            dtype,
            rows,
            row_elements,
            role,
            image,
        });
        self
    }

    /// A read-only input of `rows` rows of `row_elements`, holding `image`.
    pub fn input(self, name: &str, dtype: DType, dims: (Rows, usize), image: Vec<u8>) -> Self {
        self.buffer(name, dtype, dims, Role::Input, image)
    }

    /// An input of I32 IDs (`ids`, -1 for a hole) indexing rows of
    /// `(pool, stride bytes)` of each named pool. With pools, a rerun moves
    /// the IDs to the top rows of the pools; `holes` names a row that
    /// reruns with every ID -1.
    pub fn ids(
        self,
        name: &str,
        dims: (Rows, usize),
        ids: &[i32],
        pools: &[(&str, usize)],
        holes: Holes,
    ) -> Self {
        let pools = pools
            .iter()
            .map(|&(pool, stride)| (pool.to_owned(), stride))
            .collect();
        let image = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        self.buffer(name, DType::I32, dims, Role::Ids { pools, holes }, image)
    }

    /// An output poisoned with [`POISON`] that must write `write`.
    pub fn output(self, name: &str, dtype: DType, dims: (Rows, usize), write: Write<'a>) -> Self {
        self.output_with(name, dtype, dims, write, &[POISON])
    }

    /// An output filled with `pattern`, for outputs where all-ones elements
    /// are valid values, such as -1 IDs.
    pub fn output_with(
        self,
        name: &str,
        dtype: DType,
        dims: (Rows, usize),
        write: Write<'a>,
        pattern: &[u8],
    ) -> Self {
        let role = Role::Output {
            write,
            pattern: pattern.to_vec(),
        };
        self.buffer(name, dtype, dims, role, Vec::new())
    }

    /// Scratch of `bytes`, poisoned before every launch, that may write only
    /// `may_write`.
    pub fn scratch(
        self,
        name: &str,
        bytes: usize,
        may_write: impl Fn(&Images, u32) -> LabResult<Vec<Range<usize>>> + 'a,
    ) -> Self {
        let role = Role::Scratch(Box::new(may_write));
        self.buffer(name, DType::U8, (Rows::Fixed(1), bytes), role, Vec::new())
    }

    /// In-place state holding `initial` before every launch: `write_only` is
    /// poisoned and must be written, and bytes outside it and `updated`
    /// (read-modify-write ranges) must keep `initial`. `updated` permits
    /// writes without requiring one.
    pub fn state(
        self,
        name: &str,
        dtype: DType,
        dims: (Rows, usize),
        initial: Vec<u8>,
        write_only: Write<'a>,
        updated: impl Fn(&Images, u32) -> LabResult<Vec<Range<usize>>> + 'a,
    ) -> Self {
        let role = Role::State {
            write_only,
            updated: Box::new(updated),
        };
        self.buffer(name, dtype, dims, role, initial)
    }

    pub fn scalar(mut self, name: &str, value: Argument) -> Self {
        self.scalars.insert(name.into(), value);
        self
    }

    /// The input a copy producer rewrites before each launch: it is
    /// poisoned first, so a launch that reads it before its PDL wait, while
    /// the producer has not yet copied it, reads poison.
    pub fn produced(mut self, name: &str) -> Self {
        self.produced = Some(name.into());
        self
    }

    /// The expected bytes at each live count. At every live count, its
    /// results must cover every output and state that writes there, each
    /// whole ([`Expected`]).
    pub fn oracle(
        mut self,
        oracle: impl Fn(&Images, u32) -> LabResult<Vec<(String, Expected)>> + 'a,
    ) -> Self {
        self.oracle = Some(Box::new(oracle));
        self
    }

    /// Declares that output or state `name` writes nothing at the positive
    /// count `live`, for `reason`, such as rows a later row supersedes.
    /// Every other output and state must write at every positive count.
    pub fn no_write(mut self, name: &str, live: u32, reason: &str) -> Self {
        self.no_write.insert((name.into(), live), reason.into());
        self
    }

    /// Declares that the chain computes its results without stage
    /// `symbol`, for `reason`, such as a stage whose work a later stage
    /// redoes when it finds none: the null phase then does not require the
    /// chain without it to fail, and the receipt lists it with the reason.
    /// The stage still runs in every other phase.
    pub fn redundant(mut self, symbol: &str, reason: &str) -> Self {
        self.redundant.insert(symbol.into(), reason.into());
        self
    }

    /// Checks output `name`'s write map when the negative control omits
    /// its producer `symbol`, before launching later consumers. Missing
    /// metadata then fails on the host instead of reaching a kernel as
    /// poisoned indices or counts. If an earlier stage wrote the output,
    /// the suffix still runs and must fail the normal checks. Every stage
    /// runs without this host check in eager, graph and PDL phases.
    pub fn omission_guard(mut self, symbol: &str, name: &str) -> Self {
        self.omission_guards.insert(symbol.into(), name.into());
        self
    }

    fn omit_stage(
        &self,
        symbols: &[String],
        skip: usize,
        images: &Images,
        live: u32,
        read: impl Fn(&str) -> LabResult<Vec<u8>>,
        launch: impl Fn(usize) -> LabResult<()>,
    ) -> LabResult<Option<String>> {
        for (at, symbol) in symbols.iter().enumerate() {
            if at != skip {
                launch(at)?;
                continue;
            }
            if let Some(name) = self.omission_guards.get(symbol) {
                let buffer = self.find(name)?;
                let Role::Output { write, pattern } = &buffer.role else {
                    return Err(format!("omission guard {name} is not an output"));
                };
                if let Err(error) = poison::check(
                    &format!("omitted {symbol}: {name}"),
                    &read(name)?,
                    &self.map(buffer, write, images, live)?,
                    poison::Fill::Pattern(pattern),
                ) {
                    return Ok(Some(error));
                }
            }
        }
        Ok(None)
    }

    /// Declares the device live count a step input, for `reason`: the host
    /// writes it with stream-ordered copies before the step, as it does a
    /// step's lengths and tables, so a launch may read it before its PDL
    /// wait. The live-count producer then holds only the produced input
    /// (the "PDL input producer"), which every launch must still read
    /// after its wait; the receipt names the reason.
    pub fn live_count_before_wait(mut self, reason: &str) -> Self {
        self.step_live = Some(reason.into());
        self
    }

    /// The producers of the PDL phases. A produced input gets two: the copy
    /// producer, which rewrites it after releasing the launch, and the
    /// live-count producer, which zeroes both the device live count and the
    /// produced input before releasing the launch and publishes both after
    /// its tail. Without a produced input the live-count producer holds
    /// the count alone.
    ///
    /// These phases are probes, not guaranteed detection. A launch that
    /// reads the count or the input before its PDL wait fails only when
    /// that read lands while the producer holds them: an early count read
    /// then sees zero and writes nothing at a positive count, failing the
    /// map check, and an early input read sees zeros or poison and differs
    /// from eager. CUDA does not promise that the launch overlaps the
    /// producer, so a launch that starts after the producer's tail, or
    /// reaches the misplaced read after the producer restores the count and
    /// input, passes. No deliberately misordered kernel has yet been run
    /// against them on a GPU.
    fn pdl_producers(&self) -> Vec<PdlProducer> {
        let produced = self.produced.is_some();
        let count = self.step_live.is_none();
        let live = PdlProducer::Live { produced, count };
        match (produced, count) {
            (true, _) => vec![PdlProducer::Copy, live],
            (false, true) => vec![live],
            // A step input count and no produced input: nothing to hold.
            (false, false) => Vec::new(),
        }
    }

    /// The positive live counts at which a launch that writes nothing must
    /// fail the checks: every one at which some output or state writes.
    /// At the others every output and state is declared no-write, and the
    /// replays there check that each keeps its poison or initial bytes.
    fn null_counts(&self, images: &Images) -> LabResult<Vec<u32>> {
        let mut counts = Vec::new();
        for &live in self.live_counts.iter().filter(|&&live| live > 0) {
            let mut writes = false;
            for buffer in self.buffers.iter().filter(|buffer| buffer.checked()) {
                writes |= self.writes(buffer, images, live)? > 0;
            }
            if writes {
                counts.push(live);
            }
        }
        Ok(counts)
    }

    /// The largest live count, at which the high-id and hole reruns run.
    fn top(&self) -> u32 {
        self.live_counts.iter().max().copied().unwrap_or(0)
    }

    /// `images` with the IDs of `buffer` moved to the top rows of its
    /// `pools` and the pools' rows moved with them, and the permutation.
    /// Refused when no ID moves, since the rerun would then repeat the
    /// first run.
    fn high_images(
        &self,
        images: &Images,
        buffer: &Buffer<'_>,
        pools: &[(String, usize)],
    ) -> LabResult<(Images, Vec<usize>)> {
        let name = &buffer.name;
        let ids = images.i32s(name)?;
        let used = ids
            .iter()
            .filter_map(|&id| usize::try_from(id).ok())
            .collect::<BTreeSet<_>>();
        if used.is_empty() {
            return Err(format!(
                "{name} holds no ID, so its high-id rerun moves nothing"
            ));
        }
        let order = high_permutation(&used, self.pool_rows(pools)?)?;
        if used.iter().all(|&id| order[id] == id) {
            return Err(format!(
                "{name} already holds the top rows of its pools, so its high-id rerun moves nothing"
            ));
        }
        let mut moved = images.clone();
        moved.0.insert(name.clone(), remap_ids(&ids, &order));
        for (pool, stride) in pools {
            if let Some(image) = moved.0.get_mut(pool) {
                *image = permute_rows(image, *stride, &order, false);
            }
        }
        Ok((moved, order))
    }

    /// The hole row of `buffer` and `images` with every ID in it -1.
    fn hole_images(
        &self,
        images: &Images,
        buffer: &Buffer<'_>,
        holes: Holes,
    ) -> LabResult<Option<(usize, Images)>> {
        let name = &buffer.name;
        let row = match (holes, buffer.rows) {
            (Holes::Never, _) => return Ok(None),
            (Holes::LastLive, Rows::Live) => (self.top() as usize)
                .checked_sub(1)
                .ok_or_else(|| format!("{name} has no live row"))?,
            (Holes::LastLive, Rows::Fixed(_)) => {
                return Err(format!(
                    "{name} has fixed rows; name the hole row the entry reads (Holes::Row)"
                ));
            }
            (Holes::Row(row), _) => row,
        };
        let rows = buffer.rows(self.capacity);
        if row >= rows {
            return Err(format!("{name} hole row {row} lies past its {rows} rows"));
        }
        let mut moved = images.clone();
        let width = buffer.row_elements * 4;
        let bytes = moved
            .0
            .get_mut(name)
            .ok_or_else(|| format!("no image named {name}"))?;
        for id in bytes[row * width..][..width].chunks_exact_mut(4) {
            id.copy_from_slice(&(-1_i32).to_le_bytes());
        }
        Ok(Some((row, moved)))
    }

    /// Checks the high-id and hole reruns of every ID input at the top
    /// count, on exact results only ([`Case::exact_results`]): the high-id
    /// rerun's results, moved back, match the first run's, and one output
    /// or state exact in the first run, the rerun and the old IDs over the
    /// moved pools (which a launch that ignores the moved IDs reads) is the
    /// same in the first two and differs in the last; the hole row changes
    /// the results; and the maps of both reruns hold.
    fn validate_reruns(&self, images: &Images) -> LabResult<()> {
        let top = self.top();
        let (first, mut numeric) = self.exact_results(images, top)?;
        for buffer in &self.buffers {
            let Role::Ids { pools, holes } = &buffer.role else {
                continue;
            };
            let name = &buffer.name;
            if !pools.is_empty() {
                let (moved, order) = self.high_images(images, buffer, pools)?;
                self.check_maps(&moved, top)
                    .map_err(|e| format!("with high {name}: {e}"))?;
                // The run moves pool outputs and states back before it
                // compares them with the first run.
                let mut back = |(mut results, other): (BTreeMap<String, Vec<u8>>, _)| {
                    for (pool, stride) in pools {
                        if let Some(bytes) = results.get_mut(pool) {
                            *bytes = permute_rows(bytes, *stride, &order, true);
                        }
                    }
                    numeric.extend(other);
                    results
                };
                let high = back(
                    self.exact_results(&moved, top)
                        .map_err(|e| format!("with high {name}: {e}"))?,
                );
                if let Some(output) = differs(&first, &high) {
                    return Err(format!(
                        "{name} high-id rerun: the oracle's {output} on the moved IDs and pools, \
                         moved back, differs from the first run's, so a launch that passes the \
                         first run fails the rerun's bitwise comparison"
                    ));
                }
                // The stale read the rerun must refuse: the old IDs over
                // the moved pools, as a launch that ignores the moved IDs
                // reads them.
                let mut stale = moved.clone();
                stale.0.insert(name.clone(), images.bytes(name)?.to_vec());
                let stale = back(
                    self.exact_results(&stale, top)
                        .map_err(|e| format!("with the old {name} over moved pools: {e}"))?,
                );
                if witness(&first, &high, &stale).is_none() {
                    return Err(format!(
                        "{name} high-id rerun moves no ID the entry reads at live {top}: \
                         no output or state has exact results in the first run, the rerun and \
                         the old IDs over the moved pools, the same in the first two and \
                         different in the last{}, so a launch that ignores the moved IDs passes \
                         the rerun (give an output or state the IDs select exact bits in every \
                         run, with IDs in the rows the entry reads and moved pool rows that \
                         differ from the rows they displace, or declare {name} without pools)",
                        numeric_note(&numeric)
                    ));
                }
            }
            if let Some((row, moved)) = self.hole_images(images, buffer, *holes)? {
                let at = |e: String| format!("{name} row {row} all holes: {e}");
                self.check_maps(&moved, top).map_err(at)?;
                let (results, other) = self.exact_results(&moved, top).map_err(at)?;
                numeric.extend(other);
                if differs(&first, &results).is_none() {
                    return Err(format!(
                        "{name} row {row} all holes leaves the oracle's results at live {top} \
                         unchanged in every exact byte the run compares{}, so a launch that \
                         ignores the row passes (give an output or state the row changes exact \
                         bits, or declare Holes::Never)",
                        numeric_note(&numeric)
                    ));
                }
            }
        }
        Ok(())
    }

    /// The bytes a launch that passes the run's checks on `images` at
    /// `live` leaves in each output and state they fix exactly, by name:
    /// one the oracle gives bits for, an output's fill with those bits in
    /// its write map (the only bytes the map and bit checks accept) and a
    /// state's bits whole; or one that writes nothing, an output's fill and
    /// a state's initial bytes. Only write maps, never a state's update
    /// ranges, which permit writes without requiring them, shape them.
    ///
    /// Buffers with a numeric reference ([`Expected::Values`]) are left
    /// out and returned apart: two tolerance gates can accept one result,
    /// so a numeric reference never tells two runs apart. So are buffers
    /// written without an oracle.
    fn exact_results(&self, images: &Images, live: u32) -> LabResult<(Results, BTreeSet<String>)> {
        let mut expected = match self.oracle {
            Some(_) => self.expected(images, live)?.into_iter().collect(),
            None => BTreeMap::new(),
        };
        let mut exact = BTreeMap::new();
        let mut numeric = BTreeSet::new();
        for buffer in self.buffers.iter().filter(|buffer| buffer.checked()) {
            let name = &buffer.name;
            let bits = match expected.remove(name) {
                Some(Expected::Bits(bits)) => Some(bits),
                Some(Expected::Values(_) | Expected::Judge(_)) => {
                    numeric.insert(name.clone());
                    continue;
                }
                None if self.writes(buffer, images, live)? == 0 => None,
                None => continue,
            };
            let image = match (&buffer.role, bits) {
                (Role::Output { write, pattern }, bits) => {
                    if pattern.is_empty() {
                        return Err(format!("{name} has an empty fill pattern"));
                    }
                    let bytes = buffer.bytes(self.capacity)?;
                    let mut image = pattern
                        .iter()
                        .copied()
                        .cycle()
                        .take(bytes)
                        .collect::<Vec<_>>();
                    if let Some(bits) = bits {
                        for range in self.map(buffer, write, images, live)?.ranges() {
                            image[range.clone()].copy_from_slice(&bits[range.clone()]);
                        }
                    }
                    image
                }
                (_, Some(bits)) => bits,
                (_, None) => images.bytes(name)?.to_vec(),
            };
            exact.insert(name.clone(), image);
        }
        Ok((exact, numeric))
    }

    /// Checks that every output and state map at `live` lies in its buffer
    /// and counts elements of its dtype, whatever rule built it: at live 0
    /// nothing is written or updated, and a buffer of live rows is written
    /// or updated only below the live count. Maps of fixed rows, such as a
    /// pool an ID input indexes, may lie anywhere in the buffer.
    fn check_maps(&self, images: &Images, live: u32) -> LabResult<()> {
        for buffer in &self.buffers {
            let Some(map) = self.written(buffer, images, live)? else {
                continue;
            };
            let name = &buffer.name;
            let updated = match &buffer.role {
                Role::State { updated, .. } => union(updated(images, live)?),
                _ => Vec::new(),
            };
            let bytes = buffer.bytes(self.capacity)?;
            let end = |ranges: &[Range<usize>]| ranges.iter().map(|range| range.end).max();
            if end(map.ranges()).is_some_and(|end| end > bytes) {
                return Err(format!(
                    "{name} map at live {live} lies outside its {bytes} bytes"
                ));
            }
            if end(&updated).is_some_and(|end| end > bytes) {
                return Err(format!(
                    "{name} updates at live {live} lie outside its {bytes} bytes"
                ));
            }
            let every_count = matches!(
                &buffer.role,
                Role::Output {
                    write: Write::EveryCount(_),
                    ..
                }
            );
            if live == 0 {
                let written = map.written_bytes();
                if written > 0 && every_count && self.oracle.is_none() {
                    return Err(format!(
                        "{name} maps {written} bytes at live 0, which only the oracle's live-0 \
                         values check"
                    ));
                }
                if written > 0 && !every_count {
                    return Err(format!(
                        "{name} maps {written} bytes at live 0, where a launch writes nothing"
                    ));
                }
                let updated = updated.iter().map(Range::len).sum::<usize>();
                if updated > 0 {
                    return Err(format!(
                        "{name} updates {updated} bytes at live 0, where a launch writes nothing"
                    ));
                }
            }
            if buffer.rows == Rows::Live {
                let rows = (live as usize).min(buffer.rows(self.capacity));
                let live_bytes = rows * buffer.row_elements * map.element_bytes();
                if let Some(end) = end(map.ranges()).into_iter().chain(end(&updated)).max()
                    && end > live_bytes
                {
                    return Err(format!(
                        "{name} map at live {live} reaches byte {end}, past its {rows} live rows \
                         ({live_bytes} bytes)"
                    ));
                }
            }
        }
        Ok(())
    }

    fn find(&self, name: &str) -> LabResult<&Buffer<'a>> {
        self.buffers
            .iter()
            .find(|buffer| buffer.name == name)
            .ok_or_else(|| format!("no buffer named {name}"))
    }

    pub(crate) fn images(&self) -> Images {
        Images(
            self.buffers
                .iter()
                .filter(|buffer| buffer.has_image())
                .map(|buffer| (buffer.name.clone(), buffer.image.clone()))
                .collect(),
        )
    }

    /// Every buffer's bytes and what it holds before a launch over
    /// `images`, by name: an image (inputs, ID inputs, states) or a fill
    /// (an output's pattern, scratch's poison), for runs the case does not
    /// launch itself (a collective's ranks). The device live count is
    /// [`Case::live_name`], not among them.
    pub(crate) fn fills(&self, images: &Images) -> LabResult<Vec<(String, usize, Fill)>> {
        self.buffers
            .iter()
            .map(|buffer| {
                let fill = match &buffer.role {
                    Role::Output { pattern, .. } => Fill::Pattern(pattern.clone()),
                    Role::Scratch(_) => Fill::Pattern(vec![POISON]),
                    _ => Fill::Image(images.bytes(&buffer.name)?.to_vec()),
                };
                Ok((buffer.name.clone(), buffer.bytes(self.capacity)?, fill))
            })
            .collect()
    }

    /// The name of the device live count's buffer.
    pub(crate) fn live_name(&self) -> &str {
        &self.live
    }

    /// Judges `results`, a launch's bytes over `images` at `live` by
    /// buffer name, as the conformance run judges its eager bytes, for runs
    /// the case does not launch itself: each output keeps its fill outside
    /// its write map, scratch keeps its poison outside what it may write,
    /// `expected` covers every output that writes, and each compares as
    /// its kind says. Returns the comparisons' receipt.
    pub(crate) fn judge(
        &self,
        expected: Vec<(String, Expected)>,
        images: &Images,
        live: u32,
        results: &Results,
    ) -> LabResult<Value> {
        let named = expected
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>();
        for buffer in &self.buffers {
            let name = &buffer.name;
            let (kept, fill) = match &buffer.role {
                Role::Output { write, pattern } => {
                    let map = self.map(buffer, write, images, live)?;
                    if map.written_bytes() > 0 && !named.contains(name.as_str()) {
                        return Err(format!("the oracle gives no {name}, which writes"));
                    }
                    (map.ranges().to_vec(), pattern.clone())
                }
                Role::Scratch(may_write) => (union(may_write(images, live)?), vec![POISON]),
                _ => continue,
            };
            let actual = results
                .get(name)
                .ok_or_else(|| format!("the launch gave no {name}"))?;
            let mut inside = vec![false; actual.len()];
            for range in kept {
                inside[range.start.min(actual.len())..range.end.min(actual.len())].fill(true);
            }
            if let Some(at) =
                (0..actual.len()).find(|&at| !inside[at] && actual[at] != fill[at % fill.len()])
            {
                return Err(format!(
                    "{name} byte {at} at live {live} is written outside its map"
                ));
            }
        }
        let mut receipt = BTreeMap::new();
        for (name, expected) in expected {
            let actual = results
                .get(&name)
                .ok_or_else(|| format!("the launch gave no {name}"))?;
            self.check_expected(&name, expected, actual, images, live, &mut receipt)?;
        }
        Ok(json!(receipt))
    }

    /// Check just the output and state values of a unit launch. The unit
    /// runner has already checked its written regions and inactive bytes;
    /// it returns no scratch image to judge.
    pub(crate) fn judge_outputs(
        &self,
        images: &Images,
        live: u32,
        results: &Results,
    ) -> LabResult<Value> {
        let mut receipt = BTreeMap::new();
        for (name, expected) in self.expected(images, live)? {
            let actual = results
                .get(&name)
                .ok_or_else(|| format!("the launch gave no {name}"))?;
            self.check_expected(&name, expected, actual, images, live, &mut receipt)?;
        }
        Ok(json!(receipt))
    }

    /// `images` with the elements every state's write map names at `live`
    /// poisoned, so a launch that leaves one unwritten shows
    /// ([`Case::check_unit_writes`]).
    pub(crate) fn poison_state_maps(&self, images: &Images, live: u32) -> LabResult<Images> {
        let mut poisoned = images.clone();
        for buffer in &self.buffers {
            let Role::State { write_only, .. } = &buffer.role else {
                continue;
            };
            let map = self.map(buffer, write_only, images, live)?;
            let mut bytes = images.bytes(&buffer.name)?.to_vec();
            for range in map.ranges() {
                bytes[range.clone()].fill(POISON);
            }
            poisoned = poisoned.with(&buffer.name, bytes);
        }
        Ok(poisoned)
    }

    /// Checks a launch's `results` at `live` over `images` (whose state
    /// maps [`Case::poison_state_maps`] poisoned), for the invariance
    /// adapters: every element of an output's or state's write map is
    /// written (no longer its fill) and, for BF16 and F32 elements, finite;
    /// every other byte of an output keeps its fill, and of a state its
    /// image, but for the state's read-modify-write ranges.
    pub(crate) fn check_unit_writes(
        &self,
        images: &Images,
        live: u32,
        results: &Results,
    ) -> LabResult<()> {
        let finite = |dtype: DType, element: &[u8]| match dtype {
            DType::Bf16 => u16::from_le_bytes([element[0], element[1]]) & 0x7f80 != 0x7f80,
            DType::F32 => f32::from_le_bytes(element.try_into().expect("4 bytes")).is_finite(),
            _ => true,
        };
        for buffer in &self.buffers {
            let name = &buffer.name;
            let (map, kept) = match &buffer.role {
                Role::Output { write, pattern } => (
                    self.map(buffer, write, images, live)?,
                    Fill::Pattern(pattern.clone()),
                ),
                Role::State {
                    write_only,
                    updated,
                } => {
                    let mut image = images.bytes(name)?.to_vec();
                    for range in union(updated(images, live)?) {
                        let after = results
                            .get(name)
                            .ok_or_else(|| format!("the launch gave no {name}"))?;
                        image[range.clone()].copy_from_slice(&after[range]);
                    }
                    (
                        self.map(buffer, write_only, images, live)?,
                        Fill::Image(image),
                    )
                }
                _ => continue,
            };
            let actual = results
                .get(name)
                .ok_or_else(|| format!("the launch gave no {name}"))?;
            match &kept {
                Fill::Pattern(pattern) => {
                    poison::check(name, actual, &map, poison::Fill::Pattern(pattern))?
                }
                Fill::Image(image) => {
                    let mut cursor = 0;
                    for range in map.ranges() {
                        if let Some(at) = (cursor..range.start).find(|&at| actual[at] != image[at])
                        {
                            return Err(format!("{name} byte {at} outside its write map changed"));
                        }
                        for element in range.clone().step_by(map.element_bytes()) {
                            let end = element + map.element_bytes();
                            if actual[element..end].iter().all(|&byte| byte == POISON) {
                                return Err(format!(
                                    "{name} byte {element} in its write map holds the poison"
                                ));
                            }
                        }
                        cursor = range.end;
                    }
                    if let Some(at) = (cursor..actual.len()).find(|&at| actual[at] != image[at]) {
                        return Err(format!("{name} byte {at} outside its write map changed"));
                    }
                }
            }
            for range in map.ranges() {
                for element in range.clone().step_by(map.element_bytes()) {
                    let end = element + map.element_bytes();
                    if !finite(buffer.dtype, &actual[element..end]) {
                        return Err(format!("{name} element at byte {element} is not finite"));
                    }
                }
            }
        }
        Ok(())
    }

    /// The bytes of in-place state `name` a launch at `live` over `images`
    /// writes (its write map) or may rewrite (its read-modify-write
    /// ranges), for the kit's property checks.
    pub(crate) fn state_writes(
        &self,
        name: &str,
        images: &Images,
        live: u32,
    ) -> LabResult<Vec<Range<usize>>> {
        let buffer = self
            .buffers
            .iter()
            .find(|buffer| buffer.name == name)
            .ok_or_else(|| format!("no buffer named {name}"))?;
        let Role::State {
            write_only,
            updated,
        } = &buffer.role
        else {
            return Err(format!("{name} is not an in-place state"));
        };
        let mut ranges = self
            .map(buffer, write_only, images, live)?
            .ranges()
            .to_vec();
        ranges.extend(updated(images, live)?);
        Ok(union(ranges))
    }

    /// Bytes required to be written, excluding optional state updates.
    pub(crate) fn required_writes(
        &self,
        name: &str,
        images: &Images,
        live: u32,
    ) -> LabResult<Vec<Range<usize>>> {
        let buffer = self
            .buffers
            .iter()
            .find(|buffer| buffer.name == name)
            .ok_or_else(|| format!("no buffer named {name}"))?;
        self.written(buffer, images, live)?
            .map(|map| map.ranges().to_vec())
            .ok_or_else(|| format!("{name} is not an output or state"))
    }

    fn map(
        &self,
        buffer: &Buffer<'_>,
        write: &Write<'_>,
        images: &Images,
        live: u32,
    ) -> LabResult<WriteMap> {
        let elements = WriteMap::of(buffer.dtype);
        match write {
            Write::LiveRows => elements
                .prefix((live as usize).min(buffer.rows(self.capacity)) * buffer.row_elements),
            Write::All if live == 0 => elements.none(),
            Write::All => elements.prefix(buffer.elements(self.capacity)),
            Write::Map(rule) | Write::EveryCount(rule) => {
                // A map of wider elements than the dtype's passes elements
                // the launch wrote only in part.
                let map = rule(images, live)?;
                let bytes = elements.bytes()?;
                if map.element_bytes() != bytes {
                    return Err(format!(
                        "{} map counts {}-byte elements, its {:?} dtype {bytes}",
                        buffer.name,
                        map.element_bytes(),
                        buffer.dtype
                    ));
                }
                Ok(map)
            }
        }
    }

    /// The bytes an output or state must write at `live`.
    fn written(
        &self,
        buffer: &Buffer<'_>,
        images: &Images,
        live: u32,
    ) -> LabResult<Option<WriteMap>> {
        match &buffer.role {
            Role::Output { write, .. }
            | Role::State {
                write_only: write, ..
            } => self.map(buffer, write, images, live).map(Some),
            _ => Ok(None),
        }
    }

    /// The bytes an output or state must write or may update at `live`.
    fn writes(&self, buffer: &Buffer<'_>, images: &Images, live: u32) -> LabResult<usize> {
        let updated = match &buffer.role {
            Role::State { updated, .. } => {
                union(updated(images, live)?).iter().map(Range::len).sum()
            }
            _ => 0,
        };
        Ok(self
            .written(buffer, images, live)?
            .map_or(0, |map| map.written_bytes())
            + updated)
    }

    /// The oracle's results at `live` as a launch would hold them, with
    /// each buffer's dtype: bits, and values rounded to their BF16 or F32
    /// buffer; for CPU witnesses of the kit's mappings.
    pub(crate) fn oracle_bytes(
        &self,
        images: &Images,
        live: u32,
    ) -> LabResult<BTreeMap<String, (DType, Vec<u8>)>> {
        let mut held = BTreeMap::new();
        for (name, expected) in self.expected(images, live)? {
            let dtype = self.find(&name)?.dtype;
            let bytes = match (expected, dtype) {
                (Expected::Bits(bits), _) => bits,
                (Expected::Values(output), DType::Bf16) => output
                    .values
                    .iter()
                    .flat_map(|&v| kernel_core::codec::f32_to_bf16(v as f32).to_le_bytes())
                    .collect(),
                (Expected::Values(output), DType::F32) => output
                    .values
                    .iter()
                    .flat_map(|&v| (v as f32).to_le_bytes())
                    .collect(),
                _ => return Err(format!("{name}: the oracle's result has no host bytes")),
            };
            held.insert(name, (dtype, bytes));
        }
        Ok(held)
    }

    /// The oracle's results at `live`, refused unless they cover every
    /// output and state that writes at `live`, each whole: the buffer's
    /// bits, or one value per element.
    pub(crate) fn expected(
        &self,
        images: &Images,
        live: u32,
    ) -> LabResult<Vec<(String, Expected)>> {
        let oracle = self.oracle.as_ref().ok_or("the case has no oracle")?;
        let results = oracle(images, live)?;
        let mut named = BTreeSet::new();
        for (name, expected) in &results {
            let buffer = self.find(name)?;
            if !buffer.checked() {
                return Err(format!("the oracle names {name}, not an output or state"));
            }
            if !named.insert(name.as_str()) {
                return Err(format!("the oracle names {name} twice"));
            }
            match expected {
                Expected::Bits(bits) => {
                    let bytes = buffer.bytes(self.capacity)?;
                    if bits.len() != bytes {
                        return Err(format!(
                            "the oracle's {name} holds {} bytes of {bytes}",
                            bits.len()
                        ));
                    }
                }
                Expected::Judge(_) => {}
                Expected::Values(reference) => {
                    if !matches!(buffer.dtype, DType::Bf16 | DType::F32 | DType::Fp8E4M3) {
                        return Err(format!(
                            "{name} is {:?}; the oracle compares its bits, not values",
                            buffer.dtype
                        ));
                    }
                    let elements = buffer.elements(self.capacity);
                    if [
                        reference.values.len(),
                        reference.sum_abs_products.len(),
                        reference.reduction_lengths.len(),
                    ]
                    .iter()
                    .any(|&len| len != elements)
                    {
                        return Err(format!(
                            "the oracle's {name} holds {} values of {elements}",
                            reference.values.len()
                        ));
                    }
                }
            }
        }
        for buffer in self.buffers.iter().filter(|buffer| buffer.checked()) {
            let bytes = self.writes(buffer, images, live)?;
            if bytes > 0 && !named.contains(buffer.name.as_str()) {
                return Err(format!(
                    "the oracle omits {}, which writes {bytes} bytes at live {live}",
                    buffer.name
                ));
            }
        }
        Ok(results)
    }

    /// Checks the declaration without a GPU: the live counts keep 0, a
    /// partial count and the capacity, every parameter is bound and every
    /// binding is a parameter, images have their declared sizes, IDs lie in
    /// their pools, the produced input can be told from poison, maps count
    /// their buffer's elements and lie in it, every output and state writes
    /// at every positive count it is not declared no-write at, the oracle
    /// covers what writes, and the reruns move an ID and read the hole row.
    pub fn validate(&self) -> LabResult<()> {
        let symbol = self.symbol();
        let context = |error: String| format!("{symbol}: {error}");
        if let Source::Entries(stages) = &self.source {
            for stage in stages {
                let at = match stages.len() {
                    1 => String::new(),
                    _ => format!("{} ", stage.symbol()),
                };
                stage
                    .plan(self.capacity)
                    .map_err(|e| context(format!("{at}plan at capacity {}: {e}", self.capacity)))?;
            }
        }
        if let Some(live) = self.live_counts.iter().find(|&&live| live > self.capacity) {
            return Err(context(format!(
                "live count {live} exceeds capacity {}",
                self.capacity
            )));
        }
        // Live 0 checks that nothing is written, a partial count that rows
        // past the live count keep their poison, and the capacity that the
        // graph's prepared rows are all written.
        let counts = &self.live_counts;
        let mut missing = Vec::new();
        if !counts.contains(&0) {
            missing.push("0");
        }
        if self.capacity > 1 && !counts.iter().any(|&live| live > 0 && live < self.capacity) {
            missing.push("a partial count");
        }
        if !counts.contains(&self.capacity) {
            missing.push("the capacity");
        }
        if !missing.is_empty() {
            return Err(context(format!(
                "live counts {counts:?} omit {}",
                missing.join(", ")
            )));
        }
        let mut names = BTreeSet::from([self.live.as_str()]);
        for buffer in &self.buffers {
            if !names.insert(&buffer.name) {
                return Err(context(format!("{} is declared twice", buffer.name)));
            }
            let bytes = buffer.bytes(self.capacity).map_err(context)?;
            if bytes == 0 {
                return Err(context(format!("{} holds no bytes", buffer.name)));
            }
            if buffer.has_image() == buffer.image.is_empty()
                || (buffer.has_image() && buffer.image.len() != bytes)
            {
                return Err(context(format!(
                    "{} has a {}-byte image for {bytes} bytes",
                    buffer.name,
                    buffer.image.len()
                )));
            }
        }
        match &self.source {
            Source::Entries(stages) => {
                bind_chain(stages, &names, &self.scalars).map_err(context)?;
                if !self.omission_guards.is_empty() {
                    let launched = self
                        .stages(self.capacity, &|name| token(&names, name))
                        .map_err(context)?;
                    self.validate_launched(&launched).map_err(context)?;
                }
            }
            Source::Prepared { .. } => {
                let launched = self
                    .stages(self.capacity, &|name| token(&names, name))
                    .map_err(context)?;
                self.validate_launched(&launched).map_err(context)?;
            }
        }
        self.validate_links().map_err(context)?;
        self.validate_ids().map_err(context)?;
        if let Some(name) = &self.produced {
            let buffer = self.find(name).map_err(context)?;
            if !matches!(buffer.role, Role::Input) {
                return Err(context(format!("produced {name} is not a plain input")));
            }
            let element = WriteMap::of(buffer.dtype).bytes()?;
            if buffer.image.len() % 4 != 0 {
                return Err(context(format!("produced {name} is not whole u32 words")));
            }
            if let Some(at) = buffer
                .image
                .chunks(element)
                .position(|value| value.iter().all(|&byte| byte == POISON))
            {
                return Err(context(format!(
                    "produced {name} element {at} is the poison pattern, which reads as unwritten"
                )));
            }
        }
        let symbols: Vec<String> = match &self.source {
            Source::Entries(stages) => stages.iter().map(|stage| stage.symbol()).collect(),
            Source::Prepared { symbols, .. } => symbols.clone(),
        };
        if self
            .step_live
            .as_ref()
            .is_some_and(|reason| reason.trim().is_empty())
        {
            return Err(context(
                "a live count read before the PDL wait gives no reason".into(),
            ));
        }
        for (stage, reason) in &self.redundant {
            if !symbols.contains(stage) {
                return Err(context(format!(
                    "redundant stage {stage} is not a symbol of the case"
                )));
            }
            if reason.trim().is_empty() {
                return Err(context(format!("redundant stage {stage} gives no reason")));
            }
        }
        let images = self.images();
        for ((name, live), reason) in &self.no_write {
            if !self.find(name).map_err(context)?.checked() {
                return Err(context(format!(
                    "no-write {name} is not an output or state"
                )));
            }
            if *live == 0 || !self.live_counts.contains(live) {
                return Err(context(format!(
                    "no-write {name} at live {live} is not a positive live count of the case"
                )));
            }
            if reason.trim().is_empty() {
                return Err(context(format!(
                    "no-write {name} at live {live} gives no reason"
                )));
            }
        }
        // Each output and state must write at every positive count, so a
        // launch that skips it fails there; another buffer's writes do not
        // stand in for it.
        for &live in &self.live_counts {
            self.check_maps(&images, live).map_err(context)?;
            if live == 0 {
                continue;
            }
            for buffer in self.buffers.iter().filter(|buffer| buffer.checked()) {
                let name = &buffer.name;
                let bytes = self.writes(buffer, &images, live).map_err(context)?;
                let declared = self.no_write.contains_key(&(name.clone(), live));
                if bytes == 0 && !declared {
                    return Err(context(format!(
                        "{name} writes nothing at live {live}, so a launch that skips it would pass; \
                         declare it no_write, with the reason, if the entry writes none of it there"
                    )));
                }
                if bytes > 0 && declared {
                    return Err(context(format!(
                        "{name} is declared no-write at live {live} but writes {bytes} bytes"
                    )));
                }
                let mapped = self
                    .written(buffer, &images, live)
                    .map_err(context)?
                    .map_or(0, |map| map.written_bytes());
                if bytes > 0 && mapped == 0 && self.oracle.is_none() {
                    return Err(context(format!(
                        "{name} only updates bytes in place at live {live}, which only an oracle checks"
                    )));
                }
            }
        }
        if self.oracle.is_some() {
            for &live in &self.live_counts {
                self.expected(&images, live).map_err(context)?;
            }
        }
        if self.oracle.is_none()
            && self.buffers.iter().any(
                |buffer| matches!(&buffer.role, Role::Ids { holes, .. } if *holes != Holes::Never),
            )
        {
            return Err(context("an all-hole row needs an oracle".into()));
        }
        self.validate_reruns(&images).map_err(context)?;
        Ok(())
    }

    /// Checks a prepared op's launches at the capacity: it plans at least
    /// one, and some launch takes every output.
    fn validate_launched(&self, stages: &[Stage]) -> LabResult<()> {
        if stages.is_empty() {
            return Err(format!(
                "the op plans no launch at capacity {}",
                self.capacity
            ));
        }
        let taken = stages
            .iter()
            .flat_map(|stage| stage.names.values().map(String::as_str))
            .collect::<BTreeSet<_>>();
        if let Some(buffer) = self.buffers.iter().find(|buffer| {
            matches!(buffer.role, Role::Output { .. }) && !taken.contains(buffer.name.as_str())
        }) {
            return Err(format!("no launch takes output {}", buffer.name));
        }
        for (producer, name) in &self.omission_guards {
            let buffer = self.find(name)?;
            if !matches!(buffer.role, Role::Output { .. }) {
                return Err(format!("omission guard {name} is not an output"));
            }
            let at = stages
                .iter()
                .position(|stage| stage.symbol == *producer)
                .ok_or_else(|| format!("omission guard producer {producer} is not launched"))?;
            let takes = |stage: &Stage| stage.names.values().any(|path| path == name);
            if !takes(&stages[at]) || !stages[at + 1..].iter().any(takes) {
                return Err(format!(
                    "omission guard {name} needs its producer and a later consumer"
                ));
            }
        }
        Ok(())
    }

    /// One eager launch of the stages at `rows` over `images`, with the
    /// device live count `live`: inputs and states hold their images,
    /// outputs their fill and scratch poison. Returns every output's and
    /// state's bytes. Property checks compare these across row counts and
    /// row orders.
    pub(crate) fn execute(
        &self,
        session: &Session,
        functions: &dyn Fn(&str) -> LabResult<Function>,
        images: &Images,
        rows: u32,
        live: u32,
    ) -> LabResult<Results> {
        let mut device = BTreeMap::new();
        for buffer in &self.buffers {
            let target = session.allocate(buffer.bytes(self.capacity)?)?;
            match &buffer.role {
                Role::Output { pattern, .. } => poison::fill_all(session, &target, pattern)?,
                Role::Scratch(_) => poison::fill_all(session, &target, &[POISON])?,
                _ => poison::write(session, &target, 0, images.bytes(&buffer.name)?)?,
            }
            device.insert(buffer.name.clone(), target);
        }
        let live_buffer = session.allocate(16)?;
        poison::write(session, &live_buffer, 0, &live.to_le_bytes())?;
        for stage in self.stages(rows, &address(&device, &live_buffer, &self.live))? {
            let at = |e: String| format!("{} at {rows} rows: {e}", stage.symbol);
            let function = functions(&stage.symbol).map_err(at)?;
            if stage.plan.dynamic_shared_bytes > DEFAULT_SHARED_BYTES {
                function
                    .set_max_dynamic_shared_memory(stage.plan.dynamic_shared_bytes)
                    .map_err(|e| at(e.to_string()))?;
            }
            let bindings = bindings(&stage.names, &device, &live_buffer, &self.live);
            launch::launch(
                &session.context,
                &function,
                &session.stream,
                stage.plan,
                &stage.arguments,
                &bindings,
            )
            .map_err(at)?;
            // For the invariance receipts, which name what a case ran.
            crate::launched(&stage.symbol);
        }
        session.stream.synchronize().map_err(|e| e.to_string())?;
        self.buffers
            .iter()
            .filter(|buffer| buffer.checked())
            .map(|buffer| {
                Ok((
                    buffer.name.clone(),
                    poison::read(session, &device[&buffer.name])?,
                ))
            })
            .collect()
    }

    fn validate_ids(&self) -> LabResult<()> {
        let mut pooled = BTreeSet::new();
        for buffer in &self.buffers {
            let Role::Ids { pools, .. } = &buffer.role else {
                continue;
            };
            let ids = self.images().i32s(&buffer.name)?;
            if let Some(id) = ids.iter().find(|&&id| id < -1) {
                return Err(format!("{} holds ID {id}", buffer.name));
            }
            let rows = self.pool_rows(pools)?;
            if let Some(id) = ids.iter().find(|&&id| id >= 0 && id as usize >= rows) {
                return Err(format!(
                    "{} ID {id} lies past its pools' {rows} rows",
                    buffer.name
                ));
            }
            for (pool, _) in pools {
                if !pooled.insert(pool.as_str()) {
                    return Err(format!("{pool} is the pool of two ID inputs"));
                }
            }
        }
        Ok(())
    }

    /// The rows every pool of an ID input holds.
    fn pool_rows(&self, pools: &[(String, usize)]) -> LabResult<usize> {
        let mut rows = usize::MAX;
        for (pool, stride) in pools {
            let buffer = self.find(pool)?;
            if matches!(buffer.role, Role::Ids { .. } | Role::Scratch(_)) {
                return Err(format!("pool {pool} must be an input, output or state"));
            }
            let bytes = buffer.bytes(self.capacity)?;
            if *stride == 0 || bytes % stride != 0 {
                return Err(format!(
                    "pool {pool} of {bytes} bytes is not rows of {stride}"
                ));
            }
            rows = rows.min(bytes / stride);
        }
        Ok(rows)
    }

    /// The poisoned checks of one launch at `live`, the produced input
    /// among them when `produced`.
    fn replay<'s>(
        &self,
        session: &'s Session,
        device: &'s BTreeMap<String, DeviceBuffer>,
        images: &Images,
        live: u32,
        produced: bool,
    ) -> LabResult<Replay<'s>> {
        let mut replay = Replay::new(session);
        for buffer in &self.buffers {
            let name = buffer.name.as_str();
            let target = &device[name];
            replay = match &buffer.role {
                Role::Input if produced && self.produced.as_deref() == Some(name) => replay.state(
                    name,
                    target,
                    images.bytes(name)?.to_vec(),
                    WriteMap::of(buffer.dtype).prefix(buffer.elements(self.capacity))?,
                    Vec::new(),
                ),
                Role::Input | Role::Ids { .. } => replay,
                Role::Output { write, pattern } => replay.output_with(
                    name,
                    target,
                    self.map(buffer, write, images, live)?,
                    pattern,
                ),
                Role::Scratch(may_write) => {
                    replay.scratch_within(name, target, may_write(images, live)?)
                }
                Role::State {
                    write_only,
                    updated,
                } => replay.state(
                    name,
                    target,
                    images.bytes(name)?.to_vec(),
                    self.map(buffer, write_only, images, live)?,
                    updated(images, live)?,
                ),
            };
        }
        Ok(replay)
    }

    /// Compares the eager bytes with the oracle at `live`.
    fn compare(
        &self,
        replay: &Replay<'_>,
        images: &Images,
        live: u32,
        receipt: &mut BTreeMap<String, Value>,
    ) -> LabResult<()> {
        if self.oracle.is_none() {
            return Ok(());
        }
        for (name, expected) in self.expected(images, live)? {
            let eager = replay
                .expected(&name)
                .ok_or_else(|| format!("{name} is not an output or state"))?;
            self.check_expected(&name, expected, eager, images, live, receipt)?;
        }
        Ok(())
    }

    /// Compares one output's or state's `actual` bytes with the oracle.
    fn check_expected(
        &self,
        name: &str,
        expected: Expected,
        actual: &[u8],
        images: &Images,
        live: u32,
        receipt: &mut BTreeMap<String, Value>,
    ) -> LabResult<()> {
        let buffer = self.find(name)?;
        match expected {
            Expected::Bits(bits) => {
                let ranges = match &buffer.role {
                    Role::Output { write, .. } => {
                        self.map(buffer, write, images, live)?.ranges().to_vec()
                    }
                    _ => std::iter::once(0..actual.len()).collect(),
                };
                compare_bits(name, actual, &bits, &ranges)?;
            }
            Expected::Judge(judge) => {
                judge(actual).map_err(|e| format!("{name} at live {live}: {e}"))?;
                receipt.insert(format!("{name}_oracle"), json!({"judged": true}));
            }
            Expected::Values(reference) => {
                let values = decode(buffer.dtype, actual)?;
                if reference.values.len() != values.len() {
                    return Err(format!(
                        "{name} holds {} values, the reference {}",
                        values.len(),
                        reference.values.len()
                    ));
                }
                // The elements the launch writes, at their own indices: an
                // output's write map, a whole state.
                let element = WriteMap::of(buffer.dtype).bytes()?;
                let indices = match &buffer.role {
                    Role::Output { write, .. } => self
                        .map(buffer, write, images, live)?
                        .ranges()
                        .iter()
                        .flat_map(|range| range.start / element..range.end / element)
                        .collect(),
                    _ => (0..values.len()).collect::<Vec<_>>(),
                };
                if indices.is_empty() {
                    return Ok(());
                }
                let pick = |all: &[f64]| indices.iter().map(|&at| all[at]).collect();
                let reference = OracleOutput {
                    values: pick(&reference.values),
                    sum_abs_products: pick(&reference.sum_abs_products),
                    reduction_lengths: indices
                        .iter()
                        .map(|&at| reference.reduction_lengths[at])
                        .collect(),
                };
                let actual = indices.iter().map(|&at| values[at]).collect::<Vec<_>>();
                let mut check = BTreeMap::new();
                if matches!(buffer.dtype, DType::Bf16 | DType::Fp8E4M3) {
                    rounding_admissible(buffer.dtype, &actual, &reference)
                        .map_err(|e| format!("{name} at live {live}: {e}"))?;
                    check.insert("correct".into(), json!(true));
                } else {
                    check_output(&actual, &reference, buffer.dtype, &mut check)
                        .map_err(|e| format!("{name} at live {live}: {e}"))?;
                }
                receipt.insert(format!("{name}_oracle"), json!(check));
            }
        }
        Ok(())
    }

    /// The eager bytes of every output and state.
    fn eager(&self, replay: &Replay<'_>) -> LabResult<BTreeMap<String, Vec<u8>>> {
        self.buffers
            .iter()
            .filter(|buffer| buffer.checked())
            .map(|buffer| {
                let bytes = replay
                    .expected(&buffer.name)
                    .ok_or_else(|| format!("{} has no eager bytes", buffer.name))?;
                Ok((buffer.name.clone(), bytes.to_vec()))
            })
            .collect()
    }

    /// The runs whose exact checks refuse `mutant`, a host model of a wrong
    /// launch giving the bytes it leaves in some outputs and in-place
    /// states: `live N` at each live count, and `high <ids>` and `holes
    /// <ids>` for the reruns of each ID input at the largest count, on that
    /// run's images (a high-id run's pools stay moved). A run refuses the
    /// mutant when the bytes it gives differ from those the run compares:
    /// an output's inside its write map against the oracle's bits, and a
    /// state's whole, against the oracle's bits or, where the state writes
    /// nothing, its initial bytes. A witness test asserts that a mutant
    /// every other check passes is refused somewhere, so a fixture that
    /// stops telling them apart fails on the CPU. Refused when the mutant
    /// gives a buffer that is neither an output nor a state, or one the
    /// oracle gives no bits for where it writes.
    pub fn refusing_runs(
        &self,
        mutant: impl Fn(&Images, u32) -> LabResult<Vec<(String, Vec<u8>)>>,
    ) -> LabResult<Vec<String>> {
        let images = self.images();
        let top = self.top();
        let mut runs = self
            .live_counts
            .iter()
            .map(|&live| (format!("live {live}"), images.clone(), live))
            .collect::<Vec<_>>();
        for buffer in &self.buffers {
            let Role::Ids { pools, holes } = &buffer.role else {
                continue;
            };
            if !pools.is_empty() {
                let (moved, _) = self.high_images(&images, buffer, pools)?;
                runs.push((format!("high {}", buffer.name), moved, top));
            }
            if let Some((_, moved)) = self.hole_images(&images, buffer, *holes)? {
                runs.push((format!("holes {}", buffer.name), moved, top));
            }
        }
        let mut refusing = Vec::new();
        for (run, images, live) in runs {
            let expected = self
                .expected(&images, live)?
                .into_iter()
                .collect::<BTreeMap<_, _>>();
            for (name, bytes) in mutant(&images, live)? {
                let buffer = self.find(&name)?;
                // The bytes the run compares: an output's write map against
                // the oracle's bits, and a state whole, against the oracle's
                // bits or, where it writes nothing, its initial bytes.
                let (bits, compared) = match &buffer.role {
                    Role::Output { write, .. } => {
                        let map = self.map(buffer, write, &images, live)?;
                        if map.written_bytes() == 0 {
                            continue;
                        }
                        let Some(Expected::Bits(bits)) = expected.get(&name) else {
                            return Err(format!("the oracle gives no bits for {name} at {run}"));
                        };
                        (bits.as_slice(), map.ranges().to_vec())
                    }
                    Role::State { .. } => {
                        let bits = match expected.get(&name) {
                            Some(Expected::Bits(bits)) => bits.as_slice(),
                            None if self.writes(buffer, &images, live)? == 0 => {
                                images.bytes(&name)?
                            }
                            _ => {
                                return Err(format!(
                                    "the oracle gives no bits for {name} at {run}"
                                ));
                            }
                        };
                        (bits, std::iter::once(0..bits.len()).collect())
                    }
                    _ => {
                        return Err(format!(
                            "the mutant gives {name}, which is not an output or state"
                        ));
                    }
                };
                if bytes.len() != bits.len() {
                    return Err(format!(
                        "the mutant's {name} holds {} bytes of {}",
                        bytes.len(),
                        bits.len()
                    ));
                }
                if compared
                    .iter()
                    .any(|range| bytes[range.clone()] != bits[range.clone()])
                {
                    refusing.push(run);
                    break;
                }
            }
        }
        Ok(refusing)
    }

    /// Runs every check on `session`. `module` holds the entries; `lab`,
    /// the latent lab module, holds the copy producer of a produced input.
    /// Each entry's function first opts in to its plan's dynamic shared
    /// memory above the default 48 KiB. Returns the evidence for a receipt.
    pub fn run(
        &self,
        session: &Session,
        module: &Module,
        lab: Option<&Module>,
    ) -> LabResult<Value> {
        self.run_with(
            session,
            &|symbol| module.function(symbol).map_err(|e| e.to_string()),
            // The copy producer loads only for a produced input, as before.
            Some(&Probes::sm120(
                session,
                lab.filter(|_| self.produced.is_some()),
            )?),
        )
    }

    /// [`Case::run`] for a chain whose entries live in several modules:
    /// each stage's function from the first of `modules` that holds it.
    pub fn run_modules(
        &self,
        session: &Session,
        modules: &[&Module],
        lab: Option<&Module>,
    ) -> LabResult<Value> {
        self.run_with(
            session,
            &|symbol| {
                modules
                    .iter()
                    .find_map(|module| module.function(symbol).ok())
                    .ok_or_else(|| format!("no module holds {symbol}"))
            },
            // The copy producer loads only for a produced input, as in `run`.
            Some(&Probes::sm120(
                session,
                lab.filter(|_| self.produced.is_some()),
            )?),
        )
    }

    /// [`Case::run`] with each stage's function from `functions`, by
    /// symbol, and the PDL phases behind `probes`; without probes the run
    /// skips them and its produced input is a plain input. A prepared op's
    /// views take the device addresses of the case's buffers as handles.
    pub(crate) fn run_with(
        &self,
        session: &Session,
        functions: &dyn Fn(&str) -> LabResult<Function>,
        probes: Option<&Probes>,
    ) -> LabResult<Value> {
        self.validate()?;
        let symbol = self.symbol();
        let mut device = BTreeMap::new();
        for buffer in &self.buffers {
            device.insert(
                buffer.name.clone(),
                session.allocate(buffer.bytes(self.capacity)?)?,
            );
        }
        let live_buffer = session.allocate(16)?;
        // What the live-count producer holds in place of a step input count:
        // a count no launch reads.
        let held = session.allocate(16)?;
        poison::write(session, &held, 0, &0_u32.to_le_bytes())?;
        let staged = match (&self.produced, probes) {
            (Some(name), Some(_)) => {
                Some((name.as_str(), session.allocate(device[name].byte_len())?))
            }
            _ => None,
        };
        let copy = match (&staged, probes) {
            (
                Some(_),
                Some(Probes {
                    copy: Some(copy), ..
                }),
            ) => Some(copy),
            (Some(_), _) => {
                return Err(format!("{symbol}: a produced input needs the lab module"));
            }
            (None, _) => None,
        };
        let produced = staged.is_some();
        let rewrites = match &staged {
            Some((name, staged)) => Some(Rewrites::new(session, &[(staged, &device[*name])])?),
            None => None,
        };
        let upload = |images: &Images| -> LabResult<()> {
            for buffer in self.buffers.iter().filter(|buffer| buffer.uploaded()) {
                poison::write(
                    session,
                    &device[&buffer.name],
                    0,
                    images.bytes(&buffer.name)?,
                )?;
            }
            if let Some((name, staged)) = &staged {
                poison::write(session, staged, 0, images.bytes(name)?)?;
            }
            Ok(())
        };
        let handle = address(&device, &live_buffer, &self.live);
        // The stages at a row count, each function loaded and opted in to
        // its plan's dynamic shared memory.
        let load = |rows: u32| -> LabResult<Vec<Loaded<'_>>> {
            let mut loaded = Vec::new();
            for stage in self.stages(rows, &handle)? {
                let Stage {
                    symbol,
                    plan,
                    arguments,
                    names,
                } = stage;
                let at = |e: String| format!("{symbol}: {e}");
                let function = functions(&symbol).map_err(at)?;
                if plan.dynamic_shared_bytes > DEFAULT_SHARED_BYTES {
                    function
                        .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                        .map_err(|e| at(e.to_string()))?;
                }
                let bindings = bindings(&names, &device, &live_buffer, &self.live);
                loaded.push((symbol, function, plan, arguments, bindings));
            }
            Ok(loaded)
        };
        // The stages in order but the one at `skip`, each with PDL when
        // `pdl` and its plan allow.
        let launch_stages =
            |stages: &[Loaded<'_>], stream: &Stream, pdl: bool, skip: Option<usize>| {
                for (at, (symbol, function, plan, arguments, bindings)) in stages.iter().enumerate()
                {
                    if Some(at) == skip {
                        continue;
                    }
                    let plan = LaunchPlan {
                        pdl: pdl && plan.pdl,
                        ..*plan
                    };
                    launch::launch(
                        &session.context,
                        function,
                        stream,
                        plan,
                        arguments,
                        bindings,
                    )
                    .map_err(|e| format!("{symbol}: {e}"))?;
                }
                Ok(())
            };
        let run = |stages: &[Loaded<'_>], stream: &Stream, pdl: bool| {
            launch_stages(stages, stream, pdl, None)
        };
        let produce = |stream: &Stream| match (&staged, &copy) {
            (Some((name, staged)), Some(copy)) => copy.copy(stream, staged, &device[*name]),
            _ => Ok(()),
        };
        let chain = |stages: &[Loaded<'_>], stream: &Stream, pdl: bool| {
            produce(stream)?;
            run(stages, stream, pdl)
        };
        let stream = &session.stream;
        let set_live = |live: u32| poison::write(session, &live_buffer, 0, &live.to_le_bytes());
        let images = self.images();
        upload(&images)?;
        // The stages and graph of each row count the run plans at: the
        // capacity, or each positive live count when the stages follow
        // the host's rows.
        let rows_at = |live: u32| if self.host_rows { live } else { self.capacity };
        let mut planned = BTreeMap::new();
        for &live in &self.live_counts {
            let rows = rows_at(live);
            if rows == 0 || planned.contains_key(&rows) {
                continue;
            }
            let stages = load(rows)?;
            let graph = session.capture(|stream| chain(&stages, stream, true))?;
            planned.insert(rows, (stages, graph));
        }
        let top = *self
            .live_counts
            .iter()
            .max()
            .expect("validated live counts");
        let mut evidence = BTreeMap::new();
        let mut baseline = BTreeMap::new();
        for &live in &self.live_counts {
            let at = |e: String| format!("{symbol} live {live}: {e}");
            set_live(live)?;
            let mut receipt = BTreeMap::new();
            let Some((stages, graph)) = planned.get(&rows_at(live)) else {
                // The host launches no rows: every output and state keeps
                // its poison or initial bytes, which no launch can break.
                receipt.insert("launched".into(), json!(false));
                evidence.insert(format!("live_{live}"), json!(receipt));
                continue;
            };
            // Whether the first stage waits on a PDL producer.
            let pdl = stages[0].2.pdl;
            let mut replay = self.replay(session, &device, &images, live, produced)?;
            replay.eager(|| chain(stages, stream, false)).map_err(at)?;
            self.compare(&replay, &images, live, &mut receipt)
                .map_err(at)?;
            replay
                .relaunch(|| chain(stages, stream, false))
                .map_err(at)?;
            replay.graph(graph).map_err(at)?;
            let mut producers = Vec::new();
            // Without probes the run has no PDL phases.
            let phases = probes.map_or_else(Vec::new, |probes| {
                self.pdl_producers()
                    .into_iter()
                    .map(|producer| (producer, &probes.live))
                    .collect()
            });
            for (producer, live_producer) in phases {
                let count = match producer {
                    PdlProducer::Live { count: false, .. } => &held,
                    _ => &live_buffer,
                };
                let launch = || match (producer, &rewrites) {
                    (PdlProducer::Copy, _) => produce(stream),
                    (PdlProducer::Live { produced: true, .. }, Some(rewrites)) => {
                        live_producer.launch_rewriting(session, count, rewrites)
                    }
                    _ => live_producer.launch(session, count),
                };
                replay
                    .pdl(launch, || run(stages, stream, true), pdl)
                    .map_err(|e| at(format!("{}: {e}", producer.describe())))?;
                producers.push(producer.describe());
            }
            replay.record(&mut receipt);
            receipt.insert("pdl_producers".into(), json!(producers));
            if live == top {
                baseline = self.eager(&replay)?;
            }
            evidence.insert(format!("live_{live}"), json!(receipt));
        }

        // A launch that writes nothing must fail the map checks or the
        // oracle at every count where something writes, and so must a
        // chain without any one of its entries.
        let null_counts = self.null_counts(&images)?;
        let mut guarded_omissions = Vec::new();
        for &live in &null_counts {
            set_live(live)?;
            let mut replay = self.replay(session, &device, &images, live, false)?;
            if replay.eager(|| Ok(())).is_ok()
                && self
                    .compare(&replay, &images, live, &mut BTreeMap::new())
                    .is_ok()
            {
                return Err(format!(
                    "{symbol}: a launch that writes nothing passes the checks at live {live}"
                ));
            }
            let (stages, _) = &planned[&rows_at(live)];
            let symbols = stages
                .iter()
                .map(|stage| stage.0.clone())
                .collect::<Vec<_>>();
            let dropped = if stages.len() > 1 {
                0..stages.len()
            } else {
                0..0
            };
            for skip in dropped {
                if self.redundant.contains_key(&stages[skip].0) {
                    continue;
                }
                let mut replay = self.replay(session, &device, &images, live, false)?;
                replay.prepare(crate::poison::Phase::Eager)?;
                produce(stream)?;
                let rejected = self.omit_stage(
                    &symbols,
                    skip,
                    &images,
                    live,
                    |name| poison::read(session, &device[name]),
                    |at| launch_stages(&stages[at..=at], stream, false, None),
                )?;
                // A launch or read failure is not an expected negative
                // control. Fence it before interpreting write-map errors.
                stream
                    .synchronize()
                    .map_err(|e| format!("omitting {} at live {live}: {e}", stages[skip].0))?;
                if let Some(error) = rejected {
                    guarded_omissions.push(json!({
                        "stage": stages[skip].0, "live": live, "rejected": error,
                    }));
                    continue;
                }
                let passes = replay.finish(crate::poison::Phase::Eager, false).is_ok()
                    && self
                        .compare(&replay, &images, live, &mut BTreeMap::new())
                        .is_ok();
                if passes {
                    return Err(format!(
                        "{symbol}: the chain without {} passes the checks at live {live}",
                        stages[skip].0
                    ));
                }
            }
        }
        let no_write = self
            .no_write
            .iter()
            .map(|((name, live), reason)| json!({"buffer": name, "live": live, "reason": reason}))
            .collect::<Vec<_>>();

        let (stages, _) = planned
            .get(&rows_at(top))
            .ok_or_else(|| format!("{symbol}: no launch at the largest live count"))?;
        let mut high = Vec::new();
        let mut holes = Vec::new();
        for buffer in &self.buffers {
            let Role::Ids { pools, holes: hole } = &buffer.role else {
                continue;
            };
            let name = buffer.name.as_str();
            if !pools.is_empty() {
                let (moved, order) = self.high_images(&images, buffer, pools)?;
                upload(&moved)?;
                set_live(top)?;
                let mut replay = self.replay(session, &device, &moved, top, produced)?;
                replay
                    .eager(|| chain(stages, stream, false))
                    .map_err(|e| format!("{symbol} high {name}: {e}"))?;
                for (output, mut bytes) in self.eager(&replay)? {
                    if let Some((_, stride)) = pools.iter().find(|(pool, _)| *pool == output) {
                        bytes = permute_rows(&bytes, *stride, &order, true);
                    }
                    compare_bits(
                        &format!("{output} with high {name}"),
                        &bytes,
                        &baseline[&output],
                        std::slice::from_ref(&(0..bytes.len())),
                    )
                    .map_err(|e| format!("{symbol}: {e}"))?;
                }
                high.push(json!({"ids": name, "exercised": true, "top": order.len()}));
            } else {
                high.push(json!({"ids": name, "exercised": false, "reason": "no pools"}));
            }
            if let Some((row, moved)) = self.hole_images(&images, buffer, *hole)? {
                upload(&moved)?;
                set_live(top)?;
                let mut receipt = BTreeMap::new();
                let mut replay = self.replay(session, &device, &moved, top, produced)?;
                let at = |e: String| format!("{symbol} {name} row {row} all holes: {e}");
                replay.eager(|| chain(stages, stream, false)).map_err(at)?;
                self.compare(&replay, &moved, top, &mut receipt)
                    .map_err(at)?;
                holes.push(json!({"ids": name, "exercised": true, "row": row, "receipt": receipt}));
            } else {
                holes.push(json!({"ids": name, "exercised": false, "reason": "Holes::Never"}));
            }
            upload(&images)?;
        }
        let stages = stages
            .iter()
            .map(|(symbol, _, plan, ..)| {
                json!({"symbol": symbol, "dynamic_shared_bytes": plan.dynamic_shared_bytes})
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "symbol": symbol,
            "stages": stages,
            "capacity": self.capacity,
            "rows": if self.host_rows { "the live count" } else { "the capacity" },
            "live_counts": evidence,
            "null_launch_refused_at": null_counts,
            "no_write": no_write,
            "redundant_stages": self.redundant,
            "omission_guards": self.omission_guards,
            "guarded_omissions": guarded_omissions,
            "live_count_before_wait": self.step_live,
            "high_ids": high,
            "hole_rows": holes,
        }))
    }
}

/// The handle a CPU check gives buffer `name`: its place among `names`,
/// from 1.
fn token(names: &BTreeSet<&str>, name: &str) -> u64 {
    names
        .iter()
        .position(|&other| other == name)
        .map_or(0, |at| at as u64 + 1)
}

/// The handle a run gives buffer `name`: its device address.
fn address<'d>(
    device: &'d BTreeMap<String, DeviceBuffer>,
    live_buffer: &'d DeviceBuffer,
    live: &'d str,
) -> impl Fn(&str) -> u64 + 'd {
    move |name| {
        let buffer = if name == live {
            live_buffer
        } else {
            match device.get(name) {
                Some(buffer) => buffer,
                None => return 0,
            }
        };
        buffer.device_address_at(0, buffer.byte_len()).unwrap_or(0)
    }
}

/// The device buffers a stage's handles name.
fn bindings<'d>(
    names: &BTreeMap<u64, String>,
    device: &'d BTreeMap<String, DeviceBuffer>,
    live_buffer: &'d DeviceBuffer,
    live: &str,
) -> launch::Bindings<'d> {
    launch::Bindings {
        buffers: names
            .iter()
            .map(|(&handle, name)| {
                let buffer = if name == live {
                    live_buffer
                } else {
                    &device[name]
                };
                (handle, buffer)
            })
            .collect(),
    }
}

/// Every device handle `arguments` carry, with the argument holding it:
/// buffers, tensor-map bases and pointer-array entries.
fn handles(arguments: &ArgumentList) -> Vec<(usize, u64)> {
    let mut handles = Vec::new();
    for (at, argument) in arguments.0.iter().enumerate() {
        match argument {
            Argument::DeviceBuffer(handle) => handles.push((at, *handle)),
            Argument::TensorMap(map) => handles.push((at, map.base_buffer)),
            Argument::DevicePointerArray(array) => {
                handles.extend(array.iter().map(|&handle| (at, handle)));
            }
            _ => {}
        }
    }
    handles
}

/// Binds `params` by name: device pointers to the buffers in `buffers`
/// (handles 1, 2, ... in parameter order, returned by name), scalars to
/// `scalars` of the same kind. Every buffer and scalar must be a parameter.
pub fn bind(
    params: &[EntryParam],
    buffers: &BTreeSet<&str>,
    scalars: &BTreeMap<String, Argument>,
) -> LabResult<(Vec<Argument>, Vec<String>)> {
    let mut used = BTreeSet::new();
    let bound = bind_used(params, buffers, scalars, &mut used)?;
    unused(buffers, scalars, &used)?;
    Ok(bound)
}

/// Binds each entry's parameters as [`bind`] does; every buffer and scalar
/// must be a parameter of some entry.
fn bind_chain(
    stages: &[&dyn LabEntry],
    buffers: &BTreeSet<&str>,
    scalars: &BTreeMap<String, Argument>,
) -> LabResult<Vec<(Vec<Argument>, Vec<String>)>> {
    let mut used = BTreeSet::new();
    let bound = stages
        .iter()
        .map(|stage| {
            bind_used(&stage.params(), buffers, scalars, &mut used).map_err(|e| {
                if stages.len() > 1 {
                    format!("{}: {e}", stage.symbol())
                } else {
                    e
                }
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    unused(buffers, scalars, &used)?;
    Ok(bound)
}

/// Refuses a buffer or scalar outside `used`.
fn unused(
    buffers: &BTreeSet<&str>,
    scalars: &BTreeMap<String, Argument>,
    used: &BTreeSet<String>,
) -> LabResult<()> {
    match buffers
        .iter()
        .map(|name| name.to_string())
        .chain(scalars.keys().cloned())
        .find(|name| !used.contains(name))
    {
        Some(unused) => Err(format!("{unused} is not a parameter")),
        None => Ok(()),
    }
}

/// [`bind`] without the check that every buffer and scalar is a parameter;
/// adds the names it binds to `used`.
fn bind_used(
    params: &[EntryParam],
    buffers: &BTreeSet<&str>,
    scalars: &BTreeMap<String, Argument>,
    used: &mut BTreeSet<String>,
) -> LabResult<(Vec<Argument>, Vec<String>)> {
    let mut arguments = Vec::with_capacity(params.len());
    let mut bound = Vec::new();
    let mut names = BTreeSet::new();
    for param in params {
        let argument = match param {
            EntryParam::DevicePointer { name, .. } => {
                if !buffers.contains(name.as_str()) {
                    return Err(format!("parameter {name} has no buffer"));
                }
                bound.push(name.clone());
                Argument::DeviceBuffer(bound.len() as u64)
            }
            EntryParam::U32 { name } | EntryParam::U64 { name } | EntryParam::I32 { name } => {
                let value = scalars
                    .get(name)
                    .ok_or_else(|| format!("parameter {name} has no scalar"))?;
                let kind = matches!(
                    (param, value),
                    (EntryParam::U32 { .. }, Argument::U32(_))
                        | (EntryParam::U64 { .. }, Argument::U64(_))
                        | (EntryParam::I32 { .. }, Argument::I32(_))
                );
                if !kind {
                    return Err(format!(
                        "scalar {name} is {value:?}, another kind than its parameter"
                    ));
                }
                value.clone()
            }
            other => return Err(format!("parameter {other:?} cannot be declared")),
        };
        let name = match param {
            EntryParam::DevicePointer { name, .. }
            | EntryParam::U32 { name }
            | EntryParam::U64 { name }
            | EntryParam::I32 { name } => name.as_str(),
            _ => unreachable!("refused above"),
        };
        if !names.insert(name.to_owned()) {
            return Err(format!("parameter {name} appears twice"));
        }
        arguments.push(argument);
    }
    used.extend(names);
    Ok((arguments, bound))
}

/// A permutation of a pool's `rows` (`order[old] == new`) moving the `used`
/// rows, in order, to the top of the pool; the rows they displace take the
/// freed places in order.
pub fn high_permutation(used: &BTreeSet<usize>, rows: usize) -> LabResult<Vec<usize>> {
    if let Some(&last) = used.last()
        && last >= rows
    {
        return Err(format!("row {last} lies past a pool of {rows} rows"));
    }
    let top = rows - used.len();
    let mut order = vec![usize::MAX; rows];
    for (index, &row) in used.iter().enumerate() {
        order[row] = top + index;
    }
    let mut free = (0..rows).filter(|row| !(top..rows).contains(row));
    for slot in order.iter_mut().filter(|slot| **slot == usize::MAX) {
        *slot = free.next().expect("as many free rows as unused ones");
    }
    Ok(order)
}

/// LE I32 `ids` moved by `order`; -1 holes stay.
pub fn remap_ids(ids: &[i32], order: &[usize]) -> Vec<u8> {
    ids.iter()
        .map(|&id| usize::try_from(id).map_or(id, |id| order[id] as i32))
        .flat_map(i32::to_le_bytes)
        .collect()
}

/// `image` with its first `order.len()` rows of `stride` bytes moved to
/// `order[row]`, or back when `inverse`; bytes past those rows stay.
pub fn permute_rows(image: &[u8], stride: usize, order: &[usize], inverse: bool) -> Vec<u8> {
    let mut moved = image.to_vec();
    for (row, &to) in order.iter().enumerate() {
        let (from, to) = if inverse { (to, row) } else { (row, to) };
        moved[to * stride..][..stride].copy_from_slice(&image[from * stride..][..stride]);
    }
    moved
}

/// The first output or state in both `first` and `other` whose bytes
/// differ.
fn differs<'m>(
    first: &'m BTreeMap<String, Vec<u8>>,
    other: &BTreeMap<String, Vec<u8>>,
) -> Option<&'m str> {
    first
        .iter()
        .find(|(name, bytes)| other.get(*name).is_some_and(|other| other != *bytes))
        .map(|(name, _)| name.as_str())
}

/// The first output or state whose exact results in `first` and `high`
/// are the same and differ from those in `stale`. A buffer exact only in
/// the reruns is no witness: the run checks the first launch against its
/// numeric reference and the rerun only against the first launch's bytes,
/// so the rerun's exact bits are never compared.
fn witness<'m>(
    first: &'m BTreeMap<String, Vec<u8>>,
    high: &BTreeMap<String, Vec<u8>>,
    stale: &BTreeMap<String, Vec<u8>>,
) -> Option<&'m str> {
    first
        .iter()
        .find(|(name, bytes)| {
            high.get(*name) == Some(*bytes) && stale.get(*name).is_some_and(|stale| stale != *bytes)
        })
        .map(|(name, _)| name.as_str())
}

/// Why buffers with numeric references gave no witness, for an error.
fn numeric_note(numeric: &BTreeSet<String>) -> String {
    if numeric.is_empty() {
        return String::new();
    }
    let names = numeric.iter().cloned().collect::<Vec<_>>().join(", ");
    format!(
        " (the numeric references of {names} never count: one result can pass two \
         overlapping tolerance gates)"
    )
}

/// `ranges` sorted, with empty ranges dropped and overlapping or adjacent
/// ones joined, so the same bytes give the same ranges in any order.
fn union(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.retain(|range| range.start < range.end);
    ranges.sort_by_key(|range| range.start);
    let mut joined: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match joined.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => joined.push(range),
        }
    }
    joined
}

/// Checks `actual` against `expected` bit for bit inside `ranges`.
fn compare_bits(
    name: &str,
    actual: &[u8],
    expected: &[u8],
    ranges: &[Range<usize>],
) -> LabResult<()> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{name} holds {} bytes, the reference {}",
            actual.len(),
            expected.len()
        ));
    }
    for range in ranges {
        if let Some(at) = actual[range.clone()]
            .iter()
            .zip(&expected[range.clone()])
            .position(|(a, b)| a != b)
        {
            let at = range.start + at;
            return Err(format!(
                "{name} byte {at} is {:#04x}, the reference {:#04x}",
                actual[at], expected[at]
            ));
        }
    }
    Ok(())
}

/// The value `dtype`'s correct rounding gives `value`: BF16 or E4M3 (in
/// the code's units, saturating at 448), ties to even, in one rounding.
pub(crate) fn round_to(dtype: DType, value: f64) -> LabResult<f64> {
    Ok(match dtype {
        DType::Bf16 => f64::from(bf16_to_f32(kernel_core::codec::f64_to_bf16(value))),
        DType::Fp8E4M3 => kernel_core::kv_fp8::round_e4m3(value),
        other => return Err(format!("{other:?} values are not rounded")),
    })
}

/// The lowest and highest BF16 or E4M3 values (E4M3 in the code's units)
/// correct rounding ([`round_to`]) gives an F32 within `bound` of `value`.
/// A device quantizes an F32, so the interval's F64 ends round inward to
/// the F32 values it holds before quantizing; an interval holding none
/// (narrower than the F32 spacing, as with no bound) takes its ends'
/// nearest F32 values. Rounding is monotonic, so the values between the
/// two are exactly those of the interval's F32 values.
pub(crate) fn admitted_range(dtype: DType, value: f64, bound: f64) -> LabResult<(f64, f64)> {
    let (low, high) = (value - bound, value + bound);
    let (mut up, mut down) = (low as f32, high as f32);
    if f64::from(up) < low {
        up = up.next_up();
    }
    if f64::from(down) > high {
        down = down.next_down();
    }
    if up > down {
        (up, down) = (low as f32, high as f32);
    }
    Ok((
        round_to(dtype, f64::from(up))?,
        round_to(dtype, f64::from(down))?,
    ))
}

/// Checks BF16 or E4M3 values (E4M3 in the code's units) against
/// `reference`: each must be a value correct rounding gives some F32
/// within the element's bound `2^-22 K S` of the reference
/// ([`admitted_range`]).
pub(crate) fn rounding_admissible(
    dtype: DType,
    actual: &[f32],
    reference: &OracleOutput,
) -> LabResult<()> {
    let label = match dtype {
        DType::Bf16 => "BF16",
        _ => "E4M3",
    };
    for (at, &value) in actual.iter().enumerate() {
        let (expected, bound) = (
            reference.values[at],
            kernel_core::oracle::accumulation_bound(
                reference.reduction_lengths[at],
                reference.sum_abs_products[at],
            ),
        );
        let (low, high) = admitted_range(dtype, expected, bound)?;
        let value = f64::from(value);
        if !(low <= value && value <= high) {
            return Err(format!(
                "{label} element {at} is {value}, outside {low} to {high}, the roundings of \
                 {expected} within {bound}"
            ));
        }
    }
    Ok(())
}

/// The values of a BF16, F32 or FP8 E4M3 image, codes in their own units.
fn decode(dtype: DType, bytes: &[u8]) -> LabResult<Vec<f32>> {
    Ok(match dtype {
        DType::Fp8E4M3 => bytes
            .iter()
            .map(|&code| kernel_core::codec::SmallFloat::E4M3.decode_f64(code) as f32)
            .collect(),
        DType::Bf16 => bytes
            .chunks_exact(2)
            .map(|half| bf16_to_f32(u16::from_le_bytes([half[0], half[1]])))
            .collect(),
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("4 bytes")))
            .collect(),
        other => return Err(format!("{other:?} outputs compare bits, not values")),
    })
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;
    use kernel_core::codec::f32_to_bf16;
    use kernel_core::contract::Refusal;

    /// E4M3 values pass only as codes correct rounding gives some real
    /// within the oracle's bound of the reference: without a bound, the
    /// correctly rounded code alone, saturated at 448.
    #[test]
    fn e4m3_values_are_the_codes_of_the_reference_within_its_bound() {
        let check = |reference: f64, bound: f64, actual: u8| {
            // K = 1, so 2^-22 K S is the bound.
            let reference =
                OracleOutput::new(vec![reference], vec![bound * 2_f64.powi(22)], 1).unwrap();
            rounding_admissible(
                DType::Fp8E4M3,
                &decode(DType::Fp8E4M3, &[actual]).unwrap(),
                &reference,
            )
        };
        // 1.0 is 0x38; 0x37 is 0.9375 and 0x39 1.125.
        assert!(check(1.0, 0.0, 0x38).is_ok());
        assert!(check(1.0, 0.0, 0x37).is_err());
        assert!(check(1.0, 0.0, 0x39).is_err());
        // 1.06 rounds to 1.0, and the midpoint 1.0625 to the even 1.0.
        assert!(check(1.06, 0.0, 0x38).is_ok());
        assert!(check(1.06, 0.0, 0x39).is_err());
        assert!(check(1.0625, 0.0, 0x38).is_ok());
        assert!(check(1.0625, 0.0, 0x39).is_err());
        // Past 448 the code saturates (0x7e is 448).
        assert!(check(500.0, 0.0, 0x7e).is_ok());
        assert!(check(-500.0, 0.0, 0xfe).is_ok());
        // A bound of 0.1 admits the codes of 0.9 to 1.1: 0.875 to 1.125.
        assert!(check(1.0, 0.1, 0x37).is_ok());
        assert!(check(1.0, 0.1, 0x39).is_ok());
        assert!(check(1.0, 0.1, 0x36).is_ok());
        assert!(check(1.0, 0.1, 0x35).is_err());
        assert!(check(1.0, 0.1, 0x3a).is_err());
    }

    /// BF16 values pass only as values correct rounding gives some real
    /// within the oracle's bound of the reference: past a binade boundary a
    /// step of the next binade, and nothing a rounding of the interval
    /// cannot give.
    #[test]
    fn bf16_values_are_the_roundings_of_the_reference_within_its_bound() {
        let check = |reference: f64, bound: f64, actual: f64| {
            let reference =
                OracleOutput::new(vec![reference], vec![bound * 2_f64.powi(22)], 1).unwrap();
            rounding_admissible(DType::Bf16, &[actual as f32], &reference)
        };
        let step = |exponent: i32| 2_f64.powi(exponent);
        // Both ends of 0.9999 within 0.0002 round to 1, so the value below
        // fails although it is within half a step of 0.9999 and the bound.
        assert!(check(0.9999, 0.0002, 1.0).is_ok());
        assert!(check(0.9999, 0.0002, 1.0 - step(-8)).is_err());
        // Up past 1, a real within the bound rounds to 1 + one step of
        // [1, 2), more than half a step at the reference; two steps fail.
        assert!(check(0.999_684_7, 0.004_811, 1.0 + step(-7)).is_ok());
        assert!(check(0.999_684_7, 0.004_811, 1.0 + step(-6)).is_err());
        // Down past 1, the interval's low end still rounds to 1.
        assert!(check(1.0001, 0.0003, 1.0).is_ok());
        assert!(check(1.0001, 0.0003, 1.0 - step(-8)).is_err());
        // An end on a midpoint rounds to the even value: 1 + 2^-8 to 1.
        assert!(check(1.0, step(-8), 1.0).is_ok());
        assert!(check(1.0, step(-8), 1.0 + step(-7)).is_err());
        // Without a bound, the correctly rounded value alone.
        let nearest = f64::from(bf16_to_f32(f32_to_bf16(-0.3)));
        assert!(check(-0.3, 0.0, nearest).is_ok());
        assert!(check(-0.3, 0.0, nearest + step(-9)).is_err());
    }

    /// The gate rounds the F32 values within the bound, not its F64 ends:
    /// an interval ending less than an F32 step past a midpoint holds the
    /// midpoint at most, which rounds to even, so the value past it fails
    /// although the F64 end rounds to it. An interval holding no F32 takes
    /// the rounding of its ends' nearest F32 values.
    #[test]
    fn the_gate_rounds_the_f32_values_within_the_bound() {
        let check = |dtype: DType, reference: f64, bound: f64, actual: f64| {
            let reference =
                OracleOutput::new(vec![reference], vec![bound * 2_f64.powi(22)], 1).unwrap();
            rounding_admissible(dtype, &[actual as f32], &reference)
        };
        let step = |exponent: i32| 2_f64.powi(exponent);
        let (below, past) = (step(-23), step(-23) + step(-26));
        // BF16: 1 + 2^-8 is the midpoint of 1 and 1 + 2^-7.
        let mid = 1.0 + step(-8);
        assert_eq!(
            round_to(DType::Bf16, mid - below + past).unwrap(),
            1.0 + step(-7)
        );
        assert_eq!(
            admitted_range(DType::Bf16, mid - below, past).unwrap(),
            (1.0, 1.0)
        );
        assert!(check(DType::Bf16, mid - below, past, 1.0).is_ok());
        assert!(check(DType::Bf16, mid - below, past, 1.0 + step(-7)).is_err());
        // E4M3: 1.0625 is the midpoint of codes 1.0 and 1.125.
        let mid = 1.0625;
        assert_eq!(round_to(DType::Fp8E4M3, mid - below + past).unwrap(), 1.125);
        assert!(check(DType::Fp8E4M3, mid - below, past, 1.0).is_ok());
        assert!(check(DType::Fp8E4M3, mid - below, past, 1.125).is_err());
        // An interval just past the midpoint holding no F32: both ends'
        // nearest F32 is the midpoint, so both round to 1.
        let (reference, bound) = (mid + step(-26), step(-27));
        assert_eq!(round_to(DType::Fp8E4M3, reference).unwrap(), 1.125);
        assert_eq!(
            admitted_range(DType::Fp8E4M3, reference, bound).unwrap(),
            (1.0, 1.0)
        );
        // The low end rounds inward too: up from just below the midpoint
        // 1.1875 to the midpoint, which rounds to the even 1.25.
        let mid = 1.1875;
        assert_eq!(round_to(DType::Fp8E4M3, mid - step(-26)).unwrap(), 1.125);
        assert_eq!(
            admitted_range(DType::Fp8E4M3, mid + step(-24), step(-24) + step(-26)).unwrap(),
            (1.25, 1.25)
        );
    }

    /// An entry gathering pool rows by ID into its live output rows.
    struct Gather;

    impl LabEntry for Gather {
        fn symbol(&self) -> String {
            "chop_gather_v1".into()
        }

        fn params(&self) -> Vec<EntryParam> {
            let pointer = |name: &str| EntryParam::DevicePointer {
                name: name.into(),
                cxx_type: "uint16_t".into(),
            };
            vec![
                pointer("ids"),
                pointer("pool"),
                pointer("output"),
                pointer("live_rows"),
                EntryParam::U32 {
                    name: "width".into(),
                },
            ]
        }

        fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
            if prepared_rows == 0 || prepared_rows > 8 {
                return Err(Refusal("rows exceed the capacity"));
            }
            Ok(LaunchPlan {
                grid: [prepared_rows, 1, 1],
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: true,
            })
        }
    }

    const WIDTH: usize = 4;

    #[test]
    fn omitted_metadata_producer_is_rejected_before_its_consumer() {
        use std::cell::Cell;
        let entry = Gather;
        let make = || {
            Case::new(&entry, 8).output(
                "padded_rows",
                DType::I32,
                (Rows::Fixed(1), 1),
                Write::EveryCount(Box::new(|_, _| WriteMap::of(DType::I32).prefix(1))),
            )
        };
        let symbols = vec!["route".into(), "down".into()];
        let poisoned_count = vec![POISON; 4];
        assert_eq!(
            i32::from_le_bytes(poisoned_count.clone().try_into().unwrap()),
            -1
        );
        let consumer_calls = Cell::new(0);
        let consumer = |at| {
            assert_eq!(at, 1);
            consumer_calls.set(consumer_calls.get() + 1);
            Err("simulated device trap: padded row count is not a whole tile".into())
        };
        // Without a guard the dropped route feeds -1 to the down kernel.
        let unguarded = make();
        let error = unguarded
            .omit_stage(
                &symbols,
                0,
                &Images::default(),
                8,
                |_| Ok(poisoned_count.clone()),
                consumer,
            )
            .unwrap_err();
        assert!(error.contains("simulated device trap"));
        assert_eq!(consumer_calls.get(), 1);

        consumer_calls.set(0);
        let guarded = make().omission_guard("route", "padded_rows");
        let error = guarded
            .omit_stage(
                &symbols,
                0,
                &Images::default(),
                8,
                |_| Ok(poisoned_count.clone()),
                consumer,
            )
            .unwrap()
            .expect("the unwritten metadata must be rejected");
        assert!(error.contains("the launch did not write it"), "{error}");
        assert_eq!(consumer_calls.get(), 0);

        // If a prior stage already wrote the output, still run the suffix:
        // the guard must not hide a genuinely redundant producer.
        guarded
            .omit_stage(
                &symbols,
                0,
                &Images::default(),
                8,
                |_| Ok(vec![0; 4]),
                |at| {
                    assert_eq!(at, 1);
                    consumer_calls.set(consumer_calls.get() + 1);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(consumer_calls.get(), 1);

        let error = guarded
            .omit_stage(
                &symbols,
                0,
                &Images::default(),
                8,
                |_| Err("injected metadata read failure".into()),
                consumer,
            )
            .unwrap_err();
        assert_eq!(error, "injected metadata read failure");
        assert_eq!(consumer_calls.get(), 1);
    }

    #[test]
    fn omission_guards_require_a_bound_output_and_later_consumer() {
        let entry = Gather;
        let mut guarded = case(&entry).omission_guard("producer", "output");
        let stage = |symbol: &str, names: &[&str]| super::Stage {
            symbol: symbol.into(),
            plan: entry.plan(8).unwrap(),
            arguments: ArgumentList(vec![]),
            names: names
                .iter()
                .enumerate()
                .map(|(i, name)| (i as u64 + 1, (*name).into()))
                .collect(),
        };
        let stages = [
            stage("producer", &["output"]),
            stage("consumer", &["output"]),
        ];
        guarded.validate_launched(&stages).unwrap();
        assert!(
            guarded
                .validate_launched(&stages[..1])
                .unwrap_err()
                .contains("later consumer")
        );
        guarded.omission_guards = BTreeMap::from([("missing".into(), "output".into())]);
        assert!(
            guarded
                .validate_launched(&stages)
                .unwrap_err()
                .contains("not launched")
        );
        guarded.omission_guards = BTreeMap::from([("producer".into(), "pool".into())]);
        assert!(
            guarded
                .validate_launched(&stages)
                .unwrap_err()
                .contains("not an output")
        );
    }

    fn pool() -> Vec<u8> {
        (0..16 * WIDTH as u16).flat_map(u16::to_le_bytes).collect()
    }

    fn case(entry: &dyn LabEntry) -> Case<'_> {
        Case::new(entry, 8)
            .ids(
                "ids",
                (Rows::Live, 1),
                &[3, 0, 9, 9, -1, 5, 1, 2],
                &[("pool", WIDTH * 2)],
                Holes::LastLive,
            )
            .input("pool", DType::Bf16, (Rows::Fixed(16), WIDTH), pool())
            .output("output", DType::Bf16, (Rows::Live, WIDTH), Write::LiveRows)
            .scalar("width", Argument::U32(WIDTH as u32))
            .oracle(gather)
    }

    /// The gather's output: each live row's pool row, zeros for a hole.
    fn gather(images: &Images, live: u32) -> LabResult<Vec<(String, Expected)>> {
        let ids = images.i32s("ids")?;
        let pool = images.bytes("pool")?;
        let row = WIDTH * 2;
        let mut output = vec![0; 8 * row];
        for (at, &id) in ids.iter().take(live as usize).enumerate() {
            if let Ok(id) = usize::try_from(id) {
                output[at * row..][..row].copy_from_slice(&pool[id * row..][..row]);
            }
        }
        Ok(vec![("output".into(), Expected::Bits(output))])
    }

    /// The unit write checks accept a launch that wrote its live output rows
    /// and its state map, and refuse one that left an element of either
    /// unwritten, wrote a NaN, wrote past its live rows or changed a state
    /// byte outside its map; the state map is poisoned before the launch.
    #[test]
    fn unit_write_checks_refuse_what_equal_units_accept() {
        let entry = Gather;
        let case = Case::new(&entry, 8)
            .input(
                "x",
                DType::Bf16,
                (Rows::Live, WIDTH),
                vec![1; 8 * WIDTH * 2],
            )
            .output("output", DType::Bf16, (Rows::Live, WIDTH), Write::LiveRows)
            .state(
                "state",
                DType::Bf16,
                (Rows::Fixed(4), WIDTH),
                vec![7; 4 * WIDTH * 2],
                Write::Map(Box::new(|_, live| {
                    WriteMap::of(DType::Bf16).rows(WIDTH, 0..live as usize)
                })),
                |_, _| Ok(Vec::new()),
            );
        let images = case.poison_state_maps(&case.images(), 2).unwrap();
        let state = images.bytes("state").unwrap();
        assert!(state[..2 * WIDTH * 2].iter().all(|&b| b == POISON));
        assert!(state[2 * WIDTH * 2..].iter().all(|&b| b == 7));
        let one = 0x3f80_u16.to_le_bytes();
        let mut output = vec![POISON; 8 * WIDTH * 2];
        output[..2 * WIDTH * 2].copy_from_slice(&one.repeat(2 * WIDTH));
        let mut written = state.to_vec();
        written[..2 * WIDTH * 2].copy_from_slice(&one.repeat(2 * WIDTH));
        let results = |output: &[u8], state: &[u8]| {
            Results::from([
                ("output".to_owned(), output.to_vec()),
                ("state".to_owned(), state.to_vec()),
            ])
        };
        let check = |output: &[u8], state: &[u8]| {
            case.check_unit_writes(&images, 2, &results(output, state))
        };
        assert_eq!(check(&output, &written), Ok(()));
        let mut hole = output.clone();
        hole[2..4].fill(POISON);
        assert!(check(&hole, &written).is_err());
        let mut nan = output.clone();
        nan[..2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
        assert!(check(&nan, &written).is_err());
        let mut past = output.clone();
        past[2 * WIDTH * 2] = 0;
        assert!(check(&past, &written).is_err());
        assert!(check(&output, state).is_err());
        let mut stray = written.clone();
        stray[3 * WIDTH * 2] = 0;
        assert!(check(&output, &stray).is_err());
    }

    fn refused(case: Case<'_>, reason: &str) {
        let error = case.validate().expect_err(reason);
        assert!(error.contains(reason), "{error}");
    }

    #[test]
    fn a_declared_case_validates() {
        case(&Gather).validate().expect("valid case");
    }

    /// A run refuses a mutant only where an output it gives differs from
    /// the oracle's bits inside the write map.
    #[test]
    fn runs_refuse_a_mutant_where_its_mapped_bits_differ() {
        let entry = Gather;
        let case = case(&entry);
        let bits = |images: &Images, live| -> LabResult<Vec<u8>> {
            match gather(images, live)?.remove(0).1 {
                Expected::Bits(bits) => Ok(bits),
                Expected::Values(_) | Expected::Judge(_) => unreachable!("the gather gives bits"),
            }
        };
        let exact = case
            .refusing_runs(|images, live| Ok(vec![("output".into(), bits(images, live)?)]))
            .expect("an exact mutant");
        assert!(exact.is_empty(), "{exact:?}");
        // Bytes past the live rows lie outside the map.
        let tail = case
            .refusing_runs(|images, live| {
                let mut bytes = bits(images, live)?;
                if let Some(byte) = bytes.get_mut(live as usize * WIDTH * 2) {
                    *byte ^= 1;
                }
                Ok(vec![("output".into(), bytes)])
            })
            .expect("a tail mutant");
        assert!(tail.is_empty(), "{tail:?}");
        // A gather that reads a hole as pool row 0.
        let holes = case
            .refusing_runs(|images, live| {
                let mut ids = images.i32s("ids")?;
                for id in &mut ids {
                    *id = (*id).max(0);
                }
                let mut moved = images.clone();
                moved.0.insert("ids".into(), i32_image(&ids));
                Ok(vec![("output".into(), bits(&moved, live)?)])
            })
            .expect("a hole mutant");
        assert!(holes.contains(&"holes ids".to_string()), "{holes:?}");
        assert!(!holes.contains(&"live 0".to_string()), "{holes:?}");
        assert!(!holes.contains(&"live 1".to_string()), "{holes:?}");
        let error = case
            .refusing_runs(|_, _| Ok(vec![("pool".into(), pool())]))
            .expect_err("a mutant input");
        assert!(
            error.contains("pool, which is not an output or state"),
            "{error}"
        );
    }

    fn i32_image(values: &[i32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn every_parameter_is_bound_and_every_binding_a_parameter() {
        let entry = Gather;
        let mut unbound = case(&entry);
        unbound.scalars.clear();
        refused(unbound, "width has no scalar");
        refused(
            case(&entry).scalar("height", Argument::U32(1)),
            "height is not a parameter",
        );
        refused(
            case(&entry).scalar("width", Argument::U64(4)),
            "another kind",
        );
        refused(case(&entry).live("count"), "live_rows has no buffer");
        let stray = case(&entry).input("extra", DType::U8, (Rows::Fixed(1), 4), vec![0; 4]);
        refused(stray, "extra is not a parameter");
    }

    #[test]
    fn images_must_have_their_declared_size() {
        let entry = Gather;
        let mut short = case(&entry);
        short.buffers[1].image.pop();
        refused(short, "pool has a 127-byte image for 128 bytes");
    }

    /// The gather with a second output.
    struct Pair;

    impl LabEntry for Pair {
        fn symbol(&self) -> String {
            "chop_gather_pair_v1".into()
        }

        fn params(&self) -> Vec<EntryParam> {
            let mut params = Gather.params();
            params.insert(
                3,
                EntryParam::DevicePointer {
                    name: "second".into(),
                    cxx_type: "uint16_t".into(),
                },
            );
            params
        }

        fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
            Gather.plan(prepared_rows)
        }
    }

    /// The gather's output map, empty at live 4.
    fn skips_live_4() -> Write<'static> {
        Write::Map(Box::new(|_, live| {
            let rows = if live == 4 { 0 } else { live as usize };
            WriteMap::of(DType::Bf16).prefix(rows * WIDTH)
        }))
    }

    /// Review probe: an output that writes at the largest live count but
    /// not at a smaller one validated, so a launch that skipped it there
    /// passed every check.
    #[test]
    fn every_output_writes_at_every_positive_live_count_unless_declared() {
        let entry = Gather;
        let mut empty = case(&entry);
        empty.buffers[2].role = Role::Output {
            write: Write::Map(Box::new(|_, _| WriteMap::of(DType::Bf16).none())),
            pattern: vec![POISON],
        };
        refused(empty, "output writes nothing at live 8");
        let mut skipped = case(&entry);
        skipped.buffers[2].role = Role::Output {
            write: skips_live_4(),
            pattern: vec![POISON],
        };
        refused(skipped, "output writes nothing at live 4");
        let mut declared = case(&entry).no_write("output", 4, "row 3 is superseded");
        declared.buffers[2].role = Role::Output {
            write: skips_live_4(),
            pattern: vec![POISON],
        };
        declared.validate().expect("live 4 declared no-write");
        assert_eq!(declared.null_counts(&declared.images()).unwrap(), [8, 1, 7]);
        refused(
            case(&entry).no_write("output", 4, "stale"),
            "output is declared no-write at live 4 but writes 32 bytes",
        );
        refused(
            case(&entry).no_write("output", 3, "not a count"),
            "no-write output at live 3 is not a positive live count of the case",
        );
        refused(
            case(&entry).no_write("pool", 4, "an input"),
            "no-write pool is not an output or state",
        );
        refused(
            case(&entry).no_write("output", 4, ""),
            "no-write output at live 4 gives no reason",
        );
        case(&entry)
            .redundant("chop_gather_v1", "a later stage redoes it")
            .validate()
            .expect("a redundant stage of the case");
        refused(
            case(&entry).redundant("chop_other_v1", "unknown"),
            "redundant stage chop_other_v1 is not a symbol of the case",
        );
        refused(
            case(&entry).redundant("chop_gather_v1", " "),
            "redundant stage chop_gather_v1 gives no reason",
        );
        refused(
            case(&entry).live_counts(vec![9]),
            "live count 9 exceeds capacity 8",
        );
    }

    /// A second output that writes does not cover one that writes nothing.
    #[test]
    fn one_written_output_does_not_cover_another() {
        let entry = Pair;
        let skipped = |write: Write<'static>| {
            case(&entry)
                .output("second", DType::Bf16, (Rows::Live, WIDTH), write)
                .oracle(|images, live| {
                    let mut expected = gather(images, live)?;
                    let output = gather(images, live)?.remove(0).1;
                    expected.push(("second".into(), output));
                    Ok(expected)
                })
        };
        refused(skipped(skips_live_4()), "second writes nothing at live 4");
        skipped(Write::LiveRows)
            .validate()
            .expect("both outputs write");
    }

    /// Review probe: a list of the capacity alone validated, dropping the
    /// live-0 and partial-count checks.
    #[test]
    fn live_counts_keep_zero_a_partial_count_and_the_capacity() {
        let entry = Gather;
        refused(
            case(&entry).live_counts(vec![8]),
            "live counts [8] omit 0, a partial count",
        );
        refused(
            case(&entry).live_counts(vec![1]),
            "live counts [1] omit 0, the capacity",
        );
        refused(
            case(&entry).live_counts(vec![0]),
            "live counts [0] omit a partial count, the capacity",
        );
        refused(
            case(&entry).live_counts(vec![8, 0]),
            "live counts [8, 0] omit a partial count",
        );
        case(&entry)
            .live_counts(vec![0, 3, 8])
            .validate()
            .expect("zero, partial and capacity");
    }

    /// Review probe: a BF16 output with an F32 map over the same bytes
    /// passed the map check with half its elements still poisoned.
    #[test]
    fn maps_count_elements_of_the_buffer_dtype() {
        let entry = Gather;
        let f32_map = || -> Write<'static> {
            Write::Map(Box::new(|_, live| {
                WriteMap::of(DType::F32).prefix(live as usize * WIDTH / 2)
            }))
        };
        let mut output = case(&entry);
        output.buffers[2].role = Role::Output {
            write: f32_map(),
            pattern: vec![POISON],
        };
        refused(
            output,
            "output map counts 4-byte elements, its Bf16 dtype 2",
        );
        let mut state = case(&entry);
        state.buffers[2].image = vec![0; 8 * WIDTH * 2];
        state.buffers[2].role = Role::State {
            write_only: f32_map(),
            updated: Box::new(|_, _| Ok(Vec::new())),
        };
        refused(state, "output map counts 4-byte elements, its Bf16 dtype 2");
    }

    /// Review probe: `Write::All` on live rows mapped every byte at live 0
    /// and past every partial count, so a launch that ignored the live
    /// count passed; a custom map could do the same.
    #[test]
    fn maps_leave_live_0_and_the_rows_past_live_unwritten() {
        let entry = Gather;
        let whole = |images: &Images, _| gather(images, 8);
        let mut all = case(&entry).oracle(whole);
        all.buffers[2].role = Role::Output {
            write: Write::All,
            pattern: vec![POISON],
        };
        refused(
            all,
            "output map at live 1 reaches byte 64, past its 1 live rows (8 bytes)",
        );
        let mut past = case(&entry).oracle(whole);
        past.buffers[2].role = Role::Output {
            write: Write::Map(Box::new(|_, live| {
                let rows = if live == 0 { 0 } else { 8 };
                WriteMap::of(DType::Bf16).prefix(rows * WIDTH)
            })),
            pattern: vec![POISON],
        };
        refused(
            past,
            "output map at live 1 reaches byte 64, past its 1 live rows (8 bytes)",
        );
        let mut updated = case(&entry);
        updated.buffers[2].image = vec![0; 8 * WIDTH * 2];
        updated.buffers[2].role = Role::State {
            write_only: Write::LiveRows,
            updated: Box::new(|_, _| Ok(vec![0..2])),
        };
        refused(updated, "output updates 2 bytes at live 0");
        // A buffer of fixed rows, such as a pool, keeps its explicit map or
        // all its elements, except at live 0.
        let fixed = |write: Write<'static>| {
            let mut fixed = case(&entry).oracle(|images, live| {
                let mut expected = gather(images, live)?;
                if let Expected::Bits(bits) = &mut expected[0].1 {
                    bits.resize(16 * WIDTH * 2, 0);
                }
                Ok(expected)
            });
            fixed.buffers[2].rows = Rows::Fixed(16);
            fixed.buffers[2].role = Role::Output {
                write,
                pattern: vec![POISON],
            };
            fixed
        };
        fixed(Write::Map(Box::new(|_, live| {
            WriteMap::of(DType::Bf16).prefix(live as usize * WIDTH)
        })))
        .validate()
        .expect("fixed rows with an explicit map");
        fixed(Write::All)
            .validate()
            .expect("fixed rows written whole at positive counts");
        refused(
            fixed(Write::Map(Box::new(|_, _| {
                WriteMap::of(DType::Bf16).prefix(16 * WIDTH)
            }))),
            "output maps 128 bytes at live 0",
        );
    }

    /// Metadata written at every count, live 0 included, maps bytes there
    /// only with an oracle, whose live-0 values the run then compares.
    #[test]
    fn every_count_maps_write_at_live_0_against_the_oracle() {
        let entry = Gather;
        let every = || {
            Write::EveryCount(Box::new(|_, _| {
                WriteMap::of(DType::Bf16).prefix(16 * WIDTH)
            }))
        };
        let mut checked = case(&entry).oracle(|images, live| {
            let mut expected = gather(images, live)?;
            if let Expected::Bits(bits) = &mut expected[0].1 {
                bits.resize(16 * WIDTH * 2, 0);
            }
            Ok(expected)
        });
        checked.buffers[2].rows = Rows::Fixed(16);
        checked.buffers[2].role = Role::Output {
            write: every(),
            pattern: vec![POISON],
        };
        checked
            .validate()
            .expect("an every-count map with an oracle");
        let mut unchecked = case(&entry);
        unchecked.oracle = None;
        unchecked.buffers[2].rows = Rows::Fixed(16);
        unchecked.buffers[2].role = Role::Output {
            write: every(),
            pattern: vec![POISON],
        };
        refused(
            unchecked,
            "output maps 128 bytes at live 0, which only the oracle's live-0 values check",
        );
    }

    /// Review probe: an empty oracle satisfied the all-hole requirement and
    /// checked nothing, and a numeric reference shorter than the buffer was
    /// compared only over its own length.
    #[test]
    fn the_oracle_covers_every_written_output_whole() {
        let entry = Gather;
        refused(
            case(&entry).oracle(|_, _| Ok(Vec::new())),
            "the oracle omits output, which writes 64 bytes at live 8",
        );
        refused(
            case(&entry).oracle(|_, _| Ok(vec![("output".into(), Expected::Bits(vec![0; 16]))])),
            "the oracle's output holds 16 bytes of 64",
        );
        refused(
            case(&entry).oracle(|images, live| {
                let mut expected = gather(images, live)?;
                expected.push(("pool".into(), Expected::Bits(vec![0; 128])));
                Ok(expected)
            }),
            "the oracle names pool, not an output or state",
        );
        refused(
            case(&entry).oracle(|images, live| {
                let mut expected = gather(images, live)?;
                expected.extend(gather(images, live)?);
                Ok(expected)
            }),
            "the oracle names output twice",
        );
        let values = |count: usize| {
            OracleOutput::new(vec![0.0; count], vec![0.0; count], 1).expect("reference")
        };
        refused(
            case(&entry)
                .oracle(move |_, _| Ok(vec![("output".into(), Expected::Values(values(0)))])),
            "the oracle's output holds 0 values of 32",
        );
        refused(
            case(&entry)
                .oracle(move |_, _| Ok(vec![("output".into(), Expected::Values(values(4)))])),
            "the oracle's output holds 4 values of 32",
        );
        // A constant reference reads no ID, so it reruns no hole row and
        // no high IDs.
        let mut constant = with_ids(&entry, Rows::Live, &[3, 0, 9, 9, -1, 5, 1, 2], Holes::Never)
            .oracle(move |_, _| Ok(vec![("output".into(), Expected::Values(values(32)))]));
        constant.buffers[0].role = Role::Ids {
            pools: Vec::new(),
            holes: Holes::Never,
        };
        constant.validate().expect("a whole numeric reference");
        // An output that writes nothing at a live count may be left out.
        case(&entry)
            .oracle(|images, live| match live {
                0 => Ok(Vec::new()),
                _ => gather(images, live),
            })
            .validate()
            .expect("nothing written at live 0");
    }

    /// Numeric references compare the elements of the write map, at their
    /// own indices, and nothing past it.
    #[test]
    fn numeric_references_compare_the_mapped_elements() {
        let entry = Gather;
        let checked = case(&entry);
        let images = checked.images();
        // Live 2 maps elements 0..8 of 32; the rest hold poison.
        let mut actual = vec![POISON; 64];
        for (index, value) in actual[..16].chunks_exact_mut(2).enumerate() {
            value.copy_from_slice(&f32_to_bf16(index as f32).to_le_bytes());
        }
        let reference = |values: Vec<f64>| {
            let count = values.len();
            Expected::Values(OracleOutput::new(values, vec![1.0; count], 1).expect("reference"))
        };
        let whole = (0..32).map(|index| index as f64).collect::<Vec<_>>();
        let mut receipt = BTreeMap::new();
        checked
            .check_expected(
                "output",
                reference(whole.clone()),
                &actual,
                &images,
                2,
                &mut receipt,
            )
            .expect("mapped elements match");
        assert_eq!(receipt["output_oracle"]["correct"], json!(true));
        let mut wrong = whole;
        wrong[7] = 100.0;
        let error = checked
            .check_expected(
                "output",
                reference(wrong),
                &actual,
                &images,
                2,
                &mut receipt,
            )
            .expect_err("element 7 differs");
        assert!(error.contains("output at live 2"), "{error}");
        let error = checked
            .check_expected(
                "output",
                reference(vec![0.0; 8]),
                &actual,
                &images,
                2,
                &mut receipt,
            )
            .expect_err("a short reference");
        assert!(
            error.contains("holds 32 values, the reference 8"),
            "{error}"
        );
    }

    /// A judge sees the whole buffer, and its refusal fails the output.
    #[test]
    fn a_judge_sees_the_whole_buffer() {
        let entry = Gather;
        let checked = case(&entry);
        let images = checked.images();
        let actual = vec![POISON; 64];
        let judge = |wanted: usize| {
            Expected::Judge(Box::new(move |bytes: &[u8]| match bytes.len() {
                length if length == wanted => Ok(()),
                length => Err(format!("{length} bytes")),
            }))
        };
        let mut receipt = BTreeMap::new();
        checked
            .check_expected("output", judge(64), &actual, &images, 2, &mut receipt)
            .expect("the judge accepts");
        assert_eq!(receipt["output_oracle"], json!({"judged": true}));
        let error = checked
            .check_expected("output", judge(8), &actual, &images, 2, &mut receipt)
            .expect_err("the judge refuses");
        assert_eq!(error, "output at live 2: 64 bytes");
    }

    /// `case` with the IDs replaced by `ids` of `rows`.
    fn with_ids<'a>(entry: &'a dyn LabEntry, rows: Rows, ids: &[i32], holes: Holes) -> Case<'a> {
        let mut case = case(entry);
        case.buffers[0].rows = rows;
        case.buffers[0].image = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        case.buffers[0].role = Role::Ids {
            pools: vec![("pool".into(), WIDTH * 2)],
            holes,
        };
        case
    }

    /// Review probe: a fixed ID input reran its last row as holes, a row
    /// the entry never read, so an entry broken on all-hole rows passed.
    #[test]
    fn the_hole_row_is_one_the_entry_reads() {
        let entry = Gather;
        let ids = [3, 0, 9, 9, 7, 5, 1, 2, 4, 6];
        refused(
            with_ids(&entry, Rows::Fixed(10), &ids, Holes::LastLive),
            "ids has fixed rows; name the hole row the entry reads (Holes::Row)",
        );
        refused(
            with_ids(&entry, Rows::Fixed(10), &ids, Holes::Row(9)),
            "ids row 9 all holes leaves the oracle's results at live 8 unchanged",
        );
        refused(
            with_ids(&entry, Rows::Fixed(10), &ids, Holes::Row(10)),
            "ids hole row 10 lies past its 10 rows",
        );
        with_ids(&entry, Rows::Fixed(10), &ids, Holes::Row(2))
            .validate()
            .expect("row 2 is read at live 8");
        // A row that is already all holes reruns as itself.
        refused(
            with_ids(
                &entry,
                Rows::Live,
                &[3, 0, 9, 9, -1, 5, 1, 2],
                Holes::Row(4),
            ),
            "ids row 4 all holes leaves the oracle's results at live 8 unchanged",
        );
    }

    /// Review probe: a whole-buffer reference that also computed rows the
    /// run never compares changed there when an unread ID row became holes,
    /// and that change passed for the hole row being read. A numeric
    /// reference whose tolerance alone changed passed the same way.
    #[test]
    fn the_hole_row_changes_bytes_the_run_compares() {
        let entry = Gather;
        let ids = [3, 0, 9, 9, 7, 5, 1, 2, 4, 6];
        let padded = |holes: Holes| {
            // Every row of all ten IDs, though at most eight are live.
            let mut padded = with_ids(&entry, Rows::Fixed(10), &ids, holes).oracle(|images, _| {
                let pool = images.bytes("pool")?;
                let row = WIDTH * 2;
                let mut output = vec![0; 10 * row];
                for (at, id) in images.i32s("ids")?.into_iter().enumerate() {
                    if let Ok(id) = usize::try_from(id) {
                        output[at * row..][..row].copy_from_slice(&pool[id * row..][..row]);
                    }
                }
                Ok(vec![("output".into(), Expected::Bits(output))])
            });
            padded.buffers[2].rows = Rows::Fixed(10);
            padded
        };
        refused(
            padded(Holes::Row(9)),
            "ids row 9 all holes leaves the oracle's results at live 8 unchanged",
        );
        padded(Holes::Row(7))
            .validate()
            .expect("row 7 is read and compared at live 8");
        let tolerance =
            with_ids(&entry, Rows::Fixed(10), &ids, Holes::Row(9)).oracle(|images, live| {
                let Expected::Bits(bits) = gather(images, live)?.remove(0).1 else {
                    unreachable!("the gather's bits")
                };
                let values = decode(DType::Bf16, &bits)?
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                let sum = f64::from(images.i32s("ids")?[9]) + 2.0;
                let count = values.len();
                let reference = OracleOutput::new(values, vec![sum; count], 1).expect("reference");
                Ok(vec![("output".into(), Expected::Values(reference))])
            });
        // Pool values the numeric gate tells apart.
        let mut tolerance = tolerance;
        tolerance.buffers[1].image = (0..16 * WIDTH)
            .flat_map(|value| f32_to_bf16(value as f32).to_le_bytes())
            .collect();
        // A numeric reference never witnesses a rerun, so the high-id
        // rerun is refused first.
        refused(tolerance, "the numeric references of output never count");
    }

    /// Maps on the images of a rerun are checked as the first run's are.
    #[test]
    fn rerun_maps_count_elements_of_the_buffer_dtype() {
        let entry = Gather;
        let mut holed = case(&entry);
        holed.buffers[2].role = Role::Output {
            write: Write::Map(Box::new(|images, live| {
                if images.i32s("ids")?[7] == -1 {
                    WriteMap::of(DType::F32).prefix(live as usize * WIDTH / 2)
                } else {
                    WriteMap::of(DType::Bf16).prefix(live as usize * WIDTH)
                }
            })),
            pattern: vec![POISON],
        };
        refused(
            holed,
            "ids row 7 all holes: output map counts 4-byte elements, its Bf16 dtype 2",
        );
    }

    /// Review probe: fixed IDs whose rows the entry reads were all holes,
    /// and only unread rows held IDs, so the high-id rerun moved IDs the
    /// entry never loads and matched without exercising a high row.
    #[test]
    fn the_high_id_rerun_moves_an_id_the_entry_reads() {
        let entry = Gather;
        let mut ids = [-1; 16];
        ids[8..].copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
        refused(
            with_ids(&entry, Rows::Fixed(16), &ids, Holes::Never),
            "ids high-id rerun moves no ID the entry reads at live 8",
        );
        ids[7] = 0;
        with_ids(&entry, Rows::Fixed(16), &ids, Holes::Never)
            .validate()
            .expect("row 7 reads ID 0, which moves");
        // Without an oracle no output has exact bits, so nothing witnesses
        // the rerun, not even a map that follows the IDs.
        let mut blind = with_ids(&entry, Rows::Fixed(16), &ids, Holes::Never);
        blind.oracle = None;
        refused(
            blind,
            "ids high-id rerun moves no ID the entry reads at live 8",
        );
        // Rows that read the same bytes from every pool row show nothing.
        let mut flat = with_ids(&entry, Rows::Fixed(16), &ids, Holes::Never);
        flat.buffers[1].image = vec![1; 128];
        refused(
            flat,
            "ids high-id rerun moves no ID the entry reads at live 8",
        );
    }

    /// A model of a launch with F32 buffers and these parameters, at most
    /// 4 rows.
    struct Probe(&'static [&'static str]);

    impl LabEntry for Probe {
        fn symbol(&self) -> String {
            "chop_probe_v1".into()
        }

        fn params(&self) -> Vec<EntryParam> {
            self.0
                .iter()
                .map(|&name| EntryParam::DevicePointer {
                    name: name.into(),
                    cxx_type: "float".into(),
                })
                .collect()
        }

        fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
            Gather.plan(prepared_rows)
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    /// Review probe: state update ranges are permissions, not writes the
    /// launch must make, yet a change in them alone passed for reading the
    /// hole row (a wider range, or the same ranges reordered) or a moved ID
    /// (a range following the ID). A launch that ignores every ID, writing
    /// output ones and state[0] = 3, passes both runs of the hole probes;
    /// the high-id probe's state never changes.
    #[test]
    fn optional_state_updates_witness_no_rerun() {
        let entry = Probe(&["ids", "output", "state", "live_rows"]);
        for reordered in [false, true] {
            let probe = Case::new(&entry, 4)
                .ids("ids", (Rows::Live, 1), &[0; 4], &[], Holes::LastLive)
                .output("output", DType::F32, (Rows::Live, 1), Write::LiveRows)
                .state(
                    "state",
                    DType::F32,
                    (Rows::Fixed(2), 1),
                    f32_bytes(&[2.0, 2.0]),
                    Write::Map(Box::new(|_, _| WriteMap::of(DType::F32).none())),
                    move |images, live| {
                        if live == 0 {
                            return Ok(Vec::new());
                        }
                        let hole = images.i32s("ids")?[3] < 0;
                        Ok(match (reordered, hole) {
                            (true, true) => vec![4..8, 0..4],
                            (true, false) => vec![0..4, 4..8],
                            (false, true) => vec![0..8],
                            (false, false) => vec![0..4],
                        })
                    },
                )
                .oracle(|_, live| {
                    let first = if live > 0 { 3.0 } else { 2.0 };
                    Ok(vec![
                        ("output".into(), Expected::Bits(f32_bytes(&[1.0; 4]))),
                        ("state".into(), Expected::Bits(f32_bytes(&[first, 2.0]))),
                    ])
                });
            refused(
                probe,
                "ids row 3 all holes leaves the oracle's results at live 4 unchanged",
            );
        }
        let probe = Case::new(&entry, 4)
            .ids(
                "ids",
                (Rows::Live, 1),
                &[0; 4],
                &[("state", 4)],
                Holes::Never,
            )
            .output("output", DType::F32, (Rows::Live, 1), Write::LiveRows)
            .state(
                "state",
                DType::F32,
                (Rows::Fixed(8), 1),
                f32_bytes(&[2.0; 8]),
                Write::Map(Box::new(|_, _| WriteMap::of(DType::F32).none())),
                |images, live| {
                    if live == 0 {
                        return Ok(Vec::new());
                    }
                    let at = images.i32s("ids")?[0] as usize * 4;
                    Ok(vec![at..at + 4])
                },
            )
            .oracle(|images, _| {
                Ok(vec![
                    ("output".into(), Expected::Bits(f32_bytes(&[1.0; 4]))),
                    (
                        "state".into(),
                        Expected::Bits(images.bytes("state")?.to_vec()),
                    ),
                ])
            });
        refused(
            probe,
            "ids high-id rerun moves no ID the entry reads at live 4",
        );
    }

    /// A state that row `r` writes at row `ids[r]` of 8 F32 rows, `r + 10`;
    /// a -1 ID writes nothing. `by_row` models a launch that ignores the IDs
    /// and writes row `r`.
    fn scatter_state(images: &Images, live: u32, by_row: bool) -> LabResult<Vec<u8>> {
        let mut state = images.f32s("state")?;
        for (row, &id) in images.i32s("ids")?.iter().enumerate().take(live as usize) {
            if id >= 0 || by_row {
                state[if by_row { row } else { id as usize }] = row as f32 + 10.0;
            }
        }
        Ok(f32_bytes(&state))
    }

    fn scatter(entry: &Probe) -> Case<'_> {
        Case::new(entry, 4)
            .ids(
                "ids",
                (Rows::Live, 1),
                &[0, 1, 2, 3],
                &[("state", 4)],
                Holes::LastLive,
            )
            .output("output", DType::F32, (Rows::Live, 1), Write::LiveRows)
            .state(
                "state",
                DType::F32,
                (Rows::Fixed(8), 1),
                f32_bytes(&[2.0; 8]),
                Write::Map(Box::new(|images, live| {
                    let ids = images.i32s("ids")?;
                    let rows = ids.iter().take(live as usize).filter(|&&id| id >= 0);
                    WriteMap::of(DType::F32).rows(1, rows.map(|&id| id as usize))
                })),
                |_, _| Ok(Vec::new()),
            )
            .oracle(|images, live| {
                Ok(vec![
                    ("output".into(), Expected::Bits(f32_bytes(&[1.0; 4]))),
                    (
                        "state".into(),
                        Expected::Bits(scatter_state(images, live, false)?),
                    ),
                ])
            })
    }

    /// A state mutant is compared whole: a scatter that writes row `r` in
    /// place of row `ids[r]` passes every first run, whose IDs are the rows,
    /// and fails the high-id rerun and the hole rerun; a state changed at
    /// live 0, where nothing writes, fails there.
    #[test]
    fn runs_refuse_a_state_mutant_where_its_bytes_differ() {
        let entry = Probe(&["ids", "output", "state", "live_rows"]);
        let case = scatter(&entry);
        case.validate().expect("the scatter case");
        let state = |by_row| {
            move |images: &Images, live| {
                Ok(vec![("state".into(), scatter_state(images, live, by_row)?)])
            }
        };
        assert!(case.refusing_runs(state(false)).unwrap().is_empty());
        assert_eq!(
            case.refusing_runs(state(true)).unwrap(),
            ["high ids", "holes ids"]
        );
        let at_zero = case
            .refusing_runs(|images, live| {
                let mut state = scatter_state(images, live, false)?;
                if live == 0 {
                    state[0] ^= 1;
                }
                Ok(vec![("state".into(), state)])
            })
            .unwrap();
        assert_eq!(at_zero, ["live 0"]);
        let short = case
            .refusing_runs(|_, _| Ok(vec![("state".into(), vec![0; 4])]))
            .expect_err("a short state");
        assert!(short.contains("holds 4 bytes of 32"), "{short}");
    }

    /// An F32 reference whose numeric gate admits `bound` either side.
    fn within(values: Vec<f64>, bound: f64) -> OracleOutput {
        let count = values.len();
        // The F32 gate is the sum of absolute products times 2^-22.
        OracleOutput::new(values, vec![bound * 4_194_304.0; count], 1).expect("reference")
    }

    /// Review probe: numeric gates can overlap. The references 1 and 0,
    /// each within 0.75, differ in their central values, yet 0.5 passes
    /// both, so a launch that writes 0.5 everywhere passes the first run and
    /// the hole rerun without reading an ID. Only exact bits witness a
    /// rerun; the same case with bits validates.
    #[test]
    fn numeric_references_witness_no_rerun() {
        let entry = Probe(&["ids", "pool", "output", "live_rows"]);
        let probe = |exact: bool, pools: &'static [(&'static str, usize)], holes| {
            Case::new(&entry, 4)
                .ids("ids", (Rows::Live, 1), &[1, 0, 1, 0], pools, holes)
                .input(
                    "pool",
                    DType::F32,
                    (Rows::Fixed(4), 1),
                    f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
                )
                .output("output", DType::F32, (Rows::Live, 1), Write::LiveRows)
                .oracle(move |images, live| {
                    let ids = images.i32s("ids")?;
                    let pool = images.f32s("pool")?;
                    let mut output = vec![0.0; 4];
                    for (row, &id) in ids.iter().take(live as usize).enumerate() {
                        if let Ok(id) = usize::try_from(id) {
                            output[row] = pool[id];
                        }
                    }
                    let expected = if exact {
                        Expected::Bits(f32_bytes(&output))
                    } else {
                        Expected::Values(within(output.into_iter().map(f64::from).collect(), 0.75))
                    };
                    Ok(vec![("output".into(), expected)])
                })
        };
        let first = within(vec![2.0, 1.0, 2.0, 1.0], 0.75);
        let holed = within(vec![2.0, 1.0, 2.0, 0.0], 0.75);
        let wrong = [2.0, 1.0, 2.0, 0.5];
        kernel_core::oracle::check_numeric(&wrong, &first, DType::F32).expect("first gate");
        kernel_core::oracle::check_numeric(&wrong, &holed, DType::F32).expect("hole gate");
        refused(
            probe(false, &[], Holes::LastLive),
            "ids row 3 all holes leaves the oracle's results at live 4 unchanged",
        );
        refused(
            probe(false, &[("pool", 4)], Holes::Never),
            "ids high-id rerun moves no ID the entry reads at live 4",
        );
        probe(true, &[], Holes::LastLive)
            .validate()
            .expect("exact hole row");
        probe(true, &[("pool", 4)], Holes::Never)
            .validate()
            .expect("exact high IDs");
    }

    /// Review probe: the witness moved the IDs over unmoved pools, but the
    /// rerun moves both. With IDs all 0 and pool [1, 1, 3, .., 8], the
    /// permutation [7, 0, .., 6] moves row 0 to row 7 and row 1, which also
    /// holds 1, to row 0. The witness saw 1 against 8, yet a launch that
    /// reads row 0 whatever the IDs say reads 1 in both runs and passes.
    /// The rerun must now tell its result from the stale one: the old IDs
    /// over the moved pools.
    #[test]
    fn the_high_id_rerun_refuses_the_old_ids_over_the_moved_pools() {
        let entry = Probe(&["ids", "pool", "output", "live_rows"]);
        let probe = |pool: [f32; 8]| {
            Case::new(&entry, 4)
                .ids(
                    "ids",
                    (Rows::Live, 1),
                    &[0; 4],
                    &[("pool", 4)],
                    Holes::LastLive,
                )
                .input("pool", DType::F32, (Rows::Fixed(8), 1), f32_bytes(&pool))
                .output("output", DType::F32, (Rows::Live, 1), Write::LiveRows)
                .oracle(|images, live| {
                    let pool = images.f32s("pool")?;
                    let ids = images.i32s("ids")?;
                    let mut output = vec![0.0; 4];
                    for (row, &id) in ids.iter().take(live as usize).enumerate() {
                        if let Ok(id) = usize::try_from(id) {
                            output[row] = pool[id];
                        }
                    }
                    Ok(vec![("output".into(), Expected::Bits(f32_bytes(&output)))])
                })
        };
        let order = high_permutation(&BTreeSet::from([0]), 8).unwrap();
        assert_eq!(order, [7, 0, 1, 2, 3, 4, 5, 6]);
        let pool = [1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let moved = permute_rows(&f32_bytes(&pool), 4, &order, false);
        assert_eq!(moved[..4], moved[28..32], "rows 0 and 7 both hold 1");
        refused(
            probe(pool),
            "ids high-id rerun moves no ID the entry reads at live 4",
        );
        probe([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .validate()
            .expect("the displaced row differs");
        // An oracle that follows the ID values, not just the rows they
        // select, fails a correct launch in the rerun's bitwise comparison.
        let mut numbered = probe([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        numbered.oracle = Some(Box::new(|images, _| {
            let ids = images.i32s("ids")?;
            let output = ids.iter().map(|&id| id as f32).collect::<Vec<_>>();
            Ok(vec![("output".into(), Expected::Bits(f32_bytes(&output)))])
        }));
        refused(
            numbered,
            "the oracle's output on the moved IDs and pools, moved back, differs",
        );
    }

    /// Review probe: the witness compared the first run with the high-id
    /// rerun, and the rerun with the stale read, each on the buffers exact
    /// in both, so a buffer exact only in the reruns witnessed alone. With
    /// IDs all 0 and pool [1, .., 8], the oracle gives a numeric reference
    /// of 1 within 1 while pool row 0 holds 1, and bits otherwise: 1 for the
    /// high-id rerun and 2 for the stale read. A launch that writes 2 to
    /// every live row passes the first run's gate and matches its own bytes
    /// in the rerun; the rerun's bits of 1 are never compared. The witness
    /// must be one buffer exact in all three, the same in the first run and
    /// the rerun. Other numeric outputs may remain beside it.
    #[test]
    fn the_high_id_witness_is_exact_in_the_first_run_too() {
        let alone = Probe(&["ids", "pool", "output", "live_rows"]);
        let beside = Probe(&["ids", "pool", "output", "witness", "live_rows"]);
        let probe = |witness: bool| {
            let entry: &dyn LabEntry = if witness { &beside } else { &alone };
            let case = Case::new(entry, 4)
                .ids(
                    "ids",
                    (Rows::Live, 1),
                    &[0; 4],
                    &[("pool", 4)],
                    Holes::Never,
                )
                .input(
                    "pool",
                    DType::F32,
                    (Rows::Fixed(8), 1),
                    f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                )
                .output("output", DType::F32, (Rows::Live, 1), Write::LiveRows);
            let case = if witness {
                case.output("witness", DType::F32, (Rows::Live, 1), Write::LiveRows)
            } else {
                case
            };
            case.oracle(move |images, live| {
                let ids = images.i32s("ids")?;
                let pool = images.f32s("pool")?;
                let mut output = vec![0.0; 4];
                for (row, &id) in ids.iter().take(live as usize).enumerate() {
                    output[row] = pool[id as usize];
                }
                let mut expected = vec![(
                    "output".into(),
                    if pool[0] == 1.0 {
                        Expected::Values(within(
                            output.iter().map(|&v| f64::from(v)).collect(),
                            1.0,
                        ))
                    } else {
                        Expected::Bits(f32_bytes(&output))
                    },
                )];
                if witness {
                    expected.push(("witness".into(), Expected::Bits(f32_bytes(&output))));
                }
                Ok(expected)
            })
        };
        let order = high_permutation(&BTreeSet::from([0]), 8).unwrap();
        assert_eq!(order, [7, 0, 1, 2, 3, 4, 5, 6]);
        let wrong = [2.0; 4];
        let first = within(vec![1.0; 4], 1.0);
        kernel_core::oracle::check_numeric(&wrong, &first, DType::F32).expect("first gate");
        refused(
            probe(false),
            "ids high-id rerun moves no ID the entry reads at live 4",
        );
        probe(true)
            .validate()
            .expect("a stable exact witness beside a numeric output");
    }

    /// Review probe: IDs that are all holes moved nothing and still
    /// reported a high-id rerun.
    #[test]
    fn the_high_id_rerun_moves_an_id() {
        let entry = Gather;
        refused(
            with_ids(&entry, Rows::Live, &[-1; 8], Holes::Never),
            "ids holds no ID, so its high-id rerun moves nothing",
        );
        refused(
            with_ids(
                &entry,
                Rows::Live,
                &[8, 9, 10, 11, 12, 13, 14, 15],
                Holes::Never,
            ),
            "ids already holds the top rows of its pools, so its high-id rerun moves nothing",
        );
        with_ids(
            &entry,
            Rows::Live,
            &[8, 9, 10, 11, 12, 13, 14, 0],
            Holes::Never,
        )
        .validate()
        .expect("ID 0 moves");
    }

    #[test]
    fn ids_must_lie_in_their_pools() {
        let entry = Gather;
        let mut past = case(&entry);
        past.buffers[0].image[..4].copy_from_slice(&16_i32.to_le_bytes());
        refused(past, "ID 16 lies past its pools' 16 rows");
        let mut negative = case(&entry);
        negative.buffers[0].image[..4].copy_from_slice(&(-2_i32).to_le_bytes());
        refused(negative, "holds ID -2");
        let mut ragged = case(&entry);
        ragged.buffers[0].role = Role::Ids {
            pools: vec![("pool".into(), 24)],
            holes: Holes::Never,
        };
        refused(ragged, "is not rows of 24");
    }

    #[test]
    fn a_produced_input_must_differ_from_poison() {
        let entry = Gather;
        case(&entry)
            .produced("pool")
            .validate()
            .expect("no poison element");
        let mut poisoned = case(&entry).produced("pool");
        poisoned.buffers[1].image[6..8].fill(POISON);
        refused(poisoned, "element 3 is the poison pattern");
        refused(case(&entry).produced("ids"), "is not a plain input");
    }

    /// Review finding (source inspection): a case with a produced input
    /// ran its PDL phase only behind the copy producer, which never delays
    /// the live count, so a launch that read the count before its PDL wait
    /// passed. It now also runs behind the producer that holds both.
    #[test]
    fn a_produced_input_keeps_the_live_count_pdl_phase() {
        let entry = Gather;
        assert_eq!(
            case(&entry).pdl_producers(),
            [PdlProducer::Live {
                produced: false,
                count: true
            }]
        );
        assert_eq!(
            case(&entry).produced("pool").pdl_producers(),
            [
                PdlProducer::Copy,
                PdlProducer::Live {
                    produced: true,
                    count: true
                }
            ]
        );
    }

    /// A live count the host writes before the step may be read before
    /// the PDL wait: the live-count producer then holds the produced input
    /// alone, which a launch must still read after its wait, and a case
    /// without a produced input has no PDL phase left. The reason is
    /// required.
    #[test]
    fn a_step_input_live_count_leaves_the_produced_input_held() {
        let entry = Gather;
        let reason = "the host writes the count before the step";
        assert_eq!(
            case(&entry)
                .produced("pool")
                .live_count_before_wait(reason)
                .pdl_producers(),
            [
                PdlProducer::Copy,
                PdlProducer::Live {
                    produced: true,
                    count: false
                }
            ]
        );
        assert_eq!(
            PdlProducer::Live {
                produced: true,
                count: false
            }
            .describe(),
            "PDL input producer"
        );
        assert!(
            case(&entry)
                .live_count_before_wait(reason)
                .pdl_producers()
                .is_empty()
        );
        case(&entry)
            .produced("pool")
            .live_count_before_wait(reason)
            .validate()
            .expect("a step input count with a reason");
        refused(
            case(&entry).live_count_before_wait(" "),
            "a live count read before the PDL wait gives no reason",
        );
    }

    /// Only an error from a PDL phase counts against the PDL check.
    #[test]
    fn pdl_phase_errors_name_their_producer() {
        assert!(pdl_phase_error(
            "chop_x_v1 live 16: PDL live-count and input producer: output PDL: element 0"
        ));
        assert!(pdl_phase_error(
            "chop_x_v1 live 4: PDL input producer: output PDL: x"
        ));
        assert!(pdl_phase_error(
            "chop_x_v1 live 4: PDL copy producer: output PDL: x"
        ));
        assert!(!pdl_phase_error(
            "chop_x_v1 live 16: cache.k at live 16: numeric gate failed at element 7"
        ));
        assert!(!pdl_phase_error(
            "chop_x_v1 live 16: cache.pool eager: write-only element at byte 2 still holds poison"
        ));
    }

    #[test]
    fn hole_rows_need_an_oracle() {
        let entry = Gather;
        let mut blind = case(&entry);
        blind.oracle = None;
        refused(blind, "an all-hole row needs an oracle");
    }

    #[test]
    fn binding_follows_the_parameter_order() {
        let names = BTreeSet::from(["output", "pool", "ids", "live_rows"]);
        let scalars = BTreeMap::from([("width".to_owned(), Argument::U32(4))]);
        let (arguments, bound) = bind(&Gather.params(), &names, &scalars).unwrap();
        assert_eq!(bound, ["ids", "pool", "output", "live_rows"]);
        assert_eq!(
            arguments,
            vec![
                Argument::DeviceBuffer(1),
                Argument::DeviceBuffer(2),
                Argument::DeviceBuffer(3),
                Argument::DeviceBuffer(4),
                Argument::U32(4),
            ]
        );
    }

    #[test]
    fn high_ids_take_the_top_of_the_pool_and_move_back() {
        let order = high_permutation(&BTreeSet::from([0, 3, 5]), 8).unwrap();
        assert_eq!(order, vec![5, 0, 1, 6, 2, 7, 3, 4]);
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..8).collect::<Vec<_>>());
        assert_eq!(
            remap_ids(&[5, -1, 0, 3], &order),
            [7, -1, 5, 6]
                .iter()
                .flat_map(|id: &i32| id.to_le_bytes())
                .collect::<Vec<_>>()
        );
        let image = (0..20_u8).collect::<Vec<_>>();
        let moved = permute_rows(&image, 2, &order, false);
        assert_eq!(moved[14..16], image[10..12]);
        assert_eq!(moved[16..], image[16..]);
        assert_eq!(permute_rows(&moved, 2, &order, true), image);
        assert!(high_permutation(&BTreeSet::from([8]), 8).is_err());
    }

    #[test]
    fn update_ranges_join_in_any_order() {
        assert_eq!(union(vec![4..8, 0..4, 6..6]), [0..8]);
        assert_eq!(union(vec![10..12, 0..6, 2..4]), [0..6, 10..12]);
        assert_eq!(union(vec![]), []);
    }

    #[test]
    fn bits_compare_only_inside_the_given_ranges() {
        let expected = [1, 2, 3, 4];
        assert!(compare_bits("out", &[1, 2, 9, 9], &expected, &[0..2]).is_ok());
        let error = compare_bits("out", &[1, 2, 9, 4], &expected, &[0..4]).unwrap_err();
        assert!(error.contains("byte 2"), "{error}");
        assert!(compare_bits("out", &[1, 2], &expected, &[]).is_err());
        assert_eq!(
            decode(DType::Bf16, &0x3f80_u16.to_le_bytes()).unwrap(),
            [1.0]
        );
        assert!(decode(DType::I32, &[0; 4]).is_err());
    }

    /// A chain's entries: one gathering pool rows by ID into a partial, one
    /// copying the partial's live rows to the output.
    struct Stage(&'static str, &'static [(&'static str, &'static str)]);

    const SPLIT: Stage = Stage(
        "chop_gather_split_v1",
        &[
            ("ids", "const int32_t"),
            ("pool", "const uint16_t"),
            ("partial", "uint16_t"),
            ("live_rows", "const uint32_t"),
        ],
    );
    const MERGE: Stage = Stage(
        "chop_gather_merge_v1",
        &[
            ("partial", "const uint16_t"),
            ("output", "uint16_t"),
            ("live_rows", "const uint32_t"),
        ],
    );

    impl LabEntry for Stage {
        fn symbol(&self) -> String {
            self.0.into()
        }

        fn params(&self) -> Vec<EntryParam> {
            let mut params = self
                .1
                .iter()
                .map(|&(name, cxx_type)| EntryParam::DevicePointer {
                    name: name.into(),
                    cxx_type: cxx_type.into(),
                })
                .collect::<Vec<_>>();
            params.push(EntryParam::U32 {
                name: "width".into(),
            });
            params
        }

        fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
            Gather.plan(prepared_rows)
        }
    }

    /// The gather as a chain, its partial declared as scratch the split
    /// writes in its live rows.
    fn chain<'a>(first: &'a Stage, second: &'a Stage) -> Case<'a> {
        case(first)
            .then(second)
            .scratch("partial", 8 * WIDTH * 2, |_, live| {
                Ok(vec![0..live as usize * WIDTH * 2])
            })
    }

    #[test]
    fn a_declared_chain_validates() {
        chain(&SPLIT, &MERGE).validate().expect("valid chain");
        assert_eq!(
            chain(&SPLIT, &MERGE).symbol(),
            "chop_gather_split_v1 then chop_gather_merge_v1"
        );
        // The partial as an output the oracle covers.
        case(&SPLIT)
            .then(&MERGE)
            .output("partial", DType::Bf16, (Rows::Live, WIDTH), Write::LiveRows)
            .oracle(|images, live| {
                let mut results = gather(images, live)?;
                let Expected::Bits(bits) = &results[0].1 else {
                    unreachable!("the gather gives bits")
                };
                let partial = Expected::Bits(bits.clone());
                results.push(("partial".into(), partial));
                Ok(results)
            })
            .validate()
            .expect("a checked partial");
    }

    #[test]
    fn every_entry_of_a_chain_binds_and_plans() {
        refused(
            case(&SPLIT).then(&MERGE),
            "chop_gather_split_v1: parameter partial has no buffer",
        );
        refused(
            chain(&SPLIT, &MERGE).scalar("height", Argument::U32(1)),
            "height is not a parameter",
        );
        let mut deep = chain(&SPLIT, &MERGE);
        deep.capacity = 9;
        deep.live_counts = vec![0, 4, 9];
        refused(
            deep,
            "chop_gather_split_v1 then chop_gather_merge_v1: chop_gather_split_v1 plan at capacity 9",
        );
    }

    /// The link rule asks one read of an earlier write per later entry, not
    /// a producer for every read: a merge that also reads scratch no entry
    /// writes validates.
    #[test]
    fn a_link_needs_one_read_of_an_earlier_write_not_every_read() {
        const READS_BOTH: Stage = Stage(
            "chop_gather_merge_v1",
            &[
                ("partial", "const uint16_t"),
                ("unwritten", "const uint16_t"),
                ("output", "uint16_t"),
                ("live_rows", "const uint32_t"),
            ],
        );
        chain(&SPLIT, &READS_BOTH)
            .scratch("unwritten", 64, |_, _| Ok(vec![]))
            .validate()
            .expect("one link suffices");
    }

    /// An entry that reads nothing the chain wrote before it would pass
    /// launched first or alone; a buffer an entry writes but the case keeps
    /// unpoisoned and unchecked hides what it writes.
    #[test]
    fn each_later_entry_reads_what_an_earlier_one_writes() {
        let swapped = case(&MERGE)
            .then(&SPLIT)
            .scratch("partial", 64, |_, _| Ok(vec![0..64]));
        refused(
            swapped,
            "chop_gather_split_v1 reads nothing an earlier entry of the chain writes",
        );
        let input = case(&SPLIT).then(&MERGE).input(
            "partial",
            DType::Bf16,
            (Rows::Live, WIDTH),
            vec![0; 8 * WIDTH * 2],
        );
        refused(
            input,
            "chop_gather_split_v1 writes partial, which is declared an input",
        );
    }
}
