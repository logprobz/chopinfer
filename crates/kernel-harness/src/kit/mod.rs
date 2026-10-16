//! The conformance kit: runs any backend's prepared ops by contract roles
//! against the contracts' CPU oracles.
//!
//! A backend developer lists the ops a model requires ([`Required`]: a
//! name, how a backend prepares the op, and the case it runs) and calls
//! [`conform`] with their [`Backend`] and the [`Modules`] that load its
//! symbols. For each op the report says whether the backend prepared it or
//! refused it (with the reason), and the result of every check:
//!
//! - `declaration`: the case checked on the CPU ([`Case::validate`]), and
//!   the op's launches at the capacity against its [`OpInfo`]: each symbol
//!   is one the info lists, at most its dynamic shared memory, and every
//!   role the info sizes is a view of at least those bytes;
//! - `conformance`: every check [`Case::run`] performs for a lab entry
//!   (poisoned eager, relaunch, graph and PDL runs at each live count, write
//!   maps, the null launch, bounded scratch, the oracle), over the op's
//!   launches;
//! - one check per exactness property the op guarantees ([`properties`]),
//!   or the reason the kit cannot check it; one-row bits of an op over
//!   items of several rows compare a verify step with decode launches
//!   ([`decode`]).
//!
//! An empty [`Report::failures`] is not conformance evidence: a refused op
//! not marked required ([`Required::required`]) and a check reported as not
//! exercised (every GPU check without a session) are not failures. An
//! acceptance run marks the ops its model needs required, asserts the
//! backend prepared every one, and on a GPU asserts that each op's
//! conformance run, its PDL check unless inapplicable
//! ([`Check::inapplicable`]) and each property its `Need` requests passed.
//!
//! A case declares the op's role record over [`Spec`]s, so the kit binds
//! each role by its path (`x`, `residual.input`, `scratch.partial`) and
//! the oracle reads and returns host images by those paths. The kit
//! allocates one device buffer per role, passes views whose handles are
//! the buffers' device addresses, calls [`PreparedOp::launches`], loads
//! each symbol through [`Modules`] and launches the plans in order.

pub mod cases;
pub mod collective;
pub mod decode;
#[cfg(feature = "host-sampling")]
pub mod host;
pub mod modules;
pub mod properties;
pub mod units;

use crate::conformance::{Case, Expected, Holes, Images, Probes, Rows, Write};
use crate::poison::WriteMap;
use crate::{LabResult, Session};
use kernel_core::allocation::Traversal;
use kernel_core::backend::{Backend, Counts, Launch, Need, OpInfo, Prepared, Refused, View};
use kernel_core::contract::DType;
use kernel_core::properties::{Invariance, Property};
use serde_json::{Value, json};
use std::rc::Rc;

pub use decode::Decoded;
pub use modules::{Lookup, Modules};

/// What a role holds and how the run treats it.
enum Kind<'a> {
    /// Rows the launch reads, holding an image.
    Input(Vec<u8>),
    /// I32 IDs, -1 for a hole, indexing rows of `(role path, stride
    /// bytes)` of each pool ([`Case::ids`]).
    Ids {
        ids: Vec<i32>,
        pools: Vec<(String, usize)>,
        holes: Holes,
    },
    /// A persistent weight the launch reads, holding an image.
    Weight(Vec<u8>),
    /// Rows the launch writes, poisoned before each launch.
    Output(Write<'a>),
    /// Output rows filled with a pattern of the case's choice before each
    /// launch, for outputs whose valid values include all-ones elements.
    Filled(Write<'a>, Vec<u8>),
    /// In-place state holding an image before each launch, such as a KV
    /// cache: the launch must write the elements its map names, which are
    /// poisoned first, may rewrite the read-modify-write byte ranges, which
    /// keep their image, and keeps every other byte ([`Case::state`]).
    State(Vec<u8>, Write<'a>, Option<Updated<'a>>),
    /// Zeroed once before the first launch; every launch must leave it
    /// zero, as semaphores a graph replays must be.
    Zeroed,
    /// Scratch the launch may write anywhere (`true`) or must not write.
    Scratch(bool),
    /// The device count of live rows.
    Live,
}

/// The byte ranges of a state a launch reads and rewrites at a live count
/// over the images ([`Spec::state_updating`]).
type Updated<'a> = Box<dyn Fn(&Images, u32) -> LabResult<Vec<std::ops::Range<usize>>> + 'a>;

/// A role's host buffer: its dtype, rows and kind. A role record over
/// specs (`RmsNormBuffers<Spec>`) declares an op's case.
pub struct Spec<'a> {
    dtype: DType,
    rows: Rows,
    row_elements: usize,
    kind: Kind<'a>,
    /// Whether the role is route metadata ([`Spec::route_metadata`]).
    route_metadata: bool,
}

impl<'a> Spec<'a> {
    /// Input rows of `row_elements`, the case's live rows or a fixed count,
    /// holding `image`.
    pub fn input(dtype: DType, (rows, row_elements): (Rows, usize), image: Vec<u8>) -> Self {
        Self {
            dtype,
            rows,
            row_elements,
            kind: Kind::Input(image),
            route_metadata: false,
        }
    }

    /// I32 `ids` of `row_elements` per row indexing rows of each pool,
    /// `(role path, stride bytes)`: the run reruns them moved to the top of
    /// the pools and, with `holes`, with a row of -1 holes ([`Case::ids`]).
    pub fn ids(
        (rows, row_elements): (Rows, usize),
        ids: Vec<i32>,
        pools: &[(&str, usize)],
        holes: Holes,
    ) -> Self {
        let pools = pools
            .iter()
            .map(|&(pool, stride)| (pool.to_owned(), stride))
            .collect();
        Self {
            dtype: DType::I32,
            rows,
            row_elements,
            kind: Kind::Ids { ids, pools, holes },
            route_metadata: false,
        }
    }

    /// A persistent weight of `image`'s bytes.
    pub fn weight(image: Vec<u8>) -> Self {
        Self {
            dtype: DType::U8,
            rows: Rows::Fixed(1),
            row_elements: image.len(),
            kind: Kind::Weight(image),
            route_metadata: false,
        }
    }

    /// Output rows that every launch writes below the live count.
    pub fn output(dtype: DType, dims: (Rows, usize)) -> Self {
        Self::output_with(dtype, dims, Write::LiveRows)
    }

    /// Output rows that every launch writes as `write` says.
    pub fn output_with(
        dtype: DType,
        (rows, row_elements): (Rows, usize),
        write: Write<'a>,
    ) -> Self {
        Self {
            dtype,
            rows,
            row_elements,
            kind: Kind::Output(write),
            route_metadata: false,
        }
    }

    /// This output filled with `pattern` before each launch in place of
    /// the default poison, for an output whose valid values include
    /// all-ones elements (-1 IDs or padding), which the default fill
    /// cannot tell from an unwritten element ([`Case::output_with`]).
    /// Any other role is returned unchanged.
    pub fn filled_with(self, pattern: &[u8]) -> Self {
        match self.kind {
            Kind::Output(write) => Self {
                kind: Kind::Filled(write, pattern.to_vec()),
                ..self
            },
            _ => self,
        }
    }

    /// This output marked route metadata: data of a routed operation that
    /// depends on every routed row of a launch by design (expert counts,
    /// offsets, sorted rows), which claims of the routed scope
    /// (`Scope::RoutedOperation`) leave out of the property checks. Only
    /// marked outputs are left out.
    pub fn route_metadata(self) -> Self {
        Self {
            route_metadata: true,
            ..self
        }
    }

    /// In-place state of `rows` rows of `row_elements` holding `image`
    /// before each launch: every launch writes the elements `write` names
    /// (poisoned first) and keeps every other byte, as a KV cache's planes.
    /// The oracle gives the whole state.
    pub fn state(
        dtype: DType,
        (rows, row_elements): (Rows, usize),
        image: Vec<u8>,
        write: Write<'a>,
    ) -> Self {
        Self {
            dtype,
            rows,
            row_elements,
            kind: Kind::State(image, write, None),
            route_metadata: false,
        }
    }

    /// [`Spec::state`] whose launch also reads and rewrites the byte ranges
    /// `updated` gives, such as a recurrent slot a step advances in place:
    /// they keep the image before each launch (only `write` is poisoned),
    /// the oracle's whole state covers them, and the property checks
    /// compare them with the bytes `write` names.
    pub fn state_updating(
        dtype: DType,
        (rows, row_elements): (Rows, usize),
        image: Vec<u8>,
        write: Write<'a>,
        updated: impl Fn(&Images, u32) -> LabResult<Vec<std::ops::Range<usize>>> + 'a,
    ) -> Self {
        Self {
            dtype,
            rows,
            row_elements,
            kind: Kind::State(image, write, Some(Box::new(updated))),
            route_metadata: false,
        }
    }

    /// `bytes` zeroed before the first launch, which every launch must
    /// leave zero.
    pub fn zeroed(bytes: usize) -> Self {
        Self::bytes(bytes, Kind::Zeroed)
    }

    /// Scratch of `bytes` the launch may write.
    pub fn scratch(bytes: usize) -> Self {
        Self::bytes(bytes, Kind::Scratch(true))
    }

    /// A view of `bytes` the launch must not write, such as a role the op
    /// lists no bytes for.
    pub fn untouched(bytes: usize) -> Self {
        Self::bytes(bytes, Kind::Scratch(false))
    }

    /// The device count of live rows. A case declares at most one; the
    /// case of an op that takes none (an all-reduce, a ring prefix copy)
    /// launches each live count's rows, as [`Counts::rows`].
    pub fn live() -> Self {
        Self::bytes(4, Kind::Live)
    }

    fn bytes(bytes: usize, kind: Kind<'a>) -> Self {
        Self {
            dtype: DType::U8,
            rows: Rows::Fixed(1),
            row_elements: bytes,
            kind,
            route_metadata: false,
        }
    }
}

impl<'a> Spec<'a> {
    /// The tag the properties see this role by.
    fn tag(&self) -> Tag {
        match self.kind {
            Kind::Input(_) | Kind::Ids { .. } => Tag::Input,
            Kind::Weight(_) => Tag::Weight,
            Kind::Output(_) | Kind::Filled(..) => Tag::Output,
            Kind::State(..) => Tag::State,
            Kind::Zeroed => Tag::Zeroed,
            Kind::Scratch(_) => Tag::Scratch,
            Kind::Live => Tag::Live,
        }
    }

    /// This role's leaf at `path` in a case of `capacity` rows.
    fn leaf(&self, path: &str) -> LabResult<Leaf> {
        Ok(Leaf {
            path: path.into(),
            rows: self.rows,
            row_bytes: self.row_elements * WriteMap::of(self.dtype).bytes()?,
            tag: self.tag(),
            route_metadata: self.route_metadata,
        })
    }

    /// This role bound at `path` in `builder`, a case of `capacity` rows.
    fn bind(self, builder: Case<'a>, path: &str, capacity: u32) -> LabResult<Case<'a>> {
        let dims = (self.rows, self.row_elements);
        let bytes = self.leaf(path)?.row_bytes
            * match self.rows {
                Rows::Live => capacity as usize,
                Rows::Fixed(rows) => rows,
            };
        Ok(match self.kind {
            Kind::Input(image) | Kind::Weight(image) => {
                builder.input(path, self.dtype, dims, image)
            }
            Kind::Ids { ids, pools, holes } => {
                let pools = pools
                    .iter()
                    .map(|(pool, stride)| (pool.as_str(), *stride))
                    .collect::<Vec<_>>();
                builder.ids(path, dims, &ids, &pools, holes)
            }
            Kind::Output(write) => builder.output(path, self.dtype, dims, write),
            Kind::Filled(write, pattern) => {
                builder.output_with(path, self.dtype, dims, write, &pattern)
            }
            Kind::State(image, write, updated) => {
                builder.state(path, self.dtype, dims, image, write, move |images, live| {
                    updated
                        .as_ref()
                        .map_or_else(|| Ok(Vec::new()), |updated| updated(images, live))
                })
            }
            Kind::Zeroed => builder.state(
                path,
                self.dtype,
                dims,
                vec![0; bytes],
                Write::Map(Box::new(|_, _| WriteMap::of(DType::U8).none())),
                move |_: &Images, live: u32| {
                    Ok(if live == 0 {
                        Vec::new()
                    } else {
                        std::iter::once(0..bytes).collect()
                    })
                },
            ),
            Kind::Scratch(writes) => builder.scratch(path, bytes, move |_, _| {
                Ok(if writes {
                    std::iter::once(0..bytes).collect()
                } else {
                    Vec::new()
                })
            }),
            Kind::Live => builder.live(path),
        })
    }
}

/// How the properties see one role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Tag {
    Input,
    Weight,
    Output,
    State,
    Zeroed,
    Scratch,
    Live,
}

/// One role of a case: its path, layout and tag.
#[derive(Clone, Debug)]
pub(crate) struct Leaf {
    pub path: String,
    pub rows: Rows,
    /// Bytes of one row.
    pub row_bytes: usize,
    pub tag: Tag,
    /// Whether the case marks the role route metadata.
    pub route_metadata: bool,
}

/// A prepared op's launches at a row count, over views whose handles the
/// first argument gives by role path.
type Launches<'a> = Rc<dyn Fn(&dyn Fn(&str) -> u64, u32) -> LabResult<Vec<Launch>> + 'a>;

/// The check that the op's guarantees provide the requested exactness.
const NEED: &str = "need";

/// The check of reads before the PDL wait, behind a producer that rewrites
/// an input.
pub const PDL: &str = "pdl dependency order";

/// Why the PDL check does not apply to an op: no launch of it waits on a
/// producer, so no dependency order exists to probe.
const NO_PDL: &str = "the op's first launch does not wait on a PDL producer";

/// The check of launches at the capacity over fewer device-live rows.
const LIVE_COUNT: &str = "device live count";

/// The live count's name in the conformance run of a case that declares
/// none: no role path, so no launch takes it.
const NO_LIVE: &str = "#no device live count";

/// The role path of the op's generic workspace ([`OpInfo::workspace`]),
/// when it asks for one: distinct from every family role, some of which
/// are named `workspace` (attention's partials).
pub const WORKSPACE: &str = "backend_workspace";

/// A prepared op's conformance case.
pub struct OpCase<'a> {
    case: Case<'a>,
    info: OpInfo,
    capacity: u32,
    leaves: Vec<Leaf>,
    launches: Launches<'a>,
    /// A launch that does not read the device live count, if any, or the
    /// first launch when the case declares no live count.
    blind: Option<String>,
    /// Whether the case declares a device live count ([`Spec::live`]).
    counted: bool,
    /// Whether the case declares an oracle ([`OpCase::oracle`]).
    oracle: bool,
    /// The input the PDL copy producer rewrites, if the case has one.
    produced: Option<String>,
    /// Why the row property checks do not apply to the case, if they do
    /// The property checks that do not apply to the case, each with its
    /// reason ([`OpCase::without_properties`]).
    excluded: std::collections::BTreeMap<Property, String>,
    /// Why a case row is an item of several rows, if it is
    /// ([`OpCase::item_rows`]).
    item_rows: Option<String>,
    /// How its verify items map to decode launches, or why they do not
    /// ([`OpCase::decoded`]).
    decoded: Option<Result<Box<Decoded<'a>>, String>>,
    /// Whether the first launch waits on a PDL producer.
    pdl: bool,
}

impl<'a> OpCase<'a> {
    /// The case of `op` at `capacity` rows over `roles`, launched with
    /// [`Counts::rows`].
    pub fn new<S, R, B>(op: Prepared<B>, capacity: u32, roles: S) -> LabResult<Self>
    where
        S: Traversal<Spec<'a>, Mapped<usize> = R>,
        R: Traversal<usize, Mapped<View> = B> + Clone + 'a,
        B: 'a,
    {
        Self::with_counts(op, capacity, roles, Counts::rows)
    }

    /// [`OpCase::new`] with the counts of each row count from `counts`,
    /// for ops whose methods read more than [`Counts::rows`].
    pub fn with_counts<S, R, B>(
        op: Prepared<B>,
        capacity: u32,
        roles: S,
        counts: impl Fn(u32) -> Counts + 'a,
    ) -> LabResult<Self>
    where
        S: Traversal<Spec<'a>, Mapped<usize> = R>,
        R: Traversal<usize, Mapped<View> = B> + Clone + 'a,
        B: 'a,
    {
        let info = op.info().clone();
        // The specs in declaration order, and the record of their places.
        let mut specs = Vec::new();
        let places = roles.map(|spec| {
            specs.push(spec);
            specs.len() - 1
        });
        let mut paths = vec![String::new(); specs.len()];
        places.visit_named(&mut |path, &at| paths[at] = path.to_owned());
        if info.workspace > 0 {
            paths.push(WORKSPACE.into());
            specs.push(Spec::scratch(info.workspace as usize));
        }
        let symbols = info
            .symbols
            .iter()
            .map(|(symbol, _)| symbol.clone())
            .collect();
        let mut leaves = Vec::with_capacity(specs.len());
        let mut sizes = Vec::with_capacity(specs.len());
        for (path, spec) in paths.iter().zip(&specs) {
            let leaf = spec.leaf(path)?;
            let rows = match spec.rows {
                Rows::Live => capacity as usize,
                Rows::Fixed(rows) => rows,
            };
            sizes.push((rows * leaf.row_bytes) as u64);
            leaves.push(leaf);
        }
        let rows_of = |leaf: &Leaf| match leaf.rows {
            Rows::Live => capacity as usize,
            Rows::Fixed(rows) => rows,
        };
        let counted = match leaves.iter().filter(|leaf| leaf.tag == Tag::Live).count() {
            0 => false,
            1 => true,
            _ => return Err("a case declares at most one live count (Spec::live)".into()),
        };
        let workspace = (info.workspace > 0).then(|| sizes[sizes.len() - 1]);
        let launch_paths = paths.clone();
        let launches: Launches<'a> = Rc::new(move |handle: &dyn Fn(&str) -> u64, rows| {
            let views = places.clone().map(|at| View {
                handle: handle(&launch_paths[at]),
                bytes: sizes[at],
            });
            let workspace = workspace.map_or(View::EMPTY, |bytes| View {
                handle: handle(WORKSPACE),
                bytes,
            });
            op.launches(&views, workspace, counts(rows))
                .map_err(|refusal| format!("the op refuses {rows} rows: {refusal}"))
        });
        // An op some launch of which does not read the device live count
        // follows the host's rows alone, so its case plans each live count.
        let token = |path: &str| {
            leaves
                .iter()
                .position(|leaf| leaf.path == path)
                .map_or(0, |at| at as u64 + 1)
        };
        let live_path = leaves
            .iter()
            .find(|leaf| leaf.tag == Tag::Live)
            .map(|leaf| leaf.path.clone());
        let launched = launches(&token, capacity).unwrap_or_default();
        let blind = launched
            .iter()
            .find(|launch| {
                live_path.as_ref().is_none_or(|path| {
                    !launch.arguments.0.iter().any(|argument| {
                        *argument == kernel_core::contract::Argument::DeviceBuffer(token(path))
                    })
                })
            })
            .map(|launch| launch.symbol.clone());
        let pdl = launched.first().is_some_and(|launch| launch.plan.pdl);
        // The PDL copy producer rewrites a plain input of whole words that
        // the first launch, the one waiting on the producer, takes (a
        // pointer, tensor-map base or pointer-array entry), rows of the live
        // count first, so a launch that reads it before its PDL wait reads
        // poison.
        let first = launched
            .first()
            .map(|launch| taken(&launch.arguments))
            .unwrap_or_default();
        let produced = specs
            .iter()
            .zip(&leaves)
            .filter(|(spec, leaf)| {
                matches!(spec.kind, Kind::Input(_))
                    && (leaf.row_bytes * rows_of(leaf)) % 4 == 0
                    && first.contains(&token(&leaf.path))
            })
            .min_by_key(|(_, leaf)| leaf.rows != Rows::Live)
            .map(|(_, leaf)| leaf.path.clone());
        let plan = Rc::clone(&launches);
        let mut builder = Case::prepared(
            symbols,
            capacity,
            Box::new(move |handle, rows| plan(handle, rows)),
        );
        if !counted {
            builder = builder.live(NO_LIVE);
        }
        for (path, spec) in paths.iter().zip(specs) {
            builder = spec.bind(builder, path, capacity)?;
        }
        // Zeroed roles must come back zero.
        let zeroed = leaves
            .iter()
            .filter(|leaf| leaf.tag == Tag::Zeroed)
            .map(|leaf| (leaf.path.clone(), leaf.row_bytes))
            .collect::<Vec<_>>();
        if !zeroed.is_empty() {
            builder = builder.oracle(move |_, _| {
                Ok(zeroed
                    .iter()
                    .map(|(path, bytes)| (path.clone(), Expected::Bits(vec![0; *bytes])))
                    .collect())
            });
        }
        if blind.is_some() {
            builder = builder.host_rows();
        }
        if let Some(path) = &produced {
            builder = builder.produced(path);
        }
        Ok(Self {
            case: builder,
            info,
            capacity,
            leaves,
            launches,
            blind,
            counted,
            oracle: false,
            produced,
            excluded: std::collections::BTreeMap::new(),
            item_rows: None,
            decoded: None,
            pdl,
        })
    }

    /// The expected results at each live count, by role path: they must
    /// cover every output that writes there ([`Case::oracle`]). Zeroed
    /// roles are added.
    pub fn oracle(
        mut self,
        oracle: impl Fn(&Images, u32) -> LabResult<Vec<(String, Expected)>> + 'a,
    ) -> Self {
        let zeroed = self
            .leaves
            .iter()
            .filter(|leaf| leaf.tag == Tag::Zeroed)
            .map(|leaf| (leaf.path.clone(), leaf.row_bytes))
            .collect::<Vec<_>>();
        self.case = self.case.oracle(move |images, live| {
            let mut results = oracle(images, live)?;
            results.extend(
                zeroed
                    .iter()
                    .map(|(path, bytes)| (path.clone(), Expected::Bits(vec![0; *bytes]))),
            );
            Ok(results)
        });
        self.oracle = true;
        self
    }

    /// Reports the checks of `properties` as not exercised for `reason`,
    /// where the case's rows cannot show them: for items of several query
    /// rows each, moving a case row moves a whole item, so a comparison of
    /// one query row against itself alone (rows) or of a split between
    /// query rows (chunk) does not exist, while copying whole items (batch)
    /// still checks that items ignore each other.
    pub fn without_properties(mut self, properties: &[Property], reason: &str) -> Self {
        for &property in properties {
            self.excluded.insert(property, reason.into());
        }
        self
    }

    /// Reports every property check as not exercised for `reason`
    /// ([`OpCase::without_properties`] of them all).
    pub fn without_row_properties(self, reason: &str) -> Self {
        self.without_properties(
            &[
                Property::Chunk,
                Property::Rows,
                Property::Batch,
                Property::Ties,
                Property::Anchors,
            ],
            reason,
        )
    }

    /// Declares that a case row is an item of several rows, for `reason`:
    /// the chunk check, which compares single rows, is reported not
    /// exercised, and one-row bits the op claims or its need requests are
    /// checked against decode launches through the case's [`Decoded`]
    /// mapping, which the case must then give ([`OpCase::rows_mapping`]).
    /// The batch check still runs over whole items.
    pub fn item_rows(mut self, reason: &str) -> Self {
        self.item_rows = Some(reason.into());
        self.without_properties(&[Property::Chunk], reason)
    }

    /// How the case's verify items map to one-row decode launches, or why
    /// the backend gives no decode op for them ([`decode`]).
    pub fn decoded(mut self, decoded: LabResult<Decoded<'a>>) -> Self {
        self.decoded = Some(decoded.map(Box::new));
        self
    }

    /// Refuses a case whose rows are items of several rows when the op
    /// claims, or `need` requests, one-row bits and the case has no decode
    /// mapping (or its decode case does not declare): the rows check would
    /// compare nothing.
    pub fn rows_mapping(&self, need: Need) -> LabResult<()> {
        let one_row = |rows| matches!(rows, kernel_core::properties::Rows::OneRow { .. });
        let wanted = one_row(self.info.guarantees.rows) || one_row(need.exact.rows);
        let (Some(reason), true) = (&self.item_rows, wanted) else {
            return Ok(());
        };
        match &self.decoded {
            None => Err(format!(
                "the op's one-row bits cannot be checked: {reason}, and the case maps no verify \
                 item to decode launches"
            )),
            Some(Err(error)) => Err(format!(
                "the op's one-row bits cannot be checked: its decode op: {error}"
            )),
            Some(Ok(decoded)) => {
                decoded
                    .case
                    .validate()
                    .map_err(|error| format!("the decode case does not declare: {error}"))?;
                // The verify step must take each claim's rows.
                let images = self.case.images();
                for rows in [self.info.guarantees.rows, need.exact.rows] {
                    if let kernel_core::properties::Rows::OneRow { max } = rows {
                        decoded.verify_images(&images, max).map_err(|error| {
                            format!(
                                "the verify step cannot take one-row bits of {max} rows: {error}"
                            )
                        })?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Declares the device live count a step input the host writes before
    /// the step, which a launch may read before its PDL wait, for `reason`
    /// ([`Case::live_count_before_wait`]): the PDL phases then probe the
    /// produced input alone.
    pub fn live_count_before_wait(mut self, reason: &str) -> Self {
        self.case = self.case.live_count_before_wait(reason);
        self
    }

    /// Declares that the op computes its results without its launch of
    /// `symbol`, for `reason` ([`Case::redundant`]), so the null phase does
    /// not drop it.
    pub fn redundant_launch(mut self, symbol: &str, reason: &str) -> Self {
        self.case = self.case.redundant(symbol, reason);
        self
    }

    /// Live counts in place of the default ([`Case::live_counts`]).
    pub fn live_counts(mut self, counts: Vec<u32>) -> Self {
        self.case = self.case.live_counts(counts);
        self
    }

    /// Declares that output or state `name` writes nothing at the positive
    /// count `live`, for `reason` ([`Case::no_write`]).
    pub fn no_write(mut self, name: &str, live: u32, reason: &str) -> Self {
        self.case = self.case.no_write(name, live, reason);
        self
    }

    /// What the backend chose for the op.
    pub fn info(&self) -> &OpInfo {
        &self.info
    }

    /// Checks the declaration on the CPU: the case ([`Case::validate`]),
    /// which with an oracle requires its results for every output and
    /// state that writes, an oracle declared, and the launches at the
    /// capacity against the op's info. Returns the roles no launch takes,
    /// for the report.
    pub fn validate(&self) -> LabResult<Value> {
        if !self.oracle {
            return Err(
                "the case declares no oracle: a prepared op's case gives the oracle's \
                        result for every output and state the op writes"
                    .into(),
            );
        }
        self.case.validate()?;
        let token = |path: &str| {
            self.leaves
                .iter()
                .position(|leaf| leaf.path == path)
                .map_or(0, |at| at as u64 + 1)
        };
        let launches = (self.launches)(&token, self.capacity)?;
        for launch in &launches {
            let Some(&(_, shared)) = self
                .info
                .symbols
                .iter()
                .find(|(symbol, _)| *symbol == launch.symbol)
            else {
                return Err(format!(
                    "{} is not among the symbols the op lists",
                    launch.symbol
                ));
            };
            if launch.plan.dynamic_shared_bytes > shared {
                return Err(format!(
                    "{} plans {} bytes of dynamic shared memory; the op lists {shared}",
                    launch.symbol, launch.plan.dynamic_shared_bytes
                ));
            }
            if launch.rows == 0 {
                return Err(format!("{} records no rows", launch.symbol));
            }
        }
        for (path, bytes) in &self.info.bytes {
            let leaf = self
                .leaves
                .iter()
                .find(|leaf| leaf.path == *path)
                .ok_or_else(|| format!("the op sizes {path}, which the case does not bind"))?;
            let held = leaf.row_bytes
                * match leaf.rows {
                    Rows::Live => self.capacity as usize,
                    Rows::Fixed(rows) => rows,
                };
            if (held as u64) < *bytes {
                return Err(format!(
                    "{path} holds {held} bytes; the op sizes it {bytes}"
                ));
            }
        }
        let taken = launches
            .iter()
            .flat_map(|launch| taken(&launch.arguments))
            .collect::<std::collections::BTreeSet<_>>();
        let untaken = self
            .leaves
            .iter()
            .enumerate()
            .filter(|(at, _)| !taken.contains(&(*at as u64 + 1)))
            .map(|(_, leaf)| leaf.path.clone())
            .collect::<Vec<_>>();
        Ok(json!({
            "launches": launches.iter().map(|launch| &launch.symbol).collect::<Vec<_>>(),
            "roles_no_launch_takes": untaken,
            "rows_follow": match self.blind {
                Some(_) => "the host's rows",
                None => "the device live count",
            },
            "pdl_produced_input": self.produced,
        }))
    }

    /// Why the PDL phases cannot probe the op's dependency order, if they
    /// cannot: no producer overlaps a first launch without PDL, and the
    /// copy producer needs an input that launch takes.
    fn pdl_gap(&self) -> Option<&'static str> {
        if !self.pdl {
            Some(NO_PDL)
        } else if self.produced.is_none() {
            Some(
                "the op's first launch takes no plain input of whole words for a producer to \
                 rewrite",
            )
        } else {
            None
        }
    }

    /// Runs every check on `session` with each symbol's function from
    /// `functions`: the declaration, whether the op's guarantees provide
    /// the exactness `need` requests, the conformance run, the dependency
    /// order behind the PDL producer `probes` (not exercised without them),
    /// and the properties the op guarantees or `need` requests.
    pub fn checks(
        &self,
        need: Need,
        session: Option<&Session>,
        functions: Option<&Lookup<'_>>,
        probes: Option<&Probes>,
    ) -> Vec<Check> {
        let declaration = self.validate();
        let declared = declaration.is_ok();
        let mut checks = vec![Check::of("declaration", declaration)];
        let guaranteed = self.info.guarantees;
        checks.push(Check::of(
            NEED,
            if guaranteed.provides(need.exact) {
                Ok(json!({"needed": format!("{:?}", need.exact)}))
            } else {
                Err(format!(
                    "the op guarantees {guaranteed:?}, which does not provide the needed {:?}; \
                     the backend should have refused it",
                    need.exact
                ))
            },
        ));
        if let Some(symbol) = &self.blind {
            let reason = if self.counted {
                format!("{symbol} does not read the device live count")
            } else {
                "the op takes no device live count".to_owned()
            };
            checks.push(Check::not_exercised(
                LIVE_COUNT,
                &format!(
                    "{reason}, so the run launches each live count's rows; a graph stage captured \
                     at a larger bucket is not checked"
                ),
            ));
        }
        // The properties the op claims, then those only `need` requests,
        // each checked at its own claim.
        let claimed = guaranteed.claims();
        let requested = need
            .exact
            .claims()
            .into_iter()
            .filter(|property| !claimed.contains(property))
            .collect::<Vec<_>>();
        let (Some(session), Some(functions), true) = (session, functions, declared) else {
            let reason = if declared {
                "no GPU session"
            } else {
                "the declaration fails"
            };
            checks.push(Check::not_exercised("conformance", reason));
            checks.push(Check::not_exercised(PDL, self.pdl_gap().unwrap_or(reason)));
            for property in claimed.iter().chain(&requested) {
                let reason = self.excluded.get(property).map_or(reason, String::as_str);
                checks.push(Check::not_exercised(&properties::name(*property), reason));
            }
            return checks;
        };
        // The PDL phases run behind the backend's probes, with a copy
        // producer for a produced input; otherwise the run has none.
        let probes = probes.filter(|probes| self.produced.is_none() || probes.copy.is_some());
        let conformance = self.case.run_with(session, functions, probes);
        checks.push(Check {
            name: PDL.into(),
            outcome: match (self.pdl_gap(), probes, &self.produced, &conformance) {
                (Some(reason), ..) => Outcome::NotExercised(reason.into()),
                (None, None, ..) => Outcome::NotExercised(
                    "the backend's modules supply no PDL producer probes".into(),
                ),
                (None, Some(_), Some(input), Ok(_)) => Outcome::Passed(json!({"produced": input})),
                (None, Some(_), Some(_), Err(error))
                    if crate::conformance::pdl_phase_error(error) =>
                {
                    Outcome::Failed(format!("a PDL phase of the conformance run fails: {error}"))
                }
                // Not passed: the run stopped outside its PDL phases, which
                // then did not all run.
                (None, Some(_), Some(_), Err(_)) => Outcome::NotExercised(
                    "the conformance run fails outside its PDL phases (see conformance), so \
                     they did not all run"
                        .into(),
                ),
                (None, Some(_), None, _) => unreachable!("pdl_gap names a missing input"),
            },
        });
        checks.push(Check::of("conformance", conformance));
        for property in claimed.iter().chain(&requested) {
            if let Some(reason) = self.excluded.get(property) {
                checks.push(Check::not_exercised(&properties::name(*property), reason));
            }
        }
        let included = |properties: &[Property]| {
            properties
                .iter()
                .copied()
                .filter(|property| !self.excluded.contains_key(property))
                .collect::<Vec<_>>()
        };
        let (mut claimed, mut requested) = (included(&claimed), included(&requested));
        let images = self.case.images();
        let execute =
            |images: &Images, rows: u32| self.case.execute(session, functions, images, rows, rows);
        // One-row bits of items of several rows compare decode launches.
        if self.item_rows.is_some() {
            for (list, rows) in [
                (&mut claimed, guaranteed.rows),
                (&mut requested, need.exact.rows),
            ] {
                if !list.contains(&Property::Rows) {
                    continue;
                }
                list.retain(|&property| property != Property::Rows);
                checks.push(self.decode_check(session, functions, &images, rows));
            }
        }
        let written =
            |state: &str, images: &Images, live: u32| self.case.state_writes(state, images, live);
        checks.extend(properties::check(
            guaranteed,
            &claimed,
            &self.leaves,
            &images,
            self.capacity,
            &execute,
            &written,
        ));
        checks.extend(properties::check(
            need.exact,
            &requested,
            &self.leaves,
            &images,
            self.capacity,
            &execute,
            &written,
        ));
        checks
    }
}

impl OpCase<'_> {
    /// The rows check of an op over items of several rows at its claim
    /// `rows`: the verify step over every item, cut to the claim's rows,
    /// against its rows as decode launches.
    fn decode_check(
        &self,
        session: &Session,
        functions: &Lookup<'_>,
        images: &Images,
        rows: kernel_core::properties::Rows,
    ) -> Check {
        let name = properties::name(Property::Rows);
        let kernel_core::properties::Rows::OneRow { max } = rows else {
            return Check::of(&name, Err("no one-row claim to check".into()));
        };
        let Some(Ok(decoded)) = &self.decoded else {
            return Check::of(
                &name,
                Err("the case maps no verify item to decode launches".into()),
            );
        };
        let live = self.capacity;
        let execute = |images: &Images| self.case.execute(session, functions, images, live, live);
        Check::of(
            &name,
            decoded.check(session, functions, images, live, max, &execute),
        )
    }
}

/// The device handles `arguments` take: pointers, tensor-map bases and
/// pointer-array entries.
fn taken(arguments: &kernel_core::contract::ArgumentList) -> std::collections::BTreeSet<u64> {
    use kernel_core::contract::Argument;
    let mut taken = std::collections::BTreeSet::new();
    for argument in &arguments.0 {
        match argument {
            Argument::DeviceBuffer(handle) => {
                taken.insert(*handle);
            }
            Argument::TensorMap(map) => {
                taken.insert(map.base_buffer);
            }
            Argument::DevicePointerArray(handles) => taken.extend(handles),
            _ => {}
        }
    }
    taken
}

/// The outcome of one check.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// The check ran and passed; the evidence.
    Passed(Value),
    Failed(String),
    /// The check did not run, for this reason. It is never a pass.
    NotExercised(String),
}

/// One check of an op.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub name: String,
    pub outcome: Outcome,
}

impl Check {
    /// Whether the check does not apply to the op, as opposed to not having
    /// run: the PDL check of an op whose first launch waits on no producer.
    /// Every other check not exercised is a gap in the evidence.
    pub fn inapplicable(&self) -> bool {
        self.name == PDL && self.outcome == Outcome::NotExercised(NO_PDL.into())
    }

    pub(crate) fn of(name: &str, result: LabResult<Value>) -> Self {
        Self {
            name: name.into(),
            outcome: match result {
                Ok(evidence) => Outcome::Passed(evidence),
                Err(error) => Outcome::Failed(error),
            },
        }
    }

    pub(crate) fn not_exercised(name: &str, reason: &str) -> Self {
        Self {
            name: name.into(),
            outcome: Outcome::NotExercised(reason.into()),
        }
    }
}

/// How a backend prepares one required op, and its case over the
/// prepared op: `Err` for a refusal, `Ok(Err)` for a case the kit could not
/// declare.
type PrepareCase<'a> =
    Box<dyn Fn(&dyn Backend, Need) -> Result<LabResult<OpCase<'a>>, Refused> + 'a>;

/// One op of a model's required set: its name, how a backend prepares it
/// (the contract op and its `Need`), and its case generator.
pub struct Required<'a> {
    pub name: String,
    /// The rows and exactness the model needs: the backend prepares the op
    /// for it, and the kit checks the prepared op against it.
    pub need: Need,
    /// Whether a refusal fails the run. Otherwise a refusal is a result.
    pub required: bool,
    prepare: PrepareCase<'a>,
}

impl<'a> Required<'a> {
    /// The op `name` at `need`, which `prepare` prepares on a backend for
    /// that need and `case` declares over the prepared op. A refusal is a
    /// result.
    pub fn new<B: 'a>(
        name: impl Into<String>,
        need: Need,
        prepare: impl Fn(&dyn Backend, Need) -> Result<Prepared<B>, Refused> + 'a,
        case: impl Fn(Prepared<B>) -> LabResult<OpCase<'a>> + 'a,
    ) -> Self {
        Self {
            name: name.into(),
            need,
            required: false,
            prepare: Box::new(move |backend, need| prepare(backend, need).map(&case)),
        }
    }

    /// Gives the case its decode mapping from `decode`, which prepares the
    /// decode op its verify rows refer to on the same backend
    /// ([`OpCase::decoded`]); a refusal is kept as the reason the case has
    /// no mapping, which fails only an op whose one-row bits need it.
    pub fn with_decode(
        self,
        decode: impl Fn(&dyn Backend) -> Result<LabResult<Decoded<'a>>, Refused> + 'a,
    ) -> Self {
        let prepare = self.prepare;
        Self {
            prepare: Box::new(move |backend, need| {
                Ok(prepare(backend, need)?.map(|case| {
                    case.decoded(decode(backend).unwrap_or_else(|refused| {
                        Err(format!("the backend refuses it: {refused}"))
                    }))
                }))
            }),
            ..self
        }
    }

    /// Makes a refusal fail the run.
    pub fn required(self) -> Self {
        Self {
            required: true,
            ..self
        }
    }
}

/// What happened to one required op.
#[derive(Clone, Debug, PartialEq)]
pub enum OpResult {
    Refused(String),
    /// The backend prepared the op, but the kit could not declare its case.
    Undeclared(String),
    Prepared {
        implementation: String,
        guarantees: Invariance,
        checks: Vec<Check>,
    },
}

/// One required op's report.
#[derive(Clone, Debug, PartialEq)]
pub struct OpReport {
    pub name: String,
    pub required: bool,
    pub result: OpResult,
}

/// A backend's report over a required op set.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub backend: String,
    pub ops: Vec<OpReport>,
}

impl Report {
    /// Every failure: a refused required op, an undeclared case, a failed
    /// check. An empty list is not conformance evidence: refusals of ops
    /// not marked required and checks not exercised are not failures (see
    /// [`Report::not_exercised`] and [`Report::prepared`]).
    pub fn failures(&self) -> Vec<String> {
        let mut failures = Vec::new();
        for op in &self.ops {
            match &op.result {
                OpResult::Refused(reason) if op.required => {
                    failures.push(format!("{}: required, refused: {reason}", op.name));
                }
                OpResult::Refused(_) => {}
                OpResult::Undeclared(reason) => {
                    failures.push(format!("{}: case not declared: {reason}", op.name));
                }
                OpResult::Prepared { checks, .. } => {
                    for check in checks {
                        if let Outcome::Failed(error) = &check.outcome {
                            failures.push(format!("{} {}: {error}", op.name, check.name));
                        }
                    }
                }
            }
        }
        failures
    }

    /// The checks that did not run, by op, with the reason.
    pub fn not_exercised(&self) -> Vec<String> {
        let mut skipped = Vec::new();
        for op in &self.ops {
            if let OpResult::Prepared { checks, .. } = &op.result {
                for check in checks {
                    if let Outcome::NotExercised(reason) = &check.outcome {
                        skipped.push(format!("{} {}: {reason}", op.name, check.name));
                    }
                }
            }
        }
        skipped
    }

    /// The names of the ops the backend prepared.
    pub fn prepared(&self) -> Vec<&str> {
        self.ops
            .iter()
            .filter(|op| matches!(op.result, OpResult::Prepared { .. }))
            .map(|op| op.name.as_str())
            .collect()
    }

    /// The report as JSON, for a receipt.
    pub fn json(&self) -> Value {
        let ops = self
            .ops
            .iter()
            .map(|op| {
                let result = match &op.result {
                    OpResult::Refused(reason) => json!({"refused": reason}),
                    OpResult::Undeclared(reason) => json!({"undeclared": reason}),
                    OpResult::Prepared {
                        implementation,
                        guarantees,
                        checks,
                    } => {
                        let checks = checks
                            .iter()
                            .map(|check| {
                                let (outcome, detail) = match &check.outcome {
                                    Outcome::Passed(evidence) => ("passed", evidence.clone()),
                                    Outcome::Failed(error) => ("failed", json!(error)),
                                    Outcome::NotExercised(reason) => {
                                        ("not exercised", json!(reason))
                                    }
                                };
                                json!({"check": check.name, "outcome": outcome, "detail": detail})
                            })
                            .collect::<Vec<_>>();
                        json!({
                            "implementation": implementation,
                            "guarantees": format!("{guarantees:?}"),
                            "checks": checks,
                        })
                    }
                };
                json!({"op": op.name, "required": op.required, "result": result})
            })
            .collect::<Vec<_>>();
        json!({"backend": self.backend, "ops": ops})
    }
}

/// Runs `ops` against `backend`: prepares each op, declares its case and,
/// with a GPU `session`, runs every check with symbols `modules` loads.
/// Without a session the CPU checks run and the rest are reported as not
/// exercised.
pub fn conform(
    backend: &dyn Backend,
    modules: &dyn Modules,
    session: Option<&Session>,
    ops: &[Required<'_>],
) -> LabResult<Report> {
    let (functions, probes) = match session {
        Some(session) => (Some(modules.load(session)?), modules.probes(session)?),
        None => (None, None),
    };
    let mut reports = Vec::with_capacity(ops.len());
    for op in ops {
        let result = match (op.prepare)(backend, op.need) {
            Err(refused) => OpResult::Refused(refused.to_string()),
            Ok(Err(error)) => OpResult::Undeclared(error),
            Ok(Ok(case)) => match case.rows_mapping(op.need) {
                Err(error) => OpResult::Undeclared(error),
                Ok(()) => OpResult::Prepared {
                    implementation: case.info().implementation.clone(),
                    guarantees: case.info().guarantees,
                    checks: case.checks(op.need, session, functions.as_deref(), probes.as_ref()),
                },
            },
        };
        reports.push(OpReport {
            name: op.name.clone(),
            required: op.required,
            result,
        });
    }
    Ok(Report {
        backend: backend.name().into(),
        ops: reports,
    })
}

/// The host bytes of BF16 `values`.
pub fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// `count` BF16 values of magnitude 0.25 to 2 and either sign from `seed`:
/// no zero, so no output's sign of zero depends on the order of operations,
/// and sums of two stay exact in F32.
pub fn bf16_values(seed: u64, count: usize) -> Vec<u16> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            let magnitude = 0.25 + 1.75 * unit;
            let value = if state & 1 == 0 {
                magnitude
            } else {
                -magnitude
            };
            kernel_core::codec::f32_to_bf16(value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::modules::Functions;
    use super::*;
    use kernel_core::backend::PreparedOp;
    use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
    use kernel_core::rmsnorm::RmsNormBuffers;

    /// What a fake op gets wrong.
    #[derive(Clone, Copy, PartialEq)]
    enum Fault {
        None,
        /// It launches a symbol its info does not list.
        Symbol,
        /// It plans more dynamic shared memory than its info lists.
        Shared,
        /// It passes a handle the host never bound.
        Handle,
        /// It sizes a role the case does not bind.
        Role,
        /// It ignores the device live count.
        Blind,
        /// It claims no exactness.
        Claimless,
        /// It waits on a PDL producer and takes neither input.
        Unread,
        /// It waits on a PDL producer and takes only the weight.
        WeightOnly,
        /// It asks for a generic workspace.
        Workspace,
    }

    /// Modules a run without a session never loads.
    struct Unloaded;

    impl Modules for Unloaded {
        fn load<'s>(&'s self, _: &'s Session) -> LabResult<Functions<'s>> {
            Err("no modules".into())
        }
    }

    /// A one-launch RMSNorm-shaped op whose launch passes its roles.
    struct Fake {
        fault: Fault,
        info: OpInfo,
    }

    impl Fake {
        fn new(fault: Fault) -> Self {
            let claims = match fault {
                Fault::Claimless => Invariance::NONE,
                _ => Invariance::any_rows(4),
            };
            let mut info = OpInfo::new("fake.v1", claims, vec![("chop_fake_v1".into(), 64)]);
            if fault == Fault::Role {
                info.bytes.push(("scratch.partial".into(), 16));
            }
            if fault == Fault::Workspace {
                info.workspace = 32;
            }
            Self { fault, info }
        }
    }

    impl PreparedOp<RmsNormBuffers<View>> for Fake {
        fn info(&self) -> &OpInfo {
            &self.info
        }

        fn launches(
            &self,
            buffers: &RmsNormBuffers<View>,
            _workspace: View,
            counts: Counts,
        ) -> Result<Vec<Launch>, Refusal> {
            let handle = |view: View| Argument::DeviceBuffer(view.handle);
            let mut arguments = vec![
                handle(buffers.x),
                handle(buffers.weight),
                handle(buffers.y),
                handle(buffers.live_rows),
            ];
            if let Some(residual) = &buffers.residual {
                arguments.extend([handle(residual.input), handle(residual.output)]);
            }
            match self.fault {
                Fault::Unread => arguments.drain(..2).for_each(drop),
                Fault::WeightOnly => {
                    arguments.remove(0);
                }
                Fault::Handle => arguments.push(Argument::DeviceBuffer(999)),
                Fault::Blind => {
                    arguments.pop();
                }
                _ => {}
            }
            Ok(vec![Launch {
                symbol: match self.fault {
                    Fault::Symbol => "chop_other_v1".into(),
                    _ => "chop_fake_v1".into(),
                },
                rows: counts.rows,
                plan: LaunchPlan {
                    grid: [counts.rows, 1, 1],
                    block: [32, 1, 1],
                    dynamic_shared_bytes: if self.fault == Fault::Shared { 128 } else { 64 },
                    workspace_bytes: 0,
                    cluster: [1, 1, 1],
                    pdl: matches!(self.fault, Fault::Unread | Fault::WeightOnly),
                },
                arguments: ArgumentList(arguments),
                collective: false,
                persistent: false,
            }])
        }
    }

    const WIDTH: usize = 8;

    fn case(fault: Fault) -> LabResult<OpCase<'static>> {
        Ok(declared(fault)?.oracle(|images, _| {
            Ok(vec![(
                "y".into(),
                Expected::Bits(images.bytes("x")?.to_vec()),
            )])
        }))
    }

    /// The fake op's case without an oracle.
    fn declared(fault: Fault) -> LabResult<OpCase<'static>> {
        let rows = |seed| {
            Spec::input(
                DType::Bf16,
                (Rows::Live, WIDTH),
                bf16_bytes(&bf16_values(seed, 4 * WIDTH)),
            )
        };
        let roles = RmsNormBuffers {
            x: rows(1),
            weight: Spec::input(
                DType::Bf16,
                (Rows::Fixed(1), WIDTH),
                bf16_bytes(&bf16_values(2, WIDTH)),
            ),
            residual: None,
            y: Spec::output(DType::Bf16, (Rows::Live, WIDTH)),
            live_rows: Spec::live(),
        };
        let op: Prepared<RmsNormBuffers<View>> = Box::new(Fake::new(fault));
        OpCase::new(op, 4, roles)
    }

    /// A state's read-modify-write ranges reach the case: the property
    /// checks compare them with the write map, and only the write map is
    /// poisoned before a launch.
    #[test]
    fn a_state_updating_passes_its_ranges_to_the_case() {
        let row_bytes = WIDTH * 2;
        // Live rows are written; the row after them is read and rewritten.
        let case = declared_with_state(Spec::state_updating(
            DType::Bf16,
            (Rows::Fixed(5), WIDTH),
            bf16_bytes(&bf16_values(3, 5 * WIDTH)),
            Write::Map(Box::new(|_, live| {
                WriteMap::of(DType::Bf16).rows(WIDTH, 0..live as usize)
            })),
            move |_, live| {
                let live = live as usize;
                Ok(std::iter::once(live * row_bytes..(live + 1) * row_bytes)
                    .filter(|_| live > 0)
                    .collect())
            },
        ))
        .oracle(|images, _| {
            Ok(vec![(
                "y".into(),
                Expected::Bits(images.bytes("y")?.to_vec()),
            )])
        });
        case.validate()
            .expect("a state with updated ranges declares");
        let images = case.case.images();
        assert_eq!(
            case.case.state_writes("y", &images, 2).unwrap(),
            vec![0..3 * row_bytes]
        );
        assert_eq!(
            case.case.state_writes("y", &images, 1).unwrap(),
            vec![0..2 * row_bytes]
        );
        // A plain state's writes are its write map alone.
        let plain = declared_with_state(Spec::state(
            DType::Bf16,
            (Rows::Fixed(5), WIDTH),
            vec![0; 5 * row_bytes],
            Write::Map(Box::new(|_, live| {
                WriteMap::of(DType::Bf16).rows(WIDTH, 0..live as usize)
            })),
        ));
        assert_eq!(
            plain.case.state_writes("y", &images, 2).unwrap(),
            vec![0..2 * row_bytes]
        );
    }

    /// The fake op's case with `y` as its output role.
    fn declared_with_state(y: Spec<'static>) -> OpCase<'static> {
        let roles = RmsNormBuffers {
            x: Spec::input(
                DType::Bf16,
                (Rows::Live, WIDTH),
                bf16_bytes(&bf16_values(1, 4 * WIDTH)),
            ),
            weight: Spec::input(
                DType::Bf16,
                (Rows::Fixed(1), WIDTH),
                bf16_bytes(&bf16_values(2, WIDTH)),
            ),
            residual: None,
            y,
            live_rows: Spec::live(),
        };
        let op: Prepared<RmsNormBuffers<View>> = Box::new(Fake::new(Fault::None));
        OpCase::new(op, 4, roles).unwrap()
    }

    #[test]
    fn a_faithful_op_declares_and_reads_the_device_live_count() {
        let case = case(Fault::None).unwrap();
        let evidence = case.validate().unwrap();
        assert_eq!(evidence["rows_follow"], "the device live count");
        assert_eq!(evidence["launches"], json!(["chop_fake_v1"]));
        // The copy producer rewrites the live input rows.
        assert_eq!(evidence["pdl_produced_input"], "x");
        let checks = case.checks(Need::rows(4), None, None, None);
        assert!(matches!(checks[0].outcome, Outcome::Passed(_)));
        assert_eq!(checks[1].name, NEED);
        assert!(matches!(checks[1].outcome, Outcome::Passed(_)));
        // Without a GPU nothing else counts as passed.
        assert!(
            checks[2..]
                .iter()
                .all(|check| matches!(check.outcome, Outcome::NotExercised(_)))
        );
        assert!(checks.iter().any(|check| check.name == PDL));
        assert_eq!(checks.len(), 4 + Invariance::any_rows(4).claims().len());
    }

    /// The copy producer rewrites only an input the first launch takes;
    /// without one the dependency order is not exercised, never passed.
    #[test]
    fn the_pdl_producer_rewrites_an_input_the_first_launch_takes() {
        let case = case(Fault::WeightOnly).unwrap();
        assert_eq!(case.validate().unwrap()["pdl_produced_input"], "weight");
        let case = case_unread();
        assert!(case.validate().unwrap()["pdl_produced_input"].is_null());
        let checks = case.checks(Need::rows(4), None, None, None);
        let pdl = checks.iter().find(|check| check.name == PDL).unwrap();
        assert!(
            matches!(&pdl.outcome, Outcome::NotExercised(reason) if reason.contains("takes no plain input")),
            "{pdl:?}"
        );
    }

    /// Only a PDL check of an op without PDL is inapplicable; one the
    /// backend's probes could not run is a gap.
    #[test]
    fn only_a_pdl_check_without_pdl_is_inapplicable() {
        let check = |outcome| Check {
            name: PDL.into(),
            outcome,
        };
        assert!(check(Outcome::NotExercised(NO_PDL.into())).inapplicable());
        let missing = "the backend's modules supply no PDL producer probes";
        assert!(!check(Outcome::NotExercised(missing.into())).inapplicable());
        assert!(!check(Outcome::Passed(json!({}))).inapplicable());
        let other = Check {
            name: "conformance".into(),
            outcome: Outcome::NotExercised(NO_PDL.into()),
        };
        assert!(!other.inapplicable());
        // A fake op whose first launch has no PDL reports exactly that.
        let checks = case(Fault::None)
            .unwrap()
            .checks(Need::rows(4), None, None, None);
        let pdl = checks.iter().find(|check| check.name == PDL).unwrap();
        assert!(pdl.inapplicable(), "{pdl:?}");
    }

    /// A case whose rows are not the claims' rows reports each property as
    /// not exercised with its reason.
    #[test]
    fn a_case_without_row_properties_names_its_reason() {
        let case = case(Fault::None)
            .unwrap()
            .without_row_properties("rows are items");
        let checks = case.checks(Need::rows(4), None, None, None);
        for property in Invariance::any_rows(4).claims() {
            let check = checks
                .iter()
                .find(|check| check.name == properties::name(property))
                .unwrap();
            assert_eq!(
                check.outcome,
                Outcome::NotExercised("rows are items".into())
            );
        }
    }

    /// A case of items of several rows declares one-row bits only with a
    /// decode mapping that declares and takes the claim's rows: without
    /// one, with the decode op refused, or with a verify step that refuses
    /// the claim, a claiming op or a needing one refuses to declare; an op
    /// neither claiming nor needing them declares without one.
    #[test]
    fn one_row_bits_of_item_rows_need_a_decode_mapping() {
        let items = || case(Fault::None).unwrap().item_rows("rows are items");
        let needs = Need::rows(4);
        assert!(case(Fault::None).unwrap().rows_mapping(needs).is_ok());
        let error = items().rows_mapping(needs).unwrap_err();
        assert!(error.contains("maps no verify item"), "{error}");
        let claimless = case(Fault::Claimless).unwrap().item_rows("rows are items");
        assert!(claimless.rows_mapping(needs).is_ok());
        let needed = needs.with_exact(Invariance::one_row(4));
        let claimless = case(Fault::Claimless).unwrap().item_rows("rows are items");
        assert!(claimless.rows_mapping(needed).is_err());
        let refused = items().decoded(Err("the backend refuses it".into()));
        let error = refused.rows_mapping(needs).unwrap_err();
        assert!(error.contains("the backend refuses it"), "{error}");
        let mapped = |limit: u32| {
            items().decoded(Ok(Decoded::new(
                case(Fault::None).unwrap(),
                move |images: &Images, max| {
                    if max < limit {
                        Err(format!("items hold {limit} rows"))
                    } else {
                        Ok(images.clone())
                    }
                },
                |_: &decode::Verified<'_>, _: &Images| Ok(Vec::new()),
            )))
        };
        assert!(mapped(4).rows_mapping(needs).is_ok());
        let error = mapped(8).rows_mapping(needs).unwrap_err();
        assert!(error.contains("items hold 8 rows"), "{error}");
        // The chunk check is excluded with the reason; rows are not.
        let checks = items().checks(needs, None, None, None);
        let reason = |property| {
            checks
                .iter()
                .find(|check| check.name == properties::name(property))
                .map(|check| check.outcome.clone())
        };
        assert_eq!(
            reason(Property::Chunk),
            Some(Outcome::NotExercised("rows are items".into()))
        );
        assert_eq!(
            reason(Property::Rows),
            Some(Outcome::NotExercised("no GPU session".into()))
        );
    }

    /// Excluding some properties leaves the others to their own checks:
    /// without a GPU they say so, while the excluded ones give their
    /// reason.
    #[test]
    fn excluded_properties_name_their_reason_and_others_run() {
        let case = case(Fault::None)
            .unwrap()
            .without_properties(&[Property::Rows, Property::Chunk], "query rows");
        let checks = case.checks(Need::rows(4), None, None, None);
        let outcome = |property| {
            checks
                .iter()
                .find(|check| check.name == properties::name(property))
                .unwrap()
                .outcome
                .clone()
        };
        assert_eq!(
            outcome(Property::Rows),
            Outcome::NotExercised("query rows".into())
        );
        assert_eq!(
            outcome(Property::Chunk),
            Outcome::NotExercised("query rows".into())
        );
        assert_eq!(
            outcome(Property::Batch),
            Outcome::NotExercised("no GPU session".into())
        );
    }

    /// The unread fake's case: its oracle gives zeros, which no input sets.
    fn case_unread() -> OpCase<'static> {
        declared(Fault::Unread)
            .unwrap()
            .oracle(|_, _| Ok(vec![("y".into(), Expected::Bits(vec![0; 4 * WIDTH * 2]))]))
    }

    /// A state binds as in-place state: its image, a write map the oracle
    /// covers, and no row property compares it.
    #[test]
    fn a_state_role_declares_with_its_write_map() {
        let image = bf16_bytes(&bf16_values(3, 4 * WIDTH));
        let rows = |seed| {
            Spec::input(
                DType::Bf16,
                (Rows::Live, WIDTH),
                bf16_bytes(&bf16_values(seed, 4 * WIDTH)),
            )
        };
        let roles = RmsNormBuffers {
            x: rows(1),
            weight: rows(2),
            residual: Some(kernel_core::rmsnorm::FusedResidual {
                input: rows(4),
                output: Spec::state(
                    DType::Bf16,
                    (Rows::Live, WIDTH),
                    image.clone(),
                    Write::LiveRows,
                ),
            }),
            y: Spec::output(DType::Bf16, (Rows::Live, WIDTH)),
            live_rows: Spec::live(),
        };
        let op: Prepared<RmsNormBuffers<View>> = Box::new(Fake::new(Fault::None));
        let case = OpCase::new(op, 4, roles).unwrap().oracle(|images, _| {
            Ok(vec![
                ("y".into(), Expected::Bits(images.bytes("x")?.to_vec())),
                (
                    "residual.output".into(),
                    Expected::Bits(images.bytes("residual.output")?.to_vec()),
                ),
            ])
        });
        case.validate().unwrap();
        let leaf = case
            .leaves
            .iter()
            .find(|leaf| leaf.path == "residual.output")
            .unwrap();
        assert_eq!(leaf.tag, Tag::State);
        // Without the state's result the oracle does not cover it.
        let roles = RmsNormBuffers {
            x: rows(1),
            weight: rows(2),
            residual: Some(kernel_core::rmsnorm::FusedResidual {
                input: rows(4),
                output: Spec::state(DType::Bf16, (Rows::Live, WIDTH), image, Write::LiveRows),
            }),
            y: Spec::output(DType::Bf16, (Rows::Live, WIDTH)),
            live_rows: Spec::live(),
        };
        let op: Prepared<RmsNormBuffers<View>> = Box::new(Fake::new(Fault::None));
        let case = OpCase::new(op, 4, roles).unwrap().oracle(|images, _| {
            Ok(vec![(
                "y".into(),
                Expected::Bits(images.bytes("x")?.to_vec()),
            )])
        });
        let error = case.validate().unwrap_err();
        assert!(error.contains("residual.output"), "{error}");
    }

    /// The generic workspace binds under its own path, beside any family
    /// role named `workspace`.
    #[test]
    fn the_generic_workspace_binds_under_a_reserved_path() {
        let case = case(Fault::Workspace).unwrap();
        case.validate().unwrap();
        let leaf = case
            .leaves
            .iter()
            .find(|leaf| leaf.path == WORKSPACE)
            .expect("the workspace's role");
        assert_eq!((leaf.tag, leaf.row_bytes), (Tag::Scratch, 32));
        assert_eq!(WORKSPACE, "backend_workspace");
    }

    /// A case without an oracle passes a launch that copies its input.
    #[test]
    fn a_case_without_an_oracle_fails_the_declaration() {
        let error = declared(Fault::None).unwrap().validate().unwrap_err();
        assert!(error.contains("declares no oracle"), "{error}");
    }

    /// A backend that prepares an op for a need it cannot meet, instead of
    /// refusing it.
    struct IgnoresNeed;

    impl Backend for IgnoresNeed {
        fn name(&self) -> &str {
            "ignores-need"
        }

        fn rmsnorm(
            &self,
            _: &kernel_core::rmsnorm::RmsNormRequirement,
            _: Need,
        ) -> Result<Prepared<RmsNormBuffers<View>>, Refused> {
            Ok(Box::new(Fake::new(Fault::Claimless)))
        }
    }

    #[test]
    fn an_op_short_of_its_need_fails_and_its_needed_properties_are_planned() {
        let need = Need::rows(4).with_exact(Invariance::any_rows(4));
        let required = Required::new(
            "norm",
            need,
            |backend, need| {
                let requirement = kernel_core::rmsnorm::RmsNormRequirement::bf16(
                    WIDTH as u32,
                    1e-6,
                    kernel_core::contract::RmsNormKind::Plain,
                    false,
                    kernel_core::rmsnorm::RmsNormNumerics::TREE_256,
                );
                backend.rmsnorm(&requirement, need)
            },
            |_| case(Fault::Claimless),
        );
        let report = conform(&IgnoresNeed, &Unloaded, None, &[required]).unwrap();
        let failures = report.failures();
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("norm need"), "{failures:?}");
        // The needed properties are planned though the op claims none.
        let OpResult::Prepared { checks, .. } = &report.ops[0].result else {
            panic!("prepared");
        };
        for property in need.exact.claims() {
            assert!(
                checks
                    .iter()
                    .any(|check| check.name == properties::name(property))
            );
        }
    }

    #[test]
    fn launches_that_break_the_op_info_fail_the_declaration() {
        for (fault, reason) in [
            (Fault::Symbol, "not among the symbols"),
            (Fault::Shared, "dynamic shared memory"),
            (Fault::Handle, "names no buffer of the case"),
            (Fault::Role, "does not bind"),
        ] {
            let error = case(fault).unwrap().validate().expect_err(reason);
            assert!(error.contains(reason), "{error}");
        }
    }

    /// An op that takes no device live count (an all-reduce, a ring prefix
    /// copy) declares no Spec::live and runs the host's rows.
    #[test]
    fn a_case_without_a_live_count_runs_the_hosts_rows() {
        let roles = RmsNormBuffers {
            x: Spec::input(
                DType::Bf16,
                (Rows::Live, WIDTH),
                bf16_bytes(&bf16_values(1, 4 * WIDTH)),
            ),
            weight: Spec::input(
                DType::Bf16,
                (Rows::Fixed(1), WIDTH),
                bf16_bytes(&bf16_values(2, WIDTH)),
            ),
            residual: None,
            y: Spec::output(DType::Bf16, (Rows::Live, WIDTH)),
            live_rows: Spec::untouched(4),
        };
        let op: Prepared<RmsNormBuffers<View>> = Box::new(Fake::new(Fault::None));
        let case = OpCase::new(op, 4, roles).unwrap().oracle(|images, _| {
            Ok(vec![(
                "y".into(),
                Expected::Bits(images.bytes("x")?.to_vec()),
            )])
        });
        assert_eq!(case.validate().unwrap()["rows_follow"], "the host's rows");
        let checks = case.checks(Need::rows(4), None, None, None);
        assert!(checks.iter().any(|check| check.name == LIVE_COUNT
            && matches!(&check.outcome, Outcome::NotExercised(reason) if reason.contains("takes no device live count"))));
        // Two live counts are refused.
        let roles = RmsNormBuffers {
            x: Spec::input(
                DType::Bf16,
                (Rows::Live, WIDTH),
                bf16_bytes(&bf16_values(1, 4 * WIDTH)),
            ),
            weight: Spec::live(),
            residual: None,
            y: Spec::output(DType::Bf16, (Rows::Live, WIDTH)),
            live_rows: Spec::live(),
        };
        let op: Prepared<RmsNormBuffers<View>> = Box::new(Fake::new(Fault::None));
        let Err(error) = OpCase::new(op, 4, roles) else {
            panic!("two live counts declare");
        };
        assert!(error.contains("at most one live count"), "{error}");
    }

    #[test]
    fn an_op_blind_to_the_live_count_runs_the_hosts_rows() {
        let case = case(Fault::Blind).unwrap();
        assert_eq!(case.validate().unwrap()["rows_follow"], "the host's rows");
        let checks = case.checks(Need::rows(4), None, None, None);
        assert!(checks.iter().any(|check| check.name == LIVE_COUNT
            && matches!(&check.outcome, Outcome::NotExercised(reason) if reason.contains("chop_fake_v1"))));
    }

    #[test]
    fn a_report_fails_on_required_refusals_and_failed_checks_only() {
        let report = |required, result| Report {
            backend: "fake".into(),
            ops: vec![OpReport {
                name: "op".into(),
                required,
                result,
            }],
        };
        let refused = || OpResult::Refused("no".into());
        assert!(report(false, refused()).failures().is_empty());
        assert_eq!(report(true, refused()).failures().len(), 1);
        let checks = |outcome| OpResult::Prepared {
            implementation: "fake.v1".into(),
            guarantees: Invariance::NONE,
            checks: vec![Check {
                name: "conformance".into(),
                outcome,
            }],
        };
        let skipped = report(false, checks(Outcome::NotExercised("no GPU".into())));
        assert!(skipped.failures().is_empty());
        assert_eq!(skipped.not_exercised().len(), 1);
        assert_eq!(
            report(false, checks(Outcome::Failed("bits".into()))).failures(),
            vec!["op conformance: bits".to_owned()]
        );
        assert_eq!(
            report(false, OpResult::Undeclared("no packer".into()))
                .failures()
                .len(),
            1
        );
    }

    #[test]
    fn generated_values_are_nonzero_and_vary() {
        let values = bf16_values(7, 64);
        assert!(values.iter().all(|&bits| bits & 0x7fff != 0));
        assert!(
            values
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                > 32
        );
        assert_eq!(values, bf16_values(7, 64));
    }
}
