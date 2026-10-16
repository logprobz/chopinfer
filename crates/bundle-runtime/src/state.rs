//! Declared per-request state.
//!
//! A model declares, for one instance configuration, what each sequence
//! keeps beyond its paged KV ([`StateBuffer`]) and the streams its KV pages
//! hold ([`PagedLayers`] of [`KvStream`]s). The declaration fixes where a
//! slot's entry lies in each buffer, what a prefix snapshot copies and in
//! which order, how a step's rejected verify rows leave each buffer, and
//! what a cancelled step puts back. The meaning of the bytes stays with the
//! model; the mechanics that read the declaration are shared.
//!
//! A buffer holds `layers` layers one after another. A layer holds plane 0
//! of every entry, then plane 1 of every entry, and so on. A shadow is a
//! second copy of these working planes, after them or in a buffer of its
//! own. Pooled checkpoints and anchor spares are further entries after a
//! buffer's own; separate ones are buffers of the same layout. A snapshot
//! lays the captured buffers' pieces end to end in declaration order, layer
//! by layer and plane by plane within a buffer, followed by the model's
//! header.

use std::collections::BTreeSet;

/// How many entries each plane of a buffer holds in one instance, before
/// any pooled checkpoints and anchor spares.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Entries {
    /// One per served sequence (`Items::get`).
    Served,
    /// One per item of the largest graph bucket (`Items::capacity`).
    Bucket,
    /// A fixed count, whatever the instance serves.
    Fixed(u32),
}

/// What one entry holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scope {
    /// The sequence's state at its committed boundary.
    Slot,
    /// The last `tokens` positions, each at `position % tokens`.
    WindowRing { tokens: u64 },
    /// A circular history of `rows` rows, each at `position % rows`.
    Rows { rows: u64 },
}

impl Scope {
    /// Positions an entry holds, if entries are addressed by position.
    pub const fn positions(self) -> Option<u64> {
        match self {
            Self::Slot => None,
            Self::WindowRing { tokens } => Some(tokens),
            Self::Rows { rows } => Some(rows),
        }
    }
}

/// What a prefix snapshot holds of an entry. A capture copies the whole
/// entry; the kind states which part of it the restored request reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Capture {
    /// The buffer is not part of a snapshot.
    None,
    /// The whole state of a [`Scope::Slot`] entry.
    Full,
    /// The last `tokens` positions of a [`Scope::WindowRing`].
    RingWindow { tokens: u64 },
    /// The last `rows` rows of a [`Scope::Rows`] history.
    LastRows { rows: u64 },
}

/// Where a buffer's shadow copy of its working planes lies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Shadow {
    None,
    /// After the working planes, in the same buffer.
    Adjacent,
    /// At the same offsets of a separate buffer of the working planes' size.
    Separate,
}

/// The items of a step that take checkpoints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointItems {
    /// Up to this many drafting items, in verify-item order, capped at the
    /// buffer's own entries.
    Drafting(u32),
    /// Every item of the largest graph bucket, by its index in the step.
    Bucket,
}

/// Checkpoint entries each checkpointing item takes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointRows {
    /// The instance's verify rows per item.
    Verify,
    /// A fixed stride, at least the instance's verify rows.
    Fixed(u32),
}

/// Per-row checkpoints of a verify step: entry `item * rows + row` of the
/// checkpoints holds the state after row `row` of checkpointing item `item`.
/// A step has them when the instance verifies (two or more rows per item).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Checkpoints {
    pub items: CheckpointItems,
    pub rows: CheckpointRows,
    /// After the buffer's own entries, or in a separate buffer of the same
    /// layout.
    pub pooled: bool,
}

/// How a verify step's rejected rows leave the buffer when the step commits
/// a prefix of an item's rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rollback {
    /// Rejected rows never write it.
    None,
    /// Rejected rows write positions past the committed end, and the next
    /// step rewrites them before any row reads them: rollback only moves the
    /// committed end.
    Truncate,
    /// The step saves the entries it touches to the shadow before it runs
    /// (a [`Region::Touched`] cancel save), and commit copies the rejected
    /// rows' positions back.
    RestoreTail,
    /// The verify rows write their states to checkpoints instead of the
    /// slot, and commit copies the last kept row's into the slot. This
    /// returns any per-row state, a partial group's tail included, to the
    /// accepted prefix.
    Checkpoint(Checkpoints),
}

/// Where a verify item's slot entry comes from at commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Commit {
    /// The working entry stays as the step left it.
    Working,
    /// The working entry, with the positions of rows `kept..rows` copied
    /// back from the shadow.
    RestoreTail { kept: u32, rows: u32 },
    /// Checkpoint row `row` of the item.
    Checkpoint { row: u32 },
}

impl Rollback {
    /// Where a verify item that keeps `kept` of its `rows` rows (at least
    /// one) finds its committed entry.
    pub const fn commit(self, rows: u32, kept: u32) -> Commit {
        match self {
            Self::None | Self::Truncate => Commit::Working,
            Self::RestoreTail if kept < rows => Commit::RestoreTail { kept, rows },
            Self::RestoreTail => Commit::Working,
            Self::Checkpoint(_) => Commit::Checkpoint { row: kept - 1 },
        }
    }
}

/// Which work of a step writes a buffer, for a cancel save.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Writer {
    /// Any step.
    Any,
    /// A step with a prefill phase.
    Prefill,
    /// A step in which the drafter runs.
    Drafter,
}

/// What a cancel save covers of the working planes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Region {
    /// The entries the step touches, which the step lists.
    Touched,
    /// All of them.
    All,
    /// Layers `first..first + count`.
    Layers { first: u64, count: u64 },
}

/// One save before a step: when `writer` runs in it, `region` goes to the
/// shadow.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Save {
    pub writer: Writer,
    pub region: Region,
}

/// What a cancelled step puts back.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Cancel {
    /// Nothing: the step's writes lie past the committed end, or reach the
    /// entry only at commit.
    None,
    /// The entries the step touched become invalid: their requests retire
    /// and start again.
    Invalidate,
    /// Before a step, the first save whose writer runs in it copies its
    /// region to the shadow, and cancel copies the region back.
    Restore(Vec<Save>),
}

/// Which request may restore a snapshot that holds the buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Owner {
    /// Any request whose tokens match the snapshot's.
    Shared,
    /// Only the request that captured it, when the drafter's frontier in
    /// the header lags into a page before the last: the continuation would
    /// rewrite drafter KV in pages other requests share.
    FrontierOwner,
}

/// Whether a step keeps the state at prefix anchors inside its rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Anchors {
    None,
    /// Up to `per_step` prompt-ending items per step keep their state at the
    /// anchor in spare entries, which the kernels that produce the state
    /// write. The spares follow the buffer's own entries and checkpoints when
    /// `pooled`; otherwise they are a separate buffer of the same layout
    /// with `per_step` entries.
    Spare {
        per_step: u32,
        pooled: bool,
    },
}

/// One buffer of per-request state on one rank.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateBuffer {
    pub name: &'static str,
    pub layers: u64,
    /// Bytes of one entry's planes in one layer on one rank, in layout
    /// order. They add up to the entry's bytes per layer and rank.
    pub planes: Vec<u64>,
    pub entries: Entries,
    pub scope: Scope,
    pub shadow: Shadow,
    pub capture: Capture,
    pub rollback: Rollback,
    pub cancel: Cancel,
    pub owner: Owner,
    /// Whether the drafter owns it: it exists only with the drafter loaded.
    pub drafter: bool,
    pub anchors: Anchors,
}

/// What an instance serves, which sizes its buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Extent {
    /// Served sequences (`Items::get`).
    pub served: u32,
    /// Items of the largest graph bucket (`Items::capacity`).
    pub bucket: u32,
    /// Verify rows per item: zero without speculation, else at least two.
    pub verify_rows: u32,
}

impl StateBuffer {
    /// Bytes of one entry in one layer.
    pub fn bytes_per_layer_rank(&self) -> u64 {
        self.planes.iter().sum()
    }

    /// Bytes of one entry over every layer: what a snapshot copies.
    pub fn entry_bytes(&self) -> u64 {
        self.layers * self.bytes_per_layer_rank()
    }

    /// Entries of each plane: the buffer's own, then any pooled
    /// checkpoints, then any pooled anchor spares.
    pub fn entries(&self, extent: Extent) -> u64 {
        let checkpoints = match self.checkpoints() {
            Some(checkpoints) if checkpoints.pooled => self.checkpoint_entries(extent),
            _ => 0,
        };
        self.own_entries(extent) + checkpoints + self.spares(true)
    }

    /// The buffer's own entries, one per slot.
    pub fn own_entries(&self, extent: Extent) -> u64 {
        u64::from(match self.entries {
            Entries::Served => extent.served,
            Entries::Bucket => extent.bucket,
            Entries::Fixed(count) => count,
        })
    }

    fn checkpoints(&self) -> Option<Checkpoints> {
        match self.rollback {
            Rollback::Checkpoint(checkpoints) => Some(checkpoints),
            _ => None,
        }
    }

    /// Checkpointing items and entries per item in an instance of `extent`;
    /// none when it does not verify.
    fn checkpoint_shape(&self, extent: Extent) -> Option<(u64, u64)> {
        let checkpoints = self.checkpoints().filter(|_| extent.verify_rows > 1)?;
        let items = match checkpoints.items {
            CheckpointItems::Drafting(items) => self.own_entries(extent).min(u64::from(items)),
            CheckpointItems::Bucket => u64::from(extent.bucket),
        };
        let rows = match checkpoints.rows {
            CheckpointRows::Verify => extent.verify_rows,
            CheckpointRows::Fixed(rows) => rows,
        };
        Some((items, u64::from(rows)))
    }

    /// Checkpoint entries of one step, pooled or separate.
    pub fn checkpoint_entries(&self, extent: Extent) -> u64 {
        self.checkpoint_shape(extent)
            .map_or(0, |(items, rows)| items * rows)
    }

    /// The entry that holds the state after row `row` of checkpointing item
    /// `item`: in this buffer when the checkpoints are pooled, otherwise in
    /// the separate checkpoint buffer.
    pub fn checkpoint(&self, extent: Extent, item: u32, row: u32) -> Option<u64> {
        let (items, rows) = self.checkpoint_shape(extent)?;
        let base = if self.checkpoints()?.pooled {
            self.own_entries(extent)
        } else {
            0
        };
        (u64::from(item) < items && u64::from(row) < rows)
            .then(|| base + u64::from(item) * rows + u64::from(row))
    }

    /// Bytes of the separate checkpoint buffer, if the checkpoints have one.
    pub fn checkpoint_bytes(&self, extent: Extent) -> u64 {
        match self.checkpoints() {
            Some(checkpoints) if !checkpoints.pooled => {
                self.layers * self.layer_stride(self.checkpoint_entries(extent))
            }
            _ => 0,
        }
    }

    /// Anchor spare entries, kept in this buffer (`pooled`) or in a
    /// separate one.
    fn spares(&self, pooled: bool) -> u64 {
        match self.anchors {
            Anchors::Spare {
                per_step,
                pooled: own,
            } if own == pooled => u64::from(per_step),
            _ => 0,
        }
    }

    /// Bytes between one layer and the next in a buffer of `entries`
    /// entries.
    pub fn layer_stride(&self, entries: u64) -> u64 {
        entries * self.bytes_per_layer_rank()
    }

    /// Bytes of the working planes on one rank.
    pub fn working_bytes(&self, extent: Extent) -> u64 {
        self.layers * self.layer_stride(self.entries(extent))
    }

    /// Bytes of the buffer on one rank, an adjacent shadow included.
    pub fn bytes(&self, extent: Extent) -> u64 {
        let working = self.working_bytes(extent);
        if self.shadow == Shadow::Adjacent {
            2 * working
        } else {
            working
        }
    }

    /// Bytes of the separate shadow buffer, if the shadow has one.
    pub fn shadow_bytes(&self, extent: Extent) -> u64 {
        if self.shadow == Shadow::Separate {
            self.working_bytes(extent)
        } else {
            0
        }
    }

    /// Whether every size of the buffer in an instance of `extent` fits a
    /// `u64`: its working planes and shadow, and its separate checkpoint
    /// and spare buffers.
    fn checked_sizes(&self, extent: Extent) -> Option<()> {
        let per_layer = self
            .planes
            .iter()
            .try_fold(0_u64, |total, &plane| total.checked_add(plane))?;
        let bytes = |entries: u64| entries.checked_mul(per_layer)?.checked_mul(self.layers);
        let checkpoints = self
            .checkpoint_shape(extent)
            .map_or(Some(0), |(items, rows)| items.checked_mul(rows))?;
        bytes(self.entries(extent))?.checked_mul(2)?;
        bytes(checkpoints)?;
        bytes(self.spares(false))?;
        Some(())
    }

    /// Bytes of the separate spare buffer, if the anchor spares have one.
    pub fn spare_bytes(&self) -> u64 {
        self.layers * self.layer_stride(self.spares(false))
    }

    /// The working bytes `region` covers, as `(offset, length)`; `None` for
    /// the touched entries, which the step lists.
    pub fn region(&self, extent: Extent, region: Region) -> Option<(u64, u64)> {
        let layer = self.layer_stride(self.entries(extent));
        match region {
            Region::Touched => None,
            Region::All => Some((0, self.layers * layer)),
            Region::Layers { first, count } => Some((first * layer, count * layer)),
        }
    }

    /// Where the shadow copy of working byte `offset` lies: whether in the
    /// separate shadow buffer, and its offset there.
    pub fn shadow_offset(&self, extent: Extent, offset: u64) -> Option<(bool, u64)> {
        match self.shadow {
            Shadow::None => None,
            Shadow::Adjacent => Some((false, self.working_bytes(extent) + offset)),
            Shadow::Separate => Some((true, offset)),
        }
    }

    /// The pieces of entry `entry` in a buffer of `entries` entries, in
    /// snapshot order: `(offset, length)` by layer, then plane.
    fn pieces(&self, entry: u64, entries: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        let stride = self.layer_stride(entries);
        (0..self.layers).flat_map(move |layer| {
            let mut plane_start = layer * stride;
            self.planes.iter().map(move |&plane| {
                let piece = (plane_start + entry * plane, plane);
                plane_start += entries * plane;
                piece
            })
        })
    }
}

/// One stream of a paged layer: an entry of `entry_bytes` per
/// `tokens_per_entry` tokens. A stream written at a fraction of the token
/// rate (block keys, compressed KV) has more than one token per entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvStream {
    pub name: &'static str,
    pub tokens_per_entry: u64,
    pub entry_bytes: u64,
}

impl KvStream {
    /// Bytes of the stream in one page of one layer.
    pub fn page_bytes(&self, page_tokens: u64) -> u64 {
        page_tokens / self.tokens_per_entry * self.entry_bytes
    }
}

/// `layers` paged layers of the same streams. A serving page holds one page
/// of tokens of each group's layers in declaration order, layer after
/// layer, and each layer's streams in order (`crate::pages`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PagedLayers {
    pub name: &'static str,
    pub layers: u64,
    pub streams: Vec<KvStream>,
}

/// Where one stream of one paged layer lies in a page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagePlane {
    pub layers: &'static str,
    pub layer: u64,
    pub stream: &'static str,
    pub offset: u64,
    pub bytes: u64,
}

/// Whose state a snapshot copies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Holder {
    /// A slot's committed entry.
    Slot(u32),
    /// An anchor item's spares, which the last step kept at its anchor.
    Anchor(u32),
}

/// One contiguous piece of a snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Piece {
    /// The buffer's index in the declaration.
    pub buffer: usize,
    /// Whether the piece is in the buffer's separate spare buffer.
    pub spare: bool,
    pub offset: u64,
    pub length: u64,
}

/// A declaration that breaks a rule: the stream or buffer, and the rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Invalid {
    pub name: &'static str,
    pub rule: &'static str,
}

/// Everything one instance configuration keeps per request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestState {
    pub page_tokens: u64,
    pub pages: Vec<PagedLayers>,
    pub buffers: Vec<StateBuffer>,
    /// Bytes the host writes after the pieces of a snapshot. The driver
    /// copies it whole, so any length works.
    pub header_bytes: u64,
}

/// Snapshot copies move whole 16-byte vectors.
const VECTOR_BYTES: u64 = 16;

impl RequestState {
    /// Where every stream of every paged layer lies in a page.
    pub fn page_planes(&self) -> Vec<PagePlane> {
        let mut offset = 0;
        let mut planes = Vec::new();
        for group in &self.pages {
            for layer in 0..group.layers {
                for stream in &group.streams {
                    let bytes = stream.page_bytes(self.page_tokens);
                    planes.push(PagePlane {
                        layers: group.name,
                        layer,
                        stream: stream.name,
                        offset,
                        bytes,
                    });
                    offset += bytes;
                }
            }
        }
        planes
    }

    /// Bytes of one serving page.
    pub fn page_bytes(&self) -> u64 {
        self.page_planes().iter().map(|plane| plane.bytes).sum()
    }

    fn captured(&self) -> impl Iterator<Item = (usize, &StateBuffer)> {
        self.buffers
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.capture != Capture::None)
    }

    /// Bytes of one snapshot, its header included.
    pub fn snapshot_bytes(&self) -> u64 {
        self.captured()
            .map(|(_, buffer)| buffer.entry_bytes())
            .sum::<u64>()
            + self.header_bytes
    }

    /// Serving pages one snapshot takes.
    pub fn snapshot_pages(&self) -> u64 {
        self.snapshot_bytes().div_ceil(self.page_bytes())
    }

    /// The pieces `holder`'s snapshot copies, in snapshot order, in an
    /// instance of `extent`.
    pub fn pieces(&self, extent: Extent, holder: Holder) -> Result<Vec<Piece>, Invalid> {
        self.pieces_with_entries(extent, holder, &[])
    }

    /// Resolve a logical slot through physical entry overrides keyed by buffer
    /// index. Anchors keep their declared spare addresses.
    pub fn pieces_with_entries(
        &self,
        extent: Extent,
        holder: Holder,
        mapped: &[(usize, u64)],
    ) -> Result<Vec<Piece>, Invalid> {
        let mut seen = BTreeSet::new();
        for &(index, entry) in mapped {
            let buffer = self
                .buffers
                .get(index)
                .ok_or(invalid("", "unknown mapped buffer"))?;
            if !matches!(holder, Holder::Slot(_))
                || !seen.insert(index)
                || buffer.capture == Capture::None
                || entry >= buffer.entries(extent) - buffer.spares(true)
            {
                return Err(invalid(buffer.name, "invalid mapped entry"));
            }
        }
        let mut pieces = Vec::new();
        for (index, buffer) in self.captured() {
            let (spare, entry, entries) = match holder {
                Holder::Slot(slot) => {
                    if u64::from(slot) >= buffer.own_entries(extent) {
                        return Err(invalid(buffer.name, "slot outside the buffer"));
                    }
                    let entry = mapped
                        .iter()
                        .find(|&&(buffer, _)| buffer == index)
                        .map_or(u64::from(slot), |&(_, entry)| entry);
                    (false, entry, buffer.entries(extent))
                }
                Holder::Anchor(anchor) => match buffer.anchors {
                    Anchors::Spare { per_step, pooled } if anchor < per_step => {
                        if pooled {
                            let entries = buffer.entries(extent);
                            (false, entries - u64::from(per_step - anchor), entries)
                        } else {
                            (true, u64::from(anchor), u64::from(per_step))
                        }
                    }
                    _ => return Err(invalid(buffer.name, "anchor without a spare")),
                },
            };
            pieces.extend(buffer.pieces(entry, entries).map(|(offset, length)| Piece {
                buffer: index,
                spare,
                offset,
                length,
            }));
        }
        Ok(pieces)
    }

    /// Checks the declaration's rules, so the shared mechanics need no
    /// special case for any model.
    pub fn validate(&self) -> Result<(), Invalid> {
        if self.page_tokens == 0 || self.pages.is_empty() {
            return Err(invalid("", "pages hold no tokens or no layer"));
        }
        let mut names = BTreeSet::new();
        for group in &self.pages {
            if group.name.is_empty() || !names.insert(group.name) {
                return Err(invalid(group.name, "names must be unique and not empty"));
            }
            let mut streams = BTreeSet::new();
            if group.layers == 0 || group.streams.is_empty() {
                return Err(invalid(group.name, "paged layers without a stream"));
            }
            for stream in &group.streams {
                if stream.name.is_empty() || !streams.insert(stream.name) {
                    return Err(invalid(stream.name, "names must be unique and not empty"));
                }
                if stream.entry_bytes == 0 || stream.tokens_per_entry == 0 {
                    return Err(invalid(stream.name, "an empty stream"));
                }
                if !self.page_tokens.is_multiple_of(stream.tokens_per_entry) {
                    return Err(invalid(
                        stream.name,
                        "page_tokens is not a multiple of tokens_per_entry",
                    ));
                }
            }
        }
        let page_bytes = self
            .pages
            .iter()
            .try_fold(0_u64, |total, group| {
                group.streams.iter().try_fold(total, |total, stream| {
                    (self.page_tokens / stream.tokens_per_entry)
                        .checked_mul(stream.entry_bytes)?
                        .checked_mul(group.layers)?
                        .checked_add(total)
                })
            })
            .ok_or(invalid("", "page sizes overflow"))?;
        // Snapshot pieces continue across pages.
        if !page_bytes.is_multiple_of(VECTOR_BYTES) {
            return Err(invalid("", "pages are not whole 16-byte vectors"));
        }
        let mut snapshot = Some(self.header_bytes);
        for buffer in &self.buffers {
            if buffer.name.is_empty() || !names.insert(buffer.name) {
                return Err(invalid(buffer.name, "names must be unique and not empty"));
            }
            check(buffer)?;
            if buffer.capture != Capture::None {
                snapshot = snapshot.and_then(|bytes| bytes.checked_add(buffer.entry_bytes()));
            }
        }
        if snapshot.is_none() {
            return Err(invalid("", "snapshot sizes overflow"));
        }
        // An anchor capture copies every captured piece from spares.
        let mut anchors = self.captured().map(|(_, buffer)| match buffer.anchors {
            Anchors::None => None,
            Anchors::Spare { per_step, .. } => Some(per_step),
        });
        if let Some(first) = anchors.next()
            && anchors.any(|per_step| per_step != first)
        {
            return Err(invalid(
                "",
                "every captured buffer must keep the same anchor spares",
            ));
        }
        Ok(())
    }

    /// Checks that an instance of `extent` can hold this state: every
    /// served slot has an entry, and the rows of one verify item occupy
    /// distinct positions and checkpoints.
    pub fn validate_extent(&self, extent: Extent) -> Result<(), Invalid> {
        if extent.served == 0 || extent.served > extent.bucket || extent.verify_rows == 1 {
            return Err(invalid(
                "",
                "an extent serves 1..=bucket slots, 0 or 2+ rows",
            ));
        }
        for buffer in &self.buffers {
            let fail = |rule| Err(invalid(buffer.name, rule));
            if buffer.checked_sizes(extent).is_none() {
                return fail("the buffer's sizes overflow");
            }
            if buffer.own_entries(extent) < u64::from(extent.served) {
                return fail("fewer entries than served slots");
            }
            if matches!(buffer.rollback, Rollback::Truncate | Rollback::RestoreTail)
                && buffer
                    .scope
                    .positions()
                    .is_some_and(|positions| u64::from(extent.verify_rows) > positions)
            {
                return fail("a verify item's rows must fit the positions");
            }
            if let Some(Checkpoints {
                rows: CheckpointRows::Fixed(rows),
                ..
            }) = buffer.checkpoints()
                && extent.verify_rows > rows
            {
                return fail("a verify item's rows must fit its checkpoints");
            }
        }
        Ok(())
    }
}

fn invalid(name: &'static str, rule: &'static str) -> Invalid {
    Invalid { name, rule }
}

/// The rules of one buffer.
fn check(buffer: &StateBuffer) -> Result<(), Invalid> {
    let fail = |rule| Err(invalid(buffer.name, rule));
    // The sizes below and every entry size derived later must fit a `u64`.
    if buffer
        .planes
        .iter()
        .try_fold(0_u64, |total, &plane| total.checked_add(plane))
        .and_then(|per_layer| per_layer.checked_mul(buffer.layers))
        .is_none()
    {
        return fail("the buffer's sizes overflow");
    }
    if buffer.layers == 0
        || buffer.planes.is_empty()
        || buffer.entries == Entries::Fixed(0)
        || buffer
            .planes
            .iter()
            .any(|&plane| plane == 0 || !plane.is_multiple_of(VECTOR_BYTES))
    {
        return fail("planes must be whole 16-byte vectors in at least one layer");
    }
    if buffer.scope.positions().is_some_and(|count| {
        count == 0
            || buffer
                .planes
                .iter()
                .any(|plane| !plane.is_multiple_of(count))
    }) {
        return fail("a positional plane must hold whole positions");
    }
    let captured = match (buffer.capture, buffer.scope) {
        (Capture::None, _) => false,
        (Capture::Full, Scope::Slot) => true,
        (Capture::RingWindow { tokens }, Scope::WindowRing { tokens: ring })
        | (Capture::LastRows { rows: tokens }, Scope::Rows { rows: ring })
            if tokens > 0 && tokens <= ring =>
        {
            true
        }
        _ => return fail("the capture does not fit the scope"),
    };
    let saves = match &buffer.cancel {
        Cancel::Restore(saves) => saves.as_slice(),
        Cancel::None | Cancel::Invalidate => &[],
    };
    if (buffer.shadow == Shadow::None) != saves.is_empty() {
        return fail("a shadow holds exactly what cancel saves");
    }
    for save in saves {
        let fits = match save.region {
            Region::Touched => buffer.scope.positions().is_some(),
            Region::All => true,
            Region::Layers { first, count } => {
                count > 0 && first < buffer.layers && count <= buffer.layers - first
            }
        };
        if !fits {
            return fail("a cancel save must be touched positions or whole layers");
        }
    }
    match buffer.rollback {
        Rollback::None => {}
        Rollback::Truncate | Rollback::RestoreTail if buffer.scope.positions().is_none() => {
            return fail("only a positional buffer can roll back by position");
        }
        Rollback::RestoreTail
            if !saves.contains(&Save {
                writer: Writer::Any,
                region: Region::Touched,
            }) =>
        {
            return fail("a tail restore needs every step's touched entries saved");
        }
        Rollback::Checkpoint(Checkpoints { items, rows, .. })
            if buffer.scope != Scope::Slot
                || items == CheckpointItems::Drafting(0)
                || rows == CheckpointRows::Fixed(0) =>
        {
            return fail("checkpoints hold a slot's state for at least one item");
        }
        Rollback::Truncate | Rollback::RestoreTail | Rollback::Checkpoint(_) => {}
    }
    if buffer.owner == Owner::FrontierOwner && !(buffer.drafter && captured) {
        return fail("only captured drafter state has a frontier owner");
    }
    match buffer.anchors {
        Anchors::None => Ok(()),
        Anchors::Spare { per_step, .. } if per_step > 0 && captured => Ok(()),
        Anchors::Spare { .. } => fail("anchor spares need captured state"),
    }
}

/// A small declaration of every buffer kind, for the shared mechanics
/// tests: 256-byte pages, a three-layer window ring, a slot buffer with
/// pooled verify checkpoints, a drafter history with a frontier owner, and
/// a buffer no snapshot holds.
#[cfg(test)]
pub(crate) fn synthetic() -> RequestState {
    let stream = |name| KvStream {
        name,
        tokens_per_entry: 1,
        entry_bytes: 16,
    };
    let buffer = |name, layers, planes| StateBuffer {
        name,
        layers,
        planes,
        entries: Entries::Served,
        scope: Scope::Slot,
        shadow: Shadow::None,
        capture: Capture::Full,
        rollback: Rollback::None,
        cancel: Cancel::None,
        owner: Owner::Shared,
        drafter: false,
        anchors: Anchors::None,
    };
    RequestState {
        page_tokens: 4,
        pages: vec![PagedLayers {
            name: "kv",
            layers: 2,
            streams: vec![stream("k"), stream("v")],
        }],
        buffers: vec![
            StateBuffer {
                entries: Entries::Bucket,
                scope: Scope::WindowRing { tokens: 8 },
                shadow: Shadow::Adjacent,
                capture: Capture::RingWindow { tokens: 8 },
                rollback: Rollback::RestoreTail,
                cancel: Cancel::Restore(vec![Save {
                    writer: Writer::Any,
                    region: Region::Touched,
                }]),
                ..buffer("window", 3, vec![8 * 32, 8 * 16])
            },
            StateBuffer {
                rollback: Rollback::Checkpoint(Checkpoints {
                    items: CheckpointItems::Drafting(2),
                    rows: CheckpointRows::Verify,
                    pooled: true,
                }),
                cancel: Cancel::Invalidate,
                ..buffer("slot", 2, vec![48])
            },
            StateBuffer {
                scope: Scope::Rows { rows: 6 },
                capture: Capture::LastRows { rows: 4 },
                rollback: Rollback::Truncate,
                owner: Owner::FrontierOwner,
                drafter: true,
                ..buffer("history", 1, vec![6 * 32])
            },
            StateBuffer {
                capture: Capture::None,
                ..buffer("scratch", 1, vec![16])
            },
        ],
        header_bytes: 64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every step saves the entries it touches; commit restores a rejected
    /// tail from them, and cancel all of them.
    fn touched() -> Cancel {
        Cancel::Restore(vec![Save {
            writer: Writer::Any,
            region: Region::Touched,
        }])
    }

    /// A sliding-window ring per layer: K, then V.
    fn ring(name: &'static str, layers: u64, tokens: u64, drafter: bool) -> StateBuffer {
        StateBuffer {
            name,
            layers,
            planes: vec![tokens * 576, tokens * 512],
            entries: Entries::Bucket,
            scope: Scope::WindowRing { tokens },
            shadow: Shadow::Adjacent,
            capture: Capture::RingWindow { tokens },
            rollback: Rollback::RestoreTail,
            cancel: touched(),
            owner: Owner::Shared,
            drafter,
            anchors: Anchors::None,
        }
    }

    /// The compressor tail's verify checkpoints: one per row of each
    /// drafting item, after the slots.
    const TAIL_CHECKPOINTS: Checkpoints = Checkpoints {
        items: CheckpointItems::Drafting(32),
        rows: CheckpointRows::Verify,
        pooled: true,
    };

    /// A compressed-attention architecture that neither served model has,
    /// declared without a special case. Pages hold the compressed KV and
    /// index keys of three ratio-2 sources at one entry per two tokens and
    /// of one ratio-1 source at one per token: 288-byte FP4 KV entries (512
    /// values and an E4M3 scale per 16) and 68-byte MXFP4 index keys. Each
    /// slot keeps a 128-token FP8 window ring (512 values and 16 UE8M0 scale
    /// bytes per token) in each of 40 layers and the drafter's 3, and an F32
    /// compressor tail of one token (kv and score) per ratio-2 source.
    fn compressed() -> RequestState {
        let window = |name, layers, drafter| StateBuffer {
            planes: vec![128 * (512 + 16)],
            ..ring(name, layers, 128, drafter)
        };
        let csa = |name, layers, tokens_per_entry| PagedLayers {
            name,
            layers,
            streams: vec![
                KvStream {
                    name: "kv",
                    tokens_per_entry,
                    entry_bytes: 512 / 2 + 512 / 16,
                },
                KvStream {
                    name: "index.keys",
                    tokens_per_entry,
                    entry_bytes: 128 / 2 + 128 / 32,
                },
            ],
        };
        RequestState {
            page_tokens: 64,
            pages: vec![csa("csa.ratio2", 3, 2), csa("csa.ratio1", 1, 1)],
            buffers: vec![
                window("swa.ring", 40, false),
                window("drafter.ring", 3, true),
                StateBuffer {
                    name: "compress.tail",
                    layers: 3,
                    planes: vec![2 * 512 * 4],
                    entries: Entries::Served,
                    scope: Scope::Slot,
                    shadow: Shadow::None,
                    capture: Capture::Full,
                    rollback: Rollback::Checkpoint(TAIL_CHECKPOINTS),
                    cancel: Cancel::Invalidate,
                    owner: Owner::Shared,
                    drafter: false,
                    anchors: Anchors::None,
                },
            ],
            header_bytes: 64,
        }
    }

    const EXTENT: Extent = Extent {
        served: 64,
        bucket: 64,
        verify_rows: 6,
    };

    #[test]
    fn a_compressed_attention_declaration_validates_without_a_special_case() {
        let state = compressed();
        assert_eq!(state.validate(), Ok(()));
        assert_eq!(state.validate_extent(EXTENT), Ok(()));
        // 890 bytes of global KV per token.
        assert_eq!(state.page_bytes(), 890 * 64);
        assert_eq!(state.page_bytes(), 56_960);
        let planes = state
            .page_planes()
            .iter()
            .map(|plane| (plane.layers, plane.layer, plane.stream, plane.offset))
            .collect::<Vec<_>>();
        let ratio2 = 32 * (288 + 68);
        assert_eq!(
            planes,
            [
                ("csa.ratio2", 0, "kv", 0),
                ("csa.ratio2", 0, "index.keys", 32 * 288),
                ("csa.ratio2", 1, "kv", ratio2),
                ("csa.ratio2", 1, "index.keys", ratio2 + 32 * 288),
                ("csa.ratio2", 2, "kv", 2 * ratio2),
                ("csa.ratio2", 2, "index.keys", 2 * ratio2 + 32 * 288),
                ("csa.ratio1", 0, "kv", 3 * ratio2),
                ("csa.ratio1", 0, "index.keys", 3 * ratio2 + 64 * 288),
            ]
        );
        // 2.77 MiB of rings and 12 KiB of tails per anchor, as the fit
        // estimates.
        assert_eq!(
            state.snapshot_bytes(),
            43 * 128 * 528 + 3 * 4096 + state.header_bytes
        );
        let pieces = state.pieces(EXTENT, Holder::Slot(5)).unwrap();
        assert_eq!(pieces.len(), 40 + 3 + 3);
        assert_eq!(pieces[1].offset, 64 * 128 * 528 + 5 * 128 * 528);
        // Each tail layer holds the slots, then 32 items' six checkpoints.
        let tail = &state.buffers[2];
        assert_eq!(tail.entries(EXTENT), 64 + 32 * 6);
        assert_eq!(
            (pieces[45].buffer, pieces[45].offset),
            (2, (2 * (64 + 192) + 5) * 4096)
        );
        assert_eq!(tail.checkpoint(EXTENT, 3, 5), Some(64 + 3 * 6 + 5));
        assert_eq!(tail.checkpoint(EXTENT, 32, 0), None);
        assert_eq!(state.buffers[0].bytes(EXTENT), 2 * 40 * 64 * 128 * 528);
    }

    /// A ratio-2 compressor in the reference's order: a row at an even
    /// position becomes the tail; a row at an odd position pools with the
    /// tail into compressed entry `position / 2` and empties it.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    struct Compressor {
        tail: Option<u32>,
    }

    impl Compressor {
        fn row(&mut self, position: u64, value: u32, entries: &mut Vec<(u64, u32)>) {
            if position.is_multiple_of(2) {
                self.tail = Some(value);
            } else {
                let tail = self.tail.take().expect("an odd row follows its pair");
                entries.retain(|&(entry, _)| entry != position / 2);
                entries.push((position / 2, tail.wrapping_mul(31) ^ value));
            }
        }
    }

    /// A verify step on the compressor tail, run by the declared rollback:
    /// from every committed length (both parities of the tail) and with
    /// every row count and kept length, the committed tail and compressed
    /// entries equal a serial compressor's over the kept tokens. The state
    /// from before the step, which a whole-entry shadow would restore, does
    /// not.
    #[test]
    fn checkpoints_return_the_compressor_tail_to_the_accepted_prefix() {
        let state = compressed();
        let tail = &state.buffers[2];
        let token = |position: u64| (position as u32).wrapping_mul(2_654_435_761);
        let serial = |end: u64| {
            let (mut compressor, mut entries) = (Compressor::default(), Vec::new());
            for position in 0..end {
                compressor.row(position, token(position), &mut entries);
            }
            (compressor, entries)
        };
        let mut before_step_differs = false;
        for committed in 0..6 {
            for rows in 2..=EXTENT.verify_rows {
                for kept in 1..=rows {
                    let (slot, mut entries) = serial(committed);
                    // The step: each row's state goes to its checkpoint,
                    // the slot entry is not written.
                    let mut checkpoints = vec![Compressor::default(); rows as usize];
                    let mut working = slot;
                    for row in 0..rows {
                        let position = committed + u64::from(row);
                        working.row(position, token(position), &mut entries);
                        let entry = tail.checkpoint(EXTENT, 0, row).unwrap();
                        assert_eq!(entry, 64 + u64::from(row));
                        checkpoints[row as usize] = working;
                    }
                    let committed_tail = match tail.rollback.commit(rows, kept) {
                        Commit::Checkpoint { row } => checkpoints[row as usize],
                        other => panic!("{other:?}"),
                    };
                    let end = committed + u64::from(kept);
                    // Entries past the committed end are rewritten later.
                    entries.retain(|&(entry, _)| 2 * entry + 1 < end);
                    entries.sort_unstable();
                    let (expected, expected_entries) = serial(end);
                    assert_eq!(committed_tail, expected, "{committed} {rows} {kept}");
                    assert_eq!(entries, expected_entries, "{committed} {rows} {kept}");
                    before_step_differs |= slot != expected;
                }
            }
        }
        assert!(before_step_differs);
    }

    #[test]
    fn page_tokens_must_hold_whole_entries_and_pages_whole_vectors() {
        let mut state = compressed();
        state.page_tokens = 63;
        assert_eq!(
            state.validate().unwrap_err().rule,
            "page_tokens is not a multiple of tokens_per_entry"
        );
        state.page_tokens = 64;
        state.pages[0].streams[1].tokens_per_entry = 128;
        assert_eq!(state.validate().unwrap_err().name, "index.keys");
        // Two-token pages of these streams take 1,780 bytes.
        let mut odd = compressed();
        odd.page_tokens = 2;
        assert_eq!(
            odd.validate().unwrap_err().rule,
            "pages are not whole 16-byte vectors"
        );
        // The host writes the header, so it may be short.
        let mut short = compressed();
        short.header_bytes = 8;
        assert_eq!(short.validate(), Ok(()));
    }

    #[test]
    fn extents_must_fit_slots_rows_and_checkpoints() {
        let state = compressed();
        let extent = |served, bucket, verify_rows| Extent {
            served,
            bucket,
            verify_rows,
        };
        assert_eq!(state.validate_extent(extent(100, 128, 8)), Ok(()));
        assert!(state.validate_extent(extent(0, 1, 0)).is_err());
        assert!(state.validate_extent(extent(129, 128, 0)).is_err());
        assert!(state.validate_extent(extent(1, 1, 1)).is_err());
        // A verify item longer than the ring would revisit its own rows.
        assert_eq!(
            state.validate_extent(extent(4, 4, 129)),
            Err(invalid(
                "swa.ring",
                "a verify item's rows must fit the positions"
            ))
        );
        let mut fixed = compressed();
        fixed.buffers[0].entries = Entries::Fixed(32);
        fixed.buffers[2].rollback = Rollback::Checkpoint(Checkpoints {
            rows: CheckpointRows::Fixed(4),
            ..TAIL_CHECKPOINTS
        });
        assert_eq!(
            fixed.validate_extent(extent(33, 64, 0)).unwrap_err().rule,
            "fewer entries than served slots"
        );
        assert_eq!(
            fixed.validate_extent(extent(32, 64, 8)).unwrap_err().rule,
            "a verify item's rows must fit its checkpoints"
        );
    }

    #[test]
    fn anchors_need_captured_state_and_the_same_spares_everywhere() {
        let spare = Anchors::Spare {
            per_step: 2,
            pooled: false,
        };
        let mut state = compressed();
        state.buffers[0].anchors = spare;
        assert_eq!(
            state.validate().unwrap_err().rule,
            "every captured buffer must keep the same anchor spares"
        );
        for buffer in &mut state.buffers {
            buffer.anchors = spare;
        }
        assert_eq!(state.validate(), Ok(()));
        state.buffers[2].capture = Capture::None;
        assert_eq!(
            state.validate().unwrap_err(),
            invalid("compress.tail", "anchor spares need captured state")
        );
        state.buffers[2].anchors = Anchors::None;
        let extent = Extent {
            served: 8,
            bucket: 8,
            verify_rows: 0,
        };
        let pieces = state.pieces(extent, Holder::Anchor(1)).unwrap();
        assert!(pieces.iter().all(|piece| piece.spare));
        assert_eq!(pieces[1].offset, 2 * 128 * 528 + 128 * 528);
        assert!(state.pieces(extent, Holder::Anchor(2)).is_err());
        assert!(state.pieces(extent, Holder::Slot(8)).is_err());
    }

    #[test]
    fn checkpoints_and_spares_follow_the_slots_or_take_their_own_buffer() {
        let linear = StateBuffer {
            name: "linear",
            layers: 2,
            planes: vec![64],
            entries: Entries::Served,
            scope: Scope::Slot,
            shadow: Shadow::None,
            capture: Capture::Full,
            rollback: Rollback::Checkpoint(Checkpoints {
                items: CheckpointItems::Drafting(32),
                rows: CheckpointRows::Verify,
                pooled: true,
            }),
            cancel: Cancel::Invalidate,
            owner: Owner::Shared,
            drafter: false,
            anchors: Anchors::Spare {
                per_step: 2,
                pooled: true,
            },
        };
        let entry = StateBuffer {
            name: "entry",
            layers: 1,
            rollback: Rollback::Checkpoint(Checkpoints {
                items: CheckpointItems::Bucket,
                rows: CheckpointRows::Fixed(8),
                pooled: false,
            }),
            cancel: Cancel::None,
            anchors: Anchors::Spare {
                per_step: 2,
                pooled: false,
            },
            ..linear.clone()
        };
        let state = RequestState {
            page_tokens: 64,
            pages: vec![PagedLayers {
                name: "kv",
                layers: 1,
                streams: vec![KvStream {
                    name: "k",
                    tokens_per_entry: 1,
                    entry_bytes: 16,
                }],
            }],
            buffers: vec![linear, entry],
            header_bytes: 0,
        };
        assert_eq!(state.validate(), Ok(()));
        let extent = |served, bucket, verify_rows| Extent {
            served,
            bucket,
            verify_rows,
        };
        let [linear, entry] = [&state.buffers[0], &state.buffers[1]];
        // Slots, then 32 drafting items' checkpoints, then two spares.
        assert_eq!(linear.entries(extent(100, 128, 8)), 100 + 256 + 2);
        assert_eq!(linear.entries(extent(4, 4, 8)), 4 + 32 + 2);
        assert_eq!(linear.entries(extent(4, 4, 0)), 4 + 2);
        assert_eq!(linear.checkpoint(extent(4, 4, 4), 3, 2), Some(4 + 14));
        let anchor = state.pieces(extent(4, 4, 8), Holder::Anchor(1)).unwrap();
        assert_eq!(anchor[1].offset, 38 * 64 + 37 * 64);
        assert!(!anchor[1].spare);
        // A separate pool of every bucket item's eight rows.
        assert_eq!(entry.entries(extent(100, 128, 4)), 100);
        assert_eq!(entry.checkpoint_bytes(extent(100, 128, 4)), 128 * 8 * 64);
        assert_eq!(entry.checkpoint_bytes(extent(100, 128, 0)), 0);
        assert_eq!(
            entry.checkpoint(extent(100, 128, 4), 127, 7),
            Some(127 * 8 + 7)
        );
        assert_eq!(entry.checkpoint(extent(100, 128, 4), 128, 0), None);
        assert_eq!(entry.spare_bytes(), 2 * 64);
    }

    #[test]
    fn cancel_saves_name_their_region_and_shadow() {
        let mut state = compressed();
        let ring = &mut state.buffers[1];
        ring.shadow = Shadow::Separate;
        ring.rollback = Rollback::None;
        ring.cancel = Cancel::Restore(vec![
            Save {
                writer: Writer::Prefill,
                region: Region::All,
            },
            Save {
                writer: Writer::Drafter,
                region: Region::Layers { first: 2, count: 1 },
            },
        ]);
        assert_eq!(state.validate(), Ok(()));
        let ring = &state.buffers[1];
        let layer = 64 * 128 * 528;
        assert_eq!(
            ring.region(EXTENT, Region::Layers { first: 2, count: 1 }),
            Some((2 * layer, layer))
        );
        assert_eq!(ring.region(EXTENT, Region::Touched), None);
        assert_eq!(ring.shadow_offset(EXTENT, 7), Some((true, 7)));
        assert_eq!(ring.shadow_bytes(EXTENT), 3 * layer);
        assert_eq!(
            state.buffers[0].shadow_offset(EXTENT, 7),
            Some((false, 40 * layer + 7))
        );
        assert_eq!(
            Rollback::RestoreTail.commit(6, 2),
            Commit::RestoreTail { kept: 2, rows: 6 }
        );
        assert_eq!(Rollback::RestoreTail.commit(6, 6), Commit::Working);
    }

    /// Sizes that overflow a `u64` are validation errors, not panics.
    #[test]
    fn overflowing_declarations_are_invalid() {
        let mut range = compressed();
        range.buffers[1].cancel = Cancel::Restore(vec![
            Save {
                writer: Writer::Any,
                region: Region::Touched,
            },
            Save {
                writer: Writer::Drafter,
                region: Region::Layers {
                    first: u64::MAX,
                    count: 2,
                },
            },
        ]);
        assert_eq!(
            range.validate(),
            Err(invalid(
                "drafter.ring",
                "a cancel save must be touched positions or whole layers"
            ))
        );
        range.buffers[1].layers = 1;
        assert!(range.validate().is_err());
        let mut page = compressed();
        page.pages[0].streams[0].entry_bytes = u64::MAX / 2;
        assert_eq!(page.validate().unwrap_err().rule, "page sizes overflow");
        let mut planes = compressed();
        planes.buffers[2].planes = vec![u64::MAX - 15, 16];
        assert_eq!(
            planes.validate(),
            Err(invalid("compress.tail", "the buffer's sizes overflow"))
        );
        planes.buffers[2].planes = vec![1 << 63];
        assert_eq!(
            planes.validate().unwrap_err().rule,
            "the buffer's sizes overflow"
        );
        let mut snapshot = compressed();
        snapshot.header_bytes = u64::MAX;
        assert_eq!(
            snapshot.validate().unwrap_err().rule,
            "snapshot sizes overflow"
        );
        // Fits per entry, overflows over an instance's entries.
        let mut extent = compressed();
        extent.buffers[2].planes = vec![1 << 60];
        assert_eq!(extent.validate(), Ok(()));
        assert_eq!(
            extent.validate_extent(EXTENT),
            Err(invalid("compress.tail", "the buffer's sizes overflow"))
        );
    }

    #[test]
    fn each_rule_names_its_buffer() {
        type Change = fn(&mut StateBuffer);
        let rules: [(Change, &str); 10] = [
            (
                |buffer| buffer.planes = vec![24],
                "planes must be whole 16-byte vectors in at least one layer",
            ),
            (
                |buffer| buffer.scope = Scope::WindowRing { tokens: 96 },
                "the capture does not fit the scope",
            ),
            (
                |buffer| buffer.capture = Capture::Full,
                "the capture does not fit the scope",
            ),
            (
                |buffer| buffer.shadow = Shadow::None,
                "a shadow holds exactly what cancel saves",
            ),
            (
                |buffer| buffer.cancel = Cancel::Invalidate,
                "a shadow holds exactly what cancel saves",
            ),
            (
                |buffer| {
                    buffer.cancel = Cancel::Restore(vec![Save {
                        writer: Writer::Prefill,
                        region: Region::All,
                    }])
                },
                "a tail restore needs every step's touched entries saved",
            ),
            (
                |buffer| {
                    buffer.cancel = Cancel::Restore(vec![Save {
                        writer: Writer::Any,
                        region: Region::Layers {
                            first: 40,
                            count: 1,
                        },
                    }])
                },
                "a cancel save must be touched positions or whole layers",
            ),
            (
                |buffer| buffer.rollback = Rollback::Checkpoint(TAIL_CHECKPOINTS),
                "checkpoints hold a slot's state for at least one item",
            ),
            (
                |buffer| buffer.owner = Owner::FrontierOwner,
                "only captured drafter state has a frontier owner",
            ),
            (
                |buffer| buffer.planes = vec![128 * 528 + 16],
                "a positional plane must hold whole positions",
            ),
        ];
        for (change, rule) in rules {
            let mut state = compressed();
            change(&mut state.buffers[0]);
            assert_eq!(state.validate(), Err(invalid("swa.ring", rule)));
        }
        let mut slot = compressed();
        slot.buffers[2].rollback = Rollback::Truncate;
        assert_eq!(
            slot.validate().unwrap_err().rule,
            "only a positional buffer can roll back by position"
        );
        slot.buffers[2].name = "swa.ring";
        assert_eq!(
            slot.validate().unwrap_err().rule,
            "names must be unique and not empty"
        );
    }
}
