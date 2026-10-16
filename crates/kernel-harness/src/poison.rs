//! Poisoned relaunch, graph and PDL checks.
//!
//! A replay compared against an eager result that still sits in the same
//! buffers passes even when it writes nothing. Fill every buffer a launch
//! writes with [`POISON`] first, check afterwards that each element the
//! contract says is written no longer holds its fill and that every other
//! byte still does, and only then compare bits with the eager launch.
//!
//! Call sites follow these rules:
//!
//! - Poison every buffer a launched unit writes and the next launch fully
//!   rewrites before reading it: outputs, inter-kernel intermediates, and
//!   split-K or stream-K partial workspaces that a reducing CTA or kernel
//!   reads after other CTAs write them. Poison before the eager launch, every
//!   relaunch, every graph launch and every PDL producer.
//! - Restore bytes a launch reads before writing (in-place state) from their
//!   initial image, and poison only the bytes it writes without reading.
//! - Never poison self-resetting semaphores or counters, or comm inboxes,
//!   flags, epochs and generations.
//! - Poison only outside stream capture: a fill is a host copy followed by a
//!   stream synchronization. Poison before a PDL producer, never between the
//!   producer and its dependent. Across ranks, poison only after every rank's
//!   stream is idle ([`synchronize`]).
//! - Leave timing-only launches that compare nothing unpoisoned.
//! - Take write maps from the engine README and the kernel-core contract;
//!   where those are silent, from the CUDA source, and say so where the map
//!   is built.
//! - When an existing kernel fails a poisoned check, report it with a minimal
//!   reproduction. Never widen a map to make it pass.

use crate::{LabResult, Session, launch};
use kernel_core::contract::DType;
use runtime_cuda::{
    DeviceBuffer, Function, GraphExec, KernelArgument, Module, PinnedHostBuffer, Stream,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::ops::Range;

/// All-ones bytes: NaN in BF16, F32, F64 and FP8 E4M3, and no ID, index,
/// count or status the lab's kernels write.
pub const POISON: u8 = 0xff;

/// Host bytes per copy of a fill.
const CHUNK_BYTES: usize = 16 << 20;

/// Writes the bytes `pattern` puts at `offset..`, repeated from the buffer's
/// first byte.
fn write_pattern(offset: usize, bytes: &mut [u8], pattern: &[u8]) {
    if let [byte] = pattern {
        bytes.fill(*byte);
        return;
    }
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = pattern[(offset + index) % pattern.len()];
    }
}

#[cfg(test)]
fn pattern_bytes(offsets: Range<usize>, pattern: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0; offsets.len()];
    write_pattern(offsets.start, &mut bytes, pattern);
    bytes
}

/// Waits until every rank's stream is idle, so no peer still writes into a
/// buffer about to be poisoned.
pub fn synchronize(sessions: &[Session]) -> LabResult<()> {
    for session in sessions {
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Writes `bytes` at `offset` of `buffer` and waits for the copy.
pub fn write(
    session: &Session,
    buffer: &DeviceBuffer,
    offset: usize,
    bytes: &[u8],
) -> LabResult<()> {
    if bytes.is_empty() {
        // Nothing to copy; still fence the stream's earlier work.
        return session.stream.synchronize().map_err(|e| e.to_string());
    }
    // The copy waits for its own DMA, which the stream orders after that work.
    buffer
        .copy_from_bytes_at(offset, bytes, &session.stream)
        .map_err(|e| e.to_string())
}

/// Fills byte `ranges` of `buffer` with `pattern`, repeated from the buffer's
/// first byte, and waits for the copies. Never call it during stream capture.
pub fn fill(
    session: &Session,
    buffer: &DeviceBuffer,
    ranges: &[Range<usize>],
    pattern: &[u8],
) -> LabResult<()> {
    if pattern.is_empty() {
        return Err("empty poison pattern".into());
    }
    if let Some(range) = ranges
        .iter()
        .find(|range| range.start > range.end || range.end > buffer.byte_len())
    {
        return Err(format!(
            "fill range {range:?} lies outside a {}-byte buffer",
            buffer.byte_len()
        ));
    }
    let largest = ranges.iter().map(|range| range.end - range.start).max();
    let Some(largest) = largest.filter(|&bytes| bytes > 0) else {
        return session.stream.synchronize().map_err(|e| e.to_string());
    };
    // Each chunk is a stream-ordered DMA from page-locked staging, waited
    // for, so it lands after every earlier write on the stream.
    let mut staging = PinnedHostBuffer::new(&session.context, largest.min(CHUNK_BYTES))
        .map_err(|e| e.to_string())?;
    for range in ranges {
        let mut offset = range.start;
        while offset < range.end {
            let end = range.end.min(offset + staging.len());
            write_pattern(offset, &mut staging.as_mut_slice()[..end - offset], pattern);
            buffer
                .copy_from_pinned_at_async(offset, &staging, 0, end - offset, &session.stream)
                .map_err(|e| e.to_string())?
                .finish();
            offset = end;
        }
    }
    Ok(())
}

/// Fills all of `buffer` with `pattern`.
pub fn fill_all(session: &Session, buffer: &DeviceBuffer, pattern: &[u8]) -> LabResult<()> {
    fill(
        session,
        buffer,
        std::slice::from_ref(&(0..buffer.byte_len())),
        pattern,
    )
}

/// Fills whole buffers with [`POISON`].
pub fn poison(session: &Session, buffers: &[&DeviceBuffer]) -> LabResult<()> {
    for buffer in buffers {
        fill_all(session, buffer, &[POISON])?;
    }
    Ok(())
}

/// Every byte of `buffer`.
pub fn read(session: &Session, buffer: &DeviceBuffer) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

/// The byte ranges of one buffer that a launch's contract says it writes, in
/// whole elements of its dtype ([`WriteMap::of`]); every other byte must keep
/// its fill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteMap {
    element_bytes: usize,
    written: Vec<Range<usize>>,
}

impl WriteMap {
    /// Sorts `ranges` and joins adjacent ones. Ranges must be aligned to
    /// whole elements and must not overlap.
    fn new(
        element_bytes: usize,
        ranges: impl IntoIterator<Item = Range<usize>>,
    ) -> LabResult<Self> {
        if element_bytes == 0 {
            return Err("write map elements must have a size".into());
        }
        let mut sorted = ranges
            .into_iter()
            .filter(|range| range.start != range.end)
            .collect::<Vec<_>>();
        sorted.sort_by_key(|range| range.start);
        let mut written: Vec<Range<usize>> = Vec::with_capacity(sorted.len());
        for range in sorted {
            if range.start > range.end
                || range.start % element_bytes != 0
                || range.end % element_bytes != 0
            {
                return Err(format!(
                    "write range {range:?} does not cover whole {element_bytes}-byte elements"
                ));
            }
            match written.last_mut() {
                Some(last) if range.start < last.end => {
                    return Err(format!("write ranges {last:?} and {range:?} overlap"));
                }
                Some(last) if range.start == last.end => last.end = range.end,
                _ => written.push(range),
            }
        }
        Ok(Self {
            element_bytes,
            written,
        })
    }

    /// Maps over elements of `dtype`, counted in elements: the element size
    /// comes from the type, so a BF16 map cannot be counted in bytes or
    /// checked in 1-byte elements. There is no constructor that takes a byte
    /// count with a separate element size, so that mistake does not compile:
    ///
    /// ```compile_fail,E0599
    /// // 64 bytes of BF16 checked as 1-byte elements.
    /// kernel_harness::poison::WriteMap::prefix(64, 1);
    /// ```
    pub fn of(dtype: DType) -> Elements {
        Elements(Unit::Dtype(dtype))
    }

    /// Maps over `BYTES`-byte records that no [`DType`] names, such as
    /// 8-byte (score, index) pairs or 16-byte ring cells, counted in
    /// records. Sizes of 1, 2 or 4 bytes are dtypes and do not compile:
    ///
    /// ```compile_fail,E0080
    /// kernel_harness::poison::WriteMap::of_records::<2>();
    /// ```
    pub fn of_records<const BYTES: usize>() -> Elements {
        const {
            assert!(
                !matches!(BYTES, 0 | 1 | 2 | 4),
                "records of 1, 2 or 4 bytes are dtypes: use WriteMap::of"
            )
        };
        Elements(Unit::Record(BYTES))
    }

    pub fn element_bytes(&self) -> usize {
        self.element_bytes
    }

    pub fn ranges(&self) -> &[Range<usize>] {
        &self.written
    }

    pub fn written_bytes(&self) -> usize {
        self.written.iter().map(|range| range.len()).sum()
    }
}

/// The element type of a [`WriteMap`] ([`WriteMap::of`],
/// [`WriteMap::of_records`]); its constructors take counts and offsets in
/// elements of that type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Elements(Unit);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unit {
    Dtype(DType),
    Record(usize),
}

impl Elements {
    /// Bytes per element. Sub-byte codes have no element of their own: map
    /// packed E2M1 or FP6 bytes as [`DType::U8`].
    pub fn bytes(self) -> LabResult<usize> {
        let dtype = match self.0 {
            Unit::Dtype(dtype) => dtype,
            Unit::Record(bytes) => return Ok(bytes),
        };
        Ok(match dtype {
            DType::U8 | DType::Fp8E4M3 | DType::Fp8E5M2 | DType::Ue8M0 | DType::Ue4M3 => 1,
            DType::Bf16 | DType::F16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::I64 => 8,
            DType::Fp4E2M1 | DType::Fp6E2M3 | DType::Fp6E3M2 => {
                return Err(format!(
                    "{dtype:?} codes are smaller than a byte; map their packed bytes as U8"
                ));
            }
        })
    }

    /// Nothing is written.
    pub fn none(self) -> LabResult<WriteMap> {
        WriteMap::new(self.bytes()?, [])
    }

    /// The first `elements`, such as the rows below a live count, are
    /// written.
    pub fn prefix(self, elements: usize) -> LabResult<WriteMap> {
        self.ranges(std::iter::once(0..elements))
    }

    /// The first `bytes`, such as a whole buffer's `byte_len()`, are
    /// written; they must cover whole elements.
    pub fn byte_prefix(self, bytes: usize) -> LabResult<WriteMap> {
        self.byte_ranges(std::iter::once(0..bytes))
    }

    /// Rows of `row_elements` each at the given indices are written.
    pub fn rows(
        self,
        row_elements: usize,
        rows: impl IntoIterator<Item = usize>,
    ) -> LabResult<WriteMap> {
        self.ranges(
            rows.into_iter()
                .map(|row| row * row_elements..(row + 1) * row_elements),
        )
    }

    /// Element ranges; they are sorted, adjacent ones joined, and they must
    /// not overlap.
    pub fn ranges(self, ranges: impl IntoIterator<Item = Range<usize>>) -> LabResult<WriteMap> {
        let bytes = self.bytes()?;
        WriteMap::new(
            bytes,
            ranges
                .into_iter()
                .map(|range| range.start * bytes..range.end * bytes),
        )
    }

    /// Byte ranges, for maps placed by byte strides such as padded pages;
    /// each must cover whole elements.
    pub fn byte_ranges(
        self,
        ranges: impl IntoIterator<Item = Range<usize>>,
    ) -> LabResult<WriteMap> {
        WriteMap::new(self.bytes()?, ranges)
    }
}

/// The bytes a buffer held before a launch.
#[derive(Clone, Copy, Debug)]
pub enum Fill<'a> {
    /// `pattern` repeated from the buffer's first byte.
    Pattern(&'a [u8]),
    /// `pattern` outside the write map and, inside it, the bitwise complement
    /// of `expected`, a whole-buffer image. For dtypes without an invalid
    /// bit pattern, such as packed E2M1 codes, where 0xff is two valid -6
    /// codes: a correct write never equals its fill.
    Complement {
        pattern: &'a [u8],
        expected: &'a [u8],
    },
}

impl Fill<'_> {
    fn pattern(self) -> &'static str {
        match self {
            Fill::Pattern(_) => "pattern",
            Fill::Complement { .. } => "complement",
        }
    }

    fn byte(self, offset: usize, written: bool) -> u8 {
        match self {
            Fill::Pattern(pattern) => pattern[offset % pattern.len()],
            Fill::Complement { expected, .. } if written => !expected[offset],
            Fill::Complement { pattern, .. } => pattern[offset % pattern.len()],
        }
    }
}

/// Checks one buffer's bytes after a launch against the image it was filled
/// with: every element in `map` must differ from its fill in at least one
/// byte, and every byte outside `map` must still hold its fill.
pub fn check(label: &str, bytes: &[u8], map: &WriteMap, fill: Fill<'_>) -> Result<(), String> {
    match fill {
        Fill::Pattern(pattern) | Fill::Complement { pattern, .. } if pattern.is_empty() => {
            return Err(format!("{label}: empty fill pattern"));
        }
        Fill::Complement { expected, .. } if expected.len() < bytes.len() => {
            return Err(format!(
                "{label}: complement image holds {} of {} bytes",
                expected.len(),
                bytes.len()
            ));
        }
        _ => {}
    }
    if let Some(last) = map.written.last()
        && last.end > bytes.len()
    {
        return Err(format!(
            "{label}: write range {last:?} lies outside {} bytes",
            bytes.len()
        ));
    }
    let mut cursor = 0;
    for range in &map.written {
        untouched(label, bytes, cursor..range.start, fill)?;
        for element in range.clone().step_by(map.element_bytes) {
            let end = element + map.element_bytes;
            if (element..end).all(|offset| bytes[offset] == fill.byte(offset, true)) {
                return Err(format!(
                    "{label}: element {} (byte {element}) holds its {} fill {:02x?}; the launch did not write it",
                    element / map.element_bytes,
                    fill.pattern(),
                    &bytes[element..end]
                ));
            }
        }
        cursor = range.end;
    }
    untouched(label, bytes, cursor..bytes.len(), fill)
}

fn untouched(
    label: &str,
    bytes: &[u8],
    offsets: Range<usize>,
    fill: Fill<'_>,
) -> Result<(), String> {
    let changed = match fill {
        Fill::Pattern([byte])
        | Fill::Complement {
            pattern: [byte], ..
        } => bytes[offsets.clone()]
            .iter()
            .position(|value| value != byte)
            .map(|at| offsets.start + at),
        _ => offsets
            .clone()
            .find(|&offset| bytes[offset] != fill.byte(offset, false)),
    };
    match changed {
        None => Ok(()),
        Some(offset) => Err(format!(
            "{label}: byte {offset} outside the write map changed from {:#04x} to {:#04x}",
            fill.byte(offset, false),
            bytes[offset]
        )),
    }
}

/// Checks that every byte outside `may_write` still holds [`POISON`].
pub fn check_within(label: &str, bytes: &[u8], may_write: &[Range<usize>]) -> Result<(), String> {
    let mut allowed = may_write.to_vec();
    allowed.sort_by_key(|range| range.start);
    if let Some(range) = allowed
        .iter()
        .find(|range| range.start > range.end || range.end > bytes.len())
    {
        return Err(format!(
            "{label}: allowed range {range:?} lies outside {} bytes",
            bytes.len()
        ));
    }
    let mut cursor = 0;
    for range in allowed
        .into_iter()
        .chain(std::iter::once(bytes.len()..bytes.len()))
    {
        let gap = cursor..range.start.max(cursor);
        if let Some(at) = bytes[gap.clone()].iter().position(|&byte| byte != POISON) {
            let offset = gap.start + at;
            return Err(format!(
                "{label}: byte {offset} outside the bytes the live items may write changed from {POISON:#04x} to {:#04x}",
                bytes[offset]
            ));
        }
        cursor = cursor.max(range.end);
    }
    Ok(())
}

/// Checks in-place state after a launch: write-only elements no longer hold
/// poison, and bytes outside the write-only and read-modify-write ranges
/// still hold their initial image.
fn check_state(
    label: &str,
    bytes: &[u8],
    initial: &[u8],
    write_only: &WriteMap,
    updated: &[Range<usize>],
) -> Result<(), String> {
    if bytes.len() != initial.len() {
        return Err(format!(
            "{label}: state holds {} bytes, its initial image {}",
            bytes.len(),
            initial.len()
        ));
    }
    for range in write_only.ranges() {
        if range.end > bytes.len() {
            return Err(format!(
                "{label}: write range {range:?} lies outside {} bytes",
                bytes.len()
            ));
        }
        for element in range.clone().step_by(write_only.element_bytes) {
            let end = element + write_only.element_bytes;
            if bytes[element..end].iter().all(|&byte| byte == POISON) {
                return Err(format!(
                    "{label}: write-only element at byte {element} still holds poison; the launch did not write it"
                ));
            }
        }
    }
    if let Some(range) = updated
        .iter()
        .find(|range| range.start > range.end || range.end > bytes.len())
    {
        return Err(format!(
            "{label}: updated range {range:?} lies outside {} bytes",
            bytes.len()
        ));
    }
    let mut free = write_only
        .ranges()
        .iter()
        .chain(updated)
        .cloned()
        .collect::<Vec<_>>();
    free.sort_by_key(|range| range.start);
    let mut cursor = 0;
    for range in free
        .into_iter()
        .chain(std::iter::once(bytes.len()..bytes.len()))
    {
        let gap = cursor..range.start.max(cursor);
        if bytes[gap.clone()] == initial[gap.clone()] {
            cursor = cursor.max(range.end);
            continue;
        }
        if let Some(at) = bytes[gap.clone()]
            .iter()
            .zip(&initial[gap.clone()])
            .position(|(after, before)| after != before)
        {
            let offset = gap.start + at;
            return Err(format!(
                "{label}: byte {offset} outside the written state changed from {:#04x} to {:#04x}",
                initial[offset], bytes[offset]
            ));
        }
        cursor = cursor.max(range.end);
    }
    Ok(())
}

/// A launch phase of a replay check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Eager,
    Relaunch,
    Graph,
    Pdl,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Eager => "eager",
            Phase::Relaunch => "relaunch",
            Phase::Graph => "graph",
            Phase::Pdl => "PDL",
        }
    }

    /// The receipt field a passed replay of this phase sets.
    fn field(self) -> Option<&'static str> {
        match self {
            Phase::Eager => None,
            Phase::Relaunch => Some("deterministic_bitwise"),
            Phase::Graph => Some("graph_bitwise"),
            Phase::Pdl => Some("pdl_bitwise"),
        }
    }
}

/// Replay phases that ran on poisoned buffers and matched the eager bits.
#[derive(Clone, Debug, Default)]
pub struct Evidence {
    poisoned: Vec<String>,
    passed: Vec<(Phase, bool)>,
}

impl Evidence {
    /// Evidence over the named poisoned buffers.
    pub fn new<S: Into<String>>(poisoned: impl IntoIterator<Item = S>) -> Self {
        Self {
            poisoned: poisoned.into_iter().map(Into::into).collect(),
            passed: Vec::new(),
        }
    }

    /// Records a replay phase that ran poisoned and matched eager bits.
    /// `pdl_active` says whether the dependent launch had programmatic stream
    /// serialization; without it a PDL phase proves nothing about PDL.
    pub fn passed(&mut self, phase: Phase, pdl_active: bool) {
        self.passed.push((phase, pdl_active));
    }

    /// Whether `phase` passed at least once.
    pub fn has(&self, phase: Phase) -> bool {
        self.passed.iter().any(|&(passed, _)| passed == phase)
    }

    /// Sets `deterministic_bitwise`, `graph_bitwise` and `pdl_bitwise` only
    /// for phases that passed, `pdl_bitwise` only when PDL was active, and
    /// names the poisoned buffers.
    pub fn record(&self, receipt: &mut BTreeMap<String, Value>) {
        for &(phase, pdl_active) in &self.passed {
            if phase == Phase::Pdl && !pdl_active {
                continue;
            }
            if let Some(field) = phase.field() {
                receipt.insert(field.into(), json!(true));
            }
        }
        receipt.insert("poisoned".into(), json!(self.poisoned));
    }
}

struct Output<'a> {
    name: String,
    buffer: &'a DeviceBuffer,
    map: WriteMap,
    pattern: Vec<u8>,
    /// For complement fills, the image the eager launch must write.
    complement: Option<Vec<u8>>,
}

struct Scratch<'a> {
    name: String,
    buffer: &'a DeviceBuffer,
    ranges: Vec<Range<usize>>,
    /// For bounded scratch, the byte ranges a launch may write; every other
    /// byte must keep its poison.
    may_write: Option<Vec<Range<usize>>>,
}

struct State<'a> {
    name: String,
    buffer: &'a DeviceBuffer,
    initial: Vec<u8>,
    write_only: WriteMap,
    updated: Vec<Range<usize>>,
}

/// Drives the eager launch and its relaunch, graph and PDL replays: before
/// each launch it restores in-place state, poisons outputs, scratch and
/// write-only state, and afterwards checks the write maps and compares every
/// output and state bitwise with the eager launch.
pub struct Replay<'a> {
    session: &'a Session,
    outputs: Vec<Output<'a>>,
    scratch: Vec<Scratch<'a>>,
    states: Vec<State<'a>>,
    expected: Option<Vec<Vec<u8>>>,
    evidence: Evidence,
}

impl<'a> Replay<'a> {
    pub fn new(session: &'a Session) -> Self {
        Self {
            session,
            outputs: Vec::new(),
            scratch: Vec::new(),
            states: Vec::new(),
            expected: None,
            evidence: Evidence::default(),
        }
    }

    /// An output poisoned with [`POISON`] and checked against `map`.
    pub fn output(self, name: impl Into<String>, buffer: &'a DeviceBuffer, map: WriteMap) -> Self {
        self.output_with(name, buffer, map, &[POISON])
    }

    /// An output filled with another `pattern`, for outputs where all-ones
    /// elements are valid values, such as -1 route padding.
    pub fn output_with(
        mut self,
        name: impl Into<String>,
        buffer: &'a DeviceBuffer,
        map: WriteMap,
        pattern: &[u8],
    ) -> Self {
        self.push_output(name.into(), buffer, map, pattern.to_vec(), None);
        self
    }

    /// An output whose dtype has no invalid bit pattern: its written ranges
    /// start from the complement of `expected` (the oracle's whole-buffer
    /// image) before eager, and of the eager bytes before each replay.
    pub fn output_complement(
        mut self,
        name: impl Into<String>,
        buffer: &'a DeviceBuffer,
        map: WriteMap,
        expected: Vec<u8>,
    ) -> Self {
        self.push_output(name.into(), buffer, map, vec![POISON], Some(expected));
        self
    }

    fn push_output(
        &mut self,
        name: String,
        buffer: &'a DeviceBuffer,
        map: WriteMap,
        pattern: Vec<u8>,
        complement: Option<Vec<u8>>,
    ) {
        self.evidence.poisoned.push(name.clone());
        self.outputs.push(Output {
            name,
            buffer,
            map,
            pattern,
            complement,
        });
    }

    /// A buffer poisoned before every launch and never checked, such as a
    /// workspace a merge or reducing CTA reads in the same launch.
    pub fn scratch(self, name: impl Into<String>, buffer: &'a DeviceBuffer) -> Self {
        let whole = 0..buffer.byte_len();
        self.scratch_ranges(name, buffer, vec![whole])
    }

    /// Byte ranges of a buffer poisoned before every launch and never
    /// checked.
    pub fn scratch_ranges(
        mut self,
        name: impl Into<String>,
        buffer: &'a DeviceBuffer,
        ranges: Vec<Range<usize>>,
    ) -> Self {
        let name = name.into();
        self.evidence.poisoned.push(name.clone());
        self.scratch.push(Scratch {
            name,
            buffer,
            ranges,
            may_write: None,
        });
        self
    }

    /// A whole buffer poisoned before every launch whose written bytes are
    /// not checked or compared, but which may write only `may_write`, such as
    /// the workspace rows of the live items: every other byte must still hold
    /// poison after each launch.
    pub fn scratch_within(
        mut self,
        name: impl Into<String>,
        buffer: &'a DeviceBuffer,
        may_write: Vec<Range<usize>>,
    ) -> Self {
        let name = name.into();
        self.evidence.poisoned.push(name.clone());
        self.scratch.push(Scratch {
            name,
            buffer,
            ranges: std::iter::once(0..buffer.byte_len()).collect(),
            may_write: Some(may_write),
        });
        self
    }

    /// In-place state: `initial` (the whole buffer) is restored before every
    /// launch, then `write_only` is poisoned. After a launch `write_only` must
    /// be written, and bytes outside it and `updated` (read-modify-write
    /// ranges) must still hold `initial`.
    pub fn state(
        mut self,
        name: impl Into<String>,
        buffer: &'a DeviceBuffer,
        initial: Vec<u8>,
        write_only: WriteMap,
        updated: Vec<Range<usize>>,
    ) -> Self {
        let name = name.into();
        self.evidence.poisoned.push(name.clone());
        self.states.push(State {
            name,
            buffer,
            initial,
            write_only,
            updated,
        });
        self
    }

    /// Restores in-place state and poisons everything the next launch writes.
    /// Collective callers must prepare every idle rank before launching any.
    pub fn prepare(&self, phase: Phase) -> LabResult<()> {
        let session = self.session;
        for state in &self.states {
            if state.initial.len() != state.buffer.byte_len() {
                return Err(format!(
                    "{}: initial image holds {} of {} bytes",
                    state.name,
                    state.initial.len(),
                    state.buffer.byte_len()
                ));
            }
            write(session, state.buffer, 0, &state.initial)?;
            fill(session, state.buffer, state.write_only.ranges(), &[POISON])?;
        }
        for (index, output) in self.outputs.iter().enumerate() {
            fill_all(session, output.buffer, &output.pattern)?;
            if let Some(image) = self.complement_image(index, phase)? {
                for range in output.map.ranges() {
                    let complement = image[range.clone()]
                        .iter()
                        .map(|byte| !byte)
                        .collect::<Vec<_>>();
                    write(session, output.buffer, range.start, &complement)?;
                }
            }
        }
        for scratch in &self.scratch {
            fill(session, scratch.buffer, &scratch.ranges, &[POISON])
                .map_err(|e| format!("{}: {e}", scratch.name))?;
        }
        Ok(())
    }

    /// The whole-buffer image a complement-filled output starts from.
    fn complement_image(&self, index: usize, phase: Phase) -> LabResult<Option<&[u8]>> {
        let output = &self.outputs[index];
        let Some(oracle) = &output.complement else {
            return Ok(None);
        };
        let image: &[u8] = match (phase, &self.expected) {
            (Phase::Eager, _) => oracle,
            (_, Some(expected)) => &expected[index],
            (_, None) => return Err(format!("{}: replay before eager", output.name)),
        };
        if image.len() != output.buffer.byte_len() {
            return Err(format!(
                "{}: complement image holds {} of {} bytes",
                output.name,
                image.len(),
                output.buffer.byte_len()
            ));
        }
        Ok(Some(image))
    }

    /// Checks every write map and, after eager, compares with the eager bits.
    /// Checks the completed phase. Collective callers enqueue every rank,
    /// wait for every stream, then finish every rank even after a miss.
    pub fn finish(&mut self, phase: Phase, pdl_active: bool) -> LabResult<()> {
        let session = self.session;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let mut images = Vec::with_capacity(self.outputs.len() + self.states.len());
        for (index, output) in self.outputs.iter().enumerate() {
            let bytes = read(session, output.buffer)?;
            let label = format!("{} {}", output.name, phase.name());
            let fill = match self.complement_image(index, phase)? {
                Some(expected) => Fill::Complement {
                    pattern: &output.pattern,
                    expected,
                },
                None => Fill::Pattern(&output.pattern),
            };
            check(&label, &bytes, &output.map, fill)?;
            images.push(bytes);
        }
        for scratch in &self.scratch {
            if let Some(may_write) = &scratch.may_write {
                let bytes = read(session, scratch.buffer)?;
                let label = format!("{} {}", scratch.name, phase.name());
                check_within(&label, &bytes, may_write)?;
            }
        }
        for state in &self.states {
            let bytes = read(session, state.buffer)?;
            let label = format!("{} {}", state.name, phase.name());
            check_state(
                &label,
                &bytes,
                &state.initial,
                &state.write_only,
                &state.updated,
            )?;
            images.push(bytes);
        }
        if phase == Phase::Eager {
            if self.expected.is_some() {
                return Err("the eager phase already ran".into());
            }
            self.expected = Some(images);
            return Ok(());
        }
        let expected = self
            .expected
            .as_ref()
            .ok_or_else(|| format!("{} replay before eager", phase.name()))?;
        let names = self
            .outputs
            .iter()
            .map(|output| &output.name)
            .chain(self.states.iter().map(|state| &state.name));
        for ((name, eager), actual) in names.zip(expected).zip(&images) {
            if eager == actual {
                continue;
            }
            if let Some(offset) = eager.iter().zip(actual).position(|(a, b)| a != b) {
                return Err(format!(
                    "{name} {}: byte {offset} is {:#04x}, eager wrote {:#04x}",
                    phase.name(),
                    actual[offset],
                    eager[offset]
                ));
            }
        }
        self.evidence.passed(phase, pdl_active);
        Ok(())
    }

    /// Runs one phase: prepare, `launch`, then check.
    pub fn phase(
        &mut self,
        phase: Phase,
        launch: impl FnOnce() -> LabResult<()>,
        pdl_active: bool,
    ) -> LabResult<()> {
        self.prepare(phase)?;
        launch()?;
        self.finish(phase, pdl_active)
    }

    /// The eager launch whose bytes every replay must reproduce.
    pub fn eager(&mut self, launch: impl FnOnce() -> LabResult<()>) -> LabResult<()> {
        self.phase(Phase::Eager, launch, false)
    }

    /// The same launch again (`deterministic_bitwise`).
    pub fn relaunch(&mut self, launch: impl FnOnce() -> LabResult<()>) -> LabResult<()> {
        self.phase(Phase::Relaunch, launch, false)
    }

    /// One launch of a graph captured earlier (`graph_bitwise`).
    pub fn graph(&mut self, graph: &GraphExec) -> LabResult<()> {
        let session = self.session;
        let stream = &session.stream;
        self.phase(
            Phase::Graph,
            || graph.launch(stream).map_err(|e| e.to_string()),
            false,
        )
    }

    /// `producer` then `launch`, prepared before the producer so nothing
    /// separates the PDL pair (`pdl_bitwise` when `pdl_active`).
    pub fn pdl(
        &mut self,
        producer: impl FnOnce() -> LabResult<()>,
        launch: impl FnOnce() -> LabResult<()>,
        pdl_active: bool,
    ) -> LabResult<()> {
        self.phase(
            Phase::Pdl,
            || {
                producer()?;
                launch()
            },
            pdl_active,
        )
    }

    /// The eager bytes of an output or state.
    pub fn expected(&self, name: &str) -> Option<&[u8]> {
        let expected = self.expected.as_ref()?;
        self.outputs
            .iter()
            .map(|output| &output.name)
            .chain(self.states.iter().map(|state| &state.name))
            .position(|candidate| candidate == name)
            .map(|index| expected[index].as_slice())
    }

    pub fn evidence(&self) -> &Evidence {
        &self.evidence
    }

    /// See [`Evidence::record`].
    pub fn record(&self, receipt: &mut BTreeMap<String, Value>) {
        self.evidence.record(receipt);
    }
}

type Map<'a> = Box<dyn Fn(&Session) -> LabResult<WriteMap> + 'a>;

#[derive(Clone)]
pub struct RouteGeometry {
    pub routes: usize,
    pub experts: Range<usize>,
    pub block: usize,
    pub sorted_offset: usize,
}

pub struct RouteBuffers<'a> {
    pub ids: &'a DeviceBuffer,
    pub weights: &'a DeviceBuffer,
    pub counts: &'a DeviceBuffer,
    pub offsets: &'a DeviceBuffer,
    pub sorted: &'a DeviceBuffer,
    pub padded: &'a DeviceBuffer,
    pub active: Option<&'a DeviceBuffer>,
}

struct CheckedOutput<'a> {
    name: String,
    buffer: &'a DeviceBuffer,
    map: Map<'a>,
    pattern: Vec<u8>,
}

struct CheckedScratch<'a> {
    name: String,
    buffer: &'a DeviceBuffer,
    start: usize,
    end: usize,
    map: Option<Map<'a>>,
}

/// Write checks for compared multi-kernel units, a sibling of [`Replay`]:
/// each output's map may be computed after the launch from the session
/// (for example from freshly produced route IDs), a phase checks every
/// buffer and reports all misses ([`Failures`]) instead of the first, and
/// [`phase_all`] prepares several units before one launch. Maps may depend
/// on the freshly produced route IDs, never on GPU counts, offsets or padded
/// totals. Existing router oracles still check those IDs.
pub struct Checks<'a> {
    session: &'a Session,
    outputs: Vec<CheckedOutput<'a>>,
    scratch: Vec<CheckedScratch<'a>>,
    expected: Option<Vec<Vec<u8>>>,
    evidence: Evidence,
}

impl<'a> Checks<'a> {
    pub fn new(session: &'a Session) -> Self {
        Self {
            session,
            outputs: Vec::new(),
            scratch: Vec::new(),
            expected: None,
            evidence: Evidence::default(),
        }
    }

    pub fn output(self, name: &str, buffer: &'a DeviceBuffer, map: WriteMap) -> Self {
        self.dynamic(name, buffer, move |_| Ok(map.clone()), &[POISON])
    }

    pub fn dynamic(
        mut self,
        name: &str,
        buffer: &'a DeviceBuffer,
        map: impl Fn(&Session) -> LabResult<WriteMap> + 'a,
        pattern: &[u8],
    ) -> Self {
        self.outputs.push(CheckedOutput {
            name: name.into(),
            buffer,
            map: Box::new(map),
            pattern: pattern.into(),
        });
        self
    }

    pub fn route(
        mut self,
        geometry: RouteGeometry,
        buffers: RouteBuffers<'a>,
        pattern: &[u8],
    ) -> LabResult<Self> {
        // route_parallel.cuh:267-323 and row/router.cuh:148-337.
        let local = geometry.experts.len();
        for (name, buffer, dtype, words) in [
            ("route IDs", buffers.ids, DType::I32, geometry.routes),
            (
                "route weights",
                buffers.weights,
                DType::F32,
                geometry.routes,
            ),
            ("expert counts", buffers.counts, DType::I32, local),
            ("expert offsets", buffers.offsets, DType::I32, local + 1),
            ("padded total", buffers.padded, DType::I32, 1),
        ] {
            let map = WriteMap::of(dtype).prefix(words)?;
            self = self.dynamic(name, buffer, move |_| Ok(map.clone()), pattern);
        }
        let sorted_geometry = geometry.clone();
        self = self.dynamic(
            "sorted routes and padding",
            buffers.sorted,
            move |s| {
                let counts = counts(
                    &ids(s, buffers.ids, sorted_geometry.routes)?,
                    sorted_geometry.experts.clone(),
                )?;
                sorted_map(
                    &counts,
                    sorted_geometry.block,
                    sorted_geometry.sorted_offset,
                )
            },
            pattern,
        );
        if let Some(active) = buffers.active {
            self = self.dynamic(
                "active experts",
                active,
                move |s| {
                    let counts = counts(
                        &ids(s, buffers.ids, geometry.routes)?,
                        geometry.experts.clone(),
                    )?;
                    WriteMap::of(DType::I32).prefix(1 + counts.iter().filter(|&&n| n > 0).count())
                },
                pattern,
            );
        }
        Ok(self)
    }

    /// Poison arithmetic data only; the prefix holds persistent protocol state.
    pub fn scratch(mut self, name: &str, buffer: &'a DeviceBuffer, start: usize) -> Self {
        self.scratch.push(CheckedScratch {
            name: name.into(),
            buffer,
            start,
            end: buffer.byte_len(),
            map: None,
        });
        self
    }

    pub fn scratch_map(
        self,
        name: &str,
        buffer: &'a DeviceBuffer,
        start: usize,
        map: impl Fn(&Session) -> LabResult<WriteMap> + 'a,
    ) -> Self {
        self.scratch_map_region(name, buffer, start, buffer.byte_len(), map)
    }

    pub fn scratch_map_region(
        mut self,
        name: &str,
        buffer: &'a DeviceBuffer,
        start: usize,
        end: usize,
        map: impl Fn(&Session) -> LabResult<WriteMap> + 'a,
    ) -> Self {
        self.scratch.push(CheckedScratch {
            name: name.into(),
            buffer,
            start,
            end,
            map: Some(Box::new(map)),
        });
        self
    }

    pub fn prepare(&self) -> LabResult<()> {
        for output in &self.outputs {
            fill_all(self.session, output.buffer, &output.pattern)?;
        }
        for scratch in &self.scratch {
            if scratch.start > scratch.end || scratch.end > scratch.buffer.byte_len() {
                return Err(format!("{}: invalid scratch region", scratch.name));
            }
            fill(
                self.session,
                scratch.buffer,
                std::slice::from_ref(&(scratch.start..scratch.end)),
                &[POISON],
            )?;
        }
        Ok(())
    }

    /// Check every buffer even when an earlier buffer missed a required write.
    pub fn finish(&mut self, phase: Phase, pdl: bool) -> LabResult<()> {
        let mut failures = Failures::default();
        let mut images = Vec::with_capacity(self.outputs.len());
        for output in &self.outputs {
            let bytes = read(self.session, output.buffer)?;
            let result = (output.map)(self.session).and_then(|map| {
                check(
                    &format!("{} {}", output.name, phase.name()),
                    &bytes,
                    &map,
                    Fill::Pattern(&output.pattern),
                )
            });
            failures.check(&output.name, result);
            images.push(bytes);
        }
        for scratch in &self.scratch {
            if let Some(map) = &scratch.map {
                let bytes = read(self.session, scratch.buffer)?;
                failures.check(
                    &scratch.name,
                    map(self.session).and_then(|map| {
                        check(
                            &format!("{} {}", scratch.name, phase.name()),
                            &bytes[scratch.start..scratch.end],
                            &map,
                            Fill::Pattern(&[POISON]),
                        )
                    }),
                );
            }
        }
        if let Some(expected) = &self.expected {
            for ((output, before), after) in self.outputs.iter().zip(expected).zip(&images) {
                if let Some(offset) = before.iter().zip(after).position(|(a, b)| a != b) {
                    failures.check(
                        &output.name,
                        Err(format!(
                            "{} {}: byte {offset} differs from eager",
                            output.name,
                            phase.name()
                        )),
                    );
                }
            }
        } else if phase != Phase::Eager {
            failures.check("baseline", Err("replay has no checked eager image".into()));
        }
        failures.finish()?;
        if phase == Phase::Eager {
            self.expected = Some(images);
            self.evidence = Evidence::new(
                self.outputs
                    .iter()
                    .map(|o| o.name.clone())
                    .chain(self.scratch.iter().map(|s| s.name.clone())),
            );
        } else {
            self.evidence.passed(phase, pdl);
        }
        Ok(())
    }

    pub fn phase(
        &mut self,
        phase: Phase,
        pdl: bool,
        launch: impl FnOnce() -> LabResult<()>,
    ) -> LabResult<()> {
        self.prepare()?;
        launch()?;
        self.finish(phase, pdl)
    }

    pub fn record(&self, receipt: &mut BTreeMap<String, serde_json::Value>) {
        self.evidence.record(receipt);
    }
}

/// Collect strict failures across cases, then fail the test with reproductions.
#[derive(Default)]
pub struct Failures(Vec<String>);

impl Failures {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn check(&mut self, case: &str, result: LabResult<()>) {
        if let Err(error) = result {
            let error = format!("{case}: {error}");
            eprintln!("poisoned replay failure: {error}");
            self.0.push(error);
        }
    }

    pub fn finish(self) -> LabResult<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(self.0.join("\n"))
        }
    }
}

pub fn ids(session: &Session, buffer: &DeviceBuffer, routes: usize) -> LabResult<Vec<i32>> {
    let bytes = read(session, buffer)?;
    if routes * 4 > bytes.len() {
        return Err("live routes exceed ID allocation".into());
    }
    Ok(bytes[..routes * 4]
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}

pub fn counts(ids: &[i32], experts: Range<usize>) -> LabResult<Vec<usize>> {
    let mut counts = vec![0; experts.len()];
    for &id in ids {
        let id = usize::try_from(id).map_err(|_| "route ID was not written")?;
        if experts.contains(&id) {
            counts[id - experts.start] += 1;
        }
    }
    Ok(counts)
}

pub fn phase_all(
    checks: &mut [Checks<'_>],
    phase: Phase,
    pdl: bool,
    launch: impl FnOnce() -> LabResult<()>,
) -> LabResult<()> {
    for unit in checks.iter() {
        unit.prepare()?;
    }
    launch()?;
    let mut failures = Failures::default();
    for (layer, unit) in checks.iter_mut().enumerate() {
        failures.check(
            &format!("layer {layer} {}", phase.name()),
            unit.finish(phase, pdl),
        );
    }
    failures.finish()
}

/// route_parallel.cuh and row/router.cuh: all padded sorted slots are stored,
/// including -1 padding, which needs a fill distinct from all-ones.
pub fn sorted_map(counts: &[usize], block: usize, offset: usize) -> LabResult<WriteMap> {
    let padded = counts
        .iter()
        .map(|&n| n.div_ceil(block) * block)
        .sum::<usize>();
    WriteMap::of(DType::I32).byte_ranges(std::iter::once(offset..offset + padded * 4))
}

pub fn routed_map(ids: &[i32], experts: Range<usize>, width: usize) -> LabResult<WriteMap> {
    WriteMap::of(DType::Bf16).rows(
        width,
        ids.iter()
            .enumerate()
            .filter_map(|(row, &id)| experts.contains(&(id as usize)).then_some(row)),
    )
}

/// The PDL producer probe (`probe_pdl_prefetch_sm120`): it holds the live
/// count at zero through a producer tail, then republishes it, so a
/// consumer that reads the count before its PDL wait during that tail does
/// nothing. Its
/// function and buffers live as long as the probe, through the consumer's
/// completion.
pub struct LiveProducer {
    function: Function,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
}

impl LiveProducer {
    /// SM120's producer, `chop_probe_pdl_prefetch_producer_v1` of
    /// `probe_pdl_prefetch_sm120`.
    pub fn new(session: &Session) -> LabResult<Self> {
        let module = session.module("probe_pdl_prefetch_sm120")?;
        let function = module
            .function("chop_probe_pdl_prefetch_producer_v1")
            .map_err(|e| e.to_string())?;
        Self::with_function(session, function)
    }

    /// A producer from another architecture's kernel `function`, which takes
    /// SM120's producer's parameters in order: an eviction buffer of 16-byte
    /// vectors to stream (16 zero bytes here), a checksum buffer of one u32
    /// per thread, the u64 count of eviction vectors, the rewrite
    /// descriptors ([`Rewrites`]: per input a staged-copy pointer, the input
    /// pointer, a u32 word count and u32 flags) and their u32 count, the u32
    /// live count buffer, the u32 count to republish, and a u64 tail in
    /// nanoseconds. It launches as one block of 128 threads with
    /// programmatic stream serialization and must: wait on its own
    /// predecessor, zero every rewritten input and the live count, release
    /// its dependents, spin for the tail, then restore each input from its
    /// staged copy and publish the count (`cuda/probes/pdl_prefetch.cuh`,
    /// SM120's).
    pub fn with_function(session: &Session, function: Function) -> LabResult<Self> {
        let eviction = session.allocate(16)?;
        eviction
            .copy_from_bytes(&[0; 16], &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            function,
            eviction,
            checksum: session.allocate(128 * 4)?,
        })
    }

    /// Zeroes the u32 count in `live`, releases its dependent, and after
    /// its tail republishes the count `live` held before the launch, read
    /// from the device so the producer can never publish prepared rows in
    /// place of live ones. The read waits for the stream, so call it before
    /// the dependent, never between the two.
    pub fn launch(&self, session: &Session, live: &DeviceBuffer) -> LabResult<()> {
        self.launch_with(session, live, &self.eviction, 0)
    }

    /// [`LiveProducer::launch`] that also zeroes each input of `rewrites`
    /// before releasing its dependent and copies its staged image back after
    /// the tail, with the count: a consumer that reads either the count or
    /// one of those inputs before its PDL wait, while the producer holds
    /// them, sees zeros. A read that lands after the tail sees the restored
    /// bytes, so this probes for misordered reads; it does not guarantee to
    /// catch them.
    pub fn launch_rewriting(
        &self,
        session: &Session,
        live: &DeviceBuffer,
        rewrites: &Rewrites,
    ) -> LabResult<()> {
        self.launch_with(session, live, &rewrites.descriptors, rewrites.count)
    }

    fn launch_with(
        &self,
        session: &Session,
        live: &DeviceBuffer,
        operands: &DeviceBuffer,
        operand_count: u32,
    ) -> LabResult<()> {
        let rows = live_count(session, live)?;
        self.function
            .launch(
                &session.stream,
                launch::simple_config(
                    (1, 1, 1),
                    (128, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::Buffer(&self.eviction),
                    KernelArgument::Buffer(&self.checksum),
                    KernelArgument::U64(1),
                    KernelArgument::Buffer(operands),
                    KernelArgument::U32(operand_count),
                    KernelArgument::Buffer(live),
                    KernelArgument::U32(rows),
                    KernelArgument::U64(15_000),
                ],
            )
            .map_err(|e| e.to_string())
    }
}

/// Inputs a [`LiveProducer`] rewrites ([`LiveProducer::launch_rewriting`]):
/// descriptors of (staged image, input) pairs of whole u32 words. The
/// descriptors hold raw addresses, so the caller keeps both buffers of every
/// pair alive while the producer runs.
pub struct Rewrites {
    descriptors: DeviceBuffer,
    count: u32,
}

impl Rewrites {
    pub fn new(session: &Session, pairs: &[(&DeviceBuffer, &DeviceBuffer)]) -> LabResult<Self> {
        let mut descriptors = Vec::new();
        for (staged, input) in pairs {
            let bytes = input.byte_len();
            if staged.byte_len() < bytes {
                return Err(format!(
                    "a staged image of {} bytes cannot rewrite {bytes}",
                    staged.byte_len()
                ));
            }
            let address = |buffer: &DeviceBuffer| {
                buffer
                    .device_address_at(0, bytes)
                    .map_err(|e| e.to_string())
            };
            descriptors.extend(rewrite_descriptor(
                address(staged)?,
                address(input)?,
                bytes,
            )?);
        }
        let count = u32::try_from(pairs.len()).map_err(|_| "too many rewrites")?;
        let buffer = session.allocate(descriptors.len().max(16))?;
        write(session, &buffer, 0, &descriptors)?;
        Ok(Self {
            descriptors: buffer,
            count,
        })
    }
}

/// One `chop::probe::PdlOperand` (`cuda/probes/pdl_prefetch.cuh`): the
/// staged and input addresses, the u32 words, and flags 0, so the producer
/// zeroes the input before its release and copies the staged words back
/// after its tail.
fn rewrite_descriptor(staged: u64, input: u64, bytes: usize) -> LabResult<Vec<u8>> {
    if bytes % 4 != 0 {
        return Err(format!(
            "a rewritten input of {bytes} bytes is not u32 words"
        ));
    }
    let words = u32::try_from(bytes / 4).map_err(|_| "a rewritten input exceeds u32 words")?;
    let mut descriptor = Vec::with_capacity(24);
    descriptor.extend(staged.to_ne_bytes());
    descriptor.extend(input.to_ne_bytes());
    descriptor.extend(words.to_ne_bytes());
    descriptor.extend(0_u32.to_ne_bytes());
    Ok(descriptor)
}

/// The u32 live count at the start of `live`.
pub fn live_count(session: &Session, live: &DeviceBuffer) -> LabResult<u32> {
    let mut count = [0; 4];
    live.copy_to_bytes(&mut count, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(u32::from_ne_bytes(count))
}

/// SM120's PDL copy producer entry, which kernel-lab builds into its latent
/// lab cubin (`latent_shapes::pdl_copy_symbol`).
pub const SM120_COPY_PRODUCER: &str = "chop_latent_latent_h16_l512_pdl_copy_v1";

/// SM120's PDL copy producer ([`SM120_COPY_PRODUCER`]): it releases its
/// dependents at entry, spins, then copies a staged input over the
/// consumer's, so a consumer that reads that input before its PDL wait
/// sees its poison or stale bytes. It writes a real input, unlike
/// [`LiveProducer`].
pub struct CopyProducer {
    function: Function,
}

impl CopyProducer {
    /// The copy entry ([`SM120_COPY_PRODUCER`]) of `module`.
    pub fn new(module: &Module) -> LabResult<Self> {
        Ok(Self::with_function(
            module
                .function(SM120_COPY_PRODUCER)
                .map_err(|e| e.to_string())?,
        ))
    }

    /// A copy producer from another architecture's kernel `function`,
    /// which takes the lab copy entry's parameters in order: the source
    /// and destination buffers, a u64 count of u32 words and a u64 spin in
    /// clock cycles. It launches as 8 blocks of 256 threads and must
    /// release its dependents at entry, spin, then copy.
    pub fn with_function(function: Function) -> Self {
        Self { function }
    }

    /// Copies all of `src` into `dst` (a whole number of u32 words), or as
    /// much of it as `dst` holds.
    pub fn copy(&self, stream: &Stream, src: &DeviceBuffer, dst: &DeviceBuffer) -> LabResult<()> {
        let bytes = src.byte_len().min(dst.byte_len());
        if bytes % 4 != 0 {
            return Err("PDL copy needs whole u32 words".into());
        }
        self.function
            .launch(
                stream,
                launch::simple_config((8, 1, 1), (256, 1, 1), 0, (1, 1, 1), false)?,
                &[
                    KernelArgument::Buffer(src),
                    KernelArgument::Buffer(dst),
                    KernelArgument::U64((bytes / 4) as u64),
                    KernelArgument::U64(200_000),
                ],
            )
            .map_err(|e| e.to_string())
    }
}

/// Runs `launch` eager, relaunched, from a graph and behind a
/// [`LiveProducer`] on `live`, checking every write map and the eager bits,
/// and records the evidence in `receipt`. The producer republishes the
/// count `live` already holds, the one the write maps cover, so a launch
/// whose prepared rows exceed its live rows is checked at its live rows in
/// every phase.
pub fn replay_all(
    session: &Session,
    mut replay: Replay<'_>,
    live: &DeviceBuffer,
    launch: impl Fn() -> LabResult<()>,
    receipt: &mut BTreeMap<String, Value>,
) -> LabResult<()> {
    let producer = LiveProducer::new(session)?;
    replay.eager(&launch)?;
    replay.relaunch(&launch)?;
    let graph = session.capture(|_| launch())?;
    replay.graph(&graph)?;
    replay.pdl(|| producer.launch(session, live), &launch, true)?;
    replay.record(receipt);
    Ok(())
}

/// The live counts a one-graph replay test covers at `capacity`: capacity,
/// 0, 1, the middle and capacity - 1, in that order without repeats, so no
/// replay can rely on the previous live count.
pub fn live_counts(capacity: u32) -> Vec<u32> {
    let mut counts = Vec::new();
    for live in [capacity, 0, 1, capacity / 2, capacity.saturating_sub(1)] {
        if live <= capacity && !counts.contains(&live) {
            counts.push(live);
        }
    }
    counts
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    fn map(element_bytes: usize, ranges: &[Range<usize>]) -> WriteMap {
        WriteMap::new(element_bytes, ranges.iter().cloned()).expect("valid map")
    }

    #[test]
    fn a_written_element_left_at_its_fill_fails() {
        let mut bytes = vec![POISON; 8];
        bytes[..2].copy_from_slice(&0x3f80_u16.to_le_bytes());
        let written = map(2, &[0..4]);
        let error = check("out", &bytes, &written, Fill::Pattern(&[POISON])).unwrap_err();
        assert!(error.contains("element 1"), "{error}");
        bytes[2..4].copy_from_slice(&0x4000_u16.to_le_bytes());
        check("out", &bytes, &written, Fill::Pattern(&[POISON])).expect("both written");
    }

    #[test]
    fn bounded_scratch_rejects_writes_outside_its_ranges() {
        let mut bytes = vec![POISON; 16];
        bytes[4..8].fill(0);
        assert!(check_within("inside", &bytes, &[4..8]).is_ok());
        assert!(check_within("overlapping", &bytes, &[6..10, 2..6]).is_ok());
        assert!(check_within("none", &bytes, &[]).is_err());
        bytes[12] = 0;
        assert!(check_within("outside", &bytes, &[4..8]).is_err());
        assert!(check_within("beyond", &bytes, &[4..20]).is_err());
    }

    #[test]
    fn a_changed_untouched_byte_fails() {
        let mut bytes = vec![POISON; 8];
        bytes[..4].copy_from_slice(&1.0_f32.to_le_bytes());
        let written = map(4, &[0..4]);
        check("out", &bytes, &written, Fill::Pattern(&[POISON])).expect("tail untouched");
        bytes[6] = 0;
        let error = check("out", &bytes, &written, Fill::Pattern(&[POISON])).unwrap_err();
        assert!(error.contains("byte 6"), "{error}");
    }

    #[test]
    fn one_changed_byte_of_an_element_counts_as_written() {
        let bytes = [0xff, 0x3f, 0xff, 0xff];
        check("out", &bytes, &map(2, &[0..2]), Fill::Pattern(&[POISON]))
            .expect("the high byte changed");
    }

    #[test]
    fn a_two_byte_pattern_aligns_to_the_buffer_start() {
        let pattern = 0x7fa5_u16.to_ne_bytes();
        assert_eq!(
            pattern_bytes(3..7, &pattern),
            vec![pattern[1], pattern[0], pattern[1], pattern[0]]
        );
        let mut bytes = pattern_bytes(0..8, &pattern);
        bytes[2..4].copy_from_slice(&0x3c00_u16.to_ne_bytes());
        check("out", &bytes, &map(2, &[2..4]), Fill::Pattern(&pattern)).expect("aligned");
        bytes[5] = pattern[0];
        assert!(check("out", &bytes, &map(2, &[2..4]), Fill::Pattern(&pattern)).is_err());
    }

    #[test]
    fn a_complement_fill_accepts_all_ones_codes_and_rejects_unwritten_ones() {
        // Packed E2M1: 0xff is two -6 codes, a valid output.
        let expected = [0xff, 0x12, 0x00, POISON];
        let written = map(1, &[0..3]);
        let fill = Fill::Complement {
            pattern: &[POISON],
            expected: &expected,
        };
        assert!((0..3).all(|offset| fill.byte(offset, true) != expected[offset]));
        check("codes", &expected, &written, fill).expect("correct codes pass");
        let unwritten = [!0xff_u8, 0x12, 0x00, POISON];
        let error = check("codes", &unwritten, &written, fill).unwrap_err();
        assert!(error.contains("element 0"), "{error}");
        // A plain poison fill cannot tell a correct 0xff code from no write.
        assert!(check("codes", &expected, &written, Fill::Pattern(&[POISON])).is_err());
    }

    #[test]
    fn maps_sort_join_and_refuse_bad_ranges() {
        assert_eq!(map(2, &[4..6, 0..2, 2..4]).ranges(), &[0..6]);
        assert!(WriteMap::new(2, [1..4]).is_err());
        assert!(WriteMap::new(4, [0..6]).is_err());
        assert!(WriteMap::new(2, [0..4, 2..6]).is_err());
        assert!(WriteMap::new(0, [0..4]).is_err());
        let rows = WriteMap::of(DType::F32).rows(2, [3, 1]).expect("rows");
        assert_eq!(rows.ranges(), &[8..16, 24..32]);
        assert_eq!(rows.written_bytes(), 16);
        assert!(check("out", &[0; 4], &map(2, &[0..6]), Fill::Pattern(&[POISON])).is_err());
    }

    #[test]
    fn state_bytes_outside_written_ranges_keep_their_initial_image() {
        let initial = [1, 2, POISON, POISON, 5, 6, 7, 8];
        let write_only = map(2, &[2..4]);
        let mut after = initial;
        after[2..4].copy_from_slice(&[9, 9]);
        after[6] = 0;
        check_state("state", &after, &initial, &write_only, &[6..7]).expect("updated byte");
        assert!(check_state("state", &after, &initial, &write_only, &[]).is_err());
        let unwritten = [1, 2, POISON, POISON, 5, 6, 0, 8];
        assert!(check_state("state", &unwritten, &initial, &write_only, &[6..7]).is_err());
    }

    #[test]
    fn receipts_name_only_phases_that_ran_poisoned() {
        let mut evidence = Evidence::new(["out", "workspace"]);
        let mut receipt = BTreeMap::new();
        evidence.record(&mut receipt);
        assert!(!receipt.contains_key("graph_bitwise"));
        assert_eq!(receipt["poisoned"], json!(["out", "workspace"]));
        evidence.passed(Phase::Graph, false);
        evidence.passed(Phase::Pdl, false);
        let mut receipt = BTreeMap::new();
        evidence.record(&mut receipt);
        assert_eq!(receipt.get("graph_bitwise"), Some(&json!(true)));
        assert!(!receipt.contains_key("pdl_bitwise"));
        assert!(!receipt.contains_key("deterministic_bitwise"));
        evidence.passed(Phase::Pdl, true);
        let mut distinct = BTreeMap::new();
        evidence.record(&mut distinct);
        assert_eq!(distinct.get("pdl_bitwise"), Some(&json!(true)));
        assert!(evidence.has(Phase::Graph) && !evidence.has(Phase::Relaunch));
    }

    #[test]
    fn live_counts_cover_the_bucket_without_repeats() {
        assert_eq!(live_counts(32), vec![32, 0, 1, 16, 31]);
        assert_eq!(live_counts(2), vec![2, 0, 1]);
        assert_eq!(live_counts(1), vec![1, 0]);
    }

    #[test]
    fn a_rewrite_is_one_pdl_operand_of_flags_zero() {
        let descriptor = rewrite_descriptor(0x1000, 0x2000, 64).unwrap();
        assert_eq!(descriptor.len(), 24);
        assert_eq!(descriptor[..8], 0x1000_u64.to_ne_bytes());
        assert_eq!(descriptor[8..16], 0x2000_u64.to_ne_bytes());
        assert_eq!(descriptor[16..20], 16_u32.to_ne_bytes());
        assert_eq!(descriptor[20..], 0_u32.to_ne_bytes());
        assert!(rewrite_descriptor(0x1000, 0x2000, 6).is_err());
    }

    #[test]
    fn sorted_map_rejects_missing_padding_and_guard_writes() {
        let map = sorted_map(&[1, 0, 2], 4, 4).unwrap();
        let mut bytes = vec![0x80; 40];
        for word in bytes[4..36].chunks_exact_mut(4) {
            word.copy_from_slice(&(-1_i32).to_ne_bytes());
        }
        assert!(check("sorted", &bytes, &map, Fill::Pattern(&[0x80])).is_ok());
        bytes[32..36].fill(0x80);
        assert!(check("missing padding", &bytes, &map, Fill::Pattern(&[0x80])).is_err());
        bytes[32..36].copy_from_slice(&(-1_i32).to_ne_bytes());
        bytes[0] = 0;
        assert!(check("guard", &bytes, &map, Fill::Pattern(&[0x80])).is_err());
    }

    #[test]
    fn routed_map_rejects_missing_local_and_written_remote_rows() {
        let map = routed_map(&[0, 3, 1], 0..2, 2).unwrap();
        let mut bytes = vec![POISON; 12];
        bytes[..4].fill(0);
        bytes[8..].fill(0);
        assert!(check("routes", &bytes, &map, Fill::Pattern(&[POISON])).is_ok());
        bytes[8..10].fill(POISON);
        assert!(check("missing", &bytes, &map, Fill::Pattern(&[POISON])).is_err());
        bytes[8..10].fill(0);
        bytes[4] = 0;
        assert!(check("remote", &bytes, &map, Fill::Pattern(&[POISON])).is_err());
    }

    #[test]
    fn failures_keep_later_cases_and_still_fail() {
        let mut failures = Failures::default();
        failures.check("first", Err("missing output".into()));
        failures.check("second", Ok(()));
        failures.check("third", Err("guard changed".into()));
        let error = failures.finish().unwrap_err();
        assert!(error.contains("first") && error.contains("third"));
    }

    #[test]
    fn typed_maps_take_their_element_size_from_the_dtype() {
        let bf16 = WriteMap::of(DType::Bf16).prefix(4).unwrap();
        assert_eq!(bf16, map(2, &[0..8]));
        assert_eq!(
            WriteMap::of(DType::I32).rows(3, [2, 0]).unwrap(),
            map(4, &[0..12, 24..36])
        );
        assert_eq!(
            WriteMap::of(DType::F32)
                .ranges([4..6, 0..2])
                .unwrap()
                .ranges(),
            &[0..8, 16..24]
        );
        assert_eq!(
            WriteMap::of(DType::Fp8E4M3).none().unwrap().element_bytes(),
            1
        );
        // BF16 values whose low byte is the poison byte are written; a map
        // of the same bytes in 1-byte elements, the old miscount, calls
        // them unwritten.
        let image = [0x3eff_u16, 0xff3e, 0x3eff, 0xff3e]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        check("bf16", &image, &bf16, Fill::Pattern(&[POISON])).expect("all written");
        let bytewise = WriteMap::new(1, bf16.ranges().to_vec()).unwrap();
        assert!(check("bytes", &image, &bytewise, Fill::Pattern(&[POISON])).is_err());
        // Byte-placed ranges must still cover whole elements.
        assert!(WriteMap::of(DType::Bf16).byte_ranges([2..6]).is_ok());
        assert!(WriteMap::of(DType::Bf16).byte_ranges([1..3]).is_err());
        assert_eq!(WriteMap::of(DType::Bf16).byte_prefix(8).unwrap(), bf16);
        assert!(WriteMap::of(DType::F32).byte_prefix(6).is_err());
        assert!(WriteMap::of(DType::Fp4E2M1).prefix(2).is_err());
        let pairs = WriteMap::of_records::<8>().rows(2, [1]).unwrap();
        assert_eq!((pairs.element_bytes(), pairs.ranges()), (8, &[16..32][..]));
    }
}
