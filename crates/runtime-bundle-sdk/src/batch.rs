use std::error::Error;
use std::fmt;

/// The fixed prefix of one V1 batch request.
pub const BATCH_REQUEST_MAGIC_V1: &[u8; 12] = b"chopbatch-v1";

/// The fixed prefix of one V2 batch request.
pub const BATCH_REQUEST_MAGIC_V2: &[u8; 12] = b"chopbatch-v2";
pub const BATCH_REQUEST_MAGIC_V3: &[u8; 12] = b"chopbatch-v3";

/// V4 wraps a V3 payload and one state slot ID per ordered item.
pub const BATCH_REQUEST_MAGIC_V4: &[u8; 12] = b"chopbatch-v4";

/// V5 wraps a V3 or V4 record and each item's explicit 3-axis positions.
pub const BATCH_REQUEST_MAGIC_V5: &[u8; 12] = b"chopbatch-v5";

/// The wire code for a scheduled prefill.
pub const BATCH_PHASE_PREFILL_V1: u8 = 0;

/// The wire code for a scheduled decode.
pub const BATCH_PHASE_DECODE_V1: u8 = 1;
pub const BATCH_PHASE_VERIFY_V3: u8 = 2;
pub const BATCH_ITEM_FLAG_BUNDLE_DRAFTS_V3: u8 = 4;
/// A V3 prefill item that ends its prompt: after the step commits, the host
/// captures a prefix anchor at the last multiple of the bundle's prefill
/// granule strictly inside the item's rows, so the bundle keeps the item's
/// state as it stands there (`BundleBatchLimitsV1::anchor_items`).
pub const BATCH_ITEM_FLAG_ANCHOR_V3: u8 = 8;

/// The largest repeated field the shared codec will allocate.
pub const MAXIMUM_BATCH_WIRE_ELEMENTS_V1: u32 = 1 << 20;

/// A bounds or shape refusal while reading a V1 batch record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchReadErrorV1 {
    Truncated {
        surface: &'static str,
        needed: usize,
        remaining: usize,
    },
    Magic {
        surface: &'static str,
    },
    UnknownFlag {
        surface: &'static str,
        actual: u8,
    },
    UnknownPhase {
        actual: u8,
    },
    UnexpectedCount {
        surface: &'static str,
        expected: u64,
        actual: u32,
    },
    CountOutOfRange {
        surface: &'static str,
        actual: u64,
    },
    TrailingBytes {
        surface: &'static str,
        remaining: usize,
    },
    InvalidVerify,
    InvalidAnchor,
    InvalidMedia,
    InvalidStateSlots,
    InvalidPositionAxes,
}

impl fmt::Display for BatchReadErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated {
                surface,
                needed,
                remaining,
            } => write!(
                formatter,
                "{surface} needs {needed} more bytes but only {remaining} remain"
            ),
            Self::Magic { surface } => write!(formatter, "{surface} has the wrong magic"),
            Self::UnknownFlag { surface, actual } => {
                write!(formatter, "{surface} flag {actual} is not zero or one")
            }
            Self::UnknownPhase { actual } => {
                write!(
                    formatter,
                    "batch item phase {actual} is not prefill or decode"
                )
            }
            Self::UnexpectedCount {
                surface,
                expected,
                actual,
            } => write!(
                formatter,
                "{surface} count {actual} does not match the required count {expected}"
            ),
            Self::CountOutOfRange { surface, actual } => {
                write!(
                    formatter,
                    "{surface} count {actual} does not fit this process"
                )
            }
            Self::TrailingBytes { surface, remaining } => {
                write!(formatter, "{surface} has {remaining} trailing bytes")
            }
            Self::InvalidVerify => formatter.write_str("invalid verify item or append"),
            Self::InvalidAnchor => {
                formatter.write_str("an anchor marks a V3 prefill item that emits logits")
            }
            Self::InvalidMedia => formatter.write_str("invalid V3 media section"),
            Self::InvalidStateSlots => formatter.write_str("state slots must be unique per item"),
            Self::InvalidPositionAxes => {
                formatter.write_str("position axes must enclose V3 or V4 and cover whole items")
            }
        }
    }
}

impl Error for BatchReadErrorV1 {}

/// A bounds-checked cursor over one current V1 batch record.
#[derive(Clone, Debug)]
pub struct BatchReaderV1<'a> {
    surface: &'static str,
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> BatchReaderV1<'a> {
    #[must_use]
    pub const fn new(surface: &'static str, bytes: &'a [u8]) -> Self {
        Self {
            surface,
            bytes,
            offset: 0,
        }
    }

    #[must_use]
    pub const fn position(&self) -> usize {
        self.offset
    }

    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    pub fn take(&mut self, length: usize) -> Result<&'a [u8], BatchReadErrorV1> {
        let remaining = self.remaining();
        let end = self
            .offset
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(BatchReadErrorV1::Truncated {
                surface: self.surface,
                needed: length,
                remaining,
            })?;
        let read = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(read)
    }

    pub fn expect_magic(&mut self, magic: &[u8]) -> Result<(), BatchReadErrorV1> {
        if self.take(magic.len())? == magic {
            Ok(())
        } else {
            Err(BatchReadErrorV1::Magic {
                surface: self.surface,
            })
        }
    }

    pub fn u8(&mut self) -> Result<u8, BatchReadErrorV1> {
        Ok(self.take(1)?[0])
    }

    pub fn flag(&mut self, surface: &'static str) -> Result<bool, BatchReadErrorV1> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            actual => Err(BatchReadErrorV1::UnknownFlag { surface, actual }),
        }
    }

    pub fn u32(&mut self) -> Result<u32, BatchReadErrorV1> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("the reader returned four bytes"),
        ))
    }

    pub fn u64(&mut self) -> Result<u64, BatchReadErrorV1> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("the reader returned eight bytes"),
        ))
    }

    pub fn exact_count(
        &mut self,
        surface: &'static str,
        expected: u64,
    ) -> Result<usize, BatchReadErrorV1> {
        let actual = self.u32()?;
        if u64::from(actual) != expected {
            return Err(BatchReadErrorV1::UnexpectedCount {
                surface,
                expected,
                actual,
            });
        }
        usize::try_from(expected).map_err(|_| BatchReadErrorV1::CountOutOfRange {
            surface,
            actual: expected,
        })
    }

    /// Reads one bounded collection length before its allocation occurs.
    pub fn count(&mut self, surface: &'static str) -> Result<usize, BatchReadErrorV1> {
        let actual = self.u32()?;
        if actual > MAXIMUM_BATCH_WIRE_ELEMENTS_V1 {
            return Err(BatchReadErrorV1::CountOutOfRange {
                surface,
                actual: u64::from(actual),
            });
        }
        usize::try_from(actual).map_err(|_| BatchReadErrorV1::CountOutOfRange {
            surface,
            actual: u64::from(actual),
        })
    }

    /// Reads the byte length of an enclosed record. Media payloads may exceed
    /// the element bound, and `take` checks the length against the input
    /// before anything is copied.
    pub fn enclosed_length(&mut self) -> Result<usize, BatchReadErrorV1> {
        let actual = self.u32()?;
        usize::try_from(actual).map_err(|_| BatchReadErrorV1::CountOutOfRange {
            surface: "enclosed batch bytes",
            actual: u64::from(actual),
        })
    }

    pub fn finish(self) -> Result<(), BatchReadErrorV1> {
        let remaining = self.remaining();
        if remaining == 0 {
            Ok(())
        } else {
            Err(BatchReadErrorV1::TrailingBytes {
                surface: self.surface,
                remaining,
            })
        }
    }
}

/// The batch record revision selected by a bundle capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchWireVersionV1 {
    V1,
    V2,
    V3,
}

impl BatchWireVersionV1 {
    #[must_use]
    pub const fn magic(self) -> &'static [u8; 12] {
        match self {
            Self::V1 => BATCH_REQUEST_MAGIC_V1,
            Self::V2 => BATCH_REQUEST_MAGIC_V2,
            Self::V3 => BATCH_REQUEST_MAGIC_V3,
        }
    }
}

/// One model-neutral scheduled item in a decoded batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchItemV1 {
    request_id: u64,
    phase: u8,
    phase_tokens: u32,
    expects_logits: bool,
    prompt_rows: bool,
    bundle_drafts: bool,
    anchor: bool,
    input_tokens: Vec<u32>,
    position_axes: Vec<[i32; 3]>,
}

impl BatchItemV1 {
    #[must_use]
    pub fn new(
        request_id: u64,
        decode: bool,
        phase_tokens: u32,
        expects_logits: bool,
        input_tokens: Vec<u32>,
    ) -> Self {
        Self {
            request_id,
            phase: if decode {
                BATCH_PHASE_DECODE_V1
            } else {
                BATCH_PHASE_PREFILL_V1
            },
            phase_tokens,
            expects_logits,
            prompt_rows: false,
            bundle_drafts: false,
            anchor: false,
            input_tokens,
            position_axes: Vec::new(),
        }
    }

    #[must_use]
    pub fn verify(request_id: u64, rows: u32, bundle_drafts: bool, input_tokens: Vec<u32>) -> Self {
        Self {
            request_id,
            phase: BATCH_PHASE_VERIFY_V3,
            phase_tokens: rows,
            expects_logits: true,
            prompt_rows: false,
            bundle_drafts,
            anchor: false,
            input_tokens,
            position_axes: Vec::new(),
        }
    }

    /// Explicit (T, H, W) positions, one per phase row; empty keeps the
    /// implied 1D positions. Any nonempty set makes the record V5.
    #[must_use]
    pub fn with_position_axes(mut self, axes: Vec<[i32; 3]>) -> Self {
        self.position_axes = axes;
        self
    }

    #[must_use]
    pub fn position_axes(&self) -> &[[i32; 3]] {
        &self.position_axes
    }

    #[must_use]
    pub fn with_prompt_rows(mut self, enabled: bool) -> Self {
        self.prompt_rows = enabled;
        self
    }

    #[must_use]
    pub const fn prompt_rows(&self) -> bool {
        self.prompt_rows
    }

    /// Marks a prompt-ending prefill item whose state the bundle keeps at
    /// the last granule multiple inside it ([`BATCH_ITEM_FLAG_ANCHOR_V3`]).
    #[must_use]
    pub fn with_anchor(mut self, enabled: bool) -> Self {
        self.anchor = enabled;
        self
    }

    #[must_use]
    pub const fn anchor(&self) -> bool {
        self.anchor
    }

    #[must_use]
    pub const fn request_id(&self) -> u64 {
        self.request_id
    }

    #[must_use]
    pub const fn is_decode(&self) -> bool {
        self.phase == BATCH_PHASE_DECODE_V1
    }

    #[must_use]
    pub const fn is_verify(&self) -> bool {
        self.phase == BATCH_PHASE_VERIFY_V3
    }

    #[must_use]
    pub const fn bundle_drafts(&self) -> bool {
        self.bundle_drafts
    }

    #[must_use]
    pub const fn phase_tokens(&self) -> u32 {
        self.phase_tokens
    }

    #[must_use]
    pub const fn expects_logits(&self) -> bool {
        self.expects_logits
    }

    #[must_use]
    pub fn input_tokens(&self) -> &[u32] {
        &self.input_tokens
    }
}

/// The copy-on-write work added to one V2 append record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchAppendCopyV2 {
    source_page: u64,
    destination_page: u64,
    initialized_tokens: u64,
}

impl BatchAppendCopyV2 {
    #[must_use]
    pub const fn new(source_page: u64, destination_page: u64, initialized_tokens: u64) -> Self {
        Self {
            source_page,
            destination_page,
            initialized_tokens,
        }
    }

    #[must_use]
    pub const fn source_page(self) -> u64 {
        self.source_page
    }

    #[must_use]
    pub const fn destination_page(self) -> u64 {
        self.destination_page
    }

    #[must_use]
    pub const fn initialized_tokens(self) -> u64 {
        self.initialized_tokens
    }
}

/// One model-neutral prepared KV append.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchAppendV1 {
    previous_tokens: u64,
    additional_tokens: u64,
    new_tokens: u64,
    page_table: Vec<u64>,
    copy: Option<BatchAppendCopyV2>,
}

impl BatchAppendV1 {
    #[must_use]
    pub fn new(
        previous_tokens: u64,
        additional_tokens: u64,
        new_tokens: u64,
        page_table: Vec<u64>,
        copy: Option<BatchAppendCopyV2>,
    ) -> Self {
        Self {
            previous_tokens,
            additional_tokens,
            new_tokens,
            page_table,
            copy,
        }
    }

    #[must_use]
    pub const fn previous_tokens(&self) -> u64 {
        self.previous_tokens
    }

    #[must_use]
    pub const fn additional_tokens(&self) -> u64 {
        self.additional_tokens
    }

    #[must_use]
    pub const fn new_tokens(&self) -> u64 {
        self.new_tokens
    }

    #[must_use]
    pub fn page_table(&self) -> &[u64] {
        &self.page_table
    }

    #[must_use]
    pub const fn copy(&self) -> Option<BatchAppendCopyV2> {
        self.copy
    }
}

/// One media encode record in the shared V3 trailing section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchMediaEncodeV3 {
    pub encoder: u32,
    pub geometry: [u32; 4],
    pub output_rows: u32,
    pub pages: Vec<u64>,
    pub payload_offset: u64,
    pub payload_bytes: u64,
}

/// One item-row injection from the media arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchMediaInjectV3 {
    pub item_index: u32,
    pub item_row: u32,
    pub rows: u32,
    pub arena_row: u64,
}

/// Media records remain opaque to the speculation codec. MM-6 owns their semantics.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchMediaSectionV3 {
    pub encodes: Vec<BatchMediaEncodeV3>,
    pub injects: Vec<BatchMediaInjectV3>,
    pub payload: Vec<u8>,
}

impl BatchMediaSectionV3 {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.encodes.is_empty() && self.injects.is_empty() && self.payload.is_empty()
    }
}

/// One decoded model-neutral batch request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchRequestV1 {
    version: BatchWireVersionV1,
    items: Vec<BatchItemV1>,
    appends: Vec<BatchAppendV1>,
    media: BatchMediaSectionV3,
    state_slots: Option<Vec<u64>>,
}

impl BatchRequestV1 {
    #[must_use]
    pub fn new(
        version: BatchWireVersionV1,
        items: Vec<BatchItemV1>,
        appends: Vec<BatchAppendV1>,
    ) -> Self {
        Self {
            version,
            items,
            appends,
            media: BatchMediaSectionV3::default(),
            state_slots: None,
        }
    }

    /// One host-owned slot ID per item; the enclosed payload remains V3.
    #[must_use]
    pub fn with_state_slots(mut self, slots: Vec<u64>) -> Self {
        self.state_slots = Some(slots);
        self
    }

    #[must_use]
    pub fn state_slots(&self) -> Option<&[u64]> {
        self.state_slots.as_deref()
    }

    /// Whether any item carries explicit positions, which V5 requires.
    #[must_use]
    pub fn has_position_axes(&self) -> bool {
        self.items.iter().any(|item| !item.position_axes.is_empty())
    }

    #[must_use]
    pub fn with_media(mut self, media: BatchMediaSectionV3) -> Self {
        self.media = media;
        self
    }

    #[must_use]
    pub const fn media(&self) -> &BatchMediaSectionV3 {
        &self.media
    }

    #[must_use]
    pub const fn version(&self) -> BatchWireVersionV1 {
        self.version
    }

    #[must_use]
    pub fn items(&self) -> &[BatchItemV1] {
        &self.items
    }

    #[must_use]
    pub fn appends(&self) -> &[BatchAppendV1] {
        &self.appends
    }

    /// Decodes either supported batch record revision.
    pub fn decode(bytes: &[u8]) -> Result<Self, BatchReadErrorV1> {
        BatchCodecV1::decode(bytes)
    }

    /// Encodes this request using its selected batch record revision.
    pub fn encode(&self) -> Result<Vec<u8>, BatchWriteErrorV1> {
        BatchCodecV1::encode(self)
    }
}

/// A shape refusal while writing a batch record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchWriteErrorV1 {
    CountOutOfRange {
        surface: &'static str,
        actual: usize,
    },
    CopyRequiresV2,
    PromptRowsRequireV2,
    VerifyRequiresV3,
    MediaRequiresV3,
    InvalidVerify,
    InvalidAnchor,
    InvalidMedia,
    InvalidStateSlots,
    InvalidPositionAxes,
}

impl fmt::Display for BatchWriteErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountOutOfRange { surface, actual } => {
                write!(formatter, "{surface} count {actual} does not fit the wire")
            }
            Self::CopyRequiresV2 => {
                formatter.write_str("a copy-on-write append requires batch wire V2")
            }
            Self::PromptRowsRequireV2 => formatter.write_str("prompt rows require batch wire V2"),
            Self::VerifyRequiresV3 => formatter.write_str("verify items require batch wire V3"),
            Self::MediaRequiresV3 => formatter.write_str("media records require batch wire V3"),
            Self::InvalidVerify => formatter.write_str("invalid verify item or append"),
            Self::InvalidAnchor => {
                formatter.write_str("an anchor marks a V3 prefill item that emits logits")
            }
            Self::InvalidMedia => formatter.write_str("invalid V3 media section"),
            Self::InvalidStateSlots => {
                formatter.write_str("state slots require V3 and one unique slot per item")
            }
            Self::InvalidPositionAxes => formatter
                .write_str("position axes require V3 and one position per phase row of each item"),
        }
    }
}

impl Error for BatchWriteErrorV1 {}

/// The shared bundle-side codec for V1, V2 and capability-gated V3 records.
pub struct BatchCodecV1;

impl BatchCodecV1 {
    /// Decodes V1 bytes with the original field order and no append extension.
    pub fn decode_v1(bytes: &[u8]) -> Result<BatchRequestV1, BatchReadErrorV1> {
        Self::decode_version(bytes, BatchWireVersionV1::V1)
    }

    /// Decodes either supported version from its fixed magic.
    pub fn decode(bytes: &[u8]) -> Result<BatchRequestV1, BatchReadErrorV1> {
        if bytes.starts_with(BATCH_REQUEST_MAGIC_V5) {
            return decode_position_axes_v5(bytes);
        }
        if bytes.starts_with(BATCH_REQUEST_MAGIC_V4) {
            let mut reader = BatchReaderV1::new("state batch", bytes);
            reader.expect_magic(BATCH_REQUEST_MAGIC_V4)?;
            let length = reader.enclosed_length()?;
            let mut request = Self::decode_version(reader.take(length)?, BatchWireVersionV1::V3)?;
            let count = reader.count("state slots")?;
            if count != request.items.len() {
                return Err(BatchReadErrorV1::UnexpectedCount {
                    surface: "state slots",
                    expected: request.items.len() as u64,
                    actual: count as u32,
                });
            }
            let mut slots = Vec::with_capacity(count);
            let mut unique = std::collections::BTreeSet::new();
            for _ in 0..count {
                let slot = reader.u64()?;
                if !unique.insert(slot) {
                    return Err(BatchReadErrorV1::InvalidStateSlots);
                }
                slots.push(slot);
            }
            reader.finish()?;
            request.state_slots = Some(slots);
            return Ok(request);
        }
        let version = if bytes.starts_with(BATCH_REQUEST_MAGIC_V1) {
            BatchWireVersionV1::V1
        } else if bytes.starts_with(BATCH_REQUEST_MAGIC_V2) {
            BatchWireVersionV1::V2
        } else if bytes.starts_with(BATCH_REQUEST_MAGIC_V3) {
            BatchWireVersionV1::V3
        } else {
            return Err(BatchReadErrorV1::Magic {
                surface: "batch request",
            });
        };
        Self::decode_version(bytes, version)
    }

    /// Encodes the selected version without host or model dependencies.
    pub fn encode(request: &BatchRequestV1) -> Result<Vec<u8>, BatchWriteErrorV1> {
        if request.has_position_axes() {
            return encode_position_axes_v5(request);
        }
        if let Some(slots) = &request.state_slots {
            if request.version != BatchWireVersionV1::V3
                || slots.len() != request.items.len()
                || slots
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != slots.len()
            {
                return Err(BatchWriteErrorV1::InvalidStateSlots);
            }
            let mut inner = request.clone();
            inner.state_slots = None;
            let payload = Self::encode(&inner)?;
            let mut bytes = BATCH_REQUEST_MAGIC_V4.to_vec();
            write_count(&mut bytes, "enclosed batch bytes", payload.len())?;
            bytes.extend_from_slice(&payload);
            write_count(&mut bytes, "state slots", slots.len())?;
            for slot in slots {
                bytes.extend_from_slice(&slot.to_le_bytes());
            }
            return Ok(bytes);
        }
        if request.version != BatchWireVersionV1::V3 && !request.media.is_empty() {
            return Err(BatchWriteErrorV1::MediaRequiresV3);
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(request.version.magic());
        write_count(&mut bytes, "batch item", request.items.len())?;
        for (index, item) in request.items.iter().enumerate() {
            if item.is_verify() {
                if request.version != BatchWireVersionV1::V3 {
                    return Err(BatchWriteErrorV1::VerifyRequiresV3);
                }
                if !valid_verify(item, request.appends.get(index)) {
                    return Err(BatchWriteErrorV1::InvalidVerify);
                }
            } else if item.bundle_drafts {
                return Err(BatchWriteErrorV1::InvalidVerify);
            }
            if item.anchor && !valid_anchor(item, request.version) {
                return Err(BatchWriteErrorV1::InvalidAnchor);
            }
            bytes.extend_from_slice(&item.request_id.to_le_bytes());
            bytes.push(item.phase);
            bytes.extend_from_slice(&item.phase_tokens.to_le_bytes());
            if request.version == BatchWireVersionV1::V1 && item.prompt_rows {
                return Err(BatchWriteErrorV1::PromptRowsRequireV2);
            }
            bytes.push(
                u8::from(item.expects_logits)
                    | (u8::from(item.prompt_rows) << 1)
                    | (u8::from(item.bundle_drafts) << 2)
                    | (u8::from(item.anchor) << 3),
            );
            write_count(&mut bytes, "batch item token", item.input_tokens.len())?;
            for token in &item.input_tokens {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
        }
        write_count(&mut bytes, "batch append", request.appends.len())?;
        for append in &request.appends {
            bytes.extend_from_slice(&append.previous_tokens.to_le_bytes());
            bytes.extend_from_slice(&append.additional_tokens.to_le_bytes());
            bytes.extend_from_slice(&append.new_tokens.to_le_bytes());
            write_count(&mut bytes, "batch page table", append.page_table.len())?;
            for page in &append.page_table {
                bytes.extend_from_slice(&page.to_le_bytes());
            }
            match request.version {
                BatchWireVersionV1::V1 if append.copy.is_some() => {
                    return Err(BatchWriteErrorV1::CopyRequiresV2);
                }
                BatchWireVersionV1::V1 => {}
                BatchWireVersionV1::V2 | BatchWireVersionV1::V3 => {
                    encode_copy_v2(&mut bytes, append.copy)
                }
            }
        }
        if request.version == BatchWireVersionV1::V3 {
            encode_media_v3(&mut bytes, &request.media)?;
        }
        Ok(bytes)
    }

    fn decode_version(
        bytes: &[u8],
        version: BatchWireVersionV1,
    ) -> Result<BatchRequestV1, BatchReadErrorV1> {
        let mut reader = BatchReaderV1::new("batch request", bytes);
        reader.expect_magic(version.magic())?;
        let item_count = reader.count("batch item")?;
        let mut items = Vec::with_capacity(item_count);
        for _ in 0..item_count {
            let request_id = reader.u64()?;
            let phase = match reader.u8()? {
                BATCH_PHASE_PREFILL_V1 => BATCH_PHASE_PREFILL_V1,
                BATCH_PHASE_DECODE_V1 => BATCH_PHASE_DECODE_V1,
                BATCH_PHASE_VERIFY_V3 if version == BatchWireVersionV1::V3 => BATCH_PHASE_VERIFY_V3,
                actual => return Err(BatchReadErrorV1::UnknownPhase { actual }),
            };
            let phase_tokens = reader.u32()?;
            let flags = reader.u8()?;
            if flags
                > if version == BatchWireVersionV1::V1 {
                    1
                } else if version == BatchWireVersionV1::V2 {
                    3
                } else {
                    15
                }
            {
                return Err(BatchReadErrorV1::UnknownFlag {
                    surface: "batch item logits",
                    actual: flags,
                });
            }
            let expects_logits = flags & 1 != 0;
            let prompt_rows = flags & 2 != 0;
            let bundle_drafts = flags & BATCH_ITEM_FLAG_BUNDLE_DRAFTS_V3 != 0;
            let anchor = flags & BATCH_ITEM_FLAG_ANCHOR_V3 != 0;
            let token_count = reader.count("batch item token")?;
            let mut input_tokens = Vec::with_capacity(token_count);
            for _ in 0..token_count {
                input_tokens.push(reader.u32()?);
            }
            items.push(BatchItemV1 {
                request_id,
                phase,
                phase_tokens,
                expects_logits,
                prompt_rows,
                bundle_drafts,
                anchor,
                input_tokens,
                position_axes: Vec::new(),
            });
        }
        let append_count = reader.count("batch append")?;
        let mut appends = Vec::with_capacity(append_count);
        for _ in 0..append_count {
            let previous_tokens = reader.u64()?;
            let additional_tokens = reader.u64()?;
            let new_tokens = reader.u64()?;
            let page_count = reader.count("batch page table")?;
            let mut page_table = Vec::with_capacity(page_count);
            for _ in 0..page_count {
                page_table.push(reader.u64()?);
            }
            let copy = match version {
                BatchWireVersionV1::V1 => None,
                BatchWireVersionV1::V2 | BatchWireVersionV1::V3 => decode_copy_v2(&mut reader)?,
            };
            appends.push(BatchAppendV1::new(
                previous_tokens,
                additional_tokens,
                new_tokens,
                page_table,
                copy,
            ));
        }
        if version == BatchWireVersionV1::V3 {
            for (index, item) in items.iter().enumerate() {
                if item.is_verify() {
                    if !valid_verify(item, appends.get(index)) {
                        return Err(BatchReadErrorV1::InvalidVerify);
                    }
                } else if item.bundle_drafts {
                    return Err(BatchReadErrorV1::InvalidVerify);
                }
                if item.anchor && !valid_anchor(item, version) {
                    return Err(BatchReadErrorV1::InvalidAnchor);
                }
            }
        }
        let media = if version == BatchWireVersionV1::V3 {
            decode_media_v3(&mut reader)?
        } else {
            BatchMediaSectionV3::default()
        };
        reader.finish()?;
        Ok(BatchRequestV1::new(version, items, appends).with_media(media))
    }
}

/// Writes the enclosed V3 or V4 record, then each item's row count (zero or
/// its phase rows) and its rows of (T, H, W) little-endian i32 positions.
fn encode_position_axes_v5(request: &BatchRequestV1) -> Result<Vec<u8>, BatchWriteErrorV1> {
    if request.version != BatchWireVersionV1::V3
        || request.items.iter().any(|item| {
            !item.position_axes.is_empty() && item.position_axes.len() != item.phase_tokens as usize
        })
    {
        return Err(BatchWriteErrorV1::InvalidPositionAxes);
    }
    let mut inner = request.clone();
    for item in &mut inner.items {
        item.position_axes.clear();
    }
    let payload = BatchCodecV1::encode(&inner)?;
    let mut bytes = BATCH_REQUEST_MAGIC_V5.to_vec();
    write_count(&mut bytes, "enclosed batch bytes", payload.len())?;
    bytes.extend_from_slice(&payload);
    write_count(&mut bytes, "position axes items", request.items.len())?;
    for item in &request.items {
        write_count(&mut bytes, "position axes rows", item.position_axes.len())?;
        for row in &item.position_axes {
            for axis in row {
                bytes.extend_from_slice(&axis.to_le_bytes());
            }
        }
    }
    Ok(bytes)
}

fn decode_position_axes_v5(bytes: &[u8]) -> Result<BatchRequestV1, BatchReadErrorV1> {
    let mut reader = BatchReaderV1::new("position batch", bytes);
    reader.expect_magic(BATCH_REQUEST_MAGIC_V5)?;
    let length = reader.enclosed_length()?;
    let enclosed = reader.take(length)?;
    if !enclosed.starts_with(BATCH_REQUEST_MAGIC_V3)
        && !enclosed.starts_with(BATCH_REQUEST_MAGIC_V4)
    {
        return Err(BatchReadErrorV1::InvalidPositionAxes);
    }
    let mut request = BatchCodecV1::decode(enclosed)?;
    reader.exact_count("position axes items", request.items.len() as u64)?;
    for item in &mut request.items {
        let rows = reader.count("position axes rows")?;
        if rows != 0 && rows != item.phase_tokens as usize {
            return Err(BatchReadErrorV1::InvalidPositionAxes);
        }
        let mut axes = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut row = [0; 3];
            for axis in &mut row {
                let bytes = reader.take(4)?;
                *axis =
                    i32::from_le_bytes(bytes.try_into().expect("the reader returned four bytes"));
            }
            axes.push(row);
        }
        item.position_axes = axes;
    }
    reader.finish()?;
    Ok(request)
}

fn write_count(
    bytes: &mut Vec<u8>,
    surface: &'static str,
    actual: usize,
) -> Result<(), BatchWriteErrorV1> {
    let count = u32::try_from(actual)
        .map_err(|_| BatchWriteErrorV1::CountOutOfRange { surface, actual })?;
    bytes.extend_from_slice(&count.to_le_bytes());
    Ok(())
}

fn valid_verify(item: &BatchItemV1, append: Option<&BatchAppendV1>) -> bool {
    let expected_tokens = if item.bundle_drafts {
        1
    } else {
        item.phase_tokens
    };
    item.phase_tokens >= 2
        && item.expects_logits
        && !item.prompt_rows
        && u32::try_from(item.input_tokens.len()) == Ok(expected_tokens)
        && append.is_some_and(|append| append.additional_tokens == u64::from(item.phase_tokens))
}

/// An anchor marks a V3 prefill item that ends its prompt (it emits the
/// prompt's logits row).
fn valid_anchor(item: &BatchItemV1, version: BatchWireVersionV1) -> bool {
    version == BatchWireVersionV1::V3 && item.phase == BATCH_PHASE_PREFILL_V1 && item.expects_logits
}

fn valid_media(media: &BatchMediaSectionV3) -> bool {
    let Ok(payload_len) = u64::try_from(media.payload.len()) else {
        return false;
    };
    media.encodes.len() <= MAXIMUM_BATCH_WIRE_ELEMENTS_V1 as usize
        && media.injects.len() <= MAXIMUM_BATCH_WIRE_ELEMENTS_V1 as usize
        && media.encodes.iter().all(|encode| {
            encode.pages.len() <= MAXIMUM_BATCH_WIRE_ELEMENTS_V1 as usize
                && encode.output_rows > 0
                && encode.payload_bytes > 0
                && encode
                    .payload_offset
                    .checked_add(encode.payload_bytes)
                    .is_some_and(|end| end <= payload_len)
        })
}

fn encode_media_v3(
    bytes: &mut Vec<u8>,
    media: &BatchMediaSectionV3,
) -> Result<(), BatchWriteErrorV1> {
    if !valid_media(media) {
        return Err(BatchWriteErrorV1::InvalidMedia);
    }
    write_count(bytes, "media encode", media.encodes.len())?;
    for encode in &media.encodes {
        bytes.extend_from_slice(&encode.encoder.to_le_bytes());
        for dimension in encode.geometry {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&encode.output_rows.to_le_bytes());
        write_count(bytes, "media page", encode.pages.len())?;
        for page in &encode.pages {
            bytes.extend_from_slice(&page.to_le_bytes());
        }
        bytes.extend_from_slice(&encode.payload_offset.to_le_bytes());
        bytes.extend_from_slice(&encode.payload_bytes.to_le_bytes());
    }
    write_count(bytes, "media inject", media.injects.len())?;
    for inject in &media.injects {
        bytes.extend_from_slice(&inject.item_index.to_le_bytes());
        bytes.extend_from_slice(&inject.item_row.to_le_bytes());
        bytes.extend_from_slice(&inject.rows.to_le_bytes());
        bytes.extend_from_slice(&inject.arena_row.to_le_bytes());
    }
    bytes.extend_from_slice(&(media.payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&media.payload);
    Ok(())
}

fn decode_media_v3(
    reader: &mut BatchReaderV1<'_>,
) -> Result<BatchMediaSectionV3, BatchReadErrorV1> {
    let encode_count = reader.count("media encode")?;
    let mut encodes = Vec::with_capacity(encode_count);
    for _ in 0..encode_count {
        let encoder = reader.u32()?;
        let mut geometry = [0; 4];
        for dimension in &mut geometry {
            *dimension = reader.u32()?;
        }
        let output_rows = reader.u32()?;
        let page_count = reader.count("media page")?;
        let mut pages = Vec::with_capacity(page_count);
        for _ in 0..page_count {
            pages.push(reader.u64()?);
        }
        encodes.push(BatchMediaEncodeV3 {
            encoder,
            geometry,
            output_rows,
            pages,
            payload_offset: reader.u64()?,
            payload_bytes: reader.u64()?,
        });
    }
    let inject_count = reader.count("media inject")?;
    let mut injects = Vec::with_capacity(inject_count);
    for _ in 0..inject_count {
        injects.push(BatchMediaInjectV3 {
            item_index: reader.u32()?,
            item_row: reader.u32()?,
            rows: reader.u32()?,
            arena_row: reader.u64()?,
        });
    }
    let payload_len = usize::try_from(reader.u64()?).map_err(|_| BatchReadErrorV1::InvalidMedia)?;
    let payload = reader.take(payload_len)?.to_vec();
    let media = BatchMediaSectionV3 {
        encodes,
        injects,
        payload,
    };
    if !valid_media(&media) {
        return Err(BatchReadErrorV1::InvalidMedia);
    }
    Ok(media)
}

fn encode_copy_v2(bytes: &mut Vec<u8>, copy: Option<BatchAppendCopyV2>) {
    bytes.push(u8::from(copy.is_some()));
    if let Some(copy) = copy {
        bytes.extend_from_slice(&copy.source_page.to_le_bytes());
        bytes.extend_from_slice(&copy.destination_page.to_le_bytes());
        bytes.extend_from_slice(&copy.initialized_tokens.to_le_bytes());
    }
}

fn decode_copy_v2(
    reader: &mut BatchReaderV1<'_>,
) -> Result<Option<BatchAppendCopyV2>, BatchReadErrorV1> {
    if !reader.flag("batch append copy")? {
        return Ok(None);
    }
    Ok(Some(BatchAppendCopyV2::new(
        reader.u64()?,
        reader.u64()?,
        reader.u64()?,
    )))
}

#[cfg(test)]
mod state_slot_tests {
    use super::*;

    #[test]
    fn state_wrapper_round_trips_and_refuses_missing_duplicate_or_truncated_slots() {
        let item = BatchItemV1::new(1, false, 1, true, vec![7]);
        let request = BatchRequestV1::new(BatchWireVersionV1::V3, vec![item], vec![])
            .with_state_slots(vec![9]);
        let encoded = request.encode().unwrap();
        assert_eq!(BatchCodecV1::decode(&encoded).unwrap(), request);
        for end in 0..encoded.len() {
            assert!(BatchCodecV1::decode(&encoded[..end]).is_err());
        }
        assert!(request.clone().with_state_slots(vec![]).encode().is_err());
        let mut duplicate = request.clone();
        duplicate.items.push(duplicate.items[0].clone());
        duplicate.state_slots = Some(vec![9, 9]);
        assert_eq!(
            duplicate.encode(),
            Err(BatchWriteErrorV1::InvalidStateSlots)
        );
        let mut trailing = encoded;
        trailing.push(0);
        assert!(BatchCodecV1::decode(&trailing).is_err());
    }
}

#[cfg(test)]
mod position_axes_tests {
    use super::*;

    fn request() -> BatchRequestV1 {
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![
                BatchItemV1::new(1, false, 2, true, vec![7, 8])
                    .with_position_axes(vec![[0, 0, 0], [1, 1, 2]]),
                BatchItemV1::new(2, true, 1, true, vec![9]),
                BatchItemV1::verify(3, 3, true, vec![4]).with_position_axes(vec![
                    [5, 5, 5],
                    [6, 6, 6],
                    [7, 7, 7],
                ]),
            ],
            vec![
                BatchAppendV1::new(0, 2, 2, vec![0], None),
                BatchAppendV1::new(4, 1, 5, vec![1], None),
                BatchAppendV1::new(8, 3, 11, vec![2], None),
            ],
        )
    }

    fn without_axes(request: &BatchRequestV1) -> BatchRequestV1 {
        let mut inner = request.clone();
        for item in &mut inner.items {
            item.position_axes.clear();
        }
        inner
    }

    #[test]
    fn v5_wraps_the_enclosed_record_and_one_row_count_per_item() {
        let request = request();
        let encoded = request.encode().unwrap();
        let enclosed = without_axes(&request).encode().unwrap();
        assert!(enclosed.starts_with(BATCH_REQUEST_MAGIC_V3));
        let mut expected = BATCH_REQUEST_MAGIC_V5.to_vec();
        expected.extend_from_slice(&(enclosed.len() as u32).to_le_bytes());
        expected.extend_from_slice(&enclosed);
        expected.extend_from_slice(&3_u32.to_le_bytes());
        for (rows, values) in [
            (2_u32, &[0, 0, 0, 1, 1, 2][..]),
            (0, &[][..]),
            (3, &[5, 5, 5, 6, 6, 6, 7, 7, 7][..]),
        ] {
            expected.extend_from_slice(&rows.to_le_bytes());
            for value in values {
                expected.extend_from_slice(&i32::to_le_bytes(*value));
            }
        }
        assert_eq!(encoded, expected);
        let decoded = BatchCodecV1::decode(&encoded).unwrap();
        assert_eq!(decoded, request);
        assert_eq!(decoded.items()[0].position_axes(), [[0, 0, 0], [1, 1, 2]]);
        assert!(decoded.items()[1].position_axes().is_empty());
        for end in 0..encoded.len() {
            assert!(BatchCodecV1::decode(&encoded[..end]).is_err());
        }
        let mut trailing = encoded;
        trailing.push(0);
        assert!(BatchCodecV1::decode(&trailing).is_err());
    }

    #[test]
    fn v5_encloses_state_slots_and_steps_without_axes_keep_their_bytes() {
        let request = request().with_state_slots(vec![3, 4, 5]);
        let encoded = request.encode().unwrap();
        let enclosed =
            &encoded[16..16 + u32::from_le_bytes(encoded[12..16].try_into().unwrap()) as usize];
        assert!(enclosed.starts_with(BATCH_REQUEST_MAGIC_V4));
        assert_eq!(BatchCodecV1::decode(&encoded).unwrap(), request);
        let plain = without_axes(&request);
        assert!(!plain.has_position_axes());
        assert!(plain.encode().unwrap().starts_with(BATCH_REQUEST_MAGIC_V4));
        let plain = without_axes(&self::request());
        assert!(plain.encode().unwrap().starts_with(BATCH_REQUEST_MAGIC_V3));
    }

    #[test]
    fn enclosed_records_may_carry_media_payloads_past_the_element_bound() {
        let payload = vec![7_u8; MAXIMUM_BATCH_WIRE_ELEMENTS_V1 as usize + 64];
        let request = request()
            .with_media(BatchMediaSectionV3 {
                encodes: vec![BatchMediaEncodeV3 {
                    encoder: 0,
                    geometry: [1, 4, 4, 0],
                    output_rows: 4,
                    pages: vec![0],
                    payload_offset: 0,
                    payload_bytes: payload.len() as u64,
                }],
                injects: vec![],
                payload,
            })
            .with_state_slots(vec![3, 4, 5]);
        let encoded = request.encode().unwrap();
        assert_eq!(BatchCodecV1::decode(&encoded).unwrap(), request);
    }

    #[test]
    fn v5_refuses_partial_rows_other_revisions_and_count_mismatches() {
        let mut short = request();
        short.items[0].position_axes.pop();
        assert_eq!(short.encode(), Err(BatchWriteErrorV1::InvalidPositionAxes));
        let mut v2 = request();
        v2.version = BatchWireVersionV1::V2;
        v2.items.truncate(2);
        v2.appends.truncate(2);
        assert_eq!(v2.encode(), Err(BatchWriteErrorV1::InvalidPositionAxes));

        let wrap = |enclosed: &[u8], tail: &[u8]| {
            let mut bytes = BATCH_REQUEST_MAGIC_V5.to_vec();
            bytes.extend_from_slice(&(enclosed.len() as u32).to_le_bytes());
            bytes.extend_from_slice(enclosed);
            bytes.extend_from_slice(tail);
            bytes
        };
        let enclosed = without_axes(&request()).encode().unwrap();
        let tail = |counts: &[u32]| {
            let mut tail = (counts.len() as u32).to_le_bytes().to_vec();
            for &rows in counts {
                tail.extend_from_slice(&rows.to_le_bytes());
                tail.extend(std::iter::repeat_n(0_u8, 12 * rows as usize));
            }
            tail
        };
        assert!(BatchCodecV1::decode(&wrap(&enclosed, &tail(&[2, 0, 3]))).is_ok());
        assert!(BatchCodecV1::decode(&wrap(&enclosed, &tail(&[0, 0, 0]))).is_ok());
        assert_eq!(
            BatchCodecV1::decode(&wrap(&enclosed, &tail(&[1, 0, 3]))),
            Err(BatchReadErrorV1::InvalidPositionAxes)
        );
        assert!(matches!(
            BatchCodecV1::decode(&wrap(&enclosed, &tail(&[2, 0]))),
            Err(BatchReadErrorV1::UnexpectedCount { .. })
        ));
        let mut v2 = without_axes(&request());
        v2.version = BatchWireVersionV1::V2;
        v2.items.truncate(2);
        v2.appends.truncate(2);
        assert_eq!(
            BatchCodecV1::decode(&wrap(&v2.encode().unwrap(), &tail(&[2, 0]))),
            Err(BatchReadErrorV1::InvalidPositionAxes)
        );
        let nested = request().encode().unwrap();
        assert_eq!(
            BatchCodecV1::decode(&wrap(&nested, &tail(&[2, 0, 3]))),
            Err(BatchReadErrorV1::InvalidPositionAxes)
        );
    }
}

#[cfg(test)]
mod historical_wire_rejection {
    use super::*;
    #[test]
    fn old_magic_is_rejected_in_outer_and_nested_records() {
        for (version, slots, axes) in [
            (BatchWireVersionV1::V1, false, false),
            (BatchWireVersionV1::V2, false, false),
            (BatchWireVersionV1::V3, false, false),
            (BatchWireVersionV1::V3, true, false),
            (BatchWireVersionV1::V3, false, true),
            (BatchWireVersionV1::V3, true, true),
        ] {
            let mut item = BatchItemV1::new(17, false, 2, true, vec![7, 9]);
            if axes {
                item = item.with_position_axes(vec![[0, 1, 2], [3, 4, 5]]);
            }
            let mut request = BatchRequestV1::new(
                version,
                vec![item],
                vec![BatchAppendV1::new(0, 2, 2, vec![13], None)],
            );
            if slots {
                request = request.with_state_slots(vec![3]);
            }
            let bytes = request.encode().unwrap();
            assert_eq!(BatchRequestV1::decode(&bytes).unwrap(), request);
            let mut positions = vec![0];
            if slots || axes {
                positions.push(16);
            }
            if slots && axes {
                positions.push(32);
            }
            for offset in positions {
                let mut old = bytes.clone();
                let mut magic = *b"\x72\x74\x78-batch-v1";
                magic[11] = bytes[offset + 11];
                old[offset..offset + 12].copy_from_slice(&magic);
                assert_eq!(old.len(), bytes.len());
                assert!(
                    BatchRequestV1::decode(&old).is_err(),
                    "legacy magic at {offset}"
                );
            }
        }
    }
}
