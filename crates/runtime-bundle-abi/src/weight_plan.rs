//! Direct weight loading: the checkpoint description a host gives a bundle,
//! and the weight plan the bundle answers with.
//!
//! A bundle that can load a checkpoint directly exports
//! `chop_bundle_weight_plan_v1`. The host describes the checkpoint (every
//! safetensors header, the small documents beside it and the package's
//! bundle metadata) and names one compiled image role, the rank count and one
//! rank. The bundle answers, for that image and rank, with the arena size and
//! alignment, the named views its create path opens, the authenticated
//! components it reads (for example a rank index), and one record per
//! placed piece: the checkpoint tensors it reads, a selection, the stored
//! shape, a transform from `kernel-core`'s closed set with its parameters,
//! and the arena offset it lands at. The host executes records with no model
//! code; the bytes are what the compiled rank image would hold. A bundle may
//! instead decline an image with an empty document; a direct load, which
//! reads no compiled image, then refuses the package.
//!
//! A planned image has no file to authenticate. Its identity for one rank is
//! the SHA-256 of [`WeightPlanV1::identity_document`], which the host serves
//! as the rank's `direct_identity` where it would serve a rank file's digest,
//! so a rank index the bundle plans can name the rank it describes. It covers
//! what determines the rank's bytes except their payload: the checkpoint's
//! header identity (names, lengths and headers, not tensor bytes), the
//! implementation that produces them ([`WeightPlanImplementationV1`]: the
//! host's shared transforms, and the bundle module when a record uses a
//! bundle-owned transform), and the plan without its components.
//!
//! A record whose transform identifier is at least
//! `kernel_core::weight_transform::BUNDLE_TRANSFORM_FIRST` is executed by the
//! bundle's optional `chop_bundle_weight_transform_v1` export instead.
//!
//! Both documents are little-endian byte strings: an 8-byte magic, a `u32`
//! layout ([`WEIGHT_PLAN_LAYOUT_V1`]), then fixed-width integers, `u32` list
//! counts and byte strings prefixed by a `u64` length. Readers refuse another
//! layout by name before reading a field, trailing bytes, and counts past the
//! stated bounds. The host also states the layout in the sink, which the
//! bundle checks before decoding the request. `include/chop_bundle_v1.h`
//! lists the field order.

use crate::{BundleByteViewV1, BundleMutByteViewV1};
pub use crate::{BundleWeightPlanEmitFnV1, BundleWeightPlanFnV1, BundleWeightTransformFnV1};
use std::ffi::c_void;
use std::ops::Range;

/// Capability a package states when its bundle exports the weight plan.
pub const WEIGHT_PLAN_CAPABILITY_V1: &str = "chop.weight-plan.v1";
/// The optional plan export, NUL-terminated for `dlsym`.
pub const WEIGHT_PLAN_SYMBOL_V1: &[u8] = b"chop_bundle_weight_plan_v1\0";
/// The optional bundle transform export, NUL-terminated for `dlsym`.
pub const WEIGHT_TRANSFORM_SYMBOL_V1: &[u8] = b"chop_bundle_weight_transform_v1\0";

const REQUEST_MAGIC: &[u8; 8] = b"CHOPWREQ";
const PLAN_MAGIC: &[u8; 8] = b"CHOPWPLN";
/// The layout of both documents, which the host also states in the sink.
/// Earlier layouts are refused: 1 before requests named an image role, 2
/// before they stated the implementation. The export and capability keep
/// their `v1` names, since no layout was released.
pub const WEIGHT_PLAN_LAYOUT_V1: u32 = 3;
/// Transform identifiers from here up belong to the bundle
/// (`kernel_core::weight_transform::BUNDLE_TRANSFORM_FIRST`).
const BUNDLE_TRANSFORM_FIRST: u32 = 0x8000_0000;

/// Longest name, role or view identifier either document carries.
pub const WEIGHT_PLAN_NAME_BYTES_MAX_V1: usize = 4096;
/// Most entries in any one list of either document.
pub const WEIGHT_PLAN_ENTRIES_MAX_V1: usize = 1 << 24;
/// Most dimensions of a stored shape.
pub const WEIGHT_PLAN_DIMENSIONS_MAX_V1: usize = 8;

/// Receives the encoded plan. The bundle calls `emit` exactly once on
/// success, with the whole document (empty to decline the image), or
/// `emit_error` at most once with UTF-8 refusal text before returning a
/// failure. The host copies the bytes before
/// either returns.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BundleWeightPlanSinkV1 {
    pub struct_size: u32,
    /// The document layout the host writes and reads,
    /// [`WEIGHT_PLAN_LAYOUT_V1`]; the bundle refuses another before decoding.
    pub layout: u32,
    pub context: *mut c_void,
    pub emit: Option<BundleWeightPlanEmitFnV1>,
    pub emit_error: Option<BundleWeightPlanEmitFnV1>,
}

/// One call of a bundle-owned transform: the record's transform identifier
/// and parameters, its whole source tensors in record order, and the output
/// bytes `[output_offset, output_offset + output.length)` of the record to
/// fill.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BundleWeightTransformCallV1 {
    pub struct_size: u32,
    pub transform: u32,
    pub params: BundleByteViewV1,
    pub sources: *const BundleByteViewV1,
    pub source_count: u64,
    pub output_offset: u64,
    pub output: BundleMutByteViewV1,
}

/// One safetensors file of the checkpoint.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CheckpointFileV1 {
    /// The file name within the checkpoint directory.
    pub name: String,
    pub byte_length: u64,
    /// Where tensor data starts: 8 plus the header length.
    pub data_offset: u64,
    /// The JSON header, exactly as stored.
    pub header: Vec<u8>,
}

/// A named byte string: a checkpoint document or a package component.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NamedBytesV1 {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// The code that turns a plan into bytes, as the host states it: its planned
/// identities cover it, so two builds that could produce different bytes for
/// one plan never share an identity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WeightPlanImplementationV1 {
    /// SHA-256 of the host's shared transform implementation: every source
    /// on its byte path (the host defines which; see
    /// `runtime_host::direct::shared_transform_implementation_v1`).
    pub shared_transforms: [u8; 32],
    /// SHA-256 of the bundle's native module, which executes bundle-owned
    /// transforms; zero when the host plans without one, and then the host
    /// refuses a plan with a bundle-owned transform.
    pub bundle_module: [u8; 32],
}

/// What the host tells the bundle about the checkpoint and the load.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WeightPlanRequestV1 {
    /// The compiled image role the host asks about, one of the package's
    /// images for the selected topology, or empty for the bundle's primary
    /// image when the package carries no rank images.
    pub image_role: String,
    pub ranks: u32,
    pub rank: u32,
    /// SHA-256 of the file names, lengths and headers (see
    /// `runtime_weights::checkpoint::header_identity`).
    pub checkpoint_identity: [u8; 32],
    /// The implementation the host executes the plan with.
    pub implementation: WeightPlanImplementationV1,
    /// Every safetensors file in the directory and its immediate
    /// subdirectories (named `dir/file`), sorted by name.
    pub files: Vec<CheckpointFileV1>,
    /// The JSON documents and other small files in the same places, sorted
    /// by name. The checkpoint identity does not cover them: a plan that
    /// stores their bytes checks them itself.
    pub documents: Vec<NamedBytesV1>,
    /// The package's bundle-metadata components (`chop.bundle-metadata.`
    /// roles), by role.
    pub components: Vec<NamedBytesV1>,
    /// View identifiers the package places in host memory for this
    /// topology. A bundle may follow or override them in its plan.
    pub host_views: Vec<String>,
}

/// Where a planned view lives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewPlacementV1 {
    Device,
    /// Page-locked, device-mapped host memory; the span is cut out of the
    /// arena as `HOST_RESIDENT_VIEWS_CAPABILITY_V1` describes.
    Host,
}

/// One named view of the compiled image, for this rank.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedViewV1 {
    pub id: String,
    pub alignment: u64,
    pub placement: ViewPlacementV1,
    pub offset: u64,
    pub byte_length: u64,
}

/// One checkpoint tensor a record reads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordSourceV1 {
    /// Index into the request's `files`.
    pub file: u32,
    pub tensor: String,
}

/// One placed piece.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightRecordV1 {
    pub arena_offset: u64,
    pub byte_length: u64,
    pub transform: u32,
    pub params: Vec<u8>,
    pub sources: Vec<RecordSourceV1>,
    /// The axis `ranges` select along, or `None` for the whole tensor.
    pub axis: Option<u32>,
    pub ranges: Vec<Range<u64>>,
    pub stored_shape: Vec<u64>,
}

/// The bundle's plan for one rank.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightPlanV1 {
    pub ranks: u32,
    pub rank: u32,
    /// The compiled-image role the bundle opens.
    pub image_role: String,
    /// The arena's alignment; every view offset is a multiple of it.
    pub alignment: u64,
    /// The rank image's bytes; bytes no record covers are zero.
    pub arena_bytes: u64,
    pub views: Vec<PlannedViewV1>,
    /// Authenticated components the bundle reads for this rank, by role.
    pub components: Vec<NamedBytesV1>,
    /// Records in increasing, non-overlapping arena order.
    pub records: Vec<WeightRecordV1>,
}

/// Prefixes [`WeightPlanV1::identity_document`].
const IDENTITY_MAGIC: &[u8; 8] = b"CHOPWPI2";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightPlanCodecError(pub String);

impl std::fmt::Display for WeightPlanCodecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "weight plan: {}", self.0)
    }
}

impl std::error::Error for WeightPlanCodecError {}

fn fail<T>(reason: impl Into<String>) -> Result<T, WeightPlanCodecError> {
    Err(WeightPlanCodecError(reason.into()))
}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
    fn count(&mut self, count: usize) {
        self.u32(u32::try_from(count).expect("bounded list length"));
    }
    fn bytes(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
        self.0.extend_from_slice(bytes);
    }
    fn name(&mut self, name: &str) {
        self.bytes(name.as_bytes());
    }
    fn named(&mut self, list: &[NamedBytesV1]) {
        self.count(list.len());
        for entry in list {
            self.name(&entry.name);
            self.bytes(&entry.bytes);
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], WeightPlanCodecError> {
        if count > self.bytes.len() {
            return fail("document ends early");
        }
        let (head, tail) = self.bytes.split_at(count);
        self.bytes = tail;
        Ok(head)
    }
    fn u32(&mut self) -> Result<u32, WeightPlanCodecError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64, WeightPlanCodecError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }
    fn count(&mut self, bound: usize) -> Result<usize, WeightPlanCodecError> {
        let count = self.u32()? as usize;
        if count > bound {
            return fail(format!("list of {count} entries exceeds {bound}"));
        }
        Ok(count)
    }
    fn bytes(&mut self) -> Result<&'a [u8], WeightPlanCodecError> {
        let length = usize::try_from(self.u64()?).map_err(|_| {
            WeightPlanCodecError("byte string longer than this host can hold".into())
        })?;
        self.take(length)
    }
    fn name(&mut self) -> Result<String, WeightPlanCodecError> {
        let bytes = self.bytes()?;
        if bytes.is_empty() || bytes.len() > WEIGHT_PLAN_NAME_BYTES_MAX_V1 {
            return fail("name empty or too long");
        }
        String::from_utf8(bytes.to_vec())
            .map_err(|_| WeightPlanCodecError("name is not UTF-8".into()))
    }
    fn named(&mut self) -> Result<Vec<NamedBytesV1>, WeightPlanCodecError> {
        let count = self.count(WEIGHT_PLAN_ENTRIES_MAX_V1)?;
        let mut list = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            list.push(NamedBytesV1 {
                name: self.name()?,
                bytes: self.bytes()?.to_vec(),
            });
        }
        if list.windows(2).any(|pair| pair[0].name >= pair[1].name) {
            return fail("names are not unique and sorted");
        }
        Ok(list)
    }
    fn magic(&mut self, magic: &[u8; 8], document: &str) -> Result<(), WeightPlanCodecError> {
        if self.take(8)? != magic {
            return fail(format!("not a {document} document"));
        }
        let layout = self.u32()?;
        if layout != WEIGHT_PLAN_LAYOUT_V1 {
            return Err(layout_mismatch(document, layout));
        }
        Ok(())
    }
    fn finish(self) -> Result<(), WeightPlanCodecError> {
        if !self.bytes.is_empty() {
            return fail("trailing bytes");
        }
        Ok(())
    }
}

/// The refusal of a document or sink in another layout.
pub(crate) fn layout_mismatch(document: &str, layout: u32) -> WeightPlanCodecError {
    WeightPlanCodecError(format!(
        "{document} layout {layout}; this build reads layout {WEIGHT_PLAN_LAYOUT_V1}"
    ))
}

fn check_rank(ranks: u32, rank: u32) -> Result<(), WeightPlanCodecError> {
    if ranks == 0 || rank >= ranks {
        return fail("rank outside the rank count");
    }
    Ok(())
}

impl WeightPlanRequestV1 {
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = Writer::default();
        writer.0.extend_from_slice(REQUEST_MAGIC);
        writer.u32(WEIGHT_PLAN_LAYOUT_V1);
        writer.bytes(self.image_role.as_bytes());
        writer.u32(self.ranks);
        writer.u32(self.rank);
        writer.0.extend_from_slice(&self.checkpoint_identity);
        writer
            .0
            .extend_from_slice(&self.implementation.shared_transforms);
        writer
            .0
            .extend_from_slice(&self.implementation.bundle_module);
        writer.count(self.files.len());
        for file in &self.files {
            writer.name(&file.name);
            writer.u64(file.byte_length);
            writer.u64(file.data_offset);
            writer.bytes(&file.header);
        }
        writer.named(&self.documents);
        writer.named(&self.components);
        writer.count(self.host_views.len());
        for view in &self.host_views {
            writer.name(view);
        }
        writer.0
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, WeightPlanCodecError> {
        let mut reader = Reader { bytes };
        reader.magic(REQUEST_MAGIC, "request")?;
        let image_role = reader.bytes()?;
        if image_role.len() > WEIGHT_PLAN_NAME_BYTES_MAX_V1 {
            return fail("image role too long");
        }
        let image_role = String::from_utf8(image_role.to_vec())
            .map_err(|_| WeightPlanCodecError("image role is not UTF-8".into()))?;
        let ranks = reader.u32()?;
        let rank = reader.u32()?;
        check_rank(ranks, rank)?;
        let checkpoint_identity = reader.take(32)?.try_into().expect("32 bytes");
        let implementation = WeightPlanImplementationV1 {
            shared_transforms: reader.take(32)?.try_into().expect("32 bytes"),
            bundle_module: reader.take(32)?.try_into().expect("32 bytes"),
        };
        let count = reader.count(WEIGHT_PLAN_ENTRIES_MAX_V1)?;
        let mut files = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let file = CheckpointFileV1 {
                name: reader.name()?,
                byte_length: reader.u64()?,
                data_offset: reader.u64()?,
                header: reader.bytes()?.to_vec(),
            };
            if file.data_offset != 8 + file.header.len() as u64
                || file.data_offset > file.byte_length
            {
                return fail(format!("file {} header does not fit the file", file.name));
            }
            files.push(file);
        }
        if files.windows(2).any(|pair| pair[0].name >= pair[1].name) {
            return fail("file names are not unique and sorted");
        }
        let documents = reader.named()?;
        let components = reader.named()?;
        let count = reader.count(WEIGHT_PLAN_ENTRIES_MAX_V1)?;
        let host_views = (0..count)
            .map(|_| reader.name())
            .collect::<Result<Vec<_>, _>>()?;
        reader.finish()?;
        Ok(Self {
            image_role,
            ranks,
            rank,
            checkpoint_identity,
            implementation,
            files,
            documents,
            components,
            host_views,
        })
    }

    /// A document by name.
    #[must_use]
    pub fn document(&self, name: &str) -> Option<&[u8]> {
        self.documents
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.bytes.as_slice())
    }

    /// A package component by role.
    #[must_use]
    pub fn component(&self, role: &str) -> Option<&[u8]> {
        self.components
            .iter()
            .find(|entry| entry.name == role)
            .map(|entry| entry.bytes.as_slice())
    }
}

impl WeightPlanV1 {
    /// Whether a record is executed by the bundle's transform export.
    #[must_use]
    pub fn uses_bundle_transforms(&self) -> bool {
        self.records
            .iter()
            .any(|record| record.transform >= BUNDLE_TRANSFORM_FIRST)
    }

    /// What the rank's planned image identity digests: a magic, the
    /// checkpoint identity, the shared transform implementation, a flag and,
    /// when a record uses a bundle-owned transform, the bundle module digest,
    /// then the plan's encoding with its components left out (a planned rank
    /// index may carry the digest). Host and bundle each take its SHA-256
    /// over the request's checkpoint identity and implementation.
    #[must_use]
    pub fn identity_document(
        &self,
        checkpoint_identity: &[u8; 32],
        implementation: &WeightPlanImplementationV1,
    ) -> Vec<u8> {
        let mut document = IDENTITY_MAGIC.to_vec();
        document.extend_from_slice(checkpoint_identity);
        document.extend_from_slice(&implementation.shared_transforms);
        if self.uses_bundle_transforms() {
            document.push(1);
            document.extend_from_slice(&implementation.bundle_module);
        } else {
            document.push(0);
        }
        document.extend_from_slice(
            &Self {
                components: Vec::new(),
                ..self.clone()
            }
            .encode(),
        );
        document
    }

    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = Writer::default();
        writer.0.extend_from_slice(PLAN_MAGIC);
        writer.u32(WEIGHT_PLAN_LAYOUT_V1);
        writer.u32(self.ranks);
        writer.u32(self.rank);
        writer.name(&self.image_role);
        writer.u64(self.alignment);
        writer.u64(self.arena_bytes);
        writer.count(self.views.len());
        for view in &self.views {
            writer.name(&view.id);
            writer.u64(view.alignment);
            writer.u32(match view.placement {
                ViewPlacementV1::Device => 0,
                ViewPlacementV1::Host => 1,
            });
            writer.u64(view.offset);
            writer.u64(view.byte_length);
        }
        writer.named(&self.components);
        writer.count(self.records.len());
        for record in &self.records {
            writer.u64(record.arena_offset);
            writer.u64(record.byte_length);
            writer.u32(record.transform);
            writer.bytes(&record.params);
            writer.count(record.sources.len());
            for source in &record.sources {
                writer.u32(source.file);
                writer.name(&source.tensor);
            }
            writer.u32(record.axis.unwrap_or(u32::MAX));
            writer.count(record.ranges.len());
            for range in &record.ranges {
                writer.u64(range.start);
                writer.u64(range.end);
            }
            writer.count(record.stored_shape.len());
            for dim in &record.stored_shape {
                writer.u64(*dim);
            }
        }
        writer.0
    }

    /// Decodes and checks the plan's own consistency: aligned views and
    /// records inside the arena, records in order without overlap. Whether
    /// each record's transform produces its stated length is the executor's
    /// check, with the checkpoint in hand.
    pub fn decode(bytes: &[u8]) -> Result<Self, WeightPlanCodecError> {
        let mut reader = Reader { bytes };
        reader.magic(PLAN_MAGIC, "plan")?;
        let ranks = reader.u32()?;
        let rank = reader.u32()?;
        check_rank(ranks, rank)?;
        let image_role = reader.name()?;
        let alignment = reader.u64()?;
        let arena_bytes = reader.u64()?;
        if !alignment.is_power_of_two()
            || arena_bytes == 0
            || !arena_bytes.is_multiple_of(alignment)
        {
            return fail("arena alignment must be a power of two dividing a nonzero arena");
        }
        let count = reader.count(WEIGHT_PLAN_ENTRIES_MAX_V1)?;
        let mut views = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let view = PlannedViewV1 {
                id: reader.name()?,
                alignment: reader.u64()?,
                placement: match reader.u32()? {
                    0 => ViewPlacementV1::Device,
                    1 => ViewPlacementV1::Host,
                    other => return fail(format!("unknown view placement {other}")),
                },
                offset: reader.u64()?,
                byte_length: reader.u64()?,
            };
            if !view.alignment.is_power_of_two()
                || !view.offset.is_multiple_of(view.alignment)
                || view.byte_length == 0
                || view
                    .offset
                    .checked_add(view.byte_length)
                    .is_none_or(|end| end > arena_bytes)
            {
                return fail(format!(
                    "view {} is misaligned or outside the arena",
                    view.id
                ));
            }
            views.push(view);
        }
        if views
            .iter()
            .enumerate()
            .any(|(index, view)| views[..index].iter().any(|other| other.id == view.id))
        {
            return fail("duplicate view identifier");
        }
        let components = reader.named()?;
        let count = reader.count(WEIGHT_PLAN_ENTRIES_MAX_V1)?;
        let mut records: Vec<WeightRecordV1> = Vec::with_capacity(count.min(1 << 20));
        let mut end = 0_u64;
        for _ in 0..count {
            let arena_offset = reader.u64()?;
            let byte_length = reader.u64()?;
            let transform = reader.u32()?;
            let params = reader.bytes()?.to_vec();
            let sources = (0..reader.count(64)?)
                .map(|_| {
                    Ok(RecordSourceV1 {
                        file: reader.u32()?,
                        tensor: reader.name()?,
                    })
                })
                .collect::<Result<Vec<_>, WeightPlanCodecError>>()?;
            let axis = match reader.u32()? {
                u32::MAX => None,
                axis => Some(axis),
            };
            let ranges = (0..reader.count(WEIGHT_PLAN_ENTRIES_MAX_V1)?)
                .map(|_| Ok(reader.u64()?..reader.u64()?))
                .collect::<Result<Vec<_>, WeightPlanCodecError>>()?;
            let stored_shape = (0..reader.count(WEIGHT_PLAN_DIMENSIONS_MAX_V1)?)
                .map(|_| reader.u64())
                .collect::<Result<Vec<_>, _>>()?;
            // Whether a record's transform reads what it names (inline bytes
            // read nothing) is the executor's check.
            if byte_length == 0
                || arena_offset < end
                || arena_offset
                    .checked_add(byte_length)
                    .is_none_or(|last| last > arena_bytes)
            {
                return fail(format!(
                    "record at {arena_offset} overlaps or leaves the arena"
                ));
            }
            end = arena_offset + byte_length;
            records.push(WeightRecordV1 {
                arena_offset,
                byte_length,
                transform,
                params,
                sources,
                axis,
                ranges,
                stored_shape,
            });
        }
        reader.finish()?;
        Ok(Self {
            ranks,
            rank,
            image_role,
            alignment,
            arena_bytes,
            views,
            components,
            records,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> WeightPlanRequestV1 {
        WeightPlanRequestV1 {
            image_role: "model.rank-images.v1".into(),
            ranks: 4,
            rank: 2,
            checkpoint_identity: [7; 32],
            implementation: WeightPlanImplementationV1 {
                shared_transforms: [1; 32],
                bundle_module: [2; 32],
            },
            files: vec![CheckpointFileV1 {
                name: "model-00001.safetensors".into(),
                byte_length: 1000,
                data_offset: 8 + 5,
                header: b"{\"a\"}".to_vec(),
            }],
            documents: vec![NamedBytesV1 {
                name: "config.json".into(),
                bytes: b"{}".to_vec(),
            }],
            components: vec![],
            host_views: vec!["ngram".into()],
        }
    }

    fn plan() -> WeightPlanV1 {
        WeightPlanV1 {
            ranks: 4,
            rank: 2,
            image_role: "model.rank-images.v1".into(),
            alignment: 4096,
            arena_bytes: 3 * 4096,
            views: vec![PlannedViewV1 {
                id: "ngram".into(),
                alignment: 4096,
                placement: ViewPlacementV1::Host,
                offset: 4096,
                byte_length: 100,
            }],
            components: vec![NamedBytesV1 {
                name: "model.rank-index.2".into(),
                bytes: b"{}".to_vec(),
            }],
            records: vec![
                WeightRecordV1 {
                    arena_offset: 0,
                    byte_length: 64,
                    transform: 1,
                    params: vec![],
                    sources: vec![RecordSourceV1 {
                        file: 0,
                        tensor: "w".into(),
                    }],
                    axis: Some(0),
                    ranges: vec![0..4, 8..12],
                    stored_shape: vec![8, 8],
                },
                WeightRecordV1 {
                    arena_offset: 4096,
                    byte_length: 100,
                    transform: 3,
                    params: 2_u32.to_le_bytes().to_vec(),
                    sources: vec![RecordSourceV1 {
                        file: 0,
                        tensor: "v".into(),
                    }],
                    axis: None,
                    ranges: vec![],
                    stored_shape: vec![10, 10],
                },
            ],
        }
    }

    #[test]
    fn documents_round_trip() {
        let request = request();
        assert_eq!(
            WeightPlanRequestV1::decode(&request.encode()).unwrap(),
            request
        );
        let plan = plan();
        assert_eq!(WeightPlanV1::decode(&plan.encode()).unwrap(), plan);
    }

    #[test]
    fn decoding_refuses_malformed_documents() {
        let bytes = plan().encode();
        for cut in [0, 8, 12, bytes.len() - 1] {
            assert!(WeightPlanV1::decode(&bytes[..cut]).is_err(), "{cut}");
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(WeightPlanV1::decode(&trailing).is_err());
        for broken in [
            WeightPlanV1 { rank: 4, ..plan() },
            WeightPlanV1 {
                alignment: 3000,
                ..plan()
            },
            WeightPlanV1 {
                arena_bytes: 4096,
                ..plan()
            },
        ] {
            assert!(WeightPlanV1::decode(&broken.encode()).is_err());
        }
        let mut overlapping = plan();
        overlapping.records[1].arena_offset = 32;
        assert!(WeightPlanV1::decode(&overlapping.encode()).is_err());
        let mut outside = plan();
        outside.records[1].arena_offset = 3 * 4096 - 50;
        assert!(WeightPlanV1::decode(&outside.encode()).is_err());
        let mut misaligned = plan();
        misaligned.views[0].offset = 100;
        assert!(WeightPlanV1::decode(&misaligned.encode()).is_err());
        let mut unsorted = request();
        unsorted.files.push(unsorted.files[0].clone());
        assert!(WeightPlanRequestV1::decode(&unsorted.encode()).is_err());
        let mut header = request();
        header.files[0].data_offset = 9;
        assert!(WeightPlanRequestV1::decode(&header.encode()).is_err());
        // An empty role asks for the bundle's primary image.
        let mut primary = request();
        primary.image_role.clear();
        assert_eq!(
            WeightPlanRequestV1::decode(&primary.encode()).unwrap(),
            primary
        );
        let mut long = request();
        long.image_role = "r".repeat(WEIGHT_PLAN_NAME_BYTES_MAX_V1 + 1);
        assert!(WeightPlanRequestV1::decode(&long.encode()).is_err());
    }

    /// A request in the layout before image roles (layout 1: ranks right
    /// after the header) is refused by name instead of misread, and so is a
    /// plan document of another layout.
    #[test]
    fn documents_of_another_layout_are_refused_by_name() {
        let mut older = REQUEST_MAGIC.to_vec();
        older.extend_from_slice(&1_u32.to_le_bytes());
        older.extend_from_slice(&4_u32.to_le_bytes());
        older.extend_from_slice(&2_u32.to_le_bytes());
        older.extend_from_slice(&[7; 32]);
        for _ in 0..4 {
            older.extend_from_slice(&0_u32.to_le_bytes());
        }
        let error = WeightPlanRequestV1::decode(&older).unwrap_err().to_string();
        assert!(error.contains("request layout 1"), "{error}");
        // Earlier layouts over today's fields are refused as well, including
        // layout 2, whose requests stated no implementation.
        for layout in [1_u32, 2] {
            let mut relabelled = request().encode();
            relabelled[8..12].copy_from_slice(&layout.to_le_bytes());
            let error = WeightPlanRequestV1::decode(&relabelled)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&format!("request layout {layout}")),
                "{error}"
            );
        }
        let mut plan = plan().encode();
        plan[8..12].copy_from_slice(&(WEIGHT_PLAN_LAYOUT_V1 + 1).to_le_bytes());
        let error = WeightPlanV1::decode(&plan).unwrap_err().to_string();
        let expected = format!("plan layout {}", WEIGHT_PLAN_LAYOUT_V1 + 1);
        assert!(error.contains(&expected), "{error}");
        let error = WeightPlanV1::decode(&request().encode())
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a plan document"), "{error}");
    }

    const IMPLEMENTATION: WeightPlanImplementationV1 = WeightPlanImplementationV1 {
        shared_transforms: [3; 32],
        bundle_module: [4; 32],
    };

    /// Two builds that could produce different bytes for one plan never
    /// share an identity: the shared transforms always count, the bundle
    /// module whenever a record uses a bundle-owned transform.
    #[test]
    fn identity_covers_the_implementation_that_produces_the_bytes() {
        let shared = plan();
        assert!(!shared.uses_bundle_transforms());
        let identity = shared.identity_document(&[7; 32], &IMPLEMENTATION);
        let other_host = WeightPlanImplementationV1 {
            shared_transforms: [5; 32],
            ..IMPLEMENTATION
        };
        assert_ne!(shared.identity_document(&[7; 32], &other_host), identity);
        let other_module = WeightPlanImplementationV1 {
            bundle_module: [6; 32],
            ..IMPLEMENTATION
        };
        assert_eq!(shared.identity_document(&[7; 32], &other_module), identity);
        let mut owned = plan();
        owned.records[0].transform = BUNDLE_TRANSFORM_FIRST + 1;
        assert!(owned.uses_bundle_transforms());
        assert_ne!(
            owned.identity_document(&[7; 32], &other_module),
            owned.identity_document(&[7; 32], &IMPLEMENTATION)
        );
        // The request carries the implementation to the bundle.
        let mut stated = request();
        stated.implementation = IMPLEMENTATION;
        assert_eq!(
            WeightPlanRequestV1::decode(&stated.encode())
                .unwrap()
                .implementation,
            IMPLEMENTATION
        );
    }

    #[test]
    fn identity_covers_the_plan_but_not_its_components() {
        let plan = plan();
        let identity = plan.identity_document(&[7; 32], &IMPLEMENTATION);
        assert!(identity.starts_with(IDENTITY_MAGIC));
        let mut indexed = plan.clone();
        indexed.components[0].bytes = b"{\"sha256\": \"...\"}".to_vec();
        assert_eq!(
            indexed.identity_document(&[7; 32], &IMPLEMENTATION),
            identity
        );
        assert_ne!(plan.identity_document(&[8; 32], &IMPLEMENTATION), identity);
        let mut moved = plan.clone();
        moved.records[0].ranges = vec![0..4, 4..8];
        assert_ne!(moved.identity_document(&[7; 32], &IMPLEMENTATION), identity);
        let mut inline = plan;
        inline.records[1].sources.clear();
        assert_eq!(WeightPlanV1::decode(&inline.encode()).unwrap(), inline);
    }
}
