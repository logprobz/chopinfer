#![forbid(unsafe_code)]

pub mod checkpoint;
mod hash;
mod json;
pub mod source_file;

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs::{self, File, Metadata};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::{FileExt, MetadataExt};
#[cfg(windows)]
use std::os::windows::fs::{FileExt, MetadataExt};

pub use hash::Sha256State;

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

type Result<T> = std::result::Result<T, ArtifactError>;

pub fn sha256_reader(reader: &mut impl Read) -> Result<String> {
    hash::reader(reader)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactError(String);

impl ArtifactError {
    fn new(message: String) -> Self {
        Self(message)
    }

    fn io(action: &str, path: &Path, error: io::Error) -> Self {
        Self(format!("cannot {action} {}: {error}", path.display()))
    }

    #[must_use]
    pub fn validation(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ArtifactError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    Bf16,
    Bool,
    F16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
    F8E8M0,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl DType {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "BF16" => Ok(Self::Bf16),
            "BOOL" => Ok(Self::Bool),
            "F16" => Ok(Self::F16),
            "F32" => Ok(Self::F32),
            "F64" => Ok(Self::F64),
            "F8_E4M3" => Ok(Self::F8E4M3),
            "F8_E5M2" => Ok(Self::F8E5M2),
            "F8_E8M0" => Ok(Self::F8E8M0),
            "I8" => Ok(Self::I8),
            "I16" => Ok(Self::I16),
            "I32" => Ok(Self::I32),
            "I64" => Ok(Self::I64),
            "U8" => Ok(Self::U8),
            "U16" => Ok(Self::U16),
            "U32" => Ok(Self::U32),
            "U64" => Ok(Self::U64),
            _ => Err(ArtifactError::new(format!(
                "unsupported Safetensors dtype '{value}'"
            ))),
        }
    }

    const fn byte_width(self) -> u64 {
        match self {
            Self::Bf16 | Self::F16 | Self::I16 | Self::U16 => 2,
            Self::F32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 | Self::I8 | Self::U8 | Self::Bool => 1,
            Self::I32 | Self::U32 => 4,
        }
    }

    #[must_use]
    pub const fn safetensors_name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Bool => "BOOL",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E8M0 => "F8_E8M0",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::U64 => "U64",
        }
    }
}

pub fn decode_safetensors_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(ArtifactError::validation(format!(
            "Safetensors F32 payload has {} bytes, expected a multiple of 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk width is four")))
        .collect())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInfo {
    dtype: DType,
    shape: Vec<u64>,
    data_start: u64,
    data_end: u64,
}

impl TensorInfo {
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.data_end - self.data_start
    }

    #[must_use]
    pub const fn data_offsets(&self) -> (u64, u64) {
        (self.data_start, self.data_end)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardHeader {
    header_bytes: u64,
    data_offset: u64,
    data_bytes: u64,
    tensors: BTreeMap<String, TensorInfo>,
}

impl ShardHeader {
    #[must_use]
    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    #[must_use]
    pub const fn header_bytes(&self) -> u64 {
        self.header_bytes
    }

    #[must_use]
    pub const fn data_offset(&self) -> u64 {
        self.data_offset
    }

    #[must_use]
    pub const fn data_bytes(&self) -> u64 {
        self.data_bytes
    }
}

pub fn parse_safetensors_header(reader: &mut impl Read, file_size: u64) -> Result<ShardHeader> {
    let mut prefix = [0_u8; 8];
    reader.read_exact(&mut prefix).map_err(|error| {
        ArtifactError::new(format!(
            "Safetensors file does not contain a complete 8-byte header length: {error}"
        ))
    })?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
        return Err(ArtifactError::new(format!(
            "Safetensors header length {header_bytes} is outside 1..={MAX_HEADER_BYTES}"
        )));
    }
    let data_offset = 8_u64
        .checked_add(header_bytes)
        .ok_or_else(|| ArtifactError::new("Safetensors header offset overflow".to_owned()))?;
    let data_bytes = file_size.checked_sub(data_offset).ok_or_else(|| {
        ArtifactError::new(format!(
            "Safetensors header ends at {data_offset}, beyond file size {file_size}"
        ))
    })?;
    let header_len = usize::try_from(header_bytes)
        .map_err(|_| ArtifactError::new("Safetensors header does not fit memory".to_owned()))?;
    let mut bytes = vec![0_u8; header_len];
    reader.read_exact(&mut bytes).map_err(|error| {
        ArtifactError::new(format!(
            "cannot read complete Safetensors JSON header: {error}"
        ))
    })?;
    let root = json::parse(&bytes)?;
    let object = root.object("Safetensors header")?;
    let mut tensors = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            validate_metadata(value)?;
            continue;
        }
        if name.is_empty() {
            return Err(ArtifactError::new(
                "Safetensors tensor name cannot be empty".to_owned(),
            ));
        }
        tensors.insert(name.clone(), parse_tensor(name, value)?);
    }
    validate_spans(&tensors, data_bytes)?;
    Ok(ShardHeader {
        header_bytes,
        data_offset,
        data_bytes,
        tensors,
    })
}

fn validate_metadata(value: &json::Value) -> Result<()> {
    let metadata = value.object("Safetensors __metadata__")?;
    for (key, value) in metadata {
        value.string(&format!("Safetensors metadata '{key}'"))?;
    }
    Ok(())
}

fn parse_tensor(name: &str, value: &json::Value) -> Result<TensorInfo> {
    let object = value.object(&format!("tensor '{name}'"))?;
    let expected: BTreeSet<&str> = ["data_offsets", "dtype", "shape"].into_iter().collect();
    let actual: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    if actual != expected {
        return Err(ArtifactError::new(format!(
            "tensor '{name}' must contain exactly dtype, shape, and data_offsets"
        )));
    }
    let dtype = DType::parse(
        object
            .get("dtype")
            .expect("key set checked")
            .string(&format!("tensor '{name}' dtype"))?,
    )?;
    let shape = object
        .get("shape")
        .expect("key set checked")
        .array(&format!("tensor '{name}' shape"))?
        .iter()
        .enumerate()
        .map(|(axis, value)| value.number(&format!("tensor '{name}' shape axis {axis}")))
        .collect::<Result<Vec<_>>>()?;
    let element_count = shape
        .iter()
        .enumerate()
        .try_fold(1_u64, |count, (axis, size)| {
            count.checked_mul(*size).ok_or_else(|| {
                ArtifactError::new(format!("tensor '{name}' shape overflows at axis {axis}"))
            })
        })?;
    let expected_bytes = element_count
        .checked_mul(dtype.byte_width())
        .ok_or_else(|| ArtifactError::new(format!("tensor '{name}' byte length overflow")))?;
    let offsets = object
        .get("data_offsets")
        .expect("key set checked")
        .array(&format!("tensor '{name}' data_offsets"))?;
    if offsets.len() != 2 {
        return Err(ArtifactError::new(format!(
            "tensor '{name}' data_offsets must have two entries"
        )));
    }
    let data_start = offsets[0].number(&format!("tensor '{name}' start offset"))?;
    let data_end = offsets[1].number(&format!("tensor '{name}' end offset"))?;
    let actual_bytes = data_end
        .checked_sub(data_start)
        .ok_or_else(|| ArtifactError::new(format!("tensor '{name}' offsets are reversed")))?;
    if actual_bytes != expected_bytes {
        return Err(ArtifactError::new(format!(
            "tensor '{name}' declares {actual_bytes} bytes, expected {expected_bytes} from dtype and shape"
        )));
    }
    Ok(TensorInfo {
        dtype,
        shape,
        data_start,
        data_end,
    })
}

fn validate_spans(tensors: &BTreeMap<String, TensorInfo>, data_bytes: u64) -> Result<()> {
    let mut spans: Vec<(&str, u64, u64)> = tensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor.data_start, tensor.data_end))
        .collect();
    spans.sort_by_key(|(name, start, end)| (*start, *end, *name));
    let mut cursor = 0_u64;
    for (name, start, end) in spans {
        if start < cursor {
            return Err(ArtifactError::new(format!(
                "tensor '{name}' data offsets overlap a prior tensor"
            )));
        }
        if start > cursor {
            return Err(ArtifactError::new(format!(
                "tensor '{name}' leaves a gap before offset {start}"
            )));
        }
        cursor = end;
    }
    if cursor != data_bytes {
        return Err(ArtifactError::new(format!(
            "Safetensors data ends at {cursor}, but file contains {data_bytes} payload bytes"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsIndex {
    total_size: u64,
    weight_map: BTreeMap<String, String>,
    shards: Vec<String>,
}

impl SafetensorsIndex {
    #[must_use]
    pub const fn total_size(&self) -> u64 {
        self.total_size
    }

    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn shards(&self) -> impl Iterator<Item = &str> {
        self.shards.iter().map(String::as_str)
    }

    #[must_use]
    pub fn shard_for(&self, tensor: &str) -> Option<&str> {
        self.weight_map.get(tensor).map(String::as_str)
    }
}

pub fn parse_safetensors_index(bytes: &[u8]) -> Result<SafetensorsIndex> {
    let root = json::parse(bytes)?;
    let object = root.object("Safetensors index")?;
    let expected: BTreeSet<&str> = ["metadata", "weight_map"].into_iter().collect();
    let actual: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    if actual != expected {
        return Err(ArtifactError::new(
            "Safetensors index must contain exactly metadata and weight_map".to_owned(),
        ));
    }
    let metadata = object
        .get("metadata")
        .expect("key set checked")
        .object("Safetensors index metadata")?;
    let total_size = metadata
        .get("total_size")
        .ok_or_else(|| ArtifactError::new("index metadata is missing total_size".to_owned()))?
        .number("index metadata total_size")?;
    let mapping = object
        .get("weight_map")
        .expect("key set checked")
        .object("Safetensors weight_map")?;
    if mapping.is_empty() {
        return Err(ArtifactError::new(
            "Safetensors weight_map cannot be empty".to_owned(),
        ));
    }
    let mut weight_map = BTreeMap::new();
    let mut shards = BTreeSet::new();
    for (tensor, value) in mapping {
        if tensor.is_empty() {
            return Err(ArtifactError::new(
                "Safetensors index tensor name cannot be empty".to_owned(),
            ));
        }
        let shard = value.string(&format!("weight_map entry '{tensor}'"))?;
        validate_file_name(shard)?;
        if !shard.ends_with(".safetensors") {
            return Err(ArtifactError::new(format!(
                "weight_map shard '{shard}' is not a .safetensors file"
            )));
        }
        weight_map.insert(tensor.clone(), shard.to_owned());
        shards.insert(shard.to_owned());
    }
    Ok(SafetensorsIndex {
        total_size,
        weight_map,
        shards: shards.into_iter().collect(),
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileIdentity {
    name: String,
    size: u64,
    sha256: String,
}

impl FileIdentity {
    pub fn new(name: &str, size: u64, sha256: &str) -> Result<Self> {
        validate_file_name(name)?;
        if sha256.len() != 64
            || !sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(ArtifactError::new(format!(
                "file '{name}' SHA-256 must be 64 lowercase hexadecimal characters"
            )));
        }
        Ok(Self {
            name: name.to_owned(),
            size,
            sha256: sha256.to_owned(),
        })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactContract {
    repository: String,
    revision: String,
    config: FileIdentity,
    index: FileIdentity,
    shards: Vec<FileIdentity>,
    total_tensor_bytes: u64,
    tensor_count: usize,
    topology: ShardTopology,
}

impl ArtifactContract {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        repository: &str,
        revision: &str,
        config: FileIdentity,
        index: FileIdentity,
        mut shards: Vec<FileIdentity>,
        total_tensor_bytes: u64,
        tensor_count: usize,
        topology: ShardTopology,
    ) -> Result<Self> {
        if repository.is_empty() || revision.is_empty() {
            return Err(ArtifactError::new(
                "artifact repository and revision cannot be empty".to_owned(),
            ));
        }
        if config.name != "config.json" || index.name != "model.safetensors.index.json" {
            return Err(ArtifactError::new(
                "artifact contract requires config.json and model.safetensors.index.json"
                    .to_owned(),
            ));
        }
        if shards.is_empty() || tensor_count == 0 || total_tensor_bytes == 0 {
            return Err(ArtifactError::new(
                "artifact contract requires nonempty shards and tensors".to_owned(),
            ));
        }
        shards.sort_by(|left, right| left.name.cmp(&right.name));
        for pair in shards.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(ArtifactError::new(format!(
                    "duplicate shard identity '{}'",
                    pair[0].name
                )));
            }
        }
        Ok(Self {
            repository: repository.to_owned(),
            revision: revision.to_owned(),
            config,
            index,
            shards,
            total_tensor_bytes,
            tensor_count,
            topology,
        })
    }

    #[must_use]
    pub fn config(&self) -> &FileIdentity {
        &self.config
    }

    #[must_use]
    pub fn index(&self) -> &FileIdentity {
        &self.index
    }

    #[must_use]
    pub fn shards(&self) -> &[FileIdentity] {
        &self.shards
    }

    #[must_use]
    pub const fn total_tensor_bytes(&self) -> u64 {
        self.total_tensor_bytes
    }

    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    #[must_use]
    pub const fn topology(&self) -> ShardTopology {
        self.topology
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShardTopology {
    rank_count: u32,
}

impl ShardTopology {
    pub fn new(rank_count: u32) -> Result<Self> {
        if !matches!(rank_count, 1 | 2 | 4 | 8) {
            return Err(ArtifactError::new(
                "local shard topology rank count must be 1, 2, 4, or 8".to_owned(),
            ));
        }
        Ok(Self { rank_count })
    }

    #[must_use]
    pub const fn rank_count(self) -> u32 {
        self.rank_count
    }

    pub fn rank(self, ordinal: u32) -> Result<LocalRank> {
        if ordinal >= self.rank_count {
            return Err(ArtifactError::new(format!(
                "local rank {ordinal} is outside 0..{}",
                self.rank_count
            )));
        }
        Ok(LocalRank { ordinal })
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LocalRank {
    ordinal: u32,
}

impl LocalRank {
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self.ordinal
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactSource {
    repository: String,
    revision: String,
}

impl ArtifactSource {
    #[must_use]
    pub fn new(repository: &str, revision: &str) -> Self {
        Self {
            repository: repository.to_owned(),
            revision: revision.to_owned(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorManifest {
    shard: String,
    dtype: DType,
    shape: Vec<u64>,
    file_start: u64,
    file_end: u64,
}

impl TensorManifest {
    #[must_use]
    pub fn shard(&self) -> &str {
        &self.shard
    }

    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.file_end - self.file_start
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardManifest {
    identity: FileIdentity,
    header_bytes: u64,
    data_offset: u64,
    tensor_count: usize,
}

impl ShardManifest {
    #[must_use]
    pub fn identity(&self) -> &FileIdentity {
        &self.identity
    }

    #[must_use]
    pub const fn header_bytes(&self) -> u64 {
        self.header_bytes
    }

    #[must_use]
    pub const fn data_offset(&self) -> u64 {
        self.data_offset
    }

    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadManifest {
    repository: String,
    revision: String,
    config: FileIdentity,
    index: FileIdentity,
    shards: Vec<ShardManifest>,
    tensors: BTreeMap<String, TensorManifest>,
    total_tensor_bytes: u64,
    topology: ShardTopology,
}

impl LoadManifest {
    #[must_use]
    pub fn repository(&self) -> &str {
        &self.repository
    }

    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    #[must_use]
    pub fn config(&self) -> &FileIdentity {
        &self.config
    }

    #[must_use]
    pub fn index(&self) -> &FileIdentity {
        &self.index
    }

    #[must_use]
    pub fn shards(&self) -> &[ShardManifest] {
        &self.shards
    }

    #[must_use]
    pub fn tensors(&self) -> &BTreeMap<String, TensorManifest> {
        &self.tensors
    }

    #[must_use]
    pub const fn total_tensor_bytes(&self) -> u64 {
        self.total_tensor_bytes
    }

    #[must_use]
    pub const fn topology(&self) -> ShardTopology {
        self.topology
    }
}

#[derive(Debug)]
pub enum PayloadReadError {
    UnknownTensor,
    EmptyRead,
    BufferLengthOverflow,
    RangeEndOverflow {
        offset: u64,
        length: u64,
    },
    RangeOutOfBounds {
        offset: u64,
        length: u64,
        tensor_length: u64,
    },
    FileOffsetOverflow,
    ManifestMismatch,
    ShardLengthChanged {
        expected: u64,
        actual: u64,
    },
    ShortRead {
        expected: usize,
        actual: usize,
    },
    Io(io::Error),
}

impl fmt::Display for PayloadReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTensor => {
                formatter.write_str("tensor is not present in the verified manifest")
            }
            Self::EmptyRead => formatter.write_str("tensor payload read buffer cannot be empty"),
            Self::BufferLengthOverflow => {
                formatter.write_str("tensor payload buffer length does not fit u64")
            }
            Self::RangeEndOverflow { offset, length } => write!(
                formatter,
                "tensor-relative range offset {offset} with length {length} overflows"
            ),
            Self::RangeOutOfBounds {
                offset,
                length,
                tensor_length,
            } => write!(
                formatter,
                "tensor-relative range offset {offset} with length {length} exceeds tensor length {tensor_length}"
            ),
            Self::FileOffsetOverflow => {
                formatter.write_str("tensor payload file offset overflows u64")
            }
            Self::ManifestMismatch => {
                formatter.write_str("verified payload descriptors do not match the load manifest")
            }
            Self::ShardLengthChanged { expected, actual } => write!(
                formatter,
                "verified shard length changed from {expected} to {actual} after authentication"
            ),
            Self::ShortRead { expected, actual } => write!(
                formatter,
                "tensor payload read returned {actual} bytes, expected {expected}"
            ),
            Self::Io(error) => write!(formatter, "cannot read verified tensor payload: {error}"),
        }
    }
}

impl Error for PayloadReadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct OpenShard {
    file: File,
    path: PathBuf,
    expected_size: u64,
    expected_sha256: String,
    authenticated_metadata: Metadata,
}

/// Authenticated tensor metadata and the exact shard descriptors used to verify it.
///
/// Reads remain attached to the authenticated open inode if a shard path is renamed
/// or replaced. Call [`Self::reauthenticate_sources`] before and after consuming
/// payloads to detect source mutation. The production contract still requires a
/// read-only model mount because portable userspace checks cannot exclude mutation
/// concurrent with a read.
#[derive(Debug)]
pub struct VerifiedArtifacts {
    manifest: LoadManifest,
    shards: BTreeMap<String, OpenShard>,
}

impl VerifiedArtifacts {
    #[must_use]
    pub const fn manifest(&self) -> &LoadManifest {
        &self.manifest
    }

    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorManifest> {
        self.manifest.tensors.get(name)
    }

    pub fn tensor_byte_len(&self, name: &str) -> std::result::Result<u64, PayloadReadError> {
        self.tensor(name)
            .map(TensorManifest::byte_len)
            .ok_or(PayloadReadError::UnknownTensor)
    }

    /// Reauthenticates every retained checkpoint shard through its original descriptor.
    ///
    /// Hashing uses bounded memory and positional reads, so it does not change the
    /// descriptor cursor used during initial Safetensors parsing. Descriptor and path
    /// identity are checked before and after hashing.
    pub fn reauthenticate_sources(&self) -> Result<()> {
        for (name, shard) in &self.shards {
            reauthenticate_shard(name, shard)?;
        }
        Ok(())
    }

    pub fn read_tensor(
        &self,
        name: &str,
        tensor_offset: u64,
        buffer: &mut [u8],
    ) -> std::result::Result<(), PayloadReadError> {
        if buffer.is_empty() {
            return Err(PayloadReadError::EmptyRead);
        }
        let tensor = self.tensor(name).ok_or(PayloadReadError::UnknownTensor)?;
        let length =
            u64::try_from(buffer.len()).map_err(|_| PayloadReadError::BufferLengthOverflow)?;
        let range_end =
            tensor_offset
                .checked_add(length)
                .ok_or(PayloadReadError::RangeEndOverflow {
                    offset: tensor_offset,
                    length,
                })?;
        let tensor_length = tensor
            .file_end
            .checked_sub(tensor.file_start)
            .ok_or(PayloadReadError::ManifestMismatch)?;
        if range_end > tensor_length {
            return Err(PayloadReadError::RangeOutOfBounds {
                offset: tensor_offset,
                length,
                tensor_length,
            });
        }
        let file_offset = tensor
            .file_start
            .checked_add(tensor_offset)
            .ok_or(PayloadReadError::FileOffsetOverflow)?;
        let file_end = file_offset
            .checked_add(length)
            .ok_or(PayloadReadError::FileOffsetOverflow)?;
        if file_end > tensor.file_end {
            return Err(PayloadReadError::ManifestMismatch);
        }
        let shard = self
            .shards
            .get(&tensor.shard)
            .ok_or(PayloadReadError::ManifestMismatch)?;
        if tensor.file_end > shard.expected_size {
            return Err(PayloadReadError::ManifestMismatch);
        }
        let actual_size = shard.file.metadata().map_err(PayloadReadError::Io)?.len();
        if actual_size != shard.expected_size {
            return Err(PayloadReadError::ShardLengthChanged {
                expected: shard.expected_size,
                actual: actual_size,
            });
        }
        read_exact_at(&shard.file, buffer, file_offset)
    }

    #[must_use]
    pub fn into_manifest(self) -> LoadManifest {
        self.manifest
    }
}

pub fn inspect_artifacts(
    root: &Path,
    source: &ArtifactSource,
    contract: &ArtifactContract,
) -> Result<LoadManifest> {
    open_artifacts(root, source, contract).map(VerifiedArtifacts::into_manifest)
}

pub fn open_artifacts(
    root: &Path,
    source: &ArtifactSource,
    contract: &ArtifactContract,
) -> Result<VerifiedArtifacts> {
    if source.repository != contract.repository {
        return Err(ArtifactError::new(format!(
            "repository '{}' does not match pinned '{}'",
            source.repository, contract.repository
        )));
    }
    if source.revision != contract.revision {
        return Err(ArtifactError::new(format!(
            "revision '{}' does not match pinned '{}'",
            source.revision, contract.revision
        )));
    }
    let root_metadata = fs::symlink_metadata(root)
        .map_err(|error| ArtifactError::io("inspect artifact root", root, error))?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(ArtifactError::new(format!(
            "artifact root {} must be a real directory, not a symlink",
            root.display()
        )));
    }
    drop(open_verified_file(root, &contract.config)?);
    let mut index_file = open_verified_file(root, &contract.index)?;
    let mut index_bytes = Vec::new();
    index_file.read_to_end(&mut index_bytes).map_err(|error| {
        ArtifactError::new(format!("cannot read verified Safetensors index: {error}"))
    })?;
    let index = parse_safetensors_index(&index_bytes)?;
    if index.total_size != contract.total_tensor_bytes {
        return Err(ArtifactError::new(format!(
            "index total_size {} does not match pinned {}",
            index.total_size, contract.total_tensor_bytes
        )));
    }
    if index.tensor_count() != contract.tensor_count {
        return Err(ArtifactError::new(format!(
            "index has {} tensors, expected {}",
            index.tensor_count(),
            contract.tensor_count
        )));
    }
    let expected_shards: BTreeSet<&str> = contract
        .shards
        .iter()
        .map(|shard| shard.name.as_str())
        .collect();
    let indexed_shards: BTreeSet<&str> = index.shards().collect();
    if indexed_shards != expected_shards {
        return Err(ArtifactError::new(
            "index shard set does not match the pinned shard identities".to_owned(),
        ));
    }

    let mut tensors = BTreeMap::new();
    let mut shards = Vec::with_capacity(contract.shards.len());
    let mut open_shards = BTreeMap::new();
    let mut total_tensor_bytes = 0_u64;
    for identity in &contract.shards {
        let mut file = open_verified_file(root, identity)?;
        let header = parse_safetensors_header(&mut file, identity.size)?;
        total_tensor_bytes = total_tensor_bytes
            .checked_add(header.data_bytes)
            .ok_or_else(|| ArtifactError::new("total tensor byte length overflow".to_owned()))?;
        for (name, tensor) in &header.tensors {
            if tensors.contains_key(name) {
                return Err(ArtifactError::new(format!(
                    "tensor '{name}' appears in more than one shard"
                )));
            }
            let mapped_shard = index.shard_for(name).ok_or_else(|| {
                ArtifactError::new(format!(
                    "tensor '{name}' exists in shard '{}' but is missing from the index",
                    identity.name
                ))
            })?;
            if mapped_shard != identity.name {
                return Err(ArtifactError::new(format!(
                    "tensor '{name}' is stored in '{}' but indexed in '{mapped_shard}'",
                    identity.name
                )));
            }
            let file_start = header
                .data_offset
                .checked_add(tensor.data_start)
                .ok_or_else(|| ArtifactError::new(format!("tensor '{name}' offset overflow")))?;
            let file_end = header
                .data_offset
                .checked_add(tensor.data_end)
                .ok_or_else(|| ArtifactError::new(format!("tensor '{name}' offset overflow")))?;
            let entry = TensorManifest {
                shard: identity.name.clone(),
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                file_start,
                file_end,
            };
            let previous = tensors.insert(name.clone(), entry);
            debug_assert!(previous.is_none());
        }
        shards.push(ShardManifest {
            identity: identity.clone(),
            header_bytes: header.header_bytes,
            data_offset: header.data_offset,
            tensor_count: header.tensors.len(),
        });
        let path = root.join(&identity.name);
        let authenticated_metadata = file
            .metadata()
            .map_err(|error| ArtifactError::io("inspect authenticated shard", &path, error))?;
        open_shards.insert(
            identity.name.clone(),
            OpenShard {
                authenticated_metadata,
                file,
                path,
                expected_size: identity.size,
                expected_sha256: identity.sha256.clone(),
            },
        );
    }
    for (name, shard) in &index.weight_map {
        if !tensors.contains_key(name) {
            return Err(ArtifactError::new(format!(
                "index tensor '{name}' is missing from shard '{shard}'"
            )));
        }
    }
    if total_tensor_bytes != contract.total_tensor_bytes {
        return Err(ArtifactError::new(format!(
            "shard payloads contain {total_tensor_bytes} bytes, expected {}",
            contract.total_tensor_bytes
        )));
    }
    Ok(VerifiedArtifacts {
        manifest: LoadManifest {
            repository: contract.repository.clone(),
            revision: contract.revision.clone(),
            config: contract.config.clone(),
            index: contract.index.clone(),
            shards,
            tensors,
            total_tensor_bytes,
            topology: contract.topology,
        },
        shards: open_shards,
    })
}

/// Fill a buffer by positional reads, retrying interrupts and checking offsets.
pub fn read_exact_at(
    file: &File,
    buffer: &mut [u8],
    offset: u64,
) -> std::result::Result<(), PayloadReadError> {
    let expected = buffer.len();
    let mut actual = 0_usize;
    while actual < expected {
        let progress = u64::try_from(actual).map_err(|_| PayloadReadError::FileOffsetOverflow)?;
        let position = offset
            .checked_add(progress)
            .ok_or(PayloadReadError::FileOffsetOverflow)?;
        match positional_read(file, &mut buffer[actual..], position) {
            Ok(0) => return Err(PayloadReadError::ShortRead { expected, actual }),
            Ok(count) => {
                actual = actual
                    .checked_add(count)
                    .ok_or(PayloadReadError::BufferLengthOverflow)?;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(PayloadReadError::Io(error)),
        }
    }
    Ok(())
}

fn reauthenticate_shard(name: &str, shard: &OpenShard) -> Result<()> {
    verify_shard_descriptor(name, shard, "before reauthentication")?;
    verify_shard_path(name, shard, "before reauthentication")?;
    let actual_sha256 = hash_descriptor(&shard.file, shard.expected_size, &shard.path)?;
    verify_shard_descriptor(name, shard, "after reauthentication")?;
    verify_shard_path(name, shard, "after reauthentication")?;
    if actual_sha256 != shard.expected_sha256 {
        return Err(ArtifactError::new(format!(
            "verified shard '{name}' has SHA-256 {actual_sha256}, expected {} during reauthentication",
            shard.expected_sha256
        )));
    }
    Ok(())
}

fn verify_shard_descriptor(name: &str, shard: &OpenShard, phase: &str) -> Result<()> {
    let metadata = shard.file.metadata().map_err(|error| {
        ArtifactError::io("inspect retained shard descriptor", &shard.path, error)
    })?;
    if !metadata.is_file()
        || metadata.len() != shard.expected_size
        || !same_file(&metadata, &shard.authenticated_metadata)
    {
        return Err(ArtifactError::new(format!(
            "verified shard '{name}' descriptor identity or length changed {phase}"
        )));
    }
    Ok(())
}

fn verify_shard_path(name: &str, shard: &OpenShard, phase: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(&shard.path)
        .map_err(|error| ArtifactError::io("inspect retained shard path", &shard.path, error))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() != shard.expected_size
        || !same_file(&metadata, &shard.authenticated_metadata)
    {
        return Err(ArtifactError::new(format!(
            "verified shard '{name}' path no longer names its authenticated descriptor {phase}"
        )));
    }
    Ok(())
}

/// Hash a range without changing the descriptor's cursor.
pub fn hash_range(file: &File, mut offset: u64, mut bytes: u64) -> Result<String> {
    let mut hasher = Sha256State::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    while bytes != 0 {
        let count = bytes.min(buffer.len() as u64) as usize;
        read_exact_at(file, &mut buffer[..count], offset)
            .map_err(|error| ArtifactError::new(error.to_string()))?;
        hasher.update(&buffer[..count])?;
        offset = offset
            .checked_add(count as u64)
            .ok_or_else(|| ArtifactError::new("range hash offset overflow".into()))?;
        bytes -= count as u64;
    }
    Ok(hasher.finalize())
}

fn hash_descriptor(file: &File, size: u64, path: &Path) -> Result<String> {
    hash_range(file, 0, size).map_err(|error| {
        ArtifactError::new(format!("hash retained shard {}: {error}", path.display()))
    })
}

/// Scan the header of an opened safetensors file at its current cursor.
/// Model catalogs retain responsibility for tensor names, shapes and placement.
pub fn read_safetensors_header(mut file: &File) -> Result<ShardHeader> {
    let size = file
        .metadata()
        .map_err(|error| ArtifactError::new(error.to_string()))?
        .len();
    parse_safetensors_header(&mut file, size)
}

/// Open a regular input file without following a final symlink or blocking on a FIFO.
pub fn open_nofollow(path: &Path) -> io::Result<File> {
    open_regular_nofollow(path, fs::OpenOptions::new().read(true))
}

/// Apply no-follow and nonblocking flags to caller-selected read/write options,
/// then reject anything other than a regular file.
pub fn open_regular_nofollow(path: &Path, options: &mut fs::OpenOptions) -> io::Result<File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(io::Error::other(format!(
            "not a regular file: {}",
            path.display()
        )));
    }
    Ok(file)
}

#[cfg(unix)]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(windows)]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    left.volume_serial_number() == right.volume_serial_number()
        && left.file_index() == right.file_index()
}

#[cfg(not(any(unix, windows)))]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
        && left.created().ok() == right.created().ok()
}

#[cfg(unix)]
fn positional_read(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    file.read_at(buffer, offset)
}

#[cfg(windows)]
fn positional_read(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    file.seek_read(buffer, offset)
}

#[cfg(test)]
mod payload_read_tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn nonregular_file_errors_identify_the_path_without_assuming_input() {
        let path = std::env::temp_dir();
        let error = open_regular_nofollow(&path, fs::OpenOptions::new().read(true))
            .unwrap_err()
            .to_string();
        assert_eq!(error, format!("not a regular file: {}", path.display()));
    }

    #[test]
    fn range_hashes_cover_only_the_requested_bytes_without_moving_the_cursor() {
        let path =
            std::env::temp_dir().join(format!("runtime-weights-range-{}", std::process::id()));
        let bytes: Vec<_> = (0..1024 * 1024 + 37).map(|i| (i % 251) as u8).collect();
        fs::write(&path, &bytes).unwrap();
        let mut file = open_nofollow(&path).unwrap();
        file.seek(SeekFrom::Start(7)).unwrap();
        assert_eq!(
            hash_range(&file, 11, bytes.len() as u64 - 23).unwrap(),
            sha256_bytes(&bytes[11..bytes.len() - 12])
        );
        assert_eq!(file.stream_position().unwrap(), 7);
        assert_eq!(hash_range(&file, 17, 0).unwrap(), sha256_bytes(&[]));
        assert!(hash_range(&file, bytes.len() as u64 - 1, 2).is_err());
        drop(file);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reports_a_short_positional_read() {
        let path =
            std::env::temp_dir().join(format!("runtime-weights-short-read-{}", std::process::id()));
        fs::write(&path, [1_u8, 2]).unwrap();
        let file = File::open(&path).unwrap();
        let mut buffer = [0_u8; 3];
        let error = read_exact_at(&file, &mut buffer, 0).unwrap_err();
        assert!(matches!(
            error,
            PayloadReadError::ShortRead {
                expected: 3,
                actual: 2
            }
        ));
        drop(file);
        fs::remove_file(path).unwrap();
    }
}

fn open_verified_file(root: &Path, identity: &FileIdentity) -> Result<File> {
    let path = root.join(&identity.name);
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| ArtifactError::io("inspect pinned artifact", &path, error))?;
    if metadata.file_type().is_symlink() {
        return Err(ArtifactError::new(format!(
            "pinned artifact {} is a symlink",
            path.display()
        )));
    }
    if !metadata.is_file() {
        return Err(ArtifactError::new(format!(
            "pinned artifact {} is not a regular file",
            path.display()
        )));
    }
    let mut file = File::open(&path)
        .map_err(|error| ArtifactError::io("open pinned artifact", &path, error))?;
    let opened_metadata = file
        .metadata()
        .map_err(|error| ArtifactError::io("inspect opened artifact", &path, error))?;
    if !opened_metadata.is_file() || opened_metadata.len() != metadata.len() {
        return Err(ArtifactError::new(format!(
            "pinned artifact {} changed while it was opened",
            path.display()
        )));
    }
    if opened_metadata.len() != identity.size {
        return Err(ArtifactError::new(format!(
            "pinned artifact {} has size {}, expected {}",
            path.display(),
            opened_metadata.len(),
            identity.size
        )));
    }
    let actual_hash = hash::reader(&mut file)?;
    if actual_hash != identity.sha256 {
        return Err(ArtifactError::new(format!(
            "pinned artifact {} has SHA-256 {actual_hash}, expected {}",
            path.display(),
            identity.sha256
        )));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| ArtifactError::io("rewind verified artifact", &path, error))?;
    Ok(file)
}

fn validate_file_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    let mut components = path.components();
    let valid_component = matches!(components.next(), Some(Component::Normal(_)));
    if name.is_empty()
        || name.contains(['/', '\\'])
        || !valid_component
        || components.next().is_some()
        || path.file_name().and_then(|value| value.to_str()) != Some(name)
    {
        return Err(ArtifactError::new(format!(
            "artifact path '{name}' must be one portable file name"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardOwnershipPlan {
    topology: ShardTopology,
    assignments: BTreeMap<String, LocalRank>,
}

impl ShardOwnershipPlan {
    #[must_use]
    pub fn owner(&self, shard: &str) -> Option<LocalRank> {
        self.assignments.get(shard).copied()
    }

    #[must_use]
    pub fn assignments(&self) -> &BTreeMap<String, LocalRank> {
        &self.assignments
    }

    #[must_use]
    pub const fn topology(&self) -> ShardTopology {
        self.topology
    }
}

pub fn plan_shard_ownership(manifest: &LoadManifest) -> Result<ShardOwnershipPlan> {
    let rank_count = usize::try_from(manifest.topology.rank_count)
        .expect("supported local rank counts fit usize");
    let assignments = manifest
        .shards
        .iter()
        .enumerate()
        .map(|(index, shard)| {
            (
                shard.identity.name.clone(),
                manifest
                    .topology
                    .rank(u32::try_from(index % rank_count).expect("local rank fits u32"))
                    .expect("modulo rank count produces a valid local rank"),
            )
        })
        .collect();
    Ok(ShardOwnershipPlan {
        topology: manifest.topology,
        assignments,
    })
}

#[must_use]
pub fn sha256_bytes(bytes: &[u8]) -> String {
    hash::bytes(bytes)
}
