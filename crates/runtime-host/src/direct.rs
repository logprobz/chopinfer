//! Direct checkpoint loading: a bundle's weight plan executed into rank arenas.
//!
//! The host describes a safetensors checkpoint to the bundle (headers, small
//! documents, identity), receives one plan per image and rank, and serves the planned
//! compiled image like a packaged one: every read of a rank image range runs
//! the records covering it on worker threads, straight into the caller's
//! buffer (the pinned upload chunk). Plain copies read from the checkpoint
//! into that buffer; fragment records read the rows they pack into a worker's
//! scratch first. Records that several ranks share byte for byte, and records
//! only produced whole, are produced once and copied into each rank. A
//! read-ahead thread per rank advises the kernel of the source ranges two
//! chunks ahead, so page-cache reclaim happens off the packing threads.
//! With direct reads, a source range not wholly in the page cache is read
//! with `O_DIRECT` instead, past the cache's copy and reclaim; ranges the
//! cache holds are still read from it.
//!
//! Identity is the digest of the safetensors headers and file lengths; tensor
//! bytes are not hashed on this path.

use std::collections::BTreeMap;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, channel, sync_channel};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

mod io_stats;
mod packed_sources;
mod plan_cache;
mod stats;
mod workers;
use packed_sources::{PackedBuffers, PackedSources};
pub use stats::DirectLoadStatsV1;
use stats::SourceRanges;

/// Additional timing is opt-in so ordinary reads avoid extra clock calls.
pub(crate) fn detailed_timing_v1() -> bool {
    std::env::var_os("CHOPINFER_DIRECT_LOAD_TIMING").is_some_and(|value| value == "1")
}

use kernel_core::weight_transform::{
    PreparedRecord, SourceTensor, WeightRecord, WeightTransform, WeightTransformError,
};
use runtime_bundle_abi::weight_plan::{
    BundleWeightTransformFnV1, CheckpointFileV1, NamedBytesV1, PlannedViewV1, ViewPlacementV1,
    WeightPlanImplementationV1, WeightPlanRequestV1, WeightPlanV1, WeightRecordV1,
};
use runtime_weights::checkpoint::{CheckpointFile, header_identity};

use crate::host_rows::MappedRowsV1;
use crate::page_cache::{CacheAdviceV1, advise_v1, cached_pages_v1, direct_io_alignment_v1};

use crate::services::{
    CompiledImageDescriptionV1, CompiledImageViewV1, CompiledRankImageV1, CompiledViewSpanV1,
};

/// The largest JSON document the host reads from a checkpoint directory.
pub const DOCUMENT_BYTES_MAX_V1: u64 = 64 << 20;
/// The largest other non-safetensors file the host describes as a document,
/// such as a small tensor archive a plan stores inline after checking it.
pub const SMALL_FILE_BYTES_MAX_V1: u64 = 64 << 10;
/// Output bytes one worker task produces at most.
const TASK_BYTES: u64 = 4 << 20;
/// The largest replicated record produced once and held for the other ranks.
const SHARED_RECORD_BYTES_MAX: u64 = 64 << 20;
/// The largest source tensor read once for the several records (typically
/// one per rank) that each keep a slice of it along an inner axis.
const SHARED_SOURCE_BYTES_MAX: u64 = 16 << 20;
/// How far ahead of the current upload chunk the read-ahead threads advise.
const READ_AHEAD_BYTES: u64 = 128 << 20;
/// Read-ahead targets queued per rank while its thread advises; a read
/// finding the queue full sends none (the next target covers it).
const READ_AHEAD_PENDING: usize = 4;
/// The most source bytes one read-ahead advice call covers; a thread whose
/// image was dropped stops between calls.
const ADVICE_BYTES: u64 = 8 << 20;
/// How long dropping an image waits for its read-ahead threads to stop.
/// Past it, a thread still in an advice call is left to finish that call
/// alone and issues no other. That call may bring pages into the cache
/// after the drop: production never evicts a checkpoint's pages, so it only
/// warms the cache; a caller that evicts waits for
/// [`DirectCheckpointV1::wait_for_read_ahead`] first.
const READ_AHEAD_STOP_WAIT: Duration = Duration::from_secs(1);
/// Direct reads align their buffer, offset and length to this, unless the
/// file system reports its own alignment (`statx` `STATX_DIOALIGN`).
const DIRECT_READ_ALIGNMENT: u64 = 4096;
/// A file whose direct I/O needs a larger alignment is read through the
/// page cache.
const DIRECT_READ_ALIGNMENT_MAX: u64 = 1 << 20;
/// The most bytes one direct read moves through a worker's aligned buffer.
const DIRECT_READ_BYTES: u64 = 4 << 20;
/// A source range the page cache holds only in part is halved down to this
/// size, so its cached runs are still read from the cache.
const RESIDENCY_BYTES: u64 = 256 << 10;

/// One safetensors file of the checkpoint.
struct ShardV1 {
    name: String,
    file: CheckpointFile,
    /// The same file opened for direct reads, on first use; `None` when the
    /// file system refuses `O_DIRECT`.
    direct: OnceLock<Option<DirectFileV1>>,
    byte_length: u64,
    data_offset: u64,
    header: Vec<u8>,
    tensors: BTreeMap<String, (SourceTensor, u64)>,
}

/// A count of running read-ahead threads, and a signal as each exits.
#[derive(Default)]
struct AdvisersV1 {
    running: Mutex<usize>,
    exited: Condvar,
}

/// One running read-ahead thread, counted until dropped. Dropped, it
/// releases its image before it counts the thread out, so a waiter that
/// sees no thread running also sees none holding the image.
struct AdviserV1(Option<Arc<InnerV1>>);

impl AdviserV1 {
    fn new(inner: Arc<InnerV1>) -> Self {
        let advisers = &inner.checkpoint.advisers;
        *advisers
            .running
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
        Self(Some(inner))
    }

    fn inner(&self) -> &InnerV1 {
        self.0
            .as_deref()
            .expect("an adviser holds its image until dropped")
    }
}

impl Drop for AdviserV1 {
    fn drop(&mut self) {
        let inner = self.0.take().expect("an adviser is dropped once");
        let checkpoint = inner.checkpoint.clone();
        drop(inner);
        let advisers = &checkpoint.advisers;
        *advisers
            .running
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) -= 1;
        advisers.exited.notify_all();
    }
}

/// A checkpoint directory opened for direct loading.
pub struct DirectCheckpointV1 {
    root: PathBuf,
    shards: Vec<ShardV1>,
    documents: Vec<NamedBytesV1>,
    identity: [u8; 32],
    /// Read-ahead threads of images over this checkpoint still running,
    /// dropped images' included.
    advisers: AdvisersV1,
    /// Runs before each read-ahead advice call, as if inside it.
    #[cfg(test)]
    advice_hook: OnceLock<Box<dyn Fn() + Send + Sync>>,
    #[cfg(test)]
    read_refusal: OnceLock<Range<u64>>,
}

impl std::fmt::Debug for DirectCheckpointV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DirectCheckpointV1")
            .field("root", &self.root)
            .field("shards", &self.shards.len())
            .field("identity", &hex(&self.identity))
            .finish()
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn read_header(file: &CheckpointFile, byte_length: u64) -> Result<(u64, Vec<u8>), String> {
    let mut prefix = [0_u8; 8];
    file.read_exact_at(&mut prefix, 0)
        .map_err(|e| format!("read header length: {e}"))?;
    let length = u64::from_le_bytes(prefix);
    if length == 0 || length > 256 << 20 || 8 + length > byte_length {
        return Err("safetensors header length is outside the file".into());
    }
    let mut header = vec![0; length as usize];
    file.read_exact_at(&mut header, 8)
        .map_err(|e| format!("read header: {e}"))?;
    Ok((8 + length, header))
}

/// The names in `directory`, and whether each is a directory (not a link).
fn entries(directory: &Path) -> Result<Vec<(String, bool)>, String> {
    std::fs::read_dir(directory)
        .map_err(|e| format!("read checkpoint directory {}: {e}", directory.display()))?
        .map(|entry| {
            let entry = entry.map_err(|e| e.to_string())?;
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            let name = entry.file_name().into_string().map_err(|name| {
                format!(
                    "checkpoint name {} is not UTF-8",
                    Path::new(&name).display()
                )
            })?;
            Ok((name, kind.is_dir()))
        })
        .collect()
}

impl DirectCheckpointV1 {
    /// Opens every safetensors file, JSON document and other small file of
    /// `root` and of its immediate subdirectories (a drafter's or an audio
    /// tokenizer's, for example), named `dir/file`. Hidden directories are
    /// skipped.
    pub fn open(root: &Path) -> Result<Self, String> {
        let mut names = Vec::new();
        for (name, directory) in entries(root)? {
            if !directory {
                names.push((None, name));
            } else if !name.starts_with('.') {
                for (inner, nested) in entries(&root.join(&name))? {
                    if !nested {
                        names.push((Some(name.clone()), inner));
                    }
                }
            }
        }
        names.sort_by(|left, right| {
            let full = |(directory, name): &(Option<String>, String)| match directory {
                Some(directory) => format!("{directory}/{name}"),
                None => name.clone(),
            };
            full(left).cmp(&full(right))
        });
        let mut shards = Vec::new();
        let mut documents = Vec::new();
        for (directory, file_name) in names {
            let (parent, name) = match &directory {
                Some(directory) => (root.join(directory), format!("{directory}/{file_name}")),
                None => (root.to_path_buf(), file_name.clone()),
            };
            let open = || {
                CheckpointFile::open(&parent, &file_name)
                    .map_err(|e| format!("open checkpoint file {name}: {e}"))
            };
            if name.ends_with(".safetensors") {
                let file = open()?;
                let byte_length = file.metadata().map_err(|e| e.to_string())?.len();
                let (data_offset, header) =
                    read_header(&file, byte_length).map_err(|e| format!("{name}: {e}"))?;
                let mut prefix = ((data_offset - 8).to_le_bytes()).to_vec();
                prefix.extend_from_slice(&header);
                let parsed =
                    runtime_weights::parse_safetensors_header(&mut prefix.as_slice(), byte_length)
                        .map_err(|e| format!("{name}: {e}"))?;
                let tensors = parsed
                    .tensors()
                    .iter()
                    .map(|(tensor, info)| {
                        (
                            tensor.clone(),
                            (
                                SourceTensor {
                                    dtype: info.dtype().safetensors_name().into(),
                                    shape: info.shape().to_vec(),
                                    byte_length: info.byte_len(),
                                },
                                data_offset + info.data_offsets().0,
                            ),
                        )
                    })
                    .collect();
                shards.push(ShardV1 {
                    name,
                    file,
                    direct: OnceLock::new(),
                    byte_length,
                    data_offset,
                    header,
                    tensors,
                });
            } else {
                // JSON documents, and other small files a plan may check and
                // store; larger non-JSON files are skipped.
                let limit = if name.ends_with(".json") {
                    DOCUMENT_BYTES_MAX_V1
                } else {
                    SMALL_FILE_BYTES_MAX_V1
                };
                let Ok(metadata) = std::fs::metadata(parent.join(&file_name)) else {
                    continue;
                };
                if !metadata.is_file() || (limit != DOCUMENT_BYTES_MAX_V1 && metadata.len() > limit)
                {
                    continue;
                }
                let file = open()?;
                documents.push(NamedBytesV1 {
                    bytes: file
                        .read_bounded(limit)
                        .map_err(|e| format!("read {name}: {e}"))?,
                    name,
                });
            }
        }
        if shards.is_empty() {
            return Err(format!("{} holds no safetensors files", root.display()));
        }
        let identity = header_identity(shards.iter().map(|shard| {
            (
                shard.name.as_str(),
                shard.byte_length,
                shard.header.as_slice(),
            )
        }));
        Ok(Self {
            root: root.into(),
            shards,
            documents,
            identity,
            advisers: AdvisersV1::default(),
            #[cfg(test)]
            advice_hook: OnceLock::new(),
            #[cfg(test)]
            read_refusal: OnceLock::new(),
        })
    }

    /// Waits, without a limit, until no read-ahead thread of any image over
    /// this checkpoint runs, so none advises its pages into the cache
    /// after. Whoever evicts the checkpoint's pages calls it first.
    #[cfg(test)]
    fn wait_for_read_ahead(&self) {
        let mut running = self.advisers.running.lock().unwrap();
        while *running > 0 {
            running = self.advisers.exited.wait(running).unwrap();
        }
    }

    #[must_use]
    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }

    #[must_use]
    pub fn identity_hex(&self) -> String {
        hex(&self.identity)
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The request for one rank of a `ranks`-rank load of the compiled image
    /// `image_role`, executed by `implementation` ([`implementation_v1`]).
    #[must_use]
    pub fn request(
        &self,
        image_role: &str,
        ranks: u32,
        rank: u32,
        implementation: WeightPlanImplementationV1,
        components: Vec<NamedBytesV1>,
        host_views: Vec<String>,
    ) -> WeightPlanRequestV1 {
        WeightPlanRequestV1 {
            image_role: image_role.into(),
            ranks,
            rank,
            implementation,
            checkpoint_identity: self.identity,
            files: self
                .shards
                .iter()
                .map(|shard| CheckpointFileV1 {
                    name: shard.name.clone(),
                    byte_length: shard.byte_length,
                    data_offset: shard.data_offset,
                    header: shard.header.clone(),
                })
                .collect(),
            documents: self.documents.clone(),
            components,
            host_views,
        }
    }

    /// Refuses a checkpoint whose files moved, changed length or header since
    /// it was opened.
    pub fn verify_unchanged(&self) -> Result<(), String> {
        self.shards.iter().try_for_each(ShardV1::verify_unchanged)
    }
}

impl ShardV1 {
    /// Refuses a file that moved, changed length or header since it was
    /// opened.
    fn verify_unchanged(&self) -> Result<(), String> {
        self.file
            .verify()
            .map_err(|e| format!("checkpoint file {} changed: {e}", self.name))?;
        let length = self.file.metadata().map_err(|e| e.to_string())?.len();
        if length != self.byte_length || read_header(&self.file, length)?.1 != self.header {
            return Err(format!(
                "checkpoint file {} changed during the load",
                self.name
            ));
        }
        Ok(())
    }
}

/// The checkpoint bytes one mapped view names.
struct MappedSourceV1 {
    /// The plan's one record covering the view.
    record: usize,
    shard: usize,
    file_offset: u64,
    byte_length: u64,
}

/// Checks that exactly one record covers the mapped view `view` of `plan`,
/// and that it is a plain copy of one tensor, whole or one range of its
/// leading axis, stored unchanged; answers the file bytes it reads.
fn mapped_source_v1(
    checkpoint: &DirectCheckpointV1,
    plan: &WeightPlanV1,
    view: &PlannedViewV1,
) -> Result<MappedSourceV1, String> {
    let refuse = |reason: &str| {
        Err(format!(
            "rank {} mapped view {}: {reason}",
            plan.rank, view.id
        ))
    };
    let end = view.offset.saturating_add(view.byte_length);
    let mut covering = plan.records.iter().enumerate().filter(|(_, record)| {
        record.arena_offset < end
            && view.offset < record.arena_offset.saturating_add(record.byte_length)
    });
    let (Some((index, record)), None) = (covering.next(), covering.next()) else {
        return refuse("exactly one record must cover it");
    };
    if record.arena_offset != view.offset || record.byte_length != view.byte_length {
        return refuse("its record must cover exactly the view");
    }
    if WeightTransform::decode(record.transform, &record.params) != Ok(WeightTransform::Copy) {
        return refuse("its record must be a plain copy");
    }
    let [source] = &record.sources[..] else {
        return refuse("its record must read exactly one tensor");
    };
    let shard = checkpoint
        .shards
        .get(source.file as usize)
        .ok_or_else(|| format!("record names file {}", source.file))?;
    let (tensor, begin) = shard
        .tensors
        .get(&source.tensor)
        .ok_or_else(|| format!("tensor {} is not in {}", source.tensor, shard.name))?;
    let (rows, kept) = match (record.axis, &record.ranges[..]) {
        (None, []) => (
            0..tensor.shape.first().copied().unwrap_or(1),
            tensor.shape.clone(),
        ),
        (Some(0), [range]) => {
            let Some((&leading, rest)) = tensor.shape.split_first() else {
                return refuse("a range selects rows of a tensor with a leading axis");
            };
            if range.start >= range.end || range.end > leading {
                return refuse("its rows lie outside the tensor");
            }
            (
                range.clone(),
                [&[range.end - range.start][..], rest].concat(),
            )
        }
        _ => return refuse("it maps a whole tensor or one range of its leading axis"),
    };
    if record.stored_shape != kept {
        return refuse("it must store the rows it reads unchanged");
    }
    let leading = tensor.shape.first().copied().unwrap_or(1);
    if leading == 0 || !tensor.byte_length.is_multiple_of(leading) {
        return refuse("its tensor's bytes are not whole rows");
    }
    let row_bytes = tensor.byte_length / leading;
    let byte_length = (rows.end - rows.start) * row_bytes;
    if byte_length != view.byte_length {
        return refuse("its length is not its rows' bytes");
    }
    Ok(MappedSourceV1 {
        record: index,
        shard: source.file as usize,
        file_offset: begin + rows.start * row_bytes,
        byte_length,
    })
}

/// How a planned image is produced.
#[derive(Clone, Copy, Debug)]
pub struct DirectLoadOptionsV1 {
    /// Worker threads per rank read.
    pub threads_per_rank: usize,
    /// Whether read-ahead threads advise the kernel of upcoming source ranges.
    pub read_ahead: bool,
    /// Whether source ranges not wholly in the page cache are read with
    /// `O_DIRECT`.
    pub direct_reads: bool,
    /// Host bytes shared outputs may hold between their consumers, over all
    /// ranks.
    pub shared_bytes: u64,
}

impl DirectLoadOptionsV1 {
    /// The default budget for shared outputs held between consumers.
    pub const SHARED_BYTES_DEFAULT: u64 = 1 << 30;

    /// The most workers per rank the default gives.
    pub const DEFAULT_THREADS_PER_RANK_MAX: usize = 16;

    /// Workers per rank, by default twice the process's CPUs
    /// ([`process_cpus_v1`]) divided between the ranks that upload at
    /// once, at most [`Self::DEFAULT_THREADS_PER_RANK_MAX`]: workers spend
    /// most of a load waiting on checkpoint reads, and more reads in flight
    /// load a cold checkpoint faster. `serve --load-threads` and then
    /// `CHOPINFER_DIRECT_LOAD_THREADS` override the count.
    /// `CHOPINFER_DIRECT_LOAD_DIRECT_READS=0` turns direct reads off. Read-ahead
    /// is on only without them, since advising a cold range reads it into
    /// the cache they bypass; `CHOPINFER_DIRECT_LOAD_READ_AHEAD=0` or `=1`
    /// overrides that.
    #[must_use]
    pub fn for_ranks(ranks: usize) -> Self {
        let threads_per_rank = THREADS_PER_RANK.get().copied().unwrap_or_else(|| {
            std::env::var("CHOPINFER_DIRECT_LOAD_THREADS")
                .ok()
                .and_then(|value| value.parse().ok())
                .filter(|threads| *threads > 0)
                .unwrap_or_else(|| default_threads_per_rank_v1(process_cpus_v1(), ranks))
        });
        let direct_reads =
            std::env::var("CHOPINFER_DIRECT_LOAD_DIRECT_READS").as_deref() != Ok("0");
        Self {
            threads_per_rank,
            read_ahead: match std::env::var("CHOPINFER_DIRECT_LOAD_READ_AHEAD").as_deref() {
                Ok("0") => false,
                Ok("1") => true,
                _ => !direct_reads,
            },
            direct_reads,
            shared_bytes: std::env::var("CHOPINFER_DIRECT_LOAD_SHARED_BYTES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(Self::SHARED_BYTES_DEFAULT),
        }
    }
}

/// The CPUs this process may run on: its affinity set, less any cgroup quota.
#[must_use]
pub fn process_cpus_v1() -> usize {
    std::thread::available_parallelism().map_or(8, usize::from)
}

/// Twice `cpus` divided between `ranks`, from 1 to
/// [`DirectLoadOptionsV1::DEFAULT_THREADS_PER_RANK_MAX`].
#[must_use]
pub fn default_threads_per_rank_v1(cpus: usize, ranks: usize) -> usize {
    (2 * cpus / ranks.max(1)).clamp(1, DirectLoadOptionsV1::DEFAULT_THREADS_PER_RANK_MAX)
}

/// One source tensor of a record: its shard and absolute file offset.
#[derive(Clone, Copy, Debug)]
struct SourceAtV1 {
    shard: usize,
    begin: u64,
    length: u64,
}

struct RecordV1 {
    consumer: usize,
    packed_sources: Vec<Option<usize>>,
    /// A matching code section is part of the image admitted by full upload.
    /// Scale-only plans must read every referenced source themselves.
    has_code_section: bool,
    offset: u64,
    /// A bundle transform's parameters, passed through unread.
    bundle_params: Vec<u8>,
    prepared: PreparedRecord,
    sources: Vec<SourceAtV1>,
    /// The shared output this record copies from, if any.
    shared: Option<usize>,
    /// The shared source tensor this record reads its slice from, if any.
    source_share: Option<usize>,
}

/// A source tensor several records (the ranks' K slices of one matrix) each
/// read a strided part of: read whole once and held while it still owes
/// those records bytes and the shared budget has room.
struct SharedSourceV1 {
    state: Mutex<SharedSourceStateV1>,
}

struct SharedSourceStateV1 {
    bytes: Option<Arc<RetainedBytesV1>>,
    /// Output bytes its records have yet to produce.
    remaining: u64,
}

/// Bytes held against the shared budget: a shared source tensor or a
/// shared output's section. Tests count the bytes held until they are freed.
struct RetainedBytesV1 {
    bytes: Vec<u8>,
    range: Range<usize>,
    charge: u64,
    pool: Option<Arc<PackedBuffers>>,
    #[cfg(test)]
    held: Arc<HeldBytesV1>,
}

impl RetainedBytesV1 {
    fn new(inner: &InnerV1, bytes: Vec<u8>) -> Self {
        Self::make(inner, bytes, false)
    }

    fn pooled(inner: &InnerV1, bytes: Vec<u8>) -> Self {
        Self::make(inner, bytes, true)
    }

    fn make(inner: &InnerV1, bytes: Vec<u8>, pooled: bool) -> Self {
        let charge = if pooled {
            bytes.capacity()
        } else {
            bytes.len()
        } as u64;
        #[cfg(test)]
        {
            let now = inner.retained_held.now.fetch_add(charge, Ordering::AcqRel);
            inner
                .retained_held
                .peak
                .fetch_max(now + charge, Ordering::AcqRel);
        }
        Self {
            range: 0..bytes.len(),
            bytes,
            charge,
            pool: pooled.then(|| Arc::clone(&inner.packed_buffers)),
            #[cfg(test)]
            held: inner.retained_held.clone(),
        }
    }
}

impl Drop for RetainedBytesV1 {
    fn drop(&mut self) {
        #[cfg(test)]
        self.held.now.fetch_sub(self.charge, Ordering::AcqRel);
        if let Some(pool) = &self.pool {
            pool.give(std::mem::take(&mut self.bytes));
        }
    }
}

/// Bytes held against the shared budget now, and the most held at once; a
/// pause after each reservation is returned widens the window in which
/// another tensor or section may reserve.
#[cfg(test)]
#[derive(Default)]
struct HeldBytesV1 {
    now: AtomicU64,
    peak: AtomicU64,
    release_pause_micros: AtomicU64,
}

#[cfg(test)]
impl HeldBytesV1 {
    fn pause(&self) {
        let micros = self.release_pause_micros.load(Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_micros(micros));
    }
}

/// A record output produced once and copied into every rank that places it.
struct SharedV1 {
    state: Mutex<SharedStateV1>,
}

/// A shared output's sections (a packed operand's three, or one) and the
/// bytes each still owes its consumers. A section is kept only while some
/// consumer still needs it and the budget has room; otherwise the next
/// consumer produces the record again.
struct SharedStateV1 {
    sections: [Option<Arc<RetainedBytesV1>>; 3],
    remaining: [u64; 3],
    produced_sections: [bool; 3],
}

struct RankV1 {
    arena_bytes: u64,
    records: Vec<RecordV1>,
}

struct InnerV1 {
    checkpoint: Arc<DirectCheckpointV1>,
    ranks: Vec<RankV1>,
    offsets: Vec<u64>,
    shared: Vec<SharedV1>,
    shared_sources: Vec<SharedSourceV1>,
    packed_sources: PackedSources,
    packed_buffers: Arc<PackedBuffers>,
    transform: Option<BundleWeightTransformFnV1>,
    options: DirectLoadOptionsV1,
    source_bytes: AtomicU64,
    /// Positional reads of the checkpoint.
    source_reads: AtomicU64,
    file_read_bytes: AtomicU64,
    unique_source_bytes: AtomicU64,
    source_ranges: Option<Mutex<SourceRanges>>,
    source_cache_hits: AtomicU64,
    source_cache_bytes: AtomicU64,
    shared_output_hits: AtomicU64,
    shared_output_productions: AtomicU64,
    shared_output_reproductions: AtomicU64,
    produced_bytes: AtomicU64,
    advised_bytes: AtomicU64,
    /// Set when the image is dropped: read-ahead threads advise nothing more.
    read_ahead_stopped: AtomicBool,
    /// Source bytes read with `O_DIRECT`.
    direct_bytes: AtomicU64,
    /// Nanoseconds in source reads, in record production (reads included),
    /// waiting for or producing shared outputs, and in rank reads.
    read_nanos: AtomicU64,
    produce_nanos: AtomicU64,
    shared_nanos: AtomicU64,
    rank_nanos: AtomicU64,
    detailed_timing: bool,
    io_timing: Option<io_stats::IoCounters>,
    output_wait_nanos: AtomicU64,
    materialize_nanos: AtomicU64,
    input_wait_nanos: AtomicU64,
    /// Bytes shared outputs hold now, and the most they held.
    retained: AtomicU64,
    retained_peak: AtomicU64,
    first_read: OnceLock<Instant>,
    #[cfg(test)]
    retained_held: Arc<HeldBytesV1>,
    /// Worker scratch buffers, kept across reads: allocating them per read
    /// maps, faults and unmaps large buffers, and the unmaps serialize the
    /// other workers' page faults.
    scratch: Mutex<Vec<Vec<u8>>>,
    workers: Mutex<Option<Arc<workers::Workers>>>,
    /// Workers' aligned buffers for direct reads, kept the same way.
    direct_buffers: Mutex<Vec<Vec<u8>>>,
}

/// What warming the prewarmed mapped views at load did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrewarmV1 {
    /// Bytes the prewarmed mappings map, each counted once.
    pub bytes: u64,
    pub seconds: f64,
    /// Their resident bytes afterwards, in whole pages.
    pub resident_bytes: u64,
    /// `MemAvailable` before warming, when the host reports it.
    pub available_bytes: Option<u64>,
    /// The memory the host wants left beyond the prewarmed bytes.
    pub margin_bytes: u64,
}

/// Threads that prewarm mapped views, at most.
const PREWARM_THREADS: usize = 8;

/// Bytes one prewarm step reads ahead and touches.
const PREWARM_WINDOW_BYTES: usize = 64 << 20;

/// The memory the host wants left beyond the prewarmed bytes, by default:
/// room for the rest of the process and the system while the page cache
/// holds the tables. `CHOPINFER_PREWARM_MARGIN_BYTES` overrides it.
pub const PREWARM_MARGIN_BYTES_DEFAULT: u64 = 16 << 30;

/// `MemAvailable` in bytes, from `/proc/meminfo`.
fn available_memory_v1() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

/// Reads `mappings` into the page cache and maps them, window by window on
/// up to [`PREWARM_THREADS`] threads, after checking that the host has room
/// for them: when `MemAvailable` falls short of their bytes plus the margin,
/// it warns by name and warms anyway, since gathers stay correct when pages
/// are evicted, only slower.
fn prewarm_v1(mappings: &[Arc<MappedRowsV1>]) -> Result<PrewarmV1, String> {
    let bytes: u64 = mappings.iter().map(|rows| rows.len()).sum();
    let margin_bytes = std::env::var("CHOPINFER_PREWARM_MARGIN_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(PREWARM_MARGIN_BYTES_DEFAULT);
    let available_bytes = available_memory_v1();
    if let Some(available) = available_bytes
        && available < bytes.saturating_add(margin_bytes)
    {
        eprintln!(
            "startup prewarm: warning host-ram-short: {available} bytes available, below the \
             {bytes} prewarmed bytes plus a {margin_bytes}-byte margin; gathers read evicted \
             rows from the file"
        );
    }
    let started = Instant::now();
    let windows: Vec<(&MappedRowsV1, usize, usize)> = mappings
        .iter()
        .flat_map(|rows| {
            rows.windows(PREWARM_WINDOW_BYTES)
                .into_iter()
                .map(move |(start, length)| (rows.as_ref(), start, length))
        })
        .collect();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(PREWARM_THREADS)
        .min(windows.len());
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| -> std::io::Result<()> {
                    while let Some(&(rows, start, length)) =
                        windows.get(next.fetch_add(1, Ordering::Relaxed))
                    {
                        rows.prewarm_window(start, length)?;
                    }
                    Ok(())
                })
            })
            .collect();
        workers.into_iter().try_for_each(|worker| {
            worker
                .join()
                .map_err(|_| "a prewarm thread panicked".to_owned())?
                .map_err(|e| format!("prewarm mapped views: {e}"))
        })
    })?;
    let page = crate::host_rows::page_bytes_v1() as u64;
    let resident_bytes = mappings
        .iter()
        .map(|rows| rows.resident_pages().map(|(resident, _)| resident * page))
        .sum::<std::io::Result<u64>>()
        .map_err(|e| format!("prewarm residency: {e}"))?;
    Ok(PrewarmV1 {
        bytes,
        seconds: started.elapsed().as_secs_f64(),
        resident_bytes,
        available_bytes,
        margin_bytes,
    })
}

/// A compiled image produced from the checkpoint by the bundle's plans.
pub struct PlannedImageV1 {
    inner: Arc<InnerV1>,
    description: CompiledImageDescriptionV1,
    image_role: String,
    components: Vec<NamedBytesV1>,
    /// Each rank's image identity, hex SHA-256 of its plan's identity document.
    identities: Vec<String>,
    /// The mapped views' bytes, by rank and view identifier; ranks naming the
    /// same bytes share one mapping.
    mapped: BTreeMap<(u32, Vec<u8>), Arc<MappedRowsV1>>,
    /// What prewarming the mapped views that ask for it did, if any do.
    prewarm: Option<PrewarmV1>,
    /// Each rank's read-ahead queue: the arena offsets to advise up to.
    read_ahead: Vec<SyncSender<u64>>,
    /// The read-ahead threads, stopped on drop so a dropped image no longer
    /// advises the kernel of its checkpoint's pages.
    read_ahead_threads: Vec<std::thread::JoinHandle<()>>,
    /// Disconnected once every read-ahead thread has exited.
    read_ahead_exits: Mutex<Receiver<()>>,
}

impl Drop for PlannedImageV1 {
    /// Stops the read-ahead threads, waiting at most [`READ_AHEAD_STOP_WAIT`]
    /// for one blocked in an advice call.
    fn drop(&mut self) {
        self.inner.read_ahead_stopped.store(true, Ordering::Release);
        self.read_ahead.clear();
        let exits = self
            .read_ahead_exits
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match exits.recv_timeout(READ_AHEAD_STOP_WAIT) {
            Err(RecvTimeoutError::Disconnected) => {
                for thread in self.read_ahead_threads.drain(..) {
                    let _ = thread.join();
                }
            }
            Ok(()) | Err(RecvTimeoutError::Timeout) => eprintln!(
                "direct load: a read-ahead thread is still advising after {READ_AHEAD_STOP_WAIT:?}; leaving it to finish that call"
            ),
        }
    }
}

impl std::fmt::Debug for PlannedImageV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PlannedImageV1")
            .field("image_role", &self.image_role)
            .field("ranks", &self.inner.ranks.len())
            .finish()
    }
}

fn transform_error(record: u64) -> impl Fn(WeightTransformError) -> String {
    move |error| format!("record at arena offset {record}: {error}")
}

include!(concat!(env!("OUT_DIR"), "/direct_sources.rs"));

/// SHA-256 of the host's shared transform implementation: every source on
/// the direct load's byte path ([`DIRECT_SOURCES`], listed and digested by
/// `build.rs`: the whole source trees, build scripts and manifests of this
/// crate and of every workspace crate it links, transitively, and
/// `Cargo.lock`), by name, length and bytes. Any change to them gives every
/// planned rank a new identity.
#[must_use]
pub fn shared_transform_implementation_v1() -> [u8; 32] {
    DIRECT_SOURCES_DIGEST
}

/// The implementation a host states in its plan requests and covers in its
/// planned identities: its shared transforms, and `bundle_module` (the
/// SHA-256 of the bundle's native module, zero without one) for plans with
/// bundle-owned transforms.
#[must_use]
pub fn implementation_v1(bundle_module: [u8; 32]) -> WeightPlanImplementationV1 {
    WeightPlanImplementationV1 {
        shared_transforms: shared_transform_implementation_v1(),
        bundle_module,
    }
}

impl PlannedImageV1 {
    /// Checks one plan per rank against the checkpoint and prepares every
    /// record. `transform` is the bundle's transform export, needed only by
    /// plans with bundle-owned transforms; `implementation` is what the
    /// requests stated, which each rank's identity covers.
    pub fn new(
        checkpoint: Arc<DirectCheckpointV1>,
        plans: Vec<WeightPlanV1>,
        transform: Option<BundleWeightTransformFnV1>,
        implementation: WeightPlanImplementationV1,
        options: DirectLoadOptionsV1,
    ) -> Result<Self, String> {
        Self::new_with_range_tracking(
            checkpoint,
            &plans,
            transform,
            implementation,
            options,
            std::env::var_os("CHOPINFER_DIRECT_LOAD_TRACK_RANGES")
                .is_some_and(|value| value == "1"),
        )
    }

    fn new_with_range_tracking(
        checkpoint: Arc<DirectCheckpointV1>,
        plans: &[WeightPlanV1],
        transform: Option<BundleWeightTransformFnV1>,
        implementation: WeightPlanImplementationV1,
        options: DirectLoadOptionsV1,
        track_ranges: bool,
    ) -> Result<Self, String> {
        let first = plans.first().ok_or("no rank plans")?;
        // An identity that covered no module would let two bundle builds
        // share it for different bytes.
        if implementation.bundle_module == [0; 32]
            && plans.iter().any(WeightPlanV1::uses_bundle_transforms)
        {
            return Err(
                "a plan with bundle-owned transforms needs the bundle module's digest".into(),
            );
        }
        let (image_role, alignment, count) = (
            first.image_role.clone(),
            first.alignment,
            first.ranks as usize,
        );
        if plans.len() != count
            || plans.iter().enumerate().any(|(rank, plan)| {
                plan.rank as usize != rank
                    || plan.ranks as usize != count
                    || plan.image_role != image_role
                    || plan.alignment != alignment
            })
        {
            return Err("rank plans disagree on the rank count, role or alignment".into());
        }
        // A mapped view's record maps checkpoint bytes instead of producing
        // them, so no rank read executes it.
        let mut mapped_sources = Vec::new();
        let mut unproduced: Vec<std::collections::BTreeSet<usize>> =
            vec![Default::default(); count];
        for plan in plans {
            for view in plan
                .views
                .iter()
                .filter(|view| matches!(view.placement, ViewPlacementV1::Mapped { .. }))
            {
                let source = mapped_source_v1(&checkpoint, plan, view)?;
                unproduced[plan.rank as usize].insert(source.record);
                let prewarm = view.placement == ViewPlacementV1::Mapped { prewarm: true };
                mapped_sources.push((plan.rank, view.id.clone().into_bytes(), source, prewarm));
            }
        }
        let mut shared_keys: BTreeMap<String, usize> = BTreeMap::new();
        let mut shared_users: Vec<([u64; 3], usize)> = Vec::new();
        let mut ranks = Vec::with_capacity(count);
        let mut components: Vec<NamedBytesV1> = Vec::new();
        let mut views: BTreeMap<
            Vec<u8>,
            (u64, ViewPlacementV1, BTreeMap<u32, CompiledViewSpanV1>),
        > = BTreeMap::new();
        let mut offsets = Vec::with_capacity(count);
        let mut next = 0_u64;
        // Each rank's records are checked and prepared, and its identity
        // digested, on its own thread; grouping shared records stays in rank
        // order, so the result does not depend on the threads.
        let prepared_ranks = std::thread::scope(|scope| {
            let handles: Vec<_> = plans
                .iter()
                .map(|plan| {
                    let checkpoint = &checkpoint;
                    let implementation = &implementation;
                    let unproduced = &unproduced[plan.rank as usize];
                    scope.spawn(move || -> Result<_, String> {
                        let identity = runtime_weights::sha256_bytes(
                            &plan.identity_document(&checkpoint.identity, implementation),
                        );
                        let records = plan
                            .records
                            .iter()
                            .enumerate()
                            .filter(|(index, _)| !unproduced.contains(index))
                            .map(|(_, record)| prepare_record_v1(checkpoint, transform, record))
                            .collect::<Result<Vec<_>, String>>()?;
                        Ok((identity, records))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err("record preparation panicked".into()))
                })
                .collect::<Result<Vec<_>, String>>()
        })?;
        let mut identities = Vec::with_capacity(count);
        for (plan, (identity, prepared)) in plans.iter().zip(prepared_ranks) {
            identities.push(identity);
            offsets.push(next);
            next += plan.arena_bytes;
            for component in &plan.components {
                if components.iter().any(|other| other.name == component.name) {
                    return Err(format!("component {} is planned twice", component.name));
                }
                components.push(component.clone());
            }
            for view in &plan.views {
                let entry = views.entry(view.id.clone().into_bytes()).or_insert((
                    view.alignment,
                    view.placement,
                    BTreeMap::new(),
                ));
                if entry.0 != view.alignment || entry.1 != view.placement {
                    return Err(format!("view {} differs between ranks", view.id));
                }
                entry.2.insert(
                    plan.rank,
                    CompiledViewSpanV1 {
                        offset: view.offset,
                        byte_length: view.byte_length,
                    },
                );
            }
            let mut records = Vec::with_capacity(prepared.len());
            for (key, record) in prepared {
                let group = *shared_keys.entry(key).or_insert_with(|| {
                    shared_users.push(([0; 3], 0));
                    shared_users.len() - 1
                });
                let length = record.prepared.byte_length();
                // Large streamed records never consume shared sections. Counting
                // them here would retain an output no shared reader can release.
                if record.prepared.is_whole() || length <= SHARED_RECORD_BYTES_MAX {
                    shared_users[group].0[record.prepared.section()] += length;
                }
                shared_users[group].1 += 1;
                records.push((group, record));
            }
            let code_groups: std::collections::BTreeSet<_> = records
                .iter()
                .filter(|(_, record)| record.prepared.section() == 0)
                .map(|(group, _)| *group)
                .collect();
            for (group, record) in &mut records {
                record.has_code_section = code_groups.contains(group);
            }
            ranks.push((plan.arena_bytes, records));
        }
        // A group is shared when it is produced whole, or when several ranks
        // place it and it is at most a staging chunk; a larger replicated
        // record is produced by range in each rank instead of held whole.
        let mut shared = Vec::new();
        let mut slot = vec![None; shared_users.len()];
        let ranks = ranks
            .into_iter()
            .map(|(arena_bytes, records)| RankV1 {
                arena_bytes,
                records: records
                    .into_iter()
                    .map(|(group, mut record)| {
                        if record.prepared.is_whole()
                            || (shared_users[group].1 > 1
                                && record.prepared.byte_length() <= SHARED_RECORD_BYTES_MAX)
                        {
                            let index = *slot[group].get_or_insert_with(|| {
                                shared.push(SharedV1 {
                                    state: Mutex::new(SharedStateV1 {
                                        sections: [None, None, None],
                                        remaining: shared_users[group].0,
                                        produced_sections: [false; 3],
                                    }),
                                });
                                shared.len() - 1
                            });
                            record.shared = Some(index);
                        }
                        record
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let (mut ranks, shared_sources) = share_sources_v1(ranks);
        let packed_sources = PackedSources::new(&mut ranks);
        let description = CompiledImageDescriptionV1::new(
            alignment,
            ranks
                .iter()
                .zip(&offsets)
                .map(|(rank, offset)| CompiledRankImageV1 {
                    offset: *offset,
                    byte_length: rank.arena_bytes,
                })
                .collect(),
            views
                .into_iter()
                .map(|(id, (alignment, placement, spans))| {
                    let view = CompiledImageViewV1::new(alignment, spans);
                    let view = match placement {
                        ViewPlacementV1::Device => view,
                        ViewPlacementV1::Host => view.on_host(),
                        ViewPlacementV1::Mapped { .. } => view.mapped(),
                    };
                    (id, view)
                })
                .collect(),
        );
        // Each mapping reads the file as its header was when it was opened.
        let mut mapped = BTreeMap::new();
        let mut prewarmed: Vec<Arc<MappedRowsV1>> = Vec::new();
        for (rank, view, source, prewarm) in mapped_sources {
            let shard = &checkpoint.shards[source.shard];
            shard.verify_unchanged()?;
            let rows = crate::host_rows::map_authenticated_rows_v1(
                &shard.file,
                source.file_offset,
                source.byte_length,
                checkpoint.identity(),
            )
            .map_err(|e| {
                format!(
                    "map {} for view {}: {e}",
                    shard.name,
                    String::from_utf8_lossy(&view)
                )
            })?;
            if prewarm && !prewarmed.iter().any(|known| Arc::ptr_eq(known, &rows)) {
                prewarmed.push(Arc::clone(&rows));
            }
            mapped.insert((rank, view), rows);
        }
        let prewarm = (!prewarmed.is_empty())
            .then(|| prewarm_v1(&prewarmed))
            .transpose()?;
        let inner = Arc::new(InnerV1 {
            checkpoint,
            ranks,
            offsets,
            shared,
            shared_sources,
            packed_sources,
            packed_buffers: Arc::new(PackedBuffers::new(options.shared_bytes)),
            transform,
            options,
            source_bytes: AtomicU64::new(0),
            source_reads: AtomicU64::new(0),
            file_read_bytes: AtomicU64::new(0),
            unique_source_bytes: AtomicU64::new(0),
            source_ranges: track_ranges.then(Mutex::default),
            source_cache_hits: AtomicU64::new(0),
            source_cache_bytes: AtomicU64::new(0),
            shared_output_hits: AtomicU64::new(0),
            shared_output_productions: AtomicU64::new(0),
            shared_output_reproductions: AtomicU64::new(0),
            produced_bytes: AtomicU64::new(0),
            advised_bytes: AtomicU64::new(0),
            read_ahead_stopped: AtomicBool::new(false),
            direct_bytes: AtomicU64::new(0),
            read_nanos: AtomicU64::new(0),
            produce_nanos: AtomicU64::new(0),
            shared_nanos: AtomicU64::new(0),
            rank_nanos: AtomicU64::new(0),
            detailed_timing: detailed_timing_v1(),
            io_timing: detailed_timing_v1().then(io_stats::IoCounters::default),
            output_wait_nanos: AtomicU64::new(0),
            materialize_nanos: AtomicU64::new(0),
            input_wait_nanos: AtomicU64::new(0),
            retained: AtomicU64::new(0),
            retained_peak: AtomicU64::new(0),
            first_read: OnceLock::new(),
            #[cfg(test)]
            retained_held: Arc::default(),
            scratch: Mutex::new(Vec::new()),
            workers: Mutex::new(None),
            direct_buffers: Mutex::new(Vec::new()),
        });
        let mut read_ahead = Vec::with_capacity(count);
        let mut read_ahead_threads = Vec::new();
        let (exited, read_ahead_exits) = channel();
        for rank in 0..count {
            let (sender, receiver) = sync_channel::<u64>(READ_AHEAD_PENDING);
            if options.read_ahead {
                let adviser = AdviserV1::new(inner.clone());
                let exited = exited.clone();
                let thread = std::thread::Builder::new()
                    .name(format!("chopinfer-read-ahead-{rank}"))
                    .spawn(move || {
                        let _exited = exited;
                        read_ahead_v1(adviser.inner(), rank, &receiver);
                    })
                    .map_err(|e| format!("start read-ahead thread: {e}"))?;
                read_ahead_threads.push(thread);
                let _ = sender.try_send(READ_AHEAD_BYTES);
            }
            read_ahead.push(sender);
        }
        drop(exited);
        Ok(Self {
            inner,
            description,
            image_role,
            components,
            identities,
            mapped,
            prewarm,
            read_ahead,
            read_ahead_threads,
            read_ahead_exits: Mutex::new(read_ahead_exits),
        })
    }

    /// Rank `rank`'s identity: the hex SHA-256 of its plan's
    /// [`WeightPlanV1::identity_document`], served as the rank's
    /// `direct_identity` where a rank file's digest would be.
    #[must_use]
    pub fn identity(&self, rank: usize) -> Option<&str> {
        self.identities.get(rank).map(String::as_str)
    }

    #[must_use]
    pub fn image_role(&self) -> &str {
        &self.image_role
    }

    #[must_use]
    pub fn description(&self) -> &CompiledImageDescriptionV1 {
        &self.description
    }

    /// The authenticated components the plans declare, by role.
    #[must_use]
    pub fn components(&self) -> &[NamedBytesV1] {
        &self.components
    }

    #[must_use]
    pub fn checkpoint(&self) -> &DirectCheckpointV1 {
        &self.inner.checkpoint
    }

    /// The bytes of rank `rank`'s mapped view `view`, if the plans map one.
    #[must_use]
    pub fn mapped_view(&self, rank: u32, view: &[u8]) -> Option<&Arc<MappedRowsV1>> {
        self.mapped.get(&(rank, view.to_vec()))
    }

    /// What prewarming the mapped views that ask for it did at load, if any
    /// do.
    #[must_use]
    pub const fn prewarm(&self) -> Option<PrewarmV1> {
        self.prewarm
    }

    /// Bytes the mapped views map, each mapping counted once however many
    /// ranks share it. Neither the device nor page-locked memory holds them.
    #[must_use]
    pub fn mapped_bytes(&self) -> u64 {
        let mut seen: Vec<*const MappedRowsV1> = Vec::new();
        self.mapped
            .values()
            .filter(|rows| {
                let pointer = Arc::as_ptr(rows);
                let fresh = !seen.contains(&pointer);
                seen.push(pointer);
                fresh
            })
            .map(|rows| rows.len())
            .sum()
    }

    /// Rank `rank`'s arena length.
    #[must_use]
    pub fn arena_bytes(&self, rank: usize) -> u64 {
        self.inner.ranks[rank].arena_bytes
    }

    /// Frees retained inputs, output sections and idle worker buffers.
    /// Call after readers have joined; later reads reproduce the same bytes.
    pub fn release_buffers(&self) {
        let workers = self
            .inner
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        drop(workers);
        for shared in &self.inner.shared {
            let mut state = shared
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for section in &mut state.sections {
                if let Some(bytes) = section.take() {
                    release_retained_v1(&self.inner, bytes);
                }
            }
        }
        for source in &self.inner.shared_sources {
            let mut state = source
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(bytes) = state.bytes.take() {
                release_retained_v1(&self.inner, bytes);
            }
        }
        self.inner.packed_sources.clear(&self.inner);
        self.inner.packed_buffers.clear();
        for pool in [&self.inner.scratch, &self.inner.direct_buffers] {
            if let Ok(mut pool) = pool.lock() {
                *pool = Vec::new();
            }
        }
    }

    #[must_use]
    pub fn stats(&self) -> DirectLoadStatsV1 {
        DirectLoadStatsV1 {
            produced_bytes: self.inner.produced_bytes.load(Ordering::Relaxed),
            source_bytes: self.inner.source_bytes.load(Ordering::Relaxed),
            source_reads: self.inner.source_reads.load(Ordering::Relaxed),
            requested_source_bytes: self.inner.source_bytes.load(Ordering::Relaxed)
                + self.inner.source_cache_bytes.load(Ordering::Relaxed),
            file_read_bytes: self.inner.file_read_bytes.load(Ordering::Relaxed),
            unique_source_bytes: self.inner.unique_source_bytes.load(Ordering::Relaxed),
            source_ranges_tracked: self.inner.source_ranges.is_some(),
            source_cache_hits: self.inner.source_cache_hits.load(Ordering::Relaxed),
            source_cache_bytes: self.inner.source_cache_bytes.load(Ordering::Relaxed),
            shared_output_hits: self.inner.shared_output_hits.load(Ordering::Relaxed),
            shared_output_productions: self.inner.shared_output_productions.load(Ordering::Relaxed),
            shared_output_reproductions: self
                .inner
                .shared_output_reproductions
                .load(Ordering::Relaxed),
            advised_bytes: self.inner.advised_bytes.load(Ordering::Relaxed),
            direct_bytes: self.inner.direct_bytes.load(Ordering::Relaxed),
            read_seconds: seconds(&self.inner.read_nanos),
            produce_seconds: seconds(&self.inner.produce_nanos),
            shared_seconds: seconds(&self.inner.shared_nanos),
            rank_seconds: seconds(&self.inner.rank_nanos),
            shared_timing: self.inner.detailed_timing.then(|| stats::SharedTiming {
                output_wait_seconds: seconds(&self.inner.output_wait_nanos),
                materialize_seconds: seconds(&self.inner.materialize_nanos),
                input_wait_seconds: seconds(&self.inner.input_wait_nanos),
            }),
            io_timing: self
                .inner
                .io_timing
                .as_ref()
                .map(io_stats::IoCounters::snapshot),
            buffer_timing: self.inner.packed_buffers.timing(),
            retained_peak_bytes: self.inner.retained_peak.load(Ordering::Relaxed),
            pooled_peak_bytes: self.inner.packed_buffers.peak(),
        }
    }

    /// Fills `destination` with the bytes at `offset` of the ranks laid end
    /// to end, the compiled image's component coordinates.
    pub fn read(&self, mut offset: u64, mut destination: &mut [u8]) -> Result<(), String> {
        self.inner.first_read.get_or_init(Instant::now);
        while !destination.is_empty() {
            let rank = self.inner.offsets.partition_point(|start| *start <= offset) - 1;
            let local = offset - self.inner.offsets[rank];
            let arena = self.inner.ranks[rank].arena_bytes;
            if local >= arena {
                return Err("read past the planned image".into());
            }
            let count = (arena - local).min(destination.len() as u64) as usize;
            let (head, tail) = destination.split_at_mut(count);
            if self.inner.options.read_ahead {
                self.request_read_ahead(rank, local + count as u64 + READ_AHEAD_BYTES);
            }
            read_rank_v1(&self.inner, rank, local, head)?;
            offset += count as u64;
            destination = tail;
        }
        Ok(())
    }

    /// Asks rank `rank`'s read-ahead thread to advise up to arena offset
    /// `to`, unless its queue is full; whether it was queued.
    fn request_read_ahead(&self, rank: usize, to: u64) -> bool {
        self.read_ahead[rank].try_send(to).is_ok()
    }
}

fn seconds(nanos: &AtomicU64) -> f64 {
    nanos.load(Ordering::Relaxed) as f64 * 1e-9
}

fn elapsed_nanos(counter: &AtomicU64, since: Instant) {
    counter.fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Time CPU packing after inputs are acquired. Streamed production below
/// instead subtracts its own nested read time, never another worker's totals.
fn packing_timed_v1<T>(inner: &InnerV1, work: impl FnOnce() -> T) -> T {
    let start = inner.io_timing.as_ref().map(|_| Instant::now());
    let result = work();
    if let (Some(timing), Some(start)) = (&inner.io_timing, start) {
        elapsed_nanos(&timing.packing_nanos, start);
    }
    result
}

fn read_buffered_v1(
    file: &std::fs::File,
    buffer: &mut [u8],
    at: u64,
    timing: Option<&io_stats::IoCounters>,
) -> std::io::Result<()> {
    match timing {
        Some(timing) => io_stats::read_exact(
            |buffer, at| timing.read(false, || file.read_at(buffer, at)),
            buffer,
            at,
        ),
        None => file.read_exact_at(buffer, at),
    }
}

/// Reads source bytes of one record.
fn read_source_v1(
    inner: &InnerV1,
    source: SourceAtV1,
    offset: u64,
    buffer: &mut [u8],
) -> Result<(), WeightTransformError> {
    read_source_storage_v1(inner, source, offset, buffer, 0..buffer.len())
}

/// Owned inputs can include alignment padding around their logical range.
/// Cold runs then land in that storage directly, without a bounce copy.
fn read_source_storage_v1(
    inner: &InnerV1,
    source: SourceAtV1,
    offset: u64,
    storage: &mut [u8],
    range: Range<usize>,
) -> Result<(), WeightTransformError> {
    let length = range.len() as u64;
    if range.end > storage.len()
        || offset
            .checked_add(length)
            .is_none_or(|end| end > source.length)
    {
        return Err(WeightTransformError(
            "source read outside its tensor".into(),
        ));
    }
    let shard = &inner.checkpoint.shards[source.shard];
    let started = Instant::now();
    let at = source.begin + offset;
    #[cfg(test)]
    if inner
        .checkpoint
        .read_refusal
        .get()
        .is_some_and(|range| at < range.end && range.start < at + length)
    {
        return Err(WeightTransformError(format!(
            "read {}: {}",
            shard.name,
            std::io::Error::from_raw_os_error(5)
        )));
    }
    let residency_started = inner.io_timing.as_ref().map(|_| Instant::now());
    // Runs the cache holds are copied from it; cold runs are read past it.
    let direct = inner
        .options
        .direct_reads
        .then(|| shard.direct_file())
        .flatten();
    let runs = match direct {
        Some(_) => residency_runs_v1(&shard.file, at, length),
        None => vec![(length, false)],
    };
    if let (Some(timing), Some(start)) = (&inner.io_timing, residency_started) {
        elapsed_nanos(&timing.residency_nanos, start);
        timing.request(source.shard, at, length);
    }
    let mut begin = range.start;
    let mut run_at = at;
    for (length, cold) in runs {
        let part = begin..begin + length as usize;
        // A refusal earlier in this read turns direct reads off for the rest
        // of it too.
        let read_direct = direct
            .filter(|direct| cold && !direct.off.load(Ordering::Relaxed))
            .map(|direct| {
                read_direct_storage_v1(
                    &inner.direct_buffers,
                    direct,
                    run_at,
                    storage,
                    part.clone(),
                    inner.io_timing.as_ref(),
                )
            });
        match read_direct {
            Some(Ok(bytes)) => {
                inner.file_read_bytes.fetch_add(bytes, Ordering::Relaxed);
                inner.direct_bytes.fetch_add(length, Ordering::Relaxed);
            }
            // A file system may refuse a direct read it opened for (an
            // alignment it did not report, a range it cannot serve): that
            // run, and the rest of this shard, is read through the cache.
            Some(Err(error)) if refused_direct_read(&error) => {
                if let Some(direct) = direct {
                    direct.refused.fetch_add(1, Ordering::Relaxed);
                }
                if !direct.is_some_and(|direct| direct.off.swap(true, Ordering::Relaxed)) {
                    eprintln!(
                        "direct load: {} refused a direct read ({error}); reading it through the page cache",
                        shard.name
                    );
                }
                read_buffered_v1(
                    &shard.file,
                    &mut storage[part.clone()],
                    run_at,
                    inner.io_timing.as_ref(),
                )
                .map_err(|e| WeightTransformError(format!("read {}: {e}", shard.name)))?;
                inner.file_read_bytes.fetch_add(length, Ordering::Relaxed);
            }
            Some(Err(error)) => {
                return Err(WeightTransformError(format!(
                    "direct read {}: {error}",
                    shard.name
                )));
            }
            None => {
                read_buffered_v1(
                    &shard.file,
                    &mut storage[part.clone()],
                    run_at,
                    inner.io_timing.as_ref(),
                )
                .map_err(|e| WeightTransformError(format!("read {}: {e}", shard.name)))?;
                inner.file_read_bytes.fetch_add(length, Ordering::Relaxed);
            }
        }
        begin = part.end;
        run_at += length;
    }
    elapsed_nanos(&inner.read_nanos, started);
    inner.source_bytes.fetch_add(length, Ordering::Relaxed);
    inner.source_reads.fetch_add(1, Ordering::Relaxed);
    if let Some(ranges) = &inner.source_ranges {
        let unique = ranges
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(source.shard, at..at + length);
        inner
            .unique_source_bytes
            .fetch_add(unique, Ordering::Relaxed);
    }
    Ok(())
}

/// `length` bytes at `at` of `file` as consecutive runs, each with whether
/// the page cache lacks it: a range the cache holds wholly or not at all is
/// one run, and a mixed one is halved down to [`RESIDENCY_BYTES`], below
/// which it counts as cached when at least half its pages are. Without a
/// residency report the whole range counts as cached.
fn residency_runs_v1(file: &std::fs::File, at: u64, length: u64) -> Vec<(u64, bool)> {
    fn split(file: &std::fs::File, at: u64, length: u64, runs: &mut Vec<(u64, bool)>) {
        let cold = match cached_pages_v1(file, at, length) {
            Err(_) => false,
            Ok((cached, pages)) if cached == pages => false,
            Ok((0, _)) => true,
            Ok((cached, pages)) if length <= RESIDENCY_BYTES => cached * 2 < pages,
            Ok(_) => {
                let half = (length / 2).next_multiple_of(DIRECT_READ_ALIGNMENT);
                split(file, at, half, runs);
                split(file, at + half, length - half, runs);
                return;
            }
        };
        match runs.last_mut() {
            Some((run, kind)) if *kind == cold => *run += length,
            _ => runs.push((length, cold)),
        }
    }
    let mut runs = Vec::new();
    split(file, at, length, &mut runs);
    runs
}

/// Whether a direct read failed because the file system refused direct I/O
/// for it, rather than for a fault the buffered read would meet too.
fn refused_direct_read(error: &std::io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(code) if code == libc::EINVAL || code == libc::EOPNOTSUPP
    )
}

/// A shard's file opened again with `O_DIRECT`, the alignment its reads
/// keep, and whether a refused read turned direct reads off for it.
struct DirectFileV1 {
    file: std::fs::File,
    alignment: u64,
    off: std::sync::atomic::AtomicBool,
    /// Direct reads the file system refused.
    refused: AtomicU64,
}

impl ShardV1 {
    fn direct_file(&self) -> Option<&DirectFileV1> {
        self.direct
            .get_or_init(|| reopen_direct_v1(&self.file))
            .as_ref()
            .filter(|direct| !direct.off.load(Ordering::Relaxed))
    }
}

/// `file` opened again with `O_DIRECT`, through its open descriptor so it
/// is the same file, at the alignment its file system reports (4096 bytes
/// when it reports none); `None` where direct I/O is refused or needs more
/// than [`DIRECT_READ_ALIGNMENT_MAX`].
#[cfg(target_os = "linux")]
fn reopen_direct_v1(file: &std::fs::File) -> Option<DirectFileV1> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let direct = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(format!("/proc/self/fd/{}", file.as_raw_fd()))
        .ok()?;
    let (opened, reopened) = (file.metadata().ok()?, direct.metadata().ok()?);
    if opened.dev() != reopened.dev() || opened.ino() != reopened.ino() {
        return None;
    }
    let alignment = match direct_io_alignment_v1(&direct) {
        Ok(reported) => reported?,
        Err(_) => DIRECT_READ_ALIGNMENT,
    };
    (alignment.is_power_of_two() && alignment <= DIRECT_READ_ALIGNMENT_MAX).then(|| DirectFileV1 {
        file: direct,
        alignment,
        off: std::sync::atomic::AtomicBool::new(false),
        refused: AtomicU64::new(0),
    })
}

#[cfg(not(target_os = "linux"))]
fn reopen_direct_v1(_: &std::fs::File) -> Option<DirectFileV1> {
    None
}

/// Fills `buffer` from `at` of a file opened with `O_DIRECT`, through an
/// aligned buffer taken from and returned to `buffers`: whole aligned blocks
/// around the range are read and the range copied out of them.
fn read_direct_v1(
    buffers: &Mutex<Vec<Vec<u8>>>,
    direct: &DirectFileV1,
    at: u64,
    buffer: &mut [u8],
    timing: Option<&io_stats::IoCounters>,
) -> std::io::Result<u64> {
    let (file, align) = (&direct.file, direct.alignment);
    let needed = (DIRECT_READ_BYTES + align) as usize;
    let mut aligned = buffers
        .lock()
        .ok()
        .and_then(|mut pool| pool.pop())
        .filter(|pooled| pooled.len() >= needed)
        .unwrap_or_else(|| vec![0; needed]);
    let result = (|| {
        let base = aligned.as_ptr().align_offset(align as usize);
        let end = at + buffer.len() as u64;
        let mut done = 0;
        let mut read_bytes = 0;
        while done < buffer.len() {
            let from = at + done as u64;
            let start = from / align * align;
            let stop = end.next_multiple_of(align).min(start + DIRECT_READ_BYTES);
            let window = &mut aligned[base..base + (stop - start) as usize];
            // Reads stay aligned until the one that meets the end of the
            // file, which is short.
            let mut filled = 0;
            while filled < window.len() {
                let mut read = || file.read_at(&mut window[filled..], start + filled as u64);
                let count = match timing {
                    Some(timing) => timing.read(true, read),
                    None => read(),
                }?;
                filled += count;
                read_bytes += count as u64;
                if count == 0 || count as u64 % align != 0 {
                    break;
                }
            }
            let skip = (from - start) as usize;
            if filled <= skip {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            let count = (filled - skip).min(buffer.len() - done);
            let copying = timing.map(|_| Instant::now());
            buffer[done..done + count].copy_from_slice(&window[skip..skip + count]);
            if let (Some(timing), Some(start)) = (timing, copying) {
                timing
                    .bounce_bytes
                    .fetch_add(count as u64, Ordering::Relaxed);
                elapsed_nanos(&timing.bounce_nanos, start);
            }
            done += count;
        }
        Ok(read_bytes)
    })();
    if let Ok(mut pool) = buffers.lock() {
        pool.push(aligned);
    }
    result
}

/// Use caller-owned alignment padding when it can hold the complete direct
/// window. Ordinary destinations retain the bounded bounce-buffer path.
fn read_direct_storage_v1(
    buffers: &Mutex<Vec<Vec<u8>>>,
    direct: &DirectFileV1,
    at: u64,
    storage: &mut [u8],
    range: Range<usize>,
    timing: Option<&io_stats::IoCounters>,
) -> std::io::Result<u64> {
    if range.is_empty() {
        return Ok(0);
    }
    let align = direct.alignment;
    let prefix = (at % align) as usize;
    let end = at + range.len() as u64;
    let suffix = (end.next_multiple_of(align) - end) as usize;
    let window = range
        .start
        .checked_sub(prefix)
        .zip(range.end.checked_add(suffix))
        .filter(|&(start, end)| {
            end <= storage.len() && storage[start..].as_ptr().align_offset(align as usize) == 0
        });
    let Some((start, end)) = window else {
        return read_direct_v1(buffers, direct, at, &mut storage[range], timing);
    };
    let window = &mut storage[start..end];
    let file_start = at - prefix as u64;
    let required = prefix + range.len();
    let mut filled = 0;
    while filled < window.len() {
        let stop = window.len().min(filled + DIRECT_READ_BYTES as usize);
        let mut read = || {
            direct
                .file
                .read_at(&mut window[filled..stop], file_start + filled as u64)
        };
        let result = match timing {
            Some(timing) => timing.read(true, read),
            None => read(),
        };
        let count = match result {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            other => other?,
        };
        filled += count;
        if count == 0 || count as u64 % align != 0 {
            break;
        }
    }
    if filled < required {
        return Err(std::io::ErrorKind::UnexpectedEof.into());
    }
    Ok(filled as u64)
}

/// The whole output of a shared record, as sections.
fn produce_sections_v1(
    inner: &InnerV1,
    record: &RecordV1,
    wanted: [bool; 3],
) -> Result<Vec<Vec<u8>>, String> {
    let error = transform_error(record.offset);
    let mut read = |index: usize, offset: u64, buffer: &mut [u8]| {
        read_source_v1(inner, record.sources[index], offset, buffer)
    };
    if let WeightTransform::Bundle(id) = &record.prepared.record().transform {
        let mut sources = Vec::with_capacity(record.sources.len());
        for index in 0..record.sources.len() {
            let mut bytes = vec![0; record.sources[index].length as usize];
            read(index, 0, &mut bytes).map_err(&error)?;
            sources.push(bytes);
        }
        let views: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
        let mut out = vec![0; record.prepared.byte_length() as usize];
        runtime_bundle_abi::call_weight_transform_v1(
            inner.transform.ok_or("no bundle transform export")?,
            *id,
            &record.bundle_params,
            &views,
            0,
            &mut out,
        )?;
        return Ok(vec![out]);
    }
    if let WeightTransform::PackedOperand(operand) = &record.prepared.record().transform {
        let described = record.prepared.record();
        let mut scale_input = None;
        if record.has_code_section
            && let Some(index) = operand.independent_scale_source(described)
        {
            let input = inner
                .packed_sources
                .read(inner, record, index)
                .map_err(&error)?;
            let buffer = inner
                .packed_buffers
                .take(record.prepared.byte_length() as usize);
            if let Some(scales) = packing_timed_v1(inner, || {
                operand.independent_scales(described, input.bytes(), buffer)
            })
            .map_err(&error)?
            {
                return Ok(vec![Vec::new(), scales, Vec::new()]);
            }
            scale_input = Some((index, input));
        }
        let inputs = (0..record.sources.len())
            .map(|index| {
                if scale_input
                    .as_ref()
                    .is_some_and(|(source, _)| *source == index)
                {
                    Ok(scale_input.take().expect("matched scale source").1)
                } else {
                    inner.packed_sources.read(inner, record, index)
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(&error)?;
        let views = inputs.iter().map(|input| input.bytes()).collect::<Vec<_>>();
        let sizes = operand.plan.section_bytes().map_err(|e| e.to_string())?;
        let sizes = [sizes.codes, sizes.local_scales, sizes.global_scales];
        let buffers = std::array::from_fn(|section| {
            if wanted[section] {
                inner.packed_buffers.take(sizes[section] as usize)
            } else {
                Vec::new()
            }
        });
        return packing_timed_v1(inner, || {
            operand.execute_needed_sections_reusing(described, &views, buffers, wanted)
        })
        .map(|sections| sections.into())
        .map_err(error);
    }
    record.prepared.produce_sections(&mut read).map_err(error)
}

/// Checks one planned record against the checkpoint and prepares it, with
/// the key that groups records every rank places identically.
fn prepare_record_v1(
    checkpoint: &DirectCheckpointV1,
    transform: Option<BundleWeightTransformFnV1>,
    record: &WeightRecordV1,
) -> Result<(String, RecordV1), String> {
    let mut sources = Vec::with_capacity(record.sources.len());
    let mut described = Vec::with_capacity(record.sources.len());
    for source in &record.sources {
        let shard = checkpoint
            .shards
            .get(source.file as usize)
            .ok_or_else(|| format!("record names file {}", source.file))?;
        let (tensor, begin) = shard
            .tensors
            .get(&source.tensor)
            .ok_or_else(|| format!("tensor {} is not in {}", source.tensor, shard.name))?;
        sources.push(SourceAtV1 {
            shard: source.file as usize,
            begin: *begin,
            length: tensor.byte_length,
        });
        described.push(tensor.clone());
    }
    let transform_id = WeightTransform::decode(record.transform, &record.params)
        .map_err(transform_error(record.arena_offset))?;
    if matches!(transform_id, WeightTransform::Bundle(_)) && transform.is_none() {
        return Err("the plan uses a bundle transform the bundle does not export".into());
    }
    // A packed operand's section records share one packing, so their key
    // leaves the section out.
    let params = match &transform_id {
        WeightTransform::PackedOperand(operand) => {
            let mut operand = operand.clone();
            operand.section = kernel_core::weight_transform::PackedSection::Codes;
            WeightTransform::PackedOperand(operand)
                .encode_params()
                .map_err(transform_error(record.arena_offset))?
        }
        _ => record.params.clone(),
    };
    let key = format!(
        "{:?}",
        (
            &record.sources,
            record.transform,
            &params,
            record.axis,
            &record.ranges,
            &record.stored_shape,
            matches!(transform_id, WeightTransform::PackedOperand(_))
                .then_some(0)
                .unwrap_or(record.byte_length),
        )
    );
    let bundle = matches!(transform_id, WeightTransform::Bundle(_));
    let prepared = WeightRecord {
        transform: transform_id,
        sources: described,
        axis: record.axis,
        ranges: record.ranges.clone(),
        stored_shape: record.stored_shape.clone(),
    }
    .prepare(record.byte_length)
    .map_err(transform_error(record.arena_offset))?;
    Ok((
        key,
        RecordV1 {
            consumer: 0,
            packed_sources: Vec::new(),
            has_code_section: false,
            offset: record.arena_offset,
            bundle_params: if bundle {
                record.params.clone()
            } else {
                Vec::new()
            },
            prepared,
            sources,
            shared: None,
            source_share: None,
        },
    ))
}

/// Marks records that each keep a slice along an inner axis of the same
/// small source tensor, when there are several (the ranks' K slices of one
/// matrix), to read that tensor once for all of them.
fn share_sources_v1(mut ranks: Vec<RankV1>) -> (Vec<RankV1>, Vec<SharedSourceV1>) {
    let strided = |record: &RecordV1| {
        let described = record.prepared.record();
        record.shared.is_none()
            && record.sources.len() == 1
            && described.transform.gathers()
            && record.sources[0].length <= SHARED_SOURCE_BYTES_MAX
            && described.axis.is_some_and(|axis| {
                let shape = &described.sources[0].shape;
                axis > 0
                    && shape[..(axis as usize).min(shape.len())]
                        .iter()
                        .product::<u64>()
                        > 1
            })
    };
    let mut users: BTreeMap<(usize, u64), (usize, u64)> = BTreeMap::new();
    for record in ranks.iter().flat_map(|rank| &rank.records) {
        if strided(record) {
            let source = record.sources[0];
            let entry = users.entry((source.shard, source.begin)).or_default();
            entry.0 += 1;
            entry.1 += record.prepared.byte_length();
        }
    }
    let mut shared = Vec::new();
    let mut slots: BTreeMap<(usize, u64), usize> = BTreeMap::new();
    for record in ranks.iter_mut().flat_map(|rank| &mut rank.records) {
        if !strided(record) {
            continue;
        }
        let key = (record.sources[0].shard, record.sources[0].begin);
        let (count, remaining) = users[&key];
        if count < 2 {
            continue;
        }
        let index = *slots.entry(key).or_insert_with(|| {
            shared.push(SharedSourceV1 {
                state: Mutex::new(SharedSourceStateV1 {
                    bytes: None,
                    remaining,
                }),
            });
            shared.len() - 1
        });
        record.source_share = Some(index);
    }
    (ranks, shared)
}

/// Produces `range` of a record from its shared source tensor: the first
/// record to need it reads it whole, within the shared budget; the others
/// copy from memory. Without room in the budget the record reads its slice
/// itself. The tensor is released once its records owe no more bytes.
fn produce_from_shared_source_v1(
    inner: &InnerV1,
    record: &RecordV1,
    index: usize,
    range: Range<u64>,
    out: &mut [u8],
    scratch: &mut Vec<u8>,
) -> Result<(), String> {
    let share = &inner.shared_sources[index];
    let length = range.end - range.start;
    let source = record.sources[0];
    let bytes = {
        let mut state = share.state.lock().map_err(|_| "shared source poisoned")?;
        match state.bytes.clone() {
            Some(bytes) => Some(bytes),
            None if state.remaining > length && reserve_v1(inner, source.length) => {
                let mut whole = RetainedBytesV1::new(inner, vec![0; source.length as usize]);
                if let Err(error) = read_source_v1(inner, source, 0, &mut whole.bytes) {
                    release_retained_v1(inner, Arc::new(whole));
                    return Err(transform_error(record.offset)(error));
                }
                let whole = Arc::new(whole);
                state.bytes = Some(Arc::clone(&whole));
                Some(whole)
            }
            None => None,
        }
    };
    let result = match &bytes {
        Some(bytes) => {
            let mut read = |_: usize, offset: u64, buffer: &mut [u8]| {
                let from = offset as usize;
                buffer.copy_from_slice(&bytes.bytes[from..from + buffer.len()]);
                if let Some(timing) = &inner.io_timing {
                    timing
                        .source_copy_bytes
                        .fetch_add(buffer.len() as u64, Ordering::Relaxed);
                }
                Ok(())
            };
            record.prepared.produce(range, out, &mut read, scratch)
        }
        None => {
            let mut read = |index: usize, offset: u64, buffer: &mut [u8]| {
                read_source_v1(inner, record.sources[index], offset, buffer)
            };
            record.prepared.produce(range, out, &mut read, scratch)
        }
    }
    .map_err(transform_error(record.offset));
    // This task's reference goes before its bytes are counted, and the
    // tensor's own once they all are, under the lock: the last reference
    // frees the tensor before its reservation returns to the budget.
    let mut state = share.state.lock().map_err(|_| "shared source poisoned")?;
    if let Some(bytes) = bytes {
        release_retained_v1(inner, bytes);
    }
    state.remaining = state.remaining.saturating_sub(length);
    if state.remaining == 0
        && let Some(done) = state.bytes.take()
    {
        release_retained_v1(inner, done);
    }
    drop(state);
    result
}

/// Drops a reference to bytes held against the shared budget. The last one
/// frees them and then returns their reservation, so the budget never
/// counts less than the bytes held.
fn release_retained_v1(inner: &InnerV1, bytes: Arc<RetainedBytesV1>) {
    if let Some(bytes) = Arc::into_inner(bytes) {
        let length = bytes.charge;
        drop(bytes);
        inner.retained.fetch_sub(length, Ordering::AcqRel);
        #[cfg(test)]
        inner.retained_held.pause();
    }
}

/// Reserves `bytes` of the shared-output budget, if it has room.
fn reserve_v1(inner: &InnerV1, bytes: u64) -> bool {
    let reserved = inner
        .retained
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
            held.checked_add(bytes)
                .filter(|total| *total <= inner.options.shared_bytes)
        });
    if let Ok(held) = reserved {
        inner
            .retained_peak
            .fetch_max(held + bytes, Ordering::AcqRel);
    }
    reserved.is_ok()
}

/// Copies `range` of a shared record's output. The first consumer produces
/// the record; each section is kept for later consumers only while it still
/// owes them bytes and fits the budget, and is freed once it owes none and
/// its last consumer has copied from it.
/// A consumer that finds its section gone produces the record again, so the
/// held bytes never exceed `DirectLoadOptionsV1::shared_bytes`.
fn copy_shared_v1(
    inner: &InnerV1,
    record: &RecordV1,
    index: usize,
    range: Range<u64>,
    out: &mut [u8],
) -> Result<(), String> {
    let length = range.end - range.start;
    let wanted = record.prepared.section();
    let bytes = {
        let waiting = inner.detailed_timing.then(Instant::now);
        let mut state = inner.shared[index]
            .state
            .lock()
            .map_err(|_| "shared output poisoned")?;
        if let Some(waiting) = waiting {
            elapsed_nanos(&inner.output_wait_nanos, waiting);
        }
        let bytes = match state.sections[wanted].clone() {
            Some(bytes) => {
                inner.shared_output_hits.fetch_add(1, Ordering::Relaxed);
                SectionV1::Retained(bytes)
            }
            None => {
                let producing = inner.detailed_timing.then(Instant::now);
                let needed = std::array::from_fn(|section| {
                    section == wanted
                        || state.remaining[section] > 0 && state.sections[section].is_none()
                });
                let produced = produce_sections_v1(inner, record, needed);
                if let Some(producing) = producing {
                    elapsed_nanos(&inner.materialize_nanos, producing);
                }
                let produced = produced?;
                inner
                    .shared_output_productions
                    .fetch_add(1, Ordering::Relaxed);
                if state.produced_sections[wanted] {
                    inner
                        .shared_output_reproductions
                        .fetch_add(1, Ordering::Relaxed);
                }
                let mut wanted_bytes = None;
                for (section, data) in produced.into_iter().enumerate().take(3) {
                    if data.is_empty() {
                        continue;
                    }
                    state.produced_sections[section] = true;
                    let pooled = matches!(
                        record.prepared.record().transform,
                        WeightTransform::PackedOperand(_)
                    );
                    let charge = if pooled { data.capacity() } else { data.len() } as u64;
                    let owed = if section == wanted {
                        state.remaining[section] > length
                    } else {
                        state.remaining[section] > 0
                    };
                    let retained =
                        owed && state.sections[section].is_none() && reserve_v1(inner, charge);
                    if section != wanted
                        && !retained
                        && let Some(timing) = &inner.io_timing
                    {
                        timing
                            .unused_section_bytes
                            .fetch_add(data.len() as u64, Ordering::Relaxed);
                    }
                    let data = if retained {
                        let held = Arc::new(if pooled {
                            RetainedBytesV1::pooled(inner, data)
                        } else {
                            RetainedBytesV1::new(inner, data)
                        });
                        state.sections[section] = Some(Arc::clone(&held));
                        SectionV1::Retained(held)
                    } else {
                        SectionV1::Own(data)
                    };
                    if section == wanted {
                        wanted_bytes = Some(data);
                    }
                }
                wanted_bytes.ok_or("shared record lacks its section")?
            }
        };
        state.remaining[wanted] = state.remaining[wanted].saturating_sub(length);
        if state.remaining[wanted] == 0
            && let Some(done) = state.sections[wanted].take()
        {
            release_retained_v1(inner, done);
        }
        bytes
    };
    let result = if bytes.bytes().len() as u64 == record.prepared.byte_length() {
        out.copy_from_slice(&bytes.bytes()[range.start as usize..range.end as usize]);
        if let Some(timing) = &inner.io_timing {
            timing
                .output_copy_bytes
                .fetch_add(out.len() as u64, Ordering::Relaxed);
        }
        Ok(())
    } else {
        Err(transform_error(record.offset)(WeightTransformError(
            "shared section differs from its record".into(),
        )))
    };
    // A section the state gave up above is freed, and its reservation
    // returned, by whichever consumer drops the last reference.
    match bytes {
        SectionV1::Retained(held) => release_retained_v1(inner, held),
        SectionV1::Own(bytes)
            if matches!(
                record.prepared.record().transform,
                WeightTransform::PackedOperand(_)
            ) =>
        {
            inner.packed_buffers.give(bytes)
        }
        SectionV1::Own(_) => {}
    }
    result
}

/// A consumer's section of a shared output: one kept for later consumers,
/// or its own copy when the budget had no room.
enum SectionV1 {
    Retained(Arc<RetainedBytesV1>),
    Own(Vec<u8>),
}

impl SectionV1 {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Retained(held) => &held.bytes,
            Self::Own(bytes) => bytes,
        }
    }
}

/// One part of one record, written to its own disjoint output slice.
struct TaskV1<'a> {
    record: &'a RecordV1,
    range: Range<u64>,
    out: &'a mut [u8],
}

fn read_rank_v1(inner: &InnerV1, rank: usize, start: u64, out: &mut [u8]) -> Result<(), String> {
    let started = Instant::now();
    let result = read_rank_parts_v1(inner, rank, start, out);
    elapsed_nanos(&inner.rank_nanos, started);
    result
}

fn read_rank_parts_v1(
    inner: &InnerV1,
    rank: usize,
    start: u64,
    out: &mut [u8],
) -> Result<(), String> {
    let records = &inner.ranks[rank].records;
    let end = start + out.len() as u64;
    let first = records.partition_point(|r| r.offset + r.prepared.byte_length() <= start);
    let mut tasks = Vec::new();
    let mut rest = out;
    let mut at = start;
    for record in &records[first..] {
        if record.offset >= end {
            break;
        }
        if record.offset > at {
            let (gap, tail) = rest.split_at_mut((record.offset - at) as usize);
            gap.fill(0);
            rest = tail;
            at = record.offset;
        }
        let local_end = (record.offset + record.prepared.byte_length()).min(end) - record.offset;
        let mut local = at - record.offset;
        let unit = record.prepared.unit_bytes();
        while local < local_end {
            let mut stop = local_end;
            if record.shared.is_none() && local_end - local > TASK_BYTES {
                let target = local + TASK_BYTES;
                let aligned = target / unit * unit;
                stop = if aligned > local { aligned } else { target }.min(local_end);
            }
            let (part, tail) = rest.split_at_mut((stop - local) as usize);
            tasks.push(TaskV1 {
                record,
                range: local..stop,
                out: part,
            });
            rest = tail;
            local = stop;
        }
        at = record.offset + local_end;
    }
    rest.fill(0);
    let produced = tasks.iter().map(|task| task.out.len() as u64).sum::<u64>();
    let run = |task: TaskV1<'_>, scratch: &mut Vec<u8>| -> Result<(), String> {
        let record = task.record;
        let started = Instant::now();
        if let Some(index) = record.shared {
            let result = copy_shared_v1(inner, record, index, task.range, task.out);
            elapsed_nanos(&inner.shared_nanos, started);
            return result;
        }
        let result = if let Some(index) = record.source_share {
            produce_from_shared_source_v1(inner, record, index, task.range, task.out, scratch)
        } else {
            let mut reading = Duration::ZERO;
            let mut read = |index: usize, offset: u64, buffer: &mut [u8]| {
                let start = inner.io_timing.as_ref().map(|_| Instant::now());
                let result = read_source_v1(inner, record.sources[index], offset, buffer);
                if let Some(start) = start {
                    reading += start.elapsed();
                }
                result
            };
            let result = record
                .prepared
                .produce(task.range, task.out, &mut read, scratch)
                .map_err(transform_error(record.offset));
            if let Some(timing) = &inner.io_timing {
                timing.packing_nanos.fetch_add(
                    started.elapsed().saturating_sub(reading).as_nanos() as u64,
                    Ordering::Relaxed,
                );
            }
            result
        };
        elapsed_nanos(&inner.produce_nanos, started);
        result
    };
    let workers = inner.options.threads_per_rank.min(tasks.len()).max(1);
    if workers == 1 {
        let mut scratch = take_scratch_v1(inner);
        let result = tasks
            .into_iter()
            .try_for_each(|task| run(task, &mut scratch));
        give_scratch_v1(inner, scratch);
        result?;
    } else {
        // Larger tasks first, so the pool finishes together.
        tasks.sort_by_key(|task| std::cmp::Reverse(task.out.len()));
        let queue = Mutex::new(tasks.into_iter());
        let failed = AtomicUsize::new(0);
        let error = Mutex::new(None);
        let pool = workers_v1(inner)?;
        let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.pool().in_place_scope(|scope| {
                for _ in 0..workers {
                    scope.spawn(|_| {
                        let mut scratch = take_scratch_v1(inner);
                        while failed.load(Ordering::Relaxed) == 0 {
                            let Some(task) = queue.lock().ok().and_then(|mut queue| queue.next())
                            else {
                                break;
                            };
                            if let Err(message) = run(task, &mut scratch) {
                                failed.store(1, Ordering::Relaxed);
                                if let Ok(mut error) = error.lock() {
                                    error.get_or_insert(message);
                                }
                            }
                        }
                        give_scratch_v1(inner, scratch);
                    });
                }
            })
        }));
        if completed.is_err() {
            return Err("a direct-load worker panicked".into());
        }
        if let Some(message) = error.into_inner().ok().flatten() {
            return Err(message);
        }
        if failed.load(Ordering::Relaxed) != 0 {
            return Err("a direct-load worker failed".into());
        }
    }
    inner.produced_bytes.fetch_add(produced, Ordering::Relaxed);
    Ok(())
}

/// One bounded executor serves every rank and chunk until buffers are released.
fn workers_v1(inner: &InnerV1) -> Result<Arc<workers::Workers>, String> {
    let mut workers = inner
        .workers
        .lock()
        .map_err(|_| "weight workers poisoned")?;
    if workers.is_none() {
        let count = inner
            .options
            .threads_per_rank
            .max(1)
            .checked_mul(inner.ranks.len())
            .ok_or("weight worker count overflow")?;
        *workers = Some(Arc::new(workers::Workers::new(count)?));
    }
    Ok(Arc::clone(
        workers.as_ref().expect("weight workers initialized"),
    ))
}

fn take_scratch_v1(inner: &InnerV1) -> Vec<u8> {
    inner
        .scratch
        .lock()
        .ok()
        .and_then(|mut pool| pool.pop())
        .unwrap_or_default()
}

fn give_scratch_v1(inner: &InnerV1, scratch: Vec<u8>) {
    if let Ok(mut pool) = inner.scratch.lock() {
        pool.push(scratch);
    }
}

/// Advises the kernel of the source ranges of the arena up to each queued
/// offset, from [`READ_AHEAD_BYTES`] before it, until the image is dropped.
/// Of the offsets queued while it advised, only the newest counts: the
/// reads have moved past the others.
fn read_ahead_v1(inner: &InnerV1, rank: usize, targets: &Receiver<u64>) {
    let records = &inner.ranks[rank].records;
    let stopped = || inner.read_ahead_stopped.load(Ordering::Acquire);
    let mut advised_to = 0_u64;
    while let Ok(mut to) = targets.recv() {
        while let Ok(newer) = targets.try_recv() {
            to = newer;
        }
        if stopped() {
            return;
        }
        let start = to.saturating_sub(READ_AHEAD_BYTES).max(advised_to);
        let end = to.min(inner.ranks[rank].arena_bytes);
        if start >= end {
            continue;
        }
        advised_to = end;
        let first = records.partition_point(|r| r.offset + r.prepared.byte_length() <= start);
        let mut ranges: Vec<(usize, u64, u64)> = Vec::new();
        for record in &records[first..] {
            if record.offset >= end {
                break;
            }
            let local = start.saturating_sub(record.offset)
                ..(end - record.offset).min(record.prepared.byte_length());
            let Ok(pieces) = record.prepared.source_ranges(local) else {
                continue;
            };
            for (index, offset, length) in pieces {
                let source = record.sources[index];
                let begin = source.begin + offset;
                match ranges.last_mut() {
                    Some((shard, at, count)) if *shard == source.shard && *at + *count == begin => {
                        *count += length;
                    }
                    _ => ranges.push((source.shard, begin, length)),
                }
            }
        }
        for (shard, offset, length) in ranges {
            let file = &inner.checkpoint.shards[shard].file;
            for at in (offset..offset + length).step_by(ADVICE_BYTES as usize) {
                if stopped() {
                    return;
                }
                #[cfg(test)]
                if let Some(hook) = inner.checkpoint.advice_hook.get() {
                    hook();
                }
                let piece = ADVICE_BYTES.min(offset + length - at);
                if advise_v1(file, at, piece, CacheAdviceV1::WillNeed).is_ok() {
                    inner.advised_bytes.fetch_add(piece, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Asks the bundle for every rank's plan of the compiled image `image_role`
/// and prepares the planned image, or `None` when the bundle declines it.
pub fn plan_image_v1(
    library: &runtime_bundle_abi::BundleLibraryV1,
    bundle_module: [u8; 32],
    checkpoint: Arc<DirectCheckpointV1>,
    image_role: &str,
    ranks: u32,
    components: Vec<NamedBytesV1>,
    host_views: Vec<String>,
) -> Result<Option<PlannedImageV1>, String> {
    let started = Instant::now();
    let implementation = implementation_v1(bundle_module);
    let cache = plan_cache::Cache::selected();
    let mut validation_seconds = 0.0;
    let result = plan_cache::prepare(
        cache.as_ref(),
        ranks as usize,
        |rank| {
            checkpoint.request(
                image_role,
                ranks,
                rank as u32,
                implementation,
                components.clone(),
                host_views.clone(),
            )
        },
        |request| {
            library
                .weight_plan(request)
                .map_err(|error| format!("{image_role} rank {} weight plan: {error}", request.rank))
        },
        |plans| {
            let validation = Instant::now();
            let image = PlannedImageV1::new_with_range_tracking(
                checkpoint.clone(),
                plans,
                library.weight_transform(),
                implementation,
                DirectLoadOptionsV1::for_ranks(ranks as usize),
                std::env::var_os("CHOPINFER_DIRECT_LOAD_TRACK_RANGES")
                    .is_some_and(|value| value == "1"),
            );
            validation_seconds += validation.elapsed().as_secs_f64();
            image
        },
    )?;
    let Some((image, records, cache_hits)) = result else {
        return Ok(None);
    };
    let planned_seconds = started.elapsed().as_secs_f64() - validation_seconds;
    eprintln!(
        "startup weight_plan: {}",
        serde_json::json!({
            "image_role": image_role,
            "seconds": started.elapsed().as_secs_f64(), "plan_seconds": planned_seconds,
            "ranks": ranks, "records": records, "shared_outputs": image.inner.shared.len(),
            "metadata_cache_enabled": cache.is_some(), "metadata_cache_hits": cache_hits,
            "checkpoint_identity": image.checkpoint().identity_hex(),
            "threads_per_rank": image.inner.options.threads_per_rank,
            "cpus": process_cpus_v1(),
            "read_ahead": image.inner.options.read_ahead,
            "direct_reads": image.inner.options.direct_reads,
        })
    );
    Ok(Some(image))
}

/// The checkpoint this process loads (`serve --model` or `--checkpoint`),
/// selected once per process.
static SELECTED: OnceLock<Arc<DirectCheckpointV1>> = OnceLock::new();

/// `serve --load-threads`: packing workers per rank, selected once per
/// process; it takes precedence over `CHOPINFER_DIRECT_LOAD_THREADS`.
static THREADS_PER_RANK: OnceLock<usize> = OnceLock::new();

/// Selects the packing workers per rank of every direct load in this process.
pub fn select_threads_per_rank_v1(threads: usize) -> Result<(), String> {
    if threads == 0 {
        return Err("a direct load needs at least one packing thread per rank".to_owned());
    }
    THREADS_PER_RANK
        .set(threads)
        .map_err(|_| "direct-load threads are already selected".to_owned())
}

/// Selects the checkpoint every instance of this process loads directly.
pub fn select_checkpoint_v1(root: &Path) -> Result<(), String> {
    let checkpoint = Arc::new(DirectCheckpointV1::open(root)?);
    SELECTED
        .set(checkpoint)
        .map_err(|_| "a direct-load checkpoint is already selected".to_owned())
}

/// The selected checkpoint, if any.
pub(crate) fn selected_checkpoint_v1() -> Result<Option<Arc<DirectCheckpointV1>>, String> {
    Ok(SELECTED.get().cloned())
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    mod packed;
    use super::*;
    use kernel_core::transform::Transform;
    use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
    use runtime_bundle_abi::weight_plan::{PlannedViewV1, RecordSourceV1, WeightRecordV1};

    /// The digest covers every source on the direct load's byte path: every
    /// host module reachable from the executor, the read path and the CUDA
    /// upload (`direct`, `page_cache`, `services`, `cuda`), every workspace
    /// crate those modules name, every workspace crate a covered crate links
    /// through any of its dependency tables, and every Rust file of each. A
    /// module added outside the digest fails here instead of leaving
    /// identities stale.
    #[test]
    fn the_implementation_digest_covers_every_module_the_direct_load_uses() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest.parent().unwrap().parent().unwrap();
        let covered: std::collections::BTreeSet<&str> = DIRECT_SOURCES.iter().copied().collect();
        // Workspace crates by package name, from the root manifest.
        let root = std::fs::read_to_string(workspace.join("Cargo.toml")).unwrap();
        let members: BTreeMap<String, PathBuf> = root
            .lines()
            .filter_map(|line| {
                let (name, rest) = line.split_once(" = { path = \"")?;
                Some((
                    name.trim().to_owned(),
                    workspace.join(rest.split('"').next()?),
                ))
            })
            .collect();
        let code = |path: &Path| -> String {
            let whole = std::fs::read_to_string(path).unwrap();
            // The code, not its tests (this test names patterns itself).
            whole.split("#[cfg(test)]").next().unwrap().to_owned()
        };
        let mut crates: Vec<String> = Vec::new();
        // Host modules, transitively through their `crate::` paths.
        let mut modules = vec![
            "direct".to_owned(),
            "page_cache".to_owned(),
            "services".to_owned(),
            "cuda".to_owned(),
        ];
        let mut seen_modules = std::collections::BTreeSet::new();
        while let Some(module) = modules.pop() {
            if !seen_modules.insert(module.clone()) {
                continue;
            }
            // A name without its own file is an item of the crate root.
            let file = [format!("src/{module}.rs"), format!("src/{module}/mod.rs")]
                .into_iter()
                .find(|file| manifest.join(file).is_file())
                .unwrap_or_else(|| "src/lib.rs".to_owned());
            assert!(
                covered.contains(format!("runtime-host/{file}").as_str()),
                "host module {module} is outside the digest"
            );
            let text = code(&manifest.join(&file));
            for (at, _) in text.match_indices("crate::") {
                if at > 0 && text.as_bytes()[at - 1] == b'$' {
                    continue;
                }
                let rest = &text[at + 7..];
                let named: Vec<String> = match rest.strip_prefix('{') {
                    Some(group) => group[..group.find('}').unwrap()]
                        .split(',')
                        .map(|item| item.trim().split("::").next().unwrap().to_owned())
                        .collect(),
                    None => vec![
                        rest.chars()
                            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                            .collect(),
                    ],
                };
                modules.extend(named.into_iter().filter(|m| !m.is_empty() && m != "self"));
            }
            for (at, _) in text.match_indices("::") {
                let start = text[..at]
                    .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .map_or(0, |index| index + 1);
                let name = text[start..at].replace('_', "-");
                if members.contains_key(&name) {
                    crates.push(name);
                }
            }
        }
        // Workspace crates, transitively through every linked table.
        let mut seen = std::collections::BTreeSet::new();
        while let Some(name) = crates.pop() {
            if !seen.insert(name.clone()) {
                continue;
            }
            let root = &members[&name];
            assert!(
                covered.contains(format!("{name}/Cargo.toml").as_str()),
                "the direct load uses {name}, which the digest leaves out"
            );
            let mut stack = vec![root.join("src")];
            while let Some(dir) = stack.pop() {
                for entry in std::fs::read_dir(&dir).unwrap() {
                    let path = entry.unwrap().path();
                    if path.is_dir() {
                        stack.push(path);
                    } else if path.extension().is_some_and(|e| e == "rs") {
                        let relative = path.strip_prefix(root).unwrap().display().to_string();
                        let source = format!("{name}/{relative}");
                        assert!(covered.contains(source.as_str()), "{source} is outside");
                    }
                }
            }
            let cargo = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
            let mut table = String::new();
            for line in cargo.lines().map(str::trim) {
                if line.starts_with('[') {
                    table = line.to_owned();
                    if let Some((_, dependency)) =
                        table.trim_end_matches(']').split_once("dependencies.")
                        && !table.contains("dev-dependencies")
                    {
                        crates.push(dependency.to_owned());
                    }
                    continue;
                }
                let linked =
                    table.ends_with("dependencies]") && !table.contains("dev-dependencies");
                if let Some(dependency) = line.split(['=', '.']).next().map(str::trim)
                    && linked
                    && members.contains_key(dependency)
                {
                    crates.push(dependency.to_owned());
                }
            }
        }
        // Reached through services, the CUDA upload and a target-specific,
        // optional table.
        for name in [
            "runtime-host",
            "kernel-core",
            "runtime-weights",
            "runtime-cuda",
        ] {
            assert!(
                seen.contains(name) || name == "runtime-host",
                "{name} was not reached"
            );
        }
        assert!(seen_modules.contains("cuda") && seen_modules.contains("executor"));
        assert!(covered.contains("Cargo.lock"));
    }

    /// Twice the process's CPUs between the ranks, at least one and at most
    /// sixteen per rank: the GPU queue's 16 CPUs give TP4 eight per rank.
    #[test]
    fn the_default_worker_count_doubles_the_cpu_share_within_bounds() {
        assert_eq!(default_threads_per_rank_v1(16, 4), 8);
        assert_eq!(default_threads_per_rank_v1(32, 4), 16);
        assert_eq!(default_threads_per_rank_v1(48, 4), 16);
        assert_eq!(default_threads_per_rank_v1(16, 8), 4);
        assert_eq!(default_threads_per_rank_v1(1, 8), 1);
        assert_eq!(default_threads_per_rank_v1(3, 0), 6);
    }

    struct Temp(PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// One safetensors file holding `tensors` (name, dtype, shape, bytes).
    /// Drops every shard's pages from the page cache, once no read-ahead
    /// thread can advise them back.
    fn evict(checkpoint: &DirectCheckpointV1) {
        checkpoint.wait_for_read_ahead();
        for shard in &checkpoint.shards {
            shard.file.sync_all().unwrap();
            advise_v1(&shard.file, 0, 0, CacheAdviceV1::DontNeed).unwrap();
        }
    }

    /// Whether this host and file system can read `checkpoint` directly: its
    /// kernel reports page-cache residency and its files open with `O_DIRECT`.
    fn direct_reads_supported(checkpoint: &DirectCheckpointV1) -> bool {
        let shard = &checkpoint.shards[0];
        let supported = cached_pages_v1(&shard.file, 0, 1).is_ok() && shard.direct_file().is_some();
        assert!(
            supported || std::env::var_os("CHOPINFER_REQUIRE_DIRECT_IO_TESTS").is_none(),
            "this qualification requires residency reports and direct I/O"
        );
        supported
    }

    /// An image over `checkpoint` with no records and direct reads on, for
    /// reading sources through `read_source_v1`.
    fn direct_reading_image(checkpoint: Arc<DirectCheckpointV1>) -> PlannedImageV1 {
        let plans = vec![WeightPlanV1 {
            image_role: "r".into(),
            ranks: 1,
            rank: 0,
            alignment: 256,
            arena_bytes: 0,
            views: vec![],
            components: vec![],
            records: vec![],
        }];
        PlannedImageV1::new(
            checkpoint,
            plans,
            None,
            implementation_v1([0; 32]),
            DirectLoadOptionsV1 {
                threads_per_rank: 1,
                read_ahead: false,
                direct_reads: true,
                shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
            },
        )
        .unwrap()
    }

    /// A direct read the file system refuses after opening (here, one kept to
    /// a one-byte alignment it does not serve) is read through the page cache
    /// instead, and turns direct reads off for that shard; a read that fails
    /// for another reason (past the end of the file) still fails.
    #[test]
    fn refused_direct_reads_fall_back_to_the_page_cache() {
        let length = 1 << 20;
        let data: Vec<u8> = (0..length).map(|i| (i * 7 + 3) as u8).collect();
        let root = checkpoint(&[("t", "U8", vec![length as u64], data)]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
        if !direct_reads_supported(&checkpoint) {
            eprintln!("skipped: no residency report or O_DIRECT on this file system");
            return;
        }
        let shard = &checkpoint.shards[0];
        let (tensor, begin) = shard.tensors["t"].clone();
        let source = SourceAtV1 {
            shard: 0,
            begin,
            length: tensor.byte_length,
        };
        let mut expected = vec![0; 1001];
        shard.file.read_exact_at(&mut expected, begin + 3).unwrap();
        // An aligned read of the same file succeeds directly.
        let image = direct_reading_image(checkpoint.clone());
        evict(&checkpoint);
        let mut read = vec![0; 1001];
        read_source_v1(&image.inner, source, 3, &mut read).unwrap();
        assert!(read == expected);
        assert_eq!(image.stats().direct_bytes, 1001);
        // One-byte alignment: the kernel refuses the odd offset.
        let direct = shard.direct.get().unwrap().as_ref().unwrap();
        let aligned = direct.alignment;
        let misaligned = DirectFileV1 {
            file: direct.file.try_clone().unwrap(),
            alignment: 1,
            off: std::sync::atomic::AtomicBool::new(false),
            refused: AtomicU64::new(0),
        };
        let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
        assert!(checkpoint.shards[0].direct.set(Some(misaligned)).is_ok());
        let image = direct_reading_image(checkpoint.clone());
        evict(&checkpoint);
        let mut read = vec![0; 1001];
        read_source_v1(&image.inner, source, 3, &mut read).unwrap();
        assert!(read == expected);
        assert_eq!(image.stats().direct_bytes, 0);
        assert!(checkpoint.shards[0].direct_file().is_none());
        // Past the end of the file is a fault either way.
        let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
        let image = direct_reading_image(checkpoint.clone());
        evict(&checkpoint);
        let beyond = SourceAtV1 {
            length: source.length + 2 * aligned,
            ..source
        };
        let mut past = vec![0; 16];
        assert!(read_source_v1(&image.inner, beyond, source.length + aligned, &mut past).is_err());
        assert!(checkpoint.shards[0].direct_file().is_some());
    }

    /// A read whose cold runs sit on either side of a cached run tries a
    /// direct read once: the first cold run's refusal turns direct reads off
    /// for the second as well.
    #[test]
    fn a_refused_direct_read_is_not_retried_within_one_read() {
        let length: u64 = 4 << 20;
        let data: Vec<u8> = (0..length).map(|i| (i * 13 + 1) as u8).collect();
        let root = checkpoint(&[("t", "U8", vec![length], data)]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
        if !direct_reads_supported(&checkpoint) {
            eprintln!("skipped: no residency report or O_DIRECT on this file system");
            return;
        }
        let shard = &checkpoint.shards[0];
        let (tensor, begin) = shard.tensors["t"].clone();
        let source = SourceAtV1 {
            shard: 0,
            begin,
            length: tensor.byte_length,
        };
        // One-byte alignment: the odd start and end are refused.
        let direct = shard.direct.get().unwrap().as_ref().unwrap();
        let misaligned = DirectFileV1 {
            file: direct.file.try_clone().unwrap(),
            alignment: 1,
            off: std::sync::atomic::AtomicBool::new(false),
            refused: AtomicU64::new(0),
        };
        let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
        let shard = &checkpoint.shards[0];
        assert!(shard.direct.set(Some(misaligned)).is_ok());
        let image = direct_reading_image(checkpoint.clone());
        evict(&checkpoint);
        // Cache a run in the middle: cold, cached, cold.
        let mut middle = vec![0; 256 << 10];
        shard
            .file
            .read_exact_at(&mut middle, begin + (1 << 20))
            .unwrap();
        let runs = residency_runs_v1(&shard.file, begin + 3, length - 6);
        assert!(runs.len() >= 3 && runs[0].1 && !runs[1].1 && runs.last().unwrap().1);
        let mut expected = vec![0; (length - 6) as usize];
        shard.file.read_exact_at(&mut expected, begin + 3).unwrap();
        evict(&checkpoint);
        shard
            .file
            .read_exact_at(&mut middle, begin + (1 << 20))
            .unwrap();
        let mut read = vec![0; (length - 6) as usize];
        read_source_v1(&image.inner, source, 3, &mut read).unwrap();
        assert!(read == expected);
        let direct = shard.direct.get().unwrap().as_ref().unwrap();
        assert_eq!(direct.refused.load(Ordering::Relaxed), 1);
        assert_eq!(image.stats().direct_bytes, 0);
    }

    /// Direct reads of any range, aligned or not, across the aligned buffer's
    /// size and up to the end of the file, equal buffered reads; a range past
    /// the end fails.
    #[test]
    fn direct_reads_equal_buffered_reads_at_any_range() {
        let length = 3 * DIRECT_READ_BYTES as usize + 777;
        let data: Vec<u8> = (0..length).map(|i| (i * 131 + i / 4099) as u8).collect();
        let root = checkpoint(&[("t", "U8", vec![length as u64], data)]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
        let shard = &checkpoint.shards[0];
        let Some(direct) = shard.direct_file() else {
            eprintln!("skipped: the temporary file system refuses O_DIRECT");
            return;
        };
        let buffers = Mutex::new(Vec::new());
        let file_length = shard.byte_length;
        let block = DIRECT_READ_ALIGNMENT;
        let chunk = DIRECT_READ_BYTES;
        for (at, count) in [
            (0, 1),
            (0, block),
            (1, block),
            (block - 1, 2),
            (5, chunk),
            (chunk - 3, chunk + 7),
            (17, 3 * chunk),
            (file_length - 10, 10),
            (file_length - block - 5, block + 5),
            (0, file_length),
        ] {
            let mut buffered = vec![0; count as usize];
            shard.file.read_exact_at(&mut buffered, at).unwrap();
            let mut read = vec![0xee; count as usize];
            read_direct_v1(&buffers, direct, at, &mut read, None).unwrap();
            assert!(read == buffered, "{at} + {count}");
        }
        let mut past = vec![0; 2];
        assert!(read_direct_v1(&buffers, direct, file_length - 1, &mut past, None).is_err());
    }

    #[test]
    fn owned_direct_windows_cover_unaligned_eof_and_mixed_residency_without_copies() {
        let length = 2 * DIRECT_READ_BYTES as usize + 777;
        let data: Vec<_> = (0..length).map(|i| (i * 131 + i / 4099) as u8).collect();
        let root = checkpoint(&[("t", "U8", vec![length as u64], data.clone())]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
        if !direct_reads_supported(&checkpoint) {
            eprintln!("skipped: no residency report or O_DIRECT on this file system");
            return;
        }
        let shard = &checkpoint.shards[0];
        let direct = shard.direct_file().unwrap();
        let align = direct.alignment as usize;
        let source = SourceAtV1 {
            shard: 0,
            begin: shard.tensors["t"].1,
            length: length as u64,
        };
        let mut image = direct_reading_image(checkpoint.clone());
        Arc::get_mut(&mut image.inner).unwrap().io_timing = Some(io_stats::IoCounters::default());
        for mixed in [false, true] {
            let before = image.stats();
            evict(&checkpoint);
            if mixed {
                let mut middle = vec![0; 256 << 10];
                shard
                    .file
                    .read_exact_at(&mut middle, source.begin + (1 << 20))
                    .unwrap();
                let runs = residency_runs_v1(&shard.file, source.begin, source.length);
                assert!(runs.iter().any(|(_, cold)| *cold) && runs.iter().any(|(_, cold)| !cold));
            }
            let prefix = source.begin as usize % align;
            let mut storage = vec![0xa5; (length + prefix).next_multiple_of(align) + align];
            let start = storage.as_ptr().align_offset(align) + prefix;
            read_source_storage_v1(&image.inner, source, 0, &mut storage, start..start + length)
                .unwrap();
            assert_eq!(&storage[start..start + length], data);
            let delta = image.stats().since(before);
            assert!(delta.direct_bytes > 0);
            let io = delta.io_timing.unwrap();
            assert!(io.direct_calls > 1);
            assert_eq!(io.buffered_calls > 0, mixed);
            assert_eq!(io.bounce_bytes, 0);
        }
        evict(&checkpoint);
        let beyond = SourceAtV1 {
            length: source.length + align as u64,
            ..source
        };
        let mut storage = vec![0xa5; length + 4 * align];
        let start = storage.as_ptr().align_offset(align) + source.begin as usize % align;
        let before = image.stats();
        let error = read_source_storage_v1(
            &image.inner,
            beyond,
            0,
            &mut storage,
            start..start + beyond.length as usize,
        )
        .unwrap_err();
        assert!(
            error
                .0
                .contains(&std::io::Error::from(std::io::ErrorKind::UnexpectedEof).to_string()),
            "{error}"
        );
        assert!(image.stats().since(before).io_timing.unwrap().direct_calls > 0);
        assert!(shard.direct_file().is_some());
    }

    /// A range the cache holds in the middle splits into a cold run, a
    /// cached run and a cold run, together covering the range.
    #[test]
    fn mixed_ranges_split_into_cached_and_cold_runs() {
        let length = 4 << 20;
        let root = checkpoint(&[("t", "U8", vec![length], vec![7; length as usize])]);
        let checkpoint = DirectCheckpointV1::open(&root.0).unwrap();
        let file = &checkpoint.shards[0].file;
        if cached_pages_v1(file, 0, 1).is_err() {
            eprintln!("skipped: this kernel reports no page-cache residency");
            return;
        }
        evict(&checkpoint);
        let mut middle = vec![0; 512 << 10];
        file.read_exact_at(&mut middle, 1 << 20).unwrap();
        let file_length = checkpoint.shards[0].byte_length;
        let runs = residency_runs_v1(file, 0, file_length);
        assert_eq!(
            runs.iter().map(|(length, _)| length).sum::<u64>(),
            file_length
        );
        let at = |offset: u64| {
            let mut end = 0;
            runs.iter()
                .find(|(length, _)| {
                    end += length;
                    offset < end
                })
                .unwrap()
                .1
        };
        assert!(at(0), "{runs:?}");
        assert!(!at((1 << 20) + (256 << 10)), "{runs:?}");
        assert!(at(file_length - 1), "{runs:?}");
        assert_eq!(
            residency_runs_v1(file, 1 << 20, 256 << 10),
            [(256 << 10, false)]
        );
    }

    fn checkpoint(tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Temp {
        let root = std::env::temp_dir().join(format!(
            "direct-load-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            header.insert(
                (*name).into(),
                serde_json::json!({"dtype": dtype, "shape": shape,
                    "data_offsets": [payload.len(), payload.len() + bytes.len()]}),
            );
            payload.extend_from_slice(bytes);
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(root.join("model.safetensors"), file).unwrap();
        std::fs::write(root.join("config.json"), b"{}").unwrap();
        Temp(root)
    }

    #[test]
    fn hub_snapshot_accepts_linked_subdirectory_files_and_checks_replacements() {
        use std::os::unix::fs::symlink;
        let root = checkpoint(&[("vector", "BF16", vec![6], bytes(12, 1))]);
        let snapshot = root.0.join("snapshots/revision");
        std::fs::create_dir_all(snapshot.join("inference")).unwrap();
        std::fs::create_dir(root.0.join("blobs")).unwrap();
        for name in ["model.safetensors", "config.json"] {
            std::fs::rename(root.0.join(name), root.0.join("blobs").join(name)).unwrap();
            symlink(format!("../../blobs/{name}"), snapshot.join(name)).unwrap();
            symlink(
                format!("../../../blobs/{name}"),
                snapshot.join("inference").join(name),
            )
            .unwrap();
        }
        let opened = DirectCheckpointV1::open(&snapshot).unwrap();
        assert_eq!(opened.shards.len(), 2);
        assert!(
            opened
                .documents
                .iter()
                .any(|d| d.name == "inference/config.json" && d.bytes == b"{}")
        );
        opened.verify_unchanged().unwrap();
        let linked = snapshot.join("inference/model.safetensors");
        std::fs::remove_file(&linked).unwrap();
        symlink("../../../blobs/config.json", &linked).unwrap();
        assert!(opened.verify_unchanged().is_err());
        std::fs::remove_file(&linked).unwrap();
        symlink("../../../blobs/model.safetensors", &linked).unwrap();
        let linked = snapshot.join("inference/config.json");
        std::fs::write(root.0.join("config.json"), b"{}").unwrap();
        std::fs::remove_file(&linked).unwrap();
        symlink("../../../blobs/../config.json", &linked).unwrap();
        assert!(DirectCheckpointV1::open(&snapshot).is_err());
    }

    fn bytes(count: usize, seed: u8) -> Vec<u8> {
        (0..count)
            .map(|i| (i as u8).wrapping_mul(29) ^ seed)
            .collect()
    }

    fn record(
        offset: u64,
        length: u64,
        transform: WeightTransform,
        tensor: &str,
        axis: Option<u32>,
        ranges: Vec<Range<u64>>,
        stored_shape: Vec<u64>,
    ) -> WeightRecordV1 {
        WeightRecordV1 {
            arena_offset: offset,
            byte_length: length,
            transform: transform.id(),
            params: transform.encode_params().unwrap(),
            sources: vec![RecordSourceV1 {
                file: 0,
                tensor: tensor.into(),
            }],
            axis,
            ranges,
            stored_shape,
        }
    }

    /// Safetensors files and small files of immediate subdirectories are
    /// described as `dir/file`; hidden directories and large non-JSON files
    /// are not.
    #[test]
    fn subdirectories_and_small_files_are_described() {
        let root = checkpoint(&[("vector", "BF16", vec![6], bytes(12, 1))]);
        let drafter = root.0.join("drafter");
        std::fs::create_dir(&drafter).unwrap();
        std::fs::copy(
            root.0.join("model.safetensors"),
            drafter.join("draft.safetensors"),
        )
        .unwrap();
        std::fs::write(drafter.join("config.json"), b"{\"d\": 1}").unwrap();
        std::fs::write(drafter.join("mask.pt"), [7_u8; 100]).unwrap();
        std::fs::write(
            drafter.join("weights.bin"),
            vec![0_u8; SMALL_FILE_BYTES_MAX_V1 as usize + 1],
        )
        .unwrap();
        std::fs::create_dir_all(root.0.join(".cache/deep")).unwrap();
        std::fs::write(root.0.join(".cache/x.json"), b"{}").unwrap();
        let opened = DirectCheckpointV1::open(&root.0).unwrap();
        let request = opened.request("role", 1, 0, implementation_v1([0; 32]), vec![], vec![]);
        let names = |list: Vec<&str>| list.into_iter().map(String::from).collect::<Vec<_>>();
        assert_eq!(
            request
                .files
                .iter()
                .map(|f| f.name.clone())
                .collect::<Vec<_>>(),
            names(vec!["drafter/draft.safetensors", "model.safetensors"])
        );
        assert_eq!(
            request
                .documents
                .iter()
                .map(|d| d.name.clone())
                .collect::<Vec<_>>(),
            names(vec![
                "config.json",
                "drafter/config.json",
                "drafter/mask.pt"
            ])
        );
        assert_eq!(request.documents[2].bytes, [7_u8; 100]);
        WeightPlanRequestV1::decode(&request.encode()).unwrap();
    }

    /// Plans `ranks` ranks that each keep their equal K slice of every
    /// `(name, rows, row bytes, bytes)` matrix, one record after another,
    /// with each rank's expected image.
    fn k_slice_plans(
        matrices: &[(&str, u64, u64, &[u8])],
        ranks: u64,
    ) -> (Vec<WeightPlanV1>, Vec<Vec<u8>>) {
        let mut plans = Vec::new();
        let mut expected = Vec::new();
        for rank in 0..ranks {
            let mut image = Vec::new();
            let mut records = Vec::new();
            for (name, rows, row_bytes, matrix) in matrices {
                let slice = row_bytes / ranks;
                let keep = rank * slice..(rank + 1) * slice;
                let offset = image.len() as u64;
                for row in 0..*rows {
                    let base = (row * row_bytes) as usize;
                    image.extend_from_slice(
                        &matrix[base + keep.start as usize..base + keep.end as usize],
                    );
                }
                records.push(record(
                    offset,
                    rows * slice,
                    WeightTransform::Copy,
                    name,
                    Some(1),
                    vec![keep],
                    vec![*rows, slice],
                ));
            }
            plans.push(WeightPlanV1 {
                ranks: ranks as u32,
                rank: rank as u32,
                image_role: "fixture.rank-images.v1".into(),
                alignment: 256,
                arena_bytes: image.len() as u64,
                views: vec![],
                components: vec![],
                records,
            });
            expected.push(image);
        }
        (plans, expected)
    }

    fn sharing_image(
        checkpoint: &Arc<DirectCheckpointV1>,
        plans: &[WeightPlanV1],
        threads_per_rank: usize,
        shared_bytes: u64,
    ) -> PlannedImageV1 {
        PlannedImageV1::new_with_range_tracking(
            checkpoint.clone(),
            plans,
            None,
            implementation_v1([0; 32]),
            DirectLoadOptionsV1 {
                threads_per_rank,
                read_ahead: false,
                direct_reads: false,
                shared_bytes,
            },
            true,
        )
        .unwrap()
    }

    /// Reads every rank's image on its own thread, in `chunk`-byte reads.
    fn read_ranks_concurrently(
        image: &PlannedImageV1,
        chunk: usize,
    ) -> Vec<Result<Vec<u8>, String>> {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..image.inner.ranks.len())
                .map(|rank| {
                    scope.spawn(move || {
                        let mut out = vec![0xee; image.inner.ranks[rank].arena_bytes as usize];
                        let start = image.inner.offsets[rank];
                        for (index, part) in out.chunks_mut(chunk).enumerate() {
                            image.read(start + (index * chunk) as u64, part)?;
                        }
                        Ok(out)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    /// Four ranks each keep a K slice of one matrix: with room in the
    /// budget the matrix is read once for all of them, without it each rank
    /// reads its slice through covering reads; the bytes are the same.
    #[test]
    fn k_slices_of_one_tensor_read_it_once_for_every_rank() {
        let (rows, row_bytes) = (512_u64, 256_u64);
        let matrix = bytes((rows * row_bytes) as usize, 9);
        let _root = checkpoint(&[("matrix", "U8", vec![rows, row_bytes], matrix.clone())]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&_root.0).unwrap());
        let (plans, expected) = k_slice_plans(&[("matrix", rows, row_bytes, &matrix)], 4);
        let whole: Vec<u8> = expected.concat();
        for shared_bytes in [DirectLoadOptionsV1::SHARED_BYTES_DEFAULT, 0] {
            for threads in [1, 3] {
                for chunk in [7_usize, 1000, 8192, 1 << 20] {
                    let image = sharing_image(&checkpoint, &plans, threads, shared_bytes);
                    assert_eq!(image.inner.shared_sources.len(), 1);
                    let mut produced = vec![0xee; whole.len()];
                    for (index, part) in produced.chunks_mut(chunk).enumerate() {
                        image.read((index * chunk) as u64, part).unwrap();
                    }
                    assert_eq!(
                        produced, whole,
                        "budget {shared_bytes} threads {threads} chunk {chunk}"
                    );
                    let stats = image.stats();
                    if shared_bytes > 0 && chunk >= 8192 {
                        // One read of the whole matrix served all four ranks.
                        assert_eq!(stats.source_reads, 1, "chunk {chunk}");
                    }
                    if chunk >= 8192 {
                        // Covering reads, not one read per row.
                        assert!(stats.source_reads < rows, "{stats:?}");
                    }
                    assert_eq!(image.inner.retained.load(Ordering::Relaxed), 0);
                }
            }
        }
    }

    /// Eight ranks read two shared sources at once with a budget of one
    /// source: the tensors held never exceed it, not even while one is
    /// released and the other reserved, and every reservation is returned.
    #[test]
    fn shared_sources_stay_within_a_one_source_budget_across_concurrent_ranks() {
        let (rows, row_bytes) = (512_u64, 4096_u64);
        let first = bytes((rows * row_bytes) as usize, 3);
        let second = bytes((rows * row_bytes) as usize, 5);
        let _root = checkpoint(&[
            ("first", "U8", vec![rows, row_bytes], first.clone()),
            ("second", "U8", vec![rows, row_bytes], second.clone()),
        ]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&_root.0).unwrap());
        let (plans, expected) = k_slice_plans(
            &[
                ("first", rows, row_bytes, &first),
                ("second", rows, row_bytes, &second),
            ],
            8,
        );
        let budget = rows * row_bytes;
        for round in 0..20 {
            let image = sharing_image(&checkpoint, &plans, 2, budget);
            assert_eq!(image.inner.shared_sources.len(), 2);
            image
                .inner
                .retained_held
                .release_pause_micros
                .store(2000, Ordering::Relaxed);
            let produced = read_ranks_concurrently(&image, 16 << 10);
            for (rank, produced) in produced.into_iter().enumerate() {
                assert_eq!(
                    produced.unwrap(),
                    expected[rank],
                    "round {round} rank {rank}"
                );
            }
            let held = &image.inner.retained_held;
            assert!(
                held.peak.load(Ordering::Acquire) <= budget,
                "round {round}: held {} over a budget of {budget}",
                held.peak.load(Ordering::Acquire)
            );
            assert_eq!(held.now.load(Ordering::Acquire), 0);
            assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
            let stats = image.stats();
            assert!(stats.retained_peak_bytes <= budget);
            // Each source is read whole once, or each 16 KiB read's part of
            // it (128 KiB of rows) in one or two covering reads, not per row.
            let parts = (rows * row_bytes / 8).div_ceil(16 << 10) + 1;
            assert!(
                (2..=2 * (1 + 8 * parts * 2)).contains(&stats.source_reads),
                "round {round}: {stats:?}"
            );
        }
    }

    /// Eight ranks copy two replicated records at once with a budget of one
    /// record: the sections kept never exceed it, not even while one is
    /// released and the other kept, and every reservation is returned.
    #[test]
    fn shared_outputs_stay_within_a_one_output_budget_across_concurrent_ranks() {
        let (rows, row_bytes) = (256_u64, 4096_u64);
        let length = rows * row_bytes;
        let first = bytes(length as usize, 13);
        let second = bytes(length as usize, 17);
        let _root = checkpoint(&[
            ("first", "U8", vec![rows, row_bytes], first.clone()),
            ("second", "U8", vec![rows, row_bytes], second.clone()),
        ]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&_root.0).unwrap());
        let plans: Vec<_> = (0..8)
            .map(|rank| WeightPlanV1 {
                ranks: 8,
                rank,
                image_role: "fixture.rank-images.v1".into(),
                alignment: 256,
                arena_bytes: 2 * length,
                views: vec![],
                components: vec![],
                records: ["first", "second"]
                    .into_iter()
                    .enumerate()
                    .map(|(index, name)| {
                        record(
                            index as u64 * length,
                            length,
                            WeightTransform::Copy,
                            name,
                            None,
                            vec![],
                            vec![rows, row_bytes],
                        )
                    })
                    .collect(),
            })
            .collect();
        let expected = [first, second].concat();
        for round in 0..8 {
            let image = sharing_image(&checkpoint, &plans, 2, length);
            assert_eq!(image.inner.shared.len(), 2);
            image
                .inner
                .retained_held
                .release_pause_micros
                .store(2000, Ordering::Relaxed);
            let produced = read_ranks_concurrently(&image, 16 << 10);
            for (rank, produced) in produced.into_iter().enumerate() {
                assert!(produced.unwrap() == expected, "round {round} rank {rank}");
            }
            let held = &image.inner.retained_held;
            assert!(
                held.peak.load(Ordering::Acquire) <= length,
                "round {round}: held {} over a budget of {length}",
                held.peak.load(Ordering::Acquire)
            );
            assert_eq!(held.now.load(Ordering::Acquire), 0);
            assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
            assert!(image.stats().retained_peak_bytes <= length);
        }
    }

    /// A source of exactly the sharing limit is read once for both ranks;
    /// one a byte larger is read by each rank in covering reads.
    #[test]
    fn sources_up_to_the_sharing_limit_are_shared() {
        use kernel_core::weight_transform::execute::COVERING_READ_BYTES;
        for (rows, row_bytes) in [(SHARED_SOURCE_BYTES_MAX / 256, 256), (65281, 257)] {
            let length = rows * row_bytes;
            let shared = length <= SHARED_SOURCE_BYTES_MAX;
            assert!(length == SHARED_SOURCE_BYTES_MAX || length == SHARED_SOURCE_BYTES_MAX + 1);
            let matrix = bytes(length as usize, 7);
            let _root = checkpoint(&[("matrix", "U8", vec![rows, row_bytes], matrix.clone())]);
            let checkpoint = Arc::new(DirectCheckpointV1::open(&_root.0).unwrap());
            let (plans, expected) = k_slice_plans(&[("matrix", rows, row_bytes, &matrix)], 2);
            let image = sharing_image(
                &checkpoint,
                &plans,
                2,
                DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
            );
            assert_eq!(image.inner.shared_sources.len(), usize::from(shared));
            for (rank, produced) in read_ranks_concurrently(&image, 1 << 20)
                .into_iter()
                .enumerate()
            {
                assert!(produced.unwrap() == expected[rank], "{length} rank {rank}");
            }
            let stats = image.stats();
            if shared {
                assert_eq!(stats.source_reads, 1);
                assert_eq!(stats.retained_peak_bytes, length);
            } else {
                // Per rank: a read per cap's worth of rows, and one more
                // where each 1 MiB read or 4 MiB task ends a run.
                let slice = row_bytes / 2;
                let per_read = (COVERING_READ_BYTES - slice) / row_bytes + 1;
                let bound = 2 * (rows.div_ceil(per_read) + (rows * slice).div_ceil(1 << 20));
                assert!(
                    (2 * rows.div_ceil(per_read)..=bound).contains(&stats.source_reads),
                    "{stats:?} outside a bound of {bound}"
                );
                assert_eq!(stats.retained_peak_bytes, 0);
            }
            assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
            assert_eq!(image.inner.retained_held.now.load(Ordering::Acquire), 0);
        }
    }

    /// A checkpoint cut short after it was opened fails every rank's read
    /// with an error, and returns every reservation it took.
    #[test]
    fn a_checkpoint_truncated_after_opening_fails_and_releases_its_reservations() {
        let (rows, row_bytes) = (512_u64, 4096_u64);
        let matrix = bytes((rows * row_bytes) as usize, 11);
        let _root = checkpoint(&[("matrix", "U8", vec![rows, row_bytes], matrix.clone())]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&_root.0).unwrap());
        let (plans, _) = k_slice_plans(&[("matrix", rows, row_bytes, &matrix)], 4);
        let file = _root.0.join("model.safetensors");
        let full = std::fs::metadata(&file).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&file)
            .unwrap()
            .set_len(full - rows * row_bytes / 2)
            .unwrap();
        for shared_bytes in [DirectLoadOptionsV1::SHARED_BYTES_DEFAULT, 0] {
            let image = sharing_image(&checkpoint, &plans, 2, shared_bytes);
            for (rank, produced) in read_ranks_concurrently(&image, 64 << 10)
                .into_iter()
                .enumerate()
            {
                let error = produced.unwrap_err();
                assert!(
                    error.contains("model.safetensors"),
                    "budget {shared_bytes} rank {rank}: {error}"
                );
            }
            assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
            assert_eq!(image.inner.retained_held.now.load(Ordering::Acquire), 0);
        }
    }

    /// Dropping an image joins its read-ahead threads, so none advises the
    /// kernel of the checkpoint's pages after the drop (a later eviction
    /// stays evicted). An unjoined thread outlives the drop about half the
    /// time, hence the repeats.
    #[test]
    fn a_dropped_image_has_stopped_its_read_ahead() {
        let root = checkpoint(&[("t", "U8", vec![64], bytes(64, 1))]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
        for _ in 0..50 {
            dropped_image_is_released(checkpoint.clone());
        }
    }

    fn dropped_image_is_released(checkpoint: Arc<DirectCheckpointV1>) {
        let plans = (0..2)
            .map(|rank| WeightPlanV1 {
                image_role: "r".into(),
                ranks: 2,
                rank,
                alignment: 256,
                arena_bytes: 0,
                views: vec![],
                components: vec![],
                records: vec![],
            })
            .collect();
        let image = PlannedImageV1::new(
            checkpoint,
            plans,
            None,
            implementation_v1([0; 32]),
            DirectLoadOptionsV1 {
                threads_per_rank: 1,
                read_ahead: true,
                direct_reads: false,
                shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
            },
        )
        .unwrap();
        let inner = Arc::downgrade(&image.inner);
        drop(image);
        assert!(inner.upgrade().is_none());
    }

    /// A one-rank image copying a tensor of three advice calls, with read-ahead
    /// whose advice blocks until released: the calls entered so far, a signal
    /// on each entry, and the release.
    fn blocked_adviser_image() -> (
        Temp,
        PlannedImageV1,
        Arc<AtomicUsize>,
        Receiver<()>,
        std::sync::mpsc::Sender<()>,
    ) {
        let length = 2 * ADVICE_BYTES + 1;
        let root = checkpoint(&[("t", "U8", vec![length], bytes(length as usize, 5))]);
        let checkpoint = DirectCheckpointV1::open(&root.0).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let (entered, entries) = channel();
        let (release, released) = channel::<()>();
        let (entered, released) = (Mutex::new(entered), Mutex::new(released));
        let counted = calls.clone();
        let hook = move || {
            counted.fetch_add(1, Ordering::SeqCst);
            let _ = entered.lock().unwrap().send(());
            let _ = released.lock().unwrap().recv();
        };
        assert!(checkpoint.advice_hook.set(Box::new(hook)).is_ok());
        let plans = vec![WeightPlanV1 {
            image_role: "r".into(),
            ranks: 1,
            rank: 0,
            alignment: 256,
            arena_bytes: length.next_multiple_of(256),
            views: vec![],
            components: vec![],
            records: vec![record(
                0,
                length,
                WeightTransform::Copy,
                "t",
                None,
                vec![],
                vec![length],
            )],
        }];
        let image = PlannedImageV1::new(
            Arc::new(checkpoint),
            plans,
            None,
            implementation_v1([0; 32]),
            DirectLoadOptionsV1 {
                threads_per_rank: 1,
                read_ahead: true,
                direct_reads: false,
                shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
            },
        )
        .unwrap();
        entries.recv_timeout(Duration::from_secs(30)).unwrap();
        // Targets requested behind the blocked advice: the queue keeps a few.
        let queued = (0..1000)
            .filter(|at| image.request_read_ahead(0, at + length))
            .count();
        assert_eq!(queued, READ_AHEAD_PENDING);
        (root, image, calls, entries, release)
    }

    /// A read-ahead thread blocked in an advice call holds at most
    /// [`READ_AHEAD_PENDING`] targets (the helper checks it); released, it
    /// takes them all at once and the queue accepts targets again.
    #[test]
    fn read_ahead_queues_a_bounded_number_of_targets() {
        let (_root, image, calls, _entries, release) = blocked_adviser_image();
        // Disconnected, the hook no longer blocks.
        drop(release);
        let deadline = Instant::now() + Duration::from_secs(30);
        while !image.request_read_ahead(0, 1) {
            assert!(Instant::now() < deadline, "the queue did not drain");
            std::thread::sleep(Duration::from_millis(1));
        }
        drop(image);
        // The first window's three calls; the queued targets were within it.
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// An image dropped while its read-ahead thread is in an advice call,
    /// with targets queued behind it, waits for that call and no other: the
    /// rest of the window and the queued targets are not advised.
    #[test]
    fn a_dropped_image_stops_advising_after_the_call_in_progress() {
        let (_root, image, calls, _entries, release) = blocked_adviser_image();
        let inner = image.inner.clone();
        let releasing = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            release.send(()).unwrap();
            release
        });
        let started = Instant::now();
        drop(image);
        assert!(started.elapsed() < READ_AHEAD_STOP_WAIT);
        drop(releasing.join().unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.advised_bytes.load(Ordering::Relaxed), ADVICE_BYTES);
        assert_eq!(Arc::strong_count(&inner), 1);
    }

    /// Dropping an image whose read-ahead thread stays blocked in an advice
    /// call returns after [`READ_AHEAD_STOP_WAIT`]. Waiting for the
    /// checkpoint's read-ahead (as eviction does) lasts until the thread,
    /// released later, has finished that call, advised nothing more and
    /// exited.
    #[test]
    fn dropping_an_image_waits_a_bounded_time_for_a_blocked_adviser() {
        let (_root, image, calls, _entries, release) = blocked_adviser_image();
        let inner = image.inner.clone();
        let checkpoint = inner.checkpoint.clone();
        let started = Instant::now();
        drop(image);
        let waited = started.elapsed();
        assert!(waited >= READ_AHEAD_STOP_WAIT, "{waited:?}");
        assert!(
            waited < READ_AHEAD_STOP_WAIT + Duration::from_secs(10),
            "{waited:?}"
        );
        let waited_for = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let waiter = {
            let waited_for = waited_for.clone();
            std::thread::spawn(move || {
                checkpoint.wait_for_read_ahead();
                waited_for.store(true, Ordering::SeqCst);
            })
        };
        std::thread::sleep(Duration::from_millis(200));
        assert!(!waited_for.load(Ordering::SeqCst));
        release.send(()).unwrap();
        waiter.join().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.advised_bytes.load(Ordering::Relaxed), ADVICE_BYTES);
        assert_eq!(Arc::strong_count(&inner), 1);
    }

    /// Two ranks: a replicated widened vector, a row-sharded fragment matrix
    /// and a copied table placed in host memory; every chunk size gives the
    /// bytes computed from the reference packers.
    #[test]
    fn planned_reads_equal_the_reference_bytes_at_every_chunk_size() {
        // Valid E4M3 codes: a NaN code's low bit cleared.
        let matrix: Vec<u8> = bytes(64 * 32, 3)
            .into_iter()
            .map(|b| if b & 0x7f == 0x7f { b ^ 1 } else { b })
            .collect();
        let vector = bytes(2 * 6, 5);
        let table = bytes(40 * 8, 7);
        let _root = checkpoint(&[
            ("matrix", "F8_E4M3", vec![64, 32], matrix.clone()),
            ("vector", "BF16", vec![6], vector.clone()),
            ("table", "U8", vec![40, 8], table.clone()),
        ]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&_root.0).unwrap());
        let mut expected = Vec::new();
        let mut plans = Vec::new();
        for rank in 0..2_u64 {
            let rows = rank * 32..(rank + 1) * 32;
            // 20 of the 32 rows, padded to 32.
            let kept = rows.start..rows.start + 20;
            let mut raw = matrix[(kept.start * 32) as usize..(kept.end * 32) as usize].to_vec();
            raw.resize(32 * 32, 0);
            let fragment = FragmentWeightLayout {
                rows: 32,
                k: 32,
                format: FragmentWeightFormat::Fp8E4M3,
            }
            .pack(&raw)
            .unwrap()
            .bytes;
            let widened: Vec<u8> = vector
                .chunks_exact(2)
                .flat_map(|v| [0, 0, v[0], v[1]])
                .collect();
            let rows_of_table = &table[(rank * 20 * 8) as usize..((rank + 1) * 20 * 8) as usize];
            let mut image = vec![0_u8; 4 * 1024];
            image[..fragment.len()].copy_from_slice(&fragment);
            image[1024..1024 + widened.len()].copy_from_slice(&widened);
            image[2048..2048 + rows_of_table.len()].copy_from_slice(rows_of_table);
            expected.push(image);
            plans.push(WeightPlanV1 {
                ranks: 2,
                rank: rank as u32,
                image_role: "fixture.rank-images.v1".into(),
                alignment: 1024,
                arena_bytes: 4 * 1024,
                views: vec![PlannedViewV1 {
                    id: "table".into(),
                    alignment: 1024,
                    placement: ViewPlacementV1::Host,
                    offset: 2048,
                    byte_length: 160,
                }],
                components: vec![NamedBytesV1 {
                    name: format!("fixture.index.{rank}"),
                    bytes: b"{}".to_vec(),
                }],
                records: vec![
                    record(
                        0,
                        fragment.len() as u64,
                        WeightTransform::Fragment(FragmentWeightFormat::Fp8E4M3),
                        "matrix",
                        Some(0),
                        vec![kept.clone()],
                        vec![32, 32],
                    ),
                    record(
                        1024,
                        24,
                        WeightTransform::WidenBf16F32,
                        "vector",
                        None,
                        vec![],
                        vec![6],
                    ),
                    record(
                        2048,
                        160,
                        WeightTransform::Copy,
                        "table",
                        Some(0),
                        vec![rank * 20..(rank + 1) * 20],
                        vec![20, 8],
                    ),
                ],
            });
        }
        let whole: Vec<u8> = expected.concat();
        for (threads, direct_reads) in [(1, false), (3, false), (1, true), (3, true)] {
            for chunk in [1_usize, 7, 100, 512, 1024, 3000, 8192] {
                // The earlier images' read-ahead threads ended with them, so
                // nothing advises the pages back in after this.
                if direct_reads {
                    evict(&checkpoint);
                }
                let image = PlannedImageV1::new(
                    checkpoint.clone(),
                    plans.clone(),
                    None,
                    implementation_v1([0; 32]),
                    DirectLoadOptionsV1 {
                        threads_per_rank: threads,
                        read_ahead: threads > 1 && !direct_reads,
                        direct_reads,
                        shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
                    },
                )
                .unwrap();
                let mut produced = vec![0xee; whole.len()];
                for (index, part) in produced.chunks_mut(chunk).enumerate() {
                    image.read((index * chunk) as u64, part).unwrap();
                }
                assert_eq!(
                    produced, whole,
                    "threads {threads} chunk {chunk} direct {direct_reads}"
                );
                if direct_reads && direct_reads_supported(&checkpoint) {
                    assert!(image.stats().direct_bytes > 0, "chunk {chunk}");
                }
                assert_eq!(image.components().len(), 2);
                assert!(image.description().has_host_views());
                // The replicated vector was produced once and released.
                assert!(
                    image.inner.shared.iter().all(|s| s
                        .state
                        .lock()
                        .unwrap()
                        .sections
                        .iter()
                        .all(Option::is_none))
                );
            }
        }
        checkpoint.verify_unchanged().unwrap();
    }

    /// Replicated records copied into one rank and much later into the other
    /// keep at most the shared-output budget between their consumers, and
    /// still produce the same bytes.
    #[test]
    fn streamed_codes_do_not_owe_shared_section_bytes() {
        use kernel_core::contract::{DType, SwigluColumns, WeightFormat};
        use kernel_core::precision::layout::PackingPlan;
        use kernel_core::precision::{DecodeOrder, NibbleOrder, WeightLayout};
        use kernel_core::weight_transform::{PackedMatrixSource, PackedOperand, PackedSection};
        let rows = SHARED_RECORD_BYTES_MAX / 32 + 1;
        let length = rows * 32;
        let root = checkpoint(&[
            (
                "codes",
                "F8_E4M3",
                vec![rows, 32],
                vec![0x38; length as usize],
            ),
            ("scale", "BF16", vec![1], vec![0x80, 0x3f]),
        ]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
        let mut operand = PackedOperand {
            plan: PackingPlan {
                layout: WeightLayout::row_major(),
                format: WeightFormat::Fp8PerTensor,
                experts: 1,
                projections: 1,
                rows: rows as u32,
                k: 32,
                columns: SwigluColumns::ConcatenatedHalves,
                local_scale_dtype: Some(DType::Bf16),
            },
            section: PackedSection::Codes,
            decode_order: DecodeOrder::ScaleThenCode,
            matrices: vec![PackedMatrixSource {
                expert: 0,
                projection: 0,
                format: WeightFormat::Fp8PerTensor,
                rows,
                k: 32,
                row_origin: 0,
                k_origin: 0,
                nibble_order: NibbleOrder::EvenLow,
                codes: 0,
                local: Some(1),
                local_dtype: Some(DType::Bf16),
                global: None,
                source_expert: 0,
                source_experts: 1,
                rows_kept: vec![0..rows],
                k_kept: 0..32,
            }],
        };
        let mut records = Vec::new();
        for (section, offset, size) in [
            (PackedSection::Codes, 0, length),
            (PackedSection::LocalScales, length, 2),
        ] {
            operand.section = section;
            let mut item = record(
                offset,
                size,
                WeightTransform::PackedOperand(Box::new(operand.clone())),
                "codes",
                None,
                vec![],
                vec![],
            );
            item.sources.push(RecordSourceV1 {
                file: 0,
                tensor: "scale".into(),
            });
            records.push(item);
        }
        let image = PlannedImageV1::new(
            checkpoint,
            vec![WeightPlanV1 {
                ranks: 1,
                rank: 0,
                image_role: "fixture.rank-images.v1".into(),
                alignment: 32,
                arena_bytes: length + 32,
                views: vec![],
                components: vec![],
                records,
            }],
            None,
            implementation_v1([0; 32]),
            DirectLoadOptionsV1 {
                threads_per_rank: 1,
                read_ahead: false,
                direct_reads: false,
                shared_bytes: length * 2,
            },
        )
        .unwrap();
        assert!(image.inner.ranks[0].records[0].shared.is_none());
        assert!(image.inner.ranks[0].records[0].packed_sources.is_empty());
        let index = image.inner.ranks[0].records[1].shared.unwrap();
        assert_eq!(
            image.inner.shared[index].state.lock().unwrap().remaining,
            [0, 2, 0]
        );
        let mut out = [0; 2];
        image.read(length, &mut out).unwrap();
        assert_eq!(out, [0x80, 0x3f]);
        assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
    }

    #[test]
    fn shared_outputs_stay_within_their_budget() {
        let tensors: Vec<(String, Vec<u8>)> = (0..8)
            .map(|i| (format!("t{i}"), bytes(512, i as u8)))
            .collect();
        let named: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = tensors
            .iter()
            .map(|(name, data)| (name.as_str(), "U8", vec![512], data.clone()))
            .collect();
        let _root = checkpoint(&named);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&_root.0).unwrap());
        let plans: Vec<_> = (0..2)
            .map(|rank| WeightPlanV1 {
                ranks: 2,
                rank,
                image_role: "fixture.rank-images.v1".into(),
                alignment: 512,
                arena_bytes: 8 * 512,
                views: vec![],
                components: vec![],
                records: (0..8)
                    .map(|i| {
                        record(
                            i * 512,
                            512,
                            WeightTransform::Copy,
                            &format!("t{i}"),
                            None,
                            vec![],
                            vec![512],
                        )
                    })
                    .collect(),
            })
            .collect();
        let expected: Vec<u8> = tensors.iter().flat_map(|(_, data)| data.clone()).collect();
        let image = PlannedImageV1::new(
            checkpoint,
            plans,
            None,
            implementation_v1([0; 32]),
            DirectLoadOptionsV1 {
                threads_per_rank: 1,
                read_ahead: false,
                direct_reads: false,
                shared_bytes: 1024,
            },
        )
        .unwrap();
        for rank in 0..2_u64 {
            let mut produced = vec![0; 8 * 512];
            for (part, out) in produced.chunks_mut(700).enumerate() {
                image.read(rank * 8 * 512 + part as u64 * 700, out).unwrap();
            }
            assert_eq!(produced, expected, "rank {rank}");
        }
        assert!(
            image.stats().retained_peak_bytes <= 1024,
            "{} bytes held",
            image.stats().retained_peak_bytes
        );
        assert_eq!(image.inner.retained.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn plans_naming_missing_tensors_or_wrong_lengths_are_refused() {
        let _root = checkpoint(&[("vector", "BF16", vec![6], bytes(12, 1))]);
        let checkpoint = Arc::new(DirectCheckpointV1::open(&_root.0).unwrap());
        let plan = |record| WeightPlanV1 {
            ranks: 1,
            rank: 0,
            image_role: "fixture.rank-images.v1".into(),
            alignment: 64,
            arena_bytes: 64,
            views: vec![],
            components: vec![],
            records: vec![record],
        };
        let options = DirectLoadOptionsV1 {
            threads_per_rank: 1,
            read_ahead: false,
            direct_reads: false,
            shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
        };
        let good = record(
            0,
            12,
            WeightTransform::Copy,
            "vector",
            None,
            vec![],
            vec![6],
        );
        assert!(
            PlannedImageV1::new(
                checkpoint.clone(),
                vec![plan(good.clone())],
                None,
                implementation_v1([0; 32]),
                options
            )
            .is_ok()
        );
        let missing = record(
            0,
            12,
            WeightTransform::Copy,
            "absent",
            None,
            vec![],
            vec![6],
        );
        assert!(
            PlannedImageV1::new(
                checkpoint.clone(),
                vec![plan(missing)],
                None,
                implementation_v1([0; 32]),
                options
            )
            .is_err()
        );
        let long = WeightRecordV1 {
            byte_length: 16,
            ..good.clone()
        };
        assert!(
            PlannedImageV1::new(
                checkpoint.clone(),
                vec![plan(long)],
                None,
                implementation_v1([0; 32]),
                options
            )
            .is_err()
        );
        let bundle = WeightRecordV1 {
            transform: 0x8000_0000,
            params: vec![],
            ..good
        };
        // A bundle-owned transform needs the module's digest in the identity,
        // and then the bundle's transform export.
        let error = PlannedImageV1::new(
            checkpoint.clone(),
            vec![plan(bundle.clone())],
            None,
            implementation_v1([0; 32]),
            options,
        )
        .err()
        .unwrap();
        assert!(error.contains("bundle module's digest"), "{error}");
        assert!(
            PlannedImageV1::new(
                checkpoint,
                vec![plan(bundle)],
                None,
                implementation_v1([9; 32]),
                options
            )
            .is_err()
        );
    }
}
