//! Bounded resident copies of authenticated mapped rows, shared across ranks.
mod batch;
mod publish;
pub(crate) mod replay;
mod storage;

use super::{MappedRowsV1, RowsKeyV1};
use crate::device::mapped_gather::MappedGatherPlaneV1;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use storage::LockedBytes;

const SHARDS: usize = 64;
const WAYS: usize = 8;
const ENTRY_META: usize = 24; // Row tag, last use and valid-plane mask, all U64.
const HEADER_BYTES: usize = 64 * 1024;
const MAX_PLANES: usize = 64;
const DEFAULT_SET_BYTES: usize = 1 << 30;
const DEFAULT_TOTAL_BYTES: usize = 2 << 30;
const HOST_RESERVE_BYTES: u64 = 16 << 30;

#[derive(Clone, Copy, Debug)]
struct Config {
    set_bytes: usize,
    total_bytes: usize,
}

impl Config {
    fn environment() -> Result<Option<Self>, String> {
        Self::parse(|name| std::env::var_os(name))
    }

    fn parse(get: impl Fn(&str) -> Option<std::ffi::OsString>) -> Result<Option<Self>, String> {
        if get("CHOPINFER_MAPPED_HOT_ROWS").is_some_and(|v| v != "1") {
            return Ok(None);
        }
        let bytes = |name: &str, default| -> Result<usize, String> {
            match get(name) {
                Some(value) => value
                    .to_str()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| format!("{name} requires an integer byte budget")),
                None => Ok(default),
            }
        };
        let config = Self {
            set_bytes: bytes("CHOPINFER_MAPPED_HOT_ROWS_SET_BYTES", DEFAULT_SET_BYTES)?,
            total_bytes: bytes("CHOPINFER_MAPPED_HOT_ROWS_BYTES", DEFAULT_TOTAL_BYTES)?,
        };
        if config.set_bytes < HEADER_BYTES + SHARDS * super::page_bytes_v1()
            || config.total_bytes < config.set_bytes
        {
            return Err("resident mapped-row budgets cannot hold one cache set".into());
        }
        Ok(Some(config))
    }
}

struct Budget {
    limit: usize,
    used: AtomicU64,
    skipped: AtomicU64,
}

struct Charge {
    budget: Arc<Budget>,
    bytes: usize,
}

impl Budget {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Option<Charge> {
        let reserved = self
            .used
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(bytes as u64)
                    .filter(|&next| next <= self.limit as u64)
            });
        if reserved.is_err() {
            self.skipped.fetch_add(1, Ordering::Relaxed);
            None
        } else {
            Some(Charge {
                budget: Arc::clone(self),
                bytes,
            })
        }
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.budget
            .used
            .fetch_sub(self.bytes as u64, Ordering::Relaxed);
    }
}

type Key = Vec<(RowsKeyV1, usize)>;
struct Registry {
    config: Config,
    budget: Arc<Budget>,
    caches: Mutex<BTreeMap<Key, Weak<Cache>>>,
    fallback_logged: AtomicBool,
    allocation_unavailable: AtomicBool,
}

impl Registry {
    fn fallback(&self, reason: &str) {
        if !self.fallback_logged.swap(true, Ordering::Relaxed) {
            report_fallback(reason);
        }
    }
}

fn report_fallback(reason: &str) {
    runtime_engine::logging::line(
        serde_json::json!({
            "schema":"chopinfer.mapped-row-cache.v1", "phase":"uncached",
            "reason":reason,
        })
        .to_string(),
    );
}

static REGISTRY: OnceLock<Result<Option<Registry>, String>> = OnceLock::new();

fn registry() -> Result<Option<&'static Registry>, String> {
    match REGISTRY.get_or_init(|| {
        Config::environment()
            .or_else(|error| {
                report_fallback(&error);
                Ok(None)
            })
            .map(|config| {
                config.map(|config| Registry {
                    config,
                    budget: Arc::new(Budget {
                        limit: config.total_bytes,
                        used: AtomicU64::new(0),
                        skipped: AtomicU64::new(0),
                    }),
                    caches: Mutex::new(BTreeMap::new()),
                    fallback_logged: AtomicBool::new(false),
                    allocation_unavailable: AtomicBool::new(false),
                })
            })
    }) {
        Ok(value) => Ok(value.as_ref()),
        Err(error) => Err(error.clone()),
    }
}

struct Shard {
    memory: LockedBytes,
    clock: u64,
}

pub(crate) struct Plane {
    cache: Arc<Cache>,
    index: usize,
    pub(crate) width: usize,
}

struct Cache {
    id: u64,
    key: Key,
    planes: Vec<(usize, usize)>,
    row_bytes: usize,
    locked_bytes: usize,
    bucket_bytes: usize,
    buckets_per_shard: usize,
    shards: Vec<Mutex<Shard>>,
    charge: Charge,
    hits: AtomicU64,
    misses: AtomicU64,
    insertions: AtomicU64,
    evictions: AtomicU64,
    contention: AtomicU64,
    copies: AtomicU64,
}

fn word(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("cache word"))
}
fn put(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

impl Cache {
    fn reset_for_measurement(&self) -> Result<(), String> {
        for shard in &self.shards {
            let mut shard = shard.lock().map_err(|_| "resident row shard is poisoned")?;
            let bytes = shard.memory.bytes();
            for bucket in 0..self.buckets_per_shard {
                let begin = bucket * self.bucket_bytes;
                bytes[begin..begin + WAYS * ENTRY_META].fill(0);
            }
            shard.clock = 0;
        }
        for counter in [
            &self.hits,
            &self.misses,
            &self.insertions,
            &self.evictions,
            &self.contention,
            &self.copies,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        Ok(())
    }

    fn measurement_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "cache_id":self.id, "row_bytes":self.row_bytes,
            "capacity_rows":SHARDS*self.buckets_per_shard*WAYS,
            "locked_bytes":self.locked_bytes, "charged_bytes":self.charge.bytes,
            "hits":self.hits.load(Ordering::Relaxed), "misses":self.misses.load(Ordering::Relaxed),
            "insertions":self.insertions.load(Ordering::Relaxed),
            "evictions":self.evictions.load(Ordering::Relaxed),
            "contention":self.contention.load(Ordering::Relaxed),
            "copies":self.copies.load(Ordering::Relaxed),
        })
    }

    fn new(key: Key, charge: Charge) -> Result<Self, String> {
        Self::allocate(key, charge, LockedBytes::new)
    }

    fn allocate(
        key: Key,
        charge: Charge,
        mut allocate: impl FnMut(usize) -> std::io::Result<LockedBytes>,
    ) -> Result<Self, String> {
        let mut row_bytes = 0_usize;
        let mut planes = Vec::new();
        for &(_, width) in &key {
            planes.push((row_bytes, width));
            row_bytes = row_bytes
                .checked_add(width)
                .ok_or("resident row width overflow")?;
        }
        let bucket_bytes = row_bytes
            .checked_add(ENTRY_META)
            .and_then(|n| n.checked_mul(WAYS))
            .ok_or("resident bucket overflow")?;
        let page = super::page_bytes_v1();
        let shard_bytes = ((charge.bytes - HEADER_BYTES) / SHARDS) / page * page;
        let buckets_per_shard = shard_bytes / bucket_bytes;
        if buckets_per_shard == 0 {
            return Err("resident mapped-row set is too small for its row width".into());
        }
        let shards = (0..SHARDS)
            .map(|_| {
                allocate(shard_bytes)
                    .map(|memory| Mutex::new(Shard { memory, clock: 0 }))
                    .map_err(|error| format!("lock resident mapped-row memory: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Ok(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            key,
            planes,
            row_bytes,
            locked_bytes: SHARDS * shard_bytes,
            bucket_bytes,
            buckets_per_shard,
            shards,
            charge,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            insertions: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            contention: AtomicU64::new(0),
            copies: AtomicU64::new(0),
        })
    }

    fn bucket(&self, row: u32) -> (usize, usize) {
        let mut hash = u64::from(row);
        hash = (hash ^ (hash >> 16)).wrapping_mul(0x45d9f3b);
        hash = (hash ^ (hash >> 16)).wrapping_mul(0x45d9f3b);
        hash ^= hash >> 16;
        (
            hash as usize % SHARDS,
            (hash as usize / SHARDS % self.buckets_per_shard) * self.bucket_bytes,
        )
    }

    fn report(&self, phase: &str) {
        runtime_engine::logging::line(serde_json::json!({
            "schema":"chopinfer.mapped-row-cache.v1", "phase":phase, "cache_id":self.id,
            "mapping_offsets":self.key.iter().map(|(key,_)| key.offset).collect::<Vec<_>>(),
            "row_bytes":self.row_bytes, "planes":self.planes.len(), "locked_bytes":self.locked_bytes,
            "budget_bytes":self.charge.budget.limit, "charged_bytes":self.charge.bytes,
            "process_charged_bytes":self.charge.budget.used.load(Ordering::Relaxed),
            "capacity_rows":SHARDS*self.buckets_per_shard*WAYS,
            "hits":self.hits.load(Ordering::Relaxed), "misses":self.misses.load(Ordering::Relaxed),
            "insertions":self.insertions.load(Ordering::Relaxed), "evictions":self.evictions.load(Ordering::Relaxed),
            "contention":self.contention.load(Ordering::Relaxed), "copies":self.copies.load(Ordering::Relaxed),
            "budget_skips":self.charge.budget.skipped.load(Ordering::Relaxed),
            "counter_scope":"row-plane copies, including rank fanout; partial-row misses do not insert",
            "snapshot_consistency":"independent atomic counters"
        }).to_string());
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        self.report("release");
    }
}

impl Plane {
    /// Copy only the exact requested bytes. A contended shard falls back to
    /// the original mapping. No lock is held while a source page may fault.
    pub(crate) fn copy(&self, table: &[u8], row: u32, within: usize, out: &mut [u8]) {
        let cache = &self.cache;
        let (shard, bucket) = cache.bucket(row);
        let tag = u64::from(row) + 1;
        let bit = 1_u64 << self.index;
        let (offset, width) = cache.planes[self.index];
        let mut hit = false;
        if let Ok(mut shard) = cache.shards[shard].try_lock() {
            shard.clock = shard.clock.saturating_add(1);
            let clock = shard.clock;
            let bytes = shard.memory.bytes();
            for way in 0..WAYS {
                let meta = bucket + way * ENTRY_META;
                if word(bytes, meta) == tag && word(bytes, meta + 16) & bit != 0 {
                    let from = bucket + WAYS * ENTRY_META + way * cache.row_bytes + offset + within;
                    out.copy_from_slice(&bytes[from..from + out.len()]);
                    put(bytes, meta + 8, clock);
                    hit = true;
                    break;
                }
            }
        } else {
            cache.contention.fetch_add(1, Ordering::Relaxed);
        }
        if hit {
            cache.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            cache.misses.fetch_add(1, Ordering::Relaxed);
            let source = row as usize * width + within;
            out.copy_from_slice(&table[source..source + out.len()]);
            if within == 0 && out.len() == width {
                self.store_full_row(row, out);
            }
        }
        if cache.copies.fetch_add(1, Ordering::Relaxed) % 65536 == 65535 {
            cache.report("sample");
        }
    }
}

fn attach(registry: &Registry, mappings: Vec<(Arc<MappedRowsV1>, usize)>) -> Result<(), String> {
    attach_with(registry, mappings, Cache::new)
}

fn attach_with(
    registry: &Registry,
    mappings: Vec<(Arc<MappedRowsV1>, usize)>,
    allocate: impl FnOnce(Key, Charge) -> Result<Cache, String>,
) -> Result<(), String> {
    let mut members = BTreeMap::new();
    for (mapping, width) in mappings {
        if mapping.key.identity.is_none() {
            registry.fallback("resident rows require an authenticated mapping");
            return Ok(());
        }
        members.entry((mapping.key, width)).or_insert(mapping);
    }
    if members.is_empty() || members.len() > MAX_PLANES {
        return Err("resident row set has an unsupported plane count".into());
    }
    let key = members.keys().copied().collect::<Key>();
    if key
        .windows(2)
        .any(|pair| pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1)
    {
        return Err("one mapped plane cannot have two resident row widths".into());
    }
    if key.iter().any(|(_, width)| *width == 0) {
        return Err("resident rows require a nonzero plane width".into());
    }
    if key.iter().any(|(k, w)| {
        *w == 0 || !k.length.is_multiple_of(*w as u64) || k.identity != key[0].0.identity
    }) {
        return Err("resident row set planes have incompatible identities or row widths".into());
    }
    for ((_, width), mapping) in &members {
        if let Some(plane) = mapping.resident.get()
            && (plane.width != *width || plane.cache.key != key)
        {
            return Err("mapping already belongs to another resident row set".into());
        }
    }
    let mut caches = registry
        .caches
        .lock()
        .map_err(|_| "resident registry is poisoned")?;
    caches.retain(|_, cache| cache.strong_count() > 0);
    let cache = if let Some(cache) = caches.get(&key).and_then(Weak::upgrade) {
        cache
    } else {
        if registry.allocation_unavailable.load(Ordering::Relaxed) {
            return Ok(());
        }
        let Some(charge) = registry.budget.reserve(registry.config.set_bytes) else {
            registry.fallback("resident row budget exhausted");
            return Ok(());
        };
        let cache = match allocate(key.clone(), charge) {
            Ok(cache) => Arc::new(cache),
            Err(error) => {
                registry
                    .allocation_unavailable
                    .store(true, Ordering::Relaxed);
                registry.fallback(&error);
                return Ok(());
            }
        };
        cache.report("ready");
        caches.insert(key.clone(), Arc::downgrade(&cache));
        cache
    };
    for (index, ((_, width), mapping)) in members.into_iter().enumerate() {
        if let Some(plane) = mapping.resident.get() {
            if plane.width != width || plane.cache.key != key {
                return Err("mapping already belongs to another resident row set".into());
            }
        } else {
            mapping
                .resident
                .set(Plane {
                    cache: Arc::clone(&cache),
                    index,
                    width,
                })
                .map_err(|_| "resident mapping registration raced")?;
        }
    }
    Ok(())
}

/// Register planes sharing the same checked ID span, deduplicating rank views.
/// The first prepared group has first claim on the process memory budget.
pub(crate) fn register(planes: &[MappedGatherPlaneV1]) -> Result<(), String> {
    let Some(registry) = registry()? else {
        return Ok(());
    };
    match available_memory() {
        Some(available) => register_checked(registry, planes, available),
        None => {
            registry.fallback("host memory availability is unknown");
            Ok(())
        }
    }
}

fn available_memory() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("MemAvailable:")?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024)
            })
        })
}

/// Register a CPU-only row set before either synchronous or asynchronous copies.
pub(crate) fn register_group(mappings: Vec<(Arc<MappedRowsV1>, usize)>) -> Result<(), String> {
    let Some(registry) = registry()? else {
        return Ok(());
    };
    match available_memory() {
        Some(available) if memory_permits(registry, available) => attach(registry, mappings),
        Some(_) => Ok(()),
        None => {
            registry.fallback("host memory availability is unknown");
            Ok(())
        }
    }
}

fn register_checked(
    registry: &Registry,
    planes: &[MappedGatherPlaneV1],
    available: u64,
) -> Result<(), String> {
    if !memory_permits(registry, available) {
        return Ok(());
    }
    let mut groups = BTreeMap::<(usize, u64), Vec<_>>::new();
    for plane in planes {
        if plane.row_bytes == 0 {
            return Err("resident rows require a nonzero plane width".into());
        }
        groups
            .entry((
                plane.index_offset,
                plane.destination.length / plane.row_bytes as u64,
            ))
            .or_default()
            .push((Arc::clone(&plane.mapping), plane.row_bytes));
    }
    for group in groups.into_values() {
        attach(registry, group)?;
    }
    Ok(())
}

fn memory_permits(registry: &Registry, available: u64) -> bool {
    let unreserved =
        (registry.budget.limit as u64).saturating_sub(registry.budget.used.load(Ordering::Relaxed));
    if available < unreserved.saturating_add(HOST_RESERVE_BYTES) {
        registry.fallback("resident mapped rows would consume the host memory reserve");
        return false;
    }
    true
}

#[cfg(test)]
mod tests;

/// The explicit measurement controller clears tags under the copy-path locks.
/// It neither changes row bytes nor releases any memory, mapping or device work.
pub(super) fn reset_for_measurement() -> Result<(), String> {
    let Some(Ok(Some(registry))) = REGISTRY.get() else {
        return Err("no resident row registry to reset".into());
    };
    let caches = registry
        .caches
        .lock()
        .map_err(|_| "resident registry is poisoned")?
        .values()
        .filter_map(Weak::upgrade)
        .collect::<Vec<_>>();
    if caches.is_empty() {
        return Err("no resident row sets to reset".into());
    }
    for cache in caches {
        cache.reset_for_measurement()?;
    }
    Ok(())
}

pub(super) fn measurement_snapshot() -> serde_json::Value {
    let Some(Ok(Some(registry))) = REGISTRY.get() else {
        return serde_json::json!({"enabled":false,"sets":[]});
    };
    let Ok(caches) = registry.caches.lock() else {
        return serde_json::json!({"enabled":false,"sets":[],"reason":"registry poisoned"});
    };
    let sets = caches
        .values()
        .filter_map(Weak::upgrade)
        .map(|cache| cache.measurement_snapshot())
        .collect::<Vec<_>>();
    serde_json::json!({
        "enabled":true, "sets":sets,
        "allocation_unavailable":registry.allocation_unavailable.load(Ordering::Relaxed),
    })
}
