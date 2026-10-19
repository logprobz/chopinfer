//! A model-free kernel test harness for SM120 cubins: GPU selection and
//! sessions ([`gpu_test`], [`Session`]), launches with bound buffers
//! ([`launch`]), poisoned replays and write maps ([`poison`]), lab-entry
//! conformance ([`conformance`]), the conformance kit over any backend
//! ([`kit`]), encoder conformance ([`encoder`]), receipts and timing.
//!
//! It names no model and builds no cubin. A consumer passes what it tests as
//! data: the cubins it embeds ([`Cubins`]), the catalogs it selects from
//! (`&kernels_sm120::Catalogs`), and its declarations and cases.
//! kernel-lab runs the in-tree models' cases on it; a model runs its own
//! from its own test crate.

pub mod conformance;
pub mod encoder;
pub mod eviction;
mod gpu_state;
pub mod kit;
pub mod lab_entry;
pub mod launch;
pub mod poison;
pub mod topology;

use kernel_core::contract::DType;
use kernel_core::oracle::{OracleOutput, check_numeric, numeric_bound};
use runtime_cuda::{
    Context, DeviceBuffer, Driver, GraphExec, KernelArgument, LaunchConfig, Module, Stream,
};
use serde_json::{Value, json};
use std::any::Any;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

thread_local! {
    static TIMING_LABEL: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
    static TIMING_ROUNDS: std::cell::RefCell<BTreeMap<String, Vec<Value>>> = const { std::cell::RefCell::new(BTreeMap::new()) };
}

static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

thread_local! {
    /// The entry symbols launched on this thread since the last
    /// `take_launched`.
    static LAUNCHED: std::cell::RefCell<std::collections::BTreeSet<String>> =
        const { std::cell::RefCell::new(std::collections::BTreeSet::new()) };
}

/// Records a launch of entry `symbol`. Test helpers call it where they
/// launch a named entry, so the generated invariance tests can require that
/// a case ran the entry it reports on.
pub fn launched(symbol: &str) {
    LAUNCHED.with_borrow_mut(|set| {
        set.insert(symbol.to_owned());
    });
}

/// The symbols `launched` recorded since the last call, which it clears.
pub fn take_launched() -> std::collections::BTreeSet<String> {
    LAUNCHED.with_borrow_mut(std::mem::take)
}

pub type LabResult<T> = Result<T, String>;

/// The cubins a consumer embeds, by spec name, with each one's entry
/// symbols: what [`Session::module`] loads by name. SM120's probe kernels
/// (`probe_sm120`, `probe_pdl_prefetch_sm120`) are among them for the PDL
/// checks and timing that launch them.
#[derive(Clone, Copy)]
pub struct Cubins {
    bytes: &'static [(&'static str, &'static [u8])],
    entries: &'static [(&'static str, &'static [&'static str])],
}

impl Cubins {
    /// No cubins: a consumer that loads only bytes it holds
    /// ([`Session::load_cubin`]).
    pub const NONE: Cubins = Cubins::new(&[], &[]);

    /// The cubins `bytes` by spec name and their entry symbols `entries`.
    pub const fn new(
        bytes: &'static [(&'static str, &'static [u8])],
        entries: &'static [(&'static str, &'static [&'static str])],
    ) -> Cubins {
        Cubins { bytes, entries }
    }

    /// The bytes of the cubin `name`.
    pub fn bytes(self, name: &str) -> LabResult<&'static [u8]> {
        self.bytes
            .iter()
            .find(|(cubin, _)| *cubin == name)
            .map(|(_, bytes)| *bytes)
            .ok_or_else(|| format!("unknown cubin {name}"))
    }

    /// The entry symbols of the cubin `name`.
    pub fn entries(self, name: &str) -> LabResult<&'static [&'static str]> {
        self.entries
            .iter()
            .find(|(cubin, _)| *cubin == name)
            .map(|(_, entries)| *entries)
            .ok_or_else(|| format!("unknown cubin {name}"))
    }

    /// Whether every entry of the cubin `name` is a comm entry, whose grids
    /// spin on their peer ranks' grids.
    pub fn comm(self, name: &str) -> LabResult<bool> {
        let entries = self.entries(name)?;
        Ok(!entries.is_empty() && entries.iter().all(|entry| entry.starts_with("chop_comm_")))
    }

    /// The cubin whose entries hold `symbol`.
    pub fn holding(self, symbol: &str) -> Option<&'static str> {
        self.entries
            .iter()
            .find(|(_, entries)| entries.contains(&symbol))
            .map(|&(name, _)| name)
    }
}

/// The caller may open CUDA only after selecting the GPU with this gate.
pub fn selected_ordinals() -> LabResult<Option<Vec<i32>>> {
    let single = std::env::var("CHOPINFER_KERNEL_GPU").ok();
    let multi = std::env::var("CHOPINFER_KERNEL_GPUS").ok();
    let input = match (single, multi) {
        (None, None) => return Ok(None),
        (Some(_), Some(_)) => {
            return Err("set exactly one of CHOPINFER_KERNEL_GPU and CHOPINFER_KERNEL_GPUS".into());
        }
        (Some(value), None) | (None, Some(value)) => value,
    };
    topology::parse_ordinals(&input).map(Some)
}

/// A deterministic sample of `0..len`: every `stride`-th index, the first and
/// last, and both sides of each boundary (`b - 1` and `b`), so a stride,
/// segment, strip, tile or shard indexing error still shows. Sorted, unique,
/// all below `len`.
pub fn sample_indices(len: usize, stride: usize, boundaries: &[usize]) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    let mut picked = (0..len).step_by(stride.max(1)).collect::<Vec<_>>();
    picked.push(len - 1);
    for &b in boundaries {
        picked.extend([b.saturating_sub(1), b].into_iter().filter(|&i| i < len));
    }
    picked.sort_unstable();
    picked.dedup();
    picked
}

/// Seeded BF16 values over 13 binades, so the F32 sums of their products
/// round and any change of accumulation order changes output bits (values
/// on a narrow grid sum exactly in F32 at these widths).
pub fn wide_values(len: usize, seed: u64) -> Vec<f32> {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let binade = ((state >> 8) % 13) as i32;
            let unit = (state >> 40) as f32 / (1_u64 << 24) as f32;
            let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
            bf16_to_f32(f32_to_bf16(sign * (1.0 + unit) * 2_f32.powi(-binade)))
        })
        .collect()
}

/// Whether `CHOPINFER_KERNEL_REQUIRE=1` asks for every GPU test to run.
pub fn require_all() -> LabResult<bool> {
    parse_require(std::env::var("CHOPINFER_KERNEL_REQUIRE").ok().as_deref())
}

fn parse_require(value: Option<&str>) -> LabResult<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("CHOPINFER_KERNEL_REQUIRE must be 0 or 1".into()),
    }
}

/// Reports a test that cannot run for a missing prerequisite. It prints a
/// skip notice and returns `Ok`, or fails under `CHOPINFER_KERNEL_REQUIRE=1`
/// so a skipped test cannot count as a pass.
pub fn skip(reason: &str) -> LabResult<()> {
    skip_unless(require_all()?, reason)
}

fn skip_unless(require: bool, reason: &str) -> LabResult<()> {
    if require {
        return Err(format!(
            "CHOPINFER_KERNEL_REQUIRE=1 forbids skipping {reason}"
        ));
    }
    eprintln!("skipping {reason}");
    Ok(())
}

/// Runs `run` on a session of the first selected GPU, or skips ([`skip`])
/// when none is selected. The session loads no cubin by name.
pub fn gpu_test<T>(
    need_bytes: u64,
    run: impl FnOnce(&Session) -> LabResult<T>,
) -> LabResult<Option<T>> {
    gpu_test_with(Cubins::NONE, need_bytes, run)
}

/// [`gpu_test`] on a session that loads `cubins` by name.
pub fn gpu_test_with<T>(
    cubins: Cubins,
    need_bytes: u64,
    run: impl FnOnce(&Session) -> LabResult<T>,
) -> LabResult<Option<T>> {
    let Some(ordinals) = selected_ordinals()? else {
        skip("GPU test: set CHOPINFER_KERNEL_GPU or CHOPINFER_KERNEL_GPUS to idle GPUs")?;
        return Ok(None);
    };
    let _guard = GPU_TEST_MUTEX.lock().map_err(|e| e.to_string())?;
    let session = Session::open(cubins, ordinals[0], need_bytes)?;
    run(&session).map(Some)
}

/// Runs `run` on one session per selected GPU, ranks in order, with peer
/// access between every pair, or skips ([`skip`]) when fewer than two are
/// selected. The sessions load no cubin by name.
pub fn gpu_test_multi<T>(
    need_bytes_per_rank: u64,
    run: impl FnOnce(&[Session]) -> LabResult<T>,
) -> LabResult<Option<T>> {
    gpu_test_multi_with(Cubins::NONE, need_bytes_per_rank, run)
}

/// [`gpu_test_multi`] on sessions that load `cubins` by name.
pub fn gpu_test_multi_with<T>(
    cubins: Cubins,
    need_bytes_per_rank: u64,
    run: impl FnOnce(&[Session]) -> LabResult<T>,
) -> LabResult<Option<T>> {
    let Some(ordinals) = selected_ordinals()? else {
        skip("multi-GPU test: set CHOPINFER_KERNEL_GPUS to idle GPUs")?;
        return Ok(None);
    };
    if ordinals.len() < 2 {
        skip("multi-GPU test: at least two GPU ordinals are required")?;
        return Ok(None);
    }
    let _guard = GPU_TEST_MUTEX.lock().map_err(|e| e.to_string())?;
    let watchdog = topology::Watchdog::start(&ordinals, topology::timeout()?)?;
    let sessions = ordinals
        .iter()
        .enumerate()
        .map(|(rank, &ordinal)| {
            let mut session = Session::open(cubins, ordinal, need_bytes_per_rank)?;
            session.metadata.insert("rank".into(), json!(rank));
            session
                .metadata
                .insert("rank_devices".into(), json!(ordinals));
            session.metadata.insert("folded".into(), json!(false));
            Ok(session)
        })
        .collect::<LabResult<Vec<_>>>()?;
    for source in &sessions {
        for dest in &sessions {
            if source.ordinal == dest.ordinal {
                continue;
            }
            if !source
                .context
                .can_access_peer(&dest.context)
                .map_err(|e| e.to_string())?
            {
                return Err(format!(
                    "GPU {} cannot access peer {}",
                    source.ordinal, dest.ordinal
                ));
            }
            source
                .context
                .enable_peer_access(&dest.context)
                .map_err(|e| e.to_string())?;
            source
                .context
                .default_memory_pool()
                .map_err(|e| e.to_string())?
                .set_peer_access(dest.ordinal, true)
                .map_err(|e| e.to_string())?;
        }
    }
    watchdog.attach(&sessions);
    run(&sessions).map(Some)
}

pub struct Session {
    cubins: Cubins,
    /// What consumers keep for the session's lifetime ([`Session::state`]).
    state: RefCell<Vec<Rc<dyn Any>>>,
    pub context: Context,
    pub stream: Stream,
    pub ordinal: i32,
    pub metadata: BTreeMap<String, Value>,
}

impl Session {
    /// A session on the selected GPU `ordinal` with `need_bytes` free, which
    /// loads `cubins` by name.
    pub fn open(cubins: Cubins, ordinal: i32, need_bytes: u64) -> LabResult<Self> {
        if !selected_ordinals()?.is_some_and(|items| items.contains(&ordinal)) {
            return Err("GPU access requires a selected ordinal".into());
        }
        let driver = Driver::initialize().map_err(|e| e.to_string())?;
        let context = driver
            .retain_primary_context(ordinal)
            .map_err(|e| e.to_string())?;
        let (free, total) = context.memory_info().map_err(|e| e.to_string())?;
        if free < need_bytes {
            return Err(format!(
                "GPU {ordinal} has {free} free bytes, test requires {need_bytes}"
            ));
        }
        let stream = context.stream().map_err(|e| e.to_string())?;
        let mut metadata = BTreeMap::new();
        metadata.insert("ordinal".into(), json!(ordinal));
        metadata.insert(
            "uuid".into(),
            json!(driver.device_uuid(ordinal).map_err(|e| e.to_string())?),
        );
        metadata.insert(
            "compute_capability".into(),
            json!(format!(
                "{}.{}",
                context.device().compute_major(),
                context.device().compute_minor()
            )),
        );
        metadata.insert(
            "l2_cache_bytes".into(),
            json!(gpu_state::l2_bytes(ordinal)?),
        );
        metadata.insert("memory_free_bytes".into(), json!(free));
        metadata.insert("memory_total_bytes".into(), json!(total));
        let driver_version = driver.version().map_err(|e| e.to_string())?;
        metadata.insert("cuda_driver_version".into(), json!(driver_version));
        metadata.insert(
            "cuda_version".into(),
            json!(format!(
                "{}.{}",
                driver_version / 1000,
                (driver_version % 1000) / 10
            )),
        );
        metadata.insert(
            "sm_count".into(),
            json!(context.device().multiprocessor_count()),
        );
        // CUDA ordinals can be remapped by CUDA_VISIBLE_DEVICES. nvidia-smi
        // uses physical indices, so query the UUID obtained from CUDA.
        let uuid = metadata["uuid"]
            .as_str()
            .ok_or("missing device UUID")?
            .to_owned();
        // nvidia-smi does not report the SM count; the driver attribute does.
        let fields = "name,clocks.current.sm,clocks.current.memory,driver_version";
        let gpu = smi(&[
            &format!("--query-gpu={fields}"),
            "--format=csv,noheader,nounits",
            &format!("--id={uuid}"),
        ])?;
        let parts: Vec<_> = gpu.trim().split(',').map(str::trim).collect();
        if parts.len() != 4 {
            return Err(format!("unexpected nvidia-smi GPU metadata: {gpu}"));
        }
        for (key, value) in [
            "device_name",
            "clock_sm_mhz",
            "clock_memory_mhz",
            "driver_version",
        ]
        .into_iter()
        .zip(parts)
        {
            metadata.insert(key.into(), json!(value));
        }
        let residents = resident_processes(&uuid)?;
        metadata.insert("compute_processes".into(), json!(residents));
        metadata.insert(
            "measurement_label".into(),
            json!(if residents.is_empty()
                && std::env::var("CHOPINFER_KERNEL_SHARED_DEVICE").as_deref() != Ok("1")
            {
                "exclusive-device"
            } else {
                "shared-device"
            }),
        );
        Ok(Self {
            cubins,
            state: RefCell::default(),
            context,
            stream,
            ordinal,
            metadata,
        })
    }

    pub fn allocate(&self, bytes: usize) -> LabResult<DeviceBuffer> {
        self.context
            .default_memory_pool()
            .map_err(|e| e.to_string())?
            .allocate(&self.stream, bytes)
            .map_err(|e| e.to_string())
    }

    pub fn upload_f32(&self, values: &[f32]) -> LabResult<DeviceBuffer> {
        let buffer = self.allocate(std::mem::size_of_val(values))?;
        buffer
            .copy_from_f32(values, &self.stream)
            .map_err(|e| e.to_string())?;
        Ok(buffer)
    }

    pub fn upload_bf16(&self, values: &[u16]) -> LabResult<DeviceBuffer> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let buffer = self.allocate(bytes.len())?;
        buffer
            .copy_from_bytes(&bytes, &self.stream)
            .map_err(|e| e.to_string())?;
        Ok(buffer)
    }

    pub fn download_bf16(&self, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<u16>> {
        let mut bytes = vec![0; len.checked_mul(2).ok_or("BF16 download length overflow")?];
        buffer
            .copy_to_bytes(&mut bytes, &self.stream)
            .map_err(|e| e.to_string())?;
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    pub fn download_f32(&self, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<f32>> {
        let mut values = vec![0.0; len];
        buffer
            .copy_to_f32(&mut values, &self.stream)
            .map_err(|e| e.to_string())?;
        Ok(values)
    }

    /// The value of type `T` a consumer keeps for this session's lifetime,
    /// such as modules it loads once per session: `init`'s on first use.
    /// A failed `init` keeps nothing, so the next call tries again.
    pub fn state<T: 'static>(&self, init: impl FnOnce() -> LabResult<T>) -> LabResult<Rc<T>> {
        let held = self
            .state
            .borrow()
            .iter()
            .find_map(|value| value.clone().downcast::<T>().ok());
        if let Some(value) = held {
            return Ok(value);
        }
        let value = Rc::new(init()?);
        self.state.borrow_mut().push(value.clone());
        Ok(value)
    }

    /// The cubins this session loads by name.
    pub fn cubins(&self) -> Cubins {
        self.cubins
    }

    /// Loads a cubin another crate embeds, such as a bundle's own kernels.
    pub fn load_cubin(&self, bytes: &[u8]) -> LabResult<Module> {
        self.context.load_module(bytes).map_err(|e| e.to_string())
    }

    /// Loads the cubin `name` of this session's [`Cubins`].
    pub fn module(&self, name: &str) -> LabResult<Module> {
        self.load_cubin(self.cubins.bytes(name)?)
    }

    pub fn capture(
        &self,
        launch: impl FnOnce(&Stream) -> LabResult<()>,
    ) -> LabResult<runtime_cuda::GraphExec> {
        self.stream.begin_capture().map_err(|e| e.to_string())?;
        launch(&self.stream)?;
        self.stream
            .end_capture()
            .map_err(|e| e.to_string())?
            .instantiate()
            .map_err(|e| e.to_string())
    }

    pub fn receipt(&self, kind: &str, mut data: BTreeMap<String, Value>) -> LabResult<PathBuf> {
        merge_metadata(&mut data, &self.metadata);
        let uuid = self.metadata["uuid"]
            .as_str()
            .ok_or("missing device UUID")?;
        let telemetry = take_timing_records(uuid);
        data.insert("timing_rounds".into(), json!(telemetry));
        let end_processes = resident_processes(uuid)?;
        let start_shared = self.metadata["measurement_label"] == "shared-device";
        data.insert("compute_processes_end".into(), json!(end_processes));
        if start_shared || !end_processes.is_empty() {
            data.insert("measurement_label".into(), json!("shared-device"));
        }
        write_receipt(kind, self.ordinal, &data)
    }
}

/// Merges session metadata while preserving the test's execution mode.
fn merge_metadata(data: &mut BTreeMap<String, Value>, metadata: &BTreeMap<String, Value>) {
    let declared = data.get("execution_mode").cloned();
    for (key, value) in metadata {
        data.insert(key.clone(), value.clone());
    }
    if let Some(mode) = declared {
        data.insert("execution_mode".into(), mode);
    }
}

fn smi(args: &[&str]) -> LabResult<String> {
    let output = Command::new("nvidia-smi")
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi failed: {}{}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}

fn resident_processes(uuid: &str) -> LabResult<Vec<Value>> {
    let processes = smi(&[
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])?;
    Ok(processes
        .lines()
        .filter_map(|line| {
            let fields: Vec<_> = line.split(',').map(str::trim).collect();
            (fields.len() == 4
                && fields[0] == uuid
                && fields[1].parse::<u32>().ok() != Some(std::process::id()))
            .then(|| {
                json!({"pid": fields[1], "process_name": fields[2],
                "used_memory_mib": fields[3]})
            })
        })
        .collect())
}

/// Where GPU receipts are written and probe receipts are read:
/// `CHOPINFER_KERNEL_RECEIPTS`, else `$CHOPINFER_DATA/kernel-lab`, else a
/// directory under the system temporary directory. Never the source tree.
pub fn receipt_root() -> PathBuf {
    if let Some(root) = std::env::var_os("CHOPINFER_KERNEL_RECEIPTS") {
        return PathBuf::from(root);
    }
    match std::env::var_os("CHOPINFER_DATA") {
        Some(data) => PathBuf::from(data).join("kernel-lab"),
        None => std::env::temp_dir().join("chopinfer-kernel-lab"),
    }
}

pub fn write_receipt(
    kind: &str,
    ordinal: i32,
    fields: &BTreeMap<String, Value>,
) -> LabResult<PathBuf> {
    if kind.is_empty()
        || !kind
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("invalid receipt kind".into());
    }
    let root = receipt_root().join(kind);
    fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let path = root.join(format!("{nanos}-gpu{ordinal}.json"));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| e.to_string())?;
    let sorted = sort_json(serde_json::to_value(fields).map_err(|e| e.to_string())?);
    serde_json::to_writer_pretty(&mut file, &sorted).map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())?;
    Ok(path)
}

fn sort_json(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<_> = map.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut result = serde_json::Map::new();
            for (key, value) in entries {
                result.insert(key, sort_json(value));
            }
            Value::Object(result)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sort_json).collect()),
        other => other,
    }
}

pub fn check_output(
    actual: &[f32],
    reference: &OracleOutput,
    dtype: DType,
    receipt: &mut BTreeMap<String, Value>,
) -> LabResult<()> {
    let actual: Vec<f64> = actual.iter().map(|&v| f64::from(v)).collect();
    if let Some((index, (&value, &expected))) = actual
        .iter()
        .zip(&reference.values)
        .enumerate()
        .max_by(|left, right| {
            let a = (left.1.0 - left.1.1).abs();
            let b = (right.1.0 - right.1.1).abs();
            a.total_cmp(&b)
        })
    {
        receipt.insert(
            "worst_element".into(),
            json!({"index": index,
            "actual": value, "reference": expected, "error": (value - expected).abs()}),
        );
    }
    match check_numeric(&actual, reference, dtype) {
        Ok(()) => {
            receipt.insert("correct".into(), json!(true));
            Ok(())
        }
        Err(worst) => {
            receipt.insert("correct".into(), json!(false));
            receipt.insert(
                "worst_element".into(),
                json!({"index": worst.index,
                "actual": worst.actual, "reference": worst.reference,
                "bound": worst.bound, "error": worst.error}),
            );
            Err(format!("numeric gate failed at element {}", worst.index))
        }
    }
}

/// FP8 KV contract: attention over FP8 E4M3 KV against the attention of the
/// unrounded BF16 keys and values (`exact`). Each BF16 output element may
/// differ by the numerical contract's bound on `stored`, the oracle over the stored
/// codes, plus `TOLERANCE_SIGMAS` standard deviations `sigma` of the
/// storage error (`kernel_core::kv_fp8::attention_sigma`).
///
/// The sigma term is a statistical allowance that assumes independent
/// rounding errors, which holds for the tests' random operands. Correlated
/// rounding, such as one value repeated across many keys, can exceed it with
/// correct arithmetic. The receipt therefore reports the device arithmetic
/// error (`actual` against `stored`, in units of its numerical bound) apart
/// from the storage error (`stored` against `exact`, in sigmas), and a
/// refusal names the part that exceeded.
pub fn check_fp8_kv_output(
    actual: &[f32],
    exact: &[f64],
    sigma: &[f64],
    stored: &OracleOutput,
    receipt: &mut BTreeMap<String, Value>,
) -> LabResult<()> {
    if actual.len() != exact.len()
        || exact.len() != sigma.len()
        || exact.len() != stored.values.len()
    {
        return Err("FP8 KV check shapes differ".into());
    }
    let sigmas = kernel_core::kv_fp8::TOLERANCE_SIGMAS;
    let (mut worst, mut worst_excess, mut max_sigmas) = (0, f64::MIN, 0.0_f64);
    // A non-finite error or unit is infinitely many units, never dropped.
    let ratio = |error: f64, unit: f64| {
        if error.is_finite() && unit.is_finite() {
            error / unit.max(f64::MIN_POSITIVE)
        } else {
            f64::INFINITY
        }
    };
    let (mut max_device_bounds, mut max_storage_sigmas) = (0.0_f64, 0.0_f64);
    for index in 0..actual.len() {
        let error = (f64::from(actual[index]) - exact[index]).abs();
        let arithmetic = numeric_bound(stored, index, DType::Bf16);
        let bound = arithmetic + sigmas * sigma[index];
        let deviation = sigma[index].max(f64::MIN_POSITIVE);
        max_sigmas = max_sigmas.max(error / deviation);
        let device = (f64::from(actual[index]) - stored.values[index]).abs();
        max_device_bounds = max_device_bounds.max(ratio(device, arithmetic));
        let storage = (stored.values[index] - exact[index]).abs();
        max_storage_sigmas = max_storage_sigmas.max(ratio(storage, deviation));
        let excess = if actual[index].is_finite() && bound.is_finite() && exact[index].is_finite() {
            error - bound
        } else {
            f64::INFINITY
        };
        if excess > worst_excess {
            (worst, worst_excess) = (index, excess);
        }
    }
    // The arithmetic gate itself decides whether the device output is correct.
    let device: Vec<f64> = actual.iter().map(|&v| f64::from(v)).collect();
    let device_correct = check_numeric(&device, stored, DType::Bf16).is_ok();
    receipt.insert(
        "fp8_kv_vs_bf16".into(),
        json!({"tolerance_sigmas": sigmas, "max_error_sigmas": max_sigmas,
            "storage_model": "independent uniform E4M3 rounding, first order",
            "max_device_error_bounds": max_device_bounds,
            "device_within_bound": device_correct,
            "max_storage_error_sigmas": max_storage_sigmas,
            "worst_index": worst, "worst_actual": actual.get(worst),
            "worst_exact": exact.get(worst), "worst_sigma": sigma.get(worst),
            "worst_excess": worst_excess, "correct": worst_excess <= 0.0}),
    );
    if worst_excess <= 0.0 {
        Ok(())
    } else if device_correct {
        Err(format!(
            "FP8 KV storage error exceeds the independent-rounding allowance at element \
             {worst} ({max_storage_sigmas:.1} sigmas); arithmetic over the stored codes \
             is within its bound"
        ))
    } else {
        Err(format!(
            "FP8 KV storage tolerance failed at element {worst}; arithmetic over the stored \
             codes exceeds its bound by up to {max_device_bounds:.2} times"
        ))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub median_ms: f32,
    pub min_ms: f32,
    pub rounds: usize,
}

pub fn race(left: Timing, right: Timing) -> Option<std::cmp::Ordering> {
    if left.median_ms * 1.10 < right.median_ms {
        Some(std::cmp::Ordering::Less)
    } else if right.median_ms * 1.10 < left.median_ms {
        Some(std::cmp::Ordering::Greater)
    } else {
        None
    }
}

/// Label all measurements made by `run`, restoring the previous label on exit.
pub fn with_timing_label<T>(operation: &str, run: impl FnOnce() -> T) -> T {
    struct Restore(String);
    impl Drop for Restore {
        fn drop(&mut self) {
            TIMING_LABEL.with_borrow_mut(|label| *label = std::mem::take(&mut self.0));
        }
    }
    let _restore = Restore(
        TIMING_LABEL.with_borrow_mut(|label| std::mem::replace(label, operation.to_owned())),
    );
    run()
}
fn timing_label() -> String {
    TIMING_LABEL.with_borrow(|label| {
        if label.is_empty() {
            "unlabelled".to_owned()
        } else {
            label.clone()
        }
    })
}
fn next_measurement() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
/// Time an operation with its own label, independent of the receiving receipt.
pub fn time_named(
    session: &Session,
    operation: &str,
    warmup: usize,
    rounds: usize,
    work: impl FnMut(&Stream) -> LabResult<()>,
) -> LabResult<Timing> {
    with_timing_label(operation, || time(session, warmup, rounds, work))
}

/// Cold-cache timing. Use `time_prepared` to produce warm activations after eviction.
pub fn time(
    session: &Session,
    warmup: usize,
    rounds: usize,
    mut work: impl FnMut(&Stream) -> LabResult<()>,
) -> LabResult<Timing> {
    if std::env::var("CHOPINFER_KERNEL_EVICTION_AB").as_deref() == Ok("1") {
        let paired = time_ab(
            session,
            warmup,
            rounds,
            [eviction::EvictionMode::Write, eviction::EvictionMode::Read],
            |_, stream| work(stream),
        )?;
        return Ok(selected_arm(&paired, eviction::EvictionMode::selected()?));
    }
    time_prepared(
        session,
        warmup,
        rounds,
        eviction::EvictionMode::selected()?,
        |_| Ok(()),
        work,
    )
}

/// Evict, produce activations, then time only the target. Preparation is outside events.
pub fn time_prepared(
    session: &Session,
    warmup: usize,
    rounds: usize,
    mode: eviction::EvictionMode,
    mut produce: impl FnMut(&Stream) -> LabResult<()>,
    mut work: impl FnMut(&Stream) -> LabResult<()>,
) -> LabResult<Timing> {
    validate_timing(warmup, rounds)?;
    let eviction = eviction::Eviction::new(session, mode)?;
    let monitor = monitor(session);
    let measurement = next_measurement();
    let mut samples = Vec::with_capacity(rounds);
    for iteration in 0..warmup + rounds {
        let elapsed = timing_round(session, &eviction, &mut produce, &mut work)?;
        if iteration >= warmup {
            record_round(
                session,
                &monitor,
                json!({"measurement": measurement, "round": iteration - warmup,
                "eviction": mode.name(), "eviction_bytes": eviction.bytes, "elapsed_ms": elapsed}),
            );
            samples.push(elapsed);
        }
    }
    Ok(summarize_samples(samples))
}
fn validate_timing(warmup: usize, rounds: usize) -> LabResult<()> {
    if rounds < 3 || warmup == 0 {
        return Err("timing requires warmup and at least three rounds".into());
    }
    Ok(())
}
fn monitor(session: &Session) -> LabResult<gpu_state::Monitor> {
    gpu_state::Monitor::open(
        session.metadata["uuid"]
            .as_str()
            .ok_or("missing device UUID")?,
    )
}
fn record_round(session: &Session, monitor: &LabResult<gpu_state::Monitor>, data: Value) {
    store_round(
        session.metadata["uuid"].as_str().unwrap_or(""),
        round_data(session, monitor, data),
    );
}
fn take_timing_records(uuid: &str) -> Vec<Value> {
    TIMING_ROUNDS.with_borrow_mut(|rounds| rounds.remove(uuid).unwrap_or_default())
}
fn store_round(uuid: &str, data: Value) {
    TIMING_ROUNDS.with_borrow_mut(|rounds| rounds.entry(uuid.to_owned()).or_default().push(data));
}
fn round_data(
    session: &Session,
    monitor: &LabResult<gpu_state::Monitor>,
    mut data: Value,
) -> Value {
    let baseline = session.metadata["clock_sm_mhz"]
        .as_str()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    data["operation"] = json!(timing_label());
    data["gpu_state"] = match monitor {
        Ok(monitor) => monitor.snapshot(baseline),
        Err(error) => json!({"error": error, "flags": ["telemetry_unavailable"]}),
    };
    data
}
fn graph_round(measurement: u64, round: usize, ranks: Vec<Value>) -> Value {
    let flags = ranks
        .iter()
        .flat_map(|rank| rank["gpu_state"]["flags"].as_array().into_iter().flatten())
        .filter_map(Value::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    json!({"kind": "multi_graph", "operation": timing_label(), "measurement": measurement, "round": round, "flags": flags, "ranks": ranks})
}
fn timing_round(
    session: &Session,
    eviction: &eviction::Eviction,
    produce: &mut impl FnMut(&Stream) -> LabResult<()>,
    work: &mut impl FnMut(&Stream) -> LabResult<()>,
) -> LabResult<f32> {
    let module = session.module("probe_sm120")?;
    let gate_fn = module
        .function("chop_probe_gate_v1")
        .map_err(|e| e.to_string())?;
    let gate = session.allocate(4)?;
    let release = session.context.stream().map_err(|e| e.to_string())?;
    gate.copy_from_bytes(&0_u32.to_ne_bytes(), &session.stream)
        .map_err(|e| e.to_string())?;
    eviction.run(&session.stream)?;
    produce(&session.stream)?;
    let start = session.context.timed_event().map_err(|e| e.to_string())?;
    let end = session.context.timed_event().map_err(|e| e.to_string())?;
    gate_fn
        .launch(
            &session.stream,
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?,
            &[KernelArgument::Buffer(&gate)],
        )
        .map_err(|e| e.to_string())?;
    let scheduled = (|| {
        start.record(&session.stream).map_err(|e| e.to_string())?;
        work(&session.stream)?;
        end.record(&session.stream).map_err(|e| e.to_string())
    })();
    gate.copy_from_bytes(&1_u32.to_ne_bytes(), &release)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    scheduled?;
    start.elapsed_ms(&end).map_err(|e| e.to_string())
}

#[derive(Debug)]
pub struct PairedTiming {
    pub arms: [Timing; 2],
    /// Median of paired B/A ratios, rather than a ratio of medians.
    pub paired_ratio: f32,
}
fn arm_order(round: usize) -> [usize; 2] {
    if round % 2 == 0 { [0, 1] } else { [1, 0] }
}
fn summarize_pairs(samples: [Vec<f32>; 2]) -> LabResult<PairedTiming> {
    if samples[0].len() < 3
        || samples[0].len() != samples[1].len()
        || samples
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err("paired timing needs at least three finite positive pairs".into());
    }
    let ratios = samples[0]
        .iter()
        .zip(&samples[1])
        .map(|(a, b)| b / a)
        .collect();
    let [a, b] = samples;
    Ok(PairedTiming {
        arms: [summarize_samples(a), summarize_samples(b)],
        paired_ratio: summarize_samples(ratios).median_ms,
    })
}
/// Alternate A/B and B/A each round; independently evict before each arm.
pub fn time_ab(
    session: &Session,
    warmup: usize,
    rounds: usize,
    modes: [eviction::EvictionMode; 2],
    mut work: impl FnMut(usize, &Stream) -> LabResult<()>,
) -> LabResult<PairedTiming> {
    time_ab_prepared(session, warmup, rounds, modes, |_, _| Ok(()), &mut work)
}
fn selected_arm(paired: &PairedTiming, mode: eviction::EvictionMode) -> Timing {
    paired.arms[match mode {
        eviction::EvictionMode::Write => 0,
        eviction::EvictionMode::Read => 1,
    }]
}
fn collect_pairs(
    warmup: usize,
    rounds: usize,
    mut sample: impl FnMut(usize) -> LabResult<f32>,
    mut record: impl FnMut(usize, usize, f32),
) -> LabResult<PairedTiming> {
    let mut samples = [Vec::with_capacity(rounds), Vec::with_capacity(rounds)];
    for iteration in 0..warmup + rounds {
        let round = if iteration < warmup {
            iteration
        } else {
            iteration - warmup
        };
        for arm in arm_order(round) {
            let elapsed = sample(arm)?;
            if iteration >= warmup {
                samples[arm].push(elapsed);
                record(arm, round, elapsed);
            }
        }
    }
    summarize_pairs(samples)
}
/// Time alternating A/B arms with an untimed producer before each sample.
/// Eviction precedes the producer; only `work` is enclosed by CUDA events.
pub fn time_ab_prepared(
    session: &Session,
    warmup: usize,
    rounds: usize,
    modes: [eviction::EvictionMode; 2],
    mut produce: impl FnMut(usize, &Stream) -> LabResult<()>,
    mut work: impl FnMut(usize, &Stream) -> LabResult<()>,
) -> LabResult<PairedTiming> {
    validate_timing(warmup, rounds)?;
    let evictions = [
        eviction::Eviction::new(session, modes[0])?,
        eviction::Eviction::new(session, modes[1])?,
    ];
    let monitor = monitor(session);
    let measurement = next_measurement();
    let result = collect_pairs(
        warmup,
        rounds,
        |arm| {
            timing_round(
                session,
                &evictions[arm],
                &mut |stream| produce(arm, stream),
                &mut |stream| work(arm, stream),
            )
        },
        |arm, round, elapsed| {
            record_round(
                session,
                &monitor,
                json!({"measurement": measurement, "round": round, "arm": arm, "order": arm_order(round), "eviction": modes[arm].name(), "elapsed_ms": elapsed}),
            );
        },
    )?;
    TIMING_ROUNDS.with_borrow_mut(|records| {
        records
            .entry(session.metadata["uuid"].as_str().unwrap_or("").to_owned())
            .or_default()
            .push(json!({
                "kind": "paired_summary", "operation": timing_label(), "measurement": measurement,
                "arm_median_ms": result.arms.map(|arm| arm.median_ms),
                "paired_ratio": result.paired_ratio, "rounds": rounds,
            }))
    });
    Ok(result)
}

/// Target-graph microseconds after a predecessor, with L2 flushed before both.
/// Record events outside capture: ordinary captured events represent graph
/// dependencies, not externally observable timing nodes. The lab stream gate
/// holds execution until both graphs and events have been enqueued.
pub fn time_after(
    session: &Session,
    mut prefix: impl FnMut() -> LabResult<()>,
    mut target: impl FnMut() -> LabResult<()>,
) -> LabResult<f32> {
    let prefix_graph = session.capture(|_| prefix())?;
    let target_graph = session.capture(|_| target())?;
    let mode = eviction::EvictionMode::selected()?;
    let timing = if std::env::var("CHOPINFER_KERNEL_EVICTION_AB").as_deref() == Ok("1") {
        let paired = time_ab_prepared(
            session,
            1,
            15,
            [eviction::EvictionMode::Write, eviction::EvictionMode::Read],
            |_, stream| prefix_graph.launch(stream).map_err(|e| e.to_string()),
            |_, stream| target_graph.launch(stream).map_err(|e| e.to_string()),
        )?;
        selected_arm(&paired, mode)
    } else {
        time_prepared(
            session,
            1,
            15,
            mode,
            |stream| prefix_graph.launch(stream).map_err(|e| e.to_string()),
            |stream| target_graph.launch(stream).map_err(|e| e.to_string()),
        )?
    };
    Ok(timing.median_ms * 1000.0)
}

#[derive(Debug)]
pub struct GraphTiming {
    pub calls_per_graph: usize,
    pub critical_path: Timing,
    pub per_rank: Vec<Timing>,
    pub gpu_rounds: Vec<Value>,
}

fn summarize_samples(mut samples: Vec<f32>) -> Timing {
    samples.sort_by(f32::total_cmp);
    Timing {
        median_ms: if samples.len() % 2 == 0 {
            samples[samples.len() / 2 - 1] * 0.5 + samples[samples.len() / 2] * 0.5
        } else {
            samples[samples.len() / 2]
        },
        min_ms: samples[0],
        rounds: samples.len(),
    }
}

fn summarize_multi_graph(by_rank: Vec<Vec<f32>>, calls_per_graph: usize) -> GraphTiming {
    let rounds = by_rank[0].len();
    let critical = (0..rounds)
        .map(|round| {
            by_rank
                .iter()
                .map(|samples| samples[round])
                .fold(0.0_f32, f32::max)
        })
        .collect();
    GraphTiming {
        calls_per_graph,
        critical_path: summarize_samples(critical),
        per_rank: by_rank.into_iter().map(summarize_samples).collect(),
        gpu_rounds: Vec::new(),
    }
}

/// Time a captured sequence on every rank after stream gates release together.
/// Each round uses the slowest rank as its critical path after shared eviction.
pub fn time_multi_graph(
    sessions: &[Session],
    graphs: &[GraphExec],
    calls_per_graph: usize,
    warmup: usize,
    rounds: usize,
) -> LabResult<GraphTiming> {
    if sessions.is_empty()
        || sessions.len() != graphs.len()
        || calls_per_graph == 0
        || warmup == 0
        || rounds < 3
    {
        return Err(
            "multi-graph timing requires matching ranks, calls, warmup, and three rounds".into(),
        );
    }
    let evictions = sessions
        .iter()
        .map(|s| eviction::Eviction::new(s, eviction::EvictionMode::selected()?))
        .collect::<LabResult<Vec<_>>>()?;
    let monitors = sessions.iter().map(monitor).collect::<Vec<_>>();
    let modules = sessions
        .iter()
        .map(|session| session.module("probe_sm120"))
        .collect::<LabResult<Vec<_>>>()?;
    // Rank 0's gate releases the others with peer stores; the host sets only
    // rank 0's flag, so ranks start within about one P2P latency.
    let gates = modules
        .iter()
        .enumerate()
        .map(|(rank, module)| {
            module
                .function(if rank == 0 {
                    "chop_probe_gate_release_v1"
                } else {
                    "chop_probe_gate_v1"
                })
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let flags = sessions
        .iter()
        .map(|session| session.allocate(4))
        .collect::<LabResult<Vec<_>>>()?;
    // A pointer array needs one slot; one rank passes its own flag, count zero.
    let peer_flags = flags[if flags.len() > 1 { 1 } else { 0 }..]
        .iter()
        .map(|flag| flag.device_address_at(0, 4).map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    let release = sessions[0].context.stream().map_err(|e| e.to_string())?;
    let mut by_rank = vec![Vec::with_capacity(rounds); sessions.len()];
    let mut gpu_rounds = Vec::with_capacity(rounds);
    let measurement = next_measurement();
    for iteration in 0..warmup + rounds {
        for ((session, flag), eviction) in sessions.iter().zip(&flags).zip(&evictions) {
            eviction.run(&session.stream)?;
            flag.copy_from_bytes(&0_u32.to_ne_bytes(), &session.stream)
                .map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        let mut starts = Vec::with_capacity(sessions.len());
        let mut ends = Vec::with_capacity(sessions.len());
        let scheduled = (|| -> LabResult<()> {
            for (rank, session) in sessions.iter().enumerate() {
                let config = launch::simple_config(
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    LaunchConfig::NO_CLUSTER,
                    false,
                )?;
                let count = (flags.len() - 1) as u32;
                let releasing = [
                    KernelArgument::Buffer(&flags[rank]),
                    KernelArgument::DevicePointerArray(&peer_flags),
                    KernelArgument::U32(count),
                ];
                let waiting = [KernelArgument::Buffer(&flags[rank])];
                gates[rank]
                    .launch(
                        &session.stream,
                        config,
                        if rank == 0 {
                            &releasing[..]
                        } else {
                            &waiting[..]
                        },
                    )
                    .map_err(|e| e.to_string())?;
                let start = session.context.timed_event().map_err(|e| e.to_string())?;
                let end = session.context.timed_event().map_err(|e| e.to_string())?;
                start.record(&session.stream).map_err(|e| e.to_string())?;
                graphs[rank]
                    .launch(&session.stream)
                    .map_err(|e| e.to_string())?;
                end.record(&session.stream).map_err(|e| e.to_string())?;
                starts.push(start);
                ends.push(end);
            }
            Ok(())
        })();
        flags[0]
            .copy_from_bytes(&1_u32.to_ne_bytes(), &release)
            .map_err(|e| e.to_string())?;
        scheduled?;
        let mut rank_states = Vec::with_capacity(sessions.len());
        for (rank, end) in ends.iter().enumerate() {
            end.synchronize().map_err(|e| e.to_string())?;
            let elapsed = starts[rank].elapsed_ms(end).map_err(|e| e.to_string())?;
            if iteration >= warmup {
                by_rank[rank].push(elapsed);
                rank_states.push(round_data(&sessions[rank], &monitors[rank],
                    json!({"rank": rank, "ordinal": sessions[rank].ordinal, "eviction": evictions[rank].mode.name(), "elapsed_ms": elapsed})));
            }
        }
        if iteration >= warmup {
            let data = graph_round(measurement, iteration - warmup, rank_states);
            store_round(
                sessions[0].metadata["uuid"]
                    .as_str()
                    .ok_or("missing device UUID")?,
                data.clone(),
            );
            gpu_rounds.push(data);
        }
    }
    let mut result = summarize_multi_graph(by_rank, calls_per_graph);
    result.gpu_rounds = gpu_rounds;
    Ok(result)
}

#[cfg(test)]
mod graph_timing_tests {
    use super::*;

    #[test]
    fn operation_labels_are_scoped_and_measurements_are_unique() {
        assert_eq!(timing_label(), "unlabelled");
        with_timing_label("launch_floor", || {
            assert_eq!(timing_label(), "launch_floor");
            with_timing_label("target", || assert_eq!(timing_label(), "target"));
            assert_eq!(timing_label(), "launch_floor");
        });
        assert_eq!(timing_label(), "unlabelled");
        assert_ne!(next_measurement(), next_measurement());
    }
    #[test]
    fn aggregate_receipt_keeps_throttled_peer_state() {
        let data = graph_round(
            9,
            0,
            vec![
                json!({"rank": 0, "gpu_state": {"flags": []}}),
                json!({"rank": 1, "gpu_state": {"flags": ["power_or_thermal_throttle"]}}),
            ],
        );
        store_round("aggregate-test", data.clone());
        let records = take_timing_records("aggregate-test");
        assert_eq!(records, vec![data]);
        assert_eq!(records[0]["flags"], json!(["power_or_thermal_throttle"]));
        assert_eq!(records[0]["ranks"][1]["rank"], 1);
        assert!(take_timing_records("aggregate-test").is_empty());
    }
    #[test]
    fn prepared_pairs_keep_target_samples_in_the_selected_arm() {
        let mut order = Vec::new();
        let paired = collect_pairs(
            1,
            3,
            |arm| Ok([2.0, 7.0][arm]),
            |arm, round, elapsed| order.push((arm, round, elapsed)),
        )
        .unwrap();
        assert_eq!(
            order,
            vec![
                (0, 0, 2.0),
                (1, 0, 7.0),
                (1, 1, 7.0),
                (0, 1, 2.0),
                (0, 2, 2.0),
                (1, 2, 7.0)
            ]
        );
        assert_eq!(
            selected_arm(&paired, eviction::EvictionMode::Write).median_ms,
            2.0
        );
        assert_eq!(
            selected_arm(&paired, eviction::EvictionMode::Read).median_ms,
            7.0
        );
        assert_eq!(paired.paired_ratio, 3.5);
    }
    #[test]
    fn even_rounds_average_the_middle_samples() {
        assert_eq!(summarize_samples(vec![7.0, 1.0, 5.0, 3.0]).median_ms, 4.0);
    }
    #[test]
    fn balanced_pairs_and_statistics() {
        assert_eq!(
            (0..4).map(arm_order).collect::<Vec<_>>(),
            vec![[0, 1], [1, 0], [0, 1], [1, 0]]
        );
        let paired = summarize_pairs([vec![1.0, 10.0, 100.0], vec![50.0, 2.0, 30.0]]).unwrap();
        assert_eq!(paired.arms[0].median_ms, 10.0);
        assert_eq!(paired.arms[1].median_ms, 30.0);
        assert_eq!(paired.paired_ratio, 0.3);
        assert!(summarize_pairs([vec![0.0; 3], vec![1.0; 3]]).is_err());
        assert!(summarize_pairs([vec![1.0; 3], vec![1.0; 2]]).is_err());
    }
    #[test]
    fn critical_path_uses_each_rounds_slowest_rank() {
        let timing = summarize_multi_graph(vec![vec![1.0, 10.0, 2.0], vec![9.0, 2.0, 3.0]], 64);
        assert_eq!(timing.critical_path.median_ms, 9.0);
        assert_eq!(timing.per_rank[0].median_ms, 2.0);
        assert_eq!(timing.per_rank[1].median_ms, 3.0);
    }
}

#[cfg(test)]
mod sample_tests {
    use super::sample_indices;

    #[test]
    fn samples_keep_ends_strides_and_both_sides_of_boundaries() {
        assert_eq!(
            sample_indices(100, 32, &[50, 99, 100, 0]),
            vec![0, 32, 49, 50, 64, 96, 98, 99]
        );
        assert_eq!(sample_indices(3, 0, &[]), vec![0, 1, 2]);
        assert!(sample_indices(0, 8, &[4]).is_empty());
    }
}

#[cfg(test)]
mod skip_tests {
    use super::{parse_require, skip_unless};

    #[test]
    fn require_mode_turns_skips_into_failures() {
        assert_eq!(parse_require(None), Ok(false));
        assert_eq!(parse_require(Some("0")), Ok(false));
        assert_eq!(parse_require(Some("1")), Ok(true));
        assert!(parse_require(Some("yes")).is_err());
        assert_eq!(skip_unless(false, "test: no GPU"), Ok(()));
        let error = skip_unless(true, "test: no GPU").unwrap_err();
        assert!(error.contains("test: no GPU"), "{error}");
    }
}

#[cfg(test)]
mod receipt_tests {
    use super::*;

    #[test]
    fn receipts_keep_the_declared_execution_mode() {
        for folded in [None, Some(false)] {
            let mut metadata =
                BTreeMap::from([("execution_mode".to_owned(), json!("eager-no-pdl"))]);
            if let Some(folded) = folded {
                metadata.insert("folded".to_owned(), json!(folded));
            }
            let mut graph = BTreeMap::from([
                ("execution_mode".to_owned(), json!("graph-no-pdl")),
                ("graph_replays".to_owned(), json!(56)),
            ]);
            merge_metadata(&mut graph, &metadata);
            assert_eq!(graph["execution_mode"], json!("graph-no-pdl"));
            assert_eq!(graph["graph_replays"], json!(56));
            assert_eq!(
                graph.get("folded"),
                folded.map(|value| json!(value)).as_ref()
            );
            assert!(!graph.contains_key("pdl"));
            assert!(!graph.contains_key("eager_replays"));
        }
    }
}
