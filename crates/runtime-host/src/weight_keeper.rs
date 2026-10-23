//! Rank images kept on the device across server processes (opt-in).
//!
//! A keeper (`chopinfer weights keep`) loads a package's planned rank images
//! once, through the same host services and upload path a server uses, into
//! device memory another process can map, and lends that memory over a Unix
//! socket in a private directory. A server started with `--weight-keeper
//! DIR` asks the keeper for the rank images it is about to load. The keeper
//! lends one only when its role, rank, identity, length and device equal
//! what the server planned, and both processes opened the same checkpoint
//! files, unchanged. The identity is the SHA-256 of the plan's identity
//! document, which covers the checkpoint, the host's transform sources and
//! the bundle module, so a keeper built from other sources never matches.
//! The server maps what it is lent instead of loading it and loads the rest
//! itself; under `--weight-keeper-policy require` anything not lent fails
//! the start instead.
//!
//! A lease lasts as long as the server's connection. The driver keeps lent
//! memory alive while any process maps it, so a keeper that exits takes no
//! memory from a server that maps it.
//!
//! Every message is one line of JSON; descriptors travel with the keeper's
//! answer. The directory must be the caller's own and closed to everyone
//! else, and each side checks the other runs as the same user.

pub(crate) mod fds;
pub(crate) mod load;

use std::io::Write;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// The protocol and registry schema.
pub const SCHEMA_V1: &str = "chopinfer.weight-keeper.v1";
/// The keeper's socket, in its directory.
pub const SOCKET_NAME_V1: &str = "keeper.sock";
/// What the keeper holds and how many leases are open, in its directory.
pub const REGISTRY_NAME_V1: &str = "registry.json";

const MESSAGE_LIMIT: usize = 1 << 20;
/// How long a server waits for the keeper's answer.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a server waits for a keeper it released to exit.
const RELEASE_TIMEOUT: Duration = Duration::from_secs(60);
const ACCEPT_POLL: Duration = Duration::from_millis(20);

/// What a server does when the keeper does not lend an image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeeperPolicyV1 {
    /// Load it itself. When nothing is lent and the keeper holds no other
    /// lease, ask the keeper to release its memory first.
    Prefer,
    /// Fail the start.
    Require,
}

impl KeeperPolicyV1 {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "prefer" => Ok(Self::Prefer),
            "require" => Ok(Self::Require),
            _ => Err("--weight-keeper-policy must be prefer or require".to_owned()),
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Prefer => "prefer",
            Self::Require => "require",
        }
    }
}

/// Where one host instance's rank arenas come from.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ArenaSourceV1 {
    /// Allocated from each context's pool and loaded.
    #[default]
    Load,
    /// Allocated as memory another process can map, and loaded: a keeper.
    Lend,
    /// Mapped from the keeper in `directory` where it lends a rank image,
    /// loaded otherwise.
    Borrow {
        directory: PathBuf,
        policy: KeeperPolicyV1,
    },
}

static SELECTED: OnceLock<ArenaSourceV1> = OnceLock::new();

/// Selects where every instance of this process takes its rank arenas from.
/// Selecting the same source again changes nothing; another is refused.
pub fn select_arena_source_v1(source: ArenaSourceV1) -> Result<(), String> {
    match SELECTED.get() {
        Some(selected) if *selected == source => Ok(()),
        Some(_) => Err("another rank arena source is already selected".to_owned()),
        None => SELECTED
            .set(source)
            .map_err(|_| "the rank arena source is already selected".to_owned()),
    }
}

/// What `weights keep` asked for: the directory a keeper serves, and
/// whether borrowers map its images read-only.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeepRequestV1 {
    pub directory: PathBuf,
    pub read_only: bool,
}

static KEEP: OnceLock<KeepRequestV1> = OnceLock::new();

/// Makes this process a keeper: `serve` loads its rank images for lending
/// and serves them from `request.directory` instead of serving requests.
pub fn select_keep_v1(request: KeepRequestV1) -> Result<(), String> {
    select_arena_source_v1(ArenaSourceV1::Lend)?;
    KEEP.set(request)
        .map_err(|_| "this process already keeps weights".to_owned())
}

/// The keep request of this process, if it is a keeper.
pub(crate) fn keep_request_v1() -> Option<&'static KeepRequestV1> {
    KEEP.get()
}

/// The selected arena source; loading unless one was selected.
pub(crate) fn selected_arena_source_v1() -> ArenaSourceV1 {
    SELECTED.get().cloned().unwrap_or_default()
}

/// One rank image as a server plans it and a keeper holds it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RankImageV1 {
    pub image_role: String,
    pub rank: u32,
    /// Hex SHA-256 of the rank's plan identity document.
    pub identity: String,
    /// The physical device, named the same in every process on the host.
    pub device: String,
    /// The arena's device bytes.
    pub byte_length: u64,
}

/// The checkpoint a server or keeper opened: its header identity and each
/// file's stamp ([`crate::direct::DirectCheckpointV1::stamps`]).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointV1 {
    pub identity: String,
    pub stamps: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum OperationV1 {
    Lease,
    Status,
    Release,
    Stop,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RequestV1 {
    schema: String,
    op: OperationV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checkpoint: Option<CheckpointV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    images: Vec<RankImageV1>,
}

/// The keeper's answer for one image asked for. Lent answers carry the
/// descriptors of the reply, in order.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct AnswerV1 {
    lent: bool,
    #[serde(default)]
    mapped_bytes: u64,
    #[serde(default)]
    read_only: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReplyV1 {
    schema: String,
    ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    images: Vec<AnswerV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    registry: Option<serde_json::Value>,
}

impl ReplyV1 {
    fn new(ok: bool, reason: Option<String>) -> Self {
        Self {
            schema: SCHEMA_V1.to_owned(),
            ok,
            reason,
            images: Vec::new(),
            registry: None,
        }
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// Refuses a directory that is a link, not this user's, or open to others.
fn check_private_directory(directory: &Path) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(directory)
        .map_err(|error| format!("keeper directory {}: {error}", directory.display()))?;
    if !metadata.is_dir() {
        return Err(format!(
            "keeper directory {} is not a directory",
            directory.display()
        ));
    }
    if metadata.uid() != fds::effective_uid() || metadata.mode() & 0o077 != 0 {
        return Err(format!(
            "keeper directory {} must be this user's and closed to group and others",
            directory.display()
        ));
    }
    Ok(())
}

/// Refuses an entry of the keeper directory that is not this user's.
fn check_own_entry(path: &Path, socket: bool) -> Result<(), String> {
    use std::os::unix::fs::FileTypeExt;
    let metadata =
        std::fs::symlink_metadata(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let kind = metadata.file_type();
    let expected = if socket {
        kind.is_socket()
    } else {
        kind.is_file()
    };
    if !expected || metadata.uid() != fds::effective_uid() {
        return Err(format!(
            "{} is not this user's keeper {}",
            path.display(),
            if socket { "socket" } else { "registry" }
        ));
    }
    Ok(())
}

fn read_registry(directory: &Path) -> Result<serde_json::Value, String> {
    let path = directory.join(REGISTRY_NAME_V1);
    check_own_entry(&path, false)?;
    let file = runtime_weights::open_nofollow(&path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let registry: serde_json::Value =
        serde_json::from_reader(std::io::Read::take(file, MESSAGE_LIMIT as u64))
            .map_err(|error| format!("{}: {error}", path.display()))?;
    if registry.get("schema").and_then(serde_json::Value::as_str) != Some(SCHEMA_V1) {
        return Err(format!("{} is not a {SCHEMA_V1} registry", path.display()));
    }
    Ok(registry)
}

/// Connects to the keeper in `directory`, after checking the directory, the
/// socket and that the peer is this user's registered keeper.
fn connect(directory: &Path) -> Result<UnixStream, String> {
    check_private_directory(directory)?;
    let socket = directory.join(SOCKET_NAME_V1);
    check_own_entry(&socket, true)?;
    let registry = read_registry(directory)?;
    let stream = UnixStream::connect(&socket)
        .map_err(|error| format!("connect {}: {error}", socket.display()))?;
    let peer = fds::peer(&stream).map_err(|error| format!("keeper credentials: {error}"))?;
    if peer.uid != fds::effective_uid() {
        return Err("the keeper runs as another user".to_owned());
    }
    if registry.get("pid").and_then(serde_json::Value::as_i64) != Some(i64::from(peer.pid)) {
        return Err("the keeper is not the process its registry names".to_owned());
    }
    stream
        .set_read_timeout(Some(ANSWER_TIMEOUT))
        .and_then(|()| stream.set_write_timeout(Some(ANSWER_TIMEOUT)))
        .map_err(|error| error.to_string())?;
    Ok(stream)
}

fn send(
    stream: &UnixStream,
    message: &impl Serialize,
    fds: &[BorrowedFd<'_>],
) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(message).map_err(std::io::Error::other)?;
    line.push(b'\n');
    if fds.is_empty() {
        (&*stream).write_all(&line)
    } else {
        fds::send_with_fds(stream, &line, fds)
    }
}

fn receive<T: for<'de> Deserialize<'de>>(stream: &UnixStream) -> Result<(T, Vec<OwnedFd>), String> {
    let (line, fds) = fds::receive_line(stream, MESSAGE_LIMIT)
        .map_err(|error| format!("keeper message: {error}"))?;
    let message =
        serde_json::from_slice(&line).map_err(|error| format!("keeper message: {error}"))?;
    Ok((message, fds))
}

fn request(
    stream: &UnixStream,
    operation: OperationV1,
    checkpoint: Option<CheckpointV1>,
    images: Vec<RankImageV1>,
) -> Result<(ReplyV1, Vec<OwnedFd>), String> {
    let request = RequestV1 {
        schema: SCHEMA_V1.to_owned(),
        op: operation,
        checkpoint,
        images,
    };
    send(stream, &request, &[]).map_err(|error| format!("keeper request: {error}"))?;
    let (reply, fds): (ReplyV1, _) = receive(stream)?;
    if reply.schema != SCHEMA_V1 {
        return Err(format!("the keeper answered with schema {}", reply.schema));
    }
    Ok((reply, fds))
}

/// One rank image a keeper lent.
#[derive(Debug)]
pub(crate) struct LentImageV1 {
    pub(crate) image: RankImageV1,
    pub(crate) fd: OwnedFd,
    pub(crate) mapped_bytes: u64,
    pub(crate) read_only: bool,
}

/// What a keeper answered a lease request: the images it lent, with the
/// connection that holds them, and the reason for each it did not.
#[derive(Debug)]
pub(crate) struct LeaseV1 {
    /// Open while the lease lasts; `None` when nothing was lent.
    pub(crate) connection: Option<UnixStream>,
    pub(crate) lent: Vec<LentImageV1>,
    pub(crate) refused: Vec<(RankImageV1, String)>,
}

/// Asks the keeper in `directory` to lend `images` of `checkpoint`.
pub(crate) fn lease_v1(
    directory: &Path,
    checkpoint: CheckpointV1,
    images: Vec<RankImageV1>,
) -> Result<LeaseV1, String> {
    let stream = connect(directory)?;
    let (reply, fds) = request(
        &stream,
        OperationV1::Lease,
        Some(checkpoint),
        images.clone(),
    )?;
    if !reply.ok {
        return Err(reply
            .reason
            .unwrap_or_else(|| "the keeper refused".to_owned()));
    }
    if reply.images.len() != images.len()
        || fds.len() != reply.images.iter().filter(|answer| answer.lent).count()
    {
        return Err("the keeper's answer does not match the request".to_owned());
    }
    let mut fds = fds.into_iter();
    let mut lease = LeaseV1 {
        connection: None,
        lent: Vec::new(),
        refused: Vec::new(),
    };
    for (image, answer) in images.into_iter().zip(reply.images) {
        if !answer.lent {
            let reason = answer.reason.unwrap_or_else(|| "not lent".to_owned());
            lease.refused.push((image, reason));
            continue;
        }
        let fd = fds.next().ok_or("the keeper sent too few descriptors")?;
        if answer.mapped_bytes < image.byte_length {
            return Err("the keeper lent a mapping shorter than its image".to_owned());
        }
        lease.lent.push(LentImageV1 {
            image,
            fd,
            mapped_bytes: answer.mapped_bytes,
            read_only: answer.read_only,
        });
    }
    if !lease.lent.is_empty() {
        // The lease lasts as long as the connection, without a deadline.
        stream
            .set_read_timeout(None)
            .map_err(|error| error.to_string())?;
        lease.connection = Some(stream);
    }
    Ok(lease)
}

/// Asks the keeper in `directory` to free its memory and exit, which it
/// does only while it holds no lease, and waits until it has.
pub(crate) fn release_v1(directory: &Path) -> Result<(), String> {
    exit_v1(directory, OperationV1::Release)
}

/// Asks the keeper in `directory` to stop whatever leases are open, and
/// waits until it has freed its memory. Open leases keep their mappings.
pub(crate) fn stop_v1(directory: &Path) -> Result<(), String> {
    exit_v1(directory, OperationV1::Stop)
}

fn exit_v1(directory: &Path, operation: OperationV1) -> Result<(), String> {
    let stream = connect(directory)?;
    let (reply, _) = request(&stream, operation, None, Vec::new())?;
    if !reply.ok {
        return Err(reply
            .reason
            .unwrap_or_else(|| "the keeper refused".to_owned()));
    }
    // The connection closes as the keeper exits.
    stream
        .set_read_timeout(Some(RELEASE_TIMEOUT))
        .map_err(|error| error.to_string())?;
    let mut rest = [0_u8; 1];
    match std::io::Read::read(&mut &stream, &mut rest) {
        Ok(0) => Ok(()),
        Ok(_) => Err("the keeper sent more after releasing".to_owned()),
        Err(error) => Err(format!("the released keeper did not exit: {error}")),
    }
}

/// The keeper's registry, as it answers a status request.
pub(crate) fn status_v1(directory: &Path) -> Result<serde_json::Value, String> {
    let stream = connect(directory)?;
    let (reply, _) = request(&stream, OperationV1::Status, None, Vec::new())?;
    reply
        .registry
        .ok_or_else(|| "the keeper sent no registry".to_owned())
}

/// One rank image a keeper holds: what it was planned as, a descriptor of
/// the device memory that holds it, and that memory, held without a mapping
/// or a context so the keeper is no compute process while it lends.
pub(crate) struct KeptImageV1 {
    pub(crate) image: RankImageV1,
    pub(crate) fd: OwnedFd,
    pub(crate) mapped_bytes: u64,
    pub(crate) held: Option<crate::device::HeldDeviceMemoryV1>,
}

impl std::fmt::Debug for KeptImageV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KeptImageV1")
            .field("image", &self.image)
            .field("fd", &self.fd)
            .field("mapped_bytes", &self.mapped_bytes)
            .field("held", &self.held.is_some())
            .finish()
    }
}

/// Fresh stamps of the keeper's checkpoint files, or why they are not the
/// files it loaded any more.
pub(crate) type StampsV1 = Box<dyn Fn() -> Result<Vec<String>, String> + Send + Sync>;

/// What a keeper lends.
pub(crate) struct KeeperV1 {
    pub(crate) directory: PathBuf,
    pub(crate) checkpoint_identity: String,
    pub(crate) stamps: StampsV1,
    pub(crate) images: Vec<KeptImageV1>,
    /// Whether borrowers map the images read-only.
    pub(crate) read_only: bool,
    /// More registry fields, such as the package served.
    pub(crate) about: serde_json::Map<String, serde_json::Value>,
}

/// Why a keeper stopped serving.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KeeperExitV1 {
    /// A server asked it to free its memory while it held no lease.
    Released,
    /// Asked to stop, over the socket or by SIGTERM or SIGINT.
    Stopped,
}

/// A keeper that stopped serving.
#[derive(Debug)]
pub(crate) struct KeeperExitedV1 {
    pub(crate) exit: KeeperExitV1,
    /// The connection of whoever asked it to stop, which waits for it to
    /// close: close it once the memory is freed. A signal leaves none.
    pub(crate) waiting: Option<UnixStream>,
}

#[derive(Debug, Default)]
struct KeeperStateV1 {
    leases: usize,
    last_lease_end_unix: u64,
    /// Why the keeper no longer lends, once its checkpoint changed.
    stale: Option<String>,
}

struct SharedV1 {
    keeper: KeeperV1,
    state: Mutex<KeeperStateV1>,
    started_unix: u64,
    exit: Mutex<Option<(KeeperExitV1, UnixStream)>>,
    exiting: AtomicBool,
}

impl SharedV1 {
    fn state(&self) -> std::sync::MutexGuard<'_, KeeperStateV1> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn registry(&self, state: &KeeperStateV1) -> serde_json::Value {
        let keeper = &self.keeper;
        let mut devices: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
        for kept in &keeper.images {
            *devices.entry(kept.image.device.as_str()).or_default() += kept.mapped_bytes;
        }
        let mut registry = serde_json::json!({
            "schema": SCHEMA_V1,
            "pid": std::process::id(),
            "state": if self.exiting.load(Ordering::Acquire) { "exiting" } else if state.stale.is_some() { "stale" } else { "ready" },
            "stale_reason": state.stale,
            "socket": keeper.directory.join(SOCKET_NAME_V1),
            "checkpoint_identity": keeper.checkpoint_identity,
            "read_only": keeper.read_only,
            "devices": devices.iter().map(|(device, bytes)| serde_json::json!({"device": device, "bytes": bytes})).collect::<Vec<_>>(),
            "images": keeper.images.iter().map(|kept| serde_json::json!({
                "image_role": kept.image.image_role, "rank": kept.image.rank,
                "identity": kept.image.identity, "device": kept.image.device,
                "byte_length": kept.image.byte_length, "mapped_bytes": kept.mapped_bytes,
            })).collect::<Vec<_>>(),
            "leases": state.leases,
            "started_unix": self.started_unix,
            "last_lease_end_unix": state.last_lease_end_unix,
        });
        if let Some(fields) = registry.as_object_mut() {
            for (key, value) in &keeper.about {
                fields.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        registry
    }

    /// Replaces the registry file in one rename.
    fn write_registry(&self, state: &KeeperStateV1) -> Result<(), String> {
        let directory = &self.keeper.directory;
        let temporary = directory.join(format!(".{REGISTRY_NAME_V1}.{}", std::process::id()));
        let _ = std::fs::remove_file(&temporary);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&temporary)
            .map_err(|error| format!("{}: {error}", temporary.display()))?;
        let mut bytes =
            serde_json::to_vec_pretty(&self.registry(state)).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .and_then(|()| std::fs::rename(&temporary, directory.join(REGISTRY_NAME_V1)))
            .map_err(|error| format!("keeper registry: {error}"))
    }

    fn answer(&self, request: &RequestV1) -> (Vec<AnswerV1>, Vec<BorrowedFd<'_>>) {
        let refuse_all = |reason: String| {
            let answers = request
                .images
                .iter()
                .map(|_| AnswerV1 {
                    lent: false,
                    mapped_bytes: 0,
                    read_only: false,
                    reason: Some(reason.clone()),
                })
                .collect();
            (answers, Vec::new())
        };
        let keeper = &self.keeper;
        if let Some(reason) = &self.state().stale {
            return refuse_all(format!("the keeper's checkpoint changed: {reason}"));
        }
        let Some(checkpoint) = &request.checkpoint else {
            return refuse_all("the request names no checkpoint".to_owned());
        };
        if checkpoint.identity != keeper.checkpoint_identity {
            return refuse_all("the checkpoint identity differs".to_owned());
        }
        match (keeper.stamps)() {
            Err(reason) => {
                // Its memory no longer holds what its checkpoint says.
                let mut state = self.state();
                state.stale = Some(reason.clone());
                let _ = self.write_registry(&state);
                return refuse_all(format!("the keeper's checkpoint changed: {reason}"));
            }
            Ok(stamps) if stamps != checkpoint.stamps => {
                return refuse_all("the checkpoint files differ".to_owned());
            }
            Ok(_) => {}
        }
        let mut fds = Vec::new();
        let answers = request
            .images
            .iter()
            .map(|asked| {
                let kept = keeper.images.iter().find(|kept| {
                    kept.image.image_role == asked.image_role && kept.image.rank == asked.rank
                });
                let reason = match kept {
                    None => Some("the keeper does not hold this image"),
                    Some(kept) if kept.image.identity != asked.identity => {
                        Some("the image identity differs")
                    }
                    Some(kept) if kept.image.device != asked.device => Some("the device differs"),
                    Some(kept) if kept.image.byte_length != asked.byte_length => {
                        Some("the image length differs")
                    }
                    Some(_) => None,
                };
                match (kept, reason) {
                    (Some(kept), None) => {
                        fds.push(kept.fd.as_fd());
                        AnswerV1 {
                            lent: true,
                            mapped_bytes: kept.mapped_bytes,
                            read_only: keeper.read_only,
                            reason: None,
                        }
                    }
                    (_, reason) => AnswerV1 {
                        lent: false,
                        mapped_bytes: 0,
                        read_only: false,
                        reason: reason.map(str::to_owned),
                    },
                }
            })
            .collect();
        (answers, fds)
    }

    /// Serves one connection: one request, then for a lease, the lease
    /// until the borrower closes it.
    fn serve(&self, stream: UnixStream) {
        let Ok(peer) = fds::peer(&stream) else {
            return;
        };
        if peer.uid != fds::effective_uid() {
            return;
        }
        let _ = stream.set_read_timeout(Some(ANSWER_TIMEOUT));
        let _ = stream.set_write_timeout(Some(ANSWER_TIMEOUT));
        let request: RequestV1 = match receive(&stream) {
            Ok((request, _)) => request,
            Err(reason) => {
                let _ = send(&stream, &ReplyV1::new(false, Some(reason)), &[]);
                return;
            }
        };
        if request.schema != SCHEMA_V1 {
            let reason = format!("the keeper speaks {SCHEMA_V1}");
            let _ = send(&stream, &ReplyV1::new(false, Some(reason)), &[]);
            return;
        }
        match request.op {
            OperationV1::Status => {
                let mut reply = ReplyV1::new(true, None);
                reply.registry = Some(self.registry(&self.state()));
                let _ = send(&stream, &reply, &[]);
            }
            OperationV1::Lease => self.lend(&stream, peer.pid, &request),
            OperationV1::Release | OperationV1::Stop => {
                let state = self.state();
                if request.op == OperationV1::Release && state.leases > 0 {
                    let reason = format!("{} leases are open", state.leases);
                    let _ = send(&stream, &ReplyV1::new(false, Some(reason)), &[]);
                    return;
                }
                if self.exiting.swap(true, Ordering::AcqRel) {
                    let reason = "the keeper is exiting".to_owned();
                    let _ = send(&stream, &ReplyV1::new(false, Some(reason)), &[]);
                    return;
                }
                let _ = self.write_registry(&state);
                // Held until the reply is sent, so no lease starts meanwhile.
                let _ = send(&stream, &ReplyV1::new(true, None), &[]);
                drop(state);
                let exit = if request.op == OperationV1::Release {
                    KeeperExitV1::Released
                } else {
                    KeeperExitV1::Stopped
                };
                // The asker waits for this connection to close, which the
                // caller of `serve_v1` does once the memory is freed.
                *self
                    .exit
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((exit, stream));
            }
        }
    }

    fn lend(&self, stream: &UnixStream, pid: i32, request: &RequestV1) {
        if self.exiting.load(Ordering::Acquire) {
            let _ = send(
                stream,
                &ReplyV1::new(false, Some("the keeper is exiting".to_owned())),
                &[],
            );
            return;
        }
        let (answers, fds) = self.answer(request);
        let lent = answers.iter().filter(|answer| answer.lent).count();
        let mut reply = ReplyV1::new(true, None);
        reply.images = answers;
        let mut state = self.state();
        if self.exiting.load(Ordering::Acquire) {
            drop(state);
            let _ = send(
                stream,
                &ReplyV1::new(false, Some("the keeper is exiting".to_owned())),
                &[],
            );
            return;
        }
        if send(stream, &reply, &fds).is_err() || lent == 0 {
            return;
        }
        state.leases += 1;
        let _ = self.write_registry(&state);
        drop(state);
        eprintln!("weight keeper: lent {lent} images to process {pid}");
        // The lease lasts until the borrower closes the connection or exits.
        let _ = stream.set_read_timeout(None);
        let mut rest = [0_u8; 64];
        while matches!(std::io::Read::read(&mut &*stream, &mut rest), Ok(read) if read > 0) {}
        let mut state = self.state();
        state.leases -= 1;
        state.last_lease_end_unix = unix_seconds();
        let _ = self.write_registry(&state);
        eprintln!("weight keeper: process {pid} returned its lease");
    }
}

/// Creates the keeper directory if it is missing, closed to everyone else,
/// and refuses one that is not private.
pub(crate) fn prepare_directory_v1(directory: &Path) -> Result<(), String> {
    match std::fs::DirBuilder::new().mode(0o700).create(directory) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(format!("keeper directory {}: {error}", directory.display())),
    }
    check_private_directory(directory)
}

/// Lends `keeper`'s images until a server releases them or it is asked to
/// stop; the caller exits then, freeing the memory. Refuses to start while
/// another live keeper serves the directory.
pub(crate) fn serve_v1(keeper: KeeperV1) -> Result<KeeperExitedV1, String> {
    let directory = keeper.directory.clone();
    prepare_directory_v1(&directory)?;
    let socket = directory.join(SOCKET_NAME_V1);
    if std::fs::symlink_metadata(&socket).is_ok() {
        check_own_entry(&socket, true)?;
        if UnixStream::connect(&socket).is_ok() {
            return Err(format!("another keeper serves {}", directory.display()));
        }
        std::fs::remove_file(&socket).map_err(|error| format!("{}: {error}", socket.display()))?;
    }
    let listener = UnixListener::bind(&socket)
        .map_err(|error| format!("bind {}: {error}", socket.display()))?;
    std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))
        .map_err(|error| format!("{}: {error}", socket.display()))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| error.to_string())?;
    let shared = Arc::new(SharedV1 {
        keeper,
        state: Mutex::new(KeeperStateV1::default()),
        started_unix: unix_seconds(),
        exit: Mutex::new(None),
        exiting: AtomicBool::new(false),
    });
    shared.write_registry(&shared.state())?;
    eprintln!(
        "weight keeper: ready, {} images, {}",
        shared.keeper.images.len(),
        socket.display()
    );
    let (exit, waiting) = loop {
        if let Some((exit, waiting)) = shared
            .exit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            break (exit, Some(waiting));
        }
        if crate::signals::stopping() {
            shared.exiting.store(true, Ordering::Release);
            break (KeeperExitV1::Stopped, None);
        }
        match listener.accept() {
            Ok((stream, _)) => {
                if stream.set_nonblocking(false).is_err() {
                    continue;
                }
                let shared = Arc::clone(&shared);
                let _ = std::thread::Builder::new()
                    .name("chopinfer-weight-keeper".to_owned())
                    .spawn(move || shared.serve(stream));
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(ACCEPT_POLL);
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(format!("keeper socket: {error}")),
        }
    };
    let _ = std::fs::remove_file(&socket);
    let _ = std::fs::remove_file(directory.join(REGISTRY_NAME_V1));
    // The images, and the memory they hold, go before the caller answers
    // whoever waits for that: connection threads still finishing hold them,
    // so wait for those, but not for open leases, whose borrowers map the
    // memory until they return it anyway.
    let deadline = Instant::now() + ANSWER_TIMEOUT + ACCEPT_POLL;
    while Arc::strong_count(&shared) > 1 + shared.state().leases && Instant::now() < deadline {
        std::thread::sleep(ACCEPT_POLL);
    }
    drop(shared);
    Ok(KeeperExitedV1 { exit, waiting })
}

/// What one server start took from a keeper, for its `startup weight_keeper`
/// line.
pub(crate) fn report_v1(
    policy: KeeperPolicyV1,
    started: Instant,
    lent: &[(RankImageV1, u64, bool)],
    refused: &[(RankImageV1, String)],
    error: Option<&str>,
    released: Option<&Result<(), String>>,
) -> String {
    serde_json::json!({
        "policy": policy.name(),
        "seconds": started.elapsed().as_secs_f64(),
        "lent": lent.iter().map(|(image, mapped, read_only)| serde_json::json!({
            "image_role": image.image_role, "rank": image.rank, "identity": image.identity,
            "byte_length": image.byte_length, "mapped_bytes": mapped, "read_only": read_only,
        })).collect::<Vec<_>>(),
        "refused": refused.iter().map(|(image, reason)| serde_json::json!({
            "image_role": image.image_role, "rank": image.rank, "reason": reason,
        })).collect::<Vec<_>>(),
        "error": error,
        "released": released.map(|result| result.as_ref().err().map_or("released", String::as_str)),
    })
    .to_string()
}

#[cfg(all(test, target_os = "linux"))]
mod tests;
