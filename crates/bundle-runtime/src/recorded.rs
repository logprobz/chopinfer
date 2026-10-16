//! Immutable device programs with uploads staged before any compute.
//!
//! An upload is staged (its bytes copied into the program, uploaded to a
//! staging buffer in the upload windows, then copied to its destination in
//! command order) or direct (borrowed, and written straight to its
//! destination in the upload windows). Direct uploads take no staging
//! memory, so they hold an encoder's payload; they precede every command.

use crate::{dispatch::Resolved, submission::RankSubmission};
use runtime_bundle_abi::{BundleStreamHandleV1, DeviceBufferSpanV1 as Span, HostServicesV1};
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    hash::{Hash, Hasher},
};

/// One serving pinned-ring window. Round-robin uploads overlap rank transfers;
/// reuse can wait for an earlier transfer, never for this record's compute.
pub const UPLOAD_CHUNK: usize = 4 * 1024 * 1024;

/// Explicit canonical-view to rank-view bindings. Missing or undersized views
/// fail before submission. Offsets and lengths inside each view are preserved.
#[derive(Default)]
pub struct Relocations {
    spans: BTreeMap<(u64, u64), (Span, Span)>,
    symbols: HashMap<
        runtime_bundle_abi::BundleCudaSymbolHandleV1,
        runtime_bundle_abi::BundleCudaSymbolHandleV1,
    >,
}
impl Relocations {
    pub fn span(&mut self, from: Span, to: Span) -> Result<()> {
        if from.length != to.length {
            return Err("rank view lengths differ".into());
        }
        if let Some(previous) = self
            .spans
            .insert((from.buffer.raw, from.offset), (from, to))
            && previous != (from, to)
        {
            return Err("ambiguous rank view binding".into());
        }
        Ok(())
    }
    pub fn symbol(
        &mut self,
        from: runtime_bundle_abi::BundleCudaSymbolHandleV1,
        to: runtime_bundle_abi::BundleCudaSymbolHandleV1,
    ) {
        self.symbols.insert(from, to);
    }
    fn view(&self, view: Span) -> Result<Span> {
        let (_, &(from, to)) = self
            .spans
            .range(..=(view.buffer.raw, view.offset))
            .next_back()
            .ok_or("missing rank view")?;
        let offset = view
            .offset
            .checked_sub(from.offset)
            .ok_or("rank view offset")?;
        if from.buffer != view.buffer
            || offset
                .checked_add(view.length)
                .is_none_or(|end| end > from.length)
        {
            return Err("rank view exceeds binding".into());
        }
        Ok(Span::new(to.buffer, to.offset + offset, view.length))
    }
}

/// A fresh process namespace preserves old captures across server restarts.
pub fn capture_root(root: &std::path::Path) -> std::path::PathBuf {
    static RUN: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    root.join(RUN.get_or_init(|| {
        format!(
            "run-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        )
    }))
}

/// Filesystem failures in optional diagnostics must not kill a serving worker.
/// Device readback errors still propagate through the normal poison path.
pub fn write_capture(services: &HostServicesV1, path: &std::path::Path, bytes: &[u8]) -> bool {
    let result = write_new_capture(path, bytes);
    if let Err(error) = result {
        let _ = services
            .log(format!("encoder capture disabled: {}: {error}", path.display()).as_bytes());
        false
    } else {
        true
    }
}

fn write_new_capture(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)
}

type Result<T> = std::result::Result<T, String>;

/// A launch's live rows and the rows it was prepared for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rows {
    pub live: u32,
    pub capacity: u32,
}

/// What test builds keep of a launch for `observe`: its symbol and rows.
#[derive(Clone)]
struct Label {
    #[cfg(any(test, feature = "test-support"))]
    symbol: String,
    #[cfg(any(test, feature = "test-support"))]
    rows: Rows,
}

/// Test builds: an observer of the launches this thread's bound programs
/// submit, so a bundle can record them beside its own.
#[cfg(any(test, feature = "test-support"))]
pub mod observe {
    use super::Rows;
    use runtime_bundle_abi::BundleStreamHandleV1;
    use std::cell::RefCell;

    type Observer = Box<dyn FnMut(BundleStreamHandleV1, &str, Rows)>;

    thread_local! {
        static OBSERVER: RefCell<Option<Observer>> = const { RefCell::new(None) };
    }

    /// Calls `observer` with each launch this thread's bound programs
    /// submit, before the host receives it, until another call replaces it.
    pub fn launches(observer: Option<Observer>) {
        OBSERVER.with(|slot| *slot.borrow_mut() = observer);
    }

    pub(super) fn launched(stream: BundleStreamHandleV1, symbol: &str, rows: Rows) {
        OBSERVER.with(|slot| {
            if let Some(observer) = slot.borrow_mut().as_mut() {
                observer(stream, symbol, rows);
            }
        });
    }
}

#[derive(Clone)]
enum Command {
    Copy {
        source: Span,
        destination: Span,
    },
    Launch(Resolved, Label),
    Capture {
        source: Span,
        path: std::path::PathBuf,
    },
    /// Reads `source` back into the bound program's taps under `name`.
    Tap {
        source: Span,
        name: String,
    },
}

/// A program owns its staged upload bytes, and borrows its direct ones, until
/// every rank has finished. Duplicate tables share staging storage; repeated
/// writes keep their original order.
pub struct Program<'a> {
    staging: Option<Span>,
    bytes: Vec<u8>,
    uploads: HashMap<(usize, u64), Vec<u64>>,
    direct: Vec<(Span, &'a [u8])>,
    commands: Vec<Command>,
    capture_id: u64,
}
impl<'a> Program<'a> {
    pub fn new(staging: Span) -> Self {
        Self::staged(Some(staging))
    }
    /// A program without staging: every upload is direct.
    pub fn direct() -> Self {
        Self::staged(None)
    }
    fn staged(staging: Option<Span>) -> Self {
        static NEXT_CAPTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self {
            capture_id: NEXT_CAPTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            staging,
            // Reserve once so adding tables cannot recopy a large payload.
            bytes: Vec::with_capacity(staging.map_or(0, |staging| staging.length as usize)),
            uploads: HashMap::new(),
            direct: Vec::new(),
            commands: Vec::new(),
        }
    }
    /// Writes `bytes` to `destination` in the upload windows, before every
    /// command, in windows of at most `UPLOAD_CHUNK` bytes.
    pub fn upload_direct(&mut self, destination: Span, bytes: &'a [u8]) -> Result<()> {
        if !self.commands.is_empty() {
            return Err("direct uploads precede every command".into());
        }
        if destination.length != bytes.len() as u64 {
            return Err("direct upload length differs from its destination".into());
        }
        for (index, chunk) in bytes.chunks(UPLOAD_CHUNK).enumerate() {
            let offset = (index * UPLOAD_CHUNK) as u64;
            self.direct.push((
                Span::new(
                    destination.buffer,
                    destination.offset + offset,
                    chunk.len() as u64,
                ),
                chunk,
            ));
        }
        Ok(())
    }
    pub fn upload(&mut self, destination: Span, bytes: &[u8]) -> Result<()> {
        let staging = self.staging.ok_or("this recorded program has no staging")?;
        // Hash keys never own payload copies. Verify equality on hash collisions.
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hash);
        let key = (bytes.len(), hash.finish());
        let found = self.uploads.get(&key).and_then(|offsets| {
            offsets.iter().copied().find(|&offset| {
                self.bytes[offset as usize..offset as usize + bytes.len()] == *bytes
            })
        });
        let offset = if let Some(offset) = found {
            offset
        } else {
            let offset = self.bytes.len().next_multiple_of(256) as u64;
            if offset + bytes.len() as u64 > staging.length {
                return Err("recorded uploads exceed staging capacity".into());
            }
            self.bytes.resize(offset as usize, 0);
            self.bytes.extend_from_slice(bytes);
            self.uploads.entry(key).or_default().push(offset);
            offset
        };
        self.copy(
            Span::new(staging.buffer, staging.offset + offset, bytes.len() as u64),
            destination,
        );
        Ok(())
    }
    pub fn copy(&mut self, source: Span, destination: Span) {
        self.commands.push(Command::Copy {
            source,
            destination,
        });
    }
    /// Launches `launch`, `symbol` with `rows`.
    pub fn launch(&mut self, launch: Resolved, symbol: &str, rows: Rows) {
        #[cfg(not(any(test, feature = "test-support")))]
        let _ = (symbol, rows);
        let label = Label {
            #[cfg(any(test, feature = "test-support"))]
            symbol: symbol.to_owned(),
            #[cfg(any(test, feature = "test-support"))]
            rows,
        };
        self.commands.push(Command::Launch(launch, label));
    }
    pub fn capture(&mut self, source: Span, path: std::path::PathBuf) {
        let path = capture_root(path.parent().unwrap_or_else(|| std::path::Path::new(".")))
            .join(format!(
                "record-{}/stage-{}",
                self.capture_id,
                self.commands.len()
            ))
            .join(path.file_name().expect("capture filename"));
        self.commands.push(Command::Capture { source, path });
    }
    /// Reads `source` back, in command order, into the first bound rank's
    /// taps under `name` (`Bound::take_taps`).
    pub fn tap(&mut self, source: Span, name: String) {
        self.commands.push(Command::Tap { source, name });
    }
    pub fn staged_bytes(&self) -> usize {
        self.bytes.len()
    }
    pub fn bind<'b>(
        &'b self,
        services: &'b HostServicesV1,
        stream: BundleStreamHandleV1,
    ) -> Bound<'b> {
        Bound {
            program: self,
            direct: Cow::Borrowed(&self.direct),
            commands: Cow::Borrowed(&self.commands),
            staging: self.staging,
            capture_failed: Default::default(),
            taps: Default::default(),
            services,
            stream,
        }
    }
    /// Reuse the same immutable bytes and command topology on another rank.
    /// Only the first rank captures layers, matching the serial runner.
    pub fn bind_relocated<'b>(
        &'b self,
        services: &'b HostServicesV1,
        stream: BundleStreamHandleV1,
        bindings: &Relocations,
        capture: bool,
    ) -> Result<Bound<'b>> {
        let direct = self
            .direct
            .iter()
            .map(|&(destination, bytes)| Ok((bindings.view(destination)?, bytes)))
            .collect::<Result<Vec<_>>>()?;
        let commands = self
            .commands
            .iter()
            .filter_map(|command| {
                Some(match command {
                    Command::Copy {
                        source,
                        destination,
                    } => bindings.view(*source).and_then(|source| {
                        Ok(Command::Copy {
                            source,
                            destination: bindings.view(*destination)?,
                        })
                    }),
                    Command::Launch(launch, label) => launch
                        .relocate(
                            |view| bindings.view(view),
                            |symbol| {
                                bindings
                                    .symbols
                                    .get(&symbol)
                                    .copied()
                                    .ok_or_else(|| "missing rank symbol".to_owned())
                            },
                        )
                        .map(|launch| Command::Launch(launch, label.clone())),
                    Command::Capture { source, path } if capture => {
                        bindings.view(*source).map(|source| Command::Capture {
                            source,
                            path: path.clone(),
                        })
                    }
                    Command::Capture { .. } => return None,
                    Command::Tap { source, name } if capture => {
                        bindings.view(*source).map(|source| Command::Tap {
                            source,
                            name: name.clone(),
                        })
                    }
                    Command::Tap { .. } => return None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Bound {
            program: self,
            direct: Cow::Owned(direct),
            commands: Cow::Owned(commands),
            staging: self
                .staging
                .map(|staging| bindings.view(staging))
                .transpose()?,
            capture_failed: Default::default(),
            taps: Default::default(),
            services,
            stream,
        })
    }
}

/// Upload windows precede commands on every rank: the direct windows, then
/// the staged ones. All ranks complete each window before advancing, so
/// transfers overlap without waiting on compute.
pub struct Bound<'a> {
    program: &'a Program<'a>,
    direct: Cow<'a, [(Span, &'a [u8])]>,
    commands: Cow<'a, [Command]>,
    staging: Option<Span>,
    capture_failed: std::cell::Cell<bool>,
    taps: std::cell::RefCell<Vec<(String, Vec<u8>)>>,
    services: &'a HostServicesV1,
    stream: BundleStreamHandleV1,
}
impl Bound<'_> {
    /// The rows the program's taps read back, in command order.
    pub fn take_taps(&self) -> Vec<(String, Vec<u8>)> {
        std::mem::take(&mut self.taps.borrow_mut())
    }
}

impl RankSubmission for Bound<'_> {
    type Error = String;
    fn stages(&self) -> usize {
        self.direct.len() + self.commands.len() + self.program.bytes.len().div_ceil(UPLOAD_CHUNK)
    }
    fn submit(&self, stage: usize) -> Result<()> {
        if let Some(&(destination, bytes)) = self.direct.get(stage) {
            return self
                .services
                .copy_host_to_device(self.stream, destination, bytes)
                .map_err(|e| e.to_string());
        }
        let stage = stage - self.direct.len();
        let uploads = self.program.bytes.len().div_ceil(UPLOAD_CHUNK);
        if stage < uploads {
            let staging = self.staging.ok_or("staged bytes without staging")?;
            let offset = stage * UPLOAD_CHUNK;
            let end = (offset + UPLOAD_CHUNK).min(self.program.bytes.len());
            return self
                .services
                .copy_host_to_device(
                    self.stream,
                    Span::new(
                        staging.buffer,
                        staging.offset + offset as u64,
                        (end - offset) as u64,
                    ),
                    &self.program.bytes[offset..end],
                )
                .map_err(|e| e.to_string());
        }
        match &self.commands[stage - uploads] {
            Command::Copy {
                source,
                destination,
            } => self
                .services
                .copy_device_to_device(self.stream, *destination, *source)
                .map_err(|e| e.to_string()),
            Command::Launch(launch, label) => {
                #[cfg(any(test, feature = "test-support"))]
                observe::launched(self.stream, &label.symbol, label.rows);
                #[cfg(not(any(test, feature = "test-support")))]
                let _ = label;
                launch
                    .submit(self.services, self.stream)
                    .map_err(|e| format!("{e:?}"))
            }
            Command::Tap { source, name } => {
                let mut bytes = vec![0; source.length as usize];
                self.services
                    .copy_device_to_host(self.stream, &mut bytes, *source)
                    .map_err(|e| e.to_string())?;
                self.taps.borrow_mut().push((name.clone(), bytes));
                Ok(())
            }
            Command::Capture { source, path } => {
                if self.capture_failed.get() {
                    return Ok(());
                }
                let mut bytes = vec![0; source.length as usize];
                self.services
                    .copy_device_to_host(self.stream, &mut bytes, *source)
                    .map_err(|e| e.to_string())?;
                if !write_capture(self.services, path, &bytes) {
                    self.capture_failed.set(true);
                }
                Ok(())
            }
        }
    }
    fn drain(&self) -> Result<()> {
        self.services
            .synchronize_stream(self.stream)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1 as Handle;
    fn span(offset: u64, length: u64) -> Span {
        Span::new(Handle::from_parts(1, 1), offset, length)
    }

    #[test]
    fn dedup_preserves_intervening_writes_and_alignment() {
        let mut p = Program::new(span(4096, 1024));
        for bytes in [&b"abc"[..], &b"changed"[..], &b"abc"[..]] {
            p.upload(span(0, bytes.len() as u64), bytes).unwrap();
        }
        assert_eq!(p.staged_bytes(), 263);
        assert_eq!(p.uploads.len(), 2);
        let sources = p
            .commands
            .iter()
            .map(|c| match c {
                Command::Copy {
                    source,
                    destination,
                } => {
                    assert_eq!(destination.offset, 0);
                    source.offset
                }
                _ => panic!("copy"),
            })
            .collect::<Vec<_>>();
        assert_eq!(sources, [4096, 4352, 4096]);
    }

    #[test]
    fn direct_uploads_window_their_bytes_and_precede_every_command() {
        let payload = vec![7; UPLOAD_CHUNK + 5];
        let mut p = Program::direct();
        p.upload_direct(span(1000, payload.len() as u64), &payload)
            .unwrap();
        assert_eq!(
            p.direct
                .iter()
                .map(|(span, bytes)| (span.offset, span.length, bytes.len()))
                .collect::<Vec<_>>(),
            [
                (1000, UPLOAD_CHUNK as u64, UPLOAD_CHUNK),
                (1000 + UPLOAD_CHUNK as u64, 5, 5)
            ]
        );
        assert!(p.upload_direct(span(0, 4), b"abc").is_err());
        // Without staging, staged uploads are refused.
        assert!(p.upload(span(0, 3), b"abc").is_err());
        p.copy(span(0, 4), span(8, 4));
        assert!(p.upload_direct(span(0, 3), b"abc").is_err());
        assert_eq!(p.direct.len(), 2);
    }

    #[test]
    fn capacity_refusal_leaves_existing_commands_intact() {
        let mut p = Program::new(span(0, 256));
        p.upload(span(512, 4), b"1234").unwrap();
        assert!(p.upload(span(512, 1), b"x").is_err());
        assert_eq!(p.commands.len(), 1);
        assert_eq!(p.staged_bytes(), 4);
        p.upload(span(512, 4), b"1234").unwrap();
        assert_eq!(p.commands.len(), 2);
    }

    #[test]
    fn hash_collisions_require_byte_equality() {
        let mut p = Program::new(span(0, 1024));
        p.upload(span(2048, 3), b"abc").unwrap();
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        b"xyz"[..].hash(&mut hash);
        p.uploads.insert((3, hash.finish()), vec![0]);
        p.upload(span(2048, 3), b"xyz").unwrap();
        assert_eq!(&p.bytes[256..], b"xyz");
    }

    #[test]
    fn relocations_preserve_subviews_and_refuse_unbound_or_short_views() {
        let mut r = Relocations::default();
        r.span(span(100, 100), span(1000, 100)).unwrap();
        assert_eq!(r.view(span(120, 30)).unwrap(), span(1020, 30));
        assert!(r.view(span(90, 10)).is_err());
        assert!(r.view(span(190, 11)).is_err());
        assert!(r.span(span(200, 100), span(2000, 99)).is_err());
        assert!(r.span(span(100, 100), span(3000, 100)).is_err());
    }

    #[test]
    fn capture_identity_is_unique_and_existing_evidence_is_preserved() {
        let root = std::env::temp_dir().join(format!("recorded-test-{}", std::process::id()));
        let mut a = Program::new(span(0, 1024));
        let mut b = Program::new(span(0, 1024));
        a.capture(span(0, 2), root.join("layer.bin"));
        a.capture(span(0, 2), root.join("layer.bin"));
        b.capture(span(0, 2), root.join("layer.bin"));
        let paths = a
            .commands
            .iter()
            .chain(&b.commands)
            .map(|c| match c {
                Command::Capture { path, .. } => path,
                _ => panic!("capture"),
            })
            .collect::<Vec<_>>();
        assert_ne!(paths[0], paths[1]);
        assert_ne!(paths[0], paths[2]);
        assert!(
            paths[0]
                .display()
                .to_string()
                .contains(&format!("run-{}-", std::process::id()))
        );
        write_new_capture(paths[0], b"ok").unwrap();
        assert_eq!(
            write_new_capture(paths[0], b"bad").unwrap_err().kind(),
            std::io::ErrorKind::AlreadyExists
        );
        assert_eq!(std::fs::read(paths[0]).unwrap(), b"ok");
        assert!(write_new_capture(&paths[0].join("child"), b"bad").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
