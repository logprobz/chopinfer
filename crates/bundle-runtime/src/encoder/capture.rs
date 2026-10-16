//! Opt-in qualification captures.
//!
//! `CAPTURE_ENV`: each step that carries media owns an immutable directory
//! `step-<n>` holding rank 0's output rows of every encode (`encode-<i>.bin`,
//! read from the arena pages the record wrote), for an encoder whose stage
//! joins runs the rows it joined (`joined-<i>.bin`, such as a tokenizer's
//! codes),
//! the arena rows of every inject (`inject-<i>.bin`, with its request and
//! prompt row, so injected rows keep their positions across chunk
//! boundaries), and a manifest written last.
//!
//! `ARENA_ENV`: every rank's arena rows of every encoded record, as
//! `request-<n>-record-<i>-rank-<r>.bin` under a fresh run directory.
//!
//! Captures are diagnostics: a file system failure disables them and never
//! fails the step (`Failed::File`); a failed readback is a device fault.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use runtime_bundle_abi::{BundleStreamHandleV1, DeviceBufferSpanV1 as Span, HostServicesV1};
use runtime_bundle_sdk::{BatchMediaEncodeV3, BatchRequestV1};
use serde_json::{Value, json};

use crate::media::MediaArena;
use crate::recorded::{capture_root, write_capture};

/// Rank 0 writes each step's encoded and injected rows here when set.
pub const CAPTURE_ENV: &str = "CHOPINFER_ENCODER_CAPTURE_DIR";
/// Every rank writes each encoded record's arena rows here when set.
pub const ARENA_ENV: &str = "CHOPINFER_ENCODER_ARENA_DIR";

/// Why a capture failed.
#[derive(Debug)]
pub(super) enum Failed {
    /// A device readback failed and may have queued work.
    Device(String),
    /// The file system refused a directory or a file.
    File(String),
}

pub(super) struct Capture {
    dir: PathBuf,
    row_bytes: u64,
    encodes: Vec<Value>,
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| e.to_string())
}

impl Capture {
    pub fn begin(root: &Path, row_bytes: u64) -> Result<Self, Failed> {
        let file = |e: std::io::Error| Failed::File(format!("{}: {e}", root.display()));
        fs::create_dir_all(root).map_err(file)?;
        for step in 0..u64::MAX {
            let dir = root.join(format!("step-{step:020}"));
            match fs::create_dir(&dir) {
                Ok(()) => {
                    return Ok(Self {
                        dir,
                        row_bytes,
                        encodes: Vec::new(),
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(file(e)),
            }
        }
        Err(Failed::File("encoder capture step IDs exhausted".into()))
    }

    fn readback(
        &self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        span: Span,
        name: &str,
    ) -> Result<(), Failed> {
        let device = |e: runtime_bundle_abi::BundleAbiError| Failed::Device(e.to_string());
        let mut data = vec![0; span.length as usize];
        services
            .copy_device_to_host(stream, &mut data, span)
            .map_err(device)?;
        services.synchronize_stream(stream).map_err(device)?;
        write_new(&self.dir.join(name), &data).map_err(Failed::File)
    }

    /// Writes `record`'s output rows and, when given, its `joined` rows
    /// (their span and the bytes of one row).
    pub fn encode(
        &mut self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        (arena, layout): (Span, MediaArena),
        record: &BatchMediaEncodeV3,
        joined: Option<(Span, u64)>,
    ) -> Result<(), Failed> {
        let device = |e: runtime_bundle_abi::BundleAbiError| Failed::Device(e.to_string());
        let row_bytes = self.row_bytes;
        let mut data = vec![0; record.output_rows as usize * row_bytes as usize];
        for run in layout
            .runs(&record.pages, record.output_rows, 0, record.output_rows)
            .map_err(|e| Failed::Device(e.to_string()))?
        {
            let start = (u64::from(run.source_row) * row_bytes) as usize;
            let length = u64::from(run.rows) * row_bytes;
            services
                .copy_device_to_host(
                    stream,
                    &mut data[start..start + length as usize],
                    Span::new(
                        arena.buffer,
                        arena.offset + run.arena_row * row_bytes,
                        length,
                    ),
                )
                .map_err(device)?;
        }
        services.synchronize_stream(stream).map_err(device)?;
        let index = self.encodes.len();
        let file = format!("encode-{index}.bin");
        write_new(&self.dir.join(&file), &data).map_err(Failed::File)?;
        let mut entry = json!({"file": file, "rows": record.output_rows,
            "geometry": record.geometry, "pages": record.pages});
        if let Some((span, row_bytes)) = joined {
            let file = format!("joined-{index}.bin");
            self.readback(services, stream, span, &file)?;
            entry["joined"] =
                json!({"file": file, "rows": span.length / row_bytes, "row_bytes": row_bytes});
        }
        self.encodes.push(entry);
        Ok(())
    }

    /// Reads every inject's arena rows and writes the manifest of `schema`.
    pub fn finish(
        self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        arena: (Span, MediaArena),
        batch: &BatchRequestV1,
        schema: &str,
    ) -> Result<(), Failed> {
        let (span, layout) = arena;
        let row_bytes = self.row_bytes;
        let mut injects = Vec::new();
        for (index, inject) in batch.media().injects.iter().enumerate() {
            let file = format!("inject-{index}.bin");
            self.readback(
                services,
                stream,
                Span::new(
                    span.buffer,
                    span.offset + inject.arena_row * row_bytes,
                    u64::from(inject.rows) * row_bytes,
                ),
                &file,
            )?;
            let page_rows = layout.page_rows;
            let encoded = batch.media().encodes.iter().position(|encode| {
                encode.pages.iter().enumerate().any(|(page, &arena_page)| {
                    let first = arena_page * page_rows;
                    let count =
                        (u64::from(encode.output_rows) - page as u64 * page_rows).min(page_rows);
                    inject.arena_row >= first
                        && inject.arena_row + u64::from(inject.rows) <= first + count
                })
            });
            let item = inject.item_index as usize;
            injects.push(json!({"file": file, "rows": inject.rows,
                "request_id": batch.items()[item].request_id(),
                "prompt_row": batch.appends()[item].previous_tokens() + u64::from(inject.item_row),
                "arena_row": inject.arena_row, "encode_record": encoded,
                "origin": if encoded.is_some() { "encoded" } else { "resident" }}));
        }
        let manifest = json!({"schema": schema, "width": row_bytes / 2,
            "encodes": self.encodes, "injects": injects});
        let bytes =
            serde_json::to_vec_pretty(&manifest).map_err(|e| Failed::File(e.to_string()))?;
        write_new(&self.dir.join("manifest.json.partial"), &bytes).map_err(Failed::File)?;
        fs::rename(
            self.dir.join("manifest.json.partial"),
            self.dir.join("manifest.json"),
        )
        .map_err(|e| Failed::File(e.to_string()))
    }
}

/// Writes each rank's arena rows of every record of `records` under
/// `root` (`ARENA_ENV`); `ranks` holds each rank's stream and arena. A
/// file system failure stops the captures and is logged, not returned.
pub fn arena(
    services: &HostServicesV1,
    root: &Path,
    layout: MediaArena,
    row_bytes: u64,
    records: &[BatchMediaEncodeV3],
    ranks: &[(BundleStreamHandleV1, Span)],
) -> Result<(), String> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let request = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let root = capture_root(root);
    for (index, record) in records.iter().enumerate() {
        for (rank, &(stream, arena)) in ranks.iter().enumerate() {
            let mut bytes = vec![0; record.output_rows as usize * row_bytes as usize];
            for run in layout
                .runs(&record.pages, record.output_rows, 0, record.output_rows)
                .map_err(|e| format!("{e:?}"))?
            {
                let start = (u64::from(run.source_row) * row_bytes) as usize;
                let length = u64::from(run.rows) * row_bytes;
                services
                    .copy_device_to_host(
                        stream,
                        &mut bytes[start..start + length as usize],
                        Span::new(
                            arena.buffer,
                            arena.offset + run.arena_row * row_bytes,
                            length,
                        ),
                    )
                    .map_err(|e| e.to_string())?;
            }
            let name = format!("request-{request}-record-{index}-rank-{rank}.bin");
            if !write_capture(services, &root.join(name), &bytes) {
                return Ok(());
            }
        }
    }
    Ok(())
}
