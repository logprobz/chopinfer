//! MiMo weight images planned from the checkpoint.
//!
//! The bundle's weight plan (`model_mimo26_contract::weight_plan`) runs
//! through the host's direct-load executor (`runtime_host::direct`), so the
//! lab reads the bytes a served load uploads straight from the checkpoint,
//! on demand and in memory. Nothing is written to disk, so planning every
//! layer costs no more than planning one.

#![allow(dead_code, reason = "each lab test uses a subset of the helpers")]

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

use kernel_lab::LabResult;
use model_mimo26_contract::weight_plan::{self, DRAFTER_KIND_ROLE, Drafter, index_role};
use runtime_bundle_abi::weight_plan::NamedBytesV1;
use runtime_host::direct::{DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1};
use sha2::{Digest, Sha256};

/// One view of a planned rank index.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct PlannedView {
    pub name: String,
    /// Offset within the rank's arena.
    pub offset: u64,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub layout_id: String,
}

/// A planned rank index. It names the plan's identity (`direct_identity`)
/// where a rank image file's digest would be.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct PlannedIndex {
    pub schema: String,
    pub rank: usize,
    pub image_bytes: u64,
    pub direct_identity: String,
    pub views: Vec<PlannedView>,
}

impl PlannedIndex {
    pub fn view(&self, name: &str) -> LabResult<&PlannedView> {
        self.views
            .iter()
            .find(|view| view.name == name)
            .ok_or_else(|| format!("rank {} missing view {name}", self.rank))
    }
}

/// One read of planned bytes: where, how much, the view it read if named,
/// and the SHA-256 of the bytes it returned.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Read {
    rank: usize,
    offset: u64,
    bytes: u64,
    view: Option<String>,
    sha256: String,
}

/// One planned image of every rank, with each rank's index.
pub struct PlannedImages {
    pub image: PlannedImageV1,
    pub indexes: Vec<PlannedIndex>,
    /// Each rank's index JSON as the plan produced it.
    pub index_bytes: Vec<Vec<u8>>,
    /// Every read so far: the plan identities name the checkpoint's headers
    /// and the plan, not tensor bytes, so receipts state the content of the
    /// bytes a run actually used.
    reads: Mutex<Vec<Read>>,
}

impl PlannedImages {
    /// Plans image `role` of a `ranks`-rank load of `checkpoint`. `layers`
    /// limits a text image to those layers (with the embedding, final norm
    /// and head); `None` plans every layer. `drafter` picks the drafter
    /// image's kind; without it the plan picks DFlash when the checkpoint
    /// has it.
    pub fn plan(
        checkpoint: &Path,
        role: &str,
        ranks: usize,
        layers: Option<&BTreeSet<usize>>,
        drafter: Option<Drafter>,
    ) -> LabResult<Self> {
        let checkpoint = Arc::new(
            DirectCheckpointV1::open(checkpoint)
                .map_err(|e| format!("open checkpoint {}: {e}", checkpoint.display()))?,
        );
        let components = drafter
            .map(|kind| {
                let kind = match kind {
                    Drafter::Mtp => "mtp",
                    Drafter::Dflash => "dflash",
                };
                NamedBytesV1 {
                    name: DRAFTER_KIND_ROLE.into(),
                    bytes: serde_json::to_vec(&serde_json::json!({ "drafter": kind }))
                        .expect("drafter kind JSON"),
                }
            })
            .into_iter()
            .collect::<Vec<_>>();
        // Planning through the contract, not a built module.
        let implementation = runtime_host::direct::implementation_v1([0; 32]);
        let plans = (0..ranks as u32)
            .map(|rank| {
                let request = checkpoint.request(
                    role,
                    ranks as u32,
                    rank,
                    implementation,
                    components.clone(),
                    vec![],
                );
                weight_plan::plan_layers(&request, layers)
                    .map_err(|e| format!("plan {role} rank {rank}: {e}"))?
                    .ok_or_else(|| format!("the MiMo planner declined {role}"))
            })
            .collect::<LabResult<Vec<_>>>()?;
        let image = PlannedImageV1::new(
            checkpoint,
            plans,
            None,
            implementation,
            DirectLoadOptionsV1::for_ranks(ranks),
        )?;
        let mut indexes = Vec::new();
        let mut index_bytes = Vec::new();
        for rank in 0..ranks {
            let name = index_role(role, rank as u32);
            let bytes = image
                .components()
                .iter()
                .find(|component| component.name == name)
                .ok_or_else(|| format!("the plan has no {name}"))?
                .bytes
                .clone();
            let index: PlannedIndex = serde_json::from_slice(&bytes)
                .map_err(|e| format!("planned rank {rank} index: {e}"))?;
            if index.rank != rank
                || index.image_bytes != image.arena_bytes(rank)
                || image.identity(rank) != Some(index.direct_identity.as_str())
            {
                return Err(format!(
                    "planned rank {rank} index differs from the executed plan"
                ));
            }
            indexes.push(index);
            index_bytes.push(bytes);
        }
        Ok(Self {
            image,
            indexes,
            index_bytes,
            reads: Mutex::new(Vec::new()),
        })
    }

    pub fn ranks(&self) -> usize {
        self.indexes.len()
    }

    /// `bytes` bytes at `offset` of rank `rank`'s arena.
    pub fn read(&self, rank: usize, offset: u64, bytes: u64) -> LabResult<Vec<u8>> {
        self.read_as(rank, offset, bytes, None)
    }

    fn read_as(
        &self,
        rank: usize,
        offset: u64,
        bytes: u64,
        view: Option<&str>,
    ) -> LabResult<Vec<u8>> {
        if offset + bytes > self.image.arena_bytes(rank) {
            return Err(format!("read past rank {rank}'s planned arena"));
        }
        let base = self.image.description().rank_images()[rank].offset;
        let mut data = vec![0; bytes as usize];
        self.image.read(base + offset, &mut data)?;
        self.reads
            .lock()
            .map_err(|_| "planned read record poisoned")?
            .push(Read {
                rank,
                offset,
                bytes,
                view: view.map(str::to_owned),
                sha256: format!("{:x}", Sha256::digest(&data)),
            });
        Ok(data)
    }

    pub fn read_view(&self, rank: usize, view: &PlannedView) -> LabResult<Vec<u8>> {
        self.read_as(rank, view.offset, view.bytes, Some(&view.name))
    }

    /// The view `name` of rank `rank`.
    pub fn read_named(&self, rank: usize, name: &str) -> LabResult<Vec<u8>> {
        self.read_view(rank, self.indexes[rank].view(name)?)
    }

    /// Rank `rank`'s whole arena.
    pub fn read_rank(&self, rank: usize) -> LabResult<Vec<u8>> {
        self.read(rank, 0, self.image.arena_bytes(rank))
    }

    /// The SHA-256 of rank `rank`'s index JSON as the plan produced it.
    pub fn index_sha256(&self, rank: usize) -> String {
        format!("{:x}", Sha256::digest(&self.index_bytes[rank]))
    }

    /// The content of the bytes read from rank `rank` so far: each distinct
    /// read's offset, length, view and SHA-256 in offset order, and one
    /// SHA-256 over those lines. A whole-arena read's digest is the
    /// `bytes_sha256` a direct-load ledger records for the rank.
    pub fn read_record(&self, rank: usize) -> serde_json::Value {
        let mut reads: Vec<Read> = self
            .reads
            .lock()
            .map(|reads| reads.iter().filter(|r| r.rank == rank).cloned().collect())
            .unwrap_or_default();
        reads.sort();
        reads.dedup();
        let lines: String = reads
            .iter()
            .map(|r| format!("{} {} {}\n", r.offset, r.bytes, r.sha256))
            .collect();
        let arena = self.image.arena_bytes(rank);
        serde_json::json!({
            "content_sha256": format!("{:x}", Sha256::digest(lines.as_bytes())),
            "bytes_sha256": reads.iter()
                .find(|r| r.offset == 0 && r.bytes == arena)
                .map(|r| r.sha256.clone()),
            "reads": reads.iter().map(|r| serde_json::json!({
                "offset": r.offset, "bytes": r.bytes, "view": r.view, "sha256": r.sha256,
            })).collect::<Vec<_>>(),
        })
    }

    /// What the load is: the checkpoint's header identity and, per rank, the
    /// planned identity, arena length and index digest, and the content of
    /// the bytes read from it ([`Self::read_record`]).
    pub fn record(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": "planned",
            "image_role": self.image.image_role(),
            "checkpoint_identity": self.image.checkpoint().identity_hex(),
            "images": (0..self.ranks()).map(|rank| serde_json::json!({
                "rank": rank,
                "direct_identity": self.indexes[rank].direct_identity,
                "bytes": self.image.arena_bytes(rank),
                "index_sha256": self.index_sha256(rank),
                "read": self.read_record(rank),
            })).collect::<Vec<_>>(),
        })
    }

    /// Refuses a checkpoint whose files changed since it was planned.
    pub fn verify_unchanged(&self) -> LabResult<()> {
        self.image.checkpoint().verify_unchanged()
    }
}
