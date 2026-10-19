//! The lifecycle's synchronous draft readback and Engram staging adapter.
//! The mapped-gather host call owns CPU gathering and enqueues its H2D
//! copy before returning. It runs outside captured target graphs.

use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::own_ops::engram::HashTables;
use runtime_bundle_abi::{
    BundleCompiledImageHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1 as Span, HostServicesV1,
};
use std::collections::BTreeMap;

use super::calibration::{self, Context};
use super::device::{DECODE_BASE, slice};
use super::execution::{Diagnostic, DraftInput, Host, TargetRows};
use super::state::{BLOCK_ROWS, CommitPlan, Kind, Projection};
use crate::engram_rows::{self, Destination, EngramRows, Hashed, Mapped, Step, Table};

/// The prepared program's device work. Projection callbacks may enqueue work;
/// target, prefix-capture and commit boundaries validate saved status before
/// publishing results. A failed validation poisons the lifecycle.
pub trait Program {
    fn project_committed(&mut self, pass: Projection) -> Result<(), String>;
    fn draft(&mut self, inputs: &[DraftInput]) -> Result<(), String>;
    fn target(&mut self, rows: &[TargetRows]) -> Result<(), String>;
    fn commit_features(&mut self, plan: &CommitPlan) -> Result<(), String>;
}

pub struct Rank {
    pub stream: BundleStreamHandleV1,
    pub image: BundleCompiledImageHandleV1,
    pub rank: u32,
    pub tables: Vec<Table>,
    /// Capacity spans, sliced to the actual target-row count before IO.
    pub destinations: Vec<Destination>,
}

pub struct Readback {
    pub stream: BundleStreamHandleV1,
    pub draft_ids: Span,
    pub confidence: Span,
}

struct Pending {
    rows: TargetRows,
    text: Option<Vec<bool>>,
}

impl Pending {
    fn step(&self) -> Step<'_> {
        Step {
            request: self.rows.rows.request,
            tokens: &self.rows.tokens,
            text: self.text.as_deref(),
        }
    }
}

pub struct Staging {
    rows: EngramRows,
    columns: usize,
    layers: usize,
    head_dim: u64,
    pending: Vec<Pending>,
}

pub struct Gathered {
    phases: Vec<GatheredPhase>,
}

struct GatheredPhase {
    hashed: Hashed,
    destinations: Vec<Vec<Destination>>,
}

fn prefix(span: Span, bytes: u64) -> Result<Span, String> {
    if bytes > span.length || span.offset.checked_add(bytes).is_none() {
        return Err("DSpark staging span is too small or overflows".into());
    }
    Ok(Span::new(span.buffer, span.offset, bytes))
}

impl Staging {
    pub fn new(g: &Geometry) -> Result<Self, String> {
        let hash = HashTables::of(g)?;
        Ok(Self {
            columns: hash.columns(),
            layers: g.shape.engram.layers.len(),
            head_dim: u64::from(g.shape.engram.head_dim),
            rows: EngramRows::new(hash),
            pending: Vec::new(),
        })
    }

    pub fn restore(
        &mut self,
        request: u64,
        tokens: &[u32],
        text: Option<&[bool]>,
    ) -> Result<(), String> {
        self.rows
            .restore(request, tokens, text)
            .map_err(|e| format!("Engram restore: {e:?}"))
    }

    pub fn forget(&mut self, request: u64) {
        self.rows.forget(request);
    }

    /// Cancellation publishes no history. The next hash replaces the
    /// provisional rows; explicit clearing also releases their host memory.
    pub fn cancel(&mut self) {
        self.pending.clear();
    }

    fn hash(
        &mut self,
        rows: &[TargetRows],
        text: &BTreeMap<u64, Vec<bool>>,
        ranks: &[Rank],
    ) -> Result<Gathered, String> {
        if ranks.is_empty() {
            return Err("DSpark Engram staging needs every rank".into());
        }
        let mut pending = Vec::with_capacity(rows.len());
        for row in rows {
            let mask = match text.get(&row.rows.request) {
                None => None,
                Some(mask) => {
                    let known = match row.rows.kind {
                        Kind::Prefill { rows } | Kind::Verify { rows } => rows as usize,
                        Kind::Decode { .. } => 1,
                    };
                    if mask.len() != known {
                        return Err(
                            "DSpark text mask does not cover the supplied target tokens".into()
                        );
                    }
                    let mut mask = mask.clone();
                    mask.resize(row.tokens.len(), true);
                    Some(mask)
                }
            };
            pending.push(Pending {
                rows: row.clone(),
                text: mask,
            });
        }
        let phases = [true, false]
            .into_iter()
            .map(|prefill| {
                let steps = pending
                    .iter()
                    .filter(|row| matches!(row.rows.rows.kind, Kind::Prefill { .. }) == prefill)
                    .map(Pending::step)
                    .collect::<Vec<_>>();
                self.hash_phase(&steps, prefill, ranks)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.pending = pending;
        Ok(Gathered { phases })
    }

    fn hash_phase(
        &self,
        steps: &[Step<'_>],
        prefill: bool,
        ranks: &[Rank],
    ) -> Result<GatheredPhase, String> {
        let hashed = self
            .rows
            .hash(steps)
            .map_err(|e| format!("Engram hash: {e:?}"))?;
        if prefill && hashed.rows() > DECODE_BASE as usize {
            return Err("DSpark prefill exceeds its Engram staging region".into());
        }
        let base = if prefill { 0 } else { u64::from(DECODE_BASE) };
        let offset = base * self.columns as u64;
        let count = u64::try_from(
            hashed
                .rows()
                .checked_mul(self.columns)
                .ok_or("Engram row count overflows")?,
        )
        .map_err(|_| "Engram rows exceed U64")?;
        let destinations = ranks
            .iter()
            .enumerate()
            .map(|(index, rank)| {
                if rank.rank as usize != index
                    || rank.tables.len() != self.layers
                    || rank.tables.len() != rank.destinations.len()
                {
                    return Err("DSpark Engram ranks or table counts differ".into());
                }
                rank.tables
                    .iter()
                    .zip(&rank.destinations)
                    .map(|(table, dest)| {
                        if table.codes.row_bytes != self.head_dim
                            || table.scales.row_bytes != self.head_dim / 32
                        {
                            return Err("DSpark Engram plane width differs from geometry".into());
                        }
                        Ok(Destination {
                            codes: slice(
                                dest.codes,
                                offset * table.codes.row_bytes,
                                count
                                    .checked_mul(table.codes.row_bytes)
                                    .ok_or("Engram code bytes overflow")?,
                            )?,
                            scales: slice(
                                dest.scales,
                                offset * table.scales.row_bytes,
                                count
                                    .checked_mul(table.scales.row_bytes)
                                    .ok_or("Engram scale bytes overflow")?,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(GatheredPhase {
            hashed,
            destinations,
        })
    }

    fn validate_commit(&self, plan: &CommitPlan) -> Result<(), String> {
        if plan.items().len() != self.pending.len() {
            return Err("Engram commit differs from the staged target batch".into());
        }
        for item in plan.items() {
            let pending = self
                .pending
                .iter()
                .find(|p| p.rows.rows == item.rows)
                .ok_or("Engram commit has no matching target rows")?;
            if item.kept as usize > pending.rows.tokens.len() {
                return Err("Engram commit exceeds the verified rows".into());
            }
        }
        Ok(())
    }

    fn commit(&mut self, plan: &CommitPlan) -> Result<(), String> {
        self.validate_commit(plan)?;
        for item in plan.items() {
            let pending = self
                .pending
                .iter()
                .find(|p| p.rows.rows == item.rows)
                .expect("validated target rows");
            self.rows
                .commit(&pending.step(), item.kept as usize)
                .map_err(|e| format!("Engram commit: {e:?}"))?;
        }
        self.pending.clear();
        Ok(())
    }
}

/// Construct this borrow for each execute, commit or cancel callback, over
/// persistent `Staging` and the lifecycle's persistent `Runtime`.
pub struct Adapter<'a, P> {
    pub program: &'a mut P,
    pub services: &'a HostServicesV1,
    pub staging: &'a mut Staging,
    pub ranks: &'a [Rank],
    pub readback: &'a Readback,
    /// Supplied target-token masks only; emitted drafts are always text.
    pub text: &'a BTreeMap<u64, Vec<bool>>,
    pub diagnostic: Context,
}

impl<P: Program> Host for Adapter<'_, P> {
    type Error = String;
    type Gathered = Gathered;

    fn project_committed(&mut self, pass: Projection) -> Result<(), String> {
        self.program.project_committed(pass)
    }
    fn draft(&mut self, inputs: &[DraftInput]) -> Result<(), String> {
        self.program.draft(inputs)
    }
    fn read_draft_ids(&mut self, ids: &mut [[u32; BLOCK_ROWS as usize]]) -> Result<(), String> {
        let source = prefix(
            self.readback.draft_ids,
            (ids.len() * BLOCK_ROWS as usize * 4) as u64,
        )?;
        engram_rows::read_draft_ids(self.services, self.readback.stream, source, ids)
            .map_err(|e| format!("DSpark draft readback: {e:?}"))
    }
    fn read_confidence(&mut self, scores: &mut [[f32; BLOCK_ROWS as usize]]) -> Result<(), String> {
        let mut bytes = vec![0; scores.len() * BLOCK_ROWS as usize * 4];
        let source = prefix(self.readback.confidence, bytes.len() as u64)?;
        self.services
            .copy_device_to_host(self.readback.stream, &mut bytes, source)
            .map_err(|e| e.to_string())?;
        for (score, bytes) in scores.iter_mut().flatten().zip(bytes.chunks_exact(4)) {
            *score = f32::from_bits(u32::from_le_bytes(bytes.try_into().expect("one F32 word")));
        }
        Ok(())
    }
    fn hash_and_gather(&mut self, rows: &[TargetRows]) -> Result<Gathered, String> {
        self.staging.hash(rows, self.text, self.ranks)
    }
    fn upload_engram(&mut self, gathered: Gathered) -> Result<(), String> {
        for phase in gathered.phases {
            for (rank, destinations) in self.ranks.iter().zip(&phase.destinations) {
                self.staging
                    .rows
                    .gather(
                        self.services,
                        rank.stream,
                        Mapped {
                            image: rank.image,
                            rank: rank.rank,
                            tables: &rank.tables,
                        },
                        destinations,
                        &phase.hashed,
                    )
                    .map_err(|e| format!("Engram gather/upload: {e:?}"))?;
            }
        }
        Ok(())
    }
    fn target(&mut self, rows: &[TargetRows]) -> Result<(), String> {
        self.program.target(rows)
    }
    fn commit_features(&mut self, plan: &CommitPlan) -> Result<(), String> {
        self.staging.validate_commit(plan)?;
        self.program.commit_features(plan)?;
        self.staging.commit(plan)
    }
    fn record_diagnostics(&mut self, diagnostics: &[Diagnostic]) {
        calibration::record(self.diagnostic, diagnostics, |message| {
            self.services.log(message)
        });
        self.staging.cancel();
    }
}

#[cfg(test)]
mod tests;
