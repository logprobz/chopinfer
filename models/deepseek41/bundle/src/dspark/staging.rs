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
use super::execution::{Diagnostic, DraftInput, Host, Input, TargetRows};
use super::state::{BLOCK_ROWS, CommitPlan, Kind, Projection, RowProjection};
use crate::engram_rows::{self, Destination, EngramRows, Hashed, Mapped, Step, Table};

/// Bounded diagnostics returned after device target execution. Confidence
/// stays as raw words so NaN payloads and signed zero survive unchanged.
pub struct DeviceTarget {
    pub ids: Vec<[u32; BLOCK_ROWS as usize]>,
    pub confidence_bits: Vec<[u32; BLOCK_ROWS as usize]>,
}

/// The prepared program's device work. Projection callbacks may enqueue work;
/// target, prefix-capture and commit boundaries validate saved status before
/// publishing results. A failed validation poisons the lifecycle.
pub trait Program {
    fn project_committed(&mut self, pass: Projection) -> Result<(), String>;
    fn project_rows(&mut self, passes: &[RowProjection]) -> Result<(), String>;
    fn draft(&mut self, inputs: &[DraftInput]) -> Result<(), String>;
    fn target(&mut self, rows: &[TargetRows]) -> Result<(), String>;
    /// Device-resident decode target, with committed lookback and text masks.
    /// None is a refusal before work; successful results follow verification.
    fn device_target(
        &mut self,
        _inputs: &[Input],
        _drafts: &[DraftInput],
        _history: &[i64],
        _text: &[u8],
    ) -> Result<Option<DeviceTarget>, String> {
        Ok(None)
    }

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
    lookback: usize,
    layers: usize,
    head_dim: u64,
    pending: Vec<Pending>,
    confidence: Option<Vec<[u32; BLOCK_ROWS as usize]>>,
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
            lookback: hash.engram().max_ngram as usize - 1,
            layers: g.shape.engram.layers.len(),
            head_dim: u64::from(g.shape.engram.head_dim),
            rows: EngramRows::new(hash),
            pending: Vec::new(),
            confidence: None,
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
        self.confidence = None;
    }

    fn pending_rows(
        rows: &[TargetRows],
        text: &BTreeMap<u64, Vec<bool>>,
    ) -> Result<Vec<Pending>, String> {
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
        Ok(pending)
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
        let pending = Self::pending_rows(rows, text)?;
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

/// Host wall time of each synchronous phase of one execute, in microseconds.
/// Catch-up, drafting and readback end in device synchronizations, so they
/// include their device time; target launch returns before its device work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhaseTimes {
    pub catch_up_us: u64,
    pub catch_up_items: u32,
    pub catch_up_rows: u32,
    pub bulk_us: u64,
    pub bulk_passes: u32,
    pub draft_us: u64,
    pub read_ids_us: u64,
    pub read_confidence_us: u64,
    pub gather_us: u64,
    pub upload_us: u64,
    pub target_us: u64,
}

impl PhaseTimes {
    pub fn record(&self, context: Context, items: usize, drafting: usize) -> String {
        format!(
            concat!(
                "{{\"schema\":\"chopinfer.deepseek41.step-shape.v1\",",
                "\"instance_id\":\"{}\",\"step_id\":\"{}\",\"items\":{},\"drafting\":{},",
                "\"catch_up_items\":{},\"catch_up_rows\":{},\"bulk_passes\":{},",
                "\"micros\":{{\"catch_up\":{},\"bulk\":{},\"draft\":{},\"read_ids\":{},",
                "\"read_confidence\":{},\"gather\":{},\"upload\":{},\"target\":{}}}}}"
            ),
            context.instance_id,
            context.step_id,
            items,
            drafting,
            self.catch_up_items,
            self.catch_up_rows,
            self.bulk_passes,
            self.catch_up_us,
            self.bulk_us,
            self.draft_us,
            self.read_ids_us,
            self.read_confidence_us,
            self.gather_us,
            self.upload_us,
            self.target_us,
        )
    }
}

/// Times `work` into the field `phase` selects when timing is enabled.
fn timed<T>(
    times: &mut Option<PhaseTimes>,
    phase: fn(&mut PhaseTimes) -> &mut u64,
    work: impl FnOnce() -> T,
) -> T {
    let started = times.is_some().then(std::time::Instant::now);
    let result = work();
    if let (Some(times), Some(started)) = (times.as_mut(), started) {
        *phase(times) += started.elapsed().as_micros() as u64;
    }
    result
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
    /// Phase times of this callback, when step tracing is enabled.
    pub times: Option<PhaseTimes>,
}

impl<P> Adapter<'_, P> {
    fn upload(&mut self, gathered: Gathered) -> Result<(), String> {
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
}

impl<P: Program> Host for Adapter<'_, P> {
    type Error = String;
    type Gathered = Gathered;

    fn project_committed(&mut self, pass: Projection) -> Result<(), String> {
        if let Some(times) = &mut self.times {
            times.bulk_passes += 1;
        }
        let program = &mut *self.program;
        timed(
            &mut self.times,
            |t| &mut t.bulk_us,
            || program.project_committed(pass),
        )
    }
    fn project_rows(&mut self, passes: &[RowProjection]) -> Result<(), String> {
        if let Some(times) = &mut self.times {
            times.catch_up_items += passes.len() as u32;
            times.catch_up_rows += passes.iter().map(|pass| pass.count).sum::<u32>();
        }
        let program = &mut *self.program;
        timed(
            &mut self.times,
            |t| &mut t.catch_up_us,
            || program.project_rows(passes),
        )
    }
    fn draft(&mut self, inputs: &[DraftInput]) -> Result<(), String> {
        let program = &mut *self.program;
        timed(
            &mut self.times,
            |t| &mut t.draft_us,
            || program.draft(inputs),
        )
    }
    fn read_draft_ids(&mut self, ids: &mut [[u32; BLOCK_ROWS as usize]]) -> Result<(), String> {
        let source = prefix(
            self.readback.draft_ids,
            (ids.len() * BLOCK_ROWS as usize * 4) as u64,
        )?;
        let (services, stream) = (self.services, self.readback.stream);
        timed(
            &mut self.times,
            |t| &mut t.read_ids_us,
            || {
                engram_rows::read_draft_ids(services, stream, source, ids)
                    .map_err(|e| format!("DSpark draft readback: {e:?}"))
            },
        )
    }
    fn read_confidence(&mut self, scores: &mut [[f32; BLOCK_ROWS as usize]]) -> Result<(), String> {
        if let Some(words) = self.staging.confidence.take() {
            if words.len() != scores.len() {
                return Err("DSpark confidence block count differs".into());
            }
            for (to, from) in scores.iter_mut().zip(words) {
                *to = from.map(f32::from_bits);
            }
            return Ok(());
        }
        let mut bytes = vec![0; scores.len() * BLOCK_ROWS as usize * 4];
        let source = prefix(self.readback.confidence, bytes.len() as u64)?;
        let (services, stream) = (self.services, self.readback.stream);
        timed(
            &mut self.times,
            |t| &mut t.read_confidence_us,
            || services.copy_device_to_host(stream, &mut bytes, source),
        )
        .map_err(|e| e.to_string())?;
        for (score, bytes) in scores.iter_mut().flatten().zip(bytes.chunks_exact(4)) {
            *score = f32::from_bits(u32::from_le_bytes(bytes.try_into().expect("one F32 word")));
        }
        Ok(())
    }
    fn device_target(
        &mut self,
        inputs: &[Input],
        drafts: &[DraftInput],
    ) -> Result<Option<Vec<[u32; BLOCK_ROWS as usize]>>, String> {
        if drafts.is_empty()
            || inputs
                .iter()
                .any(|input| !matches!(input.rows.kind, Kind::Decode { .. }))
        {
            return Ok(None);
        }
        let lookback = self.staging.lookback;
        let mut history = Vec::with_capacity(inputs.len() * lookback);
        let mut text = Vec::new();
        for input in inputs {
            let past = self.staging.rows.history(input.rows.request);
            if past.len() > lookback {
                return Err("Engram committed lookback exceeds its bound".into());
            }
            history.extend(std::iter::repeat_n(
                model_deepseek41_contract::own_ops::engram::DEAD,
                lookback - past.len(),
            ));
            history.extend_from_slice(past);
            let anchor = match self.text.get(&input.rows.request) {
                None => true,
                Some(mask) if mask.len() == 1 => mask[0],
                Some(_) => return Err("DSpark text mask does not cover the decode anchor".into()),
            };
            text.push(u8::from(anchor));
            text.extend(std::iter::repeat_n(1, input.rows.kind.drafts() as usize));
        }
        let Some(result) = self
            .program
            .device_target(inputs, drafts, &history, &text)?
        else {
            return Ok(None);
        };
        if result.ids.len() != drafts.len() || result.confidence_bits.len() != drafts.len() {
            return Err("DSpark device diagnostics differ from the draft block count".into());
        }
        self.staging.confidence = Some(result.confidence_bits);
        Ok(Some(result.ids))
    }
    fn retain_target_rows(&mut self, rows: &[TargetRows]) -> Result<(), String> {
        self.staging.pending = Staging::pending_rows(rows, self.text)?;
        Ok(())
    }
    fn hash_and_gather(&mut self, rows: &[TargetRows]) -> Result<Gathered, String> {
        let (staging, text, ranks) = (&mut *self.staging, self.text, self.ranks);
        timed(
            &mut self.times,
            |t| &mut t.gather_us,
            || staging.hash(rows, text, ranks),
        )
    }
    fn upload_engram(&mut self, gathered: Gathered) -> Result<(), String> {
        let started = self.times.is_some().then(std::time::Instant::now);
        let result = self.upload(gathered);
        if let (Some(times), Some(started)) = (self.times.as_mut(), started) {
            times.upload_us += started.elapsed().as_micros() as u64;
        }
        result
    }
    fn target(&mut self, rows: &[TargetRows]) -> Result<(), String> {
        let program = &mut *self.program;
        timed(
            &mut self.times,
            |t| &mut t.target_us,
            || program.target(rows),
        )
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
