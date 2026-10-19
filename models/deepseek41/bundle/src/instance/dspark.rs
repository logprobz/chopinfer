//! Connect the model-owned lifecycle to one synchronous bundle instance.

mod cycle;
mod features;
mod graphs;
mod launch;
#[cfg(feature = "diagnostic-taps")]
mod state_capture;

use std::collections::BTreeMap;

use crate::dspark::execution::{Input, TargetRows};
use crate::dspark::staging::{Adapter, Program, Rank, Readback};
use crate::dspark::state::{BLOCK_ROWS, CommitPlan, Kind, Projection, Rows, Snapshot};
use bundle_runtime::inputs::{PackedStep, PhaseInputs};
use bundle_runtime::ledger::CommitPlan as LedgerCommit;
use runtime_bundle_abi::{BundleStatusV1, ExecuteResultSinkV1, HostServicesV1};
use runtime_bundle_sdk::BatchCodecV1;

use super::Instance;

type TextMasks = BTreeMap<u64, Vec<bool>>;

struct Work<'a> {
    instance: &'a mut Instance,
    request: &'a [u8],
    step: Option<PackedStep>,
    sink: Option<&'a mut ExecuteResultSinkV1>,
}

impl Program for Work<'_> {
    fn project_committed(&mut self, pass: Projection) -> Result<(), String> {
        self.instance.dspark_project(pass)
    }

    fn draft(&mut self, inputs: &[crate::dspark::execution::DraftInput]) -> Result<(), String> {
        self.instance.dspark_draft(inputs)
    }

    fn target(&mut self, rows: &[TargetRows]) -> Result<(), String> {
        let mut step = self.step.take().ok_or("DSpark target has no packed step")?;
        let sink = self
            .sink
            .as_deref_mut()
            .ok_or("DSpark target has no sink")?;
        let mut position = 0;
        for phase in [&mut step.prefill, &mut step.decode] {
            for item in 0..phase.request_ids.len() {
                let target = rows
                    .get(position)
                    .ok_or("DSpark target rows are incomplete")?;
                let start = phase.q_start[item] as usize;
                let count = phase.q_len[item] as usize;
                if target.rows.request != phase.request_ids[item] || target.tokens.len() != count {
                    return Err("DSpark target rows differ from the packed step".into());
                }
                phase.token_ids[start..start + count].copy_from_slice(&target.tokens);
                if target.rows.kind.drafts() > 0 {
                    sink.emit_draft_tokens(target.rows.request, &target.tokens[1..])
                        .map_err(|e| format!("DSpark draft result: {e:?}"))?;
                }
                position += 1;
            }
        }
        if position != rows.len() {
            return Err("DSpark target has surplus rows".into());
        }
        let status = self.instance.execute_packed(self.request, sink, step);
        if status != BundleStatusV1::Success {
            return Err(format!("DSpark target execute: {status:?}"));
        }
        Ok(())
    }

    fn device_target(
        &mut self,
        inputs: &[Input],
        drafts: &[crate::dspark::execution::DraftInput],
        history: &[i64],
        text: &[u8],
    ) -> Result<Option<crate::dspark::staging::DeviceTarget>, String> {
        let step = self.step.take().ok_or("DSpark target has no packed step")?;
        let sink = self
            .sink
            .as_deref_mut()
            .ok_or("DSpark target has no sink")?;
        self.instance
            .device_dspark_target(
                self.request,
                sink,
                step,
                cycle::TargetContext {
                    inputs,
                    drafts,
                    history,
                    text,
                },
            )
            .map(Some)
    }

    fn commit_features(&mut self, plan: &CommitPlan) -> Result<(), String> {
        self.instance.dspark_commit_features(plan)
    }
}

fn inputs(phase: &PhaseInputs, drafts: &[u32]) -> Result<Vec<Input>, String> {
    (0..phase.request_ids.len())
        .map(|item| {
            let start = phase.q_start[item] as usize;
            let count = phase.q_len[item] as usize;
            let budget = drafts.get(item).copied().unwrap_or(0);
            let kind = if !phase.decode {
                Kind::Prefill { rows: count as u32 }
            } else if budget > 0 {
                Kind::Decode { drafts: budget }
            } else if phase.verify_rows.get(item).copied().unwrap_or(false) {
                Kind::Verify { rows: count as u32 }
            } else {
                Kind::Decode { drafts: 0 }
            };
            let known = if budget > 0 { 1 } else { count };
            Ok(Input {
                rows: Rows {
                    request: phase.request_ids[item],
                    slot: u32::try_from(phase.slot_ids[item])
                        .map_err(|_| "DSpark slot is negative")?,
                    start: phase.prefix_len[item] as u64,
                    kind,
                },
                tokens: phase
                    .token_ids
                    .get(start..start + known)
                    .ok_or("DSpark input rows exceed tokens")?
                    .to_vec(),
            })
        })
        .collect()
}

impl Instance {
    fn dspark_adapter_parts(&self) -> Result<(Vec<Rank>, Readback), String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let image = self.image.ok_or("DSpark rank image is absent")?;
        let tables = self
            .engram
            .as_ref()
            .and_then(|engram| engram.tables())
            .ok_or("DSpark Engram tables are absent")?;
        let ranks = self
            .ranks
            .iter()
            .zip(&self.buffers)
            .enumerate()
            .map(|(rank, (handles, buffers))| {
                Ok(Rank {
                    stream: handles.2,
                    image,
                    rank: rank as u32,
                    tables: tables.to_vec(),
                    destinations: device.engram_destinations(&buffers.scratch)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let first = device
            .ranks
            .first()
            .ok_or("DSpark device state is absent")?;
        Ok((
            ranks,
            Readback {
                stream: self.ranks[0].2,
                draft_ids: first.draft_ids,
                confidence: first.confidence,
            },
        ))
    }

    fn dspark_with<T>(
        &mut self,
        work: impl FnOnce(
            &mut crate::dspark::execution::Runtime,
            &mut Adapter<'_, Work<'_>>,
        ) -> Result<T, String>,
        request: &[u8],
        step: Option<PackedStep>,
        sink: Option<&mut ExecuteResultSinkV1>,
        text: TextMasks,
    ) -> Result<T, String> {
        let (ranks, readback) = self.dspark_adapter_parts()?;
        let services = std::rc::Rc::clone(&self.services);
        let device = self.dspark.as_mut().ok_or("DSpark is absent")?;
        let diagnostic = device.diagnostic;
        if device.runtime.is_none() || device.staging.is_none() {
            return Err("DSpark lifecycle or staging is borrowed".into());
        }
        let mut runtime = device.runtime.take().expect("checked lifecycle");
        let mut staging = device.staging.take().expect("checked staging");
        let mut program = Work {
            instance: self,
            request,
            step,
            sink,
        };
        let mut host = Adapter {
            program: &mut program,
            services: &services,
            staging: &mut staging,
            ranks: &ranks,
            readback: &readback,
            text: &text,
            diagnostic,
        };
        let result = work(&mut runtime, &mut host);
        let device = self
            .dspark
            .as_mut()
            .expect("instance keeps drafter storage");
        device.runtime = Some(runtime);
        device.staging = Some(staging);
        result
    }

    pub(super) fn execute_dspark(
        &mut self,
        request: &[u8],
        sink: &mut ExecuteResultSinkV1,
        step: PackedStep,
    ) -> BundleStatusV1 {
        let prepare = || -> Result<(Vec<Input>, TextMasks), String> {
            let batch =
                BatchCodecV1::decode(request).map_err(|e| format!("DSpark batch: {e:?}"))?;
            let mut rows = inputs(&step.prefill, &[])?;
            rows.extend(inputs(&step.decode, &step.bundle_drafts(&batch))?);
            let image = self
                .config
                .media
                .then_some(self.config.shape.geometry.shape.tokens.image);
            let text = rows
                .iter()
                .filter_map(|input| {
                    let image = image?;
                    input.tokens.contains(&image).then(|| {
                        (
                            input.rows.request,
                            input.tokens.iter().map(|&token| token != image).collect(),
                        )
                    })
                })
                .collect();
            Ok((rows, text))
        };
        let (inputs, text) = match prepare() {
            Ok(value) => value,
            Err(error) => {
                self.log(&error);
                return BundleStatusV1::InvalidArgument;
            }
        };
        #[cfg(feature = "diagnostic-taps")]
        if self.active_dspark_state_capture().is_some() {
            for phase in [&step.prefill, &step.decode] {
                for (&request, pages) in phase.request_ids.iter().zip(&phase.block_tables) {
                    self.state_capture_pages.insert(request, pages.clone());
                }
            }
        } else {
            self.state_capture_pages.clear();
        }
        let device = self.dspark.as_mut().expect("enabled drafter");
        let Some(id) = device.diagnostic.step_id.checked_add(1) else {
            return BundleStatusV1::InternalError;
        };
        device.diagnostic.step_id = id;
        device.verified = inputs
            .iter()
            .filter(|input| input.rows.kind.is_verify())
            .map(|input| input.rows.request)
            .collect();
        let result = self.dspark_with(
            |runtime, host| {
                runtime
                    .execute(host, &inputs)
                    .map(|_| ())
                    .map_err(|e| format!("DSpark execute: {e:?}"))
            },
            request,
            Some(step),
            Some(sink),
            text,
        );
        match result {
            Ok(()) => BundleStatusV1::Success,
            Err(error) => {
                self.log(&error);
                BundleStatusV1::BundleError
            }
        }
    }

    pub(super) fn dspark_commit(&mut self, plan: &LedgerCommit) -> Result<(), BundleStatusV1> {
        let Some(device) = &self.dspark else {
            return Ok(());
        };
        let kept = device
            .verified
            .iter()
            .map(|request| {
                plan.items()
                    .iter()
                    .find(|item| item.request == *request)
                    .map(|item| (*request, item.kept))
                    .ok_or(BundleStatusV1::InvalidState)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.dspark_with(
            |runtime, host| {
                runtime
                    .commit(host, &kept)
                    .map(|_| ())
                    .map_err(|e| format!("DSpark commit: {e:?}"))
            },
            &[],
            None,
            None,
            BTreeMap::new(),
        )
        .map_err(|error| {
            self.log(&error);
            BundleStatusV1::BundleError
        })
    }

    pub(super) fn dspark_cancel(&mut self) -> Result<(), BundleStatusV1> {
        if self.dspark.is_none() {
            return Ok(());
        }
        self.dspark_with(
            |runtime, host| {
                runtime
                    .cancel(host)
                    .map(|_| ())
                    .map_err(|e| format!("DSpark cancel: {e:?}"))
            },
            &[],
            None,
            None,
            BTreeMap::new(),
        )
        .map_err(|error| {
            self.log(&error);
            BundleStatusV1::BundleError
        })?;
        self.dspark
            .as_mut()
            .expect("enabled drafter")
            .staging
            .as_mut()
            .expect("returned staging")
            .cancel();
        Ok(())
    }

    pub(super) fn dspark_catch_up(&mut self, request: u64) -> Result<(), BundleStatusV1> {
        if self.dspark.is_none() {
            return Ok(());
        }
        let result = self
            .dspark_with(
                |runtime, host| {
                    runtime
                        .catch_up(host, request)
                        .map_err(|e| format!("DSpark snapshot catch-up: {e:?}"))
                },
                &[],
                None,
                None,
                BTreeMap::new(),
            )
            .and_then(|()| self.check_dspark_status());
        if result.is_err() {
            self.dspark
                .as_mut()
                .expect("enabled drafter")
                .runtime
                .as_mut()
                .expect("returned lifecycle")
                .poison();
        }
        result.map_err(|error| {
            self.log(&error);
            BundleStatusV1::BundleError
        })
    }

    pub(super) fn dspark_restored(
        &mut self,
        request: u64,
        slot: u32,
        tokens: &[u32],
    ) -> Result<(), BundleStatusV1> {
        let Some(device) = &mut self.dspark else {
            return Ok(());
        };
        let end = tokens.len() as u64;
        let snapshot = Snapshot {
            start: end.saturating_sub(u64::from(device.config.window)),
            end,
            window: device.config.window,
            stages: device.config.stages,
        };
        let image = self
            .config
            .media
            .then_some(self.config.shape.geometry.shape.tokens.image);
        let text = image.map(|image| {
            tokens
                .iter()
                .map(|&token| token != image)
                .collect::<Vec<_>>()
        });
        device
            .staging
            .as_mut()
            .expect("unborrowed staging")
            .restore(request, tokens, text.as_deref())
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        device
            .runtime
            .as_mut()
            .expect("unborrowed lifecycle")
            .restore_after_copy(request, slot, snapshot)
            .map_err(|_| BundleStatusV1::InvalidState)
    }

    pub(super) fn dspark_validate_restore(
        &self,
        request: u64,
        slot: u32,
        end: u64,
    ) -> Result<(), BundleStatusV1> {
        let Some(device) = &self.dspark else {
            return Ok(());
        };
        let snapshot = Snapshot {
            start: end.saturating_sub(u64::from(device.config.window)),
            end,
            window: device.config.window,
            stages: device.config.stages,
        };
        device
            .runtime
            .as_ref()
            .ok_or(BundleStatusV1::InvalidState)?
            .state()
            .validate_restore(request, slot, snapshot)
            .map_err(|_| BundleStatusV1::InvalidState)
    }

    pub(super) fn dspark_poison(&mut self) {
        if let Some(device) = &mut self.dspark {
            device
                .runtime
                .as_mut()
                .expect("unborrowed lifecycle")
                .poison();
        }
    }

    pub(super) fn dspark_forget(&mut self, request: u64) {
        if let Some(device) = &mut self.dspark {
            device
                .staging
                .as_mut()
                .expect("unborrowed staging")
                .forget(request);
            if let Err(error) = device
                .runtime
                .as_mut()
                .expect("unborrowed lifecycle")
                .retire(request)
            {
                device
                    .runtime
                    .as_mut()
                    .expect("unborrowed lifecycle")
                    .poison();
                self.log(&format!("DSpark retirement: {error}"));
            }
        }
    }
}
