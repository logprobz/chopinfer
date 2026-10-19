//! Bounds-checked binding of the measured direct MoE stage calls.
use super::*;
use kernel_core::backend::{Launch, Need, OpInfo};
use kernel_core::contract::Argument;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Point {
    Route,
    Gate,
    Down,
    Sum,
}

pub(crate) enum Source {
    Workspace,
    Tickets,
    View(String),
}
pub(crate) struct Pointer {
    pub index: usize,
    pub source: Source,
    pub bytes: u64,
    pub alignment: u64,
}
pub(crate) struct Call {
    pub point: Point,
    pub launch: Launch,
    pub pointers: Vec<Pointer>,
}
pub(crate) struct Prepared {
    pub info: OpInfo,
    calls: Vec<Call>,
    ticket_offset: u64,
}
impl Prepared {
    pub(crate) fn new(info: OpInfo, calls: Vec<Call>, ticket_offset: u64) -> Self {
        Self {
            info,
            calls,
            ticket_offset,
        }
    }
    pub(crate) fn record(&self, asks: &Asks, phase: ExecutionPhase, rows: u32) {
        asks.record(
            format!("moe.direct {phase:?}-{rows}"),
            Need::rows(rows),
            &self.info,
        );
    }
    pub(crate) fn launch(
        &self,
        point: Point,
        workspace: DeviceBufferSpanV1,
        mut view: impl FnMut(&str) -> Result<DeviceBufferSpanV1, String>,
    ) -> Result<Vec<Lowered>, String> {
        if workspace.length < self.info.workspace || !workspace.offset.is_multiple_of(256) {
            return Err("MoE pipeline workspace is too small or unaligned".into());
        }
        let mut bind = Bind::default();
        let mut result = Vec::new();
        for call in self.calls.iter().filter(|c| c.point == point) {
            let mut launch = call.launch.clone();
            for pointer in &call.pointers {
                let needed = pointer.bytes;
                let mut span = match &pointer.source {
                    Source::Workspace => workspace,
                    Source::Tickets => DeviceBufferSpanV1::new(
                        workspace.buffer,
                        workspace
                            .offset
                            .checked_add(self.ticket_offset)
                            .ok_or("MoE ticket offset overflows")?,
                        256,
                    ),
                    Source::View(role) => view(role)?,
                };
                if span.length < needed || span.offset.checked_add(needed).is_none() {
                    return Err(format!(
                        "MoE pipeline argument {} needs {needed} bytes",
                        pointer.index
                    ));
                }
                if !span.offset.is_multiple_of(pointer.alignment) {
                    return Err("MoE pipeline pointer is unaligned".into());
                }
                span.length = needed;
                launch.arguments.0[pointer.index] = Argument::DeviceBuffer(bind.view(span).handle);
            }
            result.push(bind.lowered(launch));
        }
        Ok(result)
    }
}
