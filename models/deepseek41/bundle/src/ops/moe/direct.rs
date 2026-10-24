//! Bounds-checked binding of the measured direct MoE stage calls.
use super::*;
use kernel_core::backend::{Launch, Need, OpInfo};
use kernel_core::contract::Argument;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Point {
    RouterPrepare,
    Route,
    Gate,
    Down,
    Sum,
}

pub(crate) enum Source {
    WorkspaceAt(u64),
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
    router_planes: Option<RouterPlanes>,
}

#[derive(Clone, Copy)]
pub(crate) struct RouterPlanes {
    pub domain_bytes: u64,
    pub partial_offset: u64,
    pub partial_bytes: u64,
}

impl RouterPlanes {
    fn bind(
        self,
        workspace: DeviceBufferSpanV1,
    ) -> Result<(DeviceBufferSpanV1, DeviceBufferSpanV1), String> {
        let end = self
            .partial_offset
            .checked_add(self.partial_bytes)
            .ok_or("router partial span overflows")?;
        if self.domain_bytes == 0
            || self.partial_bytes == 0
            || self.domain_bytes > self.partial_offset
            || !workspace.offset.is_multiple_of(256)
            || !self.partial_offset.is_multiple_of(256)
            || end > workspace.length
        {
            return Err("router planes are short, overlapping or unaligned".into());
        }
        workspace
            .offset
            .checked_add(end)
            .ok_or("router workspace span overflows")?;
        Ok((
            DeviceBufferSpanV1::new(workspace.buffer, workspace.offset, self.domain_bytes),
            DeviceBufferSpanV1::new(
                workspace.buffer,
                workspace.offset + self.partial_offset,
                self.partial_bytes,
            ),
        ))
    }
}

impl Prepared {
    pub(crate) fn with_router_planes(mut self, planes: RouterPlanes) -> Self {
        self.router_planes = Some(planes);
        self
    }

    pub(crate) fn router_spans(
        &self,
        workspace: DeviceBufferSpanV1,
    ) -> Result<Option<(DeviceBufferSpanV1, DeviceBufferSpanV1)>, String> {
        self.router_planes.map(|p| p.bind(workspace)).transpose()
    }

    pub(crate) fn inplace_scales(&self, gate_up: bool, width: u32) -> bool {
        let point = if gate_up { Point::Gate } else { Point::Down };
        let role = if gate_up {
            "gate_scales"
        } else {
            "down_scales"
        };
        let readers: Vec<_> = self
            .calls
            .iter()
            .filter(|c| c.point == point)
            .filter(|c| {
                c.pointers
                    .iter()
                    .any(|p| matches!(&p.source, Source::View(r) if r == role))
            })
            .collect();
        !readers.is_empty()
            && readers
                .iter()
                .all(|c| crate::Kernels::moe_scale_variant(&c.launch.symbol, width).is_some())
    }

    pub(crate) fn fuses_router(&self) -> bool {
        self.calls.iter().any(|c| c.point == Point::RouterPrepare)
    }
    pub(crate) fn new(info: OpInfo, calls: Vec<Call>, ticket_offset: u64) -> Self {
        Self {
            info,
            calls,
            ticket_offset,
            router_planes: None,
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
                    Source::WorkspaceAt(offset) => DeviceBufferSpanV1::new(
                        workspace.buffer,
                        workspace
                            .offset
                            .checked_add(*offset)
                            .ok_or("MoE workspace offset overflows")?,
                        workspace
                            .length
                            .checked_sub(*offset)
                            .ok_or("MoE workspace offset exceeds allocation")?,
                    ),
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

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    #[test]
    fn router_planes_refuse_short_overlapping_and_overflowing_spans() {
        let planes = RouterPlanes {
            domain_bytes: 256,
            partial_offset: 256,
            partial_bytes: 61440 * 64,
        };
        let workspace = DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, 1),
            256,
            256 + planes.partial_bytes,
        );
        let (domain, partials) = planes.bind(workspace).unwrap();
        assert_eq!(domain.offset + domain.length, partials.offset);
        for bad in [
            DeviceBufferSpanV1 {
                length: workspace.length - 1,
                ..workspace
            },
            DeviceBufferSpanV1 {
                offset: 1,
                ..workspace
            },
            DeviceBufferSpanV1 {
                offset: u64::MAX - 255,
                ..workspace
            },
        ] {
            assert!(planes.bind(bad).is_err());
        }
        assert!(
            RouterPlanes {
                domain_bytes: 257,
                ..planes
            }
            .bind(workspace)
            .is_err()
        );
        assert!(
            RouterPlanes {
                partial_bytes: u64::MAX,
                ..planes
            }
            .bind(workspace)
            .is_err()
        );
    }
}
