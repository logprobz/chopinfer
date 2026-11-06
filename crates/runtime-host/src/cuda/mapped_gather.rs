//! CUDA ownership for one prepared mapped-row gather.
use super::*;
use crate::device::mapped_gather::{MappedGatherPlanV1, gather_planes_v1, plane_outputs};
use runtime_bundle_abi::MappedGatherStatsV1;
use runtime_cuda::{HostTransform, HostTransformInit, HostTransformTarget};

const INIT_ENV: &str = "CHOPINFER_MAPPED_GATHER_INIT";

/// `CHOPINFER_MAPPED_GATHER_INIT`: when a prepared gather's callback zeroes
/// its pinned outputs for the uploads, which copy each plane's live rows.
/// Unset or `extents` (the default) zeroes each stream's output before the
/// gather up to its last plane's live end, the unused capacity of the planes
/// before that included; `failure` zeroes the live rows only after a gather
/// that fails, since one that succeeds has written every byte of them
/// ([`gather_planes_v1`]). Anything else keeps the default. The uploaded
/// bytes are the same either way.
fn configured_init(value: Option<std::ffi::OsString>) -> HostTransformInit {
    match value.as_ref().and_then(|value| value.to_str()) {
        Some("failure") => HostTransformInit::OnFailure,
        _ => HostTransformInit::Extents,
    }
}

fn init() -> HostTransformInit {
    static INIT: std::sync::OnceLock<HostTransformInit> = std::sync::OnceLock::new();
    *INIT.get_or_init(|| configured_init(std::env::var_os(INIT_ENV)))
}

pub(super) struct Gather {
    transform: HostTransform,
    plan: MappedGatherPlanV1,
    input: DeviceBuffer,
    producer: Stream,
    streams: Vec<(DeviceObjectV1, Stream)>,
    targets: Vec<HostTransformTarget>,
    /// Plane destinations are packed per output stream, in declaration order.
    positions: Vec<(usize, usize)>,
    failed: bool,
}

impl Gather {
    pub(super) fn new(
        host: &CudaHostDeviceV1,
        plan: MappedGatherPlanV1,
    ) -> HostDeviceResultV1<Self> {
        plan.validate()?;
        let (input, _, _) = host.range(plan.indices)?;
        let producer = host.stream(plan.producer)?.clone();
        let mut streams: Vec<(DeviceObjectV1, Stream)> = Vec::new();
        let mut capacities: Vec<usize> = Vec::new();
        let mut targets = Vec::new();
        let mut positions = Vec::new();
        for plane in &plan.planes {
            let index = match streams
                .iter()
                .position(|(object, _)| *object == plane.stream)
            {
                Some(index) => index,
                None => {
                    streams.push((plane.stream, host.stream(plane.stream)?.clone()));
                    capacities.push(0);
                    streams.len() - 1
                }
            };
            let (buffer, offset, length) = host.range(plane.destination)?;
            positions.push((index, capacities[index]));
            targets.push(HostTransformTarget {
                buffer: buffer.clone(),
                offset,
                output: index,
                source_offset: capacities[index],
                length,
            });
            capacities[index] += length;
        }
        let outputs = streams
            .iter()
            .zip(&capacities)
            .map(|((_, stream), &bytes)| (stream, bytes))
            .collect::<Vec<_>>();
        let mut transform =
            HostTransform::new(&producer.context(), plan.indices.length as usize, &outputs)
                .map_err(driver_status)?;
        transform.set_init(init());
        Ok(Self {
            transform,
            plan,
            input: input.clone(),
            producer,
            streams,
            targets,
            positions,
            failed: false,
        })
    }

    pub(super) fn enqueue(&mut self, rows: u64) -> HostDeviceResultV1<()> {
        if self.failed {
            return Err(BundleStatusV1::InvalidState);
        }
        self.plan.validate_rows(rows)?;
        for (target, plane) in self.targets.iter_mut().zip(&self.plan.planes) {
            target.length = rows as usize * plane.row_bytes;
        }
        let planes = self.plan.planes.clone();
        let positions = self.positions.clone();
        let result = self
            .transform
            .enqueue(
                &self.input,
                self.plan.indices.offset as usize,
                self.plan.indices.length as usize,
                &self.producer,
                &self.targets,
                move |input, outputs| {
                    let lengths = planes
                        .iter()
                        .map(|plane| rows as usize * plane.row_bytes)
                        .collect::<Vec<_>>();
                    let mut outputs = plane_outputs(outputs, &positions, &lengths)?;
                    gather_planes_v1(&planes, input, &mut outputs)
                },
            )
            .map_err(driver_status);
        self.failed |= result.is_err();
        result
    }

    pub(super) fn wait(&mut self, stream: DeviceObjectV1) -> HostDeviceResultV1<()> {
        if self.failed {
            return Err(BundleStatusV1::InvalidState);
        }
        let index = self
            .streams
            .iter()
            .position(|(object, _)| *object == stream)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let result = self
            .transform
            .wait_on(index, &self.streams[index].1)
            .map_err(driver_status);
        self.failed |= result.is_err();
        result
    }

    pub(super) fn complete(&mut self) -> HostDeviceResultV1<Option<MappedGatherStatsV1>> {
        if self.failed {
            return Err(BundleStatusV1::InvalidState);
        }
        let Some(stats) = self.transform.complete().map_err(driver_status)? else {
            return Ok(None);
        };
        if stats.failure.is_some() {
            self.failed = true;
            return Err(BundleStatusV1::HostError);
        }
        let exposed = stats
            .exposed_milliseconds
            .into_iter()
            .fold(0.0_f32, f32::max);
        Ok(Some(MappedGatherStatsV1 {
            host_nanoseconds: stats.host_nanoseconds,
            exposed_nanoseconds: (f64::from(exposed) * 1_000_000.0).ceil() as u64,
        }))
    }

    pub(super) fn pinned_bytes(&self) -> u64 {
        self.transform.pinned_bytes() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_initialization_reads_its_two_modes_and_defaults_otherwise() {
        for (value, init) in [
            (None, HostTransformInit::Extents),
            (Some("extents"), HostTransformInit::Extents),
            (Some("failure"), HostTransformInit::OnFailure),
            (Some("targets"), HostTransformInit::Extents),
            (Some("Failure"), HostTransformInit::Extents),
            (Some(""), HostTransformInit::Extents),
        ] {
            assert_eq!(configured_init(value.map(Into::into)), init, "{value:?}");
        }
    }
}
