//! CUDA ownership for one prepared mapped-row gather.
use super::*;
use crate::device::mapped_gather::{MappedGatherPlanV1, gather_plane_v1};
use runtime_bundle_abi::MappedGatherStatsV1;
use runtime_cuda::{HostTransform, HostTransformTarget};

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
        let transform =
            HostTransform::new(&producer.context(), plan.indices.length as usize, &outputs)
                .map_err(driver_status)?;
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
                    for (plane, (output, start)) in planes.iter().zip(positions) {
                        let length = rows as usize * plane.row_bytes;
                        gather_plane_v1(plane, input, &mut outputs[output][start..start + length])?;
                    }
                    Ok(())
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
