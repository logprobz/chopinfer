//! Delayed uploads for testing the prepared gather's consumption boundary.
use super::*;
use crate::device::mapped_gather::{MappedGatherPlanV1, gather_plane_v1};
use runtime_bundle_abi::MappedGatherStatsV1;

pub(super) struct Gather {
    plan: MappedGatherPlanV1,
    pending: Option<Vec<(DeviceObjectV1, DeviceRangeV1, Vec<u8>)>>,
    waited: BTreeSet<u64>,
    failed: bool,
}

impl FakeHostDeviceV1 {
    pub(super) fn prepare_gather(
        &mut self,
        plan: MappedGatherPlanV1,
    ) -> HostDeviceResultV1<DeviceObjectV1> {
        plan.validate()?;
        self.require(plan.producer, FakeObjectKindV1::Stream)?;
        self.range(plan.indices)?;
        for plane in &plan.planes {
            self.require(plane.stream, FakeObjectKindV1::Stream)?;
            self.range(plane.destination)?;
        }
        let object = self.create(FakeObjectKindV1::MappedGather, Vec::new())?;
        self.mapped_gathers.insert(
            object.get(),
            Gather {
                plan,
                pending: None,
                waited: BTreeSet::new(),
                failed: false,
            },
        );
        Ok(object)
    }

    pub(super) fn enqueue_gather(
        &mut self,
        object: DeviceObjectV1,
        rows: u64,
    ) -> HostDeviceResultV1<()> {
        let gather = self
            .mapped_gathers
            .get(&object.get())
            .ok_or(BundleStatusV1::StaleHandle)?;
        if gather.pending.is_some() || gather.failed {
            return Err(BundleStatusV1::InvalidState);
        }
        gather.plan.validate_rows(rows)?;
        let plan = gather.plan.clone();
        // The CPU fixture computes now but withholds every upload until wait.
        // There is no implicit stream synchronization in this operation.
        let (start, end) = self.range(plan.indices)?;
        let input = &self
            .require(plan.indices.object, FakeObjectKindV1::Buffer)?
            .bytes[start..end];
        let mut pending = Vec::new();
        let mut failed = false;
        for plane in &plan.planes {
            let mut bytes = vec![0; rows as usize * plane.row_bytes];
            failed |= gather_plane_v1(plane, input, &mut bytes).is_err();
            let range = DeviceRangeV1 {
                length: bytes.len() as u64,
                ..plane.destination
            };
            pending.push((plane.stream, range, bytes));
        }
        let gather = self
            .mapped_gathers
            .get_mut(&object.get())
            .expect("retained gather");
        gather.pending = Some(pending);
        gather.waited.clear();
        gather.failed = failed;
        Ok(())
    }

    pub(super) fn wait_gather(
        &mut self,
        object: DeviceObjectV1,
        stream: DeviceObjectV1,
    ) -> HostDeviceResultV1<()> {
        let gather = self
            .mapped_gathers
            .get_mut(&object.get())
            .ok_or(BundleStatusV1::StaleHandle)?;
        let pending = gather
            .pending
            .as_ref()
            .ok_or(BundleStatusV1::InvalidState)?;
        if !pending.iter().any(|(owner, _, _)| *owner == stream) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if !gather.waited.insert(stream.get()) {
            return Err(BundleStatusV1::InvalidState);
        }
        let copies = pending
            .iter()
            .filter(|(owner, _, _)| *owner == stream)
            .cloned()
            .collect::<Vec<_>>();
        for (_, range, bytes) in copies {
            self.copy_host_to_device(stream, range, &bytes)?;
        }
        Ok(())
    }

    pub(super) fn complete_gather(
        &mut self,
        object: DeviceObjectV1,
    ) -> HostDeviceResultV1<Option<MappedGatherStatsV1>> {
        let gather = self
            .mapped_gathers
            .get_mut(&object.get())
            .ok_or(BundleStatusV1::StaleHandle)?;
        let pending = gather
            .pending
            .as_ref()
            .ok_or(BundleStatusV1::InvalidState)?;
        if pending
            .iter()
            .any(|(stream, _, _)| !gather.waited.contains(&stream.get()))
        {
            return Ok(None);
        }
        if gather.failed {
            return Err(BundleStatusV1::HostError);
        }
        gather.pending = None;
        Ok(Some(MappedGatherStatsV1::default()))
    }

    pub(super) fn release_gather(&mut self, object: DeviceObjectV1) -> HostDeviceResultV1<()> {
        self.destroy(object, FakeObjectKindV1::MappedGather)?;
        self.mapped_gathers
            .remove(&object.get())
            .ok_or(BundleStatusV1::StaleHandle)?;
        Ok(())
    }
}

impl Gather {
    pub(super) fn pinned_bytes(&self) -> u64 {
        self.plan.indices.length
            + self
                .plan
                .planes
                .iter()
                .map(|p| p.destination.length)
                .sum::<u64>()
    }
}
