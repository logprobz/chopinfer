//! Target hooks: raw feature taps, packed retained rows and Engram staging.

use super::Instance;
use crate::dspark::device::{DECODE_BASE, slice};
use crate::engram_rows::Destination;
use crate::lower::Lowering;
use crate::ops::Lowered;
use crate::program::Phase;
use runtime_bundle_abi::BundleAbiError;

/// A retained prefill interval, in original packed and absolute coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PrefillFeatureRange {
    pub packed_start: u32,
    pub rows: u32,
    pub position: u64,
    pub end: u64,
}

impl Instance {
    pub(crate) fn dspark_feature_range(
        &self,
        base: &Lowering<'_>,
        layer: u32,
        range: PrefillFeatureRange,
    ) -> Result<Vec<Lowered>, String> {
        if range.position.checked_add(u64::from(range.rows)) != Some(range.end) {
            return Err("DSpark retained feature interval has a different end".into());
        }
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        device.prepared.feature_range(
            base,
            layer,
            range.packed_start,
            range.rows,
            device.ranks[base.rank as usize].taps,
        )
    }

    pub(crate) fn dspark_pack_feature_range(
        &self,
        rank: usize,
        range: PrefillFeatureRange,
        counts: &[runtime_bundle_abi::DeviceBufferSpanV1],
    ) -> Result<Vec<Lowered>, String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let stored = &device.ranks[rank];
        let targets = device.shape.geometry.shape.dspark.targets.len() as u64;
        let width = device.features.row_bytes();
        device
            .prepared
            .pack_features(
                range.rows,
                stored.taps,
                slice(
                    stored.feature_indices,
                    u64::from(range.packed_start) * targets * 4,
                    u64::from(range.rows) * targets * 4,
                )?,
                counts,
                slice(
                    stored.packed,
                    u64::from(range.packed_start) * width,
                    u64::from(range.rows) * width,
                )?,
            )
            .map_err(|e| e.to_string())
    }

    pub(crate) fn dspark_feature(
        &self,
        base: &Lowering<'_>,
        phase: Phase,
        layer: u32,
    ) -> Result<Vec<Lowered>, String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        device.prepared.feature(
            base,
            layer,
            phase.rows(),
            device.ranks[base.rank as usize].taps,
        )
    }

    pub(crate) fn dspark_engram(
        &self,
        base: &Lowering<'_>,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let index = device
            .shape
            .geometry
            .shape
            .engram
            .layers
            .iter()
            .position(|&id| id == layer)
            .ok_or("DSpark target names no such Engram layer")?;
        let full = device.engram_destinations(&base.buffers.scratch)?;
        let offset = if phase.prefill() {
            0
        } else {
            u64::from(DECODE_BASE)
        };
        let width = u64::from(crate::ops::dense::engram_k(device.shape.geometry));
        let destination = Destination {
            codes: slice(
                full[index].codes,
                offset * width,
                u64::from(phase.rows()) * width,
            )?,
            scales: slice(
                full[index].scales,
                offset * width / 32,
                u64::from(phase.rows()) * width / 32,
            )?,
        };
        base.engram_rows_staged(phase, layer, destination)
    }

    pub(crate) fn dspark_pack_features(
        &self,
        rank: usize,
        phase: Phase,
    ) -> Result<Vec<Lowered>, String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let stored = &device.ranks[rank];
        let width = device.features.row_bytes();
        let offset = if phase.prefill() {
            0
        } else {
            u64::from(DECODE_BASE)
        };
        let counts = device.prepared.feature_pack_counts(phase.rows(), 0)?.len();
        device
            .prepared
            .pack_features(
                phase.rows(),
                stored.taps,
                stored.feature_indices,
                &stored.feature_counts[..counts],
                slice(
                    stored.packed,
                    offset * width,
                    u64::from(phase.rows()) * width,
                )?,
            )
            .map_err(|e| e.to_string())
    }

    pub(crate) fn dspark_pack_counts(&self, phase: Phase, live: u32) -> Result<(), BundleAbiError> {
        let Some(device) = &self.dspark else {
            return Ok(());
        };
        let counts = device
            .prepared
            .feature_pack_counts(phase.rows(), live)
            .map_err(|_| BundleAbiError::InvalidRange("DSpark feature packing rows"))?;
        for (rank, stored) in device.ranks.iter().enumerate() {
            for (&count, &span) in counts.iter().zip(&stored.feature_counts) {
                self.services.copy_host_to_device(
                    self.ranks[rank].2,
                    span,
                    &count.to_le_bytes(),
                )?;
            }
        }
        Ok(())
    }
}
