//! Prepare the model-neutral retained-window gather.
use super::select::{Candidate, choose};
use crate::latent::window_stage::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument as A, ArgumentList, NumericalContract, Refusal, Validate};
use kernel_core::latent_attention::{LatentWindowBuffers, LatentWindowStorage};
use kernel_core::latent_window_stage::{LatentWindowStage, LatentWindowStageBuffers};
use kernel_core::properties::Invariance;

pub(super) fn prepare(
    entries: &[Entry],
    query: &LatentWindowStage,
    need: Need,
) -> Result<Prepared<LatentWindowStageBuffers<View>>, Refused> {
    query.validate()?;
    if query.latent != 512 || need.capacity == 0 || need.capacity > super::MAX_CAPACITY {
        return Err(
            Refusal("SM120 window stage requires 512 channels and a bounded item count").into(),
        );
    }
    let entry = choose(
        "window stage",
        entries.iter().filter(|e| e.mutation.is_none()).map(|e| {
            Candidate::new(
                e.clone(),
                &e.symbol,
                format!("{:?}", e.spec()),
                Invariance::any_rows(e.capacity),
            )
            .capacity(e.capacity)
        }),
        need,
    )?
    .entry;
    super::check_plan(&entry.plan(need.capacity, query.window)?)?;
    let mut info = OpInfo::new(
        format!("sm120.{}", query.numerical_contract_id()?),
        Invariance::any_rows(need.capacity),
        vec![(entry.symbol.clone(), 0)],
    );
    let n = u64::from(need.capacity);
    info.bytes = vec![
        ("ring.codes".into(), u64::from(query.ring_slots) * 512),
        ("ring.scales".into(), u64::from(query.ring_slots) * 16),
        ("ring.item_slots".into(), n * 4),
        (
            "staged.codes".into(),
            n * u64::from(query.staged_slots) * 512,
        ),
        (
            "staged.scales".into(),
            n * u64::from(query.staged_slots) * 16,
        ),
        ("first_positions".into(), n * 4),
        ("last_positions".into(), 4),
        ("live_items".into(), 4),
    ];
    need.check(&info)?;
    Ok(Box::new(Stage {
        entry,
        query: *query,
        capacity: need.capacity,
        info,
    }))
}
struct Stage {
    entry: Entry,
    query: LatentWindowStage,
    capacity: u32,
    info: OpInfo,
}
fn extent(view: LatentWindowBuffers<View>, slots: u32) -> Result<(), Refusal> {
    let actual = match view.storage {
        LatentWindowStorage::Ring128 => 128,
        LatentWindowStorage::Staged { slots } => slots,
    };
    if actual != slots || view.sequences == 0 {
        return Err(Refusal(
            "window stage plane geometry differs from its descriptor",
        ));
    }
    for (buffer, stride, width) in [
        (view.codes, view.code_stride, 512),
        (view.scales, view.scale_stride, 16),
    ] {
        let plane = u64::from(slots) * width;
        if stride < plane {
            return Err(Refusal("window stage plane stride is too short"));
        }
        let size = u64::from(view.sequences - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(plane))
            .ok_or(Refusal("window stage plane extent overflows"))?;
        buffer.covers(size)?;
    }
    Ok(())
}
fn ptr(view: View, bytes: u64) -> Result<A, Refusal> {
    view.covers(bytes)?;
    Ok(A::DeviceBuffer(view.handle))
}
impl PreparedOp<LatentWindowStageBuffers<View>> for Stage {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &LatentWindowStageBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows > self.capacity
            || b.ring.items < counts.rows
            || b.staged.sequences < counts.rows
        {
            return Err(Refusal("window stage item views are too short"));
        }
        if b.staged.storage
            != (LatentWindowStorage::Staged {
                slots: self.query.staged_slots,
            })
        {
            return Err(Refusal("window stage destination must use staged storage"));
        }
        extent(b.ring, self.query.ring_slots)?;
        extent(b.staged, self.query.staged_slots)?;
        workspace.covers(self.info.workspace)?;
        let reads = [
            b.ring.codes.handle,
            b.ring.scales.handle,
            b.ring.item_slots.handle,
            b.first_positions.handle,
            b.last_positions.handle,
            b.live_items.handle,
        ];
        if b.staged.codes.handle == b.staged.scales.handle
            || reads.contains(&b.staged.codes.handle)
            || reads.contains(&b.staged.scales.handle)
        {
            return Err(Refusal(
                "window stage source, metadata and destination must not alias",
            ));
        }
        let n = u64::from(counts.rows);
        let arguments = ArgumentList(vec![
            A::DeviceBuffer(b.ring.codes.handle),
            A::DeviceBuffer(b.ring.scales.handle),
            ptr(b.ring.item_slots, n * 4)?,
            ptr(b.first_positions, n * 4)?,
            ptr(b.last_positions, u64::from(b.ring.sequences) * 4)?,
            A::DeviceBuffer(b.staged.codes.handle),
            A::DeviceBuffer(b.staged.scales.handle),
            ptr(b.live_items, 4)?,
            A::U32(counts.rows),
            A::U32(b.ring.sequences),
            A::U32(self.query.window),
            A::U32(self.query.ring_slots),
            A::U64(b.ring.code_stride),
            A::U64(b.ring.scale_stride),
            A::U64(b.staged.code_stride),
            A::U64(b.staged.scale_stride),
        ]);
        arguments.validate()?;
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows: counts.rows,
            plan: self.entry.plan(counts.rows, self.query.window)?,
            arguments,
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use kernel_core::backend::Backend;
    fn query() -> LatentWindowStage {
        LatentWindowStage {
            latent: 512,
            window: 128,
            ring_slots: 136,
            staged_slots: 256,
        }
    }
    fn plane(code: u64, scale: u64, slots: u32, sequences: u32) -> LatentWindowBuffers<View> {
        LatentWindowBuffers {
            codes: view(code, u64::MAX),
            scales: view(scale, u64::MAX),
            item_slots: view(3, 8),
            items: 2,
            sequences,
            code_stride: (1_u64 << 32) + u64::from(slots) * 512 + 66,
            scale_stride: (1_u64 << 33) + u64::from(slots) * 16 + 23,
            storage: LatentWindowStorage::Staged { slots },
            key_lists: View::EMPTY,
            key_stride: 0,
            budget: 0,
        }
    }
    fn buffers() -> LatentWindowStageBuffers<View> {
        LatentWindowStageBuffers {
            ring: plane(1, 2, 136, 3),
            staged: plane(4, 5, 256, 2),
            first_positions: view(6, 8),
            last_positions: view(7, 12),
            live_items: view(8, 4),
        }
    }
    #[test]
    fn gather_keeps_four_u64_strides_and_refuses_invalid_extents_and_aliases() {
        let backend = Sm120::new().with_latent_window_stage(Entry {
            capacity: 16,
            symbol: "stage".into(),
            mutation: None,
        });
        let op = backend
            .latent_window_stage(&query(), Need::rows(2))
            .unwrap();
        let b = buffers();
        let launches = op.launches(&b, View::EMPTY, Counts::rows(2)).unwrap();
        let args = &launches[0].arguments.0;
        for (at, stride) in [
            b.ring.code_stride,
            b.ring.scale_stride,
            b.staged.code_stride,
            b.staged.scale_stride,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(args[12 + at], A::U64(stride));
        }
        assert_eq!(launches[0].plan.grid, [2, 127, 1]);
        assert!(launches[0].plan.pdl);
        let mut bad = b;
        bad.ring.items = 1;
        assert!(op.launches(&bad, View::EMPTY, Counts::rows(2)).is_err());
        let mut bad = b;
        bad.staged.scales = b.ring.codes;
        assert!(op.launches(&bad, View::EMPTY, Counts::rows(2)).is_err());
        let mut bad = b;
        bad.ring.code_stride = 135 * 512;
        assert!(op.launches(&bad, View::EMPTY, Counts::rows(2)).is_err());
        let mut bad = b;
        bad.staged.scale_stride = u64::MAX;
        assert!(op.launches(&bad, View::EMPTY, Counts::rows(2)).is_err());
        let mut bad = b;
        bad.first_positions = view(6, 4);
        assert!(op.launches(&bad, View::EMPTY, Counts::rows(2)).is_err());
        assert!(op.launches(&b, View::EMPTY, Counts::rows(3)).is_err());
    }
}
