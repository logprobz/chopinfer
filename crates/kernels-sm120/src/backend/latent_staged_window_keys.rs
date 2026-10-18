use super::select::{Candidate, choose};
use crate::latent::staged_window_keys::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument as A, ArgumentList, NumericalContract, Refusal, Validate};
use kernel_core::latent_staged_window_keys::{
    LatentStagedWindowKeyBuffers, LatentStagedWindowKeys,
};
use kernel_core::properties::Invariance;
pub(super) fn prepare(
    entries: &[Entry],
    query: &LatentStagedWindowKeys,
    need: Need,
) -> Result<Prepared<LatentStagedWindowKeyBuffers<View>>, Refused> {
    query.validate()?;
    if need.capacity == 0 || need.capacity > super::MAX_CAPACITY {
        return Err(Refusal("staged window lists require bounded rows").into());
    }
    let entry = choose(
        "staged window keys",
        entries.iter().map(|e| {
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
    super::check_plan(&entry.plan(need.capacity)?)?;
    let mut info = OpInfo::new(
        format!("sm120.{}", query.numerical_contract_id()?),
        Invariance::any_rows(need.capacity),
        vec![(entry.symbol.clone(), 0)],
    );
    info.bytes = vec![
        ("positions".into(), u64::from(need.capacity) * 4),
        ("row_items".into(), u64::from(need.capacity) * 4),
        ("chunk_first".into(), 4),
        ("chunk_last".into(), 4),
        ("item_slots".into(), 4),
        ("ring_last".into(), 4),
        (
            "key_lists".into(),
            u64::from(need.capacity) * u64::from(query.window) * 4,
        ),
        ("live_rows".into(), 4),
    ];
    need.check(&info)?;
    Ok(Box::new(Keys {
        entry,
        query: *query,
        capacity: need.capacity,
        info,
    }))
}
struct Keys {
    entry: Entry,
    query: LatentStagedWindowKeys,
    capacity: u32,
    info: OpInfo,
}
fn ptr(v: View, n: u64) -> Result<A, Refusal> {
    v.covers(n)?;
    Ok(A::DeviceBuffer(v.handle))
}
impl PreparedOp<LatentStagedWindowKeyBuffers<View>> for Keys {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &LatentStagedWindowKeyBuffers<View>,
        _: View,
        c: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if c.rows > self.capacity || b.items == 0 || b.sequences == 0 {
            return Err(Refusal("staged window key metadata extents are invalid"));
        }
        if [
            b.positions,
            b.row_items,
            b.chunk_first,
            b.chunk_last,
            b.item_slots,
            b.ring_last,
            b.live_rows,
        ]
        .iter()
        .any(|v| v.handle == b.key_lists.handle)
        {
            return Err(Refusal("staged window keys must not alias input metadata"));
        }
        let r = u64::from(c.rows) * 4;
        let items = u64::from(b.items) * 4;
        let a = ArgumentList(vec![
            ptr(b.positions, r)?,
            ptr(b.row_items, r)?,
            ptr(b.chunk_first, items)?,
            ptr(b.chunk_last, items)?,
            ptr(b.item_slots, items)?,
            ptr(b.ring_last, u64::from(b.sequences) * 4)?,
            ptr(b.key_lists, r * u64::from(self.query.window))?,
            ptr(b.live_rows, 4)?,
            A::U32(b.items),
            A::U32(b.sequences),
            A::U32(self.query.window),
            A::U32(self.query.ring_slots),
            A::U32(self.query.staged_slots),
        ]);
        a.validate()?;
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows: c.rows,
            plan: self.entry.plan(c.rows)?,
            arguments: a,
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
    #[test]
    fn staged_list_bounds_and_output_aliases_are_checked() {
        let b = Sm120::new().with_latent_staged_window_keys(Entry {
            capacity: 32,
            symbol: "keys".into(),
        });
        let q = LatentStagedWindowKeys {
            window: 128,
            ring_slots: 136,
            staged_slots: 256,
        };
        let op = b.latent_staged_window_keys(&q, Need::rows(8)).unwrap();
        let buffers = LatentStagedWindowKeyBuffers {
            positions: view(1, 32),
            row_items: view(2, 32),
            chunk_first: view(3, 16),
            chunk_last: view(4, 16),
            item_slots: view(5, 16),
            ring_last: view(6, 8),
            items: 4,
            sequences: 2,
            key_lists: view(7, 8 * 128 * 4),
            live_rows: view(8, 4),
        };
        let l = op.launches(&buffers, View::EMPTY, Counts::rows(8)).unwrap();
        assert_eq!(l[0].plan.grid, [8, 1, 1]);
        assert_eq!(l[0].arguments.0[8], A::U32(4));
        let mut short = buffers;
        short.ring_last = view(6, 4);
        assert!(op.launches(&short, View::EMPTY, Counts::rows(8)).is_err());
        let mut alias = buffers;
        alias.key_lists = alias.positions;
        assert!(op.launches(&alias, View::EMPTY, Counts::rows(8)).is_err());
        let mut short = buffers;
        short.chunk_last = view(4, 4);
        assert!(op.launches(&short, View::EMPTY, Counts::rows(8)).is_err());
        assert!(op.launches(&buffers, View::EMPTY, Counts::rows(9)).is_err());
    }
}
