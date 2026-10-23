//! The draft noise (`kernel_core::draft_noise::DraftNoise`) through the
//! draft noise entries a model's build compiles (`row::draft_noise`). An op
//! claims what its entry's declaration claims: [`row::draft_noise::INVARIANCE`]
//! for the shared symbol, nothing for an undeclared one. Entries of
//! different symbols are a tie.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, Refusal, Validate};
use kernel_core::draft_noise::{DraftNoise, DraftNoiseBuffers, NOISE_BYTES};

use super::select::{Candidate, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::row::draft_noise::{self as row, Entry};

/// Rank `rank`'s draft noise `op` through a registered entry.
pub(super) fn draft_noise(
    entries: &[Entry],
    op: &DraftNoise,
    rank: u32,
    need: Need,
) -> Result<Prepared<DraftNoiseBuffers<View>>, Refused> {
    op.validate()?;
    if rank >= op.ranks {
        return Err(Refused::Rule(
            "the rank is past the draft noise's ranks".into(),
        ));
    }
    if need.capacity == 0 || need.capacity > MAX_CAPACITY {
        return Err(Refused::Rule(format!(
            "an SM120 draft noise serves 1 to {MAX_CAPACITY} rows"
        )));
    }
    let chosen = choose(
        "draft noise",
        entries.iter().map(|entry| {
            Candidate::new(
                entry.clone(),
                &entry.symbol,
                entry.symbol.clone(),
                entry.guarantees(),
            )
        }),
        need,
    )?;
    check_plan(&chosen.entry.plan(need.capacity, op.vocab_per_rank)?)?;
    let info = OpInfo::new(
        "sm120.row.draft-noise.f32.v1",
        chosen.guarantees,
        vec![(chosen.entry.symbol.clone(), 0)],
    );
    need.check(&info)?;
    Ok(Box::new(NoiseOp {
        entry: chosen.entry,
        capacity: need.capacity,
        vocab: op.vocab_per_rank,
        vocab_offset: rank * op.vocab_per_rank,
        info,
    }))
}

struct NoiseOp {
    entry: Entry,
    capacity: u32,
    vocab: u32,
    vocab_offset: u32,
    info: OpInfo,
}

impl PreparedOp<DraftNoiseBuffers<View>> for NoiseOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &DraftNoiseBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        if rows == 0 || rows > self.capacity {
            return Err(Refusal("row count outside the prepared capacity"));
        }
        let rows64 = u64::from(rows);
        buffers.logits.covers(
            rows64
                .checked_mul(u64::from(self.vocab) * 4)
                .ok_or(Refusal("draft noise logits overflow"))?,
        )?;
        buffers.noise.covers(rows64 * NOISE_BYTES as u64)?;
        buffers.live_rows.covers(4)?;
        buffers.live_width.covers(4)?;
        let arguments = row::arguments(
            [
                buffers.logits,
                buffers.noise,
                buffers.live_rows,
                buffers.live_width,
            ]
            .map(|view| Argument::DeviceBuffer(view.handle)),
            self.vocab,
            self.vocab_offset,
            buffers.position,
        )?;
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows,
            plan: self.entry.plan(rows, self.vocab)?,
            arguments,
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use kernel_core::backend::Backend;
    use kernel_core::properties::Invariance;

    use super::*;
    use crate::backend::{Sm120, view};

    const OP: DraftNoise = DraftNoise {
        vocab_per_rank: 32_320,
        ranks: 4,
    };

    fn buffers(rows: u64, position: u32) -> DraftNoiseBuffers<View> {
        DraftNoiseBuffers {
            logits: view(1, rows * 32_320 * 4),
            live_width: view(2, 4),
            noise: view(3, rows * 32),
            live_rows: view(4, 4),
            position,
        }
    }

    #[test]
    fn the_shared_entry_perturbs_its_ranks_shard_at_the_position() {
        let backend = Sm120::new().with_draft_noise(Entry::new(row::SYMBOL));
        let need = Need::rows(32).with_exact(Invariance::any_rows(32));
        let prepared = backend.draft_noise(&OP, 2, need).unwrap();
        assert_eq!(prepared.info().symbols, [(row::SYMBOL.to_owned(), 0)]);
        let [launch] = &prepared
            .launches(&buffers(5, 3), View::EMPTY, Counts::rows(5))
            .unwrap()[..]
        else {
            panic!("one launch");
        };
        assert_eq!(
            (launch.rows, launch.plan.grid, launch.plan.block),
            (5, [5, 16, 1], [row::THREADS, 1, 1])
        );
        assert_eq!(
            launch.arguments.0,
            [
                Argument::DeviceBuffer(1),
                Argument::DeviceBuffer(3),
                Argument::DeviceBuffer(4),
                Argument::DeviceBuffer(2),
                Argument::U32(32_320),
                Argument::U32(64_640),
                Argument::U32(3),
            ]
        );
        assert!(
            prepared
                .launches(&buffers(4, 0), View::EMPTY, Counts::rows(5))
                .is_err()
        );
        assert!(
            prepared
                .launches(&buffers(40, 0), View::EMPTY, Counts::rows(33))
                .is_err()
        );
    }

    #[test]
    fn a_backend_refuses_without_an_entry_a_rank_or_a_declaration() {
        let need = Need::rows(8).with_exact(Invariance::any_rows(8));
        assert!(Sm120::new().draft_noise(&OP, 0, need).is_err());
        let shared = Sm120::new().with_draft_noise(Entry::new(row::SYMBOL));
        assert!(shared.draft_noise(&OP, 4, need).is_err());
        assert!(shared.draft_noise(&OP, 0, Need::rows(0)).is_err());
        let undeclared = Sm120::new().with_draft_noise(Entry::new("chop_draft_noise_other_v1"));
        assert!(matches!(
            undeclared.draft_noise(&OP, 0, need),
            Err(Refused::Inexact { .. })
        ));
        assert!(undeclared.draft_noise(&OP, 0, Need::rows(8)).is_ok());
    }
}
