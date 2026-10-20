//! A selected-key cache around the validated native source and ordered merge.
use super::{Entry, GroupedEntry};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::latent_attention::LatentSourcesBuffers;
pub use kernel_core::latent_sources_cache::Layout;
use std::collections::{BTreeMap, BTreeSet};
pub type Buffers = kernel_core::latent_sources_cache::Buffers<View>;

/// Only listed, GPU-qualified row counts may use this prepared cache.
pub fn prepare(
    reference: Prepared<LatentSourcesBuffers<View>>,
    entry: Entry,
    need: Need,
    rows: &[u32],
) -> Result<Prepared<Buffers>, Refused> {
    prepare_grouped(reference, entry, need, rows, &[])
}

/// Override the cached head group only at explicitly admitted row counts.
/// Other rows retain the original cached entry and every row keeps its merge.
pub fn prepare_grouped(
    reference: Prepared<LatentSourcesBuffers<View>>,
    entry: Entry,
    need: Need,
    rows: &[u32],
    head_groups: &[(u32, u32)],
) -> Result<Prepared<Buffers>, Refused> {
    entry.validate()?;
    need.check(reference.info())?;
    let selected: BTreeSet<_> = rows.iter().copied().collect();
    let groups: BTreeMap<_, _> = head_groups.iter().copied().collect();
    if groups.len() != head_groups.len() || groups.keys().any(|r| !selected.contains(r)) {
        return Err(Refusal("cached head groups require unique admitted rows").into());
    }
    for &heads in groups.values() {
        GroupedEntry {
            cache: entry.clone(),
            heads,
        }
        .validate()?;
    }
    if entry.profile
        || selected.is_empty()
        || selected.len() != rows.len()
        || selected.iter().any(|&r| r == 0 || r > need.capacity)
        || reference.info().workspace != 0
        || reference.info().symbols
            != vec![
                (entry.source.symbol(), entry.source.dynamic_shared_bytes()),
                (entry.source.merge_symbol(), 0),
            ]
    {
        return Err(Refusal("selected cache requires its exact source and admitted rows").into());
    }
    let maximum = *selected.last().unwrap();
    let (decoded, valid) = entry.cache_bytes(maximum)?;
    let mut info = reference.info().clone();
    for (name, _) in &mut info.bytes {
        *name = format!("native.{name}");
    }
    info.bytes
        .extend([("decoded".into(), decoded), ("valid".into(), valid)]);
    info.symbols = vec![
        (entry.decode_symbol(false), 0),
        (entry.symbol(), entry.plan(maximum)?.dynamic_shared_bytes),
        (entry.source.merge_symbol(), 0),
    ];
    for heads in groups.values().copied().collect::<BTreeSet<_>>() {
        let grouped = GroupedEntry {
            cache: entry.clone(),
            heads,
        };
        info.symbols.push((
            grouped.symbol(),
            grouped.plan(maximum)?.dynamic_shared_bytes,
        ));
    }
    let identity = if groups.is_empty() {
        format!("{entry:?}/{selected:?}")
    } else {
        format!("{entry:?}/{selected:?}/{groups:?}")
    };
    let digest = kernel_build::digest(identity.as_bytes());
    info.implementation
        .push_str(&format!(":selected-cache#{}", &digest[..16]));
    need.check(&info)?;
    Ok(Box::new(Cached {
        reference,
        entry,
        selected,
        groups,
        info,
    }))
}

struct Cached {
    reference: Prepared<LatentSourcesBuffers<View>>,
    entry: Entry,
    selected: BTreeSet<u32>,
    groups: BTreeMap<u32, u32>,
    info: OpInfo,
}

impl PreparedOp<Buffers> for Cached {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        b: &Buffers,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if !self.selected.contains(&counts.rows) {
            return Err(Refusal("selected cache row count was not qualified"));
        }
        b.decoded.covers(self.info.bytes_of("decoded").unwrap())?;
        b.valid.covers(self.info.bytes_of("valid").unwrap())?;
        let old = self.reference.launches(&b.native, workspace, counts)?;
        if old.len() != 2
            || old[1].symbol != self.entry.source.merge_symbol()
            || old[1].plan != self.entry.source.merge_plan(counts.rows)?
        {
            return Err(Refusal(
                "selected cache requires the original ordered merge",
            ));
        }
        let mut calls = if let Some(&heads) = self.groups.get(&counts.rows) {
            GroupedEntry {
                cache: self.entry.clone(),
                heads,
            }
            .launches(&old[0], b.decoded, b.valid, None)?
        } else {
            self.entry.launches(&old[0], b.decoded, b.valid, None)?
        };
        calls.push(old[1].clone());
        Ok(calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Sm120;
    use crate::latent::sources::{self, Schedule};
    use kernel_core::backend::Backend;
    use kernel_core::contract::AttentionPhase;
    use kernel_core::latent_attention::{
        LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat, LatentWindowBuffers,
        LatentWindowStorage,
    };

    fn entry() -> Entry {
        let profile = crate::Profile::synthetic(crate::ProfileNames {
            name: "selected-cache",
            tag: "selected_cache",
        });
        let query = LatentAttentionQuery::Attention {
            heads: 16,
            latent: 512,
            max_keys: 128,
            softmax_scale_bits: (512_f32).powf(-0.5).to_bits(),
            page_size: 64,
            kv: LatentKvFormat::Fp8E4M3Ue8M0Block32,
            phase: AttentionPhase::Decode,
            output: LatentAttentionOutput::Normalized,
            sink: true,
            tail_rope_dims: 64,
            second_fp4_keys: false,
            capacity: 32,
        };
        let mut source = sources::entries(profile, query).unwrap().remove(1);
        source.schedule = Schedule::IndependentHeads;
        Entry {
            source,
            profile: false,
        }
    }
    fn reference(e: &Entry) -> Prepared<LatentSourcesBuffers<View>> {
        Sm120::new()
            .with_latent_sources(e.source.clone())
            .latent_attention_sources(&e.source.query, Need::rows(32))
            .unwrap()
    }
    fn buffers(e: &Entry) -> Buffers {
        let v = |handle, bytes| View { handle, bytes };
        Buffers {
            native: LatentSourcesBuffers {
                q: v(1, 32 * 16 * 512 * 2),
                row_batch: v(2, 32 * 4),
                window: LatentWindowBuffers {
                    codes: v(3, 128 * 512),
                    scales: v(4, 128 * 16),
                    item_slots: v(5, 4),
                    items: 1,
                    sequences: 1,
                    code_stride: 128 * 512,
                    scale_stride: 128 * 16,
                    storage: LatentWindowStorage::Ring128,
                    key_lists: v(6, 32 * 128 * 4),
                    key_stride: 128,
                    budget: 128,
                },
                compressed: None,
                sinks: Some(v(7, 16 * 4)),
                cos_sin: Some(v(8, 32 * 64 * 4)),
                output: Some(v(9, 32 * 16 * 512 * 2)),
                partial: None,
                workspace: v(10, e.source.workspace_bytes(32)),
                live_rows: v(11, 4),
            },
            decoded: v(20, e.cache_bytes(8).unwrap().0),
            valid: v(21, e.cache_bytes(8).unwrap().1),
        }
    }

    #[test]
    fn prepared_cache_preserves_the_reference_merge_and_opt_in() {
        let e = entry();
        let b = buffers(&e);
        let old_op = reference(&e);
        let op = prepare(reference(&e), e.clone(), Need::rows(32), &[1, 8]).unwrap();
        assert_eq!(op.info().symbols[1], (e.symbol(), 73984));
        for rows in [1, 8] {
            let old = old_op
                .launches(&b.native, View::EMPTY, Counts::rows(rows))
                .unwrap();
            let new = op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
            assert_eq!(new.len(), 3);
            assert_eq!(new[2], old[1]);
            assert_eq!(new[0].plan, e.decode_plan(rows).unwrap());
            assert_eq!(new[1].plan, e.plan(rows).unwrap());
            assert_eq!(
                &new[1].arguments.0[..old[0].arguments.0.len()],
                old[0].arguments.0.as_slice()
            );
        }
        for rows in [0, 2, 6, 16, 32] {
            assert!(op.launches(&b, View::EMPTY, Counts::rows(rows)).is_err());
        }
    }

    #[test]
    fn prepared_groups_change_only_the_selected_consumer_grid_and_symbol() {
        let e = entry();
        let b = buffers(&e);
        let baseline = prepare(reference(&e), e.clone(), Need::rows(32), &[1, 4, 8]).unwrap();
        let grouped = GroupedEntry {
            cache: e.clone(),
            heads: 8,
        };
        let candidate = prepare_grouped(
            reference(&e),
            e.clone(),
            Need::rows(32),
            &[1, 4, 8],
            &[(4, 8)],
        )
        .unwrap();
        assert_eq!(candidate.info().bytes, baseline.info().bytes);
        assert_eq!(
            &candidate.info().symbols[..3],
            baseline.info().symbols.as_slice()
        );
        assert_eq!(candidate.info().symbols[3], (grouped.symbol(), 73984));
        for rows in [1, 4, 8] {
            let mut old = baseline
                .launches(&b, View::EMPTY, Counts::rows(rows))
                .unwrap();
            let new = candidate
                .launches(&b, View::EMPTY, Counts::rows(rows))
                .unwrap();
            if rows == 4 {
                old[1].symbol = grouped.symbol();
                old[1].plan = grouped.plan(rows).unwrap();
                assert_eq!(new[1].plan.grid, [4, 2, 8]);
            }
            assert_eq!(new, old);
        }
        for groups in [&[(4, 0)][..], &[(4, 3)], &[(2, 8)], &[(4, 4), (4, 8)]] {
            assert!(
                prepare_grouped(reference(&e), e.clone(), Need::rows(32), &[1, 4, 8], groups)
                    .is_err()
            );
        }
        assert!(
            candidate
                .launches(&b, View::EMPTY, Counts::rows(2))
                .is_err()
        );
    }

    #[test]
    fn prepared_cache_retains_native_bounds_and_rejects_cache_aliases() {
        let e = entry();
        let op = prepare(reference(&e), e.clone(), Need::rows(32), &[1, 8]).unwrap();
        for bad in 0..5 {
            let mut b = buffers(&e);
            match bad {
                0 => b.decoded.bytes -= 1,
                1 => b.valid.bytes -= 1,
                2 => b.valid.handle = b.decoded.handle,
                3 => b.decoded.handle = b.native.q.handle,
                _ => b.native.q.bytes = 0,
            }
            assert!(
                op.launches(&b, View::EMPTY, Counts::rows(8)).is_err(),
                "control {bad}"
            );
        }
        for rows in [&[][..], &[1, 1], &[0], &[33]] {
            assert!(prepare(reference(&e), e.clone(), Need::rows(32), rows).is_err());
        }
    }

    #[test]
    fn cache_regions_are_disjoint_aligned_and_checked() {
        let l = Layout::new(257, 17).unwrap();
        assert_eq!(l.decoded, 0..257);
        assert_eq!(l.valid, 512..529);
        assert_eq!(l.bytes, 529);
        assert!(Layout::new(0, 1).is_err());
        assert!(Layout::new(u64::MAX, 1).is_err());
        assert!(Layout::new(1, u64::MAX).is_err());
    }
}
