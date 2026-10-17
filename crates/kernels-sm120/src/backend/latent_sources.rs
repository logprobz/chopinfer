//! Preparation and binding for native latent sources. Selection uses the
//! shared rules; this module only matches and validates its own family.

use kernel_build::EntryParam;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};
use kernel_core::latent_attention::{
    LATENT_WINDOW_SLOTS, LatentAttentionQuery, LatentKvFormat, LatentSourcesBuffers,
    LatentWindowStorage,
};

use super::select::{Candidate, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::latent::sources::{Entry, Shape, shape};

fn bytes(factors: &[u64]) -> Result<u64, Refusal> {
    factors
        .iter()
        .try_fold(1_u64, |a, &b| a.checked_mul(b))
        .ok_or(Refusal("native latent byte count overflows"))
}
fn buffer(view: View) -> Argument {
    Argument::DeviceBuffer(view.handle)
}

fn launch_key(entry: &Entry) -> String {
    format!("{:?} {:?}", entry.query, entry.entry_specs())
}

pub(super) fn prepare(
    entries: &[Entry],
    query: &LatentAttentionQuery,
    need: Need,
) -> Result<Prepared<LatentSourcesBuffers<View>>, Refused> {
    let s = shape(*query)?;
    if need.capacity == 0 || need.capacity > MAX_CAPACITY {
        return Err(Refused::Rule(
            "native latent capacity must be in 1..=MAX_CAPACITY".into(),
        ));
    }
    let selected = choose(
        "latent sources",
        entries.iter().filter(|e| e.query == *query).map(|e| {
            Candidate::new(e, e.symbol(), launch_key(e), e.invariance()).capacity(e.capacity())
        }),
        need,
    )?;
    let entry = selected.entry.clone();
    entry.contract()?;
    let mut plans = Vec::new();
    for rows in 1..=need.capacity {
        let pair = [entry.plan(rows)?, entry.merge_plan(rows)?];
        for plan in &pair {
            check_plan(plan)?;
        }
        plans.push(pair);
    }
    let digest = kernel_build::digest(launch_key(&entry).as_bytes());
    let mut info = OpInfo::new(
        format!(
            "sm120.latent-sources.v1[{}]#{}",
            entry.symbol(),
            &digest[..16]
        ),
        selected.guarantees,
        vec![
            (entry.symbol(), entry.dynamic_shared_bytes()),
            (entry.merge_symbol(), 0),
        ],
    );
    info.bytes
        .push(("workspace".into(), entry.workspace_bytes(need.capacity)));
    need.check(&info)?;
    let params = [entry.params()?, entry.merge_params()?];
    Ok(Box::new(Sources {
        entry,
        shape: s,
        plans,
        params,
        info,
    }))
}

struct Sources {
    entry: Entry,
    shape: Shape,
    plans: Vec<[LaunchPlan; 2]>,
    params: [Vec<EntryParam>; 2],
    info: OpInfo,
}

impl PreparedOp<LatentSourcesBuffers<View>> for Sources {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        b: &LatentSourcesBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        if rows == 0 || rows as usize > self.plans.len() {
            return Err(Refusal("native latent live capacity exceeds preparation"));
        }
        let s = self.shape;
        let r = u64::from(rows);
        let width = u64::from(s.latent);
        let elements = bytes(&[r, u64::from(s.heads), width])?;
        b.q.covers(bytes(&[elements, 2])?)?;
        b.row_batch.covers(bytes(&[r, 4])?)?;
        b.live_rows.covers(4)?;
        b.workspace.covers(self.entry.workspace_bytes(rows))?;
        let slots = match b.window.storage {
            LatentWindowStorage::Ring128 => LATENT_WINDOW_SLOTS,
            LatentWindowStorage::Staged { slots } => slots,
        };
        let window_bytes = LatentKvFormat::Fp8E4M3Ue8M0Block32.page_bytes(s.latent, slots);
        if slots == 0
            || b.window.sequences == 0
            || b.window.sequence_stride < window_bytes
            || b.window.budget > b.window.key_stride
        {
            return Err(Refusal("native latent window geometry or budget mismatch"));
        }
        let last = bytes(&[u64::from(b.window.sequences - 1), b.window.sequence_stride])?;
        b.window.pool.covers(
            last.checked_add(window_bytes)
                .ok_or(Refusal("native latent window bytes overflow"))?,
        )?;
        b.window
            .key_lists
            .covers(bytes(&[r, u64::from(b.window.key_stride), 4])?)?;
        if s.second != b.compressed.is_some() {
            return Err(Refusal(
                "native latent compressed source presence differs from query",
            ));
        }
        let compressed_budget = b.compressed.map_or(0, |c| c.budget);
        if b.window
            .budget
            .checked_add(compressed_budget)
            .is_none_or(|n| n > s.max_keys)
        {
            return Err(Refusal("native latent combined key budget exceeds query"));
        }
        if s.sink != b.sinks.is_some() {
            return Err(Refusal("native latent sink presence differs from query"));
        }
        if let Some(sinks) = b.sinks {
            sinks.covers(u64::from(s.heads) * 4)?;
        }
        let rotate = s.tail != 0 && !s.partial;
        if rotate != b.cos_sin.is_some() {
            return Err(Refusal(
                "native latent inverse rotation table presence differs from output",
            ));
        }
        if let Some(table) = b.cos_sin {
            table.covers(bytes(&[r, u64::from(s.tail), 4])?)?;
        }
        let mut bound = vec![
            ("q", buffer(b.q)),
            ("row_batch", buffer(b.row_batch)),
            ("live_rows", buffer(b.live_rows)),
            ("window_pool", buffer(b.window.pool)),
            ("window_keys", buffer(b.window.key_lists)),
            ("workspace", buffer(b.workspace)),
            ("window_stride", Argument::U64(b.window.sequence_stride)),
            ("window_slots", Argument::U32(slots)),
            ("window_sequences", Argument::U32(b.window.sequences)),
            ("window_key_stride", Argument::U32(b.window.key_stride)),
            ("window_budget", Argument::U32(b.window.budget)),
            ("scale_bits", Argument::U32(s.scale_bits)),
        ];
        if let Some(c) = b.compressed {
            c.block_tables.check()?;
            let page_bytes = LatentKvFormat::Fp4E2M1E4M3Block16.page_bytes(s.latent, s.page);
            if c.cache.pages == 0
                || c.cache.page_stride < page_bytes
                || c.budget > c.key_stride
                || c.block_tables.sequences != b.window.sequences
            {
                return Err(Refusal(
                    "native latent compressed geometry or budget mismatch",
                ));
            }
            let last = bytes(&[u64::from(c.cache.pages - 1), c.cache.page_stride])?;
            c.cache.pool.covers(
                last.checked_add(page_bytes)
                    .ok_or(Refusal("native latent compressed bytes overflow"))?,
            )?;
            c.key_lists
                .covers(bytes(&[r, u64::from(c.key_stride), 4])?)?;
            bound.extend([
                ("compressed_pool", buffer(c.cache.pool)),
                ("compressed_keys", buffer(c.key_lists)),
                ("block_tables", buffer(c.block_tables.tables)),
                ("page_stride", Argument::U64(c.cache.page_stride)),
                ("max_pages", Argument::U32(c.block_tables.max_pages)),
                ("compressed_pages", Argument::U32(c.cache.pages)),
                (
                    "compressed_sequences",
                    Argument::U32(c.block_tables.sequences),
                ),
                ("compressed_key_stride", Argument::U32(c.key_stride)),
                ("compressed_budget", Argument::U32(c.budget)),
            ]);
        }
        match (s.partial, b.output, b.partial) {
            (false, Some(out), None) => {
                out.covers(bytes(&[elements, 2])?)?;
                bound.push(("out", buffer(out)));
                if let Some(sinks) = b.sinks {
                    bound.push(("sinks", buffer(sinks)));
                }
                if let Some(table) = b.cos_sin {
                    bound.push(("cos_sin", buffer(table)));
                }
            }
            (true, None, Some(part)) => {
                part.max.covers(bytes(&[r, u64::from(s.heads), 4])?)?;
                part.den.covers(bytes(&[r, u64::from(s.heads), 4])?)?;
                part.num.covers(bytes(&[elements, 4])?)?;
                bound.extend([
                    ("out_max", buffer(part.max)),
                    ("out_den", buffer(part.den)),
                    ("out_num", buffer(part.num)),
                ]);
            }
            _ => return Err(Refusal("native latent output roles differ from query")),
        }
        let symbols = [self.entry.symbol(), self.entry.merge_symbol()];
        self.params
            .iter()
            .enumerate()
            .map(|(i, params)| {
                let args = params
                    .iter()
                    .map(|p| {
                        let name = match p {
                            EntryParam::DevicePointer { name, .. }
                            | EntryParam::U32 { name }
                            | EntryParam::U64 { name } => name,
                            _ => {
                                return Err(Refusal("native latent parameter kind is unsupported"));
                            }
                        };
                        bound
                            .iter()
                            .find(|(role, _)| *role == name)
                            .map(|(_, a)| a.clone())
                            .ok_or(Refusal("native latent parameter has no bound role"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let arguments = ArgumentList(args);
                arguments.validate()?;
                Ok(Launch {
                    symbol: symbols[i].clone(),
                    rows,
                    plan: self.plans[rows as usize - 1][i],
                    arguments,
                    collective: false,
                    persistent: false,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Sm120;
    use crate::{Profile, ProfileNames};
    use kernel_core::backend::Backend;
    use kernel_core::contract::AttentionPhase;
    use kernel_core::latent_attention::{
        BlockTables, LatentAttentionOutput, LatentCompressedBuffers, LatentKvCache, LatentPartial,
        LatentWindowBuffers,
    };

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "latent-sources-test",
        tag: "latent_sources_test",
    });
    fn entry() -> Entry {
        Entry {
            profile: PROFILE,
            name: "test".into(),
            splits: 8,
            query: LatentAttentionQuery::Attention {
                heads: 4,
                latent: 128,
                max_keys: 160,
                softmax_scale_bits: (128_f32).powf(-0.5).to_bits(),
                page_size: 16,
                kv: LatentKvFormat::Fp8E4M3Ue8M0Block32,
                phase: AttentionPhase::Verify,
                output: LatentAttentionOutput::Normalized,
                sink: true,
                tail_rope_dims: 64,
                second_fp4_keys: true,
                capacity: 8,
            },
        }
    }
    fn view(handle: u64, bytes: u64) -> View {
        View { handle, bytes }
    }
    fn buffers() -> LatentSourcesBuffers<View> {
        let e = entry();
        let stride = LatentKvFormat::Fp8E4M3Ue8M0Block32.page_bytes(128, 128);
        let page = LatentKvFormat::Fp4E2M1E4M3Block16.page_bytes(128, 16);
        LatentSourcesBuffers {
            q: view(1, 8 * 4 * 128 * 2),
            row_batch: view(2, 8 * 4),
            window: LatentWindowBuffers {
                pool: view(3, stride * 2),
                sequences: 2,
                sequence_stride: stride,
                storage: LatentWindowStorage::Ring128,
                key_lists: view(4, 8 * 128 * 4),
                key_stride: 128,
                budget: 128,
            },
            compressed: Some(LatentCompressedBuffers {
                cache: LatentKvCache {
                    pool: view(5, 4 * page),
                    pages: 4,
                    page_stride: page,
                },
                block_tables: BlockTables {
                    tables: view(6, 2 * 4 * 4),
                    sequences: 2,
                    max_pages: 4,
                },
                key_lists: view(7, 8 * 32 * 4),
                key_stride: 32,
                budget: 32,
            }),
            sinks: Some(view(8, 4 * 4)),
            cos_sin: Some(view(9, 8 * 64 * 4)),
            output: Some(view(10, 8 * 4 * 128 * 2)),
            partial: None,
            workspace: view(11, e.workspace_bytes(8)),
            live_rows: view(12, 4),
        }
    }

    #[test]
    fn native_sources_prepare_and_bind_both_launches() {
        let e = entry();
        let backend = Sm120::new().with_latent_sources(e.clone());
        let op = backend
            .latent_attention_sources(&e.query, Need::rows(8).with_exact(e.invariance()))
            .unwrap();
        assert_eq!(op.info().bytes_of("workspace"), Some(e.workspace_bytes(8)));
        for rows in [1, 3, 8] {
            let launches = op
                .launches(&buffers(), View::default(), Counts::rows(rows))
                .unwrap();
            assert_eq!(launches.len(), 2);
            assert_eq!(launches[0].plan, e.plan(rows).unwrap());
            assert_eq!(launches[1].plan, e.merge_plan(rows).unwrap());
            assert_eq!(launches[0].symbol, e.symbol());
            assert_eq!(launches[1].symbol, e.merge_symbol());
            assert!(launches.iter().all(|l| l.plan.pdl));
            assert!(!launches[0].arguments.0.contains(&Argument::DeviceBuffer(8)));
            assert!(launches[1].arguments.0.contains(&Argument::DeviceBuffer(8)));
            assert!(launches[1].arguments.0.contains(&Argument::DeviceBuffer(9)));
        }
    }

    #[test]
    fn native_sources_validate_each_buffer_budget_and_output_role() {
        let e = entry();
        let backend = Sm120::new().with_latent_sources(e.clone());
        let op = backend
            .latent_attention_sources(&e.query, Need::rows(8))
            .unwrap();
        let base = buffers();
        let mut bad = Vec::new();
        let mut push = |f: fn(&mut LatentSourcesBuffers<View>)| {
            let mut b = base;
            f(&mut b);
            bad.push(b);
        };
        push(|b| b.q.bytes -= 1);
        push(|b| b.row_batch.bytes -= 1);
        push(|b| b.window.pool.bytes -= 1);
        push(|b| b.window.sequence_stride -= 1);
        push(|b| b.window.sequences = 0);
        push(|b| b.window.key_lists.bytes -= 1);
        push(|b| b.window.budget += 1);
        push(|b| b.window.storage = LatentWindowStorage::Staged { slots: 0 });
        push(|b| b.compressed.as_mut().unwrap().cache.pool.bytes -= 1);
        push(|b| b.compressed.as_mut().unwrap().cache.pages = 0);
        push(|b| b.compressed.as_mut().unwrap().cache.page_stride -= 1);
        push(|b| b.compressed.as_mut().unwrap().key_lists.bytes -= 1);
        push(|b| b.compressed.as_mut().unwrap().budget += 1);
        push(|b| b.compressed.as_mut().unwrap().block_tables.tables.bytes -= 1);
        push(|b| b.compressed.as_mut().unwrap().block_tables.sequences = 1);
        push(|b| b.compressed = None);
        push(|b| b.sinks = None);
        push(|b| b.sinks.as_mut().unwrap().bytes -= 1);
        push(|b| b.cos_sin = None);
        push(|b| b.cos_sin.as_mut().unwrap().bytes -= 1);
        push(|b| b.output = None);
        push(|b| b.output.as_mut().unwrap().bytes -= 1);
        push(|b| b.workspace.bytes -= 1);
        push(|b| b.live_rows.bytes -= 1);
        for (i, b) in bad.iter().enumerate() {
            assert!(
                op.launches(b, View::default(), Counts::rows(8)).is_err(),
                "mutation {i}"
            );
        }
        for rows in [0, 9] {
            assert!(
                op.launches(&base, View::default(), Counts::rows(rows))
                    .is_err()
            );
        }
        let mut overflow = base;
        overflow.window.sequences = u32::MAX;
        overflow.window.sequence_stride = u64::MAX;
        overflow.window.pool.bytes = u64::MAX;
        assert!(
            op.launches(&overflow, View::default(), Counts::rows(8))
                .is_err()
        );
    }

    #[test]
    fn native_sources_partial_merge_never_binds_sink_or_rotation() {
        let mut e = entry();
        if let LatentAttentionQuery::Attention { output, .. } = &mut e.query {
            *output = LatentAttentionOutput::Partial;
        }
        let backend = Sm120::new().with_latent_sources(e.clone());
        let op = backend
            .latent_attention_sources(&e.query, Need::rows(8))
            .unwrap();
        let mut b = buffers();
        b.output = None;
        b.cos_sin = None;
        b.partial = Some(LatentPartial {
            max: view(13, 8 * 4 * 4),
            den: view(14, 8 * 4 * 4),
            num: view(15, 8 * 4 * 128 * 4),
        });
        let launches = op.launches(&b, View::default(), Counts::rows(8)).unwrap();
        assert_eq!(launches.len(), 2);
        assert!(
            launches
                .iter()
                .all(|l| !l.arguments.0.contains(&Argument::DeviceBuffer(8)))
        );
        assert!(
            launches[1]
                .arguments
                .0
                .contains(&Argument::DeviceBuffer(15))
        );
    }

    #[test]
    fn native_sources_selection_refuses_ambiguity_and_is_order_independent() {
        let e = entry();
        let other = Entry {
            name: "other".into(),
            ..e.clone()
        };
        assert!(
            Sm120::new()
                .latent_attention_sources(&e.query, Need::rows(1))
                .is_err()
        );
        for entries in [[e.clone(), other.clone()], [other.clone(), e.clone()]] {
            let backend = entries
                .into_iter()
                .fold(Sm120::new(), Sm120::with_latent_sources);
            assert!(
                backend
                    .latent_attention_sources(&e.query, Need::rows(1))
                    .is_err()
            );
        }
        let once = Sm120::new().with_latent_sources(e.clone());
        let twice = Sm120::new()
            .with_latent_sources(e.clone())
            .with_latent_sources(e.clone());
        assert_eq!(
            once.latent_attention_sources(&e.query, Need::rows(1))
                .unwrap()
                .info()
                .implementation,
            twice
                .latent_attention_sources(&e.query, Need::rows(1))
                .unwrap()
                .info()
                .implementation
        );
        assert!(
            once.latent_attention_sources(&e.query, Need::rows(9))
                .is_err()
        );
    }
}
