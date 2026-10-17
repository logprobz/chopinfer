//! Row quantization through registered typed entries, with fixed scale rules.

use super::select::{Candidate, choose};
use crate::row::block_quantize::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{DType, Refusal, Validate};
use kernel_core::precision::quantizer::{BlockQuantizeBuffers, BlockQuantizeQuery, QuantizeOutput};

fn key(entry: &Entry) -> String {
    format!("{:?}", entry.spec())
}

pub(super) fn prepare(
    entries: &[Entry],
    query: &BlockQuantizeQuery,
    need: Need,
) -> Result<Prepared<BlockQuantizeBuffers<View>>, Refused> {
    query.validate()?;
    if need.capacity == 0 || need.capacity > super::MAX_CAPACITY {
        return Err(Refusal("block quantization preparation capacity exceeds SM120 bounds").into());
    }
    let chosen = choose(
        "block quantize",
        entries.iter().filter(|e| e.query == *query).map(|entry| {
            Candidate::new(entry.clone(), &entry.symbol, key(entry), entry.invariance())
                .capacity(entry.capacity)
        }),
        need,
    )?;
    let entry = chosen.entry;
    entry.validate()?;
    let mut info = OpInfo::new(
        format!("sm120.block-quantize.v1:{query:?}"),
        chosen
            .guarantees
            .meet(kernel_core::properties::Invariance::any_rows(need.capacity)),
        vec![(entry.symbol.clone(), 0)],
    );
    info.bytes = vec![
        ("input".into(), entry.input_bytes(need.capacity)),
        ("output".into(), entry.output_bytes(need.capacity)),
        ("live_rows".into(), 4),
    ];
    if query.output == QuantizeOutput::Codes {
        info.bytes
            .push(("scales".into(), entry.scale_bytes(need.capacity)));
    }
    need.check(&info)?;
    Ok(Box::new(Op {
        entry,
        capacity: need.capacity,
        info,
    }))
}

struct Op {
    entry: Entry,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<BlockQuantizeBuffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &BlockQuantizeBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows > self.capacity {
            return Err(Refusal(
                "block quantization launch exceeds prepared capacity",
            ));
        }
        let entry = &self.entry;
        buffers.input.covers(entry.input_bytes(counts.rows))?;
        buffers.output.covers(entry.output_bytes(counts.rows))?;
        buffers.live_rows.covers(4)?;
        match (entry.query.output, buffers.scales) {
            (QuantizeOutput::Codes, Some(scales)) => {
                scales.covers(entry.scale_bytes(counts.rows))?
            }
            (QuantizeOutput::Bf16, None) => {}
            _ => {
                return Err(Refusal(
                    "block quantization scale role differs from the output form",
                ));
            }
        }
        if (buffers.input.handle == buffers.output.handle
            && !(entry.query.input == DType::Bf16 && entry.query.output == QuantizeOutput::Bf16))
            || buffers.output.handle == buffers.live_rows.handle
            || buffers.scales.is_some_and(|s| {
                [
                    buffers.input.handle,
                    buffers.output.handle,
                    buffers.live_rows.handle,
                ]
                .contains(&s.handle)
            })
        {
            return Err(Refusal(
                "block quantization buffers overlap incompatible roles",
            ));
        }
        Ok(vec![Launch {
            symbol: entry.symbol.clone(),
            rows: counts.rows,
            plan: entry.plan(counts.rows)?,
            arguments: entry.arguments(
                buffers.input.handle,
                buffers.output.handle,
                buffers.scales.map(|v| v.handle),
                buffers.live_rows.handle,
            )?,
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Sm120;
    use kernel_core::backend::Backend;
    use kernel_core::precision::quantizer::BlockQuantization;

    fn entry(capacity: u32, symbol: &str) -> Entry {
        Entry::new(
            BlockQuantizeQuery {
                width: 576,
                input: DType::F32,
                format: BlockQuantization::E4M3_CEIL32,
                output: QuantizeOutput::Codes,
            },
            capacity,
            symbol,
        )
        .unwrap()
    }
    fn buffers() -> BlockQuantizeBuffers<View> {
        let v = |handle| View {
            handle,
            bytes: 16 * 576 * 4,
        };
        BlockQuantizeBuffers {
            input: v(1),
            output: v(2),
            scales: Some(v(3)),
            live_rows: v(4),
        }
    }
    #[test]
    fn prepares_complete_ceil_and_fake_quant_contracts_and_checks_roles() {
        for entry in crate::row::block_quantize::lab_entries() {
            let query = entry.query;
            let backend = Sm120::new().with_block_quantize(entry);
            let op = backend.block_quantize(&query, Need::rows(8)).unwrap();
            let mut buffers = buffers();
            if query.output == QuantizeOutput::Bf16 {
                buffers.scales = None;
            }
            for rows in [0, 1, 7, 8] {
                let launches = op
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                assert_eq!(launches.len(), 1);
                assert!(launches[0].plan.pdl);
                assert_eq!(
                    launches[0].plan.grid[1],
                    query.width / query.format.group_size()
                );
            }
            assert!(op.launches(&buffers, View::EMPTY, Counts::rows(9)).is_err());
            buffers.output.bytes = 1;
            assert!(op.launches(&buffers, View::EMPTY, Counts::rows(8)).is_err());
        }
    }
    #[test]
    fn selection_is_order_independent_and_refuses_another_scale_rule() {
        let small = entry(8, "small");
        let large = entry(16, "large");
        for entries in [
            vec![small.clone(), large.clone()],
            vec![large.clone(), small.clone()],
        ] {
            let backend = entries
                .into_iter()
                .fold(Sm120::new(), Sm120::with_block_quantize);
            assert_eq!(
                backend
                    .block_quantize(&small.query, Need::rows(8))
                    .unwrap()
                    .info()
                    .symbols[0]
                    .0,
                "small"
            );
            let mut q = small.query;
            q.format = BlockQuantization::E4M3Ceil32 {
                amax_floor_bits: 2e-4_f32.to_bits(),
            };
            assert!(backend.block_quantize(&q, Need::rows(8)).is_err());
        }
        let tied = Sm120::new()
            .with_block_quantize(small.clone())
            .with_block_quantize(entry(8, "other"));
        assert!(tied.block_quantize(&small.query, Need::rows(8)).is_err());
    }
    #[test]
    fn selection_key_tracks_every_entry_field() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let base = entry(16, "original");
        let query = base.query;
        let fields: Vec<Field<Entry>> = vec![
            (
                "query",
                Class::Selects,
                Box::new(|e| e.query.input = DType::Bf16),
            ),
            // Selection settles the entry capacity; launches use the requested capacity.
            ("capacity", Class::Inert, Box::new(|e| e.capacity = 32)),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "other".into()),
            ),
        ];
        keys_follow_launches(
            &base,
            field_names!(Entry {
                query,
                capacity,
                symbol
            }),
            fields,
            key,
            |entry| {
                let op = Sm120::new()
                    .with_block_quantize(entry.clone())
                    .block_quantize(&query, Need::rows(8))?;
                let launches = [1, 8].map(|rows| {
                    op.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((format!("{:?}", entry.spec()), launches))
            },
        );
    }
}
