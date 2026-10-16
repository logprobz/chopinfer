//! The row copies (`kernel_core::row_copy`) through the copy entries a
//! model's build compiles (`row::copy`): the stream broadcast in 16-byte
//! vectors and the column slice as the launch-shape layout copy over
//! 16-bit words. Both move bytes only; an op claims what its entry's
//! declaration claims (`row::copy::declared_entries` for the shared
//! symbols), and nothing for an undeclared symbol. An entry serves every shape of its pattern,
//! so a pattern's entries of different symbols are a tie. Each op is
//! refused when prepared if its stated view alignment is below what its
//! kernel's accesses need or its footprint at the capacity overflows u64.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{ArgumentList, LaunchPlan, Refusal};
use kernel_core::row_copy::{ColumnSlice, RowCopyBuffers, StreamBroadcast};

use super::select::{Candidate, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::row::copy::{Broadcast, Entry, Pattern, declared_entries, slice};
use crate::row::precision::LayoutCopy;
use kernel_core::properties::Invariance;

/// The stream broadcast `op` through the registered broadcast entry.
pub(super) fn broadcast_streams(
    entries: &[Entry],
    op: &StreamBroadcast,
    need: Need,
) -> Result<Prepared<RowCopyBuffers<View>>, Refused> {
    let (symbol, guarantees) = symbol(entries, Pattern::BroadcastStreams, need)?;
    check_capacity(need.capacity)?;
    let copy = Broadcast::new(*op, need.capacity, &symbol)?;
    check_plan(&copy.plan(need.capacity)?)?;
    let op = BroadcastOp {
        info: OpInfo::new("sm120.row.copy.broadcast.v1", guarantees, vec![(symbol, 0)]),
        copy,
        capacity: need.capacity,
    };
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// The column slice `op` through the registered slice entry.
pub(super) fn column_slice(
    entries: &[Entry],
    op: &ColumnSlice,
    need: Need,
) -> Result<Prepared<RowCopyBuffers<View>>, Refused> {
    let (symbol, guarantees) = symbol(entries, Pattern::ColumnSlice, need)?;
    check_capacity(need.capacity)?;
    let copy = slice(op, need.capacity, &symbol)?;
    let plan = copy.stage().plan(need.capacity)?;
    check_plan(&plan)?;
    let op = SliceOp {
        info: OpInfo::new(
            format!(
                "sm120.row.copy.slice.v1[{}]",
                copy.stage().numerical_contract()
            ),
            guarantees,
            vec![(symbol, 0)],
        ),
        op: *op,
        copy,
        capacity: need.capacity,
        plan,
    };
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// The symbol of the entry of `pattern` that serves the op and what the
/// op may claim through it.
fn symbol(
    entries: &[Entry],
    pattern: Pattern,
    need: Need,
) -> Result<(String, Invariance), Refused> {
    let chosen = choose(
        "row copy",
        entries
            .iter()
            .filter(|entry| entry.pattern == pattern)
            .map(|entry| {
                let guarantees = declared(&entry.symbol);
                Candidate::new(entry.clone(), &entry.symbol, launch(entry), guarantees)
            }),
        need,
    )?;
    Ok((chosen.entry.symbol, chosen.guarantees))
}

/// The guarantees the shared copies' declarations claim for `symbol`; none
/// for an undeclared entry.
fn declared(symbol: &str) -> Invariance {
    declared_entries()
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .map_or(Invariance::NONE, |entry| entry.invariance)
}

/// What a copy entry launches: its symbol, whose code its pattern fixes.
fn launch(entry: &Entry) -> String {
    let Entry {
        pattern: _, // the op selects by it
        symbol,
    } = entry;
    symbol.clone()
}

fn check_capacity(capacity: u32) -> Result<(), Refused> {
    if capacity == 0 || capacity > MAX_CAPACITY {
        return Err(Refused::Rule(format!(
            "an SM120 row copy serves 1 to {MAX_CAPACITY} rows"
        )));
    }
    Ok(())
}

/// The launch's rows, after checking them against the prepared capacity.
fn checked(rows: u32, capacity: u32) -> Result<u32, Refusal> {
    if rows == 0 || rows > capacity {
        return Err(Refusal("row copy rows exceed the prepared capacity"));
    }
    Ok(rows)
}

fn launch_of(symbol: &str, rows: u32, plan: LaunchPlan, arguments: ArgumentList) -> Launch {
    Launch {
        symbol: symbol.into(),
        rows,
        plan,
        arguments,
        collective: false,
        persistent: false,
    }
}

/// `chop::row::copy::broadcast_streams`: one CTA per row of the launch.
struct BroadcastOp {
    copy: Broadcast,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<RowCopyBuffers<View>> for BroadcastOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &RowCopyBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = checked(counts.rows, self.capacity)?;
        let op = self.copy.op();
        buffers.source.covers(op.source_bytes(rows)?)?;
        buffers.destination.covers(op.destination_bytes(rows)?)?;
        buffers.live_rows.covers(4)?;
        let arguments = self.copy.arguments(
            buffers.source.handle,
            buffers.destination.handle,
            buffers.live_rows.handle,
        )?;
        Ok(vec![launch_of(
            self.copy.symbol(),
            rows,
            self.copy.plan(rows)?,
            arguments,
        )])
    }
}

/// The launch-shape layout copy over the slice's words: one CTA per row of
/// the launch, as the copy built at the launch's rows plans it.
struct SliceOp {
    op: ColumnSlice,
    copy: LayoutCopy,
    capacity: u32,
    /// The plan at the prepared capacity; a launch grids its own rows.
    plan: LaunchPlan,
    info: OpInfo,
}

impl PreparedOp<RowCopyBuffers<View>> for SliceOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &RowCopyBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = checked(counts.rows, self.capacity)?;
        buffers.source.covers(self.op.source_bytes(rows)?)?;
        buffers
            .destination
            .covers(self.op.destination_bytes(rows)?)?;
        buffers.live_rows.covers(4)?;
        let mut plan = self.plan;
        plan.grid[0] = rows;
        let arguments = self.copy.arguments(
            buffers.source.handle,
            buffers.destination.handle,
            buffers.live_rows.handle,
        )?;
        Ok(vec![launch_of(
            &self.copy.stage().entry().symbol,
            rows,
            plan,
            arguments,
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::select::orders;
    use crate::backend::{Sm120, view};
    use crate::row::copy::{BROADCAST_SYMBOL, INVARIANCE, SLICE_SYMBOL, entries};
    use kernel_core::backend::Backend;
    use kernel_core::contract::DType;
    use kernel_core::properties::{Invariance, Ties};

    fn shared() -> Sm120 {
        entries()
            .into_iter()
            .fold(Sm120::new(), |backend, entry| backend.with_row_copy(entry))
    }

    /// The broadcasts the models serve: BF16 rows of 2560 and 4096 into four
    /// streams.
    fn broadcasts() -> [StreamBroadcast; 2] {
        [2560, 4096].map(|width| StreamBroadcast {
            dtype: DType::Bf16,
            width,
            streams: 4,
            alignment_bytes: 16,
        })
    }

    /// The slices the models serve: the router's 288 of 320 F32 scores, and
    /// each shard of a 4096-column BF16 row at 2048 and 4096 columns.
    fn slices() -> Vec<ColumnSlice> {
        let mut slices = vec![ColumnSlice {
            dtype: DType::F32,
            width: 288,
            stride: 320,
            offset: 0,
            alignment_bytes: 16,
        }];
        for width in [2048, 4096] {
            for shard in 0..4096 / width {
                slices.push(ColumnSlice {
                    dtype: DType::Bf16,
                    width,
                    stride: 4096,
                    offset: shard * width,
                    alignment_bytes: 16,
                });
            }
        }
        slices
    }

    fn buffers(source: u64, destination: u64) -> RowCopyBuffers<View> {
        RowCopyBuffers {
            source: view(1, source),
            destination: view(2, destination),
            live_rows: view(3, 4),
        }
    }

    #[test]
    fn broadcasts_launch_what_the_builder_plans() {
        for op in broadcasts() {
            let prepared = shared().broadcast_streams(&op, Need::rows(64)).unwrap();
            assert_eq!(
                prepared.info().implementation,
                "sm120.row.copy.broadcast.v1"
            );
            assert_eq!(prepared.info().guarantees, INVARIANCE);
            assert_eq!(prepared.info().symbols, [(BROADCAST_SYMBOL.to_owned(), 0)]);
            let row = u64::from(op.width) * 2;
            let full = buffers(64 * row, 64 * 4 * row);
            for rows in [1, 16, 64] {
                let launches = prepared
                    .launches(&full, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let built = Broadcast::new(op, rows, BROADCAST_SYMBOL).unwrap();
                assert_eq!(
                    launches,
                    [Launch {
                        symbol: BROADCAST_SYMBOL.into(),
                        rows,
                        plan: built.plan(rows).unwrap(),
                        arguments: built.arguments(1, 2, 3).unwrap(),
                        collective: false,
                        persistent: false,
                    }]
                );
            }
            for rows in [0, 65] {
                assert!(
                    prepared
                        .launches(&full, View::EMPTY, Counts::rows(rows))
                        .is_err()
                );
            }
            // Each view covers the launch's rows.
            for short in [
                buffers(16 * row - 1, 16 * 4 * row),
                buffers(16 * row, 16 * 4 * row - 1),
                RowCopyBuffers {
                    live_rows: view(3, 3),
                    ..full
                },
            ] {
                assert!(
                    prepared
                        .launches(&short, View::EMPTY, Counts::rows(16))
                        .is_err()
                );
            }
            assert!(
                prepared
                    .launches(
                        &buffers(16 * row, 16 * 4 * row),
                        View::EMPTY,
                        Counts::rows(16)
                    )
                    .is_ok()
            );
            let aliased = RowCopyBuffers {
                destination: view(1, 64 * 4 * row),
                ..full
            };
            assert!(
                prepared
                    .launches(&aliased, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
        }
    }

    #[test]
    fn slices_launch_what_the_layout_copy_plans() {
        for op in slices() {
            let prepared = shared().column_slice(&op, Need::rows(64)).unwrap();
            assert_eq!(
                prepared.info().implementation,
                "sm120.row.copy.slice.v1[layout-copy.bf16.exact.v1]"
            );
            assert_eq!(prepared.info().guarantees, INVARIANCE);
            assert_eq!(prepared.info().symbols, [(SLICE_SYMBOL.to_owned(), 0)]);
            let row = op.destination_row_bytes().unwrap();
            let full = buffers(op.source_bytes(64).unwrap(), 64 * row);
            for rows in [1, 16, 64] {
                let launches = prepared
                    .launches(&full, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                // Today's layout copy built at the launch's rows.
                let built = slice(&op, rows, SLICE_SYMBOL).unwrap();
                assert_eq!(
                    launches,
                    [Launch {
                        symbol: SLICE_SYMBOL.into(),
                        rows,
                        plan: built.stage().plan(rows).unwrap(),
                        arguments: built.arguments(1, 2, 3).unwrap(),
                        collective: false,
                        persistent: false,
                    }]
                );
                // The views the copy sizes are the slice's.
                assert_eq!(built.source_bytes(), op.source_bytes(rows).unwrap());
                assert_eq!(built.destination_bytes(), u64::from(rows) * row);
            }
            for rows in [0, 65] {
                assert!(
                    prepared
                        .launches(&full, View::EMPTY, Counts::rows(rows))
                        .is_err()
                );
            }
            let source = op.source_bytes(16).unwrap();
            for short in [buffers(source - 1, 16 * row), buffers(source, 16 * row - 1)] {
                assert!(
                    prepared
                        .launches(&short, View::EMPTY, Counts::rows(16))
                        .is_err()
                );
            }
            assert!(
                prepared
                    .launches(&buffers(source, 16 * row), View::EMPTY, Counts::rows(16))
                    .is_ok()
            );
        }
    }

    #[test]
    fn copies_refuse_unserved_inexact_and_over_limit_ops() {
        let [broadcast, _] = broadcasts();
        let slice = slices()[0];
        // No registered entry of the pattern.
        let only_slices =
            Sm120::new().with_row_copy(Entry::new(Pattern::ColumnSlice, SLICE_SYMBOL));
        assert!(matches!(
            only_slices.broadcast_streams(&broadcast, Need::rows(4)),
            Err(Refused::Rule(rule)) if rule.contains("no SM120 row copy entry")
        ));
        assert!(only_slices.column_slice(&slice, Need::rows(4)).is_ok());
        // A tie rule no copy has.
        let ties = Invariance::NONE.with_ties(Ties::LowerIndex);
        assert!(matches!(
            shared().broadcast_streams(&broadcast, Need::rows(4).with_exact(ties)),
            Err(Refused::Inexact { .. })
        ));
        assert!(matches!(
            shared().column_slice(&slice, Need::rows(4).with_exact(ties)),
            Err(Refused::Inexact { .. })
        ));
        // Exactness a copy keeps, which the shared symbols declare; an
        // undeclared symbol claims nothing, and the op claims no more than
        // its declaration, which stops at SM120's capacity.
        let prefill = Invariance::any_rows(8192);
        assert!(
            shared()
                .broadcast_streams(&broadcast, Need::rows(8192).with_exact(prefill))
                .is_ok()
        );
        let own = Sm120::new().with_row_copy(Entry::new(
            Pattern::BroadcastStreams,
            "chop_row_copy_own_v1",
        ));
        assert_eq!(
            own.broadcast_streams(&broadcast, Need::rows(4))
                .unwrap()
                .info()
                .guarantees,
            Invariance::NONE
        );
        assert!(matches!(
            own.broadcast_streams(&broadcast, Need::rows(4).with_exact(prefill)),
            Err(Refused::Inexact { .. })
        ));
        assert!(matches!(
            shared().broadcast_streams(&broadcast, Need::rows(4).with_exact(Invariance::COPY)),
            Err(Refused::Inexact { .. })
        ));
        // SM120's limits: the views' alignment, whole vectors, whole words,
        // footprints within u64, the capacity.
        let narrow = StreamBroadcast {
            alignment_bytes: 8,
            ..broadcast
        };
        assert!(matches!(
            shared().broadcast_streams(&narrow, Need::rows(4)),
            Err(Refused::Rule(rule)) if rule.contains("16-byte alignment")
        ));
        let unaligned = ColumnSlice {
            alignment_bytes: 1,
            ..slice
        };
        assert!(matches!(
            shared().column_slice(&unaligned, Need::rows(4)),
            Err(Refused::Rule(rule)) if rule.contains("2-byte alignment")
        ));
        let huge = StreamBroadcast {
            dtype: DType::I64,
            width: 1 << 30,
            streams: 1 << 30,
            alignment_bytes: 16,
        };
        assert!(matches!(
            shared().broadcast_streams(&huge, Need::rows(2)),
            Err(Refused::Rule(rule)) if rule.contains("overflow")
        ));
        let odd = StreamBroadcast {
            width: 2564,
            ..broadcast
        };
        assert!(matches!(
            shared().broadcast_streams(&odd, Need::rows(4)),
            Err(Refused::Rule(rule)) if rule.contains("16-byte vectors")
        ));
        let bytes = ColumnSlice {
            dtype: DType::U8,
            ..slice
        };
        assert!(matches!(
            shared().column_slice(&bytes, Need::rows(4)),
            Err(Refused::Rule(rule)) if rule.contains("16-bit words")
        ));
        for capacity in [0, MAX_CAPACITY + 1] {
            assert!(
                shared()
                    .broadcast_streams(&broadcast, Need::rows(capacity))
                    .is_err()
            );
            assert!(shared().column_slice(&slice, Need::rows(capacity)).is_err());
        }
        // The contract's own rules.
        let outside = ColumnSlice {
            offset: 33,
            ..slice
        };
        assert!(matches!(
            shared().column_slice(&outside, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
    }

    #[test]
    fn copy_entries_prepare_alike_in_every_order() {
        let op = broadcasts()[0];
        let buffers = buffers(4 * 5120, 4 * 4 * 5120);
        let launches = |registered: &[Entry]| {
            registered
                .iter()
                .cloned()
                .fold(Sm120::new(), |backend, entry| backend.with_row_copy(entry))
                .broadcast_streams(&op, Need::rows(4))
                .map(|op| op.launches(&buffers, View::EMPTY, Counts::rows(4)).unwrap())
        };
        let one = Entry::new(Pattern::BroadcastStreams, "chop_row_copy_a_v1");
        let slice = Entry::new(Pattern::ColumnSlice, "chop_row_copy_slice_v1");
        // A duplicate registration is one entry, and another pattern's
        // entry is not the op's.
        let expected = launches(std::slice::from_ref(&one)).unwrap();
        for order in orders(&[one.clone(), one.clone(), slice]) {
            assert_eq!(launches(&order).unwrap(), expected);
        }
        // Two symbols of one pattern are a tie.
        let other = Entry::new(Pattern::BroadcastStreams, "chop_row_copy_b_v1");
        for order in orders(&[one.clone(), other, one]) {
            assert!(matches!(
                launches(&order),
                Err(Refused::Rule(rule))
                    if rule.contains("chop_row_copy_a_v1 and chop_row_copy_b_v1")
            ));
        }
    }

    /// A copy entry is its pattern, which the op selects by, and its
    /// symbol, the one key field: it is the launch's symbol.
    #[test]
    fn the_launch_key_is_the_launched_symbol() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let buffers = buffers(16 * 5120, 16 * 4 * 5120);
        let fields: Vec<Field<Entry>> = vec![
            (
                "pattern",
                Class::Selects,
                Box::new(|e| e.pattern = Pattern::ColumnSlice),
            ),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_row_copy_other_v1".into()),
            ),
        ];
        keys_follow_launches(
            &Entry::new(Pattern::BroadcastStreams, BROADCAST_SYMBOL),
            field_names!(Entry { pattern, symbol }),
            fields,
            launch,
            |entry| {
                let op = Sm120::new()
                    .with_row_copy(entry.clone())
                    .broadcast_streams(&broadcasts()[0], Need::rows(16))?;
                let launches = [1, 16].map(|rows| {
                    op.launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((format!("{:?}", entry.spec()), launches))
            },
        );
    }
}
