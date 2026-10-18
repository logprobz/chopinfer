//! Prepared paged FP4 writes with independent code and scale plane bounds.

use kernel_core::attention::BlockTables;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::compressed_kv::{
    CompressedKvStorage, CompressedKvWriteBuffers, CompressedKvWriteQuery,
};
use kernel_core::contract::{Argument, ArgumentList, Refusal, Validate};
use kernel_core::pooled_indexer::{
    IndexerKeyFormat, PositionIndexerQuery, PositionKeyWriteBuffers,
};
use kernel_core::properties::Invariance;

use super::select::{Candidate, choose};
use crate::gated_pooling::key_write::Entry;

fn select(
    entries: &[Entry],
    need: Need,
    matches: impl Fn(&Entry) -> bool,
) -> Result<&Entry, Refused> {
    Ok(choose(
        "FP4 key write",
        entries.iter().filter(|e| matches(e)).map(|entry| {
            Candidate::new(
                entry,
                entry.symbol(),
                format!("{:?}", entry.entry_spec()),
                Invariance::any_rows(need.capacity),
            )
            .capacity(entry.capacity())
        }),
        need,
    )?
    .entry)
}

pub(super) fn index(
    entries: &[Entry],
    query: &PositionIndexerQuery,
    need: Need,
) -> Result<Prepared<PositionKeyWriteBuffers<View>>, Refused> {
    index_with_storage(entries, query, IndexerKeyFormat::Fp4, need)
}

pub(super) fn index_with_storage(
    entries: &[Entry],
    query: &PositionIndexerQuery,
    storage: IndexerKeyFormat,
    need: Need,
) -> Result<Prepared<PositionKeyWriteBuffers<View>>, Refused> {
    query.validate()?;
    check_need(query.capacity, need)?;
    let entry = select(entries, need, |e| e.matches_index_storage(query, storage))?;
    Ok(Box::new(IndexWrite(Write::new(entry, need)?)))
}

pub(super) fn compressed(
    entries: &[Entry],
    query: &CompressedKvWriteQuery,
    need: Need,
) -> Result<Prepared<CompressedKvWriteBuffers<View>>, Refused> {
    query.validate()?;
    check_need(query.capacity, need)?;
    let entry = select(entries, need, |e| e.matches_compressed(query))?;
    Ok(Box::new(CompressedWrite {
        write: Write::new(entry, need)?,
        query: *query,
    }))
}

fn check_need(capacity: u32, need: Need) -> Result<(), Refused> {
    if need.capacity == 0 || need.capacity > capacity {
        return Err(Refused::Rule(
            "FP4 key-write need exceeds the query capacity".into(),
        ));
    }
    Ok(())
}

struct Write {
    entry: Entry,
    capacity: u32,
    info: OpInfo,
}

impl Write {
    fn new(entry: &Entry, need: Need) -> Result<Self, Refused> {
        let info = OpInfo::new(
            "sm120.fp4-key-write.f32-tail-rope-status-or-zero.v2",
            Invariance::any_rows(need.capacity),
            vec![(entry.symbol().to_owned(), 0)],
        );
        need.check(&info)?;
        Ok(Self {
            entry: entry.clone(),
            capacity: need.capacity,
            info,
        })
    }

    fn inputs(
        &self,
        key: View,
        cos_sin: View,
        positions: View,
        row_batch: View,
        live_rows: View,
        counts: Counts,
    ) -> Result<(), Refusal> {
        if counts.rows == 0 || counts.rows > self.capacity || counts.positions == 0 {
            return Err(Refusal(
                "FP4 key writes need bounded rows and a nonempty rotary table",
            ));
        }
        key.covers(u64::from(counts.rows) * u64::from(self.entry.width()) * 4)?;
        cos_sin.covers(u64::from(counts.positions) * u64::from(self.entry.rotary()) * 4)?;
        positions.covers(u64::from(counts.rows) * 4)?;
        row_batch.covers(u64::from(counts.rows) * 4)?;
        live_rows.covers(4)
    }

    fn launch(&self, arguments: Vec<Argument>, counts: Counts) -> Result<Vec<Launch>, Refusal> {
        let arguments = ArgumentList(arguments);
        arguments.validate()?;
        Ok(vec![Launch {
            symbol: self.entry.symbol().to_owned(),
            rows: counts.rows,
            plan: self.entry.plan(counts.rows)?,
            arguments,
            collective: false,
            persistent: false,
        }])
    }
}

fn plane(view: View, pages: u32, stride: u64, minimum: u64) -> Result<(), Refusal> {
    if pages == 0 || stride < minimum {
        return Err(Refusal("FP4 key-write pages or plane stride are too small"));
    }
    view.covers(
        u64::from(pages - 1)
            .checked_mul(stride)
            .and_then(|bytes| bytes.checked_add(minimum))
            .ok_or(Refusal("FP4 key-write plane size overflows"))?,
    )
}

fn status_buffer(status: View, offset: u64, rows: u32, operands: &[View]) -> Result<(), Refusal> {
    status.covers(kernel_core::precision::quantizer::status_extent(
        offset, rows,
    )?)?;
    if operands
        .iter()
        .any(|v| v.bytes > 0 && v.handle == status.handle)
    {
        return Err(Refusal("FP4 key-write status overlaps another operand"));
    }
    Ok(())
}

fn pointer(view: View) -> Argument {
    Argument::DeviceBuffer(view.handle)
}

fn page_arguments(
    tables: BlockTables<View>,
    pages: u32,
    code_stride: u64,
    scale_stride: u64,
    counts: Counts,
    eps_bits: u32,
) -> Vec<Argument> {
    vec![
        Argument::U32(tables.sequences),
        Argument::U32(pages),
        Argument::U32(tables.max_pages),
        Argument::U64(code_stride),
        Argument::U64(scale_stride),
        Argument::U32(counts.positions),
        Argument::U32(eps_bits),
    ]
}

struct IndexWrite(Write);

impl PreparedOp<PositionKeyWriteBuffers<View>> for IndexWrite {
    fn info(&self) -> &OpInfo {
        &self.0.info
    }

    fn launches(
        &self,
        b: &PositionKeyWriteBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let write = &self.0;
        write.inputs(
            b.key,
            b.cos_sin,
            b.positions,
            b.row_batch,
            b.live_rows,
            counts,
        )?;
        b.norm_weight.covers(u64::from(write.entry.width()) * 2)?;
        let k = b.keys;
        k.page_tables.check()?;
        let elements = u64::from(write.entry.page_size()) * u64::from(write.entry.width());
        if write.entry.packed() {
            plane(k.codes, k.pages, k.code_page_stride, elements / 2)?;
            plane(k.scales, k.pages, k.scale_page_stride, elements / 32)?;
        } else {
            if k.code_page_stride % 2 != 0 {
                return Err(Refusal("BF16 index-key page stride must be even"));
            }
            if k.scales != View::EMPTY || k.scale_page_stride != 0 {
                return Err(Refusal("BF16 index keys have no scale plane"));
            }
            plane(k.codes, k.pages, k.code_page_stride, elements * 2)?;
        }
        status_buffer(
            b.status,
            b.status_row_offset,
            counts.rows,
            &[
                b.key,
                b.norm_weight,
                b.cos_sin,
                b.positions,
                b.row_batch,
                b.live_rows,
                k.page_tables.tables,
                k.codes,
                k.scales,
            ],
        )?;
        let mut args = vec![
            pointer(b.key),
            pointer(b.norm_weight),
            pointer(b.cos_sin),
            pointer(b.positions),
            pointer(b.row_batch),
            pointer(b.live_rows),
            pointer(k.page_tables.tables),
            pointer(k.codes),
        ];
        if write.entry.packed() {
            args.push(pointer(k.scales));
        }
        args.extend(page_arguments(
            k.page_tables,
            k.pages,
            k.code_page_stride,
            k.scale_page_stride,
            counts,
            write.entry.eps_bits(),
        ));
        args.extend([pointer(b.status), Argument::U64(b.status_row_offset)]);
        write.launch(args, counts)
    }
}

struct CompressedWrite {
    write: Write,
    query: CompressedKvWriteQuery,
}

impl PreparedOp<CompressedKvWriteBuffers<View>> for CompressedWrite {
    fn info(&self) -> &OpInfo {
        &self.write.info
    }

    fn launches(
        &self,
        b: &CompressedKvWriteBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        self.write.inputs(
            b.latent,
            b.cos_sin,
            b.positions,
            b.row_batch,
            b.live_rows,
            counts,
        )?;
        let k = b.cache;
        k.page_tables.check()?;
        let (values, scales) = self.query.page_bytes()?;
        plane(k.values, k.pages, k.value_page_stride, values)?;
        let scale = match self.query.storage {
            CompressedKvStorage::Packed => {
                let view = k
                    .scales
                    .ok_or(Refusal("packed compressed KV needs E4M3 scales"))?;
                plane(view, k.pages, k.scale_page_stride, scales)?;
                view
            }
            CompressedKvStorage::Bf16 => {
                if k.value_page_stride % 2 != 0 {
                    return Err(Refusal("BF16 compressed KV page stride must be even"));
                }
                View::EMPTY
            }
        };
        status_buffer(
            b.status,
            b.status_row_offset,
            counts.rows,
            &[
                b.latent,
                b.cos_sin,
                b.positions,
                b.row_batch,
                b.live_rows,
                k.page_tables.tables,
                k.values,
                scale,
            ],
        )?;
        let mut args = vec![
            pointer(b.latent),
            pointer(b.cos_sin),
            pointer(b.positions),
            pointer(b.row_batch),
            pointer(b.live_rows),
            pointer(k.page_tables.tables),
            pointer(k.values),
        ];
        if self.query.storage == CompressedKvStorage::Packed {
            args.push(pointer(scale));
        }
        args.extend(page_arguments(
            k.page_tables,
            k.pages,
            k.value_page_stride,
            if scales == 0 { 0 } else { k.scale_page_stride },
            counts,
            0,
        ));
        args.extend([pointer(b.status), Argument::U64(b.status_row_offset)]);
        self.write.launch(args, counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Sm120;
    use kernel_core::backend::Backend;
    use kernel_core::compressed_kv::CompressedKvPages;
    use kernel_core::pooled_indexer::{IndexerCandidates, PositionKeyPages};

    fn index_query(ratio: u32) -> PositionIndexerQuery {
        PositionIndexerQuery {
            heads: 32,
            head_dim: 128,
            compression_ratio: ratio,
            tail_rope_dims: 64,
            key_norm_eps_bits: 1e-20f32.to_bits(),
            max_top_k: 512,
            page_size: 64,
            candidates: IndexerCandidates::All,
            capacity: 8,
        }
    }

    fn compressed_query(storage: CompressedKvStorage, ratio: u32) -> CompressedKvWriteQuery {
        CompressedKvWriteQuery {
            width: 512,
            tail_rope_dims: 64,
            compression_ratio: ratio,
            page_size: 64,
            storage,
            capacity: 8,
        }
    }

    fn view(handle: u64) -> View {
        View {
            handle,
            bytes: 1 << 38,
        }
    }

    fn index_buffers() -> PositionKeyWriteBuffers<View> {
        PositionKeyWriteBuffers {
            status: view(20),
            status_row_offset: 3,
            key: view(1),
            norm_weight: view(2),
            cos_sin: view(3),
            positions: view(4),
            row_batch: view(5),
            live_rows: view(6),
            keys: PositionKeyPages {
                codes: view(7),
                scales: view(8),
                pages: 3,
                code_page_stride: (1 << 32) + 4096,
                scale_page_stride: (1 << 33) + 256,
                page_tables: BlockTables {
                    tables: view(9),
                    sequences: 2,
                    max_pages: 16,
                },
            },
        }
    }

    fn compressed_buffers() -> CompressedKvWriteBuffers<View> {
        CompressedKvWriteBuffers {
            status: view(20),
            status_row_offset: 3,
            latent: view(1),
            cos_sin: view(2),
            positions: view(3),
            row_batch: view(4),
            live_rows: view(5),
            cache: CompressedKvPages {
                values: view(6),
                scales: Some(view(7)),
                pages: 3,
                value_page_stride: (1 << 32) + 65536,
                scale_page_stride: (1 << 33) + 2048,
                page_tables: BlockTables {
                    tables: view(8),
                    sequences: 2,
                    max_pages: 16,
                },
            },
        }
    }

    #[test]
    fn backend_prepares_both_ratios_and_all_required_writers() {
        let mut prepared = 0;
        for ratio in [1, 2] {
            let q = index_query(ratio);
            let backend =
                Sm120::new().with_fp4_key_writes([Entry::index(&q, "chop_index_test_v1").unwrap()]);
            let op = backend
                .position_indexer_key_write(&q, Need::rows(8))
                .unwrap();
            let b = index_buffers();
            let launches = op
                .launches(&b, View::EMPTY, Counts::rows(8).with_positions(1024))
                .unwrap();
            assert_eq!(launches.len(), 1);
            assert_eq!(
                launches[0].arguments.0[12],
                Argument::U64(b.keys.code_page_stride)
            );
            assert_eq!(
                launches[0].arguments.0[13],
                Argument::U64(b.keys.scale_page_stride)
            );
            prepared += 1;
            for storage in [CompressedKvStorage::Packed, CompressedKvStorage::Bf16] {
                let q = compressed_query(storage, ratio);
                let backend =
                    Sm120::new()
                        .with_fp4_key_writes([Entry::compressed(&q, "chop_kv_test_v1").unwrap()]);
                let op = backend.compressed_kv_write(&q, Need::rows(8)).unwrap();
                let mut b = compressed_buffers();
                if storage == CompressedKvStorage::Bf16 {
                    b.cache.scales = None;
                }
                let launches = op
                    .launches(&b, View::EMPTY, Counts::rows(8).with_positions(1024))
                    .unwrap();
                assert_eq!(launches.len(), 1);
                assert_eq!(
                    launches[0].arguments.0[if storage == CompressedKvStorage::Packed {
                        11
                    } else {
                        10
                    }],
                    Argument::U64(b.cache.value_page_stride)
                );
                assert!(launches[0].plan.pdl);
                prepared += 1;
            }
        }
        assert_eq!(prepared, 6);
    }

    #[test]
    fn bf16_index_storage_has_exact_extent_no_scales_and_native_refusals() {
        let q = index_query(2);
        let native = Entry::index(&q, "native_v1").unwrap();
        let bf16 = Entry::index_with_storage(&q, IndexerKeyFormat::Bf16, "bf16_v1").unwrap();
        let backend = Sm120::new().with_fp4_key_writes([native.clone(), bf16.clone()]);
        let op = backend
            .position_indexer_key_write_with_storage(&q, IndexerKeyFormat::Bf16, Need::rows(8))
            .unwrap();
        assert_eq!(op.info().symbols[0].0, "bf16_v1");
        assert_eq!(
            backend
                .position_indexer_key_write(&q, Need::rows(8))
                .unwrap()
                .info()
                .symbols[0]
                .0,
            "native_v1"
        );
        assert!(
            Sm120::new()
                .with_fp4_key_writes([native])
                .position_indexer_key_write_with_storage(&q, IndexerKeyFormat::Bf16, Need::rows(8))
                .is_err()
        );
        let spec = bf16.entry_spec();
        assert!(!spec.params.iter().any(
            |p| matches!(p, kernel_build::EntryParam::DevicePointer{name,..} if name=="scales")
        ));
        assert!(spec.body.contains("128,64,2,64,true,false,false>"));
        let counts = Counts::rows(8).with_positions(1024);
        let mut b = index_buffers();
        b.keys.scales = View::EMPTY;
        b.keys.scale_page_stride = 0;
        let extent = 2 * b.keys.code_page_stride + 64 * 128 * 2;
        b.keys.codes.bytes = extent;
        op.launches(&b, View::EMPTY, counts).unwrap();
        let mut bad = b;
        bad.keys.codes.bytes -= 1;
        assert!(op.launches(&bad, View::EMPTY, counts).is_err());
        bad = b;
        bad.keys.code_page_stride += 1;
        assert!(op.launches(&bad, View::EMPTY, counts).is_err());
        bad = b;
        bad.keys.scales = view(8);
        assert!(op.launches(&bad, View::EMPTY, counts).is_err());
        bad = b;
        bad.keys.scale_page_stride = 32;
        assert!(op.launches(&bad, View::EMPTY, counts).is_err());
        bad = b;
        bad.status.handle = bad.keys.codes.handle;
        assert!(op.launches(&bad, View::EMPTY, counts).is_err());
        bad = b;
        bad.status.bytes = (3 + 8) * 4 - 1;
        assert!(op.launches(&bad, View::EMPTY, counts).is_err());
        bad = b;
        bad.keys.code_page_stride = u64::MAX - 1;
        assert!(op.launches(&bad, View::EMPTY, counts).is_err());
    }

    #[test]
    fn minimal_plane_extents_accept_embedded_pages_and_reject_a_missing_last_byte() {
        let stride = (1u64 << 32) + 256;
        let extent = 2 * stride + 64;
        let view = View {
            handle: 16,
            bytes: extent,
        };
        plane(view, 3, stride, 64).unwrap();
        assert!(
            plane(
                View {
                    bytes: extent - 1,
                    ..view
                },
                3,
                stride,
                64
            )
            .is_err()
        );
        plane(View { bytes: 64, ..view }, 1, u64::MAX, 64).unwrap();
        assert!(plane(view, 3, u64::MAX, 64).is_err());
    }

    #[test]
    fn rejects_missing_scales_overflow_and_short_planes() {
        let q = compressed_query(CompressedKvStorage::Packed, 2);
        let backend =
            Sm120::new().with_fp4_key_writes([Entry::compressed(&q, "chop_kv_test_v1").unwrap()]);
        let op = backend.compressed_kv_write(&q, Need::rows(8)).unwrap();
        let counts = Counts::rows(8).with_positions(1024);
        let mut b = compressed_buffers();
        b.cache.scales = None;
        assert!(op.launches(&b, View::EMPTY, counts).is_err());
        b = compressed_buffers();
        b.cache.value_page_stride = u64::MAX;
        assert!(op.launches(&b, View::EMPTY, counts).is_err());
        b = compressed_buffers();
        b.cache.scale_page_stride = 2047;
        assert!(op.launches(&b, View::EMPTY, counts).is_err());
        b = compressed_buffers();
        b.cos_sin.bytes = 1024 * 64 * 4 - 1;
        assert!(op.launches(&b, View::EMPTY, counts).is_err());
    }

    #[test]
    fn status_extents_offsets_and_aliases_are_checked_for_both_writers() {
        let counts = Counts::rows(8).with_positions(1024);
        let q = index_query(2);
        let op = Sm120::new()
            .with_fp4_key_writes([Entry::index(&q, "status_index_v1").unwrap()])
            .position_indexer_key_write(&q, Need::rows(8))
            .unwrap();
        let mut b = index_buffers();
        b.status.bytes = (3 + 8) * 4;
        let launch = op.launches(&b, View::EMPTY, counts).unwrap();
        assert_eq!(launch[0].arguments.0.last(), Some(&Argument::U64(3)));
        b.status.bytes -= 1;
        assert!(op.launches(&b, View::EMPTY, counts).is_err());
        b = index_buffers();
        b.status_row_offset = u64::MAX;
        assert!(op.launches(&b, View::EMPTY, counts).is_err());
        for handle in 1..=9 {
            let mut b = index_buffers();
            b.status.handle = handle;
            assert!(op.launches(&b, View::EMPTY, counts).is_err());
        }
        for storage in [CompressedKvStorage::Packed, CompressedKvStorage::Bf16] {
            let q = compressed_query(storage, 2);
            let op = Sm120::new()
                .with_fp4_key_writes([Entry::compressed(&q, "status_kv_v1").unwrap()])
                .compressed_kv_write(&q, Need::rows(8))
                .unwrap();
            let mut b = compressed_buffers();
            if storage == CompressedKvStorage::Bf16 {
                b.cache.scales = None;
            }
            op.launches(&b, View::EMPTY, counts).unwrap();
            for handle in [1, 2, 3, 4, 5, 6, 8] {
                let mut bad = b;
                bad.status.handle = handle;
                assert!(op.launches(&bad, View::EMPTY, counts).is_err());
            }
            b.status.bytes = 43;
            assert!(op.launches(&b, View::EMPTY, counts).is_err());
        }
    }

    #[test]
    fn formats_and_group_positions_cannot_bind_interchangeably() {
        let q = index_query(2);
        let backend =
            Sm120::new().with_fp4_key_writes([Entry::index(&q, "chop_index_test_v1").unwrap()]);
        assert!(
            backend
                .position_indexer_key_write(&index_query(1), Need::rows(8))
                .is_err()
        );
        assert!(
            backend
                .compressed_kv_write(
                    &compressed_query(CompressedKvStorage::Packed, 2),
                    Need::rows(8)
                )
                .is_err()
        );
        assert!(
            Entry::index(
                &PositionIndexerQuery {
                    tail_rope_dims: 32,
                    ..q
                },
                "chop_bad"
            )
            .is_err()
        );
    }
}
