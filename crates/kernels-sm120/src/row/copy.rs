//! The row copies (`kernel_core::row_copy`) on SM120: the stream broadcast
//! (`cuda/row/copy.cuh`), which moves 16-byte vectors, and the column
//! slice, the launch-shape layout copy ([`LayoutCopy`],
//! `precision::launch::copy_rows`) over 16-bit words. Each grids one CTA
//! per row of its launch and moves bytes only, so every launch of up to
//! the backend's capacity keeps every row, chunk and batch claim
//! ([`INVARIANCE`], declared for the shared symbols by
//! [`declared_entries`]). Launch-shape entries serve every shape: a build
//! compiles each pattern once, under the symbols of [`entries`] or its
//! own, which it declares itself.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};
use kernel_core::properties::Invariance;
use kernel_core::row_copy::{ColumnSlice, StreamBroadcast, element_bytes};

use super::precision::{CopyRows, LayoutCopy, RowResources, ShapeBinding};
use super::{ptr, u32_arg};
use crate::invariance::{Declared, OpKind};

/// The broadcast's header, relative to the CUDA source root.
pub const HEADER: &str = "row/copy.cuh";
/// Threads of the broadcast entry.
pub const THREADS: u32 = 256;
/// Bytes of one broadcast vector: a broadcast row holds whole vectors.
pub const VECTOR_BYTES: u64 = 16;
/// Bytes of one word the slice copies: a sliced element holds whole words.
pub const WORD_BYTES: u64 = 2;
/// The shared broadcast entry.
pub const BROADCAST_SYMBOL: &str = "chop_row_copy_broadcast_streams_v1";
/// The shared slice entry.
pub const SLICE_SYMBOL: &str = "chop_row_copy_column_slice_v1";
/// The lab's cubin of the shared entries.
pub const LAB_CUBIN: &str = "row_copy_sm120";
/// What every copy launch keeps: rows move without arithmetic, so each
/// row's bytes are its source row's at any row count up to the most rows
/// an SM120 op serves, in any batch or chunk split.
pub const INVARIANCE: Invariance = Invariance::any_rows(crate::backend::MAX_CAPACITY);

/// The copy an entry performs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Pattern {
    BroadcastStreams,
    ColumnSlice,
}

/// A copy entry a build compiles: its pattern, under `symbol`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub pattern: Pattern,
    pub symbol: String,
}

impl Entry {
    pub fn new(pattern: Pattern, symbol: impl Into<String>) -> Self {
        Self {
            pattern,
            symbol: symbol.into(),
        }
    }

    /// The entry's code under its symbol.
    pub fn spec(&self) -> EntrySpec {
        match self.pattern {
            Pattern::BroadcastStreams => EntrySpec::new(
                self.symbol.clone(),
                vec![
                    ptr("source", "const uint4"),
                    ptr("destination", "uint4"),
                    ptr("live_rows", "const uint32_t"),
                    u32_arg("vectors"),
                    u32_arg("streams"),
                ],
                "chop::row::copy::broadcast_streams(source,destination,live_rows,vectors,streams);"
                    .into(),
                THREADS,
                1,
            ),
            // A launch-shape copy's code names no shape.
            Pattern::ColumnSlice => {
                let one = ColumnSlice {
                    dtype: kernel_core::contract::DType::Bf16,
                    width: 1,
                    stride: 1,
                    offset: 0,
                    alignment_bytes: 2,
                };
                slice(&one, 1, &self.symbol)
                    .expect("a one-word slice")
                    .stage()
                    .entry()
                    .clone()
            }
        }
    }
}

/// The shared entries under their kernel-base symbols.
pub fn entries() -> Vec<Entry> {
    vec![
        Entry::new(Pattern::BroadcastStreams, BROADCAST_SYMBOL),
        Entry::new(Pattern::ColumnSlice, SLICE_SYMBOL),
    ]
}

/// The declarations of the shared entries ([`entries`]): row entries every
/// profile shares, claiming [`INVARIANCE`].
pub fn declared_entries() -> Vec<Declared> {
    entries()
        .into_iter()
        .map(|entry| Declared {
            symbol: entry.symbol,
            profile: None,
            op: OpKind::Row,
            invariance: INVARIANCE,
        })
        .collect()
}

/// A cubin `name` holding `entries`.
pub fn cubin_spec(name: impl Into<String>, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![
            "#include \"row/precision_launch.cuh\"".into(),
            format!("#include \"{HEADER}\""),
        ],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    }
}

/// The lab's cubin of the shared entries ([`LAB_CUBIN`]).
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    vec![cubin_spec(LAB_CUBIN, &entries())]
}

/// A stream broadcast of up to `capacity` rows through entry `symbol`.
#[derive(Clone, Debug)]
pub struct Broadcast {
    op: StreamBroadcast,
    capacity: u32,
    vectors: u32,
    symbol: String,
}

impl Broadcast {
    /// Refuses views aligned below a vector, a row of a partial vector,
    /// widths past the entry's loop bounds and footprints past u64: SM120's
    /// limits, not the contract's.
    pub fn new(op: StreamBroadcast, capacity: u32, symbol: &str) -> Result<Self, Refusal> {
        op.validate()?;
        if u64::from(op.alignment_bytes) < VECTOR_BYTES {
            return Err(Refusal(
                "an SM120 stream broadcast needs views of 16-byte alignment",
            ));
        }
        let row = op.source_row_bytes()?;
        if !row.is_multiple_of(VECTOR_BYTES) {
            return Err(Refusal(
                "an SM120 stream broadcast moves rows of whole 16-byte vectors",
            ));
        }
        let vectors = row / VECTOR_BYTES;
        if vectors > i32::MAX as u64 || op.streams > i32::MAX as u32 {
            return Err(Refusal(
                "an SM120 stream broadcast's width exceeds its loop bounds",
            ));
        }
        if capacity == 0 || capacity > i32::MAX as u32 {
            return Err(Refusal(
                "an SM120 stream broadcast's capacity exceeds its grid",
            ));
        }
        op.destination_bytes(capacity)?;
        Ok(Self {
            op,
            capacity,
            vectors: vectors as u32,
            symbol: symbol.into(),
        })
    }

    pub fn op(&self) -> StreamBroadcast {
        self.op
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// One CTA per row of `rows`, the launch's rows.
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if rows == 0 || rows > self.capacity {
            return Err(Refusal(
                "stream broadcast rows exceed the prepared capacity",
            ));
        }
        let plan = LaunchPlan {
            grid: [rows, 1, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn arguments(
        &self,
        source: u64,
        destination: u64,
        live_rows: u64,
    ) -> Result<ArgumentList, Refusal> {
        if source == destination {
            return Err(Refusal(
                "a stream broadcast requires nonoverlapping source and destination",
            ));
        }
        let args = ArgumentList(vec![
            Argument::DeviceBuffer(source),
            Argument::DeviceBuffer(destination),
            Argument::DeviceBuffer(live_rows),
            Argument::U32(self.vectors),
            Argument::U32(self.op.streams),
        ]);
        args.validate()?;
        Ok(args)
    }
}

/// The column slice `op` of up to `capacity` rows through the launch-shape
/// layout copy `symbol`, over 16-bit words: bitwise the slice of its
/// elements. Refuses views aligned below a word, elements of an odd byte
/// count, word counts past the copy's bounds and footprints past u64,
/// SM120's limits.
pub fn slice(op: &ColumnSlice, capacity: u32, symbol: &str) -> Result<LayoutCopy, Refusal> {
    op.validate()?;
    if u64::from(op.alignment_bytes) < WORD_BYTES {
        return Err(Refusal(
            "an SM120 column slice needs views of 2-byte alignment",
        ));
    }
    let element = element_bytes(op.dtype).ok_or(Refusal("a row copy moves whole bytes"))?;
    if !u64::from(element).is_multiple_of(WORD_BYTES) {
        return Err(Refusal(
            "an SM120 column slice moves elements of whole 16-bit words",
        ));
    }
    op.source_bytes(capacity)?;
    op.destination_bytes(capacity)?;
    let words = |elements: u32| {
        u32::try_from(u64::from(elements) * u64::from(element) / WORD_BYTES)
            .map_err(|_| Refusal("an SM120 column slice's row exceeds its word bounds"))
    };
    let source = CopyRows {
        row_stride: words(op.stride)?,
        column_stride: 1,
        column_offset: words(op.offset)?,
    };
    let destination = CopyRows {
        row_stride: words(op.width)?,
        column_stride: 1,
        column_offset: 0,
    };
    LayoutCopy::new(
        words(op.width)?,
        capacity,
        source,
        destination,
        symbol,
        ShapeBinding::Launch,
        RowResources::SM120F,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::DType;

    fn broadcast(dtype: DType, width: u32) -> StreamBroadcast {
        StreamBroadcast {
            dtype,
            width,
            streams: 4,
            alignment_bytes: 16,
        }
    }

    #[test]
    fn broadcasts_plan_one_cta_per_row_of_whole_vectors() {
        let built = Broadcast::new(broadcast(DType::Bf16, 2560), 32, BROADCAST_SYMBOL).unwrap();
        let plan = built.plan(8).unwrap();
        assert_eq!(
            (plan.grid, plan.block, plan.pdl),
            ([8, 1, 1], [256, 1, 1], true)
        );
        assert_eq!(
            built.arguments(1, 2, 3).unwrap().0[3..],
            [Argument::U32(320), Argument::U32(4)]
        );
        assert!(built.plan(33).is_err());
        assert!(built.plan(0).is_err());
        assert!(built.arguments(1, 1, 3).is_err());
        // Rows of a partial vector are SM120's to refuse; the contract
        // takes them.
        let odd = broadcast(DType::Bf16, 2564);
        assert_eq!(odd.validate(), Ok(()));
        assert!(Broadcast::new(odd, 32, BROADCAST_SYMBOL).is_err());
        assert!(Broadcast::new(broadcast(DType::F32, 4), 32, BROADCAST_SYMBOL).is_ok());
        assert!(Broadcast::new(broadcast(DType::Bf16, 2560), 0, BROADCAST_SYMBOL).is_err());
        // Views aligned below a vector, which the contract takes.
        let narrow = StreamBroadcast {
            alignment_bytes: 8,
            ..broadcast(DType::Bf16, 2560)
        };
        assert_eq!(narrow.validate(), Ok(()));
        assert!(Broadcast::new(narrow, 32, BROADCAST_SYMBOL).is_err());
        // A footprint past u64 is refused when built, not at launch.
        let huge = StreamBroadcast {
            dtype: DType::I64,
            width: 1 << 30,
            streams: 1 << 30,
            alignment_bytes: 16,
        };
        assert!(Broadcast::new(huge, 1, BROADCAST_SYMBOL).is_ok());
        assert!(Broadcast::new(huge, 2, BROADCAST_SYMBOL).is_err());
    }

    #[test]
    fn slices_copy_their_elements_as_words() {
        let f32 = ColumnSlice {
            dtype: DType::F32,
            width: 288,
            stride: 320,
            offset: 16,
            alignment_bytes: 16,
        };
        let copy = slice(&f32, 32, SLICE_SYMBOL).unwrap();
        let (columns, source, destination) = copy.layout();
        assert_eq!(columns, 576);
        assert_eq!(
            source,
            CopyRows {
                row_stride: 640,
                column_stride: 1,
                column_offset: 32
            }
        );
        assert_eq!(destination.row_stride, 576);
        assert_eq!(copy.source_bytes(), f32.source_bytes(32).unwrap());
        assert_eq!(copy.destination_bytes(), 32 * 288 * 4);
        let byte = ColumnSlice {
            dtype: DType::U8,
            ..f32
        };
        assert_eq!(byte.validate(), Ok(()));
        assert!(slice(&byte, 32, SLICE_SYMBOL).is_err());
        // Views aligned below a word, which the contract takes.
        let narrow = ColumnSlice {
            alignment_bytes: 1,
            ..f32
        };
        assert_eq!(narrow.validate(), Ok(()));
        assert!(slice(&narrow, 32, SLICE_SYMBOL).is_err());
        assert!(
            slice(
                &ColumnSlice {
                    alignment_bytes: 2,
                    ..f32
                },
                32,
                SLICE_SYMBOL
            )
            .is_ok()
        );
        let wide = ColumnSlice {
            dtype: DType::I64,
            width: 1 << 30,
            stride: 1 << 30,
            offset: 0,
            alignment_bytes: 16,
        };
        assert!(slice(&wide, 1, SLICE_SYMBOL).is_err());
    }

    #[test]
    fn the_shared_entries_declare_row_claims_up_to_the_capacity() {
        use kernel_core::properties::Property;
        let declared = declared_entries();
        let symbols: Vec<_> = declared.iter().map(|d| d.symbol.as_str()).collect();
        assert_eq!(symbols, [BROADCAST_SYMBOL, SLICE_SYMBOL]);
        for entry in &declared {
            assert_eq!((entry.op, entry.profile), (OpKind::Row, None));
            assert_eq!(entry.invariance, INVARIANCE);
        }
        assert_eq!(
            INVARIANCE.claims(),
            [Property::Chunk, Property::Rows, Property::Batch]
        );
        assert!(!INVARIANCE.one_row_at(crate::backend::MAX_CAPACITY + 1));
    }

    #[test]
    fn the_lab_cubin_holds_both_patterns_under_their_symbols() {
        let [spec] = &lab_cubin_specs()[..] else {
            panic!("one lab cubin");
        };
        assert_eq!(spec.name, LAB_CUBIN);
        let symbols: Vec<_> = spec.entries.iter().map(|e| e.symbol.as_str()).collect();
        assert_eq!(symbols, [BROADCAST_SYMBOL, SLICE_SYMBOL]);
        assert!(spec.entries[1].body.contains("launch::copy_rows"));
        let renamed = Entry::new(Pattern::ColumnSlice, "chop_row_copy_other_v1").spec();
        assert_eq!(renamed.body, spec.entries[1].body);
        assert_eq!(renamed.params, spec.entries[1].params);
    }
}
