//! The MiMo bundle's own DFlash row kernels at every rank count, run from
//! the cubins the bundle embeds: the row copy, circular cache write and
//! draft argmax over the ranks.
//! Each runs through the declarative conformance runner: poisoned eager,
//! relaunch, graph and PDL launches at every live count, a copy producer
//! rewriting a real input, and a launch that writes nothing failing.

use bundle_mimo26::{
    embedded_cubin,
    topology::{Tp, bundle_symbol, entry_threads, kernel_tag},
};
use kernel_build::EntryParam;
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan, Refusal};
use kernel_lab::conformance::{Case, Expected, Holes, Images, Rows, Write};
use kernel_lab::lab_entry::{LabEntry, pack_by_kind};
use kernel_lab::launch::simple_config;
use kernel_lab::poison::WriteMap;
use kernel_lab::{LabResult, Session, latent_shapes};
use model_mimo26_contract::own_ops::dflash;
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig, Module};

fn module(session: &Session, tp: Tp, name: &str) -> LabResult<Module> {
    let cubin = format!("{}_{name}", kernel_tag(tp));
    session.load_cubin(embedded_cubin(&cubin).ok_or(format!("missing {cubin}"))?)
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(16))?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn download(session: &Session, buffer: &DeviceBuffer, bytes: usize) -> LabResult<Vec<u8>> {
    let mut host = vec![0_u8; bytes];
    buffer
        .copy_to_bytes(&mut host, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(host)
}

fn words<T: Copy>(values: &[T], bytes: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    values.iter().flat_map(|&value| bytes(value)).collect()
}

fn launch(
    session: &Session,
    module: &Module,
    symbol: &str,
    grid: u32,
    block: u32,
    arguments: &[KernelArgument<'_>],
) -> LabResult<()> {
    module
        .function(symbol)
        .map_err(|e| e.to_string())?
        .launch(
            &session.stream,
            simple_config(
                (grid, 1, 1),
                (block, 1, 1),
                0,
                LaunchConfig::NO_CLUSTER,
                true,
            )?,
            arguments,
        )
        .map_err(|e| e.to_string())
}

/// A bundle row entry as the bundle plans it: a block of the entry's
/// declared threads per `rows_per_block` rows of a bucket, with PDL. Its
/// parameters are the ones `build.rs` declares: pointers, then u32 scalars.
struct BundleRow {
    symbol: String,
    params: Vec<EntryParam>,
    threads: u32,
    rows_per_block: u32,
}

impl BundleRow {
    fn new(
        tp: Tp,
        name: &str,
        pointers: &[&str],
        scalars: &[&str],
        rows_per_block: u32,
    ) -> LabResult<Self> {
        let symbol = bundle_symbol(tp, name);
        let threads = entry_threads(&symbol).ok_or(format!("no entry {symbol}"))?;
        let params = pointers
            .iter()
            .map(|&name| EntryParam::DevicePointer {
                name: name.into(),
                cxx_type: "void".into(),
            })
            .chain(
                scalars
                    .iter()
                    .map(|&name| EntryParam::U32 { name: name.into() }),
            )
            .collect();
        Ok(Self {
            symbol,
            params,
            threads,
            rows_per_block,
        })
    }
}

impl LabEntry for BundleRow {
    fn symbol(&self) -> String {
        self.symbol.clone()
    }

    fn params(&self) -> Vec<EntryParam> {
        self.params.clone()
    }

    fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        if prepared_rows == 0 {
            return Err(Refusal("a bundle row launch needs rows"));
        }
        Ok(LaunchPlan {
            grid: [prepared_rows.div_ceil(self.rows_per_block), 1, 1],
            block: [self.threads, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        })
    }
}

/// The DFlash row copy writes the first `width` columns of each live row,
/// unscaled here, and nothing past them or past the live rows.
struct CopyRows {
    entry: BundleRow,
    input: Vec<u8>,
}

impl CopyRows {
    const CAPACITY: u32 = 64;
    const WIDTH: usize = 1024;
    const INPUT_STRIDE: usize = 4096;
    const OUTPUT_STRIDE: usize = 1536;

    fn new(tp: Tp) -> LabResult<Self> {
        let entry = BundleRow::new(
            tp,
            "dflash_copy_rows",
            &["input", "output", "live_rows"],
            &["width", "input_stride", "output_stride", "scale_bits"],
            1,
        )?;
        let input = (0..Self::CAPACITY as usize * Self::INPUT_STRIDE)
            .flat_map(|index| {
                ((index as u32).wrapping_mul(2_654_435_761) as u16 & 0x7f7f).to_le_bytes()
            })
            .collect();
        Ok(Self { entry, input })
    }

    fn case(&self) -> Case<'_> {
        self.case_of(&self.entry)
    }

    /// The copy's case run through `entry`.
    fn case_of<'a>(&'a self, entry: &'a dyn LabEntry) -> Case<'a> {
        let (width, input_stride, output_stride) =
            (Self::WIDTH, Self::INPUT_STRIDE, Self::OUTPUT_STRIDE);
        Case::new(entry, Self::CAPACITY)
            .input(
                "input",
                DType::Bf16,
                (Rows::Live, input_stride),
                self.input.clone(),
            )
            .produced("input")
            .output(
                "output",
                DType::Bf16,
                (Rows::Live, output_stride),
                Write::Map(Box::new(move |_, live| {
                    WriteMap::of(DType::Bf16).ranges(
                        (0..live as usize)
                            .map(|row| row * output_stride..row * output_stride + width),
                    )
                })),
            )
            .scalar("width", Argument::U32(width as u32))
            .scalar("input_stride", Argument::U32(input_stride as u32))
            .scalar("output_stride", Argument::U32(output_stride as u32))
            .scalar("scale_bits", Argument::U32(1.0_f32.to_bits()))
            .oracle(move |images, live| {
                let input = images.bytes("input")?;
                let mut expected = vec![0; Self::CAPACITY as usize * output_stride * 2];
                let op = dflash::CopyRows {
                    width: width as u32,
                    input_stride: Some(input_stride as u32),
                    output_stride: output_stride as u32,
                    scale_bits: 1.0_f32.to_bits(),
                };
                dflash::copy_rows(&op, Some(input), &mut expected, live, Self::CAPACITY)
                    .map_err(|e| e.0.to_string())?;
                Ok(vec![("output".into(), Expected::Bits(expected))])
            })
    }
}

/// A broken entry for the runner's negative controls: the row copy with
/// its live count read from `stale_rows`, a constant input, in place of the
/// device live count.
struct StaleCount(BundleRow);

impl LabEntry for StaleCount {
    fn symbol(&self) -> String {
        self.0.symbol()
    }

    fn params(&self) -> Vec<EntryParam> {
        let mut params = self.0.params();
        params.push(EntryParam::DevicePointer {
            name: "stale_rows".into(),
            cxx_type: "void".into(),
        });
        params
    }

    fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.0.plan(prepared_rows)
    }

    fn pack_arguments(&self, mut values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        let stale = values.pop().ok_or(Refusal("no stale count"))?;
        let live = self
            .0
            .params
            .iter()
            .position(|param| matches!(param, EntryParam::DevicePointer { name, .. } if name == "live_rows"))
            .ok_or(Refusal("no live count parameter"))?;
        values[live] = stale;
        pack_by_kind(&self.0.params, values)
    }
}

/// Where the runner must refuse a broken copy: the live count, then the
/// output and phase the error names, then the defect it reports.
struct Refused {
    live: u32,
    at: &'static str,
    reason: &'static str,
}

impl Refused {
    /// Whether `error` is this refusal: the runner's errors read
    /// `<symbol> live <n>: <output> <phase>: ... <reason> ...`.
    fn matches(&self, error: &str) -> bool {
        error.contains(&format!(" live {}: {}: ", self.live, self.at))
            && error.contains(self.reason)
    }
}

/// Broken row copies the runner must refuse: (name, the refusal, the
/// case). One writes nothing (width 0), so the first count (64) leaves the
/// output poisoned; one reads a constant count of every row, so it matches
/// at 64 and writes past the live rows at 0, the next count; one reads a
/// constant count of half the rows, so it ignores the device live count and
/// leaves rows 32 to 63 poisoned at 64. Each fails its eager launch.
fn broken_copies<'a>(
    copy: &'a CopyRows,
    stale: &'a StaleCount,
) -> Vec<(&'static str, Refused, Case<'a>)> {
    let count = |rows: u32| words(&[rows as i32], i32::to_le_bytes);
    let capacity = CopyRows::CAPACITY;
    vec![
        (
            "writes nothing",
            Refused {
                live: capacity,
                at: "output eager",
                reason: "did not write",
            },
            copy.case().scalar("width", Argument::U32(0)),
        ),
        (
            "writes past the live rows",
            Refused {
                live: 0,
                at: "output eager",
                reason: "outside the write map changed",
            },
            copy.case_of(stale).input(
                "stale_rows",
                DType::I32,
                (Rows::Fixed(1), 1),
                count(capacity),
            ),
        ),
        (
            "ignores the device live count",
            Refused {
                live: capacity,
                at: "output eager",
                reason: "did not write",
            },
            copy.case_of(stale).input(
                "stale_rows",
                DType::I32,
                (Rows::Fixed(1), 1),
                count(capacity / 2),
            ),
        ),
    ]
}

const BF16: fn(f32) -> u16 = |value| (value.to_bits() >> 16) as u16;

/// The DFlash cache write stores each live row's K and scaled V at its
/// slot's `position % 2048` cell, and skips rows a later row of the same
/// item overwrites. Slots index the caches, so they rerun moved to the
/// top slots.
struct CacheWrite {
    entry: BundleRow,
    width: usize,
    k: Vec<u16>,
    v: Vec<u16>,
}

impl CacheWrite {
    const SLOTS: usize = 32;
    const CACHE: usize = 2048;
    /// (item, position) per row; item 1's first row is 2048 behind its
    /// last.
    const ROWS: [(i32, i32); 5] = [(0, 7), (0, 2055), (1, 3), (1, 2051), (2, 4095)];
    const SLOT_IDS: [i32; 3] = [5, 31, 0];
    const LAST_POSITIONS: [i64; 3] = [2055, 2051, 4095];
    const SCALE: f32 = 0.5;

    fn new(tp: Tp) -> LabResult<Self> {
        let width = tp.dflash_kv_heads() as usize * 128;
        let entry = BundleRow::new(
            tp,
            "dflash_cache_write",
            &[
                "k",
                "v",
                "k_cache",
                "v_cache",
                "slot_ids",
                "token_batch",
                "positions",
                "last_positions",
                "live_rows",
            ],
            &["k_stride", "v_stride", "v_scale_bits"],
            1,
        )?;
        if entry.threads as usize != width / 4 {
            return Err(format!(
                "{} declares {} threads for entry {width}",
                entry.symbol, entry.threads
            ));
        }
        let rows = Self::ROWS.len();
        let k = (0..rows * (width + 64))
            .map(|i| BF16(((i % 97) as f32 - 48.0) / 8.0))
            .collect();
        let v = (0..rows * (width + 32))
            .map(|i| BF16(((i % 89) as f32 - 44.0) / 4.0))
            .collect();
        Ok(Self { entry, width, k, v })
    }

    /// (row, first cache element) of each live row the write keeps.
    fn cells(&self, images: &Images, live: u32) -> LabResult<Vec<(usize, usize)>> {
        let slots = images.i32s("slot_ids")?;
        let batch = images.i32s("token_batch")?;
        let positions = images.i32s("positions")?;
        let last = images
            .bytes("last_positions")?
            .chunks_exact(8)
            .map(|word| i64::from_le_bytes(word.try_into().expect("8 bytes")))
            .collect::<Vec<_>>();
        Ok((0..live as usize)
            .filter_map(|row| {
                let (item, position) = (batch[row] as usize, positions[row]);
                (position >= 0 && i64::from(position) + Self::CACHE as i64 > last[item]).then(
                    || {
                        let cell =
                            slots[item] as usize * Self::CACHE + position as usize % Self::CACHE;
                        (row, cell * self.width)
                    },
                )
            })
            .collect())
    }

    fn map(&self, images: &Images, live: u32) -> LabResult<WriteMap> {
        WriteMap::of(DType::Bf16).ranges(
            self.cells(images, live)?
                .into_iter()
                .map(|(_, cell)| cell..cell + self.width),
        )
    }

    fn case(&self) -> Case<'_> {
        let width = self.width;
        let (k_stride, v_stride) = (width + 64, width + 32);
        let cache = (Rows::Fixed(Self::SLOTS * Self::CACHE), width);
        let slot_bytes = Self::CACHE * width * 2;
        let superseded = "row 0, the only live row, is superseded by item 0's row 1";
        Case::new(&self.entry, Self::ROWS.len() as u32)
            .no_write("k_cache", 1, superseded)
            .no_write("v_cache", 1, superseded)
            .input("k", DType::Bf16, (Rows::Live, k_stride), bytes_u16(&self.k))
            .produced("k")
            .input("v", DType::Bf16, (Rows::Live, v_stride), bytes_u16(&self.v))
            .output(
                "k_cache",
                DType::Bf16,
                cache,
                Write::Map(Box::new(|images, live| self.map(images, live))),
            )
            .output(
                "v_cache",
                DType::Bf16,
                cache,
                Write::Map(Box::new(|images, live| self.map(images, live))),
            )
            .ids(
                "slot_ids",
                (Rows::Fixed(Self::SLOT_IDS.len()), 1),
                &Self::SLOT_IDS,
                &[("k_cache", slot_bytes), ("v_cache", slot_bytes)],
                Holes::Never,
            )
            .input(
                "token_batch",
                DType::I32,
                (Rows::Live, 1),
                words(&Self::ROWS.map(|row| row.0), i32::to_le_bytes),
            )
            .input(
                "positions",
                DType::I32,
                (Rows::Live, 1),
                words(&Self::ROWS.map(|row| row.1), i32::to_le_bytes),
            )
            .input(
                "last_positions",
                DType::I64,
                (Rows::Fixed(Self::LAST_POSITIONS.len()), 1),
                Self::LAST_POSITIONS
                    .iter()
                    .flat_map(|position| position.to_le_bytes())
                    .collect(),
            )
            .scalar("k_stride", Argument::U32(k_stride as u32))
            .scalar("v_stride", Argument::U32(v_stride as u32))
            .scalar("v_scale_bits", Argument::U32(Self::SCALE.to_bits()))
            .oracle(move |images, live| {
                let cache = Self::SLOTS * Self::CACHE * width * 2;
                let (mut expected_k, mut expected_v) = (vec![0_u8; cache], vec![0_u8; cache]);
                let op = dflash::CacheWrite {
                    kv_heads: (width / 128) as u32,
                    head_dim: 128,
                    tokens: Self::CACHE as u32,
                    slots: Self::SLOTS as u32,
                    k_stride: k_stride as u32,
                    v_stride: v_stride as u32,
                    v_scale_bits: Self::SCALE.to_bits(),
                };
                let last = images
                    .bytes("last_positions")?
                    .chunks_exact(8)
                    .map(|word| i64::from_le_bytes(word.try_into().expect("8 bytes")))
                    .collect::<Vec<_>>();
                dflash::cache_write(
                    &op,
                    images.bytes("k")?,
                    images.bytes("v")?,
                    &mut expected_k,
                    &mut expected_v,
                    &images.i32s("slot_ids")?,
                    &images.i32s("token_batch")?,
                    &images.i32s("positions")?,
                    &last,
                    live,
                    Self::ROWS.len() as u32,
                )
                .map_err(|e| e.0.to_string())?;
                Ok(vec![
                    ("k_cache".into(), Expected::Bits(expected_k)),
                    ("v_cache".into(), Expected::Bits(expected_v)),
                ])
            })
    }
}

/// Each draft row takes the best score over every rank's (score, ID) pair,
/// the lowest ID on ties; rows at their item's prefix are not drafts and
/// keep their bytes.
struct Drafts {
    entry: BundleRow,
    ranks: usize,
    gathered: Vec<f32>,
}

impl Drafts {
    const CAPACITY: usize = 40;

    fn new(tp: Tp) -> LabResult<Self> {
        let ranks = tp.count();
        let entry = BundleRow::new(
            tp,
            "dflash_drafts",
            &[
                "gathered",
                "token_ids",
                "positions",
                "prefix_len",
                "token_batch",
                "live_rows",
            ],
            &["capacity"],
            32,
        )?;
        if entry.threads != 32 {
            return Err(format!(
                "{} declares {} threads",
                entry.symbol, entry.threads
            ));
        }
        let capacity = Self::CAPACITY;
        let mut gathered = vec![0.0_f32; ranks * 2 * capacity];
        for rank in 0..ranks {
            for row in 0..capacity {
                // Rows divisible by 3 tie across every rank on the score.
                let score = if row % 3 == 0 {
                    1.0
                } else {
                    ((row * 7 + rank * 13) % 11) as f32
                };
                gathered[rank * 2 * capacity + row] = score;
                gathered[rank * 2 * capacity + capacity + row] =
                    f32::from_bits((1000 + (ranks - rank) * 100 + row) as u32);
            }
        }
        Ok(Self {
            entry,
            ranks,
            gathered,
        })
    }

    /// The live rows that are drafts: not at their item's prefix.
    fn drafts(images: &Images, live: u32) -> LabResult<Vec<usize>> {
        let positions = images.i32s("positions")?;
        let batch = images.i32s("token_batch")?;
        let prefix = images.i32s("prefix_len")?;
        Ok((0..live as usize)
            .filter(|&row| positions[row] != prefix[batch[row] as usize])
            .collect())
    }

    fn case(&self) -> Case<'_> {
        let (capacity, ranks) = (Self::CAPACITY, self.ranks);
        // One item per 8 rows; its prefix is the first row's position.
        let positions = (0..capacity as i32).collect::<Vec<_>>();
        let token_batch = (0..capacity as i32).map(|row| row / 8).collect::<Vec<_>>();
        let prefix_len = (0..capacity as i32).step_by(8).collect::<Vec<_>>();
        Case::new(&self.entry, capacity as u32)
            .live_counts(vec![40, 0, 1, 20, 37, 39])
            .no_write(
                "token_ids",
                1,
                "row 0, the only live row, is its item's prefix",
            )
            .input(
                "gathered",
                DType::F32,
                (Rows::Fixed(ranks * 2), capacity),
                words(&self.gathered, f32::to_le_bytes),
            )
            .produced("gathered")
            .output(
                "token_ids",
                DType::I32,
                (Rows::Live, 1),
                Write::Map(Box::new(|images, live| {
                    WriteMap::of(DType::I32).rows(1, Self::drafts(images, live)?)
                })),
            )
            .input(
                "positions",
                DType::I32,
                (Rows::Live, 1),
                words(&positions, i32::to_le_bytes),
            )
            .input(
                "prefix_len",
                DType::I32,
                (Rows::Fixed(prefix_len.len()), 1),
                words(&prefix_len, i32::to_le_bytes),
            )
            .input(
                "token_batch",
                DType::I32,
                (Rows::Live, 1),
                words(&token_batch, i32::to_le_bytes),
            )
            .scalar("capacity", Argument::U32(capacity as u32))
            .oracle(move |images, live| {
                let mut expected = vec![0_i32; capacity];
                dflash::drafts(
                    &dflash::Drafts {
                        ranks: ranks as u32,
                    },
                    &images.f32s("gathered")?,
                    &mut expected,
                    &images.i32s("positions")?,
                    &images.i32s("prefix_len")?,
                    &images.i32s("token_batch")?,
                    live,
                    capacity as u32,
                )
                .map_err(|e| e.0.to_string())?;
                Ok(vec![(
                    "token_ids".into(),
                    Expected::Bits(words(&expected, i32::to_le_bytes)),
                )])
            })
    }
}

fn bytes_u16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Eager prefill launches the DFlash row copy at the smallest row bucket
/// holding the live rows instead of 8,192 rows (`Phase::row_grid`). The
/// output must match the capacity launch byte for byte, including rows past
/// the live count, which neither launch writes. Sizes include the halves
/// pipelined prefill gives 1,908 (960 + 948) and 3,743 (1,920 + 1,823) tokens.
fn dflash_copy_rows_grids(session: &Session, tp: Tp) -> LabResult<()> {
    const CAPACITY: usize = 8192;
    let symbol = bundle_symbol(tp, "dflash_copy_rows");
    let module = module(session, tp, "dflash")?;
    // A tap's shard of the hidden row into a wider row, and a V copy scaled.
    let (width, input_stride, output_stride) = (1024_usize, 4096_usize, 1536_usize);
    let input = (0..CAPACITY * input_stride)
        .flat_map(|index| {
            ((index as u32).wrapping_mul(2_654_435_761) as u16 & 0x7f7f).to_le_bytes()
        })
        .collect::<Vec<_>>();
    let input_buffer = upload(session, &input)?;
    let marker = vec![0x5a_u8; CAPACITY * output_stride * 2];
    for live in [
        1_u32, 50, 63, 65, 691, 948, 960, 1023, 1025, 1823, 1920, 3743,
    ] {
        let bucket = kernels_sm120::row::smallest_bucket(live).ok_or("row bucket")?;
        let live_rows = upload(session, &live.to_le_bytes())?;
        for scale in [1.0_f32, 0.707] {
            let mut outputs = Vec::new();
            for grid in [CAPACITY as u32, bucket] {
                let output = upload(session, &marker)?;
                launch(
                    session,
                    &module,
                    &symbol,
                    grid,
                    256,
                    &[
                        KernelArgument::Buffer(&input_buffer),
                        KernelArgument::Buffer(&output),
                        KernelArgument::Buffer(&live_rows),
                        KernelArgument::U32(width as u32),
                        KernelArgument::U32(input_stride as u32),
                        KernelArgument::U32(output_stride as u32),
                        KernelArgument::U32(scale.to_bits()),
                    ],
                )?;
                outputs.push(download(session, &output, marker.len())?);
            }
            if outputs[0] != outputs[1] {
                return Err(format!(
                    "{tp:?} DFlash row copy of {live} rows at grid {bucket} differs from 8192"
                ));
            }
            if scale == 1.0 {
                let row = live as usize - 1;
                let copied = &outputs[1][row * output_stride * 2..][..width * 2];
                if copied != &input[row * input_stride * 2..][..width * 2] {
                    return Err(format!("{tp:?} DFlash row copy lost row {row}"));
                }
            }
            if outputs[1][live as usize * output_stride * 2..]
                != marker[live as usize * output_stride * 2..]
            {
                return Err(format!("{tp:?} DFlash row copy wrote past {live} rows"));
            }
        }
    }
    Ok(())
}

/// Every declared case binds each parameter of its entry, sizes its images
/// and writes something at its largest live count, at every rank count.
#[test]
fn bundle_row_cases_are_declared_whole() {
    for tp in Tp::SUPPORTED {
        CopyRows::new(tp).unwrap().case().validate().expect("copy");
        CacheWrite::new(tp)
            .unwrap()
            .case()
            .validate()
            .expect("cache write");
        Drafts::new(tp).unwrap().case().validate().expect("drafts");
    }
}

/// Negative controls: each broken copy is declared whole, so on a GPU only
/// the runner's checks can refuse it, and each refusal names its place.
#[test]
fn broken_row_copies_are_declared_whole() {
    let refusal = Refused {
        live: 0,
        at: "output eager",
        reason: "outside the write map changed",
    };
    assert!(refusal.matches(
        "chop_x live 0: output eager: byte 3 outside the write map changed from 0xff to 0x00"
    ));
    assert!(!refusal.matches(
        "chop_x live 64: output eager: byte 3 outside the write map changed from 0xff to 0x00"
    ));
    assert!(!refusal.matches(
        "chop_x live 0: PDL copy producer: output PDL: byte 3 outside the write map changed"
    ));
    let tp = Tp::SUPPORTED[0];
    let copy = CopyRows::new(tp).unwrap();
    let stale = StaleCount(CopyRows::new(tp).unwrap().entry);
    for (name, _, case) in broken_copies(&copy, &stale) {
        case.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// The runner refuses each broken copy with the error its defect causes, at
/// the live count, output and phase where it first shows.
#[test]
fn the_runner_refuses_broken_row_copies_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let lab = session.module(&latent_shapes::cubin_name())?;
        let tp = Tp::SUPPORTED[0];
        let dflash = module(session, tp, "dflash")?;
        let copy = CopyRows::new(tp)?;
        let stale = StaleCount(CopyRows::new(tp)?.entry);
        for (name, refused, case) in broken_copies(&copy, &stale) {
            match case.run(session, &dflash, Some(&lab)) {
                Ok(_) => return Err(format!("the runner passed a copy that {name}")),
                Err(error) if refused.matches(&error) => eprintln!("{name}: refused: {error}"),
                Err(error) => {
                    return Err(format!(
                        "a copy that {name} failed otherwise than \"live {}: {}: ... {}\": {error}",
                        refused.live, refused.at, refused.reason
                    ));
                }
            }
        }
        Ok(())
    })
    .expect("broken row copies");
}

#[test]
fn mimo_bundle_row_kernels_at_every_rank_count_gpu() {
    kernel_lab::gpu_test(2 << 30, |session| {
        let lab = session.module(&latent_shapes::cubin_name())?;
        for tp in Tp::SUPPORTED {
            let dflash = module(session, tp, "dflash")?;
            CopyRows::new(tp)?
                .case()
                .run(session, &dflash, Some(&lab))
                .map_err(|e| format!("{tp:?} DFlash row copy: {e}"))?;
            CacheWrite::new(tp)?
                .case()
                .run(session, &dflash, Some(&lab))
                .map_err(|e| format!("{tp:?} DFlash cache write: {e}"))?;
            Drafts::new(tp)?
                .case()
                .run(session, &dflash, Some(&lab))
                .map_err(|e| format!("{tp:?} DFlash drafts: {e}"))?;
            dflash_copy_rows_grids(session, tp)?;
        }
        Ok(())
    })
    .expect("MiMo bundle row kernels");
}
