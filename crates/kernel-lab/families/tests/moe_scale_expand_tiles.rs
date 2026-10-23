//! The tiled paired scale expansion against the paired entry followed by the
//! row-tile transform, byte for byte, and against the codec. Build the
//! `moe_scale_expand` and `moe_scale_expand_tiles` families with
//! `build_family`; set CHOPINFER_SCALE_EXPAND_CUBINS and
//! CHOPINFER_SCALE_EXPAND_TILES_CUBINS to their external output directories
//! when running on an idle GPU.

use kernel_core::backend::View;
use kernel_core::contract::Argument;
use kernel_core::precision::scale_offset::{self as codec, decode_bank_matrix};
use kernel_core::transform::RowTileLayout;
use kernel_harness::{LabResult, Session, launch, poison};
use kernel_lab_families::{moe_scale_expand, moe_scale_expand_tiles};
use kernels_sm120::moe::scale_expand::{self as expand, PairBank};
use runtime_cuda::{DeviceBuffer, KernelArgument, Module};
use std::path::PathBuf;

const BANK_GUARD: usize = 16;
// The tiled entry writes words: the outputs stay 4-byte aligned.
const OUTPUT_GUARD: usize = 16;
const TAIL_GUARD: usize = 36;

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Constant,
    Random,
    EveryByte,
    Wrapping,
}

/// One bank of `experts * bank.projections` matrices: its encoded bytes
/// and raw scales.
struct Fixture {
    bank: PairBank,
    experts: u32,
    encoded: Vec<u8>,
    raw: Vec<u8>,
}

impl Fixture {
    fn new(bank: PairBank, experts: u32, pattern: Pattern) -> Self {
        let (rows, groups) = (bank.rows as usize, bank.groups as usize);
        let matrix_bytes = rows * groups;
        let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ u64::from(bank.width);
        let mut blocks = Vec::new();
        let mut raw = Vec::new();
        for matrix in 0..(experts * bank.projections) as usize {
            let block = if matches!(pattern, Pattern::Wrapping) {
                // Bases and offsets that wrap, and exceptions on both sides
                // of every strip edge and at each tile column.
                let fixed = codec::fixed_bytes(rows as u64, groups as u64, bank.width).unwrap();
                let row_bytes = codec::plane_row_bytes(groups as u64, bank.width) as usize;
                let mut block = vec![0; fixed as usize];
                block[..rows].fill(255);
                block[rows..rows + rows * row_bytes].fill(255);
                let mut indices = vec![0, 3, 4, groups - 1, matrix_bytes - 1];
                for strip in (64..rows).step_by(64) {
                    indices.extend([strip * groups - 1, strip * groups, strip * groups + 5]);
                }
                indices.sort_unstable();
                indices.dedup();
                for (n, index) in indices.into_iter().enumerate() {
                    block.extend_from_slice(
                        &(((index as u32) << 8) | (253 + n as u32 % 3)).to_le_bytes(),
                    );
                }
                block
            } else {
                let bytes: Vec<u8> = (0..matrix_bytes)
                    .map(|i| match pattern {
                        Pattern::Constant => (matrix * 31) as u8,
                        Pattern::Random => {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            state as u8
                        }
                        Pattern::EveryByte => (i + matrix * 37) as u8,
                        Pattern::Wrapping => unreachable!(),
                    })
                    .collect();
                codec::encode(&bytes, rows as u64, groups as u64, bank.width).unwrap()
            };
            raw.extend(codec::decode(&block, rows as u64, groups as u64, bank.width).unwrap());
            blocks.push(block);
        }
        let layout =
            codec::bank_layout(&blocks.iter().map(|b| b.len() as u64).collect::<Vec<_>>()).unwrap();
        let mut encoded = layout.table;
        for block in &blocks {
            encoded.extend_from_slice(block);
        }
        Self {
            bank,
            experts,
            encoded,
            raw,
        }
    }

    /// The row-tile transform of this bank's raw scales: the
    /// `row-tiles-64x128-groups` scale section over every matrix.
    fn tiles(&self) -> RowTileLayout {
        RowTileLayout {
            rows: (self.experts * self.bank.projections * self.bank.rows) as usize,
            row_bytes: self.bank.groups as usize,
            tile_rows: 64,
            tile_bytes: 4,
        }
    }

    /// The guarded output image before and after an expansion of the
    /// experts `ids` lists (all without a list), rows or tiled: the codec's
    /// matrices where listed, each other byte its initial value, which
    /// differs from every expected write.
    fn images(&self, ids: Option<&[u32]>, tiled: bool) -> (Vec<u8>, Vec<u8>) {
        let matrix_bytes = (self.bank.rows * self.bank.groups) as usize;
        let mut initial = vec![0; self.raw.len()];
        let mut expected = initial.clone();
        for expert in 0..self.experts {
            let listed = ids.is_none_or(|ids| ids.contains(&expert));
            for projection in 0..self.bank.projections {
                let matrix = (expert * self.bank.projections + projection) as usize;
                let at = matrix * matrix_bytes..(matrix + 1) * matrix_bytes;
                let decoded = decode_bank_matrix(
                    &self.encoded,
                    matrix,
                    self.bank.rows.into(),
                    self.bank.groups.into(),
                    self.bank.width,
                )
                .unwrap();
                assert_eq!(decoded, self.raw[at.clone()]);
                for ((before, after), byte) in initial[at.clone()]
                    .iter_mut()
                    .zip(&mut expected[at.clone()])
                    .zip(decoded)
                {
                    *before = !byte;
                    *after = if listed { byte } else { !byte };
                }
            }
        }
        if tiled {
            let mut scratch = Vec::new();
            for image in [&mut initial, &mut expected] {
                self.tiles().pack_in_place(image, &mut scratch).unwrap();
            }
        }
        let guard = |image: Vec<u8>| {
            let mut guarded = vec![0xa5; OUTPUT_GUARD];
            guarded.extend(image);
            guarded.resize(guarded.len() + TAIL_GUARD, 0x5a);
            guarded
        };
        (guard(initial), guard(expected))
    }
}

/// One case: its expert count and its two banks, either of which may be
/// absent.
type Case = (u32, [Option<(PairBank, Pattern)>; 2]);

/// Gate/up-like first banks of whole tiles at every width, among them
/// Flash's TP4 gate/up bank (576 rows, 160 groups, 2 projections) beside
/// its down bank: one-strip tiles, multi-strip tiles, a last tile shorter
/// than the rest, a lone first bank, and record lists from none to nearly
/// every scale.
fn cases() -> Vec<Case> {
    let bank = |width, rows, groups, projections| PairBank {
        width,
        rows,
        groups,
        projections,
    };
    let mut cases = Vec::new();
    for (first, second) in [(1, 4), (2, 1), (4, 2)] {
        cases.push((
            5,
            [
                Some((bank(first, 576, 160, 2), Pattern::Random)),
                Some((bank(second, 5120, 18, 1), Pattern::Random)),
            ],
        ));
        cases.push((
            3,
            [
                Some((bank(first, 192, 64, 2), Pattern::Wrapping)),
                Some((bank(second, 65, 17, 1), Pattern::EveryByte)),
            ],
        ));
        cases.push((
            2,
            [Some((bank(first, 1088, 4, 1), Pattern::EveryByte)), None],
        ));
    }
    cases.push((4, [Some((bank(1, 640, 12, 2), Pattern::Wrapping)), None]));
    cases.push((
        2,
        [
            Some((bank(2, 64, 160, 2), Pattern::Constant)),
            Some((bank(4, 512, 18, 1), Pattern::Constant)),
        ],
    ));
    cases
}

fn active_lists(experts: u32) -> Vec<Option<Vec<u32>>> {
    vec![
        Some((0..experts).rev().collect()),
        Some(vec![experts - 1, 0]),
        Some(vec![]),
        None,
    ]
}

fn active_bytes(experts: u32, ids: &[u32]) -> Vec<u8> {
    let mut words = vec![-1_i32; experts as usize + 1];
    words[0] = ids.len() as i32;
    for (word, id) in words[1..].iter_mut().zip(ids) {
        *word = *id as i32;
    }
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

#[test]
fn cpu_tiled_fixtures_cover_strips_tiles_and_records() {
    let cases = cases();
    assert_eq!(cases.len(), 11);
    let firsts: Vec<PairBank> = cases
        .iter()
        .filter_map(|(_, [first, _])| first.map(|(bank, _)| bank))
        .collect();
    assert_eq!(firsts.len(), cases.len());
    for bank in &firsts {
        assert!(bank.rows.is_multiple_of(64) && bank.groups.is_multiple_of(4));
        assert!(expand::pair_tiles_plan([Some(*bank), None], 2, 2).is_ok());
    }
    // One-strip tiles, multi-strip tiles, and a last tile shorter than the
    // rest.
    assert!(firsts.iter().any(|b| b.tile_rows() == 64));
    assert!(firsts.iter().any(|b| b.tile_rows() > 64));
    assert!(firsts.iter().any(|b| !b.rows.is_multiple_of(b.tile_rows())));
    for width in [1, 2, 4] {
        assert!(firsts.iter().any(|b| b.width == width));
    }
    let (count, mut runs) = (cases.len(), 0);
    for (experts, banks) in cases {
        runs += active_lists(experts).len();
        // Only the first bank is written in tiles.
        for (at, (bank, pattern)) in banks
            .into_iter()
            .enumerate()
            .filter_map(|(at, bank)| Some((at, bank?)))
        {
            let f = Fixture::new(bank, experts, pattern);
            let layouts: &[bool] = if at == 0 { &[false, true] } else { &[false] };
            for ids in active_lists(experts) {
                for &tiled in layouts {
                    let (initial, expected) = f.images(ids.as_deref(), tiled);
                    let core = OUTPUT_GUARD..OUTPUT_GUARD + f.raw.len();
                    let writes = initial[core.clone()]
                        .iter()
                        .zip(&expected[core.clone()])
                        .filter(|(a, b)| a != b)
                        .count();
                    let listed = ids.as_ref().map_or(experts as usize, Vec::len);
                    assert_eq!(writes, f.raw.len() / experts as usize * listed);
                }
                if at != 0 {
                    continue;
                }
                // The tiled image is the row image's transform.
                let mut rows = f.images(ids.as_deref(), false).1;
                f.tiles()
                    .pack_in_place(
                        &mut rows[OUTPUT_GUARD..OUTPUT_GUARD + f.raw.len()],
                        &mut Vec::new(),
                    )
                    .unwrap();
                assert_eq!(rows, f.images(ids.as_deref(), true).1);
            }
        }
    }
    // The GPU test's runs: both entries over each case and active list.
    assert_eq!(runs, 44);
    println!("scale_expand_tiles_plan {{\"cases\":{count},\"runs\":{runs}}}");
}

fn load(session: &Session, variable: &str, cubin: &str) -> LabResult<Module> {
    let directory = std::env::var_os(variable)
        .map(PathBuf::from)
        .ok_or(format!("{variable} is required"))?;
    kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
        .and_then(|r| r.verify_current())
        .map_err(|e| e.to_string())?;
    let bytes =
        std::fs::read(directory.join(format!("{cubin}.cubin"))).map_err(|e| e.to_string())?;
    session.load_cubin(&bytes)
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len())?;
    poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

fn download(session: &Session, buffer: &DeviceBuffer) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

/// One launch of the paired (`tiled` false) or tiled paired entry over
/// `fixtures` with active list `ids`, from each output's initial image: the
/// guarded outputs after it.
fn expand_once(
    session: &Session,
    function: &runtime_cuda::Function,
    fixtures: [Option<&Fixture>; 2],
    ids: Option<&[u32]>,
    tiled: bool,
) -> LabResult<Vec<Vec<u8>>> {
    let present: Vec<&Fixture> = fixtures.iter().flatten().copied().collect();
    let experts = present[0].experts;
    let mut banks = Vec::new();
    let mut outputs = Vec::new();
    for (at, f) in fixtures.iter().enumerate() {
        let Some(f) = f else { continue };
        let mut image = vec![0x6d; BANK_GUARD];
        image.extend_from_slice(&f.encoded);
        image.resize(image.len() + TAIL_GUARD, 0x6d);
        banks.push(upload(session, &image)?);
        outputs.push(upload(session, &f.images(ids, tiled && at == 0).0)?);
    }
    let active = upload(session, &active_bytes(experts, ids.unwrap_or(&[])))?;
    let at = |b: usize| fixtures[..b].iter().flatten().count() as u64;
    let views = |b: usize| {
        fixtures[b].map(|f| {
            (
                f.bank,
                View {
                    handle: 10 + at(b),
                    bytes: f.encoded.len() as u64,
                },
                View {
                    handle: 20 + at(b),
                    bytes: f.raw.len() as u64,
                },
            )
        })
    };
    let slots = ids.map_or(experts, |ids| ids.len().max(1) as u32);
    let list = ids.is_some().then_some(View {
        handle: 2,
        bytes: active.byte_len() as u64,
    });
    let banks_only = [0, 1].map(|b| views(b).map(|v| v.0));
    let (plan, args) = if tiled {
        (
            expand::pair_tiles_plan(banks_only, experts, slots),
            expand::pair_tiles_arguments([views(0), views(1)], list, experts),
        )
    } else {
        (
            expand::pair_plan(banks_only, experts, slots),
            expand::pair_arguments([views(0), views(1)], list, experts),
        )
    };
    let config = launch::config(plan.map_err(|e| e.to_string())?)?;
    let args = args
        .map_err(|e| e.to_string())?
        .0
        .iter()
        .map(|a| match a {
            Argument::DeviceBuffer(2) => KernelArgument::Buffer(&active),
            Argument::DeviceBuffer(h) if (10..20).contains(h) => {
                let at = (*h - 10) as usize;
                KernelArgument::Range {
                    buffer: &banks[at],
                    offset: BANK_GUARD,
                    byte_len: present[at].encoded.len(),
                }
            }
            Argument::DeviceBuffer(h) => {
                let at = (*h - 20) as usize;
                KernelArgument::Range {
                    buffer: &outputs[at],
                    offset: OUTPUT_GUARD,
                    byte_len: present[at].raw.len(),
                }
            }
            Argument::U32(v) => KernelArgument::U32(*v),
            Argument::NullDevicePointer => KernelArgument::NullBuffer,
            _ => unreachable!("paired scale expansion ABI"),
        })
        .collect::<Vec<_>>();
    launch::launch_with(function, &session.stream, config, &args)?;
    let mut images = Vec::new();
    for output in &outputs {
        images.push(download(session, output)?);
    }
    for (bank, f) in banks.iter().zip(&present) {
        assert_eq!(
            &download(session, bank)?[BANK_GUARD..][..f.encoded.len()],
            f.encoded
        );
    }
    Ok(images)
}

#[test]
fn tiled_pair_is_the_paired_entry_then_row_tiles_on_gpu() {
    let ran = kernel_harness::gpu_test(256 << 20, |session| {
        let rows_module = load(
            session,
            "CHOPINFER_SCALE_EXPAND_CUBINS",
            moe_scale_expand::CUBIN,
        )?;
        let tiles_module = load(
            session,
            "CHOPINFER_SCALE_EXPAND_TILES_CUBINS",
            moe_scale_expand_tiles::CUBIN,
        )?;
        let rows = rows_module
            .function(expand::PAIR_SYMBOL)
            .map_err(|e| e.to_string())?;
        let tiles = tiles_module
            .function(expand::PAIR_TILES_SYMBOL)
            .map_err(|e| e.to_string())?;
        let mut runs = 0;
        for (experts, case) in cases() {
            let fixtures = case.map(|c| c.map(|(bank, p)| Fixture::new(bank, experts, p)));
            let fixtures = [fixtures[0].as_ref(), fixtures[1].as_ref()];
            for ids in active_lists(experts) {
                let ids = ids.as_deref();
                let by_rows = expand_once(session, &rows, fixtures, ids, false)?;
                let by_tiles = expand_once(session, &tiles, fixtures, ids, true)?;
                for (at, f) in fixtures.iter().flatten().enumerate() {
                    let tiled = at == 0;
                    let what = format!("bank {at} of {:?}, ids={ids:?}", f.bank);
                    assert_eq!(by_rows[at], f.images(ids, false).1, "rows, {what}");
                    // The paired entry's output, then the transform.
                    let mut transformed = by_rows[at].clone();
                    if tiled {
                        f.tiles()
                            .pack_in_place(
                                &mut transformed[OUTPUT_GUARD..OUTPUT_GUARD + f.raw.len()],
                                &mut Vec::new(),
                            )
                            .map_err(|e| e.to_string())?;
                    }
                    assert_eq!(by_tiles[at], transformed, "tiles, {what}");
                    assert_eq!(by_tiles[at], f.images(ids, tiled).1, "codec, {what}");
                }
                runs += 1;
            }
        }
        Ok(runs)
    })
    .unwrap();
    // Each run compares both entries over one case and active list.
    if let Some(runs) = ran {
        assert_eq!(runs, 44);
        let count = cases().len();
        println!("scale_expand_tiles_done {{\"cases\":{count},\"runs\":{runs}}}");
    }
}
