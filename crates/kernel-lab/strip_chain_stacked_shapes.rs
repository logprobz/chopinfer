//! Lab-only stacked strip-chain GEMM instances (`gemm::stacked`,
//! strip_chains_stacked.cuh), L2 weight prefetch entries (`gemm::prefetch`)
//! and the entries they are checked and timed against, shared by the
//! `strip_chains_stacked` lock family and `tests/strip_chains_stacked.rs`.
//!
//! - Stacks: BF16 decode projections that read one activation at K 4096,
//!   by shape: three n2048 parts writing thirds of one `[rows, 6144]`
//!   buffer (`n2048x3`), and those with two n128 parts and an n64 part
//!   whose 16 leading columns are written compacted (`n2048x3_n128_n64w16_n128`).
//!   Each at 1, 2, 4, 8, 16 and 32 rows in 1, 2 and 4 strips at one split
//!   and 1 and 2 strips at two splits, each without and with the L2
//!   prefetch of its later chunks. One cubin per stack.
//! - Parts: per part shape and capacity, the one-split `Tiled` entry whose
//!   bits every strip schedule keeps (`one`) and the strip-chain schedule a
//!   decode bucket of that shape runs today (`served`, kernel-lab family
//!   `strip_chains`' measured fastest), with the exact layout copies that
//!   place today's contiguous outputs where a stack writes them.
//! - Prefetch: the bulk and per-line L2 prefetch entries, the bulk entry
//!   that never waits for its predecessor (probe), and an idle window
//!   kernel standing in for the kernels a prefetch overlaps.
//! - Mutants: the stacked kernel's deliberate defects
//!   (`chop::gemm::StackedMutation`) at one and two splits.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{DType, DenseKernel, InstantiationSpec, Target};
use kernels_sm120::gemm::prefetch::{self, Mode, Prefetch};
use kernels_sm120::gemm::stacked::{self, Part, Stacked};
use kernels_sm120::gemm::{self, DenseRequest};
use kernels_sm120::precision::generated;
use kernels_sm120::row::precision::{self as row, LayoutCopy};

/// The lock family of the cubins in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "strip_chains_stacked";

/// The part references' cubin.
pub const PARTS_CUBIN: &str = "strip_chains_stacked_parts_sm120";
/// The layout copies' cubin.
pub const COPIES_CUBIN: &str = "strip_chains_stacked_copies_sm120";
/// The prefetch entries' cubin.
pub const PREFETCH_CUBIN: &str = "strip_chains_stacked_prefetch_sm120";
/// The idle window's cubin.
pub const WINDOW_CUBIN: &str = "strip_chains_stacked_window_sm120";
/// The mutants' cubin.
pub const MUTANTS_CUBIN: &str = "strip_chains_stacked_mutants_sm120";

pub const K: u32 = 4096;
/// Row capacities: every decode and verify bucket.
pub const CAPACITIES: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// (strips, splits) of the stacked candidates.
pub const SCHEDULES: [(u8, u8); 5] = [(1, 1), (2, 1), (4, 1), (1, 2), (2, 2)];
/// Most threads per prefetch CTA.
pub const PREFETCH_THREADS: u32 = 256;
pub const IDLE_WINDOW: &str = "chop_lab_idle_window_v1";
pub const PREFETCH_BULK: &str = "chop_lab_prefetch_bulk_v1";
pub const PREFETCH_LINES: &str = "chop_lab_prefetch_lines_v1";
pub const PREFETCH_NO_WAIT: &str = "chop_lab_prefetch_no_wait_v1";

/// `chop::gemm::StackedMutation` values; the first is the unmodified kernel.
pub const MUTATIONS: [&str; 6] = [
    "none",
    "drop_wait",
    "late_wait",
    "shared_slots",
    "keep_ticket",
    "first_map",
];
/// The mutations that apply at one split.
pub const ONE_SPLIT_MUTATIONS: [&str; 4] = ["none", "drop_wait", "late_wait", "first_map"];

/// The stacks by tag: their parts in launch order.
pub fn stacks() -> Vec<(&'static str, Vec<Part>)> {
    let third = Part {
        rows: 2048,
        width: 2048,
        row_stride: 3 * 2048,
    };
    vec![
        ("n2048x3", vec![third; 3]),
        (
            "n2048x3_n128_n64w16_n128",
            vec![
                third,
                third,
                third,
                Part::contiguous(128),
                Part {
                    rows: 64,
                    width: 16,
                    row_stride: 16,
                },
                Part::contiguous(128),
            ],
        ),
    ]
}

/// The cubin of stack `tag`.
pub fn cubin_name(tag: &str) -> String {
    format!("strip_chains_stacked_{tag}_sm120")
}

/// The strip schedule (strips, splits) a decode bucket of `capacity` rows
/// over an `n` x 4096 BF16 weight runs today: family `strip_chains`'
/// fastest measured candidate.
pub fn served_schedule(n: u32, capacity: u32) -> (u8, u8) {
    match n {
        2048 => (2, 1),
        128 if capacity <= 8 => (1, 23),
        128 => (1, 47),
        64 => (1, 64),
        _ => panic!("no served strip schedule for n{n}"),
    }
}

fn make(query: kernel_core::contract::GemmQuery, kernel: DenseKernel, symbol: String) -> InstantiationSpec {
    gemm::instantiate(DenseRequest {
        query,
        kernel,
        target: Target::Sm120f,
        symbol,
        knobs: None,
    })
    .expect("lab stacked-part instances are valid")
}

/// A part shape's reference entries at one capacity.
pub struct PartEntries {
    pub n: u32,
    pub capacity: u32,
    /// One-split `Tiled`: the bits every strip schedule keeps.
    pub one: InstantiationSpec,
    /// Today's strip-chain schedule of the shape and bucket.
    pub served: InstantiationSpec,
}

/// The part references of `n` rows at `capacity` rows.
pub fn part_entries(n: u32, capacity: u32) -> PartEntries {
    let query = gemm::bf16_query(capacity, n, K, DType::Bf16);
    let name = |what: &str| format!("chop_lab_stacked_part_n{n}_m{capacity}_{what}_v1");
    let (strips, splits) = served_schedule(n, capacity);
    PartEntries {
        n,
        capacity,
        one: make(query, DenseKernel::Tiled { splits: 1 }, name("one")),
        served: make(
            query,
            DenseKernel::StripChains { strips, splits },
            name("served"),
        ),
    }
}

/// The distinct part shapes of every stack.
pub fn part_shapes() -> Vec<u32> {
    let mut shapes: Vec<u32> = stacks()
        .into_iter()
        .flat_map(|(_, parts)| parts.into_iter().map(|p| p.rows))
        .collect();
    shapes.sort_unstable();
    shapes.dedup();
    shapes
}

/// The exact copy of a part's contiguous `[rows, n]` output into the
/// place a stack writes it (`None` where the stack's place is contiguous):
/// a third of a wider row, or the leading columns compacted.
pub fn layout_copy(part: &Part, column_offset: u32, capacity: u32) -> Option<LayoutCopy> {
    if part.width == part.rows && part.row_stride == part.rows {
        return None;
    }
    let (copy, _) = generated::layout_copy(
        part.width,
        capacity,
        row::CopyRows {
            row_stride: part.rows,
            column_stride: 1,
            column_offset: 0,
        },
        row::CopyRows {
            row_stride: part.row_stride,
            column_stride: 1,
            column_offset,
        },
        row::ShapeBinding::Compiled,
        &[],
    )
    .expect("lab layout copies are valid");
    Some(copy)
}

/// Each part's output buffer slot (parts sharing a strided buffer share
/// one) and column offset in it: parts with the same row stride above
/// their width fill one buffer side by side.
pub fn placements(parts: &[Part]) -> Vec<(usize, u32)> {
    let mut placed: Vec<(usize, u32)> = Vec::new();
    let mut buffers: Vec<(u32, u32)> = Vec::new(); // (row stride, used columns)
    for part in parts {
        let shared = part.row_stride > part.width;
        let slot = buffers
            .iter()
            .position(|&(stride, used)| shared && stride == part.row_stride && used < stride);
        match slot {
            Some(slot) => {
                placed.push((slot, buffers[slot].1));
                buffers[slot].1 += part.width;
            }
            None => {
                placed.push((buffers.len(), 0));
                buffers.push((part.row_stride, part.width));
            }
        }
    }
    placed
}

/// One stack at one capacity.
pub struct Case {
    pub stack: &'static str,
    pub tag: String,
    pub capacity: u32,
    pub parts: Vec<Part>,
    pub candidates: Vec<Stacked>,
}

/// Every stacked candidate, stacks then capacities.
pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (stack, parts) in stacks() {
        for capacity in CAPACITIES {
            let candidates = SCHEDULES
                .iter()
                .flat_map(|&(strips, splits)| {
                    [false, true].map(|prefetch| Stacked {
                        symbol: format!(
                            "chop_lab_stacked_{stack}_m{capacity}_s{strips}x{splits}{}_v1",
                            if prefetch { "_pf" } else { "" }
                        ),
                        capacity,
                        k: K,
                        output: DType::Bf16,
                        parts: parts.clone(),
                        strips,
                        splits,
                        prefetch,
                    })
                })
                .collect();
            cases.push(Case {
                stack,
                tag: format!("{stack}_m{capacity}"),
                capacity,
                parts: parts.clone(),
                candidates,
            });
        }
    }
    cases
}

/// The mutant instances: the three-part stack at 4 rows in 2 strips at
/// two splits (every mutation) and at one split (those that apply).
pub fn mutants() -> Vec<(String, Stacked)> {
    let (stack, parts) = stacks().remove(0);
    [(2u8, &MUTATIONS[..]), (1, &ONE_SPLIT_MUTATIONS[..])]
        .into_iter()
        .flat_map(|(splits, mutations)| {
            let parts = parts.clone();
            mutations.iter().map(move |&mutation| {
                (
                    mutation.to_owned(),
                    Stacked {
                        symbol: format!("chop_lab_stacked_{mutation}_{stack}_m4_s2x{splits}_v1"),
                        capacity: 4,
                        k: K,
                        output: DType::Bf16,
                        parts: parts.clone(),
                        strips: 2,
                        splits,
                        prefetch: false,
                    },
                )
            })
        })
        .collect()
}

/// The prefetch entries: bulk, per line, and bulk without its wait.
pub fn prefetches() -> Vec<Prefetch> {
    vec![
        Prefetch {
            symbol: PREFETCH_BULK.into(),
            mode: Mode::Bulk,
        },
        Prefetch {
            symbol: PREFETCH_LINES.into(),
            mode: Mode::Lines,
        },
        Prefetch {
            symbol: PREFETCH_NO_WAIT.into(),
            mode: Mode::Bulk,
        },
    ]
}

/// The family's cubins: one per stack, the parts' references and copies,
/// the prefetch entries, the idle window and the mutants.
#[allow(dead_code)] // Read by the build script only.
pub fn cubins() -> Vec<CubinSpec> {
    let cases = cases();
    let mut cubins: Vec<CubinSpec> = stacks()
        .into_iter()
        .map(|(stack, _)| {
            let entries: Vec<Stacked> = cases
                .iter()
                .filter(|case| case.stack == stack)
                .flat_map(|case| case.candidates.iter().cloned())
                .collect();
            stacked::cubin_spec(&cubin_name(stack), &entries).expect("lab stacked cubin is valid")
        })
        .collect();
    let references: Vec<InstantiationSpec> = part_shapes()
        .into_iter()
        .flat_map(|n| CAPACITIES.map(|capacity| part_entries(n, capacity)))
        .flat_map(|entries| [entries.one, entries.served])
        .collect();
    let mut parts = gemm::cubin_spec(&references).expect("lab stacked-part cubin is valid");
    parts.name = PARTS_CUBIN.into();
    // Each distinct copy once (uses at several capacities share a symbol).
    let mut copies = std::collections::BTreeMap::new();
    for (_, stack_parts) in stacks() {
        let placed = placements(&stack_parts);
        for capacity in CAPACITIES {
            for (part, &(_, offset)) in stack_parts.iter().zip(&placed) {
                if let Some(copy) = layout_copy(part, offset, capacity) {
                    copies
                        .entry(copy.stage().entry().symbol.clone())
                        .or_insert_with(|| copy.stage().clone());
                }
            }
        }
    }
    let stages: Vec<_> = copies.into_values().collect();
    cubins.push(parts);
    cubins.push(row::cubin_spec(COPIES_CUBIN, &stages).expect("lab copy cubin is valid"));
    let mut prefetch = prefetch::cubin_spec(PREFETCH_CUBIN, &prefetches(), PREFETCH_THREADS);
    for entry in &mut prefetch.entries {
        if entry.symbol == PREFETCH_NO_WAIT {
            entry.body = entry
                .body
                .replace(">(ranges", ", chop::gemm::PrefetchMutation::no_wait>(ranges");
        }
    }
    cubins.push(prefetch);
    cubins.push(CubinSpec {
        name: WINDOW_CUBIN.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_dirs: vec![],
        headers: vec!["#include \"kernel-lab/cuda/probes/idle_window.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            IDLE_WINDOW.into(),
            vec![EntryParam::U64 { name: "ns".into() }],
            "chop::probe::idle_window(ns);".into(),
            32,
            1,
        )],
    });
    let mutants = mutants();
    let specs: Vec<Stacked> = mutants.iter().map(|(_, spec)| spec.clone()).collect();
    let mut cubin =
        stacked::cubin_spec(MUTANTS_CUBIN, &specs).expect("lab stacked mutant cubin is valid");
    for (entry, (mutation, _)) in cubin.entries.iter_mut().zip(&mutants) {
        assert_eq!(entry.body.matches(">(parts").count(), 1, "{}", entry.symbol);
        entry.body = entry.body.replace(
            ">(parts",
            &format!(", chop::gemm::StackedMutation::{mutation}>(parts"),
        );
    }
    cubins.push(cubin);
    cubins
}

/// Every cubin name the family declares.
#[allow(dead_code)] // Read by the family file only.
pub fn cubin_names() -> Vec<String> {
    stacks()
        .into_iter()
        .map(|(stack, _)| cubin_name(stack))
        .chain(
            [
                PARTS_CUBIN,
                COPIES_CUBIN,
                PREFETCH_CUBIN,
                WINDOW_CUBIN,
                MUTANTS_CUBIN,
            ]
            .map(str::to_owned),
        )
        .collect()
}
