//! CPU design checks, not a CUDA candidate or proof of compiled GPU bits.
//! Keep the current m16n8k16 atoms; change only which independent heads
//! share a CTA or a loaded A fragment. GPU comparison remains mandatory.
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy)]
enum Layout {
    PairedRows,
    HeadGroups,
    HeadFragments,
}

#[derive(Clone, Copy)]
struct Worker {
    row: usize,
    head0: usize,
    warp: usize,
    lane: usize,
    fragments: usize,
}

fn workers(layout: Layout, live: usize) -> Vec<Worker> {
    let (rows_per_cta, groups, fragments, head_ctas) = match layout {
        Layout::PairedRows => (2, 2, 1, 2),
        Layout::HeadGroups => (1, 2, 1, 1),
        Layout::HeadFragments => (1, 1, 2, 1),
    };
    let mut result = Vec::new();
    for x in 0..live.div_ceil(rows_per_cta) {
        for y in 0..head_ctas {
            for tid in 0..groups * 128 {
                let group = tid / 128;
                let (row, head0) = match layout {
                    Layout::PairedRows => (x * 2 + group, y * 8),
                    Layout::HeadGroups => (x, group * 8),
                    Layout::HeadFragments => (x, 0),
                };
                if row < live {
                    result.push(Worker {
                        row,
                        head0,
                        warp: tid % 128 / 32,
                        lane: tid % 32,
                        fragments,
                    });
                }
            }
        }
    }
    result
}

type Address = (usize, usize, usize);
fn ownership(layout: Layout, origins: &[usize], query: bool) -> BTreeMap<Address, usize> {
    let mut result = BTreeMap::new();
    for w in workers(layout, origins.len()) {
        let row = origins[w.row];
        let g = w.lane / 4;
        let c = w.lane % 4;
        for n in 0..w.fragments {
            let head0 = w.head0 + n * 8;
            if query {
                for ks in 0..8 {
                    for d in [0, 1, 8, 9] {
                        *result
                            .entry((row, head0 + g, w.warp * 128 + ks * 16 + 2 * c + d))
                            .or_default() += 1;
                    }
                }
            } else {
                for r in 0..2 {
                    let head = head0 + 2 * c + r;
                    if w.warp == 0 && g == 0 {
                        for field in 0..2 {
                            *result.entry((row, head, field)).or_default() += 1;
                        }
                    }
                    for m in 0..8 {
                        for half in 0..2 {
                            *result
                                .entry((row, head, 2 + w.warp * 128 + m * 16 + g + half * 8))
                                .or_default() += 1;
                        }
                    }
                }
            }
        }
    }
    result
}

#[test]
fn head_sharing_preserves_every_query_operand_and_partial_owner() {
    // Origins can be nonconsecutive after retention or from either row part.
    let origins = [0, 4095, 24576, 28672, 32767];
    for live in 0..=origins.len() {
        for query in [false, true] {
            let reference = ownership(Layout::PairedRows, &origins[..live], query);
            assert_eq!(reference.len(), live * 16 * if query { 512 } else { 514 });
            assert!(reference.values().all(|&owners| owners == 1));
            for layout in [Layout::HeadGroups, Layout::HeadFragments] {
                assert_eq!(ownership(layout, &origins[..live], query), reference);
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum Atom {
    Qk { keys: [i32; 16], first_dim: usize },
    Pv { keys: [i32; 16], first_dim: usize },
}

type Chain = (usize, usize, usize, usize, usize);
fn chains(layout: Layout, lists: &[Vec<i32>], pv: bool) -> BTreeMap<Chain, Vec<Atom>> {
    let mut result: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for w in workers(layout, lists.len())
        .into_iter()
        .filter(|w| w.lane == 0)
    {
        for (unit, keys) in lists[w.row].chunks_exact(64).enumerate() {
            for m in 0..if pv { 8 } else { 4 } {
                for k in 0..if pv { 4 } else { 8 } {
                    for n in 0..w.fragments {
                        let head = w.head0 + n * 8;
                        let start = if pv { k } else { m } * 16;
                        let key_atom = keys[start..start + 16].try_into().unwrap();
                        let dim = w.warp * 128 + if pv { m } else { k } * 16;
                        result
                            .entry((w.row, head, w.warp, unit, m))
                            .or_default()
                            .push(if pv {
                                Atom::Pv {
                                    keys: key_atom,
                                    first_dim: dim,
                                }
                            } else {
                                Atom::Qk {
                                    keys: key_atom,
                                    first_dim: dim,
                                }
                            });
                    }
                }
            }
        }
    }
    result
}

#[test]
fn head_sharing_keeps_each_mma_chain_and_original_key_unit() {
    // Include holes, duplicate keys and unrelated neighboring selections.
    let lists = (0..2)
        .map(|row| {
            (0..640)
                .map(|k| {
                    if k % 19 == 0 {
                        -1
                    } else {
                        (k * (row * 2 + 3)) % 337
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for pv in [false, true] {
        let reference = chains(Layout::PairedRows, &lists, pv);
        assert_eq!(reference.len(), 2 * 2 * 4 * 10 * if pv { 8 } else { 4 });
        assert!(
            reference
                .values()
                .all(|chain| chain.len() == if pv { 4 } else { 8 })
        );
        for layout in [Layout::HeadGroups, Layout::HeadFragments] {
            assert_eq!(chains(layout, &lists, pv), reference);
        }
        let mut reordered = lists.clone();
        for row in &mut reordered {
            for unit in row.chunks_exact_mut(64) {
                unit.sort_unstable();
            }
        }
        assert_ne!(chains(Layout::HeadGroups, &reordered, pv), reference);
    }
}

#[derive(Debug, Eq, PartialEq)]
struct OnlineUnit {
    unit: usize,
    qk_fold: [usize; 4],
    max_shuffles: [usize; 2],
    ordered_sum: Vec<i32>,
    probability_pairs: Vec<[i32; 2]>,
}

type OnlineChain = (usize, usize, usize, usize);
fn online_chains(
    layout: Layout,
    lists: &[Vec<i32>],
    reverse_units: bool,
) -> BTreeMap<OnlineChain, Vec<OnlineUnit>> {
    let mut result: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for w in workers(layout, lists.len()) {
        let c = w.lane % 4;
        for n in 0..w.fragments {
            let head = w.head0 + n * 8 + w.lane / 4;
            let mut units = lists[w.row]
                .chunks_exact(64)
                .enumerate()
                .collect::<Vec<_>>();
            if reverse_units {
                units.reverse();
            }
            for (unit, keys) in units {
                let ordered_sum = (0..4)
                    .flat_map(|kt| (0..4).map(move |i| keys[kt * 16 + 2 * c + i % 2 + (i / 2) * 8]))
                    .collect::<Vec<_>>();
                let probability_pairs = ordered_sum
                    .chunks_exact(2)
                    .map(|pair| [pair[0], pair[1]])
                    .collect();
                result
                    .entry((w.row, head, w.warp, c))
                    .or_default()
                    .push(OnlineUnit {
                        unit,
                        qk_fold: [0, 1, 2, 3],
                        max_shuffles: [1, 2],
                        ordered_sum,
                        probability_pairs,
                    });
            }
            // Output rescale uses the correction from lane (2c+r)*4.
            for r in 0..2 {
                let output_head = w.head0 + n * 8 + 2 * c + r;
                let source_lane = (2 * c + r) * 4;
                assert_eq!(w.head0 + n * 8 + source_lane / 4, output_head);
            }
        }
    }
    result
}

#[test]
fn head_sharing_keeps_online_softmax_and_output_update_order() {
    let lists = vec![(0..640).collect(), (0..640).rev().collect()];
    let reference = online_chains(Layout::PairedRows, &lists, false);
    assert_eq!(reference.len(), 2 * 16 * 4 * 4);
    // Each vector is the carry chain: maximum, denominator FMA, BF16 P,
    // PV and output rescale/add complete before the next 64-key unit.
    assert!(
        reference
            .values()
            .all(|chain| chain.iter().map(|u| u.unit).eq(0..10))
    );
    for layout in [Layout::HeadGroups, Layout::HeadFragments] {
        assert_eq!(online_chains(layout, &lists, false), reference);
        assert_ne!(online_chains(layout, &lists, true), reference);
    }
}

#[test]
fn union_order_is_not_a_general_f32_reduction_identity() {
    // Put these nonzero products in distinct MMA atoms. Even preserving the
    // set and the F32 accumulator type does not preserve the chain result.
    let tiny = 2.0f32.powi(-20);
    let ordered = [1024.0f32, tiny, -1024.0]
        .into_iter()
        .fold(0.0, |sum, x| sum + x);
    let reordered = [1024.0f32, -1024.0, tiny]
        .into_iter()
        .fold(0.0, |sum, x| sum + x);
    assert_eq!(ordered.to_bits(), 0.0f32.to_bits());
    assert_eq!(reordered.to_bits(), tiny.to_bits());
}

fn tile_sector(key: usize, dim: usize, phase: usize) -> usize {
    (dim / 64) * (64 * 128) + key * 128 + (((dim % 64) / 8) ^ ((key + phase) % 8)) * 16
}

#[test]
fn shared_key_halves_are_disjoint_swizzled_sectors_not_contiguous_buffers() {
    for phase in 0..8 {
        let halves = [0, 32].map(|first| {
            (first..first + 32)
                .flat_map(|key| {
                    (0..512)
                        .step_by(8)
                        .map(move |dim| tile_sector(key, dim, phase))
                })
                .collect::<BTreeSet<_>>()
        });
        assert_eq!(halves[0].len(), 2048);
        assert_eq!(halves[1].len(), 2048);
        assert!(halves[0].is_disjoint(&halves[1]));
        assert_eq!(
            halves[0].union(&halves[1]).copied().collect::<Vec<_>>(),
            (0..65536).step_by(16).collect::<Vec<_>>()
        );
        // A contiguous 32 KiB overwrite would corrupt the old second half.
        assert!(halves[1].iter().any(|&sector| sector < 32768));
        assert!(halves[0].iter().any(|&sector| sector >= 32768));
    }
    let tile = 0..65536;
    let exchange = tile.end..tile.end + 2 * 4 * 64 * 8 * 4;
    let valid = exchange.end..exchange.end + 64 * 4;
    let codes = valid.end..valid.end + 64 * 8;
    let scales = codes.end..codes.end + 64 * 8;
    assert_eq!(scales.end, 83200);
    assert!(scales.end <= 101376);
    assert_eq!(codes.start % 8, 0);
}

#[test]
fn half_tile_pipeline_must_wait_for_every_column_chain() {
    // Each head group's four dimension warps own eight column chains.
    let readers = 2 * 4 * 8;
    let consumed = |visits: &[(usize, usize)]| {
        let mut seen = vec![[false; 4]; readers];
        for &(reader, key_atom) in visits {
            seen[reader][key_atom] = true;
        }
        seen.into_iter().all(|r| r[0] && r[1])
    };
    let old_order = (0..readers)
        .flat_map(|reader| (0..4).map(move |k| (reader, k)))
        .collect::<Vec<_>>();
    let interleaved = (0..4)
        .flat_map(|k| (0..readers).map(move |reader| (reader, k)))
        .collect::<Vec<_>>();
    assert!(!consumed(&old_order[..2]));
    assert!(!consumed(&interleaved[..readers * 2 - 1]));
    assert!(consumed(&interleaved[..readers * 2]));
    for reader in 0..readers {
        let trace = |order: &[(usize, usize)]| {
            order
                .iter()
                .filter_map(|&(r, k)| (r == reader).then_some(k))
                .collect::<Vec<_>>()
        };
        assert_eq!(trace(&old_order), trace(&interleaved));
    }
}
