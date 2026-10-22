//! The column-block HC producer and its split finish
//! (`cuda/sinkhorn_hc/round_cols.cuh`, `finish_split.cuh`): ownership maps,
//! the transposed butterfly's bits against `warp_sum` on the CPU, the RMS
//! lane order against the served producers, compilation and the backend's
//! bucket selection.
use kernel_core::{
    contract::RmsNormKind,
    rmsnorm::{RmsNormNumerics, RmsNormRequirement},
    rmsnorm_quantize::RmsNormQuantizeQuery,
    sinkhorn_expand_mix::SinkhornExpandMixQuery,
};
use kernels_sm120::{
    row::{rmsnorm_quantize, rmsnorm_quantize_cached},
    sinkhorn_hc::{
        expand_mix, finish_norm,
        finish_split::{self, Bucket},
        lagged, round_chain_partials, round_cols, round_rows,
    },
};

const H: usize = 5120;
const N: usize = 4;
const FLAT: usize = N * H;
const MIXES: usize = 24;
const CHAINS: usize = FLAT / 128;

fn norm() -> rmsnorm_quantize_cached::Entry {
    rmsnorm_quantize_cached::Entry {
        reference: rmsnorm_quantize::Entry::new(
            RmsNormQuantizeQuery {
                norm: RmsNormRequirement::bf16(
                    5120,
                    1e-20,
                    RmsNormKind::Plain,
                    false,
                    RmsNormNumerics::TREE_256,
                ),
                amax_floor_bits: 1e-4_f32.to_bits(),
            },
            32,
            "norm_reference",
        )
        .unwrap(),
        prefix: "norm_cached".into(),
        early_release: false,
    }
}
fn hc() -> round_chain_partials::Entry {
    round_chain_partials::Entry {
        reference: expand_mix::Entry {
            query: SinkhornExpandMixQuery {
                mix: lagged::LAB.mix,
            },
            capacity: 32,
            symbol: "round_reference".into(),
            probe: false,
        },
        prefix: "round_chain".into(),
    }
}
fn finish_entry() -> finish_norm::Entry {
    finish_norm::Entry {
        mixer: hc().reference,
        norm: norm(),
        prefix: "finish_norm".into(),
    }
}
fn producer(mutation: round_cols::Mutation) -> round_cols::Entry {
    round_cols::Entry {
        reference: hc(),
        prefix: "round_cols".into(),
        mutation,
    }
}
fn split(mutation: finish_split::Mutation) -> finish_split::Entry {
    finish_split::Entry {
        reference: finish_entry(),
        prefix: "finish_split".into(),
        mutation,
    }
}

/// Every expanded value, rounded column and chain sum of a bucket has
/// exactly one owner: (CTA column block, row tile, item or warp, mix slot).
#[test]
fn column_blocks_own_every_output_once() {
    for rows in [16usize, 32] {
        for (tile, transposed) in round_cols::TILES {
            let tile = tile as usize;
            let p = producer(round_cols::Mutation::None)
                .plan(rows as u32, rows as u32, tile as u32, transposed)
                .unwrap();
            assert_eq!(p.grid, [40, rows.div_ceil(tile) as u32, 1]);
            assert_eq!(p.block, [256, 1, 1]);
            assert_eq!(
                p.workspace_bytes,
                hc().plan(rows as u32, rows as u32, 4)
                    .unwrap()
                    .workspace_bytes
            );
            let mut expanded = vec![0u8; rows * FLAT];
            let mut rounded = vec![0u8; rows * H];
            let mut sums = vec![0u8; rows * MIXES * CHAINS];
            for block in 0..40 {
                for t in 0..rows.div_ceil(tile) {
                    for item in 0..tile * 32 {
                        let (row, col) = (t * tile + item / 32, block * 128 + 4 * (item % 32));
                        if row >= rows {
                            continue;
                        }
                        for c in col..col + 4 {
                            rounded[row * H + c] += 1;
                            for j in 0..N {
                                expanded[row * FLAT + j * H + c] += 1;
                            }
                        }
                    }
                    for warp in 0..8 {
                        let chain = (warp % N) * (H / 128) + block;
                        for i in 0..12 {
                            let m = warp / N + 2 * i;
                            for r in 0..tile {
                                let row = t * tile + r;
                                if row < rows {
                                    sums[row * MIXES * CHAINS + m * CHAINS + chain] += 1;
                                }
                            }
                        }
                    }
                }
            }
            assert!(
                expanded
                    .iter()
                    .chain(&rounded)
                    .chain(&sums)
                    .all(|&n| n == 1)
            );
        }
    }
    for (rows, live, tile, tx) in [
        (0, 0, 4, false),
        (16, 17, 4, false),
        (16, 16, 16, false),
        (16, 16, 1, true),
    ] {
        assert!(
            producer(round_cols::Mutation::None)
                .plan(rows, live, tile, tx)
                .is_err()
        );
    }
    assert!(
        producer(round_cols::Mutation::ReverseLevels)
            .plan(32, 32, 8, true)
            .is_err()
    );
}

/// The served lane's butterfly: every lane ends with the same bits.
fn warp_sum(mut v: [f32; 32]) -> [f32; 32] {
    for offset in [16, 8, 4, 2, 1] {
        let old = v;
        for lane in 0..32 {
            v[lane] = old[lane] + old[lane ^ offset];
        }
    }
    v
}

/// The producer's transposed butterfly over `batch` sums per lane: the
/// value each lane ends with and the sum index it holds.
fn transposed(values: &[Vec<f32>], batch: usize) -> Vec<(usize, f32)> {
    let levels = batch.trailing_zeros() as usize;
    let mut v: Vec<Vec<f32>> = values.to_vec();
    for level in 0..levels {
        let offset = 16 >> level;
        let half = (batch >> level) / 2;
        let old = v.clone();
        for lane in 0..32 {
            let upper = lane & offset != 0;
            for k in 0..half {
                let send = |l: usize| {
                    if l & offset != 0 {
                        old[l][k]
                    } else {
                        old[l][k + half]
                    }
                };
                let keep = if upper {
                    old[lane][k + half]
                } else {
                    old[lane][k]
                };
                v[lane][k] = keep + send(lane ^ offset);
            }
        }
    }
    let mut value: Vec<f32> = v.iter().map(|x| x[0]).collect();
    let mut offset = 16 >> levels;
    while offset > 0 {
        let old = value.clone();
        for lane in 0..32 {
            value[lane] = old[lane] + old[lane ^ offset];
        }
        offset >>= 1;
    }
    (0..32)
        .map(|lane| (lane >> (5 - levels), value[lane]))
        .collect()
}

#[test]
fn transposed_butterfly_keeps_warp_sum_bits() {
    let mut state = 0x2545f4914f6cdd1d_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Wide binades and both signs, so a different tree changes bits.
        let mantissa = (state >> 40) as u32 & 0x7fffff;
        let exponent = 100 + (state >> 20) as u32 % 50;
        f32::from_bits(((state & 1) as u32) << 31 | exponent << 23 | mantissa)
    };
    for batch in [8, 16, 32] {
        for _ in 0..200 {
            let values: Vec<Vec<f32>> = (0..32)
                .map(|_| (0..batch).map(|_| next()).collect())
                .collect();
            let got = transposed(&values, batch);
            for (lane, &(index, value)) in got.iter().enumerate() {
                let mut column = [0f32; 32];
                for (l, c) in column.iter_mut().enumerate() {
                    *c = values[l][index];
                }
                let want = warp_sum(column);
                assert_eq!(
                    value.to_bits(),
                    want[lane].to_bits(),
                    "batch {batch} lane {lane}"
                );
            }
            // Every sum index is held by 32 / batch lanes.
            for index in 0..batch {
                assert_eq!(got.iter().filter(|(i, _)| *i == index).count(), 32 / batch);
            }
        }
    }
}

/// The split finish's RMS lane t adds the squares of the elements the
/// served producers' slot t adds, in the same order.
#[test]
fn rms_lanes_keep_the_served_producers_order() {
    for slot in 0..256 {
        let served: Vec<usize> = (slot..FLAT / 8)
            .step_by(256)
            .flat_map(|v| (0..4).flat_map(move |j| [8 * v + 2 * j, 8 * v + 2 * j + 1]))
            .collect();
        let split: Vec<usize> = (0..FLAT / 8 / 256)
            .flat_map(|i| {
                let v = slot + 256 * i;
                (0..8).map(move |e| 8 * v + e)
            })
            .collect();
        assert_eq!(served, split);
    }
    let finish = include_str!("../cuda/sinkhorn_hc/finish_split.cuh");
    let producer = include_str!("../cuda/sinkhorn_hc/round_cols.cuh");
    for source in [finish, producer] {
        for forbidden in ["blockIdx", "griddep_", "atom"] {
            assert!(!source.contains(forbidden), "{forbidden}");
        }
    }
}

#[test]
fn column_pair_compiles_without_spills() {
    let tc = kernel_build::Toolchain::detect().unwrap();
    assert!(!tc.skipped);
    let root = std::env::temp_dir().join(format!("column-producers-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut producers: Vec<_> = round_cols::TILES
        .iter()
        .map(|&(tile, tx)| (producer(round_cols::Mutation::None), tile, tx))
        .collect();
    producers.push((producer(round_cols::Mutation::ReverseLevels), 4, false));
    let mut finishes: Vec<_> = finish_split::WARPS
        .iter()
        .map(|&w| (split(finish_split::Mutation::None), w))
        .collect();
    for m in [
        finish_split::Mutation::ClearStatus,
        finish_split::Mutation::WrongFloor,
        finish_split::Mutation::ReverseFold,
        finish_split::Mutation::ReverseSquares,
    ] {
        let mut e = split(m);
        e.prefix = format!("finish_split_{m:?}").to_lowercase();
        finishes.push((e, 32));
    }
    producers[6].0.prefix = "round_cols_reverse_levels".into();
    let built = kernel_build::build(
        &tc,
        &[
            round_cols::cubin_spec("round_cols", &producers).unwrap(),
            finish_split::cubin_spec("finish_split", &finishes).unwrap(),
        ],
        &root.join("out"),
        &root.join("lock"),
        kernel_build::BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    assert_eq!(built.iter().map(|m| m.entries.len()).sum::<usize>(), 14);
    for m in built {
        for e in m.entries {
            assert_eq!((e.local_bytes, e.stack_bytes), (0, 0), "{}", e.symbol);
            eprintln!(
                "COLUMN_PAIR_RESOURCE {} regs={} shared={} local={} stack={}",
                e.symbol, e.registers, e.shared_static_bytes, e.local_bytes, e.stack_bytes
            );
        }
    }
}

fn finish_backend() -> kernels_sm120::backend::Sm120 {
    kernels_sm120::backend::Sm120::default()
        .with_sinkhorn_round_expand_mix(hc().reference.clone())
        .with_sinkhorn_round_chain(hc(), vec![(1, 8), (8, 4)])
        .with_rmsnorm_quantize(norm().reference.clone())
        .with_rmsnorm_quantize_cached(norm(), vec![(1, 32), (8, 32)])
}
fn finish_buffers()
-> kernel_core::sinkhorn_expand_mix::SinkhornRoundNormBuffers<kernel_core::backend::View> {
    use kernel_core::{
        backend::View,
        rmsnorm_quantize::RmsNormQuantizeBuffers as Norm,
        sinkhorn_expand_mix::{
            SinkhornRoundExpandMixBuffers as Hc, SinkhornRoundNormBuffers as Buffers,
        },
    };
    let v = |handle| View {
        handle,
        bytes: 1 << 30,
    };
    Buffers {
        hc: Hc {
            x: v(1),
            sum: v(2),
            rounded: v(3),
            post_in: v(4),
            comb_in: v(5),
            expanded: v(6),
            fn_: v(7),
            base: v(8),
            scale: v(9),
            pre_in: v(10),
            collapsed: v(11),
            pre: v(12),
            post: v(4),
            comb: v(5),
            live_rows: v(13),
        },
        norm: Norm {
            input: v(11),
            weight: v(14),
            normed: v(15),
            widened: Some(v(16)),
            codes: v(17),
            scales: v(18),
            live_rows: v(13),
            status: v(19),
            status_row_offset: 3,
        },
        tickets: None,
    }
}

#[test]
fn column_pair_replaces_both_launches_only_at_measured_buckets() {
    use kernel_core::{
        backend::{Backend, Counts, Need, View},
        sinkhorn_expand_mix::SinkhornRoundNormQuery,
    };
    let e = finish_entry();
    let query = SinkhornRoundNormQuery {
        mixer: e.mixer.query,
        norm: e.norm.reference.query,
    };
    let rows = round_rows::Entry {
        reference: hc(),
        prefix: "rounded_rows".into(),
    };
    let base = || {
        finish_backend()
            .with_sinkhorn_finish_norm(e.clone(), vec![8, 16, 32])
            .with_sinkhorn_finish_norm_rows(rows.clone(), vec![(32, 2)])
    };
    let measured = vec![
        Bucket {
            rows: 16,
            tile: 4,
            transposed: true,
            warps: 32,
        },
        Bucket {
            rows: 32,
            tile: 8,
            transposed: false,
            warps: 16,
        },
    ];
    let old = base()
        .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
        .unwrap();
    let new = base()
        .with_sinkhorn_finish_norm_cols(
            producer(round_cols::Mutation::None),
            split(finish_split::Mutation::None),
            measured.clone(),
        )
        .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
        .unwrap();
    assert_eq!(new.info().workspace, old.info().workspace);
    assert_eq!(new.info().bytes, old.info().bytes);
    for symbol in [
        "round_cols_r4_tx_v1",
        "round_cols_r8_v1",
        "finish_split_w32_v1",
        "finish_split_w16_v1",
    ] {
        assert!(
            new.info().symbols.iter().any(|(s, _)| s == symbol),
            "{symbol}"
        );
    }
    let workspace = View {
        handle: 20,
        bytes: old.info().workspace,
    };
    for count in [1, 6, 8, 12, 16, 32] {
        let expected = old
            .launches(&finish_buffers(), workspace, Counts::rows(count))
            .unwrap();
        let actual = new
            .launches(&finish_buffers(), workspace, Counts::rows(count))
            .unwrap();
        match measured.iter().find(|b| b.rows == count) {
            None => assert_eq!(actual, expected, "{count}"),
            Some(b) => {
                assert_eq!(expected.len(), 2);
                assert_eq!(actual.len(), 2);
                assert_eq!(
                    actual[0].symbol,
                    producer(round_cols::Mutation::None).symbol(b.tile, b.transposed)
                );
                assert_eq!(actual[0].plan.grid, [40, count.div_ceil(b.tile), 1]);
                assert_eq!(actual[0].arguments, expected[0].arguments);
                assert_eq!(
                    actual[0].plan.workspace_bytes,
                    expected[0].plan.workspace_bytes
                );
                assert_eq!(
                    actual[1].symbol,
                    split(finish_split::Mutation::None).symbol(b.warps)
                );
                assert_eq!(actual[1].plan.grid, [2, count, 1]);
                assert_eq!(actual[1].plan.block, [b.warps * 32, 1, 1]);
                assert_eq!(actual[1].arguments, expected[1].arguments);
            }
        }
    }
    let bucket = |rows, tile, transposed, warps| Bucket {
        rows,
        tile,
        transposed,
        warps,
    };
    let refused = |p: round_cols::Entry, f: finish_split::Entry, b: Vec<Bucket>| {
        base()
            .with_sinkhorn_finish_norm_cols(p, f, b)
            .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
            .is_err()
    };
    let none = || producer(round_cols::Mutation::None);
    let plain = || split(finish_split::Mutation::None);
    // No bucket, unordered, outside the served tail, bad tiles or warps.
    for b in [
        vec![],
        vec![bucket(32, 8, false, 32), bucket(16, 4, false, 32)],
        vec![bucket(4, 2, false, 32)],
        vec![bucket(16, 16, false, 32)],
        vec![bucket(16, 4, false, 4)],
    ] {
        assert!(refused(none(), plain(), b));
    }
    // Controls never serve.
    assert!(refused(
        producer(round_cols::Mutation::ReverseLevels),
        plain(),
        measured.clone()
    ));
    assert!(refused(
        none(),
        split(finish_split::Mutation::ReverseSquares),
        measured.clone()
    ));
    // Another mixer's pair, or a bucket the two-CTA finish already takes.
    let mut other = none();
    other.reference.reference.symbol.push_str("_different");
    assert!(refused(other, plain(), measured.clone()));
    assert!(
        base()
            .with_sinkhorn_finish_norm_twin(
                kernels_sm120::sinkhorn_hc::finish_twin::Entry {
                    reference: finish_entry(),
                    prefix: "twin".into(),
                    mutation: kernels_sm120::sinkhorn_hc::completion::Mutation::None,
                },
                vec![(16, 32)],
            )
            .with_sinkhorn_finish_norm_cols(none(), plain(), measured.clone())
            .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
            .is_err()
    );
    assert!(
        base()
            .with_sinkhorn_finish_norm_cols(none(), plain(), measured.clone())
            .with_sinkhorn_finish_norm_cols(none(), plain(), measured)
            .sinkhorn_round_expand_mix_norm(&query, Need::rows(32))
            .is_err()
    );
}
