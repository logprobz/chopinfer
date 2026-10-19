//! Experimental MMA scores, complete bit comparisons and sampled f64 errors.
use kernel_core::{
    attention::BlockTables,
    backend::{Backend, Counts, Need},
    pooled_indexer::{PositionKeyPages, PositionSelectBuffers},
};
use kernel_harness::LabResult;
use kernel_lab_families::position_mma_probe as family;
use kernels_sm120::backend::Sm120;
use std::path::PathBuf;
#[path = "position_mma_probe/oracle.rs"]
mod oracle;
#[path = "position_query_parallel/support.rs"]
mod support;
use support::*;

pub const OFFSETS: [u32; 3] = [0, 32768, 131072];
pub const DECODE: [u32; 7] = [1, 2, 4, 6, 8, 16, 32];

#[test]
fn cpu_mma_shapes_keep_every_requested_row() {
    for ratio in [1, 2] {
        let specs = family::cubins().remove(0);
        assert!(
            specs
                .entries
                .iter()
                .any(|e| e.symbol == family::symbol(ratio))
        );
        for offset in OFFSETS {
            let rows: Vec<_> = (0..2)
                .flat_map(|shard| (0..4096).map(move |r| offset + shard * 4096 + r))
                .collect();
            assert_eq!(rows.len(), 8192);
            assert_eq!(rows[8191], offset + 8191);
        }
    }
}

fn words(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
}
fn mismatches(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len());
    if a == b {
        0
    } else {
        words(a).zip(words(b)).filter(|(a, b)| a != b).count()
    }
}

fn block(
    s: &kernel_harness::Session,
    m: &runtime_cuda::Module,
    rows: u32,
    first: u32,
    offset: u32,
    ratio: u32,
    prefill: bool,
) -> LabResult<serde_json::Value> {
    let width = (offset + 8192).div_ceil(ratio).div_ceil(64) * 64 + 64;
    let mut a = Arena {
        session: s,
        buffers: vec![],
    };
    let mut x = oracle::Operands::new(
        rows as usize,
        width as usize,
        0x812654329 ^ u64::from(first),
    );
    x.witness();
    let q = a.input(&x.q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let weights = a.input(
        &x.weights
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let codes = a.input(&x.codes)?;
    let scales = a.input(&x.scales)?;
    let page = 64 / ratio;
    let pages = width / page;
    let tables = a.input(&i32s(
        (0..pages).map(|p| if p == 2 { -1 } else { p as i32 }),
    ))?;
    let positions: Vec<_> = (0..rows)
        .map(|r| {
            if prefill {
                (offset + first + r) as i32
            } else {
                (offset + 8191) as i32
            }
        })
        .collect();
    let b = PositionSelectBuffers {
        q,
        head_weights: weights,
        keys: PositionKeyPages {
            codes,
            scales,
            pages,
            code_page_stride: u64::from(page) * 64,
            scale_page_stride: u64::from(page) * 4,
            page_tables: BlockTables {
                tables,
                sequences: 1,
                max_pages: pages,
            },
        },
        positions: a.input(&i32s(positions.clone()))?,
        row_batch: a.input(&i32s(vec![0; rows as usize]))?,
        live_rows: a.input(&rows.to_le_bytes())?,
        candidate_mask: None,
        position_capacity: width,
        scores: a.output(rows as usize * width as usize * 4)?,
        status: a.output(rows as usize * 4)?,
        output: a.output(rows as usize * 512 * 4)?,
    };
    let op = Sm120::new()
        .with_position_indexer([family::entry(ratio)])
        .with_live_position_indexer()
        .position_indexer(&family::entry(ratio).query, 512, width, Need::rows(rows))
        .map_err(|e| e.to_string())?;
    let scratch = a.output(op.info().workspace as usize)?;
    let old = op
        .launches(&b, scratch, Counts::rows(rows))
        .map_err(|e| e.to_string())?;
    let mut new = old.clone();
    new[0].symbol = family::symbol(ratio);
    let mut planes = vec![];
    for calls in [&old, &new] {
        a.poison(&[b.scores, b.output, b.status, scratch])?;
        let graph = s.capture(|_| fire(s, m, &a, calls))?;
        graph.launch(&s.stream).map_err(|e| e.to_string())?;
        s.stream.synchronize().map_err(|e| e.to_string())?;
        planes.push([a.read(b.scores)?, a.read(b.output)?, a.read(b.status)?]);
    }
    let milliseconds = time_pair(s, m, &a, [&old[..1], &new[..1]], 3)?;
    // Same masks, holes and -inf tail in both arms. Arithmetic changes are
    // reported; they are never silently treated as bitwise conformance.
    for arm in &planes {
        assert!(words(&arm[2]).all(|v| v == 0));
        for (row, &position) in positions.iter().enumerate() {
            let count = (position as u32 + 1) / ratio;
            for (index, value) in
                words(&arm[0][row * width as usize * 4..(row + 1) * width as usize * 4]).enumerate()
            {
                if index >= count as usize || index / page as usize == 2 {
                    assert_eq!(value, f32::NEG_INFINITY.to_bits());
                }
            }
        }
    }
    let mut squared = [0_f64; 2];
    let mut maximum = [0_f64; 2];
    let mut samples = 0;
    let mut signed = [0_f64; 2];
    let mut normalized = [0_f64; 2];
    let mut reference_squared = 0_f64;
    for row in [0, rows as usize - 1] {
        let count = (positions[row] as u32 + 1) / ratio;
        for key in [0, 1, count / 2, count.saturating_sub(1)] {
            if key >= count || key / page == 2 {
                continue;
            }
            let at = (row * width as usize + key as usize) * 4;
            let (reference, magnitude) = x.reference_stats(row, key as usize);
            reference_squared += reference * reference;
            let bound = 2_f64.powi(-22) * 161. * magnitude;
            for arm in 0..2 {
                let value = f32::from_bits(u32::from_le_bytes(
                    planes[arm][0][at..at + 4].try_into().unwrap(),
                ));
                let exact = x.score(row, key as usize, arm == 1);
                assert_eq!(
                    value.to_bits(),
                    exact.to_bits(),
                    "oracle arm={arm} row={row} key={key}"
                );
                let error = (f64::from(value) - reference).abs();
                assert!(
                    error <= bound,
                    "score exceeds the f64 contract: arm={arm} error={error} bound={bound}"
                );
                signed[arm] += f64::from(value) - reference;
                if bound > 0. {
                    normalized[arm] = normalized[arm].max(error / bound);
                }
                squared[arm] += error * error;
                maximum[arm] = maximum[arm].max(error);
            }
            samples += 1;
        }
    }
    let control = (positions[0] as u32 + 1) / ratio > 0;
    if control {
        assert_eq!(
            f32::from_bits(words(&planes[0][0]).next().unwrap()),
            1. / 64.
        );
        assert_eq!(f32::from_bits(words(&planes[1][0]).next().unwrap()), 0.);
    }
    Ok(
        serde_json::json!({"first":first,"rows":rows,"offset":offset,"ratio":ratio,"prefill":prefill,"width":width,
        "score_bit_mismatches":mismatches(&planes[0][0],&planes[1][0]),"selection_mismatches":mismatches(&planes[0][1],&planes[1][1]),
        "samples":samples,"squared_error":squared,"signed_error":signed,"max_bound_fraction":normalized,"reference_squared":reference_squared,"max_abs_error":maximum,"old_ms":milliseconds[0],"new_ms":milliseconds[1],
        "negative_control_rejected":control,"exclusive":true,"serving":false}),
    )
}

#[test]
#[ignore = "requires the staged pair GPU job"]
fn gpu_mma_scores() {
    let ran = gpu_test(2 << 30, |s| {
        let root =
            PathBuf::from(std::env::var_os("CHOPINFER_MMA_CUBINS").ok_or("MMA cubins required")?);
        let m = s.load_cubin(
            &std::fs::read(root.join(format!("{}.cubin", family::CUBIN)))
                .map_err(|e| e.to_string())?,
        )?;
        let shard: u32 = std::env::var("CHOPINFER_L4_SHARD")
            .map_err(|e| e.to_string())?
            .parse()
            .map_err(|_| "invalid shard")?;
        if shard > 1 {
            return Err("invalid shard".into());
        }
        let mut blocks = 0;
        for (i, rows) in DECODE
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i as u32 % 2 == shard)
        {
            for ratio in [1, 2] {
                let record = block(s, &m, rows, i as u32, 0, ratio, false)?;
                println!("MMA_BLOCK {record}");
                blocks += 1;
            }
        }
        for offset in OFFSETS {
            for ratio in [1, 2] {
                for first in (shard * 4096..(shard + 1) * 4096).step_by(64) {
                    let record = block(s, &m, 64, first, offset, ratio, true)?;
                    println!("MMA_BLOCK {record}");
                    blocks += 1;
                }
            }
        }
        println!("MMA_COMPLETE shard={shard} blocks={blocks}");
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some(), "GPU test skipped");
}
