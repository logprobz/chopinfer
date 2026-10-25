use super::*;
pub(super) fn source(
    a: &mut Arena<'_>,
    case: Case,
    live: View,
) -> LabResult<(View, View, Vec<Launch>)> {
    use kernel_core::precision::quantizer::BlockQuantizeBuffers;
    let input = a.filled(u64::from(case.rows) * 5120, 0)?;
    let scales = a.filled(u64::from(case.rows) * 160, 0)?;
    let status = a.filled(u64::from(case.rows) * 4, 0)?;
    let values = if a.session.is_some() {
        let mut seed = 0xc837_6251_aa19_46d3u64;
        let mut uniform = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 32) as u32 as f64 + 0.5) / (u32::MAX as f64 + 1.)
        };
        (0..u64::from(case.rows) * 5120)
            .flat_map(|i| {
                let normal =
                    (-2. * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
                let value = normal
                    * (0.5 + (i / 32 % 5) as f64 * 0.25)
                    * if i % 997 == 0 { 6. } else { 1. };
                kernel_core::codec::f32_to_bf16(value as f32).to_le_bytes()
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let bf16 = if a.session.is_some() {
        a.bytes(&values)?
    } else {
        a.filled(u64::from(case.rows) * 5120 * 2, 0)?
    };
    let backend = registration::quantizers(&FLASH, 4)?
        .into_iter()
        .fold(Sm120::new(), Sm120::with_block_quantize);
    let q = backend
        .block_quantize(&moe::quantizer(5120), Need::rows(case.rows))
        .map_err(|e| e.to_string())?;
    let ops = q
        .launches(
            &BlockQuantizeBuffers {
                input: bf16,
                output: input,
                scales: Some(scales),
                live_rows: live,
                status,
                status_row_offset: 0,
            },
            View::EMPTY,
            Counts::rows(case.rows),
        )
        .map_err(|e| e.to_string())?;
    Ok((input, scales, ops))
}
pub(super) fn equal(a: &Arena<'_>, case: Case, old: &Arm, new: &Arm, live: u32) -> LabResult<()> {
    for (i, (&left, &right)) in old.outputs.iter().zip(&new.outputs).enumerate() {
        let x = a.read(left)?;
        let y = a.read(right)?;
        if x != y {
            return Err(format!(
                "{} live={live} output={i}: bitwise mismatch at byte {:?}",
                case.label(),
                x.iter().zip(&y).position(|(x, y)| x != y)
            ));
        }
        let bytes = left.bytes as usize * live as usize / case.rows as usize;
        let data = &x[..bytes];
        let finite = match i {
            1 | 5 | 6 => data
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            2 => data
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80),
            3 => data.iter().all(|b| b & 0x7f != 0x7f),
            4 => !data.contains(&255),
            _ => true,
        };
        if !finite {
            return Err(format!("nonfinite or unwritten live output {i}"));
        }
    }
    for arm in [old, new] {
        if a.read(arm.status)?.iter().any(|&x| x != 0) {
            return Err("nonzero sticky quantizer status".into());
        }
    }
    Ok(())
}
pub(super) fn poison(a: &Arena<'_>, arm: &Arm) -> LabResult<()> {
    let s = &a.session.unwrap().stream;
    for v in &arm.poison {
        a.buffers[&v.handle]
            .copy_from_bytes(&vec![255; v.bytes as usize], s)
            .map_err(|e| e.to_string())?;
    }
    a.buffers[&arm.status.handle]
        .copy_from_bytes(&vec![0; arm.status.bytes as usize], s)
        .map_err(|e| e.to_string())
}
pub(super) fn clock_table(a: &Arena<'_>, arm: &Arm) -> LabResult<serde_json::Value> {
    let mut parts = serde_json::Map::new();
    for (part, view) in ["gateup", "down"].into_iter().zip(&arm.clocks) {
        let values = a
            .read(*view)?
            .chunks_exact(8)
            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        let queues=values.chunks_exact(family::CTAS as usize*8).enumerate().map(|(q,ctas)| {
   let mut cycles=[0_u64;7];let mut items=0;let mut active=0;let mut max=0;
   for row in ctas.chunks_exact(8) {if row[7]>0{active+=1;}items+=row[7];
    max=max.max(row[..7].iter().sum::<u64>());for i in 0..7{cycles[i]+=row[i];}}
   serde_json::json!({"rows":([8,32,64,128][q]),"cta_phase_cycles":cycles,"items":items,"active_ctas":active,"max_cta_cycles":max})
  }).collect::<Vec<_>>();
        parts.insert(part.into(), queues.into());
    }
    Ok(
        serde_json::json!({"phase_order":["setup_prologue","wait_barrier","decode_barrier","mma","fold_barrier","enqueue","epilogue_barrier"],"scope":"lane zero in every launched CTA; diagnostic kernels only; summed phase cycles are not exclusive wall time","parts":parts}),
    )
}
pub(super) fn utilization(
    a: &Arena<'_>,
    old: &Arm,
    v: family::Variant,
) -> LabResult<serde_json::Value> {
    let counts = a
        .read(old.route_counts)?
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut tiles = [0_u64; 4];
    let mut actual = 0;
    for n in counts {
        actual += u64::from(n);
        let full = n / v.rows;
        let tail = n % v.rows;
        tiles[if v.rows == 64 { 2 } else { 3 }] += u64::from(full);
        if tail > 0 {
            tiles[if tail <= 8 {
                0
            } else if tail <= 32 {
                1
            } else if tail <= 64 {
                2
            } else {
                3
            }] += 1;
        }
    }
    let allocated = tiles
        .iter()
        .zip([8, 32, 64, 128])
        .map(|(&n, r)| n * r)
        .sum::<u64>();
    Ok(
        serde_json::json!({"tiles":tiles,"live_routes":actual,"allocated_rows":allocated,"row_utilization":actual as f64/allocated as f64}),
    )
}
