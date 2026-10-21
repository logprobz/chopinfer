//! Host emulation of the router-in-norm launches, for running every phase
//! of the job on CPU. It models each launch's reads, writes, flags and
//! paths, not its arithmetic: the norm is a sequential F32 RMSNorm with
//! stand-in codes and scales, and each logit sums every 64th product,
//! tensor-path operands outside the tensor domain counting as zero (so a
//! row certified by mistake changes its logits, as on the device).
use super::*;

fn handles(op: &Launch) -> Vec<u64> {
    op.arguments
        .0
        .iter()
        .map(|a| match a {
            Argument::DeviceBuffer(h) => *h,
            Argument::U32(v) => u64::from(*v),
            Argument::U64(v) => *v,
            _ => 0,
        })
        .collect()
}

fn u16s(v: &[u8]) -> Vec<u16> {
    v.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn put_u32(v: &mut [u8], at: usize, x: u32) {
    v[at * 4..at * 4 + 4].copy_from_slice(&x.to_le_bytes());
}

const STRIDE: usize = 64;

fn dot(x: impl Fn(usize) -> f32, w: &[u16], f: usize, tensor: bool) -> f32 {
    let mut total = 0f32;
    for k in (0..K as usize).step_by(STRIDE) {
        let a = x(k);
        let a = if tensor && !tensor_operand(a) { 0.0 } else { a };
        total += a * widen(w[f * K as usize + k]);
    }
    total
}

pub fn run(a: &mut Arena<'_>, ops: &[Launch]) -> LabResult<()> {
    for op in ops {
        let h = handles(op);
        let host = a.host.as_mut().unwrap();
        let live_of = |host: &BTreeMap<u64, Vec<u8>>, handle: u64| u32s(&host[&handle])[0];
        if op.symbol.contains("prefill_norm_ceil") {
            let rows = op.plan.grid[0];
            let live = live_of(host, h[6]).min(rows) as usize;
            let input = u16s(&host[&h[0]]);
            let weight = u16s(&host[&h[1]]);
            let eps = f32::from_bits(h[9] as u32);
            let certify = op.symbol.contains("certif");
            let all = op.symbol.contains("certify_all");
            let (mut normed, mut widened, mut codes, mut scales, mut status) = (
                host[&h[2]].clone(),
                host[&h[3]].clone(),
                host[&h[4]].clone(),
                host[&h[5]].clone(),
                host[&h[7]].clone(),
            );
            let mut flags = if certify {
                host[&h[10]].clone()
            } else {
                Vec::new()
            };
            for r in 0..live {
                let row = &input[r * K as usize..(r + 1) * K as usize];
                let square: f32 = row.iter().map(|&x| widen(x) * widen(x)).sum();
                let inv = 1.0 / (square / K as f32 + eps).sqrt();
                let out: Vec<u16> = row
                    .iter()
                    .zip(&weight)
                    .map(|(&x, &w)| bf16(widen(x) * inv * widen(w)))
                    .collect();
                let flagged = out.iter().any(|&x| !widen(x).is_finite());
                let prior = u32s(&status)[r];
                let st = prior | u32::from(flagged);
                put_u32(&mut status, r, st);
                for (k, &x) in out.iter().enumerate() {
                    let at = r * K as usize + k;
                    normed[at * 2..at * 2 + 2].copy_from_slice(&x.to_le_bytes());
                    widened[at * 4..at * 4 + 4].copy_from_slice(&widen(x).to_le_bytes());
                    codes[at] = if st != 0 { 0 } else { x as u8 };
                }
                for g in 0..(K / 32) as usize {
                    scales[r * (K / 32) as usize + g] = if st != 0 {
                        127
                    } else {
                        (out[g * 32] >> 7) as u8
                    };
                }
                if certify {
                    put_u32(
                        &mut flags,
                        r,
                        u32::from(all || out.iter().all(|&x| tensor_operand(widen(x)))),
                    );
                }
            }
            for (handle, value) in [
                (h[2], normed),
                (h[3], widened),
                (h[4], codes),
                (h[5], scales),
                (h[7], status),
            ] {
                host.insert(handle, value);
            }
            if certify {
                host.insert(h[10], flags);
            }
        } else if op.symbol.ends_with("_zero_prepare_v1") {
            let rows = h[2] as usize;
            let live = (live_of(host, h[1]) as usize).min(rows);
            let act = u32s(&host[&h[0]]);
            let mut work = host[&h[3]].clone();
            let components = (rows * 4).next_multiple_of(16);
            let single = components + rows * 3 * K as usize * 2;
            for r in 0..rows {
                let row = if r < live {
                    &act[r * K as usize..(r + 1) * K as usize]
                } else {
                    &[][..]
                };
                let valid = r < live && row.iter().all(|&b| tensor_operand(f32::from_bits(b)));
                let exact = row.iter().all(|&b| b & 0xffff == 0);
                for (k, &b) in row.iter().enumerate() {
                    let at = components + (r * 3 * K as usize + k) * 2;
                    work[at..at + 2].copy_from_slice(&((b >> 16) as u16).to_le_bytes());
                }
                if valid && !exact {
                    return Err("emulation: residual components are not modelled".into());
                }
                put_u32(&mut work, r, u32::from(valid));
                put_u32(&mut work[single..], r, u32::from(valid && exact));
                put_u32(
                    &mut work[single + rows * 4..],
                    r,
                    u32::from(valid && !exact),
                );
            }
            host.insert(h[3], work);
        } else if op.symbol.contains("_zero_m32_v1") || op.symbol.contains("_certified_m32_v1") {
            let certified = op.symbol.contains("_certified_");
            // zero: activation, weight, workspace, output, live, rows, domain;
            // certified: activation, normed, certified, weight, output, live, rows, domain.
            let (act_h, w_h, out_h, live_h, rows, dom_h) = if certified {
                (h[0], h[3], h[4], h[5], h[6] as usize, h[7])
            } else {
                (h[0], h[1], h[3], h[4], h[5] as usize, h[6])
            };
            let live = (live_of(host, live_h) as usize).min(rows);
            let act = u32s(&host[&act_h]);
            let w = u16s(&host[&w_h]);
            let domain = u32s(&host[&dom_h]);
            let (plane, plane_stride, single, row_domain): (Vec<u16>, usize, Vec<u32>, Vec<u32>) =
                if certified {
                    let flags = u32s(&host[&h[2]]);
                    (u16s(&host[&h[1]]), K as usize, flags.clone(), flags)
                } else {
                    let work = &host[&h[2]];
                    let components = (rows * 4).next_multiple_of(16);
                    let single = components + rows * 3 * K as usize * 2;
                    (
                        u16s(&work[components..single]),
                        3 * K as usize,
                        u32s(&work[single..single + rows * 4]),
                        u32s(&work[..rows * 4]),
                    )
                };
            let mut out = host[&out_h].clone();
            for r in 0..live {
                for f in 0..N as usize {
                    let wd = domain[f / 16] != 0;
                    let value = if wd && single[r] != 0 {
                        dot(|k| widen(plane[r * plane_stride + k]), &w, f, true)
                    } else if !wd || row_domain[r] == 0 {
                        dot(|k| f32::from_bits(act[r * K as usize + k]), &w, f, false)
                    } else {
                        return Err("emulation: residual rows are not modelled".into());
                    };
                    put_u32(&mut out, r * N as usize + f, value.to_bits());
                }
            }
            host.insert(out_h, out);
        } else {
            return Err(format!("emulation has no model of {}", op.symbol));
        }
    }
    Ok(())
}
