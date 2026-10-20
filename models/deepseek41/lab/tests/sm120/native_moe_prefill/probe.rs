//! Criterion 1: one native instruction from +0. Each output carries one
//! nonzero product, so every finite E4M3 code (subnormals included) by
//! every E2M1 code must give its exact product; small-integer sums must be
//! exact; the K slots and scale lanes follow the PTX fragment layout the
//! host emulation assumes. A rounding probe of long sums is report-only.
use super::*;
use kernel_core::codec::{SmallFloat, decode_ue8m0_f64};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};

/// One instruction's operands: A (16 weight rows by 32 K slots, byte
/// containers with the E2M1 code in bits 5:2), B (32 slots by 8 routed
/// columns, E4M3 bytes), one scale per A row and per B column.
#[derive(Clone)]
pub struct Mma {
    pub a: [[u8; 32]; 16],
    pub b: [[u8; 8]; 32],
    pub sa: [u8; 16],
    pub sb: [u8; 8],
}

impl Mma {
    /// The lanes' registers in the PTX m16n8k32 layout: lane 4g + t holds
    /// A rows g and g + 8 at slots 4t.. and 16 + 4t.., B column g at the
    /// same slots, the scale of A row g + 8 (t & 1) and of B column g.
    pub fn registers(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut a = vec![0; 128];
        let mut b = vec![0; 64];
        let mut s = vec![0; 64];
        for lane in 0..32 {
            let (g, t) = (lane / 4, lane % 4);
            for i in 0..4 {
                for j in 0..4 {
                    let byte = self.a[g + 8 * (i & 1)][4 * t + 16 * (i >> 1) + j];
                    a[4 * lane + i] |= u32::from(byte) << (8 * j);
                }
            }
            for i in 0..2 {
                for j in 0..4 {
                    b[2 * lane + i] |= u32::from(self.b[4 * t + 16 * i + j][g]) << (8 * j);
                }
            }
            s[2 * lane] = u32::from(self.sa[g + 8 * (t & 1)]);
            s[2 * lane + 1] = u32::from(self.sb[g]);
        }
        (a, b, s)
    }
    /// D[row][column] from the lanes' accumulators.
    pub fn outputs(d: &[f32]) -> [[f32; 8]; 16] {
        let mut out = [[0.0; 8]; 16];
        for lane in 0..32 {
            let (g, t) = (lane / 4, lane % 4);
            for i in 0..4 {
                out[g + 8 * (i >> 1)][2 * t + (i & 1)] = d[4 * lane + i];
            }
        }
        out
    }
    /// The exact f64 result of each output.
    pub fn exact(&self) -> [[f64; 8]; 16] {
        let mut out = [[0.0; 8]; 16];
        for (r, row) in out.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                for k in 0..32 {
                    *v += SmallFloat::E2M1.decode_f64(self.a[r][k] >> 2)
                        * decode_ue8m0_f64(self.sa[r])
                        * SmallFloat::E4M3.decode_f64(self.b[k][c])
                        * decode_ue8m0_f64(self.sb[c]);
                }
            }
        }
        out
    }
}

/// Single-product cases: A row r is nonzero only at slot (r + launch) % 32,
/// B is dense, so output (r, c) is one product. Each row holds one E2M1
/// code and eight E4M3 codes; over all launches every finite E4M3 code
/// meets every E2M1 code, rows walk every slot, and the row and column
/// scales differ so a misrouted scale lane shows.
pub fn single_products(scales: (u8, u8)) -> Vec<Mma> {
    let e4m3: Vec<u8> = (0..=255u8).filter(|c| c & 0x7f != 0x7f).collect();
    let rows: Vec<(u8, Vec<u8>)> = (0..16u8)
        .flat_map(|w| e4m3.chunks(8).map(move |c| (w, c.to_vec())))
        .collect();
    rows.chunks(16)
        .enumerate()
        .map(|(launch, rows)| {
            let mut m = Mma {
                a: [[0; 32]; 16],
                // Unused slots hold a nonzero code so a misplaced A slot shows.
                b: [[0x38; 8]; 32],
                sa: std::array::from_fn(|r| scales.0 + (r % 4) as u8),
                sb: std::array::from_fn(|c| scales.1 + (c % 3) as u8),
            };
            for (r, (w, codes)) in rows.iter().enumerate() {
                let slot = (r + launch) % 32;
                m.a[r][slot] = w << 2;
                for (c, &code) in codes.iter().enumerate() {
                    m.b[slot][c] = code;
                }
                for c in codes.len()..8 {
                    m.b[slot][c] = 0;
                }
            }
            m
        })
        .collect()
}

/// Small-integer sums: every slot nonzero, exact in F32.
pub fn integer_sums() -> Vec<Mma> {
    (0..8u8)
        .map(|s| {
            let mut m = Mma {
                a: [[0; 32]; 16],
                b: [[0; 8]; 32],
                sa: [127; 16],
                sb: [127 + s; 8],
            };
            for r in 0..16 {
                for k in 0..32 {
                    // E2M1 1, 2, 3, 4 or 6 and their negatives.
                    let code =
                        [2u8, 4, 5, 6, 7][(r + k) % 5] | if (r * 7 + k) % 3 == 0 { 8 } else { 0 };
                    m.a[r][k] = code << 2;
                }
            }
            for k in 0..32 {
                for c in 0..8 {
                    // E4M3 integers 1 through 8.
                    m.b[k][c] = [0x38u8, 0x40, 0x44, 0x48, 0x4a, 0x4c, 0x4e, 0x50][(k + 3 * c) % 8];
                }
            }
            m
        })
        .collect()
}

/// Report-only: 31 products of 2688 and one small product, whose exact
/// sums need up to 27 bits, against round to nearest.
pub fn rounding() -> Mma {
    let mut m = Mma {
        a: [[7 << 2; 32]; 16],
        b: [[0x7e; 8]; 32],
        sa: std::array::from_fn(|r| 127 - (r % 4) as u8),
        sb: [127; 8],
    };
    for r in 0..16 {
        m.a[r][31] = (1 + (r % 7) as u8) << 2;
    }
    for c in 0..8 {
        m.b[31][c] = 1 + c as u8;
    }
    m
}

pub const SCALE_PAIRS: [(u8, u8); 4] = [(127, 127), (120, 134), (100, 150), (150, 100)];

fn launch_plan() -> LaunchPlan {
    LaunchPlan {
        grid: [1, 1, 1],
        block: [32, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: false,
    }
}

fn run(runner: &mut Runner<'_>, cases: &[Mma]) -> LabResult<Vec<[[f32; 8]; 16]>> {
    let mut out = Vec::new();
    for m in cases {
        let mut a = Arena::new(Some(runner.session));
        let (ra, rb, rs) = m.registers();
        let bytes = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let va = a.bytes(&bytes(&ra))?;
        let vb = a.bytes(&bytes(&rb))?;
        let vs = a.bytes(&bytes(&rs))?;
        let vd = a.filled(128 * 4, 255)?;
        let op = Launch {
            symbol: native::PROBE.into(),
            rows: 1,
            plan: launch_plan(),
            arguments: ArgumentList(
                [va, vb, vs, vd]
                    .iter()
                    .map(|v| Argument::DeviceBuffer(v.handle))
                    .collect(),
            ),
            collective: false,
            persistent: false,
        };
        runner.run(&a, &[op])?;
        out.push(Mma::outputs(&floats(&a.read(vd)?)));
    }
    Ok(out)
}

/// The probe's executed cases and its exactness failures.
pub fn gpu(runner: &mut Runner<'_>) -> LabResult<serde_json::Value> {
    let mut executed = 0_u64;
    let mut failures = Vec::new();
    for pair in SCALE_PAIRS {
        let cases = single_products(pair);
        for (m, d) in cases.iter().zip(run(runner, &cases)?) {
            let exact = m.exact();
            for r in 0..16 {
                for c in 0..8 {
                    let products = (0..32)
                        .filter(|&k| {
                            m.a[r][k] != 0 && SmallFloat::E4M3.decode_f64(m.b[k][c]) != 0.0
                        })
                        .count();
                    assert!(products <= 1);
                    executed += 1;
                    let want = exact[r][c] as f32;
                    // Zero products compare as values: the chain starts at +0.
                    let equal = if want == 0.0 {
                        d[r][c] == 0.0
                    } else {
                        d[r][c].to_bits() == want.to_bits()
                    };
                    if f64::from(want) != exact[r][c] || !equal {
                        failures.push(serde_json::json!({"case":"single","scales":pair,"row":r,"column":c,"got":d[r][c],"want":exact[r][c]}));
                    }
                }
            }
        }
    }
    let cases = integer_sums();
    for (m, d) in cases.iter().zip(run(runner, &cases)?) {
        let exact = m.exact();
        for r in 0..16 {
            for c in 0..8 {
                executed += 1;
                if f64::from(d[r][c]) != exact[r][c] {
                    failures.push(serde_json::json!({"case":"integer_sum","row":r,"column":c,"got":d[r][c],"want":exact[r][c]}));
                }
            }
        }
    }
    let m = rounding();
    let d = run(runner, std::slice::from_ref(&m))?.remove(0);
    let exact = m.exact();
    let mut nearest = 0;
    let mut errors = Vec::new();
    for r in 0..16 {
        for c in 0..8 {
            nearest += u32::from(d[r][c] == exact[r][c] as f32);
            errors.push((f64::from(d[r][c]) - exact[r][c]) / exact[r][c].abs());
        }
    }
    Ok(serde_json::json!({
        "executed_outputs": executed,
        "failures": failures.len(),
        "first_failures": failures.iter().take(8).collect::<Vec<_>>(),
        "rounding_report_only": {"outputs":128,"round_to_nearest":nearest,"relative_errors":errors},
    }))
}

#[test]
fn probe_cases_isolate_one_product_and_cover_the_domain() {
    let cases = single_products((127, 127));
    // 16 E2M1 codes by 32 rows of up to eight E4M3 codes, 16 rows per launch.
    assert_eq!(cases.len(), 16 * 32 / 16);
    let mut seen = std::collections::BTreeSet::new();
    let mut slots = std::collections::BTreeSet::new();
    for m in &cases {
        for r in 0..16 {
            let nonzero: Vec<usize> = (0..32).filter(|&k| m.a[r][k] != 0).collect();
            assert!(nonzero.len() <= 1);
            for &k in &nonzero {
                slots.insert((r, k));
                for c in 0..8 {
                    seen.insert((m.a[r][k] >> 2, m.b[k][c]));
                }
            }
        }
    }
    // The +0 E2M1 code leaves a row empty; -0 (code 8) is a nonzero byte
    // and is placed like any other code.
    for w in 1..16u8 {
        for c in (0..=255u8).filter(|c| c & 0x7f != 0x7f) {
            assert!(seen.contains(&(w, c)), "{w} {c}");
        }
    }
    // Launches 0 and 1 hold the +0 code; the others walk all slots.
    assert_eq!(slots.len(), 16 * 32 - 2 * 16);
    assert!((0..32).all(|k| slots.iter().any(|&(_, s)| s == k)));
    // Registers round-trip through the layout model.
    let (a, b, _) = cases[3].registers();
    assert_eq!((a.len(), b.len()), (128, 64));
    let d: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let out = Mma::outputs(&d);
    let mut back = vec![0.0; 128];
    for lane in 0..32 {
        for i in 0..4 {
            back[4 * lane + i] = out[lane / 4 + 8 * (i >> 1)][2 * (lane % 4) + (i & 1)];
        }
    }
    assert_eq!(back, d);
    // Integer sums are exact in F32.
    for m in integer_sums() {
        for row in m.exact() {
            for v in row {
                assert_eq!(f64::from(v as f32), v);
            }
        }
    }
}
