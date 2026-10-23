//! Unselected exact native FP8 staging matrix against the serving route.
use bundle_deepseek41::sm120::{fold_cubin_for_symbol, mxfp8_fold as registration};
use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers as Buffers;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{LabResult, Session};
use kernels_sm120::{backend::Sm120, gemm::mxfp8_prefill_compact::Entry};
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::FLASH};
use runtime_cuda::{DeviceBuffer, Function, Stream};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
#[path = "../../../sm120/compact_dense.rs"]
mod family;
mod matrix;
#[path = "../exact_prefill/timing.rs"]
mod timing;
const ENV: &str = "CHOPINFER_DEEPSEEK41_COMPACT_DENSE_CUBINS";
const ROWS: &[u32] = &[
    1, 16, 17, 32, 33, 64, 65, 127, 128, 129, 257, 512, 2048, 4096, 8192,
];
const TIMED: &[u32] = &[128, 512, 2048, 4096, 8192];
struct Arena<'a> {
    session: Option<&'a Session>,
    next: u64,
    buffers: BTreeMap<u64, DeviceBuffer>,
}
impl<'a> Arena<'a> {
    fn new(session: Option<&'a Session>) -> Self {
        Self {
            session,
            next: 0,
            buffers: BTreeMap::new(),
        }
    }
    fn filled(&mut self, bytes: u64, value: u8) -> LabResult<View> {
        self.next += 1;
        let v = View {
            handle: self.next,
            bytes,
        };
        if let Some(s) = self.session {
            // The PDL rewrite probe copies whole u32 words. Keep the
            // logical view exact while padding its allocation, including
            // odd row counts with K=576 (18 scale bytes per row).
            let b = s.allocate((bytes as usize).div_ceil(4) * 4)?;
            kernel_harness::poison::fill_all(s, &b, &[value])?;
            self.buffers.insert(v.handle, b);
        }
        Ok(v)
    }
    fn bytes(&mut self, bytes: &[u8]) -> LabResult<View> {
        let v = self.filled(bytes.len() as u64, 0)?;
        self.write(v, bytes)?;
        Ok(v)
    }
    fn write(&self, v: View, bytes: &[u8]) -> LabResult<()> {
        if let Some(s) = self.session {
            if bytes.len() != v.bytes as usize {
                return Err("host write extent differs".into());
            }
            self.buffers[&v.handle]
                .copy_from_bytes(bytes, &s.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn fill(&self, v: View, value: u8) -> LabResult<()> {
        kernel_harness::poison::fill_all(self.session.unwrap(), &self.buffers[&v.handle], &[value])
    }
    fn read(&self, v: View) -> LabResult<Vec<u8>> {
        kernel_harness::poison::read(self.session.unwrap(), &self.buffers[&v.handle])
    }
}
fn seed(count: usize, scale: bool, salt: u32) -> Vec<u8> {
    (0..count)
        .map(|i| {
            let mut x = (i as u32)
                .wrapping_add(salt)
                .wrapping_mul(747796405)
                .wrapping_add(2891336453);
            x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
            x ^= x >> 22;
            if scale {
                120 + (x % 13) as u8
            } else {
                (x % 127) as u8 | ((x >> 16) as u8 & 128)
            }
        })
        .collect()
}
fn topology(e: &Entry) -> Vec<u32> {
    SUPPORTED_RANKS
        .iter()
        .copied()
        .filter(|&r| {
            registration::prefill_entries(&FLASH, r)
                .unwrap()
                .iter()
                .any(|x| x.query == e.base.query)
        })
        .collect()
}
fn buffers(
    a: &mut Arena<'_>,
    e: &Entry,
    rows: u32,
    weights: [View; 2],
) -> LabResult<Buffers<View>> {
    let b = e.base.bytes(rows);
    Ok(Buffers {
        activations: a.filled(b.activations, 0)?,
        activation_scales: a.filled(b.activation_scales, 127)?,
        weights: weights[0],
        weight_scales: weights[1],
        output: a.filled(b.output, 255)?,
        live_rows: a.bytes(&rows.to_le_bytes())?,
        status: a.filled(b.status + 20, 0)?,
        status_row_offset: 3,
    })
}
fn prepare(
    a: &mut Arena<'_>,
    e: &Entry,
    rows: u32,
    b: &Buffers<View>,
    candidate: bool,
) -> LabResult<Vec<Launch>> {
    let mut b = *b;
    if !candidate {
        b.weights = if a.session.is_some() {
            let q = e.base.query;
            let codes = a.read(b.weights)?;
            let packed = kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout {
                groups: q.groups as usize,
                n: q.n as usize,
                k: q.k as usize,
            }
            .pack(&codes[..b.weights.bytes as usize])
            .map_err(|e| e.to_string())?;
            a.bytes(&packed)?
        } else {
            a.filled(b.weights.bytes, 0)?
        };
    }
    let op = if candidate {
        e.prepare(rows).map_err(|e| e.to_string())?
    } else {
        let rank = topology(e)[0];
        let backend = registration::register(Sm120::new(), &FLASH, rank)?;
        let backend = bundle_deepseek41::sm120::prefill_matrix::register(backend, &FLASH, rank)?;
        let backend =
            bundle_deepseek41::sm120::prefill_tma_scales::register(backend, &FLASH, rank)?
                .with_packed_fold_weights();
        if rows >= 128 {
            backend.mxfp8_prefill_gemm(&e.base.query, Need::rows(rows))
        } else {
            backend.mxfp8_fold_gemm(&e.base.query, Need::rows(rows))
        }
        .map_err(|e| e.to_string())?
    };
    op.launches(&b, View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.to_string())
}
fn functions(s: &Session, ops: &[Launch]) -> LabResult<BTreeMap<String, Function>> {
    let mut modules = BTreeMap::new();
    let mut out = BTreeMap::new();
    for op in ops {
        if out.contains_key(&op.symbol) {
            continue;
        }
        let name = if op.symbol.contains("_compact_") && !op.symbol.ends_with("_packed_v1") {
            family::CUBIN
        } else {
            fold_cubin_for_symbol(&op.symbol)
                .ok_or_else(|| format!("missing baseline {}", op.symbol))?
                .0
        };
        if !modules.contains_key(name) {
            let bytes = if name == family::CUBIN {
                std::fs::read(
                    std::path::Path::new(&std::env::var(ENV).map_err(|e| e.to_string())?)
                        .join(format!("{name}.cubin")),
                )
                .map_err(|e| e.to_string())?
            } else {
                fold_cubin_for_symbol(&op.symbol).unwrap().1.to_vec()
            };
            modules.insert(name, s.load_cubin(&bytes)?);
        }
        let f = modules[name]
            .function(&op.symbol)
            .map_err(|e| e.to_string())?;
        if op.plan.dynamic_shared_bytes > 0 {
            f.set_max_dynamic_shared_memory(op.plan.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
        }
        out.insert(op.symbol.clone(), f);
    }
    Ok(out)
}
fn issue(
    s: &Session,
    a: &Arena<'_>,
    f: &BTreeMap<String, Function>,
    ops: &[Launch],
    stream: &Stream,
) -> LabResult<()> {
    let b = Bindings {
        buffers: a.buffers.iter().map(|(&h, b)| (h, b)).collect(),
    };
    for op in ops {
        launch(
            &s.context,
            &f[&op.symbol],
            stream,
            op.plan,
            &op.arguments,
            &b,
        )?;
    }
    Ok(())
}
fn status(rows: u32, live: u32, bad: bool) -> Vec<u8> {
    (0..rows + 5)
        .flat_map(|i| {
            let v = if i < 3 || i >= rows + 3 {
                0xabcddcba_u32
            } else if bad && i == 7 && live > 4 {
                0x80
            } else {
                0
            };
            v.to_le_bytes()
        })
        .collect()
}
fn reset(a: &Arena<'_>, b: &Buffers<View>, rows: u32, live: u32, bad: bool) -> LabResult<()> {
    a.fill(b.output, 255)?;
    a.write(b.status, &status(rows, live, bad))?;
    a.write(b.live_rows, &live.to_le_bytes())
}
fn equality(
    a: &Arena<'_>,
    old: &Buffers<View>,
    new: &Buffers<View>,
    rows: u32,
    live: u32,
    bad: bool,
) -> LabResult<Vec<u8>> {
    let x = a.read(old.output)?;
    let y = a.read(new.output)?;
    let xs = a.read(old.status)?;
    let ys = a.read(new.status)?;
    if x != y || xs != ys {
        return Err(format!(
            "bitwise mismatch at rows={rows} live={live}: output={:?}, status={}",
            x.iter().zip(&y).position(|(a, b)| a != b),
            xs != ys
        ));
    }
    let width = x.len() / rows as usize;
    if x[live as usize * width..].iter().any(|&b| b != 255) {
        return Err("inactive output overwritten".into());
    }
    if x[..live as usize * width]
        .chunks_exact(4)
        .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
    {
        return Err("nonfinite live output".into());
    }
    let mut expected = status(rows, live, bad);
    if bad {
        for row in [1, 2] {
            if row < live {
                expected[(row as usize + 3) * 4] = 1;
            }
        }
    }
    if xs != expected {
        return Err("admission or sticky status differs".into());
    }
    if bad {
        for row in [1, 2, 4] {
            if row < live
                && x[row as usize * width..(row as usize + 1) * width]
                    .iter()
                    .any(|&b| b != 0)
            {
                return Err("refused row not zero".into());
            }
        }
    }
    Ok(x)
}
#[test]
fn cpu_matrix() {
    let specs = family::cubins();
    let symbols = specs[0]
        .entries
        .iter()
        .map(|e| e.symbol.clone())
        .collect::<BTreeSet<_>>();
    let mut embedded = BTreeMap::<String, Value>::new();
    let mut cases = Vec::new();
    for e in family::entries() {
        let mut a = Arena::new(None);
        let b = e.base.bytes(1);
        let weights = [
            a.filled(b.weights, 0).unwrap(),
            a.filled(b.weight_scales, 0).unwrap(),
        ];
        for &rows in ROWS {
            let b = buffers(&mut a, &e, rows, weights).unwrap();
            let old = prepare(&mut a, &e, rows, &b, false).unwrap();
            let new = prepare(&mut a, &e, rows, &b, true).unwrap();
            for op in old.iter().chain(&new) {
                op.plan.validate_geometry().unwrap();
                assert_eq!(op.plan.workspace_bytes, 0);
                if op.symbol.contains("_compact_") && !op.symbol.ends_with("_packed_v1") {
                    assert!(symbols.contains(&op.symbol));
                    assert!(op.plan.dynamic_shared_bytes <= 101376);
                } else {
                    let (name, bytes) = fold_cubin_for_symbol(&op.symbol).unwrap();
                    assert!(!bytes.is_empty());
                    embedded.insert(name.into(), json!({"sha256":kernel_build::digest(bytes)}));
                }
            }
            cases.push(json!({"n":e.base.query.n,"k":e.base.query.k,"groups":e.base.query.groups,"stages":e.stages,"rows":rows,"topologies":topology(&e),"baseline_symbols":old.iter().map(|o|&o.symbol).collect::<Vec<_>>(),"candidate_symbols":new.iter().map(|o|&o.symbol).collect::<Vec<_>>()}));
        }
    }
    let (bytes, exports) = bundle_deepseek41::sm120::dependency_probes();
    assert!(exports.contains(&kernels_sm120::probes::LIVE));
    embedded.insert(
        kernels_sm120::probes::CUBIN.into(),
        json!({"sha256":kernel_build::digest(bytes)}),
    );
    assert!(!cases.is_empty());
    println!(
        "compact_dense_inventory {}",
        json!({"family":family::FAMILY,"cases":cases,"required_ordinals":1,"candidate_module":family::CUBIN,"symbols":symbols,"embedded_modules":embedded,"timing_rows":TIMED,"correctness_rows":ROWS,"serving_enabled":false})
    );
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_COMPACT_DENSE_CUBINS"]
fn bitwise_gpu() {
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 8 << 30, |s| {
        matrix::run(s, false)
    })
    .unwrap();
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_COMPACT_DENSE_CUBINS"]
fn timing_gpu() {
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 8 << 30, |s| {
        matrix::run(s, true)
    })
    .unwrap();
}
