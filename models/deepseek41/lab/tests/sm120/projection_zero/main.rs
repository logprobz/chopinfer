//! Bitwise projection specialization against the unchanged prefill dispatch.
use bundle_deepseek41::sm120::{f32_prefill, fold_cubin_for_symbol};
use kernel_core::contract::LaunchPlan;
use kernel_harness::{LabResult, Session, poison};
use kernels_sm120::f32_projection::{Entry, zero_decode, zero_prefill as zero};
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Module, Stream};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};
mod cpu;
mod data;
#[path = "../../../sm120/projection_zero.rs"]
mod family;
mod sweep;
mod timing;
fn n() -> usize {
    family::entry().query().n as usize
}
const K: usize = 5120;
fn rows() -> Vec<u32> {
    vec![1, 8, 32, 448, 512, 960, 2048, 2496, 4096, 8191, 8192]
}
fn tiles(rows: u32) -> Vec<u32> {
    if rows <= 32 { vec![32] } else { vec![32, 64] }
}
fn cases(mem: bool) -> Vec<(u32, &'static str)> {
    if mem {
        return ["bf16", "residual", "mixed", "weight", "edges"]
            .into_iter()
            .flat_map(|p| [(8, p), (129, p)])
            .collect();
    }
    let mut v = rows().into_iter().map(|r| (r, "bf16")).collect::<Vec<_>>();
    for r in [8, 32, 129, 512] {
        for p in ["residual", "mixed", "weight", "edges"] {
            v.push((r, p));
        }
    }
    v
}
fn old(rows: u32) -> Entry {
    let e = family::entry();
    let selected = f32_prefill::serving_entry(&FLASH, e.clone().with_split_decode())
        .unwrap()
        .entry_for_rows(rows);
    if rows <= 32 || selected.uses_prefill_tiles() {
        selected
    } else {
        e
    }
}
fn stage(grid: [u32; 3]) -> LaunchPlan {
    LaunchPlan {
        grid,
        block: [128, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1; 3],
        pdl: true,
    }
}
fn old_plan(e: &Entry, r: u32) -> LaunchPlan {
    if e.uses_prefill_tiles() {
        e.prefill_plan(r).unwrap()
    } else {
        e.plan(r).unwrap()
    }
}
fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_PROJECTION_ZERO_ROOT")
        .expect("staged root")
        .into()
}
struct Functions {
    map: BTreeMap<String, Function>,
    _modules: Vec<Module>,
}
impl Functions {
    fn load(s: &Session) -> LabResult<Self> {
        let mut map = BTreeMap::new();
        let mut modules = Vec::new();
        let module = s.load_cubin(
            &std::fs::read(
                root()
                    .join("cubins")
                    .join(format!("{}.cubin", family::CUBIN)),
            )
            .map_err(|e| e.to_string())?,
        )?;
        for spec in family::cubins() {
            for e in spec.entries {
                map.insert(
                    e.symbol.clone(),
                    module.function(&e.symbol).map_err(|e| e.to_string())?,
                );
            }
        }
        modules.push(module);
        for (name, bytes, symbols) in cpu::embedded() {
            let module = s.load_cubin(bytes)?;
            for symbol in symbols {
                map.insert(
                    symbol.clone(),
                    module
                        .function(&symbol)
                        .map_err(|e| format!("{name} {symbol}: {e}"))?,
                );
            }
            modules.push(module);
        }
        for (symbol, p) in cpu::launches() {
            let f = &map[&symbol];
            f.set_max_dynamic_shared_memory(p.dynamic_shared_bytes)
                .map_err(|e| {
                    format!(
                        "{symbol}: MAX_DYNAMIC_SHARED_SIZE_BYTES={} block={:?}: {e}",
                        p.dynamic_shared_bytes, p.block
                    )
                })?;
        }
        for m in [32, 64] {
            let symbol = zero::symbol(&family::entry(), m);
            let p = zero::plan(&family::entry(), m, 8192).unwrap();
            let resident = map[&symbol]
                .max_active_clusters(kernel_harness::launch::config(p)?, false)
                .map_err(|e| e.to_string())?;
            if resident == 0 {
                return Err(format!("{symbol}: zero occupancy"));
            }
            println!(
                "PROJECTION_ZERO_OCCUPANCY {}",
                json!({"tile":m,"symbol":symbol,"resident_ctas":resident,"threads":p.block[0],"shared_bytes":p.dynamic_shared_bytes})
            );
        }
        Ok(Self {
            map,
            _modules: modules,
        })
    }
    fn call(&self, symbol: &str, stream: &Stream, p: LaunchPlan, args: &[A]) -> LabResult<()> {
        self.map[symbol].launch(stream,kernel_harness::launch::config(p)?,args).map_err(|e|format!("{symbol} grid={:?} block={:?} cluster={:?} MAX_DYNAMIC_SHARED_SIZE_BYTES={}: {e}",p.grid,p.block,p.cluster,p.dynamic_shared_bytes))
    }
}
struct Buffers {
    reference: Entry,
    x: DeviceBuffer,
    w: DeviceBuffer,
    domain: DeviceBuffer,
    work: [DeviceBuffer; 2],
    live: DeviceBuffer,
    out: [DeviceBuffer; 2],
}
impl Buffers {
    fn new(s: &Session, x: &[f32], w: &[u16], rows: u32) -> LabResult<Self> {
        let e = family::entry();
        let work = [
            s.allocate(
                e.prefill_workspace_bytes(rows)
                    .max(e.clone().with_decode().decode_workspace_bytes(rows))
                    as usize,
            )?,
            s.allocate(
                zero::workspace_bytes(&e, rows)
                    .max(e.clone().with_decode().decode_workspace_bytes(rows))
                    as usize,
            )?,
        ];
        for b in &work {
            poison::fill_all(s, b, &[0xa5])?;
        }
        Ok(Self {
            reference: old(rows),
            x: s.upload_f32(x)?,
            w: s.upload_bf16(w)?,
            domain: s.allocate(n() / 16 * 4)?,
            work,
            live: s.allocate(4)?,
            out: [
                s.allocate((rows as usize * n() + 16) * 4)?,
                s.allocate((rows as usize * n() + 16) * 4)?,
            ],
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        f: &Functions,
        stream: &Stream,
        rows: u32,
        m: u32,
        arm: usize,
        pdl: bool,
        mutation: Option<&str>,
        prep_only: bool,
    ) -> LabResult<()> {
        let e = family::entry();
        let old = &self.reference;
        let mut prepared = stage([rows, 1, 1]);
        prepared.pdl = pdl;
        if rows <= 32 && mutation != Some("omit_chain") && mutation != Some("force_certificate") {
            let cached = e.clone().with_decode();
            f.call(
                &cached.stage_symbol("_rows"),
                stream,
                prepared,
                &[
                    A::Buffer(&self.x),
                    A::Buffer(&self.live),
                    A::U32(rows),
                    A::Buffer(&self.work[arm]),
                ],
            )?;
            if prep_only {
                return Ok(());
            }
            let name = if arm == 0 {
                cached.stage_symbol("_decode")
            } else if mutation == Some("force_decode") {
                family::control(&e, "force_decode")
            } else {
                zero_decode::symbol(&e)
            };
            let mut p = cached.decode_plan(rows).unwrap();
            p.pdl = pdl;
            f.call(
                &name,
                stream,
                p,
                &[
                    A::Buffer(&self.x),
                    A::Buffer(&self.w),
                    A::Buffer(&self.work[arm]),
                    A::Buffer(&self.live),
                    A::U32(rows),
                    A::Buffer(&self.domain),
                ],
            )?;
            let mut p = stage([(rows * n() as u32).div_ceil(128), 1, 1]);
            p.pdl = pdl;
            return f.call(
                &cached.stage_symbol("_reduce"),
                stream,
                p,
                &[
                    A::Buffer(&self.work[arm]),
                    A::Buffer(&self.out[arm]),
                    A::Buffer(&self.live),
                    A::U32(rows),
                ],
            );
        }
        if arm == 1 || old.uses_prefill_tiles() {
            let name = if arm == 0 {
                old.stage_symbol("_prepare")
            } else if mutation == Some("force_certificate") {
                family::control(&e, "force_certificate")
            } else {
                zero::prepare_symbol(&e)
            };
            f.call(
                &name,
                stream,
                prepared,
                &[
                    A::Buffer(&self.x),
                    A::Buffer(&self.live),
                    A::U32(rows),
                    A::Buffer(&self.work[arm]),
                ],
            )?;
        }
        if prep_only {
            return Ok(());
        }
        let (name, mut p) = if arm == 0 {
            (old.symbol().to_string(), old_plan(old, rows))
        } else {
            (
                if mutation == Some("omit_chain") {
                    family::control(&e, "omit_chain")
                } else {
                    zero::symbol(&e, m)
                },
                zero::plan(&e, m, rows).unwrap(),
            )
        };
        p.pdl = pdl;
        if arm == 0 && !old.uses_prefill_tiles() {
            f.call(
                &name,
                stream,
                p,
                &[
                    A::Buffer(&self.x),
                    A::Buffer(&self.w),
                    A::Buffer(&self.out[arm]),
                    A::Buffer(&self.live),
                    A::U32(rows),
                ],
            )
        } else {
            f.call(
                &name,
                stream,
                p,
                &[
                    A::Buffer(&self.x),
                    A::Buffer(&self.w),
                    A::Buffer(&self.work[arm]),
                    A::Buffer(&self.out[arm]),
                    A::Buffer(&self.live),
                    A::U32(rows),
                    A::Buffer(&self.domain),
                ],
            )
        }
    }
    fn classify(&self, f: &Functions, s: &Session) -> LabResult<()> {
        f.call(
            &family::entry()
                .with_prefill_tiles(32)
                .unwrap()
                .stage_symbol("_weights"),
            &s.stream,
            stage([n() as u32 / 16, 1, 1]),
            &[A::Buffer(&self.w), A::Buffer(&self.domain)],
        )
    }
}
#[allow(clippy::too_many_arguments)]
fn compare(
    s: &Session,
    f: &Functions,
    x: &[f32],
    w: &[u16],
    rows: u32,
    m: u32,
    label: &str,
    timed: bool,
) -> LabResult<Vec<u8>> {
    compare_with_reference(s, f, x, w, rows, m, label, timed, old(rows))
}
#[allow(clippy::too_many_arguments)]
fn compare_with_reference(
    s: &Session,
    f: &Functions,
    x: &[f32],
    w: &[u16],
    rows: u32,
    m: u32,
    label: &str,
    timed: bool,
    reference: Entry,
) -> LabResult<Vec<u8>> {
    let mut b = Buffers::new(s, x, w, rows)?;
    b.reference = reference;
    b.classify(f, s)?;
    let graphs = [
        s.capture(|st| b.run(f, st, rows, m, 0, true, None, false))?,
        s.capture(|st| b.run(f, st, rows, m, 1, true, None, false))?,
    ];
    let mut result = Vec::new();
    for live in [rows, rows - 1, 0, rows] {
        b.live
            .copy_from_bytes(&live.to_le_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
        for graph in [false, true] {
            for o in &b.out {
                poison::fill_all(s, o, &[0xa5])?;
            }
            for (arm, g) in graphs.iter().enumerate() {
                if graph {
                    g.launch(&s.stream).map_err(|e| e.to_string())?;
                } else {
                    b.run(f, &s.stream, rows, m, arm, false, None, false)?;
                }
            }
            let out = [poison::read(s, &b.out[0])?, poison::read(s, &b.out[1])?];
            if out[0] != out[1] {
                let at = out[0]
                    .iter()
                    .zip(&out[1])
                    .position(|(a, b)| a != b)
                    .unwrap();
                return Err(format!(
                    "rows={rows} m={m} {label} live={live} graph={graph}: mismatch byte {at}"
                ));
            }
            if out[1][live as usize * n() * 4..].iter().any(|&b| b != 0xa5) {
                return Err("dead output or guard overwritten".into());
            }
            if live > 0 && out[1][..live as usize * n() * 4].iter().all(|&b| b == 0xa5) {
                return Err("live output left poisoned".into());
            }
            result = out[1][..rows as usize * n() * 4].to_vec();
        }
    }
    if timed {
        let pair = timing::pair(s, false, |arm, st| {
            graphs[arm].launch(st).map_err(|e| e.to_string())
        })?;
        let cold = timing::pair(s, true, |arm, st| {
            graphs[arm].launch(st).map_err(|e| e.to_string())
        })?;
        let prepare = if b.reference.uses_prefill_tiles() {
            Some(timing::pair(s, false, |arm, st| {
                b.run(f, st, rows, m, arm, true, None, true)
            })?)
        } else {
            None
        };
        println!(
            "PROJECTION_ZERO_TIME {}",
            json!({"rows":rows,"tile":m,"label":label,"arms":["base_prefill_dispatch","zero_components"],"reference":b.reference.symbol(),"includes_prepare":true,"timing":pair,"cold":cold,"prepare":prepare})
        );
    }
    Ok(result)
}
fn negative(s: &Session, f: &Functions, mutation: &str) -> LabResult<()> {
    let rows = if mutation == "force_decode" { 8 } else { 129 };
    let value = if mutation == "force_certificate" || mutation == "force_decode" {
        f32::from_bits(0x3f800080)
    } else {
        1.
    };
    let b = Buffers::new(
        s,
        &vec![value; rows as usize * K],
        &vec![0x3f80; n() * K],
        rows,
    )?;
    b.classify(f, s)?;
    b.live
        .copy_from_bytes(&rows.to_le_bytes(), &s.stream)
        .map_err(|e| e.to_string())?;
    for arm in 0..2 {
        b.run(f, &s.stream, rows, 32, arm, false, Some(mutation), false)?;
    }
    if poison::read(s, &b.out[0])?[..rows as usize * n() * 4]
        == poison::read(s, &b.out[1])?[..rows as usize * n() * 4]
    {
        return Err(format!("ineffective negative control {mutation}"));
    }
    println!(
        "PROJECTION_ZERO_NEGATIVE {}",
        json!({"mutation":mutation,"changed":true})
    );
    Ok(())
}
fn run(s: &Session) -> LabResult<()> {
    let worker: usize = std::env::var("CHOPINFER_PROJECTION_ZERO_WORKER")
        .map_err(|e| e.to_string())?
        .parse()
        .map_err(|_| "worker")?;
    let mem = std::env::var("CHOPINFER_PROJECTION_ZERO_MEMCHECK").as_deref() == Ok("1");
    let f = Functions::load(s)?;
    for (rows, pattern) in cases(mem) {
        let (x, w) = data::synthetic(rows, pattern, 91 + worker as u32);
        for m in tiles(rows) {
            compare(s, &f, &x, &w, rows, m, pattern, !mem && pattern == "bf16")?;
            println!(
                "PROJECTION_ZERO_BITS {}",
                json!({"rows":rows,"tile":m,"label":pattern,"comparisons":8,"bitwise":true})
            );
        }
    }
    for mutation in ["omit_chain", "force_certificate", "force_decode"] {
        negative(s, &f, mutation)?;
    }
    // Row zero is certified in both batches. The next row needs residual MMAs.
    for m in [32, 64] {
        let (mut x, w) = data::synthetic(129, "bf16", 701);
        let alone = compare(s, &f, &x[..K], &w, 1, m, "isolation", false)?;
        x[K] = f32::from_bits(0x3f800080);
        x[2 * K] = f32::from_bits(1);
        let mixed = compare(s, &f, &x, &w, 129, m, "isolation", false)?;
        if alone != mixed[..n() * 4] {
            return Err("neighbor changed row bits".into());
        }
        println!(
            "PROJECTION_ZERO_ISOLATION {}",
            json!({"tile":m,"bitwise":true})
        );
    }
    if !mem {
        let layer = [0, 39][worker];
        let (x, w) = data::captured(s, &f, layer)?;
        for rows in [1, 8, 32] {
            compare(
                s,
                &f,
                &x[..rows * K],
                &w,
                rows as u32,
                32,
                "capture_prefix",
                true,
            )?;
            println!(
                "PROJECTION_ZERO_CAPTURE_PREFIX {}",
                json!({"layer":layer,"rows":rows,"comparisons":8,"bitwise":true})
            );
        }
        for m in [32, 64] {
            let full = compare(s, &f, &x, &w, 8192, m, "capture", true)?;
            println!(
                "PROJECTION_ZERO_CAPTURE {}",
                json!({"layer":layer,"rows":8192,"tile":m,"bitwise":true,"comparisons":8})
            );
            for (first, count) in [(0, 129), (129, 320), (449, 2048), (2497, 5695)] {
                let part = compare(
                    s,
                    &f,
                    &x[first * K..(first + count) * K],
                    &w,
                    count as u32,
                    m,
                    "partition",
                    false,
                )?;
                if part != full[first * n() * 4..(first + count) * n() * 4] {
                    return Err("chunk partition changed output".into());
                }
                println!(
                    "PROJECTION_ZERO_PARTITION {}",
                    json!({"layer":layer,"tile":m,"first":first,"rows":count,"bitwise":true})
                );
            }
        }
    }
    println!(
        "PROJECTION_ZERO_DONE {}",
        json!({"n":n(),"worker":worker,"memcheck":mem,"cases":cases(mem).iter().map(|(r,_)|tiles(*r).len()).sum::<usize>(),"comparisons":cases(mem).iter().map(|(r,_)|tiles(*r).len()*8).sum::<usize>(),"capture":if mem{0}else{2},"negative":3,"isolation":2,"partitions":if mem{0}else{8}})
    );
    Ok(())
}
#[test]
#[ignore = "requires CHOPINFER_PROJECTION_ZERO_ROOT and an assigned idle GPU"]
fn projection_gpu() {
    gpu_run(run);
}
fn gpu_run(run: fn(&Session) -> LabResult<()>) {
    use std::sync::OnceLock;
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    static TABLE: OnceLock<Vec<(&'static str, &'static [u8])>> = OnceLock::new();
    let bytes = BYTES.get_or_init(|| {
        std::fs::read(root().join("cubins/probe_sm120.cubin")).expect("frozen timing probe")
    });
    let cubins = kernel_harness::Cubins::new(
        TABLE.get_or_init(|| vec![("probe_sm120", bytes)]),
        &[(
            "probe_sm120",
            &["chop_probe_flush_v1", "chop_probe_read_v1"],
        )],
    );
    assert!(
        kernel_harness::gpu_test_with(cubins, 4 << 30, run)
            .unwrap()
            .is_some()
    );
}
