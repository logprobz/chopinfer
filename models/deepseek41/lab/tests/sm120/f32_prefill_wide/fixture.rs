use super::{catalog::*, root};
use kernel_core::contract::LaunchPlan;
use kernel_harness::{LabResult, Session, eviction::EvictionMode, launch, poison};
use kernels_sm120::f32_projection::{Entry, blocked, wide};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Module, Stream};
use serde_json::json;
use std::collections::BTreeMap;

pub struct Modules(BTreeMap<String, Module>);
impl Modules {
    pub fn load(s: &Session) -> LabResult<Self> {
        let mut names = vec![CUBIN.to_owned(), baseline_module()];
        for e in entries().iter().filter(|e| matches!(e.query().n, 32 | 384)) {
            names.push(generic(e).0.into());
        }
        names.sort();
        names.dedup();
        let mut modules = BTreeMap::new();
        for name in names {
            let bytes =
                std::fs::read(root().join(format!("{name}.cubin"))).map_err(|e| e.to_string())?;
            modules.insert(name, s.load_cubin(&bytes)?);
        }
        Ok(Self(modules))
    }
    fn function(&self, module: &str, symbol: &str) -> LabResult<Function> {
        self.0
            .get(module)
            .ok_or_else(|| format!("missing module {module}"))?
            .function(symbol)
            .map_err(|e| format!("{module}/{symbol}: {e}"))
    }
}
fn call(f: &Function, s: &Stream, p: LaunchPlan, args: &[A]) -> LabResult<()> {
    f.launch(s, launch::config(p)?, args).map_err(|e| {
        format!(
            "grid={:?} block={:?} cluster={:?} shared={} MAX_DYNAMIC_SHARED_SIZE_BYTES={}: {e}",
            p.grid, p.block, p.cluster, p.dynamic_shared_bytes, p.dynamic_shared_bytes
        )
    })
}
fn configure(f: &Function, p: LaunchPlan, symbol: &str) -> LabResult<()> {
    f.set_max_dynamic_shared_memory(p.dynamic_shared_bytes)
        .map_err(|e| {
            format!(
                "{symbol}: MAX_DYNAMIC_SHARED_SIZE_BYTES={} block={:?}: {e}",
                p.dynamic_shared_bytes, p.block
            )
        })
}
struct Buffers {
    x: DeviceBuffer,
    w: DeviceBuffer,
    domain: DeviceBuffer,
    work: DeviceBuffer,
    live: DeviceBuffer,
    out: [DeviceBuffer; 2],
}
fn operands(e: &Entry, rows: u32, pattern: &str) -> (Vec<f32>, Vec<f32>) {
    let q = e.query();
    let (mut a, mut w) = super::super::f32_decode::random_operands(rows, q.n, q.k, 9817 + rows);
    if pattern == "bf16" {
        for x in &mut a {
            *x = f32::from_bits(x.to_bits() & 0xffff0000);
        }
    }
    if pattern == "row_fallback" {
        for row in 0..rows as usize {
            a[row * q.k as usize + q.k as usize - 1] = f32::from_bits(1);
        }
    }
    if pattern == "weight_fallback" || pattern == "edges" {
        for col in 0..q.n as usize {
            w[col * q.k as usize + q.k as usize - 1] = f32::from_bits(0x00800000);
        }
    }
    if pattern == "mixed" {
        for row in (1..rows as usize).step_by(3) {
            a[row * q.k as usize + q.k as usize - 1] = f32::from_bits(1);
        }
        for col in (16..q.n as usize).step_by(32) {
            w[col * q.k as usize + q.k as usize - 1] = f32::from_bits(0x00800000);
        }
    }
    if pattern == "edges" {
        let bits = [
            0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x7f7fffff, 0x7f800000, 0xff800000,
            0x7fc12345,
        ];
        for row in 0..rows as usize {
            a[row * q.k as usize + 127] = f32::from_bits(bits[row % bits.len()]);
        }
        let bits = [
            0, 0x80000000, 0x00010000, 0x007f0000, 0x7f7f0000, 0x7f800000, 0xff800000, 0x7fc10000,
            0x7f810000,
        ];
        for col in 0..q.n as usize {
            w[col * q.k as usize + 127] = f32::from_bits(bits[col % bits.len()]);
        }
    }
    (a, w)
}
impl Buffers {
    fn new(s: &Session, e: &Entry, rows: u32, pattern: &str) -> LabResult<Self> {
        let (a, w) = operands(e, rows, pattern);
        let live = s.allocate(4)?;
        poison::write(s, &live, 0, &rows.to_ne_bytes())?;
        Ok(Self {
            x: s.upload_f32(&a)?,
            w: s.upload_bf16(
                &w.iter()
                    .map(|v| (v.to_bits() >> 16) as u16)
                    .collect::<Vec<_>>(),
            )?,
            domain: s.allocate(e.weight_domain_bytes() as usize)?,
            work: s.allocate(
                e.clone()
                    .with_prefill_tiles(32)
                    .unwrap()
                    .prefill_workspace_bytes(rows) as usize,
            )?,
            live,
            out: [
                s.allocate((rows as usize * e.query().n as usize + 16) * 4)?,
                s.allocate((rows as usize * e.query().n as usize + 16) * 4)?,
            ],
        })
    }
}
fn classify(s: &Session, m: &Modules, e: &Entry, b: &Buffers) -> LabResult<()> {
    let symbol = e
        .clone()
        .with_prefill_tiles(32)
        .unwrap()
        .stage_symbol("_weights");
    call(
        &m.function(&baseline_module(), &symbol)?,
        &s.stream,
        stage_plan([e.query().n.div_ceil(16), 1, 1]),
        &[A::Buffer(&b.w), A::Buffer(&b.domain)],
    )
}
#[allow(
    clippy::too_many_arguments,
    reason = "the fixture binds both prepared and generic launch ABIs"
)]
fn prepared_run(
    m: &Modules,
    s: &Stream,
    e: &Entry,
    b: &Buffers,
    rows: u32,
    f: &Function,
    p: LaunchPlan,
    arm: usize,
) -> LabResult<()> {
    let symbol = e
        .clone()
        .with_prefill_tiles(32)
        .unwrap()
        .stage_symbol("_prepare");
    call(
        &m.function(&baseline_module(), &symbol)?,
        s,
        stage_plan([rows, 1, 1]),
        &[
            A::Buffer(&b.x),
            A::Buffer(&b.live),
            A::U32(rows),
            A::Buffer(&b.work),
        ],
    )?;
    call(
        f,
        s,
        p,
        &[
            A::Buffer(&b.x),
            A::Buffer(&b.w),
            A::Buffer(&b.work),
            A::Buffer(&b.out[arm]),
            A::Buffer(&b.live),
            A::U32(rows),
            A::Buffer(&b.domain),
        ],
    )
}
fn check_output(output: &[Vec<u8>; 2], live: u32, n: u32, label: &str) -> LabResult<()> {
    if output[0] != output[1] {
        let at = output[0]
            .iter()
            .zip(&output[1])
            .position(|(a, b)| a != b)
            .unwrap();
        return Err(format!("{label}: mismatch byte {at}"));
    }
    let live_bytes = live as usize * n as usize * 4;
    if live > 0
        && output[0][..live_bytes]
            .chunks_exact(4)
            .any(|v| v == [0xa5; 4])
    {
        return Err(format!("{label}: live output remains poisoned"));
    }
    if output[1][live_bytes..].iter().any(|&v| v != 0xa5) {
        return Err(format!("{label}: inactive output or guard overwritten"));
    }
    Ok(())
}
pub fn compare(s: &Session, m: &Modules, c: Case) -> LabResult<()> {
    let base_module = baseline_module();
    let e = &entries()[c.entry];
    let q = e.query();
    let label = c.json().to_string();
    let b = Buffers::new(s, e, c.rows, c.pattern)?;
    let f = [
        if c.wide {
            m.function(&baseline_module(), &blocked::symbol(e, 32))?
        } else {
            m.function(generic(e).0, e.baseline_symbol())?
        },
        m.function(if c.wide { CUBIN } else { &base_module }, &c.symbol(e))?,
    ];
    let p = [
        if c.wide {
            blocked::plan(e, 32, c.rows).unwrap()
        } else {
            e.plan(c.rows).unwrap()
        },
        c.plan(e),
    ];
    for arm in 0..2 {
        configure(&f[arm], p[arm], &label)?;
    }
    classify(s, m, e, &b)?;
    let run = |arm: usize, stream: &Stream| -> LabResult<()> {
        if arm == 0 && !c.wide {
            call(
                &f[arm],
                stream,
                p[arm],
                &[
                    A::Buffer(&b.x),
                    A::Buffer(&b.w),
                    A::Buffer(&b.out[arm]),
                    A::Buffer(&b.live),
                    A::U32(c.rows),
                ],
            )
        } else {
            prepared_run(m, stream, e, &b, c.rows, &f[arm], p[arm], arm)
        }
        .map_err(|e| format!("{label} arm={arm}: {e}"))
    };
    let graphs = [
        s.capture(|stream| run(0, stream))?,
        s.capture(|stream| run(1, stream))?,
    ];
    for (phase, live) in [c.rows, c.rows - 1, 0, c.rows].into_iter().enumerate() {
        poison::write(s, &b.live, 0, &live.to_ne_bytes())?;
        for graph in [false, true] {
            for (arm, executable) in graphs.iter().enumerate() {
                poison::fill_all(s, &b.out[arm], &[0xa5])?;
                poison::fill_all(s, &b.work, &[0xa5])?;
                if graph {
                    executable.launch(&s.stream).map_err(|e| e.to_string())?;
                } else {
                    run(arm, &s.stream)?;
                }
            }
            s.stream.synchronize().map_err(|e| e.to_string())?;
            check_output(
                &[poison::read(s, &b.out[0])?, poison::read(s, &b.out[1])?],
                live,
                q.n,
                &label,
            )?;
            println!(
                "F32_PREFILL_BITS {}",
                json!({"case":c.json(),"phase":phase,"live":live,"graph":graph})
            );
        }
    }
    if c.timed() {
        let paired = kernel_harness::time_ab(s, 2, 7, [EvictionMode::Write; 2], |arm, stream| {
            graphs[arm].launch(stream).map_err(|e| e.to_string())
        })?;
        let bytes = u64::from(c.rows) * u64::from(q.k) * 4
            + e.weight_bytes()
            + u64::from(c.rows) * u64::from(q.n) * 4;
        println!(
            "F32_PREFILL_TIMING {}",
            json!({"case":c.json(),"old_us":paired.arms[0].median_ms*1000.,"new_us":paired.arms[1].median_ms*1000.,"ratio":paired.paired_ratio,"rounds":7,"unique_bytes":bytes,"dram_floor_us":bytes as f64/1_600_000.})
        );
    }
    Ok(())
}

pub fn occupancy(m: &Modules) -> LabResult<()> {
    let base_module = baseline_module();
    for e in entries() {
        for (kind, tile) in [(false, 32), (false, 64), (true, 64), (true, 128)] {
            let symbol = if kind {
                wide::symbol(&e, tile)
            } else {
                blocked::symbol(&e, tile)
            };
            let p = if kind {
                wide::plan(&e, tile, 8192)
            } else {
                blocked::plan(&e, tile, 8192)
            }
            .unwrap();
            let f = m.function(if kind { CUBIN } else { &base_module }, &symbol)?;
            configure(&f, p, &symbol)?;
            let mut query = launch::config(p)?;
            query.grid = (65536, 1, 1);
            let active = f.max_active_clusters(query, false).map_err(|e| {
                format!(
                    "{symbol}: occupancy block={:?} dynamic_shared={}: {e}",
                    p.block, p.dynamic_shared_bytes
                )
            })?;
            if active == 0 {
                return Err(format!("{symbol}: no resident CTA"));
            }
            println!(
                "F32_PREFILL_OCCUPANCY {}",
                json!({"symbol":symbol,"block":p.block,"shared":p.dynamic_shared_bytes,"active_unit_clusters":active,"sm_count":188,"warps_per_sm":f64::from(active)*f64::from(p.block[0])/32./188.})
            );
        }
    }
    Ok(())
}

pub fn controls(s: &Session, m: &Modules) -> LabResult<()> {
    let e = entries().into_iter().find(|e| e.query().n == 384).unwrap();
    let rows = 129;
    for (tile, mutation, pattern) in [
        (64, "chain", "weight_fallback"),
        (128, "neighbor", "tensor"),
    ] {
        let b = Buffers::new(s, &e, rows, pattern)?;
        if mutation == "neighbor" {
            poison::write(
                s,
                &b.x,
                (13 * e.query().k as usize + 127) * 4,
                &1u32.to_ne_bytes(),
            )?;
        }
        classify(s, m, &e, &b)?;
        let p = wide::plan(&e, tile, rows).unwrap();
        let functions = [
            m.function(CUBIN, &wide::symbol(&e, tile))?,
            m.function(CUBIN, &format!("chop_f32_wide_{mutation}_negative_v1"))?,
        ];
        for (arm, function) in functions.iter().enumerate() {
            configure(function, p, mutation)?;
            poison::fill_all(s, &b.work, &[0xa5])?;
            poison::fill_all(s, &b.out[arm], &[0xa5])?;
            prepared_run(m, &s.stream, &e, &b, rows, function, p, arm)?;
        }
        let out = [poison::read(s, &b.out[0])?, poison::read(s, &b.out[1])?];
        let width = e.query().n as usize * 4;
        let differs = (0..rows as usize)
            .filter(|&row| mutation != "neighbor" || row != 13)
            .any(|row| {
                out[0][row * width..(row + 1) * width] != out[1][row * width..(row + 1) * width]
            });
        if !differs {
            return Err(format!("{mutation}: negative control accepted"));
        }
        println!(
            "F32_PREFILL_NEGATIVE {}",
            json!({"mutation":mutation,"rows":rows,"tile":tile})
        );
    }
    Ok(())
}
pub fn isolation(s: &Session, m: &Modules) -> LabResult<()> {
    let base_module = baseline_module();
    let e = entries().into_iter().find(|e| e.query().n == 384).unwrap();
    let rows = 129;
    for (kind, tile) in [(false, 32), (false, 64), (true, 64), (true, 128)] {
        let b = Buffers::new(s, &e, rows, "tensor")?;
        classify(s, m, &e, &b)?;
        let symbol = if kind {
            wide::symbol(&e, tile)
        } else {
            blocked::symbol(&e, tile)
        };
        let f = m.function(if kind { CUBIN } else { &base_module }, &symbol)?;
        let p = if kind {
            wide::plan(&e, tile, rows)
        } else {
            blocked::plan(&e, tile, rows)
        }
        .unwrap();
        configure(&f, p, &symbol)?;
        for (change, bits) in [1u32, 0x7fc12345].into_iter().enumerate() {
            for arm in 0..2 {
                let original =
                    operands(&e, rows, "tensor").0[13 * e.query().k as usize + 127].to_bits();
                poison::write(
                    s,
                    &b.x,
                    (13 * e.query().k as usize + 127) * 4,
                    &if arm == 0 { original } else { bits }.to_ne_bytes(),
                )?;
                poison::fill_all(s, &b.out[arm], &[0xa5])?;
                poison::fill_all(s, &b.work, &[0xa5])?;
                prepared_run(m, &s.stream, &e, &b, rows, &f, p, arm)?;
            }
            let out = [poison::read(s, &b.out[0])?, poison::read(s, &b.out[1])?];
            let width = e.query().n as usize * 4;
            for row in (0..rows as usize).filter(|&r| r != 13) {
                if out[0][row * width..(row + 1) * width] != out[1][row * width..(row + 1) * width]
                {
                    return Err(format!("{symbol}: neighboring row changed output {row}"));
                }
            }
            println!(
                "F32_PREFILL_ISOLATION {}",
                json!({"symbol":symbol,"rows":rows,"changed_row":13,"change":change})
            );
        }
    }
    Ok(())
}
