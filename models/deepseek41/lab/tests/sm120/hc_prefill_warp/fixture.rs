//! Guarded row views, including original packed origins for carried tensors.
use super::artifacts::{self, Loaded};
use kernel_core::contract::LaunchPlan;
use kernel_harness::{LabResult, Session, launch, poison};
use runtime_cuda::{DeviceBuffer, KernelArgument, Stream};
pub const H: usize = 5120;
pub const N: usize = 4;
pub const MIXES: usize = 24;
pub const GUARD: usize = 16;
const FILL: u8 = 0xa5;
static POISON: [u8; 65536] = [FILL; 65536];
fn filled(bytes: &[u8]) -> bool {
    bytes.chunks(POISON.len()).all(|c| c == &POISON[..c.len()])
}

pub struct Span {
    pub buffer: DeviceBuffer,
    pub start: usize,
    pub bytes: usize,
    pub stride: usize,
}
impl Span {
    pub fn new(s: &Session, stride: usize, rows: u32, origin: u32) -> LabResult<Self> {
        let start = GUARD + origin as usize * stride;
        let bytes = rows as usize * stride;
        let span = Self {
            buffer: s.allocate(start + bytes + GUARD)?,
            start,
            bytes,
            stride,
        };
        span.reset(s)?;
        Ok(span)
    }
    pub fn reset(&self, s: &Session) -> LabResult<()> {
        poison::fill_all(s, &self.buffer, &[FILL])
    }
    pub fn argument(&self) -> KernelArgument<'_> {
        KernelArgument::Range {
            buffer: &self.buffer,
            offset: self.start,
            byte_len: self.bytes,
        }
    }
    pub fn slice(&self, first: u32, rows: u32) -> KernelArgument<'_> {
        let offset = first as usize * self.stride;
        let bytes = rows as usize * self.stride;
        assert!(offset + bytes <= self.bytes);
        KernelArgument::Range {
            buffer: &self.buffer,
            offset: self.start + offset,
            byte_len: bytes,
        }
    }
    pub fn upload(&self, s: &Session, bytes: &[u8]) -> LabResult<()> {
        if bytes.len() != self.bytes {
            return Err("fixture input size differs".into());
        }
        poison::write(s, &self.buffer, self.start, bytes)
    }
    pub fn read(&self, s: &Session) -> LabResult<Vec<u8>> {
        poison::read(s, &self.buffer)
    }
    pub fn guards(&self, bytes: &[u8], live: u32) -> LabResult<()> {
        let tail = self.start + live as usize * self.stride;
        if bytes.len() != self.buffer.byte_len()
            || !filled(&bytes[..self.start])
            || !filled(&bytes[tail..])
        {
            return Err(format!(
                "write escaped live rows: start={} live={live} stride={}",
                self.start, self.stride
            ));
        }
        Ok(())
    }
}
pub struct Output {
    pub mix: Vec<Span>,
    pub expanded: Span,
}
impl Output {
    pub fn new(s: &Session, rows: u32, origin: u32) -> LabResult<Self> {
        Ok(Self {
            mix: [
                (H * 2, 0),
                (N * 4, origin),
                (N * 4, origin),
                (N * N * 4, origin),
                (MIXES * 4, 0),
                (4, 0),
                (N * N * 4, 0),
            ]
            .into_iter()
            .map(|(stride, start)| Span::new(s, stride, rows, start))
            .collect::<LabResult<_>>()?,
            expanded: Span::new(s, N * H * 2, rows, origin)?,
        })
    }
    pub fn reset(&self, s: &Session, expand: bool, probe: bool) -> LabResult<()> {
        if expand {
            self.expanded.reset(s)
        } else {
            for out in self.mix.iter().take(if probe { 7 } else { 4 }) {
                out.reset(s)?;
            }
            Ok(())
        }
    }
    pub fn read(
        &self,
        s: &Session,
        expand: bool,
        probe: bool,
        live: u32,
    ) -> LabResult<Vec<Vec<u8>>> {
        let spans = if expand {
            vec![&self.expanded]
        } else {
            self.mix.iter().take(if probe { 7 } else { 4 }).collect()
        };
        spans
            .into_iter()
            .map(|span| {
                let bytes = span.read(s)?;
                span.guards(&bytes, live)?;
                Ok(bytes)
            })
            .collect()
    }
}
pub struct Fixture {
    pub rows: u32,
    pub origin: u32,
    pub x: Span,
    pub y: Span,
    pub fn_: Span,
    pub base: Span,
    pub scale: Span,
    pub pre: Span,
    pub post: Span,
    pub comb: Span,
    pub live: DeviceBuffer,
    pub outputs: [Output; 2],
}
fn random(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed as u32 as f32 / u32::MAX as f32) - 0.5
}
fn f32_data(count: usize, mut seed: u64, scale: f32) -> Vec<f32> {
    (0..count).map(|_| random(&mut seed) * scale).collect()
}
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn rows_data(rows: u32, width: usize, mut seed: u64, pattern: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows as usize * width * 2);
    for i in 0..rows as usize * width {
        let v = random(&mut seed) * 4.;
        let v = match pattern {
            "zero" => -0.,
            "tiny" => f32::from_bits(if i % 2 == 0 { 0x0001_0000 } else { 0x8001_0000 }),
            "wide" => v * if i % 2 == 0 { 1e30 } else { 1e-30 },
            "nan" if i % width == 0 => f32::from_bits(0x7fc1_0000),
            "inf" if i % width == width - 1 => f32::NEG_INFINITY,
            "cancellation" => {
                if i % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            }
            _ => v,
        };
        bytes.extend(kernel_core::codec::f32_to_bf16(v).to_le_bytes());
    }
    bytes
}
impl Fixture {
    pub fn new(s: &Session, rows: u32, origin: u32, seed: u64, pattern: &str) -> LabResult<Self> {
        let f = Self {
            rows,
            origin,
            x: Span::new(s, N * H * 2, rows, origin)?,
            y: Span::new(s, H * 2, rows, origin)?,
            fn_: Span::new(s, MIXES * N * H * 4, 1, 0)?,
            base: Span::new(s, MIXES * 4, 1, 0)?,
            scale: Span::new(s, 12, 1, 0)?,
            pre: Span::new(s, N * 4, rows, origin)?,
            post: Span::new(s, N * 4, rows, origin)?,
            comb: Span::new(s, N * N * 4, rows, origin)?,
            live: s.allocate(4)?,
            outputs: [Output::new(s, rows, origin)?, Output::new(s, rows, origin)?],
        };
        f.x.upload(s, &rows_data(rows, N * H, seed + origin as u64, pattern))?;
        f.y.upload(s, &rows_data(rows, H, seed + 5, pattern))?;
        let mut weights = f32_data(MIXES * N * H, seed + 17, 0.02);
        let mut post = f32_data(rows as usize * N, seed + 31, 1.);
        let mut comb = f32_data(rows as usize * N * N, seed + 47, 0.5);
        let mut pre = f32_data(rows as usize * N, seed + 59, 1.);
        if pattern == "weight_nonfinite" {
            weights[0] = f32::NAN;
            weights[MIXES * N * H - 1] = f32::INFINITY;
            post[0] = f32::INFINITY;
            comb[N * N - 1] = f32::NAN;
            pre[0] = f32::NAN;
        }
        f.fn_.upload(s, &floats(&weights))?;
        f.base
            .upload(s, &floats(&f32_data(MIXES, seed + 61, 0.1)))?;
        f.scale.upload(s, &floats(&[0.001, 0.01, 0.002]))?;
        f.pre.upload(s, &floats(&pre))?;
        f.post.upload(s, &floats(&post))?;
        f.comb.upload(s, &floats(&comb))?;
        f.set_live(s, rows)?;
        Ok(f)
    }
    pub fn set_live(&self, s: &Session, live: u32) -> LabResult<()> {
        if live > self.rows {
            return Err("live count exceeds allocation".into());
        }
        poison::write(s, &self.live, 0, &live.to_le_bytes())
    }
    pub fn arguments(
        &self,
        arm: usize,
        expand: bool,
        probe: bool,
        first: u32,
        rows: u32,
    ) -> Vec<KernelArgument<'_>> {
        let output = &self.outputs[arm];
        let mut args = if expand {
            vec![
                self.x.slice(first, rows),
                self.y.slice(first, rows),
                self.post.slice(first, rows),
                self.comb.slice(first, rows),
                output.expanded.slice(first, rows),
            ]
        } else {
            let mut args = vec![
                self.x.slice(first, rows),
                self.fn_.argument(),
                self.base.argument(),
                self.scale.argument(),
                self.pre.slice(first, rows),
            ];
            args.extend(
                output
                    .mix
                    .iter()
                    .take(if probe { 7 } else { 4 })
                    .map(|p| p.slice(first, rows)),
            );
            args
        };
        args.push(KernelArgument::Buffer(&self.live));
        args
    }
    #[allow(clippy::too_many_arguments)]
    pub fn fire(
        &self,
        _s: &Session,
        m: &Loaded,
        stream: &Stream,
        symbol: &str,
        plan: LaunchPlan,
        arm: usize,
        probe: bool,
        pdl: bool,
    ) -> LabResult<()> {
        self.fire_range(m, stream, symbol, plan, arm, probe, pdl, 0, self.rows)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn fire_range(
        &self,
        m: &Loaded,
        stream: &Stream,
        symbol: &str,
        mut plan: LaunchPlan,
        arm: usize,
        probe: bool,
        pdl: bool,
        first: u32,
        rows: u32,
    ) -> LabResult<()> {
        plan.pdl = pdl;
        artifacts::check(symbol, plan, &m.configured)?;
        let expand = symbol.contains("expand_");
        let attribute = m.configured.get(symbol).copied().unwrap_or(48 * 1024);
        m.functions.get(symbol).ok_or_else(||format!("missing {symbol}"))?.launch(stream,launch::config(plan)?,&self.arguments(arm,expand,probe,first,rows))
            .map_err(|e|format!("{symbol} rows={} origin={} grid={:?} block={:?} cluster={:?} shared={} MAX_DYNAMIC_SHARED_SIZE_BYTES={attribute}: {e}",self.rows,self.origin,plan.grid,plan.block,plan.cluster,plan.dynamic_shared_bytes))
    }
    pub fn staged_inputs(&self, s: &Session) -> LabResult<Vec<DeviceBuffer>> {
        self.rewritten_inputs()
            .into_iter()
            .map(|b| {
                let copy = s.allocate(b.byte_len())?;
                poison::write(s, &copy, 0, &poison::read(s, b)?)?;
                Ok(copy)
            })
            .collect()
    }
    pub fn rewritten_inputs(&self) -> Vec<&DeviceBuffer> {
        vec![
            &self.x.buffer,
            &self.fn_.buffer,
            &self.pre.buffer,
            &self.y.buffer,
            &self.post.buffer,
            &self.comb.buffer,
        ]
    }
}
pub fn compare(old: &[Vec<u8>], new: &[Vec<u8>]) -> LabResult<()> {
    if old.len() != new.len() {
        return Err("output role count differs".into());
    }
    for (role, (a, b)) in old.iter().zip(new).enumerate() {
        if a == b {
            continue;
        }
        let at = a.iter().zip(b).position(|(a, b)| a != b).unwrap_or(0);
        return Err(format!(
            "role={role} byte={at} old={:?} new={:?}",
            a.get(at),
            b.get(at)
        ));
    }
    Ok(())
}

/// Poison every neighboring row while preserving the selected row exactly.
pub fn neighbors(bytes: &[u8], stride: usize, keep: u32) -> Vec<u8> {
    assert!(bytes.len().is_multiple_of(stride));
    let at = keep as usize * stride;
    assert!(at + stride <= bytes.len());
    let mut result = vec![0xff; bytes.len()];
    result[at..at + stride].copy_from_slice(&bytes[at..at + stride]);
    result
}
#[test]
fn neighbor_poison_and_full_buffer_checks_detect_boundary_mutations() {
    let data: Vec<u8> = (0..33 * 16).map(|i| i as u8).collect();
    for keep in [0, 7, 8, 15, 16, 32] {
        let altered = neighbors(&data, 16, keep);
        for (r, row) in altered.chunks(16).enumerate() {
            if r == keep as usize {
                assert_eq!(row, &data[r * 16..r * 16 + 16]);
            } else {
                assert_eq!(row, &[0xff; 16]);
            }
        }
    }
    let a = vec![vec![FILL; 65537]; 7];
    compare(&a, &a).unwrap();
    for role in 0..7 {
        for at in [0, 15, 65535, 65536] {
            let mut bad = a.clone();
            bad[role][at] ^= 1;
            assert!(compare(&a, &bad).is_err());
        }
    }
    assert!(filled(&a[0]));
    let mut bad = a[0].clone();
    bad[65536] = 0;
    assert!(!filled(&bad));
}
