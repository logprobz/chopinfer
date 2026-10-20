use super::{FLAT, OFFSET, ROLES, WIDTH, artifacts, artifacts::Site, catalog};
use kernel_core::codec::f32_to_bf16;
use kernel_harness::{LabResult, Session, launch};
use kernels_sm120::sinkhorn_hc::prefill_fused::Entry;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Module, Stream};
pub const STRIDES: [usize; 9] = [
    FLAT * 2,
    WIDTH * 2,
    16,
    16,
    64,
    WIDTH * 2,
    WIDTH * 4,
    WIDTH,
    WIDTH / 32,
];
pub struct Fixture {
    pub rows: u32,
    pub bytes: Vec<Vec<u8>>,
}
impl Fixture {
    pub fn new(site: &Site, rows: u32, seed: u64, pattern: &str) -> Self {
        let mut state = seed;
        let mut bytes = vec![Vec::new(); 9];
        for row in 0..rows as usize {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let source = state as usize % 8;
            let rotation = (state >> 32) as usize % WIDTH;
            for (index, role, width) in [(0, "streams", FLAT), (1, "sum", WIDTH)] {
                for col in 0..width {
                    let at =
                        source * width + (col / WIDTH) * WIDTH + (col % WIDTH + rotation) % WIDTH;
                    let original = if index == 0 {
                        u16::from_le_bytes(
                            site.inputs[role][at * 2..at * 2 + 2].try_into().unwrap(),
                        )
                    } else {
                        f32_to_bf16(f32::from_le_bytes(
                            site.inputs[role][at * 4..at * 4 + 4].try_into().unwrap(),
                        ))
                    };
                    let value = match pattern {
                        "zero" => {
                            if (row + col) % 2 == 0 {
                                0
                            } else {
                                0x8000
                            }
                        }
                        "tiny" => ((row + col) % 127 + 1) as u16 | (original & 0x8000),
                        "maximum" => 0x7f7f | (original & 0x8000),
                        "nan" if col % WIDTH == 0 => 0x7fc1,
                        "inf" if col % WIDTH == WIDTH - 1 => 0xff80,
                        "cancellation" => {
                            if col % 2 == 0 {
                                0x4780
                            } else {
                                0xb780
                            }
                        }
                        _ => original,
                    };
                    bytes[index].extend(value.to_le_bytes());
                }
            }
            for (i, role, stride) in [(2, "post_in", 16), (3, "comb_in", 64), (7, "pre_in", 16)] {
                bytes[i].extend(&site.inputs[role][source * stride..(source + 1) * stride]);
            }
        }
        for (i, role) in [(4, "fn"), (5, "base"), (6, "scale"), (8, "norm_weight")] {
            bytes[i] = site.inputs[role].clone();
        }
        if pattern == "weight_nan" {
            bytes[8][0..2].copy_from_slice(&0x7fc1u16.to_le_bytes());
        }
        Self { rows, bytes }
    }
    pub fn select(&self, indices: &[usize]) -> Self {
        let mut bytes = self.bytes.clone();
        for (index, stride) in [(0, FLAT * 2), (1, WIDTH * 2), (2, 16), (3, 64), (7, 16)] {
            bytes[index] = indices
                .iter()
                .flat_map(|r| {
                    self.bytes[index][r * stride..(r + 1) * stride]
                        .iter()
                        .copied()
                })
                .collect();
        }
        Self {
            rows: indices.len() as u32,
            bytes,
        }
    }
}
fn upload(b: &DeviceBuffer, bytes: &[u8], stream: &Stream) -> LabResult<()> {
    b.copy_from_bytes(bytes, stream).map_err(|e| e.to_string())
}
pub struct Output {
    pub buffers: Vec<DeviceBuffer>,
    sizes: Vec<usize>,
}
impl Output {
    fn new(s: &Session, rows: u32) -> LabResult<Self> {
        let sizes: Vec<_> = STRIDES
            .into_iter()
            .map(|n| n * rows as usize)
            .chain([(rows as usize + OFFSET + 2) * 4])
            .collect();
        Ok(Self {
            buffers: sizes
                .iter()
                .map(|&n| s.allocate(n))
                .collect::<LabResult<_>>()?,
            sizes,
        })
    }
    pub fn read(&self, s: &Session) -> LabResult<Vec<Vec<u8>>> {
        self.buffers
            .iter()
            .zip(&self.sizes)
            .map(|(b, &size)| {
                let mut bytes = vec![0; size];
                b.copy_to_bytes(&mut bytes, &s.stream)
                    .map_err(|e| e.to_string())?;
                Ok(bytes)
            })
            .collect()
    }
}
pub struct Device<'a> {
    pub s: &'a Session,
    pub entry: Entry,
    pub rows: u32,
    pub widen: bool,
    pub alias: bool,
    inputs: Vec<DeviceBuffer>,
    pub live: DeviceBuffer,
    pub outputs: [Output; 2],
    host_coefficients: [Vec<u8>; 3],
    functions: Vec<Function>,
    _modules: Vec<Module>,
}
impl<'a> Device<'a> {
    pub fn new(
        s: &'a Session,
        entry: &Entry,
        f: &Fixture,
        widen: bool,
        alias: bool,
    ) -> LabResult<Self> {
        entry.plan(f.rows, f.rows).map_err(|e| e.to_string())?;
        let expected = [
            FLAT * 2 * f.rows as usize,
            WIDTH * 2 * f.rows as usize,
            16 * f.rows as usize,
            64 * f.rows as usize,
            24 * FLAT * 4,
            96,
            12,
            16 * f.rows as usize,
            WIDTH * 2,
        ];
        if f.bytes.iter().map(Vec::len).collect::<Vec<_>>() != expected {
            return Err("fixture extents differ".into());
        }
        let inputs = f
            .bytes
            .iter()
            .map(|b| {
                let d = s.allocate(b.len())?;
                upload(&d, b, &s.stream)?;
                Ok(d)
            })
            .collect::<LabResult<Vec<_>>>()?;
        let live = s.allocate(4)?;
        upload(&live, &f.rows.to_ne_bytes(), &s.stream)?;
        let symbols = [
            catalog::expand(f.rows).0.symbol,
            catalog::mix(f.rows).0.symbol,
            catalog::norm().reference.symbol,
            entry.symbol.clone(),
        ];
        let modules = symbols
            .iter()
            .map(|sym| s.load_cubin(&artifacts::module_for(sym)?))
            .collect::<LabResult<Vec<_>>>()?;
        let functions = modules
            .iter()
            .zip(symbols)
            .map(|(m, s)| m.function(&s).map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        let outputs = [Output::new(s, f.rows)?, Output::new(s, f.rows)?];
        // Every concurrent invocation owns distinct inputs and outputs. The
        // candidate has no global workspace or synchronization tickets.
        let mut ranges = inputs
            .iter()
            .chain(outputs.iter().flat_map(|o| &o.buffers))
            .map(|b| {
                b.device_address_at(0, b.byte_len())
                    .map(|address| (address, b.byte_len() as u64))
                    .map_err(|e| e.to_string())
            })
            .collect::<LabResult<Vec<_>>>()?;
        ranges.sort_unstable();
        if ranges.windows(2).any(|p| p[0].0 + p[0].1 > p[1].0) {
            return Err("overlapping allocation".into());
        }
        Ok(Self {
            s,
            entry: entry.clone(),
            rows: f.rows,
            widen,
            alias,
            inputs,
            live,
            outputs,
            host_coefficients: [f.bytes[7].clone(), f.bytes[2].clone(), f.bytes[3].clone()],
            functions,
            _modules: modules,
        })
    }
    pub fn set_live(&self, live: u32) -> LabResult<()> {
        self.entry
            .plan(self.rows, live)
            .map_err(|e| e.to_string())?;
        upload(&self.live, &live.to_ne_bytes(), &self.s.stream)
    }
    pub fn reset(&self, arm: usize, sticky: bool) -> LabResult<()> {
        for (i, b) in self.outputs[arm].buffers.iter().enumerate() {
            let bytes = if i == 9 {
                (0..self.rows as usize + OFFSET + 2)
                    .flat_map(|j| {
                        let value = if j < OFFSET || j >= OFFSET + self.rows as usize {
                            0x1357_2468u32
                        } else if sticky {
                            0x80
                        } else {
                            0
                        };
                        value.to_ne_bytes()
                    })
                    .collect()
            } else if self.alias && (2..5).contains(&i) {
                self.host_coefficients[i - 2].clone()
            } else {
                vec![0xa5; self.outputs[arm].sizes[i]]
            };
            upload(b, &bytes, &self.s.stream)?;
        }
        Ok(())
    }
    pub fn launch(&self, arm: usize, stream: &Stream) -> LabResult<()> {
        let i = &self.inputs;
        let o = &self.outputs[arm].buffers;
        let pre = if self.alias { &o[2] } else { &i[7] };
        let post = if self.alias { &o[3] } else { &i[2] };
        let comb = if self.alias { &o[4] } else { &i[3] };
        let wide = if self.widen {
            A::Buffer(&o[6])
        } else {
            A::NullBuffer
        };
        let eps = self.entry.query.norm.norm.epsilon_bits;
        if arm == 1 {
            self.functions[3]
                .launch(
                    stream,
                    launch::config(self.entry.plan(self.rows, 0).unwrap())?,
                    &[
                        A::Buffer(&i[0]),
                        A::Buffer(&i[1]),
                        A::Buffer(post),
                        A::Buffer(comb),
                        A::Buffer(&o[0]),
                        A::Buffer(&i[4]),
                        A::Buffer(&i[5]),
                        A::Buffer(&i[6]),
                        A::Buffer(pre),
                        A::Buffer(&o[1]),
                        A::Buffer(&o[2]),
                        A::Buffer(&o[3]),
                        A::Buffer(&o[4]),
                        A::Buffer(&i[8]),
                        A::Buffer(&o[5]),
                        wide,
                        A::Buffer(&o[7]),
                        A::Buffer(&o[8]),
                        A::Buffer(&self.live),
                        A::Buffer(&o[9]),
                        A::U64(OFFSET as u64),
                        A::U32(eps),
                    ],
                )
                .map_err(|e| e.to_string())
        } else {
            self.functions[0]
                .launch(
                    stream,
                    launch::config(catalog::expand(self.rows).1)?,
                    &[
                        A::Buffer(&i[0]),
                        A::Buffer(&i[1]),
                        A::Buffer(post),
                        A::Buffer(comb),
                        A::Buffer(&o[0]),
                        A::Buffer(&self.live),
                    ],
                )
                .map_err(|e| e.to_string())?;
            self.functions[1]
                .launch(
                    stream,
                    launch::config(catalog::mix(self.rows).1)?,
                    &[
                        A::Buffer(&o[0]),
                        A::Buffer(&i[4]),
                        A::Buffer(&i[5]),
                        A::Buffer(&i[6]),
                        A::Buffer(pre),
                        A::Buffer(&o[1]),
                        A::Buffer(&o[2]),
                        A::Buffer(&o[3]),
                        A::Buffer(&o[4]),
                        A::Buffer(&self.live),
                    ],
                )
                .map_err(|e| e.to_string())?;
            self.functions[2]
                .launch(
                    stream,
                    launch::config(catalog::norm().plan(self.rows).unwrap())?,
                    &[
                        A::Buffer(&o[1]),
                        A::Buffer(&i[8]),
                        A::Buffer(&o[5]),
                        wide,
                        A::Buffer(&o[7]),
                        A::Buffer(&o[8]),
                        A::Buffer(&self.live),
                        A::Buffer(&o[9]),
                        A::U64(OFFSET as u64),
                        A::U32(eps),
                    ],
                )
                .map_err(|e| e.to_string())
        }
    }
    pub fn once(&self, arm: usize, sticky: bool) -> LabResult<Vec<Vec<u8>>> {
        self.reset(arm, sticky)?;
        self.launch(arm, &self.s.stream)?;
        self.s.stream.synchronize().map_err(|e| e.to_string())?;
        self.outputs[arm].read(self.s)
    }
    pub fn guards(&self, values: &[Vec<u8>], live: u32, sticky: bool) -> LabResult<()> {
        if !self.alias {
            return super::fusion_checks::guards(
                &ROLES,
                values,
                self.rows,
                live,
                if self.widen { &[] } else { &[6] },
                OFFSET,
                sticky,
            );
        }
        let mut copy = values.to_vec();
        if self.alias {
            for i in 2..5 {
                let start = STRIDES[i] * live as usize;
                if copy[i][start..] != self.host_coefficients[i - 2][start..] {
                    return Err("inactive coefficient alias changed".into());
                }
                copy[i][start..].fill(0xa5);
            }
        }
        super::fusion_checks::guards(
            &ROLES,
            &copy,
            self.rows,
            live,
            if self.widen { &[] } else { &[6] },
            OFFSET,
            sticky,
        )
    }
}
