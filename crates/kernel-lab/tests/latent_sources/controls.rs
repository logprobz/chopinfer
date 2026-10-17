//! Numerical witnesses over the same prepared source and merge launches.

use kernel_core::backend::{Backend, Need};
use kernel_core::codec::{SmallFloat, f32_to_bf16};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionQuery, LatentKvFormat};
use kernel_core::latent_attention_oracle::decode_native_page;
use kernel_lab::families::latent_sources_controls as family;
use kernel_lab::kit::{LabModules, Modules, OpCase, cases};
use kernel_lab::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::latent::sources::Entry;
use std::collections::BTreeMap;

const ROWS: usize = 4;
const PAD: usize = 64;
type Outputs = BTreeMap<String, Vec<u8>>;

fn bf16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

struct Fixture {
    case: OpCase<'static>,
    images: kernel_lab::conformance::Images,
    width: usize,
    heads: usize,
    slots: usize,
    window_stride: usize,
    page: usize,
    page_stride: usize,
}

impl Fixture {
    fn new(entry: &Entry) -> LabResult<Self> {
        let LatentAttentionQuery::Attention {
            heads,
            latent,
            page_size,
            ..
        } = entry.query
        else {
            return Err("control requires attention".into());
        };
        let prepared = Sm120::new()
            .with_latent_sources(entry.clone())
            .latent_attention_sources(&entry.query, Need::rows(ROWS as u32))
            .map_err(|e| e.to_string())?;
        let case = cases::latent_sources::numerical_case(entry.query, ROWS as u32, prepared)?;
        let mut images = case.images();
        let (width, heads, page) = (latent as usize, heads as usize, page_size as usize);
        let window_stride = images.bytes("window.pool")?.len() / 2;
        let slots = (window_stride - PAD) / (width + width / 32);
        let page_stride =
            LatentKvFormat::Fp4E2M1E4M3Block16.page_bytes(latent, page_size) as usize + PAD;
        let mut window = vec![0; images.bytes("window.pool")?.len()];
        for sequence in 0..2 {
            let first = sequence * window_stride + slots * width;
            window[first..first + slots * width / 32].fill(127);
        }
        let mut compressed = vec![0; images.bytes("compressed.cache.pool")?.len()];
        for physical in 0..compressed.len() / page_stride {
            let first = physical * page_stride + page * width / 2;
            compressed[first..first + page * width / 16].fill(0x38);
        }
        images = images
            .with("window.pool", window)
            .with("compressed.cache.pool", compressed)
            .with("q", vec![0; ROWS * heads * width * 2])
            .with("row_batch", i32_bytes(&[0; ROWS]))
            .with("window.key_lists", i32_bytes(&vec![-1; ROWS * 128]))
            .with("compressed.key_lists", i32_bytes(&vec![-1; ROWS * 128]))
            .with("sinks", f32_bytes(&vec![0.0; heads]))
            .with("cos_sin", f32_bytes(&[1.0, 0.0].repeat(ROWS * 32)));
        Ok(Self {
            case,
            images,
            width,
            heads,
            slots,
            window_stride,
            page,
            page_stride,
        })
    }

    fn select(
        &mut self,
        row: usize,
        window: &[(usize, i32)],
        compressed: &[(usize, i32)],
    ) -> LabResult<()> {
        for (name, selected) in [
            ("window.key_lists", window),
            ("compressed.key_lists", compressed),
        ] {
            let mut ids = self.images.i32s(name)?;
            ids[row * 128..(row + 1) * 128].fill(-1);
            for &(at, position) in selected {
                ids[row * 128 + at] = position;
            }
            self.images = self.images.with(name, i32_bytes(&ids));
        }
        Ok(())
    }

    fn query(&mut self, row: usize, values: &[f32]) -> LabResult<()> {
        assert_eq!(values.len(), self.width);
        let mut q = bf16(self.images.bytes("q")?);
        for head in 0..self.heads {
            let start = (row * self.heads + head) * self.width;
            for (out, &value) in q[start..start + self.width].iter_mut().zip(values) {
                *out = f32_to_bf16(value);
            }
        }
        self.images = self.images.with("q", kernel_lab::kit::bf16_bytes(&q));
        Ok(())
    }

    fn sink(&mut self, value: f32) {
        self.images = self
            .images
            .with("sinks", f32_bytes(&vec![value; self.heads]));
    }

    fn window_raw(&mut self, slot: usize, codes: &[u8], scales: &[u8]) -> LabResult<()> {
        assert_eq!((codes.len(), scales.len()), (self.width, self.width / 32));
        let mut pool = self.images.bytes("window.pool")?.to_vec();
        pool[slot * self.width..(slot + 1) * self.width].copy_from_slice(codes);
        let at = self.slots * self.width + slot * (self.width / 32);
        pool[at..at + scales.len()].copy_from_slice(scales);
        self.images = self.images.with("window.pool", pool);
        Ok(())
    }

    fn window(&mut self, slot: usize, values: &[f32]) -> LabResult<()> {
        let codes = values
            .iter()
            .map(|&v| SmallFloat::E4M3.encode_f32(v))
            .collect::<Vec<_>>();
        self.window_raw(slot, &codes, &vec![127; self.width / 32])
    }

    fn compressed_raw(&mut self, position: usize, codes: &[u8], scales: &[u8]) -> LabResult<()> {
        assert_eq!(
            (codes.len(), scales.len()),
            (self.width / 2, self.width / 16)
        );
        let tables = self.images.i32s("compressed.block_tables.tables")?;
        let physical = tables[position / self.page] as usize;
        let slot = position % self.page;
        let mut pool = self.images.bytes("compressed.cache.pool")?.to_vec();
        let first = physical * self.page_stride + slot * self.width / 2;
        pool[first..first + codes.len()].copy_from_slice(codes);
        let first =
            physical * self.page_stride + self.page * self.width / 2 + slot * self.width / 16;
        pool[first..first + scales.len()].copy_from_slice(scales);
        self.images = self.images.with("compressed.cache.pool", pool);
        Ok(())
    }

    fn compressed(&mut self, position: usize, values: &[f32]) -> LabResult<()> {
        let codes = values
            .chunks_exact(2)
            .map(|pair| {
                SmallFloat::E2M1.encode_f32(pair[0]) | SmallFloat::E2M1.encode_f32(pair[1]) << 4
            })
            .collect::<Vec<_>>();
        self.compressed_raw(position, &codes, &vec![0x38; self.width / 16])
    }

    fn run(
        &self,
        session: &Session,
        functions: &kernel_lab::kit::Lookup<'_>,
        live: u32,
    ) -> LabResult<Outputs> {
        self.case.launch(session, functions, &self.images, live)
    }
    fn judge(&self, outputs: &Outputs, live: u32) -> LabResult<()> {
        self.case
            .judge_outputs(&self.images, live, outputs)
            .map(|_| ())
    }
}

#[test]
fn native_numerical_controls_prepare() {
    let entries = family::catalog();
    assert_eq!(entries.len(), family::REQUIRED);
    for entry in &entries {
        Fixture::new(entry).unwrap().case.validate().unwrap();
    }
    let specs = family::cubins();
    assert_eq!(specs[0].entries.len(), 5);
    assert!(entries[0].sink_per_split_entry_spec().is_err());
    assert!(entries[1].sink_per_split_entry_spec().is_ok());
}

#[test]
fn native_sink_is_applied_once_and_per_split_control_is_rejected() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let entry = family::catalog()
            .into_iter()
            .find(|e| e.splits > 1)
            .unwrap();
        let mut fixture = Fixture::new(&entry)?;
        let mut q = vec![0.0; fixture.width];
        q[0] = 1.0;
        let mut first = q.clone();
        first[1] = 2.0;
        let mut second = q.clone();
        second[1] = 6.0;
        fixture.query(0, &q)?;
        fixture.window(0, &first)?;
        fixture.compressed(0, &second)?;
        fixture.select(0, &[(0, 0)], &[(0, 0)])?;
        let LatentAttentionQuery::Attention {
            softmax_scale_bits, ..
        } = entry.query
        else {
            unreachable!()
        };
        fixture.sink(f32::from_bits(softmax_scale_bits));
        let functions = LabModules.load(session)?;
        let correct = fixture.run(session, functions.as_ref(), 1)?;
        fixture.judge(&correct, 1)?;
        assert_eq!(bf16(&correct["output"])[1], f32_to_bf16(8.0 / 3.0));
        let mutant = entry.sink_per_split_entry_spec().unwrap().symbol;
        let merge = entry.merge_symbol();
        let wrong = |symbol: &str| functions(if symbol == merge { &mutant } else { symbol });
        let result = fixture.run(session, &wrong, 1)?;
        assert_eq!(
            bf16(&result["output"])[1],
            f32_to_bf16(8.0 / (2.0 + entry.splits as f32))
        );
        let failure = fixture
            .judge(&result, 1)
            .expect_err("the numerical gate must reject the per-split sink");
        assert!(
            failure.contains("BF16 element") && failure.contains("outside"),
            "{failure}"
        );
        session.receipt(
            "native-latent-sink-control",
            BTreeMap::from([
                ("splits".into(), serde_json::json!(entry.splits)),
                ("negative_control".into(), serde_json::json!(failure)),
            ]),
        )?;
        Ok(())
    })
    .expect("native sink ownership");
}

#[test]
fn native_probability_cancellation_meets_the_tile_independent_bound() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let entry = family::catalog()
            .into_iter()
            .find(|e| {
                matches!(
                    e.query,
                    LatentAttentionQuery::Attention {
                        phase: AttentionPhase::Prefill,
                        ..
                    }
                )
            })
            .unwrap();
        let functions = LabModules.load(session)?;
        let mut count = 0;
        for positions in [(0, 1), (63, 64), (126, 127)] {
            let mut f = Fixture::new(&entry)?;
            let mut q = vec![0.0; f.width];
            q[0] = -22.625;
            let mut a = vec![0.0; f.width];
            a[1] = 1.0;
            let mut b = vec![0.0; f.width];
            b[0] = 1.0;
            b[1] = -2.75;
            f.query(0, &q)?;
            f.window(0, &a)?;
            f.window(1, &b)?;
            f.select(0, &[(positions.0, 0), (positions.1, 1)], &[])?;
            f.judge(&f.run(session, functions.as_ref(), 1)?, 1)?;
            count += 1;
        }
        let mut f = Fixture::new(&entry)?;
        let mut q = vec![0.0; f.width];
        q[0] = -22.625;
        let mut a = vec![0.0; f.width];
        a[1] = 1.0;
        let mut b = vec![0.0; f.width];
        b[0] = 1.0;
        b[1] = -3.0;
        f.query(0, &q)?;
        f.window(0, &a)?;
        f.compressed(0, &b)?;
        f.select(0, &[(127, 0)], &[(0, 0)])?;
        f.judge(&f.run(session, functions.as_ref(), 1)?, 1)?;
        count += 1;
        assert_eq!(count, 4);
        session.receipt(
            "native-latent-probability-bound",
            BTreeMap::from([("cases".into(), serde_json::json!(count))]),
        )?;
        Ok(())
    })
    .expect("native probability rounding");
}

#[test]
fn native_code_scale_products_reach_the_bf16_tile_exactly() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let functions = LabModules.load(session)?;
        let mut checked = 0;
        for entry in family::catalog() {
            for compressed in [false, true] {
                let mut f = Fixture::new(&entry)?;
                f.sink(-1000.0);
                let mut expected = Vec::new();
                for row in 0..ROWS {
                    if compressed {
                        let codes = (0..f.width / 2)
                            .map(|i| ((2 * i % 16) | ((2 * i + 1) % 16) << 4) as u8)
                            .collect::<Vec<_>>();
                        let scales = (0..f.width / 16)
                            .map(|b| ((row * 32 + b) % 126 + 1) as u8)
                            .collect::<Vec<_>>();
                        f.compressed_raw(row, &codes, &scales)?;
                        f.select(row, &[], &[(0, row as i32)])?;
                        let physical =
                            f.images.i32s("compressed.block_tables.tables")?[row / f.page] as usize;
                        expected.push(
                            decode_native_page(
                                LatentKvFormat::Fp4E2M1E4M3Block16,
                                f.width,
                                f.page,
                                &f.images.bytes("compressed.cache.pool")?
                                    [physical * f.page_stride..],
                                row % f.page,
                            )
                            .map_err(|e| e.to_string())?,
                        );
                    } else {
                        let codes = (0..f.width)
                            .map(|i| {
                                let v = (i + row * 17) % 254;
                                (v + usize::from(v >= 127)) as u8
                            })
                            .collect::<Vec<_>>();
                        let scales = (0..f.width / 32)
                            .map(|b| [105, 119, 126, 127, 128, 136, 144, 160][(b + row) % 8])
                            .collect::<Vec<_>>();
                        f.window_raw(row, &codes, &scales)?;
                        f.select(row, &[(0, row as i32)], &[])?;
                        expected.push(
                            decode_native_page(
                                LatentKvFormat::Fp8E4M3Ue8M0Block32,
                                f.width,
                                f.slots,
                                &f.images.bytes("window.pool")?[..f.window_stride],
                                row,
                            )
                            .map_err(|e| e.to_string())?,
                        );
                    }
                }
                let result = f.run(session, functions.as_ref(), ROWS as u32)?;
                let actual = bf16(&result["output"]);
                for (row, expected) in expected.iter().enumerate() {
                    for head in 0..f.heads {
                        for (d, &value) in expected.iter().enumerate() {
                            let got = actual[(row * f.heads + head) * f.width + d];
                            let want = f32_to_bf16(value as f32);
                            assert_eq!(
                                if want & 0x7fff == 0 {
                                    got & 0x7fff
                                } else {
                                    got
                                },
                                if want & 0x7fff == 0 { 0 } else { want },
                                "compressed={compressed} row={row} head={head} channel={d}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 2 * 2 * ROWS * 4 * 512);
        session.receipt(
            "native-latent-decoding",
            BTreeMap::from([("bf16_values".into(), serde_json::json!(checked))]),
        )?;
        Ok(())
    })
    .expect("native codes and scales decode exactly");
}

#[test]
fn native_inverse_rotation_uses_the_observed_bf16_attention_boundary() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let functions = LabModules.load(session)?;
        for entry in family::catalog() {
            let mut f = Fixture::new(&entry)?;
            let mut a = vec![0.0; f.width];
            let mut b = vec![0.0; f.width];
            for d in f.width - 64..f.width {
                a[d] = ((d % 9) as f32 - 4.0) * 0.25;
                b[d] = ((d % 7) as f32 - 3.0) * 0.5;
            }
            a[f.width - 64] = 1.0;
            a[f.width - 63] = 3.0;
            b[f.width - 64] = 0.0;
            b[f.width - 63] = 0.0;
            f.window(0, &a)?;
            f.window(1, &b)?;
            f.select(0, &[(0, 0), (1, 1)], &[])?;
            let identity = f.run(session, functions.as_ref(), 1)?;
            f.judge(&identity, 1)?;
            let before = bf16(&identity["output"]);
            let mut table = Vec::new();
            for row in 0..ROWS {
                for pair in 0..32 {
                    let angle = (row * 17 + pair * 7 + 1) as f32 / 13.0;
                    let (s, c) = angle.sin_cos();
                    table.extend([c, s]);
                }
            }
            table[..2].copy_from_slice(&[0.6, 0.8]);
            f.images = f.images.with("cos_sin", f32_bytes(&table));
            let rotated = bf16(&f.run(session, functions.as_ref(), 1)?["output"]);
            cases::latent_sources::check_inverse_tail(
                &before[..f.heads * f.width],
                &rotated[..f.heads * f.width],
                f.width,
                f.heads,
                64,
                &table[..64],
            )?;
            assert_eq!(rotated[f.width - 63], f32_to_bf16(0.33203125));
        }
        session.receipt(
            "native-latent-inverse-boundary",
            BTreeMap::from([("entries".into(), serde_json::json!(family::REQUIRED))]),
        )?;
        Ok(())
    })
    .expect("native inverse rotation boundary");
}
