//! Seeded operands that expose rounding in channel and head reductions.
use super::*;

pub(super) const SEED: u64 = 0x91e1_0da5_c79e_7b1d;
const KEY_TILE: usize = 2048;

struct Random(u64);
impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn bf16(&mut self) -> u16 {
        let bits = self.next();
        let exponent = 115 + self.next() % 25;
        ((bits >> 16) as u16 & 0x807f) | ((exponent as u16) << 7)
    }
    fn weight(&mut self) -> f32 {
        let bits = self.next();
        let exponent = 121 + self.next() % 13;
        f32::from_bits((bits & 0x807fffff) | (exponent << 23))
    }
}

pub(super) struct Operands {
    pub queries: Vec<u16>,
    pub weights: Vec<f32>,
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
}
impl Operands {
    pub fn new(rows: usize, width: usize, bf16: bool, seed: u64) -> Self {
        assert!(width > 0 && seed != 0);
        let mut rng = Random(seed);
        let mut queries = vec![0; rows * 32 * 128];
        let mut weights = Vec::with_capacity(rows * 32);
        for row in 0..rows {
            // Equal head pairs and nearly opposite weights force cancellation
            // after the rectified dot, as well as inside each channel dot.
            for pair in 0..16 {
                let first = (row * 32 + pair * 2) * 128;
                for channel in 0..128 {
                    let value = rng.bf16();
                    queries[first + channel] = value;
                    queries[first + 128 + channel] = value;
                }
                let weight = rng.weight();
                weights.extend([weight, -weight * (1.0 + 2.0_f32.powi(-20))]);
            }
        }
        let tile = width.min(KEY_TILE);
        let stride = if bf16 { 256 } else { 64 };
        let mut codes = Vec::with_capacity(width * stride);
        if bf16 {
            for _ in 0..tile * 128 {
                codes.extend(rng.bf16().to_le_bytes());
            }
        } else {
            for _ in 0..tile * 64 {
                codes.push(rng.next() as u8);
            }
        }
        let mut scales = Vec::new();
        if !bf16 {
            scales.reserve(width * 4);
            for _ in 0..tile * 4 {
                scales.push((115 + rng.next() % 25) as u8);
            }
        }
        // A large random tile keeps full-capacity fixture construction cheap;
        // identical keys in later tiles also exercise cross-shard score ties.
        while codes.len() < width * stride {
            let take = codes.len().min(width * stride - codes.len());
            codes.extend_from_within(..take);
        }
        if !bf16 {
            while scales.len() < width * 4 {
                let take = scales.len().min(width * 4 - scales.len());
                scales.extend_from_within(..take);
            }
        }
        Self {
            queries,
            weights,
            codes,
            scales,
        }
    }
    pub fn query_bytes(&self, reverse: bool) -> Vec<u8> {
        self.queries
            .chunks_exact(32 * 128)
            .flat_map(|row| {
                (0..32).flat_map(move |h| {
                    let head = if reverse { 31 - h } else { h };
                    row[head * 128..(head + 1) * 128]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                })
            })
            .collect()
    }
    pub fn weight_bytes(&self, reverse: bool) -> Vec<u8> {
        self.weights
            .chunks_exact(32)
            .flat_map(|row| {
                (0..32).flat_map(move |h| row[if reverse { 31 - h } else { h }].to_le_bytes())
            })
            .collect()
    }
    fn key(&self, position: usize, bf16: bool) -> [f32; 128] {
        std::array::from_fn(|c| {
            if bf16 {
                let at = (position * 128 + c) * 2;
                f32::from_bits(
                    u32::from(u16::from_le_bytes([self.codes[at], self.codes[at + 1]])) << 16,
                )
            } else {
                let byte = self.codes[position * 64 + c / 2];
                let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                kernel_core::codec::SmallFloat::E2M1.decode_f32(code)
                    * kernel_core::codec::decode_ue8m0_f32(self.scales[position * 4 + c / 32])
            }
        })
    }
}

#[derive(Clone, Copy)]
enum Order {
    Native,
    ReverseHeads,
    ReverseChannels,
    SplitHeadFma,
}

// The existing scalar scores_f32 oracle has a serial dot, so it cannot be
// used for bit equality with the GPU's four-term lanes and shuffle tree.
fn score(operands: &Operands, key: &[f32; 128], order: Order) -> f32 {
    let scale =
        kernel_core::position_indexer_oracle::weight_scale(family::dense(1).query.selection())
            .unwrap();
    let mut score = 0.0_f32;
    for h in 0..32 {
        let head = if matches!(order, Order::ReverseHeads) {
            31 - h
        } else {
            h
        };
        let mut lanes = [0.0_f32; 32];
        for (lane, sum) in lanes.iter_mut().enumerate() {
            for part in 0..4 {
                let part = if matches!(order, Order::ReverseChannels) {
                    3 - part
                } else {
                    part
                };
                let channel = part * 32 + lane;
                let q = f32::from_bits(u32::from(operands.queries[head * 128 + channel]) << 16);
                *sum = q.mul_add(key[channel], *sum);
            }
        }
        for offset in [16, 8, 4, 2, 1] {
            for lane in 0..offset {
                lanes[lane] += lanes[lane + offset];
            }
        }
        let dot = lanes[0].max(0.);
        let weight = operands.weights[head] * scale;
        score = if matches!(order, Order::SplitHeadFma) {
            dot * weight + score
        } else {
            dot.mul_add(weight, score)
        };
    }
    score
}

#[test]
fn cpu_arithmetic_controls_detect_reordering_and_split_fma() {
    for bf16 in [false, true] {
        for seed in [SEED, SEED ^ 0x1b03_27fa_6a72_95cd] {
            let operands = Operands::new(1, 128, bf16, seed);
            let mut mismatches = [0; 3];
            for position in 0..128 {
                let key = operands.key(position, bf16);
                let native = score(&operands, &key, Order::Native);
                assert!(native.is_finite());
                for (i, order) in [
                    Order::ReverseHeads,
                    Order::ReverseChannels,
                    Order::SplitHeadFma,
                ]
                .into_iter()
                .enumerate()
                {
                    let control = score(&operands, &key, order);
                    assert!(control.is_finite());
                    mismatches[i] += usize::from(native.to_bits() != control.to_bits());
                }
            }
            assert!(
                mismatches.iter().all(|&n| n > 0),
                "bf16={bf16} seed={seed:x} mismatches={mismatches:?}"
            );
            println!(
                "arithmetic controls bf16={bf16} seed={seed:x}: head/channel/split-FMA mismatches={mismatches:?}"
            );
        }
    }
}

pub(super) fn control_scores(operands: &Operands, bf16: bool) -> (Vec<u32>, Vec<u32>) {
    let native = (0..128)
        .map(|i| score(operands, &operands.key(i, bf16), Order::Native).to_bits())
        .collect();
    let reverse = (0..128)
        .map(|i| score(operands, &operands.key(i, bf16), Order::ReverseHeads).to_bits())
        .collect();
    (native, reverse)
}

#[test]
#[ignore = "requires an allocated idle GPU via the staged job"]
fn gpu_arithmetic_reorder_control() {
    let result = kernel_harness::gpu_test(32 << 20, |session| {
        let root = PathBuf::from(
            std::env::var_os("CHOPINFER_POSITION_LIVE_CUBINS").ok_or("prebuilt cubins required")?,
        );
        let module = session.load_cubin(
            &std::fs::read(root.join(format!("{}.cubin", family::CUBIN)))
                .map_err(|e| e.to_string())?,
        )?;
        let bf16 =
            std::env::var("CHOPINFER_POSITION_LIVE_STORAGE").map_err(|e| e.to_string())? == "bf16";
        for seed in [SEED, SEED ^ 0x1b03_27fa_6a72_95cd] {
            let operands = Operands::new(1, 128, bf16, seed);
            let (native, reversed) = control_scores(&operands, bf16);
            let native: Vec<_> = native.into_iter().flat_map(u32::to_le_bytes).collect();
            let reversed: Vec<_> = reversed.into_iter().flat_map(u32::to_le_bytes).collect();
            assert!(compare_bytes(&native, &reversed, "CPU reversed-head control").is_err());
            let mut a = Arena {
                session,
                buffers: Vec::new(),
            };
            let codes = a.input(&operands.codes)?;
            let scales = if bf16 {
                View::EMPTY
            } else {
                a.input(&operands.scales)?
            };
            let b = PositionSelectBuffers {
                q: a.input(&operands.query_bytes(false))?,
                head_weights: a.input(&operands.weight_bytes(false))?,
                keys: PositionKeyPages {
                    codes,
                    scales,
                    pages: 2,
                    code_page_stride: if bf16 { 16384 } else { 4096 },
                    scale_page_stride: if bf16 { 0 } else { 256 },
                    page_tables: BlockTables {
                        tables: a.input(&i32s([0, 1]))?,
                        sequences: 1,
                        max_pages: 2,
                    },
                },
                positions: a.input(&i32s([127]))?,
                row_batch: a.input(&i32s([0]))?,
                live_rows: a.input(&1_u32.to_le_bytes())?,
                candidate_mask: None,
                position_capacity: 128,
                scores: a.output(128 * 4)?,
                status: a.output(4)?,
                output: a.output(512 * 4)?,
            };
            for new in [false, true] {
                let op = backend(new)
                    .position_indexer_with_storage(
                        &family::dense(1).query,
                        if bf16 {
                            IndexerKeyFormat::Bf16
                        } else {
                            IndexerKeyFormat::Fp4
                        },
                        512,
                        128,
                        Need::rows(1),
                    )
                    .map_err(|e| e.to_string())?;
                let scratch = a.output((op.info().workspace as usize).max(8))?;
                let calls = op
                    .launches(&b, scratch, Counts::rows(1))
                    .map_err(|e| e.to_string())?;
                let graph = session.capture(|_| fire(session, &module, &a, &calls))?;
                for reverse in [false, true] {
                    a.write(b.q, &operands.query_bytes(reverse))?;
                    a.write(b.head_weights, &operands.weight_bytes(reverse))?;
                    a.poison(&[b.scores, b.status, b.output, scratch])?;
                    graph.launch(&session.stream).map_err(|e| e.to_string())?;
                    session.stream.synchronize().map_err(|e| e.to_string())?;
                    let actual = a.read(b.scores)?;
                    compare_bytes(
                        if reverse { &reversed } else { &native },
                        &actual,
                        "GPU score versus fixed-order CPU witness",
                    )?;
                    assert_eq!(a.read(b.status)?, 0_i32.to_le_bytes());
                    assert_eq!(
                        compare_bytes(&native, &actual, "GPU reversed-head control").is_err(),
                        reverse
                    );
                }
            }
        }
        println!("arithmetic negative control rejected on both old and live GPU paths");
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "required arithmetic control skipped");
}
