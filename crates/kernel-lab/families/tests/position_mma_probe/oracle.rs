use kernel_core::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};

pub struct Operands {
    pub q: Vec<u16>,
    pub weights: Vec<f32>,
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
}
struct Random(u64);
impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}
impl Operands {
    pub fn new(rows: usize, width: usize, seed: u64) -> Self {
        let mut rng = Random(seed);
        let mut q = vec![0; rows * 4096];
        for group in q.chunks_exact_mut(32) {
            let scale = 2_f32.powi(rng.next() as i32 % 13);
            for value in group.iter_mut() {
                *value = f32_to_bf16(SmallFloat::E2M1.decode_f32((rng.next() & 15) as u8) * scale);
            }
            group[0] = f32_to_bf16(6. * scale);
        }
        let mut weights = Vec::with_capacity(rows * 32);
        for _ in 0..rows * 32 {
            let bits = rng.next();
            weights.push(f32::from_bits(
                (bits & 0x807f_ffff) | ((120 + rng.next() % 15) << 23),
            ));
        }
        let tile = width.min(2048);
        let mut codes: Vec<_> = (0..tile * 64).map(|_| rng.next() as u8).collect();
        let mut scales: Vec<_> = (0..tile * 4)
            .map(|_| (115 + rng.next() % 25) as u8)
            .collect();
        while codes.len() < width * 64 {
            let n = codes.len().min(width * 64 - codes.len());
            codes.extend_from_within(..n);
        }
        while scales.len() < width * 4 {
            let n = scales.len().min(width * 4 - scales.len());
            scales.extend_from_within(..n);
        }
        Self {
            q,
            weights,
            codes,
            scales,
        }
    }
    pub fn witness(&mut self) {
        self.q[..4096].fill(0);
        self.weights[..32].fill(0.);
        self.weights[0] = 1.;
        // One channel's large terms cancel before the lane tree. A separate
        // lane's unit survives there but is lost by block-first accumulation.
        self.q[0] = f32_to_bf16(2_f32.powi(25));
        self.q[33] = f32_to_bf16(1.);
        self.q[64] = f32_to_bf16(-2_f32.powi(25));
        self.codes[..64].fill(0x22); // E2M1 1,1
        self.scales[..4].fill(127);
    }
    pub fn key(&self, index: usize) -> Vec<f32> {
        (0..128)
            .map(|c| {
                let byte = self.codes[index * 64 + c / 2];
                SmallFloat::E2M1.decode_f32((byte >> (4 * (c % 2))) & 15)
                    * 2_f32.powi(i32::from(self.scales[index * 4 + c / 32]) - 127)
            })
            .collect()
    }
    pub fn score(&self, row: usize, index: usize, block_order: bool) -> f32 {
        let k = self.key(index);
        let mut score = 0_f32;
        for h in 0..32 {
            let q = &self.q[(row * 32 + h) * 128..(row * 32 + h + 1) * 128];
            let mut dot = 0_f32;
            if block_order {
                for block in 0..4 {
                    let exact: f64 = (block * 32..block * 32 + 32)
                        .map(|c| f64::from(bf16_to_f32(q[c])) * f64::from(k[c]))
                        .sum();
                    dot += exact as f32;
                }
            } else {
                let mut lanes = [0_f32; 32];
                for (lane, sum) in lanes.iter_mut().enumerate() {
                    for part in 0..4 {
                        let c = part * 32 + lane;
                        *sum = bf16_to_f32(q[c]).mul_add(k[c], *sum);
                    }
                }
                for offset in [16, 8, 4, 2, 1] {
                    for lane in 0..offset {
                        lanes[lane] += lanes[lane + offset];
                    }
                }
                dot = lanes[0];
            }
            score = dot
                .max(0.)
                .mul_add(self.weights[row * 32 + h] * (1. / 64.), score);
        }
        score
    }
    pub fn reference_stats(&self, row: usize, index: usize) -> (f64, f64) {
        let query = kernel_lab_families::position_mma_probe::entry(1)
            .query
            .selection();
        let q = self.q[row * 4096..(row + 1) * 4096]
            .iter()
            .map(|&v| f64::from(bf16_to_f32(v)))
            .collect::<Vec<_>>();
        let weights = self.weights[row * 32..(row + 1) * 32]
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>();
        let key = self
            .key(index)
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        let result =
            kernel_core::position_indexer_oracle::scores(query, &q, &weights, &key).unwrap();
        (result.values[0], result.sum_abs_products[0])
    }
    pub fn reference(&self, row: usize, index: usize) -> f64 {
        self.reference_stats(row, index).0
    }
}

#[test]
fn cpu_mma_block_order_is_not_the_served_order() {
    let mut x = Operands::new(1, 64, 123456789);
    x.witness();
    assert_eq!(x.reference(0, 0), 1. / 64.);
    assert_eq!(x.score(0, 0, false), 1. / 64.);
    assert_eq!(x.score(0, 0, true), 0.);
    let quantized = kernel_core::precision::quantizer::BlockQuantization::E2M1Ceil32
        .fake_quant(
            &x.q.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<_>>(),
            1,
            4096,
        )
        .unwrap();
    assert_eq!(
        quantized, x.q,
        "counterexample must survive the trained fake quantizer"
    );
    println!(
        "MMA_ORDER_CONTROL rejected: old=f64=0.015625 block_mma=0; candidate error exceeds the current build"
    );
}

#[test]
fn cpu_mma_randomized_deviations_are_reported_against_f64() {
    for seed in [0x197acd43, 0x762314b9] {
        let x = Operands::new(2, 128, seed);
        let values = x.q.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<_>>();
        let roundtrip = kernel_core::precision::quantizer::BlockQuantization::E2M1Ceil32
            .fake_quant(&values, 2, 4096)
            .unwrap();
        assert_eq!(
            roundtrip, x.q,
            "random queries must be trained-quantizer outputs"
        );
        let mut squared = [0_f64; 2];
        let mut maximum = [0_f64; 2];
        let mut bits = 0;
        for row in 0..2 {
            for key in 0..128 {
                let old = x.score(row, key, false);
                let new = x.score(row, key, true);
                let reference = x.reference(row, key);
                bits += usize::from(old.to_bits() != new.to_bits());
                for (arm, value) in [old, new].into_iter().enumerate() {
                    let e = (f64::from(value) - reference).abs();
                    squared[arm] += e * e;
                    maximum[arm] = maximum[arm].max(e);
                }
            }
        }
        assert!(bits > 0, "random fixtures did not expose reordered scores");
        println!(
            "MMA_CPU_DEVIATION {}",
            serde_json::json!({"seed":seed,"samples":256,"mismatches":bits,"max_abs":maximum,"rmse":squared.map(|x|(x/256.).sqrt()),"serving":false})
        );
    }
}
