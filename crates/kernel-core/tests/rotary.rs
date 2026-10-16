use kernel_core::contract::{DType, RotaryStyle};
use kernel_core::rotary::{
    AngleRule, Axes, ComputedTable, FreqTable, RotaryArithmetic, RotaryPositions, RotaryRounding,
    RotarySlice, Rotation, Yarn,
};

fn values(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn close(a: f64, b: f64) -> bool {
    close_at(a, b, 0.0)
}

/// Frequencies from different f64 formulas differ by ulps, which a large
/// position multiplies; allow that in the angle.
fn close_at(a: f64, b: f64, angle: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + b.abs()) + 1e-15 * angle.abs()
}

/// DeepSeek V3 `precompute_freqs_cis` YaRN frequencies, in f64.
fn deepseek_yarn_frequencies(dim: usize, base: f64, yarn: Yarn) -> Vec<f64> {
    let correction_dim = |rotations: f64| {
        dim as f64
            * (yarn.original_max_position as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * base.ln())
    };
    let low = correction_dim(yarn.beta_fast).floor().max(0.0);
    let high = correction_dim(yarn.beta_slow).ceil().min(dim as f64 - 1.0);
    let high = if low == high { high + 0.001 } else { high };
    (0..dim / 2)
        .map(|i| {
            let frequency = 1.0 / base.powf((2 * i) as f64 / dim as f64);
            let smooth = 1.0 - ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            frequency / yarn.factor * (1.0 - smooth) + frequency * smooth
        })
        .collect()
}

const DEEPSEEK_YARN: Yarn = Yarn {
    factor: 16.0,
    original_max_position: 65536,
    beta_fast: 32.0,
    beta_slow: 1.0,
};

fn deepseek_tail(inverse: bool) -> Rotation {
    Rotation {
        slice: RotarySlice {
            offset: 448,
            len: 64,
        },
        pairing: RotaryStyle::GptJ,
        inverse,
        rounding: RotaryRounding::F64,
    }
}

const DEEPSEEK_TABLE: FreqTable<'static> = FreqTable::Computed(ComputedTable {
    theta: 160000.0,
    rule: AngleRule::YarnF64(DEEPSEEK_YARN),
    axes: Axes::Linear,
});

#[test]
fn deepseek_gptj_tail_with_yarn_matches_direct_f64_math() {
    // A 512-wide latent whose last 64 values carry interleaved RoPE.
    let positions = [0, 1, 37, 65535, 65536, 1_000_000];
    let input = values(positions.len() * 512, 3);
    let rotated = deepseek_tail(false)
        .apply(
            &input,
            512,
            RotaryPositions::Linear(&positions),
            &DEEPSEEK_TABLE,
        )
        .unwrap();
    let frequencies = deepseek_yarn_frequencies(64, 160000.0, DEEPSEEK_YARN);
    // The ramp blends: the fastest pair keeps its frequency, the slowest is
    // divided by the factor.
    assert_eq!(frequencies[0], 1.0);
    assert!(close(
        frequencies[31],
        1.0 / 160000_f64.powf(62.0 / 64.0) / 16.0
    ));
    for (row, &position) in positions.iter().enumerate() {
        let x = &input[row * 512..(row + 1) * 512];
        let y = &rotated[row * 512..(row + 1) * 512];
        assert_eq!(x[..448], y[..448]);
        for (pair, &frequency) in frequencies.iter().enumerate() {
            let angle = position as f64 * frequency;
            let (a, b) = (x[448 + 2 * pair], x[448 + 2 * pair + 1]);
            assert!(close_at(
                y[448 + 2 * pair],
                a * angle.cos() - b * angle.sin(),
                angle
            ));
            assert!(close_at(
                y[448 + 2 * pair + 1],
                a * angle.sin() + b * angle.cos(),
                angle
            ));
        }
    }
}

#[test]
fn inverse_rotation_undoes_the_forward_rotation() {
    let positions = [5, 70000, 3];
    let input = values(positions.len() * 512, 4);
    let forward = deepseek_tail(false)
        .apply(
            &input,
            512,
            RotaryPositions::Linear(&positions),
            &DEEPSEEK_TABLE,
        )
        .unwrap();
    let back = deepseek_tail(true)
        .apply(
            &forward,
            512,
            RotaryPositions::Linear(&positions),
            &DEEPSEEK_TABLE,
        )
        .unwrap();
    for (a, b) in back.iter().zip(&input) {
        assert!(close(*a, *b), "{a} {b}");
    }
    // The inverse is the rotation by the negative angle, sine negated exactly.
    let frequencies = deepseek_yarn_frequencies(64, 160000.0, DEEPSEEK_YARN);
    let inverse = deepseek_tail(true)
        .apply(
            &input,
            512,
            RotaryPositions::Linear(&positions),
            &DEEPSEEK_TABLE,
        )
        .unwrap();
    for (row, &position) in positions.iter().enumerate() {
        for (pair, &frequency) in frequencies.iter().enumerate() {
            let angle = position as f64 * frequency;
            let i = row * 512 + 448 + 2 * pair;
            let (a, b) = (input[i], input[i + 1]);
            assert!(close_at(
                inverse[i],
                a * angle.cos() + b * angle.sin(),
                angle
            ));
            assert!(close_at(
                inverse[i + 1],
                -a * angle.sin() + b * angle.cos(),
                angle
            ));
        }
    }
}

#[test]
fn computed_rules_and_axes_follow_their_formulas() {
    let pairs = 16;
    let theta = 10000.0;
    let div = ComputedTable {
        theta,
        rule: AngleRule::DivPowF64,
        axes: Axes::Linear,
    };
    let mul = ComputedTable {
        rule: AngleRule::MulNegPowF64,
        ..div
    };
    for pair in 0..pairs {
        let exponent = 2.0 * pair as f64 / 32.0;
        let (sin, cos) = (9.0 / theta.powf(exponent)).sin_cos();
        assert_eq!(div.cos_sin([9, 0, 0], pair, pairs).unwrap(), (cos, sin));
        let (sin, cos) = (9.0 * theta.powf(-exponent)).sin_cos();
        assert_eq!(mul.cos_sin([9, 0, 0], pair, pairs).unwrap(), (cos, sin));
    }
    // Axial 2D: first half of the pairs on axis 0, second on axis 1, each
    // with pairs / 2 frequencies.
    let axial = ComputedTable {
        axes: Axes::Axial2d,
        ..mul
    };
    let row = axial.row([3, 11, 99], pairs).unwrap();
    for (pair, &entry) in row.iter().enumerate() {
        let (axis, index) = (pair / 8, pair % 8);
        let position = [3.0, 11.0][axis];
        let (sin, cos) = (position * theta.powf(-(2.0 * index as f64) / 16.0)).sin_cos();
        assert_eq!(entry, (cos, sin));
    }
    // Interleaved M-RoPE sections pick H and W pairs.
    let mrope = ComputedTable {
        axes: Axes::Mrope([2, 2, 2]),
        ..mul
    };
    let row = mrope.row([1, 2, 3], 6).unwrap();
    for (pair, axis) in [0, 1, 2, 0, 1, 2].into_iter().enumerate() {
        let position = [1.0, 2.0, 3.0][axis];
        let (sin, cos) = (position * theta.powf(-(2.0 * pair as f64) / 12.0)).sin_cos();
        assert_eq!(row[pair], (cos, sin));
    }
}

#[test]
fn rounding_points_and_arithmetic_are_explicit() {
    let rotation = |arithmetic, table, output| Rotation {
        slice: RotarySlice { offset: 1, len: 2 },
        pairing: RotaryStyle::Neox,
        inverse: false,
        rounding: RotaryRounding {
            table,
            arithmetic,
            output,
        },
    };
    let entries = [(0.3, 0.7)];
    let table = FreqTable::Captured(&entries);
    let input = [9.0, 1.001, -0.3333, 5.0];
    let positions = RotaryPositions::Linear(&[0]);
    let f64_out = rotation(RotaryArithmetic::F64, None, None)
        .apply(&input, 4, positions, &table)
        .unwrap();
    assert_eq!(
        f64_out,
        [
            9.0,
            1.001 * 0.3 - -0.3333 * 0.7,
            -0.3333 * 0.3 + 1.001 * 0.7,
            5.0
        ]
    );
    let fma = rotation(RotaryArithmetic::F32Fma, None, None)
        .apply(&input, 4, positions, &table)
        .unwrap();
    let product = -0.3333_f32 * 0.7_f32;
    assert_eq!(fma[1], f64::from(1.001_f32.mul_add(0.3, -product)));
    let bf16 = |v: f64| kernel_core::oracle::bf16_round(v);
    let rounded = rotation(RotaryArithmetic::Bf16Products, Some(DType::Bf16), None)
        .apply(&input, 4, positions, &table)
        .unwrap();
    let (c, s) = (bf16(0.3), bf16(0.7));
    assert_eq!(rounded[1], bf16(bf16(1.001 * c) - bf16(-0.3333 * s)));
    assert_eq!(rounded[2], bf16(bf16(-0.3333 * c) + bf16(1.001 * s)));
    let stored = rotation(RotaryArithmetic::F64, None, Some(DType::Bf16))
        .apply(&input, 4, positions, &table)
        .unwrap();
    assert_eq!(stored, [9.0, bf16(f64_out[1]), bf16(f64_out[2]), 5.0]);
    assert!(
        rotation(RotaryArithmetic::F64, Some(DType::Fp8E4M3), None)
            .apply(&input, 4, positions, &table)
            .is_err()
    );
}

#[test]
fn unsupported_tables_and_shapes_are_refused() {
    let rotation = |offset, len| Rotation {
        slice: RotarySlice { offset, len },
        pairing: RotaryStyle::GptJ,
        inverse: false,
        rounding: RotaryRounding::F64,
    };
    let input = values(128, 5);
    let positions = RotaryPositions::Linear(&[0, 1]);
    let hf = |theta, axes| {
        FreqTable::Computed(ComputedTable {
            theta,
            rule: AngleRule::HfF32AngleBf16CosSin,
            axes,
        })
    };
    // The HF rule accepts table geometries with finite positive F32 theta.
    assert!(
        rotation(0, 64)
            .apply(&input, 64, positions, &hf(1e7, Axes::Linear))
            .is_ok()
    );
    for (len, theta, axes) in [
        (32, f64::MAX, Axes::Linear),
        (64, f64::MIN_POSITIVE, Axes::Linear),
        (64, 1e4, Axes::Axial2d),
    ] {
        let err = rotation(0, len)
            .apply(&input, 64, positions, &hf(theta, axes))
            .unwrap_err();
        assert_eq!(err.0, "HF RoPE geometry not implemented");
    }
    let yarn = |yarn| {
        FreqTable::Computed(ComputedTable {
            theta: 1e4,
            rule: AngleRule::YarnF64(yarn),
            axes: Axes::Linear,
        })
    };
    for bad in [
        Yarn {
            factor: 0.0,
            ..DEEPSEEK_YARN
        },
        Yarn {
            original_max_position: 0,
            ..DEEPSEEK_YARN
        },
        Yarn {
            beta_slow: f64::NAN,
            ..DEEPSEEK_YARN
        },
    ] {
        assert!(
            rotation(0, 64)
                .apply(&input, 64, positions, &yarn(bad))
                .is_err()
        );
    }
    let axial = FreqTable::Computed(ComputedTable {
        theta: 1e4,
        rule: AngleRule::MulNegPowF64,
        axes: Axes::Axial2d,
    });
    assert!(rotation(0, 6).apply(&input, 64, positions, &axial).is_err());
    let linear = FreqTable::Computed(ComputedTable {
        theta: 1e4,
        rule: AngleRule::DivPowF64,
        axes: Axes::Linear,
    });
    for (offset, len) in [(0, 0), (0, 3), (40, 32), (usize::MAX, 2)] {
        assert!(
            rotation(offset, len)
                .apply(&input, 64, positions, &linear)
                .is_err()
        );
    }
    assert!(
        rotation(0, 2)
            .apply(&input[..127], 64, positions, &linear)
            .is_err()
    );
    // Captured tables are indexed by linear position and must cover it.
    let entries = [(1.0, 0.0); 32];
    let captured = FreqTable::Captured(&entries);
    assert!(
        rotation(0, 64)
            .apply(&input, 64, positions, &captured)
            .is_err()
    );
    assert!(
        rotation(0, 64)
            .apply(&input, 64, RotaryPositions::Linear(&[0, 0]), &captured)
            .is_ok()
    );
    assert!(
        rotation(0, 64)
            .apply(&input, 64, RotaryPositions::Axes(&[[0; 3]; 2]), &captured)
            .is_err()
    );
}

#[test]
fn yarn_outside_the_usual_band_is_refused() {
    let table = |theta, yarn| ComputedTable {
        theta,
        rule: AngleRule::YarnF64(yarn),
        axes: Axes::Linear,
    };
    // ln(theta) = 0 would make every correction bound, and so every entry,
    // NaN, even at position zero.
    assert_eq!(
        table(1.0, DEEPSEEK_YARN).row([0; 3], 32).unwrap_err().0,
        "YaRN parameters invalid"
    );
    for (theta, yarn) in [
        (0.5, DEEPSEEK_YARN),
        (
            160000.0,
            Yarn {
                beta_fast: 1.0,
                beta_slow: 32.0,
                ..DEEPSEEK_YARN
            },
        ),
        (
            160000.0,
            Yarn {
                beta_fast: 1.0,
                beta_slow: 1.0,
                ..DEEPSEEK_YARN
            },
        ),
        (
            160000.0,
            Yarn {
                beta_fast: f64::INFINITY,
                ..DEEPSEEK_YARN
            },
        ),
        // A context so long that the band starts above the last frequency.
        (
            160000.0,
            Yarn {
                original_max_position: usize::MAX,
                ..DEEPSEEK_YARN
            },
        ),
    ] {
        assert!(table(theta, yarn).cos_sin([0; 3], 0, 32).is_err());
    }
    let valid = table(160000.0, DEEPSEEK_YARN).row([7; 3], 32).unwrap();
    assert!(valid.iter().all(|&(c, s)| c.is_finite() && s.is_finite()));
}

/// Each legacy F64 order decides which of two NaN terms propagates in an
/// unoptimized x86-64 build (addition returns its first NaN operand); an
/// optimized build may commute the addition.
#[cfg(all(debug_assertions, target_arch = "x86_64"))]
#[test]
fn legacy_f64_orders_keep_their_nan_operand() {
    let (left, right) = (
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::from_bits(0x7ff8_0000_0000_0002),
    );
    let rotated = kernel_core::oracle::rope(
        &[left, right],
        &[0],
        2,
        2,
        1e4,
        RotaryStyle::Neox,
        Some(&[(1.0, 1.0)]),
    )
    .unwrap();
    // `oracle::rope`: second = left * sin + right * cos.
    assert_eq!(rotated[1].to_bits(), left.to_bits());
    let scratch = kernel_core::multimodal_oracle::scratch_rope(
        &[left, right],
        1,
        2,
        2,
        RotaryStyle::Neox,
        &[1.0],
        &[1.0],
    )
    .unwrap();
    // `scratch_rope`: second = right * cos + left * sin.
    assert_eq!(scratch.values[1].to_bits(), right.to_bits());
    for (arithmetic, expected) in [
        (RotaryArithmetic::F64, left),
        (RotaryArithmetic::F64CosFirst, right),
    ] {
        let second = arithmetic.element(right, left, 1.0, 1.0, true);
        assert_eq!(second.to_bits(), expected.to_bits());
        // The first element is `x cos - y sin` in both.
        let first = arithmetic.element(left, right, 1.0, 1.0, false);
        assert_eq!(first.to_bits(), left.to_bits());
    }
}

#[test]
fn f32_angle_rules_keep_their_f32_steps() {
    let table = |rule, axes| ComputedTable {
        theta: 1e4,
        rule,
        axes,
    };
    for position in [0_usize, 1, 7, 127, 4095, 1 << 20] {
        let p = position as f32;
        let div = table(AngleRule::DivPowF32, Axes::Linear)
            .row([position; 3], 32)
            .unwrap();
        let recip = table(AngleRule::MulRecipPowF32, Axes::Linear)
            .row([position; 3], 32)
            .unwrap();
        let f64_power = table(AngleRule::MulRecipF64PowF32, Axes::Linear)
            .row([position; 3], 32)
            .unwrap();
        for i in 0..32 {
            let power = 1e4_f32.powf(2.0 * i as f32 / 64.0);
            let angle = p / power;
            assert_eq!(div[i], (f64::from(angle.cos()), f64::from(angle.sin())));
            let angle = p * (1.0 / power);
            assert_eq!(recip[i], (f64::from(angle.cos()), f64::from(angle.sin())));
            let frequency = 1.0 / 1e4_f64.powf(f64::from((2 * i) as f32 / 64.0)) as f32;
            let angle = f64::from(p * frequency);
            assert_eq!(f64_power[i], (angle.cos(), angle.sin()));
        }
    }
    // Axial 2D: the first half of the pairs reads axis 0, the second axis 1,
    // each over d = pairs.
    let axial = table(AngleRule::MulRecipPowF32, Axes::Axial2d)
        .row([3, 5, 0], 36)
        .unwrap();
    for (pair, &(cos, _)) in axial.iter().enumerate() {
        let (axis, i) = (pair / 18, pair % 18);
        let angle = [3.0_f32, 5.0][axis] * (1.0 / 1e4_f32.powf(2.0 * i as f32 / 36.0));
        assert_eq!(cos, f64::from(angle.cos()));
    }
    let overflow = ComputedTable {
        theta: 1e39,
        rule: AngleRule::DivPowF32,
        axes: Axes::Linear,
    };
    assert!(overflow.row([1; 3], 4).is_err());
}

#[test]
fn hf_fallback_at_64_pairs_matches_pinned_frequencies_on_even_pairs() {
    for theta in [1e4, 1e7] {
        let table = ComputedTable {
            theta,
            rule: AngleRule::HfF32AngleBf16CosSin,
            axes: Axes::Linear,
        };
        for position in [1, 513, 262143] {
            let row = table.row([position; 3], 64).unwrap();
            for pair in 0..32 {
                assert_eq!(
                    row[2 * pair],
                    kernel_core::hf_rope::cos_sin(position, pair, theta)
                );
            }
        }
    }
}
