use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{ArgmaxLayout, DType, RmsNormKind, RotaryStyle, RowQuery, Validate};
use kernel_core::oracle::{
    NormArithmetic, NormOffset, NormScale, RmsNorm, rmsnorm, rope, sharded_argmax,
    sharded_argmax_final, sharded_argmax_partials,
};

/// DFlash's Q/K RMSNorm: one BF16 weight vector for every head, F32
/// arithmetic and one BF16 rounding of the output.
const QK_NORM: RmsNorm = RmsNorm {
    eps: 1e-6,
    offset: NormOffset::Plain,
    arithmetic: NormArithmetic::F32,
    scale: NormScale::MultiplyReciprocal,
    before_weight: None,
    output: Some(DType::Bf16),
};

#[test]
fn qk_norm_reuses_one_weight_vector_for_each_head() {
    let one = f32_to_bf16(1.0);
    let two = f32_to_bf16(2.0);
    let input = [one, two, one, two, two, one, two, one];
    let weights = [one, two, one, two];
    let output = QK_NORM.apply_bf16(&input, &weights, 4).unwrap();
    let input_f64 = input
        .iter()
        .map(|&x| f64::from(bf16_to_f32(x)))
        .collect::<Vec<_>>();
    let repeated = weights
        .repeat(2)
        .iter()
        .map(|&x| f64::from(bf16_to_f32(x)))
        .collect::<Vec<_>>();
    let reference = rmsnorm(
        &input_f64,
        8,
        &repeated,
        1e-6,
        RmsNormKind::QkPerHead { head_dim: 4 },
        None,
    )
    .unwrap();
    for (actual, expected) in output.iter().zip(reference.values) {
        assert!((f64::from(bf16_to_f32(*actual)) - expected).abs() < 0.02);
    }
    assert!(QK_NORM.apply_bf16(&input, &weights[..3], 4).is_err());
}

#[test]
fn partial_neox_rope_keeps_unrotated_half() {
    let input = (0..128).map(f64::from).collect::<Vec<_>>();
    let output = rope(&input, &[3], 128, 64, 1e4, RotaryStyle::Neox, None).unwrap();
    assert_eq!(&output[64..], &input[64..]);
    assert_ne!(&output[..64], &input[..64]);
    RowQuery::Rope {
        head_dim: 128,
        rotary_dim: 64,
        theta: 1e4,
        style: RotaryStyle::Neox,
        source: kernel_core::contract::RotarySource::Computed,
        fused_paged_kv_write: false,
    }
    .validate()
    .unwrap();
}

#[test]
fn tp4_argmax_uses_global_ids_and_lowest_id_for_ties() {
    let (one, high) = (1.0, 9.0);
    let shards = [
        &[one, high][..],
        &[high, one][..],
        &[one, one][..],
        &[high, one][..],
    ];
    let partials = sharded_argmax_partials(&shards).unwrap();
    assert_eq!(
        partials.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
        [1, 2, 4, 6]
    );
    assert_eq!(sharded_argmax_final(&partials).unwrap(), 1);
    assert_eq!(sharded_argmax(&shards).unwrap(), 1);
    assert!(sharded_argmax(&[&[]]).is_err());
    assert_eq!(sharded_argmax(&[&[f32::NAN, f32::NAN]]), Ok(0));
    RowQuery::ShardedArgmaxPartial {
        vocab_per_rank: 38144,
        ranks: 4,
        logits: DType::Bf16,
    }
    .validate()
    .unwrap();
    RowQuery::ShardedArgmaxFinal {
        ranks: 4,
        pairs: ArgmaxLayout::RowMajor,
    }
    .validate()
    .unwrap();
    assert!(
        RowQuery::ShardedArgmaxFinal {
            ranks: 0,
            pairs: ArgmaxLayout::RowMajor,
        }
        .validate()
        .is_err()
    );
}
