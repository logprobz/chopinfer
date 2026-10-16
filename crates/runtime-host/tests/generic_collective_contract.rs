//! The generic collective contracts this host answers to, and their numerics.

use std::rc::Rc;

use runtime_bundle_abi::{
    BundleAbiError, BundleDeviceBufferHandleV1, BundleStatusV1, CollectiveRequestV1,
    DeviceBufferSpanV1, HostServicesTableV1, KernelArgumentV1,
};
use runtime_host::fake::{FAKE_COLLECTIVE_SUM_BF16_V1, FakeHostDeviceV1};
use runtime_host::{
    COLLECTIVE_SUM_BF16_V1, COLLECTIVE_SUM_F32_V1, GENERIC_COLLECTIVE_CONTRACTS_V1,
    GenericCollectiveV1, HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1,
    assert_generic_collective_ordering_contract_v1, bf16_bits_from_f32_v1, f32_from_bf16_bits_v1,
};

/// The ordinary tensor-parallel width this contract was added for.
const RANKS: u32 = 4;

/// The lanes each rank's span carries, chosen to need more than one byte pair.
const LANES: u64 = 6;

const LANE_BYTES: u64 = 2;

fn table() -> HostServicesTableV1 {
    let config = HostInstanceConfigV1::new(
        HostTopologyV1::new(vec![0, 1, 2, 3]).expect("the test topology is valid"),
    );
    HostServicesTableV1::new(Rc::new(HostServicesV1Backend::new(
        FakeHostDeviceV1::new(),
        config,
    )))
}

fn status(error: BundleAbiError) -> BundleStatusV1 {
    match error {
        BundleAbiError::Status(status) => status,
        other => panic!("expected a host status, got {other}"),
    }
}

fn span(buffer: BundleDeviceBufferHandleV1) -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(buffer, 0, LANES * LANE_BYTES)
}

fn argument(buffer: BundleDeviceBufferHandleV1) -> KernelArgumentV1 {
    KernelArgumentV1::DeviceBuffer(span(buffer))
}

/// The whole set a bundle may name, written down once.
///
/// A contract that is not in this list is unsupported everywhere, so what the
/// list holds is the host's answer to what it can be asked for. Naming the set
/// here is what keeps a later entry from arriving without a decision.
#[test]
fn the_host_answers_to_exactly_the_two_generic_collective_contracts() {
    assert_eq!(
        GENERIC_COLLECTIVE_CONTRACTS_V1,
        [COLLECTIVE_SUM_F32_V1, COLLECTIVE_SUM_BF16_V1]
    );
    assert_eq!(COLLECTIVE_SUM_F32_V1, b"sum-f32-v1");
    assert_eq!(COLLECTIVE_SUM_BF16_V1, b"sum-bf16-v1");

    for contract in GENERIC_COLLECTIVE_CONTRACTS_V1 {
        let collective = GenericCollectiveV1::from_contract_v1(contract)
            .expect("every listed contract names a collective this host performs");
        assert_eq!(collective.contract(), *contract);
    }
    assert_eq!(
        GenericCollectiveV1::from_contract_v1(COLLECTIVE_SUM_F32_V1),
        Some(GenericCollectiveV1::SumF32)
    );
    assert_eq!(
        GenericCollectiveV1::from_contract_v1(COLLECTIVE_SUM_BF16_V1),
        Some(GenericCollectiveV1::SumBf16)
    );
    assert_eq!(GenericCollectiveV1::SumF32.lane_bytes(), 4);
    assert_eq!(GenericCollectiveV1::SumBf16.lane_bytes(), 2);
}

/// A name this host does not serve names nothing, however close it looks.
#[test]
fn a_contract_this_host_does_not_serve_names_no_collective() {
    for unserved in [
        &b""[..],
        b"sum-f16-v1",
        b"sum-bf16",
        b"sum-bf16-v2",
        b"sum-bf16-v1 ",
        b"product-bf16-v1",
        b"SUM-BF16-V1",
    ] {
        assert_eq!(
            GenericCollectiveV1::from_contract_v1(unserved),
            None,
            "{} must name no generic collective",
            String::from_utf8_lossy(unserved)
        );
    }
}

/// The exact `bf16` every accumulated `f32` sum rounds to.
///
/// These are fixtures, not a restatement of the implementation: each row is the
/// value a round-to-nearest-ties-to-even conversion produces, including the
/// three places a truncating conversion would differ, which are the ties, the
/// overflow to infinity, and the signalling NaN.
#[test]
fn every_accumulated_sum_rounds_to_the_bf16_the_contract_declares() {
    for (sum, expected, why) in [
        (0.0_f32, 0x0000_u16, "zero"),
        (-0.0_f32, 0x8000, "the sign of a zero survives"),
        (1.0, 0x3f80, "an exactly representable value is unchanged"),
        (-1.0, 0xbf80, "and so is its negation"),
        (1.5, 0x3fc0, "and a value inside the 8-bit significand"),
        (256.0, 0x4380, "a power of two is exact"),
        (257.0, 0x4380, "a tie rounds down to the even significand"),
        (259.0, 0x4382, "and this tie rounds up to the even one"),
        (258.0, 0x4381, "the odd value between them is exact"),
        (
            f32::from_bits(0x3f80_8000),
            0x3f80,
            "the tie above one rounds to the even significand",
        ),
        (
            f32::from_bits(0x3f81_8000),
            0x3f82,
            "and the next tie rounds up to the even one",
        ),
        (
            f32::from_bits(0x3f80_9000),
            0x3f81,
            "more than half an ulp rounds up",
        ),
        (
            f32::from_bits(0x3f80_7fff),
            0x3f80,
            "and less than half an ulp rounds down",
        ),
        (
            f32::MAX,
            0x7f80,
            "a finite sum too large for bf16 becomes an infinity",
        ),
        (f32::INFINITY, 0x7f80, "an infinity stays one"),
        (f32::NEG_INFINITY, 0xff80, "with its sign"),
        (
            f32::from_bits(0x0080_0000),
            0x0080,
            "the smallest normal is shared by both formats",
        ),
        (
            f32::from_bits(0x0000_8000),
            0x0000,
            "half the smallest bf16 subnormal is a tie down to zero",
        ),
        (
            f32::from_bits(0x0000_c000),
            0x0001,
            "and three quarters of it rounds up to that subnormal",
        ),
        (
            f32::from_bits(0x8000_0100),
            0x8000,
            "a sum too small to represent keeps its sign",
        ),
    ] {
        assert_eq!(
            bf16_bits_from_f32_v1(sum),
            expected,
            "{sum:e} must round to {expected:#06x}: {why}"
        );
    }

    // A NaN sum stays a NaN rather than becoming an infinity, and a signalling
    // one arrives quiet, which is the case truncation alone gets wrong.
    for nan in [f32::NAN, f32::from_bits(0x7f80_0001)] {
        let rounded = bf16_bits_from_f32_v1(nan);
        assert!(
            f32_from_bf16_bits_v1(rounded).is_nan(),
            "{rounded:#06x} must still be a NaN"
        );
        assert_eq!(rounded & 0x0040, 0x0040, "and must be quiet");
    }
}

/// Widening a `bf16` is exact, so a lane that never changes never drifts.
#[test]
fn widening_a_bf16_lane_and_rounding_it_back_returns_the_same_bits() {
    for bits in 0..=u16::MAX {
        let widened = f32_from_bf16_bits_v1(bits);
        if widened.is_nan() {
            // Only the quiet bit is asserted for a NaN; the payload is not.
            assert_eq!(bf16_bits_from_f32_v1(widened) & 0x0040, 0x0040);
            continue;
        }
        assert_eq!(
            bf16_bits_from_f32_v1(widened),
            bits,
            "{bits:#06x} must survive the round trip unchanged"
        );
    }
}

/// Folds one lane over the ranks the way the contract declares.
///
/// Rank 0 seeds the fold and ranks 1 through N-1 are added to it, which is the
/// same split the plan performs; only the reading of the bytes differs.
fn fold_lane(contributions: &[f32]) -> Vec<u8> {
    let collective = GenericCollectiveV1::SumBf16;
    let lanes = |value: f32| {
        let lane = collective.encode_lanes_v1(&[value]);
        assert_eq!(lane.len(), 2, "every contribution is one bf16 lane");
        lane
    };
    let (first, rest) = contributions
        .split_first()
        .expect("a fold has at least rank 0");
    let mut totals = collective
        .seed_lanes_v1(1, &lanes(*first))
        .expect("rank 0 seeds one lane");
    for contribution in rest {
        collective
            .accumulate_lanes_v1(&mut totals, &lanes(*contribution))
            .expect("one lane of bf16 accumulates into one total");
    }
    collective.encode_lanes_v1(&totals)
}

/// The `f32` fold is never narrowed back to `bf16` between ranks.
///
/// The discriminator is a lane whose ranks each round away individually: three
/// contributions of one against a fourth of 256 stay at 256 if the running total
/// is narrowed to `bf16` after every rank, and reach 260 when the one narrowing
/// happens after the last rank. This is what the wider accumulator buys, and it
/// is all it buys: the additions inside the fold still round to `f32`.
#[test]
fn the_bf16_fold_narrows_once_rather_than_once_per_rank() {
    assert_eq!(
        fold_lane(&[256.0, 1.0, 1.0, 1.0]),
        0x4382_u16.to_le_bytes(),
        "259 narrows up to 260, the even significand"
    );

    // Narrowing each rank in turn would have stayed at 256 the whole way.
    let mut narrowed_each_time = 0.0_f32;
    for contribution in [256.0_f32, 1.0, 1.0, 1.0] {
        narrowed_each_time =
            f32_from_bf16_bits_v1(bf16_bits_from_f32_v1(narrowed_each_time + contribution));
    }
    assert_eq!(narrowed_each_time, 256.0);
}

/// The fold is in rank order, and the answer depends on that order.
///
/// Each addition rounds to `f32`, so the contract is a fixed left fold over the
/// ranks and not an exactly rounded sum. Over these four lanes the rank-order
/// fold answers 1, because 2^30 + 1 rounds back to 2^30 and the total returns to
/// zero before the last rank is added. Two other groupings of the same four
/// contributions answer 2 and 0, which is why the order is part of the contract
/// rather than an implementation detail a later backend could change.
#[test]
fn the_bf16_fold_is_in_rank_order_and_a_regrouping_answers_differently() {
    const HUGE: f32 = 1_073_741_824.0; // 2^30, and exactly a bf16
    let contributions = [HUGE, 1.0, -HUGE, 1.0];

    assert_eq!(
        fold_lane(&contributions),
        bf16_bits_from_f32_v1(1.0).to_le_bytes(),
        "the rank-order fold answers 1"
    );

    // Pairing rank 0 with rank 2 and rank 1 with rank 3 answers 2, because
    // neither pair loses its small term before the pairs are added.
    let crosswise = (contributions[0] + contributions[2]) + (contributions[1] + contributions[3]);
    assert_eq!(crosswise, 2.0);
    // Pairing the adjacent ranks, which is what a tree reduction would do,
    // answers 0: each pair loses its small term to the large one.
    let adjacent = (contributions[0] + contributions[1]) + (contributions[2] + contributions[3]);
    assert_eq!(adjacent, 0.0);
}

/// Rank 0 seeds the `bf16` fold, and no zero is an identity that hides it.
///
/// `+0.0 + -0.0` is `+0.0`, so a lane whose every rank contributes `-0.0` is the
/// one non-NaN input that tells the two seeds apart. This contract widens rank 0
/// and answers `-0.0`, which is what the pinned upstream `packed_reduce` answers
/// with `tmp = upcast(ptrs[0][idx])`. Seeding with `+0.0` would answer `+0.0`,
/// so the seed cannot drift without this failing.
///
/// A NaN input is not a second discriminator, because it is not a promise: what
/// an addition does to a NaN's sign and payload is unspecified, so the seeds are
/// not compared on one. See
/// `a_nan_keeps_its_sign_and_payload_through_the_narrowing_but_not_the_fold`.
#[test]
fn the_bf16_fold_is_seeded_by_rank_zero_and_keeps_a_negative_zero() {
    assert_eq!(
        fold_lane(&[-0.0, -0.0, -0.0, -0.0]),
        0x8000_u16.to_le_bytes(),
        "widening rank 0 keeps the negative zero every rank contributed"
    );
    // The seed a `+0.0` accumulator would have produced, for contrast.
    let plus_zero_seeded = [-0.0_f32, -0.0, -0.0, -0.0]
        .into_iter()
        .fold(0.0_f32, |total, contribution| total + contribution);
    assert_eq!(bf16_bits_from_f32_v1(plus_zero_seeded), 0x0000);

    // A single negative zero among positive zeroes still sums to a positive
    // zero, so the fixture above is about the seed and not about sign handling.
    assert_eq!(fold_lane(&[-0.0, 0.0, 0.0, 0.0]), 0x0000_u16.to_le_bytes());
    assert_eq!(
        fold_lane(&[0.0, -0.0, -0.0, -0.0]),
        0x0000_u16.to_le_bytes()
    );

    // `sum-f32-v1` keeps the `+0.0` seed it shipped with. Among non-NaN inputs
    // this is the only place the two contracts answer differently for the same
    // contributions; a NaN is outside that comparison, because what an addition
    // does to a NaN's sign and payload is unspecified.
    let negative_zero_f32 = GenericCollectiveV1::SumF32
        .seed_lanes_v1(1, &(-0.0_f32).to_le_bytes())
        .expect("rank 0 seeds one f32 lane");
    assert_eq!(negative_zero_f32[0].to_bits(), 0.0_f32.to_bits());
    let negative_zero_bf16 = GenericCollectiveV1::SumBf16
        .seed_lanes_v1(1, &0x8000_u16.to_le_bytes())
        .expect("rank 0 seeds one bf16 lane");
    assert_eq!(negative_zero_bf16[0].to_bits(), (-0.0_f32).to_bits());
}

/// What the narrowing promises about a NaN, and what nothing promises.
///
/// [`bf16_bits_from_f32_v1`] is a pure function of the bits it is given, so its
/// sign and payload handling is exact and pinned here: the sign survives, the
/// top of the payload survives, and the quiet bit is set so a signalling NaN
/// cannot arrive as an infinity.
///
/// The fold is a different matter. Rust does not specify the sign or the payload
/// of a NaN that came out of an addition, so this host promises neither and
/// nothing here asserts them. What a NaN contribution does promise is that the
/// lane stays a NaN, and that it arrives quiet.
#[test]
fn a_nan_keeps_its_sign_and_payload_through_the_narrowing_but_not_the_fold() {
    for (bits, expected, why) in [
        (0x7fc0_0000_u32, 0x7fc0_u16, "a positive quiet NaN"),
        (0xffc0_0000, 0xffc0, "a negative quiet NaN keeps its sign"),
        (0x7fc1_2300, 0x7fc1, "the top of a payload survives"),
        (0xffc1_2300, 0xffc1, "with its sign"),
        (0x7f80_0001, 0x7fc0, "a signalling NaN arrives quiet"),
        (0xff80_0001, 0xffc0, "and keeps its sign doing so"),
    ] {
        assert_eq!(
            bf16_bits_from_f32_v1(f32::from_bits(bits)),
            expected,
            "{bits:#010x} must narrow to {expected:#06x}: {why}"
        );
    }

    // Through the fold, only NaN-ness and quietness are promised.
    let collective = GenericCollectiveV1::SumBf16;
    let mut totals = collective
        .seed_lanes_v1(1, &0x7fc1_u16.to_le_bytes())
        .expect("rank 0 seeds one bf16 lane");
    assert!(totals[0].is_nan(), "widening a NaN lane is exact");
    for contribution in [1.0_f32, -1.0, 0.0] {
        collective
            .accumulate_lanes_v1(&mut totals, &collective.encode_lanes_v1(&[contribution]))
            .expect("one lane accumulates");
    }
    let narrowed = u16::from_le_bytes([
        collective.encode_lanes_v1(&totals)[0],
        collective.encode_lanes_v1(&totals)[1],
    ]);
    assert!(
        f32_from_bf16_bits_v1(narrowed).is_nan(),
        "the lane is a NaN"
    );
    assert_eq!(narrowed & 0x0040, 0x0040, "and it is quiet");
}

/// A contribution that is not whole lanes of the declared width is refused.
#[test]
fn accumulating_a_contribution_of_the_wrong_length_is_refused() {
    let mut totals = [0.0_f32; 2];

    for (collective, contribution) in [
        (GenericCollectiveV1::SumBf16, vec![0_u8; 3]),
        (GenericCollectiveV1::SumBf16, vec![0_u8; 8]),
        (GenericCollectiveV1::SumF32, vec![0_u8; 6]),
        (GenericCollectiveV1::SumF32, vec![0_u8; 4]),
    ] {
        assert!(
            collective
                .accumulate_lanes_v1(&mut totals, &contribution)
                .is_err(),
            "{} bytes are not two lanes of {collective:?}",
            contribution.len()
        );
    }
    assert_eq!(totals, [0.0; 2], "a refused contribution adds nothing");

    assert_eq!(
        GenericCollectiveV1::SumBf16.accumulate_lanes_v1(&mut totals, &[0_u8; 4]),
        Ok(())
    );
    assert_eq!(
        GenericCollectiveV1::SumF32.accumulate_lanes_v1(&mut totals, &[0_u8; 8]),
        Ok(())
    );
}

/// Encodes one rank's contribution, refusing a value the width cannot hold.
///
/// Every contribution these tests use is exactly representable, so the encoding
/// rounds nothing and the reduction is the only place a rounding happens.
fn lanes_of(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            let bits = bf16_bits_from_f32_v1(*value);
            // Compared as bits, so the sign of a zero is checked rather than
            // compared away by `-0.0 == 0.0`.
            assert_eq!(
                f32_from_bf16_bits_v1(bits).to_bits(),
                value.to_bits(),
                "{value} must be an exact bf16 contribution"
            );
            bits.to_le_bytes()
        })
        .collect()
}

fn lane_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|lane| u16::from_le_bytes([lane[0], lane[1]]))
        .collect()
}

/// Four ranks, folded lane-wise in rank order, written back to every one.
///
/// The first lane is what the `f32` accumulator buys: narrowed after every rank
/// it stays at 256, and with the one narrowing after the last rank it is 260.
/// The fourth lane is the other direction, a total that narrows back down onto
/// its largest contribution. The fifth lane is the fold order itself: in rank
/// order it answers 1, and it would answer 2 or 0 under either pairing of the
/// same four contributions. The last lane is the seed: every rank contributes
/// `-0.0`, so it answers `-0.0` because rank 0 is widened rather than added to a
/// `+0.0` accumulator, which would have answered `+0.0`.
#[test]
fn a_bf16_collective_folds_four_ranks_and_writes_the_result_to_every_one() {
    const HUGE: f32 = 1_073_741_824.0; // 2^30, and exactly a bf16
    let contributions = [
        [256.0_f32, 1.0, -0.5, 1024.0, HUGE, -0.0],
        [1.0, 1.0, 0.25, 1.0, 1.0, -0.0],
        [1.0, 1.0, 0.25, 1.0, -HUGE, -0.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, -0.0],
    ];
    let expected = [
        0x4382_u16, // 259 narrows up to 260, the even significand
        0x4080,     // 4, exactly
        0x0000,     // the negative and the positive contributions cancel
        0x4480,     // 1027 narrows back down to 1024
        0x3f80,     // 1: 2^30 + 1 rounds back to 2^30, so only the last 1 stands
        0x8000,     // -0.0: rank 0 is widened, not added to a +0.0 accumulator
    ];

    let table = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let mut buffers = Vec::new();
    for contribution in contributions {
        let buffer = host
            .allocate_device_buffer(pool, stream, LANES * LANE_BYTES, 64)
            .unwrap();
        host.copy_host_to_device(stream, span(buffer), &lanes_of(&contribution))
            .unwrap();
        buffers.push(buffer);
    }
    let collective = host
        .create_collective(&CollectiveRequestV1 {
            flags: 0,
            rank_count: u64::from(RANKS),
            semantic_contract: FAKE_COLLECTIVE_SUM_BF16_V1.to_vec(),
        })
        .unwrap();
    let arguments = buffers.iter().copied().map(argument).collect::<Vec<_>>();

    host.execute_collective(collective, stream, COLLECTIVE_SUM_BF16_V1, &arguments)
        .unwrap();

    // Every rank receives the same bytes, because the fold ran once and its
    // result was written back to all of them.
    for buffer in &buffers {
        let mut read = vec![0_u8; (LANES * LANE_BYTES) as usize];
        host.copy_device_to_host(stream, &mut read, span(*buffer))
            .unwrap();
        assert_eq!(lane_bits(&read), expected);
    }

    host.release_collective(collective).unwrap();
    for buffer in buffers.into_iter().rev() {
        host.release_device_buffer(buffer).unwrap();
    }
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

/// Every way of asking wrongly, and the contribution each one leaves alone.
///
/// The refusal is what matters, and so is what did not happen: a request the
/// host will not perform reads no rank buffer and writes none, so the bundle's
/// own contributions are still there to retry with. The last case proves that,
/// by reducing correctly through the same handle every refusal was made on.
#[test]
fn a_refused_bf16_collective_leaves_every_contribution_where_it_was() {
    const BUFFER_BYTES: u64 = 16;
    const SPAN_BYTES: u64 = LANES * LANE_BYTES;

    let table = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let mut buffers = Vec::new();
    for rank in 0..RANKS {
        let buffer = host
            .allocate_device_buffer(pool, stream, BUFFER_BYTES, 64)
            .unwrap();
        // Distinct bytes per rank, so any write at all is visible.
        let filled = (0..BUFFER_BYTES as u8)
            .map(|byte| byte.wrapping_add(rank as u8 * 32))
            .collect::<Vec<_>>();
        host.copy_host_to_device(
            stream,
            DeviceBufferSpanV1::new(buffer, 0, BUFFER_BYTES),
            &filled,
        )
        .unwrap();
        buffers.push((buffer, filled));
    }
    let collective = host
        .create_collective(&CollectiveRequestV1 {
            flags: 0,
            rank_count: u64::from(RANKS),
            semantic_contract: FAKE_COLLECTIVE_SUM_BF16_V1.to_vec(),
        })
        .unwrap();
    let whole = |rank: usize| DeviceBufferSpanV1::new(buffers[rank].0, 0, SPAN_BYTES);
    let sound = (0..RANKS as usize)
        .map(|rank| KernelArgumentV1::DeviceBuffer(whole(rank)))
        .collect::<Vec<_>>();
    let replacing = |rank: usize, span: DeviceBufferSpanV1| {
        let mut arguments = sound.clone();
        arguments[rank] = KernelArgumentV1::DeviceBuffer(span);
        arguments
    };

    for (operation, arguments, expected, why) in [
        (
            &b"sum-f16-v1"[..],
            sound.clone(),
            BundleStatusV1::Unsupported,
            "a contract this host does not serve",
        ),
        (
            COLLECTIVE_SUM_F32_V1,
            sound.clone(),
            BundleStatusV1::Unsupported,
            "an operation that is not the one this collective was created for",
        ),
        (
            COLLECTIVE_SUM_BF16_V1,
            sound[..RANKS as usize - 1].to_vec(),
            BundleStatusV1::InvalidArgument,
            "one span short of the rank count",
        ),
        (
            COLLECTIVE_SUM_BF16_V1,
            {
                let mut arguments = sound.clone();
                arguments.push(KernelArgumentV1::DeviceBuffer(whole(0)));
                arguments
            },
            BundleStatusV1::InvalidArgument,
            "one span more than the rank count",
        ),
        (
            COLLECTIVE_SUM_BF16_V1,
            {
                let mut arguments = sound.clone();
                arguments[2] = KernelArgumentV1::ScalarU32(0);
                arguments
            },
            BundleStatusV1::InvalidArgument,
            "a scalar where a rank's span belongs",
        ),
        (
            COLLECTIVE_SUM_BF16_V1,
            replacing(3, DeviceBufferSpanV1::new(buffers[3].0, 0, SPAN_BYTES - 2)),
            BundleStatusV1::InvalidArgument,
            "a span shorter than the others",
        ),
        (
            COLLECTIVE_SUM_BF16_V1,
            replacing(3, DeviceBufferSpanV1::new(buffers[3].0, 0, SPAN_BYTES + 2)),
            BundleStatusV1::InvalidArgument,
            "a span longer than the others",
        ),
        (
            COLLECTIVE_SUM_BF16_V1,
            (0..RANKS as usize)
                .map(|rank| {
                    KernelArgumentV1::DeviceBuffer(DeviceBufferSpanV1::new(buffers[rank].0, 0, 3))
                })
                .collect(),
            BundleStatusV1::InvalidArgument,
            "a length that is not whole bf16 lanes",
        ),
        (
            COLLECTIVE_SUM_BF16_V1,
            (0..RANKS as usize)
                .map(|rank| {
                    KernelArgumentV1::DeviceBuffer(DeviceBufferSpanV1::new(buffers[rank].0, 0, 0))
                })
                .collect(),
            BundleStatusV1::InvalidArgument,
            "a span carrying no lane at all",
        ),
        (
            COLLECTIVE_SUM_BF16_V1,
            replacing(1, DeviceBufferSpanV1::new(buffers[1].0, 1, SPAN_BYTES)),
            BundleStatusV1::InvalidArgument,
            "a span that starts inside a lane",
        ),
        (
            COLLECTIVE_SUM_BF16_V1,
            replacing(1, whole(0)),
            BundleStatusV1::InvalidArgument,
            "two ranks contributing the same bytes",
        ),
        (
            COLLECTIVE_SUM_BF16_V1,
            replacing(
                1,
                DeviceBufferSpanV1::new(buffers[0].0, LANE_BYTES, SPAN_BYTES),
            ),
            BundleStatusV1::InvalidArgument,
            "two ranks whose spans share a lane",
        ),
        (
            COLLECTIVE_SUM_BF16_V1,
            replacing(
                1,
                DeviceBufferSpanV1::new(buffers[1].0, BUFFER_BYTES, SPAN_BYTES),
            ),
            BundleStatusV1::InvalidArgument,
            "a span that runs past its allocation",
        ),
    ] {
        assert_eq!(
            host.execute_collective(collective, stream, operation, &arguments)
                .map_err(status),
            Err(expected),
            "{why} must be refused"
        );
        for (buffer, filled) in &buffers {
            let mut read = vec![0_u8; BUFFER_BYTES as usize];
            host.copy_device_to_host(
                stream,
                &mut read,
                DeviceBufferSpanV1::new(*buffer, 0, BUFFER_BYTES),
            )
            .unwrap();
            assert_eq!(&read, filled, "{why} must leave every contribution alone");
        }
    }

    // The handle every one of those was refused on still reduces.
    host.execute_collective(collective, stream, COLLECTIVE_SUM_BF16_V1, &sound)
        .unwrap();
    let mut read = vec![0_u8; SPAN_BYTES as usize];
    host.copy_device_to_host(stream, &mut read, whole(0))
        .unwrap();
    assert_ne!(read, buffers[0].1[..SPAN_BYTES as usize]);

    host.release_collective(collective).unwrap();
    for (buffer, _) in buffers.into_iter().rev() {
        host.release_device_buffer(buffer).unwrap();
    }
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

/// Two ranks may share an allocation as long as they share no byte.
///
/// The rule the refusals above enforce is about the bytes an in-place sum reads
/// and overwrites, not about which allocation they came from, so four spans laid
/// out end to end in one buffer are a sound request.
#[test]
fn four_disjoint_spans_of_one_allocation_are_a_sound_bf16_collective() {
    const SPAN_BYTES: u64 = LANES * LANE_BYTES;

    let table = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let buffer = host
        .allocate_device_buffer(pool, stream, SPAN_BYTES * u64::from(RANKS), 64)
        .unwrap();
    for rank in 0..u64::from(RANKS) {
        host.copy_host_to_device(
            stream,
            DeviceBufferSpanV1::new(buffer, rank * SPAN_BYTES, SPAN_BYTES),
            &lanes_of(&[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
        )
        .unwrap();
    }
    let collective = host
        .create_collective(&CollectiveRequestV1 {
            flags: 0,
            rank_count: u64::from(RANKS),
            semantic_contract: FAKE_COLLECTIVE_SUM_BF16_V1.to_vec(),
        })
        .unwrap();
    let arguments = (0..u64::from(RANKS))
        .map(|rank| {
            KernelArgumentV1::DeviceBuffer(DeviceBufferSpanV1::new(
                buffer,
                rank * SPAN_BYTES,
                SPAN_BYTES,
            ))
        })
        .collect::<Vec<_>>();

    host.execute_collective(collective, stream, COLLECTIVE_SUM_BF16_V1, &arguments)
        .unwrap();

    for rank in 0..u64::from(RANKS) {
        let mut read = vec![0_u8; SPAN_BYTES as usize];
        host.copy_device_to_host(
            stream,
            &mut read,
            DeviceBufferSpanV1::new(buffer, rank * SPAN_BYTES, SPAN_BYTES),
        )
        .unwrap();
        assert_eq!(read, lanes_of(&[4.0, 8.0, 16.0, 32.0, 64.0, 128.0]));
    }

    host.release_collective(collective).unwrap();
    host.release_device_buffer(buffer).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

/// A released collective reduces nothing, and it is released in order.
#[test]
fn a_bf16_collective_handle_is_released_in_order_and_then_reduces_nothing() {
    let table = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let mut buffers = Vec::new();
    for _ in 0..RANKS {
        buffers.push(
            host.allocate_device_buffer(pool, stream, LANES * LANE_BYTES, 64)
                .unwrap(),
        );
    }
    let collective = host
        .create_collective(&CollectiveRequestV1 {
            flags: 0,
            rank_count: u64::from(RANKS),
            semantic_contract: FAKE_COLLECTIVE_SUM_BF16_V1.to_vec(),
        })
        .unwrap();
    let arguments = buffers.iter().copied().map(argument).collect::<Vec<_>>();
    host.execute_collective(collective, stream, COLLECTIVE_SUM_BF16_V1, &arguments)
        .unwrap();

    // The collective is the newest resource, so nothing older goes first.
    assert_eq!(
        host.release_stream(stream).map_err(status),
        Err(BundleStatusV1::InvalidState)
    );
    host.release_collective(collective).unwrap();
    assert_eq!(
        host.execute_collective(collective, stream, COLLECTIVE_SUM_BF16_V1, &arguments)
            .map_err(status),
        Err(BundleStatusV1::StaleHandle)
    );
    assert_eq!(
        host.release_collective(collective).map_err(status),
        Err(BundleStatusV1::StaleHandle)
    );

    for buffer in buffers.into_iter().rev() {
        host.release_device_buffer(buffer).unwrap();
    }
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

/// The `bf16` contract owes the same stream ordering the `f32` one owes.
#[test]
fn a_bf16_collective_that_honours_its_stream_passes_the_ordering_contract() {
    let mut device = FakeHostDeviceV1::new();

    assert_eq!(
        assert_generic_collective_ordering_contract_v1(
            &mut device,
            0,
            GenericCollectiveV1::SumBf16
        ),
        Ok(())
    );
    assert_eq!(device.live_object_count(), 0);
}

#[test]
fn a_bf16_collective_still_honours_its_stream_when_the_work_on_it_is_pending() {
    let mut device = FakeHostDeviceV1::new();
    // The writes that fill the rank buffers are now work ordered on the
    // collective's stream rather than work already done.
    device.defer_host_transfers(true);

    assert_eq!(
        assert_generic_collective_ordering_contract_v1(
            &mut device,
            0,
            GenericCollectiveV1::SumBf16
        ),
        Ok(()),
        "the collective must let its own stream's work land before reading"
    );
    assert_eq!(device.live_object_count(), 0);
}

#[test]
fn a_bf16_collective_that_reads_before_its_stream_lands_fails_the_ordering_contract() {
    let mut device = FakeHostDeviceV1::new();
    device.defer_host_transfers(true);
    // A backend that stages through some other stream behaves this way, and
    // the reduction it returns is a plausible number over stale bytes.
    device.skip_collective_stream_ordering(true);

    assert_eq!(
        assert_generic_collective_ordering_contract_v1(
            &mut device,
            0,
            GenericCollectiveV1::SumBf16
        ),
        Err(BundleStatusV1::InvalidState)
    );
    assert_eq!(device.live_object_count(), 0);
}
