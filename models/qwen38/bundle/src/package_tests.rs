//! Native allocation and graph checks against model-owned package declarations.
use crate::contract::*;
use model_compiler_sdk::package_draft;

use model_qwen38_contract::package::{
    DEVICE_BYTES, Encoders, device_weight_bytes, maximum_sequences, ngram_in_host_memory,
    rank_budget, working_bytes,
};

/// What an instance can verify: nothing, host drafts at two, four or eight
/// verify rows per item, and the MTP drafter at eight. Every package can
/// verify with host drafts; MTP packages also with their drafter.
fn speculations() -> [(&'static str, crate::scratch::Speculation); 5] {
    let host = |verify_rows| crate::scratch::Speculation {
        verify_rows,
        drafter: false,
    };
    [
        ("none", Default::default()),
        ("host drafts, 2 rows", host(2)),
        ("host drafts, 4 rows", host(4)),
        (
            "host drafts, 8 rows",
            host(crate::contract::MAX_VERIFY_ROWS),
        ),
        (
            "MTP, 8 rows",
            crate::scratch::Speculation {
                verify_rows: crate::contract::MAX_VERIFY_ROWS,
                drafter: true,
            },
        ),
    ]
}

/// Device bytes one rank's bundle needs besides its weights when it serves
/// `sequences` sequences with each speculation, in [`speculations`] order:
/// its scratch (the buffer table `create` allocates, verify checkpoints
/// included), the recurrent state of every slot, the ring state of every
/// state slot, and a 1 GiB KV floor.
fn working_requirements(shape: Shape, sequences: u32) -> [u64; 5] {
    use crate::scratch::sizes;
    let items = crate::contract::Items::new(sequences).unwrap();
    let served = crate::scratch::Served::new(items);
    let target = crate::instance::target(shape, false);
    let workspace = target.workspace_bytes(served).unwrap();
    let state = shape
        .recurrent_state_layouts()
        .iter()
        .map(|(_, bytes)| bytes)
        .sum::<u64>()
        * u64::from(sequences);
    let ring = u64::from(items.capacity()) * shape.ring_state_slot_bytes(true);
    speculations().map(|(_, speculation)| {
        let scratch: u64 = sizes(
            &target.comm_bytes(served),
            workspace,
            speculation,
            shape.page_bytes_bf16_mtp(),
            shape,
            served,
        )
        .iter()
        .map(|(_, bytes)| bytes)
        .sum();
        scratch + state + ring + (1 << 30)
    })
}

/// The contract's working memory, which sizes the declared minimum and the
/// package maximum, bounds the buffer table's for every speculation at every
/// served count up to 256, and is within 1.5 GiB of the largest at the
/// graph buckets it is modeled on.
#[test]
fn contract_working_bytes_bound_the_buffer_table_at_every_count() {
    for shape in model_qwen38_contract::geometry::GEOMETRIES
        .iter()
        .flat_map(|g| {
            [Ranks::TP2, Ranks::TP4]
                .map(|ranks| Shape::of_geometry(g, ranks).expect("served geometry"))
        })
    {
        for sequences in 1..=DECODE_BATCH {
            let bound = working_bytes(shape, sequences);
            let exact = working_requirements(shape, sequences);
            for ((name, _), exact) in speculations().iter().zip(exact) {
                assert!(
                    exact <= bound,
                    "{shape:?} C{sequences} {name}: {exact} > {bound}"
                );
            }
            if sequences.is_power_of_two() && sequences >= 32 {
                let largest = exact.into_iter().max().unwrap();
                assert!(
                    bound - largest < 3 << 29,
                    "{shape:?} C{sequences}: {largest} < {bound}"
                );
            }
        }
    }
}

/// The declared minimum covers a rank's needs before KV at every served
/// count the package admits (its maximum sequences), for every speculation,
/// host drafts included, and with the encoder, leaving at least 4 GiB for
/// the host's arenas, modules, sampling workspace and 1 GiB reserve; and it
/// fits its device. At 256 sequences TP4 fits; TP2's needs exceed its
/// device, so its package admits fewer.
#[test]
fn declared_minimum_covers_the_maximum_sequences_and_fits_the_device() {
    let gib = |bytes: u64| bytes as f64 / f64::from(1 << 30);
    for shape in model_qwen38_contract::geometry::GEOMETRIES
        .iter()
        .flat_map(|g| {
            [Ranks::TP2, Ranks::TP4].map(|r| Shape::of_geometry(g, r).expect("served geometry"))
        })
    {
        let ngram_host = ngram_in_host_memory(shape);
        let weights = device_weight_bytes(shape, ngram_host);
        for encoders in Encoders::SETS {
            let (media, encoder) = (encoders.vision, encoders.media_bytes_for(shape));
            for mtp in [false, true] {
                let admitted = maximum_sequences(shape, mtp, encoders);
                let declared =
                    model_qwen38_contract::package::minimum_device_memory(shape, mtp, encoders);
                assert!(declared <= rank_budget(), "{shape:?}");
                let widest = working_requirements(shape, admitted)
                    .into_iter()
                    .max()
                    .unwrap();
                let needed = weights + encoder + widest;
                eprintln!(
                    "TP{} MTP {mtp} media {media}: C{admitted} needs {:.2} GiB, declares {:.0} GiB",
                    shape.ranks(),
                    gib(needed),
                    gib(declared)
                );
                assert!(needed + (4 << 30) <= declared, "{shape:?} C{admitted}");
            }
        }
        let widest = working_requirements(shape, DECODE_BATCH)
            .into_iter()
            .max()
            .unwrap();
        let fits = weights + widest + (4 << 30) <= rank_budget();
        assert_eq!(
            fits,
            maximum_sequences(shape, false, Encoders::NONE) == DECODE_BATCH
        );
        eprintln!(
            "TP{} C256: needs {:.2} GiB, budget {:.0} GiB",
            shape.ranks(),
            gib(weights + widest),
            gib(rank_budget())
        );
    }
    assert!(
        model_qwen38_contract::package::minimum_device_memory(Shape::TP2, false, Encoders::NONE)
            <= DEVICE_BYTES
    );
}

#[test]
fn sdk_decode_buckets_match_the_bundle_at_both_batch_limits() {
    for drafter in [false, true] {
        let limit = if drafter {
            model_qwen38_contract::package::MTP_QUALIFIED_SEQUENCES
        } else {
            crate::contract::DECODE_BATCH
        };
        let module: std::collections::BTreeSet<_> = (1..=limit)
            .map(|live| crate::graphs::decode_bucket(live).unwrap())
            .collect();
        assert_eq!(
            package_draft::decode_buckets(limit).collect::<Vec<_>>(),
            module.into_iter().collect::<Vec<_>>()
        );
    }
}
