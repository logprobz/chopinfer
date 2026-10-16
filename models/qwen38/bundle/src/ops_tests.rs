//! Every prepared op claims the exactness its key needs.

use super::*;

/// The topologies and served counts the ops are checked at.
fn instances() -> Vec<(Shape, Served)> {
    [Shape::TP4, Shape::TP2]
        .into_iter()
        .flat_map(|shape| {
            [1, 32, DECODE_BATCH]
                .into_iter()
                .filter_map(move |items| Some((shape, Served::new(Items::new(items)?))))
        })
        .collect()
}

/// Every key of an instance serving `served`.
fn keys(served: Served) -> Vec<Key> {
    std::iter::once(Key::Prefill)
        .chain(buckets(served).flat_map(|bucket| [Key::Decode(bucket), Key::Verify(bucket)]))
        .collect()
}

fn prepared(shape: Shape, served: Served) -> Ops {
    Ops::prepare(
        crate::instance::target(shape, false).as_ref(),
        shape,
        served,
        true,
        1,
        true,
        false,
    )
    .unwrap_or_else(|refused| panic!("{shape:?}: {refused}"))
}

/// Every prepared op claims what its key needs: a prefill op keeps the
/// manifest's chunk granule, so a backend that cannot refuses create.
#[test]
fn prepared_ops_keep_the_prefill_granule_and_exact_verify_rows() {
    for (shape, served) in instances() {
        let ops = prepared(shape, served);
        for key in keys(served) {
            let stage = ops.stage(key).unwrap();
            for info in [
                stage.norm.info(),
                stage.bottleneck.info(),
                stage.mix.info(),
                stage.ple_gate.info(),
                stage.head_rows.info(),
                stage.gated_norm.info(),
            ] {
                assert!(
                    info.guarantees
                        .provides(exact(key, key.capacity(), Anchors::NotApplicable)),
                    "{key:?} {}",
                    info.implementation
                );
            }
            for info in [stage.conv.info(), stage.delta.info()] {
                assert!(
                    info.guarantees
                        .provides(exact(key, key.capacity(), Anchors::Spare)),
                    "{key:?} {}",
                    info.implementation
                );
            }
        }
        for key in keys(served) {
            let attention = ops.attention(key).unwrap();
            let pieces = match key {
                Key::Prefill => SCORE_ROWS,
                Key::Decode(rows) | Key::Verify(rows) => rows,
            };
            let infos = [
                (
                    attention.kv_write(false).info(),
                    key.capacity(),
                    Anchors::NotApplicable,
                ),
                (
                    attention.kv_write(true).info(),
                    key.capacity(),
                    Anchors::NotApplicable,
                ),
                (
                    attention.index_query.info(),
                    key.capacity(),
                    Anchors::NotApplicable,
                ),
                (attention.key_write.info(), key.capacity(), ANCHORED_RING),
                (attention.scores.info(), pieces, Anchors::NotApplicable),
                (attention.top_k.info(), pieces, Anchors::NotApplicable),
                (
                    attention.attention(false).info(),
                    key.capacity(),
                    Anchors::NotApplicable,
                ),
                (
                    attention.attention(true).info(),
                    key.capacity(),
                    Anchors::NotApplicable,
                ),
                (
                    attention.output_gate.info(),
                    key.capacity(),
                    Anchors::NotApplicable,
                ),
            ];
            for (info, rows, anchors) in infos {
                assert!(
                    info.guarantees.provides(exact(key, rows, anchors)),
                    "{key:?} {}",
                    info.implementation
                );
            }
            let moe = ops.moe(key).unwrap();
            let need = exact(key, key.capacity(), Anchors::NotApplicable);
            let route_need = need.of_routed_operation();
            for rank in 0..shape.ranks() {
                let route = moe.route(rank).unwrap().info();
                assert!(
                    route.guarantees.provides(route_need),
                    "{key:?} {}",
                    route.implementation
                );
                let combine = moe.combine(rank).unwrap().info();
                assert!(
                    combine.guarantees.provides(need),
                    "{key:?} {}",
                    combine.implementation
                );
            }
            for info in [moe.gate_up.info(), moe.down.info()] {
                assert!(
                    info.guarantees.provides(need),
                    "{key:?} {}",
                    info.implementation
                );
            }
            let collectives = ops.collectives(key).unwrap();
            for rank in 0..shape.ranks() {
                let info = collectives.reduce(rank, false).unwrap().info();
                assert!(
                    info.guarantees.provides(need),
                    "{key:?} {}",
                    info.implementation
                );
            }
        }
        let prefill = ops.stage(Key::Prefill).unwrap();
        assert!(prefill.delta.info().guarantees.chunks_at(PREFILL_GRANULE));
    }
}
