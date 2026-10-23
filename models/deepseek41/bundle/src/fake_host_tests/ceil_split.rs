//! Flash TP4 quantizes one row of the WO_a and WO_b inputs with the split
//! ceil twins (`sm120::ceil_split`), bound exactly as the served
//! quantizers, wherever a quantizer launch takes them: a one-row prefill.
//! One-row decode quantizes both inputs in its status folds
//! (`sm120::fold_fused`), so it launches neither form. Every other launch,
//! and every wider step, keeps its bindings.
use super::query_window::{arguments, canonical, decode_steps, first_difference, stream, symbol};
use super::*;

/// Each served one-row quantizer and its split twin.
const TWINS: [(&str, &str); 2] = [
    (
        "chop_deepseek41_dense_ceil_w8192_cached_v1",
        "chop_deepseek41_dense_ceil_w8192_split8_v1",
    ),
    (
        "chop_deepseek41_dense_ceil_f32_w2048_parallel_v1",
        "chop_deepseek41_dense_ceil_f32_w2048_split8_v1",
    ),
];

fn replaced(line: &str) -> bool {
    TWINS
        .iter()
        .any(|&(served, split)| symbol(line) == served || symbol(line) == split)
}

fn run_case(layout: KvLayout, served: bool) -> Vec<Vec<String>> {
    crate::sm120::ceil_split::with_served(served, || {
        let (cubins, steps) = decode_steps(layout);
        assert_eq!(
            cubins
                .iter()
                .any(|name| crate::sm120::ceil_split::is_cubin(name)),
            served
        );
        steps
    })
}

/// The launch lines of a one-token prompt's prefill on a Flash TP4
/// instance of `layout`, with the twins served or not.
fn prefill_case(layout: KvLayout, served: bool) -> Vec<String> {
    crate::sm120::ceil_split::with_served(served, || {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, layout);
        let mut instance = create(&host, 8);
        assert_eq!(
            host.device
                .module_cubins
                .borrow()
                .values()
                .any(|name| crate::sm120::ceil_split::is_cubin(name)),
            served
        );
        let from = host.device.journal.borrow().len();
        step(
            &mut instance,
            &prompt(1, 0, 0, 1, true),
            &mut Views::default(),
        );
        let lines = host.device.journal.borrow()[from..]
            .iter()
            .filter(|l| l.starts_with("launch "))
            .cloned()
            .collect();
        instance.retire_request(1).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        lines
    })
}

#[test]
fn one_row_prefill_quantizes_the_wo_inputs_with_the_split_twins() {
    for layout in KV_LAYOUTS {
        let reference = canonical(&[prefill_case(layout, false)], replaced).remove(0);
        let serving = canonical(&[prefill_case(layout, true)], replaced).remove(0);
        let count =
            |lines: &[String], name: &str| lines.iter().filter(|l| symbol(l) == name).count();
        for (served, split) in TWINS {
            // Every layer of every rank quantizes each input once.
            assert_eq!(count(&reference, served), 40 * 4, "{served}");
            assert_eq!(count(&reference, split), 0);
            assert_eq!(count(&serving, served), 0, "{served}");
            assert_eq!(count(&serving, split), 40 * 4, "{split}");
            // Each twin takes its served launch's stream and operands.
            let served: Vec<_> = reference.iter().filter(|l| symbol(l) == served).collect();
            let split: Vec<_> = serving.iter().filter(|l| symbol(l) == split).collect();
            for (served, split) in served.iter().zip(&split) {
                assert_eq!(stream(served), stream(split));
                assert_eq!(arguments(served), arguments(split));
            }
        }
        let others = |lines: &[String]| {
            lines
                .iter()
                .filter(|l| !replaced(l))
                .cloned()
                .collect::<Vec<_>>()
        };
        first_difference("prefill launches", &others(&reference), &others(&serving));
    }
}

#[test]
fn one_row_decode_quantizes_the_wo_inputs_in_its_status_folds() {
    for layout in KV_LAYOUTS {
        let reference = canonical(&run_case(layout, false), replaced);
        let serving = canonical(&run_case(layout, true), replaced);
        // The sliced WO_a input quantizer and WO_a's merge replace both
        // quantizers at every layer of every rank.
        let sliced = |lines: &[String]| {
            lines
                .iter()
                .filter(|l| symbol(l).starts_with("chop_deepseek41_dense_ceil_w8192_sliced_"))
                .count()
        };
        assert_eq!(sliced(&serving[0]), 40 * 4);
        for (old, new) in reference.iter().zip(&serving) {
            assert!(old.iter().chain(new).all(|l| !replaced(l)));
            first_difference("decode launches", old, new);
        }
    }
}

/// With the query window twins served as well, the one-row step replaces
/// exactly the query window launches, and every wider step keeps its own.
#[test]
fn the_query_window_twins_alone_change_the_one_row_decode() {
    let either = |line: &str| replaced(line) || super::query_window::replaced(line);
    let run = |served: bool| {
        crate::sm120::query_window::with_served(served, || run_case(KV_LAYOUTS[0], served))
    };
    let reference = canonical(&run(false), either);
    let serving = canonical(&run(true), either);
    let others = |lines: &[String]| {
        lines
            .iter()
            .filter(|l| !either(l))
            .cloned()
            .collect::<Vec<_>>()
    };
    first_difference(
        "one-row launches",
        &others(&reference[0]),
        &others(&serving[0]),
    );
    let twins = serving[0]
        .iter()
        .filter(|l| symbol(l).starts_with("chop_deepseek41_query_window_twin_"))
        .count();
    let splits = serving[0]
        .iter()
        .filter(|l| symbol(l).ends_with("_split8_v1"))
        .count();
    assert_eq!((twins, splits), (40 * 4, 0));
    for (old, new) in reference.iter().zip(&serving).skip(1) {
        first_difference("wider or replayed launches", old, new);
    }
}
