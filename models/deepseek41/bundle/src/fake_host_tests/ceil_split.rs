//! Flash TP4 quantizes one row of the WO_a and WO_b inputs with the split
//! ceil twins (`sm120::ceil_split`), bound exactly as the served
//! quantizers; every other launch, and every wider step, keeps its
//! bindings.
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

#[test]
fn one_row_decode_quantizes_the_wo_inputs_with_the_split_twins() {
    for layout in KV_LAYOUTS {
        let reference = canonical(&run_case(layout, false), replaced);
        let serving = canonical(&run_case(layout, true), replaced);
        let count =
            |lines: &[String], name: &str| lines.iter().filter(|l| symbol(l) == name).count();
        let [first, two, eight, last] = [0, 1, 2, 3].map(|i| (&reference[i], &serving[i]));
        for (served, split) in TWINS {
            // Every layer of every rank quantizes each input once.
            assert_eq!(count(first.0, served), 40 * 4, "{served}");
            assert_eq!(count(first.0, split), 0);
            assert_eq!(count(first.1, served), 0, "{served}");
            assert_eq!(count(first.1, split), 40 * 4, "{split}");
            // Each twin takes its served launch's stream and operands.
            let served: Vec<_> = first.0.iter().filter(|l| symbol(l) == served).collect();
            let split: Vec<_> = first.1.iter().filter(|l| symbol(l) == split).collect();
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
        first_difference("one-row launches", &others(first.0), &others(first.1));
        // Wider steps keep the served quantizers; the replayed step
        // launches none.
        first_difference("two-row launches", two.0, two.1);
        first_difference("eight-row launches", eight.0, eight.1);
        first_difference("replayed one-row launches", last.0, last.1);
    }
}

/// With the query window twins served as well, the one-row step replaces
/// exactly both sets of launches and every wider step keeps its own.
#[test]
fn the_split_twins_serve_beside_the_query_window_twins() {
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
    assert_eq!((twins, splits), (40 * 4, 2 * 40 * 4));
    for (old, new) in reference.iter().zip(&serving).skip(1) {
        first_difference("wider or replayed launches", old, new);
    }
}
