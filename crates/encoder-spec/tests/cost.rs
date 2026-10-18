//! The cost model over Qwen3.8's vision declaration and the DeepSeek
//! fixture: GEMMs linear in rows, full attention quadratic in the sequence.

use encoder_spec::cost::Cost;
use encoder_spec::geometry::Media;
use encoder_spec::{Declaration, Environment, parse};

const QWEN: &str = include_str!("../../../models/qwen38/contract/encoders/qwen38-vision.json");
const DSV41: &str = include_str!("fixtures/dsv41-vision.json");

fn costs(declaration: &Declaration, tag: &str, grid: [u64; 3]) -> Vec<Cost> {
    let geometry = declaration.geometry(tag, Media::Grid(grid)).unwrap();
    assert_eq!(geometry.runs.len(), 1);
    declaration.cost(&geometry.runs[0]).unwrap()
}

#[test]
fn qwen_costs_follow_the_declared_math() {
    let qwen = parse(QWEN, &Environment::new("qwen38")).unwrap();
    for [h, w] in [[2, 2], [8, 12], [64, 64], [256, 254]] {
        let rows = h * w;
        let merged = rows / 4;
        let costs = costs(&qwen, "image.patch-grid.v1", [1, h, w]);
        assert_eq!(costs.len(), 7);
        assert_eq!(costs[0].flops, 2 * rows * 1536 * 1152);
        // F32 payload in, F32 rows out, the BF16 patch weight and bias.
        assert_eq!(
            costs[0].bytes,
            rows * (1536 + 1152) * 4 + (1152 * 1536 + 1152) * 2
        );
        let gemms = 2 * rows * 1152 * (3456 + 1152 + 2 * 4352);
        let attention = 4 * 16 * 72 * rows * rows;
        let norms = 2 * 8 * rows * 1152;
        assert_eq!(costs[2].flops, 27 * (gemms + attention + norms));
        assert_eq!(costs[4], Cost::default());
        assert_eq!(costs[5].flops, 2 * merged * 4608 * 4608);
        assert_eq!(costs[6].flops, 2 * merged * 4608 * 2560);
        // The merged BF16 rows leave the last GEMM.
        assert_eq!(
            costs[6].bytes,
            merged * (4608 * 4 + 2560 * 2) + (2560 * 4608 + 2560) * 2
        );
    }
    // Doubling the patches doubles the GEMMs and quadruples attention.
    let small = costs(&qwen, "image.patch-grid.v1", [1, 64, 64]);
    let large = costs(&qwen, "image.patch-grid.v1", [1, 128, 64]);
    assert_eq!(large[0].flops, 2 * small[0].flops);
    let attention = |rows: u64| 27 * 4 * 16 * 72 * rows * rows;
    assert_eq!(
        large[2].flops - attention(8192),
        2 * (small[2].flops - attention(4096))
    );
    let geometry = qwen
        .geometry("image.patch-grid.v1", Media::Grid([1, 8, 12]))
        .unwrap();
    let total = costs(&qwen, "image.patch-grid.v1", [1, 8, 12])
        .into_iter()
        .fold(Cost::default(), |sum, cost| sum + cost);
    assert_eq!(qwen.item_cost(&geometry).unwrap(), total);
}

#[test]
fn deepseek_costs_count_its_space_to_depth_rows() {
    let dsv41 = parse(DSV41, &Environment::new("dsv41")).unwrap();
    let costs = costs(&dsv41, "image.centered-pad.v1", [1, 5, 7]);
    let (rows, cells) = (35, 6);
    let swiglu = 2 * rows * (1024 * 2 * 2816 + 2816 * 1024);
    let gemms = 2 * rows * 1024 * (3 * 1024 + 1024) + swiglu;
    let attention = 4 * 16 * 64 * rows * rows;
    assert_eq!(
        costs[1].flops,
        32 * (gemms + attention + 2 * 8 * rows * 1024)
    );
    assert_eq!(costs[4].flops, 2 * cells * 9216 * 5120);
    assert_eq!(costs[5].flops, 2 * cells * 5120 * 5120);
}
