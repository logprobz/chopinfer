use super::*;

/// A small deterministic generator, so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

fn case(rng: &mut Rng) -> ([u32; RULE_WORDS], Vec<Item>, Vec<[f32; BLOCK]>) {
    let count = 1 + rng.below(32) as usize;
    let mut items = Vec::new();
    let (mut first_row, mut drafting) = (0, 0);
    let mut scores = Vec::new();
    for _ in 0..count {
        let reserved = rng.below(BLOCK as u64 + 1) as u32;
        let kind = if rng.below(2) == 0 {
            SamplingKindV1::Greedy
        } else {
            SamplingKindV1::Sampled
        };
        let score_row = if reserved == 0 { NONE } else { drafting };
        if reserved > 0 {
            drafting += 1;
            // Coarse scores make equal survivals, so ties are exercised.
            let coarse = rng.below(4) == 0;
            scores.push(std::array::from_fn(|_| match rng.below(40) {
                0 => f32::NAN,
                1 => f32::INFINITY,
                _ if coarse => (rng.below(3) as f32) - 1.0,
                _ => (rng.below(2001) as f32 - 1000.0) / 250.0,
            }));
        }
        items.push(Item {
            reserved,
            kind,
            score_row,
            first_row,
        });
        first_row += 1 + reserved;
    }
    let width = count as u32 + rng.below(u64::from(first_row) - count as u64 + 1) as u32;
    let mut rule = [0; RULE_WORDS];
    rule[0] = width;
    rule[1] = rng.below(BLOCK as u64 + 1) as u32;
    for word in &mut rule[2..] {
        *word = ((rng.below(801) as f32 - 400.0) / 100.0).to_bits();
    }
    (rule, items, scores)
}

/// The kernel's rank-count schedule equals the SDK rule on random items,
/// ties, NaN and infinite scores, and every floor and width, and the
/// layouts hold the contract's invariants.
#[test]
fn ranked_allocation_equals_the_sdk_rule() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut ties = 0;
    for _ in 0..4000 {
        let (rule, items, scores) = case(&mut rng);
        let reference = allocate_reference(&rule, &items, &scores).unwrap();
        assert_eq!(
            allocate_ranked(&rule, &items, &scores).unwrap(),
            reference,
            "{rule:?} {items:?} {scores:?}"
        );
        let reserved_rows: u32 = items.iter().map(|item| 1 + item.reserved).sum();
        let width = rule[0];
        assert_eq!(reference.rows.len() as u32, width);
        assert_eq!(reference.expand.len() as u32, reserved_rows);
        for (item, &k) in items.iter().zip(&reference.verified) {
            assert!(k <= item.reserved);
        }
        for (w, &row) in reference.rows.iter().enumerate() {
            assert_eq!(reference.expand[row as usize], w as u32);
            assert!(w == 0 || reference.rows[w - 1] < row);
        }
        for (r, &w) in reference.expand.iter().enumerate() {
            // A reserved row takes a computed row at or after its own
            // place in a width-row tail of the reserved rows.
            assert!(reserved_rows - width + w >= r as u32);
            assert!(r == 0 || reference.expand[r - 1] <= w);
        }
        ties += usize::from(scores.iter().any(|row| row.iter().any(|s| *s == 0.0)));
    }
    assert!(ties > 100);
}

/// Inconsistent items or widths are refused rather than allocated.
#[test]
fn inconsistent_inputs_are_refused() {
    let item = |reserved, score_row, first_row| Item {
        reserved,
        kind: SamplingKindV1::Sampled,
        score_row,
        first_row,
    };
    let mut rule = [0; RULE_WORDS];
    let scores = [[0.0; BLOCK]];
    let good = [item(2, 0, 0), item(0, NONE, 3)];
    for width in [2, 4] {
        rule[0] = width;
        assert!(allocate_reference(&rule, &good, &scores).is_ok());
    }
    for width in [1, 5] {
        rule[0] = width;
        assert!(
            allocate_reference(&rule, &good, &scores).is_err(),
            "{width}"
        );
    }
    rule[0] = 3;
    for bad in [
        [item(2, 0, 1), item(0, NONE, 3)],
        [item(2, NONE, 0), item(0, NONE, 3)],
        [item(2, 1, 0), item(0, NONE, 3)],
        [item(6, 0, 0), item(0, NONE, 7)],
        [item(2, 0, 0), item(0, 0, 3)],
    ] {
        assert!(allocate_reference(&rule, &bad, &scores).is_err(), "{bad:?}");
        assert!(allocate_ranked(&rule, &bad, &scores).is_err(), "{bad:?}");
    }
}

/// The rule record carries the SDK allocation's width, floor and pairs.
#[test]
fn rule_records_carry_the_allocation() {
    let calibration = VerifyCalibrationV1::new(
        &[(1.5, -0.25), (1.0, 0.0), (0.5, 0.125)],
        &[(0.75, 0.5), (2.0, -1.0), (0.25, 0.0)],
    )
    .unwrap();
    let allocation = VerifyAllocationV1::new(12, 1, calibration, vec![SamplingKindV1::Greedy]);
    let rule = encode_rule(&allocation);
    assert_eq!((rule[0], rule[1]), (12, 1));
    assert_eq!(f32::from_bits(rule[2 + 2 * 2]), 0.5);
    assert_eq!(f32::from_bits(rule[3 + 2 * BLOCK + 2]), -1.0);
    assert_eq!(rule[2 + 2 * 3..2 + 2 * BLOCK], [0; 4]);
}
