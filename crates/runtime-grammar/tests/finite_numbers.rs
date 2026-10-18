//! Finite numbers between bounds against an exact oracle: a number with an
//! exponent is admitted by its value, within the documented refusals.
mod common;

use std::cmp::Ordering;
use std::sync::Arc;

use common::Rng;
use runtime_grammar::{
    CompiledGrammar, GrammarBuilder, GrammarMatcher, JsonOptions, SchemaCompiler, TokenVocabulary,
    VocabularyToken,
};
use serde_json::{Value, json};

const WINDOW: usize = 40;

fn finite_grammar(schema: &Value) -> Arc<CompiledGrammar> {
    let mut builder = GrammarBuilder::new();
    let expr = SchemaCompiler::new(&mut builder, schema, JsonOptions::default())
        .finite_numbers()
        .schema(schema)
        .unwrap();
    let root = builder.rule(expr);
    let grammar = builder.finish(root).unwrap();
    let tokens = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    let vocabulary = Arc::new(TokenVocabulary::new(256, tokens).unwrap());
    Arc::new(CompiledGrammar::new(&grammar, vocabulary).unwrap())
}

fn admits(grammar: &Arc<CompiledGrammar>, text: &str) -> bool {
    let mut matcher = GrammarMatcher::new(Arc::clone(grammar));
    matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_complete()
}

/// A nonzero number as its significand (from the first nonzero digit) and
/// order (the power of ten of that digit).
struct Value10 {
    negative: bool,
    significand: Vec<u8>,
    order: i64,
}

/// A spelling's parts: whether it is negative, its integer and fraction
/// digits, and its exponent.
struct Spelling {
    negative: bool,
    integer: String,
    fraction: Option<String>,
    exponent: Option<i64>,
}

impl Spelling {
    fn parse(text: &str) -> Self {
        let (negative, rest) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        let (mantissa, exponent) = match rest.split_once(['e', 'E']) {
            Some((mantissa, exponent)) => (mantissa, Some(exponent.parse().unwrap())),
            None => (rest, None),
        };
        let (integer, fraction) = match mantissa.split_once('.') {
            Some((integer, fraction)) => (integer.to_owned(), Some(fraction.to_owned())),
            None => (mantissa.to_owned(), None),
        };
        Self {
            negative,
            integer,
            fraction,
            exponent,
        }
    }

    fn value(&self) -> Option<Value10> {
        let fraction = self.fraction.as_deref().unwrap_or("");
        let digits = format!("{}{fraction}", self.integer);
        let first = digits.find(|digit| digit != '0')?;
        let exponent = self.exponent.unwrap_or(0);
        let order = if self.integer != "0" {
            self.integer.len() as i64 - 1 + exponent
        } else {
            -(first as i64 - 1) - 1 + exponent
        };
        Some(Value10 {
            negative: self.negative,
            significand: digits.as_bytes()[first..].to_vec(),
            order,
        })
    }

    /// Zeros that lead a fraction below 1.
    fn leading_zeros(&self) -> usize {
        let fraction = self.fraction.as_deref().unwrap_or("");
        fraction.len() - fraction.trim_start_matches('0').len()
    }
}

/// `significand` at `order` against a positive integer's digits.
fn compare_magnitude(significand: &[u8], order: i64, bound: &[u8]) -> Ordering {
    let len = bound.len() as i64;
    match (order + 1).cmp(&len) {
        Ordering::Equal => {}
        other => return other,
    }
    for index in 0..significand.len().max(bound.len()) {
        let left = significand.get(index).copied().unwrap_or(b'0');
        let right = bound.get(index).copied().unwrap_or(b'0');
        match left.cmp(&right) {
            Ordering::Equal => {}
            other => return other,
        }
    }
    Ordering::Equal
}

/// A nonzero value against an integer bound.
fn compare(value: &Value10, bound: i64) -> Ordering {
    let digits = bound.unsigned_abs().to_string().into_bytes();
    match (value.negative, bound.cmp(&0)) {
        (false, Ordering::Less | Ordering::Equal) => Ordering::Greater,
        (true, Ordering::Greater | Ordering::Equal) => Ordering::Less,
        (false, Ordering::Greater) => compare_magnitude(&value.significand, value.order, &digits),
        (true, Ordering::Less) => {
            compare_magnitude(&value.significand, value.order, &digits).reverse()
        }
    }
}

/// A magnitude limit and whether it is included.
type Limit = Option<(u64, bool)>;

#[derive(Clone, Copy, Debug)]
struct Bounds {
    lower: Option<(i64, bool)>,
    upper: Option<(i64, bool)>,
}

impl Bounds {
    fn schema(&self) -> Value {
        let mut schema = json!({"type": "number"});
        if let Some((value, inclusive)) = self.lower {
            schema[if inclusive {
                "minimum"
            } else {
                "exclusiveMinimum"
            }] = json!(value);
        }
        if let Some((value, inclusive)) = self.upper {
            schema[if inclusive {
                "maximum"
            } else {
                "exclusiveMaximum"
            }] = json!(value);
        }
        schema
    }

    fn contains(&self, value: Option<&Value10>) -> bool {
        let order = |bound: i64| match value {
            Some(value) => compare(value, bound),
            None => 0.cmp(&bound),
        };
        self.lower.is_none_or(|(bound, inclusive)| {
            order(bound) == Ordering::Greater || (inclusive && order(bound) == Ordering::Equal)
        }) && self.upper.is_none_or(|(bound, inclusive)| {
            order(bound) == Ordering::Less || (inclusive && order(bound) == Ordering::Equal)
        })
    }

    /// The magnitude limits a sign's values meet: a minimum above 0 and a
    /// maximum, each with whether it is included.
    fn limits(&self, negative: bool) -> (Limit, Limit) {
        let magnitude = |bound: Option<(i64, bool)>, positive: bool| {
            bound
                .filter(|(value, _)| if positive { *value > 0 } else { *value < 0 })
                .map(|(value, inclusive)| (value.unsigned_abs(), inclusive))
        };
        if negative {
            (magnitude(self.upper, false), magnitude(self.lower, false))
        } else {
            (magnitude(self.lower, true), magnitude(self.upper, true))
        }
    }

    /// Whether the grammar admits `text`: its value is within the bounds,
    /// finite below 1e308 unless an integer, and not a refused spelling.
    fn expected(&self, text: &str) -> bool {
        let spelling = Spelling::parse(text);
        let Some(value) = spelling.value() else {
            return self.contains(None);
        };
        if !self.contains(Some(&value)) {
            return false;
        }
        let integer = spelling.fraction.is_none() && spelling.exponent.is_none();
        if integer {
            return true;
        }
        if value.order > 307 {
            return false;
        }
        let Some(exponent) = spelling.exponent else {
            return true;
        };
        let (lower, upper) = self.limits(value.negative);
        let len = |limit: Limit| limit.map(|(value, _)| value.to_string().len());
        let greatest = len(upper).map_or(307, |len| len as i64 - 2);
        if spelling.integer.len() > 308 {
            return false;
        }
        if spelling.integer == "0" && spelling.leading_zeros() > WINDOW {
            return lower.is_none() && exponent <= greatest + WINDOW as i64 + 2;
        }
        if (lower.is_some() || upper.is_some()) && spelling.integer.len() > WINDOW {
            return false;
        }
        if let Some((minimum, false)) = lower {
            let digits = minimum.to_string().into_bytes();
            if value.order == digits.len() as i64 - 1
                && spelling.integer.len() > digits.len()
                && value.significand.starts_with(&digits)
            {
                return false;
            }
        }
        true
    }
}

fn digits(rng: &mut Rng, count: usize) -> String {
    (0..count)
        .map(|_| char::from(b'0' + rng.below(10) as u8))
        .collect()
}

/// A spelling aimed near a threshold: a significand from a bound's digits or
/// random ones, an integer part of a length near the counting limits or a
/// fraction with leading zeros, and an exponent putting it at an order near
/// a bound's, 0 or the largest.
fn spelling(rng: &mut Rng, bounds: &Bounds) -> String {
    let mut seeds: Vec<String> = [bounds.lower, bounds.upper]
        .into_iter()
        .flatten()
        .map(|(value, _)| value.unsigned_abs().to_string())
        .filter(|digits| digits != "0")
        .collect();
    let (first, more) = (1 + rng.below(9), rng.below(4));
    seeds.push(format!("{first}{}", digits(rng, more)));
    let seed = seeds[rng.below(seeds.len())].clone();
    let significand = match rng.below(4) {
        0 => seed,
        1 => format!("{seed}{}", "0".repeat(rng.below(3))),
        2 => {
            let more = 1 + rng.below(2);
            format!("{seed}{}", digits(rng, more))
        }
        _ => {
            let mut text = seed.into_bytes();
            let at = rng.below(text.len());
            text[at] = b'0' + rng.below(10) as u8;
            if text[0] == b'0' {
                text[0] = b'1';
            }
            String::from_utf8(text).unwrap()
        }
    };
    let orders: Vec<i64> = [bounds.lower, bounds.upper]
        .into_iter()
        .flatten()
        .map(|(value, _)| value.unsigned_abs().to_string().len() as i64)
        .flat_map(|len| [len - 2, len - 1, len])
        .chain([-1, 0, 1, 306, 307, 308, -400])
        .collect();
    let order = orders[rng.below(orders.len())];
    let negative = rng.below(2) == 0;
    let (mantissa, exponent) = if rng.below(3) == 0 {
        // A fraction below 1: `zeros` leading zeros, then the significand.
        let zeros = [0, 1, 2, WINDOW - 1, WINDOW, WINDOW + 1, WINDOW + 5][rng.below(7)];
        (
            format!("0.{}{significand}", "0".repeat(zeros)),
            order + zeros as i64 + 1,
        )
    } else {
        // An integer part of `integers` digits, padded with digits.
        let lengths = [
            1,
            2,
            3,
            17,
            18,
            19,
            WINDOW - 1,
            WINDOW,
            WINDOW + 1,
            307,
            308,
            309,
        ];
        let integers = lengths[rng.below(lengths.len())];
        let mut all = significand;
        if all.len() < integers {
            let pad = if rng.below(2) == 0 {
                "0".repeat(integers - all.len())
            } else {
                digits(rng, integers - all.len())
            };
            all.push_str(&pad);
        }
        let (integer, fraction) = all.split_at(integers.min(all.len()));
        let mantissa = if fraction.is_empty() {
            integer.to_owned()
        } else {
            format!("{integer}.{fraction}")
        };
        (mantissa, order - integer.len() as i64 + 1)
    };
    let sign = if negative { "-" } else { "" };
    if exponent == 0 && rng.below(3) == 0 {
        return format!("{sign}{mantissa}");
    }
    let mark = ["e", "E"][rng.below(2)];
    let exponent_sign = match (exponent.cmp(&0), rng.below(2)) {
        (Ordering::Less, _) => "-",
        (Ordering::Equal, 0) => "-",
        (_, 0) => "+",
        _ => "",
    };
    let zeros = "0".repeat(rng.below(3));
    format!(
        "{sign}{mantissa}{mark}{exponent_sign}{zeros}{}",
        exponent.unsigned_abs()
    )
}

#[test]
fn a_number_with_an_exponent_is_compared_by_value() {
    let bounds = [
        (None, None),
        (Some((0, true)), None),
        (Some((0, false)), None),
        (None, Some((0, true))),
        (None, Some((-1, true))),
        (Some((1, true)), None),
        (Some((5, false)), None),
        (Some((9, false)), None),
        (None, Some((99, true))),
        (None, Some((100, false))),
        (Some((10, true)), Some((100, true))),
        (Some((-100, false)), Some((100, false))),
        (Some((123, true)), Some((456, false))),
        (Some((120, false)), Some((129, true))),
        (Some((7, true)), Some((7, true))),
        (Some((-5_000, true)), Some((-20, false))),
        (Some((1_000_000_000_000_000, true)), None),
    ];
    let mut rng = Rng::new(0x5eed);
    for (lower, upper) in bounds {
        let bounds = Bounds { lower, upper };
        let grammar = finite_grammar(&bounds.schema());
        let mut admitted = 0;
        for _ in 0..3000 {
            let text = spelling(&mut rng, &bounds);
            let expected = bounds.expected(&text);
            assert_eq!(admits(&grammar, &text), expected, "{bounds:?}: {text}");
            admitted += usize::from(expected);
        }
        // The spellings reach both sides of each threshold.
        assert!(
            (20..2980).contains(&admitted),
            "{bounds:?}: {admitted} admitted"
        );
    }
}
