#![allow(dead_code)]

use std::sync::Arc;

use runtime_grammar::{
    CompiledGrammar, ConstraintState, Grammar, GrammarMatcher, TokenVocabulary, VocabularyToken,
};
use serde_json::Value;

/// A small deterministic generator, so walks replay exactly.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

pub const STOP: u32 = 0;
pub const SPECIAL_OPEN: u32 = 1;

/// Every single byte, plus tokens that straddle JSON structure, escapes,
/// partial UTF-8 and whitespace runs. Token 0 is a stop token and token 1 a
/// special token spelled `<call>`.
pub fn adversarial_vocabulary() -> Arc<TokenVocabulary> {
    let mut tokens = vec![
        VocabularyToken {
            bytes: b"</s>".to_vec(),
            special: true,
        },
        VocabularyToken {
            bytes: b"<call>".to_vec(),
            special: true,
        },
    ];
    for byte in 0..=255_u8 {
        tokens.push(VocabularyToken {
            bytes: vec![byte],
            special: false,
        });
    }
    let texts: &[&[u8]] = &[
        b"{\"",
        b"\":",
        b"\": ",
        b"\",",
        b"\", \"",
        b"\"}",
        b"\"]",
        b"}]",
        b"]}",
        b"}}",
        b"]]",
        b"},{",
        b"}, {",
        b"[{",
        b"[\"",
        b"\"]}",
        b"1,",
        b"0]",
        b"12",
        b"-0",
        b"-1",
        b"1.5",
        b"e+",
        b"E-1",
        b"00",
        b".0",
        b"true",
        b"false",
        b"null",
        b"nul",
        b"tr",
        b"\\u00",
        b"\\\"",
        b"\\\\",
        b"\\n",
        b"\\u",
        b"u0041",
        b" ",
        b"  ",
        b"    ",
        b"\n",
        b"\n  ",
        b" \"",
        b"\t",
        b" }",
        b"\n}",
        b", ",
        b",\n",
        b": \"",
        b"name",
        b"\"name\"",
        b"age",
        b"\"age\": ",
        b"tags",
        b"items",
        b"red",
        b"green",
        b"blue",
        b"\"red\"",
        b"abc",
        b"xyz",
        b"</p>",
        b"</",
        b"p>",
        b"<p=",
        b"</call>",
        b"<call",
        b">",
        b"=",
        b"hello",
        b" world",
        b"\xc3",
        b"\xa9",
        b"\xc3\xa9",
        b"\xe2\x82",
        b"\xac",
        b"\xe2\x82\xac",
        b"\xf0\x9f",
        b"\x98\x80",
        b"\xf0\x9f\x98\x80",
        b"\xed\xa0\x80",
        b"\xc0\x80",
        b"12-34",
        b"2024-",
        b"-01",
        b"T12",
        b":00",
        b"Z",
        b"+05:30",
        b"aaaa",
        b"a\"b",
    ];
    for text in texts {
        tokens.push(VocabularyToken {
            bytes: text.to_vec(),
            special: false,
        });
    }
    // An empty token and a padded tail with no bytes at all.
    tokens.push(VocabularyToken {
        bytes: Vec::new(),
        special: false,
    });
    Arc::new(TokenVocabulary::new(tokens.len() + 7, tokens).unwrap())
}

pub fn compile(grammar: &Grammar, vocabulary: &Arc<TokenVocabulary>) -> Arc<CompiledGrammar> {
    let compiled = CompiledGrammar::new(grammar, Arc::clone(vocabulary)).unwrap();
    let usage = compiled.compilation_usage();
    assert!(
        usage.headroom() >= 10.0,
        "real corpus compilation needs 10x headroom: {usage:?}"
    );
    Arc::new(compiled)
}

pub fn bit(mask: &[u32], token: u32) -> bool {
    mask[token as usize / 32] & (1 << (token % 32)) != 0
}

/// The allowed set computed token by token, without caches.
pub fn brute_force(matcher: &GrammarMatcher, vocabulary: &TokenVocabulary) -> Vec<u32> {
    (0..vocabulary.size() as u32)
        .filter(|&token| matcher.accepts_token(token).unwrap())
        .collect()
}

pub fn allowed(mask: &[u32], size: usize) -> Vec<u32> {
    (0..size as u32).filter(|&token| bit(mask, token)).collect()
}

pub fn token_text(vocabulary: &TokenVocabulary, token: u32) -> Vec<u8> {
    if token == SPECIAL_OPEN {
        return b"<call>".to_vec();
    }
    vocabulary
        .ordinary_bytes(token)
        .unwrap_or_default()
        .to_vec()
}

/// Walks the constraint with random allowed tokens, checking at every step
/// that the cached mask equals the brute-force set. Returns the text and
/// whether the walk ended on the stop token.
pub fn random_walk(
    grammar: &Arc<CompiledGrammar>,
    rng: &mut Rng,
    steps: usize,
    check_masks: bool,
) -> (Vec<u8>, bool) {
    let vocabulary = Arc::clone(grammar.vocabulary());
    let mut state = ConstraintState::new(Arc::clone(grammar), None, vec![STOP]);
    let mut mask = vec![0_u32; vocabulary.mask_words()];
    let mut text = Vec::new();
    for _ in 0..steps {
        assert!(state.fill_mask(&mut mask).unwrap());
        let tokens = allowed(&mask, vocabulary.size());
        if check_masks {
            let mut expected = brute_force(state.matcher(), &vocabulary);
            if state.matcher().is_complete() {
                expected.push(STOP);
                expected.sort_unstable();
            }
            assert_eq!(
                tokens,
                expected,
                "after {:?}",
                String::from_utf8_lossy(&text)
            );
        }
        // Prefer ending once possible half the time, so walks finish.
        let token = if bit(&mask, STOP) && rng.below(2) == 0 {
            STOP
        } else {
            tokens[rng.below(tokens.len())]
        };
        for other in &tokens {
            assert!(state.allows(*other).unwrap());
        }
        state.accept(token).unwrap();
        if token == STOP {
            return (text, true);
        }
        text.extend(token_text(&vocabulary, token));
    }
    (text, false)
}

/// Parses generated JSON. Exponents beyond `f64` are legal JSON that
/// serde_json without arbitrary precision refuses; those samples return `None`.
pub fn parse_json(text: &[u8]) -> Option<Value> {
    match serde_json::from_slice(text) {
        Ok(value) => Some(value),
        Err(error) if error.to_string().contains("number out of range") => None,
        Err(error) => panic!("{error}: {}", String::from_utf8_lossy(text)),
    }
}

/// Validates the subset of JSON Schema the grammar supports, independently
/// of the grammar.
pub fn validates(value: &Value, schema: &Value, root: &Value) -> bool {
    let Value::Object(fields) = schema else {
        return schema == &Value::Bool(true);
    };
    if let Some(Value::String(reference)) = fields.get("$ref") {
        let target = root.pointer(reference.strip_prefix('#').unwrap()).unwrap();
        return validates(value, target, root);
    }
    // Every keyword holds at once, as JSON Schema defines it, independently
    // of how the compiler combines them.
    if fields
        .get("const")
        .is_some_and(|constant| value != constant)
    {
        return false;
    }
    if let Some(Value::Array(values)) = fields.get("enum")
        && !values.contains(value)
    {
        return false;
    }
    if let Some(Value::Array(branches)) = fields.get("anyOf")
        && !branches.iter().any(|branch| validates(value, branch, root))
    {
        return false;
    }
    if let Some(Value::Array(branches)) = fields.get("oneOf")
        && branches
            .iter()
            .filter(|branch| validates(value, branch, root))
            .count()
            != 1
    {
        return false;
    }
    if let Some(Value::Array(parts)) = fields.get("allOf")
        && !parts.iter().all(|part| validates(value, part, root))
    {
        return false;
    }
    let kinds: Vec<&str> = match fields.get("type") {
        Some(Value::String(kind)) => vec![kind],
        Some(Value::Array(kinds)) => kinds.iter().map(|kind| kind.as_str().unwrap()).collect(),
        _ => Vec::new(),
    };
    let matches_kind = |kind: &str| match (kind, value) {
        ("object", Value::Object(_))
        | ("array", Value::Array(_))
        | ("string", Value::String(_))
        | ("boolean", Value::Bool(_))
        | ("null", Value::Null) => true,
        ("integer", Value::Number(number)) => {
            number.is_i64() || number.is_u64() || number.as_f64().is_some_and(|v| v.fract() == 0.0)
        }
        ("number", Value::Number(_)) => true,
        _ => false,
    };
    if !kinds.is_empty() && !kinds.iter().any(|kind| matches_kind(kind)) {
        return false;
    }
    match value {
        Value::Object(members) => {
            let properties = fields.get("properties").and_then(Value::as_object);
            if let Some(Value::Array(required)) = fields.get("required")
                && required
                    .iter()
                    .any(|name| !members.contains_key(name.as_str().unwrap()))
            {
                return false;
            }
            for (name, member) in members {
                match properties.and_then(|properties| properties.get(name)) {
                    Some(schema) => {
                        if !validates(member, schema, root) {
                            return false;
                        }
                    }
                    None => match fields.get("additionalProperties") {
                        Some(Value::Bool(false)) => return false,
                        None if properties.is_some() => return false,
                        Some(schema @ Value::Object(_)) => {
                            if !validates(member, schema, root) {
                                return false;
                            }
                        }
                        _ => {}
                    },
                }
            }
            // Declared properties appear in schema order.
            if let Some(properties) = properties {
                let order: Vec<usize> = members
                    .keys()
                    .filter_map(|name| properties.keys().position(|key| key == name))
                    .collect();
                if order.windows(2).any(|pair| pair[0] >= pair[1]) {
                    return false;
                }
            }
            true
        }
        Value::Array(items) => {
            let min = fields.get("minItems").and_then(Value::as_u64).unwrap_or(0);
            let max = fields
                .get("maxItems")
                .and_then(Value::as_u64)
                .unwrap_or(u64::MAX);
            let count = items.len() as u64;
            if count < min || count > max {
                return false;
            }
            match fields.get("items") {
                Some(schema) => items.iter().all(|item| validates(item, schema, root)),
                None => true,
            }
        }
        Value::String(text) => {
            let length = text.chars().count() as u64;
            let min = fields.get("minLength").and_then(Value::as_u64).unwrap_or(0);
            let max = fields
                .get("maxLength")
                .and_then(Value::as_u64)
                .unwrap_or(u64::MAX);
            length >= min && length <= max
        }
        Value::Number(number) => {
            // Overflowed finite decimal numbers compare beyond every finite
            // bound; arbitrary precision now lets those samples reach this oracle.
            let value = number
                .as_f64()
                .unwrap_or_else(|| number.to_string().parse::<f64>().unwrap());
            let check = |key: &str, test: &dyn Fn(f64) -> bool| {
                fields.get(key).and_then(Value::as_f64).is_none_or(test)
            };
            check("minimum", &|bound| value >= bound)
                && check("maximum", &|bound| value <= bound)
                && check("exclusiveMinimum", &|bound| value > bound)
                && check("exclusiveMaximum", &|bound| value < bound)
        }
        _ => true,
    }
}

/// Isolate adversarial probes with hard time and address-space limits.
#[cfg(target_os = "linux")]
pub fn bounded(name: &str, probe: impl FnOnce()) {
    if std::env::var("GRAMMAR_RESOURCE_PROBE").as_deref() == Ok(name) {
        probe();
        return;
    }
    let result = std::process::Command::new("bash")
        .args(["-c", "ulimit -c 0; ulimit -v 524288; exec timeout 20s \"$1\" --exact \"$2\" --nocapture --test-threads=1", "grammar-probe"])
        .arg(std::env::current_exe().unwrap())
        .arg(name)
        .env("GRAMMAR_RESOURCE_PROBE", name)
        .status().unwrap();
    assert!(
        result.success(),
        "{name} exceeded 20 seconds or 512 MiB, or failed: {result}"
    );
}
