//! Engram: the n-gram hash that picks each position's table rows, and the
//! gate that writes the looked-up value into the residual streams
//! (`inference/engram.py`, `inference/model.py:296-366`).
//!
//! The hash runs on the host: its ids index the host-mapped tables, which
//! the host gathers into staging rows, so it is a contract function over
//! the package's hash resource ([`HashTables`]), exact in integers. The gate
//! is a device op ([`EngramGate`]) with an f64 oracle and bound.

use crate::geometry::{Engram, Geometry};
use kernel_core::backend::Need;
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::Refusal;
use kernel_core::oracle::{NeumaierSum, accumulation_bound};
use sha2::{Digest, Sha256};

/// The hash resource the package ships, generated offline from the pinned
/// vendor code by `models/deepseek41/engram_resource.py`: the compressed
/// token map and the multipliers of every layer id below [`LAYER_IDS`].
pub const RESOURCE: &[u8] = include_bytes!("../../resources/engram-hash.v2.bin");

/// SHA-256 of [`RESOURCE`]. [`HashTables::new`] refuses other bytes.
pub const RESOURCE_SHA256: &str =
    "584e3b3950054ebbd372fadadccb8094c3630af2d8fe22c46b4a23787fd09869";

/// The package component format of [`RESOURCE`].
pub const RESOURCE_FORMAT: &str = "deepseek41.engram-hash.v2";

/// The resource holds multipliers for every Engram layer id below this.
pub const LAYER_IDS: u32 = 64;

const MAGIC: &[u8; 8] = b"DS41ENG2";

/// A position's compressed token that takes no part in an n-gram: inside an
/// image span (the vendor's `NgramHashState.DEAD`).
pub const DEAD: i64 = -1;

/// One geometry's hash: the compressed token map, one odd multiplier per
/// (Engram layer, lookback), and the bucket primes and offsets.
#[derive(Clone, Debug, PartialEq)]
pub struct HashTables {
    engram: Engram,
    compressed_vocab: u32,
    pad: i64,
    /// `[layer][lookback]`.
    multipliers: Vec<i64>,
    /// `[layer][column]`, column `(n - 2) * heads + head` for n-gram size n.
    primes: Vec<u64>,
    /// `[layer][column]`: the start of each column's bucket range in the
    /// layer's table.
    offsets: Vec<u64>,
    token_map: Vec<u32>,
}

/// Reads little-endian fields in order.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let (head, rest) = self
            .bytes
            .split_first_chunk::<N>()
            .ok_or("the Engram hash resource is truncated")?;
        self.bytes = rest;
        Ok(*head)
    }

    fn u32(&mut self) -> Result<u32, String> {
        self.take().map(u32::from_le_bytes)
    }

    fn i64(&mut self) -> Result<i64, String> {
        self.take().map(i64::from_le_bytes)
    }
}

/// Deterministic Miller-Rabin for every u64.
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2_u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let mul = |a: u64, b: u64| (u128::from(a) * u128::from(b) % u128::from(n)) as u64;
    let pow = |mut base: u64, mut exp: u64| {
        let mut result = 1;
        while exp > 0 {
            if exp & 1 == 1 {
                result = mul(result, base);
            }
            base = mul(base, base);
            exp >>= 1;
        }
        result
    };
    let (mut d, mut s) = (n - 1, 0);
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }
    [2_u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        .iter()
        .all(|&a| {
            let mut x = pow(a, d);
            if x == 1 || x == n - 1 {
                return true;
            }
            (1..s).any(|_| {
                x = mul(x, x);
                x == n - 1
            })
        })
}

impl HashTables {
    /// The shipped resource under `geometry`'s Engram layout.
    pub fn of(geometry: &Geometry) -> Result<HashTables, String> {
        HashTables::new(RESOURCE, geometry.shape.vocab, geometry.shape.engram)
    }

    /// Parses `bytes` (refusing any but [`RESOURCE_SHA256`]) for a
    /// vocabulary of `vocab` ids and the Engram layout `engram`: every token
    /// maps into the compressed vocabulary and every compressed id is used;
    /// multipliers are odd and positive and keep `id * multiplier` within
    /// i64. The bucket primes are the vendor's draw (each the least unused
    /// prime above the bucket base, in layer, n-gram size and head order),
    /// and must sum to each layer's table rows.
    pub fn new(bytes: &[u8], vocab: u32, engram: Engram) -> Result<HashTables, String> {
        let digest: String = Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if digest != RESOURCE_SHA256 {
            return Err(format!(
                "Engram hash resource sha256 {digest} is not the pinned {RESOURCE_SHA256}"
            ));
        }
        let mut r = Reader { bytes };
        if &r.take::<8>()? != MAGIC {
            return Err("not an Engram hash resource".into());
        }
        let header = [r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?];
        let [resource_vocab, compressed, pad_raw, pad, max_ngram, ids] = header;
        if [resource_vocab, compressed, pad_raw, max_ngram, ids]
            != [
                vocab,
                engram.compressed_vocab,
                engram.pad_token,
                engram.max_ngram,
                LAYER_IDS,
            ]
        {
            return Err(format!(
                "Engram hash resource header {header:?} does not match the geometry"
            ));
        }
        if !(2..=8).contains(&max_ngram)
            || engram.heads == 0
            || engram.layers.len() != engram.rows.len()
        {
            return Err("Engram layout outside the hash's n-gram sizes or heads".into());
        }
        let all = (0..ids * max_ngram)
            .map(|_| r.i64())
            .collect::<Result<Vec<_>, _>>()?;
        let bound = i64::MAX / i64::from(compressed) / 2;
        if let Some(m) = all
            .iter()
            .find(|&&m| m <= 0 || m % 2 == 0 || m > 2 * bound + 1)
        {
            return Err(format!("Engram multiplier {m} is not odd within its bound"));
        }
        let mut multipliers = Vec::new();
        for &layer in engram.layers {
            if layer >= LAYER_IDS {
                return Err(format!(
                    "Engram layer {layer} has no multipliers in the resource"
                ));
            }
            multipliers
                .extend_from_slice(&all[(layer * max_ngram) as usize..][..max_ngram as usize]);
        }
        let columns = (max_ngram - 1) * engram.heads;
        let mut primes = Vec::new();
        for _ in 0..engram.layers.len() as u32 * (max_ngram - 1) {
            let mut current = u64::from(engram.vocab) - 1;
            for _ in 0..engram.heads {
                current += 1;
                while !is_prime(current) || primes.contains(&current) {
                    current += 1;
                }
                primes.push(current);
            }
        }
        let mut offsets = Vec::with_capacity(primes.len());
        for (layer, row) in primes.chunks(columns as usize).enumerate() {
            let mut start = 0;
            for &prime in row {
                offsets.push(start);
                start += prime;
            }
            if start != u64::from(engram.rows[layer]) {
                return Err(format!(
                    "Engram layer {layer} buckets sum to {start}, not its {} table rows",
                    engram.rows[layer]
                ));
            }
        }
        let token_map = (0..vocab).map(|_| r.u32()).collect::<Result<Vec<_>, _>>()?;
        if !r.bytes.is_empty() {
            return Err("trailing bytes after the Engram token map".into());
        }
        let mut used = vec![false; compressed as usize];
        for &id in &token_map {
            *used
                .get_mut(id as usize)
                .ok_or("an Engram token maps outside the compressed vocabulary")? = true;
        }
        if used.contains(&false) {
            return Err("an Engram compressed id has no token".into());
        }
        if token_map.get(pad_raw as usize) != Some(&pad) {
            return Err("the Engram pad id is not the pad token's compressed id".into());
        }
        Ok(HashTables {
            engram,
            compressed_vocab: compressed,
            pad: i64::from(pad),
            multipliers,
            primes,
            offsets,
            token_map,
        })
    }

    pub fn raw_vocab(&self) -> u32 {
        self.token_map.len() as u32
    }

    /// Integer constants for the device hash, with exact U64 encodings.
    /// Header: compressed vocabulary size and pad ID, then the token map,
    /// layer multipliers, primes and offsets in their contract order.
    pub fn device_words(&self) -> Vec<u64> {
        [u64::from(self.compressed_vocab), self.pad as u64]
            .into_iter()
            .chain(self.token_map.iter().map(|&id| u64::from(id)))
            .chain(self.multipliers.iter().map(|&value| value as u64))
            .chain(self.primes.iter().copied())
            .chain(self.offsets.iter().copied())
            .collect()
    }

    /// The Engram layout the tables hash for.
    pub fn engram(&self) -> Engram {
        self.engram
    }

    /// Hash ids per position and layer: `(max_ngram - 1) * heads`.
    pub fn columns(&self) -> usize {
        ((self.engram.max_ngram - 1) * self.engram.heads) as usize
    }

    /// The compressed id of raw `token`, or [`DEAD`] where the position
    /// takes no part in an n-gram (`text` false: an image span).
    pub fn compress(&self, token: u32, text: bool) -> Result<i64, Refusal> {
        let id = *self
            .token_map
            .get(token as usize)
            .ok_or(Refusal("an Engram token id is outside the vocabulary"))?;
        Ok(if text { i64::from(id) } else { DEAD })
    }

    /// The hash ids of one position into `out` (`[layer][column]`, each id
    /// a row of that layer's table). `recent[s]` is the compressed id `s`
    /// positions back (`recent[0]` the position's own), [`DEAD`] inside an
    /// image span; fewer than `max_ngram` entries mean the sequence starts
    /// there. Look-back stops at the sequence start and at the first dead
    /// token: those slots and every older one take the pad id. The running
    /// XOR of `id * multiplier` over lookbacks 0..n-1 is the n-gram's hash,
    /// reduced modulo each head's prime, plus the head's offset.
    pub fn position(&self, recent: &[i64], out: &mut [u32]) -> Result<(), Refusal> {
        let n = self.engram.max_ngram as usize;
        let heads = self.engram.heads as usize;
        let columns = self.columns();
        if out.len() != self.engram.layers.len() * columns {
            return Err(Refusal("Engram hash output is not layers * columns"));
        }
        let mut tokens = [self.pad; 8];
        let tokens = &mut tokens[..n];
        let mut blocked = false;
        for (shift, slot) in tokens.iter_mut().enumerate() {
            let source = recent.get(shift).copied();
            blocked |= source.is_none_or(|id| id == DEAD);
            if !blocked {
                let id = source.unwrap_or(self.pad);
                if !(0..i64::from(self.compressed_vocab)).contains(&id) {
                    return Err(Refusal(
                        "an Engram history id is outside the compressed vocabulary",
                    ));
                }
                *slot = id;
            }
        }
        for (layer, out) in out.chunks_mut(columns).enumerate() {
            let multipliers = &self.multipliers[layer * n..][..n];
            let mut rolling = tokens[0] * multipliers[0];
            for i in 1..n {
                rolling ^= tokens[i] * multipliers[i];
                for head in 0..heads {
                    let column = (i - 1) * heads + head;
                    let at = layer * columns + column;
                    let bucket = rolling as u64 % self.primes[at];
                    out[column] = (bucket + self.offsets[at]) as u32;
                }
            }
        }
        Ok(())
    }

    /// The hash ids of every position of a sequence of compressed ids from
    /// its start (`[position][layer][column]`), as the vendor computes them
    /// over a prefill and any later chunks.
    pub fn sequence(&self, compressed: &[i64]) -> Result<Vec<u32>, Refusal> {
        let width = self.engram.layers.len() * self.columns();
        let n = self.engram.max_ngram as usize;
        let mut out = vec![0; compressed.len() * width];
        let mut recent = Vec::with_capacity(n);
        for (p, row) in out.chunks_mut(width).enumerate() {
            recent.clear();
            recent.extend((0..n.min(p + 1)).map(|s| compressed[p - s]));
            self.position(&recent, row)?;
        }
        Ok(out)
    }
}

/// The Engram gate of one Engram layer over `streams` residual streams of
/// `width`, numerical contract `deepseek41.engram-gate.v2`.
///
/// Per row and stream s, with `h` the stream (BF16, widened), `key` the
/// row's key s and `value` its value (F32 slices of the `wkv` output),
/// `w = q_weight[s] * k_weight[s]` (BF16 products, exact in F32) and
/// `n = width`:
///
/// - `r = rsqrt(sum(h^2) / n + 1e-20) * rsqrt(sum(key^2) / n + 1e-20)`;
/// - `dot = sum(h * w * key) * r * n^-0.5`;
/// - `gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))`, zero for a
///   row whose `text` flag is zero when the op is `masked` (an image span);
/// - `output = BF16(h + gate * value)`, rounded once to nearest even.
///
/// The reduction order is the backend's. A row with a closed gate computes
/// `BF16(h + 0 * value)` exactly. Everything is computed as F32; the bound
/// is [`gate_reference`]'s. A non-finite `h`, key or weight makes that
/// stream's open gate NaN. NaN gates propagate to every element of that
/// stream. Values follow IEEE arithmetic: a NaN value produces NaN, and
/// zero times an infinite value produces NaN even for a closed gate.
/// NaN outputs compare by class, without prescribing sign or payload.
/// Other rows and streams are unaffected. No data value causes a trap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngramGate {
    pub width: u32,
    pub streams: u32,
    /// Rows carry a `text` flag; a zero flag closes the row's gate.
    pub masked: bool,
}

/// The gate's RMS epsilon (`norm_eps`).
pub const GATE_EPS: f64 = 1e-20;
/// The floor of `|dot|` before the signed square root.
pub const GATE_CLAMP: f64 = 1e-6;

impl EngramGate {
    /// The gate of `geometry`: its hidden width over its residual streams.
    pub const fn of(geometry: &Geometry, masked: bool) -> EngramGate {
        EngramGate {
            width: geometry.shape.hidden,
            streams: geometry.shape.hc.streams,
            masked,
        }
    }

    pub fn validate(&self, need: Need) -> Result<(), Refusal> {
        if self.width == 0 || self.streams == 0 || need.capacity == 0 {
            return Err(Refusal("an Engram gate has rows, streams and width"));
        }
        Ok(())
    }

    /// Elements of one row's streams.
    pub fn row_elements(&self) -> usize {
        self.width as usize * self.streams as usize
    }

    /// Elements of one row's `kv`: a key per stream, then the value.
    pub fn kv_elements(&self) -> usize {
        self.width as usize * (self.streams as usize + 1)
    }
}

kernel_core::scratch_record! {
    /// An Engram gate's buffers, every view 16-byte aligned: the BF16
    /// streams `x` (`[row][stream][width]`), the F32 `kv` rows
    /// (`[row][(streams + 1) * width]`: key per stream, then the value),
    /// the BF16 `q_weight` and `k_weight` (`[stream][width]`), the U8 `text`
    /// flag per row (bound when the op is masked), the device U32 count of
    /// `live_rows`, and the BF16 `output` streams, which may be `x` itself.
    /// Rows from `live_rows` on are not written.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EngramGateBuffers {
        pub x: value,
        pub kv: value,
        pub q_weight: value,
        pub k_weight: value,
        pub text: option,
        pub live_rows: value,
        pub output: value,
    }
}

/// Host inputs of one gate row: `x` and `kv` of one row, the weights, and
/// whether its gate is open.
#[derive(Clone, Copy, Debug)]
pub struct GateRow<'a> {
    pub x: &'a [u16],
    pub kv: &'a [f32],
    pub q_weight: &'a [u16],
    pub k_weight: &'a [u16],
    pub open: bool,
}

/// The gate's exact output and the largest distance a kernel output may
/// have from it, per element of a row's streams.
#[derive(Clone, Debug, PartialEq)]
pub struct GateReference {
    pub values: Vec<f64>,
    pub bounds: Vec<f64>,
    /// Each stream's exact gate.
    pub gates: Vec<f64>,
}

fn gate_of(dot: f64) -> f64 {
    if dot.is_nan() {
        return f64::NAN;
    }
    let z = dot.abs().max(GATE_CLAMP).sqrt().copysign(dot);
    1.0 / (1.0 + (-z).exp())
}

fn half_bf16_ulp(value: f64) -> f64 {
    let magnitude = value.abs().max(f64::from(f32::MIN_POSITIVE));
    2_f64.powi(magnitude.log2().floor() as i32 - 8)
}

/// The f64 oracle over the original F32 keys and values and widened BF16
/// streams and weights. Let `u = 2^-24`, `eta = 2^-150`, `a = 2^-22 n`,
/// `b = a + u (1 + a)`, `S = sum|h w key|`, and `T = sum|key|`.
/// The F32 key product and square can round, unlike BF16 products. Composing
/// one product rounding with the reduction allowance gives `b` in place of
/// v1's `a`. The dot error is `(b S + (T+n) eta (1+a)) r n^-0.5` plus
/// `|dot| (b + 2^-19 + 2 eta (1+a)/eps)` for normalization and scaling.
/// The eta terms cover gradual underflow in products, sums and mean squares.
/// The gate error is its largest change over that interval (including the
/// signed clamp's jump at zero), plus 2^-20 for its F32 evaluation.
/// Output error is `|value| gate_error + 2u (|h|+|value|) + eta`, then
/// half a BF16 ulp at the largest possible output magnitude. There is no
/// BF16 rounding allowance for either key or value. This arithmetic bound
/// assumes finite F32 intermediates; IEEE non-finite results compare by
/// class and infinity sign. A closed gate has zero bound and compares the
/// exact BF16 result of `h + 0 * value`, including signed zero.
pub fn gate_reference(op: &EngramGate, row: GateRow<'_>) -> Result<GateReference, Refusal> {
    let (n, streams) = (op.width as usize, op.streams as usize);
    if row.x.len() != n * streams
        || row.kv.len() != n * (streams + 1)
        || row.q_weight.len() != n * streams
        || row.k_weight.len() != n * streams
    {
        return Err(Refusal("Engram gate oracle inputs do not match the op"));
    }
    let f = |bits: u16| f64::from(bf16_to_f32(bits));
    let value = &row.kv[n * streams..];
    let mut out = GateReference {
        values: Vec::with_capacity(n * streams),
        bounds: Vec::with_capacity(n * streams),
        gates: Vec::with_capacity(streams),
    };
    let scale = (n as f64).sqrt().recip();
    for s in 0..streams {
        let (h, key) = (&row.x[s * n..][..n], &row.kv[s * n..][..n]);
        let (q, k) = (&row.q_weight[s * n..][..n], &row.k_weight[s * n..][..n]);
        let (mut hh, mut kk, mut dot) = (
            NeumaierSum::default(),
            NeumaierSum::default(),
            NeumaierSum::default(),
        );
        let mut magnitude = 0.0;
        let mut key_magnitude = 0.0;
        let finite = h.iter().chain(q).chain(k).all(|&v| f(v).is_finite())
            && key.iter().all(|v| v.is_finite());
        for i in 0..n {
            let term = f(h[i]) * (f(q[i]) * f(k[i])) * f64::from(key[i]);
            hh.add(f(h[i]) * f(h[i]));
            kk.add(f64::from(key[i]) * f64::from(key[i]));
            dot.add(term);
            magnitude += term.abs();
            key_magnitude += f64::from(key[i]).abs();
        }
        let r = (hh.value() / n as f64 + GATE_EPS).sqrt().recip()
            * (kk.value() / n as f64 + GATE_EPS).sqrt().recip();
        let exact = dot.value() * r * scale;
        let a = accumulation_bound(n, 1.0);
        let b = a + 2_f64.powi(-24) * (1.0 + a);
        let eta = 2_f64.powi(-150);
        let error = (b * magnitude + (key_magnitude + n as f64) * eta * (1.0 + a)) * r * scale
            + exact.abs() * (b + 2_f64.powi(-19) + 2.0 * eta * (1.0 + a) / GATE_EPS);
        let (gate, gate_error) = if row.open && !finite {
            (f64::NAN, 0.0)
        } else if row.open {
            let g = gate_of(exact);
            let spread = (gate_of(exact + error) - g)
                .abs()
                .max((gate_of(exact - error) - g).abs());
            (g, spread + 2_f64.powi(-20))
        } else {
            (0.0, 0.0)
        };
        out.gates.push(gate);
        for i in 0..n {
            let (x, v) = (f(h[i]), f64::from(value[i]));
            let y = x + gate * v;
            if row.open && y.is_finite() {
                let e = v.abs() * gate_error + 2_f64.powi(-23) * (x.abs() + v.abs()) + eta;
                out.values.push(y);
                out.bounds.push(e + half_bf16_ulp(y.abs() + e));
            } else {
                // F32 `x + 0 * v`: -0 + +0 is +0, as the kernel and the vendor add.
                out.values.push(y);
                out.bounds.push(0.0);
            }
        }
    }
    Ok(out)
}

/// The first element of a row whose BF16 output `actual` is farther from
/// the reference than its bound, or (for a closed gate) not bit-identical
/// to `BF16(h + 0 * value)`.
pub fn check_gate_row(actual: &[u16], reference: &GateReference) -> Result<(), usize> {
    if actual.len() != reference.values.len() {
        return Err(actual.len().min(reference.values.len()));
    }
    for (i, (&bits, (&r, &bound))) in actual
        .iter()
        .zip(reference.values.iter().zip(&reference.bounds))
        .enumerate()
    {
        let y = f64::from(bf16_to_f32(bits));
        let ok = if r.is_nan() {
            y.is_nan()
        } else if r.is_infinite() {
            y == r
        } else if bound == 0.0 {
            kernel_core::codec::f32_to_bf16(r as f32) == bits
        } else {
            y.is_finite() && (y - r).abs() <= bound
        };
        if !ok {
            return Err(i);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::codec::f32_to_bf16;

    fn vendor_cases() -> serde_json::Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../fixtures/deepseek41/engram-vendor-hashes.json"
        );
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn flash() -> HashTables {
        HashTables::of(&crate::geometry::FLASH).unwrap()
    }

    fn ids(value: &serde_json::Value) -> Vec<u32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    }

    /// Every case's hashes under `tables` equal the vendor's.
    fn check_cases(tables: &HashTables, cases: &serde_json::Value) {
        let width = tables.engram().layers.len() * tables.columns();
        for case in cases.as_array().unwrap() {
            let tokens = ids(&case["ids"]);
            let text: Vec<bool> = match case["text"].as_array() {
                Some(t) => t.iter().map(|v| v.as_bool().unwrap()).collect(),
                None => vec![true; tokens.len()],
            };
            let compressed: Vec<i64> = tokens
                .iter()
                .zip(&text)
                .map(|(&id, &t)| tables.compress(id, t).unwrap())
                .collect();
            // The vendor hashed in two chunks; the hash of a position
            // depends only on its history, so one pass gives the same ids.
            let hashes = tables.sequence(&compressed).unwrap();
            assert_eq!(hashes[..width], ids(&case["first_row"])[..]);
            let bytes: Vec<u8> = hashes
                .iter()
                .flat_map(|&h| i64::from(h).to_le_bytes())
                .collect();
            let digest: String = Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(digest, case["sha256"].as_str().unwrap(), "{case}");
        }
    }

    #[test]
    fn the_resource_is_the_vendors_map_multipliers_and_primes() {
        let tables = flash();
        assert_eq!(tables.columns(), 24);
        assert_eq!(tables.token_map.len(), 129_280);
        assert_eq!(tables.token_map.iter().max(), Some(&99_091));
        // numpy's PCG64 seeded 10007 * layer id, times two plus one.
        assert_eq!(
            tables.multipliers,
            [
                76_632_096_046_245,
                4_839_876_093_313,
                35_959_672_319_349,
                73_987_337_458_391,
                67_716_810_739_261,
                51_510_806_800_915,
                30_921_347_202_721,
                82_619_226_485_591,
            ]
        );
        assert_eq!(tables.primes[0], 16_000_057);
        assert_eq!(tables.primes[47], 16_000_889);
        let doc = vendor_cases();
        assert_eq!(doc["resource_sha256"], RESOURCE_SHA256);
    }

    #[test]
    fn hashes_equal_the_vendors_ids() {
        let doc = vendor_cases();
        assert_eq!(doc["cases"].as_array().unwrap().len(), 12);
        check_cases(&flash(), &doc["cases"]);
    }

    #[test]
    fn other_layer_ids_and_bucket_bases_hash_as_the_vendor_does() {
        let doc = vendor_cases();
        let layouts = doc["layouts"].as_object().unwrap();
        assert_eq!(layouts.len(), 2);
        for layout in layouts.values() {
            let leak = |v: &serde_json::Value| -> &'static [u32] { Box::leak(ids(v).into()) };
            let engram = Engram {
                layers: leak(&layout["layer_ids"]),
                rows: leak(&layout["table_rows"]),
                vocab: layout["bucket_base"].as_u64().unwrap() as u32,
                ..crate::geometry::FLASH.shape.engram
            };
            let tables = HashTables::new(RESOURCE, 129_280, engram).unwrap();
            check_cases(&tables, &layout["cases"]);
            // Rows that are not the draw's sum refuse.
            let rows: &'static [u32] = Box::leak(vec![engram.rows[0] + 1, engram.rows[1]].into());
            let wrong = Engram { rows, ..engram };
            assert!(
                HashTables::new(RESOURCE, 129_280, wrong)
                    .unwrap_err()
                    .contains("table rows")
            );
        }
    }

    #[test]
    fn the_resource_refuses_other_bytes_and_layouts() {
        let engram = crate::geometry::FLASH.shape.engram;
        let mut bytes = RESOURCE.to_vec();
        bytes[100] ^= 1;
        assert!(
            HashTables::new(&bytes, 129_280, engram)
                .unwrap_err()
                .contains("sha256")
        );
        let other = Engram {
            max_ngram: 3,
            ..engram
        };
        assert!(
            HashTables::new(RESOURCE, 129_280, other)
                .unwrap_err()
                .contains("header")
        );
        let late = Engram {
            layers: &[1, 64],
            ..engram
        };
        assert!(
            HashTables::new(RESOURCE, 129_280, late)
                .unwrap_err()
                .contains("no multipliers")
        );
        assert!(!is_prime(16_000_000) && is_prime(16_000_057) && !is_prime(1));
    }

    #[test]
    fn look_back_stops_at_the_start_and_at_dead_tokens() {
        let tables = flash();
        let mut a = vec![0; 48];
        let mut b = vec![0; 48];
        // A dead token one back equals the sequence starting here.
        tables.position(&[500, DEAD, 7, 9], &mut a).unwrap();
        tables.position(&[500], &mut b).unwrap();
        assert_eq!(a, b);
        // Older history past the blocked slot is ignored.
        tables.position(&[500, 12, DEAD, 9], &mut a).unwrap();
        tables.position(&[500, 12], &mut b).unwrap();
        assert_eq!(a, b);
        // The pad token as history hashes like a missing slot.
        tables.position(&[500, 2], &mut a).unwrap();
        tables.position(&[500], &mut b).unwrap();
        assert_eq!(a, b, "a missing slot takes the pad id");
        // Each id lies in its column's bucket range.
        for (column, &id) in a[..24].iter().enumerate() {
            let start = tables.offsets[column];
            assert!((start..start + tables.primes[column]).contains(&u64::from(id)));
        }
        assert!(tables.position(&[99_092], &mut a).is_err());
        assert_eq!(tables.compress(129_279, false).unwrap(), DEAD);
        assert!(tables.compress(129_280, true).is_err());
    }

    fn bf(x: f32) -> u16 {
        f32_to_bf16(x)
    }

    fn row(seed: u32, n: usize, streams: usize) -> (Vec<u16>, Vec<f32>, Vec<u16>, Vec<u16>) {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
        };
        let x = (0..n * streams).map(|_| bf(next() * 4.0)).collect();
        let kv = (0..n * (streams + 1)).map(|_| next()).collect();
        let q = (0..n * streams).map(|_| bf(1.0 + next() * 0.25)).collect();
        let k = (0..n * streams).map(|_| bf(1.0 + next() * 0.25)).collect();
        (x, kv, q, k)
    }

    /// An F32 evaluation in sequential order, as a simple kernel would;
    /// `signed_sqrt` false drops the square root (a wrong formula).
    fn f32_gate(op: &EngramGate, r: GateRow<'_>, signed_sqrt: bool) -> Vec<u16> {
        let (n, streams) = (op.width as usize, op.streams as usize);
        let f = |b: u16| bf16_to_f32(b);
        let mut out = Vec::new();
        for s in 0..streams {
            let (mut hh, mut kk, mut d) = (0f32, 0f32, 0f32);
            for i in 0..n {
                let h = f(r.x[s * n + i]);
                let key = r.kv[s * n + i];
                hh += h * h;
                kk += key * key;
                d += h * (f(r.q_weight[s * n + i]) * f(r.k_weight[s * n + i])) * key;
            }
            let rr =
                (hh / n as f32 + 1e-20).sqrt().recip() * (kk / n as f32 + 1e-20).sqrt().recip();
            let dot = d * rr * (n as f32).powf(-0.5);
            let z = if signed_sqrt {
                dot.abs().max(1e-6).sqrt().copysign(dot)
            } else {
                dot
            };
            let g = if r.open {
                1.0 / (1.0 + (-z).exp())
            } else {
                0.0
            };
            for i in 0..n {
                let v = r.kv[streams * n + i];
                out.push(bf(g.mul_add(v, f(r.x[s * n + i]))));
            }
        }
        out
    }

    #[test]
    fn an_f32_gate_meets_the_bound_and_a_wrong_formula_does_not() {
        let op = EngramGate::of(&crate::geometry::FLASH, true);
        let (n, streams) = (5120, 4);
        for seed in 0..4 {
            let (x, kv, q, k) = row(seed, n, streams);
            for open in [true, false] {
                let r = GateRow {
                    x: &x,
                    kv: &kv,
                    q_weight: &q,
                    k_weight: &k,
                    open,
                };
                let reference = gate_reference(&op, r).unwrap();
                assert_eq!(check_gate_row(&f32_gate(&op, r, true), &reference), Ok(()));
                if open {
                    // Without the signed square root the gate moves far
                    // past the bound.
                    let wrong = f32_gate(&op, r, false);
                    assert!(check_gate_row(&wrong, &reference).is_err());
                } else {
                    assert!(reference.bounds.iter().all(|&b| b == 0.0));
                }
            }
        }
    }

    #[test]
    fn rounding_f32_values_before_the_add_is_rejected() {
        let op = EngramGate::of(&crate::geometry::FLASH, false);
        let x = vec![bf(-1.0); op.row_elements()];
        let w = vec![bf(1.0); op.row_elements()];
        let mut kv = vec![-1.001; op.kv_elements()];
        kv[op.row_elements()..].fill(1.002);
        let row = GateRow {
            x: &x,
            kv: &kv,
            q_weight: &w,
            k_weight: &w,
            open: true,
        };
        let reference = gate_reference(&op, row).unwrap();
        assert_eq!(
            check_gate_row(&f32_gate(&op, row, true), &reference),
            Ok(())
        );
        let mut rounded = kv.clone();
        rounded[op.row_elements()..]
            .iter_mut()
            .for_each(|v| *v = bf16_to_f32(bf(*v)));
        let wrong = f32_gate(
            &op,
            GateRow {
                kv: &rounded,
                ..row
            },
            true,
        );
        assert!(check_gate_row(&wrong, &reference).is_err());
    }

    #[test]
    fn non_finite_gate_inputs_and_values_have_local_ieee_results() {
        let op = EngramGate {
            width: 2,
            streams: 2,
            masked: true,
        };
        let x = [bf(1.0); 4];
        let w = [bf(1.0); 4];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for operand in 0..4 {
                let mut kv = [1.001; 6];
                let mut h = x;
                let mut q = w;
                let mut k = w;
                match operand {
                    0 => kv[0] = bad,
                    1 => h[0] = bf(bad),
                    2 => q[0] = bf(bad),
                    _ => k[0] = bf(bad),
                }
                let r = GateRow {
                    x: &h,
                    kv: &kv,
                    q_weight: &q,
                    k_weight: &k,
                    open: true,
                };
                let reference = gate_reference(&op, r).unwrap();
                assert!(reference.values[..2].iter().all(|v| v.is_nan()));
                assert!(reference.values[2..].iter().all(|v| v.is_finite()));
                let closed = gate_reference(&op, GateRow { open: false, ..r }).unwrap();
                assert!(closed.bounds.iter().all(|&v| v == 0.0));
            }
            let kv = [1.001, 1.001, 1.001, 1.001, bad, 2.001];
            for open in [true, false] {
                let r = GateRow {
                    x: &x,
                    kv: &kv,
                    q_weight: &w,
                    k_weight: &w,
                    open,
                };
                let reference = gate_reference(&op, r).unwrap();
                assert_eq!(reference.values[0].is_nan(), !open || bad.is_nan());
                assert_eq!(reference.values[0].is_infinite(), open && bad.is_infinite());
                assert!(reference.values[1].is_finite());
                let got: Vec<_> = reference.values.iter().map(|&v| bf(v as f32)).collect();
                assert_eq!(check_gate_row(&got, &reference), Ok(()));
            }
        }
    }

    #[test]
    fn nan_outputs_are_checked_by_class_without_accepting_finite_values() {
        let reference = GateReference {
            values: vec![f64::NAN],
            bounds: vec![0.0],
            gates: vec![f64::NAN],
        };
        assert_eq!(check_gate_row(&[0x7fff], &reference), Ok(()));
        assert_eq!(check_gate_row(&[0xffc1], &reference), Ok(()));
        assert_eq!(check_gate_row(&[bf(0.0)], &reference), Err(0));
        assert_eq!(check_gate_row(&[bf(f32::INFINITY)], &reference), Err(0));
    }

    #[test]
    fn a_closed_gate_keeps_the_stream_bits_but_not_negative_zero() {
        let op = EngramGate {
            width: 2,
            streams: 1,
            masked: true,
        };
        let x = [0x8000, bf(3.5)];
        let kv = [1.0, 1.0, 2.0, -2.0];
        let w = [bf(1.0); 2];
        let r = GateRow {
            x: &x,
            kv: &kv,
            q_weight: &w,
            k_weight: &w,
            open: false,
        };
        let reference = gate_reference(&op, r).unwrap();
        // -0 + 0 * 2 is +0; 3.5 + 0 * -2 is 3.5.
        assert_eq!(check_gate_row(&[0x0000, bf(3.5)], &reference), Ok(()));
        assert_eq!(check_gate_row(&[0x8000, bf(3.5)], &reference), Err(0));
    }
}
