//! Engram: the n-gram hash that picks each position's table rows, and the
//! gate that writes the looked-up value into the residual streams
//! (`inference/engram.py`, `inference/model.py:296-366`).
//!
//! The hash runs on the host: its ids index the host-mapped tables, which
//! the host gathers into staging rows, so it is a contract function over
//! the package's hash resource ([`HashTables`]), exact in integers. The gate
//! is a device op ([`EngramGate`]) with an f64 oracle and bound.

use kernel_core::backend::Need;
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::Refusal;
use kernel_core::oracle::{NeumaierSum, accumulation_bound};
use sha2::{Digest, Sha256};

/// The hash resource the package ships, generated offline from the pinned
/// vendor code by `models/deepseek41/engram_resource.py`.
pub const RESOURCE: &[u8] = include_bytes!("../../resources/engram-hash.v1.bin");

/// SHA-256 of [`RESOURCE`]. [`HashTables::parse`] refuses other bytes.
pub const RESOURCE_SHA256: &str =
    "cf13df82c09decf7e4cc51e90d86e71a8bd266f254dc833c67a85b544213469c";

/// The package component format of [`RESOURCE`].
pub const RESOURCE_FORMAT: &str = "deepseek41.engram-hash.v1";

const MAGIC: &[u8; 8] = b"DS41ENG1";

/// A position's compressed token that takes no part in an n-gram: inside an
/// image span (the vendor's `NgramHashState.DEAD`).
pub const DEAD: i64 = -1;

/// The flash geometry's hash: what [`HashTables::parse`] checks the
/// resource against.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HashGeometry {
    pub vocab: u32,
    pub compressed_vocab: u32,
    pub pad_id: u32,
    /// The value each (n-gram size, head) bucket prime search starts above
    /// (`engram_vocab_size`).
    pub bucket_base: u32,
    pub layer_ids: [u32; 2],
    pub max_ngram: u32,
    pub heads: u32,
    /// Table rows of each Engram layer, the sum of its bucket primes.
    pub table_rows: [u64; 2],
}

/// DeepSeek V4.1 Flash's hash geometry (`inference/config.json`).
pub const FLASH: HashGeometry = HashGeometry {
    vocab: 129_280,
    compressed_vocab: 99_092,
    pad_id: 2,
    bucket_base: 16_000_000,
    layer_ids: [1, 14],
    max_ngram: 4,
    heads: 8,
    table_rows: [384_006_168, 384_016_682],
};

/// The parsed hash resource: the compressed token map, one odd multiplier
/// per (layer, lookback), and the bucket primes and offsets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HashTables {
    geometry: HashGeometry,
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
    /// The resource the package ships, checked against [`FLASH`].
    pub fn flash() -> Result<HashTables, String> {
        HashTables::parse(RESOURCE, FLASH)
    }

    /// Parses `bytes` and checks them against `geometry` and
    /// [`RESOURCE_SHA256`]: every token maps into the compressed vocabulary
    /// and every compressed id is used; multipliers are odd and positive
    /// and keep `id * multiplier` within i64; the primes are the vendor's
    /// draw (each the least unused prime above the bucket base, in layer,
    /// n-gram size and head order) and sum to each layer's table rows.
    pub fn parse(bytes: &[u8], geometry: HashGeometry) -> Result<HashTables, String> {
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
        let header = [
            r.u32()?,
            r.u32()?,
            r.u32()?,
            r.u32()?,
            r.u32()?,
            r.u32()?,
            r.u32()?,
            r.u32()?,
        ];
        let [
            vocab,
            compressed,
            pad_raw,
            pad,
            base,
            layers,
            max_ngram,
            heads,
        ] = header;
        if [vocab, compressed, pad_raw, base, layers, max_ngram, heads]
            != [
                geometry.vocab,
                geometry.compressed_vocab,
                geometry.pad_id,
                geometry.bucket_base,
                geometry.layer_ids.len() as u32,
                geometry.max_ngram,
                geometry.heads,
            ]
        {
            return Err(format!(
                "Engram hash resource header {header:?} does not match the geometry"
            ));
        }
        let layer_ids = [r.u32()?, r.u32()?];
        if layer_ids != geometry.layer_ids {
            return Err("Engram hash resource layer ids differ from the geometry".into());
        }
        let multipliers = (0..layers * max_ngram)
            .map(|_| r.i64())
            .collect::<Result<Vec<_>, _>>()?;
        let bound = i64::MAX / i64::from(compressed) / 2;
        if let Some(m) = multipliers
            .iter()
            .find(|&&m| m <= 0 || m % 2 == 0 || m > 2 * bound + 1)
        {
            return Err(format!("Engram multiplier {m} is not odd within its bound"));
        }
        let columns = (max_ngram - 1) * heads;
        let primes = (0..layers * columns)
            .map(|_| r.u32().map(u64::from))
            .collect::<Result<Vec<_>, _>>()?;
        let mut drawn = Vec::with_capacity(primes.len());
        for _ in 0..layers * (max_ngram - 1) {
            let mut current = u64::from(base) - 1;
            for _ in 0..heads {
                current += 1;
                while !is_prime(current) || drawn.contains(&current) {
                    current += 1;
                }
                drawn.push(current);
            }
        }
        if primes != drawn {
            return Err("Engram bucket primes are not the vendor's draw".into());
        }
        let mut offsets = Vec::with_capacity(primes.len());
        for (layer, row) in primes.chunks(columns as usize).enumerate() {
            let mut start = 0;
            for &prime in row {
                offsets.push(start);
                start += prime;
            }
            if start != geometry.table_rows[layer] || start > u64::from(u32::MAX) {
                return Err(format!(
                    "Engram layer {layer} buckets sum to {start}, not its table rows"
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
        if token_map[pad_raw as usize] != pad {
            return Err("the Engram pad id is not the pad token's compressed id".into());
        }
        Ok(HashTables {
            geometry,
            pad: i64::from(pad),
            multipliers,
            primes,
            offsets,
            token_map,
        })
    }

    pub fn geometry(&self) -> HashGeometry {
        self.geometry
    }

    /// Hash ids per position and layer: `(max_ngram - 1) * heads`.
    pub fn columns(&self) -> usize {
        ((self.geometry.max_ngram - 1) * self.geometry.heads) as usize
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
        let n = self.geometry.max_ngram as usize;
        let heads = self.geometry.heads as usize;
        let columns = self.columns();
        if out.len() != self.geometry.layer_ids.len() * columns {
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
                if !(0..i64::from(self.geometry.compressed_vocab)).contains(&id) {
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
        let width = self.geometry.layer_ids.len() * self.columns();
        let n = self.geometry.max_ngram as usize;
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
/// `width`, numerical contract `deepseek41.engram-gate.v1`.
///
/// Per row and stream s, with `h` the stream (BF16, widened), `key` the
/// row's key s and `value` its value (BF16 halves of the `wkv` output),
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
/// is [`gate_reference`]'s.
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
    /// The flash geometry's gate: 4 streams of 5120.
    pub const FLASH: EngramGate = EngramGate {
        width: 5120,
        streams: 4,
        masked: true,
    };

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
    /// streams `x` (`[row][stream][width]`), the BF16 `kv` rows
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
    pub kv: &'a [u16],
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
    let z = dot.abs().max(GATE_CLAMP).sqrt().copysign(dot);
    1.0 / (1.0 + (-z).exp())
}

fn half_bf16_ulp(value: f64) -> f64 {
    let magnitude = value.abs().max(f64::from(f32::MIN_POSITIVE));
    2_f64.powi(magnitude.log2().floor() as i32 - 8)
}

/// The f64 oracle of one row and its bound. The gate is evaluated exactly
/// in f64. Its error follows from the accumulation rule (2^-22 K S) for each
/// F32 reduction: the dot's error is `2^-22 n sum|h w key| r n^-0.5` plus a
/// relative `2^-22 n + 2^-19` for the two mean squares, reciprocal square
/// roots and products. The gate error is the gate's largest change over
/// that interval of `dot` (it is monotone) plus 2^-20 for evaluating it. An
/// element may then differ from `h + gate * value` by `|value|` times the
/// gate error, an F32 rounding of the sum, and half a BF16 ulp of the
/// output. A closed gate is exact: bound zero.
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
        for i in 0..n {
            let term = f(h[i]) * (f(q[i]) * f(k[i])) * f(key[i]);
            hh.add(f(h[i]) * f(h[i]));
            kk.add(f(key[i]) * f(key[i]));
            dot.add(term);
            magnitude += term.abs();
        }
        let r = (hh.value() / n as f64 + GATE_EPS).sqrt().recip()
            * (kk.value() / n as f64 + GATE_EPS).sqrt().recip();
        let exact = dot.value() * r * scale;
        let error = accumulation_bound(n, magnitude) * r * scale
            + exact.abs() * (accumulation_bound(n, 1.0) + 2_f64.powi(-19));
        let (gate, gate_error) = if row.open {
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
            let (x, v) = (f(h[i]), f(value[i]));
            if row.open {
                let y = x + gate * v;
                let e = v.abs() * gate_error + 2_f64.powi(-23) * (x.abs() + v.abs());
                out.values.push(y);
                out.bounds.push(e + half_bf16_ulp(y.abs() + e));
            } else {
                // F32 `x + 0 * v`: -0 + +0 is +0, as the kernel and the vendor add.
                out.values.push(x + 0.0 * v);
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
        let ok = if bound == 0.0 {
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

    #[test]
    fn the_resource_is_the_vendors_map_multipliers_and_primes() {
        let tables = HashTables::flash().unwrap();
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
        let tables = HashTables::flash().unwrap();
        let doc = vendor_cases();
        let cases = doc["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 12);
        for case in cases {
            let ids: Vec<u32> = case["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let text: Vec<bool> = match case["text"].as_array() {
                Some(t) => t.iter().map(|v| v.as_bool().unwrap()).collect(),
                None => vec![true; ids.len()],
            };
            let compressed: Vec<i64> = ids
                .iter()
                .zip(&text)
                .map(|(&id, &t)| tables.compress(id, t).unwrap())
                .collect();
            // The vendor hashed in two chunks; the hash of a position
            // depends only on its history, so one pass gives the same ids.
            let hashes = tables.sequence(&compressed).unwrap();
            let first: Vec<u32> = case["first_row"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(&hashes[..48], &first[..]);
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
    fn the_resource_refuses_other_bytes() {
        let mut bytes = RESOURCE.to_vec();
        bytes[100] ^= 1;
        assert!(
            HashTables::parse(&bytes, FLASH)
                .unwrap_err()
                .contains("sha256")
        );
        let geometry = HashGeometry { heads: 4, ..FLASH };
        assert!(
            HashTables::parse(RESOURCE, geometry)
                .unwrap_err()
                .contains("header")
        );
        assert!(!is_prime(16_000_000) && is_prime(16_000_057) && !is_prime(1));
    }

    #[test]
    fn look_back_stops_at_the_start_and_at_dead_tokens() {
        let tables = HashTables::flash().unwrap();
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

    fn row(seed: u32, n: usize, streams: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
        };
        let x = (0..n * streams).map(|_| bf(next() * 4.0)).collect();
        let kv = (0..n * (streams + 1)).map(|_| bf(next())).collect();
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
                let key = f(r.kv[s * n + i]);
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
                let v = f(r.kv[streams * n + i]);
                out.push(bf(g.mul_add(v, f(r.x[s * n + i]))));
            }
        }
        out
    }

    #[test]
    fn an_f32_gate_meets_the_bound_and_a_wrong_formula_does_not() {
        let op = EngramGate::FLASH;
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
    fn a_closed_gate_keeps_the_stream_bits_but_not_negative_zero() {
        let op = EngramGate {
            width: 2,
            streams: 1,
            masked: true,
        };
        let x = [0x8000, bf(3.5)];
        let kv = [bf(1.0), bf(1.0), bf(2.0), bf(-2.0)];
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
