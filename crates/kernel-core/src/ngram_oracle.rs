//! Reference arithmetic for the hashed n-gram embedding and the per-layer
//! embedding (PLE) gate. Hashing is integer-exact; the gate is f64.

use crate::codec::SmallFloat;
use crate::contract::{NgramQuery, Refusal, Validate};
use crate::oracle::OracleOutput;
use crate::recurrent_oracle::sigmoid;

const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const LAYER_PRIME: u64 = 10_007;

pub fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(SPLITMIX_GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

/// Hash tables of one PLE layer: one multiplier per n-gram position, and the
/// prime size and row offset of every head.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramTables {
    pub multipliers: Vec<u64>,
    pub sizes: Vec<u64>,
    pub offsets: Vec<u64>,
    /// Table rows after padding to `divisible_by`.
    pub rows: u64,
}

/// The HF derivation: multiplier i is `2 * (splitmix64(seed + 10007 *
/// layer_index + GAMMA * (i + 1)) % half_bound) + 1` with `half_bound =
/// (i64::MAX / vocab) / 2`, which keeps every product below 2^63; global
/// head g has the (g + 1)-th prime at or above `base`.
pub fn ngram_tables(
    vocab: u64,
    orders: usize,
    heads_per_order: usize,
    base: u64,
    seed: u64,
    layer_index: u64,
    divisible_by: u64,
) -> Result<NgramTables, Refusal> {
    if vocab == 0 || orders < 2 || heads_per_order == 0 || base < 2 || divisible_by == 0 {
        return Err(Refusal("invalid n-gram table parameters"));
    }
    let half_bound = ((i64::MAX as u64) / vocab / 2).max(1);
    let layer_seed = seed.wrapping_add(LAYER_PRIME.wrapping_mul(layer_index));
    let multipliers = (0..orders as u64)
        .map(|i| {
            let value = layer_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(i + 1));
            2 * (splitmix64(value) % half_bound) + 1
        })
        .collect();
    let heads = (orders - 1) * heads_per_order;
    let mut prime = base - 1;
    for _ in 0..layer_index * heads as u64 {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    let mut sizes = Vec::with_capacity(heads);
    let mut offsets = Vec::with_capacity(heads);
    let mut total = 0;
    for _ in 0..heads {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
        sizes.push(prime);
        offsets.push(total);
        total += prime;
    }
    Ok(NgramTables {
        multipliers,
        sizes,
        offsets,
        rows: total.div_ceil(divisible_by) * divisible_by,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramRows {
    /// Global table row per token and head, [tokens, heads].
    pub rows: Vec<u64>,
    /// The last `orders - 1` tokens, oldest first: the sequence's new history.
    pub history: Vec<u32>,
}

/// Table rows for `tokens` after `history` (the previous `orders - 1` ids,
/// oldest first; all `eos` at request start). A predecessor reads as `eos`
/// once the walk back from the token meets an `eos`.
pub fn ngram_rows(
    tables: &NgramTables,
    heads_per_order: usize,
    history: &[u32],
    tokens: &[u32],
    eos: u32,
) -> Result<NgramRows, Refusal> {
    let orders = tables.multipliers.len();
    let heads = (orders - 1) * heads_per_order;
    if orders < 2
        || history.len() != orders - 1
        || tables.sizes.len() != heads
        || tables.offsets.len() != heads
        || tokens.is_empty()
    {
        return Err(Refusal("n-gram history or tables do not match the orders"));
    }
    let window = [history, tokens].concat();
    let mut rows = Vec::with_capacity(tokens.len() * heads);
    for index in history.len()..window.len() {
        let mut shifted = vec![u64::from(window[index])];
        let mut crossed = false;
        for shift in 1..orders {
            let token = window[index - shift];
            let value = if crossed { eos } else { token };
            crossed |= value == eos;
            shifted.push(u64::from(value));
        }
        for order in 2..=orders {
            let mixed = (0..order).fold(0_u64, |acc, position| {
                acc ^ shifted[position].wrapping_mul(tables.multipliers[position])
            });
            for head in (order - 2) * heads_per_order..(order - 1) * heads_per_order {
                rows.push(mixed % tables.sizes[head] + tables.offsets[head]);
            }
        }
    }
    Ok(NgramRows {
        rows,
        history: window[window.len() - history.len()..].to_vec(),
    })
}

/// Gather this rank's heads. `table` holds rows `[table_start, ..)` of the
/// global table as E4M3 codes, `head_width` per row. The output is
/// [tokens, local_heads * head_width], each code times `scale`.
pub fn ngram_gather(
    query: NgramQuery,
    rows: &[u64],
    rank: usize,
    table: &[u8],
    table_start: u64,
    scale: f64,
) -> Result<Vec<f64>, Refusal> {
    query.validate()?;
    let NgramQuery::HashGather {
        orders,
        heads_per_order,
        head_width,
        local_heads,
        capacity,
    } = query
    else {
        return Err(Refusal("expected n-gram gather query"));
    };
    let heads = ((orders - 1) * heads_per_order) as usize;
    let (width, local) = (head_width as usize, local_heads as usize);
    if rows.is_empty()
        || !rows.len().is_multiple_of(heads)
        || rows.len() / heads > capacity as usize
        || (rank + 1) * local > heads
        || !table.len().is_multiple_of(width)
        || !scale.is_finite()
    {
        return Err(Refusal("n-gram gather operand shape mismatch"));
    }
    let table_rows = (table.len() / width) as u64;
    let mut output = Vec::with_capacity(rows.len() / heads * local * width);
    for token in rows.chunks_exact(heads) {
        for &row in &token[rank * local..(rank + 1) * local] {
            let local_row = row
                .checked_sub(table_start)
                .filter(|&r| r < table_rows)
                .ok_or(Refusal("n-gram row outside this rank's table"))?
                as usize;
            output.extend(
                table[local_row * width..(local_row + 1) * width]
                    .iter()
                    .map(|&code| SmallFloat::E4M3.decode_f64(code) * scale),
            );
        }
    }
    Ok(output)
}

/// key and query are grouped-normalized, [tokens, streams, width]; value is
/// [tokens, width]. The output is [tokens, streams, width].
pub fn ple_gate(
    query: NgramQuery,
    key: &[f64],
    query_states: &[f64],
    value: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let NgramQuery::PleGate {
        streams,
        width,
        capacity,
    } = query
    else {
        return Err(Refusal("expected n-gram gate query"));
    };
    let (s, h) = (streams as usize, width as usize);
    let tokens = value.len() / h;
    if value.is_empty()
        || value.len() != tokens * h
        || tokens > capacity as usize
        || key.len() != tokens * s * h
        || query_states.len() != key.len()
    {
        return Err(Refusal("n-gram gate operand shape mismatch"));
    }
    let scale = (h as f64).sqrt().recip();
    let mut values = Vec::with_capacity(key.len());
    let mut bounds = Vec::with_capacity(key.len());
    for token in 0..tokens {
        for stream in 0..s {
            let base = (token * s + stream) * h;
            let k = &key[base..base + h];
            let q = &query_states[base..base + h];
            let d = k.iter().zip(q).map(|(a, b)| a * b).sum::<f64>() * scale;
            let signed = d.signum() * d.abs().max(1e-6).sqrt();
            let gate = if d == 0.0 { 0.5 } else { sigmoid(signed) };
            for col in 0..h {
                let product = gate * value[token * h + col];
                values.push(product);
                bounds.push(product.abs());
            }
        }
    }
    OracleOutput::new(values, bounds, h)
}
