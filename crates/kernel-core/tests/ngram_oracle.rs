use kernel_core::contract::{
    ConvActivation, DType, NgramQuery, NumericalContract, RecurrentPhase, RecurrentQuery, RowQuery,
    Validate,
};
use kernel_core::ngram_oracle::{NgramTables, ngram_gather, ngram_rows, ngram_tables, ple_gate};
use kernel_core::oracle::{sharded_argmax, sharded_embedding_gather};
use kernel_core::recurrent_oracle::sigmoid;

const EOS: u32 = 248_044;

fn qwen38_tables() -> NgramTables {
    ngram_tables(248_320, 3, 8, 20_000_000, 1234, 0, 128).unwrap()
}

#[test]
fn qwen38_ngram_contracts_validate_with_stable_ids() {
    let gather = NgramQuery::HashGather {
        orders: 3,
        heads_per_order: 8,
        head_width: 160,
        local_heads: 4,
        capacity: 256,
    };
    let gate = NgramQuery::PleGate {
        streams: 4,
        width: 2560,
        capacity: 256,
    };
    assert_eq!(
        gather.numerical_contract_id().unwrap().as_str(),
        "row.ngram-hash-gather-fp8-tensor.v1"
    );
    assert_eq!(
        gate.numerical_contract_id().unwrap().as_str(),
        "row.ngram-gate-signed-sqrt-sigmoid.v1"
    );
    let conv = RecurrentQuery::CausalConv1d {
        channels: 10240,
        kernel: 4,
        dilation: 3,
        activation: ConvActivation::Silu,
        phase: RecurrentPhase::Decode,
        capacity: 256,
    };
    assert_eq!(conv.state_elements(), 9 * 10240);
    assert!(
        NgramQuery::HashGather {
            orders: 3,
            heads_per_order: 8,
            head_width: 160,
            local_heads: 5,
            capacity: 256,
        }
        .validate()
        .is_err()
    );
    // Orders are math from 2 up; a backend refuses the ones it cannot run.
    let orders = |orders, heads_per_order| NgramQuery::HashGather {
        orders,
        heads_per_order,
        head_width: 160,
        local_heads: 4,
        capacity: 256,
    };
    orders(9, 4).validate().unwrap();
    assert!(orders(1, 8).validate().is_err());
    // A head count past u32 is refused, not wrapped.
    assert!(orders(u32::MAX, 8).validate().is_err());
}

#[test]
fn qwen38_tables_match_the_checkpoint_layout() {
    let tables = qwen38_tables();
    // 128 row shards of 2,500,012 rows each in the checkpoint headers.
    assert_eq!(tables.rows, 128 * 2_500_012);
    assert_eq!(&tables.sizes[..3], &[20_000_003, 20_000_023, 20_000_033]);
    assert_eq!(tables.sizes[15], 20_000_171);
    assert_eq!(tables.offsets[1], 20_000_003);
    assert!(tables.sizes.windows(2).all(|w| w[0] < w[1]));
    // Independently computed from the HF splitmix64 derivation.
    assert_eq!(
        tables.multipliers,
        [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    );
    for &m in &tables.multipliers {
        assert_eq!(m % 2, 1);
        assert!(u128::from(m) * 248_320 < 1 << 63);
    }
}

#[test]
fn hashes_reset_at_end_of_document_tokens() {
    let tables = NgramTables {
        multipliers: vec![3, 5, 7],
        sizes: vec![11, 13],
        offsets: vec![0, 11],
        rows: 24,
    };
    let eos = 9;
    let out = ngram_rows(&tables, 1, &[eos, eos], &[1, 2, 9, 4], eos).unwrap();
    let row = |t0: u64, t1: u64, t2: u64| {
        let bigram = (t0 * 3) ^ (t1 * 5);
        let trigram = bigram ^ (t2 * 7);
        [bigram % 11, trigram % 13 + 11]
    };
    let expected = [
        row(1, 9, 9),
        row(2, 1, 9),
        row(9, 2, 1),
        // Token 4 follows the eos, so both predecessors read as eos.
        row(4, 9, 9),
    ]
    .concat();
    assert_eq!(out.rows, expected);
    assert_eq!(out.history, [9, 4]);
    // An eos two back hides everything behind it.
    let out = ngram_rows(&tables, 1, &[3, 9], &[5, 6], eos).unwrap();
    assert_eq!(out.rows, [row(5, 9, 9), row(6, 5, 9)].concat());
}

#[test]
fn split_chunks_carry_the_history() {
    let tables = qwen38_tables();
    let tokens = [17_u32, 248_044, 5, 99, 1000, 248_319, 3, 3];
    let whole = ngram_rows(&tables, 8, &[EOS, EOS], &tokens, EOS).unwrap();
    let first = ngram_rows(&tables, 8, &[EOS, EOS], &tokens[..3], EOS).unwrap();
    let second = ngram_rows(&tables, 8, &first.history, &tokens[3..], EOS).unwrap();
    assert_eq!([first.rows, second.rows].concat(), whole.rows);
    assert_eq!(second.history, whole.history);
    for (index, &row) in whole.rows.iter().enumerate() {
        let head = index % 16;
        assert!(row >= tables.offsets[head] && row < tables.offsets[head] + tables.sizes[head]);
    }
}

#[test]
fn head_sharded_ranks_all_gather_to_the_full_embedding() {
    let tables = ngram_tables(1000, 3, 8, 50, 7, 0, 4).unwrap();
    let width = 2;
    let codes = (0..tables.rows as usize * width)
        .map(|i| (i % 120) as u8)
        .collect::<Vec<_>>();
    let rows = ngram_rows(&tables, 8, &[0, 0], &[5, 6, 7], 0).unwrap().rows;
    let query = |local_heads| NgramQuery::HashGather {
        orders: 3,
        heads_per_order: 8,
        head_width: width as u32,
        local_heads,
        capacity: 8,
    };
    let full = ngram_gather(query(16), &rows, 0, &codes, 0, 0.5).unwrap();
    let mut gathered = vec![Vec::new(); 3];
    for rank in 0..4 {
        let start = tables.offsets[4 * rank];
        let end = tables
            .offsets
            .get(4 * rank + 4)
            .copied()
            .unwrap_or(tables.rows);
        let slice = &codes[start as usize * width..end as usize * width];
        let local = ngram_gather(query(4), &rows, rank, slice, start, 0.5).unwrap();
        for (token, chunk) in local.chunks_exact(4 * width).enumerate() {
            gathered[token].extend_from_slice(chunk);
        }
    }
    assert_eq!(gathered.concat(), full);
    // Code 0x38 is 1.0; the per-tensor scale multiplies.
    let one = ngram_gather(query(16), &[0; 16], 0, &[0x38; 2], 0, 0.25).unwrap();
    assert!(one.iter().all(|&x| x == 0.25));
    assert!(ngram_gather(query(4), &rows, 0, &codes[..10], 1_000_000, 1.0).is_err());
}

#[test]
fn ple_gate_uses_a_signed_square_root() {
    let gate = NgramQuery::PleGate {
        streams: 3,
        width: 4,
        capacity: 1,
    };
    let key = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    .concat();
    let query = [
        [2.0, 2.0, 0.0, 0.0],
        [-0.5, 0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0, 3.0],
    ]
    .concat();
    let out = ple_gate(gate, &key, &query, &[1.0, -2.0, 0.0, 4.0]).unwrap();
    // d = 4 / 2 = 2, then -0.5 / 2 = -0.25, then 0.
    let gates = [sigmoid(2_f64.sqrt()), sigmoid(-0.5), 0.5];
    for (stream, &g) in gates.iter().enumerate() {
        assert_eq!(
            out.values[stream * 4..(stream + 1) * 4],
            [g, -2.0 * g, 0.0, 4.0 * g]
        );
    }
    // Tiny products clamp to sqrt(1e-6).
    let tiny = ple_gate(
        NgramQuery::PleGate {
            streams: 1,
            width: 1,
            capacity: 1,
        },
        &[1e-9],
        &[1.0],
        &[1.0],
    )
    .unwrap();
    assert_eq!(tiny.values, [sigmoid(1e-6_f64.sqrt())]);
}

#[test]
fn embedding_and_logits_cover_the_248320_vocabulary() {
    RowQuery::EmbeddingGather {
        vocab: 248_320,
        width: 2560,
    }
    .validate()
    .unwrap();
    RowQuery::LogitsTopK {
        vocab: 248_320,
        k: 20,
    }
    .validate()
    .unwrap();
    RowQuery::ShardedArgmaxPartial {
        vocab_per_rank: 62_080,
        ranks: 4,
        logits: DType::Bf16,
    }
    .validate()
    .unwrap();
    assert_eq!(4 * 62_080, 248_320);
    // The last live token (248,076) and the stop IDs sit on rank 3.
    for id in [248_044_usize, 248_046, 248_076] {
        assert_eq!(id / 62_080, 3);
    }
    let table = (0..8 * 2).map(f64::from).collect::<Vec<_>>();
    let shard = sharded_embedding_gather(&table[8..], 4, 2, 4, &[1, 5, 7]).unwrap();
    assert_eq!(shard, [0.0, 0.0, 10.0, 11.0, 14.0, 15.0]);
    let (low, high) = (-1.0, 3.0);
    let shards = [[low; 4], [low; 4], [low; 4], [low, low, high, low]];
    let views = shards.iter().map(|s| &s[..]).collect::<Vec<_>>();
    assert_eq!(sharded_argmax(&views).unwrap(), 14);
}
