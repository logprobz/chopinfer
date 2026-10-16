//! QSA query and pooled-key HF tables checked against independent oracles.
use super::*;
use kernel_core::oracle;
use crate::row::rope_positions::{SEEDS, Stats, positions};

fn exercise(session: &Session, batch: usize, count: usize, bench: bool) -> LabResult<u64> {
    let decode = count == 1;
    let cached = if decode { 3 } else { 0 };
    let rows = batch * count;
    let case = Case {
        name: format!("rope_qsa_b{batch}_n{count}"),
        phase: if decode {
            AttentionPhase::Decode
        } else {
            AttentionPhase::Prefill
        },
        fp8: false,
        rounded: false,
        sequences: vec![(cached, count); batch],
        pattern: Pattern::Random,
    };
    let modules = Modules::load(session)?;
    let kernels = Kernels::load(&modules, &case)?;
    let mut fixture = prepare(session, &case, modules.topology)?;
    let geometry = model_qwen38_contract::sm120::attention::SPARSE;
    fixture.d_cos_sin = upload(session, &bytes_f32(&cos_sin_table(262144, &geometry)))?;
    let exhaustive = std::env::var("CHOPINFER_ROPE_EXHAUSTIVE").as_deref() == Ok("1");
    let selected: Vec<_> = if bench {
        vec![262143]
    } else {
        positions(exhaustive && rows > 1)
            .into_iter()
            .filter(|&p| p < 262144)
            .collect()
    };
    let seeds = if bench { &SEEDS[..1] } else { &SEEDS[..] };
    let mut failures = 0;
    for &seed in seeds {
        let mut stats = Stats::default();
        stats.oracle_description = "f64 Gemma norm and HF table/F32 rotation; OracleOutput K/S bounds; FP8 K uses the existing storage bound";
        stats.maximum_position = 262143;
        let positions_per_launch = if decode { batch } else { count / BLOCK };
        for chunk in selected.chunks(positions_per_launch) {
            let p = chunk[0];
            let ps: Vec<_> = (0..rows)
                .map(|r| chunk[(if decode { r } else { r / BLOCK }).min(chunk.len() - 1)])
                .collect();
            let mut rng = Rng(seed ^ ((p as u64 + 1) << 24));
            let projection: Vec<_> = (0..rows * WIDTH).map(|_| rng.bf16(0.7)).collect();
            let row_batch: Vec<_> = (0..batch)
                .flat_map(|b| std::iter::repeat_n(b as i32, count))
                .collect();
            let token_positions: Vec<_> = (0..batch)
                .flat_map(|_| (cached..cached + count).map(|p| p as i32))
                .collect();
            let rope: Vec<_> = ps.iter().map(|&p| [p as i32; 3]).collect();
            let exact_rope: Vec<_> = ps.iter().map(|&p| [p; 3]).collect();
            let mut ring_rope = vec![0; batch * RING * 3];
            for b in 0..batch {
                let slot = fixture.sequences[b].slot as usize;
                ring_rope[slot * RING * 3..(slot + 1) * RING * 3].fill(ps[b * count] as i32);
            }
            let ring_input: Vec<_> = (0..batch * RING * INDEX_DIM)
                .map(|_| rng.bf16(0.7))
                .collect();
            fixture
                .d_ring
                .copy_from_bytes(&bytes_u16(&ring_input), &session.stream)
                .map_err(|e| e.to_string())?;
            fixture
                .d_ring_positions
                .copy_from_bytes(&bytes_i32(&ring_rope), &session.stream)
                .map_err(|e| e.to_string())?;
            fixture
                .d_block_keys
                .copy_from_bytes(
                    &vec![0xff; fixture.d_block_keys.byte_len()],
                    &session.stream,
                )
                .map_err(|e| e.to_string())?;
            fixture.upload_rows(session, &row_batch, &token_positions, &rope, &projection)?;
            fixture
                .d_q_index
                .copy_from_bytes(&vec![0xff; fixture.d_q_index.byte_len()], &session.stream)
                .map_err(|e| e.to_string())?;
            fixture.launch_query(&session.stream, &kernels)?;
            fixture.launch_key_write(&session.stream, &kernels, fixture.prepared)?;
            let query_input: Vec<_> = projection
                .chunks_exact(WIDTH)
                .flat_map(|x| decode_bf16(&x[..INDEX_HEADS * INDEX_DIM]))
                .collect();
            let query = SparseAttentionQuery::IndexerQueryGemmaNormMrope {
                heads: 4,
                head_dim: 128,
                rotary_dim: 64,
                theta_bits: geometry.theta.to_bits(),
                mrope: geometry.mrope,
                eps_bits: geometry.eps.to_bits(),
                capacity: rows as u32,
            };
            let reference = qsa::indexer_query(
                query,
                &query_input,
                &decode_bf16(&fixture.q_norm),
                &exact_rope,
                None,
            )
            .map_err(|e| e.to_string())?;
            let actual =
                session.download_bf16(&fixture.d_q_index, rows * INDEX_HEADS * INDEX_DIM)?;
            for (i, &x) in actual.iter().enumerate() {
                if i % INDEX_DIM >= 64 {
                    continue;
                }
                stats.check_bound(
                    f64::from(bf16_to_f32(x)),
                    reference.values[i],
                    oracle::numeric_bound(&reference, i, DType::Bf16),
                    ps[i / (INDEX_HEADS * INDEX_DIM)],
                    i % 32,
                );
            }
            let block_bytes = download(
                session,
                &fixture.d_block_keys,
                fixture.d_block_keys.byte_len(),
            )?;
            for b in 0..batch {
                let slot = fixture.sequences[b].slot as usize;
                let mut ring =
                    decode_bf16(&ring_input[slot * RING * INDEX_DIM..][..RING * INDEX_DIM]);
                let mut ring_positions = vec![[ps[b * count]; 3]; RING];
                let mut block_keys = vec![0.0; (cached + count) / BLOCK * INDEX_DIM];
                let raw: Vec<_> = projection[b * count * WIDTH..(b + 1) * count * WIDTH]
                    .chunks_exact(WIDTH)
                    .flat_map(|x| decode_bf16(&x[INDEX_HEADS * INDEX_DIM..]))
                    .collect();
                let query = SparseAttentionQuery::IndexerKeyWrite {
                    head_dim: 128,
                    rotary_dim: 64,
                    theta_bits: geometry.theta.to_bits(),
                    mrope: geometry.mrope,
                    eps_bits: geometry.eps.to_bits(),
                    block: BLOCK as u32,
                    ring: RING as u32,
                    capacity: count as u32,
                };
                let expected = qsa::indexer_key_write(
                    query,
                    &raw,
                    &(cached..cached + count).collect::<Vec<_>>(),
                    &exact_rope[b * count..(b + 1) * count],
                    &decode_bf16(&fixture.k_norm),
                    IndexerKeyCache {
                        ring_keys: &mut ring,
                        ring_positions: &mut ring_positions,
                        block_keys: &mut block_keys,
                    },
                )
                .map_err(|e| e.to_string())?;
                if expected.blocks.is_empty() {
                    return Err("QSA sweep completed no key block".into());
                }
                for (i, &block) in expected.blocks.iter().enumerate() {
                    let physical = fixture.sequences[b].pages[block * BLOCK / PAGE] as usize;
                    let offset = physical * fixture.block_stride as usize
                        + (block % (PAGE / BLOCK)) * INDEX_DIM * 2;
                    let actual = as_u16(&block_bytes[offset..offset + INDEX_DIM * 2]);
                    for (dim, &x) in actual.iter().enumerate() {
                        if dim >= 64 {
                            continue;
                        }
                        let index = i * INDEX_DIM + dim;
                        stats.check_bound(
                            f64::from(bf16_to_f32(x)),
                            expected.keys.values[index],
                            oracle::numeric_bound(&expected.keys, index, DType::Bf16),
                            ps[b * count + if decode { 0 } else { block * BLOCK }],
                            dim % 32,
                        );
                    }
                }
            }
            if bench {
                let mut receipt = BTreeMap::from([
                    ("case".into(), json!(case.name)),
                    ("rows".into(), json!(rows)),
                    ("rope_position".into(), json!(p)),
                ]);
                for key in [false, true] {
                    let graph = session.capture(|stream| {
                        if key {
                            fixture.launch_key_write(stream, &kernels, fixture.prepared)
                        } else {
                            fixture.launch_query(stream, &kernels)
                        }
                    })?;
                    let timing = time(session, 3, 15, |stream| {
                        graph.launch(stream).map_err(|e| e.to_string())
                    })?;
                    receipt.insert(
                        if key { "key_us" } else { "query_us" }.into(),
                        json!(timing.median_ms * 1000.0),
                    );
                }
                session.receipt("qsa_rope_cost", receipt)?;
            }
        }
        stats.receipt(
            session,
            &format!("{}:query_and_pooled_key_hf_angles", case.name),
            rows,
            seed,
            exhaustive && !bench && rows > 1,
        )?;
        failures += stats.failures;
    }
    Ok(failures)
}

#[test]
fn qsa_rope_position_sweep() {
    gpu_test(2 << 30, |session| {
        let mut failures = 0;
        for (batch, count) in [(1, 1), (32, 1), (1, 128)] {
            failures += exercise(session, batch, count, false)?;
        }
        if failures > 0 {
            return Err(format!("QSA exact-angle sweep: {failures} failures"));
        }
        Ok(())
    })
    .expect("QSA query and pooled-key exact-angle sweep");
}

#[test]
fn qsa_rope_position_costs() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        return;
    }
    gpu_test(2 << 30, |session| {
        for (batch, count) in [
            (1, 1),
            (2, 1),
            (4, 1),
            (8, 1),
            (16, 1),
            (32, 1),
            (1, 128),
            (1, 512),
        ] {
            if exercise(session, batch, count, true)? > 0 {
                return Err("QSA benchmark input fails oracle".into());
            }
        }
        Ok(())
    })
    .expect("QSA RoPE costs");
}

#[test]
fn qsa_table_matches_qwen_cpu_fixture_bits() {
    let fixture = include_bytes!("../../../../fixtures/qwen_rope_cpu.bin");
    let table = cos_sin_table(262144, &model_qwen38_contract::sm120::attention::SPARSE);
    for row in fixture[128..].chunks_exact(17) {
        let pair = row[0] as usize;
        let position = u32::from_le_bytes(row[1..5].try_into().unwrap()) as usize;
        for component in 0..2 {
            let offset = 9 + 2 * component;
            let bits = u16::from_le_bytes(row[offset..offset + 2].try_into().unwrap()) as u32;
            assert_eq!(table[position * 64 + pair * 2 + component].to_bits(), bits << 16,
                       "position {position}, pair {pair}, component {component}");
        }
    }
}
