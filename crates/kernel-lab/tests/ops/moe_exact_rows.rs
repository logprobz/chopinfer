//! MiMo exact expert rows (`Variant::exact`, `EXACT_BUCKETS` of MiMo's contract). A
//! verify step of up to eight rows runs each decode bucket's exact entry (or
//! its decode entry where that already has bucket 1's reduction order), so
//! every token's gate/up rows and weighted down rows are those of the token
//! routed alone through bucket 1's entry, bit for bit. The checks use random
//! MXFP4 codes and scales, inputs spread over thirteen binades and routes
//! that share experts across tokens, at every MiMo rank count, full and
//! partial buckets. The bucket's decode entry must differ from bucket 1
//! somewhere on the same inputs, so the checks can see another order.
//!
//! `grouped_moe_mimo_exact_rows_bench` (`CHOPINFER_KERNEL_BENCH=1`) times the
//! served exact entries against the decode entries they replace, with the
//! tokens' routes shared or distinct.

use super::*;

/// Experts with stored weights; routes use lower IDs only.
const STORED_EXPERTS: usize = 64;

/// The entry a verify step of `tokens` rows serves (selection.rs): the
/// bucket's exact entry, or its decode entry.
fn served(
    profile: Profile,
    projection: Projection,
    tokens: u32,
    exact: bool,
) -> LabResult<Variant> {
    let find = |exact: bool| {
        moe::catalog(kernel_lab::catalogs()).into_iter().find(|v| {
            v.profile == profile
                && v.projection == projection
                && v.phase == Phase::Decode
                && v.path == InputPath::A16Gather
                && v.query.weight == WeightFormat::Mxfp4
                && v.tokens_capacity == tokens
                && v.exact == exact
                && v.query.epilogues.swiglu != Some(SwigluColumns::Interleaved)
                && v.query.epilogues.output_quantization.is_none()
        })
    };
    exact
        .then(|| find(true))
        .flatten()
        .or_else(|| find(false))
        .ok_or_else(|| format!("{} t{tokens}: no expert entry", profile.as_str()))
}

pub(super) fn mix(seed: u64) -> u64 {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^ (x >> 32)
}

/// Random MXFP4 codes, and UE8M0 scales near one, for the first
/// `STORED_EXPERTS` experts of `v`'s projection, in fragment layout.
pub(super) fn random_weights(session: &Session, v: &Variant, seed: u64) -> LabResult<GpuWeights> {
    let (rows, k) = (v.query.n as usize, v.query.k as usize);
    // Eight code bytes per draw.
    let codes = (0..STORED_EXPERTS * rows * k / 2)
        .map(|index| (mix(seed ^ (index / 8) as u64) >> (8 * (index % 8))) as u8)
        .collect::<Vec<_>>();
    let scales = (0..STORED_EXPERTS * rows * k / 32)
        .map(|index| 124 + ((mix(!seed ^ (index / 8) as u64) >> (8 * (index % 8))) % 6) as u8)
        .collect::<Vec<_>>();
    let layout = Mxfp4MoeFragmentLayout {
        experts: STORED_EXPERTS,
        rows_per_expert: rows,
        k,
    };
    Ok(GpuWeights {
        codes: upload_bytes(session, &layout.pack_codes(&codes).map_err(err)?.bytes)?,
        scales: upload_bytes(session, &layout.pack_scales(&scales).map_err(err)?.bytes)?,
        f32_scales: session.upload_f32(&[1.0])?,
    })
}

/// BF16 values spread over thirteen binades, so a changed F32 summation
/// order rounds differently.
pub(super) fn spread(count: usize, seed: u64) -> Vec<u16> {
    (0..count)
        .map(|index| {
            let binade = mix(seed ^ index as u64) % 13;
            f32_to_bf16(sample(seed ^ (index as u64) << 1) * 0.25 * (-(binade as f32)).exp2())
        })
        .collect()
}

/// Route metadata of `live` tokens whose `ids` and route `weights` are
/// token-major, eight per token, for `v`'s row tiles and capacity.
pub(super) struct Routes {
    topk: DeviceBuffer,
    offsets: DeviceBuffer,
    sorted: DeviceBuffer,
    padded: DeviceBuffer,
    live: DeviceBuffer,
    active: DeviceBuffer,
}

pub(super) fn routes(
    session: &Session,
    v: &Variant,
    ids: &[usize],
    weights: &[f32],
) -> LabResult<Routes> {
    let group = v.query.grouped.ok_or("grouped query")?;
    let experts = group.experts as usize;
    let permutation =
        oracle::expert_permute(ids, experts, usize::from(v.knobs.tile_m)).map_err(err)?;
    let capacity = match group.capacity {
        kernel_core::contract::RoutedCapacity::TotalRows(rows) => rows as usize,
        _ => return Err("expert entries need total routed-row capacity".into()),
    };
    let mut sorted = permutation
        .forward
        .iter()
        .map(|row| row.map_or(-1, |row| row as i32))
        .collect::<Vec<_>>();
    if sorted.len() > capacity {
        return Err("routes exceed the entry's routed-row capacity".into());
    }
    let padded = sorted.len() as i32;
    sorted.resize(capacity, -1);
    let mut active = (0..experts)
        .filter(|expert| ids.contains(expert))
        .map(|expert| expert as i32)
        .collect::<Vec<_>>();
    active.insert(0, active.len() as i32);
    let offsets = permutation
        .expert_offsets
        .iter()
        .map(|&offset| offset as i32)
        .collect::<Vec<_>>();
    Ok(Routes {
        topk: session.upload_f32(weights)?,
        offsets: upload_bytes(session, &i32_bytes(&offsets))?,
        sorted: upload_bytes(session, &i32_bytes(&sorted))?,
        padded: upload_bytes(session, &i32_bytes(&[padded]))?,
        live: upload_bytes(session, &i32_bytes(&[(ids.len() / TOP_K) as i32]))?,
        active: upload_bytes(session, &i32_bytes(&active))?,
    })
}

/// `v`'s routed output rows `[route][out]` for `routes` from `input`.
pub(super) struct Projected {
    v: Variant,
    functions: MoeFunctions,
    pub(super) counts: RuntimeCounts,
    pub(super) partials: DeviceBuffer,
    dummy: DeviceBuffer,
    one: DeviceBuffer,
    pub(super) output: DeviceBuffer,
    width: usize,
}

impl Projected {
    pub(super) fn new(session: &Session, v: Variant) -> LabResult<Self> {
        let counts = RuntimeCounts {
            m: v.tokens_capacity,
            batch: 0,
            context: 0,
            routed_rows: v.tokens_capacity * TOP_K as u32,
            split_k: v.knobs.split_k_capacity,
        };
        let width = match v.projection {
            Projection::GateUp => v.query.n as usize / 2,
            Projection::Down => v.query.n as usize,
        };
        Ok(Self {
            functions: load_moe(session, &v, counts)?,
            partials: session
                .allocate(v.plan(counts).map_err(err)?.workspace_bytes.max(1) as usize)?,
            dummy: upload_bytes(session, &[0])?,
            one: session.upload_f32(&[1.0])?,
            output: session.allocate(v.tokens_capacity as usize * TOP_K * width * 2)?,
            counts,
            width,
            v,
        })
    }

    pub(super) fn launch(
        &self,
        session: &Session,
        input: &DeviceBuffer,
        weights: &GpuWeights,
        routes: &Routes,
    ) -> LabResult<()> {
        let operands = MoeOperands {
            activation: input,
            activation_scales: &self.dummy,
            activation_f32_scales: &self.one,
            weights,
            output: &self.output,
            topk_weights: &routes.topk,
            offsets: &routes.offsets,
            sorted: &routes.sorted,
            padded: &routes.padded,
            live: &routes.live,
            active: &routes.active,
            quantized_output: &self.dummy,
            quantized_scales: &self.dummy,
            partials: &self.partials,
        };
        launch_moe(
            session,
            &self.v,
            &self.functions,
            self.counts,
            &operands,
            true,
        )
    }

    /// Output rows `[first, first + count)` as BF16 bits.
    pub(super) fn rows(
        &self,
        session: &Session,
        first: usize,
        count: usize,
    ) -> LabResult<Vec<u16>> {
        let all = session.download_bf16(&self.output, (first + count) * self.width)?;
        Ok(all[first * self.width..].to_vec())
    }
}

/// Token-major expert IDs of `live` tokens drawn from a pool of `pool`
/// experts, so tokens share experts, and distinct route weights.
fn token_routes(live: usize, pool: usize, seed: u64) -> (Vec<usize>, Vec<f32>) {
    let mut ids = Vec::with_capacity(live * TOP_K);
    for token in 0..live {
        let mut chosen = Vec::new();
        let mut draw = 0;
        while chosen.len() < TOP_K {
            let expert = (mix(seed ^ (token * 97 + draw) as u64) % pool as u64) as usize;
            draw += 1;
            if !chosen.contains(&expert) {
                chosen.push(expert);
            }
        }
        ids.extend(chosen);
    }
    let weights = (0..ids.len())
        .map(|route| 0.03 + 0.2 * ((mix(seed ^ 0x77 ^ route as u64) % 97) as f32 / 97.0))
        .collect();
    (ids, weights)
}

fn first_difference(left: &[u16], right: &[u16]) -> Option<usize> {
    left.iter().zip(right).position(|(a, b)| a != b)
}

#[test]
fn grouped_moe_mimo_exact_rows_equal_one_token() {
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let mut checked = 0;
        for profile in model_mimo26_contract::sm120::PROFILES {
            let one_gate = Projected::new(session, served(profile, Projection::GateUp, 1, false)?)?;
            let one_down = Projected::new(session, served(profile, Projection::Down, 1, false)?)?;
            let gate_weights = random_weights(session, &one_gate.v, 0x6a7e ^ checked)?;
            let down_weights = random_weights(session, &one_down.v, 0xd0e1 ^ checked)?;
            let intermediate = one_gate.width;
            for (tokens, live) in [(2_u32, 2_usize), (4, 3), (4, 4), (8, 6), (8, 8)] {
                let seed = u64::from(tokens) << 8 ^ live as u64 ^ checked;
                let (ids, weights) = token_routes(live, 20, seed);
                let input_host = spread(tokens as usize * HIDDEN, 0x5eed ^ seed);
                let input = session.upload_bf16(&input_host)?;
                let mut sensitive = BTreeMap::new();
                for exact in [true, false] {
                    let gate = Projected::new(
                        session,
                        served(profile, Projection::GateUp, tokens, exact)?,
                    )?;
                    let down =
                        Projected::new(session, served(profile, Projection::Down, tokens, exact)?)?;
                    let step = routes(session, &gate.v, &ids, &weights)?;
                    gate.launch(session, &input, &gate_weights, &step)?;
                    let step_down = routes(session, &down.v, &ids, &weights)?;
                    down.launch(session, &gate.output, &down_weights, &step_down)?;
                    for token in 0..live {
                        let alone_ids = &ids[token * TOP_K..(token + 1) * TOP_K];
                        let alone_weights = &weights[token * TOP_K..(token + 1) * TOP_K];
                        let alone_input =
                            session.upload_bf16(&input_host[token * HIDDEN..(token + 1) * HIDDEN])?;
                        let alone = routes(session, &one_gate.v, alone_ids, alone_weights)?;
                        one_gate.launch(session, &alone_input, &gate_weights, &alone)?;
                        let alone_down = routes(session, &one_down.v, alone_ids, alone_weights)?;
                        one_down.launch(session, &one_gate.output, &down_weights, &alone_down)?;
                        for (name, step_rows, alone_rows, width) in [
                            (
                                "gate_up",
                                gate.rows(session, token * TOP_K, TOP_K)?,
                                one_gate.rows(session, 0, TOP_K)?,
                                intermediate,
                            ),
                            (
                                "down",
                                down.rows(session, token * TOP_K, TOP_K)?,
                                one_down.rows(session, 0, TOP_K)?,
                                HIDDEN,
                            ),
                        ] {
                            let differs = first_difference(&step_rows, &alone_rows);
                            if exact {
                                if let Some(at) = differs {
                                    return Err(format!(
                                        "{} {} t{tokens} live {live} token {token} {name}: route {} column {} is {:#06x}, alone {:#06x}",
                                        profile.as_str(),
                                        (if name == "gate_up" { &gate.v } else { &down.v }).symbol,
                                        at / width,
                                        at % width,
                                        step_rows[at],
                                        alone_rows[at]
                                    ));
                                }
                            } else if differs.is_some() {
                                *sensitive.entry(name).or_insert(0) += 1;
                            }
                        }
                    }
                    if exact {
                        checked += 1;
                        let mut receipt = BTreeMap::new();
                        receipt.insert("case".into(), json!("moe_exact_rows"));
                        receipt.insert("profile".into(), json!(profile.as_str()));
                        receipt.insert("tokens".into(), json!(tokens));
                        receipt.insert("live".into(), json!(live));
                        receipt.insert("gate_up".into(), json!(gate.v.symbol));
                        receipt.insert("down".into(), json!(down.v.symbol));
                        receipt.insert("bitwise_equal_to_one_token".into(), json!(true));
                        session.receipt("exact_rows", receipt)?;
                    }
                }
                // Where the decode entry has another order, the inputs must
                // expose it for at least one token.
                for (name, projection) in [("gate_up", Projection::GateUp), ("down", Projection::Down)]
                {
                    let decode = served(profile, projection, tokens, false)?;
                    let exact = served(profile, projection, tokens, true)?;
                    if decode.symbol != exact.symbol && !sensitive.contains_key(name) {
                        return Err(format!(
                            "{} t{tokens} live {live} {name}: {} matched bucket 1 for every token, so the check cannot see another order",
                            profile.as_str(),
                            decode.symbol
                        ));
                    }
                }
            }
        }
        eprintln!("exact expert rows: {checked} bucket cases bitwise equal to one token");
        Ok(())
    })
    .expect("MiMo exact expert rows");
}

#[test]
fn grouped_moe_mimo_exact_rows_bench() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("exact expert rows bench: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs")
            .unwrap();
        return;
    }
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let profile = model_mimo26_contract::sm120::config::PROFILE;
        let gate_weights =
            random_weights(session, &served(profile, Projection::GateUp, 1, false)?, 3)?;
        let down_weights =
            random_weights(session, &served(profile, Projection::Down, 1, false)?, 5)?;
        for tokens in model_mimo26_contract::sm120::moe::EXACT_BUCKETS {
            for (routing, pool) in [("shared", TOP_K), ("distinct", STORED_EXPERTS)] {
                let live = tokens as usize;
                let (ids, weights) = if routing == "shared" {
                    token_routes(live, pool, 11)
                } else {
                    let ids = (0..live * TOP_K).collect::<Vec<_>>();
                    (ids, vec![0.125; live * TOP_K])
                };
                let input = session.upload_bf16(&spread(live * HIDDEN, 13))?;
                let gate_rows = session.upload_bf16(&spread(
                    live * TOP_K * served(profile, Projection::Down, 1, false)?.query.k as usize,
                    17,
                ))?;
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("moe_exact_rows_bench"));
                receipt.insert("tokens".into(), json!(tokens));
                receipt.insert("routing".into(), json!(routing));
                for projection in [Projection::GateUp, Projection::Down] {
                    let name = if projection == Projection::GateUp {
                        "gate_up"
                    } else {
                        "down"
                    };
                    let weights_for = if projection == Projection::GateUp {
                        &gate_weights
                    } else {
                        &down_weights
                    };
                    for exact in [false, true] {
                        let v = served(profile, projection, tokens, exact)?;
                        let projected = Projected::new(session, v)?;
                        let step = routes(session, &projected.v, &ids, &weights)?;
                        let activation = if projection == Projection::GateUp {
                            &input
                        } else {
                            &gate_rows
                        };
                        let graph = session.capture(|_| {
                            projected.launch(session, activation, weights_for, &step)
                        })?;
                        let timing =
                            time(session, 3, 7, |stream| graph.launch(stream).map_err(err))?;
                        receipt.insert(
                            format!("{name}_{}_us", if exact { "exact" } else { "decode" }),
                            json!(timing.median_ms * 1000.0),
                        );
                        receipt.insert(
                            format!("{name}_{}_symbol", if exact { "exact" } else { "decode" }),
                            json!(projected.v.symbol),
                        );
                    }
                }
                eprintln!("{}", serde_json::to_string(&receipt).map_err(err)?);
                session.receipt("bench", receipt)?;
            }
        }
        Ok(())
    })
    .expect("MiMo exact expert rows bench");
}
