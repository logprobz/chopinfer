//! Independent consumer parity and order-sensitive norm inputs.
#![allow(dead_code)] // Shared by test binaries with different consumer cases.

#[path = "../../norm_weights.rs"]
pub(crate) mod cases;
use kernel_core::{
    codec::{bf16_to_f32, f32_to_bf16},
    contract::{DType, SigmoidMeanHyperConnectionQuery as H, StaticQuery},
};
use kernel_lab::{
    LabResult, Session,
    poison::{self, Replay, WriteMap},
};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig};
use serde_json::json;
use std::collections::BTreeMap;

pub fn varied_input(ids: &[usize], width: usize) -> Vec<u16> {
    ids.iter()
        .flat_map(|&row| {
            (0..width).map(move |c| {
                let h = (((row as u64 + 1) * 0x9e3779b9) ^ ((c as u64 + 1) * 0x85ebca6b))
                    .wrapping_mul(0xc2b2ae35);
                // Full BF16 mantissas, both signs and exponents -8 through 8.
                ((h >> 32) as u16 & 0x807f) | ((119 + (h % 17) as u16) << 7)
            })
        })
        .collect()
}

pub fn sparse_input(width: usize, groups: usize) -> (Vec<u16>, Vec<u16>) {
    let mut x = vec![0; width * groups];
    for group in x.chunks_exact_mut(width) {
        group[0] = f32_to_bf16(4096.0);
        for warp in 1..8 {
            group[32 * warp] = f32_to_bf16(1.0);
        }
    }
    // This BF16 weight moves the normalized value onto a BF16 rounding
    // boundary between the two reduction trees, at both tested widths.
    (x, vec![0x3a14; width * groups])
}

pub fn square_sum(x: &[u16], xor_serial: bool) -> f32 {
    let mut threads = [0.0_f32; 256];
    for (c, &bits) in x.iter().enumerate() {
        let v = bf16_to_f32(bits);
        threads[c % 256] = v.mul_add(v, threads[c % 256]);
    }
    let mut warps = [0.0_f32; 32];
    for (warp, lanes) in threads.chunks_exact(32).enumerate() {
        let mut values: [f32; 32] = lanes.try_into().unwrap();
        for offset in [16, 8, 4, 2, 1] {
            let before = values;
            for (lane, value) in values.iter_mut().enumerate() {
                let peer = if xor_serial {
                    lane ^ offset
                } else if lane + offset < 32 {
                    lane + offset
                } else {
                    lane
                };
                *value += before[peer];
            }
        }
        warps[warp] = values[0];
    }
    if xor_serial {
        warps[..8].iter().fold(0.0, |a, b| a + b)
    } else {
        for offset in [16, 8, 4, 2, 1] {
            let before = warps;
            for (lane, value) in warps.iter_mut().enumerate() {
                *value += before[if lane + offset < 32 {
                    lane + offset
                } else {
                    lane
                }];
            }
        }
        warps[0]
    }
}

pub fn cpu_norm(x: &[u16], w: &[u16], xor_serial: bool) -> Vec<u16> {
    let inv = (square_sum(x, xor_serial) / x.len() as f32 + 1e-6)
        .sqrt()
        .recip();
    x.iter()
        .zip(w)
        .map(|(&v, &weight)| f32_to_bf16(bf16_to_f32(v) * inv * (1.0 + bf16_to_f32(weight))))
        .collect()
}

fn count(session: &Session, rows: usize) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(4)?;
    poison::write(session, &buffer, 0, &(rows as u32).to_le_bytes())?;
    Ok(buffer)
}

fn run(
    session: &Session,
    module: &str,
    symbol: &str,
    grid: (u32, u32, u32),
    args: &[KernelArgument<'_>],
    outputs: &[(&str, &DeviceBuffer, usize)],
) -> LabResult<Vec<Vec<u16>>> {
    let module = session.module(module)?;
    let function = module.function(symbol).map_err(|e| e.to_string())?;
    let config = LaunchConfig {
        grid,
        block: (256, 1, 1),
        shared_memory_bytes: 0,
        cluster: LaunchConfig::NO_CLUSTER,
        programmatic_stream_serialization: true,
    };
    let launch = || {
        function
            .launch(&session.stream, config, args)
            .map_err(|e| e.to_string())
    };
    let mut replay = Replay::new(session);
    for &(name, buffer, elements) in outputs {
        replay = replay.output(name, buffer, WriteMap::of(DType::Bf16).prefix(elements)?);
    }
    replay.eager(launch)?;
    replay.relaunch(launch)?;
    replay.graph(&session.capture(|_| launch())?)?;
    let mut receipt = BTreeMap::from([("symbol".into(), json!(symbol))]);
    replay.record(&mut receipt);
    session.receipt("norm_consumer_replay", receipt)?;
    outputs
        .iter()
        .map(|&(_, buffer, elements)| session.download_bf16(buffer, elements))
        .collect()
}

pub fn shared_row(
    session: &Session,
    width: usize,
    groups: usize,
    dtype: DType,
    wrong: bool,
    x: &[u16],
    w: &[u16],
) -> LabResult<Vec<u16>> {
    let source = session.upload_bf16(x)?;
    let weight = match dtype {
        DType::Bf16 => session.upload_bf16(w)?,
        _ => session.upload_f32(&w.iter().copied().map(bf16_to_f32).collect::<Vec<_>>())?,
    };
    let rows = x.len() / (width * groups);
    let live = count(session, rows)?;
    let output = session.allocate((rows + 1) * width * groups * 2)?;
    let mut symbol = cases::row_symbol(width as u32, groups as u32, dtype);
    if wrong {
        symbol = symbol.replace("_v1", "_wrong_tree_v1");
    }
    let module = if wrong {
        cases::WRONG_TREE_CUBIN
    } else {
        cases::CUBIN
    };
    Ok(run(
        session,
        module,
        &symbol,
        (rows as u32 + 1, groups as u32, 1),
        &[
            KernelArgument::Buffer(&source),
            KernelArgument::Buffer(&weight),
            KernelArgument::Buffer(&output),
            KernelArgument::Buffer(&live),
        ],
        &[("norm", &output, x.len())],
    )?
    .remove(0))
}

pub fn wrong_tree_gpu(session: &Session) -> LabResult<()> {
    for (width, groups) in cases::SHAPES {
        let (x, w) = sparse_input(width as usize, groups as usize);
        let correct = shared_row(
            session,
            width as usize,
            groups as usize,
            DType::Bf16,
            false,
            &x,
            &w,
        )?;
        let wrong = shared_row(
            session,
            width as usize,
            groups as usize,
            DType::Bf16,
            true,
            &x,
            &w,
        )?;
        assert_ne!(
            correct, wrong,
            "fixture must expose the opposite reduction at {width} x {groups}"
        );
        session.receipt(
            "norm_wrong_tree",
            BTreeMap::from([
                ("width".into(), json!(width)),
                ("groups".into(), json!(groups)),
                ("different_output_bits".into(), json!(true)),
            ]),
        )?;
    }
    Ok(())
}

pub fn legacy_rows(session: &Session, mtp_symbol: &str) -> LabResult<()> {
    let entries = kernels_sm120::row::hyper_connection::catalog(kernel_lab::catalogs());
    let hc = entries
        .iter()
        .find(|e| {
            matches!(
                e.query,
                StaticQuery::HyperConnection(H::GroupedNorm {
                    width: 2560,
                    streams: 4,
                    ..
                })
            )
        })
        .expect("legacy grouped norm");
    let hc_module = format!("row_{}", hc.profile.tag());
    for sparse in [true, false] {
        let fixture = |width, groups| {
            if sparse {
                sparse_input(width, groups)
            } else {
                (
                    varied_input(&[3, 7, 11], width * groups),
                    varied_input(&[19], width * groups),
                )
            }
        };
        let (hx, hw) = fixture(2560, 4);
        let rows = hx.len() / 10240;
        let source = session.upload_bf16(&hx)?;
        let weight = session.upload_bf16(&hw)?;
        let output = session.allocate((rows + 1) * 10240 * 2)?;
        let live = count(session, rows)?;
        let legacy = run(
            session,
            &hc_module,
            &hc.symbol,
            (rows as u32 + 1, 4, 1),
            &[
                KernelArgument::Buffer(&source),
                KernelArgument::Buffer(&weight),
                KernelArgument::Buffer(&output),
                KernelArgument::Buffer(&live),
            ],
            &[("norm", &output, hx.len())],
        )?
        .remove(0);
        for dtype in [DType::Bf16, DType::F32] {
            assert_eq!(
                legacy,
                shared_row(session, 2560, 4, dtype, false, &hx, &hw)?,
                "legacy grouped norm, {dtype:?}, sparse={sparse}"
            );
        }
        let (mx, mw) = fixture(10240, 1);
        let (ex, ew) = fixture(2560, 1);
        let x = session.upload_bf16(&mx)?;
        let hweight = session.upload_bf16(&mw)?;
        let embed = session.upload_bf16(&ex)?;
        let eweight = session.upload_bf16(&ew)?;
        let hidden_out = session.allocate((rows + 1) * 10240 * 2)?;
        let embed_out = session.allocate((rows + 1) * 2560 * 2)?;
        let legacy = run(
            session,
            "probe_sm120",
            mtp_symbol,
            (rows as u32 + 1, 1, 1),
            &[
                KernelArgument::Buffer(&x),
                KernelArgument::Buffer(&hweight),
                KernelArgument::Buffer(&embed),
                KernelArgument::Buffer(&eweight),
                KernelArgument::Buffer(&hidden_out),
                KernelArgument::Buffer(&embed_out),
                KernelArgument::Buffer(&live),
            ],
            &[
                ("hidden", &hidden_out, mx.len()),
                ("embedding", &embed_out, ex.len()),
            ],
        )?;
        for dtype in [DType::Bf16, DType::F32] {
            assert_eq!(
                legacy[0],
                shared_row(session, 10240, 1, dtype, false, &mx, &mw)?,
                "legacy hidden norm, {dtype:?}, sparse={sparse}"
            );
            assert_eq!(
                legacy[1],
                shared_row(session, 2560, 1, dtype, false, &ex, &ew)?,
                "legacy embedding norm, {dtype:?}, sparse={sparse}"
            );
        }
        session.receipt(
            "norm_consumer_parity",
            BTreeMap::from([
                ("sparse_adversary".into(), json!(sparse)),
                ("legacy_hc_symbol".into(), json!(hc.symbol)),
                ("legacy_mtp_symbol".into(), json!(mtp_symbol)),
                ("bf16_and_f32_bitwise".into(), json!(true)),
            ]),
        )?;
    }
    Ok(())
}

pub fn legacy_comm(session: &Session) -> LabResult<()> {
    const WIDTH: usize = 2560;
    const WIDE: usize = 4 * WIDTH;
    for fixture in 0..3 {
        let (residual, weights) = if fixture == 0 {
            sparse_input(WIDTH, 4)
        } else if fixture == 1 {
            (varied_input(&[3], WIDE), varied_input(&[19], WIDE))
        } else {
            (vec![0; WIDE], vec![0x3a14; WIDE])
        };
        let partials = if fixture == 0 {
            vec![0.0; 2 * WIDTH]
        } else if fixture == 1 {
            varied_input(&[7], 2 * WIDTH)
                .into_iter()
                .map(bf16_to_f32)
                .collect()
        } else {
            vec![1e30_f32; 2 * WIDTH]
        };
        let logits = if fixture == 2 {
            vec![f32_to_bf16(-400.0); 4]
        } else {
            [-400.0, -8.0, 0.0, 8.0].map(f32_to_bf16).to_vec()
        };
        let source = session.upload_f32(&partials)?;
        let x = session.upload_bf16(&residual)?;
        let l = session.upload_bf16(&logits)?;
        let bfweight = session.upload_bf16(&weights)?;
        let fweight =
            session.upload_f32(&weights.iter().copied().map(bf16_to_f32).collect::<Vec<_>>())?;
        let residual_out = session.allocate(2 * WIDE * 2)?;
        let normed = session.allocate(2 * WIDE * 2)?;
        let live = count(session, 1)?;
        let mut baseline = None;
        for mode in 0..3 {
            for legacy in [true, false] {
                let symbol = format!(
                    "chop_lab_norm_comm_{}_m{mode}_v1",
                    if legacy { "legacy_bf16" } else { "f32" }
                );
                let got = run(
                    session,
                    if legacy {
                        cases::LEGACY_CUBIN
                    } else {
                        cases::CUBIN
                    },
                    &symbol,
                    (2, if mode == 2 { 1 } else { 4 }, 1),
                    &[
                        KernelArgument::Buffer(&source),
                        KernelArgument::Buffer(&x),
                        KernelArgument::Buffer(&l),
                        KernelArgument::Buffer(if legacy { &bfweight } else { &fweight }),
                        KernelArgument::Buffer(&residual_out),
                        KernelArgument::Buffer(&normed),
                        KernelArgument::Buffer(&live),
                        KernelArgument::U32(2),
                    ],
                    &[("residual", &residual_out, WIDE), ("norm", &normed, WIDE)],
                )?;
                if let Some(expected) = &baseline {
                    assert_eq!(
                        expected, &got,
                        "legacy comm parity, fixture={fixture}, mode={mode}, legacy={legacy}"
                    );
                } else {
                    baseline = Some(got);
                }
            }
        }
        if fixture == 2 {
            assert!(
                baseline.as_ref().unwrap()[0]
                    .iter()
                    .all(|&v| bf16_to_f32(v) > 0.0),
                "tail product must survive underflow"
            );
        }
        session.receipt(
            "norm_consumer_parity",
            BTreeMap::from([
                ("comm_fixture".into(), json!(fixture)),
                ("legacy_bf16_vs_f32_bitwise".into(), json!(true)),
                ("cached_form_is_abi_only".into(), json!(true)),
            ]),
        )?;
    }
    Ok(())
}
