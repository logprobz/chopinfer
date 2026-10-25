//! The served Decode-16 boundary (the rounded expansion mixer's partials and
//! finish, then the BF16 input norm) and the candidate finish/norm tail,
//! both prepared from the bundle's registered ops as the bundle asks them,
//! bound to one set of device buffers.
use super::{artifacts, buffers};
pub use buffers::Input;
use bundle_deepseek41::sm120 as b;
use kernel_core::{
    backend::{Backend, Counts, Launch, Need, View},
    contract::Argument,
    properties::{Chunk, Invariance, Rows},
    rmsnorm_quantize::RmsNormQuantizeBuffers as Norm,
    sinkhorn_expand_mix::{
        SinkhornExpandMixQuery, SinkhornRoundExpandMixBuffers as Hc, SinkhornRoundNormBuffers,
        SinkhornRoundNormQuery,
    },
};
use kernel_harness::{LabResult, Session, launch};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::{
    geometry::FLASH,
    hc, rows,
    serving::{DECODE_BATCH, PREFILL_GRANULE},
};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Module, Stream};
use std::collections::BTreeMap;

/// 16 rows of 8192 scratch words cover both ops' workspace at the decode
/// capacity (`workspace`).
pub type Output = buffers::Output<8192>;
pub const ROWS: u32 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Arm {
    /// Served at Decode-16: rounded expansion mixer, then the input norm.
    Served,
    /// The finish/norm pair served at Verify-16 and Decode-32.
    Tail,
}
pub const ARMS: [Arm; 2] = [Arm::Served, Arm::Tail];

fn backend() -> LabResult<Sm120> {
    let backend = b::norm_quantize::register(Sm120::new(), &FLASH)?;
    let backend = b::norm_quantize_cached::register(backend, &FLASH);
    let backend = b::hc_round_expand_mix::register(backend, &FLASH);
    let backend = b::hc_round_chain::register(backend, &FLASH, 4);
    Ok(b::hc_finish_norm::register_pair(backend, &FLASH))
}

/// What the bundle asks of all three ops (`ops::exact::every_phase` at the
/// decode capacity).
fn need() -> Need {
    Need::rows(DECODE_BATCH).with_exact(Invariance {
        rows: Rows::OneRow { max: DECODE_BATCH },
        chunk: Chunk::Granule(PREFILL_GRANULE),
        ..Invariance::NONE
    })
}

/// Views in role order: x, sum, rounded, post_in, comb_in, expanded, fn,
/// base, scale, pre_in, collapsed, pre, post, comb, live, norm weight,
/// normed, widened, codes, scales, status, workspace.
pub fn plans(arm: Arm, views: &[View], widen: bool) -> LabResult<Vec<Launch>> {
    if views.len() != 22 {
        return Err("an HC plan needs 22 views".into());
    }
    let backend = backend()?;
    let v = |index: usize| views[index];
    let hc = Hc {
        x: v(0),
        sum: v(1),
        rounded: v(2),
        post_in: v(3),
        comb_in: v(4),
        expanded: v(5),
        fn_: v(6),
        base: v(7),
        scale: v(8),
        pre_in: v(9),
        collapsed: v(10),
        pre: v(11),
        post: v(12),
        comb: v(13),
        live_rows: v(14),
    };
    let norm = Norm {
        input: v(10),
        weight: v(15),
        normed: v(16),
        widened: widen.then(|| v(17)),
        codes: v(18),
        scales: v(19),
        live_rows: v(14),
        status: v(20),
        status_row_offset: 0,
    };
    let mixer = SinkhornExpandMixQuery {
        mix: hc::mix(&FLASH),
    };
    let input_norm = rows::norm_quantizers(&FLASH)[0];
    let counts = Counts::rows(ROWS);
    let error = |e: kernel_core::contract::Refusal| e.0.to_owned();
    match arm {
        Arm::Served => {
            let op = backend
                .sinkhorn_round_expand_mix(&mixer, need())
                .map_err(|e| format!("{e:?}"))?;
            let mut launches = op.launches(&hc, v(21), counts).map_err(error)?;
            let op = backend
                .rmsnorm_quantize(&input_norm, need())
                .map_err(|e| format!("{e:?}"))?;
            if op.info().workspace != 0 {
                return Err("the input norm takes no workspace".into());
            }
            launches.extend(op.launches(&norm, View::EMPTY, counts).map_err(error)?);
            Ok(launches)
        }
        Arm::Tail => {
            let op = backend
                .sinkhorn_round_expand_mix_norm(
                    &SinkhornRoundNormQuery {
                        mixer,
                        norm: input_norm,
                    },
                    need(),
                )
                .map_err(|e| format!("{e:?}"))?;
            op.launches(
                &SinkhornRoundNormBuffers {
                    hc,
                    norm,
                    tickets: None,
                },
                v(21),
                counts,
            )
            .map_err(error)
        }
    }
}

/// The larger workspace of the two arms' ops.
pub fn workspace() -> LabResult<u64> {
    let backend = backend()?;
    let mixer = SinkhornExpandMixQuery {
        mix: hc::mix(&FLASH),
    };
    let served = backend
        .sinkhorn_round_expand_mix(&mixer, need())
        .map_err(|e| format!("{e:?}"))?
        .info()
        .workspace;
    let tail = backend
        .sinkhorn_round_expand_mix_norm(
            &SinkhornRoundNormQuery {
                mixer,
                norm: rows::norm_quantizers(&FLASH)[0],
            },
            need(),
        )
        .map_err(|e| format!("{e:?}"))?
        .info()
        .workspace;
    Ok(served.max(tail))
}

/// Every symbol either arm launches, in first-use order.
pub fn symbols() -> LabResult<Vec<String>> {
    let views = (1..=22)
        .map(|handle| View {
            handle,
            bytes: 1 << 30,
        })
        .collect::<Vec<_>>();
    let mut symbols = Vec::new();
    for arm in ARMS {
        for widen in [false, true] {
            for l in plans(arm, &views, widen)? {
                if !symbols.contains(&l.symbol) {
                    symbols.push(l.symbol);
                }
            }
        }
    }
    Ok(symbols)
}

pub struct Kernels {
    functions: BTreeMap<String, Function>,
    _modules: Vec<Module>,
}
impl Kernels {
    pub fn load(s: &Session) -> LabResult<Self> {
        artifacts::verify()?;
        let mut functions = BTreeMap::new();
        let mut modules = Vec::new();
        for (_, (bytes, symbols)) in artifacts::images()? {
            let m = s.load_cubin(&bytes)?;
            for symbol in symbols {
                functions.insert(
                    symbol.clone(),
                    m.function(&symbol).map_err(|e| e.to_string())?,
                );
            }
            modules.push(m);
        }
        Ok(Self {
            functions,
            _modules: modules,
        })
    }
}

/// Poison the outputs (status rows hold `prior`). With `alias`, post and
/// comb start as post_in and comb_in, as the served buffers alias them.
pub fn reset(s: &Stream, i: &Input, o: &Output, prior: u32, alias: bool) -> LabResult<()> {
    o.poison(s, prior)?;
    if alias {
        for (input, output) in [("post_in", "post"), ("comb_in", "comb")] {
            let mut bytes = vec![0; i.data[input].byte_len()];
            i.data[input]
                .copy_to_bytes(&mut bytes, s)
                .map_err(|e| e.to_string())?;
            bytes.resize(o.data[output].byte_len(), 0xa5);
            o.data[output]
                .copy_from_bytes(&bytes, s)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// One arm's launches over `i` and `o`. `epsilon` replaces the norm
/// epsilon's bits (a mutation control). Returns the launch count.
#[allow(clippy::too_many_arguments)]
pub fn launch(
    s: &Stream,
    k: &Kernels,
    i: &Input,
    o: &Output,
    arm: Arm,
    widen: bool,
    alias: bool,
    epsilon: Option<u32>,
) -> LabResult<usize> {
    let d = &i.data;
    let v = &o.data;
    let post = if alias { &v["post"] } else { &d["post_in"] };
    let comb = if alias { &v["comb"] } else { &d["comb_in"] };
    let buffers: [&DeviceBuffer; 22] = [
        &d["streams"],
        &d["sum"],
        &v["rounded"],
        post,
        comb,
        &v["expanded"],
        &d["fn"],
        &d["base"],
        &d["scale"],
        &d["pre_in"],
        &v["collapsed"],
        &v["pre"],
        &v["post"],
        &v["comb"],
        &i.live,
        &d["norm_weight"],
        &v["normed"],
        &v["widened"],
        &v["codes"],
        &v["scales"],
        &v["status"],
        &o.scratch,
    ];
    if (o.scratch.byte_len() as u64) < workspace()? {
        return Err("scratch below the ops' workspace".into());
    }
    // Aliased roles share one handle, as the bundle binds one logical view.
    let mut seen: Vec<(u64, u64)> = Vec::new();
    let views = buffers
        .iter()
        .enumerate()
        .map(|(index, b)| {
            let address = b
                .device_address_at(0, b.byte_len())
                .map_err(|e| e.to_string())?;
            let handle = match seen.iter().find(|(a, _)| *a == address) {
                Some(&(_, h)) => h,
                None => {
                    let h = index as u64 + 1;
                    seen.push((address, h));
                    h
                }
            };
            Ok(View {
                handle,
                bytes: b.byte_len() as u64,
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let by_handle = |h: u64| -> LabResult<&DeviceBuffer> {
        views
            .iter()
            .position(|v| v.handle == h)
            .map(|index| buffers[index])
            .ok_or_else(|| "unbound handle".to_owned())
    };
    let plans = plans(arm, &views, widen)?;
    for plan in &plans {
        let args = plan
            .arguments
            .0
            .iter()
            .map(|arg| match *arg {
                Argument::DeviceBuffer(h) => Ok(A::Buffer(by_handle(h)?)),
                Argument::NullDevicePointer => Ok(A::NullBuffer),
                Argument::U64(value) => Ok(A::U64(value)),
                Argument::U32(value) if value == i.eps => Ok(A::U32(epsilon.unwrap_or(value))),
                _ => Err("unqualified HC argument or norm epsilon".to_owned()),
            })
            .collect::<LabResult<Vec<_>>>()?;
        k.functions
            .get(&plan.symbol)
            .ok_or("unstaged HC symbol")?
            .launch(s, launch::config(plan.plan)?, &args)
            .map_err(|e| e.to_string())?;
    }
    Ok(plans.len())
}

/// Every output role. The ops' private workspace is excluded: the arms lay
/// out different partials there and nothing reads it after the boundary.
pub fn read(s: &Stream, o: &Output) -> LabResult<BTreeMap<String, Vec<u8>>> {
    let mut result = o.read(s)?;
    result.remove("probe");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arms launch what the bundle's Decode-16 lowering launches before
    /// and after the tail binding, over the same roles; the job compares
    /// these shapes with the exported Decode-16 journals.
    #[test]
    fn served_and_tail_plans_bind_every_role() {
        let views = (1..=22)
            .map(|handle| View {
                handle,
                bytes: 1 << 30,
            })
            .collect::<Vec<_>>();
        let eps = rows::norm_quantizers(&FLASH)[0].norm.epsilon_bits;
        for widen in [false, true] {
            let mut layout = Vec::new();
            for arm in ARMS {
                let launches = plans(arm, &views, widen).unwrap();
                let symbols = launches
                    .iter()
                    .map(|l| l.symbol.rsplit_once("deepseek41_").unwrap().1)
                    .collect::<Vec<_>>();
                assert_eq!(
                    symbols,
                    match arm {
                        Arm::Served => vec![
                            "hc_round_expand_mix_partials_v1",
                            "hc_round_expand_mix_finish_v1",
                            "input_v1"
                        ],
                        Arm::Tail => vec!["hc_round_chain_w4_v1", "hc_finish_norm_w32_v1"],
                    },
                    "{arm:?}"
                );
                let mut bound = std::collections::BTreeSet::new();
                for l in &launches {
                    for a in &l.arguments.0 {
                        match *a {
                            Argument::DeviceBuffer(h) => {
                                bound.insert(h);
                            }
                            Argument::U32(value) => assert_eq!(value, eps),
                            Argument::NullDevicePointer | Argument::U64(_) => {}
                            _ => panic!("unexpected argument {a:?}"),
                        }
                    }
                }
                // Every role but the widened row (when not widening).
                let expected = (1..=22)
                    .filter(|&h| widen || h != 18)
                    .collect::<std::collections::BTreeSet<u64>>();
                assert_eq!(bound, expected, "{arm:?} widen={widen}");
                layout.push(serde_json::json!({"arm":format!("{arm:?}"),"launches":launches.iter().map(|l| serde_json::json!({
                    "symbol":l.symbol,"grid":l.plan.grid,"block":l.plan.block,
                    "dynamic_shared_bytes":l.plan.dynamic_shared_bytes})).collect::<Vec<_>>()}));
            }
            println!(
                "HC_TAIL16_PLANS {}",
                serde_json::json!({"widen":widen,"arms":layout})
            );
        }
        assert!(workspace().unwrap() <= u64::from(ROWS) * 8192 * 4);
    }
}
