//! The served wide-bucket HC boundary (the chain or row-tiled producer, then
//! the fused finish/norm tail), prepared from the bundle's registered op as
//! the bundle asks it, and candidates composed from its own validated
//! launches, bound to one set of device buffers.
use super::{artifacts, buffers, cols, twin};
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
use kernels_sm120::{
    backend::Sm120,
    sinkhorn_hc::{finish_split, finish_twin, round_cols},
};
use model_deepseek41_contract::{
    geometry::FLASH,
    hc, rows,
    serving::{DECODE_BATCH, PREFILL_GRANULE},
};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Module, Stream};
use std::collections::BTreeMap;

/// Scratch words per row: the served pair's chain sums (24 mixes x 160
/// chains) and its 256 RMS lanes.
pub const WORDS: usize = 4096;
pub const CHAIN_WORDS: usize = 3840;
pub type Output = buffers::Output<WORDS>;

/// A replacement composed from the served pair's own validated launches.
#[derive(Clone, Debug)]
pub enum Candidate {
    /// The served producer, then the tail on two CTAs per row.
    Twin(finish_twin::Entry, u32),
    /// The column-block producer (rows per CTA, transposed butterfly), then
    /// the split finish (warps per CTA).
    Cols {
        producer: round_cols::Entry,
        finish: finish_split::Entry,
        tile: u32,
        transposed: bool,
        warps: u32,
    },
}
impl Candidate {
    pub fn symbols(&self) -> Vec<String> {
        match self {
            Candidate::Twin(e, w) => vec![e.symbol(*w)],
            Candidate::Cols {
                producer,
                finish,
                tile,
                transposed,
                warps,
            } => vec![producer.symbol(*tile, *transposed), finish.symbol(*warps)],
        }
    }
    /// Whether the candidate leaves the RMS lane words of the workspace to
    /// its finish (the column-block producer does not write them).
    pub fn lanes(&self) -> bool {
        matches!(self, Candidate::Twin(..))
    }
}

/// Every candidate, by name: twin{8,16,32}, then cols_r{2,4,8}[tx]_w{8,16,32}.
pub fn variants() -> Vec<String> {
    let mut names = finish_twin::WARPS
        .iter()
        .map(|w| format!("twin{w}"))
        .collect::<Vec<_>>();
    for (tile, tx) in round_cols::TILES {
        for w in finish_split::WARPS {
            names.push(format!("cols_r{tile}{}_w{w}", if tx { "tx" } else { "" }));
        }
    }
    names
}

pub fn variant(name: &str) -> LabResult<Candidate> {
    if let Some(w) = name.strip_prefix("twin") {
        let warps: u32 = w.parse().map_err(|_| "invalid warp count")?;
        if !finish_twin::WARPS.contains(&warps) {
            return Err("unsupported warp count".into());
        }
        return Ok(Candidate::Twin(twin::served(), warps));
    }
    let rest = name
        .strip_prefix("cols_r")
        .ok_or_else(|| format!("unknown variant {name}"))?;
    let (tile, warps) = rest.split_once("_w").ok_or("invalid column variant")?;
    let (tile, transposed) = match tile.strip_suffix("tx") {
        Some(t) => (t, true),
        None => (tile, false),
    };
    let tile: u32 = tile.parse().map_err(|_| "invalid tile")?;
    let warps: u32 = warps.parse().map_err(|_| "invalid warp count")?;
    if !round_cols::TILES.contains(&(tile, transposed)) || !finish_split::WARPS.contains(&warps) {
        return Err("unsupported column variant".into());
    }
    Ok(Candidate::Cols {
        producer: cols::producer(),
        finish: cols::finish(),
        tile,
        transposed,
        warps,
    })
}

/// Every mutation control of the column pair, each with the plain other
/// half: the producer's at four rows per CTA, the finish's at 32 warps.
pub fn controls() -> Vec<Candidate> {
    cols::producer_controls()
        .into_iter()
        .map(|producer| Candidate::Cols {
            producer,
            finish: cols::finish(),
            tile: 4,
            transposed: false,
            warps: 32,
        })
        .chain(
            cols::finish_controls()
                .into_iter()
                .map(|finish| Candidate::Cols {
                    producer: cols::producer(),
                    finish,
                    tile: 4,
                    transposed: false,
                    warps: 32,
                }),
        )
        .collect()
}

/// The serving registration of the HC finish/norm op.
fn backend() -> LabResult<Sm120> {
    let backend = b::norm_quantize::register(Sm120::new(), &FLASH)?;
    let backend = b::norm_quantize_cached::register(backend, &FLASH);
    let backend = b::hc_round_expand_mix::register(backend, &FLASH);
    let backend = b::hc_round_chain::register(backend, &FLASH);
    Ok(b::hc_finish_norm::register(backend, &FLASH))
}

/// What the bundle asks of the op (`ops::exact::every_phase` at the decode
/// capacity).
fn need() -> Need {
    Need::rows(DECODE_BATCH).with_exact(Invariance {
        rows: Rows::OneRow { max: DECODE_BATCH },
        chunk: Chunk::Granule(PREFILL_GRANULE),
        ..Invariance::NONE
    })
}

fn query() -> SinkhornRoundNormQuery {
    SinkhornRoundNormQuery {
        mixer: SinkhornExpandMixQuery {
            mix: hc::mix(&FLASH),
        },
        norm: rows::norm_quantizers(&FLASH)[0],
    }
}

fn buffers(views: &[View], widen: bool) -> SinkhornRoundNormBuffers<View> {
    let v = |index: usize| views[index];
    SinkhornRoundNormBuffers {
        hc: Hc {
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
        },
        norm: Norm {
            input: v(10),
            weight: v(15),
            normed: v(16),
            widened: widen.then(|| v(17)),
            codes: v(18),
            scales: v(19),
            live_rows: v(14),
            status: v(20),
            status_row_offset: 0,
        },
        tickets: None,
    }
}

/// Views in role order: x, sum, rounded, post_in, comb_in, expanded, fn,
/// base, scale, pre_in, collapsed, pre, post, comb, live, norm weight,
/// normed, widened, codes, scales, status, workspace. The served op's
/// launches at `rows`, then `candidate` composed from them.
pub fn plans(
    rows: u32,
    views: &[View],
    widen: bool,
    candidate: Option<&Candidate>,
) -> LabResult<Vec<Launch>> {
    if views.len() != 22 {
        return Err("an HC plan needs 22 views".into());
    }
    let op = backend()?
        .sinkhorn_round_expand_mix_norm(&query(), need())
        .map_err(|e| format!("{e:?}"))?;
    let mut plans = op
        .launches(&buffers(views, widen), views[21], Counts::rows(rows))
        .map_err(|e| e.to_string())?;
    if plans.len() != 2 {
        return Err("the served HC boundary must have two launches".into());
    }
    let error = |e: kernel_core::contract::Refusal| e.0.to_owned();
    match candidate {
        None => {}
        Some(Candidate::Twin(entry, warps)) => {
            plans[1] = entry.compose(&plans[1], *warps).map_err(error)?;
        }
        Some(Candidate::Cols {
            producer,
            finish,
            tile,
            transposed,
            warps,
        }) => {
            plans[0] = producer
                .compose(&plans[0], *tile, *transposed)
                .map_err(error)?;
            plans[1] = finish.compose(&plans[1], *warps).map_err(error)?;
        }
    }
    Ok(plans)
}

/// The served op's workspace at the decode capacity.
pub fn workspace() -> LabResult<u64> {
    Ok(backend()?
        .sinkhorn_round_expand_mix_norm(&query(), need())
        .map_err(|e| format!("{e:?}"))?
        .info()
        .workspace)
}

/// Every symbol the served arm and the candidates launch at 16 and 32
/// rows, in first-use order.
pub fn symbols() -> LabResult<Vec<String>> {
    let views = (1..=22)
        .map(|handle| View {
            handle,
            bytes: 1 << 30,
        })
        .collect::<Vec<_>>();
    let mut candidates = vec![None];
    candidates.extend(
        variants()
            .iter()
            .map(|n| variant(n).map(Some))
            .collect::<LabResult<Vec<_>>>()?,
    );
    candidates.extend(controls().into_iter().map(Some));
    let mut symbols = Vec::new();
    for rows in [16, 32] {
        for c in &candidates {
            for widen in [false, true] {
                for l in plans(rows, &views, widen, c.as_ref())? {
                    if !symbols.contains(&l.symbol) {
                        symbols.push(l.symbol);
                    }
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

/// The served boundary (`candidate` none) or a candidate over `i` and `o`.
/// `epsilon` replaces the norm epsilon's bits (a mutation control).
/// Returns the launched symbols.
#[allow(clippy::too_many_arguments)]
pub fn launch(
    s: &Stream,
    k: &Kernels,
    i: &Input,
    o: &Output,
    candidate: Option<&Candidate>,
    widen: bool,
    alias: bool,
    epsilon: Option<u32>,
) -> LabResult<Vec<String>> {
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
    let plans = plans(o.rows as u32, &views, widen, candidate)?;
    if plans
        .iter()
        .any(|p| p.plan.workspace_bytes > o.scratch.byte_len() as u64)
    {
        return Err("scratch below the boundary's workspace".into());
    }
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
    Ok(plans.into_iter().map(|p| p.symbol).collect())
}

/// Every output role, the rows' chain sums ("partials") and their RMS lane
/// words ("lanes") of the workspace.
pub fn read(s: &Stream, o: &Output) -> LabResult<BTreeMap<String, Vec<u8>>> {
    let mut result = o.read(s)?;
    result.remove("probe");
    let mut scratch = vec![0; o.scratch.byte_len()];
    o.scratch
        .copy_to_bytes(&mut scratch, s)
        .map_err(|e| e.to_string())?;
    let (mut partials, mut lanes) = (Vec::new(), Vec::new());
    for row in scratch.chunks_exact(WORDS * 4).take(o.rows) {
        partials.extend_from_slice(&row[..CHAIN_WORDS * 4]);
        lanes.extend_from_slice(&row[CHAIN_WORDS * 4..]);
    }
    result.insert("partials".into(), partials);
    result.insert("lanes".into(), lanes);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The served arm launches what the bundle's Decode-16 and Decode-32
    /// lowering launches; every candidate keeps every role of it and only
    /// replaces the launches it names; the backend's own column-pair
    /// selection composes the same launches as the lab.
    #[test]
    fn served_plans_and_candidates_bind_every_role() {
        let views = (1..=22)
            .map(|handle| View {
                handle,
                bytes: 1 << 30,
            })
            .collect::<Vec<_>>();
        let eps = rows::norm_quantizers(&FLASH)[0].norm.epsilon_bits;
        let names = variants();
        assert_eq!(names.len(), 21);
        for rows in [16, 32] {
            for widen in [false, true] {
                let served = plans(rows, &views, widen, None).unwrap();
                let tail: Vec<_> = served
                    .iter()
                    .map(|l| l.symbol.rsplit_once("deepseek41_").unwrap().1)
                    .collect();
                let producer = if rows == 32 {
                    "hc_round_rows_r2_v1"
                } else {
                    "hc_round_chain_w4_v1"
                };
                assert_eq!(tail, [producer, "hc_finish_norm_w32_v1"]);
                let mut layout = vec![
                    serde_json::json!({"arm":"served","launches":served.iter().map(|l| serde_json::json!({
                    "symbol":l.symbol,"grid":l.plan.grid,"block":l.plan.block,
                    "dynamic_shared_bytes":l.plan.dynamic_shared_bytes})).collect::<Vec<_>>()}),
                ];
                let bound = |launches: &[Launch]| {
                    let mut bound = BTreeSet::new();
                    for l in launches {
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
                    bound
                };
                // Every role but the widened row (when not widening).
                let expected = (1..=22)
                    .filter(|&h| widen || h != 18)
                    .collect::<BTreeSet<u64>>();
                assert_eq!(bound(&served), expected);
                for name in names.iter().map(String::as_str) {
                    let c = variant(name).unwrap();
                    let launches = plans(rows, &views, widen, Some(&c)).unwrap();
                    assert_eq!(launches.len(), 2, "{name}");
                    assert_eq!(bound(&launches), expected, "{name}");
                    for (new, old) in launches.iter().zip(&served) {
                        assert_eq!(new.arguments, old.arguments, "{name}");
                        assert!(new.plan.workspace_bytes <= old.plan.workspace_bytes);
                        assert!(new.plan.pdl);
                    }
                    let symbols = launches
                        .iter()
                        .map(|l| l.symbol.clone())
                        .collect::<Vec<_>>();
                    match &c {
                        Candidate::Twin(..) => {
                            assert_eq!(symbols[0], served[0].symbol);
                            assert_eq!(symbols[1..], c.symbols());
                        }
                        Candidate::Cols {
                            tile,
                            transposed,
                            warps,
                            ..
                        } => {
                            assert_eq!(symbols, c.symbols());
                            assert_eq!(launches[0].plan.grid, [40, rows.div_ceil(*tile), 1]);
                            assert_eq!(launches[1].plan.grid, [2, rows, 1]);
                            // The backend's own selection at this bucket
                            // composes exactly these launches.
                            let backend = backend().unwrap().with_sinkhorn_finish_norm_cols(
                                cols::producer(),
                                cols::finish(),
                                vec![finish_split::Bucket {
                                    rows,
                                    tile: *tile,
                                    transposed: *transposed,
                                    warps: *warps,
                                }],
                            );
                            let op = backend
                                .sinkhorn_round_expand_mix_norm(&query(), need())
                                .unwrap();
                            let selected = op
                                .launches(&buffers(&views, widen), views[21], Counts::rows(rows))
                                .unwrap();
                            assert_eq!(selected, launches, "{name}");
                            let other = op
                                .launches(
                                    &buffers(&views, widen),
                                    views[21],
                                    Counts::rows(if rows == 16 { 32 } else { 16 }),
                                )
                                .unwrap();
                            assert!(other.iter().all(|l| !symbols.contains(&l.symbol)));
                        }
                    }
                    layout.push(serde_json::json!({"arm":name,"launches":launches.iter().map(|l| serde_json::json!({
                        "symbol":l.symbol,"grid":l.plan.grid,"block":l.plan.block,
                        "dynamic_shared_bytes":l.plan.dynamic_shared_bytes})).collect::<Vec<_>>()}));
                }
                for c in controls() {
                    let launches = plans(rows, &views, widen, Some(&c)).unwrap();
                    assert_eq!(bound(&launches), expected);
                    assert_eq!(
                        launches
                            .iter()
                            .map(|l| l.symbol.clone())
                            .collect::<Vec<_>>(),
                        c.symbols()
                    );
                }
                println!(
                    "HC_COLS_PLANS {}",
                    serde_json::json!({"rows":rows,"widen":widen,"arms":layout})
                );
            }
        }
        assert_eq!(controls().len(), 5);
        for name in [
            "served",
            "twin4",
            "cols_r16_w8",
            "cols_r2_w64",
            "cols_r2t_w8",
            "cols",
        ] {
            assert!(variant(name).is_err(), "{name}");
        }
        assert!(workspace().unwrap() <= u64::from(DECODE_BATCH) * WORDS as u64 * 4);
        let symbols = symbols().unwrap();
        // Served (chain, row-tiled, tail), twin, producers, finishes, controls.
        assert_eq!(symbols.len(), 3 + 3 + 6 + 3 + 5);
    }
}
