use super::{artifacts, buffers, registration};
pub use buffers::Input;
use bundle_deepseek41::sm120 as b;
use kernel_core::contract::LaunchPlan;
use kernel_harness::{LabResult, Session, launch};
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{Function, KernelArgument as A, Module, Stream};
use std::collections::BTreeMap;
pub type Output = buffers::Output<4096>;
pub const VARIANTS: [&str; 6] = [
    "cached32",
    "chain4-tail32",
    "rows2-tail32",
    "rows4-tail32",
    "rows8-tail32",
    "rows4-tail16",
];
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
    pub(super) fn call(
        &self,
        s: &Stream,
        symbol: &str,
        plan: LaunchPlan,
        args: &[A<'_>],
    ) -> LabResult<()> {
        self.functions
            .get(symbol)
            .ok_or("missing frozen symbol")?
            .launch(s, launch::config(plan)?, args)
            .map_err(|e| e.to_string())
    }
}
pub fn launch(
    s: &Stream,
    k: &Kernels,
    i: &Input,
    o: &Output,
    variant: &str,
    widen: bool,
) -> LabResult<()> {
    if variant == "served" {
        return super::served::launch(s, k, i, o, widen);
    }
    if variant != "reference" && !VARIANTS.contains(&variant) {
        return Err("unknown exact variant".into());
    }
    let rows = i.rows as u32;
    if rows == 0 || rows > 32 {
        return Err("invalid row capacity".into());
    }
    let d = &i.data;
    let v = &o.data;
    let mixer = b::hc_round_expand_mix::entry(&FLASH);
    let chain = b::hc_round_chain::entry(&FLASH);
    let (symbol, plan) = if let Some(tile) = variant
        .strip_prefix("rows")
        .and_then(|s| s.split('-').next())
        .and_then(|s| s.parse::<u32>().ok())
    {
        let e = registration::entry();
        let index = [2, 4, 8]
            .iter()
            .position(|&t| t == tile)
            .ok_or("unsupported tile")?;
        (
            e.symbols()[index].clone(),
            e.plan(rows, rows, tile).map_err(|e| e.to_string())?,
        )
    } else if variant.starts_with("chain4") || rows == 8 {
        (
            chain.symbols()[0].clone(),
            chain.plan(rows, rows, 4).map_err(|e| e.to_string())?,
        )
    } else {
        (
            mixer.symbols()[0].clone(),
            mixer.plans(rows).map_err(|e| e.to_string())?[0],
        )
    };
    k.call(
        s,
        &symbol,
        plan,
        &[
            A::Buffer(&d["streams"]),
            A::Buffer(&d["sum"]),
            A::Buffer(&v["rounded"]),
            A::Buffer(&d["post_in"]),
            A::Buffer(&d["comb_in"]),
            A::Buffer(&v["expanded"]),
            A::Buffer(&d["fn"]),
            A::Buffer(&o.scratch),
            A::Buffer(&i.live),
        ],
    )?;
    let mut args = vec![
        A::Buffer(&v["expanded"]),
        A::Buffer(&d["base"]),
        A::Buffer(&d["scale"]),
        A::Buffer(&d["pre_in"]),
        A::Buffer(&v["collapsed"]),
        A::Buffer(&v["pre"]),
        A::Buffer(&v["post"]),
        A::Buffer(&v["comb"]),
        A::Buffer(&o.scratch),
        A::Buffer(&i.live),
    ];
    let wide = if widen {
        A::Buffer(&v["widened"])
    } else {
        A::NullBuffer
    };
    if variant.contains("tail") {
        let warps = if variant.ends_with("16") { 16 } else { 32 };
        let tail = b::hc_finish_norm::entry(&FLASH);
        args.extend([
            A::Buffer(&d["norm_weight"]),
            A::Buffer(&v["normed"]),
            wide,
            A::Buffer(&v["codes"]),
            A::Buffer(&v["scales"]),
            A::Buffer(&v["status"]),
            A::U64(0),
            A::U32(i.eps),
        ]);
        k.call(
            s,
            &tail.symbols()[usize::from(warps == 32)],
            tail.plan(rows, rows, warps).map_err(|e| e.to_string())?,
            &args,
        )
    } else {
        k.call(
            s,
            &mixer.symbols()[1],
            mixer.plans(rows).map_err(|e| e.to_string())?[1],
            &args,
        )?;
        let norm = b::norm_quantize_cached::entry(&FLASH);
        let (symbol, plan) = if variant == "cached32" || rows == 8 {
            (
                norm.symbols()[2].clone(),
                norm.plan(rows, rows, 32).map_err(|e| e.to_string())?,
            )
        } else {
            (
                norm.reference.symbol.clone(),
                norm.reference.plan(rows).map_err(|e| e.to_string())?,
            )
        };
        k.call(
            s,
            &symbol,
            plan,
            &[
                A::Buffer(&v["collapsed"]),
                A::Buffer(&d["norm_weight"]),
                A::Buffer(&v["normed"]),
                wide,
                A::Buffer(&v["codes"]),
                A::Buffer(&v["scales"]),
                A::Buffer(&i.live),
                A::Buffer(&v["status"]),
                A::U64(0),
                A::U32(i.eps),
            ],
        )
    }
}
pub fn read(s: &Stream, o: &Output) -> LabResult<BTreeMap<String, Vec<u8>>> {
    let mut result = o.read(s)?;
    result.remove("probe");
    let mut scratch = vec![0; o.scratch.byte_len()];
    o.scratch
        .copy_to_bytes(&mut scratch, s)
        .map_err(|e| e.to_string())?;
    scratch.truncate(o.rows * 4096 * 4);
    result.insert("partials".into(), scratch);
    Ok(result)
}
pub fn launches(variant: &str) -> usize {
    if variant.contains("tail") { 2 } else { 3 }
}
