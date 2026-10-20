use super::{artifacts, buffers, registration, served};
pub use buffers::Input;
use kernel_core::{backend::View, contract::Argument};
use kernel_harness::{LabResult, Session, launch};
use kernels_sm120::sinkhorn_hc::completion::{Entry, Layout};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Module, Stream};
use std::collections::BTreeMap;
pub type Output = buffers::Output<4096>;
pub const VARIANTS: [&str; 5] = ["tail8", "tail16", "tail32", "split8", "split16"];
const GUARD: usize = 64;

pub fn variant(name: &str) -> LabResult<(Entry, u32)> {
    let (layout, warps) = if let Some(w) = name.strip_prefix("tail") {
        (Layout::Tail, w)
    } else if let Some(w) = name.strip_prefix("split") {
        (Layout::Split, w)
    } else {
        return Err(format!("unknown completion variant {name}"));
    };
    let warps: u32 = warps.parse().map_err(|_| "invalid warp count")?;
    if !layout.warps().contains(&warps) {
        return Err("unsupported warp count".into());
    }
    Ok((registration::entry(layout), warps))
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

/// One word per row, zeroed, followed by a poisoned guard.
pub struct Tickets {
    pub buffer: DeviceBuffer,
    rows: usize,
}
impl Tickets {
    pub fn new(s: &Session, rows: usize) -> LabResult<Self> {
        let buffer = s.allocate(rows * 4 + GUARD)?;
        let mut bytes = vec![0xa5; rows * 4 + GUARD];
        bytes[..rows * 4].fill(0);
        buffer
            .copy_from_bytes(&bytes, &s.stream)
            .map_err(|e| e.to_string())?;
        Ok(Self { buffer, rows })
    }
    /// Every ticket returned to zero and the guard is unchanged.
    pub fn check(&self, s: &Stream) -> LabResult<()> {
        let mut bytes = vec![0; self.buffer.byte_len()];
        self.buffer
            .copy_to_bytes(&mut bytes, s)
            .map_err(|e| e.to_string())?;
        if bytes[..self.rows * 4].iter().any(|&b| b != 0)
            || bytes[self.rows * 4..].iter().any(|&b| b != 0xa5)
        {
            return Err("completion ticket was not reset or its guard changed".into());
        }
        Ok(())
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

/// The served pair (`variant == "served"`) or one candidate launch composed
/// from that pair's own validated launches. Returns the launch count.
#[allow(clippy::too_many_arguments)]
pub fn launch(
    s: &Stream,
    k: &Kernels,
    i: &Input,
    o: &Output,
    t: &Tickets,
    variant: &str,
    widen: bool,
    alias: bool,
) -> LabResult<usize> {
    let candidate = match variant {
        "served" => None,
        name => Some(self::variant(name)?),
    };
    launch_with(
        s,
        k,
        i,
        o,
        t,
        candidate.as_ref().map(|(e, w)| (e, *w)),
        widen,
        alias,
    )
}

/// As `launch`, with an explicit candidate entry (or none for the served
/// pair), so mutation controls use the same composition.
#[allow(clippy::too_many_arguments)]
pub fn launch_with(
    s: &Stream,
    k: &Kernels,
    i: &Input,
    o: &Output,
    t: &Tickets,
    candidate: Option<(&Entry, u32)>,
    widen: bool,
    alias: bool,
) -> LabResult<usize> {
    let d = &i.data;
    let v = &o.data;
    let post = if alias { &v["post"] } else { &d["post_in"] };
    let comb = if alias { &v["comb"] } else { &d["comb_in"] };
    let buffers = [
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
        &t.buffer,
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
    let mut plans = served::plans(i.rows as u32, &views[..22], widen)?;
    if plans.len() != 2 {
        return Err("served HC reference must have two launches".into());
    }
    if let Some((entry, warps)) = candidate {
        plans = vec![
            entry
                .compose(&plans[0], &plans[1], views[22], warps)
                .map_err(|e| e.to_string())?,
        ];
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
                Argument::U32(value) if value == i.eps => Ok(A::U32(value)),
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

/// Every output role and the row's complete partial workspace.
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
