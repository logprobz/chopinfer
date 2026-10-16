//! The kit's adapter for kernels the host launches itself rather than
//! through a [`Backend`]: the device sampler (`kernel_core::sampling`).
//!
//! The preparer here plans the launches the host makes over the op's role
//! record, so the kit runs the kernels by role like any backend op:
//! [`sampling`] the served SM120 sampling path the host dispatches
//! (`kernels_sm120::sampling`'s planners, with `runtime_kernels`' symbol
//! list). [`HostModules`] loads the cubin the host ships
//! (`runtime_kernels::SAMPLING_CUBIN`), not the lab's own builds, and
//! [`Host`] stands in for the backend in reports. No backend prepares or
//! refuses these ops: a [`super::Required`] over them ignores the backend
//! it is given.

use super::modules::{Functions, Modules};
use crate::conformance::Probes;
use crate::{LabResult, Session};
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::{ArgumentList, DType, LaunchPlan, Refusal, RuntimeCounts, Validate};
use kernel_core::properties::Invariance;
use kernel_core::sampling::{
    BIAS_BYTES, COUNT_BYTES, META_BYTES, OUTPUT_BYTES, SamplingBuffers, SamplingRequirement,
};
use kernels_sm120::sampling::{
    self as served, FILTER_ROW_BYTES, FilteredSamplingBuffers, SIMPLE_PARTIAL_BYTES, SIMPLE_TILE,
    SamplingQuery, SimpleSamplingBuffers,
};
use runtime_cuda::Module;
use std::cell::OnceCell;

/// The launcher of host kernels in a kit report: a backend that
/// implements no family, since the host prepares these ops itself.
pub struct Host;

impl Backend for Host {
    fn name(&self) -> &str {
        "host"
    }
}

/// The cubin the host ships: the served sampler, loaded once, with
/// SM120's PDL probes from the session's cubins ([`Probes::of_session`]).
pub struct HostModules;

impl Modules for HostModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        if runtime_kernels::SAMPLING_CUBIN.is_empty() {
            return Err("the host's cubins are built on Linux only".into());
        }
        let sampling = OnceCell::<Module>::new();
        Ok(Box::new(move |symbol| {
            if sampling.get().is_none() {
                let _ = sampling.set(session.load_cubin(runtime_kernels::SAMPLING_CUBIN)?);
            }
            let module = sampling.get().expect("the module just loaded");
            module.function(symbol).map_err(|e| e.to_string())
        }))
    }

    fn probes(&self, session: &Session) -> LabResult<Option<Probes>> {
        Probes::of_session(session).map(Some)
    }
}

/// A served sampling path, as the host dispatches it over a view's rows
/// (`runtime-host`'s `SamplingDispatch`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplingPath {
    /// Every row simple, none with top logprobs: the simple partials.
    Simple,
    /// Every row simple, some with top logprobs: the top partials, with
    /// 256 threads.
    Top,
    /// Every row greedy with at most one top logprob: the greedy partials,
    /// with 256 threads when some row has one (`top`).
    Greedy { top: bool },
    /// Any rows: the six filtered stages.
    Filtered,
}

impl SamplingPath {
    pub fn name(self) -> &'static str {
        match self {
            SamplingPath::Simple => "simple",
            SamplingPath::Top => "top",
            SamplingPath::Greedy { top: false } => "greedy",
            SamplingPath::Greedy { top: true } => "greedy top",
            SamplingPath::Filtered => "filtered",
        }
    }
}

/// The host's sampler over up to `need.capacity` rows of `requirement`'s
/// logits on `path`. `gathered` is the row count of a gathered view whose
/// dispatch an owned subview keeps (`runtime-host`'s `SamplingDispatch`,
/// retained across owner partitions), or `None` for a view sampled on its
/// own: the host sizes the simple partials' blocks (256 threads with top
/// logprobs or at most eight rows, else 64) and the filtered histogram's
/// grid by the dispatch's rows, not the launched ones. It claims no row
/// property: the contract judges logprobs within a tolerance, not by their
/// bits.
pub fn sampling(
    requirement: SamplingRequirement,
    path: SamplingPath,
    gathered: Option<u32>,
    need: Need,
) -> Result<Prepared<SamplingBuffers<View>>, Refused> {
    requirement.validate()?;
    if gathered.is_some_and(|rows| rows < need.capacity || rows > served::MAX_BATCH) {
        return Err(Refused::Rule(
            "a gathered dispatch holds the launched rows and at most 64".into(),
        ));
    }
    let query = SamplingQuery {
        row_capacity: need.capacity,
        width_capacity: requirement.width,
        dtype: requirement.dtype,
    };
    query.validate()?;
    let f32 = requirement.dtype == DType::F32;
    let symbols: Vec<&str> = match path {
        SamplingPath::Filtered if f32 => runtime_kernels::SAMPLING_FILTERED_F32_SYMBOLS.to_vec(),
        SamplingPath::Filtered => runtime_kernels::SAMPLING_FILTERED_BF16_SYMBOLS.to_vec(),
        _ => {
            let partials = match (path, f32) {
                (SamplingPath::Simple, true) => {
                    runtime_kernels::SAMPLING_SIMPLE_PARTIALS_F32_SYMBOL
                }
                (SamplingPath::Simple, false) => {
                    runtime_kernels::SAMPLING_SIMPLE_PARTIALS_BF16_SYMBOL
                }
                (SamplingPath::Top, true) => {
                    runtime_kernels::SAMPLING_SIMPLE_TOP_PARTIALS_F32_SYMBOL
                }
                (SamplingPath::Top, false) => {
                    runtime_kernels::SAMPLING_SIMPLE_TOP_PARTIALS_BF16_SYMBOL
                }
                (_, true) => runtime_kernels::SAMPLING_GREEDY_PARTIALS_1024_F32_SYMBOL,
                (_, false) => runtime_kernels::SAMPLING_GREEDY_PARTIALS_1024_BF16_SYMBOL,
            };
            let finish = if f32 {
                runtime_kernels::SAMPLING_SIMPLE_FINISH_F32_SYMBOL
            } else {
                runtime_kernels::SAMPLING_SIMPLE_FINISH_BF16_SYMBOL
            };
            vec![partials, finish]
        }
    };
    let format = if f32 { "f32" } else { "bf16" };
    let dispatch = gathered.map_or_else(String::new, |rows| format!(".gathered{rows}"));
    let mut info = OpInfo::new(
        format!(
            "host.sampling.{}.{format}{dispatch}.v1",
            path.name().replace(' ', "_")
        ),
        Invariance::NONE,
        symbols
            .iter()
            .map(|symbol| (symbol.to_string(), 0))
            .collect(),
    );
    let rows = u64::from(need.capacity);
    let width = u64::from(requirement.width);
    let tiles = u64::from(requirement.width.div_ceil(SIMPLE_TILE));
    info.bytes = vec![(
        "scratch.partials".into(),
        rows * tiles * SIMPLE_PARTIAL_BYTES,
    )];
    if path == SamplingPath::Filtered {
        info.bytes.extend([
            ("scratch.filter".into(), rows * FILTER_ROW_BYTES),
            ("scratch.adjusted".into(), rows * width * 8),
            ("scratch.order".into(), rows * width * 4),
        ]);
    }
    need.check(&info)?;
    Ok(Box::new(SamplingOp {
        query,
        path,
        gathered,
        symbols: symbols.into_iter().map(str::to_owned).collect(),
        info,
    }))
}

struct SamplingOp {
    query: SamplingQuery,
    path: SamplingPath,
    /// The rows of a retained gathered dispatch ([`sampling`]).
    gathered: Option<u32>,
    /// The symbols in launch order.
    symbols: Vec<String>,
    info: OpInfo,
}

impl PreparedOp<SamplingBuffers<View>> for SamplingOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &SamplingBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        let live = buffers.live_width;
        let element = if self.query.dtype == DType::F32 { 4 } else { 2 };
        let logits = u64::from(rows.max(1) - 1)
            .checked_mul(buffers.row_stride)
            .and_then(|elements| elements.checked_add(u64::from(live)))
            .and_then(|elements| elements.checked_mul(element))
            .ok_or(Refusal("sampling logits overflow"))?;
        buffers.logits.covers(logits)?;
        let rows64 = u64::from(rows);
        buffers.metadata.covers(rows64 * META_BYTES as u64)?;
        buffers.output.covers(rows64 * OUTPUT_BYTES as u64)?;
        buffers.counts.covers(COUNT_BYTES as u64)?;
        buffers.biases.covers(BIAS_BYTES as u64)?;
        buffers.eos.covers(8)?;
        let tiles = u64::from(live.div_ceil(SIMPLE_TILE));
        buffers
            .scratch
            .partials
            .covers(rows64 * tiles * SIMPLE_PARTIAL_BYTES)?;
        let dispatch_rows = self.gathered.unwrap_or(rows);
        let runtime = RuntimeCounts {
            m: rows,
            batch: rows,
            context: 0,
            routed_rows: 0,
            split_k: 1,
        };
        let launch = |symbol: &String, plan: LaunchPlan, arguments: &ArgumentList| Launch {
            symbol: symbol.clone(),
            rows,
            plan,
            arguments: arguments.clone(),
            collective: false,
            persistent: false,
        };
        if self.path != SamplingPath::Filtered {
            let top = matches!(
                self.path,
                SamplingPath::Top | SamplingPath::Greedy { top: true }
            );
            let mut planned = served::plan_simple(
                self.query,
                runtime,
                live,
                buffers.row_stride,
                SimpleSamplingBuffers {
                    logits: buffers.logits.handle,
                    metadata: buffers.metadata.handle,
                    partials: buffers.scratch.partials.handle,
                    output: buffers.output.handle,
                },
                top,
            )?;
            // The host's block size follows its dispatch's rows.
            planned.partials.block[0] = if top || dispatch_rows <= 8 { 256 } else { 64 };
            planned.partials.validate()?;
            return Ok(vec![
                launch(&self.symbols[0], planned.partials, &planned.partial_args),
                launch(&self.symbols[1], planned.finish, &planned.finish_args),
            ]);
        }
        buffers.scratch.filter.covers(rows64 * FILTER_ROW_BYTES)?;
        buffers
            .scratch
            .adjusted
            .covers(rows64 * u64::from(live) * 8)?;
        buffers.scratch.order.covers(rows64 * u64::from(live) * 4)?;
        if let Some(masks) = buffers.masks {
            masks.covers(rows64 * u64::from(live.div_ceil(32)) * 4)?;
        }
        let mut planned = served::plan_filtered(
            self.query,
            runtime,
            live,
            buffers.row_stride,
            FilteredSamplingBuffers {
                logits: buffers.logits.handle,
                metadata: buffers.metadata.handle,
                counts: buffers.counts.handle,
                biases: buffers.biases.handle,
                eos: buffers.eos.handle,
                partials: buffers.scratch.partials.handle,
                filter: buffers.scratch.filter.handle,
                adjusted: buffers.scratch.adjusted.handle,
                order: buffers.scratch.order.handle,
                output: buffers.output.handle,
                masks: buffers.masks.map(|masks| masks.handle),
            },
        )?;
        // The host's histogram grid follows its dispatch's rows.
        planned.histogram.grid[0] =
            served::filtered_histogram_groups(live.div_ceil(SIMPLE_TILE), dispatch_rows);
        planned.histogram.validate()?;
        let stages = [
            planned.stats,
            planned.histogram,
            planned.collect,
            planned.resolve,
            planned.gumbel,
        ];
        let mut launches = stages
            .into_iter()
            .zip(&self.symbols)
            .map(|(plan, symbol)| launch(symbol, plan, &planned.tile_args))
            .collect::<Vec<_>>();
        launches.push(launch(
            &self.symbols[5],
            planned.finish,
            &planned.finish_args,
        ));
        Ok(launches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Views that cover any size, at distinct handles.
    fn views() -> SamplingBuffers<View> {
        let view = |handle| View {
            handle,
            bytes: u64::MAX,
        };
        SamplingBuffers {
            logits: view(1),
            metadata: view(2),
            counts: view(3),
            biases: view(4),
            eos: view(5),
            masks: None,
            output: view(6),
            scratch: kernel_core::sampling::SamplingScratch {
                partials: view(7),
                filter: view(8),
                adjusted: view(9),
                order: view(10),
            },
            live_width: 2600,
            row_stride: 2688,
        }
    }

    /// A logits stride whose byte size overflows is refused, not wrapped:
    /// two F32 rows 2^62 elements apart.
    #[test]
    fn an_overflowing_logits_size_is_refused() {
        let requirement = SamplingRequirement {
            dtype: DType::F32,
            width: 4096,
        };
        let op = sampling(requirement, SamplingPath::Simple, None, Need::rows(2)).unwrap();
        let buffers = SamplingBuffers {
            row_stride: 1 << 62,
            ..views()
        };
        let refusal = op
            .launches(&buffers, View::EMPTY, Counts::rows(2))
            .unwrap_err();
        assert_eq!(refusal, Refusal("sampling logits overflow"));
        op.launches(&views(), View::EMPTY, Counts::rows(2))
            .expect("a stride that fits");
    }

    /// A partition of a gathered view keeps the gathered dispatch: four
    /// rows of a 16-row simple batch launch 64 threads per block, as
    /// runtime-host does, where a view of four rows alone launches 256.
    #[test]
    fn a_partition_keeps_its_gathered_dispatch() {
        let requirement = SamplingRequirement {
            dtype: DType::Bf16,
            width: 4096,
        };
        let threads = |path, gathered| {
            let op = sampling(requirement, path, gathered, Need::rows(4)).unwrap();
            op.launches(&views(), View::EMPTY, Counts::rows(4)).unwrap()[0]
                .plan
                .block[0]
        };
        assert_eq!(threads(SamplingPath::Simple, None), 256);
        assert_eq!(threads(SamplingPath::Simple, Some(16)), 64);
        assert_eq!(threads(SamplingPath::Greedy { top: false }, Some(16)), 64);
        assert_eq!(threads(SamplingPath::Top, Some(16)), 256);
        // The filtered histogram's grid follows the gathered rows.
        let wide = SamplingRequirement {
            width: 8192,
            ..requirement
        };
        let op = sampling(wide, SamplingPath::Filtered, Some(64), Need::rows(4)).unwrap();
        let buffers = SamplingBuffers {
            live_width: 8192,
            row_stride: 8192,
            ..views()
        };
        let histogram = &op.launches(&buffers, View::EMPTY, Counts::rows(4)).unwrap()[1];
        assert_eq!(histogram.plan.grid, [6, 4, 1]);
        assert_eq!(served::filtered_histogram_groups(8, 4), 8);
        // A dispatch over fewer rows than the launch is refused.
        assert!(sampling(requirement, SamplingPath::Simple, Some(2), Need::rows(4)).is_err());
    }
}
