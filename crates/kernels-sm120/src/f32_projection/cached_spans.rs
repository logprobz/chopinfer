//! Cached exact-F32 stages with independent row-domain and partial spans.
use super::{Entry, pointer};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::{Launch, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, WeightFormat};

pub const CAPACITY: u32 = 64;
pub const DOMAIN_BYTES: u64 = CAPACITY as u64 * 4;

#[derive(Clone, Debug)]
pub struct Candidate {
    entry: Entry,
}

/// The caller checks physical span alignment and disjointness before binding
/// these opaque views. Domain and partial spans require 256-byte alignment.
#[derive(Clone, Copy)]
pub struct Buffers {
    pub activation: View,
    pub weight: View,
    pub row_domain: View,
    pub partials: View,
    pub live_rows: View,
    pub weight_domain: View,
    pub output: View,
}

/// Three validated stages. A fused consumer replaces classifier and reducer
/// explicitly while retaining the producer's argument order and row stride.
#[derive(Clone, Debug)]
pub struct CachedProjectionStages {
    classifier: Launch,
    producer: Launch,
    reducer: Launch,
    row_domain: View,
    partials: View,
}

impl CachedProjectionStages {
    pub fn row_domain(&self) -> View {
        self.row_domain
    }
    pub fn partials(&self) -> View {
        self.partials
    }
    pub fn into_launches(self) -> [Launch; 3] {
        [self.classifier, self.producer, self.reducer]
    }
    pub fn into_producer(self, rows: u32) -> Result<Launch, Refusal> {
        if rows != self.producer.rows {
            return Err(Refusal(
                "cached producer row stride differs from the consumer",
            ));
        }
        Ok(self.producer)
    }
}

impl Candidate {
    pub fn new(entry: Entry) -> Result<Self, Refusal> {
        if !entry.uses_cached_weights()
            || entry.uses_fused_split()
            || entry.uses_prefill_tiles()
            || entry.query().weight != WeightFormat::Bf16
            || entry.query().m_capacity < CAPACITY
        {
            return Err(Refusal(
                "separate cached stages require BF16 weights and capacity 64",
            ));
        }
        entry.decode_plan(1)?;
        Ok(Self { entry })
    }
    pub fn entry(&self) -> &Entry {
        &self.entry
    }
    pub fn symbols(&self) -> [String; 3] {
        ["rows", "compute", "reduce"].map(|stage| {
            format!(
                "{}_spans_m64_{stage}_v1",
                self.entry.stage_symbol("_decode")
            )
        })
    }
    pub fn partial_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        if !(33..=CAPACITY).contains(&rows) {
            return Err(Refusal(
                "separate cached stages require 33 to 64 planned rows",
            ));
        }
        let q = self.entry.query();
        u64::from(rows)
            .checked_mul(u64::from(q.n))
            .and_then(|v| v.checked_mul(u64::from(q.k.div_ceil(128))))
            .and_then(|v| v.checked_mul(4))
            .ok_or(Refusal("cached partial span overflows"))
    }
    pub fn plans(&self, rows: u32) -> Result<[LaunchPlan; 3], Refusal> {
        self.partial_bytes(rows)?;
        let q = self.entry.query();
        let elements = rows
            .checked_mul(q.n)
            .ok_or(Refusal("cached reduction grid overflows"))?;
        Ok([
            self.entry.stage_plan([rows, 1, 1])?,
            self.entry.stage_plan([
                q.n.div_ceil(16),
                rows.div_ceil(32),
                q.k.div_ceil(128).div_ceil(4),
            ])?,
            self.entry.stage_plan([elements.div_ceil(128), 1, 1])?,
        ])
    }
    pub fn specs(&self) -> [EntrySpec; 3] {
        let q = self.entry.query();
        let [classify, compute, reduce] = self.symbols();
        let rows = EntryParam::U32 {
            name: "rows".into(),
        };
        let live = pointer("live_rows", "const uint32_t");
        [
            EntrySpec::new(
                classify,
                vec![
                    pointer("activation", "const float"),
                    live.clone(),
                    rows.clone(),
                    pointer("row_domain", "uint32_t"),
                ],
                format!(
                    "chop::exact_f32::classify_rows<{}>(activation,live_rows,rows,row_domain);",
                    q.k
                ),
                128,
                1,
            ),
            EntrySpec::new(
                compute,
                vec![
                    pointer("activation", "const float"),
                    pointer("weight", "const uint16_t"),
                    pointer("row_domain", "const uint32_t"),
                    pointer("partials", "float"),
                    live.clone(),
                    rows.clone(),
                    pointer("weight_domain", "const uint32_t"),
                ],
                // The reusable task has no wait. No mutable input may be read
                // until the producer that classified these rows has completed.
                format!(
                    "__shared__ chop::exact_f32_reuse::Arena arena;const chop::exact_f32_reuse::Task task{{blockIdx.x,blockIdx.y,blockIdx.z}};chop::ptx::griddep_launch_dependents();chop::ptx::griddep_wait();chop::exact_f32_reuse::task<{},{},false>(task,arena,activation,weight,partials,live_rows,rows,weight_domain,row_domain);",
                    q.n, q.k
                ),
                128,
                1,
            ),
            EntrySpec::new(
                reduce,
                vec![
                    pointer("partials", "const float"),
                    pointer("output", "float"),
                    live,
                    rows,
                ],
                format!(
                    "chop::exact_f32::reduce<{},{}>(partials,output,live_rows,rows);",
                    q.n, q.k
                ),
                128,
                1,
            ),
        ]
    }
    pub fn stages(&self, b: Buffers, rows: u32) -> Result<CachedProjectionStages, Refusal> {
        let q = self.entry.query();
        b.activation.covers(u64::from(rows) * u64::from(q.k) * 4)?;
        b.weight.covers(self.entry.weight_bytes())?;
        b.row_domain.covers(DOMAIN_BYTES)?;
        b.partials.covers(self.partial_bytes(rows)?)?;
        b.live_rows.covers(4)?;
        b.weight_domain.covers(self.entry.weight_domain_bytes())?;
        b.output.covers(u64::from(rows) * u64::from(q.n) * 4)?;
        let views = [
            b.activation,
            b.weight,
            b.row_domain,
            b.partials,
            b.live_rows,
            b.weight_domain,
            b.output,
        ];
        for (i, a) in views.iter().enumerate() {
            if views[i + 1..].iter().any(|v| v.handle == a.handle) {
                return Err(Refusal("separate cached views alias"));
            }
        }
        let ptr = |v: View| Argument::DeviceBuffer(v.handle);
        let args = [
            vec![
                ptr(b.activation),
                ptr(b.live_rows),
                Argument::U32(rows),
                ptr(b.row_domain),
            ],
            vec![
                ptr(b.activation),
                ptr(b.weight),
                ptr(b.row_domain),
                ptr(b.partials),
                ptr(b.live_rows),
                Argument::U32(rows),
                ptr(b.weight_domain),
            ],
            vec![
                ptr(b.partials),
                ptr(b.output),
                ptr(b.live_rows),
                Argument::U32(rows),
            ],
        ];
        let calls = self
            .symbols()
            .into_iter()
            .zip(self.plans(rows)?)
            .zip(args)
            .map(|((symbol, plan), args)| Launch {
                symbol,
                rows,
                plan,
                arguments: ArgumentList(args),
                collective: false,
                persistent: false,
            })
            .collect::<Vec<_>>();
        let [classifier, producer, reducer]: [Launch; 3] =
            calls.try_into().expect("three named stages");
        Ok(CachedProjectionStages {
            classifier,
            producer,
            reducer,
            row_domain: b.row_domain,
            partials: b.partials,
        })
    }
}

pub fn cubin_spec(name: &str, candidates: &[Candidate]) -> Result<CubinSpec, Refusal> {
    if candidates.is_empty() {
        return Err(Refusal("separate cached stages require entries"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/exact_f32_reuse.cuh\"".into()],
        defines: vec![],
        entries: candidates.iter().flat_map(Candidate::specs).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery};
    use kernel_core::precision::{ExecutionPhase, OperationUse};
    fn candidate() -> Candidate {
        Candidate::new(
            Entry::new(
                GemmQuery {
                    m_capacity: 64,
                    n: 384,
                    k: 5120,
                    weight: WeightFormat::Bf16,
                    activation: ActivationFormat::F32,
                    output: DType::F32,
                    epilogues: Epilogues::default(),
                    grouped: None,
                },
                OperationUse {
                    phase: ExecutionPhase::Verify,
                    capacity: 64,
                    graph: true,
                },
                "chop_test_cached_spans",
            )
            .unwrap()
            .with_decode(),
        )
        .unwrap()
    }
    fn buffers(c: &Candidate, rows: u32) -> Buffers {
        let v = |handle, bytes| View { handle, bytes };
        Buffers {
            activation: v(1, u64::from(rows) * 5120 * 4),
            weight: v(2, c.entry.weight_bytes()),
            row_domain: v(3, 256),
            partials: v(4, c.partial_bytes(rows).unwrap()),
            live_rows: v(5, 4),
            weight_domain: v(6, 96),
            output: v(7, u64::from(rows) * 384 * 4),
        }
    }
    #[test]
    fn stages_preserve_explicit_planes_and_planned_stride() {
        let c = candidate();
        for rows in 33..=64 {
            assert_eq!(c.partial_bytes(rows).unwrap(), 61440 * u64::from(rows));
            let b = buffers(&c, rows);
            let s = c.stages(b, rows).unwrap();
            assert_eq!(s.row_domain(), b.row_domain);
            assert_eq!(s.partials(), b.partials);
            assert!(s.clone().into_producer(rows - 1).is_err());
            let p = s.into_producer(rows).unwrap();
            assert_eq!(p.plan.grid, [24, 2, 10]);
            assert!(p.plan.pdl);
            assert_eq!(
                p.arguments.0,
                vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                    Argument::DeviceBuffer(4),
                    Argument::DeviceBuffer(5),
                    Argument::U32(rows),
                    Argument::DeviceBuffer(6)
                ]
            );
            let mut short = b;
            short.partials.bytes -= 1;
            assert!(c.stages(short, rows).is_err());
            let mut short = b;
            short.row_domain.bytes = 128;
            assert!(c.stages(short, rows).is_err());
            let mut alias = b;
            alias.partials.handle = alias.row_domain.handle;
            assert!(c.stages(alias, rows).is_err());
        }
        for rows in [0, 1, 32, 65, 128] {
            assert!(c.plans(rows).is_err());
        }
    }
    #[test]
    fn producer_waits_before_the_unmodified_reuse_task() {
        let c = candidate();
        let [classify, compute, reduce] = c.specs();
        assert!(classify.body.contains("classify_rows<5120>"));
        let wait = compute.body.find("griddep_wait()").unwrap();
        assert!(compute.body.find("griddep_launch_dependents()").unwrap() < wait);
        assert!(wait < compute.body.find("exact_f32_reuse::task<").unwrap());
        assert!(compute.body.ends_with("weight_domain,row_domain);"));
        assert_eq!(
            reduce.body,
            "chop::exact_f32::reduce<384,5120>(partials,output,live_rows,rows);"
        );
        assert!(!compute.body.contains("workspace+"));
        assert_eq!(compute.params.len(), 7);
        assert!(
            c.entry
                .entry_specs()
                .iter()
                .any(|s| s.body.contains("workspace+32"))
        );
    }
}
