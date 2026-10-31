//! Device operands, every form's launches and graphs, and the bitwise
//! comparison with the split route.
use super::catalog::{self, Shape};
use bundle_deepseek41::sm120::fold_cubin_for_symbol;
use kernel_core::backend::{Backend, Counts, Launch, Need, Prepared, View};
use kernel_core::codec::f32_to_bf16;
use kernel_core::precision::buffers::{ProjectionBuffers, ProjectionScratch, QuantizedWeights};
use kernel_harness::kit::cases::f32_projection as cases;
use kernel_harness::{LabResult, Session, launch, poison};
use kernels_sm120::backend::{F32ProjectionOffers, Sm120};
use kernels_sm120::f32_projection::{Entry, cached_spans, rows};
use runtime_cuda::{DeviceBuffer, GraphExec, Module, Stream};
use std::collections::BTreeMap;

pub fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// One form a verify step's projection can take at some row counts.
#[derive(Clone, Debug)]
pub enum Form {
    /// The served backend op of the registered entry, with its decode
    /// reuse, last-producer and split routes.
    Served,
    /// The ordered split route alone, the reference.
    Split,
    /// The separate-span cached stages, 33 to 64 rows.
    Spans,
    /// The drafter's bounded cached reuse (60-row capacity).
    Bounded,
    /// The cached decode reuse at up to 32 rows.
    Reuse,
    /// A whole-K row tile.
    Rows(Box<rows::Candidate>),
    /// A row tile the bundle registers (`f32_verify_rows`), launched from
    /// the packaged cubin at the rows its tuning row serves.
    Packed(Box<rows::Candidate>, [u32; 2]),
}

impl Form {
    pub fn name(&self) -> String {
        match self {
            Self::Served => "served".into(),
            Self::Split => "split".into(),
            Self::Spans => "spans".into(),
            Self::Bounded => "bounded_reuse_m60".into(),
            Self::Reuse => "reuse32".into(),
            Self::Rows(c) | Self::Packed(c, _) => c
                .symbol()
                .split_once("_rows_")
                .map(|(_, tag)| {
                    let tag = tag.trim_end_matches("_v1");
                    if matches!(self, Self::Packed(..)) {
                        let site = if c.symbol().contains("_decode_f32_") {
                            "decode_"
                        } else {
                            ""
                        };
                        format!("packed_{site}rows_{tag}")
                    } else {
                        format!("rows_{tag}")
                    }
                })
                .expect("row tile symbol"),
        }
    }
    /// Whether the form serves `rows` rows of `s`.
    pub fn admits(&self, s: Shape, rows: u32) -> bool {
        match self {
            Self::Served | Self::Split => true,
            Self::Spans => catalog::spans(s).is_some() && (33..=64).contains(&rows),
            Self::Bounded => s.name == "head" && (33..=60).contains(&rows),
            Self::Reuse => {
                s.name == "head"
                    && (9..=32).contains(&rows)
                    && kernels_sm120::gemm::dense_decode::supported_reuse_rows(rows)
            }
            Self::Rows(c) => rows <= c.capacity(),
            Self::Packed(c, served) => {
                (served[0]..=served[1]).contains(&rows) && rows <= c.capacity()
            }
        }
    }
}

/// The form a Flash TP4 verify step of `rows` rows serves today: the
/// backend op through 32 rows, the cached spans at 33 to 64 and the split
/// route above.
pub fn current(s: Shape, rows: u32) -> Form {
    if rows > 32 && rows <= 64 && catalog::spans(s).is_some() {
        Form::Spans
    } else {
        Form::Served
    }
}

pub fn prepared(entry: &Entry, rows: u32) -> Prepared<ProjectionBuffers<View>> {
    let backend = Sm120::new().with_f32_projections(
        F32ProjectionOffers::new(
            vec![entry.clone()],
            entry.runtime_symbols().into_iter().collect(),
            "0".repeat(64),
        )
        .unwrap(),
    );
    backend
        .projection(&cases::operation(entry), entry.usage(), Need::rows(rows))
        .unwrap()
        .handle
}

fn bf16_weight_code(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0xffff == 0 {
        (bits >> 16) as u16
    } else {
        f32_to_bf16(value)
    }
}

/// Every module a form launches from: the bundle's cubins and the lab's.
pub struct Modules {
    modules: Vec<Module>,
}
impl Modules {
    pub fn load(s: &Session, lab: &'static [u8], shape: Shape) -> LabResult<Self> {
        let mut modules = vec![s.load_cubin(lab)?];
        let mut symbols = catalog::served(shape).runtime_symbols();
        symbols.extend(catalog::split(shape).runtime_symbols());
        if let Some(c) = catalog::spans(shape) {
            symbols.extend(c.symbols());
        }
        if shape.name == "head" {
            symbols.extend(bounded(shape).runtime_symbols());
        }
        for symbol in symbols {
            if modules.iter().all(|m| m.function(&symbol).is_err()) {
                let (_, bytes) = fold_cubin_for_symbol(&symbol)
                    .ok_or_else(|| format!("no bundle cubin holds {symbol}"))?;
                modules.push(s.load_cubin(bytes)?);
            }
        }
        for c in catalog::candidates(shape) {
            modules[0]
                .function(&c.symbol())
                .map_err(err)?
                .set_max_dynamic_shared_memory(c.tile().shared_bytes())
                .map_err(err)?;
        }
        for (c, _) in catalog::packed(shape) {
            for symbol in c.symbols() {
                if modules.iter().all(|m| m.function(&symbol).is_err()) {
                    let (_, bytes) = fold_cubin_for_symbol(&symbol)
                        .ok_or_else(|| format!("no bundle cubin holds {symbol}"))?;
                    modules.push(s.load_cubin(bytes)?);
                }
            }
            modules
                .iter()
                .find_map(|m| m.function(&c.symbol()).ok())
                .ok_or("missing packed row tile")?
                .set_max_dynamic_shared_memory(c.tile().shared_bytes())
                .map_err(err)?;
        }
        Ok(Self { modules })
    }
    pub fn run(
        &self,
        s: &Session,
        bindings: &launch::Bindings<'_>,
        launches: &[Launch],
        stream: &Stream,
    ) -> LabResult<()> {
        for l in launches {
            let function = self
                .modules
                .iter()
                .find_map(|m| m.function(&l.symbol).ok())
                .ok_or_else(|| format!("missing entry {}", l.symbol))?;
            launch::launch(
                &s.context,
                &function,
                stream,
                l.plan,
                &l.arguments,
                bindings,
            )?;
        }
        Ok(())
    }
}

/// The drafter's bounded reuse route of the head at the 60-row capacity.
fn bounded(s: Shape) -> Entry {
    catalog::split(s).with_bounded_reuse(vec![60]).unwrap()
}
fn bounded_at(s: Shape, rows: u32) -> Entry {
    let rows = if rows == 60 { vec![60] } else { vec![rows, 60] };
    catalog::split(s).with_bounded_reuse(rows).unwrap()
}

/// A shape's immutable operands: BF16 weights and their cached domain
/// table, classified once by the split route's create-time stage.
pub struct Weights {
    pub shape: Shape,
    pub weight: DeviceBuffer,
    pub domain: DeviceBuffer,
}
impl Weights {
    pub fn new(s: &Session, modules: &Modules, shape: Shape, b: &[f32]) -> LabResult<Self> {
        let codes: Vec<u16> = b.iter().copied().map(bf16_weight_code).collect();
        let weight = s.upload_bf16(&codes)?;
        let split = catalog::split(shape);
        let domain = s.allocate(split.weight_domain_bytes() as usize)?;
        let bindings = launch::Bindings {
            buffers: BTreeMap::from([(2, &weight), (3, &domain)]),
        };
        let view = |h| View {
            handle: h,
            bytes: bindings.buffers[&h].byte_len() as u64,
        };
        let buffers = projection(view(2), view(2), view(2), view(2), view(3));
        let init = prepared(&split, 1).initialize(&buffers).map_err(err)?;
        modules.run(s, &bindings, &init, &s.stream)?;
        s.stream.synchronize().map_err(err)?;
        let flags = poison::read(s, &domain)?;
        let tiles = b.len() / (shape.k as usize * 16);
        for tile in [0, tiles / 2, tiles - 1] {
            let expected = cases::weight_domains(
                &b[tile * shape.k as usize * 16..(tile + 1) * shape.k as usize * 16],
                shape.k as usize,
                kernel_core::contract::WeightFormat::Bf16,
            )[0];
            let actual = u32::from_le_bytes(flags[tile * 4..tile * 4 + 4].try_into().unwrap());
            if actual != expected {
                return Err(format!(
                    "{}: weight domain differs at tile {tile}",
                    shape.name
                ));
            }
        }
        Ok(Self {
            shape,
            weight,
            domain,
        })
    }
}

fn projection(
    activation: View,
    weight: View,
    output: View,
    live: View,
    domain: View,
) -> ProjectionBuffers<View> {
    ProjectionBuffers {
        activation,
        weights: QuantizedWeights {
            codes: weight,
            local_scales: None,
            global_scales: None,
        },
        output,
        live_rows: live,
        residual: None,
        activation_global: None,
        routing: None,
        scratch: ProjectionScratch {
            weight_domain: Some(domain),
            partial: View::EMPTY,
            semaphores: View::EMPTY,
            unused: View::EMPTY,
            temporary: None,
            gathered: None,
            codes: None,
            scales: None,
            activation_global: None,
            row_amax: None,
            output_codes: None,
            output_scales: None,
            quantizer_status: None,
            quantizer_status_row_offset: 0,
        },
        row_offset: 0,
        shard_rows: 0,
    }
}

/// One form's private output and workspaces.
struct Planes {
    output: DeviceBuffer,
    workspace: DeviceBuffer,
    extra: DeviceBuffer,
}

/// Every form of one shape at one planned row count over shared operands.
pub struct Step<'w> {
    pub weights: &'w Weights,
    pub rows: u32,
    pub forms: Vec<Form>,
    activation: DeviceBuffer,
    pub live: DeviceBuffer,
    planes: Vec<Planes>,
    launches: Vec<Vec<Launch>>,
}

const ACTIVATION: u64 = 1;
const WEIGHT: u64 = 2;
const DOMAIN: u64 = 3;
const LIVE: u64 = 4;
const COPIES: u64 = 1 << 20;
fn handles(form: usize) -> [u64; 3] {
    let base = 16 + 3 * form as u64;
    [base, base + 1, base + 2]
}

impl<'w> Step<'w> {
    pub fn new(
        s: &Session,
        weights: &'w Weights,
        rows: u32,
        a: &[f32],
        forms: Vec<Form>,
    ) -> LabResult<Self> {
        let shape = weights.shape;
        let q = catalog::split(shape).query();
        let activation = s.upload_f32(&a[..(rows * q.k) as usize])?;
        let live = s.allocate(4)?;
        live.copy_from_bytes(&rows.to_le_bytes(), &s.stream)
            .map_err(err)?;
        let mut planes = Vec::new();
        for form in &forms {
            let (workspace, extra) = match form {
                Form::Served => (catalog::served(shape).runtime_workspace_bytes(rows), 16),
                Form::Split => (catalog::split(shape).runtime_workspace_bytes(rows), 16),
                Form::Spans => (
                    cached_spans::DOMAIN_BYTES,
                    catalog::spans(shape)
                        .unwrap()
                        .partial_bytes(rows)
                        .map_err(err)?,
                ),
                Form::Bounded => (bounded_at(shape, rows).runtime_workspace_bytes(rows), 16),
                Form::Reuse => (reuse(shape, rows).runtime_workspace_bytes(rows), 16),
                Form::Rows(c) | Form::Packed(c, _) => (c.workspace_bytes(rows).map_err(err)?, 16),
            };
            planes.push(Planes {
                output: s.allocate((rows * q.n * 4) as usize)?,
                workspace: s.allocate(workspace.max(16) as usize)?,
                extra: s.allocate(extra as usize)?,
            });
        }
        let mut step = Self {
            weights,
            rows,
            forms,
            activation,
            live,
            planes,
            launches: Vec::new(),
        };
        step.launches = (0..step.forms.len())
            .map(|f| step.plan(f))
            .collect::<LabResult<_>>()?;
        Ok(step)
    }

    pub fn bindings(&self) -> launch::Bindings<'_> {
        let mut buffers = BTreeMap::from([
            (ACTIVATION, &self.activation),
            (WEIGHT, &self.weights.weight),
            (DOMAIN, &self.weights.domain),
            (LIVE, &self.live),
        ]);
        for (f, p) in self.planes.iter().enumerate() {
            let [o, w, e] = handles(f);
            buffers.insert(o, &p.output);
            buffers.insert(w, &p.workspace);
            buffers.insert(e, &p.extra);
        }
        launch::Bindings { buffers }
    }

    fn plan(&self, f: usize) -> LabResult<Vec<Launch>> {
        let bindings = self.bindings();
        let view = |h| View {
            handle: h,
            bytes: bindings.buffers[&h].byte_len() as u64,
        };
        let [o, w, e] = handles(f);
        let rows = self.rows;
        let shape = self.weights.shape;
        let op_launches = |entry: &Entry| {
            prepared(entry, rows)
                .launches(
                    &projection(
                        view(ACTIVATION),
                        view(WEIGHT),
                        view(o),
                        view(LIVE),
                        view(DOMAIN),
                    ),
                    view(w),
                    Counts::rows(rows),
                )
                .map_err(err)
        };
        match &self.forms[f] {
            Form::Served => op_launches(&catalog::served(shape)),
            Form::Split => op_launches(&catalog::split(shape)),
            Form::Bounded => {
                let launches = op_launches(&bounded_at(shape, rows))?;
                if !launches[1].symbol.contains("_reuse32_m60_") {
                    return Err("bounded reuse did not select the 60-row entry".into());
                }
                Ok(launches)
            }
            Form::Reuse => {
                let launches = op_launches(&reuse(shape, rows))?;
                if !launches.iter().any(|l| l.symbol.ends_with("_reuse32_v1")) {
                    return Err("decode reuse did not select its entry".into());
                }
                Ok(launches)
            }
            Form::Spans => Ok(catalog::spans(shape)
                .unwrap()
                .stages(
                    cached_spans::Buffers {
                        activation: view(ACTIVATION),
                        weight: view(WEIGHT),
                        row_domain: view(w),
                        partials: view(e),
                        live_rows: view(LIVE),
                        weight_domain: view(DOMAIN),
                        output: view(o),
                    },
                    rows,
                )
                .map_err(err)?
                .into_launches()
                .into()),
            Form::Rows(c) | Form::Packed(c, _) => Ok(c
                .launches(
                    rows::Buffers {
                        activation: view(ACTIVATION),
                        weight: view(WEIGHT),
                        workspace: view(w),
                        live_rows: view(LIVE),
                        weight_domain: view(DOMAIN),
                        output: view(o),
                    },
                    rows,
                )
                .map_err(err)?),
        }
    }

    pub fn launches(&self, f: usize) -> &[Launch] {
        &self.launches[f]
    }

    pub fn graph(&self, s: &Session, modules: &Modules, f: usize) -> LabResult<GraphExec> {
        let bindings = self.bindings();
        s.capture(|stream| modules.run(s, &bindings, &self.launches[f], stream))
    }

    /// A graph of form `f` once per weight copy in order, each call
    /// reading its own copy, as a step's layers read their own weights.
    pub fn repeated(
        &self,
        s: &Session,
        modules: &Modules,
        f: usize,
        copies: &[&DeviceBuffer],
    ) -> LabResult<GraphExec> {
        let mut bindings = self.bindings();
        for (c, copy) in copies.iter().enumerate() {
            bindings.buffers.insert(COPIES + c as u64, copy);
        }
        let launches: Vec<Launch> = (0..copies.len())
            .flat_map(|c| {
                self.launches[f].iter().cloned().map(move |mut l| {
                    for a in &mut l.arguments.0 {
                        if *a == kernel_core::contract::Argument::DeviceBuffer(WEIGHT) {
                            *a = kernel_core::contract::Argument::DeviceBuffer(COPIES + c as u64);
                        }
                    }
                    l
                })
            })
            .collect();
        s.capture(|stream| modules.run(s, &bindings, &launches, stream))
    }

    pub fn run(&self, s: &Session, modules: &Modules, f: usize) -> LabResult<()> {
        modules.run(s, &self.bindings(), &self.launches[f], &s.stream)
    }

    pub fn output(&self, s: &Session, f: usize) -> LabResult<Vec<u8>> {
        poison::read(s, &self.planes[f].output)
    }

    pub fn poison(&self, s: &Session) -> LabResult<()> {
        let buffers: Vec<_> = self
            .planes
            .iter()
            .flat_map(|p| [&p.output, &p.workspace, &p.extra])
            .collect();
        poison::poison(s, &buffers)
    }

    pub fn set_live(&self, s: &Session, live: u32) -> LabResult<()> {
        self.live
            .copy_from_bytes(&live.to_le_bytes(), &s.stream)
            .map_err(err)
    }

    /// Every form's output equals the reference form's (index 0) bit for
    /// bit over the live rows, eagerly and in a graph replay, at the
    /// planned rows, one fewer and zero live rows; rows past the live
    /// count stay poisoned.
    pub fn compare(&self, s: &Session, modules: &Modules, label: &str) -> LabResult<usize> {
        let graphs = (0..self.forms.len())
            .map(|f| self.graph(s, modules, f))
            .collect::<LabResult<Vec<_>>>()?;
        let n = self.weights.shape.n as usize;
        let mut compared = 0;
        for live in [self.rows, self.rows - 1, 0] {
            self.set_live(s, live)?;
            for graph in [false, true] {
                self.poison(s)?;
                for (f, exec) in graphs.iter().enumerate() {
                    if graph {
                        exec.launch(&s.stream).map_err(err)?;
                    } else {
                        self.run(s, modules, f)?;
                    }
                }
                let reference = self.output(s, 0)?;
                let written = live as usize * n * 4;
                for f in 0..self.forms.len() {
                    let actual = self.output(s, f)?;
                    if actual[..written] != reference[..written] {
                        let at = actual[..written]
                            .chunks_exact(4)
                            .zip(reference[..written].chunks_exact(4))
                            .position(|(a, b)| a != b)
                            .unwrap();
                        return Err(format!(
                            "{label}: {} rows={} live={live} graph={graph} differs from {} at row {} feature {}: {:?} != {:?}",
                            self.forms[f].name(),
                            self.rows,
                            self.forms[0].name(),
                            at / n,
                            at % n,
                            &actual[at * 4..at * 4 + 4],
                            &reference[at * 4..at * 4 + 4]
                        ));
                    }
                    if actual[written..].iter().any(|&b| b != poison::POISON) {
                        return Err(format!(
                            "{label}: {} wrote rows past the live count {live}",
                            self.forms[f].name()
                        ));
                    }
                    compared += 1;
                }
            }
        }
        self.set_live(s, self.rows)?;
        Ok(compared)
    }
}

fn reuse(s: Shape, rows: u32) -> Entry {
    catalog::split(s).with_decode_reuse(vec![rows]).unwrap()
}

/// `witness`'s operands of `s` at the verify capacity: rows of F32
/// activations and BF16-exact weights.
pub fn operands(s: Shape, witness: cases::Witness) -> (Vec<f32>, Vec<f32>) {
    cases::operands(&catalog::lab_entry(s), witness)
}

/// A deterministic real-valued pattern with a late subnormal that forces
/// one row's SIMT chains, and a weight tile outside the tensor domain.
pub fn mixed(s: Shape) -> (Vec<f32>, Vec<f32>) {
    let (mut a, mut b) = cases::operands(&catalog::lab_entry(s), cases::Witness::General);
    let k = s.k as usize;
    a[5 * k + k - 1] = f32::from_bits(1);
    b[17 * k + 3] = 2_f32.powi(-40);
    (a, b)
}
