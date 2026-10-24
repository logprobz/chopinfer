//! The current rank-local FP8 MoE graph and authenticated operands for lab pairs.
use super::*;
use kernel_core::contract::{InstantiationSpec, StaticQuery, WeightFormat};
use kernel_core::row_copy::ColumnSlice;
use kernel_lab::launch::Bindings;
use kernels_sm120::gemm::{self, GemmBuffers};
use kernels_sm120::precision::Compute;
use kernels_sm120::row::{copy, precision::LayoutCopy};
use runtime_cuda::{Event, Module};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

fn plain_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!(parts.next(), Some(std::path::Component::Normal(_))) && parts.next().is_none()
}

fn read(root: &Path, name: &str, bytes: usize) -> LabResult<Vec<u8>> {
    if !plain_name(name) {
        return Err("operand must be a regular direct child".into());
    }
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(root.join(name))
        .map_err(err)?;
    let metadata = file.metadata().map_err(err)?;
    if !metadata.is_file() || metadata.len() != bytes as u64 {
        return Err(format!("{name}: expected {bytes}, got {}", metadata.len()));
    }
    let mut value = Vec::with_capacity(bytes);
    file.read_to_end(&mut value).map_err(err)?;
    if value.len() != bytes {
        return Err("operand changed while reading".into());
    }
    Ok(value)
}

#[test]
fn fp8_moe_refuses_operand_path_traversal() {
    for name in [
        "",
        ".",
        "..",
        "../input.bin",
        "/input.bin",
        "case/input.bin",
    ] {
        assert!(!plain_name(name));
    }
    assert!(plain_name("input.bin"));
}

fn cases() -> LabResult<(PathBuf, Vec<(String, u32)>)> {
    let root = PathBuf::from(
        std::env::var_os("CHOPINFER_FP8_MOE_CAPTURE").ok_or("missing direct capture")?,
    );
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).map_err(err)?)
            .map_err(err)?;
    if value["program"] != "Decode" || value["bitwise_comparisons"] != 4 {
        return Err("requires actual AR captures with four output comparisons".into());
    }
    let cases = value["cases"]
        .as_array()
        .ok_or("missing cases")?
        .iter()
        .map(|v| {
            let name = v["name"].as_str().ok_or("missing case name")?;
            let rows = v["rows"].as_u64().ok_or("missing rows")? as u32;
            if !plain_name(name) || ![8, 32].contains(&rows) {
                return Err("invalid direct case".into());
            }
            Ok((name.to_owned(), rows))
        })
        .collect::<LabResult<Vec<_>>>()?;
    if cases.len() != 2 || cases[0].1 != 8 || cases[1].1 != 32 || cases[0].0 == cases[1].0 {
        return Err("requires one C8 and one C32 case".into());
    }
    Ok((root, cases))
}

fn ids(root: &Path) -> LabResult<Vec<usize>> {
    let bytes = read(root, "ids.bin", 32 * TOP_K * 4)?;
    let ids = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect::<Vec<_>>();
    for row in ids.chunks_exact(TOP_K) {
        if row.iter().any(|&e| e >= EXPERTS)
            || row.iter().collect::<std::collections::BTreeSet<_>>().len() != TOP_K
        {
            return Err("invalid captured expert IDs".into());
        }
    }
    Ok(ids)
}

#[test]
#[ignore = "requires CHOPINFER_FP8_MOE_CAPTURE and CHOPINFER_FP8_MOE_CUBIN"]
fn fp8_moe_capture_validates_on_cpu() {
    projection_bytes().unwrap();
    for rows in [8, 16, 32] {
        for router in [false, true] {
            projection_spec(rows, router).unwrap();
        }
    }
    let (root, cases) = cases().unwrap();
    for (name, _) in cases {
        let root = root.join(name);
        let ids = ids(&root).unwrap();
        for (name, bytes) in [
            ("input.bin", 32 * 4096 * 2),
            ("routes.bin", 32 * 8 * 4),
            ("router.bin", 320 * 4096 * 2),
            ("bias.bin", 288 * 4),
            ("shared-gate.bin", 1024 * 4096),
            ("shared-gate-scales.bin", 1024),
            ("shared-down.bin", 4096 * 512),
            ("shared-down-scales.bin", 512),
        ] {
            read(&root, name, bytes).unwrap();
        }
        for expert in ids.into_iter().collect::<std::collections::BTreeSet<_>>() {
            for (tag, bytes) in [
                ("gate", 1024 * 4096),
                ("gate-scales", 1024),
                ("down", 4096 * 512),
                ("down-scales", 512),
            ] {
                read(&root, &format!("e{expert}-{tag}.bin"), bytes).unwrap();
            }
        }
    }
}

/// Read the exact served module without linking a second bundle C ABI.
struct ServedProjections(Module);

fn projection_bytes() -> LabResult<Vec<u8>> {
    let path = PathBuf::from(
        std::env::var_os("CHOPINFER_FP8_MOE_CUBIN").ok_or("missing served projection cubin")?,
    );
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&path)
        .map_err(err)?;
    let size = file.metadata().map_err(err)?.len();
    if size > 128 << 20 {
        return Err("projection cubin exceeds its limit".into());
    }
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("invalid cubin name")?;
    let bytes = read(
        path.parent().ok_or("missing cubin parent")?,
        name,
        size as usize,
    )?;
    let root = path.parent().ok_or("missing cubin parent")?;
    let binding = read_json(root, "bindings.json")?;
    let lock_bytes = read_sized(root, "projections.lock", 16 << 20)?;
    if format!("{:x}", Sha256::digest(&lock_bytes))
        != binding["lock_sha256"]
            .as_str()
            .ok_or("missing lock digest")?
    {
        return Err("projection lock differs from the model-owned bindings".into());
    }
    let lock: serde_json::Value = serde_json::from_slice(&lock_bytes).map_err(err)?;
    let record = lock
        .as_array()
        .ok_or("invalid projection lock")?
        .iter()
        .find(|r| r["name"] == binding["cubin_name"])
        .ok_or("missing served module lock")?;
    if format!("{:x}", Sha256::digest(&bytes))
        != record["cubin_sha256"]
            .as_str()
            .ok_or("missing cubin digest")?
    {
        return Err("served projection cubin differs from its unchanged lock".into());
    }
    Ok(bytes)
}

fn read_sized(root: &Path, name: &str, limit: usize) -> LabResult<Vec<u8>> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(root.join(name))
        .map_err(err)?;
    let size = file.metadata().map_err(err)?.len();
    if size > limit as u64 {
        return Err("staged metadata exceeds its limit".into());
    }
    read(root, name, size as usize)
}

fn read_json(root: &Path, name: &str) -> LabResult<serde_json::Value> {
    serde_json::from_slice(&read_sized(root, name, 16 << 20)?).map_err(err)
}

fn projection_spec(rows: u32, router: bool) -> LabResult<InstantiationSpec> {
    let path =
        PathBuf::from(std::env::var_os("CHOPINFER_FP8_MOE_CUBIN").ok_or("missing served cubin")?);
    let binding = read_json(
        path.parent().ok_or("missing cubin parent")?,
        "bindings.json",
    )?;
    let role = if router { "router" } else { "shared_down" };
    let wanted = binding["bindings"][rows.to_string()][role]
        .as_str()
        .ok_or("missing model-owned projection choice")?;
    let (offers, _, _) = kernel_lab::models::projection_inventory()?;
    let specs = offers
        .iter()
        .filter_map(|o| match &o.handle().compute {
            Compute::Dense(spec) if spec.symbol == wanted => Some(spec),
            _ => None,
        })
        .collect::<Vec<_>>();
    let first = *specs
        .first()
        .ok_or("projection choice is absent from the model-owned inventories")?;
    if specs.iter().any(|s| s.key != first.key) {
        return Err("projection symbol names different compile keys".into());
    }
    let StaticQuery::Gemm(q) = first.key.query else {
        return Err("GEMM query required".into());
    };
    let expected = if router {
        (320, 4096, WeightFormat::Bf16)
    } else {
        (
            4096,
            512,
            WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32),
        )
    };
    if (q.n, q.k, q.weight) != expected
        || q.m_capacity != rows
        || q.output != DType::F32
        || q.epilogues.residual_add == router
    {
        return Err("projection choice differs from the captured graph geometry".into());
    }
    Ok(first.clone())
}

/// The bundle's exact dense offer, with private scratch for each stream.
struct ProjectionCall {
    spec: InstantiationSpec,
    function: Function,
    output: DeviceBuffer,
    partial: DeviceBuffer,
    semaphores: DeviceBuffer,
    weights: DeviceBuffer,
    scales: Option<DeviceBuffer>,
    residual: Option<DeviceBuffer>,
    rows: u32,
}
impl ProjectionCall {
    fn new(session: &Session, root: &Path, rows: u32, router: bool) -> LabResult<Self> {
        let spec = projection_spec(rows, router)?;
        let StaticQuery::Gemm(query) = spec.key.query else {
            return Err("GEMM query required".into());
        };
        let plan = gemm::plan(&spec.key, gemm::row_counts(&spec.key, rows)).map_err(err)?;
        let module =
            session.state(|| Ok(ServedProjections(session.load_cubin(&projection_bytes()?)?)))?;
        let function = module.0.function(&spec.symbol).map_err(err)?;
        if plan.dynamic_shared_bytes >= 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                .map_err(err)?;
        }
        let bytes = query.n as usize * query.k as usize * if router { 2 } else { 1 };
        let weights = upload(
            session,
            &read(
                root,
                if router {
                    "router.bin"
                } else {
                    "shared-down.bin"
                },
                bytes,
            )?,
        )?;
        let scales = if router {
            None
        } else {
            Some(upload(
                session,
                &read(root, "shared-down-scales.bin", 512)?,
            )?)
        };
        let output = zeroed(session, rows as usize * query.n as usize * 4)?;
        let residual = query
            .epilogues
            .residual_add
            .then(|| zeroed(session, output.byte_len()))
            .transpose()?;
        Ok(Self {
            weights,
            scales,
            output,
            partial: zeroed(session, plan.workspace_bytes as usize)?,
            semaphores: zeroed(
                session,
                GemmBuffers::semaphore_capacity_bytes(&spec.key).map_err(err)? as usize,
            )?,
            residual,
            spec,
            function,
            rows,
        })
    }
    fn launch(&self, session: &Session, stream: &Stream, input: &DeviceBuffer) -> LabResult<()> {
        let key = &self.spec.key;
        let counts = gemm::row_counts(key, self.rows);
        let args = gemm::arguments(
            key,
            counts,
            gemm::tensor_maps(key, 1, 2).map_err(err)?,
            GemmBuffers {
                output: 3,
                partial: 4,
                semaphores: 5,
                weight_scales: self.scales.as_ref().map(|_| 6),
                activation_scales: None,
                residual: self.residual.as_ref().map(|_| 7),
                weight_global_scale: None,
                activation_global_scale: None,
            },
        )
        .map_err(err)?;
        let mut buffers = std::collections::BTreeMap::from([
            (1, &self.weights),
            (2, input),
            (3, &self.output),
            (4, &self.partial),
            (5, &self.semaphores),
        ]);
        if let Some(scales) = &self.scales {
            buffers.insert(6, scales);
        }
        if let Some(residual) = &self.residual {
            buffers.insert(7, residual);
        }
        launch::launch(
            &session.context,
            &self.function,
            stream,
            gemm::plan(key, counts).map_err(err)?,
            &args,
            &Bindings { buffers },
        )
    }
}

struct Context {
    layer: Layer,
    shared: Dense,
    router: ProjectionCall,
    shared_down: ProjectionCall,
    gate: Projection,
    gate_fn: Function,
    compact: LayoutCopy,
    compact_fn: Function,
    sum_fn: Function,
    side: Stream,
    fork: Event,
    join: Event,
    rows: u32,
}

fn route_for_rows(prepared: LocalRoute, rows: u32) -> LabResult<LocalRoute> {
    if rows == 0 || rows > prepared.capacity {
        return Err("join rows exceed the compiled route capacity".into());
    }
    // Capacity fixes the compiled scan's thread count. Only live_tokens shrinks.
    Ok(prepared)
}

fn retire_case<T>(case: T, mut fence: impl FnMut() -> LabResult<()>) -> LabResult<()> {
    // Graph nodes may use the side stream. Fence the context before graph and
    // buffer drops, then drain their queued frees before the next case allocates.
    if let Err(error) = fence() {
        // A failed fence does not authorize freeing buffers still used by GPU
        // work. This lab exits on failure; retain the case until process exit.
        std::mem::forget(case);
        return Err(error);
    }
    drop(case);
    fence()
}

#[test]
fn fp8_moe_retires_cross_stream_case_after_fence() {
    use std::cell::RefCell;
    use std::rc::Rc;
    struct Resource(Rc<RefCell<Vec<&'static str>>>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.borrow_mut().push("release");
        }
    }
    let events = Rc::new(RefCell::new(Vec::new()));
    retire_case(Resource(events.clone()), || {
        events.borrow_mut().push("fence");
        Ok(())
    })
    .unwrap();
    assert_eq!(&*events.borrow(), &["fence", "release", "fence"]);
}

#[test]
fn fp8_moe_failed_fence_retains_the_case() {
    use std::cell::Cell;
    use std::rc::Rc;
    struct Resource(Rc<Cell<bool>>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }
    let freed = Rc::new(Cell::new(false));
    assert!(retire_case(Resource(freed.clone()), || Err("pending work".into())).is_err());
    assert!(!freed.get());
}

#[test]
fn fp8_moe_preserves_compiled_route_capacity() {
    let compiled = split::SHAPES.routes()[0];
    for rows in [8, 16, 32] {
        let selected = route_for_rows(compiled, rows).unwrap();
        assert_eq!(selected.capacity, compiled.capacity);
        assert_eq!(
            selected.plan(RouteKernel::Decode, rows).unwrap().block,
            compiled.plan(RouteKernel::Decode, rows).unwrap().block
        );
        assert_eq!(
            selected.symbol(split::PREFIX, RouteKernel::Decode),
            compiled.symbol(split::PREFIX, RouteKernel::Decode)
        );
    }
    assert!(route_for_rows(compiled, 0).is_err());
    assert!(route_for_rows(compiled, 33).is_err());
}

impl Context {
    fn new(session: &Session, root: &Path, rows: u32) -> LabResult<Self> {
        let captured_ids = ids(root)?;
        let pool = captured_ids
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let layer = Layer::routed(
            session,
            &split::SHAPES,
            Weights::Fp8Block128,
            false,
            0,
            927,
            &pool,
            Some(pool_logits(32, &pool, &mut Rng(927))),
            Schedule::WarpStreams,
        )?;
        poison::write(
            session,
            &layer.x,
            0,
            &read(root, "input.bin", 32 * 4096 * 2)?,
        )?;
        poison::write(
            session,
            &layer.routes.bias,
            0,
            &read(root, "bias.bin", 288 * 4)?,
        )?;
        for &expert in &pool {
            for (tag, bank, n, k) in [
                ("gate", &layer.gate_up_bank, 1024, 4096),
                ("down", &layer.down_bank, 4096, 512),
            ] {
                poison::write(
                    session,
                    &bank.codes,
                    expert * n * k,
                    &read(root, &format!("e{expert}-{tag}.bin"), n * k)?,
                )?;
                let size = n / 128 * (k / 128) * 4;
                poison::write(
                    session,
                    bank.block_scales.as_ref().unwrap(),
                    expert * size,
                    &read(root, &format!("e{expert}-{tag}-scales.bin"), size)?,
                )?;
            }
        }
        layer.routes.set_live(session, rows)?;
        let shared = Dense::scheduled(
            session,
            Weights::Fp8Block128,
            512,
            false,
            913,
            Schedule::BlockWarps,
        )?;
        poison::write(
            session,
            &shared.gate_up_bank.codes,
            0,
            &read(root, "shared-gate.bin", 1024 * 4096)?,
        )?;
        poison::write(
            session,
            shared.gate_up_bank.block_scales.as_ref().unwrap(),
            0,
            &read(root, "shared-gate-scales.bin", 1024)?,
        )?;
        let gate = layer.gate_up.clone();
        let gate_fn = Entry::of(&gate).load(session)?;
        let compact = copy::slice(
            &ColumnSlice {
                dtype: DType::F32,
                width: 288,
                stride: 320,
                offset: 0,
                alignment_bytes: 16,
            },
            rows,
            copy::SLICE_SYMBOL,
        )
        .map_err(err)?;
        let compact_fn = session
            .module(copy::LAB_CUBIN)?
            .function(copy::SLICE_SYMBOL)
            .map_err(err)?;
        let sum_fn = session
            .module("chop_moe_chain_e288l288k8_sum_add")?
            .function(
                &layer
                    .sum
                    .with_output(RoutedSumOutput::F32)
                    .add_symbol(split::PREFIX),
            )
            .map_err(err)?;
        Ok(Self {
            layer,
            shared,
            router: ProjectionCall::new(session, root, rows, true)?,
            shared_down: ProjectionCall::new(session, root, rows, false)?,
            gate,
            gate_fn,
            compact,
            compact_fn,
            sum_fn,
            side: session.context.stream().map_err(err)?,
            fork: session.context.event().map_err(err)?,
            join: session.context.event().map_err(err)?,
            rows,
        })
    }
    fn shared_gate(&self, stream: &Stream) -> LabResult<()> {
        let b = KernelArgument::Buffer;
        let [codes, block, local, globals] = self.shared.gate_up_bank.arguments();
        launch_with(
            stream,
            &self.shared.fns[0],
            self.shared.gate_up.plan(self.rows).map_err(err)?,
            true,
            &[
                b(&self.layer.x),
                codes,
                block,
                local,
                globals,
                KernelArgument::NullBuffer,
                KernelArgument::NullBuffer,
                KernelArgument::NullBuffer,
                b(&self.layer.routes.live),
                KernelArgument::NullBuffer,
                b(&self.shared.inter),
            ],
        )
    }
    fn launch(
        &self,
        session: &Session,
        stream: &Stream,
        early: bool,
        omit_shared: bool,
    ) -> LabResult<()> {
        self.launch_with_gate(session, stream, early, omit_shared, |context, stream| {
            context.layer.launch_projection(
                stream,
                true,
                &context.gate,
                &context.gate_fn,
                context.rows,
                true,
            )
        })
    }
    fn launch_with_gate(
        &self,
        session: &Session,
        stream: &Stream,
        early: bool,
        omit_shared: bool,
        gate: impl FnOnce(&Self, &Stream) -> LabResult<()>,
    ) -> LabResult<()> {
        self.launch_with_projections(session, stream, early, omit_shared, gate, |c, s| {
            c.layer
                .launch_projection(s, false, &c.layer.down, &c.layer.fns[1], c.rows, true)
        })
    }
    fn launch_with_projections(
        &self,
        session: &Session,
        stream: &Stream,
        early: bool,
        omit_shared: bool,
        gate: impl FnOnce(&Self, &Stream) -> LabResult<()>,
        down: impl FnOnce(&Self, &Stream) -> LabResult<()>,
    ) -> LabResult<()> {
        self.fork.record(stream).map_err(err)?;
        self.side.wait_event(&self.fork).map_err(err)?;
        if !omit_shared {
            self.shared_gate(&self.side)?;
            self.shared_down
                .launch(session, &self.side, &self.shared.inter)?;
        }
        self.join.record(&self.side).map_err(err)?;
        self.router.launch(session, stream, &self.layer.x)?;
        launch::launch(
            &session.context,
            &self.compact_fn,
            stream,
            self.compact.stage().plan(self.rows).map_err(err)?,
            &self.compact.arguments(1, 2, 3).map_err(err)?,
            &Bindings {
                buffers: std::collections::BTreeMap::from([
                    (1, &self.router.output),
                    (2, &self.layer.routes.logits),
                    (3, &self.layer.routes.live),
                ]),
            },
        )?;
        let route = route_for_rows(self.layer.route, self.rows)?;
        self.layer
            .routes
            .launch(stream, route, &self.layer.route_fns, 0, true)?;
        if early {
            stream.wait_event(&self.join).map_err(err)?;
        }
        gate(self, stream)?;
        down(self, stream)?;
        if !early {
            stream.wait_event(&self.join).map_err(err)?;
        }
        let b = KernelArgument::Buffer;
        launch_with(
            stream,
            &self.sum_fn,
            self.layer.sum.plan(self.rows).map_err(err)?,
            true,
            &[
                b(&self.layer.routed),
                b(&self.layer.routes.ids),
                b(&self.layer.routes.live),
                KernelArgument::U32(0),
                b(&self.shared_down.output),
                b(&self.layer.out_f32),
            ],
        )
    }
    fn outputs(&self) -> Vec<&DeviceBuffer> {
        let mut out = self.layer.routes.outputs().to_vec();
        out.extend([
            &self.router.output,
            &self.layer.inter,
            &self.layer.routed,
            &self.shared.inter,
            &self.shared_down.output,
            &self.layer.out_f32,
        ]);
        out
    }
}

/// Preserve every timing sample before the next comparison starts. Session's
/// receipt drains the per-round records, including clocks and process checks.
fn time_graph_pair(
    session: &Session,
    label: &str,
    graphs: [&runtime_cuda::GraphExec; 2],
) -> LabResult<()> {
    let timing = kernel_lab::with_timing_label(label, || {
        kernel_lab::time_ab(
            session,
            8,
            24,
            [kernel_lab::eviction::EvictionMode::Read; 2],
            |arm, stream| graphs[arm].launch(stream).map_err(err),
        )
    })?;
    let path = session.receipt(
        "fp8-moe-pair",
        BTreeMap::from([
            ("operation".into(), serde_json::json!(label)),
            ("rounds".into(), serde_json::json!(24)),
            (
                "base_ms".into(),
                serde_json::json!(timing.arms[0].median_ms),
            ),
            (
                "candidate_ms".into(),
                serde_json::json!(timing.arms[1].median_ms),
            ),
            (
                "paired_ratio".into(),
                serde_json::json!(timing.paired_ratio),
            ),
        ]),
    )?;
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).map_err(err)?).map_err(err)?;
    let rounds = record["timing_rounds"]
        .as_array()
        .ok_or("timing receipt omitted its rounds")?;
    let samples = rounds
        .iter()
        .filter_map(|row| Some((row["round"].as_u64()?, row["arm"].as_u64()?)))
        .collect::<Vec<_>>();
    let wanted = (0..24)
        .flat_map(|round| [(round, 0), (round, 1)])
        .collect::<std::collections::BTreeSet<_>>();
    if samples.len() != 48
        || samples
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            != wanted
    {
        return Err("timing receipt omitted a paired sample".into());
    }
    if record["measurement_label"] == "shared-device" {
        return Err("the timing receipt reports another compute process".into());
    }
    eprintln!("FP8_MOE_RECEIPT {}", path.display());
    Ok(())
}

#[test]
#[ignore = "requires authenticated operands and an exclusive GPU"]
fn served_graph_control_on_gpu() {
    let (root, cases) = cases().unwrap();
    let time = std::env::var("CHOPINFER_FP8_MOE_TIME_CONTROL").unwrap_or_else(|_| "0".into());
    assert!(["0", "1"].contains(&time.as_str()));
    gpu_test(8 << 30, |session| {
        let mut checks = 0;
        for (name, rows) in &cases {
            let c = Context::new(session, &root.join(name), *rows)?;
            let mut graphs = Vec::new();
            let result = (|| -> LabResult<()> {
                let outputs = c.outputs();
                let expected = poisoned_outputs(session, &outputs, || {
                    c.launch(session, &session.stream, false, false)
                })?;
                for _ in 0..2 {
                    graphs.push(session.capture(|s| c.launch(session, s, false, false))?);
                }
                for graph in &graphs {
                    for _ in 0..2 {
                        let actual = poisoned_outputs(session, &outputs, || {
                            graph.launch(&session.stream).map_err(err)
                        })?;
                        same_bytes("current FP8 MoE control", &expected, &actual)?;
                        checks += 1;
                    }
                }
                if time == "1" {
                    time_graph_pair(
                        session,
                        &format!("{name}/m{rows}/AA"),
                        [&graphs[0], &graphs[1]],
                    )?;
                }
                Ok(())
            })();
            let retired = retire_case((graphs, c), || session.context.synchronize().map_err(err));
            result?;
            retired?;
        }
        assert_eq!(checks, 8);
        eprintln!("FP8_MOE_CONTROL_PASS cases=2 checks={checks}");
        Ok(())
    })
    .expect("the two captures keep the current full-MoE bits");
}
