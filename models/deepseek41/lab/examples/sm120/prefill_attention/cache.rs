//! Decode-once lab. Native inputs and both attention arms stay live through
//! each graph. Only the candidate owns the guarded decoded arena.
use super::*;
use kernels_sm120::latent::sources_prefill_cache as cache;
use std::ops::Range;
use std::path::Path;

const MODULE: &str = "latent_sources_prefill_cache_deepseek41_sm120";

#[path = "heads.rs"]
pub(crate) mod heads;
#[path = "split.rs"]
pub(crate) mod split;
#[path = "stages.rs"]
pub(crate) mod stages;

fn specs() -> LabResult<Vec<kernel_build::CubinSpec>> {
    let mut specs = crate::specs()?
        .into_iter()
        .filter(|s| [REFERENCE, Schedule::Pair.module(), "probe_sm120"].contains(&s.name.as_str()))
        .collect::<Vec<_>>();
    specs.push(
        cache::cubin_spec(
            bundle_deepseek41::sm120::latent::profile(&FLASH),
            &entries()?,
        )
        .map_err(|e| e.to_string())?,
    );
    Ok(specs)
}

pub fn build(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(out)?;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()?;
    if out.canonicalize()?.starts_with(repo) {
        return Err("snapshot must be outside repository".into());
    }
    let lock = out.join("checks.lock");
    let toolchain = kernel_build::Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compiler required".into());
    }
    let specs = specs()?;
    let built = kernel_build::build(
        &toolchain,
        &specs,
        out,
        &lock,
        kernel_build::BuildOptions {
            lock_update: !lock.exists(),
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )?;
    let modules = built
        .iter()
        .map(|b| serde_json::json!({"name":b.name,"sha256":b.sha256}))
        .collect::<Vec<_>>();
    std::fs::write(
        out.join("snapshot.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"modules":modules,"rows":ROWS}))?,
    )?;
    Ok(())
}

pub fn snapshot(out: &Path) -> Result<kernel_harness::Cubins, Box<dyn std::error::Error>> {
    snapshot_with(out, specs()?)
}
fn snapshot_with(
    out: &Path,
    specs: Vec<kernel_build::CubinSpec>,
) -> Result<kernel_harness::Cubins, Box<dyn std::error::Error>> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("snapshot.json"))?)?;
    let mut bytes = Vec::new();
    let mut names = Vec::new();
    for spec in specs {
        let records = manifest["modules"].as_array().ok_or("missing modules")?;
        let record = records
            .iter()
            .find(|r| r["name"] == spec.name)
            .ok_or("missing cubin")?;
        let code = std::fs::read(out.join(format!("{}.cubin", spec.name)))?;
        if record["sha256"] != kernel_build::digest(&code) {
            return Err("cubin hash mismatch".into());
        }
        let name: &'static str = Box::leak(spec.name.into_boxed_str());
        let symbols = spec
            .entries
            .into_iter()
            .map(|e| &*Box::leak(e.symbol.into_boxed_str()))
            .collect::<Vec<_>>();
        bytes.push((name, &*Box::leak(code.into_boxed_slice())));
        names.push((name, &*Box::leak(symbols.into_boxed_slice())));
    }
    Ok(kernel_harness::Cubins::new(
        Box::leak(bytes.into_boxed_slice()),
        Box::leak(names.into_boxed_slice()),
    ))
}

struct Cached {
    rig: Rig,
    arena: DeviceBuffer,
    layout: cache::Layout,
    clear: Function,
    mark: Option<Function>,
    window: Function,
    pages: Option<Function>,
    _module: Module,
}
impl Cached {
    fn new(s: &Session, entry: sources::Entry, seed: u32) -> LabResult<Self> {
        let mut rig = Rig::scheduled(s, entry.clone(), seed, Schedule::Pair)?;
        // Compare the qualified pair against the same math over BF16 inputs.
        rig.arms[0].attention = rig._modules[1]
            .function(&Schedule::Pair.symbol(&entry))
            .map_err(|e| e.to_string())?;
        rig.arms[0].candidate = true;
        let module = s.module(MODULE)?;
        let decoded = cache::decoded_entry(&entry).map_err(|e| e.to_string())?;
        rig.arms[1].attention = module
            .function(&sources_prefill::reuse_symbol(&decoded, 2))
            .map_err(|e| e.to_string())?;
        rig.arms[1]
            .attention
            .set_max_dynamic_shared_memory(
                sources_prefill::reuse_plan(&decoded, ROWS, 2)
                    .map_err(|e| e.to_string())?
                    .dynamic_shared_bytes,
            )
            .map_err(|e| e.to_string())?;
        let layout = cache::Layout::new(WIDTH as u32, SLOTS, 1, rig.inputs.page, rig.inputs.pages)
            .map_err(|e| e.to_string())?;
        let second = rig.inputs.compressed_budget > 0;
        let result = Self {
            arena: s.allocate(layout.bytes as usize)?,
            layout,
            clear: module.function(cache::CLEAR).map_err(|e| e.to_string())?,
            mark: second
                .then(|| {
                    module
                        .function(&cache::mark_symbol(rig.inputs.page))
                        .map_err(|e| e.to_string())
                })
                .transpose()?,
            window: module
                .function(&cache::window_symbol(WIDTH as u32))
                .map_err(|e| e.to_string())?,
            pages: second
                .then(|| {
                    module
                        .function(&cache::pages_symbol(WIDTH as u32, rig.inputs.page))
                        .map_err(|e| e.to_string())
                })
                .transpose()?,
            rig,
            _module: module,
        };
        result.audit()?;
        Ok(result)
    }
    fn region(&self, r: &Range<u64>) -> Arg<'_> {
        Arg::Range {
            buffer: &self.arena,
            offset: r.start as usize,
            byte_len: (r.end - r.start) as usize,
        }
    }
    fn producer(&self, stream: &Stream) -> LabResult<()> {
        let input = &self.rig.inputs;
        let launch = |f: &Function, items: u64, args: &[Arg<'_>]| {
            launch::launch_with(
                f,
                stream,
                launch::config(cache::producer_plan(items).map_err(|e| e.to_string())?)?,
                args,
            )
        };
        launch(
            &self.clear,
            u64::from(input.pages),
            &[self.region(&self.layout.marks), Arg::U32(input.pages)],
        )?;
        if let Some(mark) = &self.mark {
            launch(
                mark,
                u64::from(ROWS) * 512,
                &[
                    Arg::Buffer(&input.row_batch),
                    Arg::Buffer(&input.live),
                    Arg::Buffer(&input.compressed_keys),
                    Arg::Buffer(&input.tables),
                    self.region(&self.layout.marks),
                    Arg::U32(ROWS),
                    Arg::U32(512),
                    Arg::U32(512),
                    Arg::U32(input.pages),
                    Arg::U32(input.pages),
                    Arg::U32(1),
                ],
            )?;
        }
        launch(
            &self.window,
            u64::from(SLOTS) * (WIDTH / 8) as u64,
            &[
                Arg::Buffer(&input.window_codes),
                Arg::Buffer(&input.window_scales),
                self.region(&self.layout.window),
                Arg::U32(SLOTS),
                Arg::U32(1),
                Arg::U64(u64::from(SLOTS) * WIDTH as u64),
                Arg::U64(u64::from(SLOTS) * (WIDTH / 32) as u64),
            ],
        )?;
        if let Some(pages) = &self.pages {
            launch(
                pages,
                u64::from(input.pages) * u64::from(input.page) * (WIDTH / 8) as u64,
                &[
                    Arg::Buffer(&input.compressed_pool),
                    self.region(&self.layout.marks),
                    self.region(&self.layout.compressed),
                    Arg::U32(input.pages),
                    Arg::U64(u64::from(input.page) * (WIDTH / 2 + WIDTH / 16) as u64),
                ],
            )?;
        }
        Ok(())
    }
    fn attention(&self, stream: &Stream) -> LabResult<()> {
        let mut args = self
            .rig
            .arguments(1, WINDOW, self.rig.inputs.compressed_budget);
        args[3] = self.region(&self.layout.window);
        args[4] = Arg::NullBuffer;
        args[8] = Arg::U64(u64::from(SLOTS) * WIDTH as u64 * 2);
        args[9] = Arg::U64(0);
        if self.pages.is_some() {
            args[16] = self.region(&self.layout.compressed);
            args[19] = Arg::U64(u64::from(self.rig.inputs.page) * WIDTH as u64 * 2);
        }
        launch::launch_with(
            &self.rig.arms[1].attention,
            stream,
            launch::config(
                Schedule::Pair
                    .plan(&self.rig.entry, ROWS)
                    .map_err(|e| e.to_string())?,
            )?,
            &args,
        )
    }
    fn combined(&self, stream: &Stream, arm: usize) -> LabResult<()> {
        if arm == 0 {
            self.rig.combined(stream, 0)
        } else {
            self.producer(stream)?;
            self.attention(stream)?;
            self.rig.merge(stream, 1)
        }
    }
    fn audit(&self) -> LabResult<()> {
        self.audit_with(&[])
    }
    fn audit_with(&self, extra: &[&DeviceBuffer]) -> LabResult<()> {
        let i = &self.rig.inputs;
        let mut buffers = vec![
            &i.q,
            &i.row_batch,
            &i.live,
            &i.window_codes,
            &i.window_scales,
            &i.item_slots,
            &i.window_keys,
            &i.compressed_pool,
            &i.compressed_keys,
            &i.tables,
            &i.sinks,
            &i.cos_sin,
            &self.arena,
        ];
        for arm in &self.rig.arms {
            buffers.extend([&arm.workspace, &arm.output]);
        }
        buffers.extend_from_slice(extra);
        let mut spans = buffers
            .into_iter()
            .map(|b| {
                b.device_address_at(0, b.byte_len())
                    .map(|p| (p, p + b.byte_len() as u64))
                    .map_err(|e| e.to_string())
            })
            .collect::<LabResult<Vec<_>>>()?;
        spans.sort_unstable();
        if spans.windows(2).any(|s| s[0].1 > s[1].0) {
            return Err("decoded cache aliases a concurrently live allocation".into());
        }
        for r in [
            &self.layout.window,
            &self.layout.compressed,
            &self.layout.marks,
        ] {
            self.arena
                .device_address_at(r.start as usize, (r.end - r.start) as usize)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn poison(&self, s: &Session) -> LabResult<()> {
        self.rig.poison(s)?;
        self.arena
            .copy_from_bytes(&vec![0xa5; self.layout.bytes as usize], &s.stream)
            .map_err(|e| e.to_string())
    }
    fn check_cache(&self, s: &Session, live: u32) -> LabResult<()> {
        let mut data = vec![0; self.layout.bytes as usize];
        self.arena
            .copy_to_bytes(&mut data, &s.stream)
            .map_err(|e| e.to_string())?;
        for g in self.layout.guards() {
            if data[g.start as usize..g.end as usize]
                .iter()
                .any(|&b| b != 0xa5)
            {
                return Err("decoded cache guard overwritten".into());
            }
        }
        let marks = &data[self.layout.marks.start as usize..self.layout.marks.end as usize];
        for (page, m) in marks.chunks_exact(4).enumerate() {
            let marked = u32::from_le_bytes(m.try_into().unwrap());
            if marked > 1 || (live == 0 && marked != 0) {
                return Err("decoded page marks were not refreshed".into());
            }
            if marked == 0 {
                let start = self.layout.compressed.start as usize
                    + page * self.rig.inputs.page as usize * WIDTH * 2;
                if data[start..start + self.rig.inputs.page as usize * WIDTH * 2]
                    .iter()
                    .any(|&b| b != 0xa5)
                {
                    return Err("unreferenced decoded page overwritten".into());
                }
            }
        }
        Ok(())
    }
}

pub(super) struct Lists {
    rows: u32,
    positions: Vec<u32>,
    selected: Vec<u8>,
    window: Option<Vec<u8>>,
    hash: String,
}
fn integer(v: &serde_json::Value, name: &str) -> LabResult<u32> {
    v[name]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| format!("invalid {name}"))
}
fn read_file(root: &Path, value: &serde_json::Value) -> LabResult<Vec<u8>> {
    let relative = Path::new(value["path"].as_str().ok_or("missing list path")?);
    if relative.is_absolute()
        || relative
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err("list path must stay inside capture root".into());
    }
    let mut path = root.to_path_buf();
    for part in relative.components() {
        path.push(part);
        if std::fs::symlink_metadata(&path)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
        {
            return Err("list capture refuses symbolic links".into());
        }
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if value["sha256"] != kernel_build::digest(&bytes) {
        return Err("selected-list digest mismatch".into());
    }
    Ok(bytes)
}
fn lists(path: &Path, selected: &str, offset: u32, samples: bool) -> LabResult<Lists> {
    let layer = match selected {
        "window" => 0,
        "mixed32" => 2,
        "mixed64" => 20,
        _ => return Err("invalid cache variant".into()),
    };
    layer_lists(path, layer, offset, samples)
}

pub(super) fn layer_lists(path: &Path, layer: u32, offset: u32, samples: bool) -> LabResult<Lists> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    layer_lists_in(path, &manifest, layer, offset, samples)
}

pub(super) fn layer_lists_in(
    path: &Path,
    manifest: &serde_json::Value,
    layer: u32,
    offset: u32,
    samples: bool,
) -> LabResult<Lists> {
    if manifest["schema"] != "chopinfer.prefill-cache-lists.v1" || manifest["samples"] != samples {
        return Err("wrong selected-list capture schema or mode".into());
    }
    let records = manifest["records"]
        .as_array()
        .ok_or("missing list records")?;
    let record = records
        .iter()
        .find(|r| r["offset"] == offset && r["layer"] == layer)
        .ok_or("required captured layer/offset absent")?;
    let rows = integer(record, "rows")?;
    if rows != if samples { 32 } else { ROWS } {
        return Err("capture must contain every requested row".into());
    }
    let positions = record["positions"]
        .as_array()
        .ok_or("missing positions")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("invalid position".into())
        })
        .collect::<LabResult<Vec<_>>>()?;
    if positions.len() != rows as usize
        || positions.iter().any(|&p| p < offset || p >= offset + ROWS)
        || (!samples
            && positions
                .iter()
                .enumerate()
                .any(|(i, &p)| p != offset + i as u32))
    {
        return Err("capture positions do not match chunk".into());
    }
    let root = path.parent().ok_or("missing capture root")?;
    let bytes = if layer == 0 {
        vec![0xff; rows as usize * 512 * 4]
    } else {
        read_file(root, &record["selected"])?
    };
    if bytes.len() != rows as usize * 512 * 4 {
        return Err("selected-list length mismatch".into());
    }
    for word in bytes.chunks_exact(4) {
        let key = i32::from_le_bytes(word.try_into().unwrap());
        if key < -1 || key >= fixture::CONTEXT as i32 {
            return Err("captured key exceeds bounded lab source pool".into());
        }
    }
    let window = record
        .get("window")
        .map(|r| read_file(root, r))
        .transpose()?;
    if window
        .as_ref()
        .is_some_and(|w| w.len() != rows as usize * WINDOW as usize * 4)
    {
        return Err("window-list length mismatch".into());
    }
    if !samples && window.is_none() {
        return Err("full capture requires actual window lists".into());
    }
    let hash = kernel_build::digest(if layer == 0 {
        window.as_deref().ok_or("window capture is missing")?
    } else {
        &bytes
    });
    Ok(Lists {
        rows,
        positions,
        hash,
        selected: bytes,
        window,
    })
}
impl Lists {
    pub(super) fn slice(&self, first: u32, rows: u32) -> LabResult<Self> {
        let end = first.checked_add(rows).ok_or("list slice overflow")?;
        if rows == 0 || end > self.rows {
            return Err("list slice exceeds captured rows".into());
        }
        let slice = |bytes: &[u8], stride: usize| {
            bytes[first as usize * stride..end as usize * stride].to_vec()
        };
        let selected = slice(&self.selected, 512 * 4);
        Ok(Self {
            rows,
            positions: self.positions[first as usize..end as usize].to_vec(),
            hash: kernel_build::digest(&selected),
            selected,
            window: self.window.as_ref().map(|w| slice(w, WINDOW as usize * 4)),
        })
    }

    pub(super) fn install(&self, s: &Session, input: &Inputs, live: u32) -> LabResult<()> {
        input
            .compressed_keys
            .copy_from_bytes(&self.selected, &s.stream)
            .map_err(|e| e.to_string())?;
        let window = self.window.clone().unwrap_or_else(|| {
            self.positions
                .iter()
                .enumerate()
                .flat_map(|(r, &p)| {
                    (0..WINDOW).flat_map(move |k| {
                        if p + k < WINDOW - 1 {
                            (-1_i32).to_le_bytes()
                        } else {
                            (r as i32 + k as i32).to_le_bytes()
                        }
                    })
                })
                .collect()
        });
        input
            .window_keys
            .copy_from_bytes(&window, &s.stream)
            .map_err(|e| e.to_string())?;
        input
            .live
            .copy_from_bytes(&live.to_le_bytes(), &s.stream)
            .map_err(|e| e.to_string())
    }
}

pub fn run(
    sessions: &[Session],
    path: &Path,
    selected: &str,
    offset: u32,
    samples: bool,
    timing: bool,
) -> LabResult<()> {
    exclusive(sessions)?;
    if timing && samples {
        return Err("sampled lists do not qualify full-row timing".into());
    }
    let lists = lists(path, selected, offset, samples)?;
    let entry = entries()?
        .into_iter()
        .find(|e| variant(e) == selected)
        .ok_or("missing cache entry")?;
    for (rank, s) in sessions.iter().enumerate() {
        let rig = Cached::new(s, entry.clone(), rank as u32 + 1)?;
        lists.install(s, &rig.rig.inputs, lists.rows)?;
        if timing {
            let graphs = [0, 1]
                .map(|arm| s.capture(|stream| rig.combined(stream, arm)))
                .into_iter()
                .collect::<LabResult<Vec<_>>>()?;
            let total =
                kernel_harness::time_ab(s, 1, 3, [EvictionMode::Write; 2], |arm, stream| {
                    graphs[arm].launch(stream).map_err(|e| e.to_string())
                })?;
            let producer = s.capture(|stream| rig.producer(stream))?;
            let produced =
                kernel_harness::time_ab(s, 1, 3, [EvictionMode::Write; 2], |_, stream| {
                    producer.launch(stream).map_err(|e| e.to_string())
                })?;
            println!(
                "TIMING prefill_cache rank={rank} variant={selected} offset={offset} rows={} paired_total_us={} cached_total_us={} producer_us={} ratio={} samples=3 cache_bytes={} actual_selected=true",
                lists.rows,
                total.arms[0].median_ms * 1000.0,
                total.arms[1].median_ms * 1000.0,
                produced.arms[0].median_ms * 1000.0,
                total.paired_ratio,
                rig.layout.bytes
            );
            continue;
        }
        oracle(s, &entry, Schedule::Pair)?;
        let graph = s.capture(|stream| {
            rig.combined(stream, 0)?;
            rig.combined(stream, 1)
        })?;
        for (generation, live) in [lists.rows, lists.rows - 1, 0, lists.rows]
            .into_iter()
            .enumerate()
        {
            rig.rig
                .inputs
                .set_case(s, generation as u32 + 11, false, live)?;
            lists.install(s, &rig.rig.inputs, live)?;
            // Change a native operand on every replay, including zero live rows.
            let code = if generation % 2 == 0 { 0x38 } else { 0xb8 };
            rig.rig
                .inputs
                .window_codes
                .copy_from_bytes(&[code; 8], &s.stream)
                .map_err(|e| e.to_string())?;
            let compressed_check = if rig.pages.is_some() {
                lists
                    .selected
                    .chunks_exact(4)
                    .take(live as usize * 512)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                    .find(|&k| k >= 0)
                    .map(|key| -> LabResult<(usize, u8, u8)> {
                        let page = rig.rig.inputs.page as usize;
                        let physical = rig.rig.inputs.pages as usize - 1 - key as usize / page;
                        let slot = key as usize % page;
                        let stride = page * (WIDTH / 2 + WIDTH / 16);
                        let code = if generation % 2 == 0 { 0x21 } else { 0xba };
                        rig.rig
                            .inputs
                            .compressed_pool
                            .copy_from_bytes_at(
                                physical * stride + slot * WIDTH / 2,
                                &[code; 4],
                                &s.stream,
                            )
                            .map_err(|e| e.to_string())?;
                        let mut scale = [0];
                        rig.rig
                            .inputs
                            .compressed_pool
                            .copy_to_bytes_at_async(
                                physical * stride + page * WIDTH / 2 + slot * WIDTH / 16,
                                &mut scale,
                                &s.stream,
                            )
                            .map_err(|e| e.to_string())?;
                        s.stream.synchronize().map_err(|e| e.to_string())?;
                        Ok((
                            rig.layout.compressed.start as usize
                                + (physical * page + slot) * WIDTH * 2,
                            code,
                            scale[0],
                        ))
                    })
                    .transpose()?
            } else {
                None
            };
            rig.poison(s)?;
            graph.launch(&s.stream).map_err(|e| e.to_string())?;
            rig.rig.compare(s, live)?;
            rig.check_cache(s, live)?;
            let mut decoded = [0; 16];
            rig.arena
                .copy_to_bytes_at_async(rig.layout.window.start as usize, &mut decoded, &s.stream)
                .map_err(|e| e.to_string())?;
            s.stream.synchronize().map_err(|e| e.to_string())?;
            let value = kernel_core::codec::f32_to_bf16(
                kernel_core::codec::SmallFloat::E4M3.decode_f64(code) as f32
                    * kernel_core::codec::decode_ue8m0_f32(121),
            );
            if decoded
                .chunks_exact(2)
                .any(|b| u16::from_le_bytes(b.try_into().unwrap()) != value)
            {
                return Err("decoded operand was not refreshed exactly".into());
            }
            if let Some((at, code, scale)) = compressed_check {
                let mut bytes = [0; 16];
                rig.arena
                    .copy_to_bytes_at_async(at, &mut bytes, &s.stream)
                    .map_err(|e| e.to_string())?;
                s.stream.synchronize().map_err(|e| e.to_string())?;
                for (i, word) in bytes.chunks_exact(2).enumerate() {
                    let value = kernel_core::codec::SmallFloat::E2M1
                        .decode_f64((code >> ((i % 2) * 4)) & 15)
                        as f32;
                    let scale = kernel_core::codec::SmallFloat::E4M3.decode_f64(scale) as f32;
                    if u16::from_le_bytes(word.try_into().unwrap())
                        != kernel_core::codec::f32_to_bf16(value * scale)
                    {
                        return Err("compressed operand was not refreshed exactly".into());
                    }
                }
            }
            if generation == 0 {
                rig.arena
                    .copy_from_bytes(&[0], &s.stream)
                    .map_err(|e| e.to_string())?;
                if rig.check_cache(s, live).is_ok() {
                    return Err("cache guard negative control was not rejected".into());
                }
            }
        }
        let page_cases = if rig.pages.is_some() {
            for page in [-1, rig.rig.inputs.pages as i32, 0] {
                rig.rig.inputs.set_case(s, 101, false, lists.rows)?;
                lists.install(s, &rig.rig.inputs, lists.rows)?;
                let tables = page.to_le_bytes().repeat(rig.rig.inputs.pages as usize);
                rig.rig
                    .inputs
                    .tables
                    .copy_from_bytes(&tables, &s.stream)
                    .map_err(|e| e.to_string())?;
                rig.poison(s)?;
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
                rig.rig.compare(s, lists.rows)?;
                rig.check_cache(s, lists.rows)?;
            }
            3
        } else {
            0
        };
        println!(
            "PASS prefill_cache rank={rank} variant={selected} offset={offset} rows={} workspace=true output=true replay=true f64=true cache_guards=true guard_control=true disjoint=true refresh=true actual_selected=true page_cases={page_cases} selected_sha256={} cache_bytes={}",
            lists.rows, lists.hash, rig.layout.bytes
        );
    }
    Ok(())
}

pub fn check_lists(path: &Path, variant: &str, offset: u32, samples: bool) -> LabResult<()> {
    let data = lists(path, variant, offset, samples)?;
    println!(
        "PASS prefill_cache_lists variant={variant} offset={offset} rows={} selected_sha256={}",
        data.rows, data.hash
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_list_slices_preserve_original_key_coordinates() {
        let data = Lists {
            rows: 4,
            positions: vec![24576, 24577, 24578, 24579],
            selected: (0..4)
                .flat_map(|r: i32| r.to_le_bytes().repeat(512))
                .collect(),
            window: Some(
                (100..104)
                    .flat_map(|r: i32| r.to_le_bytes().repeat(128))
                    .collect(),
            ),
            hash: String::new(),
        };
        let part = data.slice(2, 2).unwrap();
        assert_eq!(part.positions, [24578, 24579]);
        assert_eq!(&part.selected[..4], &2_i32.to_le_bytes());
        assert_eq!(&part.window.unwrap()[..4], &102_i32.to_le_bytes());
        for (first, rows) in [(0, 0), (3, 2), (u32::MAX, 1)] {
            assert!(data.slice(first, rows).is_err());
        }
    }

    #[test]
    fn native_decode_preserves_consumer_plan_and_partial_abi() {
        for entry in entries().unwrap() {
            let decoded = cache::decoded_entry(&entry).unwrap();
            assert_eq!(entry.workspace_bytes(ROWS), decoded.workspace_bytes(ROWS));
            assert_eq!(entry.params().unwrap(), decoded.params().unwrap());
            assert_eq!(
                Schedule::Pair.plan(&entry, ROWS).unwrap(),
                Schedule::Pair.plan(&decoded, ROWS).unwrap()
            );
            assert!(
                decoded.entry_specs().unwrap()[0]
                    .body
                    .contains("NativeAddress,true>")
            );
        }
    }
    #[test]
    fn captured_lists_reject_samples_as_full_and_check_digest_and_extent() {
        let root = std::env::temp_dir().join(format!("prefill-cache-lists-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let bytes = vec![0; 32 * 512 * 4];
        std::fs::write(root.join("selected.i32"), &bytes).unwrap();
        let manifest = root.join("lists.json");
        let data = serde_json::json!({"schema":"chopinfer.prefill-cache-lists.v1","samples":true,"records":[{
            "layer":2,"offset":0,"rows":32,"positions":(0..32).collect::<Vec<_>>(),
            "selected":{"path":"selected.i32","sha256":kernel_build::digest(&bytes)}
        }]});
        std::fs::write(&manifest, data.to_string()).unwrap();
        assert_eq!(lists(&manifest, "mixed32", 0, true).unwrap().rows, 32);
        assert!(lists(&manifest, "mixed32", 0, false).is_err());
        assert!(lists(&manifest, "mixed32", 8192, true).is_err());
        std::fs::write(root.join("selected.i32"), [1]).unwrap();
        assert!(lists(&manifest, "mixed32", 0, true).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
