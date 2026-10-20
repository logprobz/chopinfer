//! Compare original-row slices against the full-chunk route, with local scratch.
use super::*;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const PARTIAL: usize = HEADS * (WIDTH + 2) * 4;
const OUTPUT: usize = HEADS * WIDTH * 2;
const ROPE: usize = 64 * 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Span {
    first: u32,
    rows: u32,
}
impl Span {
    fn end(self) -> u32 {
        self.first + self.rows
    }
    fn live(self, full: u32) -> u32 {
        full.saturating_sub(self.first).min(self.rows)
    }
    fn window(self, offset: u32) -> std::ops::Range<u32> {
        if self.rows == 0 {
            return 0..0;
        }
        // Staged slots start at max(chunk_first - 127, 0). A cold chunk
        // has no prefix, so its part B still needs the last 127 A slots.
        let prefix = offset.min(WINDOW - 1);
        (prefix + self.first).saturating_sub(WINDOW - 1)..prefix + self.end()
    }
    fn halves(self) -> [Self; 2] {
        let a = self.rows.div_ceil(128) * 64;
        [
            Self { rows: a, ..self },
            Self {
                first: self.first + a,
                rows: self.rows - a,
            },
        ]
    }
}

struct Case {
    name: String,
    layer: u32,
    variant: &'static str,
    span: Span,
}
fn cases(offset: u32) -> LabResult<Vec<Case>> {
    let mut cases = [(0, "window"), (2, "mixed32"), (20, "mixed64")]
        .into_iter()
        .map(|(layer, variant)| Case {
            name: variant.into(),
            layer,
            variant,
            span: Span {
                first: 0,
                rows: ROWS,
            },
        })
        .collect::<Vec<_>>();
    for (layer, range) in super::super::retained::intervals(offset)?
        .into_iter()
        .skip(20)
    {
        cases.push(Case {
            name: format!("retained{layer}"),
            layer,
            variant: "mixed64",
            span: Span {
                first: range.start,
                rows: range.end - range.start,
            },
        });
    }
    Ok(cases)
}
fn cached(case: &Case, offset: u32) -> bool {
    case.variant == "window" || offset > 0
}

pub fn table(offset: u32) -> LabResult<serde_json::Value> {
    Ok(serde_json::Value::Array(cases(offset)?.into_iter().map(|case| {
        let [a,b] = case.span.halves();
        let part = |name, span: Span| {
            let window = span.window(offset);
            serde_json::json!({"part":name,"first":span.first,"rows":span.rows,
                "window_first":window.start,"window_rows":window.end-window.start})
        };
        serde_json::json!({"case":case.name,"layer":case.layer,"variant":case.variant,"offset":offset,
            "cached":cached(&case,offset),"parts":[part("whole",case.span),part("a",a),part("b",b)]})
    }).collect()))
}

fn validate(data: &Lists, case: &Case, offset: u32) -> LabResult<()> {
    let window = data.window.as_ref().ok_or("actual window list required")?;
    let ratio = if case.variant == "mixed32" { 2 } else { 1 };
    for part in [case.span, case.span.halves()[0], case.span.halves()[1]] {
        let published = part.window(offset);
        for row in part.first..part.end() {
            for word in window
                [row as usize * WINDOW as usize * 4..(row + 1) as usize * WINDOW as usize * 4]
                .chunks_exact(4)
            {
                let key = i32::from_le_bytes(word.try_into().unwrap());
                let newest = offset.min(WINDOW - 1) + row;
                if key != -1
                    && (!published.contains(&(key as u32))
                        || key > newest as i32
                        || key < newest.saturating_sub(WINDOW - 1) as i32)
                {
                    return Err(format!(
                        "{} offset={offset} row={row} key={key} escapes window {published:?}",
                        case.name
                    ));
                }
            }
            if case.variant != "window" {
                for word in data.selected[row as usize * 512 * 4..(row + 1) as usize * 512 * 4]
                    .chunks_exact(4)
                {
                    let key = i32::from_le_bytes(word.try_into().unwrap());
                    if key >= ((offset + row + 1) / ratio) as i32 {
                        return Err(format!(
                            "{} offset={offset} row={row} selected key={key} is not yet published",
                            case.name
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}
fn manifest(path: &Path) -> LabResult<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
pub fn check_inputs(path: &Path, offset: u32) -> LabResult<()> {
    let manifest = manifest(path)?;
    for case in cases(offset)? {
        let data = layer_lists_in(path, &manifest, case.layer, offset, false)?;
        validate(&data, &case, offset)?;
    }
    println!("PASS split_inputs offset={offset} cases=23 origins=true published_keys=true");
    Ok(())
}

fn region(buffer: &DeviceBuffer, first: usize, bytes: usize) -> Arg<'_> {
    Arg::Range {
        buffer,
        offset: first,
        byte_len: bytes,
    }
}
fn rows(buffer: &DeviceBuffer, span: Span, stride: usize) -> Arg<'_> {
    region(
        buffer,
        span.first as usize * stride,
        span.rows as usize * stride,
    )
}
fn progress(label: &str, stage: &str, start: Instant) {
    println!(
        "PROGRESS split_attention {label} stage={stage} elapsed_ms={} unix_ms={}",
        start.elapsed().as_millis(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
}

impl Cached {
    fn part_producer(
        &self,
        stream: &Stream,
        part: Span,
        live: &DeviceBuffer,
        offset: u32,
    ) -> LabResult<()> {
        if part.rows == 0 {
            return Ok(());
        }
        let i = &self.rig.inputs;
        let launch = |f: &Function, count, args: &[Arg<'_>]| {
            launch::launch_with(
                f,
                stream,
                launch::config(cache::producer_plan(count).map_err(|e| e.to_string())?)?,
                args,
            )
        };
        launch(
            &self.clear,
            u64::from(i.pages),
            &[self.region(&self.layout.marks), Arg::U32(i.pages)],
        )?;
        if let Some(mark) = &self.mark {
            launch(
                mark,
                u64::from(part.rows) * 512,
                &[
                    rows(&i.row_batch, part, 4),
                    Arg::Buffer(live),
                    rows(&i.compressed_keys, part, 512 * 4),
                    Arg::Buffer(&i.tables),
                    self.region(&self.layout.marks),
                    Arg::U32(part.rows),
                    Arg::U32(512),
                    Arg::U32(512),
                    Arg::U32(i.pages),
                    Arg::U32(i.pages),
                    Arg::U32(1),
                ],
            )?;
        }
        let window = part.window(offset);
        let first = window.start as usize;
        let slots = window.end - window.start;
        launch(
            &self.window,
            u64::from(slots) * (WIDTH / 8) as u64,
            &[
                region(&i.window_codes, first * WIDTH, slots as usize * WIDTH),
                region(
                    &i.window_scales,
                    first * (WIDTH / 32),
                    slots as usize * (WIDTH / 32),
                ),
                region(
                    &self.arena,
                    self.layout.window.start as usize + first * WIDTH * 2,
                    slots as usize * WIDTH * 2,
                ),
                Arg::U32(slots),
                Arg::U32(1),
                Arg::U64(u64::from(SLOTS) * WIDTH as u64),
                Arg::U64(u64::from(SLOTS) * (WIDTH / 32) as u64),
            ],
        )?;
        if let Some(pages) = &self.pages {
            launch(
                pages,
                u64::from(i.pages) * u64::from(i.page) * (WIDTH / 8) as u64,
                &[
                    Arg::Buffer(&i.compressed_pool),
                    self.region(&self.layout.marks),
                    self.region(&self.layout.compressed),
                    Arg::U32(i.pages),
                    Arg::U64(u64::from(i.page) * (WIDTH / 2 + WIDTH / 16) as u64),
                ],
            )?;
        }
        Ok(())
    }

    fn part(
        &self,
        stream: &Stream,
        part: Span,
        live: &DeviceBuffer,
        decoded: bool,
        offset: u32,
    ) -> LabResult<()> {
        if part.rows == 0 {
            return Ok(());
        }
        let i = &self.rig.inputs;
        if decoded {
            self.part_producer(stream, part, live, offset)?;
        }
        let mut args = self.rig.arguments(1, WINDOW, i.compressed_budget);
        args[0] = rows(&i.q, part, OUTPUT);
        args[1] = rows(&i.row_batch, part, 4);
        args[2] = Arg::Buffer(live);
        args[6] = rows(&i.window_keys, part, WINDOW as usize * 4);
        if self.pages.is_some() {
            args[17] = rows(&i.compressed_keys, part, 512 * 4);
        }
        if decoded {
            args[3] = self.region(&self.layout.window);
            args[4] = Arg::NullBuffer;
            args[8] = Arg::U64(u64::from(SLOTS) * WIDTH as u64 * 2);
            args[9] = Arg::U64(0);
            if self.pages.is_some() {
                args[16] = self.region(&self.layout.compressed);
                args[19] = Arg::U64(u64::from(i.page) * WIDTH as u64 * 2);
            }
        }
        launch::launch_with(
            &self.rig.arms[usize::from(decoded)].attention,
            stream,
            launch::config(
                Schedule::Pair
                    .plan(&self.rig.entry, part.rows)
                    .map_err(|e| e.to_string())?,
            )?,
            &args,
        )?;
        launch::launch_with(
            &self.rig.merge,
            stream,
            launch::config(
                self.rig
                    .entry
                    .merge_plan(part.rows)
                    .map_err(|e| e.to_string())?,
            )?,
            &[
                Arg::Buffer(&self.rig.arms[1].workspace),
                Arg::Buffer(live),
                Arg::Buffer(&self.rig.arms[1].output),
                Arg::Buffer(&i.sinks),
                rows(&i.cos_sin, part, ROPE),
            ],
        )
    }

    fn poison_reference(&self, s: &Session, part: Span, poison: &DeviceBuffer) -> LabResult<()> {
        // Only these reference rows are downloaded. Every live row is still
        // computed by the full-chunk launch. Keep fills outside timed graphs.
        for arm in &self.rig.arms {
            for (buffer, stride) in [(&arm.workspace, PARTIAL), (&arm.output, OUTPUT)] {
                buffer
                    .copy_range_from_async(
                        part.first as usize * stride,
                        poison,
                        0,
                        part.rows as usize * stride,
                        &s.stream,
                    )
                    .map_err(|e| e.to_string())?;
            }
        }
        self.arena
            .copy_range_from_async(0, poison, 0, self.layout.bytes as usize, &s.stream)
            .map_err(|e| e.to_string())
    }
    fn poison_part(&self, s: &Session, part: Span, poison: &DeviceBuffer) -> LabResult<()> {
        for (buffer, stride) in [
            (&self.rig.arms[1].workspace, PARTIAL),
            (&self.rig.arms[1].output, OUTPUT),
        ] {
            buffer
                .copy_range_from_async(
                    0,
                    poison,
                    0,
                    (part.rows + 1).min(ROWS) as usize * stride,
                    &s.stream,
                )
                .map_err(|e| e.to_string())?;
        }
        self.arena
            .copy_range_from_async(0, poison, 0, self.layout.bytes as usize, &s.stream)
            .map_err(|e| e.to_string())
    }
    fn check_window_guards(&self, s: &Session, part: Span, offset: u32) -> LabResult<()> {
        let mut bytes = vec![0; (self.layout.window.end - self.layout.window.start) as usize];
        self.arena
            .copy_to_bytes_at_async(self.layout.window.start as usize, &mut bytes, &s.stream)
            .map_err(|e| e.to_string())?;
        s.stream.synchronize().map_err(|e| e.to_string())?;
        let window = part.window(offset);
        for region in [
            &bytes[..window.start as usize * WIDTH * 2],
            &bytes[window.end as usize * WIDTH * 2..],
        ] {
            if region.iter().any(|&x| x != 0xa5) {
                return Err("window decoder wrote outside its part footprint".into());
            }
        }
        Ok(())
    }
}

fn download(s: &Session, rig: &Cached, arm: usize, span: Span) -> LabResult<[Vec<u8>; 2]> {
    let mut data = [
        vec![0; span.rows as usize * PARTIAL],
        vec![0; span.rows as usize * OUTPUT],
    ];
    for ((buffer, stride), bytes) in [
        (&rig.rig.arms[arm].workspace, PARTIAL),
        (&rig.rig.arms[arm].output, OUTPUT),
    ]
    .into_iter()
    .zip(&mut data)
    {
        buffer
            .copy_to_bytes_at_async(span.first as usize * stride, bytes, &s.stream)
            .map_err(|e| e.to_string())?;
        s.stream.synchronize().map_err(|e| e.to_string())?;
    }
    Ok(data)
}
fn compare(
    s: &Session,
    rig: &Cached,
    case: &Case,
    part: Span,
    expected: &[Vec<u8>; 2],
    live: u32,
) -> LabResult<()> {
    let checked = Span {
        first: 0,
        rows: (part.rows + 1).min(ROWS),
    };
    let actual = download(s, rig, 1, checked)?;
    for ((reference, got), stride) in expected.iter().zip(actual).zip([PARTIAL, OUTPUT]) {
        let first = (part.first - case.span.first) as usize * stride;
        let bytes = part.rows as usize * stride;
        compare_host_bytes(
            "part vs full",
            &reference[first..first + bytes],
            &got[..bytes],
            live as usize * stride,
        )?;
        compare_host_bytes(
            "part guard",
            &vec![0xa5; got.len() - bytes],
            &got[bytes..],
            0,
        )?;
    }
    Ok(())
}

fn serial(
    rig: &Cached,
    stream: &Stream,
    parts: &[(&str, Span); 3],
    lives: &[DeviceBuffer],
    decoded: bool,
    offset: u32,
    assembled: Option<(&[DeviceBuffer; 2], &DeviceBuffer)>,
) -> LabResult<()> {
    for index in 1..3 {
        let part = parts[index].1;
        if part.rows == 0 {
            continue;
        }
        if let Some((_, poison)) = assembled {
            for (destination, stride) in [
                (&rig.rig.arms[1].workspace, PARTIAL),
                (&rig.rig.arms[1].output, OUTPUT),
            ] {
                destination
                    .copy_range_from_async(
                        0,
                        poison,
                        0,
                        (part.rows + 1).min(ROWS) as usize * stride,
                        stream,
                    )
                    .map_err(|e| e.to_string())?;
            }
        }
        rig.part(stream, part, &lives[index], decoded, offset)?;
        if let Some((assembled, _)) = assembled {
            for ((destination, source), stride) in assembled
                .iter()
                .zip([&rig.rig.arms[1].workspace, &rig.rig.arms[1].output])
                .zip([PARTIAL, OUTPUT])
            {
                destination
                    .copy_range_from_async(
                        (part.first - parts[0].1.first) as usize * stride,
                        source,
                        0,
                        part.rows as usize * stride,
                        stream,
                    )
                    .map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}

fn check_marks(s: &Session, rig: &Cached, data: &Lists, part: Span, live: u32) -> LabResult<()> {
    let mut expected = vec![0_u32; rig.rig.inputs.pages as usize];
    if rig.pages.is_some() {
        let first = part.first as usize * 512 * 4;
        let end = (part.first + part.live(live)) as usize * 512 * 4;
        for key in data.selected[first..end].chunks_exact(4) {
            let key = i32::from_le_bytes(key.try_into().unwrap());
            if key >= 0 {
                expected[(rig.rig.inputs.pages - 1 - key as u32 / rig.rig.inputs.page) as usize] =
                    1;
            }
        }
    }
    let mut actual = vec![0; expected.len() * 4];
    rig.arena
        .copy_to_bytes_at_async(rig.layout.marks.start as usize, &mut actual, &s.stream)
        .map_err(|e| e.to_string())?;
    s.stream.synchronize().map_err(|e| e.to_string())?;
    if actual
        != expected
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    {
        return Err("part page marks differ from its selected-list slice".into());
    }
    Ok(())
}

pub fn run(sessions: &[Session], path: &Path, offset: u32, timing: bool) -> LabResult<()> {
    exclusive(sessions)?;
    let manifest = manifest(path)?;
    let cases = cases(offset)?;
    let start = Instant::now();
    for (rank, s) in sessions.iter().enumerate() {
        for entry in entries()? {
            let rig = Cached::new(s, entry.clone(), rank as u32 + 1)?;
            if !timing {
                oracle(s, &entry, Schedule::Pair)?;
            }
            let poison = (!timing)
                .then(|| fixture::upload(s, &vec![0xa5; ROWS as usize * PARTIAL]))
                .transpose()?;
            let rotation = (0..ROWS)
                .flat_map(|r| {
                    let pair = match r % 3 {
                        0 => [0.6_f32, 0.8],
                        1 => [-0.8, 0.6],
                        _ => [1.0, 0.0],
                    };
                    pair.repeat(32)
                })
                .collect::<Vec<_>>();
            rig.rig
                .inputs
                .cos_sin
                .copy_from_f32(&rotation, &s.stream)
                .map_err(|e| e.to_string())?;
            for case in cases.iter().filter(|c| c.variant == variant(&entry)) {
                let label = format!(
                    "rank={rank} offset={offset} case={} variant={}",
                    case.name, case.variant
                );
                progress(&label, "inputs", start);
                let data = layer_lists_in(path, &manifest, case.layer, offset, false)?;
                validate(&data, case, offset)?;
                data.install(s, &rig.rig.inputs, ROWS)?;
                let [a, b] = case.span.halves();
                let parts = [("whole", case.span), ("a", a), ("b", b)];
                let lives = parts
                    .iter()
                    .map(|(_, p)| fixture::upload(s, &p.rows.to_le_bytes()))
                    .collect::<LabResult<Vec<_>>>()?;
                rig.audit_with(&lives.iter().collect::<Vec<_>>())?;
                if timing {
                    for (index, (name, part)) in
                        parts.iter().enumerate().filter(|(_, (_, p))| p.rows > 0)
                    {
                        let graphs = [false, true]
                            .into_iter()
                            .filter(|&d| !d || cached(case, offset))
                            .map(|d| {
                                s.capture(|stream| {
                                    rig.part(stream, *part, &lives[index], d, offset)
                                })
                            })
                            .collect::<LabResult<Vec<_>>>()?;
                        let result = kernel_harness::time_ab(
                            s,
                            1,
                            3,
                            [EvictionMode::Write; 2],
                            |arm, stream| {
                                graphs[arm.min(graphs.len() - 1)]
                                    .launch(stream)
                                    .map_err(|e| e.to_string())
                            },
                        )?;
                        println!(
                            "TIMING split_attention {label} part={name} first={} rows={} paired_us={} cached_us={} cached={} samples=3 warmup=1 producer_included=true",
                            part.first,
                            part.rows,
                            result.arms[0].median_ms * 1000.0,
                            if cached(case, offset) {
                                result.arms[1].median_ms * 1000.0
                            } else {
                                0.0
                            },
                            cached(case, offset)
                        );
                    }
                    let graphs = [false, true]
                        .into_iter()
                        .filter(|&d| !d || cached(case, offset))
                        .map(|d| {
                            s.capture(|stream| {
                                serial(&rig, stream, &parts, &lives, d, offset, None)
                            })
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    let result = kernel_harness::time_ab(
                        s,
                        1,
                        3,
                        [EvictionMode::Write; 2],
                        |arm, stream| {
                            graphs[arm.min(graphs.len() - 1)]
                                .launch(stream)
                                .map_err(|e| e.to_string())
                        },
                    )?;
                    println!(
                        "TIMING split_attention {label} part=serial first={} rows={} paired_us={} cached_us={} cached={} samples=3 warmup=1 producer_included=true",
                        case.span.first,
                        case.span.rows,
                        result.arms[0].median_ms * 1000.0,
                        if cached(case, offset) {
                            result.arms[1].median_ms * 1000.0
                        } else {
                            0.0
                        },
                        cached(case, offset)
                    );
                    continue;
                }
                let native = s.capture(|stream| rig.rig.combined(stream, 0))?;
                let full_cached = s.capture(|stream| rig.combined(stream, 1))?;
                let graphs = parts
                    .iter()
                    .enumerate()
                    .map(|(index, (_, part))| {
                        if part.rows == 0 {
                            return Ok(Vec::new());
                        }
                        [false, true]
                            .into_iter()
                            .filter(|&d| !d || cached(case, offset))
                            .map(|d| {
                                s.capture(|stream| {
                                    rig.part(stream, *part, &lives[index], d, offset)
                                })
                            })
                            .collect::<LabResult<Vec<_>>>()
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                let assembled = [
                    s.allocate(case.span.rows as usize * PARTIAL)?,
                    s.allocate(case.span.rows as usize * OUTPUT)?,
                ];
                let poison = poison.as_ref().ok_or("missing correctness fill buffer")?;
                let mut extras = lives.iter().collect::<Vec<_>>();
                extras.extend(assembled.iter());
                extras.push(poison);
                rig.audit_with(&extras)?;
                let serial_graphs = [false, true]
                    .into_iter()
                    .filter(|&d| !d || cached(case, offset))
                    .map(|d| {
                        s.capture(|stream| {
                            serial(
                                &rig,
                                stream,
                                &parts,
                                &lives,
                                d,
                                offset,
                                Some((&assembled, poison)),
                            )
                        })
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                let timed_graphs = [false, true]
                    .into_iter()
                    .filter(|&d| !d || cached(case, offset))
                    .map(|d| {
                        s.capture(|stream| serial(&rig, stream, &parts, &lives, d, offset, None))
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                for (generation, live) in [ROWS, a.end() - 1, ROWS - 1, 0, ROWS]
                    .into_iter()
                    .enumerate()
                {
                    rig.rig
                        .inputs
                        .live
                        .copy_from_bytes(&live.to_le_bytes(), &s.stream)
                        .map_err(|e| e.to_string())?;
                    for (_, part) in parts.iter().filter(|(_, p)| p.rows > 0) {
                        rig.rig
                            .inputs
                            .q
                            .copy_from_bytes_at(
                                part.first as usize * OUTPUT,
                                &fixture::query(2, 17 + generation as u32),
                                &s.stream,
                            )
                            .map_err(|e| e.to_string())?;
                    }
                    rig.poison_reference(s, case.span, poison)?;
                    native.launch(&s.stream).map_err(|e| e.to_string())?;
                    let expected = download(s, &rig, 0, case.span)?;
                    if cached(case, offset) {
                        full_cached.launch(&s.stream).map_err(|e| e.to_string())?;
                        let actual = download(s, &rig, 1, case.span)?;
                        if actual != expected {
                            return Err(
                                "current full cached route differs from native full route".into()
                            );
                        }
                        rig.check_cache(s, live)?;
                    }
                    for (index, (name, part)) in parts.iter().enumerate() {
                        lives[index]
                            .copy_from_bytes(&part.live(live).to_le_bytes(), &s.stream)
                            .map_err(|e| e.to_string())?;
                        if part.rows == 0 {
                            continue;
                        }
                        for (decoded, graph) in graphs[index].iter().enumerate() {
                            rig.poison_part(s, *part, poison)?;
                            graph.launch(&s.stream).map_err(|e| e.to_string())?;
                            compare(s, &rig, case, *part, &expected, part.live(live))?;
                            if decoded == 1 {
                                rig.check_cache(s, part.live(live))?;
                                rig.check_window_guards(s, *part, offset)?;
                                check_marks(s, &rig, &data, *part, live)?;
                            }
                        }
                        progress(
                            &format!("{label} part={name} generation={generation}"),
                            "compared",
                            start,
                        );
                    }
                    for (decoded, graph) in serial_graphs.iter().enumerate() {
                        rig.poison_part(
                            s,
                            Span {
                                first: 0,
                                rows: ROWS,
                            },
                            poison,
                        )?;
                        for buffer in &assembled {
                            buffer
                                .copy_range_from_async(0, poison, 0, buffer.byte_len(), &s.stream)
                                .map_err(|e| e.to_string())?;
                        }
                        graph.launch(&s.stream).map_err(|e| e.to_string())?;
                        for ((buffer, wanted), stride) in
                            assembled.iter().zip(&expected).zip([PARTIAL, OUTPUT])
                        {
                            let mut actual = vec![0; buffer.byte_len()];
                            buffer
                                .copy_to_bytes(&mut actual, &s.stream)
                                .map_err(|e| e.to_string())?;
                            compare_host_bytes(
                                "serial parts vs full",
                                wanted,
                                &actual,
                                case.span.live(live) as usize * stride,
                            )?;
                        }
                        if decoded == 1 {
                            check_marks(s, &rig, &data, if b.rows > 0 { b } else { a }, live)?;
                            for guard in rig.layout.guards() {
                                let mut bytes = vec![0; (guard.end - guard.start) as usize];
                                rig.arena
                                    .copy_to_bytes_at_async(
                                        guard.start as usize,
                                        &mut bytes,
                                        &s.stream,
                                    )
                                    .map_err(|e| e.to_string())?;
                                s.stream.synchronize().map_err(|e| e.to_string())?;
                                if bytes.iter().any(|&x| x != 0xa5) {
                                    return Err("serial cache guard overwritten".into());
                                }
                            }
                        }
                        // Exercise the exact timed chain too, without the
                        // observation copies that preserve A's output. Its
                        // last active rows must match the full reference.
                        // Inactive B scratch can legitimately retain A data.
                        rig.poison_part(
                            s,
                            Span {
                                first: 0,
                                rows: ROWS,
                            },
                            poison,
                        )?;
                        timed_graphs[decoded]
                            .launch(&s.stream)
                            .map_err(|e| e.to_string())?;
                        let last = if b.rows > 0 { b } else { a };
                        let actual = download(
                            s,
                            &rig,
                            1,
                            Span {
                                first: 0,
                                rows: last.rows,
                            },
                        )?;
                        for ((got, wanted), stride) in
                            actual.iter().zip(&expected).zip([PARTIAL, OUTPUT])
                        {
                            let first = (last.first - case.span.first) as usize * stride;
                            let bytes = last.live(live) as usize * stride;
                            if got[..bytes] != wanted[first..first + bytes] {
                                return Err(
                                    "uninstrumented part chain differs from full reference".into(),
                                );
                            }
                        }
                    }
                    progress(
                        &format!("{label} part=serial generation={generation}"),
                        "compared",
                        start,
                    );
                }
                for (name, part) in parts
                    .iter()
                    .copied()
                    .chain([("serial", case.span)])
                    .filter(|(_, p)| p.rows > 0)
                {
                    for candidate in if cached(case, offset) {
                        &["paired", "cached"][..]
                    } else {
                        &["paired"][..]
                    } {
                        println!(
                            "PASS split_attention {label} part={name} first={} rows={} candidate={candidate} workspace=true output=true full_chunk=true replay=true f64=true guards=true origins=true timed_chain=true",
                            part.first, part.rows
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_rows_preserve_global_origins_and_cover_each_interval_once() {
        for offset in [0, 24576] {
            let cases = cases(offset).unwrap();
            assert_eq!(cases.len(), 23);
            for case in cases {
                let [a, b] = case.span.halves();
                assert_eq!(a.first, case.span.first);
                assert_eq!(a.end(), b.first);
                assert_eq!(b.end(), case.span.end());
                assert_eq!(a.rows + b.rows, case.span.rows);
                assert_eq!(a.first % 64, 0);
                assert_eq!(b.first % 64, 0);
                assert_eq!(a.live(0), 0);
                assert_eq!(b.live(0), 0);
                assert_eq!(a.live(ROWS), a.rows);
                assert_eq!(b.live(ROWS), b.rows);
                if case.span.rows == ROWS {
                    assert_eq!([a.rows, b.rows], [4096, 4096]);
                } else {
                    assert_eq!(a.rows, b.rows + 64);
                }
            }
        }
        assert!(cases(1).is_err());
    }

    #[test]
    fn window_footprints_keep_the_actual_cold_or_retained_prefix() {
        let [a, b] = (Span {
            first: 0,
            rows: ROWS,
        })
        .halves();
        assert_eq!(a.window(0), 0..4096);
        assert_eq!(b.window(0), 3969..8192);
        assert_eq!(a.window(24576), 0..4223);
        assert_eq!(b.window(24576), 4096..8319);
        let [a, b] = (Span {
            first: 8128,
            rows: 64,
        })
        .halves();
        assert_eq!(a.window(0), 8001..8192);
        assert_eq!(a.window(24576), 8128..8319);
        assert_eq!(b.window(0), 0..0);
        assert_eq!(b.window(24576), 0..0);
    }

    #[test]
    fn published_list_checks_reject_wrong_part_origins_and_future_keys() {
        let mut data = Lists {
            rows: 256,
            positions: (0..256).collect(),
            hash: String::new(),
            selected: vec![0xff; 256 * 512 * 4],
            window: Some(
                (0..256_i32)
                    .flat_map(|r| {
                        (0..128).flat_map(move |k| {
                            let key = (r - 127).max(0) + k;
                            if key <= r {
                                key.to_le_bytes()
                            } else {
                                (-1_i32).to_le_bytes()
                            }
                        })
                    })
                    .collect(),
            ),
        };
        let case = Case {
            name: "test".into(),
            layer: 20,
            variant: "mixed64",
            span: Span {
                first: 64,
                rows: 192,
            },
        };
        validate(&data, &case, 0).unwrap();
        let at = 192 * 128 * 4;
        data.window.as_mut().unwrap()[at..at + 4].copy_from_slice(&64_i32.to_le_bytes());
        assert!(validate(&data, &case, 0).is_err());
        data.window.as_mut().unwrap()[at..at + 4].copy_from_slice(&65_i32.to_le_bytes());
        data.selected[64 * 512 * 4..64 * 512 * 4 + 4].copy_from_slice(&65_i32.to_le_bytes());
        assert!(validate(&data, &case, 0).is_err());
    }
}
