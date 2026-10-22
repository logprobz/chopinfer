//! Bounded actual-list, topology and partition qualification for head staging.
use super::*;
use kernel_build::digest;
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
#[path = "heads_next.rs"]
pub mod next;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Candidate {
    Current,
    Shared,
    Async,
    Pipeline,
}
impl Candidate {
    fn name(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Shared => "shared",
            Self::Async => "async",
            Self::Pipeline => "pipeline",
        }
    }
    fn module(self, decoded: bool, tp: u32) -> &'static str {
        match self {
            Self::Current if tp == 4 && decoded => MODULE,
            Self::Current if tp == 4 => "latent_sources_prefill_reuse2_deepseek41_sm120",
            Self::Current => TOPOLOGY_REFERENCE,
            Self::Shared => "latent_sources_prefill_head_shared_deepseek41_sm120",
            Self::Async => "latent_sources_prefill_head_async_deepseek41_sm120",
            Self::Pipeline => "latent_sources_prefill_head_pipeline_deepseek41_sm120",
        }
    }
    fn symbol(self, e: &sources::Entry) -> String {
        match self {
            Self::Current
                if matches!(e.query, LatentAttentionQuery::Attention { heads: 16, .. }) =>
            {
                sources_prefill::reuse_symbol(e, 2)
            }
            Self::Current => e.symbol(),
            Self::Shared => heads::symbol(e),
            Self::Async => heads::asynchronous::symbol(e, false),
            Self::Pipeline => heads::asynchronous::symbol(e, true),
        }
    }
    fn plan(
        self,
        e: &sources::Entry,
        rows: u32,
    ) -> Result<kernel_core::contract::LaunchPlan, kernel_core::contract::Refusal> {
        if self == Self::Current
            && !matches!(e.query, LatentAttentionQuery::Attention { heads: 16, .. })
        {
            e.plan(rows)
        } else if self == Self::Current {
            sources_prefill::reuse_plan(e, rows, 2)
        } else {
            heads::plan(e, rows)
        }
    }
}
fn candidates(decoded: bool) -> Vec<Candidate> {
    if decoded {
        vec![Candidate::Shared, Candidate::Async, Candidate::Pipeline]
    } else {
        vec![Candidate::Shared]
    }
}
struct Buffers {
    partial: DeviceBuffer,
    output: DeviceBuffer,
}
struct HeadRig {
    input: Inputs,
    entry: sources::Entry,
    heads: usize,
    capacity: u32,
    decoded: bool,
    reference_native: bool,
    arena: DeviceBuffer,
    unaligned: Option<DeviceBuffer>,
    layout: cache::Layout,
    arms: [Buffers; 2],
    functions: Vec<(Candidate, Function)>,
    merge: Function,
    clear: Function,
    mark: Function,
    window: Function,
    pages: Function,
    part_live: [DeviceBuffer; 2],
    _modules: Vec<Module>,
}
fn region(buffer: &DeviceBuffer, first: usize, bytes: usize) -> Arg<'_> {
    Arg::Range {
        buffer,
        offset: first,
        byte_len: bytes,
    }
}
impl HeadRig {
    fn new(
        s: &Session,
        entry: sources::Entry,
        tp: u32,
        capacity: u32,
        decoded: bool,
        seed: u32,
        unaligned: bool,
    ) -> LabResult<Self> {
        let LatentAttentionQuery::Attention {
            heads: count,
            page_size,
            ..
        } = entry.query
        else {
            unreachable!()
        };
        let heads = count as usize;
        let input = Inputs::with_shape(s, page_size, true, seed, heads, capacity + 1)?;
        let layout = cache::Layout::new(WIDTH as u32, SLOTS, 1, page_size, input.pages)
            .map_err(|e| e.to_string())?;
        let arena = s.allocate(layout.bytes as usize)?;
        let unaligned = if unaligned {
            if !decoded {
                return Err("unaligned control requires decoded inputs".into());
            }
            Some(s.allocate(layout.bytes as usize + 4)?)
        } else {
            None
        };
        let make = || -> LabResult<Buffers> {
            Ok(Buffers {
                partial: s.allocate((capacity + 1) as usize * heads * (WIDTH + 2) * 4)?,
                output: s.allocate((capacity + 1) as usize * heads * WIDTH * 2)?,
            })
        };
        let arms = [make()?, make()?];
        let e = if decoded {
            cache::decoded_entry(&entry).map_err(|e| e.to_string())?
        } else {
            entry.clone()
        };
        let mut modules = Vec::new();
        let mut functions = Vec::new();
        for candidate in std::iter::once(Candidate::Current).chain(candidates(decoded)) {
            let module = s.module(candidate.module(decoded, tp))?;
            let function = module
                .function(
                    &candidate.symbol(if candidate == Candidate::Current && tp != 4 {
                        &entry
                    } else {
                        &e
                    }),
                )
                .map_err(|e| e.to_string())?;
            function
                .set_max_dynamic_shared_memory(
                    candidate
                        .plan(&entry, capacity)
                        .map_err(|e| e.to_string())?
                        .dynamic_shared_bytes,
                )
                .map_err(|e| e.to_string())?;
            functions.push((candidate, function));
            modules.push(module);
        }
        let merge_module = s.module(if tp == 4 {
            REFERENCE
        } else {
            TOPOLOGY_REFERENCE
        })?;
        let merge = merge_module
            .function(&entry.merge_symbol())
            .map_err(|e| e.to_string())?;
        modules.push(merge_module);
        let producers = s.module(MODULE)?;
        let clear = producers
            .function(cache::CLEAR)
            .map_err(|e| e.to_string())?;
        let mark = producers
            .function(&cache::mark_symbol(page_size))
            .map_err(|e| e.to_string())?;
        let window = producers
            .function(&cache::window_symbol(WIDTH as u32))
            .map_err(|e| e.to_string())?;
        let pages = producers
            .function(&cache::pages_symbol(WIDTH as u32, page_size))
            .map_err(|e| e.to_string())?;
        modules.push(producers);
        let result = Self {
            input,
            entry,
            heads,
            capacity,
            decoded,
            reference_native: tp != 4,
            arena,
            unaligned,
            layout,
            arms,
            functions,
            merge,
            clear,
            mark,
            window,
            pages,
            part_live: [s.allocate(4)?, s.allocate(4)?],
            _modules: modules,
        };
        result.audit()?;
        Ok(result)
    }
    fn arena_region(&self, r: &Range<u64>) -> Arg<'_> {
        region(&self.arena, r.start as usize, (r.end - r.start) as usize)
    }
    fn cache_view(&self, r: &Range<u64>) -> Arg<'_> {
        if let Some(buffer) = &self.unaligned {
            region(buffer, r.start as usize + 2, (r.end - r.start) as usize)
        } else {
            self.arena_region(r)
        }
    }
    fn audit(&self) -> LabResult<()> {
        let i = &self.input;
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
        for a in &self.arms {
            buffers.extend([&a.partial, &a.output]);
        }
        buffers.extend(&self.part_live);
        buffers.extend(self.unaligned.iter());
        let mut ranges = buffers
            .into_iter()
            .map(|b| {
                b.device_address_at(0, b.byte_len())
                    .map(|p| (p, p + b.byte_len() as u64))
                    .map_err(|e| e.to_string())
            })
            .collect::<LabResult<Vec<_>>>()?;
        ranges.sort_unstable();
        if ranges.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err("head qualification allocations overlap".into());
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
    fn producer(&self, stream: &Stream) -> LabResult<()> {
        if !self.decoded {
            return Ok(());
        }
        let i = &self.input;
        let run = |f: &Function, count, args: &[Arg<'_>]| {
            launch::launch_with(
                f,
                stream,
                launch::config(cache::producer_plan(count).map_err(|e| e.to_string())?)?,
                args,
            )
        };
        run(
            &self.clear,
            u64::from(i.pages),
            &[self.arena_region(&self.layout.marks), Arg::U32(i.pages)],
        )?;
        run(
            &self.mark,
            u64::from(self.capacity) * 512,
            &[
                Arg::Buffer(&i.row_batch),
                Arg::Buffer(&i.live),
                Arg::Buffer(&i.compressed_keys),
                Arg::Buffer(&i.tables),
                self.arena_region(&self.layout.marks),
                Arg::U32(self.capacity),
                Arg::U32(512),
                Arg::U32(512),
                Arg::U32(i.pages),
                Arg::U32(i.pages),
                Arg::U32(1),
            ],
        )?;
        run(
            &self.window,
            u64::from(SLOTS) * (WIDTH / 8) as u64,
            &[
                Arg::Buffer(&i.window_codes),
                Arg::Buffer(&i.window_scales),
                self.arena_region(&self.layout.window),
                Arg::U32(SLOTS),
                Arg::U32(1),
                Arg::U64(u64::from(SLOTS) * WIDTH as u64),
                Arg::U64(u64::from(SLOTS) * (WIDTH / 32) as u64),
            ],
        )?;
        run(
            &self.pages,
            u64::from(i.pages) * u64::from(i.page) * (WIDTH / 8) as u64,
            &[
                Arg::Buffer(&i.compressed_pool),
                self.arena_region(&self.layout.marks),
                self.arena_region(&self.layout.compressed),
                Arg::U32(i.pages),
                Arg::U64(u64::from(i.page) * (WIDTH / 2 + WIDTH / 16) as u64),
            ],
        )?;
        if let Some(buffer) = &self.unaligned {
            buffer
                .copy_range_from_async(2, &self.arena, 0, self.arena.byte_len(), stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn arguments<'a>(
        &'a self,
        candidate: Candidate,
        arm: usize,
        first: u32,
        rows: u32,
        live: &'a DeviceBuffer,
    ) -> LabResult<Vec<Arg<'a>>> {
        let i = &self.input;
        let view = |buffer, stride| region(buffer, first as usize * stride, rows as usize * stride);
        let LatentAttentionQuery::Attention {
            softmax_scale_bits, ..
        } = self.entry.query
        else {
            unreachable!()
        };
        let mut args = vec![
            view(&i.q, self.heads * WIDTH * 2),
            view(&i.row_batch, 4),
            Arg::Buffer(live),
            Arg::Buffer(&i.window_codes),
            Arg::Buffer(&i.window_scales),
            Arg::Buffer(&i.item_slots),
            view(&i.window_keys, WINDOW as usize * 4),
            view(&self.arms[arm].partial, self.heads * (WIDTH + 2) * 4),
            Arg::U64(u64::from(SLOTS) * WIDTH as u64),
            Arg::U64(u64::from(SLOTS) * (WIDTH / 32) as u64),
            Arg::U32(SLOTS),
            Arg::U32(1),
            Arg::U32(1),
            Arg::U32(WINDOW),
            Arg::U32(WINDOW),
            Arg::U32(softmax_scale_bits),
            Arg::Buffer(&i.compressed_pool),
            view(&i.compressed_keys, 512 * 4),
            Arg::Buffer(&i.tables),
            Arg::U64(u64::from(i.page) * (WIDTH / 2 + WIDTH / 16) as u64),
            Arg::U32(i.pages),
            Arg::U32(i.pages),
            Arg::U32(1),
            Arg::U32(512),
            Arg::U32(512),
        ];
        if self.decoded && !(candidate == Candidate::Current && self.reference_native) {
            args[3] = self.cache_view(&self.layout.window);
            args[4] = Arg::NullBuffer;
            args[8] = Arg::U64(u64::from(SLOTS) * WIDTH as u64 * 2);
            args[9] = Arg::U64(0);
            args[16] = self.cache_view(&self.layout.compressed);
            args[19] = Arg::U64(u64::from(i.page) * WIDTH as u64 * 2);
        }
        Ok(args)
    }
    fn attention(
        &self,
        stream: &Stream,
        candidate: Candidate,
        arm: usize,
        first: u32,
        rows: u32,
        live: &DeviceBuffer,
    ) -> LabResult<()> {
        if rows == 0 {
            return Ok(());
        }
        let args = self.arguments(candidate, arm, first, rows, live)?;
        let f = &self
            .functions
            .iter()
            .find(|(c, _)| *c == candidate)
            .ok_or("candidate not loaded")?
            .1;
        launch::launch_with(
            f,
            stream,
            launch::config(
                candidate
                    .plan(&self.entry, rows)
                    .map_err(|e| e.to_string())?,
            )?,
            &args,
        )
    }
    fn merge(
        &self,
        stream: &Stream,
        arm: usize,
        first: u32,
        rows: u32,
        live: &DeviceBuffer,
    ) -> LabResult<()> {
        if rows == 0 {
            return Ok(());
        }
        launch::launch_with(
            &self.merge,
            stream,
            launch::config(self.entry.merge_plan(rows).map_err(|e| e.to_string())?)?,
            &[
                region(
                    &self.arms[arm].partial,
                    first as usize * self.heads * (WIDTH + 2) * 4,
                    rows as usize * self.heads * (WIDTH + 2) * 4,
                ),
                Arg::Buffer(live),
                region(
                    &self.arms[arm].output,
                    first as usize * self.heads * WIDTH * 2,
                    rows as usize * self.heads * WIDTH * 2,
                ),
                Arg::Buffer(&self.input.sinks),
                region(
                    &self.input.cos_sin,
                    first as usize * 64 * 4,
                    rows as usize * 64 * 4,
                ),
            ],
        )
    }
    fn whole(&self, stream: &Stream, candidate: Candidate, arm: usize) -> LabResult<()> {
        self.attention(stream, candidate, arm, 0, self.capacity, &self.input.live)?;
        self.merge(stream, arm, 0, self.capacity, &self.input.live)
    }
    fn partitioned(&self, stream: &Stream, candidate: Candidate) -> LabResult<()> {
        let a = first_part(self.capacity);
        for (part, (first, rows)) in [(0, a), (a, self.capacity - a)].into_iter().enumerate() {
            self.attention(stream, candidate, 1, first, rows, &self.part_live[part])?;
            self.merge(stream, 1, first, rows, &self.part_live[part])?;
        }
        Ok(())
    }
    fn poison(&self, s: &Session) -> LabResult<()> {
        for buffer in self
            .arms
            .iter()
            .flat_map(|a| [&a.partial, &a.output])
            .chain([&self.arena])
            .chain(self.unaligned.iter())
        {
            buffer
                .copy_from_bytes(&vec![0xa5; buffer.byte_len()], &s.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn compare(&self, s: &Session, live: u32) -> LabResult<()> {
        self.compare_mode(s, live, true)
    }
    fn compare_mode(&self, s: &Session, live: u32, partial: bool) -> LabResult<()> {
        for (name, first, second, stride) in [
            (
                "head partial",
                &self.arms[0].partial,
                &self.arms[1].partial,
                self.heads * (WIDTH + 2) * 4,
            ),
            (
                "head output",
                &self.arms[0].output,
                &self.arms[1].output,
                self.heads * WIDTH * 2,
            ),
        ] {
            if name == "head partial" && !partial {
                let mut bytes = vec![0; second.byte_len()];
                second
                    .copy_to_bytes(&mut bytes, &s.stream)
                    .map_err(|e| e.to_string())?;
                if bytes.iter().any(|&b| b != 0xa5) {
                    return Err("timed fused path wrote the eliminated partial buffer".into());
                }
                continue;
            }
            let mut a = vec![0; first.byte_len()];
            let mut b = vec![0; second.byte_len()];
            first
                .copy_to_bytes(&mut a, &s.stream)
                .map_err(|e| e.to_string())?;
            second
                .copy_to_bytes(&mut b, &s.stream)
                .map_err(|e| e.to_string())?;
            compare_host_bytes(name, &a, &b, live as usize * stride)?;
        }
        if self.decoded {
            let mut bytes = vec![0; self.arena.byte_len()];
            self.arena
                .copy_to_bytes(&mut bytes, &s.stream)
                .map_err(|e| e.to_string())?;
            if let Some(buffer) = &self.unaligned {
                let mut copied = vec![0; buffer.byte_len()];
                buffer
                    .copy_to_bytes(&mut copied, &s.stream)
                    .map_err(|e| e.to_string())?;
                if copied[..2] != [0xa5; 2]
                    || copied[copied.len() - 2..] != [0xa5; 2]
                    || copied[2..copied.len() - 2] != bytes
                {
                    return Err("unaligned cache image or guards changed".into());
                }
            }
            if self.layout.guards().iter().any(|r| {
                bytes[r.start as usize..r.end as usize]
                    .iter()
                    .any(|&v| v != 0xa5)
            }) {
                return Err("head cache guard overwritten".into());
            }
            for (page, word) in bytes
                [self.layout.marks.start as usize..self.layout.marks.end as usize]
                .chunks_exact(4)
                .enumerate()
            {
                let mark = u32::from_le_bytes(word.try_into().unwrap());
                if mark > 1 || (live == 0 && mark != 0) {
                    return Err("head cache mark not refreshed".into());
                }
                let start = self.layout.compressed.start as usize
                    + page * self.input.page as usize * WIDTH * 2;
                if mark == 0
                    && bytes[start..start + self.input.page as usize * WIDTH * 2]
                        .iter()
                        .any(|&v| v != 0xa5)
                {
                    return Err("unreferenced head cache page overwritten".into());
                }
            }
        }
        Ok(())
    }
    fn live(&self, s: &Session, live: u32, seed: u32) -> LabResult<()> {
        self.input
            .live
            .copy_from_bytes(&live.to_le_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
        self.input
            .q
            .copy_from_bytes(
                &fixture::query_with_heads(2.min(self.capacity), self.heads, seed),
                &s.stream,
            )
            .map_err(|e| e.to_string())?;
        let a = first_part(self.capacity);
        for (buffer, n) in self
            .part_live
            .iter()
            .zip([live.min(a), live.saturating_sub(a)])
        {
            buffer
                .copy_from_bytes(&n.to_le_bytes(), &s.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

fn first_part(rows: u32) -> u32 {
    (rows.div_ceil(128) * 64).min(rows)
}

struct Case {
    tp: u32,
    name: String,
    first: u32,
    rows: u32,
}
fn cases(offset: u32, timing: bool) -> Vec<Case> {
    let mut cases = vec![Case {
        tp: 4,
        name: "full".into(),
        first: 0,
        rows: ROWS,
    }];
    if !timing {
        for tp in [1, 2] {
            cases.push(Case {
                tp,
                name: "topology".into(),
                first: 4095,
                rows: 129,
            });
        }
        if [0, 24576].contains(&offset) {
            for rows in [64, 128, 704, 1280] {
                cases.push(Case {
                    tp: 4,
                    name: format!("tail{rows}"),
                    first: ROWS - rows,
                    rows,
                });
            }
        }
    }
    if !timing && offset == 24576 {
        cases.push(Case {
            tp: 4,
            name: "unaligned".into(),
            first: 4095,
            rows: 65,
        });
    }
    cases
}

pub fn run(sessions: &[Session], lists: &Path, offset: u32, timing: bool) -> LabResult<()> {
    if ![0, 8192, 16384, 24576].contains(&offset) {
        return Err("invalid head offset".into());
    }
    exclusive(sessions)?;
    let manifest = serde_json::from_slice(&std::fs::read(lists).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let decoded = offset > 0;
    for (rank, s) in sessions.iter().enumerate() {
        for case in cases(offset, timing) {
            for entry in topology_entries(case.tp)? {
                let variant = variant(&entry);
                let layer = if variant == "mixed32" { 2 } else { 20 };
                let data = layer_lists_in(lists, &manifest, layer, offset, false)?
                    .slice(case.first, case.rows)?;
                let rig = HeadRig::new(
                    s,
                    entry,
                    case.tp,
                    case.rows,
                    decoded,
                    rank as u32 + 1,
                    case.name == "unaligned",
                )?;
                data.install(s, &rig.input, case.rows)?;
                for candidate in candidates(decoded) {
                    let label = format!(
                        "rank={rank} tp={} offset={offset} variant={variant} case={} first={} rows={} candidate={} heads={} cache_align={}",
                        case.tp,
                        case.name,
                        case.first,
                        case.rows,
                        candidate.name(),
                        rig.heads,
                        if case.name == "unaligned" { 2 } else { 16 }
                    );
                    println!(
                        "PROGRESS head_staging {label} phase={}",
                        if timing { "time" } else { "check" }
                    );
                    if !timing {
                        let graph = s.capture(|stream| {
                            rig.producer(stream)?;
                            rig.whole(stream, Candidate::Current, 0)?;
                            rig.whole(stream, candidate, 1)
                        })?;
                        for (generation, live) in [case.rows, case.rows - 1, 0, case.rows]
                            .into_iter()
                            .enumerate()
                        {
                            rig.live(s, live, 31 + generation as u32)?;
                            rig.poison(s)?;
                            graph.launch(&s.stream).map_err(|e| e.to_string())?;
                            rig.compare(s, live)?;
                        }
                        // Both parts consume one completed cache and own disjoint
                        // partial/output ranges. No producer overlaps a consumer.
                        let parts = s.capture(|stream| {
                            rig.producer(stream)?;
                            rig.whole(stream, Candidate::Current, 0)?;
                            rig.partitioned(stream, candidate)
                        })?;
                        rig.live(s, case.rows, 51)?;
                        rig.poison(s)?;
                        parts.launch(&s.stream).map_err(|e| e.to_string())?;
                        rig.compare(s, case.rows)?;
                        println!(
                            "PASS head_staging {label} partial=true output=true replay=4 poison=true partition_slots=2 part_a={} part_b={} cache_guards=true disjoint=true actual_selected=true selected_sha256={}",
                            first_part(case.rows),
                            case.rows - first_part(case.rows),
                            digest(&data.selected)
                        );
                    } else {
                        rig.live(s, case.rows, 61)?;
                        let graphs = [Candidate::Current, candidate]
                            .into_iter()
                            .enumerate()
                            .map(|(arm, c)| {
                                s.capture(|stream| {
                                    rig.producer(stream)?;
                                    rig.whole(stream, c, arm)
                                })
                            })
                            .collect::<LabResult<Vec<_>>>()?;
                        let result = kernel_harness::time_ab(
                            s,
                            1,
                            3,
                            [EvictionMode::Write; 2],
                            |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
                        )?;
                        println!(
                            "TIMING head_staging {label} current_total_us={} candidate_total_us={} samples=3 warmup=1 producer_attention_merge=true",
                            result.arms[0].median_ms * 1000.0,
                            result.arms[1].median_ms * 1000.0
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
    fn head_job_counts_every_topology_tail_and_alignment_control() {
        let gates = [0, 8192, 16384, 24576]
            .map(|offset| cases(offset, false).len() * 2 * 2 * candidates(offset > 0).len());
        assert_eq!(gates, [28, 36, 36, 96]);
        assert_eq!(gates.into_iter().sum::<usize>(), 196);
        let times = [0, 8192, 16384, 24576]
            .map(|offset| cases(offset, true).len() * 2 * 2 * candidates(offset > 0).len());
        assert_eq!(times, [4, 12, 12, 12]);
        assert_eq!(
            [64, 65, 128, 129, 704, 1280, 8192].map(first_part),
            [64, 64, 64, 128, 384, 640, 4096]
        );
        for tp in [1, 2, 4] {
            for e in topology_entries(tp).unwrap() {
                let plan = Candidate::Current.plan(&e, 129).unwrap();
                assert_eq!(plan.block[0], if tp == 4 { 256 } else { 128 });
                assert_eq!(
                    Candidate::Current.symbol(&e),
                    if tp == 4 {
                        sources_prefill::reuse_symbol(&e, 2)
                    } else {
                        e.symbol()
                    }
                );
            }
        }
    }
}
