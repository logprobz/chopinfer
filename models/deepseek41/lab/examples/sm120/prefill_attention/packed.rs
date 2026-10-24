//! Qualify paired attention, and the fused shared-head finish, on prefill
//! steps that pack several prompts, at the row counts serving produces:
//! every row bitwise equal to the reference plan, and the time of both
//! plans.
use super::*;

/// Packed prompt lengths: the served C8, C16 and C1 prefill steps of the
/// coding workload, then odd counts and boundaries between prompts at every
/// position of a paired CTA.
pub(super) const SHAPES: [&[u32]; 12] = [
    &[568; 8],
    &[554; 14],
    &[901, 902],
    &[789],
    &[1],
    &[3],
    &[1, 1, 1],
    &[63, 65],
    &[127, 129, 1],
    &[3, 5, 7, 11, 13, 17, 19, 23],
    &[64; 16],
    &[4096, 4095],
];

/// The shapes timed: the served steps and one unpacked full chunk.
const TIMED: [&[u32]; 5] = [&[568; 8], &[554; 14], &[901, 902], &[789], &[ROWS]];

fn label(lengths: &[u32]) -> String {
    let rows = lengths.iter().sum::<u32>();
    match lengths {
        [first, rest @ ..] if rest.iter().all(|length| length == first) => {
            format!("items={} length={first} rows={rows}", lengths.len())
        }
        _ => format!("items={} rows={rows}", lengths.len()),
    }
}

pub fn run(sessions: &[Session], timing: bool) -> LabResult<()> {
    exclusive(sessions)?;
    for (rank, s) in sessions.iter().enumerate() {
        for entry in entries()? {
            let LatentAttentionQuery::Attention {
                page_size,
                second_fp4_keys,
                ..
            } = entry.query
            else {
                unreachable!()
            };
            let mut rig = Rig::scheduled(s, entry.clone(), rank as u32 + 3, Schedule::Pair)?;
            if !timing {
                for rows in [4, 5, 64, 129] {
                    oracle_rows(s, &entry, Schedule::Pair, rows)?;
                }
                println!(
                    "PASS packed_attention_oracle rank={rank} variant={} rows=4,5,64,129 items=3 interleaved=true bitwise=true f64=true",
                    variant(&entry)
                );
            }
            let shapes: &[&[u32]] = if timing { &TIMED } else { &SHAPES };
            for (index, &lengths) in shapes.iter().enumerate() {
                for adversarial in [false, true] {
                    if timing && adversarial {
                        continue;
                    }
                    let seed = 200 + 40 * rank as u32 + index as u32;
                    rig.inputs =
                        Inputs::packed(s, page_size, second_fp4_keys, seed, lengths, adversarial)?;
                    let rows = lengths.iter().sum::<u32>();
                    rig.rows = rows;
                    let label = format!(
                        "rank={rank} variant={} {} adversarial={adversarial}",
                        variant(&entry),
                        label(lengths)
                    );
                    if timing {
                        let graphs = [0, 1]
                            .map(|arm| s.capture(|stream| rig.combined(stream, arm)))
                            .into_iter()
                            .collect::<LabResult<Vec<_>>>()?;
                        let total = kernel_harness::time_ab(
                            s,
                            1,
                            3,
                            [EvictionMode::Write; 2],
                            |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
                        )?;
                        println!(
                            "TIMING packed_attention {label} old_total_us={} candidate_total_us={} ratio={} samples=3 warmup=1",
                            total.arms[0].median_ms * 1000.0,
                            total.arms[1].median_ms * 1000.0,
                            total.paired_ratio
                        );
                        continue;
                    }
                    let graph = s.capture(|stream| {
                        rig.combined(stream, 0)?;
                        rig.combined(stream, 1)
                    })?;
                    // Replays with fewer live rows end inside a prompt and
                    // inside a paired CTA.
                    for live in [rows, rows - 1, rows / 2, 0, rows] {
                        rig.inputs
                            .live
                            .copy_from_bytes(&live.to_le_bytes(), &s.stream)
                            .map_err(|e| e.to_string())?;
                        rig.poison(s)?;
                        graph.launch(&s.stream).map_err(|e| e.to_string())?;
                        rig.compare(s, live)?;
                    }
                    println!(
                        "PASS packed_attention {label} workspace=true output=true replay=true guard=true"
                    );
                }
            }
        }
    }
    Ok(())
}

/// The serving fused shared-head module: one launch per row and sixteen
/// heads that writes the final output, replacing the reference attention
/// and its single-split merge.
const FINISH: &str = "latent_sources_prefill_head_finish_deepseek41_sm120";

/// The fused finish arm over a rig's inputs, launched as the bundle does:
/// the reference arguments with the partial workspace disabled, then the
/// sinks, RoPE and output views.
struct Fused {
    function: Function,
    output: DeviceBuffer,
    _module: Module,
}

impl Fused {
    fn new(s: &Session, entry: &sources::Entry) -> LabResult<Self> {
        use sources_prefill::heads;
        let module = s.module(FINISH)?;
        let function = module
            .function(&heads::next::symbol(
                entry,
                false,
                heads::next::Mode::Finish,
            ))
            .map_err(|e| e.to_string())?;
        function
            .set_max_dynamic_shared_memory(heads::SHARED_BYTES)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            function,
            output: s.allocate(ROWS as usize * HEADS * WIDTH * 2)?,
            _module: module,
        })
    }

    fn launch(&self, rig: &Rig, stream: &Stream) -> LabResult<()> {
        let mut args = rig.arguments(0, WINDOW, rig.inputs.compressed_budget);
        args[7] = Arg::NullBuffer;
        args.extend([
            Arg::Buffer(&rig.inputs.sinks),
            Arg::Buffer(&rig.inputs.cos_sin),
            Arg::Buffer(&self.output),
        ]);
        let plan = sources_prefill::heads::plan(&rig.entry, rig.rows).map_err(|e| e.to_string())?;
        launch::launch_with(&self.function, stream, launch::config(plan)?, &args)
    }

    fn poison(&self, s: &Session, rig: &Rig) -> LabResult<()> {
        let rows = rig.rows.saturating_add(1).min(ROWS) as usize;
        self.output
            .copy_from_bytes(&vec![0xa5; rows * HEADS * WIDTH * 2], &s.stream)
            .map_err(|e| e.to_string())
    }

    /// The fused output equals the reference arm's merged output on every
    /// live row, and leaves the row after the last live one untouched.
    fn compare(&self, s: &Session, rig: &Rig, live: u32) -> LabResult<()> {
        let stride = HEADS * WIDTH * 2;
        let bytes = rig.rows.saturating_add(1).min(ROWS) as usize * stride;
        let mut reference = vec![0; bytes];
        let mut fused = vec![0; bytes];
        rig.arms[0]
            .output
            .copy_to_bytes(&mut reference, &s.stream)
            .map_err(|e| e.to_string())?;
        self.output
            .copy_to_bytes(&mut fused, &s.stream)
            .map_err(|e| e.to_string())?;
        compare_host_bytes("fused output", &reference, &fused, live as usize * stride)
    }
}

/// Qualify the fused shared-head finish on packed prefill steps: on the two
/// entries with compressed keys (the only ones that have it), every live
/// row's output bitwise equal to the reference attention and merge, and the
/// time of both.
pub fn run_fused(sessions: &[Session], timing: bool) -> LabResult<()> {
    exclusive(sessions)?;
    for (rank, s) in sessions.iter().enumerate() {
        for entry in entries()? {
            let LatentAttentionQuery::Attention {
                page_size,
                second_fp4_keys,
                ..
            } = entry.query
            else {
                unreachable!()
            };
            if !second_fp4_keys {
                continue;
            }
            let mut rig = Rig::scheduled(s, entry.clone(), rank as u32 + 5, Schedule::Pair)?;
            let fused = Fused::new(s, &entry)?;
            let shapes: &[&[u32]] = if timing { &TIMED } else { &SHAPES };
            for (index, &lengths) in shapes.iter().enumerate() {
                for adversarial in [false, true] {
                    if timing && adversarial {
                        continue;
                    }
                    let seed = 600 + 40 * rank as u32 + index as u32;
                    rig.inputs =
                        Inputs::packed(s, page_size, second_fp4_keys, seed, lengths, adversarial)?;
                    let rows = lengths.iter().sum::<u32>();
                    rig.rows = rows;
                    let label = format!(
                        "rank={rank} variant={} {} adversarial={adversarial}",
                        variant(&entry),
                        label(lengths)
                    );
                    if timing {
                        let graphs = [
                            s.capture(|stream| rig.combined(stream, 0))?,
                            s.capture(|stream| fused.launch(&rig, stream))?,
                        ];
                        let total = kernel_harness::time_ab(
                            s,
                            1,
                            3,
                            [EvictionMode::Write; 2],
                            |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
                        )?;
                        println!(
                            "TIMING packed_fused {label} old_total_us={} candidate_total_us={} ratio={} samples=3 warmup=1",
                            total.arms[0].median_ms * 1000.0,
                            total.arms[1].median_ms * 1000.0,
                            total.paired_ratio
                        );
                        continue;
                    }
                    let graph = s.capture(|stream| {
                        rig.combined(stream, 0)?;
                        fused.launch(&rig, stream)
                    })?;
                    for live in [rows, rows - 1, rows / 2, 0, rows] {
                        rig.inputs
                            .live
                            .copy_from_bytes(&live.to_le_bytes(), &s.stream)
                            .map_err(|e| e.to_string())?;
                        rig.poison(s)?;
                        fused.poison(s, &rig)?;
                        graph.launch(&s.stream).map_err(|e| e.to_string())?;
                        fused.compare(s, &rig, live)?;
                    }
                    println!("PASS packed_fused {label} output=true replay=true guard=true");
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
    fn packed_shapes_fit_the_chunk_and_cross_paired_ctas() {
        for lengths in SHAPES.iter().chain(&TIMED) {
            let rows = lengths.iter().sum::<u32>();
            assert!(rows > 0 && rows <= ROWS, "{lengths:?}");
        }
        // Prompt boundaries fall on both rows of a paired CTA.
        let boundaries = SHAPES
            .iter()
            .flat_map(|lengths| {
                lengths.iter().scan(0, |end, length| {
                    *end += length;
                    Some(*end % 2)
                })
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(boundaries.len(), 2);
        assert_eq!(label(&[554; 14]), "items=14 length=554 rows=7756");
        assert_eq!(label(&[63, 65]), "items=2 rows=128");
    }
}
