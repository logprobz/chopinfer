//! Tap capture mechanics over model-owned steps, filters and occurrences.

use runtime_bundle_abi::{BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1};
use runtime_bundle_sdk::taps::{EncoderRows, EncoderSegment, TapError, TapSpec, TapStep, Taps};

use crate::encoder::BlockRows;
use crate::inputs::PhaseInputs;
use crate::media::{OccurrenceStep, Occurrences};

/// Checks both filters against all declared sites and reports text selection.
pub fn selects_text(taps: &Taps, encoders: &[TapSpec], text: &[TapSpec]) -> Result<bool, String> {
    let every = encoders.iter().chain(text).cloned().collect::<Vec<_>>();
    taps.check_layers(&every)?;
    taps.check_sites(&every)?;
    Ok(text.iter().any(|spec| taps.selected(spec)))
}

/// Whether the encoder must read back any selected block site.
pub fn selects_blocks(taps: Option<&Taps>, specs: &[TapSpec], output: &str) -> bool {
    taps.is_some_and(|taps| {
        specs
            .iter()
            .any(|spec| spec.name != output && taps.selected(spec))
    })
}

/// Opens a phase, attaching encoder rows before declaring selected text sites.
pub fn begin(
    taps: &mut Taps,
    program: &str,
    ranks: u32,
    inputs: &PhaseInputs,
    encoders: &mut Vec<EncoderRows>,
    specs: impl IntoIterator<Item = TapSpec>,
) -> Result<TapStep, String> {
    let live = inputs.live_tokens();
    let dir = taps.next_step().map_err(|error| error.to_string())?;
    let mut capture = taps
        .begin(&dir, program, ranks, live, &inputs.positions)
        .map_err(|error| error.to_string())?;
    for rows in std::mem::take(encoders) {
        capture.encoder(rows).map_err(|error| error.to_string())?;
    }
    for spec in specs {
        if taps.selected(&spec) {
            capture.declare(&spec).map_err(|error| error.to_string())?;
        }
    }
    capture
        .open(&inputs.tap_items(program, live, capture.rows(), taps.from()))
        .map_err(|error| error.to_string())?;
    Ok(capture)
}

/// Reads each selected site's shards through the model's buffer mapping.
pub fn capture(
    taps: &Taps,
    capture: &mut TapStep,
    services: &HostServicesV1,
    streams: &[BundleStreamHandleV1],
    spec: &TapSpec,
    mut buffer: impl FnMut(usize) -> DeviceBufferSpanV1,
) -> Result<(), TapError> {
    if taps.selected(spec) {
        for rank in 0..spec.sharding.ranks(streams.len() as u32) {
            let index = rank as usize;
            capture.capture(services, streams[index], spec, rank, buffer(index))?;
        }
    }
    Ok(())
}

/// Leaves the open step pending until commit or cancellation.
pub fn end(open: &mut Option<TapStep>, pending: &mut Vec<TapStep>) {
    if let Some(capture) = open.take() {
        pending.push(capture);
    }
}

/// Cancels an unfinished step and ends every pending step, even after an error.
pub fn finish(
    open: &mut Option<TapStep>,
    pending: &mut Vec<TapStep>,
    commit: bool,
) -> Result<(), String> {
    let mut result = match open.take() {
        Some(stale) => stale.cancel(),
        None => Ok(()),
    };
    for capture in std::mem::take(pending) {
        let ended = if commit {
            capture.finish()
        } else {
            capture.cancel()
        };
        result = result.and(ended);
    }
    result.map_err(|error| error.to_string())
}

/// The block taps `taps` selects of each occurrence `step` begins: the
/// encoder runtime's block rows of the step's encodes are kept per
/// encode (`Occurrences::keep`), and each occurrence gets its encode's
/// whole rows, in the item's row order, under its request, kind and
/// item, whichever step or request ran the encode.
pub fn block_rows(
    taps: &Taps,
    output: &str,
    specs: &[TapSpec],
    occurrences: &mut Occurrences,
    step: &OccurrenceStep,
    rows: Vec<BlockRows>,
) -> Result<Vec<EncoderRows>, String> {
    let spec = |name: &str| {
        specs
            .iter()
            .find(|spec| spec.name == name && name != output)
            .ok_or_else(|| format!("{name} is not a declared block tap"))
    };
    for block in rows {
        let spec = spec(&block.name)?;
        if !taps.selected(spec) {
            continue;
        }
        if block.bytes.len() as u64 % (u64::from(spec.cols) * spec.dtype.bytes()) != 0 {
            return Err(format!("{}: a partial row", block.name));
        }
        let encode = step
            .encodes
            .get(block.record)
            .ok_or_else(|| format!("{}: rows of no encode", block.name))?;
        occurrences.keep(*encode, block.name, block.bytes);
    }
    let mut out: Vec<EncoderRows> = Vec::new();
    for read in step.reads.iter().filter(|read| read.begins) {
        for (name, bytes) in occurrences.kept(read.encode) {
            let spec = spec(name)?;
            let rows = bytes.len() as u64 / (u64::from(spec.cols) * spec.dtype.bytes());
            let at = match out.iter().position(|tap| tap.spec.name == *name) {
                Some(at) => at,
                None => {
                    out.push(EncoderRows {
                        spec: spec.clone(),
                        bytes: Vec::new(),
                        segments: Vec::new(),
                    });
                    out.len() - 1
                }
            };
            let tap = &mut out[at];
            let start = tap.rows();
            tap.segments.push(EncoderSegment {
                start,
                index: 0,
                rows: rows as u32,
                ..read.segment
            });
            tap.bytes.extend_from_slice(bytes);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_sdk::taps::TapFilter;

    #[test]
    fn finish_ends_every_pending_step_after_an_earlier_error() {
        for commit in [false, true] {
            let root = std::env::temp_dir().join(format!(
                "bundle-taps-finish-{}-{commit}",
                std::process::id()
            ));
            let mut taps = Taps::new(root.clone(), TapFilter::default());
            let mut step = || {
                let dir = taps.next_step().unwrap();
                let mut capture = taps.begin(&dir, "Prefill", 1, 1, &[0]).unwrap();
                capture.open(&serde_json::json!({})).unwrap();
                (dir, capture)
            };
            let (stale_dir, stale) = step();
            let (bad_dir, bad) = step();
            let (last_dir, last) = step();
            if commit {
                std::fs::write(bad_dir.join("completion.json"), b"occupied").unwrap();
            } else {
                std::fs::remove_dir_all(&bad_dir).unwrap();
                std::fs::write(&bad_dir, b"not a directory").unwrap();
            }
            let mut open = Some(stale);
            let mut pending = vec![bad, last];
            assert!(finish(&mut open, &mut pending, commit).is_err());
            assert!(open.is_none());
            assert!(pending.is_empty());
            assert!(!stale_dir.exists());
            if commit {
                assert!(last_dir.join("completion.json").is_file());
            } else {
                assert!(!last_dir.exists());
            }
            std::fs::remove_dir_all(root).unwrap();
        }
    }
}
