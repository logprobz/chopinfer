//! Full selected-ID lists beside transactional taps for a single 32K prompt.
use crate::{
    buffers::{RankBuffers, index_selected, scratch_rows},
    program::Stage,
    shape::Shape,
};
use bundle_runtime::inputs::PhaseInputs;
use runtime_bundle_abi::{BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1};
use runtime_bundle_sdk::taps::TapError;
use std::{ffi::OsString, fs, path::Path};

pub(super) const ENV: &str = "CHOPINFER_TAPS_PREFILL_SELECTED";
pub(super) fn requested(value: Option<OsString>) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" => Ok(true),
        Some(_) => Err(format!("{ENV} must be 1 or unset")),
    }
}
fn offset(positions: &[i32], live: u32, batch: u32, decode: bool) -> Option<u32> {
    if decode || batch != 1 || live != 8192 || positions.len() < 8192 {
        return None;
    }
    let start = u32::try_from(*positions.first()?).ok()?;
    if ![0, 8192, 16384, 24576].contains(&start) {
        return None;
    }
    positions
        .iter()
        .take(8192)
        .enumerate()
        .all(|(i, &p)| u32::try_from(p).ok() == Some(start + i as u32))
        .then_some(start)
}
pub(super) fn capture(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    buffers: &RankBuffers,
    shape: &Shape,
    inputs: &PhaseInputs,
    stage: Stage,
    dir: &Path,
) -> Result<(), TapError> {
    let Stage::LatentAttention(layer) = stage else {
        return Ok(());
    };
    let Some(offset) = offset(
        &inputs.positions,
        inputs.live_tokens(),
        inputs.live_batch(),
        inputs.decode,
    ) else {
        return Ok(());
    };
    let g = shape.geometry;
    let read = |span: DeviceBufferSpanV1, bytes: u64| -> Result<Vec<u8>, TapError> {
        if bytes > span.length {
            return Err(std::io::Error::other("selected capture exceeds its span").into());
        }
        let mut output = vec![0; bytes as usize];
        services
            .copy_device_to_host(
                stream,
                &mut output,
                DeviceBufferSpanV1::new(span.buffer, span.offset, bytes),
            )
            .map_err(TapError::Host)?;
        services
            .synchronize_stream(stream)
            .map_err(TapError::Host)?;
        Ok(output)
    };
    if layer == 0 {
        fs::write(
            dir.join("window-keys.i32"),
            read(buffers.scratch.window_keys, 8192 * 128 * 4)?,
        )?;
    }
    let Some(source) = model_deepseek41_contract::indexer::sources(g, layer) else {
        return Ok(());
    };
    let index = g
        .shape
        .csa
        .index_sources
        .iter()
        .position(|&l| l == source.selection)
        .ok_or_else(|| std::io::Error::other("selected capture has no source bank"))?;
    let top_k = g.shape.indexer.top_k;
    let span = index_selected(buffers.scratch.index_selected, index, scratch_rows(), top_k);
    let sub = dir.join(format!("selected-L{layer:02}"));
    fs::create_dir(&sub)?;
    fs::write(
        sub.join("selected.i32"),
        read(span, 8192 * u64::from(top_k) * 4)?,
    )?;
    fs::write(
        sub.join("selected.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"chopinfer.prefill-selected.v1","rank":0,"replicated":true,
            "layer":layer,"selection_source":source.selection,"key_source":source.keys,
            "ratio":g.compress_ratio(source.keys),"offset":offset,"rows":8192,"top_k":top_k,
            "positions":&inputs.positions[..8192],"token_ids":&inputs.token_ids[..8192]
        }))
        .map_err(std::io::Error::other)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn four_full_chunks_include_the_last_valid_position() {
        for start in [0, 8192, 16384, 24576] {
            let positions = (start..start + 8192).collect::<Vec<i32>>();
            assert_eq!(offset(&positions, 8192, 1, false), Some(start as u32));
            assert_eq!(offset(&positions, 8191, 1, false), None);
            assert_eq!(offset(&positions, 8192, 2, false), None);
        }
        assert_eq!(
            offset(&(32768..40960).collect::<Vec<_>>(), 8192, 1, false),
            None
        );
        assert_eq!(offset(&[0], 8192, 1, false), None);
        assert!(!requested(None).unwrap());
        assert!(requested(Some("1".into())).unwrap());
        assert!(requested(Some("0".into())).is_err());
    }
}
