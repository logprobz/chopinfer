//! A step's images on an instance serving the vision encoder
//! (`crate::encoders`): the encoder runs the step's encodes before the step
//! opens, and a prefill whose rows read the arena overrides their embedding
//! rows after the embedding's all-reduce. Those rows are image rows: the
//! router selects them with `ffn.gate.bias_vl` (`bias_select` 1) and the
//! Engram gate closes on them (text flag 0), as the vendor's image mask
//! (`token_types >= 0`) does. The router's flags return to zero (text)
//! after the prefill, so the decode and verify graphs read text rows.

use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::pipeline::PreparedLaunch;
use runtime_bundle_abi::{BundleAbiError, BundleStatusV1, DeviceBufferSpanV1};
use runtime_bundle_sdk::BatchCodecV1;

use super::Instance;

/// The image and text flags of a prefill's live rows, one U8 a row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PrefillMedia {
    pub(super) image: Vec<u8>,
    pub(super) text: Vec<u8>,
}

/// The flags of a prefill whose rows read the arena at `source_rows` (-1
/// keeps the token's embedding), whose tokens are `tokens`: `None` when no
/// row reads it. Every image row carries the image token and no text row
/// does, as the vendor lays an image span; a step that differs is refused.
pub(super) fn flags(
    source_rows: &[i64],
    tokens: &[u32],
    image_token: u32,
) -> Result<Option<PrefillMedia>, String> {
    if source_rows.iter().all(|&row| row < 0) {
        return Ok(None);
    }
    if source_rows.len() != tokens.len() {
        return Err("the arena sources do not cover the prefill's tokens".into());
    }
    let image = crate::encoders::image_rows(source_rows);
    if let Some(row) = image
        .iter()
        .zip(tokens)
        .position(|(&image, &token)| (image == 1) != (token == image_token))
    {
        return Err(format!(
            "prefill row {row} reads the arena only where its token is not the image token, or the other way"
        ));
    }
    Ok(Some(PrefillMedia {
        text: crate::encoders::text_rows(source_rows),
        image,
    }))
}

impl Instance {
    /// Runs the step's encodes on every rank and uploads each rank's
    /// prefill source rows; the flags of the prefill's rows, `None` when
    /// none reads the arena or the instance serves no images.
    pub(super) fn encode(
        &self,
        request: &[u8],
        prefill: &PhaseInputs,
    ) -> Result<Option<PrefillMedia>, BundleStatusV1> {
        let Some(media) = &self.media else {
            return Ok(None);
        };
        let streams = self
            .ranks
            .iter()
            .map(|&(_, _, stream)| stream)
            .collect::<Vec<_>>();
        let injects = media.encoder.run(
            &self.services,
            &streams,
            &self.symbols,
            request,
            prefill,
            &self.collectives,
        )?;
        if !injects {
            return Ok(None);
        }
        let batch = BatchCodecV1::decode(request).map_err(|_| BundleStatusV1::InvalidArgument)?;
        let plan = crate::encoders::ARENA
            .plan(batch.media(), prefill, batch.items().len())
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let image_token = self.config.shape.geometry.shape.tokens.image;
        flags(&plan.source_rows, &prefill.token_ids, image_token).map_err(|error| {
            self.log(&format!("images: {error}"));
            BundleStatusV1::InvalidArgument
        })
    }

    /// Rank `rank`'s launch that overrides the image rows of the embedding
    /// `embedded` (`rows` rows) by their arena rows.
    pub(super) fn inject(
        &self,
        rank: usize,
        embedded: DeviceBufferSpanV1,
        live_tokens: DeviceBufferSpanV1,
        rows: u32,
    ) -> Result<PreparedLaunch, String> {
        let media = self.media.as_ref().ok_or("the instance serves no images")?;
        media.encoder.ranks[rank]
            .inject_part(embedded, live_tokens, 0, rows)
            .map_err(|refusal| format!("encoder inject: {}", refusal.0))
    }

    /// Rank `rank`'s text flags of the prefill rows.
    pub(super) fn text_flags(&self, rank: usize) -> Result<DeviceBufferSpanV1, String> {
        self.media
            .as_ref()
            .and_then(|media| media.text.get(rank).copied())
            .ok_or_else(|| "the instance serves no images".to_owned())
    }

    /// Uploads `media`'s flags to rank `rank`: the router's bias choice and
    /// the Engram gate's text flags.
    pub(super) fn upload_flags(
        &self,
        rank: usize,
        media: &PrefillMedia,
    ) -> Result<(), BundleAbiError> {
        let (_, _, stream) = self.ranks[rank];
        let scratch = &self.buffers[rank].scratch;
        bundle_runtime::resources::upload_bytes(
            &self.services,
            stream,
            scratch.bias_select,
            &media.image,
        )?;
        let text = self
            .text_flags(rank)
            .map_err(|_| BundleAbiError::InvalidRange("deepseek41 text flags"))?;
        bundle_runtime::resources::upload_bytes(&self.services, stream, text, &media.text)
    }

    /// Returns every rank's router bias choice to text after a prefill
    /// that selected image rows.
    pub(super) fn clear_flags(&self) -> Result<(), BundleAbiError> {
        for (rank, &(_, _, stream)) in self.ranks.iter().enumerate() {
            self.services
                .zero_device_buffer(stream, self.buffers[rank].scratch.bias_select)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_rows_carry_the_image_token_alone() {
        let image = 900;
        let sources = [-1, 4, 5, 6, -1];
        let tokens = [11, image, image, image, 12];
        let media = flags(&sources, &tokens, image).unwrap().unwrap();
        assert_eq!(media.image, [0, 1, 1, 1, 0]);
        assert_eq!(media.text, [1, 0, 0, 0, 1]);
        // No row reads the arena: a text prefill.
        assert_eq!(flags(&[-1, -1], &[11, 12], image).unwrap(), None);
        // An image row without the image token, or the image token on a
        // text row, is refused.
        assert!(super::flags(&sources, &[11, image, 13, image, 12], image).is_err());
        assert!(super::flags(&sources, &[image, image, image, image, 12], image).is_err());
        assert!(super::flags(&sources, &tokens[..4], image).is_err());
    }
}
