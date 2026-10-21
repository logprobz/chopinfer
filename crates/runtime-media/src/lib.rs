//! Model-neutral CPU media decoding and preprocessing.

pub mod audio;
mod core;
pub mod gif;
pub mod image;
pub mod jpeg;
pub mod mp4;
pub mod png;
pub mod recipe;
pub mod remote;
pub mod video_frames;
pub mod wav;
pub mod webp;

pub use core::{
    Alpha, DecodedAudio, MediaError, MediaLimits, MediaResult, MediaSource, Orientation, Probe,
    RgbImage, content_key, content_key_streamed, decode_base64, decode_data_url, probe,
    probe_oriented, read_source,
};
pub use image::{ImageFormat, ImagePolicy, decode_image, decode_working_bytes, probe_image};
