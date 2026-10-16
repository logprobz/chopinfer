//! Decodes the red-then-blue fixture on a real device. Run it with
//! `--ignored` on one GPU, for example `CUDA_VISIBLE_DEVICES=2`.

use runtime_cuda::{AnnexBStream, NvdecError, NvdecSession};

/// The fixture's Annex B stream: parameter sets, then 8 access units in
/// decode order (4 red frames, then 4 blue), 256x256.
const STREAM: &[u8] = include_bytes!("fixtures/red-then-blue.h264");

#[test]
#[ignore = "needs an NVIDIA GPU and libnvcuvid"]
fn fixture_decodes_to_four_red_then_four_blue_frames() {
    let session = NvdecSession::open(0).unwrap();
    let (max_width, max_height) = session.h264_capacity().unwrap();
    assert!(max_width >= 256 && max_height >= 256);

    let mut stream = AnnexBStream::new();
    stream
        .push_unit(0, |out| {
            out.extend_from_slice(STREAM);
            Ok::<(), ()>(())
        })
        .unwrap();
    let mut planes = Vec::new();
    session
        .decode_h264(&stream, 4096, 4096, &mut |_| true, &mut |frame| {
            assert_eq!((frame.width, frame.height), (256, 256));
            let luma = &frame.nv12[..256 * 256];
            let chroma = &frame.nv12[256 * 256..];
            assert!(luma.iter().all(|&y| y == luma[0]));
            planes.push((luma[0], chroma[0], chroma[1]));
            true
        })
        .unwrap();
    let red = (81, 90, 240);
    let blue = (41, 240, 110);
    assert_eq!(planes, [red, red, red, red, blue, blue, blue, blue]);

    // A limit below the stream's geometry is refused before decoding.
    let refused = session.decode_h264(&stream, 128, 128, &mut |_| true, &mut |_| true);
    assert!(matches!(refused, Err(NvdecError::Unsupported)));

    // A consumer that stops after one frame ends the decode.
    let stopped = session.decode_h264(&stream, 4096, 4096, &mut |_| true, &mut |_| false);
    assert!(matches!(stopped, Err(NvdecError::Stopped)));
}
