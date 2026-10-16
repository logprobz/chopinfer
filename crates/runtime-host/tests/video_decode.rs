//! Extracts the red-then-blue fixture's frames on a real device. Run it with
//! `--features linux-cuda-host -- --ignored` on one GPU, for example
//! `CUDA_VISIBLE_DEVICES=2`.
#![cfg(all(target_os = "linux", feature = "linux-cuda-host"))]

use runtime_host::video::{VideoDecodeServiceV1, VideoLimitsV1, prepare_video_v1};
use runtime_media::video_frames::FrameSampling;

const FIXTURE: &[u8] = include_bytes!("../../runtime-media/tests/fixtures/red-then-blue.mp4");

fn limits(maximum_frames: usize) -> VideoLimitsV1 {
    VideoLimitsV1 {
        maximum_samples: 1 << 16,
        maximum_width: 4096,
        maximum_height: 4096,
        sampling: FrameSampling {
            maximum_frames,
            seconds_per_frame: 1,
        },
    }
}

#[test]
#[ignore = "needs an NVIDIA GPU and libnvcuvid"]
fn fixture_frames_come_back_in_display_order_as_rgb() {
    let service = VideoDecodeServiceV1::start(0).unwrap();
    let never = || false;
    let (width, height) = service.capacity();
    assert!(width >= 256 && height >= 256);

    let frames = service
        .decode(
            prepare_video_v1(FIXTURE, &limits(32)).unwrap(),
            limits(32),
            &never,
        )
        .unwrap()
        .frames;
    let summary: Vec<_> = frames
        .iter()
        .map(|frame| {
            assert_eq!((frame.width, frame.height), (256, 256));
            assert!(frame.rgb.chunks(3).all(|pixel| pixel == &frame.rgb[..3]));
            (
                frame.label_seconds,
                [frame.rgb[0], frame.rgb[1], frame.rgb[2]],
            )
        })
        .collect();
    let red = [253, 0, 0];
    let blue = [0, 0, 254];
    assert_eq!(
        summary,
        [
            (0, red),
            (1, red),
            (2, red),
            (3, red),
            (4, blue),
            (5, blue),
            (6, blue),
            (7, blue)
        ]
    );

    // Three uniform frames: display frames 0, 3 and 7.
    let sparse = service
        .decode(
            prepare_video_v1(FIXTURE, &limits(3)).unwrap(),
            limits(3),
            &never,
        )
        .unwrap()
        .frames;
    let colors: Vec<_> = sparse.iter().map(|frame| frame.rgb[2]).collect();
    assert_eq!(colors, [0, 0, 254]);
}
