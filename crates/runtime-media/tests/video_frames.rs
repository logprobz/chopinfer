use runtime_media::mp4::{Mp4Limits, demux_video};
use runtime_media::video_frames::{
    FrameSampling, SampledFrame, VideoFrameError, frame_plan, nv12_to_rgb, sampling_total,
    uniform_frame_indices,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/red-then-blue.mp4");

fn flat_nv12(width: u32, height: u32, y: u8, u: u8, v: u8) -> Vec<u8> {
    let luma = (width * height) as usize;
    let mut frame = vec![y; luma];
    for _ in 0..luma / 4 {
        frame.extend_from_slice(&[u, v]);
    }
    frame
}

#[test]
fn flat_codes_convert_with_simd_rounding() {
    let cases = [
        ((81, 90, 240), [253, 0, 0]),
        ((41, 240, 110), [0, 0, 254]),
        ((16, 128, 128), [0, 0, 0]),
        ((235, 128, 128), [255, 255, 255]),
        ((126, 128, 128), [128, 128, 128]),
    ];
    for ((y, u, v), expected) in cases {
        let rgb = nv12_to_rgb(4, 2, &flat_nv12(4, 2, y, u, v)).unwrap();
        assert!(
            rgb.chunks(3).all(|pixel| pixel == expected),
            "{y} {u} {v}: {rgb:?}"
        );
    }
}

#[test]
fn each_chroma_sample_covers_one_two_by_two_block() {
    // Two chroma samples per row: the left block is red chroma, the right blue.
    let mut frame = vec![81, 81, 41, 41, 81, 81, 41, 41];
    frame.extend_from_slice(&[90, 240, 240, 110]);
    let rgb = nv12_to_rgb(4, 2, &frame).unwrap();
    for row in rgb.chunks(12) {
        assert_eq!(row, [253, 0, 0, 253, 0, 0, 0, 0, 254, 0, 0, 254]);
    }
}

#[test]
fn malformed_frames_are_refused() {
    assert_eq!(
        nv12_to_rgb(3, 2, &[0; 9]),
        Err(VideoFrameError::Geometry {
            width: 3,
            height: 2
        })
    );
    assert_eq!(
        nv12_to_rgb(4, 2, &[0; 11]),
        Err(VideoFrameError::Length {
            expected: 12,
            actual: 11
        })
    );
}

#[test]
fn sampling_matches_uniform_linspace() {
    assert_eq!(uniform_frame_indices(8, 32), (0..8).collect::<Vec<_>>());
    assert_eq!(
        uniform_frame_indices(100, 32),
        [
            0, 3, 6, 9, 12, 15, 19, 22, 25, 28, 31, 35, 38, 41, 44, 47, 51, 54, 57, 60, 63, 67, 70,
            73, 76, 79, 83, 86, 89, 92, 95, 99
        ]
    );
    assert_eq!(uniform_frame_indices(5, 1), [0]);
    assert!(uniform_frame_indices(0, 32).is_empty());
}

#[test]
fn edit_lists_replace_the_header_count_only_past_one_frame() {
    assert_eq!(sampling_total(8, 8), 8);
    assert_eq!(sampling_total(8, 7), 8);
    assert_eq!(sampling_total(8, 6), 6);
    assert_eq!(sampling_total(8, 0), 8);
}

#[test]
fn plans_drop_unreadable_indices_and_relabel() {
    let sampling = FrameSampling {
        maximum_frames: 32,
        seconds_per_frame: 1,
    };
    let plan = frame_plan(8, 7, sampling).unwrap();
    assert_eq!(plan.len(), 7);
    assert_eq!(
        plan[6],
        SampledFrame {
            display_index: 6,
            label_seconds: 6
        }
    );
    let sparse = frame_plan(
        100,
        100,
        FrameSampling {
            maximum_frames: 4,
            seconds_per_frame: 2,
        },
    )
    .unwrap();
    let labels: Vec<_> = sparse
        .iter()
        .map(|frame| (frame.display_index, frame.label_seconds))
        .collect();
    assert_eq!(labels, [(0, 0), (33, 2), (66, 4), (99, 6)]);
    assert_eq!(frame_plan(0, 0, sampling), Err(VideoFrameError::NoFrames));
}

#[test]
fn edit_list_segments_bound_presentable_frames() {
    let track = demux_video(FIXTURE, &Mp4Limits::default()).unwrap();
    assert_eq!(track.edit_duration, Some(65536));
    assert_eq!(track.presentable_order(), track.presentation_order());

    // Halve the edit segment to 2000 movie ticks (2 s): four frames remain.
    let mut trimmed = FIXTURE.to_vec();
    let duration = 1190 + 16;
    trimmed[duration..duration + 4].copy_from_slice(&2000u32.to_be_bytes());
    let track = demux_video(&trimmed, &Mp4Limits::default()).unwrap();
    assert_eq!(track.edit_duration, Some(32768));
    let presentable = track.presentable_order();
    assert_eq!(presentable, [0, 2, 3, 1]);
    let sampling = FrameSampling {
        maximum_frames: 32,
        seconds_per_frame: 1,
    };
    let plan = frame_plan(track.samples.len(), presentable.len(), sampling).unwrap();
    assert_eq!(plan.len(), 4);
}
