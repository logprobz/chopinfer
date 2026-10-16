use runtime_media::mp4::{AvcConfiguration, Mp4Error, Mp4Limits, VideoCodec, demux_video};
use sha2::{Digest, Sha256};

const FIXTURE: &[u8] = include_bytes!("fixtures/red-then-blue.mp4");

fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

fn full(kind: &[u8; 4], version: u8, body: &[u8]) -> Vec<u8> {
    let mut payload = vec![version, 0, 0, 0];
    payload.extend_from_slice(body);
    boxed(kind, &payload)
}

fn words(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect()
}

fn movie_header() -> Vec<u8> {
    full(b"mvhd", 0, &words(&[0, 0, 1000, 0]))
}

fn avc1(codec: &[u8; 4]) -> Vec<u8> {
    let mut visual = vec![0; 24];
    visual.extend_from_slice(&64u16.to_be_bytes());
    visual.extend_from_slice(&32u16.to_be_bytes());
    visual.extend_from_slice(&[0; 50]);
    let avcc = [
        1, 0x42, 0, 0x1e, 0xfd, 0xe1, 0, 2, 0x67, 1, 1, 0, 2, 0x68, 2,
    ];
    visual.extend_from_slice(&boxed(b"avcC", &avcc));
    boxed(codec, &visual)
}

/// Builds a two-sample progressive file with 64-bit chunk offsets and 8-bit
/// compact sample sizes. `extra` is appended inside `moov`.
fn synthetic(codec: &[u8; 4], extra: &[u8], timescale: u32) -> Vec<u8> {
    let samples = [0u8, 3, 0x65, 0xaa, 0xbb, 0, 2, 0x41, 0x9a];
    let ftyp = boxed(b"ftyp", b"isom\0\0\0\0isom");
    let mdat_offset = (ftyp.len() + 8) as u64;
    let mut stsd = words(&[1]);
    stsd.extend_from_slice(&avc1(codec));
    let mut stz2 = vec![0, 0, 0, 8];
    stz2.extend_from_slice(&words(&[2]));
    stz2.extend_from_slice(&[5, 4]);
    let mut co64 = words(&[1]);
    co64.extend_from_slice(&mdat_offset.to_be_bytes());
    let stbl = [
        full(b"stsd", 0, &stsd),
        full(b"stts", 0, &words(&[1, 2, 100])),
        full(b"stsc", 0, &words(&[1, 1, 2, 1])),
        full(b"stz2", 0, &stz2),
        full(b"co64", 0, &co64),
    ]
    .concat();
    let minf = boxed(b"minf", &boxed(b"stbl", &stbl));
    let mdia = [
        full(b"mdhd", 0, &words(&[0, 0, timescale, 200, 0])),
        full(
            b"hdlr",
            0,
            &[words(&[0]), b"vide".to_vec(), vec![0; 13]].concat(),
        ),
        minf,
    ]
    .concat();
    let moov = boxed(
        b"moov",
        &[
            movie_header(),
            boxed(b"trak", &boxed(b"mdia", &mdia)),
            extra.to_vec(),
        ]
        .concat(),
    );
    [ftyp, boxed(b"mdat", &samples), moov].concat()
}

#[test]
fn fixture_resolves_every_sample_in_display_order() {
    let track = demux_video(FIXTURE, &Mp4Limits::default()).unwrap();
    assert_eq!((track.width, track.height), (256, 256));
    assert_eq!((track.timescale, track.duration), (16384, 65536));
    assert_eq!(track.edit_media_time, 16384);
    let VideoCodec::Avc(config) = &track.codec;
    assert_eq!(
        (config.profile, config.level, config.length_size),
        (100, 12, 4)
    );
    assert_eq!(config.sequence_parameter_sets.len(), 1);
    assert_eq!(config.picture_parameter_sets.len(), 1);
    assert_eq!(track.samples.len(), 8);
    assert_eq!(track.samples[0].offset, 48);
    assert_eq!(track.samples[0].size, 758);
    let sync: Vec<_> = track.samples.iter().map(|sample| sample.sync).collect();
    assert_eq!(sync, [true, false, false, false, true, false, false, false]);
    let times: Vec<_> = track
        .presentation_order()
        .into_iter()
        .map(|index| track.presentation_time(&track.samples[index]))
        .collect();
    assert_eq!(times, (0..8).map(|frame| frame * 8192).collect::<Vec<_>>());
}

#[test]
fn fixture_samples_convert_to_annex_b() {
    let track = demux_video(FIXTURE, &Mp4Limits::default()).unwrap();
    let VideoCodec::Avc(config) = &track.codec;
    let mut stream = Vec::new();
    config.write_parameter_sets(&mut stream);
    assert_eq!(&stream[..5], &[0, 0, 0, 1, 0x67]);
    for index in 0..track.samples.len() {
        let before = stream.len();
        config
            .write_annex_b(track.sample_bytes(FIXTURE, index).unwrap(), &mut stream)
            .unwrap();
        assert_eq!(&stream[before..before + 4], &[0, 0, 0, 1]);
    }
    let idr = stream
        .windows(5)
        .filter(|window| window[..4] == [0, 0, 0, 1] && window[4] & 0x1f == 5)
        .count();
    assert_eq!(idr, 2);
    assert_eq!(
        format!("{:x}", Sha256::digest(&stream)),
        "9c9918bc4a45e3e8ea10ba604917be99559eb30f0a0e1cf2ec11a1b05d0ed1cb"
    );
}

#[test]
fn compact_sizes_and_wide_offsets_resolve() {
    let file = synthetic(b"avc1", &[], 1000);
    let track = demux_video(&file, &Mp4Limits::default()).unwrap();
    let VideoCodec::Avc(config) = &track.codec;
    assert_eq!(config.length_size, 2);
    assert_eq!(track.samples.len(), 2);
    assert_eq!(track.samples[1].offset, track.samples[0].offset + 5);
    assert_eq!(track.samples[1].decode_time, 100);
    assert!(track.samples.iter().all(|sample| sample.sync));
    let mut stream = Vec::new();
    config
        .write_annex_b(track.sample_bytes(&file, 0).unwrap(), &mut stream)
        .unwrap();
    assert_eq!(stream, [0, 0, 0, 1, 0x65, 0xaa, 0xbb]);
}

#[test]
fn refusals_name_the_cause() {
    let limits = Mp4Limits::default();
    assert_eq!(
        demux_video(&synthetic(b"hvc1", &[], 1000), &limits),
        Err(Mp4Error::Unsupported {
            detail: "video codec hvc1".into()
        })
    );
    assert_eq!(
        demux_video(&synthetic(b"avc1", &boxed(b"mvex", &[]), 1000), &limits),
        Err(Mp4Error::Unsupported {
            detail: "fragmented MP4".into()
        })
    );
    assert_eq!(
        demux_video(
            &synthetic(b"avc1", &[], 1000),
            &Mp4Limits { maximum_samples: 1 }
        ),
        Err(Mp4Error::Limit {
            surface: "sample",
            limit: 1,
            actual: 2
        })
    );
    assert!(matches!(
        demux_video(&synthetic(b"avc1", &[], 0), &limits),
        Err(Mp4Error::Malformed {
            surface: "mdhd",
            ..
        })
    ));
    let truncated = &FIXTURE[..FIXTURE.len() - 1];
    assert_eq!(
        demux_video(truncated, &limits),
        Err(Mp4Error::Truncated { surface: "file" })
    );
    // The single stco entry sits after the box header, version and count.
    let mut past_end = FIXTURE.to_vec();
    let entry = 1742 + 16;
    past_end[entry..entry + 4].copy_from_slice(&(FIXTURE.len() as u32 - 10).to_be_bytes());
    assert_eq!(
        demux_video(&past_end, &limits),
        Err(Mp4Error::Truncated { surface: "sample" })
    );
    assert_eq!(
        demux_video(&boxed(b"moov", &movie_header()), &limits),
        Err(Mp4Error::NoVideoTrack)
    );
}

#[test]
fn malformed_nal_lengths_are_refused() {
    let config = AvcConfiguration {
        profile: 66,
        compatibility: 0,
        level: 30,
        length_size: 4,
        sequence_parameter_sets: vec![vec![0x67]],
        picture_parameter_sets: vec![vec![0x68]],
    };
    let mut out = Vec::new();
    assert!(matches!(
        config.write_annex_b(&[0, 0, 0, 9, 0x65], &mut out),
        Err(Mp4Error::Truncated { .. })
    ));
    assert!(matches!(
        config.write_annex_b(&[0, 0, 0, 0], &mut out),
        Err(Mp4Error::Malformed { .. })
    ));
}

/// The review's repeated-offset container, with no duplicated payload storage.
fn repeated_samples(sample: &[u8], count: u32, length_size: u8) -> Vec<u8> {
    let ftyp = boxed(b"ftyp", b"isom\0\0\0\0isom");
    let offset = (ftyp.len() + 8) as u32;
    let mut description = words(&[1]);
    let mut entry = avc1(b"avc1");
    let avcc = entry.windows(4).position(|bytes| bytes == b"avcC").unwrap();
    entry[avcc + 8] = 0xfc | (length_size - 1);
    description.extend(entry);
    let mut offsets = words(&[count]);
    for _ in 0..count {
        offsets.extend(offset.to_be_bytes());
    }
    let stbl = [
        full(b"stsd", 0, &description),
        full(b"stts", 0, &words(&[1, count, 100])),
        full(b"stsc", 0, &words(&[1, 1, 1, 1])),
        full(b"stsz", 0, &words(&[sample.len() as u32, count])),
        full(b"stco", 0, &offsets),
    ]
    .concat();
    let mdia = [
        full(b"mdhd", 0, &words(&[0, 0, 1000, count * 100, 0])),
        full(
            b"hdlr",
            0,
            &[words(&[0]), b"vide".to_vec(), vec![0; 13]].concat(),
        ),
        boxed(b"minf", &boxed(b"stbl", &stbl)),
    ]
    .concat();
    [
        ftyp,
        boxed(b"mdat", sample),
        boxed(
            b"moov",
            &[movie_header(), boxed(b"trak", &boxed(b"mdia", &mdia))].concat(),
        ),
    ]
    .concat()
}

#[test]
fn repeated_offsets_cannot_expand_past_the_sample_byte_budget() {
    let mut sample = vec![0xff, 0xff];
    sample.extend(vec![0x65; 65_535]);
    let file = repeated_samples(&sample, 1024, 2);
    assert!(file.len() < 100_000);
    assert!(
        matches!(
            demux_video(
                &file,
                &Mp4Limits {
                    maximum_samples: 65_536
                }
            ),
            Err(Mp4Error::Limit {
                surface: "sample bytes",
                limit: 67_108_864,
                ..
            })
        ),
        "repeated offsets must be charged for every sample"
    );
}

#[test]
fn short_nals_cannot_expand_past_the_annex_b_byte_budget() {
    // 32 MiB of sample data becomes 80 MiB of Annex B data. The file is < 300 KiB.
    let file = repeated_samples(&[1, 0x65].repeat(256), 65_536, 1);
    assert!(file.len() < 300_000);
    assert!(
        matches!(
            demux_video(
                &file,
                &Mp4Limits {
                    maximum_samples: 65_536
                }
            ),
            Err(Mp4Error::Limit {
                surface: "Annex B bytes",
                limit: 67_108_864,
                ..
            })
        ),
        "start codes and parameter sets must count against the byte budget"
    );
}

#[test]
fn repeated_offsets_within_budget_preserve_access_units() {
    let file = repeated_samples(&[0, 3, 0x65, 0xaa, 0xbb], 1024, 2);
    let track = demux_video(&file, &Mp4Limits::default()).unwrap();
    let VideoCodec::Avc(config) = &track.codec;
    let mut stream = Vec::new();
    config.write_parameter_sets(&mut stream);
    for index in 0..track.samples.len() {
        config
            .write_annex_b(track.sample_bytes(&file, index).unwrap(), &mut stream)
            .unwrap();
    }
    let mut expected = vec![0, 0, 0, 1, 0x67, 1, 0, 0, 0, 1, 0x68, 2];
    expected.extend([0, 0, 0, 1, 0x65, 0xaa, 0xbb].repeat(1024));
    assert_eq!(stream, expected);
    assert_eq!((track.width, track.height), (64, 32));
    assert_eq!(track.samples.len(), 1024);
}

#[test]
fn parameter_sets_count_toward_the_annex_b_byte_budget() {
    let mut sample = 65_532_u32.to_be_bytes().to_vec();
    sample.extend(vec![0x65; 65_532]);
    let file = repeated_samples(&sample, 1024, 4);
    assert!(matches!(
        demux_video(&file, &Mp4Limits::default()),
        Err(Mp4Error::Limit {
            surface: "Annex B bytes",
            limit: 67_108_864,
            actual: 67_108_876,
        })
    ));
}

#[test]
fn uniform_sample_count_is_refused_before_sample_expansion() {
    let mut file = repeated_samples(&[1, 0x65], 1, 1);
    let stsz = file.windows(4).position(|bytes| bytes == b"stsz").unwrap();
    file[stsz + 12..stsz + 16].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(matches!(
        demux_video(
            &file,
            &Mp4Limits {
                maximum_samples: 65_536
            }
        ),
        Err(Mp4Error::Limit {
            surface: "sample",
            limit: 65_536,
            actual: 4_294_967_295
        })
    ));
}
