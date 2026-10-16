//! ISO BMFF (MP4) demuxing for the first video track.
//!
//! The demuxer resolves the sample tables of one progressive (non-fragmented)
//! file into byte ranges and timestamps, and converts length-prefixed H.264
//! samples to the Annex B stream a hardware decoder consumes. It decodes no
//! pixels. Every count is checked against the bytes that carry it before any
//! allocation, and every sample range is checked against the file. Repeated
//! ranges and Annex B start codes count toward a 64 MiB expanded-byte cap.

use std::error::Error;
use std::fmt;

/// Hard cap for cumulative sample bytes and for the complete Annex B stream,
/// including parameter sets. The container's own byte cap is separate.
pub const MAXIMUM_EXPANDED_BYTES: u64 = 64 << 20;

/// Bounds applied while demuxing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mp4Limits {
    /// The most samples one video track may declare.
    pub maximum_samples: u32,
}

impl Default for Mp4Limits {
    fn default() -> Self {
        Self {
            maximum_samples: 1 << 18,
        }
    }
}

/// A refusal while demuxing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Mp4Error {
    Truncated {
        surface: &'static str,
    },
    Malformed {
        surface: &'static str,
        detail: &'static str,
    },
    Unsupported {
        detail: String,
    },
    Limit {
        surface: &'static str,
        limit: u64,
        actual: u64,
    },
    NoVideoTrack,
}

impl fmt::Display for Mp4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { surface } => write!(formatter, "MP4 {surface} is truncated"),
            Self::Malformed { surface, detail } => {
                write!(formatter, "MP4 {surface} is malformed: {detail}")
            }
            Self::Unsupported { detail } => write!(formatter, "unsupported MP4 input: {detail}"),
            Self::Limit {
                surface,
                limit,
                actual,
            } => write!(
                formatter,
                "MP4 {surface} count {actual} exceeds the limit {limit}"
            ),
            Self::NoVideoTrack => formatter.write_str("MP4 input has no video track"),
        }
    }
}

impl Error for Mp4Error {}

/// The decoder configuration of a video track.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VideoCodec {
    Avc(AvcConfiguration),
}

/// An `avcC` decoder configuration record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AvcConfiguration {
    pub profile: u8,
    pub compatibility: u8,
    pub level: u8,
    /// The byte width of each NAL unit length prefix: 1, 2 or 4.
    pub length_size: u8,
    pub sequence_parameter_sets: Vec<Vec<u8>>,
    pub picture_parameter_sets: Vec<Vec<u8>>,
}

const START_CODE: [u8; 4] = [0, 0, 0, 1];

impl AvcConfiguration {
    /// Appends every SPS and PPS as Annex B NAL units.
    pub fn write_parameter_sets(&self, out: &mut Vec<u8>) {
        for unit in self
            .sequence_parameter_sets
            .iter()
            .chain(&self.picture_parameter_sets)
        {
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(unit);
        }
    }

    /// Appends one length-prefixed sample as Annex B NAL units.
    pub fn write_annex_b(&self, sample: &[u8], out: &mut Vec<u8>) -> Result<(), Mp4Error> {
        self.for_each_nal(sample, |unit| {
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(unit);
            Ok(())
        })
    }

    fn for_each_nal(
        &self,
        sample: &[u8],
        mut visit: impl FnMut(&[u8]) -> Result<(), Mp4Error>,
    ) -> Result<(), Mp4Error> {
        let width = usize::from(self.length_size);
        let mut rest = sample;
        while !rest.is_empty() {
            let (prefix, tail) = split(rest, width, "NAL length")?;
            let length = prefix
                .iter()
                .fold(0usize, |value, byte| (value << 8) | usize::from(*byte));
            if length == 0 {
                return Err(malformed("sample", "empty NAL unit"));
            }
            let (unit, tail) = split(tail, length, "NAL unit")?;
            visit(unit)?;
            rest = tail;
        }
        Ok(())
    }
}

/// One coded sample (access unit) in decode order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sample {
    /// The absolute byte offset of the sample in the file.
    pub offset: u64,
    pub size: u32,
    /// The decode timestamp in track timescale units.
    pub decode_time: u64,
    /// The composition offset in track timescale units.
    pub composition_offset: i64,
    /// Whether the sample is a sync (random access) sample.
    pub sync: bool,
}

/// The demuxed first video track of a file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoTrack {
    pub codec: VideoCodec,
    pub width: u16,
    pub height: u16,
    /// Ticks per second for every timestamp of this track.
    pub timescale: u32,
    /// The media duration in timescale units.
    pub duration: u64,
    /// The media time the edit list maps to presentation time zero.
    pub edit_media_time: i64,
    /// The length of the edit list's media segment in timescale units, when
    /// the edit list bounds it.
    pub edit_duration: Option<u64>,
    /// Samples in decode order.
    pub samples: Vec<Sample>,
}

impl VideoTrack {
    /// The presentation timestamp of one sample in timescale units.
    pub fn presentation_time(&self, sample: &Sample) -> i64 {
        sample.decode_time as i64 + sample.composition_offset - self.edit_media_time
    }

    /// Sample indices sorted by presentation time, which is display order.
    pub fn presentation_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.samples.len()).collect();
        order.sort_by_key(|index| (self.presentation_time(&self.samples[*index]), *index));
        order
    }

    /// Display-order sample indices inside the edit list's media segment.
    ///
    /// A sample is presentable when its presentation time lies in
    /// `[0, edit_duration)`. The others are still decoded, as references, but
    /// never shown. This is the window FFmpeg's MP4 reader applies.
    pub fn presentable_order(&self) -> Vec<usize> {
        self.presentation_order()
            .into_iter()
            .filter(|&index| {
                let time = self.presentation_time(&self.samples[index]);
                time >= 0 && self.edit_duration.is_none_or(|end| (time as u64) < end)
            })
            .collect()
    }

    /// The bytes of one sample within the file it was demuxed from.
    pub fn sample_bytes<'a>(&self, file: &'a [u8], index: usize) -> Result<&'a [u8], Mp4Error> {
        let sample = self
            .samples
            .get(index)
            .ok_or(malformed("sample", "index is out of range"))?;
        let start = usize::try_from(sample.offset).map_err(|_| truncated("sample"))?;
        start
            .checked_add(sample.size as usize)
            .and_then(|end| file.get(start..end))
            .ok_or(truncated("sample"))
    }
}

/// Demuxes the first video track of a progressive MP4 file.
pub fn demux_video(file: &[u8], limits: &Mp4Limits) -> Result<VideoTrack, Mp4Error> {
    let top = boxes(file, "file")?;
    if top.iter().any(|item| &item.kind == b"moof") {
        return Err(unsupported("fragmented MP4"));
    }
    let moov = only_child(&top, b"moov", "file")?;
    let moov = boxes(moov.body, "moov")?;
    if moov.iter().any(|item| &item.kind == b"mvex") {
        return Err(unsupported("fragmented MP4"));
    }
    let movie_timescale = movie_header(only_child(&moov, b"mvhd", "moov")?.body)?;
    for trak in moov.iter().filter(|item| &item.kind == b"trak") {
        if let Some(track) = video_track(trak.body, movie_timescale, file.len() as u64, limits)? {
            check_annex_b_bytes(&track, file)?;
            return Ok(track);
        }
    }
    Err(Mp4Error::NoVideoTrack)
}

fn charge_bytes(total: &mut u64, bytes: u64, surface: &'static str) -> Result<(), Mp4Error> {
    let actual = total.checked_add(bytes).ok_or(Mp4Error::Limit {
        surface,
        limit: MAXIMUM_EXPANDED_BYTES,
        actual: u64::MAX,
    })?;
    if actual > MAXIMUM_EXPANDED_BYTES {
        return Err(Mp4Error::Limit {
            surface,
            limit: MAXIMUM_EXPANDED_BYTES,
            actual,
        });
    }
    *total = actual;
    Ok(())
}

/// Count exactly what the writer will emit without allocating access units.
fn check_annex_b_bytes(track: &VideoTrack, file: &[u8]) -> Result<(), Mp4Error> {
    let VideoCodec::Avc(config) = &track.codec;
    let mut total = 0;
    let mut charge = |unit: &[u8]| {
        charge_bytes(&mut total, START_CODE.len() as u64, "Annex B bytes")?;
        charge_bytes(&mut total, unit.len() as u64, "Annex B bytes")
    };
    for unit in config
        .sequence_parameter_sets
        .iter()
        .chain(&config.picture_parameter_sets)
    {
        charge(unit)?;
    }
    for index in 0..track.samples.len() {
        config.for_each_nal(track.sample_bytes(file, index)?, &mut charge)?;
    }
    Ok(())
}

struct Mp4Box<'a> {
    kind: [u8; 4],
    body: &'a [u8],
}

fn boxes<'a>(mut bytes: &'a [u8], surface: &'static str) -> Result<Vec<Mp4Box<'a>>, Mp4Error> {
    let mut items = Vec::new();
    while !bytes.is_empty() {
        let (header, _) = split(bytes, 8, surface)?;
        let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        let kind = [header[4], header[5], header[6], header[7]];
        let (header_size, size) = match size {
            0 => (8, bytes.len() as u64),
            1 => {
                let (large, _) = split(&bytes[8..], 8, surface)?;
                (16, u64::from_be_bytes(large.try_into().unwrap_or([0; 8])))
            }
            size => (8, u64::from(size)),
        };
        let size = usize::try_from(size).map_err(|_| truncated(surface))?;
        if size < header_size {
            return Err(malformed(surface, "box size is smaller than its header"));
        }
        let (whole, rest) = split(bytes, size, surface)?;
        items.push(Mp4Box {
            kind,
            body: &whole[header_size..],
        });
        bytes = rest;
    }
    Ok(items)
}

fn only_child<'a, 'b>(
    items: &'b [Mp4Box<'a>],
    kind: &[u8; 4],
    surface: &'static str,
) -> Result<&'b Mp4Box<'a>, Mp4Error> {
    let mut matching = items.iter().filter(|item| &item.kind == kind);
    let first = matching
        .next()
        .ok_or(malformed(surface, "a required box is missing"))?;
    if matching.next().is_some() {
        return Err(malformed(surface, "a box appears twice"));
    }
    Ok(first)
}

fn optional_child<'a, 'b>(
    items: &'b [Mp4Box<'a>],
    kind: &[u8; 4],
    surface: &'static str,
) -> Result<Option<&'b Mp4Box<'a>>, Mp4Error> {
    match items.iter().filter(|item| &item.kind == kind).count() {
        0 => Ok(None),
        1 => only_child(items, kind, surface).map(Some),
        _ => Err(malformed(surface, "a box appears twice")),
    }
}

fn video_track(
    trak: &[u8],
    movie_timescale: u32,
    file_size: u64,
    limits: &Mp4Limits,
) -> Result<Option<VideoTrack>, Mp4Error> {
    let trak = boxes(trak, "trak")?;
    let mdia = boxes(only_child(&trak, b"mdia", "trak")?.body, "mdia")?;
    let mut hdlr = Reader::full(only_child(&mdia, b"hdlr", "mdia")?.body, "hdlr")?.1;
    hdlr.skip(4)?;
    if hdlr.bytes(4)? != b"vide" {
        return Ok(None);
    }
    let (timescale, duration) = media_header(only_child(&mdia, b"mdhd", "mdia")?.body)?;
    let (edit_media_time, edit_duration) = match optional_child(&trak, b"edts", "trak")? {
        Some(edts) => {
            let edts = boxes(edts.body, "edts")?;
            match optional_child(&edts, b"elst", "edts")? {
                Some(elst) => edit_list(elst.body, movie_timescale, timescale)?,
                None => (0, None),
            }
        }
        None => (0, None),
    };
    let minf = boxes(only_child(&mdia, b"minf", "mdia")?.body, "minf")?;
    let stbl = boxes(only_child(&minf, b"stbl", "minf")?.body, "stbl")?;
    let (codec, width, height) = sample_description(only_child(&stbl, b"stsd", "stbl")?.body)?;
    let samples = sample_table(&stbl, file_size, limits)?;
    Ok(Some(VideoTrack {
        codec,
        width,
        height,
        timescale,
        duration,
        edit_media_time,
        edit_duration,
        samples,
    }))
}

fn movie_header(body: &[u8]) -> Result<u32, Mp4Error> {
    let (version, mut reader) = Reader::full(body, "mvhd")?;
    reader.skip(if version == 1 { 16 } else { 8 })?;
    let timescale = reader.u32()?;
    if timescale == 0 {
        return Err(malformed("mvhd", "timescale is zero"));
    }
    Ok(timescale)
}

fn media_header(body: &[u8]) -> Result<(u32, u64), Mp4Error> {
    let (version, mut reader) = Reader::full(body, "mdhd")?;
    let (timescale, duration) = if version == 1 {
        reader.skip(16)?;
        (reader.u32()?, reader.u64()?)
    } else {
        reader.skip(8)?;
        (reader.u32()?, u64::from(reader.u32()?))
    };
    if timescale == 0 {
        return Err(malformed("mdhd", "timescale is zero"));
    }
    Ok((timescale, duration))
}

/// Returns the media time and track-timescale length of the one media
/// segment. Leading empty edits only delay presentation and are ignored; more
/// than one media segment or a rate other than one is refused. A zero segment
/// length leaves the segment unbounded.
fn edit_list(
    body: &[u8],
    movie_timescale: u32,
    track_timescale: u32,
) -> Result<(i64, Option<u64>), Mp4Error> {
    let (version, mut reader) = Reader::full(body, "elst")?;
    let count = reader.u32()?;
    let entry = if version == 1 { 20 } else { 12 };
    reader.expect_entries(count, entry)?;
    let mut segment = None;
    for _ in 0..count {
        let (duration, time) = if version == 1 {
            (reader.u64()?, reader.u64()? as i64)
        } else {
            (u64::from(reader.u32()?), i64::from(reader.u32()? as i32))
        };
        let rate = reader.u32()?;
        if time == -1 {
            continue;
        }
        if rate != 0x0001_0000 {
            return Err(unsupported("edit list rate other than one"));
        }
        if time < 0 {
            return Err(malformed("elst", "media time is negative"));
        }
        // Movie to track units, rounded to nearest as FFmpeg's av_rescale does.
        let length = (u128::from(duration) * u128::from(track_timescale)
            + u128::from(movie_timescale / 2))
            / u128::from(movie_timescale);
        let length = u64::try_from(length).map_err(|_| malformed("elst", "length overflows"))?;
        if segment
            .replace((time, (length > 0).then_some(length)))
            .is_some()
        {
            return Err(unsupported("edit list with more than one media segment"));
        }
    }
    Ok(segment.unwrap_or((0, None)))
}

fn sample_description(body: &[u8]) -> Result<(VideoCodec, u16, u16), Mp4Error> {
    let (_, mut reader) = Reader::full(body, "stsd")?;
    if reader.u32()? != 1 {
        return Err(unsupported("more than one sample description"));
    }
    let entries = boxes(reader.rest(), "stsd")?;
    let entry = entries
        .first()
        .ok_or(malformed("stsd", "sample description is missing"))?;
    if &entry.kind != b"avc1" {
        return Err(unsupported(format!(
            "video codec {}",
            String::from_utf8_lossy(&entry.kind)
        )));
    }
    let mut visual = Reader::new(entry.body, "avc1");
    visual.skip(24)?;
    let width = visual.u16()?;
    let height = visual.u16()?;
    visual.skip(50)?;
    let children = boxes(visual.rest(), "avc1")?;
    let avcc = only_child(&children, b"avcC", "avc1")?;
    Ok((
        VideoCodec::Avc(avc_configuration(avcc.body)?),
        width,
        height,
    ))
}

fn avc_configuration(body: &[u8]) -> Result<AvcConfiguration, Mp4Error> {
    let mut reader = Reader::new(body, "avcC");
    if reader.u8()? != 1 {
        return Err(unsupported("avcC configuration version"));
    }
    let profile = reader.u8()?;
    let compatibility = reader.u8()?;
    let level = reader.u8()?;
    let length_size = (reader.u8()? & 0x03) + 1;
    if length_size == 3 {
        return Err(malformed("avcC", "NAL length size is 3"));
    }
    let sps_count = reader.u8()? & 0x1f;
    let sequence_parameter_sets = parameter_sets(&mut reader, sps_count)?;
    let pps_count = reader.u8()?;
    let picture_parameter_sets = parameter_sets(&mut reader, pps_count)?;
    if sequence_parameter_sets.is_empty() || picture_parameter_sets.is_empty() {
        return Err(malformed("avcC", "parameter sets are missing"));
    }
    Ok(AvcConfiguration {
        profile,
        compatibility,
        level,
        length_size,
        sequence_parameter_sets,
        picture_parameter_sets,
    })
}

fn parameter_sets(reader: &mut Reader<'_>, count: u8) -> Result<Vec<Vec<u8>>, Mp4Error> {
    (0..count)
        .map(|_| {
            let length = usize::from(reader.u16()?);
            if length == 0 {
                return Err(malformed("avcC", "empty parameter set"));
            }
            Ok(reader.bytes(length)?.to_vec())
        })
        .collect()
}

fn sample_table(
    stbl: &[Mp4Box<'_>],
    file_size: u64,
    limits: &Mp4Limits,
) -> Result<Vec<Sample>, Mp4Error> {
    let sizes = sample_sizes(stbl, limits)?;
    let mut total = 0;
    for &size in &sizes {
        charge_bytes(&mut total, u64::from(size), "sample bytes")?;
    }
    let count = sizes.len();
    let decode_times = decode_times(only_child(stbl, b"stts", "stbl")?.body, count)?;
    let composition = match optional_child(stbl, b"ctts", "stbl")? {
        Some(ctts) => composition_offsets(ctts.body, count)?,
        None => vec![0; count],
    };
    let sync = match optional_child(stbl, b"stss", "stbl")? {
        Some(stss) => sync_samples(stss.body, count)?,
        None => vec![true; count],
    };
    let offsets = sample_offsets(stbl, &sizes, file_size)?;
    Ok((0..count)
        .map(|index| Sample {
            offset: offsets[index],
            size: sizes[index],
            decode_time: decode_times[index],
            composition_offset: composition[index],
            sync: sync[index],
        })
        .collect())
}

fn sample_sizes(stbl: &[Mp4Box<'_>], limits: &Mp4Limits) -> Result<Vec<u32>, Mp4Error> {
    let check = |count: u32| {
        if count > limits.maximum_samples {
            return Err(Mp4Error::Limit {
                surface: "sample",
                limit: u64::from(limits.maximum_samples),
                actual: u64::from(count),
            });
        }
        Ok(count as usize)
    };
    match (
        optional_child(stbl, b"stsz", "stbl")?,
        optional_child(stbl, b"stz2", "stbl")?,
    ) {
        (Some(stsz), None) => {
            let (_, mut reader) = Reader::full(stsz.body, "stsz")?;
            let uniform = reader.u32()?;
            let count = check(reader.u32()?)?;
            if uniform != 0 {
                return Ok(vec![uniform; count]);
            }
            reader.expect_entries(count as u32, 4)?;
            (0..count).map(|_| reader.u32()).collect()
        }
        (None, Some(stz2)) => {
            let (_, mut reader) = Reader::full(stz2.body, "stz2")?;
            reader.skip(3)?;
            let field = reader.u8()?;
            let count = check(reader.u32()?)?;
            match field {
                4 => {
                    let packed = reader.bytes(count.div_ceil(2))?;
                    Ok((0..count)
                        .map(|index| {
                            let byte = packed[index / 2];
                            u32::from(if index % 2 == 0 {
                                byte >> 4
                            } else {
                                byte & 0x0f
                            })
                        })
                        .collect())
                }
                8 => Ok(reader.bytes(count)?.iter().map(|&b| u32::from(b)).collect()),
                16 => {
                    reader.expect_entries(count as u32, 2)?;
                    (0..count).map(|_| reader.u16().map(u32::from)).collect()
                }
                _ => Err(malformed("stz2", "field size is not 4, 8 or 16")),
            }
        }
        (None, None) => Err(malformed("stbl", "sample sizes are missing")),
        (Some(_), Some(_)) => Err(malformed("stbl", "both stsz and stz2 are present")),
    }
}

fn decode_times(body: &[u8], count: usize) -> Result<Vec<u64>, Mp4Error> {
    let (_, mut reader) = Reader::full(body, "stts")?;
    let entries = reader.u32()?;
    reader.expect_entries(entries, 8)?;
    let mut times = Vec::with_capacity(count);
    let mut time = 0u64;
    for _ in 0..entries {
        let run = reader.u32()? as usize;
        let delta = u64::from(reader.u32()?);
        if run > count - times.len() {
            return Err(malformed("stts", "runs exceed the sample count"));
        }
        for _ in 0..run {
            times.push(time);
            time = time
                .checked_add(delta)
                .ok_or(malformed("stts", "time overflows"))?;
        }
    }
    if times.len() != count {
        return Err(malformed("stts", "runs do not cover every sample"));
    }
    Ok(times)
}

fn composition_offsets(body: &[u8], count: usize) -> Result<Vec<i64>, Mp4Error> {
    let (version, mut reader) = Reader::full(body, "ctts")?;
    let entries = reader.u32()?;
    reader.expect_entries(entries, 8)?;
    let mut offsets = Vec::with_capacity(count);
    for _ in 0..entries {
        let run = reader.u32()? as usize;
        let raw = reader.u32()?;
        let offset = if version == 1 {
            i64::from(raw as i32)
        } else {
            i64::from(raw)
        };
        if run > count - offsets.len() {
            return Err(malformed("ctts", "runs exceed the sample count"));
        }
        offsets.resize(offsets.len() + run, offset);
    }
    if offsets.len() != count {
        return Err(malformed("ctts", "runs do not cover every sample"));
    }
    Ok(offsets)
}

fn sync_samples(body: &[u8], count: usize) -> Result<Vec<bool>, Mp4Error> {
    let (_, mut reader) = Reader::full(body, "stss")?;
    let entries = reader.u32()?;
    reader.expect_entries(entries, 4)?;
    let mut sync = vec![false; count];
    let mut previous = 0;
    for _ in 0..entries {
        let number = reader.u32()? as usize;
        if number <= previous || number > count {
            return Err(malformed("stss", "sync sample numbers are out of order"));
        }
        sync[number - 1] = true;
        previous = number;
    }
    Ok(sync)
}

fn sample_offsets(
    stbl: &[Mp4Box<'_>],
    sizes: &[u32],
    file_size: u64,
) -> Result<Vec<u64>, Mp4Error> {
    let chunks = chunk_offsets(stbl)?;
    let (_, mut reader) = Reader::full(only_child(stbl, b"stsc", "stbl")?.body, "stsc")?;
    let entries = reader.u32()?;
    reader.expect_entries(entries, 12)?;
    let mut runs = Vec::with_capacity(entries as usize);
    for _ in 0..entries {
        let first_chunk = reader.u32()? as usize;
        let per_chunk = reader.u32()? as usize;
        if reader.u32()? != 1 {
            return Err(unsupported("a sample description index other than 1"));
        }
        let previous = runs.last().map_or(0, |&(first, _)| first);
        if first_chunk <= previous || first_chunk > chunks.len() {
            return Err(malformed("stsc", "chunk runs are out of order"));
        }
        runs.push((first_chunk, per_chunk));
    }
    if runs.first().map(|&(first, _)| first) != Some(1) && !sizes.is_empty() {
        return Err(malformed("stsc", "chunk runs do not start at chunk 1"));
    }
    let mut offsets = Vec::with_capacity(sizes.len());
    for (run, &(first_chunk, per_chunk)) in runs.iter().enumerate() {
        let end_chunk = runs
            .get(run + 1)
            .map_or(chunks.len() + 1, |&(next, _)| next);
        for &chunk in &chunks[first_chunk - 1..end_chunk - 1] {
            let mut offset = chunk;
            for _ in 0..per_chunk {
                let size = *sizes
                    .get(offsets.len())
                    .ok_or(malformed("stsc", "chunks hold more samples than stsz"))?;
                let end = offset
                    .checked_add(u64::from(size))
                    .filter(|&end| end <= file_size)
                    .ok_or(truncated("sample"))?;
                offsets.push(offset);
                offset = end;
            }
        }
    }
    if offsets.len() != sizes.len() {
        return Err(malformed("stsc", "chunks hold fewer samples than stsz"));
    }
    Ok(offsets)
}

fn chunk_offsets(stbl: &[Mp4Box<'_>]) -> Result<Vec<u64>, Mp4Error> {
    match (
        optional_child(stbl, b"stco", "stbl")?,
        optional_child(stbl, b"co64", "stbl")?,
    ) {
        (Some(stco), None) => {
            let (_, mut reader) = Reader::full(stco.body, "stco")?;
            let count = reader.u32()?;
            reader.expect_entries(count, 4)?;
            (0..count).map(|_| reader.u32().map(u64::from)).collect()
        }
        (None, Some(co64)) => {
            let (_, mut reader) = Reader::full(co64.body, "co64")?;
            let count = reader.u32()?;
            reader.expect_entries(count, 8)?;
            (0..count).map(|_| reader.u64()).collect()
        }
        (None, None) => Err(malformed("stbl", "chunk offsets are missing")),
        (Some(_), Some(_)) => Err(malformed("stbl", "both stco and co64 are present")),
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    surface: &'static str,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], surface: &'static str) -> Self {
        Self { bytes, surface }
    }

    /// Reads a full box header and returns its version.
    fn full(bytes: &'a [u8], surface: &'static str) -> Result<(u8, Self), Mp4Error> {
        let mut reader = Self::new(bytes, surface);
        let version = reader.u8()?;
        reader.skip(3)?;
        if version > 1 {
            return Err(unsupported(format!("{surface} version {version}")));
        }
        Ok((version, reader))
    }

    fn bytes(&mut self, count: usize) -> Result<&'a [u8], Mp4Error> {
        let (taken, rest) = split(self.bytes, count, self.surface)?;
        self.bytes = rest;
        Ok(taken)
    }

    fn skip(&mut self, count: usize) -> Result<(), Mp4Error> {
        self.bytes(count).map(|_| ())
    }

    fn rest(&mut self) -> &'a [u8] {
        std::mem::take(&mut self.bytes)
    }

    /// Refuses a table whose declared entries exceed the bytes that remain,
    /// before anything is allocated for it.
    fn expect_entries(&self, count: u32, entry_bytes: usize) -> Result<(), Mp4Error> {
        if (count as usize).saturating_mul(entry_bytes) > self.bytes.len() {
            return Err(truncated(self.surface));
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, Mp4Error> {
        Ok(self.bytes(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, Mp4Error> {
        let bytes = self.bytes(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, Mp4Error> {
        let bytes = self.bytes(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, Mp4Error> {
        Ok((u64::from(self.u32()?) << 32) | u64::from(self.u32()?))
    }
}

fn split<'a>(
    bytes: &'a [u8],
    count: usize,
    surface: &'static str,
) -> Result<(&'a [u8], &'a [u8]), Mp4Error> {
    if bytes.len() < count {
        return Err(truncated(surface));
    }
    Ok(bytes.split_at(count))
}

fn truncated(surface: &'static str) -> Mp4Error {
    Mp4Error::Truncated { surface }
}

fn malformed(surface: &'static str, detail: &'static str) -> Mp4Error {
    Mp4Error::Malformed { surface, detail }
}

fn unsupported(detail: impl Into<String>) -> Mp4Error {
    Mp4Error::Unsupported {
        detail: detail.into(),
    }
}
