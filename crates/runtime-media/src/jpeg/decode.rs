use crate::{MediaError, MediaLimits, MediaResult};

use super::entropy::{Bits, Huffman, Tables};
use super::{Header, Markers, be16, frame, tiff_orientation};

pub(super) const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
const MAX_SCANS: usize = 64;

pub(super) struct Component {
    pub id: u8,
    pub h: usize,
    pub v: usize,
    pub quant: usize,
    pub block_width: usize,
    pub block_height: usize,
    pub logical_width: usize,
    pub logical_height: usize,
    pub coeffs: Vec<[i16; 64]>,
    pub successive: [Option<u8>; 64],
}

pub(super) struct Frame {
    pub header: Header,
    pub max_h: usize,
    pub max_v: usize,
    pub mcu_cols: usize,
    pub mcu_rows: usize,
    pub components: Vec<Component>,
    pub quant: [Option<[u16; 64]>; 4],
    pub adobe_transform: Option<u8>,
}

fn div_ceil(x: usize, y: usize) -> MediaResult<usize> {
    x.checked_add(y - 1)
        .map(|n| n / y)
        .ok_or_else(|| MediaError::new("JPEG geometry overflow"))
}

impl Frame {
    fn new(header: Header, payload: &[u8], limits: MediaLimits) -> MediaResult<Self> {
        limits.check_named_image("JPEG", u32::from(header.width), u32::from(header.height))?;
        header.coefficient_bytes()?;
        let (max_h, max_v) = header.maximum_sampling();
        let mcu_cols = div_ceil(usize::from(header.width), 8 * max_h)?;
        let mcu_rows = div_ceil(usize::from(header.height), 8 * max_v)?;
        let mut components = Vec::with_capacity(header.components as usize);
        for part in payload[6..].chunks_exact(3) {
            let h = usize::from(part[1] >> 4);
            let v = usize::from(part[1] & 15);
            if max_h % h != 0 || max_v % v != 0 {
                return Err(MediaError::new("unsupported fractional JPEG sampling"));
            }
            let (block_width, block_height) = header.component_blocks(components.len());
            let logical_width = div_ceil(usize::from(header.width) * h, 8 * max_h)?;
            let logical_height = div_ceil(usize::from(header.height) * v, 8 * max_v)?;
            let count = block_width
                .checked_mul(block_height)
                .ok_or_else(|| MediaError::new("JPEG coefficient size overflow"))?;
            let mut coeffs = Vec::new();
            coeffs
                .try_reserve_exact(count)
                .map_err(|_| MediaError::new("JPEG coefficient allocation failed"))?;
            coeffs.resize(count, [0; 64]);
            components.push(Component {
                id: part[0],
                h,
                v,
                quant: usize::from(part[2]),
                block_width,
                block_height,
                logical_width,
                logical_height,
                coeffs,
                successive: [None; 64],
            });
        }
        Ok(Self {
            header,
            max_h,
            max_v,
            mcu_cols,
            mcu_rows,
            components,
            quant: [None; 4],
            adobe_transform: None,
        })
    }
}

fn parse_quant(tables: &mut [Option<[u16; 64]>; 4], mut payload: &[u8]) -> MediaResult<()> {
    while !payload.is_empty() {
        let selector = payload[0];
        let index = usize::from(selector & 15);
        let unit = match selector >> 4 {
            0 => 1,
            1 => 2,
            _ => return Err(MediaError::new("unsupported JPEG quant precision")),
        };
        let table_bytes = 64 * unit;
        let table_data = payload
            .get(1..1 + table_bytes)
            .ok_or_else(|| MediaError::new("truncated JPEG quant table"))?;
        if index > 3 {
            return Err(MediaError::new("invalid JPEG quant selector"));
        }
        let mut table = [0_u16; 64];
        for (i, natural) in ZIGZAG.iter().copied().enumerate() {
            table[natural] = if unit == 1 {
                u16::from(table_data[i])
            } else {
                be16(&table_data[i * 2..i * 2 + 2])
            };
            if table[natural] == 0 {
                return Err(MediaError::new("zero JPEG quantizer"));
            }
        }
        tables[index] = Some(table);
        payload = &payload[1 + table_bytes..];
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ScanComponent {
    index: usize,
    dc: usize,
    ac: usize,
}

struct Scan {
    components: Vec<ScanComponent>,
    first: usize,
    last: usize,
    high: u8,
    low: u8,
}

impl Scan {
    fn parse(frame: &Frame, payload: &[u8]) -> MediaResult<Self> {
        let count = payload.first().copied().map(usize::from).unwrap_or(0);
        if count == 0 || count > frame.components.len() || payload.len() != 1 + count * 2 + 3 {
            return Err(MediaError::new("invalid JPEG scan header"));
        }
        let mut components = Vec::with_capacity(count);
        for part in payload[1..1 + count * 2].chunks_exact(2) {
            let index = frame
                .components
                .iter()
                .position(|component| component.id == part[0])
                .ok_or_else(|| MediaError::new("unknown JPEG scan component"))?;
            if components
                .iter()
                .any(|existing: &ScanComponent| existing.index == index)
            {
                return Err(MediaError::new("duplicate JPEG scan component"));
            }
            let dc = usize::from(part[1] >> 4);
            let ac = usize::from(part[1] & 15);
            if dc > 3 || ac > 3 {
                return Err(MediaError::new("invalid JPEG scan table selector"));
            }
            components.push(ScanComponent { index, dc, ac });
        }
        let first = usize::from(payload[1 + count * 2]);
        let last = usize::from(payload[2 + count * 2]);
        let approximation = payload[3 + count * 2];
        let high = approximation >> 4;
        let low = approximation & 15;
        if frame.header.progressive {
            if first > last
                || last > 63
                || low > 13
                || (first == 0 && last != 0)
                || (first != 0 && count != 1)
                || (high != 0 && low + 1 != high)
            {
                return Err(MediaError::new("invalid progressive JPEG scan"));
            }
        } else if first != 0 || last != 63 || high != 0 || low != 0 {
            return Err(MediaError::new("invalid baseline JPEG scan"));
        }
        for choice in &components {
            let component = &frame.components[choice.index];
            for coefficient in first..=last {
                let previous = component.successive[coefficient];
                if (high == 0 && previous.is_some()) || (high != 0 && previous != Some(high)) {
                    return Err(MediaError::new("invalid JPEG scan progression"));
                }
            }
        }
        Ok(Self {
            components,
            first,
            last,
            high,
            low,
        })
    }

    fn record(&self, frame: &mut Frame) {
        for choice in &self.components {
            for coefficient in self.first..=self.last {
                frame.components[choice.index].successive[coefficient] = Some(self.low);
            }
        }
    }
}

fn huffman(table: &Option<Huffman>) -> MediaResult<&Huffman> {
    table
        .as_ref()
        .ok_or_else(|| MediaError::new("JPEG Huffman table is missing"))
}

/// Every stored coefficient fits 16 bits, as libjpeg's JCOEF does; a
/// stream that needs more is refused.
fn coefficient(value: i32) -> MediaResult<i16> {
    i16::try_from(value).map_err(|_| MediaError::new("JPEG coefficient exceeds 16-bit range"))
}

fn dc_first(
    block: &mut [i16; 64],
    predictor: &mut i32,
    table: &Huffman,
    bits: &mut Bits<'_>,
    shift: u8,
) -> MediaResult<()> {
    let size = table.decode(bits)?;
    if size > 11 {
        return Err(MediaError::new("invalid JPEG DC category"));
    }
    *predictor = i32::from(coefficient(
        predictor
            .checked_add(bits.signed(size)?)
            .ok_or_else(|| MediaError::new("JPEG DC predictor overflow"))?,
    )?);
    block[0] = coefficient(*predictor << shift)?;
    Ok(())
}

fn ac_baseline(block: &mut [i16; 64], table: &Huffman, bits: &mut Bits<'_>) -> MediaResult<()> {
    let mut k = 1;
    while k < 64 {
        let symbol = table.decode(bits)?;
        let run = usize::from(symbol >> 4);
        let size = symbol & 15;
        if size == 0 {
            if run == 15 {
                k += 16;
                continue;
            }
            break;
        }
        if size > 10 || k + run >= 64 {
            return Err(MediaError::new("invalid baseline JPEG AC code"));
        }
        k += run;
        // Ten magnitude bits always fit.
        block[ZIGZAG[k]] = bits.signed(size)? as i16;
        k += 1;
    }
    Ok(())
}

fn ac_first(
    block: &mut [i16; 64],
    table: &Huffman,
    bits: &mut Bits<'_>,
    scan: &Scan,
    eob: &mut u32,
) -> MediaResult<()> {
    if *eob != 0 {
        *eob -= 1;
        return Ok(());
    }
    let mut k = scan.first;
    while k <= scan.last {
        let symbol = table.decode(bits)?;
        let run = usize::from(symbol >> 4);
        let size = symbol & 15;
        if size == 0 {
            if run == 15 {
                k += 16;
                continue;
            }
            *eob = (1 << run) + bits.bits(run as u8)? - 1;
            break;
        }
        if size > 10 || k + run > scan.last {
            return Err(MediaError::new("invalid progressive JPEG AC code"));
        }
        k += run;
        block[ZIGZAG[k]] = coefficient(bits.signed(size)? << scan.low)?;
        k += 1;
    }
    Ok(())
}

fn correction(block: &mut [i16; 64], k: usize, bit: i32, bits: &mut Bits<'_>) -> MediaResult<()> {
    let value = i32::from(block[ZIGZAG[k]]);
    if value != 0 && bits.bit()? != 0 && (value & bit) == 0 {
        block[ZIGZAG[k]] = coefficient(value + if value > 0 { bit } else { -bit })?;
    }
    Ok(())
}

fn ac_refine(
    block: &mut [i16; 64],
    table: &Huffman,
    bits: &mut Bits<'_>,
    scan: &Scan,
    eob: &mut u32,
) -> MediaResult<()> {
    let bit = 1_i32 << scan.low;
    let mut k = scan.first;
    if *eob == 0 {
        while k <= scan.last {
            let symbol = table.decode(bits)?;
            let mut run = i32::from(symbol >> 4);
            let size = symbol & 15;
            let new_value = if size == 1 {
                Some(if bits.bit()? != 0 { bit } else { -bit })
            } else if size == 0 && run != 15 {
                *eob = (1 << run) + bits.bits(run as u8)?;
                break;
            } else if size == 0 {
                None
            } else {
                return Err(MediaError::new("invalid progressive JPEG refinement code"));
            };
            loop {
                if k > scan.last {
                    return Err(MediaError::new("JPEG refinement run exceeds band"));
                }
                if block[ZIGZAG[k]] != 0 {
                    correction(block, k, bit, bits)?;
                } else {
                    run -= 1;
                    if run < 0 {
                        break;
                    }
                }
                k += 1;
            }
            if let Some(value) = new_value {
                // A refinement bit is at most 1 << 13.
                block[ZIGZAG[k]] = value as i16;
            }
            k += 1;
        }
    }
    if *eob > 0 {
        for index in k..=scan.last {
            correction(block, index, bit, bits)?;
        }
        *eob -= 1;
    }
    Ok(())
}

struct DecodeState {
    predictors: [i32; 4],
    eob: u32,
    progressive: bool,
}

fn decode_block(
    block: &mut [i16; 64],
    state: &mut DecodeState,
    choice: ScanComponent,
    scan: &Scan,
    tables: &Tables,
    bits: &mut Bits<'_>,
) -> MediaResult<()> {
    if !state.progressive {
        dc_first(
            block,
            &mut state.predictors[choice.index],
            huffman(&tables.dc[choice.dc])?,
            bits,
            0,
        )?;
        ac_baseline(block, huffman(&tables.ac[choice.ac])?, bits)
    } else if scan.first == 0 {
        if scan.high == 0 {
            dc_first(
                block,
                &mut state.predictors[choice.index],
                huffman(&tables.dc[choice.dc])?,
                bits,
                scan.low,
            )
        } else {
            if bits.bit()? != 0 {
                block[0] |= 1 << scan.low;
            }
            Ok(())
        }
    } else if scan.high == 0 {
        ac_first(
            block,
            huffman(&tables.ac[choice.ac])?,
            bits,
            scan,
            &mut state.eob,
        )
    } else {
        ac_refine(
            block,
            huffman(&tables.ac[choice.ac])?,
            bits,
            scan,
            &mut state.eob,
        )
    }
}

fn decode_scan(
    frame: &mut Frame,
    tables: &Tables,
    scan: &Scan,
    bits: &mut Bits<'_>,
    restart: u16,
) -> MediaResult<()> {
    let single = scan.components.len() == 1;
    let (cols, rows) = if single {
        let component = &frame.components[scan.components[0].index];
        (component.logical_width, component.logical_height)
    } else {
        (frame.mcu_cols, frame.mcu_rows)
    };
    let total = cols * rows;
    let mut state = DecodeState {
        predictors: [0; 4],
        eob: 0,
        progressive: frame.header.progressive,
    };
    let mut restart_number = 0_u8;
    for mcu in 0..total {
        if restart != 0 && mcu != 0 && mcu % usize::from(restart) == 0 {
            bits.restart(restart_number)?;
            restart_number = (restart_number + 1) & 7;
            state.predictors = [0; 4];
            state.eob = 0;
        }
        let mx = mcu % cols;
        let my = mcu / cols;
        for &choice in &scan.components {
            let component = &mut frame.components[choice.index];
            let (h, v) = if single {
                (1, 1)
            } else {
                (component.h, component.v)
            };
            for dy in 0..v {
                for dx in 0..h {
                    let bx = mx * h + dx;
                    let by = my * v + dy;
                    let index = by * component.block_width + bx;
                    let block = &mut component.coeffs[index];
                    decode_block(block, &mut state, choice, scan, tables, bits)?;
                }
            }
        }
    }
    if state.eob != 0 {
        return Err(MediaError::new("JPEG EOB run exceeds scan"));
    }
    bits.finish();
    scan.record(frame);
    Ok(())
}

pub(super) fn coefficients(bytes: &[u8], limits: MediaLimits) -> MediaResult<Frame> {
    limits.check_bytes(bytes.len())?;
    let mut markers = Markers::new(bytes)?;
    let mut orientation = 1;
    let mut frame_data: Option<Frame> = None;
    let mut huffman = Tables::new();
    let mut pending_quant: [Option<[u16; 64]>; 4] = [None; 4];
    let mut restart = 0_u16;
    let mut scans = 0_usize;
    let mut adobe = None;
    loop {
        let (marker, payload) = markers.next()?;
        match marker {
            0xc0 | 0xc2 => {
                if frame_data.is_some() {
                    return Err(MediaError::new("duplicate JPEG frame"));
                }
                let header = frame(marker, payload, orientation)?;
                let mut new_frame = Frame::new(header, payload, limits)?;
                new_frame.quant = pending_quant;
                new_frame.adobe_transform = adobe;
                frame_data = Some(new_frame);
            }
            0xc4 => huffman.parse(payload)?,
            0xdb => {
                if let Some(current) = &mut frame_data {
                    parse_quant(&mut current.quant, payload)?;
                } else {
                    parse_quant(&mut pending_quant, payload)?;
                }
            }
            0xdd => {
                if payload.len() != 2 {
                    return Err(MediaError::new("invalid JPEG restart interval"));
                }
                restart = be16(payload);
            }
            0xe1 => {
                if let Some(value) = tiff_orientation(payload) {
                    orientation = value;
                    if let Some(current) = &mut frame_data {
                        current.header.orientation = value;
                    }
                }
            }
            0xee if payload.starts_with(b"Adobe") && payload.len() >= 12 => {
                adobe = Some(payload[11]);
                if let Some(current) = &mut frame_data {
                    current.adobe_transform = adobe;
                }
            }
            0xda => {
                let current = frame_data
                    .as_mut()
                    .ok_or_else(|| MediaError::new("JPEG SOF is missing"))?;
                if scans == MAX_SCANS {
                    return Err(MediaError::new("JPEG scan count limit exceeded"));
                }
                let scan = Scan::parse(current, payload)?;
                for choice in &scan.components {
                    if current.quant[current.components[choice.index].quant].is_none() {
                        return Err(MediaError::new("JPEG quantizer is missing"));
                    }
                }
                let mut bits = Bits::new(bytes, markers.pos);
                decode_scan(current, &huffman, &scan, &mut bits, restart)?;
                markers.pos = bits.position();
                scans += 1;
            }
            0xd9 => {
                let current = frame_data.ok_or_else(|| MediaError::new("JPEG SOF is missing"))?;
                if scans == 0
                    || current
                        .components
                        .iter()
                        .any(|component| component.successive[0].is_none())
                {
                    return Err(MediaError::new("JPEG component scan is missing"));
                }
                return Ok(current);
            }
            0xc1 | 0xc3..=0xcf if marker != 0xc4 && marker != 0xc8 && marker != 0xcc => {
                return Err(MediaError::new("unsupported JPEG frame coding"));
            }
            _ => {}
        }
    }
}
