use crate::{MediaError, MediaResult};

#[derive(Clone)]
pub(super) struct Huffman {
    counts: [u8; 17],
    first_code: [u32; 17],
    first_symbol: [usize; 17],
    symbols: Vec<u8>,
}

impl Huffman {
    fn new(counts: &[u8], symbols: &[u8]) -> MediaResult<Self> {
        if counts.len() != 16
            || symbols.len() != counts.iter().map(|&x| usize::from(x)).sum::<usize>()
            || symbols.is_empty()
            || symbols.len() > 256
        {
            return Err(MediaError::new("invalid JPEG Huffman table length"));
        }
        let mut table = Self {
            counts: [0; 17],
            first_code: [0; 17],
            first_symbol: [0; 17],
            symbols: symbols.to_vec(),
        };
        let mut code = 0_u32;
        let mut offset = 0_usize;
        for length in 1..=16 {
            code <<= 1;
            let count = u32::from(counts[length - 1]);
            if code + count > 1 << length {
                return Err(MediaError::new("oversubscribed JPEG Huffman table"));
            }
            table.counts[length] = counts[length - 1];
            table.first_code[length] = code;
            table.first_symbol[length] = offset;
            offset += count as usize;
            code += count;
        }
        Ok(table)
    }

    pub fn decode(&self, reader: &mut Bits<'_>) -> MediaResult<u8> {
        let mut code = 0_u32;
        for length in 1..=16 {
            code = (code << 1) | u32::from(reader.bit()?);
            let first = self.first_code[length];
            if code >= first && code - first < u32::from(self.counts[length]) {
                return Ok(self.symbols[self.first_symbol[length] + (code - first) as usize]);
            }
        }
        Err(MediaError::new("invalid JPEG Huffman code"))
    }
}

pub(super) struct Tables {
    pub dc: [Option<Huffman>; 4],
    pub ac: [Option<Huffman>; 4],
}

impl Tables {
    pub fn new() -> Self {
        Self {
            dc: std::array::from_fn(|_| None),
            ac: std::array::from_fn(|_| None),
        }
    }

    pub fn parse(&mut self, mut payload: &[u8]) -> MediaResult<()> {
        while !payload.is_empty() {
            if payload.len() < 17 {
                return Err(MediaError::new("truncated JPEG Huffman table"));
            }
            let selector = payload[0];
            let count: usize = payload[1..17].iter().map(|&x| usize::from(x)).sum();
            let end = 17 + count;
            let symbols = payload
                .get(17..end)
                .ok_or_else(|| MediaError::new("truncated JPEG Huffman symbols"))?;
            let table = Huffman::new(&payload[1..17], symbols)?;
            match (selector >> 4, selector & 15) {
                (0, index @ 0..=3) => self.dc[index as usize] = Some(table),
                (1, index @ 0..=3) => self.ac[index as usize] = Some(table),
                _ => return Err(MediaError::new("invalid JPEG Huffman selector")),
            }
            payload = &payload[end..];
        }
        Ok(())
    }
}

pub(super) struct Bits<'a> {
    bytes: &'a [u8],
    pos: usize,
    current: u8,
    remaining: u8,
}

impl<'a> Bits<'a> {
    pub fn new(bytes: &'a [u8], pos: usize) -> Self {
        Self {
            bytes,
            pos,
            current: 0,
            remaining: 0,
        }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn bit(&mut self) -> MediaResult<u8> {
        if self.remaining == 0 {
            let value = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| MediaError::new("truncated JPEG entropy data"))?;
            self.pos += 1;
            if value == 0xff {
                if self.bytes.get(self.pos) != Some(&0x00) {
                    return Err(MediaError::new("JPEG scan ended before expected blocks"));
                }
                self.pos += 1;
            }
            self.current = value;
            self.remaining = 8;
        }
        self.remaining -= 1;
        Ok((self.current >> self.remaining) & 1)
    }

    pub fn bits(&mut self, count: u8) -> MediaResult<u32> {
        if count > 16 {
            return Err(MediaError::new("invalid JPEG entropy bit count"));
        }
        let mut value = 0_u32;
        for _ in 0..count {
            value = (value << 1) | u32::from(self.bit()?);
        }
        Ok(value)
    }

    pub fn signed(&mut self, count: u8) -> MediaResult<i32> {
        let value = self.bits(count)? as i32;
        if count == 0 {
            Ok(0)
        } else if value < 1 << (count - 1) {
            Ok(value + 1 - (1 << count))
        } else {
            Ok(value)
        }
    }

    pub fn restart(&mut self, expected: u8) -> MediaResult<()> {
        self.remaining = 0;
        if self.bytes.get(self.pos) != Some(&0xff) {
            return Err(MediaError::new("JPEG restart marker is missing"));
        }
        while self.bytes.get(self.pos) == Some(&0xff) {
            self.pos += 1;
        }
        if self.bytes.get(self.pos) != Some(&(0xd0 + expected)) {
            return Err(MediaError::new("JPEG restart marker is out of sequence"));
        }
        self.pos += 1;
        Ok(())
    }

    pub fn finish(&mut self) {
        self.remaining = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huffman_rejects_oversubscribed_and_truncated_tables() {
        let mut tables = Tables::new();
        assert!(
            tables
                .parse(&[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3])
                .is_err()
        );
        assert!(
            tables
                .parse(&[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                .is_err()
        );
    }

    #[test]
    fn entropy_stuffing_and_restart_are_bounded() {
        let mut bits = Bits::new(&[0xff, 0, 0xff, 0xd0, 0xff, 0xd9], 0);
        assert_eq!(bits.bits(8).unwrap(), 255);
        bits.restart(0).unwrap();
        assert!(bits.bit().is_err());
    }
}
