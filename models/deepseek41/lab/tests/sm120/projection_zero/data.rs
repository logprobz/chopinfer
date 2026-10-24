use super::*;
pub fn synthetic(rows: u32, pattern: &str, seed: u32) -> (Vec<f32>, Vec<u16>) {
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        ((0x3c00 + (state % 1024)) | ((state >> 16) & 0x8000)) as u16
    };
    let mut x = (0..rows as usize * K)
        .map(|_| f32::from_bits(u32::from(next()) << 16))
        .collect::<Vec<_>>();
    let mut w = (0..n() * K).map(|_| next()).collect::<Vec<_>>();
    for r in 0..rows as usize {
        x[r * K] = if r % 2 == 0 { 0.0 } else { -0.0 };
        match pattern {
            "residual" => x[r * K + 127] = f32::from_bits(0x3f800081),
            "mixed" => match r % 3 {
                1 => x[r * K + 127] = f32::from_bits(0x3f800081),
                2 => x[r * K + 127] = f32::from_bits(1),
                _ => {}
            },
            "edges" => {
                x[r * K + 127] = f32::from_bits(
                    [
                        0,
                        0x80000000,
                        1,
                        0x007fffff,
                        0x00800000,
                        0x7f7fffff,
                        0x7f800000,
                        0xff800000,
                        0x7fc12345,
                        97 << 23,
                        (158 << 23) - 1,
                    ][r % 11],
                )
            }
            "bf16" | "weight" => {}
            _ => panic!("unknown pattern"),
        }
    }
    if pattern == "weight" {
        w[K - 1] = 1;
    }
    if pattern == "edges" {
        for c in 0..n() {
            w[c * K + 127] = [0, 0x8000, 1, 0x007f, 0x7f7f, 0x7f80, 0xff80, 0x7fc1, 0x7f81][c % 9];
        }
    }
    (x, w)
}
pub fn manifest() -> LabResult<Value> {
    serde_json::from_slice(
        &std::fs::read(root().join("inputs/manifest.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
pub fn read(layer: u32, name: &str) -> LabResult<Vec<u8>> {
    use std::{io::Read, os::unix::fs::OpenOptionsExt};
    let manifest = manifest()?;
    let key = format!("L{layer:02}-{name}");
    let expected = &manifest["files"][&key];
    // Linux O_NOFOLLOW, matching the staged-input readers.
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(0x20000)
        .open(root().join("inputs").join(&key))
        .map_err(|e| e.to_string())?;
    let bytes = expected["bytes"].as_u64().ok_or("capture length")?;
    if bytes > 8192 * K as u64 * 4 || file.metadata().map_err(|e| e.to_string())?.len() != bytes {
        return Err("capture extent differs".into());
    }
    let mut b = Vec::new();
    file.take(bytes + 1)
        .read_to_end(&mut b)
        .map_err(|e| e.to_string())?;
    if b.len() as u64 != expected["bytes"].as_u64().unwrap()
        || kernel_build::digest(&b) != expected["sha256"]
    {
        return Err(format!("capture changed: {key}"));
    }
    Ok(b)
}
pub fn normalized(
    s: &Session,
    f: &Functions,
    input: &[u8],
    weight: &[u8],
    eps: u32,
) -> LabResult<Vec<f32>> {
    if input.len() != 8192 * K * 2 || weight.len() != K * 2 {
        return Err("norm fixture extent differs".into());
    }
    let x = s.allocate(8192 * K * 2)?;
    x.copy_from_bytes(input, &s.stream)
        .map_err(|e| e.to_string())?;
    let w = s.allocate(K * 2)?;
    w.copy_from_bytes(weight, &s.stream)
        .map_err(|e| e.to_string())?;
    let y = s.allocate(8192 * K * 2)?;
    let live = s.allocate(4)?;
    live.copy_from_bytes(&8192u32.to_le_bytes(), &s.stream)
        .map_err(|e| e.to_string())?;
    f.call(
        "chop_projection_zero_replay_norm_v1",
        &s.stream,
        family::norm().plan(8192).unwrap(),
        &[
            A::Buffer(&x),
            A::Buffer(&w),
            A::Buffer(&x),
            A::Buffer(&y),
            A::Buffer(&y),
            A::Buffer(&live),
            A::U32(eps),
        ],
    )?;
    let x = poison::read(s, &y)?
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16))
        .collect();
    Ok(x)
}
pub fn captured(s: &Session, f: &Functions, layer: u32) -> LabResult<(Vec<f32>, Vec<u16>)> {
    let x = normalized(
        s,
        f,
        &read(layer, "norm-input.bf16")?,
        &read(layer, "norm-weight.bf16")?,
        manifest()?["epsilon_bits"].as_u64().ok_or("epsilon bits")? as u32,
    )?;
    let w = read(layer, &format!("weight-n{}.bf16", n()))?
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Ok((x, w))
}
