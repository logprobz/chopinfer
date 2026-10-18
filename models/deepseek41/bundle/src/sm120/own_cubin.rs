//! The own-op cubins' specifications, shared by the build script and the
//! kernel test crate, which rebuilds them against the same lock.

use super::own_entries::{ROPE_CUBIN, THREADS, gate_cubin, gate_symbol, gates, rope_symbol};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::own_ops::rope::{MAX_POSITIONS, RopeRows, RopeTable};
use std::path::Path;

/// CUDA's f64 `sincos` keeps a stack frame for its reduction of arguments
/// past 2^31, which no angle of the RoPE domain (below 2^21) reaches.
const SINCOS_FRAME_BYTES: u32 = 40;

fn ptr(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}

/// The cubin over the bundle's CUDA sources under `bundle`.
fn spec(
    bundle: &Path,
    name: String,
    entries: Vec<EntrySpec>,
    defines: Vec<(String, String)>,
) -> CubinSpec {
    CubinSpec {
        name,
        sources_root: bundle.to_path_buf(),
        include_dirs: vec!["cuda".into()],
        headers: vec!["#include \"deepseek41_own.cuh\"".into()],
        defines,
        entries,
    }
}

/// `geometry`'s Engram gates, masked and open.
pub fn gate_cubin_spec(bundle: &Path, geometry: &Geometry) -> CubinSpec {
    let ns = "chop::deepseek41";
    let entries = gates(geometry)
        .into_iter()
        .map(|op| {
            let body = format!(
                "{ns}::engram_gate<{},{},{}>(x,kv,q_weight,k_weight,text,live_rows,output);",
                op.width, op.streams, op.masked
            );
            let params = vec![
                ptr("x", "const uint16_t"),
                ptr("kv", "const float"),
                ptr("q_weight", "const uint16_t"),
                ptr("k_weight", "const uint16_t"),
                ptr("text", "const uint8_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("output", "uint16_t"),
            ];
            EntrySpec::new(
                gate_symbol(geometry.kernel_tag, &op),
                params,
                body,
                THREADS,
                1,
            )
        })
        .collect();
    spec(bundle, gate_cubin(geometry), entries, rope_defines())
}

/// One define per frequency (nvcc splits a command-line define at
/// commas); the header reads them in every own-op cubin.
fn rope_defines() -> Vec<(String, String)> {
    let mut defines = Vec::new();
    for (prefix, table) in [
        ("DS41_RW", RopeTable::Window),
        ("DS41_RC", RopeTable::Compressed),
    ] {
        for (pair, bits) in table.frequency_bits().iter().enumerate() {
            defines.push((format!("{prefix}{pair}"), format!("0x{bits:08x}u")));
        }
    }
    defines.push(("DS41_ROPE_POSITIONS".into(), format!("{MAX_POSITIONS}u")));
    defines
}

/// The RoPE rows of both tables, by range and by positions. The
/// frequencies come from the contract's pinned bits.
pub fn rope_cubin_spec(bundle: &Path) -> CubinSpec {
    let ns = "chop::deepseek41";
    let mut entries = Vec::new();
    for table in RopeTable::ALL {
        let compressed = table == RopeTable::Compressed;
        let mut rope = Vec::new();
        rope.push(EntrySpec::new(
            rope_symbol(table, RopeRows::Range),
            vec![
                EntryParam::U32 {
                    name: "first".into(),
                },
                EntryParam::U32 {
                    name: "rows".into(),
                },
                ptr("output", "float"),
            ],
            format!("{ns}::rope_range<{compressed}>(first,rows,output);"),
            THREADS,
            1,
        ));
        rope.push(EntrySpec::new(
            rope_symbol(table, RopeRows::Positions),
            vec![
                ptr("positions", "const uint32_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("output", "float"),
            ],
            format!("{ns}::rope_positions<{compressed}>(positions,live_rows,output);"),
            THREADS,
            1,
        ));
        for mut entry in rope {
            entry.allowed_local_bytes = SINCOS_FRAME_BYTES;
            entries.push(entry);
        }
    }
    spec(bundle, ROPE_CUBIN.into(), entries, rope_defines())
}
