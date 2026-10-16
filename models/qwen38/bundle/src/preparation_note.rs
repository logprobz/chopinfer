//! Native preparation and precision metadata emitted by the build.

runtime_bundle_sdk::embed_note!(
    include_bytes!(concat!(env!("OUT_DIR"), "/preparation-note.bin")),
    precision = include_bytes!(concat!(env!("OUT_DIR"), "/precision-capabilities.json"))
);

const PRECISION: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/precision-capabilities.json"));

pub fn binding_dependency() -> &'static str {
    include_str!(concat!(env!("OUT_DIR"), "/precision-dependency.txt"))
}

pub fn capabilities() -> &'static [u8] {
    PRECISION
}
