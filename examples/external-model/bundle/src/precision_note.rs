runtime_bundle_sdk::embed_note!(
    include_bytes!(concat!(env!("OUT_DIR"), "/precision-note.bin")),
    precision = include_bytes!(concat!(env!("OUT_DIR"), "/precision-capabilities.json"))
);
