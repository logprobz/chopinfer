//! The generated catalog against the checkpoint's safetensors headers. Set
//! `CHOPINFER_GLM5NEXT_CHECKPOINT` to the pinned checkpoint directory; the
//! test skips otherwise. Only headers are read, never tensor data.
use model_glm5next_contract::{admission, catalog, weight_plan};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Path;

const MAX_HEADER: u64 = 64 << 20;

fn header(path: &Path) -> serde_json::Map<String, Value> {
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut length = [0; 8];
    file.read_exact(&mut length).unwrap();
    let length = u64::from_le_bytes(length);
    assert!(
        length <= MAX_HEADER,
        "{}: header of {length} bytes",
        path.display()
    );
    let mut bytes = vec![0; length as usize];
    file.read_exact(&mut bytes).unwrap();
    let Value::Object(mut header) = serde_json::from_slice(&bytes).unwrap() else {
        panic!("{}: header is not an object", path.display());
    };
    header.remove("__metadata__");
    header
}

#[test]
fn catalog_matches_checkpoint_headers() {
    let Some(root) = std::env::var_os("CHOPINFER_GLM5NEXT_CHECKPOINT") else {
        eprintln!("skipped: CHOPINFER_GLM5NEXT_CHECKPOINT is not set");
        return;
    };
    let root = Path::new(&root);
    let matched =
        admission::admit_bytes(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
    let index: Value =
        serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let weight_map = index["weight_map"].as_object().unwrap();
    let shards = weight_map
        .values()
        .map(|file| file.as_str().unwrap())
        .collect::<BTreeSet<_>>();
    let mut observed = BTreeMap::new();
    for shard in shards {
        for (name, entry) in header(&root.join(shard)) {
            assert_eq!(weight_map[&name], shard, "{name}: indexed in another shard");
            let offsets = entry["data_offsets"].as_array().unwrap();
            let bytes = offsets[1].as_u64().unwrap() - offsets[0].as_u64().unwrap();
            let shape = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap())
                .collect::<Vec<_>>();
            let dtype = entry["dtype"].as_str().unwrap().to_owned();
            assert!(observed.insert(name, (dtype, shape, bytes)).is_none());
        }
    }
    assert_eq!(observed.len(), weight_map.len());
    let tensors = catalog::catalog(matched.geometry);
    let expected = tensors
        .iter()
        .map(|t| {
            (
                t.name.clone(),
                (t.dtype.name().to_owned(), t.shape.clone(), t.bytes()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let missing = expected
        .keys()
        .filter(|name| !observed.contains_key(*name))
        .collect::<Vec<_>>();
    let extra = observed
        .keys()
        .filter(|name| !expected.contains_key(*name))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "missing {missing:?}; extra {extra:?}"
    );
    for (name, entry) in &expected {
        assert_eq!(&observed[name], entry, "{name}");
    }
    let total = observed.values().map(|(_, _, bytes)| bytes).sum::<u64>();
    assert_eq!((observed.len(), total), (76_108, 328_326_771_576));
    let ranks = weight_plan::plan(matched.geometry, &tensors, 4).unwrap();
    for rank in &ranks {
        eprintln!(
            "rank {}: {:?}, total {}",
            rank.rank,
            rank.bytes,
            rank.bytes.total()
        );
    }
    assert_eq!(ranks[0].bytes.total(), 83_370_801_144);
}
