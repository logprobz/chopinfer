use kernel_core::contract::DType;
use model_compiler_sdk::{quantization::source::TensorInventory, source::SourceFiles};
use serde_json::json;
use std::{collections::BTreeSet, fs, path::PathBuf};

struct Source(PathBuf);
impl Source {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("quantized-source-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        let source = Self(path);
        for (file, name, dtype, shape, bytes) in [
            ("weights", "linear.weight", "U8", vec![2], vec![0x17, 0x89]),
            (
                "scales",
                "linear.weight_scale",
                "U8",
                vec![2],
                vec![127, 128],
            ),
            ("state", "state.ids", "I64", vec![2], vec![0; 16]),
            (
                "patch",
                "vision.patch",
                "BF16",
                vec![2, 3, 2, 4, 4],
                vec![0; 384],
            ),
        ] {
            let header = serde_json::to_vec(&json!({name: {
                "dtype": dtype, "shape": shape, "data_offsets": [0, bytes.len()]
            }}))
            .unwrap();
            let mut body = (header.len() as u64).to_le_bytes().to_vec();
            body.extend(header);
            body.extend(bytes);
            fs::write(source.0.join(file), body).unwrap();
        }
        source.index(
            json!({"linear.weight":"weights", "linear.weight_scale":"scales",
            "state.ids":"state", "vision.patch":"patch"}),
        );
        source
    }
    fn index(&self, map: serde_json::Value) {
        fs::write(
            self.0.join("index.json"),
            serde_json::to_vec(&json!({"weight_map":map})).unwrap(),
        )
        .unwrap();
    }
    fn capture(&self) -> SourceFiles {
        SourceFiles::capture(&self.0, &[("index.json", 4096)], &[], "index.json").unwrap()
    }
}
impl Drop for Source {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn captured_inventory_is_closed_and_reads_only_declared_representations() {
    let source = Source::new();
    let files = source.capture();
    let inventory = TensorInventory::from_captured(&files, "index.json").unwrap();
    let mut names = BTreeSet::from(["linear.weight".into()]);
    assert!(
        inventory
            .require_accounted(&names)
            .unwrap_err()
            .to_string()
            .contains("linear.weight_scale")
    );
    names.insert("linear.weight_scale".into());
    names.extend(["state.ids".into(), "vision.patch".into()]);
    inventory.require_accounted(&names).unwrap();
    assert_eq!(inventory.tensors["state.ids"].dtype, DType::I64);
    assert_eq!(
        inventory
            .read(&files, &inventory.tensors["state.ids"])
            .unwrap(),
        vec![0; 16]
    );
    assert_eq!(inventory.tensors["vision.patch"].shape, [2, 3, 2, 4, 4]);
    assert_eq!(
        inventory
            .read(&files, &inventory.tensors["vision.patch"])
            .unwrap(),
        vec![0; 384]
    );
    let mut scale = inventory.tensors["linear.weight_scale"].clone();
    scale.dtype = DType::Ue8M0;
    assert_eq!(inventory.read(&files, &scale).unwrap(), [127, 128]);
    let mut row = scale.clone();
    row.offset = 1;
    row.shape = vec![1];
    row.byte_length = 1;
    assert_eq!(inventory.read_slice(&files, &row).unwrap(), [128]);
    assert!(inventory.read(&files, &row).is_err());
    row.offset = 2;
    assert!(inventory.read_slice(&files, &row).is_err());
    let mut unaligned = inventory.tensors["state.ids"].clone();
    unaligned.shape = vec![1];
    unaligned.byte_length = 8;
    unaligned.offset = 1;
    assert!(inventory.read_slice(&files, &unaligned).is_err());
    scale.dtype = DType::Fp8E4M3;
    assert!(inventory.read(&files, &scale).is_err());
    scale.dtype = DType::U8;
    scale.shape = vec![1, 2];
    assert!(inventory.read(&files, &scale).is_err());
    let mut stale = inventory.clone();
    stale.spans.get_mut("linear.weight").unwrap().offset += 1;
    assert!(
        stale
            .read(&files, &inventory.tensors["linear.weight"])
            .unwrap_err()
            .to_string()
            .contains("offset differs")
    );
    drop(files);
    source.index(
        json!({"linear.weight":"weights", "linear.weight_scale":"scales", "missing":"weights"}),
    );
    assert!(TensorInventory::from_captured(&source.capture(), "index.json").is_err());
    source.index(json!({"linear.weight":"scales", "linear.weight_scale":"weights"}));
    assert!(
        TensorInventory::from_captured(&source.capture(), "index.json")
            .unwrap_err()
            .to_string()
            .contains("shard assignment")
    );
}
