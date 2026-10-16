//! Read supplied contracts as bounded data, with no compiler or dynamic loader.
use kernel_core::contract::{DType, Epilogues, Target, WeightFormat};
use kernel_core::precision::capability::{
    ActivationCapability, CapabilityDocument, GlobalScaleCapability, OperationCapability,
};
use kernel_core::precision::{
    ArithmeticSpec, DecodeOrder, ExecutionPhase, MatrixGeometry, OperationKind, OperationUse,
    Rounding, WeightLayout,
};
use model_compiler_sdk::native_contract::NativeModule;
use runtime_bundle::preparation::{ImplementationContract, NativeContracts, NativeProfile};
use runtime_weights::sha256_bytes;
use std::fs;

fn capabilities() -> Vec<u8> {
    serde_json::to_vec(&CapabilityDocument {
        version: 1,
        operations: vec![OperationCapability {
            kind: OperationKind::Dense,
            target: Target::Sm120f,
            usage: OperationUse {
                phase: ExecutionPhase::Decode,
                capacity: 1,
                graph: false,
            },
            geometry: MatrixGeometry { rows: 256, k: 256 },
            operands: 1,
            weight: WeightFormat::Nvfp4,
            activation: Some(ActivationCapability::Bf16),
            epilogues: Epilogues::default(),
            layout: WeightLayout::row_major(),
            local_scale_dtypes: vec![DType::Fp8E4M3],
            globals: GlobalScaleCapability::SharedScalar,
            shared_local_scale: false,
            arithmetic: ArithmeticSpec {
                weight_operand: DType::Bf16,
                accumulator: DType::F32,
                output: DType::Bf16,
                decode_order: DecodeOrder::ScaleThenCode,
                rounding: Rounding::NearestEven,
                numerical_contract: "fixture.declared-arithmetic.v1".into(),
            },
            implementation: "fixture.supplied.v1".into(),
            stages: vec!["fixture_compute_v1".into()],
        }],
    })
    .unwrap()
}

fn module(payload: &[u8], expected: Option<&[u8]>, duplicate: bool, kind: u32) -> Vec<u8> {
    let contracts = NativeContracts {
        version: 1,
        profiles: vec![NativeProfile {
            profile: "fixture".into(),
            tensor_parallel_size: 1,
            weight_contract: "a".repeat(64),
            package_contract: "b".repeat(64),
        }],
        implementation_contract: Some(ImplementationContract {
            selection: "c".repeat(64),
            dependencies: "d".repeat(64),
            capabilities: expected.map(sha256_bytes),
        }),
    };
    let json = serde_json::to_vec(&contracts).unwrap();
    let mut note = Vec::new();
    for n in [10_u32, json.len() as u32, 1] {
        note.extend(n.to_le_bytes());
    }
    note.extend(b"CHOPINFER\0\0\0");
    note.extend(json);
    note.resize(note.len().next_multiple_of(4), 0);
    let names = b"\0.shstrtab\0.note.chopinfer.model\0.chopinfer.precision\0";
    let count = if duplicate { 5 } else { 4 };
    let mut elf = vec![0; 64 + count * 64];
    elf[..6].copy_from_slice(b"\x7fELF\x02\x01");
    elf[16..18].copy_from_slice(&3_u16.to_le_bytes());
    elf[18..20].copy_from_slice(&62_u16.to_le_bytes());
    elf[40..48].copy_from_slice(&64_u64.to_le_bytes());
    elf[58..60].copy_from_slice(&64_u16.to_le_bytes());
    elf[60..62].copy_from_slice(&(count as u16).to_le_bytes());
    elf[62..64].copy_from_slice(&1_u16.to_le_bytes());
    let mut sections = vec![
        (1_u32, 3, names.as_slice()),
        (11, 7, note.as_slice()),
        (33, kind, payload),
    ];
    if duplicate {
        sections.push((33, kind, payload));
    }
    for (i, (name, kind, data)) in sections.into_iter().enumerate() {
        let base = 64 + (i + 1) * 64;
        let offset = elf.len() as u64;
        elf[base..base + 4].copy_from_slice(&name.to_le_bytes());
        elf[base + 4..base + 8].copy_from_slice(&kind.to_le_bytes());
        elf[base + 24..base + 32].copy_from_slice(&offset.to_le_bytes());
        elf[base + 32..base + 40].copy_from_slice(&(data.len() as u64).to_le_bytes());
        elf.extend_from_slice(data);
    }
    elf
}

#[test]
fn authenticates_bounded_capabilities_without_loading_or_compiling_code() {
    let root = std::env::temp_dir().join(format!("native-capabilities-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let path = root.join("module.bin");
    let original = capabilities();
    fs::write(&path, module(&original, Some(&original), false, 1)).unwrap();
    let supplied = NativeModule::inspect(&path).unwrap();
    let inventory = supplied.precision_inventory().unwrap();
    assert_eq!(inventory.document().operations.len(), 1);
    assert_eq!(inventory.document().operations[0].geometry.k, 256);
    drop(supplied);
    for (bytes, reason) in [
        (module(b"{}", Some(&original), false, 1), "digest mismatch"),
        (
            module(&original, None, false, 1),
            "no portable precision capabilities",
        ),
        (
            module(&original, Some(&original), true, 1),
            "duplicate native section",
        ),
        (
            module(&original, Some(&original), false, 8),
            "wrong ELF section type",
        ),
        (module(b"{}", Some(b"{}"), false, 1), "missing field"),
    ] {
        fs::write(&path, bytes).unwrap();
        let error = NativeModule::inspect(&path)
            .unwrap()
            .precision_inventory()
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(reason), "{error}");
    }
    fs::remove_dir_all(root).unwrap();
}
