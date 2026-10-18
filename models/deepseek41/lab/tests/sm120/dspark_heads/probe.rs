use kernel_build::{BuildOptions, CubinSpec, EntryParam, EntrySpec, LockDir, Toolchain};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const SYMBOL: &str = "chop_deepseek41_dspark_test_producer_v1";
const CUBIN: &str = "deepseek41_dspark_test_sm120";

pub fn cubin() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        if let Some(path) = std::env::var_os("CHOPINFER_DEEPSEEK41_PROBE_CUBIN") {
            return std::fs::read(path).expect("prebuilt test producer");
        }
        let lab = Path::new(env!("CARGO_MANIFEST_DIR"));
        let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let spec = CubinSpec {
            name: CUBIN.into(),
            sources_root: lab.into(),
            include_dirs: vec!["cuda".into()],
            headers: vec!["#include \"dspark_probe.cuh\"".into()],
            defines: vec![],
            entries: vec![EntrySpec::new(
                SYMBOL.into(),
                vec![
                    ptr("eviction", "const uint4"),
                    ptr("checksum", "uint32_t"),
                    EntryParam::U64 {
                        name: "vectors".into(),
                    },
                    ptr("operands", "const dspark_test::Operand"),
                    EntryParam::U32 {
                        name: "count".into(),
                    },
                    ptr("live", "uint32_t"),
                    EntryParam::U32 {
                        name: "rows".into(),
                    },
                    EntryParam::U64 {
                        name: "tail_ns".into(),
                    },
                ],
                "dspark_test::produce(eviction,checksum,vectors,operands,count,live,rows,tail_ns);"
                    .into(),
                128,
                1,
            )],
        };
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("dspark-probe");
        std::fs::create_dir_all(&out).unwrap();
        let built = kernel_build::build_in_families(
            &Toolchain::detect().unwrap(),
            &[spec],
            &out,
            &LockDir::new(lab.join("kernels.lock.d")).family("dspark_probe", [CUBIN]),
            BuildOptions {
                emit_cargo_directives: false,
                profile_line_info: false,
                ..BuildOptions::from_env()
            },
            None,
        )
        .expect("test producer cubin build and lock");
        std::fs::read(&built[0].path).unwrap()
    })
}

#[test]
fn head_cpu_producer_matches_lock() {
    assert!(
        cubin()
            .windows(SYMBOL.len())
            .any(|b| b == SYMBOL.as_bytes())
    );
}
