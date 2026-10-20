use super::*;
use std::fs::File;
use std::sync::Arc;

struct Fixture {
    path: std::path::PathBuf,
    mapping: Arc<MappedRowsV1>,
    bytes: Vec<u8>,
    width: usize,
}
impl Fixture {
    fn new(label: &str, rows: usize, width: usize) -> Self {
        let path =
            std::env::temp_dir().join(format!("mapped-fetch-{label}-{}", std::process::id()));
        let bytes = (0..rows * width)
            .map(|i| ((i * 13) ^ (i / 127) ^ (i / 4093)) as u8)
            .collect::<Vec<_>>();
        let mut file_bytes = vec![0x5a; 300];
        file_bytes.extend(&bytes);
        std::fs::write(&path, file_bytes).unwrap();
        let file = File::open(&path).unwrap();
        let mapping = super::super::map_rows_v1(&file, 300, bytes.len() as u64).unwrap();
        Self {
            path,
            mapping,
            bytes,
            width,
        }
    }
    fn expected(&self, rows: &[u32]) -> Vec<u8> {
        rows.iter()
            .flat_map(|&r| {
                self.bytes[r as usize * self.width..][..self.width]
                    .iter()
                    .copied()
            })
            .collect()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[test]
fn default_is_on_and_zero_keeps_the_legacy_choice() {
    assert!(configured(None));
    assert!(configured(Some("1".into())));
    assert!(!configured(Some("0".into())));
    assert!(!configured(Some("invalid".into())));
}

#[test]
fn queued_rows_keep_order_duplicates_offsets_and_cross_page_bytes() {
    let f = Fixture::new("ordered", 6000, 97);
    let rows = (0..5000_u32).map(|i| (i * 137) % 6000).collect::<Vec<_>>();
    let mut out = vec![0xa5; rows.len() * f.width];
    copy_rows(&f.mapping, f.width, &rows, &mut out).unwrap();
    assert_eq!(out, f.expected(&rows));
}

#[test]
fn miss_positions_preserve_prefilled_hits_and_reject_invalid_work_before_writes() {
    let f = Fixture::new("positions", 100, 264);
    let rows = [17, 2, 17, 99];
    let expected = f.expected(&rows);
    let mut out = vec![0xa5; expected.len()];
    out[..264].copy_from_slice(&expected[..264]);
    out[528..792].copy_from_slice(&expected[528..792]);
    copy_misses(&f.mapping, 264, &rows, &[1, 3], &mut out).unwrap();
    assert_eq!(out, expected);
    for misses in [vec![3, 1], vec![1, 1], vec![4]] {
        let mut poison = vec![0xa5; expected.len()];
        assert!(copy_misses(&f.mapping, 264, &rows, &misses, &mut poison).is_err());
        assert!(poison.iter().all(|&b| b == 0xa5));
    }
    let mut poison = vec![0xa5; 264];
    assert!(copy_rows(&f.mapping, 264, &[100], &mut poison).is_err());
    assert!(poison.iter().all(|&b| b == 0xa5));
}

#[test]
fn retained_descriptor_survives_path_replacement_and_io_failure_is_not_a_mapped_read() {
    let f = Fixture::new("identity", 100, 264);
    let other = f.path.with_extension("replacement");
    std::fs::write(&other, vec![0x6b; f.bytes.len() + 300]).unwrap();
    std::fs::rename(&other, &f.path).unwrap();
    let rows = [3, 7, 99];
    let mut out = vec![0; rows.len() * 264];
    copy_rows(&f.mapping, 264, &rows, &mut out).unwrap();
    assert_eq!(out, f.expected(&rows));
    std::fs::write(&other, []).unwrap();
    let empty = Arc::new(File::open(&other).unwrap());
    out.fill(0xa5);
    assert!(
        copy_misses_from(&f.mapping, 264, &rows, &[0, 1, 2], &mut out, Some(&empty))
            .unwrap_err()
            .contains("read mapped row source")
    );
    assert!(out.iter().all(|&b| b == 0xa5));
    std::fs::remove_file(other).unwrap();
}

#[test]
fn missing_descriptor_keeps_exact_legacy_bytes() {
    let f = Fixture::new("mapped-fallback", 100, 8);
    let rows = [5, 9, 5];
    let mut out = vec![0; 24];
    copy_misses_from(&f.mapping, 8, &rows, &[0, 1, 2], &mut out, None).unwrap();
    assert_eq!(out, f.expected(&rows));
}

#[cfg(target_os = "linux")]
#[test]
fn successful_descriptor_reads_never_fault_the_original_mapping() {
    use std::io::{Read, Seek, SeekFrom};
    let f = Fixture::new("no-mmap", 1000, 256);
    f.mapping
        .advise(super::super::RowAdviceV1::DontNeed)
        .unwrap();
    let present = || {
        let address = f.mapping.bytes().as_ptr() as u64;
        let mut file = File::open("/proc/self/pagemap").unwrap();
        file.seek(SeekFrom::Start(
            address / super::super::page_bytes_v1() as u64 * 8,
        ))
        .unwrap();
        let mut entry = [0; 8];
        file.read_exact(&mut entry).unwrap();
        u64::from_ne_bytes(entry) & (1 << 63) != 0
    };
    assert!(!present());
    let rows = [0, 1, 17, 99];
    let mut out = vec![0; 1024];
    copy_rows(&f.mapping, 256, &rows, &mut out).unwrap();
    assert_eq!(out, f.expected(&rows));
    assert!(
        !present(),
        "queued reads must not dereference the table mmap"
    );
}
