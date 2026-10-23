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
fn reader_window_defaults_and_accepts_only_the_reader_thread_range() {
    assert!(valid_readers(DEFAULT_READERS));
    assert_eq!(configured_readers(None), DEFAULT_READERS);
    assert_eq!(configured_readers(Some("1".into())), 1);
    let most = pool::WORKERS.to_string();
    assert_eq!(configured_readers(Some(most.into())), pool::WORKERS);
    let over = (pool::WORKERS + 1).to_string();
    for value in ["0", over.as_str(), "-1", "sixteen", ""] {
        assert_eq!(
            configured_readers(Some(value.into())),
            DEFAULT_READERS,
            "{value}"
        );
    }
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

#[test]
fn evicted_rows_are_started_together_and_copied_exactly() {
    use crate::page_cache::{CacheAdviceV1, advise_v1, cached_pages_v1};
    let f = Fixture::new("evicted", 4000, 264);
    let file = File::open(&f.path).unwrap();
    file.sync_all().unwrap();
    advise_v1(&file, 0, 0, CacheAdviceV1::DontNeed).unwrap();
    let evicted =
        cached_pages_v1(&file, 0, 300 + f.bytes.len() as u64).is_ok_and(|(cached, _)| cached == 0);
    let rows = (0..700_u64)
        .map(|i| (i * 2_654_435_761 % 4000) as u32)
        .collect::<Vec<_>>();
    let started = || COLD.load(Ordering::Relaxed);
    let before = started();
    let mut out = vec![0xa5; rows.len() * 264];
    copy_rows(&f.mapping, 264, &rows, &mut out).unwrap();
    assert_eq!(out, f.expected(&rows));
    if start_reads() && evicted {
        // Rows on distinct pages were all cold; later rows of a page may
        // find it cached once an earlier share read it.
        assert!(started() > before);
    } else {
        eprintln!("skipped started-read count: reads not started or file not evicted");
    }
    // The same rows again come from the page cache, byte for byte.
    out.fill(0xa5);
    copy_rows(&f.mapping, 264, &rows, &mut out).unwrap();
    assert_eq!(out, f.expected(&rows));
}

#[test]
fn joint_planes_read_every_planes_misses_in_one_wait_and_copy_exactly() {
    let codes = Fixture::new("joint-codes", 3000, 97);
    let scales = Fixture::new("joint-scales", 3000, 8);
    let code_rows = [5, 2999, 5, 0, 1234, 2999];
    let scale_rows = [7, 7, 0];
    let mut code_out = vec![0xa5; code_rows.len() * 97];
    let mut scale_out = vec![0xa5; scale_rows.len() * 8];
    let mut asked = Vec::new();
    copy_planes_with(
        &mut [
            PlaneCopy {
                mapping: &codes.mapping,
                width: 97,
                rows: &code_rows,
                out: &mut code_out,
            },
            PlaneCopy {
                mapping: &scales.mapping,
                width: 8,
                rows: &[],
                out: &mut [],
            },
            PlaneCopy {
                mapping: &scales.mapping,
                width: 8,
                rows: &scale_rows,
                out: &mut scale_out,
            },
        ],
        |groups| {
            asked.push(
                groups
                    .iter()
                    .map(|(_, requests)| requests.to_vec())
                    .collect::<Vec<_>>(),
            );
            pool::read_groups(groups)
        },
    )
    .unwrap();
    // One read for both tables, each distinct row once in first-use order,
    // and none for the plane without rows.
    let reads = |width: usize, rows: &[u64]| {
        rows.iter()
            .map(|&row| (300 + row * width as u64, width))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        asked,
        [vec![reads(97, &[5, 2999, 0, 1234]), reads(8, &[7, 0])]]
    );
    assert_eq!(code_out, codes.expected(&code_rows));
    assert_eq!(scale_out, scales.expected(&scale_rows));
    // The served reader copies the same bytes.
    code_out.fill(0xa5);
    scale_out.fill(0xa5);
    copy_planes(&mut [
        PlaneCopy {
            mapping: &codes.mapping,
            width: 97,
            rows: &code_rows,
            out: &mut code_out,
        },
        PlaneCopy {
            mapping: &scales.mapping,
            width: 8,
            rows: &scale_rows,
            out: &mut scale_out,
        },
    ])
    .unwrap();
    assert_eq!(code_out, codes.expected(&code_rows));
    assert_eq!(scale_out, scales.expected(&scale_rows));
    // A bad row in any plane, or a read that answers the wrong shape,
    // publishes no plane.
    type Read = fn(&[pool::Group<'_>]) -> Result<pool::GroupRows, String>;
    let refused = |rows: &[u32], read: Read| {
        let mut code_poison = vec![0xa5; code_rows.len() * 97];
        let mut scale_poison = vec![0xa5; rows.len() * 8];
        let failed = copy_planes_with(
            &mut [
                PlaneCopy {
                    mapping: &codes.mapping,
                    width: 97,
                    rows: &code_rows,
                    out: &mut code_poison,
                },
                PlaneCopy {
                    mapping: &scales.mapping,
                    width: 8,
                    rows,
                    out: &mut scale_poison,
                },
            ],
            read,
        )
        .is_err();
        failed && code_poison.iter().chain(&scale_poison).all(|&b| b == 0xa5)
    };
    assert!(refused(&[1, 3000], |_| panic!("read after a bad row")));
    assert!(refused(&[1, 2], |_| Ok((vec![Vec::new()], false))));
    assert!(refused(&[1, 2], |groups| {
        // The first group's rows are right; the second's are missing.
        let (mut rows, serial) = pool::read_groups(&groups[..1])?;
        rows.push(Vec::new());
        Ok((rows, serial))
    }));
}

#[test]
fn a_plane_without_a_descriptor_or_past_one_tile_reads_every_plane_alone() {
    let codes = Fixture::new("alone-codes", 100, 264);
    let mut scales = Fixture::new("alone-scales", 5000, 8);
    let code_rows = [17, 2, 17, 99];
    let refuse =
        |_: &[pool::Group<'_>]| -> Result<pool::GroupRows, String> { panic!("planes read alone") };
    // One more distinct row than a tile of 8-byte rows.
    let wide = (0..4097_u32).collect::<Vec<_>>();
    for scale_rows in [wide, vec![5, 9, 5]] {
        let mut code_out = vec![0xa5; code_rows.len() * 264];
        let mut scale_out = vec![0xa5; scale_rows.len() * 8];
        if scale_rows.len() < 4097 {
            // The process's mapping registry holds a weak reference; drop it
            // so this test owns the mapping alone and can close its source.
            let key = scales.mapping.key;
            super::super::REGISTRY.lock().unwrap().remove(&key);
            Arc::get_mut(&mut scales.mapping).unwrap().source = None;
        }
        copy_planes_with(
            &mut [
                PlaneCopy {
                    mapping: &codes.mapping,
                    width: 264,
                    rows: &code_rows,
                    out: &mut code_out,
                },
                PlaneCopy {
                    mapping: &scales.mapping,
                    width: 8,
                    rows: &scale_rows,
                    out: &mut scale_out,
                },
            ],
            refuse,
        )
        .unwrap();
        assert_eq!(code_out, codes.expected(&code_rows));
        assert_eq!(scale_out, scales.expected(&scale_rows));
    }
}
