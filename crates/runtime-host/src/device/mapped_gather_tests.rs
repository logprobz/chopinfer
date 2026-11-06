use super::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

struct Probe {
    clock: AtomicU64,
    step: u64,
    asked: Mutex<Vec<Vec<u32>>>,
}

impl Probe {
    fn new(step: u64) -> Self {
        Self {
            clock: AtomicU64::new(0),
            step,
            asked: Mutex::new(Vec::new()),
        }
    }
}

impl GatherProbe for Probe {
    fn now(&self) -> Duration {
        Duration::from_micros(self.clock.fetch_add(self.step, Ordering::Relaxed))
    }
    fn read_ahead(&self, rows: &[u32]) {
        self.asked.lock().unwrap().push(rows.to_vec());
    }
}

fn plane(label: &str) -> MappedGatherPlaneV1 {
    // The mapping starts inside a page; rows cross pages and carry arbitrary bytes.
    let bytes: Vec<u8> = (0..(257 * 97 + 19))
        .map(|i| (i * 31 + i / 251) as u8)
        .collect();
    let path = std::env::temp_dir().join(format!("async-gather-{label}-{}", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let mapping = crate::host_rows::map_rows_v1(&file, 19, (257 * 97) as u64).unwrap();
    std::fs::remove_file(path).unwrap();
    MappedGatherPlaneV1 {
        mapping,
        row_bytes: 97,
        index_offset: 8,
        stream: DeviceObjectV1::new(1),
        destination: DeviceRangeV1 {
            object: DeviceObjectV1::new(2),
            offset: 0,
            length: 97 * 80,
        },
    }
}

fn input(ids: &[u32]) -> Vec<u8> {
    [u32::MAX; 2]
        .into_iter()
        .chain(ids.iter().copied())
        .chain([u32::MAX; 2])
        .flat_map(u32::to_le_bytes)
        .collect()
}

fn expected(plane: &MappedGatherPlaneV1, ids: &[u32]) -> Vec<u8> {
    ids.iter()
        .flat_map(|&id| {
            plane.mapping.bytes()[id as usize * plane.row_bytes..][..plane.row_bytes].to_vec()
        })
        .collect()
}

#[test]
fn cold_async_plane_advises_only_the_unread_live_rows_once() {
    let plane = plane("cold");
    let ids: Vec<u32> = (0..53).map(|i| (i * 37 % 257) as u32).collect();
    let probe = Probe::new(50);
    let mut output = vec![0xa5; ids.len() * plane.row_bytes];
    gather_plane_with_probe(&plane, &input(&ids), &mut output, &probe, true).unwrap();
    assert_eq!(output, expected(&plane, &ids));
    assert_eq!(*probe.asked.lock().unwrap(), [ids[16..].to_vec()]);
}

#[test]
fn warm_and_short_async_planes_keep_bytes_without_advice() {
    let plane = plane("warm");
    for (count, step) in [(0, 100), (1, 100), (16, 100), (53, 0), (53, 40)] {
        // Duplicate and reversed indices must retain their exact byte order.
        let ids: Vec<u32> = (0..count).map(|i| 256 - (i / 2) as u32).collect();
        let probe = Probe::new(step);
        let mut output = vec![0xa5; count * plane.row_bytes];
        gather_plane_with_probe(&plane, &input(&ids), &mut output, &probe, true).unwrap();
        assert_eq!(output, expected(&plane, &ids));
        assert!(probe.asked.lock().unwrap().is_empty(), "{count} {step}");
    }
}

#[test]
fn invalid_async_ids_are_refused_before_copy_or_advice() {
    let plane = plane("invalid");
    let mut ids = vec![1; 53];
    ids[52] = 257;
    let probe = Probe::new(100);
    let mut output = vec![0xa5; ids.len() * plane.row_bytes];
    assert_eq!(
        gather_plane_with_probe(&plane, &input(&ids), &mut output, &probe, true),
        Err("mapped gather device index is outside the table".into())
    );
    assert!(output.iter().all(|&v| v == 0xa5));
    assert!(probe.asked.lock().unwrap().is_empty());
    assert_eq!(probe.clock.load(Ordering::Relaxed), 0);
    assert_eq!(
        gather_plane_with_probe(&plane, &[0; 8], &mut output, &probe, true),
        Err("mapped gather input is truncated".into())
    );
    assert!(output.iter().all(|&v| v == 0xa5));
}

#[test]
fn async_plane_resets_read_ahead_for_each_reused_gather() {
    let plane = plane("reuse");
    let probe = Probe::new(50);
    for ids in [vec![2; 33], vec![256; 17]] {
        let mut output = vec![0; ids.len() * plane.row_bytes];
        gather_plane_with_probe(&plane, &input(&ids), &mut output, &probe, true).unwrap();
        assert_eq!(output, expected(&plane, &ids));
    }
    assert_eq!(*probe.asked.lock().unwrap(), [vec![2; 17], vec![256; 1]]);
}

#[test]
fn disabled_async_read_ahead_copies_without_advice() {
    let plane = plane("disabled");
    let ids: Vec<u32> = (0..53).map(|i| (i * 37 % 257) as u32).collect();
    let probe = Probe::new(50);
    let mut output = vec![0xa5; ids.len() * plane.row_bytes];
    gather_plane_with_probe(&plane, &input(&ids), &mut output, &probe, false).unwrap();
    assert_eq!(output, expected(&plane, &ids));
    assert!(probe.asked.lock().unwrap().is_empty());
}

/// Whether this process maps the page holding the mapping's first byte.
#[cfg(target_os = "linux")]
fn mapped_first_page(mapping: &crate::host_rows::MappedRowsV1) -> bool {
    use std::io::{Read, Seek, SeekFrom};
    let address = mapping.bytes().as_ptr() as u64;
    let mut file = std::fs::File::open("/proc/self/pagemap").unwrap();
    file.seek(SeekFrom::Start(
        address / crate::host_rows::page_bytes_v1() as u64 * 8,
    ))
    .unwrap();
    let mut entry = [0; 8];
    file.read_exact(&mut entry).unwrap();
    u64::from_ne_bytes(entry) & (1 << 63) != 0
}

#[cfg(target_os = "linux")]
#[test]
fn default_dispatch_uses_descriptor_reads_and_zero_restores_mapping_access() {
    const CHILD: &str = "MAPPED_FETCH_DISPATCH_TEST";
    if let Some(mode) = std::env::var_os(CHILD) {
        let plane = plane("dispatch");
        let ids = [0, 17, 256, 0];
        let wanted = expected(&plane, &ids);
        plane
            .mapping
            .advise(crate::host_rows::RowAdviceV1::DontNeed)
            .unwrap();
        let present = || mapped_first_page(&plane.mapping);
        assert!(!present());
        let mut output = vec![0xa5; wanted.len()];
        gather_plane_v1(&plane, &input(&ids), &mut output).unwrap();
        assert_eq!(output, wanted);
        assert_eq!(
            present(),
            mode == "legacy",
            "dispatch must use the selected source"
        );
        return;
    }
    for mode in ["default", "legacy"] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args([
            "--exact",
            "device::mapped_gather::tests::default_dispatch_uses_descriptor_reads_and_zero_restores_mapping_access",
            "--nocapture",
        ]).env(CHILD, mode);
        if mode == "legacy" {
            command.env("CHOPINFER_MAPPED_MISS_FETCH", "0");
        } else {
            command.env_remove("CHOPINFER_MAPPED_MISS_FETCH");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{mode}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
    }
}

fn scales(label: &str) -> MappedGatherPlaneV1 {
    // A narrow plane of another, shorter file, as a codes plane has its
    // scales.
    let bytes: Vec<u8> = (0..(200 * 8 + 5)).map(|i| (i * 7 + i / 13) as u8).collect();
    let path = std::env::temp_dir().join(format!("async-scales-{label}-{}", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let mapping = crate::host_rows::map_rows_v1(&file, 5, 200 * 8).unwrap();
    std::fs::remove_file(path).unwrap();
    MappedGatherPlaneV1 {
        mapping,
        row_bytes: 8,
        index_offset: 8,
        stream: DeviceObjectV1::new(3),
        destination: DeviceRangeV1 {
            object: DeviceObjectV1::new(4),
            offset: 0,
            length: 8 * 80,
        },
    }
}

/// Four ranks' codes and scales over two shared tables, as one
/// tensor-parallel layer gather has them. The last plane reads its ids one
/// word later, so it cannot reuse a leader's rows.
fn ranks(label: &str) -> Vec<MappedGatherPlaneV1> {
    let (codes, scales) = (plane(&format!("{label}-codes")), scales(label));
    let mut planes = (0..4)
        .flat_map(|_| [codes.clone(), scales.clone()])
        .collect::<Vec<_>>();
    planes[7].index_offset = 12;
    planes
}

fn gather_ranks(planes: &[MappedGatherPlaneV1], ids: &[u32]) -> Result<Vec<Vec<u8>>, Vec<Vec<u8>>> {
    let rows = ids.len() - 1;
    let mut outputs = planes
        .iter()
        .map(|plane| vec![0xa5; rows * plane.row_bytes])
        .collect::<Vec<_>>();
    match gather_planes_v1(
        planes,
        &input(ids),
        &mut outputs
            .iter_mut()
            .map(Vec::as_mut_slice)
            .collect::<Vec<_>>(),
    ) {
        Ok(()) => Ok(outputs),
        Err(_) => Err(outputs),
    }
}

#[test]
fn every_rank_and_plane_of_a_gather_equals_its_plane_by_plane_bytes() {
    let planes = ranks("joint");
    for count in [1, 2, 53] {
        // One id more than the rows, for the plane that reads one word later.
        let ids: Vec<u32> = (0..=count).map(|i| (i * 37 % 200) as u32).collect();
        let outputs = gather_ranks(&planes, &ids).unwrap();
        for (index, (plane, output)) in planes.iter().zip(&outputs).enumerate() {
            let live = if index == 7 { &ids[1..] } else { &ids[..count] };
            assert_eq!(*output, expected(plane, live), "plane {index} rows {count}");
        }
    }
    assert!(
        gather_planes_v1(&planes, &input(&[1, 2]), &mut [&mut [0; 97][..]]).is_err(),
        "one output for eight planes"
    );
}

#[test]
fn a_bad_id_for_any_joint_plane_is_refused_before_every_write() {
    let planes = ranks("joint-invalid");
    let mut ids = vec![1; 54];
    // 250 is a codes row past the scales table, so gathering plane by plane
    // would write the first plane before refusing the second.
    for bad in [250, 257] {
        ids[52] = bad;
        let outputs = gather_ranks(&planes, &ids).unwrap_err();
        assert!(
            outputs.iter().flatten().all(|&byte| byte == 0xa5),
            "id {bad}"
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn joint_planes_read_both_tables_in_one_batch_and_zero_reads_them_alone() {
    const CHILD: &str = "MAPPED_JOINT_PLANES_TEST";
    if let Some(mode) = std::env::var_os(CHILD) {
        let planes = ranks("joint-count");
        let joint = || {
            crate::host_rows::fetch::snapshot()["joint_batches"]
                .as_u64()
                .unwrap()
        };
        let before = joint();
        let ids = [0, 17, 199, 0, 9];
        let outputs = gather_ranks(&planes, &ids).unwrap();
        for (index, (plane, output)) in planes.iter().zip(&outputs).enumerate() {
            let live = if index == 7 { &ids[1..] } else { &ids[..4] };
            assert_eq!(*output, expected(plane, live), "plane {index}");
        }
        // The rank-0 codes and scales planes lead: their misses are one
        // batch. The other planes gather on their own.
        assert_eq!(joint() - before, u64::from(mode == "joint"), "{mode:?}");
        return;
    }
    for mode in ["joint", "alone"] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "device::mapped_gather::tests::joint_planes_read_both_tables_in_one_batch_and_zero_reads_them_alone",
                "--nocapture",
            ])
            .env(CHILD, mode)
            .env_remove("CHOPINFER_MAPPED_MISS_FETCH")
            .env_remove("CHOPINFER_MAPPED_MISS_START");
        if mode == "alone" {
            command.env("CHOPINFER_MAPPED_MISS_JOINT_PLANES", "0");
        } else {
            command.env_remove("CHOPINFER_MAPPED_MISS_JOINT_PLANES");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{mode}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
    }
}

#[test]
fn plane_outputs_place_each_plane_in_its_stream_and_refuse_overlap() {
    let (mut first, mut second) = ([0_u8; 16], [0_u8; 8]);
    let mut outputs = [&mut first[..], &mut second[..]];
    // Planes in any order: two in each output, one leaving a gap.
    let places = [(1, 5), (0, 0), (0, 10), (1, 0)];
    let lengths = [3, 10, 6, 5];
    for (plane, slice) in plane_outputs(&mut outputs, &places, &lengths)
        .unwrap()
        .into_iter()
        .enumerate()
    {
        assert_eq!(slice.len(), lengths[plane]);
        slice.fill(plane as u8 + 1);
    }
    assert_eq!(first, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]);
    assert_eq!(second, [4, 4, 4, 4, 4, 1, 1, 1]);
    let mut outputs = [&mut first[..], &mut second[..]];
    for (places, lengths) in [
        (&[(0, 0), (0, 5)][..], &[10, 3][..]),
        (&[(0, 14)][..], &[5][..]),
        (&[(0, 17)][..], &[0][..]),
        (&[(2, 0)][..], &[1][..]),
        (&[(0, 0)][..], &[1, 1][..]),
    ] {
        assert!(
            plane_outputs(&mut outputs, places, lengths).is_err(),
            "{places:?} {lengths:?}"
        );
    }
}

/// The large-gather size: a gather of this many rows skips the resident cache.
const LARGE: usize = 65_536;

/// Ids for a large gather of [`ranks`], one more for its shifted plane.
fn large_ids() -> Vec<u32> {
    (0..=LARGE).map(|i| (i * 37 % 200) as u32).collect()
}

fn check_ranks(planes: &[MappedGatherPlaneV1], ids: &[u32], outputs: &[Vec<u8>]) {
    let rows = ids.len() - 1;
    for (index, (plane, output)) in planes.iter().zip(outputs).enumerate() {
        let live = if index == 7 { &ids[1..] } else { &ids[..rows] };
        assert!(*output == expected(plane, live), "plane {index}");
    }
}

#[test]
fn large_gathers_of_every_rank_and_plane_equal_their_mapped_bytes() {
    let planes = ranks("large");
    let ids = large_ids();
    check_ranks(&planes, &ids, &gather_ranks(&planes, &ids).unwrap());
    // One plane alone takes the same route.
    let mut output = vec![0xa5; LARGE * 97];
    gather_plane_v1(&planes[0], &input(&ids), &mut output).unwrap();
    assert!(output == expected(&planes[0], &ids[..LARGE]));
    // A codes row past the scales table is refused before any plane is
    // written.
    let mut bad = ids.clone();
    bad[60_000] = 250;
    let outputs = gather_ranks(&planes, &bad).unwrap_err();
    assert!(outputs.iter().flatten().all(|&byte| byte == 0xa5));
}

#[cfg(target_os = "linux")]
#[test]
fn large_gathers_read_through_descriptors_and_zero_restores_the_mapping() {
    const CHILD: &str = "MAPPED_LARGE_GATHER_TEST";
    if let Some(mode) = std::env::var_os(CHILD) {
        let planes = ranks("large-route");
        let codes = &planes[0].mapping;
        codes
            .advise(crate::host_rows::RowAdviceV1::DontNeed)
            .unwrap();
        assert!(!mapped_first_page(codes));
        let large = || {
            let snapshot = crate::host_rows::fetch::snapshot();
            let count = |key: &str| snapshot[key].as_u64().unwrap();
            (count("large_gathers"), count("large_rows"))
        };
        let before = large();
        let ids = large_ids();
        let outputs = gather_ranks(&planes, &ids).unwrap();
        let after = large();
        let counted = (after.0 - before.0, after.1 - before.1);
        // Checked before the expected bytes, which read the mapping.
        let touched = mapped_first_page(codes);
        if mode == "large" {
            // Rank 0's codes and scales are one batch and the shifted plane
            // another; the other ranks copy rank 0's bytes.
            assert_eq!(counted, (2, 3 * LARGE as u64));
            assert!(!touched, "large reads touched the mapping");
        } else {
            assert_eq!(counted, (0, 0));
            assert!(touched, "the mapped copy must read the mapping");
        }
        check_ranks(&planes, &ids, &outputs);
        return;
    }
    for mode in ["large", "mapped"] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "device::mapped_gather::tests::large_gathers_read_through_descriptors_and_zero_restores_the_mapping",
                "--nocapture",
            ])
            .env(CHILD, mode)
            .env_remove("CHOPINFER_MAPPED_MISS_FETCH")
            .env_remove("CHOPINFER_MAPPED_MISS_START")
            .env_remove("CHOPINFER_MAPPED_LARGE_GATHER_CACHE");
        if mode == "mapped" {
            command.env("CHOPINFER_MAPPED_LARGE_GATHER_FETCH", "0");
        } else {
            command.env_remove("CHOPINFER_MAPPED_LARGE_GATHER_FETCH");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{mode}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
    }
}
