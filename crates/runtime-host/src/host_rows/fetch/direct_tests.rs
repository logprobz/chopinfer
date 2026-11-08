use super::super::scripted::{GARBAGE, Kind, Scripted};
use super::*;
use std::collections::HashSet;

/// A table of `rows` rows of `width` bytes, after a header of `lead` bytes.
struct Table {
    path: std::path::PathBuf,
    mapping: Arc<crate::host_rows::MappedRowsV1>,
    bytes: Vec<u8>,
    width: usize,
    lead: u64,
}

impl Table {
    fn new(label: &str, rows: usize, width: usize, lead: usize) -> Self {
        let path =
            std::env::temp_dir().join(format!("mapped-direct-{label}-{}", std::process::id()));
        let bytes = (0..rows * width)
            .map(|i| ((i * 29) ^ (i / 251) ^ (i / 4099)) as u8)
            .collect::<Vec<_>>();
        let mut file_bytes = vec![0x5a; lead];
        file_bytes.extend(&bytes);
        std::fs::write(&path, file_bytes).unwrap();
        let file = File::open(&path).unwrap();
        let mapping =
            crate::host_rows::map_rows_v1(&file, lead as u64, bytes.len() as u64).unwrap();
        Self {
            path,
            mapping,
            bytes,
            width,
            lead: lead as u64,
        }
    }

    fn expected(&self, rows: &[u32]) -> Vec<u8> {
        rows.iter()
            .flat_map(|&row| self.bytes[row as usize * self.width..][..self.width].to_vec())
            .collect()
    }

    fn offset(&self, row: u32) -> u64 {
        self.lead + u64::from(row) * self.width as u64
    }

    /// `count` descriptions of the table's file.
    fn files(&self, count: usize) -> Vec<Arc<File>> {
        (0..count)
            .map(|_| Arc::new(File::open(&self.path).unwrap()))
            .collect()
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Two leaders, codes of an odd width with two twins and scales with one,
/// over rows with duplicates; the outputs start as garbage. The scales
/// start past the codes' last byte, so no two rows share a file offset.
struct Gather {
    codes: Table,
    scales: Table,
    rows: Vec<u32>,
}

impl Gather {
    fn new(label: &str, positions: usize) -> Self {
        Self {
            codes: Table::new(&format!("{label}-codes"), 3000, 97, 300),
            scales: Table::new(&format!("{label}-scales"), 3000, 8, 300_000),
            rows: (0..positions as u32)
                .map(|i| i.wrapping_mul(2_654_435_761) % 3000)
                .collect(),
        }
    }

    /// Runs [`copy_with`] and answers its result and every output: each
    /// leader followed by its twins.
    fn run(
        &self,
        workers: usize,
        files: usize,
        source: &dyn RowSource,
        spawn: &dyn Fn(usize) -> bool,
    ) -> (Result<(), String>, Vec<Vec<u8>>) {
        let n = self.rows.len();
        let mut outputs = [97, 97, 97, 8, 8].map(|width| vec![GARBAGE; n * width]);
        let [codes, codes_a, codes_b, scales, scales_a] = &mut outputs;
        let mut planes = [
            PlaneCopy {
                mapping: &self.codes.mapping,
                width: 97,
                rows: &self.rows,
                out: codes,
            },
            PlaneCopy {
                mapping: &self.scales.mapping,
                width: 8,
                rows: &self.rows,
                out: scales,
            },
        ];
        let mut twins = [
            vec![&mut codes_a[..], &mut codes_b[..]],
            vec![&mut scales_a[..]],
        ];
        let files = [self.codes.files(files), self.scales.files(files)];
        let result = copy_with(&mut planes, &mut twins, &files, workers, source, spawn);
        let [codes, codes_a, codes_b, scales, scales_a] = outputs;
        (result, vec![codes, codes_a, codes_b, scales, scales_a])
    }

    fn check(&self, outputs: &[Vec<u8>], label: &str) {
        let (codes, scales) = (
            self.codes.expected(&self.rows),
            self.scales.expected(&self.rows),
        );
        for (index, output) in outputs.iter().enumerate() {
            let expected = if index < 3 { &codes } else { &scales };
            assert!(output == expected, "{label}: output {index}");
        }
    }

    /// Every probe class but full, in turn, on every third position's row.
    fn cold_script(&self) -> Scripted {
        let classes = [
            Probe::Short,
            Probe::WouldBlock,
            Probe::Unsupported,
            Probe::Interrupted,
            Probe::Other,
        ];
        let mut scripted = Scripted::default();
        for (i, &row) in self.rows.iter().enumerate().step_by(3) {
            let class = classes[i / 3 % classes.len()];
            scripted.classes.insert(self.codes.offset(row), class);
            scripted.classes.insert(self.scales.offset(row), class);
        }
        scripted
    }
}

fn every(_: usize) -> bool {
    true
}

fn worker_counts() -> [usize; 4] {
    [
        1,
        2,
        7,
        policy(std::thread::available_parallelism().map_or(1, usize::from)),
    ]
}

#[test]
fn the_policy_leaves_two_threads_and_runs_one_to_24_workers() {
    for (threads, workers) in [
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (8, 6),
        (24, 22),
        (26, 24),
        (48, 24),
    ] {
        assert_eq!(policy(threads), workers, "{threads}");
    }
}

#[test]
fn direct_reads_write_every_leader_and_twin_exactly_on_any_workers() {
    let gather = Gather::new("exact", 9000);
    for workers in worker_counts() {
        for files in [1, 3, 24] {
            let (result, outputs) = gather.run(workers, files, &Descriptor, &every);
            result.unwrap();
            gather.check(&outputs, &format!("workers {workers} files {files}"));
        }
    }
}

#[test]
fn every_cold_row_is_advised_before_any_worker_blocks_and_ends_whole() {
    let gather = Gather::new("schedule", 5000);
    for workers in worker_counts() {
        let scripted = gather.cold_script();
        let (result, outputs) = gather.run(workers, 3, &scripted, &every);
        result.unwrap();
        // Garbage from short and failed probes never survives, in a leader
        // or a twin.
        gather.check(&outputs, &format!("workers {workers}"));
        let calls = scripted.calls();
        let seqs = |kind: Kind| {
            calls
                .iter()
                .enumerate()
                .filter(|(_, call)| call.kind == kind)
                .map(|(seq, _)| seq)
                .collect::<Vec<_>>()
        };
        let (advice, reads) = (seqs(Kind::Advise), seqs(Kind::Read));
        assert!(
            advice.iter().max() < reads.iter().min(),
            "workers {workers}"
        );
        // Each cold position is advised and read whole once, nothing else.
        let cold = |kind: Kind| {
            let mut offsets = calls
                .iter()
                .filter(|call| call.kind == kind)
                .map(|call| (call.offset, call.length))
                .collect::<Vec<_>>();
            offsets.sort_unstable();
            offsets
        };
        // Rows repeat: a cold row is cold at every position it holds.
        let mut wanted = gather
            .rows
            .iter()
            .filter(|&&row| scripted.classes.contains_key(&gather.codes.offset(row)))
            .flat_map(|&row| {
                [
                    (gather.codes.offset(row), 97),
                    (gather.scales.offset(row), 8),
                ]
            })
            .collect::<Vec<_>>();
        wanted.sort_unstable();
        assert_eq!(cold(Kind::Advise), wanted, "workers {workers}");
        assert_eq!(cold(Kind::Read), wanted, "workers {workers}");
        assert_eq!(scripted.in_flight(), 0);
    }
}

#[test]
fn a_source_that_never_reads_without_waiting_reads_every_row_whole() {
    let gather = Gather::new("unsupported", 3000);
    let scripted = Scripted {
        otherwise: Some(Probe::Unsupported),
        ..Scripted::default()
    };
    let (result, outputs) = gather.run(5, 2, &scripted, &every);
    result.unwrap();
    gather.check(&outputs, "unsupported");
    let calls = scripted.calls();
    let count = |kind: Kind| calls.iter().filter(|call| call.kind == kind).count();
    let rows = 2 * gather.rows.len();
    assert_eq!(
        (count(Kind::Probe), count(Kind::Advise), count(Kind::Read)),
        (rows, rows, rows)
    );
}

#[test]
fn workers_probe_disjoint_chunks_that_partition_every_leader() {
    let gather = Gather::new("partition", 7000);
    for workers in worker_counts() {
        let scripted = Scripted::default();
        let (result, outputs) = gather.run(workers, 2, &scripted, &every);
        result.unwrap();
        gather.check(&outputs, &format!("workers {workers}"));
        let calls = scripted.calls();
        for (width, base) in [(97, gather.codes.offset(0)), (8, gather.scales.offset(0))] {
            let mut probes = calls
                .iter()
                .filter(|call| call.kind == Kind::Probe && call.length == width)
                .collect::<Vec<_>>();
            assert!(probes.iter().all(|call| call.offset >= base));
            probes.sort_unstable_by_key(|call| call.at);
            // One probe per position, end to end over the output.
            assert_eq!(probes.len(), gather.rows.len());
            assert!(
                probes
                    .windows(2)
                    .all(|pair| pair[0].at + width == pair[1].at)
            );
            // Each chunk of positions is one worker's.
            for chunk in probes.chunks(CHUNK_ROWS) {
                assert!(chunk.iter().all(|call| call.thread == chunk[0].thread));
            }
        }
        let threads = calls.iter().map(|call| call.thread).collect::<HashSet<_>>();
        assert!(threads.len() <= workers, "workers {workers}");
    }
}

/// A failed gather has finished every worker: no call is in flight when it
/// returns, and none begins later.
fn finished(scripted: &Scripted) {
    assert_eq!(scripted.in_flight(), 0);
    let calls = scripted.calls().len();
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(scripted.calls().len(), calls);
}

#[test]
fn a_whole_read_failure_after_writes_fails_the_gather_after_every_worker_joined() {
    let gather = Gather::new("failure", 6000);
    for workers in worker_counts() {
        let scripted = Scripted {
            fail_after: Some(40),
            ..gather.cold_script()
        };
        let (result, _) = gather.run(workers, 3, &scripted, &every);
        assert_eq!(
            result.unwrap_err(),
            "scripted read failure",
            "workers {workers}"
        );
        finished(&scripted);
    }
}

#[test]
fn a_worker_panic_becomes_the_gathers_error() {
    let gather = Gather::new("panic", 6000);
    // A probe late in the gather, so some workers are reading when it panics.
    let late = gather.codes.offset(gather.rows[4321]);
    for workers in worker_counts() {
        let scripted = Scripted {
            panic_at: Some(late),
            ..gather.cold_script()
        };
        let (result, _) = gather.run(workers, 3, &scripted, &every);
        assert!(
            result.unwrap_err().contains("panicked"),
            "workers {workers}"
        );
        finished(&scripted);
    }
}

#[test]
fn workers_that_cannot_start_leave_exact_bytes_to_the_others() {
    let gather = Gather::new("spawn", 6000);
    let current = std::thread::current().id();
    for (label, spawn, most) in [
        ("none start", &(|_| false) as &dyn Fn(usize) -> bool, 1),
        ("odd workers fail", &|worker| worker % 2 == 0, 4),
    ] {
        let scripted = gather.cold_script();
        let (result, outputs) = gather.run(7, 3, &scripted, spawn);
        result.unwrap();
        gather.check(&outputs, label);
        let threads = scripted
            .calls()
            .iter()
            .map(|call| call.thread)
            .collect::<HashSet<_>>();
        assert!(threads.len() <= most, "{label}");
        assert!(
            threads.contains(&current),
            "{label}: the caller is worker 0"
        );
    }
}

#[test]
fn twins_of_another_length_are_refused_before_any_read() {
    let table = Table::new("twin-length", 100, 8, 300);
    let rows = vec![3; 9000];
    let mut out = vec![GARBAGE; rows.len() * 8];
    let mut short = vec![GARBAGE; rows.len() * 8 - 8];
    let route = super::super::LargeRoute {
        reader_files: 0,
        workers: 2,
    };
    let result = super::super::copy_large_twinned(
        &mut [PlaneCopy {
            mapping: &table.mapping,
            width: 8,
            rows: &rows,
            out: &mut out,
        }],
        &mut [vec![&mut short[..]]],
        route,
    );
    assert!(result.is_err());
    assert!(out.iter().chain(&short).all(|&byte| byte == GARBAGE));
}
