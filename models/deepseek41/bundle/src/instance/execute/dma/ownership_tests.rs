//! CPU model of the unchanged qualified peer-copy joins with private owners.
type Clock = [u64; 12];
fn before(a: Clock, b: Clock) -> bool {
    a.iter().zip(b).all(|(a, b)| *a <= b)
}
#[derive(Default)]
struct Gpu {
    clocks: [Clock; 12],
}
impl Gpu {
    fn step(&mut self, stream: usize) -> Clock {
        self.clocks[stream][stream] += 1;
        self.clocks[stream]
    }
    fn wait(&mut self, stream: usize, event: Clock) {
        for (a, b) in self.clocks[stream].iter_mut().zip(event) {
            *a = (*a).max(b);
        }
        self.step(stream);
    }
    fn copy(&mut self, stream: usize, source_owner: usize, destination_owner: usize) -> Clock {
        let done = self.step(stream);
        // Exactly the owner-stream conditions of the qualified DMA path.
        if source_owner != stream {
            self.wait(source_owner, done);
        }
        if destination_owner != stream && destination_owner != source_owner {
            self.wait(destination_owner, done);
        }
        done
    }
}
fn prepared(owner_base: usize) -> Gpu {
    let mut gpu = Gpu::default();
    for rank in 0..4 {
        let allocated = gpu.step(owner_base + rank);
        gpu.wait(rank, allocated);
        gpu.wait(4 + rank, allocated);
    }
    let producers = (0..4).map(|rank| gpu.step(rank)).collect::<Vec<_>>();
    for comm in 4..8 {
        for ready in &producers {
            gpu.wait(comm, *ready);
        }
    }
    gpu
}
fn permutation(gpu: &mut Gpu, owner_base: usize) -> Vec<Clock> {
    (0..4)
        .map(|destination| {
            let source = (destination + 3) % 4;
            gpu.copy(
                4 + destination,
                owner_base + source,
                owner_base + destination,
            )
        })
        .collect()
}
#[test]
fn compute_ownership_serializes_disjoint_b_rows() {
    let mut gpu = prepared(0);
    let copies = permutation(&mut gpu, 0);
    for (rank, copy) in copies.into_iter().enumerate() {
        assert!(before(copy, gpu.step(rank)));
    }
}
#[test]
fn private_owners_keep_b_independent_and_explicit_joins_order_reuse() {
    let mut gpu = prepared(8);
    let copies = permutation(&mut gpu, 8);
    for rank in 0..4 {
        let b = gpu.step(rank);
        assert!(copies.iter().all(|copy| !before(*copy, b)));
        for done in &copies {
            gpu.wait(rank, *done);
        }
        let reuse_a = gpu.step(rank);
        assert!(copies.iter().all(|copy| before(*copy, reuse_a)));
    }
}
#[test]
fn ownership_on_the_busy_communication_stream_adds_a_copy_dependency() {
    let mut gpu = prepared(4);
    let copies = permutation(&mut gpu, 4);
    assert!(before(copies[0], copies[3]));
    let mut gpu = prepared(8);
    let copies = permutation(&mut gpu, 8);
    assert!(!before(copies[0], copies[3]));
}
#[test]
fn private_allocation_must_complete_before_compute_and_copy_use() {
    let mut gpu = Gpu::default();
    let allocation = gpu.step(8);
    assert!(!before(allocation, gpu.step(0)));
    gpu.wait(0, allocation);
    gpu.wait(4, allocation);
    assert!(before(allocation, gpu.step(0)));
    assert!(before(allocation, gpu.step(4)));
}
#[test]
fn destruction_must_cover_final_compute_reads_as_well_as_dma() {
    let mut gpu = prepared(8);
    let copies = permutation(&mut gpu, 8);
    for copy in copies {
        gpu.wait(0, copy);
    }
    let final_compute = gpu.step(0);
    assert!(
        !before(final_compute, gpu.step(8)),
        "negative control: copy-only owner joins do not cover final compute"
    );
    gpu.wait(8, final_compute);
    assert!(before(final_compute, gpu.step(8)));
}
#[test]
fn relocation_adds_no_payload_and_staging_has_an_explicit_cost() {
    let mib = 1024_u64 * 1024;
    let rows = 8192;
    let hidden = 5120;
    let partial = rows * hidden * 4;
    let reduced = rows * hidden * 2;
    assert_eq!(partial, 160 * mib);
    assert_eq!(reduced, 80 * mib);
    let inbox = partial;
    assert_eq!(partial + 2 * reduced + inbox, 480 * mib);
    let extra_staging = partial + reduced;
    assert_eq!(extra_staging, 240 * mib);
    assert_eq!(extra_staging * 4, 960 * mib);
    assert_eq!(2 * extra_staging * 81, 40_768_634_880);
}
