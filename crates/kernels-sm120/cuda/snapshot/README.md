# Snapshot copies

Kernel-core contract: `snapshot_copy::SnapshotCopy`, whose oracle is
`snapshot_copy::snapshot_copy`. A prefix snapshot is a run of bytes laid
across serving-pool pages: snapshot byte `c` lives at
`pages[c / page_bytes] * page_bytes + c % page_bytes` of the pool. The
table one launch reads holds the snapshot's page IDs, then its segments,
each `length` bytes at `offset` of one of up to eight bound views, at
snapshot byte `at`. One launch per rank moves every segment a snapshot
table covers, in place of one device copy per piece and page fragment.

- `copy<Views>` (`copy.cuh`, grid `[pages * page_bytes / 16 / 256]`, 256
  threads) gives each thread one 16-byte vector of the snapshot's bytes.
  The thread finds the first segment that ends past its vector by binary
  search over the segments, which the host keeps in snapshot order and
  disjoint, and moves each segment's part of the vector. Pages hold whole
  vectors, so a vector lies in one page and needs one page lookup. A part
  that is one segment's whole vector, with both addresses 16-byte aligned,
  moves as one 16-byte access; every other part (an unaligned segment, a
  byte tail, a vector two segments share) moves byte by byte. Threads past
  the last segment, and over the header the host writes, return.

The launch is PDL: it releases dependents and waits before reading the
table, the views or the pages. It writes only the bytes the segments name,
each from one thread, so its result does not depend on scheduling. The host
checks the table (`SnapshotCopy::check`) and that the views, the pool's
snapshot pages and the table buffer do not overlap before it uploads or
launches anything (`bundle_runtime::snapshot::KernelCopy`); the kernel reads
the table's sizes from its arguments and trusts its contents.
