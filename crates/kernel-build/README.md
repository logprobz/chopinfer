# kernel-build

`kernel-build` is a build-script library that generates versioned CUDA entry
wrappers, compiles native `sm_120f` cubins with CUDA 13.2, and records their
source closure and compiler identity. It does not execute GPU code. The
compiler comes from `CHOPINFER_CUDA_HOME` or `/usr/local/cuda` and must report
release 13.2; off Linux the build is skipped.

Every cubin is audited for target architecture, exact exported entries, no
embedded PTX or unresolved device functions, and per-entry register, stack,
local, and static shared memory usage. Default builds omit `-lineinfo`, so the
cubins contain no debug PTX sections and remain byte identical across checkouts.

Each consumer keeps its own sorted `kernels.lock`. Ordinary builds require an
exact match. Set `KERNELS_LOCK_UPDATE=1` to rewrite it after a deliberate
source or toolchain change, then review and commit the diff.

## Lock directories

`build_in_families` and `KernelSelection::build_in_families` take a `LockDir`
instead of a lock file. `LockDir::family(name, cubins)` declares a *source
family*: a name (ASCII letters, digits, `_`, `-`) and the cubin names it owns.
The families of one call must cover its specs exactly once. Each family has
two files in the directory:

- `<family>.lock` gates the build. Its records are the lock-file records
  without `source_closure_sha256`: name, `line_info`, toolchain version, nvcc
  hash, flags, defines, entry ABI, `cubin_sha256` and the audited per-entry
  resources (which derive from the cubin bytes).
- `<family>.provenance.json` lists each cubin's `source_closure_sha256`. It is
  reported, never gating: a difference passes with a Cargo warning naming the
  family and cubins, and the report's `lock_families[].provenance_changed`.

A build reads and, under `KERNELS_LOCK_UPDATE=1`, writes only its own
families' files (unchanged bytes are not rewritten) and never opens any other
file in the directory. A gating difference fails with the family, cubin and
field. The report then has `lock_path` set to the directory, `lock_sha256`
null and one `lock_families` row per family with its lock digest;
`verify_current` checks each family lock against the reported identities
without their source closure. Lock-file builds keep their exact behavior,
bytes and reports (`lock_families` is omitted when empty). The validation
reporter (`ci/kernel_reports.py`) applies the same rule: it checks each family
lock's digest and records, requires the families to cover the cubins exactly
once, and never reads provenance. A validation catalog may list a lock
directory or one family's lock file.

```rust
let lock = LockDir::new(manifest.join("kernels.lock.d"))
    .family("latent_attention", latent.iter().map(|s| &s.name))
    .family("delta_rule", delta.iter().map(|s| &s.name));
let specs: Vec<_> = latent.into_iter().chain(delta).collect();
let built = kernel_build::build_in_families(
    &toolchain, &specs, &out.join("families"), &lock, BuildOptions::from_env(), Some(&context))?;
```

For local profiling, set `CHOPINFER_KERNEL_LINEINFO=1`. This adds `-lineinfo`,
uses a `.profile.cubin` filename, and writes an identity sidecar with
`"line_info": true`. Profiling builds still check architecture, symbols,
unresolved functions, resource usage, and executable PTX images. They allow
nvcc's debug PTX sections and never compare or write `kernels.lock`, even when
`KERNELS_LOCK_UPDATE=1` is set. Release builds use `"line_info": false`.

## Shared cubin cache

`CHOPINFER_NVCC_WRAPPER` names an executable that kernel-build runs as
`<wrapper> <nvcc> <args...>` in each cubin's stage directory. Source-closure
discovery and the audits call nvcc directly. For each cubin compilation,
kernel-build also exports `CHOPINFER_NVCC_CACHE_CLOSURE` (the source-closure
digest), `CHOPINFER_NVCC_CACHE_TOOLCHAIN` and `CHOPINFER_NVCC_CACHE_LOOKUP`,
which is `0` for lock updates and line-info builds, whose results no lock
comparison checks. The toolchain digest covers the nvcc version, the bytes of
nvcc, `nvcc.profile`, every program `nvcc -dryrun` shows the compilation
running (host compiler, its `cc1plus`, `cicc`, `ptxas`), the shared libraries
`ldd` resolves for them, and libdevice; it is computed once per toolkit and
build-script process. A cubin gets no variables and compiles directly when its
generated unit, closure, defines or `NVCC_PREPEND_FLAGS`/`NVCC_APPEND_FLAGS`
name `__DATE__`, `__TIME__` or `__TIMESTAMP__`; when the environment sets a
setting the dry run reports that the key does not model (for example
`PTXAS_FLAGS`, `INCLUDES` or `NVVMIR_LIBRARY_DIR`; `MODELLED_SETTINGS` in
`src/cache_key.rs` lists the modelled ones), or `LD_PRELOAD` or `LD_AUDIT`; or
when the toolchain cannot be identified.

[`tools/nvcc_cache/nvcc_cache.py`](../../tools/nvcc_cache/nvcc_cache.py) is a
content-addressed cache behind that hook, shared by worktrees and fresh
targets:

```bash
export CHOPINFER_NVCC_WRAPPER="$PWD/tools/nvcc_cache/nvcc_cache.py"
export CHOPINFER_NVCC_CACHE_DIR="$CHOPINFER_DATA/nvcc-cache"
export CHOPINFER_NVCC_CACHE_MAX_BYTES=20G   # optional, default 10G
```

Point the wrapper at a checkout that outlives the builds, since temporary
worktrees (the geometry recipe) inherit it. An entry's key is the closure and
toolchain digests, nvcc's arguments with paths below the stage directory made
relative and the output path left out, every `NVCC_*` variable, and the host
preprocessor's `SOURCE_DATE_EPOCH`, `CPATH`, `C_INCLUDE_PATH`,
`CPLUS_INCLUDE_PATH`, `GCC_EXEC_PREFIX` and `COMPILER_PATH`. Flags, defines
and include directories change the key; the checkout path does not. Only
arguments whose whole meaning the key holds are cached (the wrapper's
`SAFE_ARGUMENT` list: architecture, standard, optimization and warning flags,
defines, include directories); an options file or any other argument, in the
command line or in `NVCC_PREPEND_FLAGS`/`NVCC_APPEND_FLAGS`, compiles without
the cache and logs `bypass`. Entries live in
`$CHOPINFER_NVCC_CACHE_DIR/v1/<2 hex>/<key>`, each a header line with the
cubin's SHA-256 and size followed by the cubin. They are written to a
temporary file and renamed, so concurrent builders are safe; a hit is checked
against its recorded SHA-256, and a corrupt or partial entry is a miss that the
compile replaces. When a store takes the cache above
`CHOPINFER_NVCC_CACHE_MAX_BYTES`, the least recently used entries (by
modification time, which a hit refreshes) are removed down to 90% of it; a
reader that already opened an evicted entry still reads all of it. Without the
cache directory or the key variables the wrapper runs nvcc unchanged.

A hit replaces only the nvcc call: the audits, digests and `kernels.lock`
comparison run on the result as on a fresh compile, so an entry that does not
match the lock fails it. The lock cannot catch a key that misses an input,
since the hit returns the cubin the lock was made from; that is why
indescribable compilations are not cached. Each call appends `hit`, `miss`,
`corrupt`, `refresh`, `bypass` or `store-error` with the compile seconds to
`v1/events.log`, which is renamed to `events.log.1` past 64 MiB
(`CHOPINFER_NVCC_CACHE_LOG` names another file, for example per build);
`tools/nvcc_cache/nvcc_cache.py --stats [LOG]` prints the counts and the
compile seconds saved. `python3 -B -m unittest discover -s tools/nvcc_cache`
tests the wrapper with a fake compiler; `tests/nvcc_cache.rs` checks real
builds whose options file, `nvcc.profile` or build date changes.

## Tuning tables

`kernel_build::tuning` reads and writes schedule tables (schema
`chopinfer-tuning-v1`): rows keyed by operation, weight format, named
rank-shape sizes and batch bucket, each with a schedule name and count or
flag knobs. Files are canonical (ascending keys, one row per line) and
`Table::parse` refuses any other form, so a rewritten table diffs by row.
The module interprets no knob; the owning catalog validates rows against
its kernel. `accepts` and `best` are the replacement rule offline sweeps
share: a candidate must beat the current row by more than its A-to-A noise
band plus a margin. The grouped-MoE template's rows live with their
owners: shared entries in `crates/kernels-sm120/tuning/schedules.json` and
each model's in `models/<name>/contract/tuning/schedules.json`.

## Model-owned construction and reports

`KernelSelection` composes existing `CubinSpec` values with one `EntryDeclaration`
per entry and a `BuildContext` for package/source/selection identity and source-root
mappings. It does not interpret model programs or mathematical operation enums.
`SelectedCubin::new` requires a versioned cubin name; `existing` preserves a legacy
name. Entry symbols remain versioned. Native declarations are assertions by their
owner, not proofs of CUDA correctness. `build` remains available, while
`build_with_context` adds ownership to legacy lists without changing their recipes.

The versioned `chopinfer-kernel-build-v1` report is written beside build outputs as
`kernel-build-report.json` (`kernel-build-report.profile.json` for profiling
builds). It does not change `CubinIdentity`, the closure hash algorithm or the
lock format. It contains:

- Consumer and selection identities, declared contracts, implementations and cases.
- A deduplicated table of files from the actual `nvcc -M` closure, per-file digests,
  byte counts, stable owner-relative identities and separate local locations.
- Each cubin's ordered closure keys, include directories, headers, defines, flags,
  compiler identity, generated digests, recipe digest and audited locked identity.
- Lock digest/status and separate membership, ownership and declaration completeness.

Root IDs are caller-supplied source namespaces. Map shared SDK sources to the same
namespace across consumers, even when their `sources_root` differs. Generated
files, toolkit inputs and system inputs remain distinct; toolkit identities depend
on the recorded build environment. Paths needed to reopen files are excluded from
portable recipe/file identities. Declarations do not enter legacy lock identities.

Cache hits recompute dependency membership. A failed build or lock check removes
that output's prior report. A different consumer cannot reuse its output directory.
Profiling has a separate report and status; skipped builds have incomplete coverage.
Missing context leaves ownership incomplete, and legacy catalogs leave entry
declarations incomplete even when all CUDA sources are known.

`BuildReport::verify_current` checks schema, verified lock status, current source and
compiler hashes, every cubin/sidecar, recipe digest and the original closure digest.
Callers separately require the completeness levels they need. An `Updated` candidate
is not a verified rebuild. This is build evidence, not authentication of metadata or
a qualification certificate. Reports contain local paths and belong outside Git.

Validation consumers join stable source identities to all recorded consumers and
union previous/current membership for deletions and changed dependencies. They keep
Cargo and explicit Rust/data edges and fall back conservatively for absent or stale
reports. This crate exports that information; validation selection policy belongs
to the validation reporter. Runtime execution, laboratory instrumentation and
serving qualification remain independent consumers of declarations.
