# Host ranges

The optional host observer annotates existing engine scopes, mapped-row
gathers, uploads, kernel and collective submission, waits, sampling, result
collection and commit. It adds no device operation. The engine and bundle
callbacks run on the `bundle-engine` thread; these are nested scopes on that
thread, not two independent workers. Bundles may send bounded range notes
through the existing log callback. The instrumented bundle annotates prefill and
decode/verify planning, upload, submission, status, result emission, draft
readback and commit. Unknown or unbalanced notes remain visible in the log.

Build `nvtx_ranges.c` as a shared library against the installed NVTX headers,
outside the repository. Set `LD_PRELOAD` to that helper and
`CHOPINFER_NVTX_RANGES=1` for the profiled server. A missing helper refuses
startup. Ordinary serving does not resolve the helper or install an observer.
Keep `CHOPINFER_STEP_TRACE=1` for the existing traced scopes. Collect
`cuda,nvtx,osrt` in a bounded Nsight range after the exact request and seed
warm-up. CPU samples and context switches require a successful environment
preflight; report their absence when unavailable.

Intersect ranges with the complement of GPU activity. Inclusive host range
times overlap GPU execution and each other, so they must not be added or
subtracted directly from a step. Per-rank inactivity can overlap work on a
peer GPU. Retain the all-rank idle intersection separately. CPU samples
locate work inside unannotated bundle planning and submission; missing
samples leave that time unresolved. Per-kernel annotations add overhead,
so a profile is attribution evidence, not served timing acceptance.

This diagnostic does not admit any kernel route or change a model option.
Keep the kernel locks unchanged and compare the annotated bundle's default
journals against the unannotated bundle before profiling it.
