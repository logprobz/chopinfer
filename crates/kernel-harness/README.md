# kernel-harness

The model-free GPU test harness for SM120 kernels. It names no model and
builds no cubin: a consumer passes what it tests as data.

- `gpu_test`, `gpu_test_multi` and `Session`: GPU selection
  (`CHOPINFER_KERNEL_GPU`, `CHOPINFER_KERNEL_GPUS`), sessions, uploads,
  graph capture, receipts and timing. `gpu_test_with` opens sessions that
  load a consumer's embedded cubins by name (`Cubins`); `Session::load_cubin`
  loads any bytes, such as a model's built cubins.
- `launch`: kernel-core launch plans and arguments over bound buffers.
- `poison`: poisoned relaunch, graph replay and PDL checks with write maps.
- `conformance`: the checks every entry owes, declared per buffer.
- `kit`: runs a backend's prepared ops by contract role against their CPU
  oracles (`kit::conform`). The case families select from the catalogs
  (`&kernels_sm120::Catalogs`) or entries a consumer passes; each family's
  `checks` hold its case self-checks over them. `host-sampling` adds the
  host's own sampling kernels (`kit::host`).
- `encoder`: each kernel entry a declared encoder stage selects, run from
  its serving cubin against the stage's component oracle. A consumer passes
  its declarations, encoder catalog and embedded cubins (`encoder::cases`)
  and a table of variant IDs and runners (`encoder::run`).

PDL checks and timing launch SM120's probe kernels (`probe_sm120`,
`probe_pdl_prefetch_sm120` and `poison::SM120_COPY_PRODUCER`), which a
consumer includes in its `Cubins`; without them those checks report
themselves as not exercised or fail with the missing cubin.

kernel-lab runs the in-tree models' cases on this crate. A new model runs
its kernel tests from its own test crate, as `examples/external-model/lab`
does. GPU tests run only on otherwise
idle GPUs.
