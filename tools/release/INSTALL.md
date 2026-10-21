# Chopinfer alpha

Expect bugs and breaking formats, flags and packages. Requires Linux x86_64
with glibc 2.39 or newer. Serving is tested and supported only on NVIDIA RTX
PRO 6000 Blackwell 96 GB GPUs, two or four per model; other SM120 parts are
untested and may be refused. Models: MiMo V2.6 Flash, Qwen3.8 Flash and GLM
5.3 Flash. Weights and GPU qualification are not included.

1. Check the archive against the adjacent SHA256SUMS, extract it, then run
   `sha256sum --check SHA256SUMS` inside. Add `bin/` to PATH; use `--version`
   on either host binary to identify the build.
2. `chopinfer serve --model DIR` serves a supported checkpoint by loading it
   directly through the `bin/chopinfer-model-*` helpers and `lib/*.so`;
   `chopinfer verify --model DIR` checks its contents first. A MiMo copy
   without Hugging Face metadata needs the source lock: add `--manifest
   models/mimo26/checkpoint-files.json#mopd --allow-unrecorded`. To pin a
   package, pass the matching `lib/*.so` to `*-package --module` and serve it
   with `chopinfer serve --bundle PACKAGE --checkpoint DIR --profile NAME`.
   Libraries contain locked sm_120f cubins; they are not model packages.
3. Serving needs the real NVIDIA driver (`libcuda.so.1`), peer access between
   the selected GPUs, and sufficient device and host memory. Driver 595.84
   is the reference, not a qualified minimum.
   Follow [the preview guide](https://github.com/logprobz/chopinfer/blob/main/docs/developer-preview.md)
   at the tag matching BUILD_INFO for dependencies and installation steps.
4. `models/<model>/qual.json` and the files it pins (`fixtures/`,
   `tools/qual/modules.json`) qualify a deployment against the reference
   model: `chopinfer eval verify-references --model models/<model>` checks
   them offline, and `chopinfer eval run --model models/<model> --report DIR
   --launch --model CHECKPOINT` serves the checkpoint and runs the quick
   checks. See `docs/developer-preview.md` at the same tag.

[Report issues](https://github.com/logprobz/chopinfer/issues) with the version,
hardware and a minimal reproduction. Remove secrets and private inputs.
