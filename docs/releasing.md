# Alpha release checklist

Release artifacts are built and uploaded by a maintainer.

Use [`tools/qual/release_preview.sh`](../tools/qual/README.md#correctness-runner)
as the one validation entry point for a release of any model.

1. Choose a revision that passes `bash ci/cpu.sh` and the relevant
   [qualification gates](../tools/qual/README.md). CPU acceptance alone does
   not qualify GPU execution.
2. Set `workspace.package.version` in `Cargo.toml` and refresh `Cargo.lock`.
   Preserve ABI and schema versions. Tag the accepted revision with the
   matching version, for example `v0.1.0-alpha.2`.
3. Use the pinned CUDA builder in [deployment](deploy-common.md#build-the-image),
   or the [native toolchain](toolchain.md). In the devel container,
   `tools/release/setup-toolchain.sh` installs it as root. Set external
   `CARGO_HOME`, `RUSTUP_HOME` and `CARGO_TARGET_DIR` locations, add
   `$CARGO_HOME/bin` to PATH, then populate the locked dependency cache with
   `CARGO_NET_OFFLINE=false cargo fetch --locked`.
4. Run `bash tools/release/build.sh OUTPUT_DIRECTORY` without GPUs. The
   output directory must be fresh. Without Git metadata, set the full source
   SHA in `CHOPINFER_BUILD_REVISION`. Never regenerate kernel locks to hide
   a mismatch.
5. Check the archive and `SHA256SUMS`, then upload them by hand with the
   release notes. The archive contains binaries, native libraries, profiles,
   kernel locks, build identity, licenses and installation notes, not weights.

`python3 tools/release/notices.py OUTPUT_FILE` generates notices from the
locked Cargo graph and verified cached license files. Unknown license
expressions fail the checks. Both archives and images include the notices.
