# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:44a9504c6dfb50b1241464241b02a93871928f373de6f5a644cf5fe9f080aa63
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.2.1-base-ubuntu24.04@sha256:7be56e69d8ae7c3648b8fca009fa35980ecd9c6eefaafbff3ee8c224e0043eb5
ARG BUILD_IMAGE=toolchain
ARG RUNTIME_IMAGE=runtime-tools

FROM ${CUDA_DEVEL_IMAGE} AS toolchain
# kernels.lock pins the exact nvcc build (13.2.86), which NVIDIA publishes as
# packages but not as a 13.2.2 image, so the compiler packages are pinned here.
ARG CUDA_COMPILER_VERSION=13.2.86-1
RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
    build-essential ca-certificates curl python3 \
    cuda-nvcc-13-2=${CUDA_COMPILER_VERSION} cuda-crt-13-2=${CUDA_COMPILER_VERSION} \
    cuda-cccl-13-2=${CUDA_COMPILER_VERSION} cuda-cudart-13-2=${CUDA_COMPILER_VERSION} \
    cuda-cudart-dev-13-2=${CUDA_COMPILER_VERSION} cuda-driver-dev-13-2=${CUDA_COMPILER_VERSION} \
    cuda-cuobjdump-13-2=${CUDA_COMPILER_VERSION} cuda-nvdisasm-13-2=${CUDA_COMPILER_VERSION} \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo-home
ENV PATH=/opt/cargo-home/bin:$PATH
COPY rust-toolchain.toml /tmp/rust-toolchain.toml
COPY tools/release/install-rust.sh /tmp/install-rust.sh
RUN rust_version=$(sed -n 's/^channel = "\([^"]*\)"/\1/p' /tmp/rust-toolchain.toml) && \
    test -n "$rust_version" && \
    bash /tmp/install-rust.sh "$rust_version" && \
    sha256sum /tmp/install-rust.sh > /opt/rustup-installer.sha256 && rm /tmp/install-rust.sh

# BuildKit --build-context cargo-cache=DIR can replace this empty stage with
# a prepared Cargo home containing registry/ and any required source config.
FROM scratch AS cargo-cache
COPY deploy/docker/empty-cargo-home/ /

FROM ${BUILD_IMAGE} AS build
ARG CARGO_OFFLINE=false
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo-home
ENV PATH=/opt/cargo-home/bin:$PATH
WORKDIR /src
COPY . .
ARG CUDA_DEVEL_IMAGE
ARG CUDA_RUNTIME_IMAGE
ARG BUILD_IMAGE
ARG RUNTIME_IMAGE
ARG BUILDKIT_SYNTAX=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
ARG RUST_TOOLCHAIN
ARG CARGO_LOCK_SHA256
ARG CHOPINFER_BUILD_REVISION
ARG CUDA_COMPILER_VERSION=13.2.86-1
RUN python3 deploy/docker/build.py --check && \
    mkdir -p /opt/build-inputs && \
    python3 deploy/docker/build.py --revision > /opt/build-inputs/revision && \
    rustc --version --verbose > /opt/build-inputs/rustc.txt && \
    cargo --version > /opt/build-inputs/cargo.txt && \
    nvcc --version > /opt/build-inputs/nvcc.txt && \
    dpkg-query -W > /opt/build-inputs/build-packages.txt && \
    cp /opt/rustup-installer.sha256 /opt/build-inputs/ && \
    sha256sum Cargo.lock > /opt/build-inputs/Cargo.lock.sha256
# The host binary links libcuda.so.1 verbatim. The build image has no driver,
# so the toolkit's link stub is exposed under that name for linking only; the
# NVIDIA container runtime provides the real driver library at run time.
RUN mkdir -p /opt/cuda-link-stub && \
    ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /opt/cuda-link-stub/libcuda.so.1
ENV RUSTFLAGS="-L native=/opt/cuda-link-stub"
RUN --mount=type=bind,from=cargo-cache,target=/cargo-cache \
    cp -a /cargo-cache/. "$CARGO_HOME/" && \
    case "$CARGO_OFFLINE" in true) offline=--offline ;; false) offline= ;; *) exit 2 ;; esac && \
    cargo fetch --locked $offline && \
    export CHOPINFER_BUILD_REVISION=$(cat /opt/build-inputs/revision) && \
    CARGO_TARGET_DIR=/src/target bash tools/release/build.sh /opt/archive && \
    mkdir /opt/release && tar -xzf /opt/archive/*.tar.gz -C /opt/release --strip-components=1 && \
    python3 -B tools/release/plan.py --view libraries > /tmp/release-libraries && \
    ldd /opt/release/bin/chopinfer && \
    while IFS= read -r library; do ldd "/opt/release/$library" || exit; done < /tmp/release-libraries


FROM ${CUDA_RUNTIME_IMAGE} AS runtime-tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgcc-s1 python3 && rm -rf /var/lib/apt/lists/*

FROM ${RUNTIME_IMAGE} AS runtime
ARG CUDA_DEVEL_IMAGE
ARG CUDA_RUNTIME_IMAGE
ARG BUILD_IMAGE
ARG RUNTIME_IMAGE
ARG BUILDKIT_SYNTAX=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
ARG CUDA_COMPILER_VERSION=13.2.86-1
LABEL org.chopinfer.build.cuda-devel-image=$CUDA_DEVEL_IMAGE \
      org.chopinfer.build.cuda-runtime-image=$CUDA_RUNTIME_IMAGE \
      org.chopinfer.build.build-image=$BUILD_IMAGE \
      org.chopinfer.build.runtime-image=$RUNTIME_IMAGE \
      org.chopinfer.build.frontend-image=$BUILDKIT_SYNTAX \
      org.chopinfer.build.cuda-compiler=$CUDA_COMPILER_VERSION
COPY --from=build /opt/release/THIRD_PARTY_NOTICES /usr/share/doc/chopinfer/THIRD_PARTY_NOTICES
COPY LICENSE /usr/share/doc/chopinfer/LICENSE
COPY --from=build /opt/build-inputs/ /usr/share/chopinfer/build-inputs/
RUN dpkg-query -W > /usr/share/chopinfer/build-inputs/runtime-packages.txt
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
# The checkpoint is read-only; named volumes inherit this cache's ownership.
ENV CHOPINFER_MODEL=/data/model CHOPINFER_MODEL_CACHE=/data/model-cache
COPY --from=build /opt/release/ /opt/chopinfer/
RUN useradd --uid 10001 --create-home chopinfer && \
    mkdir -p /data/model-cache && chown chopinfer:chopinfer /data/model-cache && \
    chmod 700 /data/model-cache && chmod -R go-w /opt/chopinfer
COPY deploy/docker/deployment.py /opt/chopinfer/bin/deployment.py
COPY --chmod=755 deploy/docker/entrypoint.sh /opt/chopinfer/bin/entrypoint.sh
ENV PATH=/opt/chopinfer/bin:$PATH
USER chopinfer
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=10s --timeout=3s --start-period=600s --retries=3 \
    CMD python3 /opt/chopinfer/bin/deployment.py healthcheck
ENTRYPOINT ["/opt/chopinfer/bin/entrypoint.sh"]
