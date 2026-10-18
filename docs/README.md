# Documentation

## Try Chopinfer

- [Quickstart: checkpoint to first response](developer-preview.md), including troubleshooting and how to report a problem
- [Compatible checkpoints](compatible-checkpoints.md): `serve --model`, `verify --model` and the model cache
- [Quantized checkpoints and activation policies](quantized-checkpoints.md)
- [Operator commands](cli.md)
- [Serving profiles and alpha limits](serving-profiles.md)
- [Serving request surface](serving-request-surface.md): endpoints, parameters, refusals and queueing
- [Serving MiMo V2.6 Flash](serving-mimo26.md)

## Deploy and operate

- [Common deployment operations](deploy-common.md)
- [Deploy MiMo V2.6 Flash](deploy-mimo26.md)
- [Deploy Qwen3.8 Flash](deploy-qwen38.md)
- [Deploy GLM 5.3 Flash](deploy-glm5next.md)
- [Metrics, request traces and step profiles](operations-observability.md), and the live view [chopinfer-top](operations-observability.md#watch-running-instances-with-chopinfer-top)
- [Adaptive speculation](adaptive-speculation.md)
- [Mixed prefill and decode scheduling](mixed-scheduling.md)
- [MiMo V2.6 Flash at TP2](topology-mimo.md)
- [Qwen3.8 Flash at TP2](topology-qwen.md)
- [GLM 5.3 Flash at TP4](topology-glm5next.md)

## Contribute

- [Contributing](contributing.md), [toolchain](toolchain.md), [testing](testing.md) and the [development loop](development-loop.md)
- [Performance](performance.md): measure, profile and submit a speed change, and the [contribution tiers](contributing.md#contribution-tiers)
- [Landing a change](development-loop.md#landing-a-change) (pull requests)
- [Architecture](architecture.md) and [bundle architecture](../BUNDLE_ARCHITECTURE.md)
- [Adding a model](adding-a-model.md) and [models maintained outside core](external-models.md)
- [Direct weight loading](direct-loading.md)
- [Shared bundle runtime](bundle-runtime.md)
- [Host-managed serving KV pool](serving-kv-pool.md) and [per-request recurrent state](serving-recurrent-state.md)
- [Context-parallel attention](context-parallel.md): groups, striped pages, kernels and how a bundle opts in
- [Qwen3.8 Flash bundle](bundle-qwen38.md) and its [weight plan and package](model-qwen38.md)
- [GLM 5.3 Flash: model, bundle and status](model-glm5next.md)
- [Kernel library](kernel-library.md), including [adding a GPU backend](kernel-library.md#adding-a-gpu-backend)
- [Trained block quantization and row status](block-quantization.md)
- [Latent glue row status](latent-glue-status.md)
- [Security boundary](security-boundary.md)
- [Alpha release checklist](releasing.md)

## Qualify and debug

- [Evidence tools](testing.md#evidence-tools): [journal proofs](../tools/journal_proof/README.md), [serving smoke](../tools/smoke/README.md) and [serving A/B](../tools/ab/serving/README.md)
- [Qualification tools](../tools/qual/README.md)
- [Paired release accuracy](paired-accuracy.md)
- [Check kinds: the qualification contract](check-kinds.md)
- [Numerics](numerics.md), including the [engine rules](numerics.md#engine-rules)
- [GPU reference runner, cache and taps](reference-runner.md)
- [Diagnose the first divergence](debugging.md)
- [Qwen3.8 Flash dump checks](qwen38-oracle.md)
