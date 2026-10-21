# qwen38

`compiler/` admits checkpoints and emits packages; `contract/` owns geometry,
precision, topology and frontend declarations; `bundle/` owns native execution.
Shared runtime code stays model-neutral.

- [Model inventory](model.toml)
- [Serving profiles](profiles.json)
- [Qualification declaration](qual.json)
- [Checkpoint pins](checkpoint-files.json)
- [Model integration](../../docs/adding-a-model.md)

## Serving profiles

The profile file owns settings, status and qualified concurrency. Read the
[qualification scope and exceptions](../../docs/serving-profiles.md) before
treating a compatible checkpoint or explicit override as qualified.
