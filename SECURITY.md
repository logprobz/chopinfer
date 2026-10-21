# Security policy

Only the latest alpha tag receives security fixes.

Report suspected vulnerabilities privately through GitHub's private
vulnerability reporting (the repository's Security tab, "Report a
vulnerability"). Do not include exploit details in a public issue. Include the
affected revision, reproduction steps, and impact.

An operator selects a local package, directly (`serve --bundle DIR`) or
through the helper that prepares it (`serve --model DIR`). Verification checks
its manifest, closed file set, component digests, and ABI version before
loading native code, and serves the component bytes it hashed. A loaded
native module runs in the host process and is trusted code, not a sandbox.

## Installed preparation providers

`serve --model` discovers regular `chopinfer-model-*` executables only beside
the canonical host executable, in a directory that is not group or world
writable. It rejects helper symlinks and unsafe files, bounds discovery and
protocol records, and searches neither PATH, the checkpoint nor the current
directory. Providers are trusted installed code; checkpoints supply data, not
executables. The helper performs compatibility and integrity checks and hands
a leased canonical package to the runtime. Artifact integrity does not
establish publisher trust or model quality. See
[compatible checkpoints](docs/developer-preview.md#checkpoints) for cache and freshness
rules.

## Chat frontend

A package's chat format is code in its native module, exported as
`chop.frontend.v1`, so it is trusted bundle code like the rest of the
module. The frontend declaration and tokenizer files it reads are
authenticated data: the host opens the frontend over the component bytes it
hashed, and tokenizes with its own reader under the declared component
policy. Tokenizer assets are never executable code from a model repository,
and the host never runs a checkpoint's chat template.

Trusted is not unchecked. The host refuses client text that contains a media
placeholder before render, bounds every frontend answer before copying it,
writes media placeholders itself, and compiles a returned grammar only under
its own budgets. The frontend parser sees only bytes that passed the host's
stop filter and end on a complete UTF-8 character, and the host fails the
request when the parser's ranges, tool calls or finish break the contract
([frontend export](BUNDLE_ARCHITECTURE.md#frontend-export)). Stop strings,
UTF-8 handling and limits stay with the host whatever the format does.

## ABI and files

The C ABI carries fixed-width records, bounded byte views, and host-issued
handles. The host resolves device addresses; a bundle does not receive a raw
device address as a persistent contract. Unknown versions and unsupported
operation contracts fail closed. See
[BUNDLE_ARCHITECTURE.md](BUNDLE_ARCHITECTURE.md).

`runtime-weights` validates local file identity, sizes, tensor metadata,
offsets, and checked reads, and retains the read-only descriptors used for
authentication, so checked positional reads stay bound to those descriptors
after a path rename. Keep no-follow path checks when adapting paths. Keep
authenticated files immutable: retaining a descriptor cannot prevent
same-size in-place writes. The checkpoint identity covers safetensors headers
and sizes, not payload bytes ([direct loading](BUNDLE_ARCHITECTURE.md#direct-weight-loading));
`chopinfer verify --model DIR` checks content. SHA-256 uses the locked `sha2`
backend; CPU feature selection must preserve the digest and all tensor-range
checks.
