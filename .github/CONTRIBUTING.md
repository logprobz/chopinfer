# Contributing

See [docs/contributing.md](../docs/contributing.md) for the workflow,
[AGENTS.md](../AGENTS.md) for contributor rules, and
[docs/contributing.md#setup](../docs/contributing.md#setup) for build setup.

## What is accepted

Pull requests for bug fixes, documentation fixes, checkpoint compatibility
and small, contained improvements are welcome. Larger changes (kernels, the
scheduler, numerics, new models, the bundle ABI or package formats) need an
issue first and the maintainer's agreement before work starts. Review is best
effort, and the maintainer may close any pull request, with or without review.

## Evidence

Run `bash ci/quick.sh` on the change, and `bash ci/cpu.sh` when its
prerequisites are in place, then attach the results to the pull request
([contribution tiers](../docs/contributing.md#contribution-tiers)).

Contributions are accepted under Apache-2.0 ([LICENSE](../LICENSE), section 5).
Follow the [code of conduct](CODE_OF_CONDUCT.md) and report vulnerabilities as
[SECURITY.md](../SECURITY.md) describes.
