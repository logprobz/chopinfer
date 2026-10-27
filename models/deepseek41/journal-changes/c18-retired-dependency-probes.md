# Retire packaged dependency probes

The package no longer contains `dependency_probes_sm120`. Serving never
loaded this module, so no resolve, launch or teardown operation changes.
The declaration names exactly this package removal and is read at the head
by the journal proof.

Model-lab fixtures now use the existing `shared/dependency_probes` family.
Its kernel-lab lock and provenance are byte-identical to the retired bundle
pair, and the standalone family build reproduces the locked cubin. The
fixture loader checks the digest and both required exports without linking
kernel-lab. CPU tests can build the locked module; GPU jobs must stage it
before execution and set `CHOPINFER_DEEPSEEK41_DEPENDENCY_PROBES` to that
directory. The fold comparison still authenticates every required staged
module and rejects missing or altered timing probes.
