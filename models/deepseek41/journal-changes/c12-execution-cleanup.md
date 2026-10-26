# Execution cleanup

The attention-only prefill overlap, two-part micro schedule and explicit
full-schedule DMA executor are removed. Their opt-in trace assertions and
the overlap and micro cases in mixed-schedule fixtures are declared removals.
Upper-skip DMA and shared-expert decode overlap remain served.

At this cleanup's base, the remaining overlap and micro fixtures contain
no pinned `check_journal` calls. Full-schedule DMA selection and its dedicated
fixtures were already removed. Therefore the pinned-journal removal set is
empty. Every retained pinned journal must prove `IDENTICAL`, including its
complete compared history, with `--require identity`. The empty declaration
authorizes no resource, module, handle or operation edits and no pin updates.

The plan enum keeps skip plus prepared Engram composition. Ordinary lowering
and checked DMA subspans keep the same arguments. The executor scaffold keeps
the pipeline's per-submission poison policy and the existing Engram join
failure state. This host refactor changes no kernel or arithmetic and needs
no GPU evidence while those identity requirements hold.
