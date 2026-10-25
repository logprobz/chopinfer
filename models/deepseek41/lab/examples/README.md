# Family builds

`build_family` replaces the individual cubin builders. Run it through the
shared heavy-command lane, with output outside the repository:

```sh
tools/dev/heavy.sh cargo run -p model-deepseek41-lab --example build_family \
  --locked --offline -- --list
tools/dev/heavy.sh cargo run -p model-deepseek41-lab --example build_family \
  --locked --offline -- lab/exact_moe_prefill_vector "$OUT"
```

The first argument names a lock root and family. `bundle/` and `lab/` can
hold different recipes with the same family name. The table records each
recipe's ordered specs, dependencies and lock destinations. It retains the
comparison cubins, qualification module inventories and fold baseline
subdirectories that the individual builders produced.

Existing locks are verified. `KERNELS_LOCK_UPDATE=1` can initialize an absent
family only; it cannot replace an existing lock. Dependency locks are always
verified. The query-prefill control and router64 recipes keep their locks
under the external output directory and initialize them only when absent.
Profiling line information and skipped CUDA compilation are refused. The
routed-MoE floor recipe retains `CHOPINFER_MOE_FLOOR_LOCK_DIR` when set.

The exact-prefill scalar and vector recipes each have `lab/` and `bundle/`
entries. These replace the old implicit `sm120` example's scalar/vector and
lock-root flags. `family-build.json` records every built module, its lock,
SHA-256, symbols and resources. It is an artifact inventory, not GPU evidence.

`prepare_moe_wide` remains a separate example because it prepares packed
weight banks for the retained MoE tests.
