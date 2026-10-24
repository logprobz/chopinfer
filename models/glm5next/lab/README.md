# GLM kernel tests

The model's GPU fixtures use `kernel-harness` and the bundle's embedded
cubins. This crate is a workspace member, but not a default member, and
does not depend on `kernel-lab`.

On an idle GPU, run the retained-row copy fixture with:

```sh
CHOPINFER_KERNEL_GPU=0 CHOPINFER_KERNEL_REQUIRE=1 \
  cargo test --locked --offline -p model-glm5next-lab --test retained_rows \
  -- --nocapture --test-threads=1
```

It checks 52 ordered-copy cases against scalar copies, including strided
runs, contiguous DMA, overlapping destinations, and phase changes. Two
poison fills and eager and captured replays produce 312 byte comparisons
and 104 corruption controls. Without an idle GPU selection it skips;
`CHOPINFER_KERNEL_REQUIRE=1` makes a skipped check fail.
