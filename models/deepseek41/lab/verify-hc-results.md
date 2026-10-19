# Verify-row HC fusion

At six and 32 verify rows, the four-warp, 160-CTA rounded HC producer and
the 1024-thread finish/norm tail passed bitwise comparison against the
unfused chain on both GPUs at 250 W on 2026-10-19. Each GPU executed
104 base cases, 48 byte and eight arithmetic negative controls, and 24
concurrent scratch cases. Poisoned outputs, replay, live-row guards,
nonfinite inputs and sticky quantizer status are covered.

The fastest full-chain event times, in microseconds per call, span the two
GPUs. Measurements use batches of 32 graph calls.

| Rows | Widen | Original | Fused |
|---:|:---:|---:|---:|
| 6 | no | 31.104-31.551 | 17.088-17.280 |
| 6 | yes | 32.768-33.472 | 20.352-20.736 |
| 32 | no | 40.448-41.344 | 27.904-28.544 |
| 32 | yes | 41.920-43.329 | 31.168-31.808 |

Hot and cold exclusive profiles select the same fused tail in every case.
All four captures contain 12,288 graph launches and 3,584 complete chain
calls; eight warmups per case and variant are excluded. Rounding and
widening pad six live rows to eight CTAs. The analyzer decodes that padding
only for those entries and independently checks each subsequent row count.

Serving selects this pair at 77 eligible Flash sublayer boundaries. The
three initial or Engram-separated mixers use the qualified warp mixer and
retain their original norm chain. Verify eight and 16 rows, prefill and
other geometries keep their previous routes. The separate producer and
norm selectors are unchanged, including Decode-32: its four-warp producer
without the fused tail regressed against the existing rounded pair.

The TP4 native-KV journal drops from 2,167 to 1,889 launches per rank at
six rows, and from 2,680 to 2,402 at 32 rows. Each removes 278 launches:
30 plain five-launch chains and 47 widened six-launch chains become two
launches each. The operand audit checks every producer input, HC result,
norm output, live-row pointer, epsilon and status argument against the old
chain, and requires every unrelated launch and its order to be unchanged.
Eight- and 16-row journals are byte-identical. The scratch audit rejects
overlapping writes and finds none in the active verify fork/join regions.

The 77-boundary event projection is 1.004-1.027 ms saved at six rows and
0.882-0.925 ms at 32 rows. The three standalone warp mixers add a small
saving. These are kernel-chain projections; full-model bitwise comparison
and end-to-end verify latency remain integration checks. Existing kernel
locks and qualified cubin bytes are unchanged.
