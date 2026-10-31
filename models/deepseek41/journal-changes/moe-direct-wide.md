# Wide direct MoE behind the wide verify option

With the `wide_verify` candidate option, Flash TP4 verify steps of 65 to
192 rows at the package's 192 verify rows run the routed experts through
the wide direct queues (`bundle/src/sm120/moe_direct_wide.rs`) instead of
the 8192-row expert chains. The tuning table owns the range
(`moe.direct_wide.verify`). Without the option nothing changes: no wide
bucket is prepared and the wide module is never loaded. The router
projection and route launches, steps of 64 rows or fewer, TP1 and TP2 and
the drafter bank keep their launches either way. The
[lab qualification](../lab/moe_direct_wide.md) records the bitwise
evidence and the timings.

With the option, a TP4 instance with more than 64 verify rows loads the
locked module `bandwidth_moe_deepseek41_wide_sm120` on each rank and
resolves its 12 symbols there. No allocation, upload or workspace size
changes: the wide buckets need less workspace than the 8192-row chains
already reserve.

Only the two allocation fixtures with the option on change. Each gains 48
resolves at create (twelve symbols on each rank) and 52 bare releases at
destroy (the symbols and the module). In each of their 320 MoE layer
executions (40 layers, four ranks, two cycles), the gate/up chain, its
ceil quantizer and the down chain become the pack, absorbing gate queue,
gate finisher and two-strip down queue, and the routed sum becomes the
down finisher. The shared-expert launches between them keep their place
and operands.

| Fixture | Before count and FNV-1a | After count and FNV-1a |
|---|---|---|
| C16 odd, wide | 193206, `40f317c588987efb` | 193626, `3a5812bf4772ab54` |
| C32 maximum, wide | 345944, `8aacb4dc20b58ef8` | 346364, `3ad600cc7d668879` |

Every other journal is unchanged. No kernel lock or golden changes.
