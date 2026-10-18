# Pair pooling

`pool.cuh` implements `GatedPoolingQuery` for group 2 at widths 128 and 512.
The input key and gate projections and the two tail banks are F32. The
pooling launch computes a stable per-channel softmax and weighted sum in
group order. A second launch applies RMSNorm in F32 with exactly decoded
BF16 weights and writes F32. Keeping the F32 intermediate in scratch lets
the lab check each stage against its own oracle. Neither stage rounds its
activations to BF16.

Rows of one item form one contiguous run, with consecutive non-hole token
positions. Negative positions are holes. Only completing rows write a
latent; all live rows write a compressed position or -1. Rows beyond the
device live count remain untouched. Prefill and decode leave the last live
tail in place. Verify writes live tails to per-item, per-row checkpoints.
Completing rows have no live tail, so their checkpoint bytes stay unchanged.
Source slots of different items and all checkpoint destinations are distinct.

The first non-hole row of an item owns every access to its incoming tail
and its final in-place tail. Other completing rows read both members from
the launch's inputs. Both launches wait for PDL predecessors before reading
the live count or any input. Tail offsets use 64-bit products.

Ratio one composes a projection with F32 output and the same F32 RMSNorm
without a completion mask; it has no pooling state. Key writers keep their
input, optional key RMSNorm and tail RoPE in F32. Their sole BF16 rounding
is immediately before the trained FP4 quantizer's amax and code decisions.
Cache publication and index-before-attention write order
belong to the model's launch sequence, including steps that emit no latent.
