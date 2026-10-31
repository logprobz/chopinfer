# Direct MoE cache knobs over the merged wide verify branch

The merge of the direct MoE cache knobs (`moe_direct_stream`,
`moe_direct_keep`, `moe_direct_retire` and `moe_direct_loads`, each off by
default) into the wide verify branch changes no journal: no fixture sets a
knob, and the knob entries live in their own locked modules
(`bandwidth_moe_deepseek41_knobs_sm120` and
`bandwidth_moe_deepseek41_wide_knobs_sm120`, family
`routed_moe_direct_knobs`), which no default instance loads.

On 2026-10-31 the journal proof of the merge commit against the first
parent, the wide verify branch, with an identity spec of the first
parent's capture, was PROVEN with full history coverage: 29 journals
identical, 29 of 29 fixture histories compared.
