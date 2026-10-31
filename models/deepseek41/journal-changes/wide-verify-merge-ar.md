# AR decode replay fixes over the merged wide verify branch

The merge of the AR decode replay fixes (the DeepSeek create order that
records the AR status events before the Engram gathers, the fake host's
release order and the opt-in `default_owners` fixed-batch mode) into the
wide verify branch changes no journal, and neither does the drafter
confidence comparison of fixed-batch replays that follows it, which
changes the host's replay tool only.

On 2026-10-31 the journal proof of that tip (the confidence comparison
over the merge) against the merged exact F32 fallback fix, with an
identity specification of the earlier commit's capture, was PROVEN with
full history coverage: 29 journals identical, 29 of 29 fixture histories
compared.
