"""Bounded-memory prefill of the pinned DeepSeek V4.1 reference code.

The vendor's `Transformer.forward` (model.py:1244-1275) runs a prompt as one chunk at start
position 0 and holds whole-prompt tensors that a long prompt cannot afford:

- the indexer scores every query against every compressed position at once, a
  [b, s, heads, t] tensor (model.py:557): 128 GiB at 32K tokens in F32, 1 TiB at 128K;
- the residual stream is [b, s, 4, 5120] (model.py:1261), widened to F32 by mHC
  (model.py:954): 10 GiB at 128K tokens, with a copy per sublayer; q is [b, s, 64, 512]
  (model.py:772), 16 GiB at 128K in F32.

`prefill` runs the same steps, mirroring each vendor line it replaces, with the whole-prompt
stream held on `stream_device` (the host) between layers and moved to a layer's device one row
block at a time:

1. Per row block: engram, the mHC mixes, `hc_pre` and the attention norm, then every per-row
   attention input: q_norm(wq_a(x)), the window key (norm, RoPE, FP8 QDQ), the compressor's
   wkv and wgate rows (or the ratio-1 latent) and the indexer's head weights.
2. Per layer, over the sequence: the compressor's pooling and tail state, the index-key and
   compressed-key appends, the window ring, exactly as `Compressor.forward`,
   `Indexer.forward`, `Attention._window_kv` and `Attention._compress_kv` do at start 0.
3. Per row block: the indexer's scores in query blocks of at most `score_bytes` (with the
   candidate pre-filter of its source), sparse attention of the block's queries against the
   full-sequence keys, the output projection, `hc_post`, the FFN sublayer and the next mix.

Every step is the vendor's per-row arithmetic on fewer rows, so the result is the unsplit
run's wherever the kernels compute one row independently of the others in the call (CPU
matmuls do; the adapter's replacements must, and the tests check the toy kernels). Two
deliberate departures, recorded in `REPLACEMENTS`:

- Selection ties go to the lower index (the engine's rule) instead of `torch.topk`'s
  kernel-defined order, in the index top-k, the candidate block top-k and the MoE router's
  expert top-k. A pick that
  would land on a reachable position the candidate filter excluded (possible only when fewer
  finite scores remain than the top-k, which the released geometry never allows) depends on
  that order alone, so it is refused.
- The candidate source publishes its kept block indices per query ([b, s, blocks] int32)
  instead of the [b, s, t] boolean mask; consumers rebuild the mask per query block.

The caches end as the vendor's prefill leaves them (window rings, compressor tail state,
compressed and index keys, the engram hash history), so a vendor decode step continues from
them. `main_hidden` rows (the DSpark target inputs) come back only for `hidden_rows`.

The adapter calls `install(vendor)` once per vendor namespace, which puts the same selection
rule into the vendor's `Indexer.forward`, `select_candidate_blocks` and `Gate.forward` for the
whole-prompt forward, the prefill and every decode step, and `prefill(...)` per sequence (it calls `check`: pinned
sources, world size, hooks); `rows_context(start, stop)` (tools/refrun/taps.chunk) wraps
every row-block call, and the forward hooks of each `Block` and `Attention` fire per row block
with that block's inputs and outputs. Taps on `Compressor` or `Indexer` modules, whose work
is split across phases, are refused, as are hooks on the model root other than the grid
watchers of tools/refrun/taps.
"""

from contextlib import nullcontext
import copy
import hashlib
import inspect

import torch
import torch.nn.functional as F

# The pinned execution plan of every capture: blocks of ROWS rows at absolute positions (a
# prompt of at most ROWS rows is one block, the vendor's own whole-prompt arithmetic), and one
# query block's index scores, [rows, heads, t] with the masked copy beside them, within
# SCORE_BYTES. Library GEMMs choose their kernel by row count, so another plan rounds
# differently; `plan_difference` bounds that against a one-block run.
ROWS = 2048
SCORE_BYTES = 1 << 30
# Limits of the plan against a one-block run, pinned on 2026-10-17 before any candidate run:
# per mode, the largest difference in a scored token's log-probability (nats) and the top-1
# logit gap under which a top-1 disagreement is a near tie. BF16 is a spread variant: its
# difference is recorded, not gated.
PLAN_LIMITS = {"f32": {"nats": 1e-4, "gap": 1e-4}}

# sha256 of the vendor sources `prefill` mirrors (inspect.getsource of model.py at 1bae0e10). A
# replaced method has another source, so an adapter patch here is refused rather than bypassed.
MIRRORED = {
    "Transformer.forward": "7d36e0bfaae15a295496b34a03205837f54e464888486e1eb1b87cd63babed12",
    "Transformer.merge_image_embeddings": "21251d9af6606c31c3b26dd6c9934fae8de26682461fdaf35ea94f08f0359597",
    "Block.forward": "b93813876daa6cb0bad02feb6678599fabb80ce46023635287c0781d4b51673d",
    "Attention.forward": "46a9a58e08f01510a8f4d63879a824ed3e792379d150917822918bdb4925546d",
    "Attention._window_kv": "8752c908e579c697f069faffa970b148b437fb5615743c5d45c430bab97254b2",
    "Attention._compress_kv": "cf185e9832854ebe927c80eb619ad90b1b6b2603422f12100d6e46cab31a202b",
    "Attention._compress_topk_idxs": "60ceca3a943abcf4c79bf0eeb544469548e4df754520c38d9e8d2788511339f1",
    "Compressor.forward": "cd864ce74af1194d0a30178035efa2af92b8f7d27666724b3c63cd1ca6e5a092",
    "Indexer.forward": "a590f59752f6b0fe7ee93bcd371f016d43ded4ec121e3d3190cef2764051e9ee",
    "select_candidate_blocks": "99d70a4f2befc8302203aa247495b12aae2d04d0b2125999554d9f222e219d3d",
    "make_identity_pre_mix": "7c456645aed206a60f09e4656cf7bdbf06d82ca460ee4546fd1815f93fab1dd8",
    "Gate.forward": "1995c8aaddb90f7267ae20ae5d50691aa1a4f1c6d9cc2f2a760fe42fc5b8848c",
}

REPLACEMENTS = {
    "Indexer.forward / select_candidate_blocks (every start position)": (
        "reference_memory.install: the vendor lines with ties to the lower index in the index top-k and the "
        "candidate block top-k; a pick of a filtered reachable position is refused"),
    "Gate.forward (every start position)": (
        "reference_memory.install: the vendor lines with ties to the lower expert index in the routing top-k "
        "(torch.topk orders ties by kernel), as the serving contract routes"),
    "Transformer.forward (start_pos 0)": (
        "reference_memory.prefill: the same steps in row blocks of at most ROWS rows, the [b, s, hc_mult, dim] "
        "stream on the host between layers; logits for the last `keep` rows (full_logits)"),
    "Block.forward / Attention.forward (start_pos 0)": (
        "per-row inputs per row block; window ring, compressor pooling and key appends once per layer over the "
        "sequence; sparse attention and the FFN per row block against the full-sequence keys"),
    "Indexer.forward (start_pos 0)": (
        "scores, candidate filter and top-k in query blocks of at most SCORE_BYTES of scores; ties to the lower "
        "index (torch.topk orders ties by kernel); a pick of a filtered reachable position is refused"),
    "select_candidate_blocks (prefill)": (
        "per query block, ties to the lower block index; publishes kept block indices [b, s, blocks] int32 "
        "instead of the [b, s, t] mask"),
}


def source_digests(vendor):
    model = vendor.model
    originals = getattr(model, "REFERENCE_MEMORY_ORIGINALS", None) or {}
    digests = {}
    for name in MIRRORED:
        target = originals.get(name)
        if target is None:
            target = model
            for part in name.split("."):
                target = getattr(target, part)
        digests[name] = hashlib.sha256(inspect.getsource(target).encode()).hexdigest()
    return digests


def install(vendor):
    """Puts the selection rule of `prefill` into the vendor's own `Indexer.forward`,
    `select_candidate_blocks` and `Gate.forward` (`indexer_forward`, `select_candidate_blocks`,
    `gate_forward`), so the whole-prompt forward, the prefill and every decode step select and
    route by it. Checks the pinned sources first; calling it again does nothing."""
    model = vendor.model
    if getattr(model, "REFERENCE_MEMORY_ORIGINALS", None) is not None:
        return
    changed = sorted(name for name, digest in source_digests(vendor).items() if digest != MIRRORED[name])
    if changed:
        raise ValueError(f"vendor sources mirrored by reference_memory changed or were replaced: {changed}")
    model.REFERENCE_MEMORY_ORIGINALS = {"Indexer.forward": model.Indexer.forward,
                                        "select_candidate_blocks": model.select_candidate_blocks,
                                        "Gate.forward": model.Gate.forward}
    model.Indexer.forward = indexer_forward(model)
    model.select_candidate_blocks = select_candidate_blocks
    model.Gate.forward = gate_forward(model)
    model.REFERENCE_MEMORY_INSTALLED = {"Indexer.forward": model.Indexer.forward,
                                        "select_candidate_blocks": select_candidate_blocks,
                                        "Gate.forward": model.Gate.forward}


def check(vendor, model):
    """Refuses a model whose mirrored vendor sources changed, a sharded run, or a forward hook
    `prefill` would not fire."""
    changed = sorted(name for name, digest in source_digests(vendor).items() if digest != MIRRORED[name])
    installed = getattr(vendor.model, "REFERENCE_MEMORY_INSTALLED", None) or {}
    current = {"Indexer.forward": vendor.model.Indexer.forward,
               "select_candidate_blocks": vendor.model.select_candidate_blocks,
               "Gate.forward": vendor.model.Gate.forward}
    changed += sorted(name for name, function in installed.items() if current[name] is not function)
    if changed:
        raise ValueError(f"vendor sources mirrored by reference_memory changed or were replaced: {changed}")
    if vendor.model.world_size != 1:
        raise ValueError(f"reference_memory runs one rank, not world_size {vendor.model.world_size}")
    split = (vendor.model.Compressor, vendor.model.Indexer, vendor.model.Transformer)
    for name, module in model.named_modules():
        if not isinstance(module, split):
            continue
        hooks = [*module._forward_hooks.values(), *module._forward_pre_hooks.values()]
        if isinstance(module, vendor.model.Transformer):
            # tools/refrun/taps watches every ancestor of an encoder tap for its grid; prefill never
            # calls the root, so these never fire, and need not
            hooks = [hook for hook in hooks if not grid_watcher(hook)]
        if hooks:
            raise ValueError(f"{name or 'model'}: prefill splits this module's work, so its hooks cannot fire")


def grid_watcher(hook):
    """Whether `hook` is one of the grid hooks tools/refrun/taps.Taps puts on an encoder tap's
    ancestors (`grid_hook`, `grid_done`)."""
    function = getattr(hook, "__func__", None)
    return (getattr(function, "__module__", None) == "tools.refrun.taps"
            and getattr(function, "__qualname__", None) in ("Taps.grid_hook", "Taps.grid_done"))


def lower_topk(scores, k):
    """Indices of the k largest scores along the last dimension, ties to the lower index."""
    return torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :k]


def top_positions(index_score, topk, compress_lens, offset, where):
    """Indexer.forward's selection (model.py:578-581) with ties to the lower index: the top-k
    positions in position order, unreachable ones -1, the rest shifted by `offset`. A pick of a
    reachable position at -inf (filtered out by the candidate blocks) would be decided by the
    tie order alone, so it is refused."""
    picks = lower_topk(index_score, topk)
    refused = (index_score.gather(-1, picks) == -torch.inf) & (picks < compress_lens)
    if bool(refused.any()):
        raise ValueError(f"{where}: fewer finite index scores than the top-k leave a filtered reachable "
                         "position to the tie order")
    idxs = picks.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1).int()


def indexer_forward(m):
    """`Indexer.forward` of the vendor namespace `m` (model.py:527-581), line for line, with the
    selection of `top_positions` and the candidate blocks of `select_candidate_blocks`."""

    def forward(self, x, qr, latent, start_pos, offset):
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen
        if self.owns_k and latent is not None:
            freqs = (
                self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            k = self.k_norm(self.wk(latent))
            m.apply_rotary_emb(k[..., -rd:], freqs)
            m.fp4_act_quant(k, m.fp4_block_size, True)
            self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
        if self.owns_k:
            m.shared_attn.index_k = self.k_cache
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        m.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        m.fp4_act_quant(q, m.fp4_block_size, True)
        index_k = m.shared_attn.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if m.world_size > 1:
            m.dist.all_reduce(index_score)
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio
        if self.is_candidate_source:
            m.shared_attn.candidates = m.select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~m.shared_attn.candidates, -torch.inf)
        topk = min(self.index_topk, end_pos // ratio)
        return top_positions(index_score, topk, compress_lens, offset, f"indexer at start {start_pos}")

    return forward


def gate_forward(m):
    """`Gate.forward` of the vendor namespace `m` (model.py:811-830), line for line, with ties to
    the lower expert index in the routing top-k. Under the adapter's F64 copy (`m.WIDE`) the
    scores are F64, as its widened `.float()` casts make them."""

    def forward(self, x, image_mask=None):
        widen = torch.Tensor.double if getattr(m, "WIDE", False) else torch.Tensor.float
        scores = m.linear(widen(x), widen(self.weight)) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        indices = lower_topk(scores + bias, self.topk)
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
        weights *= self.route_scale
        return weights, indices

    return forward


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """`select_candidate_blocks` (model.py:584-611) with ties to the lower block index."""
    return candidate_mask(candidate_blocks(logits, compress_lens, topk_blocks, block_size), logits.size(-1),
                          block_size)


def query_rows(heads, width, element, budget=SCORE_BYTES):
    """Query rows per indexer block: two [rows, heads, width] score tensors within `budget`."""
    return max(1, budget // max(1, 2 * heads * width * element))


def candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """`select_candidate_blocks` (model.py:597-611) as kept block indices, -1 for a dropped pick."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    order = lower_topk(scores, min(topk_blocks, num_blocks))
    return torch.where(scores.gather(-1, order) > -torch.inf, order, -1).int()


def candidate_mask(kept, width, block_size):
    """The [..., width] boolean mask `select_candidate_blocks` returns, from kept block indices."""
    num_blocks = -(-width // block_size)
    keep = torch.zeros(*kept.shape[:-1], num_blocks + 1, dtype=torch.bool, device=kept.device)
    keep.scatter_(-1, torch.where(kept >= 0, kept, num_blocks).long(), True)
    return keep[..., :num_blocks].repeat_interleave(block_size, dim=-1)[..., :width]


def fire(module, args, output):
    """A module's forward hooks for a call `prefill` computed without calling the module, as
    `nn.Module.__call__` runs them: a pre-hook's returned inputs (the runner's device moves)
    reach the later hooks; a hook that would change the output is refused."""
    kwargs = {}
    for key, hook in module._forward_pre_hooks.items():
        if key in module._forward_pre_hooks_with_kwargs:
            result = hook(module, args, kwargs)
            if result is not None:
                if not (isinstance(result, tuple) and len(result) == 2):
                    raise ValueError(f"{type(module).__name__}: a kwargs pre-hook returned {type(result).__name__}")
                args, kwargs = result
        else:
            result = hook(module, args)
            if result is not None:
                args = result if isinstance(result, tuple) else (result,)
    for key, hook in module._forward_hooks.items():
        if key in module._forward_hooks_with_kwargs:
            result = hook(module, args, kwargs, output)
        else:
            result = hook(module, args, output)
        if result is not None:
            raise ValueError(f"{type(module).__name__}: a hook that changes outputs cannot run under prefill")


def device_of(module):
    return next(module.parameters()).device


class Sequence:
    """Whole-sequence tensors of one layer, allocated at the first row block's shape."""

    def __init__(self, bsz, seqlen):
        self.bsz, self.seqlen = bsz, seqlen
        self.tensors = {}

    def put(self, name, start, value, rows=None, device=None):
        tensor = self.tensors.get(name)
        if tensor is None:
            rows = self.seqlen if rows is None else rows
            tensor = torch.empty((self.bsz, rows, *value.shape[2:]), dtype=value.dtype,
                                 device=value.device if device is None else device)
            self.tensors[name] = tensor
        tensor[:, start:start + value.size(1)] = value.to(tensor.device)

    def __getitem__(self, name):
        return self.tensors[name]

    def get(self, name):
        return self.tensors.get(name)


class LayerAttention:
    """One layer's `Attention` at start 0 split into its three phases (module docstring)."""

    def __init__(self, vendor, attn, bsz, seqlen, shared, score_bytes):
        self.m, self.attn, self.shared, self.score_bytes = vendor.model, attn, shared, score_bytes
        self.bsz, self.seqlen = bsz, seqlen
        self.seq = Sequence(bsz, seqlen)
        self.ratio = attn.compress_ratio
        self.compress_len = seqlen // self.ratio if self.ratio else 0
        self.selects = bool(self.ratio) and attn.is_index_source and self.compress_len > 0

    def inputs(self, x, start):
        """Phase 1, rows [start, start + x.size(1)): every per-row attention input."""
        m, attn, seq = self.m, self.attn, self.seq
        rows = x.size(1)
        freqs_cis = attn.freqs_cis[start:start + rows]
        self.dtype = x.dtype
        # Attention.forward (model.py:771)
        seq.put("qr", start, attn.q_norm(attn.wq_a(x)))
        # Attention._window_kv (model.py:706-708), written into the keys sparse attention reads
        kv = attn.kv_norm(attn.wkv(x))
        m.apply_rotary_emb(kv[..., -attn.rope_head_dim:], freqs_cis)
        m.act_quant(kv, m.fp8_block_size, m.scale_fmt, m.scale_dtype, True)
        seq.put("kv", start, kv, rows=self.seqlen + self.compress_len)
        if self.ratio and attn.is_kv_source:
            compressor = attn.compressor
            if self.ratio == 1:  # Compressor.forward (model.py:461-462)
                seq.put("latent", start, compressor.norm(compressor.wkv(x)))
            else:  # Compressor.forward (model.py:464-465), widened with the adapter's F64 module
                xf = x.double() if getattr(m, "WIDE", False) else x.float()
                seq.put("score_kv", start, compressor.wkv(xf))
                seq.put("score", start, compressor.wgate(xf))
        if self.selects:  # Indexer.forward (model.py:556)
            indexer = attn.indexer
            seq.put("weights", start, indexer.weights_proj(x) * (indexer.softmax_scale * indexer.n_heads**-0.5))

    def sequence(self):
        """Phase 2: the window ring, the compressor's pooling and state, and the key appends."""
        m, attn, seq, shared = self.m, self.attn, self.seq, self.shared
        bsz, seqlen, win = self.bsz, self.seqlen, attn.window_size
        kv = seq["kv"][:, :seqlen]
        # Attention._window_kv (model.py:709-716)
        if seqlen <= win:
            attn.window_kv_cache[:bsz, :seqlen] = kv
        else:
            cutoff = seqlen % win
            attn.window_kv_cache[:bsz, cutoff:win], attn.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                [win - cutoff, cutoff], dim=1
            )
        self.window_idxs = m.get_window_topk_idxs(win, bsz, seqlen, 0).to(kv.device)
        if not self.ratio:
            return
        ratio, rd = self.ratio, attn.rope_head_dim
        # Attention._compress_kv (model.py:743-749)
        latent = None
        if attn.is_kv_source:
            latent = self.compress(attn.compressor, self.dtype)
            m.shared_attn.compress_kv = attn.compress_kv_cache
        # Attention._compress_topk_idxs (model.py:726-737) and Indexer.forward (model.py:536-549)
        if attn.is_index_source:
            if self.compress_len == 0:
                self.compress_idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=kv.device)
            else:
                indexer = attn.indexer
                if indexer.freqs_cis is None:
                    indexer.freqs_cis = attn.freqs_cis
                if indexer.owns_k and latent is not None:
                    freqs = indexer.freqs_cis[: seqlen - seqlen % ratio : ratio]
                    k = indexer.k_norm(indexer.wk(latent))
                    m.apply_rotary_emb(k[..., -rd:], freqs)
                    m.fp4_act_quant(k, m.fp4_block_size, True)
                    indexer.k_cache[:bsz, 0:k.size(1)] = k
                if indexer.owns_k:
                    m.shared_attn.index_k = indexer.k_cache
                self.compress_idxs = None  # filled per row block in phase 3
        else:
            self.compress_idxs = shared.get("topk_idxs")
            if self.compress_idxs is None:
                raise ValueError(f"layer {attn.layer_id} reads the top-k of an index source no layer before it ran")
        # Attention._compress_kv (model.py:752-764)
        if latent is not None:
            freqs = attn.freqs_cis[: seqlen - seqlen % ratio : ratio]
            m.apply_rotary_emb(latent[..., -rd:], freqs)
            m.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            attn.compress_kv_cache[:bsz, 0:latent.size(1)] = latent
        seq["kv"][:, seqlen:] = m.shared_attn.compress_kv[:bsz, :self.compress_len].to(kv.device)

    def compress(self, compressor, dtype):
        """Compressor.forward at start 0 (model.py:458-485) over the stored per-row outputs."""
        seq, bsz, seqlen, ratio = self.seq, self.bsz, self.seqlen, self.ratio
        if ratio == 1:
            return seq["latent"]
        kv, score = seq["score_kv"], seq["score"]
        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        if remainder:
            kv, compressor.kv_state[:bsz, :remainder] = kv.split([cutoff, remainder], dim=1)
            score, compressor.score_state[:bsz, :remainder] = score.split([cutoff, remainder], dim=1)
        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        if not should_compress:
            return None
        return compressor.norm(kv.to(dtype))

    def select(self, start, stop):
        """Indexer.forward (model.py:551-581) at start 0 for queries [start, stop)."""
        m, attn, seq, shared = self.m, self.attn, self.seq, self.shared
        indexer, bsz, ratio, rd = attn.indexer, self.bsz, self.ratio, attn.rope_head_dim
        width = self.compress_len
        q = indexer.wq_b(seq["qr"][:, start:stop]).unflatten(-1, (indexer.n_local_heads, indexer.index_head_dim))
        m.apply_rotary_emb(q[..., -rd:], indexer.freqs_cis[start:stop])
        m.fp4_act_quant(q, m.fp4_block_size, True)
        index_k = m.shared_attn.index_k[:bsz, :width].to(q.device)
        weights = seq["weights"][:, start:stop]
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        compress_lens = (torch.arange(start + 1, stop + 1, device=q.device) // ratio).unsqueeze(-1)
        index_score.masked_fill_(torch.arange(width, device=q.device) >= compress_lens, -torch.inf)
        if indexer.is_candidate_source:
            kept = candidate_blocks(index_score, compress_lens, indexer.candidate_topk_blocks,
                                    indexer.candidate_block_size)
            self.candidates.put("kept", start, kept, device=shared["stream_device"])
        elif indexer.uses_candidates:
            kept = shared["candidates"][:, start:stop].to(q.device)
            index_score = index_score.masked_fill(~candidate_mask(kept, width, indexer.candidate_block_size),
                                                  -torch.inf)
        return top_positions(index_score, min(indexer.index_topk, width), compress_lens, self.seqlen,
                             f"layer {attn.layer_id}")

    def rows(self, start, stop):
        """Phase 3, rows [start, stop): selection, sparse attention and the output projection."""
        m, attn, seq = self.m, self.attn, self.seq
        bsz, rd = self.bsz, attn.rope_head_dim
        qr = seq["qr"][:, start:stop]
        freqs_cis = attn.freqs_cis[start:stop]
        if self.selects:
            indexer = attn.indexer
            step = query_rows(indexer.n_local_heads, self.compress_len, qr.element_size(), self.score_bytes)
            for begin in range(start, stop, step):
                end = min(stop, begin + step)
                self.selected.put("idxs", begin, self.select(begin, end))
        # Attention.forward (model.py:772-790)
        q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        m.apply_rotary_emb(q[..., -rd:], freqs_cis)
        topk_idxs = self.window_idxs[:, start:stop]
        if self.ratio:
            compress_idxs = self.selected["idxs"] if self.selects else self.compress_idxs
            topk_idxs = torch.cat([topk_idxs, compress_idxs[:, start:stop].to(topk_idxs.device)], dim=-1)
        o = m.sparse_attn(q, seq["kv"], attn.attn_sink, topk_idxs, attn.softmax_scale)
        m.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
        o = o.view(bsz, stop - start, attn.n_local_groups, -1)
        m.act_quant(o, m.fp8_block_size, m.scale_fmt, m.scale_dtype, True)
        wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return attn.wo_b(o.flatten(2))

    def begin_rows(self):
        self.selected = Sequence(self.bsz, self.seqlen)
        self.candidates = Sequence(self.bsz, self.seqlen)

    def held(self):
        """The whole-sequence tensors of this layer, by name; `host.` names live on the stream device."""
        held = dict(self.seq.tensors)
        held["window_idxs"] = self.window_idxs
        if self.selects:
            held["selected"] = self.selected["idxs"]
        if self.candidates.get("kept") is not None:
            held["host.candidates"] = self.candidates["kept"]
        return held

    def publish(self):
        """What the layer hands down the stack (SharedAttentionRuntime, model.py:1169-1179)."""
        if self.selects:
            self.shared["topk_idxs"] = self.selected["idxs"]
        elif self.ratio and self.attn.is_index_source:
            self.shared["topk_idxs"] = self.compress_idxs
        if self.candidates.get("kept") is not None:
            self.shared["candidates"] = self.candidates["kept"]


def blocks(length, rows):
    return [(start, min(length, start + rows)) for start in range(0, length, rows)]


@torch.inference_mode()
def prefill(vendor, model, input_ids, keep=1, images=None, token_types=None, *, rows=ROWS,
            score_bytes=SCORE_BYTES, stream_device="cpu", hidden_rows=None, rows_context=None, on_layer=None):
    """`Transformer.forward(input_ids, 0, images, token_types)` in row blocks (module docstring).

    Returns the F32 logits [b, keep, vocab] of the last `keep` rows, and the DSpark target
    inputs `main_hidden` [b, len(hidden_rows), dim * targets] for the positions `hidden_rows`
    (a slice of consecutive positions), or None. `on_layer(index, held)`, when given, sees each
    layer's whole-sequence tensors (`LayerAttention.held`) and the host stream once the layer ran
    (reference_fit counts them).
    """
    check(vendor, model)
    m = vendor.model
    rows_context = rows_context or (lambda _start, _stop: nullcontext())
    bsz, seqlen = input_ids.shape
    if not 1 <= keep <= seqlen:
        raise ValueError(f"keep {keep} is outside 1..{seqlen}")
    parts = blocks(seqlen, rows)
    hc = model.hc_mult
    # Transformer.forward (model.py:1252-1255)
    image_mask = None if token_types is None else token_types >= 0
    engram_mask = None if image_mask is None else ~image_mask
    engram_hashes = None
    if model.engram_hash is not None:
        hash_device = model.engram_hash.token_map.device
        engram_hashes = model.engram_hash(
            input_ids.to(hash_device), 0, None if engram_mask is None else engram_mask.to(hash_device))
    # model.py:1256-1261: embed, merge the image spans, expand to hc_mult copies
    embed_device = device_of(model.embed)
    stream = None
    for start, stop in parts:
        with rows_context(start, stop):
            h = model.embed(input_ids[:, start:stop].to(embed_device))
        if stream is None:
            stream = torch.empty(bsz, seqlen, hc, h.size(-1), dtype=h.dtype, device=stream_device)
        stream[:, start:stop] = h.unsqueeze(2).to(stream_device)
    if images is not None:
        for i, sample in enumerate(images):
            for image in sample or ():
                span = slice(image.start, image.start + image.types.numel())
                h = stream[i:i + 1, span, 0].to(embed_device)
                shifted = copy.copy(image)
                shifted.start = 0
                model.merge_image_embeddings([[shifted]], h)
                stream[i, span] = h[0].unsqueeze(1).to(stream_device)
    pre_mix = m.make_identity_pre_mix(stream, hc)
    if hidden_rows is not None:
        hidden_rows = range(*hidden_rows.indices(seqlen))
        if hidden_rows.step != 1:
            raise ValueError("hidden_rows is a slice of consecutive positions")
    main_hiddens = []
    shared = {"stream_device": stream_device}
    for i, layer in enumerate(model.layers):
        device = device_of(layer)
        attention = LayerAttention(vendor, layer.attn, bsz, seqlen, shared, score_bytes)
        mixes = Sequence(bsz, seqlen)
        hidden = None
        for start, stop in parts:
            with rows_context(start, stop):
                h = stream[:, start:stop].to(device)
                if layer.engram is not None:  # model.py:1265-1266
                    h = layer.engram(h, engram_hashes[:, start:stop, layer.engram.layer_hash_index, :].to(device),
                                     None if engram_mask is None else engram_mask[:, start:stop].to(device))
                    stream[:, start:stop] = h.to(stream_device)
                if i in model.target_layer_ids and hidden_rows is not None:  # model.py:1268-1269
                    wanted = [r for r in hidden_rows if start <= r < stop]
                    if wanted:
                        if hidden is None:
                            hidden = torch.empty(bsz, len(hidden_rows), h.size(-1), dtype=h.dtype,
                                                 device=stream_device)
                        offset = wanted[0] - hidden_rows.start
                        local = torch.tensor([r - start for r in wanted], device=device)
                        hidden[:, offset:offset + len(wanted)] = h[:, local].mean(dim=2).to(stream_device)
                # Block.forward (model.py:984-986)
                attn_pre, attn_post, attn_comb = layer.hc_mixes(h, layer.hc_attn_fn, layer.hc_attn_scale,
                                                                layer.hc_attn_base)
                mixes.put("pre", start, attn_pre)
                mixes.put("post", start, attn_post)
                mixes.put("comb", start, attn_comb)
                x = layer.hc_pre(h, pre_mix[:, start:stop].to(device))
                x = layer.attn_norm(x)
                attention.inputs(x, start)
        if hidden is not None:
            main_hiddens.append(hidden)
        attention.sequence()
        attention.begin_rows()
        for start, stop in parts:
            with rows_context(start, stop):
                residual = stream[:, start:stop].to(device)
                x = attention.rows(start, stop)
                if layer.attn._forward_hooks or layer.attn._forward_pre_hooks:
                    # the attention input again, through forward() so the norm's own hooks stay quiet
                    attn_input = layer.attn_norm.forward(layer.hc_pre(residual, pre_mix[:, start:stop].to(device)))
                    fire(layer.attn, (attn_input, 0), x)
                # Block.forward (model.py:988-995)
                x = layer.hc_post(x, residual, mixes["post"][:, start:stop], mixes["comb"][:, start:stop])
                block_input = residual
                residual = x
                ffn_pre, ffn_post, ffn_comb = layer.hc_mixes(x, layer.hc_ffn_fn, layer.hc_ffn_scale,
                                                             layer.hc_ffn_base)
                x = layer.hc_pre(x, mixes["pre"][:, start:stop])
                x = layer.ffn_norm(x)
                x = layer.ffn(x, None if image_mask is None else image_mask[:, start:stop].to(device))
                x = layer.hc_post(x, residual, ffn_post, ffn_comb)
                fire(layer, (block_input, 0, pre_mix[:, start:stop],
                             None if image_mask is None else image_mask[:, start:stop]), (x, ffn_pre))
                stream[:, start:stop] = x.to(stream_device)
                pre_mix[:, start:stop] = ffn_pre.to(stream_device)
        attention.publish()
        if on_layer is not None:
            on_layer(i, {**attention.held(), **{f"mixes.{k}": v for k, v in mixes.tensors.items()},
                         "host.stream": stream, "host.pre_mix": pre_mix})
    m.shared_attn.topk_idxs = None
    m.shared_attn.candidates = None
    # model.py:1271-1272: hc_pre and the final norm over every row, the head over the kept rows
    norm_device, head_device = device_of(model.norm), device_of(model.head)
    first_kept, logits = seqlen - keep, []
    for start, stop in parts:
        with rows_context(start, stop):
            h = layer.hc_pre(stream[:, start:stop].to(norm_device), pre_mix[:, start:stop].to(norm_device))
            h = model.norm(h)
        if stop > first_kept:
            begin = max(start, first_kept)
            with rows_context(begin, stop):
                logits.append(model.head(h[:, begin - start:].to(head_device), full_logits=True))
    main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
    return torch.cat(logits, dim=1), main_hidden


def unsplit_logits(model, input_ids, images=None, token_types=None):
    """The vendor forward with every row's logits: `Transformer.forward` with the head asked for
    full logits, for the comparisons with `prefill`."""
    head = model.head
    original = head.forward
    head.forward = lambda x, full_logits=False: original(x, full_logits=True)
    try:
        _ids, logits, main_hidden = model(input_ids, 0, images, token_types)
    finally:
        del head.forward
    return logits, main_hidden


def plan_difference(plan_logits, one_logits, targets, gap=PLAN_LIMITS["f32"]["gap"]):
    """The pinned plan's logits against a one-block run's, [rows, vocab] each, over the rows'
    scored `targets` [rows]: the largest and mean difference in the targets' log-probabilities
    (nats), the top-1 agreement, and the disagreements whose one-block top-1 gap is at least
    `gap` (clear ones) or under it (near ties)."""
    plan_lp = torch.log_softmax(plan_logits.double(), -1)
    one_lp = torch.log_softmax(one_logits.double(), -1)
    index = targets.long().unsqueeze(-1)
    nats = (plan_lp.gather(-1, index) - one_lp.gather(-1, index)).abs().squeeze(-1)
    top = one_logits.double().topk(2, dim=-1).values
    near = (top[:, 0] - top[:, 1]) < gap
    disagree = plan_logits.argmax(-1) != one_logits.argmax(-1)
    return {"rows": int(targets.numel()), "max_nats": float(nats.max()), "mean_nats": float(nats.mean()),
            "top1_agreement": float((~disagree).double().mean()),
            "clear_top1_disagreements": int((disagree & ~near).sum()),
            "near_tie_top1_disagreements": int((disagree & near).sum())}


def plan_verdict(mode, difference):
    """The refusals of a plan difference under PLAN_LIMITS; None for a recorded-only mode."""
    limits = PLAN_LIMITS.get(mode)
    if limits is None:
        return None
    refusals = []
    if difference["max_nats"] > limits["nats"]:
        refusals.append(f"{mode}: {difference['max_nats']:.3g} nats > {limits['nats']:g}")
    if difference["clear_top1_disagreements"]:
        refusals.append(f"{mode}: {difference['clear_top1_disagreements']} top-1 disagreements with a gap "
                        f">= {limits['gap']:g}")
    return refusals


def compare_plan(vendor, model, input_ids, keep, **options):
    """The pinned plan (ROWS) and a one-block run (rows = the prompt) of the same prompt, scored
    teacher-forced on the kept rows' next tokens: `plan_difference` of their logits. `options`
    go to both prefills (images, token_types, stream_device, ...); the caller resets nothing
    between them, since a prefill starts from position 0 and rewrites every cache it reads."""
    seqlen = input_ids.size(1)
    plan, _ = prefill(vendor, model, input_ids, keep, rows=ROWS, **options)
    one, _ = prefill(vendor, model, input_ids, keep, rows=seqlen, **options)
    targets = input_ids[0, seqlen - keep + 1:].to(plan.device)
    return plan_difference(plan[0, :-1], one[0, :-1], targets)
