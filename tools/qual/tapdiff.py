"""Name the first tap where the engine leaves the reference's own spread.

    python3 -m tools.qual.tapdiff --engine DUMP --f32 RUN/taps/f32 \\
        --bf16 RUN/taps/bf16 [--f64 RUN/taps/f64] [--request ENGINE=SEQUENCE] \\
        [--ratio 3] [--limit 0.05] [--every-row GLOB] [--row-floor F] \\
        [--codes-margin 0.03] [--direction-scale GLOB] [--scale-variant NAME=DIR]
        [--workload WORKLOAD] [--json RECEIPT]

Every input is a tap directory of `step<NNNN>` subdirectories in the dump
format (index.jsonl, items.json, capture.json, `chopinfer.taps.v1`): the
engine's taps (runtime_bundle_sdk::taps) and the reference runner's taps
(tools/refrun/taps.py) per mode, under the same names.

1. Rows are aligned by (request, position) through each step's items, so
   packed prefills and chunked prefill steps line up. An encoder tap
   (`"axis": "encoder"` in index.jsonl) holds encoder output rows instead,
   aligned by (request, media kind, item, row) through the step's items.json
   `encoder` segments: `{tap: [{"request", "start", "kind", "item",
   "index", "rows"}]}`, tensor rows start.. holding rows index.. of the
   request's occurrence `item` of media `kind` (counted per kind in prompt
   order), so the order a producer stores its items in never pairs rows of
   different items. Segments must map every stored row exactly once and
   name their kind and item; an item or kind one side lacks is missing. An engine request is
   matched to the reference sequence whose tokens agree at every shared
   position (or named with --request); engine requests that match none,
   such as warmup passes, are ignored. A reference sequence whose steps
   hold encoder rows only (tools/refrun/encoder.py) stores no positions;
   --workload (the runner's workload) gives its tokens, in the reference
   and, when it is such a run too, the engine.
2. Engine rank shards are reduced as capture.json `expected` (or --declare)
   states: `replicated` (rank 0), `rank_partial` (rank-ordered F32 sum) or
   `col_shards` (columns concatenated by per-rank blocks, or each rank's
   share of every [heads, dim] block); a tap of other shards
   (`expert_partial`, a per-rank layout) is not compared. A dump of a
   model's older schema (`v1_schema` in `models/<m>/taps.json`, such as
   `chopinfer.qwen38.capture.v1`) is read through the declaration, which
   names its taps (`v1`), their sharding and comparison. A capture schema
   no declaration reads, an undeclared tap, a tap whose dtype, ranks or
   columns differ from its declaration, and a tap without a declared
   sharding in a multi-rank capture are incompatible. A `replicated` tap
   stored by several ranks (each rank computes it) must hold bitwise-equal
   copies; a copy that differs fails the tap. A chopinfer.taps.v1 capture's
   index must hold only the taps and ranks capture.json `expected` declares,
   with their declared dtype, columns and rows; an engine capture names its
   `model`, whose declaration must declare each tap's name, dtype, ranks,
   columns and sharding.
3. Per dense tap and row: e = |engine - f32| / |f32|, floor = |bf16 - f32| /
   |f32| and num = |f32 - f64| / |f64| (relative L2 of the row). A row
   exceeds when e > ratio * floor, or e > limit where it has no floor.
   Rows are those the engine and f32 share; bf16 and f64 rows are used
   where they exist (a row without bf16 has no floor, and `num` reports
   the f64 rows it covers), so an optional arm cannot remove a row. A zero
   reference row matches only a zero row, and nonfinite engine values
   always exceed; nonfinite f32 or f64 values are refused.
   Router weights (`paired`) are aligned by the expert ids of the sibling
   `router.ids` tap, which must hold one distinct expert index per
   weight: a row whose id set differs from f32's is a routing flip
   (counted, and left to the ids tap), not a weight error. An encoder
   tap's engine rows must be exactly the f32 rows of the compared
   requests (`encoder_rows`): missing rows make it incomplete, rows the
   reference lacks incompatible.
4. Per tap: median, p90 and max of e, floor, e/floor and num. A dense tap
   fails when its median excess (e / (ratio * floor), or e / limit) is
   above 1. A dense tap that --every-row names also fails when any row
   exceeds e > ratio * max(floor, row floor) (e > limit for a row without a
   bf16 row); the row floor (--row-floor, default 2^-8, the BF16 unit
   roundoff doubled) keeps a row whose bf16 run happens to equal f32 from
   demanding an exact engine. `--every-row GLOB:CODES/GROUP` holds an
   encoder row that merges GROUP rows of codes tap CODES (row i merges code
   rows GROUP * i..) only when the engine's codes there all equal f32's,
   under both rules: a row whose codes flipped is left to the codes tap and
   counted (`code_flipped_rows`). A dense tap that --aggregate names is
   also held as a whole, so a few badly wrong rows cannot hide under the
   median: its relative L2 over every compared row, |engine - f32| / |f32|,
   at most ratio times the bf16 run's over the rows bf16 covers, raised to
   ROW_FLOOR (limit without bf16 rows). A set tap (top-k ids) fails when its row
   disagreement rate exceeds ratio times bf16's (or limit when bf16's is 0).
   An encoder tap that --direction-scale names is judged as each row's
   direction and scale instead of its relative L2, since a few-ulp error
   in one massive channel can rescale a whole normalized row: direction d =
   |engine / |engine| - f32 / |f32|| against the bf16 run's (same rules as
   above: every row d <= ratio * max(floor, row floor), median excess <= 1),
   and per item (request, media kind and item) the largest row scale error
   | |engine| / |f32| - 1 | at most the largest one of the bf16 run and each
   --scale-variant over the item's rows, rounded up to 0.005. With
   --scale-policy NAME (an engine that runs an F32 policy, which variant
   NAME emulates) only NAME's largest row scale error sets the limit, at
   least POLICY_SCALE_STEP (0.001) and rounded up to it; the bf16 run's and
   the other variants' are reported. The receipt lists each item's scale
   errors side by side, and the rule.
   A codes tap (integer codes compared one by one, such as an RVQ
   quantizer's `x.codes`, [rows, stages]) fails when its agreement with f32,
   over every code or over the first stage, is below bf16's agreement over
   the same codes less --codes-margin (default 0.03), or below 1 less the
   margin without a bf16 arm. Discrete taps are reported.

5. With --router (the model's router: `score` sigmoid or softmax, `top_k`,
   `n_group`, `topk_group`, and `bias` when it adds a correction bias,
   given by --router-bias, a tools/qual/router_bias.py sidecar), routing
   flips are judged by route-near-tie.v2. Per MoE layer (each reference
   `router.ids` tap with its `router.logits`), the selection scores are
   rebuilt in binary64 from the reference's F32 logits (plus the bias). A
   flip (the engine's id set differs from F32's) has the gap of the largest
   pair it reversed under F32's scores (max over dropped experts minus min
   over added ones) and is a near tie when the gap is at most the layer's
   margin: ratio times the largest movement of F32's k-th/(k+1)-th gap in
   the bf16 run, over rows the bf16 run has not routed differently at an
   earlier layer, cumulative over earlier layers, raised to two F32 ulps of
   the layer's largest score. A row's first flip decides it. A dense tap
   that fails only its --aggregate rule passes when removing the fewest rows
   behind an earlier near-tie first flip, largest squared error first, makes
   the aggregate over the rest pass (the bf16 run's over the same rows), and
   each removed row is within ratio times the bf16 run's worst row at the tap
   (raised to ROW_FLOOR) and plausible: every value finite, its norm within
   a factor of ratio of the F32 row's and its cosine with that row positive.
   The tap's `route_explanation` records it. Without
   a bf16 run, the bias, finite scores, or every layer's taps, or for a
   group-limited router (n_group above 1, whose group mask is not tapped),
   routes are unjudged and every tap keeps the rules above. The receipt's
   `routes` holds the margins, every flip and the bf16 run's own first
   flips beyond the margin.

The verdict is the first failing tap in the reference's program order.
Without one, a step holding compared rows whose capture is incomplete
(completion.json missing or not `full`, a declared tap absent from its
index, or a tap holding fewer rows than the step's `dump_rows` or items)
makes the verdict `incomplete`, not a match.
Per position, the first exceeding tap is counted in a histogram: a kernel
error makes every position fail at one tap, a routing flip a few positions
downstream. Encoder rows are not positions: each encoder tap reports how
many of its rows exceed, apart from the histogram. The receipt (--json) binds the inputs' metadata files by sha256.
Exit status: 0 no tap fails, 1 a tap fails, 2 incompatible inputs, 3 an
incomplete capture or nothing compared.
"""

import argparse
from decimal import Decimal
import fnmatch
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.model_sdk import diagnostic, taps as tap_names  # noqa: E402

SCHEMA = "chopinfer.tapdiff.v1"
RATIO, LIMIT = 3.0, 0.05
# --every-row's row floor: twice the BF16 unit roundoff, the relative error of
# storing a row in BF16.
ROW_FLOOR = 2.0 ** -8
# --codes-margin: MiMo's legacy BF16 audio tokenizer agrees with the f64
# codes 2 points less often than the vendor's BF16 run (71.2% against 73.2%
# of the codes, the GPU encoder reference, 2026-10-01).
CODES_MARGIN = 0.03
# --direction-scale: an item's scale limit is its BF16 variants' largest row
# scale error rounded up to this step.
SCALE_STEP = 0.005
# --scale-policy: an F32-policy engine's item scale limit is its policy run's
# largest row scale error, at least this floor and rounded up to it. The F32
# policy and the exact run differ only in reduction order (at most 0.06% on
# Qwen3.8's probes), and storing merged rows in BF16 moves a row norm by
# about 2^-9 / sqrt(cols), far below it; a 1% scale error is ten times it.
POLICY_SCALE_STEP = 0.001
# --router: the rule that judges routing flips.
ROUTE_RULE = "route-near-tie.v2"
DTYPES = {"bf16": "<u2", "f32": "<f4", "f64": "<f8", "u32": "<u4", "u8": "u1"}


def compare_kind(name, dtype, declared=None):
    if declared:
        return declared
    if name.endswith(".ids") or name.endswith("logits.topk"):
        return "set"
    if name.endswith(".x.codes"):
        return "codes"
    if name.endswith("router.weights"):
        return "paired"
    return "dense" if dtype in ("bf16", "f32", "f64") else "discrete"


def read(path, entry):
    import numpy as np
    raw = np.fromfile(path, dtype=DTYPES[entry["dtype"]])
    if raw.size != entry["rows"] * entry["cols"]:
        raise ValueError(f"{path} holds {raw.size} values, index declares {entry['rows']}x{entry['cols']}")
    if entry["dtype"] == "bf16":
        raw = (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.reshape(entry["rows"], entry["cols"])


def reducible(sharding):
    return sharding in ("replicated", "rank_partial") or isinstance(sharding, dict) and (
        isinstance(sharding.get("col_shards"), list) or "blocks" in (sharding.get("col_shards") or {}))


class RankCopiesDiffer(ValueError):
    """Ranks that compute a tap alike hold different copies of it."""


def reduce(parts, sharding, ranks):
    """One [rows, cols] array from the rank arrays of a tap."""
    import numpy as np
    if sharding == "replicated":
        # A capture that holds several ranks' copies holds equal ones.
        for rank, part in enumerate(parts[1:], 1):
            if part.tobytes() != parts[0].tobytes():
                raise RankCopiesDiffer(f"rank {rank}'s copy differs from rank 0's")
        return parts[0]
    if len(parts) != ranks:
        raise ValueError(f"{sharding} tap has ranks {len(parts)} of {ranks}")
    if sharding == "rank_partial":
        total = parts[0].astype(np.float32)
        for part in parts[1:]:
            total = total + part.astype(np.float32)
        return total
    if isinstance(sharding, dict) and "col_shards" in sharding:
        blocks = sharding["col_shards"]
        layout = head_blocks(blocks["blocks"], ranks) if isinstance(blocks, dict) else blocks
        width = max(start + count for rank_blocks in layout for start, count in rank_blocks)
        out = np.zeros((parts[0].shape[0], width), dtype=np.float32)
        covered = np.zeros(width, dtype=np.int64)
        for part, rank_blocks in zip(parts, layout):
            offset = 0
            for start, count in rank_blocks:
                out[:, start:start + count] = part[:, offset:offset + count]
                covered[start:start + count] += 1
                offset += count
            if offset != part.shape[1]:
                raise ValueError(f"rank shard has {part.shape[1]} columns, blocks cover {offset}")
        if not (covered == 1).all():
            raise ValueError("column shards overlap or leave columns uncovered")
        return out
    raise ValueError(f"unsupported sharding {sharding!r}")


def head_blocks(blocks, ranks):
    """Per rank, the (start, count) column blocks of its share of every
    [heads, dim] block of a row."""
    layout, start = [[] for _ in range(ranks)], 0
    for heads, dim in blocks:
        if heads % ranks:
            raise ValueError(f"{heads} heads do not split over {ranks} ranks")
        count = heads // ranks * dim
        for rank in range(ranks):
            layout[rank].append((start + rank * count, count))
        start += heads * dim
    return layout


class Step:
    """One step directory: its index, items and the row key of each stored
    row: (request, position) for sequence rows, (request, ("enc", kind,
    item, index))
    for an encoder tap's rows (`"axis": "encoder"` in the index, segments in
    items.json `encoder`)."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.capture = json.loads((self.dir / "capture.json").read_text())
        self.ranks = self.capture.get("ranks", 1)
        items = json.loads((self.dir / "items.json").read_text())
        self.index = {}
        for line in (self.dir / "index.jsonl").read_text().splitlines():
            entry = json.loads(line)
            parts = self.index.setdefault(entry["name"], {})
            if entry["rank"] in parts or not 0 <= entry["rank"] < self.ranks:
                raise ValueError(f"{self.dir}: duplicate or invalid rank of {entry['name']}")
            parts[entry["rank"]] = entry
        self.rows = {}  # phase row -> (request, position, token)
        for item in items["items"]:
            if item.get("request") is None:
                continue
            for offset, (position, token) in enumerate(zip(item["positions"] or [], item["tokens"] or [])):
                self.rows[item["start"] + offset] = (item["request"], position, token)
        self.encoder = {}  # encoder tap -> {tensor row: (request, ("enc", kind, item, index))}
        self.malformed = {}  # encoder tap -> why its segments are not a map
        for name, segments in items.get("encoder", {}).items():
            rows = self.encoder[name] = {}
            for segment in segments:
                if segment.get("kind") is None or segment.get("item") is None:
                    self.malformed.setdefault(name, f"{name} segment at row {segment['start']} names no kind or item")
                    continue
                media = ("enc", segment["kind"], segment["item"])
                for offset in range(segment["rows"]):
                    row, key = segment["start"] + offset, (segment["request"], (*media, segment["index"] + offset))
                    if row in rows or key in rows.values():
                        self.malformed.setdefault(name, f"{name} segments overlap at row {row} or index {key}")
                    rows[row] = key
        self.requests = {key[0] for key in self.rows.values()} | {
            key[0] for rows in self.encoder.values() for key in rows.values()}
        # The rows every sequence tap holds: the declared dump span, or the items' rows.
        dump = items.get("dump_rows")
        self.span = tuple(dump) if dump else (min(self.rows), max(self.rows) + 1) if self.rows else (0, 0)
        expected = self.capture.get("expected")
        self.declared = {e["name"]: e.get("sharding") for e in expected} if isinstance(expected, list) else {}
        self.compare = {e["name"]: e.get("compare") for e in expected} if isinstance(expected, list) else {}
        self.missing = self.coverage(expected)

    def encoder_tap(self, name):
        return any(entry.get("axis") == "encoder" for entry in self.index[name].values())

    def coverage(self, expected):
        """Why the step is incomplete (its completion record, a declared tap
        the index lacks, or a tap holding fewer rows than the step declares),
        or None when it is complete."""
        path = self.dir / "completion.json"
        if not path.exists():
            return "no completion.json"
        completion = json.loads(path.read_text())
        if completion.get("completion") != "full":
            return f"completion {completion.get('completion')!r}: {completion.get('reason')}"
        stored = {(name, rank) for name, parts in self.index.items() for rank in parts}
        if isinstance(expected, list):
            absent = [e["name"] for e in expected if (e["name"], e.get("rank", 0)) not in stored]
            if absent:
                return f"declared taps absent from the index: {absent[:4]}"
        elif isinstance(expected, int) and expected != len(stored):
            return f"index holds {len(stored)} tensors, capture expects {expected}"
        for name, parts in self.index.items():
            if self.encoder_tap(name):
                # Segments map every stored row, and only stored rows.
                rows = self.encoder.get(name)
                if not rows:
                    return f"{name} holds encoder rows and items.json declares none"
                if name in self.malformed:
                    return self.malformed[name]
                for entry in parts.values():
                    stored_rows = set(range(entry["first_row"], entry["first_row"] + entry["rows"]))
                    if set(rows) != stored_rows:
                        return (f"{name} rank {entry['rank']} stores {len(stored_rows)} rows, its segments map"
                                f" {len(set(rows) & stored_rows)} of them and {len(set(rows) - stored_rows)} others")
                continue
            start, end = self.span
            for entry in parts.values():
                if entry["first_row"] > start or entry["first_row"] + entry["rows"] < end:
                    return (f"{name} rank {entry['rank']} holds rows {entry['first_row']}.."
                            f"{entry['first_row'] + entry['rows']} of {start}..{end}")
        return None

    def key(self, name, row):
        """The key of a tensor row of a stored tap, or None."""
        if self.encoder_tap(name):
            return self.encoder.get(name, {}).get(row)
        key = self.rows.get(row)
        return None if key is None else key[:2]

    def tensor(self, name, sharding):
        """(first phase row, [rows, cols] array) of a reduced tap."""
        entries = [self.index[name][rank] for rank in sorted(self.index[name])]
        if [e["rank"] for e in entries] != list(range(len(entries))):
            raise ValueError(f"{self.dir}: {name} ranks are not 0..n")
        first = {(e["first_row"], e["rows"]) for e in entries}
        if len(first) != 1:
            raise ValueError(f"{self.dir}: {name} ranks cover different rows")
        parts = [read(self.dir / f"{name}.r{e['rank']}.bin", e) for e in entries]
        return entries[0]["first_row"], reduce(parts, sharding, self.ranks), entries[0]["dtype"]


class TapDir:
    """A tap directory, with engine names translated to canonical ones."""

    def __init__(self, root, declarations=(), steps=None):
        self.root = Path(root)
        dirs = sorted(d for d in self.root.glob("step*") if (d / "index.jsonl").exists())
        if steps is not None:
            dirs = [d for d in dirs if any(fnmatch.fnmatch(d.name, s) for s in steps)]
        if not dirs:
            raise ValueError(f"{root}: no step directories")
        self.steps = [Step(d) for d in dirs]
        self.names = {}  # canonical name -> [(step, stored name, sharding)]
        self.order = []
        self.compare = {}  # name -> declared comparison
        for step in self.steps:
            schema = step.capture.get("schema")
            model = tap_names.v1_declaration(schema)
            if model is None and schema != tap_names.SCHEMA:
                raise ValueError(f"{step.dir}: no tap declaration (models/*/taps.json) reads capture schema "
                                 f"{schema!r}, which holds {', '.join(sorted(step.index)[:4])}")
            owner = None
            if model is None:
                check_index(step)
                if step.capture.get("model") is not None:
                    owner = tap_names.model_declaration(step.capture["model"])
                    if owner is None:
                        raise ValueError(f"{step.dir}: no tap declaration (models/*/taps.json) for model "
                                         f"{step.capture['model']!r} of capture schema {schema}")
            for stored, parts in step.index.items():
                if owner:
                    # An engine capture: its model declares each tap.
                    entry = owner.entry(stored)
                    if entry is None:
                        raise ValueError(f"{step.dir}: model {owner.model}'s declaration ({owner.path}) does not "
                                         f"declare {stored}")
                    check(step, stored, parts, entry, schema)
                    if step.declared.get(stored) != entry["sharding"]:
                        raise ValueError(f"{step.dir}: {stored} has sharding {step.declared.get(stored)!r} in "
                                         f"capture.json, its declaration {entry['sharding']!r}")
                if model:
                    # An older dump: the model's declaration names its taps.
                    mapped = model.canonical(stored)
                    if mapped is None:
                        raise ValueError(f"{step.dir}: capture schema {schema} ({model.path}) does not "
                                         f"declare {stored}")
                    name, entry = mapped
                    check(step, stored, parts, entry, schema)
                    sharding, self.compare[name] = entry["sharding"], entry["compare"]
                else:
                    name = stored
                    if step.compare.get(stored):
                        self.compare[name] = step.compare[stored]
                    sharding = step.declared.get(stored) or declared(declarations, name, "sharding")
                    if sharding is None:
                        if len(parts) > 1 or step.ranks > 1:
                            raise ValueError(f"{step.dir}: {stored} has {len(parts)} of {step.ranks} ranks "
                                             f"and no declared sharding (capture schema {schema})")
                        sharding = "replicated"
                if name not in self.names:
                    self.names[name] = []
                    self.order.append(name)
                self.names[name].append((step, stored, sharding))
        self.tokens = {}  # (request, position) -> token
        self.where = {}  # (request, position) -> [(step, phase row)]
        for step in self.steps:
            for row, (request, position, token) in step.rows.items():
                key = (request, position)
                if self.tokens.setdefault(key, token) != token:
                    raise ValueError(f"{self.root}: request {request} position {position} has two tokens")
                self.where.setdefault(key, []).append((step, row))
        self.encoder_names = {name for name, entries in self.names.items()
                              if any(step.encoder_tap(stored) for step, stored, _ in entries)}
        # Every row key: sequence positions and encoder rows.
        self.keys = set(self.where) | {key for step in self.steps for rows in step.encoder.values()
                                       for key in rows.values()}

    def encoder_keys(self, name, requests):
        """The (request, ("enc", kind, item, index)) keys an encoder tap's steps declare
        for `requests`."""
        return {key for step, stored, _ in self.names.get(name, []) for key in step.encoder.get(stored, {}).values()
                if key[0] in requests}

    def rows(self, name, keys):
        """{key: row vector} of a tap at the keys it covers; a key that two
        steps cover is refused."""
        found, dtype = {}, None
        for step, stored, sharding in self.names.get(name, []):
            first, values, dtype = step.tensor(stored, sharding)
            for row in range(values.shape[0]):
                key = step.key(stored, first + row)
                if key is None or key not in keys:
                    continue
                if key in found:
                    raise ValueError(f"{self.root}: {name} at request {key[0]} row {key[1]} is in two steps"
                                     " (restrict with --steps)")
                found[key] = values[row]
        return found, dtype


def check_index(step):
    """Refuses a chopinfer.taps.v1 step whose index holds a tap and rank
    capture.json does not declare, or whose declared dtype, columns or rows
    differ from the index."""
    expected = step.capture.get("expected")
    if not isinstance(expected, list):
        raise ValueError(f"{step.dir}: capture.json declares no expected taps")
    declared = {}
    for entry in expected:
        key = (entry["name"], entry.get("rank", 0))
        if key in declared:
            raise ValueError(f"{step.dir}: capture.json declares {key[0]} rank {key[1]} twice")
        declared[key] = entry
    for name, parts in step.index.items():
        for rank, part in sorted(parts.items()):
            entry = declared.get((name, rank))
            if entry is None:
                raise ValueError(f"{step.dir}: {name} rank {rank} is in the index but not in capture.json "
                                 "expected")
            for field in ("dtype", "cols", "first_row", "rows"):
                if field in entry and entry[field] != part[field]:
                    raise ValueError(f"{step.dir}: {name} rank {rank} has {field} {part[field]}, capture.json "
                                     f"declares {entry[field]}")


def check(step, stored, parts, entry, schema):
    """Refuses a tap of an older dump whose dtype, ranks or columns differ
    from its declaration."""
    where = f"{step.dir}: {stored} (capture schema {schema})"
    dtypes = sorted({part["dtype"] for part in parts.values()})
    if dtypes != [entry["dtype"]]:
        raise ValueError(f"{where} has dtype {dtypes}, its declaration {entry['dtype']}")
    sharding = entry["sharding"]
    ranks = list(range(step.ranks)) if sharding != "replicated" or entry.get("ranks") == "every" else [0]
    if sorted(parts) != ranks:
        raise ValueError(f"{where} has ranks {sorted(parts)}, its declaration {ranks}")
    if "cols" in entry:
        cols = entry["cols"] // step.ranks if isinstance(sharding, dict) else entry["cols"]
        widths = sorted({part["cols"] for part in parts.values()})
        if widths != [cols]:
            raise ValueError(f"{where} has {widths} columns, its declaration {cols} per rank")


def declared(declarations, name, field):
    for entry in declarations:
        if fnmatch.fnmatchcase(name, entry["name"]) and field in entry:
            return entry[field]
    return None


def match_requests(engine, reference, explicit):
    """{engine request: reference request} by tokens at shared positions."""
    by_request = {}
    for (request, position), token in engine.tokens.items():
        by_request.setdefault(request, {})[position] = token
    sequences = {}
    for (request, position), token in reference.tokens.items():
        sequences.setdefault(request, {})[position] = token
    mapping, ignored = {}, []
    for request, positions in sorted(by_request.items(), key=lambda item: str(item[0])):
        if str(request) in explicit:
            mapping[request] = explicit[str(request)]
            continue
        matches = [name for name, tokens in sequences.items()
                   if (shared := positions.keys() & tokens.keys())
                   and all(positions[p] == tokens[p] for p in shared)]
        if len(matches) > 1:
            raise ValueError(f"engine request {request} matches sequences {sorted(matches)}; use --request")
        if matches:
            mapping[request] = matches[0]
        else:
            ignored.append(request)
    # An encoder reference's sequences hold encoder rows and no positions.
    held = {request for step in reference.steps for request in step.requests}
    for request, name in mapping.items():
        if name not in held:
            raise ValueError(f"--request names {name}, which the reference does not hold")
        tokens = sequences.get(name, {})
        if any(tokens.get(p, t) != t for p, t in by_request[request].items()):
            raise ValueError(f"engine request {request} disagrees with {name} on tokens")
    return mapping, ignored


def sequence_tokens(tapdir, workload):
    """Gives each sequence of a runner `workload` that `tapdir`'s steps hold
    without positions (encoder rows only) its tokens, for request matching."""
    requests = {request for step in tapdir.steps for request in step.requests}
    positioned = {request for request, _ in tapdir.tokens}
    for sequence in workload["sequences"]:
        name = sequence["name"]
        if name in requests and name not in positioned:
            tapdir.tokens.update({(name, position): token for position, token in enumerate(sequence["token_ids"])})


def relative(values, reference):
    """Per-row relative L2 of values against a finite reference. A zero
    reference row gives 0 for a zero row and inf otherwise; a row with a
    nonfinite value gives inf."""
    import numpy as np
    if not np.isfinite(reference).all():
        raise ValueError("the reference holds nonfinite values")
    with np.errstate(over="ignore", invalid="ignore"):
        num = np.sqrt(np.sum((values - reference) ** 2, axis=1))
    den = np.sqrt(np.sum(reference ** 2, axis=1))
    out = np.where(num == 0, 0.0, np.inf)
    nonzero = den > 0
    out[nonzero] = num[nonzero] / den[nonzero]
    out[~np.isfinite(values).all(axis=1)] = np.inf
    return out


def stats(values):
    """Median, p90 and max of the non-NaN values of a nonnegative measure,
    JSON-safe: infinities are "inf", and quantiles never interpolate
    against one."""
    import numpy as np
    values = values[~np.isnan(values)]
    if not values.size:
        return None
    top = np.finfo(np.float64).max
    clipped = np.minimum(values, top)

    def safe(value):
        return "inf" if value >= top else float(value)
    with np.errstate(over="ignore"):
        out = {"median": safe(np.median(clipped)), "p90": safe(np.quantile(clipped, 0.9)),
               "max": safe(clipped.max())}
    if np.isinf(values).any():
        out["inf_rows"] = int(np.isinf(values).sum())
    return out


def stack(found, keys, mapping=None):
    import numpy as np
    return np.stack([found[mapping(k) if mapping else k] for k in keys]).astype(np.float64)


class Rows:
    """The rows of one tap at the compared keys, per producer: the engine's,
    f32's (every key) and the optional arms' (the keys they cover)."""

    def __init__(self, engine, reference, keys, aligned):
        import numpy as np
        self.keys = keys
        self.engine = stack(engine, keys, aligned.get)
        self.f32 = stack(reference["f32"], keys)
        self.arms = {}
        for mode in ("bf16", "f64"):
            if mode in reference:
                covered = np.array([key in reference[mode] for key in keys])
                values = np.full_like(self.f32, np.nan)
                if covered.any():
                    values[covered] = stack(reference[mode], [k for k, c in zip(keys, covered) if c])
                self.arms[mode] = (covered, values)

    def arm(self, mode):
        """(covered mask, rows with NaN where uncovered) of an optional arm."""
        import numpy as np
        return self.arms.get(mode, (np.zeros(len(self.keys), dtype=bool), np.full_like(self.f32, np.nan)))


def by_ids(weights, ids):
    """Each row's weights in ascending order of its expert ids."""
    import numpy as np
    order = np.argsort(ids, axis=1, kind="stable")
    return np.take_along_axis(weights, order, axis=1), np.take_along_axis(ids, order, axis=1)


def paired_ids(weights, ids):
    """Why an arm's router ids cannot index its weights (another width,
    non-integer, negative or repeated ids in a row), or None."""
    import numpy as np
    arms = [("engine", weights.engine, ids.engine), ("f32", weights.f32, ids.f32)]
    for mode in weights.arms:
        covered, values = weights.arm(mode)
        id_covered, id_values = ids.arm(mode)
        both = covered & id_covered
        arms.append((mode, values[both], id_values[both]))
    for label, values, arm_ids in arms:
        if values.shape != arm_ids.shape:
            return f"{label} holds {arm_ids.shape[1]} router ids for {values.shape[1]} weights"
        if (arm_ids < 0).any() or (arm_ids != np.floor(arm_ids)).any():
            return f"{label} router ids are not expert indices"
        ordered = np.sort(arm_ids, axis=1)
        if (ordered[:, 1:] == ordered[:, :-1]).any():
            return f"{label} router ids repeat an expert in a row"
    return None


def aggregate_rule(row, E, F, bf16, has, ratio, limit, row_floor):
    """Holds the whole tap: its relative L2 over every compared row against
    ratio times bf16's over the rows bf16 covers, raised to the row floor
    (`limit` without bf16 rows). Returns whether it exceeds."""
    import numpy as np
    norm = np.linalg.norm(F)
    error = np.linalg.norm(E - F) if np.isfinite(E).all() else np.inf
    e = float(error / norm) if norm > 0 else (0.0 if error == 0 else np.inf)
    floor = None
    if has.any():
        reference = np.linalg.norm(F[has])
        floor = float(np.linalg.norm(bf16 - F[has]) / reference) if reference > 0 else 0.0
    bound = ratio * max(floor, row_floor) if floor is not None else limit
    row["aggregate"] = {"e": e, "floor": floor, "bound": bound, "passed": bool(e <= bound)}
    return not e <= bound


def dense(row, rows, ratio, limit, valid=None, row_floor=None, aggregate=False):
    """Scores dense rows into `row`; returns each row's exceed flag. With a
    `row_floor`, every row is held to ratio times its floor, raised to the
    row floor, as well as the median. With `aggregate`, the whole tap is
    held too (`aggregate_rule`, raised to ROW_FLOOR)."""
    import numpy as np
    valid = np.ones(len(rows.keys), dtype=bool) if valid is None else valid
    E, F = rows.engine[valid], rows.f32[valid]
    e = relative(E, F)
    covered, bf16 = rows.arm("bf16")
    floor = np.full(len(e), np.nan)
    has = covered[valid]
    if has.any():
        floor[has] = relative(bf16[valid][has], F[has])
    floored = np.isfinite(floor) & (floor > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = np.where(floored, e / (ratio * floor), e / limit)
        to_floor = np.where(floored, e / floor, np.nan)
    row |= {"e": stats(e), "floor": stats(floor), "e_over_floor": stats(to_floor), "excess": stats(excess),
            "unfloored_rows": int((~floored).sum()), "bf16_rows": int(has.sum())}
    covered, f64 = rows.arm("f64")
    if covered.any():
        with_f64 = covered[valid]
        row["num"] = stats(relative(F[with_f64], f64[valid][with_f64])) if with_f64.any() else None
        row["f64_rows"] = int(with_f64.sum())
    row["fail"] = bool(np.median(excess) > 1) if len(e) else False
    if aggregate and len(e):
        bf16_rows = bf16[valid][has] if has.any() else None
        row["fail"] = aggregate_rule(row, E, F, bf16_rows, has, ratio, limit, ROW_FLOOR) or row["fail"]
    if row_floor is not None and len(e):
        with np.errstate(divide="ignore", invalid="ignore"):
            excess = np.where(has, e / (ratio * np.maximum(np.where(has, floor, 0.0), row_floor)), e / limit)
        row["every_row"] = {"row_floor": row_floor, "excess": stats(excess), "exceeding": int((excess > 1).sum())}
        row["fail"] = row["fail"] or bool((excess > 1).any())
    exceeds = np.zeros(len(rows.keys), dtype=bool)
    exceeds[valid] = excess > 1
    return exceeds


def every_row_rule(text):
    """(tap glob, codes tap, group) of an --every-row value `GLOB` or
    `GLOB:CODES/GROUP`."""
    glob, _, codes = text.partition(":")
    if not codes:
        return glob, None, None
    name, _, group = codes.partition("/")
    if not group.isdecimal() or int(group) < 1:
        raise ValueError(f"--every-row {text}: expected GLOB:CODES_TAP/GROUP")
    return glob, name, int(group)


def matched_codes(keys, codes, group):
    """Per encoder row key, whether the `group` codes rows it merges (rows
    group * i .. group * i + group - 1 of its item that the reference
    holds) all equal f32's in the engine."""
    import numpy as np
    found, _, reference, aligned, _ = codes
    equal = {key: key in aligned and np.array_equal(found[aligned[key]], reference["f32"][key])
             for key in reference["f32"]}
    held = []
    for request, (axis, kind, item, index) in keys:
        merged = [(request, (axis, kind, item, group * index + offset)) for offset in range(group)]
        merged = [key for key in merged if key in equal]
        held.append(bool(merged) and all(equal[key] for key in merged))
    return np.array(held, dtype=bool)


def rounded_up(value, step):
    """`value` rounded up to a multiple of `step`, both read as their shortest
    decimals."""
    step = Decimal(str(step))
    quotient, remainder = divmod(Decimal(str(value)), step)
    return float((quotient + (1 if remainder > 0 else 0)) * step)


def unit_rows(values):
    """Rows over their L2 norms; a zero row stays zero."""
    import numpy as np
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norms > 0, values / norms, 0.0)


def direction_scale(row, rows, keys, variants, ratio, limit, row_floor, step=SCALE_STEP, policy=None):
    """Scores an encoder tap's rows by direction and scale into `row` (see
    the module documentation); `variants` maps a --scale-variant name to its
    rows at `keys`. With `policy` (a variant's name) only that variant's
    scale errors set an item's limit, at least POLICY_SCALE_STEP and rounded
    up to it; the others are reported. Returns each row's exceed flag."""
    import numpy as np
    E, F = rows.engine, rows.f32
    if not np.isfinite(F).all():
        raise ValueError("the reference holds nonfinite values")
    norm_f = np.linalg.norm(F, axis=1)
    d = np.linalg.norm(unit_rows(E) - unit_rows(F), axis=1)
    d[~np.isfinite(E).all(axis=1)] = np.inf
    covered, bf16 = rows.arm("bf16")
    floor = np.full(len(d), np.nan)
    if covered.any():
        floor[covered] = np.linalg.norm(unit_rows(bf16[covered]) - unit_rows(F[covered]), axis=1)
    floored = np.isfinite(floor) & (floor > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = np.where(floored, d / (ratio * floor), d / limit)
        each = np.where(covered, d / (ratio * np.maximum(np.where(covered, floor, 0.0), row_floor)), d / limit)
        to_floor = np.where(floored, d / floor, np.nan)

        def scale(values):
            import numpy as np
            return np.abs(np.linalg.norm(values, axis=1) / norm_f - 1)
        engine_scale = np.where(norm_f > 0, scale(E), np.where(np.linalg.norm(E, axis=1) > 0, np.inf, 0.0))
        spreads = {"bf16": np.where(covered, scale(np.where(covered[:, None], bf16, F)), np.nan)}
        spreads |= {name: scale(values) for name, values in variants.items()}
    items = {}
    for index, (request, (_, kind, item, _)) in enumerate(keys):
        items.setdefault((str(request), kind, item), []).append(index)
    scales, scale_fail = {}, False
    for (request, kind, item), indices in sorted(items.items()):
        largest = {name: float(np.nanmax(values[indices])) if np.isfinite(values[indices]).any() else None
                   for name, values in spreads.items()}
        if policy is None:
            known = [value for value in largest.values() if value is not None]
            bound = rounded_up(max(known), step) if known else None
        else:
            own = largest.get(policy)
            bound = rounded_up(max(own, POLICY_SCALE_STEP), POLICY_SCALE_STEP) if own is not None else None
        worst = float(engine_scale[indices].max())
        passed = bound is not None and worst <= bound
        scale_fail |= not passed
        scales[f"{request}/{kind}/{item}"] = {"engine": worst, "engine_median": float(np.median(engine_scale[indices])),
                                              **largest, "limit": bound, "passed": passed}
    row["scale_rule"] = {"limit_from": [policy] if policy else sorted(spreads),
                         "step": POLICY_SCALE_STEP if policy else step}
    row |= {"direction": {"e": stats(d), "floor": stats(floor), "e_over_floor": stats(to_floor),
                          "excess": stats(excess), "every_row": {"row_floor": row_floor, "excess": stats(each),
                                                                 "exceeding": int((each > 1).sum())}},
            "scale": scales, "bf16_rows": int(covered.sum())}
    row["e"], row["floor"], row["e_over_floor"] = row["direction"]["e"], row["direction"]["floor"], \
        row["direction"]["e_over_floor"]
    row["fail"] = bool(np.median(excess) > 1 or (each > 1).any() or scale_fail) if len(d) else False
    return (each > 1) | np.array([not scales[f"{r!s}/{k}/{i}"]["passed"] for r, (_, k, i, _) in keys])


def codes(row, rows, margin):
    """Scores integer codes into `row`: the share of codes equal to f32's,
    over all and over the first column, against bf16's on the same rows.
    Returns each row's exceed flag (any code differs)."""
    E, F = rows.engine, rows.f32
    equal = E == F
    row |= {"agreement": float(equal.mean()), "stage0_agreement": float(equal[:, 0].mean()),
            "disagree": float(1 - equal.mean()), "codes_margin": margin}
    covered, bf16 = rows.arm("bf16")
    floor = {"agreement": 1.0, "stage0_agreement": 1.0}
    if covered.any():
        same = bf16[covered] == F[covered]
        floor = {"agreement": float(same.mean()), "stage0_agreement": float(same[:, 0].mean())}
        row |= {"bf16_agreement": floor["agreement"], "bf16_stage0_agreement": floor["stage0_agreement"],
                "bf16_disagree": float(1 - same.mean()), "bf16_rows": int(covered.sum())}
    row["fail"] = any(row[key] < floor[key] - margin for key in floor)
    return ~equal.all(axis=1)


def selection_scores(spec, logits, bias):
    """The scores a router selects on, in binary64: sigmoid or softmax of
    the logits, plus the correction bias where the router adds one."""
    import numpy as np
    x = np.asarray(logits, dtype=np.float64)
    with np.errstate(over="ignore"):
        if spec["score"] == "sigmoid":
            scores = 1.0 / (1.0 + np.exp(-x))
        else:
            shifted = np.exp(x - x.max(axis=1, keepdims=True))
            scores = shifted / shifted.sum(axis=1, keepdims=True)
    return scores if bias is None else scores + np.asarray(bias, dtype=np.float64)


def router_problem(spec, biases):
    """Why a --router spec and its biases cannot judge routes, or None."""
    if not isinstance(spec, dict) or spec.get("score") not in ("sigmoid", "softmax"):
        return "the router spec names no score function (sigmoid or softmax)"
    if not isinstance(spec.get("top_k"), int) or isinstance(spec["top_k"], bool) or spec["top_k"] < 1:
        return "the router spec names no top_k"
    if spec.get("n_group", 1) != 1 or spec.get("topk_group", 1) != 1:
        return "a group-limited router needs its group mask tapped (router.groups); routes are not judged"
    if spec.get("bias") and biases is None:
        return "the router adds a correction bias and no --router-bias was given"
    return None


def routes(engine, variants, gather, spec, biases, ratio):
    """route-near-tie.v2: the routing flips of every MoE layer (the engine's
    `router.ids` set differs from F32's), each judged a near tie or
    decisive, and the rows behind a near-tie first flip. Returns (report,
    {key: program-order index of the router.ids tap of the key's first flip}
    for keys whose first flip is a near tie).

    A flip's gap is the largest pair it reversed under F32's selection
    scores: max over dropped experts minus min over added ones. A layer's
    margin is ratio times its spread S: the largest movement of F32's
    k-th/(k+1)-th boundary gap in the BF16 run, over rows the BF16 run has
    not routed differently at an earlier layer, cumulative over the layers
    so far, raised to two F32 ulps of the layer's largest score. A flip is a
    near tie when its gap is at most the margin. A key's first flip decides
    it; later flips there are counted, not judged again. Anything missing
    leaves routes unjudged: every flip is then decisive (today's rules)."""
    import numpy as np
    order = variants["f32"].order
    report = {"rule": ROUTE_RULE, "router": spec, "status": "judged", "layers": {}, "flips": [],
              "bf16_first_flips_beyond_margin": []}

    def unjudged(reason):
        return {**report, "status": "unjudged", "reason": reason, "layers": {}, "flips": []}, {}
    problem = router_problem(spec, biases)
    if problem:
        return unjudged(problem)
    if "bf16" not in variants:
        return unjudged("no bf16 run: the margin is the BF16 run's boundary movement")
    layers = [(index, name[:-len("ids")]) for index, name in enumerate(order) if name.endswith(".router.ids")]
    if not layers:
        return unjudged("the reference taps no router.ids")
    top_k = spec["top_k"]
    spread = None
    bf16_behind, first, behind = set(), {}, {}
    for index, prefix in layers:
        ids_name, logits_name = prefix + "ids", prefix + "logits"
        label = prefix[:-len(".router.")] if prefix.endswith(".router.") else prefix[:-1]
        held = [ids_name in variants[mode].names and logits_name in variants[mode].names for mode in ("f32", "bf16")]
        if not all(held) or ids_name not in engine.names:
            return unjudged(f"{label}: the engine taps no router.ids or a reference run no router.ids "
                            "and router.logits")
        match = re.fullmatch(r"L(\d+)\.router\.", prefix)
        layer = int(match.group(1)) if match else None
        bias = None
        if spec.get("bias"):
            bias = biases.get(layer) if layer is not None else None
            if bias is None:
                return unjudged(f"the router biases hold no layer for {label}")
        ids_found, _, ids_reference, ids_aligned, keys = gather(ids_name)
        # The gap and the margin are the reference's own: the engine's
        # logits are not read.
        reference_logits = {mode: variants[mode].rows(logits_name, set(keys))[0] for mode in ("f32", "bf16")}
        if not keys or any(key not in reference_logits["f32"] for key in keys):
            return unjudged(f"{label}: the F32 run's router.logits lack rows its router.ids hold")
        ids = Rows(ids_found, ids_reference, keys, ids_aligned)
        logits_f32 = stack(reference_logits["f32"], keys)
        experts = logits_f32.shape[1]
        if bias is not None and len(bias) != experts:
            return unjudged(f"the router biases of {label} hold {len(bias)} experts, the logits {experts}")
        if experts <= top_k or ids.f32.shape[1] != top_k:
            return unjudged(f"{label}: {ids.f32.shape[1]} ids of {experts} experts, top_k {top_k}")
        if not np.isfinite(logits_f32).all():
            return unjudged(f"{label}: the F32 router logits hold nonfinite values")
        ids_covered, bf16_ids = ids.arm("bf16")
        covered = ids_covered & np.array([key in reference_logits["bf16"] for key in keys])
        bf16_logits = np.zeros_like(logits_f32)
        if covered.any():
            bf16_logits[covered] = stack(reference_logits["bf16"], [k for k, c in zip(keys, covered) if c])
        cf = selection_scores(spec, logits_f32, bias)
        cb = selection_scores(spec, bf16_logits, bias)
        if not (np.isfinite(cf).all() and np.isfinite(cb[covered]).all()):
            return unjudged(f"{label}: nonfinite selection scores")
        moves, bf16_flipped, flips = [], [], []
        for i, key in enumerate(keys):
            ranked = np.argsort(-cf[i], kind="stable")
            kth, after = ranked[top_k - 1], ranked[top_k]
            reference_ids = set(int(v) for v in ids.f32[i])
            if covered[i]:
                if key not in bf16_behind:
                    moves.append(abs((cb[i, kth] - cb[i, after]) - (cf[i, kth] - cf[i, after])))
                if set(int(v) for v in bf16_ids[i]) != reference_ids:
                    bf16_flipped.append(i)
            engine_ids = ids.engine[i]
            if set(int(v) for v in engine_ids) != reference_ids:
                flips.append(i)
        if moves:
            spread = max(spread or 0.0, float(max(moves)))
        resolution = 2.0 * float(np.spacing(np.float32(np.abs(cf).max())))
        margin = ratio * max(spread, resolution) if spread is not None else None

        def gap(i, chosen):
            """The largest pair `chosen` reversed under F32's scores (inf
            when it names an expert outside the logits)."""
            reference_ids = set(int(v) for v in ids.f32[i])
            if not all(float(v).is_integer() and 0 <= v < experts for v in chosen):
                return float("inf")
            chosen = set(int(v) for v in chosen)
            return float(max(cf[i, d] for d in reference_ids - chosen) - min(cf[i, a] for a in chosen - reference_ids))
        for i in flips:
            key, value = keys[i], gap(i, ids.engine[i])
            near = margin is not None and np.isfinite(value) and value <= margin
            is_first = key not in first
            if is_first:
                first[key] = near
                if near:
                    behind[key] = index
            report["flips"].append({"layer": label, "sequence": key[0], "position": key[1], "gap": value,
                                    "margin": margin, "near_tie": bool(near) if is_first else None,
                                    "first": is_first})
        for i in bf16_flipped:
            if keys[i] not in bf16_behind:
                value = gap(i, bf16_ids[i])
                if margin is None or not value <= margin:
                    report["bf16_first_flips_beyond_margin"].append(
                        {"layer": label, "sequence": keys[i][0], "position": keys[i][1], "gap": value,
                         "margin": margin})
                bf16_behind.add(keys[i])
        report["layers"][label] = {"spread": spread, "resolution": resolution, "margin": margin,
                                         "rounding_only_rows": len(moves), "flips": len(flips),
                                         "bf16_flips": len(bf16_flipped)}
    report["first_flips"] = {"near_tie": sum(first.values()), "decisive": sum(not v for v in first.values())}
    for flip in report["flips"]:
        if not np.isfinite(flip["gap"]):
            flip["gap"] = "inf"
    return report, behind


def aggregate_only(row):
    """Whether a dense tap fails its whole-tap aggregate and nothing else."""
    median = (row.get("excess") or {}).get("median")
    return (row.get("fail") and (row.get("aggregate") or {}).get("passed") is False
            and isinstance(median, float) and median <= 1 and not (row.get("every_row") or {}).get("exceeding"))


def plausible(values, reference, ratio):
    """route-near-tie.v2's guard on a row an explanation removes (D16):
    every value finite, its norm within a factor of ratio of the reference
    row's (inclusive) and its cosine with the reference row positive. A row
    with a non-finite value, a norm outside that range (a zeroed row among
    them) or a cosine at or below 0 does not pass. The ratio and cosine are
    None where undefined."""
    import numpy as np
    values, reference = np.asarray(values, np.float64), np.asarray(reference, np.float64)
    if not (np.isfinite(values).all() and np.isfinite(reference).all()):
        return {"plausible": False, "norm_ratio": None, "cosine": None}
    norm, base = float(np.linalg.norm(values)), float(np.linalg.norm(reference))
    cosine = float(values @ reference) / (norm * base) if norm and base else None
    return {"plausible": bool(norm <= ratio * base and base <= ratio * norm and cosine is not None and cosine > 0),
            "norm_ratio": norm / base if base else None, "cosine": cosine}


def explain(row, rows, valid, behind, index, ratio, limit):
    """route-near-tie.v2's explanation of a tap that fails only its
    aggregate (option B, minimal exclusion): remove rows behind a near-tie
    first flip at an earlier router, largest squared error first, until the
    aggregate over the rows left (and the BF16 run's over the same rows)
    passes. Each removed row must be plausible (finite, its norm within a
    factor of ratio of the F32 row's, a positive cosine with it) and within
    ratio times the BF16 run's worst row at this tap (raised to ROW_FLOOR),
    else the tap is not explained.
    Records `route_explanation`; an explained tap no longer fails."""
    import numpy as np
    valid = np.ones(len(rows.keys), dtype=bool) if valid is None else valid
    keys = [key for key, held in zip(rows.keys, valid) if held]
    E, F = rows.engine[valid], rows.f32[valid]
    covered, bf16 = rows.arm("bf16")
    has, B = covered[valid], bf16[valid]
    outcome = {"rule": ROUTE_RULE, "explained": False, "removed": [], "row_bound": None}
    row["route_explanation"] = outcome
    if not has.any():
        outcome["reason"] = "no bf16 rows: no per-row bound"
        return
    e = relative(E, F)
    floor = relative(B[has], F[has])
    outcome["row_bound"] = bound = ratio * max(float(floor.max()), ROW_FLOOR)
    with np.errstate(over="ignore", invalid="ignore"):
        error = np.where(np.isfinite(E).all(axis=1), np.sum((E - F) ** 2, axis=1), np.inf)
    candidates = sorted((i for i, key in enumerate(keys) if behind.get(key, index) < index),
                        key=lambda i: (-error[i], i))
    keep = np.ones(len(keys), dtype=bool)
    for i in candidates:
        removed = {"sequence": keys[i][0], "position": keys[i][1],
                   "e": float(e[i]) if np.isfinite(e[i]) else "inf", "bound": bound} | plausible(E[i], F[i], ratio)
        outcome["removed"].append(removed)
        if not removed["plausible"]:
            outcome["reason"] = f"row {keys[i][1]} fails the plausibility guard"
            return
        if not e[i] <= bound:
            outcome["reason"] = f"row {keys[i][1]} exceeds the per-row bound"
            return
        keep[i] = False
        trial = {}
        exceeds = aggregate_rule(trial, E[keep], F[keep], B[keep & has] if (keep & has).any() else None,
                                 has[keep], ratio, limit, ROW_FLOOR)
        outcome["aggregate"] = trial["aggregate"] | {"rows": int(keep.sum())}
        if not exceeds:
            outcome["explained"] = True
            row["fail"] = False
            return
    outcome["reason"] = ("no row behind a near-tie flip" if not candidates
                         else "the aggregate fails without every row behind a near-tie flip")


def compare(engine, variants, mapping, declarations=(), ratio=RATIO, limit=LIMIT, every_row=(),
            row_floor=ROW_FLOOR, codes_margin=CODES_MARGIN, direction_globs=(), scale_variants=None,
            aggregate_globs=(), scale_policy=None, router=None):
    """Per-tap rows in the f32 reference's program order, the first
    exceeding tap of each (sequence, position) and its histogram, per
    encoder tap its exceeding rows, and the route judgement (None without
    `router`, the (spec, biases) of --router). Dense taps matching an
    `every_row` glob are also held row by row."""
    import numpy as np
    reference_keys = variants["f32"].keys
    engine_keys = {key for key in engine.keys if key[0] in mapping}
    taps, first, encoder = [], {}, {}

    def gather(name):
        """(engine rows, dtype, per-mode reference rows, {reference key:
        engine key}, keys the engine and f32 share)."""
        reference = {mode: tapdir.rows(name, reference_keys)[0] for mode, tapdir in variants.items()}
        found, dtype = engine.rows(name, engine_keys)
        aligned = {}
        for request, position in found:
            if aligned.setdefault((mapping[request], position), (request, position)) != (request, position):
                raise ValueError(f"two engine requests hold {mapping[request]} position {position}")
        return found, dtype, reference, aligned, sorted(k for k in aligned if k in reference["f32"])

    route_report, behind = (routes(engine, variants, gather, *router, ratio) if router is not None
                            else (None, {}))
    for index, name in enumerate(variants["f32"].order):
        row = {"tap": name}
        taps.append(row)
        if name not in engine.names:
            row["status"] = "reference only"
            continue
        missing = None
        if name in variants["f32"].encoder_names:
            # An encoder tap promises every output row: the engine's rows
            # must be the reference's, whatever steps they come in.
            if name not in engine.encoder_names:
                row["status"] = "incompatible: the engine stores encoder rows as sequence rows"
                continue
            expected = variants["f32"].encoder_keys(name, set(mapping.values()))
            observed = {(mapping[request], index) for request, index in engine.encoder_keys(name, set(mapping))}
            row["encoder_rows"] = {"expected": len(expected), "observed": len(observed),
                                   "missing": len(expected - observed), "unexpected": len(observed - expected)}
            if observed - expected:
                row["status"] = (f"incompatible: the engine holds {len(observed - expected)} encoder rows"
                                 f" the reference does not ({len(observed)} observed, {len(expected)} expected)")
                continue
            if expected - observed:
                missing = f"incomplete: the engine holds {len(observed)} of {len(expected)} encoder rows"
        shardings = [sharding for _, _, sharding in engine.names[name]]
        if not all(map(reducible, shardings)):
            unreduced = next(s for s in shardings if not reducible(s))
            row["status"] = f"not compared: the engine holds {json.dumps(unreduced)} shards"
            continue
        try:
            found, dtype, reference, aligned, keys = gather(name)
        except RankCopiesDiffer as error:
            row.update(status=f"fail: {error}", fail=True)
            continue
        row["rows"] = len(keys)
        if not keys:
            row["status"] = missing or "no shared rows"
            continue
        rows = Rows(found, reference, keys, aligned)
        if rows.engine.shape != rows.f32.shape:
            row["status"] = f"incompatible: engine {rows.engine.shape[1]} columns, reference {rows.f32.shape[1]}"
            continue
        kind = row["compare"] = compare_kind(
            name, dtype, declared(declarations, name, "compare") or engine.compare.get(name))
        rule = next((rule for rule in map(every_row_rule, every_row) if fnmatch.fnmatchcase(name, rule[0])), None)
        whole = any(fnmatch.fnmatchcase(name, glob) for glob in aggregate_globs)
        strict, held = (row_floor, None) if rule else (None, None)
        if rule and rule[1] and kind == "dense":
            codes_name, group = rule[1:]
            if codes_name not in engine.names or codes_name not in variants["f32"].names:
                row["status"] = f"incompatible: {name}'s every-row rule needs {codes_name}"
                continue
            held = matched_codes(keys, gather(codes_name), group)
            row["code_flipped_rows"] = int((~held).sum())
        if kind == "dense" and any(fnmatch.fnmatchcase(name, glob) for glob in direction_globs):
            if name not in variants["f32"].encoder_names:
                row["status"] = f"incompatible: --direction-scale names {name}, which holds no encoder rows"
                continue
            scale_rows = {}
            for label, tapdir in (scale_variants or {}).items():
                found = tapdir.rows(name, reference_keys)[0]
                if any(key not in found for key in keys):
                    row["status"] = f"incompatible: scale variant {label} lacks rows of {name}"
                    break
                scale_rows[label] = stack(found, keys)
            else:
                if scale_policy is not None and scale_policy not in scale_rows:
                    row["status"] = f"incompatible: --scale-policy {scale_policy} is no --scale-variant"
                    continue
                exceeds = direction_scale(row, rows, keys, scale_rows, ratio, limit, row_floor,
                                          policy=scale_policy)
            if row.get("status"):
                continue
        elif kind == "dense":
            exceeds = dense(row, rows, ratio, limit, held, strict, whole)
            if whole and behind and aggregate_only(row):
                explain(row, rows, held, behind, index, ratio, limit)
        elif kind == "codes":
            exceeds = codes(row, rows, codes_margin)
        elif kind == "paired":
            ids_name = name[:-len("weights")] + "ids"
            if ids_name not in engine.names or ids_name not in variants["f32"].names:
                row["status"] = f"incompatible: {name} without {ids_name}"
                continue
            ids_found, _, ids_reference, ids_aligned, _ = gather(ids_name)
            if any(ids_aligned.get(k) != aligned[k] or k not in ids_reference["f32"] for k in keys):
                row["status"] = f"incompatible: {ids_name} lacks rows {name} holds"
                continue
            ids = Rows(ids_found, ids_reference, keys, ids_aligned)
            problem = paired_ids(rows, ids)
            if problem:
                row["status"] = f"incompatible: {problem}"
                continue
            rows.engine, engine_ids = by_ids(rows.engine, ids.engine)
            rows.f32, f32_ids = by_ids(rows.f32, ids.f32)
            same = (engine_ids == f32_ids).all(axis=1)
            for mode, (covered, values) in list(rows.arms.items()):
                arm_covered, arm_ids = ids.arm(mode)
                covered = covered & arm_covered
                values[covered], ordered = by_ids(values[covered], arm_ids[covered])
                # A bf16 or f64 row routed elsewhere has no weight floor.
                covered[covered] = (ordered == f32_ids[covered]).all(axis=1)
                rows.arms[mode] = (covered, values)
            row["route_flips"] = int((~same).sum())
            exceeds = (dense(row, rows, ratio, limit, same, strict, whole) if same.any()
                       else np.zeros(len(keys), dtype=bool))
            row.setdefault("fail", False)
        else:
            E, F = rows.engine, rows.f32

            def differ(values, reference):
                import numpy as np
                if kind == "set":
                    return np.array([set(a.tolist()) != set(b.tolist()) for a, b in zip(values, reference)])
                return (values != reference).any(axis=1)
            exceeds = differ(E, F)
            row["disagree"] = float(exceeds.mean())
            covered, bf16 = rows.arm("bf16")
            if covered.any():
                row["bf16_disagree"] = float(differ(bf16[covered], F[covered]).mean())
                row["bf16_rows"] = int(covered.sum())
            floor = row.get("bf16_disagree", 0.0)
            row["fail"] = kind == "set" and row["disagree"] > (ratio * floor if floor > 0 else limit)
            if kind == "discrete":
                exceeds = np.zeros(len(keys), dtype=bool)
        row["status"] = "fail" if row["fail"] else missing or "ok"
        if name in variants["f32"].encoder_names:
            encoder[name] = int(np.count_nonzero(exceeds))
            continue
        for key, bad in zip(keys, exceeds):
            if bad and key not in first:
                first[key] = name
    histogram = {}
    for name in first.values():
        histogram[name] = histogram.get(name, 0) + 1
    return taps, first, histogram, encoder, route_report


def identities(tapdir):
    out = {}
    for step in tapdir.steps:
        for file in ("capture.json", "items.json", "index.jsonl"):
            path = step.dir / file
            out[f"{step.dir.name}/{file}"] = diagnostic.sha256(path.read_bytes())
    return out


def fmt(value, key="median"):
    if not value:
        return f"{'-':>10}"
    return f"{value[key]:>10}" if isinstance(value[key], str) else f"{value[key]:10.4g}"


def table(taps):
    lines = [f"{'tap':28s} {'rows':>6} {'e med':>10} {'e p90':>10} {'floor med':>10} {'e/fl med':>10}"
             f" {'e/fl p90':>10} {'e/fl max':>10} {'num med':>10}  status"]
    for row in taps:
        if "e" in row:
            cells = (f"{fmt(row['e'])} {fmt(row['e'], 'p90')} {fmt(row['floor'])} {fmt(row['e_over_floor'])}"
                     f" {fmt(row['e_over_floor'], 'p90')} {fmt(row['e_over_floor'], 'max')} {fmt(row.get('num'))}")
        elif "disagree" in row:
            floor = row.get("bf16_disagree", float("nan"))
            cells = f"{row['disagree']:10.4g} {'':10} {floor:10.4g} {'':10} {'':10} {'':10} {'':10}"
        else:
            cells = " " * 76
        lines.append(f"{row['tap']:28s} {row.get('rows', 0):6d} {cells}  {row['status']}")
    return "\n".join(lines)


def run(args):
    for text in args.every_row:
        every_row_rule(text)
    declarations = json.loads(Path(args.declare).read_text())["taps"] if args.declare else []
    steps = args.steps.split(",") if args.steps else None
    engine = TapDir(args.engine, declarations, steps)
    variants = {"f32": TapDir(args.f32)}
    for mode in ("bf16", "f64"):
        if getattr(args, mode):
            variants[mode] = TapDir(getattr(args, mode))
    for mode, tapdir in variants.items():
        if mode != "f32" and any(variants["f32"].tokens.get(k, t) != t for k, t in tapdir.tokens.items()):
            raise ValueError(f"the {mode} taps disagree with f32 on tokens")
    for path in args.workload:
        workload = json.loads(Path(path).read_text())
        for tapdir in (engine, variants["f32"]):
            sequence_tokens(tapdir, workload)
    explicit = dict(item.split("=", 1) for item in args.request)
    mapping, ignored = match_requests(engine, variants["f32"], explicit)
    if not mapping:
        raise ValueError("no engine request matches a reference sequence")
    scale_variants = {}
    for text in args.scale_variant:
        label, _, directory = text.partition("=")
        if not label or not directory:
            raise ValueError(f"--scale-variant {text}: expected NAME=DIR")
        scale_variants[label] = TapDir(directory)
    router = None
    if args.router is not None:
        spec = json.loads(args.router)
        biases = None
        if args.router_bias is not None:
            from tools.qual import router_bias
            biases = router_bias.load(Path(args.router_bias).read_bytes())[1]
        router = (spec, biases)
    taps, first, histogram, exceeding, route_report = compare(
        engine, variants, mapping, declarations, args.ratio, args.limit, args.every_row, args.row_floor,
        args.codes_margin, args.direction_scale, scale_variants, args.aggregate, args.scale_policy, router)
    failing = next((row["tap"] for row in taps if row.get("fail")), None)
    incompatible = [row["tap"] for row in taps if row["status"].startswith("incompatible")]
    positions = sum(1 for key in variants["f32"].where if key[0] in mapping.values())
    encoder_rows = {row["tap"]: {"reference": row["encoder_rows"]["expected"], "exceeding": exceeding.get(row["tap"], 0)}
                    for row in taps if "encoder_rows" in row}
    incomplete = {label: {step.dir.name: step.missing for step in tapdir.steps
                          if step.missing and step.requests & set(requests)}
                  for label, tapdir, requests in [("engine", engine, mapping)] +
                  [(mode, tapdir, set(mapping.values())) for mode, tapdir in variants.items()]}
    incomplete = {label: steps for label, steps in incomplete.items() if steps}
    short = {row["tap"]: row["status"] for row in taps if row["status"].startswith("incomplete")}
    if short:
        incomplete["taps"] = short
    status = ("incompatible" if incompatible else "diverged" if failing else "incomplete" if incomplete
              else "insufficient" if not any(row.get("rows") for row in taps) else "match")
    report = {"schema": SCHEMA, "status": status, "first": failing, "ratio": args.ratio, "limit": args.limit,
              "every_row": list(args.every_row), "row_floor": args.row_floor, "codes_margin": args.codes_margin,
              "direction_scale": list(args.direction_scale), "scale_step": SCALE_STEP,
              "aggregate": list(args.aggregate), "scale_policy": args.scale_policy,
              "scale_variants": {label: str(tapdir.root) for label, tapdir in scale_variants.items()},
              "requests": {str(k): v for k, v in mapping.items()}, "ignored_requests": [str(r) for r in ignored],
              "taps": taps, "first_exceeding": histogram,
              "positions": {"reference": positions, "with_exceeding_tap": len(first)},
              "encoder_rows": encoder_rows,
              "engine_only": sorted(set(engine.order) - set(variants["f32"].order)),
              "incomplete_steps": incomplete, "routes": route_report,
              "inputs": {"engine": {"dir": str(args.engine), "files": identities(engine)},
                         **{mode: {"dir": str(getattr(args, mode)), "files": identities(tapdir)}
                            for mode, tapdir in variants.items()}}}
    print(table(taps))
    print(f"\nfirst exceeding tap per position ({len(first)} of {positions} positions exceed somewhere):")
    for name, count in sorted(histogram.items(), key=lambda item: -item[1]):
        print(f"  {count:6d}  {name}")
    for name, rows in encoder_rows.items():
        print(f"{name}: {rows['exceeding']} of {rows['reference']} encoder rows exceed")
    for label, steps in incomplete.items():
        for step, reason in steps.items():
            print(f"incomplete {label} {step}: {reason}")
    if route_report is not None:
        print(f"routes: {route_report['status']}" + (f" ({route_report['reason']})" if route_report.get("reason")
                                                     else ""))
    for row in taps:
        if "route_explanation" in row:
            outcome = row["route_explanation"]
            print(f"{row['tap']}: aggregate {'explained' if outcome['explained'] else 'not explained'} by route "
                  f"near ties, {len(outcome['removed'])} rows removed"
                  + (f" ({outcome['reason']})" if outcome.get("reason") else ""))
    message = {"match": "no tap outside the floor", "diverged": f"first tap outside the floor is {failing}",
               "incompatible": ", ".join(incompatible), "incomplete": "a compared step's capture is incomplete",
               "insufficient": "no rows compared"}[status]
    print(f"\n{status}: {message}")
    if args.json:
        diagnostic.write_new(Path(args.json).absolute(), report)
    return {"match": 0, "diverged": 1, "incompatible": 2, "incomplete": 3, "insufficient": 3}[status]


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True, type=Path, help="engine tap or dump directory")
    parser.add_argument("--f32", required=True, type=Path, help="reference f32 taps")
    parser.add_argument("--bf16", type=Path, help="reference bf16 taps (the floor)")
    parser.add_argument("--f64", type=Path, help="reference f64 taps")
    parser.add_argument("--request", action="append", default=[], metavar="ENGINE=SEQUENCE")
    parser.add_argument("--steps", help="comma-separated engine step globs to read (default all)")
    parser.add_argument("--declare", help="tap declarations: {\"taps\": [{\"name\": glob, \"sharding\", \"compare\"}]}")
    parser.add_argument("--ratio", type=float, default=RATIO)
    parser.add_argument("--limit", type=float, default=LIMIT)
    parser.add_argument("--every-row", action="append", default=[], metavar="GLOB[:CODES/GROUP]",
                        help="dense taps also held row by row (repeatable); with :CODES/GROUP only rows "
                             "whose merged codes match")
    parser.add_argument("--row-floor", type=float, default=ROW_FLOOR, help="--every-row's least row floor")
    parser.add_argument("--codes-margin", type=float, default=CODES_MARGIN,
                        help="how far a codes tap's agreement may sit below bf16's")
    parser.add_argument("--direction-scale", action="append", default=[], metavar="GLOB",
                        help="encoder taps judged by row direction and per-item scale (repeatable)")
    parser.add_argument("--scale-variant", action="append", default=[], metavar="NAME=DIR",
                        help="another BF16 variant's taps whose scale errors join the bf16 run's")
    parser.add_argument("--scale-policy", metavar="NAME",
                        help="the --scale-variant of an F32-policy engine: only its scale errors set the limit")
    parser.add_argument("--aggregate", action="append", default=[], metavar="GLOB",
                        help="dense taps also held as a whole by relative L2 (repeatable)")
    parser.add_argument("--workload", action="append", default=[],
                        help="runner workload whose token IDs name encoder-only reference sequences")
    parser.add_argument("--router", metavar="JSON",
                        help="the model's router: {\"score\": \"sigmoid\"|\"softmax\", \"top_k\", \"n_group\", "
                             "\"topk_group\", \"bias\"}; judges routing flips by route-near-tie.v2")
    parser.add_argument("--router-bias", metavar="FILE",
                        help="the router correction biases (tools/qual/router_bias.py sidecar)")
    parser.add_argument("--json", help="write the receipt here (never overwritten)")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        return run(parse_args(sys.argv[1:] if argv is None else argv))
    except ValueError as error:
        print(f"incompatible: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
