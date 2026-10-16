"""Measure MiMo audio encoder spread between captures.

The 24-layer audio tokenizer turns BF16 rounding noise into discrete RVQ
code flips, so correct implementations disagree on a few percent of stage-0
codes. This tool measures how far encoder outputs are from each other. The
reference spread of the encoder output comes from the GPU reference
runner's `enc.audio.out` tap in its f32, bf16 and f64 modes
(tools/qual/tapdiff.py); the CPU f64 oracle chains this tool once read are
retired.

- `spread NAME=DIR ...` prints pairwise stage-0/all-stage code agreement and
  embedding row cosines among the supplied runner capture directories.
- `layers A B` compares two runner capture directories written with
  CHOPINFER_ENCODER_CAPTURE_ALL_LAYERS=1, layer by layer: the fraction of
  BF16 values that differ and the relative L2 difference. Equal directories
  show a deterministic encoder; the first differing layer locates a change.
- `prepare --out DIR NAME=DIR ...` writes BF16 embedding and i32 code files
  for every source, ready for a diagnostic server that reads
  CHOPINFER_DIAG_AUDIO_OVERRIDE (a control file naming the override). The
  switch does not exist on main; it is a local diagnostic.

The repository holds no audio reference case (the model's audio is served
but not qualified in-tree), so there is no first-token probe of frozen
audio cases.
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from array import array
import glob
import json
import math

from encoder_parity import (bf16_bits, native, read_bf16, read_i32,
                            row_cosines)

STAGES = 20
WIDTH = 4096


def load_sources(pairs):
    """name -> (codes list, embeddings list) from explicit runner outputs."""
    sources = {}
    for pair in pairs:
        name, _, directory = pair.partition("=")
        codes = glob.glob(f"{directory}/audio-codes-*.bin")
        output = Path(directory) / "arena-output.bf16"
        if not output.exists():
            output = Path(glob.glob(f"{directory}/audio-output-*.bin")[0])
        if len(codes) != 1:
            raise SystemExit(f"{directory}: expected one audio-codes file")
        sources[name] = (list(read_i32(codes[0])), list(read_bf16(output)))
    return sources


def pair_metrics(a, b):
    codes = len(a[0]) // STAGES
    cosines = row_cosines(a[1], b[1], len(a[1]) // WIDTH, WIDTH)
    return {"stage0": sum(a[0][r * STAGES] == b[0][r * STAGES] for r in range(codes)) / codes,
            "all_stages": sum(x == y for x, y in zip(a[0], b[0])) / len(a[0]),
            "mean_row_cosine": sum(cosines) / len(cosines), "min_row_cosine": min(cosines)}


def spread(args):
    sources = load_sources(args.sources)
    names = list(sources)
    rows = {}
    print(f"{'pair':<28} stage0  all    mean_cos  min_cos")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = rows[f"{a}_vs_{b}"] = pair_metrics(sources[a], sources[b])
            print(f"{a + ' vs ' + b:<28} {m['stage0']:.4f} {m['all_stages']:.4f} "
                  f"{m['mean_row_cosine']:.6f} {m['min_row_cosine']:.6f}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return 0


def layers(args):
    first, second = Path(args.first), Path(args.second)
    names = sorted(p.name for p in first.glob("*.bin"))
    order = lambda name: [int(t) if t.isdigit() else t for t in
                          name.replace("-", " ").replace("layer", "layer ").split()]
    identical = True
    for name in sorted(names, key=order):
        other = second / name
        if not other.exists():
            print(f"{name}: missing in {second}")
            identical = False
            continue
        if "codes" in name:
            a, b = read_i32(first / name), read_i32(other)
            differ = sum(x != y for x, y in zip(a, b))
            print(f"{name}: {differ}/{len(a)} codes differ")
        else:
            a, b = read_bf16(first / name), read_bf16(other)
            differ = sum(x != y for x, y in zip(a, b))
            norm = math.sqrt(sum(x * x for x in a)) or 1.0
            relative = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / norm
            print(f"{name}: {differ}/{len(a)} values differ, relative L2 {relative:.3e}")
        identical &= differ == 0
    print("identical" if identical else "different")
    return 0 if identical else 1


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = load_sources(args.sources)
    index = {}
    for name, (codes, embeddings) in sources.items():
        code_path, embedding_path = out / f"{name}.codes.i32", out / f"{name}.embeddings.bf16"
        code_path.write_bytes(native(array("i", codes)).tobytes())
        embedding_path.write_bytes(native(bf16_bits(embeddings)).tobytes())
        index[name] = {"codes": str(code_path), "embeddings": str(embedding_path)}
    (out / "sources.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    (out / "control").write_text("")
    print(f"export CHOPINFER_DIAG_AUDIO_OVERRIDE={out / 'control'} CHOPINFER_DIAG_NO_MEDIA_CACHE=1")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("spread", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("sources", nargs="*", metavar="NAME=DIR")
        if name == "spread":
            command.add_argument("--json")
        else:
            command.add_argument("--out", required=True)
    compare = commands.add_parser("layers")
    compare.add_argument("first")
    compare.add_argument("second")
    args = parser.parse_args()
    return {"spread": spread, "layers": layers, "prepare": prepare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
