# DSpark early Engram test default

The early Engram gather already defaults on in serving. This change makes
bundle tests use that same default. No production launch selection,
arithmetic, kernel lock or served envelope changes.

The six TP4 verify fixtures add these resources during instance creation:

| Resource | Count | Size |
|---|---:|---:|
| Rank-zero early ID stream | 1 | n/a |
| Rank-zero early ID buffer | 1 | 36,864 bytes |
| Early row scratch | 1 per rank | 405,504 bytes per rank |
| Prepared layer gathers | 2 | n/a |

The buffers have 256-byte alignment. Each positional declaration binds the
resource's exact issue ordinal, creator records, rank, size and stream.
Every retained operation uses one ordered typed identity map. No operation
edit, module change or allocation resize is declared. The other ten pinned
journals keep their complete histories unchanged.

These verify fixtures receive tokens from the host, so they prepare the
early gathers without reading the added buffers before their pinned
checkpoints. The DSpark early-gather tests cover drafted steps separately:
they check each uploaded ID against host hashes, early submission before
the drafter, both gathers' joins at each layer consumer, completion at the
boundary and balanced resource release. Graph and round-trip fixtures
locate the late target gathers after the device hash.

The declaration's trace positions must be regenerated when preceding
resource creation changes. Pin updates require an exact operation audit;
CPU journals establish structure, not GPU numerical or timing evidence.
