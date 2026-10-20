The drafter processes at most six requests per forward, keeping five rows
per request and at most 30 rows per forward. This avoids the dense and MoE
dispatch change above 64 rows and the small head boundary above 32 rows.
Every position is still computed. Target sampling, rejection, RNG
advancement and publication are unchanged.

`CHOPINFER_DSPARK_DRAFT_CHUNK_ITEMS=32` selects the original whole batch for
comparison. Values from 1 through 32 are accepted. Chunk outputs occupy a
separate 1,280-byte buffer per rank. All copies and graph replays use the
existing rank streams. Each chunk retains its outputs before staging is
reused; final restoration precedes the existing drafter join and target.

Exact-width startup warmup covers 97 target keys at a 32-row budget:
all reachable widths at the ordinary concurrency waves, the widest key
at other concurrencies where every request can fit a draft, and witnesses
for each drafter count. The complete reachable set has 426 keys and would
need 1,278 target graph handles per rank before other graphs. The existing
384-handle limit stays in effect. Unselected keys remain lazy.

The diagnostic build can compare both drafter implementations on identical
inputs with `CHOPINFER_DSPARK_DRAFT_CHECK` naming an existing marker file.
It compares all five proposal IDs and all five confidence bit patterns on
every rank, failing before target verification on any mismatch. Removing
the marker disables the check permanently at the next chunked forward.
The comparison reads back outputs and is only for qualification.

`dspark_cost_job.py` uses one server per policy and chunk size, four servers
and eight C8/C16 sampled cases in total. Production verify warmup is on.
Candidate startup and full-workload warmup check drafter equality; the
marker is removed before timing. Served IDs and logprobs also compare
bitwise. The job records graph hits, capture-memory deltas, major/minor
faults, power and host-idle receipts. Timing is informational when the
host-idle rule fails.

The comparison job explicitly uses 512 graph handles per rank on both
arms. The first real startup filled the 384-handle default and denied one
additional catch-up graph per rank. Request completion tails created
additional target keys beyond the scheduled warmup witnesses. The saved
startup contains 385 unique handles per rank across retirement, about
906 MiB of capture allocations and at least 3.40 GiB free on each GPU.
The job retains the zero-retirement assertion through startup, workload
warmup and timing, and refuses more than 2 GiB of observed capture
allocations per rank. This qualification override does not change the
serving default.

The memory report distinguishes a fixed graph-handle limit from observed
CUDA allocation bytes. Driver graph memory has no fixed byte size per
handle. Fake-host results prove lifecycle, spans and graph coverage; the
staged served comparison must establish device bitwise equality and speed.
