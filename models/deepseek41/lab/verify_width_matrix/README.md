This CPU-staged job compares speculative widths on one frozen installation.
It does not queue work or change a serving default.

The provisional matrix starts seven servers: K0, balanced with maximum K1,
K2 and K3, fixed K5, adaptive K5, and balanced K5. Every server allows sixteen
requests and runs C8 and C16, alternating cohort order between servers.
All cases use temperature 1.0, top_p 1.0 and identical seeded diverse prompts.
Both device results and the resident row cache are explicitly enabled.

The existing budget is 32 total target rows. True uniform K2 and K3 at C16
need 48 and 64 rows and remain unsupported. The K2/K3-cap servers still run
C16, but their results are labeled as budgeted configurations; the balanced
planner normally selects K1 there. No case is silently presented as a wider
verify test.

Each cohort first warms 64 output tokens, then records 192 tokens per request.
The measured response itself supplies the exactness comparison, including
selected and top logprobs as binary64 bits. Token-string collisions and the
sampled token's extra top-logprob key follow the existing completion parser.
Startup verification warmup is disabled to avoid warming every possible
shape in seven processes. Workload warmups, at least twelve full-concurrency
steps, exact row reporting, device acceptance packets and both resident
cache sets are required. The first four full steps are excluded from the
step-time summary. All request/step data and process fault counts are saved.

Warmup and timed windows report the actual zero-verify fraction and the
policy's reason counts: missing AR baseline, estimated loss, realized loss,
cooldown and probes. Rates use all full-concurrency steps as the denominator;
per-plan rates and steps without a planner invocation are also explicit.
Each traced decision includes the cost table, AR estimate, candidate estimate,
realized rates and probe state before training on that step. A zero-verify
adaptive window is valid evidence. Missing diagnostics are an error.

The follow-up confirmation reads the last successful real matrix receipt,
checks its SHA-256, and selects the configuration with the largest geometric
mean throughput ratio to adaptive serving across C8 and C16. Its four servers
run adaptive, winner, winner, adaptive. The result recommends a default only
when both winner repetitions beat both adaptive repetitions at both
concurrencies, every output matches bitwise, and every measured host window
passes the existing idle-host rule. No recommendation is made for noisy or
overlapping repetitions. Defaults are changed separately after reviewing
this confirmation.

Both jobs have a 1200-second whole-machine limit and set all four visible GPUs
themselves. The runner uses a shared 1140-second execution deadline, bounded
requests, and a final cleanup reserve. It checks 250 W caps, idle GPUs before
startup, kernel Xids, owned process cleanup and positive case counts. A matrix
passes only with fourteen measured cases; confirmation needs eight. A failed
run never publishes a successful matrix pointer.

CPU checks:

```bash
python3 -B models/deepseek41/lab/verify_width_matrix/test_job.py
python3 -B models/deepseek41/lab/verify_width_matrix/test_job.py --native "$STAGE"
python3 -B models/deepseek41/lab/verify_width_matrix/test_job.py --replay "$RECORDED_X6_ARM"
python3 -B models/deepseek41/lab/verify_width_matrix/test_job.py --dry-run "$CPU_OUTPUT" token logprob inactive zero-steps
```

Run these through the shared CPU queue. The replay uses the recorded normal
capacity X6 C8 arm. The HTTP fixture executes both job modes without GPU
access and deliberately rejects wrong tokens, one-bit logprob changes,
missing device packets and zero step coverage. Unit tests also cover signal
interruption and deadline cleanup. It is selected only by the
explicit CPU dry-run flag and is never named in either GPU job launcher.

After preparing one native installation, `stage.py --configure` records its
paths and copies this harness. `test_job.py --native` validates every server
argument with that frozen binary's CPU-only option listing. The standard
checkpoint stamp verifier must pass before sealing. `stage.py --timings`
checks the time budget against a prior recorded startup and C8 cohort;
`--seal` requires passing CPU and timeout receipts, hashes the installation
and harness, and writes the matrix and confirmation job files. It does not
submit them. The confirmation refuses a missing, modified or failed matrix.
