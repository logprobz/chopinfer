"""Check the KV offload model against contract sizes and snapshot page counts."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import kv_offload_model as model  # noqa: E402


class KvOffloadModelTests(unittest.TestCase):
    def test_page_bytes_match_the_bundle_contracts(self):
        # GA_PAGE_BYTES_BF16 and GA_PAGE_BYTES_FP8 in models/mimo26/bundle/src/contract.rs.
        self.assertEqual(model.page_bytes("mimo", 4, "bf16"), 368_640)
        self.assertEqual(model.page_bytes("mimo", 4, "fp8"), 184_320)
        # PAGE_BYTES_BF16 and the FP8 branch's PAGE_BYTES_FP8 in
        # models/qwen38/bundle/src/contract.rs.
        self.assertEqual(model.page_bytes("qwen", 4, "bf16"), 835_584)
        self.assertEqual(model.page_bytes("qwen", 4, "fp8"), 442_368)
        # MiMo TP2's independent expected size: 11,520 B per token and rank.
        self.assertEqual(model.page_bytes("mimo", 2, "bf16"), 64 * 11_520)

    def test_snapshot_sizes_match_decision_0085_and_q_prefix(self):
        # prefix-cache contract, "What a bundle snapshots".
        self.assertEqual(model.snapshot_bytes("mimo", 4, "mtp"), 7_438_336)
        self.assertEqual(model.snapshot_bytes("mimo", 4, "dflash"), 16_865_280)
        self.assertEqual(model.snapshot_pages("mimo", 4, "bf16", "mtp"), 21)
        self.assertEqual(model.snapshot_pages("mimo", 4, "fp8", "mtp"), 41)
        self.assertEqual(model.snapshot_pages("mimo", 4, "bf16", "dflash"), 46)
        self.assertEqual(model.snapshot_pages("mimo", 4, "fp8", "dflash"), 92)
        # Q-PREFIX's describe_prefix_state_v1 reports 35 BF16 pages.
        self.assertEqual(model.snapshot_pages("qwen", 4, "bf16"), 35)
        # The expected per-slot state layout plus one ring copy.
        self.assertEqual(
            model.snapshot_bytes("qwen", 4),
            29_048_960 + model.QWEN_RING_COPY,
        )

    def test_request_bytes_grow_with_context_and_budgets_hold_them(self):
        for _, name, ranks, element, drafter in model.CONFIGURATIONS:
            sizes = [model.host_bytes(name, ranks, element, c, drafter) for c in model.CONTEXTS]
            self.assertEqual(sizes, sorted(sizes))
            per_token = ranks * model.page_bytes(name, ranks, element) // model.PAGE_TOKENS
            # Whole pages: never less than the tokens' bytes.
            self.assertGreaterEqual(sizes[0], model.CONTEXTS[0] * per_token)

    def test_swaps_beat_recompute_beyond_a_small_context(self):
        self.assertAlmostEqual(model.PCIE5_X16_RAW / 1e9, 63.0, places=1)
        for _, name, ranks, element, drafter in model.CONFIGURATIONS:
            for rate in model.PCIE_PLANNING:
                tokens = model.break_even_tokens(name, ranks, element, drafter, rate)
                self.assertLess(tokens, model.PAGE_TOKENS)
                copy = 2 * model.copy_seconds(name, ranks, element, tokens, drafter, rate)
                self.assertLess(copy, model.recompute_seconds(name, ranks, tokens))

    def test_main_prints_every_configuration(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            model.main()
        for name, *_ in model.CONFIGURATIONS:
            self.assertEqual(output.getvalue().count(f"| {name} |"), 3, name)


if __name__ == "__main__":
    unittest.main()
