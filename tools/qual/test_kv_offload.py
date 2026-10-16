from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import subprocess
import unittest
from unittest.mock import patch

import kv_offload as kvo


FACTS = {"kv_offload": {"budget_gib": 16, "swap_every": 16, "max_preemptions": 3,
                       "min_gated": 18, "pressure_tokens": 32768}, "depth": 3}


def metric_text(offload=True):
    text = "# ignored help\nother_histogram_sum 0.25\n"
    for name in kvo.COUNTERS.values():
        if offload or not name.startswith("chop_kv_offload_"):
            text += f"{name} 0\n"
    if offload:
        text += ('chop_kv_offload_host_pages{kind="used"} 0\n'
                 'chop_kv_offload_host_pages{kind="free"} 1024\n'
                 'chop_kv_offload_swapped_requests 0\n'
                 'chop_kv_offload_fallback_total{reason="capture"} 0\n')
    return text


class Offload(unittest.TestCase):
    def test_arm_overrides_replace_configuration_options_without_duplicates(self):
        import release_preview as runner
        configuration = {"page_tokens": 64, "serve": ["--speculative-tokens", "5", "--context-parallel-size", "2"]}
        arms = kvo.arms(FACTS, configuration)
        plain = runner.serve_overrides(configuration["serve"], arms["kv-offload-plain-off"])
        self.assertEqual(plain.count("--speculative-tokens"), 1)
        self.assertEqual(plain[plain.index("--speculative-tokens") + 1], "0")
        pressure = runner.serve_overrides(configuration["serve"], arms["kv-offload-pressure-swap"])
        self.assertEqual(pressure[pressure.index("--speculative-tokens") + 1], "5")
        self.assertEqual(pressure[pressure.index("--kv-pages") + 1], "256")

    def test_development_selection_is_explicit_and_cannot_promote(self):
        import release_preview as runner
        facts = {"full": ["first", "second"], "quick": ["first"]}
        config = {"default": {"screens": True}}
        chosen, configs = runner.select_checks(facts, config, "full", ["second", "second"])
        self.assertEqual(runner.configuration_facts(chosen, "default")["full"], ["second"])
        self.assertEqual(runner.configuration_facts(chosen, "default")["quick"], ["first"])
        self.assertFalse(configs["default"]["screens"])
        self.assertTrue(config["default"]["screens"])
        with self.assertRaisesRegex(ValueError, "outside"):
            runner.select_checks(facts, config, "quick", ["second"])
        store = Namespace(name="last-good", model="test")
        self.assertIn("subset", runner.Store.update(store, None, {"selected_checks": ["second"]}))

    def test_runner_resolves_callable_arms_and_keeps_them_out_of_quick(self):
        import release_preview as runner
        facts = {**FACTS, "model": "test", "full": ["offload_plain", "offload_pressure"], "quick": [], "checks": {
            f"offload_{scenario}": {"module": "kv_offload", "files": [
                f"kv_offload_{scenario}_off", f"kv_offload_{scenario}_swap"]}
            for scenario in ("plain", "pressure")}}
        config = {"serve": [], "features": [], "screens": False, "page_tokens": 64, "context_parallel_size": 4}
        full = runner.plan(facts, "full", "default", config)
        self.assertEqual(len(full), 5)
        pressure = next(arm for arm in full if arm["arm"] == "kv-offload-pressure-swap")
        self.assertIn("128", pressure["serve"])
        self.assertEqual(runner.plan(facts, "quick", "default", config), [])
        values = runner.predicate_values({"gates": [{"name": "c1_gated", "value": 28,
                                          "op": "ge", "limit": 18}]}, [])
        self.assertEqual(values[0]["value"], 28)

    def test_live_worker_and_replay_use_the_rust_rule(self):
        snapshot = dict.fromkeys((*kvo.COUNTERS, "host_pages_used", "swapped_requests"), 0)
        snapshot["fallbacks"] = {}
        requests = [{"name": f"item-{i}", "prompt_sha256": f"prompt-{i}", "token_ids": [1] * 32,
                     "tokens": ["a"] * 32, "logprobs": [-0.5] * 32, "top": [{"a": -0.5}] * 32,
                     "finish_reason": "length", "usage": {"prompt_tokens": 10, "completion_tokens": 32,
                     "total_tokens": 42}, "observed": deepcopy(snapshot)} for i in range(36)]
        off = {"arm": "off", "scenario": "plain", "output_tokens": 96, "concurrency": 16,
               "c1": requests, "c16": deepcopy(requests), "repeat": deepcopy(requests), "warm": [],
               "before": deepcopy(snapshot), "after": deepcopy(snapshot),
               "c16_before": deepcopy(snapshot), "c16_after": deepcopy(snapshot)}
        swap = deepcopy(off)
        swap.update(arm="swap", repeat=[])
        for row in swap["c1"]:
            row["observed"].update(preemptions=2, swap_out=2, swap_in=2, exact_resumes=2)
        for key in ("preemptions", "swap_out", "swap_in"):
            swap["c16_before"][key] = 72
            swap["c16_after"][key] = swap["after"][key] = 144
        import declaration
        policy = declaration.compile_gate({"model": "example", "cohorts": {}, "checks": {},
            "qualification": {}, "receipts": {}, "inputs": {}, "kv_offload": FACTS["kv_offload"],
            "preview": {"speculation": {"tokens": 0}}})
        reference = kvo.kinds.of("kv_offload").reference({}, policy)
        self.assertEqual(reference, {"facts": FACTS["kv_offload"], "speculative": False})
        capture = {"passed": True, "count": 36, "records": {"off": off, "swap": swap}, "policy": reference}
        check = {"module": "kv_offload", "rules": [{"path": ["passed"], "op": "eq", "value": True}]}
        kind = kvo.kinds.of("kv_offload")
        self.assertTrue(kind.score(capture, check=check)["passed"])
        self.assertEqual(kvo.kinds.rescore(kind, capture, reference, check)["status"], "pass")
        import release_gate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "off.json").write_text(json.dumps(off))
            (root / "swap.json").write_text(json.dumps(swap))
            declared = {**check, "id": "offload", "args": ["report", "--off", str(root / "off.json"),
                        "--swap", str(root / "swap.json")], "timeout_seconds": 30,
                        "receipt": "*-kv-offload-plain.json"}
            policy["checks"] = [declared]
            (root / "policy.json").write_text(json.dumps(policy))
            row = release_gate.run_check(declared, [], "full", {}, root, root / "policy.json",
                                         time.monotonic() + 30, False, None)
            self.assertTrue(row["passed"], row)
        swap["c1"][0]["logprobs"][0] = -0.5000000000000001
        result = kvo.kinds.rescore(kind, capture, reference, check)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["metrics"]["different_requests"], ["item-0"])

    def test_offload_facts_compile_without_a_preview(self):
        # Readers that drop a model's preview (the GSM8K floor fixtures)
        # still compile its declaration; the scorer reads the preview only
        # when it judges.
        import declaration
        config = declaration.compile_gate({"model": "example", "cohorts": {}, "checks": {},
            "qualification": {}, "receipts": {}, "inputs": {}, "kv_offload": FACTS["kv_offload"]})
        self.assertEqual(config["kv_offload"], FACTS["kv_offload"])
        self.assertNotIn("preview", config)

    def test_host_prefix_arm_is_a_usage_error(self):
        result = subprocess.run([str(kvo.kinds.eval_binary()), "serve", "--bundle", "unused",
                                 "--kv-offload-prefix", "on"], capture_output=True, text=True,
                                timeout=10, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires the deferred host prefix cache", result.stderr)

    def test_arms_derive_stripes_for_each_parallel_size(self):
        for parallel, pages in ((1, 512), (2, 256), (4, 128)):
            arms = kvo.arms(FACTS, {"page_tokens": 64, "context_parallel_size": parallel})
            self.assertEqual(len(arms), 4)
            for name, words in arms.items():
                options = dict(zip(words[::2], words[1::2]))
                pressure, swap = "pressure" in name, name.endswith("swap")
                self.assertEqual(options["--kv-offload-gib"], "16" if swap else "0")
                self.assertEqual(options["--prefix-cache"], "on" if pressure else "off")
                self.assertEqual(options["--speculative-tokens"], "3" if pressure else "0")
                self.assertEqual(options["--kv-offload-prefix"], "off")
                self.assertEqual(options.get("--kv-pages"), str(pages) if pressure else None)
                self.assertEqual(options.get("--debug-swap-every"), "16" if swap else None)

    def test_configuration_overrides_and_nonstandard_page_size(self):
        words = kvo.arms(FACTS, {"page_tokens": 128, "depth": 0,
                               "kv_offload": {"budget_gib": 8}})["kv-offload-pressure-swap"]
        options = dict(zip(words[::2], words[1::2]))
        self.assertEqual(options["--kv-pages"], "256")
        self.assertEqual(options["--speculative-tokens"], "0")
        self.assertEqual(options["--kv-offload-gib"], "8")
        with self.assertRaisesRegex(ValueError, "stripes"):
            kvo.arms(FACTS, {"page_tokens": 63})

    def test_runner_records_cover_exactly_the_four_arms(self):
        arms = kvo.arms(FACTS, {"page_tokens": 64})
        self.assertEqual({record[0] for record in kvo.RUNNER_RECORDS.values()}, set(arms))
        for _, script, words in kvo.RUNNER_RECORDS.values():
            self.assertEqual(script, "kv_offload.py")
            self.assertIn("600", words)

    def test_metrics_parse_and_observe_only_required_counters(self):
        before = kvo.metrics(metric_text())
        after = kvo.metrics(metric_text().replace("chop_kv_offload_swap_in_total 0", "chop_kv_offload_swap_in_total 3")
                            .replace('reason="capture"} 0', 'reason="capture"} 1'))
        seen = kvo.delta(before, after)
        self.assertEqual(seen["swap_in"], 3)
        self.assertEqual(seen["fallbacks"], {"capture": 1})
        self.assertEqual(after["host_pages_used"], 0)

    def test_off_arm_explicitly_allows_unpublished_disabled_tier_metrics(self):
        self.assertEqual(kvo.metrics(metric_text(False), offload=False)["swap_in"], 0)
        with self.assertRaisesRegex(ValueError, "missing metric"):
            kvo.metrics(metric_text(False))

    def test_missing_gauges_nonfinite_counts_and_resets_fail(self):
        for text in (metric_text().replace('chop_kv_offload_host_pages{kind="used"} 0\n', ''),
                     metric_text().replace("swap_in_total 0", "swap_in_total NaN"),
                     metric_text().replace("swap_in_total 0", "swap_in_total 1.5")):
            with self.assertRaises(ValueError):
                kvo.metrics(text)
        after = kvo.metrics(metric_text())
        before = dict(after, swap_in=1)
        with self.assertRaisesRegex(ValueError, "reset"):
            kvo.delta(before, after)

    def test_capture_requests_ids_and_preserves_logprobs_and_prompt_identity(self):
        class Client:
            def request(self, path, body):
                assert path == "/v1/completions" and body["return_token_ids"] is True
                return {"choices": [{"token_ids": [1], "logprobs": {"tokens": ["a"],
                    "token_logprobs": [-0.0], "top_logprobs": [{"a": -0.0}]},
                    "finish_reason": "length"}], "usage": {"completion_tokens": 1}}, 0.1
        row = kvo.completion(Client(), "test", ("item", "prompt"), 96)
        self.assertEqual(row["token_ids"], [1])
        self.assertEqual(row["prompt_sha256"], hashlib.sha256(b"prompt").hexdigest())
        self.assertEqual(row["logprobs"], [-0.0])

    def test_capture_c1_then_pressure_warm_measured_and_off_control(self):
        client = Namespace(identity=lambda model: {"model": "test"}, timeout=0,
                           request=lambda path, body: ({"tokens": [1]}, 0.0))
        rows = []
        def completion(client, model, job, tokens):
            rows.append(job[0])
            return {"name": job[0], "usage": {"prompt_tokens": 8066}}
        args = Namespace(url="http://unused", timeout=600, model=None, arm="off",
                         scenario="pressure", output_tokens=96, concurrency=16)
        with patch.object(kvo, "Client", return_value=client), \
             patch.object(kvo, "prompts", return_value=[("one", "prompt")]), \
             patch.object(kvo, "completion", side_effect=completion), \
             patch.object(kvo, "metrics_text", return_value=pool_text(offload=False)), \
             patch.object(kvo, "receipt") as receipt:
            self.assertEqual(kvo.record(args), 0)
        self.assertEqual([row for row in rows if not row.startswith("fill-")], ["one"] * 4)
        record = receipt.call_args.args[2]
        self.assertEqual(record["c1"][0]["observed"]["swap_in"], 0)
        for key in ("warm", "c16", "repeat"):
            self.assertEqual(len(record[key]), 1)


def pool_text(capacity=128, tokens=32768, used=0, cached=0, anchors=0, evictions=0, offload=True):
    return (metric_text(offload).replace("chop_prefix_cache_evictions_total 0", f"chop_prefix_cache_evictions_total {evictions}")
            + f"chop_kv_block_capacity {capacity}\nchop_kv_capacity_tokens {tokens}\n"
            f"chop_kv_blocks_used {used}\nchop_prefix_cache_blocks {cached}\nchop_prefix_cache_anchors {anchors}\n")


class AnchorPool:
    """runtime-kv's anchor accounting (crates/runtime-kv/src/lib.rs), in
    pages: an anchor shares its sequence's leading pages and allocates
    `state` fresh pages; an allocation that finds too few free pages evicts
    anchors least recently used first, each eviction counted, and fails only
    when none is left."""

    def __init__(self, capacity, state, anchors=()):
        self.capacity, self.state, self.evictions = capacity, state, 0
        self.anchors = list(anchors)  # [exclusive pages], least recently used first
        self.free = capacity - sum(self.anchors)
        assert self.free >= 0

    def allocate(self, pages):
        while self.free < pages and self.anchors:
            self.free += self.anchors.pop(0)
            self.evictions += 1
        if self.free < pages:
            return False
        self.free -= pages
        return True

    def request(self, pages, slack=0):
        """A sequence of `pages` that captures one anchor at its end and holds
        `slack` more pages until it finishes; its pages then belong to the
        anchor alone, or are freed if the capture found no room."""
        if not self.allocate(pages + slack):
            return False
        captured = self.allocate(self.state)
        self.free += slack
        if captured:
            self.anchors.append(pages + self.state)
        else:
            self.free += pages
        return True


def declared_pressure_pools():
    """(model, configuration, pool stripes) for every model's declared
    kv_offload facts and context-parallel size, at the packages' page size."""
    root = Path(__file__).resolve().parents[2]
    for qual in sorted(root.glob("models/*/qual.json")):
        value = json.loads(qual.read_text())
        if "kv_offload" not in value:
            continue
        sources = "".join(path.read_text() for path in (qual.parent / "contract/src").glob("*.rs"))
        assert "pub const PAGE_TOKENS: u64 = 64;" in sources, qual.parent.name
        for name, configuration in {"default": {}, **value.get("configurations", {})}.items():
            policy = value["kv_offload"] | configuration.get("kv_offload", {})
            parallel = kvo.configuration_option(configuration, "--context-parallel-size", 1)
            yield qual.parent.name, name, kvo.pressure_pages(policy, 64, parallel), policy["pressure_tokens"]


class PressureEviction(unittest.TestCase):
    def fill(self, pool, plan):
        start = pool.evictions
        for _ in range(plan["max_fillers"]):
            if pool.evictions - start >= plan["evictions"]:
                break
            self.assertTrue(pool.request(plan["filler_pages"], kvo.FILL_SLACK_PAGES))
        return pool.evictions - start >= plan["evictions"]

    def test_the_fill_forces_a_c16_eviction_for_every_declared_pool(self):
        pools = list(declared_pressure_pools())
        root = Path(__file__).resolve().parents[2]
        declaring = {path.parent.name for path in root.glob("models/*/qual.json")
                     if json.loads(path.read_text()).get("capabilities", {}).get("kv_offload")}
        self.assertTrue(declaring)
        self.assertLessEqual(declaring, {model for model, *_ in pools})
        for model, name, capacity, tokens in pools:
            stripe = tokens // capacity
            # The cohort's long prompts run about 8,000 tokens; any longest
            # prompt from 2,048 tokens to a whole pool must work.
            for longest in sorted({2048, 4096, 8066, 12288, tokens - stripe}):
                for state in range(0, capacity, 5):
                    for before in ((), (capacity - state,) if state < capacity else (), (1, 2, 3)):
                        plan = kvo.fill_plan(capacity, tokens, longest, len(before))
                        with self.subTest(model=model, configuration=name, longest=longest, state=state,
                                          before=before):
                            pages = plan["filler_pages"]
                            self.assertGreaterEqual(pages, 1)
                            self.assertLessEqual(pages * stripe, kvo.FILL_CHUNK_TOKENS)
                            pool = AnchorPool(capacity, state, before)
                            if pages + state + kvo.FILL_SLACK_PAGES > capacity:
                                # No filler anchor fits: the fill cannot reach its
                                # precondition and says so.
                                self.assertFalse(self.fill(pool, plan))
                                continue
                            self.assertTrue(self.fill(pool, plan))
                            evicted = pool.evictions
                            pool.request(plan["longest_prompt_pages"])
                            self.assertGreater(pool.evictions, evicted)

    def test_without_the_fill_c16_eviction_is_a_timing_margin(self):
        # The 2026-10-14 dcp4 run: 128 stripes of 256 tokens,
        # 82 snapshot pages an anchor, an 8,066-token longest prompt (32
        # stripes). C1's forced swaps leave no anchor behind (prefix
        # evictions 4 -> 4 over warm and C16): the longest prompt and its own
        # anchor fit, nothing is evicted, and any later eviction depends on
        # when captures and swaps meet.
        pool = AnchorPool(128, 82, ())
        self.assertTrue(pool.request(32))
        self.assertEqual(pool.evictions, 0)
        # The fill leaves one filler anchor and too few pages for the same prompt.
        plan = kvo.fill_plan(128, 32768, 8066, 0)
        pool = AnchorPool(128, 82, ())
        self.assertTrue(self.fill(pool, plan))
        evicted = pool.evictions
        self.assertTrue(pool.request(32))
        self.assertEqual(pool.evictions, evicted + 1)

    def test_plan_follows_the_pinned_pool_and_the_longest_prompt(self):
        plan = kvo.fill_plan(128, 32768, 8066, 2)
        self.assertEqual((plan["stripe_tokens"], plan["longest_prompt_pages"], plan["filler_pages"],
                          plan["filler_tokens"], plan["evictions"]), (256, 32, 8, 2048, 3))
        self.assertEqual(plan["max_fillers"], 2 * 16 + 2 + 4)
        plan = kvo.fill_plan(512, 32768, 8066, 0)
        self.assertEqual((plan["longest_prompt_pages"], plan["filler_pages"]), (127, 32))
        self.assertEqual(kvo.fill_plan(512, 32768, 300, 0)["filler_pages"], 1)
        self.assertEqual(kvo.fill_plan(512, 32768, 200, 0)["filler_pages"], 0)
        with self.assertRaisesRegex(ValueError, "whole"):
            kvo.fill_plan(100, 32768, 8066, 0)

    def test_the_pressure_record_fills_after_warm_and_before_the_c16_snapshot(self):
        for arm in ("off", "swap"):
            events, scrapes = [], {"evictions": 0}

            class Client:
                timeout = 0

                def identity(self, model):
                    return {"model": "test"}

                def request(self, path, body):
                    assert path == "/tokenize"
                    events.append(("tokenize", body["prompt"]))
                    return {"tokens": [7] if body.get("add_special_tokens") is False else [1, 2]}, 0.0

            def completion(client, model, job, tokens):
                name, prompt = job
                events.append(("completion", name, prompt, tokens))
                if name.startswith("fill-"):
                    scrapes["evictions"] += 1
                return {"name": name, "usage": {"prompt_tokens": 8066 if name == "long" else 100}}

            def metrics_text(client):
                events.append(("metrics",))
                return pool_text(anchors=2, evictions=scrapes["evictions"])

            args = Namespace(url="http://unused", timeout=600, model=None, arm=arm,
                             scenario="pressure", output_tokens=96, concurrency=16)
            with patch.object(kvo, "Client", return_value=Client()), \
                 patch.object(kvo, "prompts", return_value=[("short", "a"), ("long", "b")]), \
                 patch.object(kvo, "completion", side_effect=completion), \
                 patch.object(kvo, "metrics_text", side_effect=metrics_text), \
                 patch.object(kvo, "receipt") as receipt:
                self.assertEqual(kvo.record(args), 0)
            record = receipt.call_args.args[2]
            fill = record["fill"]
            self.assertEqual(fill["plan"]["filler_pages"], 8)
            self.assertTrue(fill["precondition"])
            self.assertEqual(fill["evictions"], 3)
            self.assertEqual([row["name"] for row in fill["rows"]], ["fill-0", "fill-1", "fill-2"])
            names = [event[1] for event in events if event[0] == "completion"]
            # C1, warm, then the fill, then C16 (and the off arm's control).
            self.assertEqual(names[:2], ["short", "long"])
            self.assertEqual(sorted(names[2:4]), ["long", "short"])
            self.assertEqual(names[4:7], ["fill-0", "fill-1", "fill-2"])
            self.assertEqual(sorted(names[7:9]), ["long", "short"])
            prompts = [event[2] for event in events if event[0] == "completion" and event[1].startswith("fill-")]
            self.assertTrue(all(len(prompt) == 2048 and prompt[:2] == [1, 2] for prompt in prompts))
            self.assertTrue(all(event[3] == 1 for event in events
                                if event[0] == "completion" and event[1].startswith("fill-")))
            texts = [event[1] for event in events if event[0] == "tokenize"]
            self.assertEqual(texts[1:], [f"Filler {index} of the offload pressure check. " for index in range(3)])
            # The C16 snapshot follows the fill.
            fill_end = max(index for index, event in enumerate(events)
                           if event[0] == "completion" and event[1] == "fill-2")
            self.assertEqual(record["c16_before"]["prefix_evictions"], 3)
            self.assertIn(("metrics",), events[fill_end:])

    def test_the_plain_record_has_no_fill(self):
        args = Namespace(url="http://unused", timeout=600, model=None, arm="off",
                         scenario="plain", output_tokens=96, concurrency=16)
        client = Namespace(identity=lambda model: {"model": "test"}, timeout=0)
        with patch.object(kvo, "Client", return_value=client), \
             patch.object(kvo, "prompts", return_value=[("one", "prompt")]), \
             patch.object(kvo, "completion", return_value={"name": "one"}), \
             patch.object(kvo, "metrics_text", return_value=metric_text(False)), \
             patch.object(kvo, "receipt") as receipt:
            self.assertEqual(kvo.record(args), 0)
        self.assertIsNone(receipt.call_args.args[2]["fill"])


if __name__ == "__main__":
    unittest.main()
