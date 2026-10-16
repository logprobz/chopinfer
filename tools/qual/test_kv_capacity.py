"""CPU checks for kv_capacity.py against a loopback server."""

from qual_test_config import configure_receipts as setUpModule

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import tempfile
import threading
import unittest
from unittest.mock import patch

import kv_capacity


def health(size=None, name="tp4-mtp", modalities=None):
    profile = {"name": name, "tensor_parallel_size": 4, "kv_cache_dtype": "fp8", "prefix_cache": True,
               "max_sequences": 32, "speculation": {"mode": "adaptive", "tokens": 3, "drafter": "mtp"}}
    if size is not None:
        profile["context_parallel_size"] = size
    value = {"status": "ready", "serving": {"bundle_id": "fixture.v1", "profile": profile,
                                            "model_options": {"checkpoint.quantization_layout": "fp8-block128"}}}
    if modalities is not None:
        value["modalities"] = modalities
    return value


def scrape(tokens, blocks, fp8=True):
    lines = ["# TYPE chop_kv_capacity_tokens gauge", f"chop_kv_capacity_tokens {tokens}",
             f"chop_kv_block_capacity {blocks}", 'chop_requests{state="active"} 3']
    if fp8:
        lines.append(f"chop_fp8_kv_capacity_tokens {tokens}")
    return "\n".join(lines) + "\n"


class MeasureTests(unittest.TestCase):
    def test_a_group_counts_its_stripes_and_the_log_agrees(self):
        log = ("startup create: 41.2s\nserving KV capacity: 8350 pages, 1068800 tokens\n"
               "context parallel: 2 ranks per group, each holding 8350 pages of 64 tokens\n")
        record, errors = kv_capacity.measure(scrape(1068800, 8350), health(2, modalities=["text", "image"]), log,
                                             planned=1069952)
        self.assertEqual(errors, [])
        self.assertEqual(record["block_tokens"], 128)
        self.assertEqual(record["context_parallel_size"], 2)
        self.assertEqual(record["startup"]["context_parallel"]["pages_per_rank"], 8350)
        self.assertAlmostEqual(record["measured_over_planned"], 1068800 / 1069952)
        self.assertEqual(record["model_options"], {"checkpoint.quantization_layout": "fp8-block128"})
        self.assertEqual(record["modalities"], ["text", "image"])

    def test_disagreement_and_missing_facts_are_errors(self):
        log = "serving KV capacity: 9242 pages, 591000 tokens\n"
        _, errors = kv_capacity.measure(scrape(591488, 9242), health(), log)
        self.assertEqual(errors, ["the log's 591000 tokens differ from the gauge's 591488"])
        _, errors = kv_capacity.measure(scrape(1069952, 8359), health(2),
                                        "serving KV capacity: 8359 pages, 1069952 tokens\n")
        self.assertEqual(errors, ["the log names no context-parallel group of 2"])
        _, errors = kv_capacity.measure("", {"status": "ready"})
        self.assertEqual(len(errors), 2, errors)
        _, errors = kv_capacity.measure(scrape(1, 1), health(), "no capacity here\n")
        self.assertEqual(errors, ["the server log has no serving KV capacity line"])

    def test_contradictory_geometry_is_refused(self):
        # Tokens match, but the log's pages and page size contradict the
        # block gauge.
        log = ("serving KV capacity: 16718 pages, 1069952 tokens\n"
               "context parallel: 2 ranks per group, each holding 16718 pages of 32 tokens\n")
        _, errors = kv_capacity.measure(scrape(1069952, 8359), health(2), log)
        self.assertIn("the log's 16718 pages differ from the gauge's 8359 blocks", errors)
        # A group in the log that the health does not report.
        log = ("serving KV capacity: 8359 pages, 1069952 tokens\n"
               "context parallel: 2 ranks per group, each holding 8359 pages of 64 tokens\n")
        _, errors = kv_capacity.measure(scrape(1069952, 8359), health(1), log)
        self.assertEqual(errors, ["the log names a context-parallel group of 2, /health reports none"])
        _, errors = kv_capacity.measure(scrape(1069952, 8359), health(4), log)
        self.assertIn("the log's group of 2 differs from /health's 4", errors)
        # Pages per rank and the token product must agree within the log.
        log = ("serving KV capacity: 8359 pages, 1069952 tokens\n"
               "context parallel: 2 ranks per group, each holding 8000 pages of 64 tokens\n")
        _, errors = kv_capacity.measure(scrape(1069952, 8359), health(2), log)
        self.assertEqual(errors, ["the log's 8000 pages per rank differ from its 8359 pages"])
        # Without a log, the gauges alone must split into pages.
        _, errors = kv_capacity.measure(scrape(1069953, 8359), health(2))
        self.assertEqual(errors, ["1069953 capacity tokens are not a whole number of 8359 blocks"])
        _, errors = kv_capacity.measure(scrape(1069952, 8359), health(2), page_tokens=32)
        self.assertEqual(errors, ["128-token blocks are not 32-token pages on 2 ranks"])
        _, errors = kv_capacity.measure(scrape(9242 * 63, 9242), health(2))
        self.assertEqual(errors, ["63-token blocks do not split into pages on 2 ranks"])
        _, errors = kv_capacity.measure(scrape(591488, 9242, fp8=False) + "chop_fp8_kv_capacity_tokens 1\n",
                                        health())
        self.assertEqual(errors, ["the FP8 gauge's 1 tokens differ from the pool's 591488"])

    def test_the_last_startup_counts(self):
        log = ("serving KV capacity: 1 pages, 64 tokens\n"
               "context parallel: 2 ranks per group, each holding 1 pages of 64 tokens\n"
               "serving KV capacity: 9242 pages, 591488 tokens\n")
        capacity, group = kv_capacity.startup_lines(log)
        self.assertEqual(capacity, {"pages": 9242, "tokens": 591488})
        self.assertIsNone(group)


class ServerTests(unittest.TestCase):
    def test_writes_a_receipt_from_a_running_server(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/metrics":
                    payload, kind = scrape(591488, 9242).encode(), "text/plain"
                else:
                    body = {"/v1/models": {"data": [{"id": "Fixture"}]}, "/health": health()}.get(self.path)
                    if body is None:
                        self.send_error(404)
                        return
                    payload, kind = json.dumps(body).encode(), "application/json"
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, \
                tempfile.NamedTemporaryFile("w", suffix=".log") as log:
            log.write("serving KV capacity: 9242 pages, 591488 tokens\n")
            log.flush()
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                argv = ["kv_capacity", "--url", url, "--server-log", log.name, "--planned", "591488",
                        "--label", "tp4-mtp"]
                with patch("sys.argv", argv), patch.object(kv_capacity, "receipt") as receipt:
                    with self.assertRaises(SystemExit) as result:
                        kv_capacity.main()
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual(result.exception.code, 0)
        check, identity, report = receipt.call_args.args[:3]
        self.assertEqual(check, "kv-capacity")
        self.assertEqual(identity["model"], "Fixture")
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["capacity_tokens"], 591488)
        self.assertEqual(report["measured_over_planned"], 1.0)
        self.assertEqual(report["label"], "tp4-mtp")


if __name__ == "__main__":
    unittest.main()
