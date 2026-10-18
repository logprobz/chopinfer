"""API thinking requests use the declared frontend transport and response field."""

from qual_test_config import configure_receipts as setUpModule

import os
import unittest
from unittest.mock import Mock, patch

import api
import common


def capture(reasoning=None):
    bodies = []

    def request(path, body, **kwargs):
        bodies.append(body)
        if "thinking" not in body and "chat_template_kwargs" not in body:
            raise RuntimeError("outside this request test")
        return {"choices": [{"message": {"content": "answer", **(reasoning or {})}}]}, 0.1

    client = Mock()
    client.identity.return_value = {"model": "m"}
    client.request.side_effect = request
    with patch("sys.argv", ["api", "--mechanics-only"]), \
            patch.object(api, "Client", return_value=client), \
            patch.object(api.urllib.request, "urlopen", side_effect=RuntimeError("no metrics")), \
            patch.object(api, "receipt") as receipt:
        try:
            api.main()
        except SystemExit as error:
            assert error.code == 1  # Unrelated API cases are deliberately unanswered.
    return [body for body in bodies if "thinking" in body or "chat_template_kwargs" in body], receipt.call_args.args[2]


class ThinkingRequests(unittest.TestCase):
    def test_declared_control_and_hidden_reasoning(self):
        for config in sorted((common.ROOT / "models").glob("*/qual.json")):
            with self.subTest(model=config.parent.name), \
                    patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(config)}):
                frontend = common.frontend()
                field = frontend["response"]["reasoning_field"]
                bodies, report = capture()
                self.assertEqual(len(bodies), 3)
                for body, enabled in zip(bodies, (True, True, False)):
                    if frontend["reasoning"].get("switch") == "thinking-object":
                        self.assertEqual(body["thinking"], {"type": "enabled" if enabled else "disabled"})
                        self.assertNotIn("chat_template_kwargs", body)
                    else:
                        self.assertEqual(body["chat_template_kwargs"],
                                         {frontend["reasoning"]["thinking_keywords"][0]: enabled})
                self.assertTrue(next(row for row in report["rows"] if row["name"] == "hidden_reasoning")["passed"])
                _, report = capture({field: "exposed"})
                self.assertFalse(next(row for row in report["rows"] if row["name"] == "hidden_reasoning")["passed"])


if __name__ == "__main__":
    unittest.main()
