"""Mixed media must reach the same prompt positions before scoring."""
import copy
import unittest
from unittest.mock import patch
import multimodal

class MixedMediaTests(unittest.TestCase):
    def test_image_placement_mismatch_fails_before_scoring(self):
        expected = {"id": "two-images", "prompt_token_ids": [1,151655,2,151655,3], "answer_token_ids": [785] * 32, "answer": "The " * 32,
                    "exact": {}, "bf16": {}, "f64": {}}
        response = {"prompt_token_ids": [1,151655,151655,2,3] + [785] * 32,
                    "prompt_logprobs": [None] * 5 + [{"785": {"logprob": -1}}] * 32}
        class Client:
            def request(self, path, body):
                self.body = body
                return response, 0
        client = Client()
        with self.assertRaisesRegex(ValueError, "prompt token IDs"):
            multimodal.evaluate_mixed(client, "model", expected, {"messages": [{"role":"user","content":"media"}]})
        self.assertTrue(client.body["continue_final_message"])
        self.assertEqual(client.body["messages"][-1], {"role":"assistant", "content":"The " * 32})

    def test_short_reference_is_refused_before_request(self):
        from unittest.mock import Mock
        client = Mock()
        client.request.return_value = ({"prompt_token_ids": [1, 785]}, 0)
        expected = {"id": "single-image", "answer": "The", "answer_token_ids": [785],
                    "prompt_token_ids": [1], "exact": {}, "bf16": {}, "f64": {}}
        with patch.object(multimodal, "score_forced", return_value={"passed": True}), \
                patch.object(multimodal, "forced_capture", return_value={}):
            with self.assertRaisesRegex(ValueError, "at least 32"):
                multimodal.evaluate_mixed(client, "model", expected, {"messages": []})
        client.request.assert_not_called()

class GeneratedAnswerBindingTests(unittest.TestCase):
    def test_forced_answers_cannot_drop_generated_positions(self):
        import refs
        names = multimodal.MIXED_CASES
        generated = {"cases": [{"id": name, "answer_token_ids": [785] * 32 + [999]} for name in names],
                     "binding": {"modes": {"f32": {"entries": [[name, 0, "key"] for name in names]}},
                                 "sequences": [{"answer": {"policy": "greedy"}}]}}
        fixture = {"provenance": {"generation": {"generation": {"stop_ids": [999]}}},
                   "cases": [{"id": name, "answer_token_ids": [785] * 32, "terminal_stop_id": 999,
                              "generation_key_sha256": "key"} for name in names]}
        refs.verify_mixed_generation(fixture, generated)
        fixture["cases"][0]["answer_token_ids"].pop()
        with self.assertRaisesRegex(ValueError, "full F32 generation"):
            refs.verify_mixed_generation(fixture, generated)
