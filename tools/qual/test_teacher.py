"""Offline parsing and scoring checks for teacher.py."""

from qual_test_config import setUpModule  # noqa: F401

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from vectors import top_columns
from common import ROOT
import declaration
import teacher
from exact_reference import check_scores, gap_top, top
from teacher import (LARGE_SET_MEAN, MAX_MEAN, MAX_TOP1, check_reference, evaluate, reference_limits, relative,
                     response_prompt_rows, score, summary, variant_scores)

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "teacher.py parsing and scoring on synthetic responses"


class RetiredCaptureTests(unittest.TestCase):
    def test_running_the_retired_capture_refuses_and_names_eval(self):
        done = subprocess.run([sys.executable, '-B', str(Path(teacher.__file__)), '--url', 'http://127.0.0.1:1',
                               '--model', 'served'], capture_output=True, text=True, timeout=60)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('chopinfer eval run', done.stderr)


class TeacherResponseTests(unittest.TestCase):
    def test_choice_fields(self):
        ids, rows = response_prompt_rows({"choices": [{
            "prompt_token_ids": [1, 2],
            "prompt_logprobs": [None, {"2": {"logprob": -0.25}}],
        }]})
        self.assertEqual(ids, [1, 2])
        self.assertEqual(rows[1]["2"]["logprob"], -0.25)

    def test_chopinfer_root_fields(self):
        ids, rows = response_prompt_rows({"choices": [{}], "prompt_token_ids": [1, 2],
                                          "prompt_logprobs": [None, {"2": {"logprob": -0.25}}]})
        self.assertEqual(ids, [1, 2])
        self.assertEqual(len(rows), 2)

    def test_missing_or_misaligned_rows_fail(self):
        for response in ({"choices": [{}]},
                         {"choices": [{"prompt_token_ids": [1, 2],
                                       "prompt_logprobs": [None]}]}):
            with self.assertRaisesRegex(ValueError, "missing prompt token IDs"):
                response_prompt_rows(response)


class TeacherScoringTests(unittest.TestCase):
    def test_score_and_relative_gate(self):
        reference = [{"exact": {"logprob": [-1.0, -2.0], "top1": [5, 6], "margin": [1.0, 0.1],
                                **top_columns([[[5, -1.0], [4, -2.0]], [[6, -2.0], [4, -2.1]]])}}]
        server = [{"name": "t", "category": "prose", "logprob": [-1.01, -2.02], "top1": [5, 6]}]
        baseline = [{"name": "t", "category": "prose", "logprob": [-1.2, -2.0], "top1": [5, 4]}]
        server_scores = score(server, reference, "exact")
        baseline_scores = score(baseline, reference, "exact")
        self.assertEqual([p["disagrees"] for p in baseline_scores], [False, True])
        self.assertEqual(baseline_scores[1]["flip"]["decisive"], False)
        self.assertAlmostEqual(baseline_scores[1]["flip"]["gap"], 0.1)
        gate = relative(server_scores, baseline_scores)
        self.assertTrue(gate["passed"])
        self.assertEqual((gate["top1_server_only"], gate["top1_baseline_only"]), (0, 1))
        worse = relative(baseline_scores, server_scores)
        self.assertFalse(worse["passed"])


class ReferenceLimitTests(unittest.TestCase):
    # A gate variant and one spread variant of a synthetic reference.
    SYNTHETIC = ("exact", "spread")

    def text(self, **variants):
        """Variants of (logprobs, top-1 tokens), each storing only its top-1
        in its top 5, so every flip against it is decisive."""
        return {"name": "t", "category": "prose", **{
            name: {"logprob": logprobs, "top1": top1, "margin": [1.0] * len(logprobs),
                   **top_columns([[[token, logprob]] for token, logprob in zip(top1, logprobs)])}
            for name, (logprobs, top1) in variants.items()}}

    def test_floors_hold_until_a_variant_exceeds_them(self):
        close = [self.text(exact=([-1.0, -2.0], [5, 6]), spread=([-1.02, -2.0], [5, 6]))]
        limits, spread = reference_limits(close, variants=self.SYNTHETIC)
        self.assertEqual(limits, {"mean_abs_dlogprob": MAX_MEAN, "top1_decisive": MAX_TOP1, "top1_near_tie": 0.08,
                                  "top1_disagreement": 0.10})
        self.assertEqual(list(spread), ["spread"])
        wide = [self.text(exact=([-1.0, -2.0], [5, 6]), spread=([-1.3, -2.12], [5, 7]))]
        limits, _ = reference_limits(wide, variants=self.SYNTHETIC)
        # Mean 0.21 rounds up to 0.25; one decisive flip in two positions.
        self.assertAlmostEqual(limits["mean_abs_dlogprob"], 0.25)
        self.assertEqual(limits["top1_decisive"], 0.5)
        # The same flip where the gate variant ranks the variant's token
        # within the near-tie limit leaves the top-1 floor.
        wide[0]["exact"]["topk_ids"][1].append(7)
        wide[0]["exact"]["topk_logits"][1].append(-2.25)
        wide[0]["exact"]["topk_logprobs"][1].append(-2.25)
        limits, spread = reference_limits(wide, variants=self.SYNTHETIC)
        self.assertEqual(limits["top1_decisive"], MAX_TOP1)
        self.assertEqual(summary(spread["spread"])["top1_disagreement"], 0.5)
        self.assertEqual(summary(spread["spread"])["top1_near_ties"], 1)

    def test_a_declared_factor_scales_the_worst_variant_mean(self):
        count = 1000
        logprobs = [-1.0] * count
        texts = [self.text(exact=(logprobs, [5] * count), spread=([value - 0.13 for value in logprobs], [5] * count))]
        self.assertAlmostEqual(reference_limits(texts, variants=self.SYNTHETIC)[0]["mean_abs_dlogprob"], 0.15)
        with patch.object(teacher, "SPREAD_MEAN_FACTOR", 1.25):
            # 0.13 x 1.25 = 0.1625, rounded up to 0.2, in limits.py and in the eval binary alike.
            self.assertAlmostEqual(reference_limits(texts, variants=self.SYNTHETIC)[0]["mean_abs_dlogprob"], 0.2)
            result = {}
            evaluate(variant_scores(texts, "exact"), texts, result, variants=self.SYNTHETIC)
        self.assertEqual(result["limits"]["mean_abs_dlogprob"], 0.2)

    def test_sets_of_a_thousand_positions_have_the_tighter_mean_floor(self):
        def texts(count, offset):
            logprobs = [-1.0] * count
            return [self.text(exact=(logprobs, [5] * count),
                              spread=([value - offset for value in logprobs], [5] * count))]
        self.assertEqual(reference_limits(texts(999, 0.0), variants=self.SYNTHETIC)[0]["mean_abs_dlogprob"], MAX_MEAN)
        self.assertEqual(reference_limits(texts(1000, 0.0), variants=self.SYNTHETIC)[0]["mean_abs_dlogprob"], LARGE_SET_MEAN)
        self.assertEqual(LARGE_SET_MEAN, 0.10)
        # The reference's own spread still raises the floor, rounded up to 0.05.
        self.assertAlmostEqual(reference_limits(texts(1000, 0.12), variants=self.SYNTHETIC)[0]["mean_abs_dlogprob"], 0.15)
        self.assertEqual(reference_limits(texts(1000, 0.0), variants=self.SYNTHETIC)[0]["top1_decisive"], MAX_TOP1)

    def test_the_reference_must_score_every_frozen_text_with_every_variant(self):
        reference = json.loads((ROOT / "fixtures/mimo26/teacher-exact-reference.json").read_text())["texts"]
        names = [text["name"] for text in
                 json.loads((ROOT / "fixtures/mimo26/teacher-texts.json").read_text())["texts"]]
        check_reference(reference, names)
        broken = {
            "a text missing": reference[1:],
            "texts reordered": [reference[1], reference[0], *reference[2:]],
            "chat removed": [text for text in reference if text["category"] != "chat"],
            "a variant missing": [*reference[:5], {key: value for key, value in reference[5].items()
                                                   if key != "bf16"}, *reference[6:]],
            "a short column": [*reference[:5], {**reference[5], "exact": {
                **reference[5]["exact"], "top1": reference[5]["exact"]["top1"][:-1]}}, *reference[6:]],
            "a moved boundary": [*reference[:5], {**reference[5], "score_from": reference[5]["score_from"] + 1},
                                 *reference[6:]],
        }
        for label, texts in broken.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    check_reference(texts, names)

    def test_systematic_near_tie_flips_fail_their_category(self):
        reference = json.loads((ROOT / "fixtures/mimo26/teacher-exact-reference.json").read_text())["texts"]
        captured = variant_scores(reference, "exact")
        # The exact logprobs everywhere, but prose's top-1 swapped with the
        # runner-up wherever the reference ranks it within 0.3 nats.
        swaps = 0
        for text, exact in zip(captured, reference):
            if text["category"] != "prose":
                continue
            top1 = list(text["top1"])
            for index, pairs in enumerate(gap_top(exact["exact"])):
                if pairs[0][1] - pairs[1][1] <= 0.3:
                    top1[index] = pairs[1][0]
                    swaps += 1
            text["top1"] = top1
        result = {}
        self.assertFalse(evaluate(captured, reference, result)[1])
        prose = result["by_category"]["prose"]
        self.assertEqual((prose["top1_near_ties"], prose["top1_decisive_flips"]), (swaps, 0))
        self.assertGreater(prose["top1_near_tie"], prose["limits"]["top1_near_tie"])
        self.assertEqual(result["failed_categories"], ["prose"])
        self.assertLessEqual(prose["mean_abs_dlogprob"], 0.0)

    def mixed_chat(self, near, decisive):
        """The reference fixture and a capture with exact logprobs everywhere and,
        in chat, the first `near` near ties and `decisive` decisive
        positions flipped."""
        reference = json.loads((ROOT / "fixtures/mimo26/teacher-exact-reference.json").read_text())["texts"]
        captured = variant_scores(reference, "exact")
        for text, exact in zip(captured, reference):
            if text["category"] != "chat":
                continue
            top1 = list(text["top1"])
            for index, pairs in enumerate(gap_top(exact["exact"])):
                if near and pairs[0][1] - pairs[1][1] <= 0.3:
                    top1[index], near = pairs[1][0], near - 1
                elif decisive and pairs[0][1] - pairs[1][1] > 0.3:
                    top1[index], decisive = 10**9, decisive - 1
            text["top1"] = top1
        self.assertEqual((near, decisive), (0, 0))
        return captured, reference

    def test_mixed_flips_beyond_the_variant_total_fall_back_to_the_bf16_run(self):
        # The confirm review's capture: in chat 45 decisive flips (the
        # variant's decisive rate) plus 38 near ties (8%), 83 of 486
        # positions against the BF16 variant's 65. The total limit is that
        # run's own rate, which has no power (variant-spread.v2), so the
        # total passes by the paired rule: 57 flips of its own against the
        # BF16 run's 39 on the same positions, within 2 sqrt(96).
        captured, reference = self.mixed_chat(38, 45)
        result = {}
        self.assertTrue(evaluate(captured, reference, result)[1])
        chat = result["by_category"]["chat"]
        self.assertEqual((chat["top1_disagreements"], chat["top1_near_ties"]), (83, 38))
        self.assertEqual(chat["limits"]["top1_disagreement"], 65 / 486)
        self.assertGreater(chat["top1_disagreement"], chat["limits"]["top1_disagreement"])
        self.assertFalse(chat["power"]["top1_disagreement"]["has_power"])
        self.assertEqual(chat["passed_by"], {"top1_decisive": "limit", "top1_near_tie": "limit",
                                             "top1_disagreement": "paired_reference"})
        paired = chat["paired_reference"]["top1_disagreement"]
        self.assertEqual((paired["test_only"], paired["baseline_only"]), (57, 39))

    def test_mixed_flips_beyond_the_bf16_run_fail_chat(self):
        # 60 decisive flips and 38 near ties, 98 of 486: 72 flips of its own
        # against the BF16 run's 39 exceed the paired margin, so the total
        # cap fails chat.
        captured, reference = self.mixed_chat(38, 60)
        result = {}
        self.assertFalse(evaluate(captured, reference, result)[1])
        self.assertEqual(result["failed_categories"], ["chat"])
        chat = result["by_category"]["chat"]
        self.assertEqual(chat["top1_disagreements"], 98)
        self.assertIsNone(chat["passed_by"]["top1_disagreement"])
        paired = chat["paired_reference"]["top1_disagreement"]
        self.assertEqual((paired["test_only"], paired["baseline_only"], paired["passed"]), (72, 39, False))
        self.assertEqual(result["failed_categories"], ["chat"])

    def test_a_bad_category_fails_however_good_the_rest_is(self):
        reference = json.loads((ROOT / "fixtures/mimo26/teacher-exact-reference.json").read_text())["texts"]
        captured = variant_scores(reference, "exact")
        result = {}
        self.assertTrue(evaluate(captured, reference, result)[1])
        self.assertEqual(result["failed_categories"], [])
        # 0.6 nats off on every chat position stays inside the pooled limit.
        worse = [{**text, "logprob": [value - 0.6 for value in text["logprob"]]}
                 if text["category"] == "chat" else text for text in captured]
        result = {}
        self.assertFalse(evaluate(worse, reference, result)[1])
        self.assertLessEqual(result["mean_abs_dlogprob"], result["limits"]["mean_abs_dlogprob"])
        self.assertEqual(result["failed_categories"], ["chat"])
        # Chat's limits are its BF16 spread: 0.4225 rounded up, and 45 decisive
        # of its 65 top-1 flips in 486 positions.
        self.assertEqual(result["by_category"]["chat"]["limits"],
                         {"mean_abs_dlogprob": 0.45, "top1_decisive": 45 / 486, "top1_near_tie": 0.08,
                          "top1_disagreement": 65 / 486})

    def test_committed_reference_keeps_the_configured_limits(self):
        config = declaration.load(declaration.path("mimo26"))
        check = next(check for check in config["checks"] if check["id"] == "teacher")
        rules = {rule["path"][0]: rule["value"] for rule in check["rules"]
                 if rule["path"][0] in ("mean_abs_dlogprob", "top1_decisive", "top1_near_tie", "top1_disagreement")}
        reference = json.loads((ROOT / config["inputs"]["teacher_reference"]).read_text())
        limits, spread = reference_limits(reference["texts"])
        self.assertEqual(limits, rules)
        self.assertTrue(spread)
        self.assertEqual(reference["provenance"]["checkpoint"]["served_model_name"], "MiMo-V2.6-Flash-MOPD")


class RunnerLayoutTests(unittest.TestCase):
    """Reference variants in the reference runner's score layout."""

    def rows(self):
        return {"logprob": [-0.25, -2.25], "top1": [7, 4], "margin": [1.5, 0.5],
                "topk_ids": [[7, 3, 9, 1, 2], [4, 8, 5, 6, 0]],
                "topk_logits": [[9.0, 7.5, 7.0, 6.0, 5.0], [8.0, 7.5, 7.0, 6.5, 6.0]],
                "topk_logprobs": [[-0.25, -1.75, -2.25, -3.25, -4.25], [-0.75, -1.25, -1.75, -2.25, -2.75]]}

    def test_a_consistent_runner_row_passes_and_lists_its_top_pairs(self):
        check_scores(self.rows(), [7, 6], "t exact")
        self.assertEqual(top(self.rows())[0][:2], [[7, -0.25], [3, -1.75]])

    def test_retired_oracle_layout_is_refused(self):
        rows = self.rows()
        legacy = {key: rows[key] for key in ("logprob", "top1", "margin")}
        legacy["top5"] = top(rows)
        with self.assertRaises(ValueError):
            check_scores(legacy, [7, 6], "retired oracle")

    def test_an_inconsistent_runner_row_fails(self):
        edits = {
            "a short column": lambda rows: rows["topk_ids"].pop(),
            "a short top-k row": lambda rows: rows["topk_logits"][0].pop(),
            "top1 not first": lambda rows: rows["top1"].__setitem__(0, 3),
            "margin not the logit gap": lambda rows: rows["margin"].__setitem__(0, 1.75),
            "logits out of order": lambda rows: rows["topk_logits"][1].reverse(),
            "a repeated token": lambda rows: rows["topk_ids"][0].__setitem__(1, 7),
            "logprob unlike its top-k entry": lambda rows: rows["logprob"].__setitem__(0, -0.3),
        }
        for label, edit in edits.items():
            with self.subTest(label):
                rows = self.rows()
                edit(rows)
                with self.assertRaises(ValueError):
                    check_scores(rows, [7, 6], "t exact")


if __name__ == "__main__":
    unittest.main()


class FloorOnlyReferenceTests(unittest.TestCase):
    """The opt-in keeps the large-set floors and refuses incomplete evidence."""

    @staticmethod
    def fixture(positions=1024):
        policy = json.loads((ROOT / 'tools/refrun/mode-policies.json').read_text())
        exact = {'logprob': [0.0] * positions, 'top1': [7] * positions, 'margin': [1.0] * positions,
                 'topk_ids': [[7, 8, 9, 10, 11] for _ in range(positions)],
                 'topk_logits': [[0.0, -1.0, -2.0, -3.0, -4.0] for _ in range(positions)],
                 'topk_logprobs': [[0.0, -1.0, -2.0, -3.0, -4.0] for _ in range(positions)]}
        text = {'name': 'long', 'category': 'long', 'prompt_token_ids': [7] * (positions + 1),
                'score_from': 1, 'exact': exact}
        # The scorer receives an already authenticated reference. Pin and
        # producer authentication are exercised separately by test_refs.
        binding = {'schema': 'chopinfer.refcache.binding.v1', 'workload_sha256': 'a' * 64,
                   'modes': {'f32': {'mode': 'f32', 'variant': 'exact', 'dequant': 'exact',
                                     'policy_sha256': policy['f32']['policy_sha256']['exact']}}}
        return {'texts': [text], 'variants': ['exact'], 'binding': binding}

    @staticmethod
    def rust(reference, allow=True, shift=0.0):
        import kinds
        check = {'constants': {'teacher': {'ALLOW_FLOOR_ONLY': {'value': allow}}}}
        capture = variant_scores(reference['texts'], 'exact')
        capture = copy.deepcopy(capture)
        for text in capture:
            text['logprob'] = [value - shift for value in text['logprob']]
        return kinds.call('score_capture', kinds.KINDS['teacher_forced'], capture, reference, check)['metrics']

    def test_single_variant_still_requires_an_explicit_opt_in(self):
        reference = self.fixture()
        config = {'qualification': {'references': {'long': {'variants': ['exact']}}}}
        with patch.object(teacher, 'qualification_config', return_value=config), \
                patch.object(teacher, 'ALLOW_FLOOR_ONLY', False, create=True):
            with self.assertRaisesRegex(ValueError, 'spread variant'):
                teacher.teacher_variants('long')
        with self.assertRaises(ValueError):
            self.rust(reference, allow=False)

    def test_explicit_floor_policy_matches_python_and_rust(self):
        reference = self.fixture()
        config = {'qualification': {'references': {'long': {'variants': ['exact']}}}}
        with patch.object(teacher, 'qualification_config', return_value=config), \
                patch.object(teacher, 'ALLOW_FLOOR_ONLY', True, create=True):
            self.assertEqual(teacher.teacher_variants('long'), ('exact',))
            teacher.floor_only_reference(reference, ('exact',))
            expected, spread = reference_limits(reference['texts'], variants=('exact',))
            self.assertEqual(spread, {})
            self.assertEqual(expected, {'mean_abs_dlogprob': 0.10, 'top1_decisive': 0.08,
                                        'top1_near_tie': 0.08, 'top1_disagreement': 0.10})
            actual = self.rust(reference)
            self.assertEqual(actual['limits'], expected)
            self.assertTrue(actual['passed'])
            self.assertFalse(self.rust(reference, shift=0.125)['passed'])
            result = {}
            _, passed = evaluate(variant_scores(reference['texts'], 'exact'), reference['texts'], result,
                                 variants=('exact',), binding=reference['binding'])
            self.assertTrue(passed)
            self.assertEqual(result['limits'], actual['limits'])

    def test_admission_and_rescore_keep_the_explicit_policy(self):
        import kinds
        from vectors import served_rows
        reference = self.fixture()
        reference.update(gate='exact', sha256='b' * 64)
        text = reference['texts'][0]
        columns = text['exact']
        row = {'name': text['name'], 'prompt_tokens': len(text['prompt_token_ids']),
               'scored': len(columns['logprob']), 'latency_seconds': 0.1,
               **served_rows(text['prompt_token_ids'][1:], columns['logprob'], columns['top1'])}
        receipt = {'schema': 'chopinfer-qualification/v1', 'errors': [], 'skipped': [],
                   'rows': [row], 'reference_sha256': reference['sha256'], **self.rust(reference)}
        check = {'constants': {'teacher': {'ALLOW_FLOOR_ONLY': {'value': True}}},
                 'rules': [{'path': ['passed'], 'op': 'eq', 'value': True, 'source': 'test'}]}
        kind = kinds.KINDS['teacher_forced']
        kind.verify(receipt, reference, check)
        tampered_limits = copy.deepcopy(receipt)
        tampered_limits['limits']['mean_abs_dlogprob'] = 99.0
        self.assertEqual(kind.score(tampered_limits, reference, check)['limits'], receipt['limits'])
        for denied in ({}, {'constants': {'teacher': {'ALLOW_FLOOR_ONLY': {'value': False}}}}):
            with self.assertRaises(ValueError):
                kind.verify(receipt, reference, denied)
        with self.assertRaises(ValueError):
            kind.verify(receipt, None, check)
        with self.assertRaises(ValueError):
            self.rust(reference, allow=1)

    def test_live_parent_scores_the_worker_against_the_declared_reference(self):
        import contextlib
        import io
        import time
        import kinds
        import release_gate
        from vectors import served_rows
        reference = self.fixture()
        reference.update(gate='exact', sha256='b' * 64)
        text = reference['texts'][0]
        columns = text['exact']
        receipt = {'schema': 'chopinfer-qualification/v1', 'errors': [], 'skipped': [],
                   'reference_sha256': reference['sha256'], **self.rust(reference),
                   'rows': [{'name': text['name'], 'prompt_tokens': len(text['prompt_token_ids']),
                             'scored': len(columns['logprob']), 'latency_seconds': 0.1,
                             **served_rows(text['prompt_token_ids'][1:], columns['logprob'], columns['top1'])}]}
        config = declaration.load(declaration.MECHANICS)
        check = {'id': 'floor', 'module': 'teacher', 'args': [], 'receipt': '*-floor.json',
                 'timeout_seconds': 60,
                 'constants': {'teacher': {'ALLOW_FLOOR_ONLY': {'value': True}}},
                 'rules': [{'path': ['passed'], 'op': 'eq', 'value': True, 'source': 'test'}]}
        kind = kinds.of('teacher')
        foreign = copy.deepcopy(reference)
        foreign['sha256'] = 'c' * 64
        for label, pinned, worker_exit, expected in (
                ('complete', reference, 0, True), ('missing', None, 0, False),
                ('foreign', foreign, 0, False), ('worker failure', reference, 1, False)):
            with self.subTest(case=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'checks').mkdir()
                policy = root / 'policy.json'
                policy.write_text(json.dumps(config))
                def capture(command, environment, log, timeout):
                    (root / 'checks' / 'fresh-floor.json').write_text(json.dumps(receipt))
                    return worker_exit
                with patch.object(kind, 'capture', side_effect=capture), \
                        patch.object(kind, 'reference', return_value=pinned) as resolve, \
                        contextlib.redirect_stdout(io.StringIO()):
                    row = release_gate.run_check(check, [], 'full', {}, root, policy,
                                                 time.monotonic() + 60, False, None)
                self.assertEqual(row['exit_code'], worker_exit)
                self.assertEqual(row['passed'], expected, row)
                if worker_exit:
                    resolve.assert_not_called()
                else:
                    resolve.assert_called_once()
                    self.assertEqual(resolve.call_args.args[0]['id'], 'floor')

    def test_floor_policy_refuses_bad_mode_missing_binding_and_partial_scores(self):
        cases = {}
        cases['short'] = self.fixture(999)
        cases['missing binding'] = self.fixture()
        cases['missing binding'].pop('binding')
        cases['wrong mode'] = self.fixture()
        cases['wrong mode']['binding']['modes']['f32']['mode'] = 'f64'
        cases['wrong policy'] = self.fixture()
        cases['wrong policy']['binding']['modes']['f32']['policy_sha256'] = '0' * 64
        cases['missing exact'] = self.fixture()
        cases['missing exact']['texts'][0].pop('exact')
        cases['partial scores'] = self.fixture()
        cases['partial scores']['texts'][0]['exact']['logprob'].pop()
        cases['nonfinite'] = self.fixture()
        cases['nonfinite']['texts'][0]['exact']['logprob'][0] = float('nan')
        for label, reference in cases.items():
            with self.subTest(case=label), patch.object(teacher, 'ALLOW_FLOOR_ONLY', True, create=True):
                with self.assertRaises(ValueError):
                    teacher.floor_only_reference(reference, ('exact',))
                if label != 'missing exact':
                    with self.assertRaises(ValueError):
                        self.rust(reference)
