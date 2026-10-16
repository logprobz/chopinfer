"""The runner end to end on a tiny random HF model, on CPU.

The runner's scores must equal those of a plain `model(...)` forward over the
same weights, decoded here by the definitions rather than by quant.py. Needs
torch and transformers; skipped under a Python without them.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.models.llama.modeling_llama import eager_attention_forward
    from tools.model_sdk.diagnostic import canonical
    from tools.refrun import adapter, cache, keys, margins, modes, ops, place, quant, run, score, stamps, tiny
    from tools.refrun.test_quant import naive
except ModuleNotFoundError:
    torch = None

PROMPTS = [[1, 5, 9, 33, 2, 7, 40, 41], [3, 4, 5]]


def setUpModule():
    if torch is None:
        return
    modes.prepare_environment(cuda=False)
    # Local filesystems tick in a few ms; a short window keeps the tests fast.
    global RACY_NS
    RACY_NS, stamps.RACY_NS = stamps.RACY_NS, 100_000_000


def tearDownModule():
    if torch is not None:
        stamps.RACY_NS = RACY_NS


def plain_model(root, dtype):
    """HF Llama with the tiny checkpoint's weights decoded element by element."""
    from safetensors.torch import load_file
    stored = load_file(str(Path(root) / "model.safetensors"))
    config = LlamaConfig(**json.loads((Path(root) / "config.json").read_text()))
    config._attn_implementation = "eager"
    state = {}
    inter, hidden = config.intermediate_size, config.hidden_size
    grid = tiny.GATE
    for name, tensor in stored.items():
        if ".mlp." not in name:
            state[name] = tensor.to(dtype)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp."
        gate_scale = stored[prefix + "gate_proj.weight_scale_inv"]
        rows = grid.scale_rows(inter)
        state[prefix + "gate_proj.weight"] = naive(
            stored[prefix + "gate_proj.weight"],
            lambda r, c: float(gate_scale[rows[r], c // grid.block_cols]), inter, hidden, "e4m3").to(dtype)
        up_scale = stored[prefix + "up_proj.weight_scale"]
        state[prefix + "up_proj.weight"] = naive(
            stored[prefix + "up_proj.weight"],
            lambda r, c: quant.E8M0[int(up_scale[r, c // 32])], inter, hidden, "e2m1").to(dtype)
        state[prefix + "down_proj.weight"] = stored[prefix + "down_proj.weight"].to(dtype)
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = LlamaForCausalLM(config).eval()
    finally:
        torch.set_default_dtype(default)
    model.load_state_dict(state, strict=True)
    return model


def write_workload(root, sequences):
    path = Path(root) / "workload.json"
    path.write_text(json.dumps({"schema": "chopinfer.refrun.workload.v1", "model": "tiny",
                                "sequences": sequences}))
    return path


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = Path(cls.directory.name)
        cls.checkpoint = tiny.write_checkpoint(root / "checkpoint")
        sequences = [{"name": "text", "category": "prose", "token_ids": PROMPTS[0], "score_from": 1},
                     {"name": "text-again", "category": "repeat", "token_ids": PROMPTS[0], "score_from": 1},
                     {"name": "answer", "category": "probe", "token_ids": PROMPTS[1], "score_from": 3,
                      "answer": {"policy": "forced", "ids": [17, 18, 19]}},
                     {"name": "greedy", "category": "probe", "token_ids": PROMPTS[1], "score_from": 2,
                      "answer": {"policy": "greedy", "max_new_tokens": 6, "stop_ids": [95]}}]
        cls.output = root / "out"
        cls.provenance = run.run(run.parse_args([
            "--adapter", str(keys_root() / "tools/refrun/tiny.py"), "--checkpoint", str(cls.checkpoint),
            "--lock", str(cls.checkpoint / tiny.LOCK), "--workload", str(write_workload(root, sequences)),
            "--modes", "f32,bf16,f64", "--devices", "cpu,cpu,cpu", "--output", str(cls.output),
            "--stamps", str(root / "stamps"), "--cache", str(root / "cache")]))
        cls.scores = {mode: json.loads((cls.output / f"{mode}.json").read_text())["entries"]
                      for mode in ("f32", "bf16", "f64")}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def entry(self, mode, name, policy="forced"):
        for entry in self.scores[mode]:
            if entry["name"] == name and entry["key"]["sequence"]["answer"]["policy"] == policy:
                return entry
        raise KeyError(name)

    def test_scores_equal_a_plain_forward(self):
        for mode, dtype in (("f32", torch.float32), ("f64", torch.float64)):
            model = plain_model(self.checkpoint, dtype)
            for name, ids, score_from in (("text", PROMPTS[0], 1), ("answer", PROMPTS[1] + [17, 18, 19], 3)):
                with torch.no_grad():
                    logits = model(input_ids=torch.tensor([ids]), use_cache=False,
                                   logits_to_keep=len(ids) - score_from + 1).logits[0]
                logprobs = torch.log_softmax(logits[:-1], dim=-1)
                tokens = torch.tensor(ids[score_from:])
                scores = self.entry(mode, name)["scores"]
                self.assertEqual(scores["token"], ids[score_from:])
                self.assertEqual(scores["logprob"], logprobs.gather(1, tokens[:, None])[:, 0].double().tolist(),
                                 (mode, name))
                order = sorted(range(logits.shape[1]), key=lambda i: (-logits[0, i].item(), i))
                self.assertEqual(scores["topk_ids"][0], order[:20])
                self.assertEqual(scores["topk_logprobs"][0], logprobs[0, order[:20]].double().tolist())
                self.assertEqual(scores["topk_logits"][0], logits[0, order[:20]].double().tolist())
                self.assertEqual(scores["logit"], logits[:-1].gather(1, tokens[:, None])[:, 0].double().tolist())
                self.assertEqual(scores["top1"], logits[:-1].argmax(dim=-1).tolist())
                self.assertEqual(scores["margin"][0], logits[0, order[0]].item() - logits[0, order[1]].item())

    def test_greedy_answer_matches_a_recomputed_argmax_and_is_then_forced(self):
        model = plain_model(self.checkpoint, torch.float64)
        ids = list(PROMPTS[1])
        for _ in range(6):
            with torch.no_grad():
                token = int(model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].argmax())
            ids.append(token)
            if token == 95:
                break
        answer = ids[len(PROMPTS[1]):]
        generated = self.entry("f32", "greedy", "greedy")["scores"]
        self.assertEqual(generated["answer"], answer)
        self.assertEqual(generated["top1"], answer)
        for mode in ("f32", "bf16", "f64"):
            forced = self.entry(mode, "greedy")["scores"]
            self.assertEqual(forced["token"], PROMPTS[1][2:] + answer, mode)
        self.assertEqual(sum(1 for e in self.scores["bf16"] if e["key"]["sequence"]["answer"]["policy"] == "greedy"),
                         0)

    def test_modes_differ_only_by_rounding(self):
        f32, f64, bf16 = (self.entry(mode, "text")["scores"]["logprob"] for mode in ("f32", "f64", "bf16"))
        self.assertLess(max(abs(a - b) for a, b in zip(f32, f64)), 1e-4)
        self.assertNotEqual(f32, f64)
        self.assertLess(max(abs(a - b) for a, b in zip(f32, bf16)), 0.5)
        self.assertNotEqual(f32, bf16)

    def test_keys_and_provenance(self):
        f32 = self.entry("f32", "text")
        self.assertEqual(f32["key"]["mode"]["name"], "f32")
        self.assertEqual(f32["key"]["environment"]["device"], "cpu")
        self.assertIn("tools/refrun/tiny.py", f32["key"]["adapter"]["files"])
        self.assertLessEqual({"tools/refrun/place.py", "tools/refrun/workload.py"}, set(f32["key"]["adapter"]["files"]))
        self.assertEqual(f32["key"]["reference_code"]["transformers_sha256"], keys.installed_transformers()[1])
        self.assertNotEqual(f32["key_sha256"], self.entry("f64", "text")["key_sha256"])
        self.assertTrue(all(m["repeat_bitwise_equal"] for m in self.provenance["modes"].values()))
        devices = {device for _, device in self.provenance["placement"]["units"]}
        self.assertEqual(devices, {"cpu"})

    def test_labels_stay_out_of_keyed_scores(self):
        for mode in ("f32", "bf16", "f64"):
            first, second = self.entry(mode, "text"), self.entry(mode, "text-again")
            self.assertEqual((first["category"], second["category"]), ("prose", "repeat"))
            self.assertEqual(first["key_sha256"], second["key_sha256"])
            self.assertEqual(canonical(first["scores"]), canonical(second["scores"]))

    def test_existing_output_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            run.run(run.parse_args([
                "--adapter", str(keys_root() / "tools/refrun/tiny.py"), "--checkpoint", str(self.checkpoint),
                "--lock", str(self.checkpoint / tiny.LOCK), "--workload", str(Path(self.directory.name) /
                                                                             "workload.json"),
                "--devices", "cpu", "--output", str(self.output), "--no-cache"]))

    def cached_run(self, name, *extra):
        root = Path(self.directory.name)
        return run.run(run.parse_args([
            "--adapter", str(keys_root() / "tools/refrun/tiny.py"), "--checkpoint", str(self.checkpoint),
            "--lock", str(self.checkpoint / tiny.LOCK), "--workload", str(root / "workload.json"),
            "--modes", "f32,bf16,f64", "--devices", "cpu,cpu,cpu", "--output", str(root / name),
            "--stamps", str(root / "stamps"), "--cache", str(root / "cache"), *extra]))

    def test_a_cached_run_loads_no_weights_and_writes_the_same_scores(self):
        with mock.patch.object(place, "Store", side_effect=AssertionError("loaded the checkpoint")):
            provenance = self.cached_run("hit")
        self.assertIsNone(provenance["load"])
        for mode, record in provenance["modes"].items():
            self.assertEqual((record["computed"], record["cache_hits"]), (0, 5 if mode == "f32" else 4))
            self.assertEqual((Path(self.directory.name) / "hit" / f"{mode}.json").read_bytes(),
                             (self.output / f"{mode}.json").read_bytes())
        entry = cache.Store(Path(self.directory.name) / "cache").get(self.entry("f32", "text")["key_sha256"])
        self.assertEqual(entry["scores"], self.entry("f32", "text")["scores"])

    def test_a_recomputed_hit_must_equal_its_entry(self):
        self.cached_run("recompute", "--recompute")
        real = score.teacher_forced

        def drifted(*args, **kwargs):
            scores = real(*args, **kwargs)
            scores["logprob"][0] += 2.0 ** -40
            return scores
        with mock.patch.object(score, "teacher_forced", side_effect=drifted), \
                self.assertRaisesRegex(cache.Nondeterministic, "differs from the stored one"):
            self.cached_run("drift", "--recompute")

    def raced_run(self, name, schedule, target=None, edit=None):
        """The tiny f32 run of a fresh checkpoint, with `schedule(checkpoint,
        real, args, kwargs)` patched over `target` (stamps.verify by default)
        as a writer would interleave, and `edit(checkpoint)` applied before
        the run. Returns the run's f32 entries."""
        root = Path(self.directory.name) / name
        checkpoint = tiny.write_checkpoint(root / "checkpoint")
        if edit is not None:
            edit(checkpoint)
        target = target or (stamps, "verify")
        real = getattr(*target)
        with mock.patch.object(*target, side_effect=lambda *a, **k: schedule(checkpoint, real, a, k)):
            run.run(run.parse_args([
                "--adapter", str(keys_root() / "tools/refrun/tiny.py"), "--checkpoint", str(checkpoint),
                "--lock", str(checkpoint / tiny.LOCK), "--workload", str(Path(self.directory.name) /
                                                                         "workload.json"),
                "--modes", "f32", "--devices", "cpu", "--output", str(root / "out"),
                "--stamps", str(root / "stamps"), "--no-cache"]))
        return json.loads((root / "out" / "f32.json").read_text())["entries"]

    def test_a_config_restored_before_verification_scores_the_verified_bytes(self):
        # The confirm review's schedule: config.json differs when the run
        # starts (rms_norm_eps 1e-01, same size) and is restored just before
        # it is verified. Parsed before verification, the changed bytes gave
        # different scores under the same key.
        def change(checkpoint):
            config = checkpoint / "config.json"
            config.write_bytes(config.read_bytes().replace(b'"rms_norm_eps": 1e-06', b'"rms_norm_eps": 1e-01'))

        def restore(checkpoint, verify, args, kwargs):
            config = checkpoint / "config.json"
            if b'"rms_norm_eps": 1e-01' in config.read_bytes():
                config.write_bytes(config.read_bytes().replace(b'"rms_norm_eps": 1e-01',
                                                               b'"rms_norm_eps": 1e-06'))
            return verify(*args, **kwargs)
        self.assertEqual(self.raced_run("restored", restore, edit=change), self.scores["f32"])

    def test_a_config_changed_after_verification_is_refused(self):
        def change_after(checkpoint, verify, args, kwargs):
            verified = verify(*args, **kwargs)
            config = checkpoint / "config.json"
            config.write_bytes(config.read_bytes().replace(b'"rms_norm_eps": 1e-06', b'"rms_norm_eps": 1e-01'))
            return verified
        with self.assertRaisesRegex(RuntimeError, "config.json changed after it was verified"):
            self.raced_run("changed-config", change_after)
        self.assertFalse((Path(self.directory.name) / "changed-config/out/f32.json").exists())

    def test_a_shard_rewritten_while_scoring_is_refused(self):
        def rewrite(checkpoint, teacher_forced, args, kwargs):
            shard = checkpoint / "model.safetensors"
            shard.write_bytes(shard.read_bytes())
            return teacher_forced(*args, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "model.safetensors changed after it was verified"):
            self.raced_run("rewritten-shard", rewrite, (score, "teacher_forced"))

    def test_changed_checkpoint_bytes_are_refused(self):
        root = Path(self.directory.name)
        changed = tiny.write_checkpoint(root / "changed")
        config = changed / "config.json"
        text = config.read_text()
        self.assertIn('"rms_norm_eps": 1e-06', text)
        config.write_text(text.replace('"rms_norm_eps": 1e-06', '"rms_norm_eps": 1e-05'))
        with self.assertRaisesRegex(ValueError, "config.json has sha256"):
            run.run(run.parse_args([
                "--adapter", str(keys_root() / "tools/refrun/tiny.py"), "--checkpoint", str(changed),
                "--lock", str(self.checkpoint / tiny.LOCK), "--workload", str(root / "workload.json"),
                "--devices", "cpu", "--output", str(root / "changed-out"), "--stamps", str(root / "stamps"),
                "--no-cache"]))


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class RankTests(unittest.TestCase):
    def test_logits_rank_where_logprobs_collapse(self):
        for gap in (2.0 ** -25, 2.0 ** -30):
            logits = torch.tensor([[0.0, gap]])
            logprobs = torch.log_softmax(logits, dim=-1)
            self.assertEqual(logprobs[0, 0].item(), logprobs[0, 1].item())
            scores = score.rows(logits, [1], torch.float32, topk=2)
            self.assertEqual(scores["top1"], [int(logits.argmax())])
            self.assertEqual(scores["topk_ids"], [[1, 0]])
            self.assertEqual(scores["margin"], [gap])
            self.assertEqual(scores["topk_logprobs"], [logprobs[0, [1, 0]].double().tolist()])

    def test_gaps_to_any_ranked_token_come_from_logits(self):
        logits = torch.tensor([[0.0, 2.0 ** -25, -1.5, -40.0, 3.0]])
        scores = score.rows(logits, [3], torch.float32, topk=3)
        self.assertEqual(scores["topk_ids"], [[4, 1, 0]])
        self.assertEqual(margins.gap(scores, 0, 4, 1), scores["margin"][0])
        self.assertEqual(margins.gap(scores, 0, 1, 0), 2.0 ** -25)
        self.assertEqual(scores["topk_logprobs"][0][1], scores["topk_logprobs"][0][2])
        self.assertEqual(margins.gap(scores, 0, 4, 3), 43.0)
        self.assertIsNone(margins.gap(scores, 0, 4, 2))

    def test_equal_logits_go_to_the_lower_id(self):
        logits = torch.tensor([[0.5, 1.0, 1.0, -2.0]], dtype=torch.bfloat16)
        scores = score.rows(logits, [2], torch.float32, topk=4)
        self.assertEqual(scores["top1"], [int(logits.argmax())])
        self.assertEqual(scores["topk_ids"], [[1, 2, 0, 3]])
        self.assertEqual(scores["margin"], [0.0])

    def test_logits_wider_than_the_log_softmax_are_refused(self):
        with self.assertRaisesRegex(ValueError, "wider"):
            score.rows(torch.zeros((1, 4), dtype=torch.float64), [0], torch.float32)


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class ModeListTests(unittest.TestCase):
    def test_a_mode_runs_under_two_rules_with_distinct_labels(self):
        policies = run.mode_policies("f32,f32:hf-bf16,bf16,f64", ["f64=hf-bf16"])
        self.assertEqual([label for label, _ in policies], ["f32", "f32-hf-bf16", "bf16", "f64"])
        self.assertEqual([p["dequant"] for _, p in policies], ["exact", "hf-bf16", "hf-bf16", "hf-bf16"])
        self.assertEqual(policies[1][1]["name"], "f32")
        with self.assertRaisesRegex(ValueError, "repeats"):
            run.mode_policies("f32,f32:exact,f32")
        with self.assertRaisesRegex(ValueError, "not run"):
            run.mode_policies("f32", ["f64=hf-bf16"])
        with self.assertRaisesRegex(ValueError, "cannot dequantize"):
            run.mode_policies("bf16:exact")


def keys_root():
    from tools.refrun import keys
    return keys.ROOT


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class PlaceTests(unittest.TestCase):
    def test_plan_is_contiguous_and_balanced(self):
        sizes = [("embed", 10), *((f"layer{i}", 20) for i in range(8)), ("head", 10)]
        placement = place.plan(sizes, ["d0", "d1", "d2", "d3"])
        devices = [device for _, device in placement.units]
        self.assertEqual(devices, sorted(devices))
        load = {d: sum(s for (_, s), (_, p) in zip(sizes, placement.units) if p == d) for d in placement.devices}
        self.assertEqual(sorted(load.values()), [40, 40, 50, 50])

    def test_checkpoint_reads_equal_safetensors(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as root:
            checkpoint_dir = tiny.write_checkpoint(Path(root) / "c", seed=3)
            checkpoint = place.Checkpoint(checkpoint_dir)
            expected = load_file(str(checkpoint_dir / "model.safetensors"))
            units = tiny.units(checkpoint)
            placement = place.plan([(u.module, checkpoint.stored_bytes(u.tensors)) for u in units], ["cpu"])
            store = place.Store(checkpoint, placement, units)
            self.assertEqual(set(store.tensors), set(expected))
            for name, tensor in expected.items():
                self.assertTrue(torch.equal(store.get(name).view(torch.uint8), tensor.view(torch.uint8)), name)
                self.assertEqual(store.get(name).dtype, tensor.dtype)
            self.assertEqual(store.read_bytes, sum(t.nbytes for t in expected.values()))
            rows = place.HostRows(checkpoint, "model.embed_tokens.weight").rows([5, 0, 5])
            self.assertTrue(torch.equal(rows, expected["model.embed_tokens.weight"][[5, 0, 5]]))

    def test_extra_shards_join_under_a_prefix_and_pinned_files_check_their_hash(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as root:
            checkpoint_dir = tiny.write_checkpoint(Path(root) / "c", seed=3)
            (checkpoint_dir / "codec").mkdir()
            table = torch.arange(6, dtype=torch.float32).reshape(2, 3)
            save_file({"embed": table}, str(checkpoint_dir / "codec/model.safetensors"))
            (checkpoint_dir / "codec/config.json").write_text("{}")
            checkpoint = place.Checkpoint(checkpoint_dir, {"codec/model.safetensors": "codec."})
            self.assertEqual(checkpoint.entries["codec.embed"].file, "codec/model.safetensors")
            self.assertNotIn("embed", checkpoint.entries)
            self.assertIn("codec/model.safetensors", checkpoint.read)
            units = [place.Unit("codec", ("codec.embed",))]
            store = place.Store(checkpoint, place.plan([("codec", 24)], ["cpu"]), units)
            self.assertTrue(torch.equal(store.get("codec.embed"), table))
            digest = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
            self.assertEqual(checkpoint.pinned("codec/config.json", digest), b"{}")
            with self.assertRaisesRegex(ValueError, "not the pinned"):
                checkpoint.pinned("codec/config.json", "0" * 64)

    def test_shards_are_read_in_parallel_runs(self):
        """Three readers, one shard each, at once (a barrier their first reads
        must pass together); runs of adjacent tensors share a slot, a tensor
        larger than the slot is read in parts, and unplaced tensors between
        them are skipped."""
        import threading
        from safetensors.torch import save_file
        generator = torch.Generator().manual_seed(5)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as root:
            root = Path(root)
            expected, weight_map = {}, {}
            for shard in range(3):
                tensors = {f"s{shard}.a": torch.randn((7, 9), generator=generator),
                           f"s{shard}.b": torch.randn((300,), generator=generator).to(torch.bfloat16),
                           f"s{shard}.skip": torch.randint(0, 255, (50,), generator=generator, dtype=torch.uint8),
                           f"s{shard}.c": torch.randint(-9, 9, (11,), generator=generator)}
                file = f"model-{shard}.safetensors"
                save_file(tensors, str(root / file))
                weight_map.update({name: file for name in tensors})
                expected.update({n: t for n, t in tensors.items() if not n.endswith(".skip")})
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
            (root / "config.json").write_text("{}")
            checkpoint = place.Checkpoint(root)
            units = [place.Unit(f"u{shard}", tuple(n for n in expected if n.startswith(f"s{shard}.")))
                     for shard in range(3)]
            placement = place.Placement(("cpu",), tuple((u.module, "cpu") for u in units))
            barrier, threads, read_into = threading.Barrier(3, timeout=30), set(), place.read_into

            def first_reads_meet(fd, view, offset, request=None):
                if threading.get_ident() not in threads:
                    threads.add(threading.get_ident())
                    barrier.wait()
                return read_into(fd, view, offset, request)
            place.read_into = first_reads_meet
            try:
                store = place.Store(checkpoint, placement, units, slot_bytes=256, slots=2, readers=3)
            finally:
                place.read_into = read_into
            self.assertEqual(len(threads), 3)
            self.assertEqual(set(store.tensors), set(expected))
            for name, tensor in expected.items():
                self.assertEqual(store.get(name).dtype, tensor.dtype)
                self.assertTrue(torch.equal(store.get(name), tensor), name)
            self.assertEqual(store.read_bytes, sum(t.nbytes for t in expected.values()))

    def test_runs_join_adjacent_tensors_up_to_the_slot(self):
        def entry(offset, nbytes):
            return place.Entry("f", "U8", (nbytes,), offset, nbytes)
        entries = [("a", entry(0, 100)), ("b", entry(100, 100)), ("c", entry(200, 300)),
                   ("d", entry(600, 50)), ("e", entry(650, 50))]
        self.assertEqual([[n for n, _ in run] for run in place.runs(entries, 256)],
                         [["a", "b"], ["c"], ["d", "e"]])
        self.assertEqual([[n for n, _ in run] for run in place.runs(entries, 1000)],
                         [["a", "b", "c"], ["d", "e"]])

    def test_unmapped_stored_tensors_are_refused(self):
        model = torch.nn.Linear(2, 2)
        reference = adapter.Reference(model, {"w": "weight"})
        adapter.check_tensor_names(reference, {"w": None})
        with self.assertRaisesRegex(ValueError, "unmapped"):
            adapter.check_tensor_names(reference, {"w": None, "b": None})
        with self.assertRaisesRegex(ValueError, "does not have"):
            adapter.check_tensor_names(adapter.Reference(model, {"w": "missing"}), {"w": None})


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class AttentionTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(11)
        self.q, self.k, self.v = (torch.randn((1, 4, 37, 16), generator=generator) for _ in range(3))
        mask = torch.full((37, 37), float("-inf")).triu(1)
        self.mask = mask[None, None]

    def test_one_block_is_hf_eager(self):
        module = torch.nn.Module()
        module.num_key_value_groups, module.training = 2, False
        k, v = self.k[:, :2], self.v[:, :2]
        expected, _ = eager_attention_forward(module, self.q, k, v, self.mask, 0.25)
        got, _ = ops.hf_attention(module, self.q, k, v, self.mask, 0.25)
        self.assertTrue(torch.equal(got, expected))

    def test_query_blocks_and_f64(self):
        whole = ops.attention(self.q.double(), self.k.double(), self.v.double(), 0.25, self.mask.double())
        blocks = ops.attention(self.q.double(), self.k.double(), self.v.double(), 0.25, self.mask.double(),
                               chunk=8)
        self.assertEqual(blocks.dtype, torch.float64)
        self.assertLess((blocks - whole).abs().max().item(), 1e-13)
        scores = (self.q.double() @ self.k.double().transpose(-1, -2)) * 0.25 + self.mask.double()
        exact = torch.softmax(scores, dim=-1) @ self.v.double()
        self.assertLess((whole - exact).abs().max().item(), 1e-13)


def tiny_llama(attention, seed=0):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    config._attn_implementation = attention
    return LlamaForCausalLM(config).eval()


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class RegisteredAttentionTests(unittest.TestCase):
    """`ops.register()` through a real HF model, so HF builds its mask."""

    def setUp(self):
        self.name = ops.register()
        self.chunk = ops.CHUNK

    def tearDown(self):
        ops.CHUNK = self.chunk

    def forward(self, model, ids):
        with torch.no_grad():
            return model(input_ids=torch.tensor([ids]), use_cache=False).logits[0]

    def test_equals_eager_and_is_causal(self):
        eager, custom = tiny_llama("eager"), tiny_llama(self.name)
        ids, changed = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7]
        self.assertTrue(torch.equal(self.forward(custom, ids), self.forward(eager, ids)))
        self.assertTrue(torch.equal(self.forward(custom, changed)[:-1], self.forward(custom, ids)[:-1]))
        ops.CHUNK = 2
        blocked = self.forward(custom, ids)
        self.assertLess((blocked - self.forward(eager, ids)).abs().max().item(), 1e-5)
        self.assertTrue(torch.equal(self.forward(custom, changed)[:-1], blocked[:-1]))

    def test_hf_masks_equal_eager_masks(self):
        from transformers import masking_utils
        embeds, padded = torch.zeros((1, 6, 8)), torch.tensor([[1, 1, 1, 1, 1, 1]])
        built = {}
        for attention in ("eager", self.name):
            config = tiny_llama(attention).config
            config.sliding_window = 3
            built[attention] = [
                masking_utils.create_causal_mask(config, embeds, padded, None),
                masking_utils.create_sliding_window_causal_mask(config, embeds, padded, None),
                masking_utils.create_bidirectional_mask(config, embeds, padded),
            ]
        causal, sliding, bidirectional = built[self.name]
        for got, expected in zip(built[self.name], built["eager"]):
            self.assertTrue(got is None and expected is None or torch.equal(got, expected))
        self.assertTrue(torch.equal(causal[0, 0] == 0, torch.ones(6, 6, dtype=torch.bool).tril()))
        self.assertTrue(torch.equal(sliding[0, 0] == 0, torch.ones(6, 6, dtype=torch.bool).tril().triu(-2)))
        self.assertTrue(bidirectional is None or not bidirectional.any())

    def test_cached_decoding_matches_full_forwards(self):
        model = tiny_llama(self.name).double()
        ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            step = model(input_ids=torch.tensor([[9]]), past_key_values=out.past_key_values, use_cache=True)
        full = self.forward(model, [1, 2, 3, 4, 9])
        self.assertLess((step.logits[0, -1] - full[-1]).abs().max().item(), 1e-12)


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class ModeFlagTests(unittest.TestCase):
    def test_active_resets_reduced_precision_switches(self):
        cuda = torch.backends.cuda
        cuda.allow_fp16_bf16_reduction_math_sdp(True)
        cuda.matmul.allow_fp16_accumulation = True
        cuda.matmul.allow_bf16_reduced_precision_reduction = True
        cuda.matmul.allow_fp16_reduced_precision_reduction = True
        for name in ("bf16", "f32", "f64"):
            with modes.active(modes.policy(name)):
                self.assertFalse(cuda.fp16_bf16_reduction_math_sdp_allowed())
                self.assertFalse(cuda.matmul.allow_fp16_accumulation)
                self.assertFalse(cuda.matmul.allow_bf16_reduced_precision_reduction)
                self.assertFalse(cuda.matmul.allow_fp16_reduced_precision_reduction)


if __name__ == "__main__":
    unittest.main()
