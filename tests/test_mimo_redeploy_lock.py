"""Exercise two dev redeploys with a CPU-only fake server, for MiMo and Qwen3.8."""

import os
import json
from pathlib import Path
import select
import signal
import socket
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def executable(path, body):
    path.write_text(body)
    path.chmod(0o755)


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def redeploy(root, port, env, model, extra=()):
    command = [
        "bash", "-c", 'cargo() { :; }; export -f cargo; exec "$@"',
        "redeploy-test", str(ROOT / f"tools/dev/{model}-redeploy.sh"),
        "--checkpoint", str(root / "checkpoint"),
        "--model-cache-dir", str(root / "model-cache"),
        "--work-dir", str(root / "work"),
        "--logs-dir", str(root / "logs"),
        "--port", str(port),
        "--startup-timeout", "8",
    ]
    if model == "mimo":
        command += ["--drafter", "none"]
    command += list(extra)
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=20)


def main():
    with tempfile.TemporaryDirectory(prefix="mimo-redeploy-lock-") as temporary:
        root = Path(temporary)
        (root / "checkpoint").mkdir()
        for name in ["tokenizer.json", "tokenizer_config.json", "config.json"]:
            (root / "checkpoint" / name).write_text("{}")
        (root / "checkpoint/model.safetensors.index.json").write_text(
            '{"weight_map": {"model.visual.patch_embed.proj.weight": "a.safetensors"}}')
        target = root / "target/release"
        target.mkdir(parents=True)
        # Cargo target directories are often group writable; the private
        # install must not be.
        target.chmod(0o775)
        for module in ["libbundle_mimo26.so", "libbundle_qwen38.so"]:
            (target / module).write_bytes(b"fake module")
            (target / module).chmod(0o664)
        for model in ["mimo26", "qwen38"]:
            executable(target / f"chopinfer-model-{model}", """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
arguments = sys.argv[1:]
assert arguments[:2] == ['prepare', '--direct'], arguments
options = dict(zip(arguments[2::2], arguments[3::2]))
here = Path(sys.argv[0]).resolve().parent
model = here.name and Path(sys.argv[0]).name.removeprefix('chopinfer-model-')
assert (here.parent / 'lib' / f'libbundle_{model}.so').is_file(), 'module is not in ../lib'
bundle = Path(options['--cache-dir']) / f'p-{model}-{options["--tensor-parallel-size"]}'
fresh = not bundle.exists()
bundle.mkdir(parents=True, exist_ok=True)
(bundle / 'manifest.json').write_text(json.dumps({'id': model}))
if fresh:
    with open(os.environ['CHOPINFER_TEST_ACTIONS'], 'a') as log:
        log.write('prepare ' + model + '\\n')
print(json.dumps({'result': 'prepared-direct', 'bundle': str(bundle), 'profile': {'name': 'fake'}}))
""")
        for package in ["mimo26-package", "qwen38-package"]:
            executable(target / package, "#!/bin/sh\nexit 3\n")
        executable(target / "chopinfer", """#!/usr/bin/env python3
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import sys

if sys.argv[1] != 'serve' or '--model' not in sys.argv:
    raise SystemExit(2)
with open(os.environ['CHOPINFER_TEST_ACTIONS'] + '.serve', 'a') as log:
    log.write(' '.join(sys.argv[1:]) + '\\n')
listen = sys.argv[sys.argv.index('--listen') + 1]
host, port = listen.rsplit(':', 1)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            body = {'status': 'ready'}
        elif self.path == '/v1/models':
            body = {'data': [{'id': 'MiMo-V2.6-Flash-MOPD'}]}
        else:
            self.send_error(404)
            return
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass

server = HTTPServer((host, int(port)), Handler)
print('{"status":"listening"}', flush=True)
server.serve_forever()
""")
        fake_bin = root / "bin"
        fake_bin.mkdir()
        executable(fake_bin / "nvidia-smi", """#!/usr/bin/env python3
import os
import sys
if sys.argv[1] == '--query-gpu=index,uuid,memory.used,memory.total':
    for index in range(4):
        print(f'{index}, GPU-fake{index}, 0, 96000')
elif sys.argv[1] == '--query-compute-apps=pid,used_memory,gpu_uuid':
    held = os.environ.get('CHOPINFER_TEST_HELD_PID')
    if held:
        print(f'{held}, 88000, GPU-fake0')
else:
    raise SystemExit(2)
""")
        env = os.environ.copy()
        env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(f"GPU-fake{index}" for index in range(4))
        env["CHOPINFER_TEST_ACTIONS"] = str(root / "actions")
        env["CARGO_TARGET_DIR"] = str(root / "target")
        mimo = ["bash", str(ROOT / "tools/dev/mimo-redeploy.sh"), "--dry-run",
                "--checkpoint", str(root / "checkpoint")]
        qwen = ["bash", str(ROOT / "tools/dev/qwen38-redeploy.sh"), "--dry-run",
                "--checkpoint", str(root / "checkpoint")]
        def run(command):
            return subprocess.run(command, capture_output=True, text=True, env=env, timeout=10)
        default = run(mimo)
        assert default.returncode == 0, default.stderr
        assert "chopinfer-model-mimo26 prepare --direct --model " in default.stdout
        assert "chopinfer serve --model " in default.stdout
        assert "--tensor-parallel-size 4 --model-cache-dir " in default.stdout
        # The tp4-dflash profile supplies the speculation and every limit.
        for word in ["--speculative-tokens", "--kv-cache-dtype", "--max-sequences", "--bundle", "--topology"]:
            assert word not in default.stdout, word
        # With no dflash/ the helper serves MTP; beside dflash/ MTP needs an
        # explicit package.
        mtp = run(mimo + ["--drafter", "mtp"])
        assert mtp.returncode == 0, mtp.stderr
        assert "bin/mimo26-package --checkpoint" not in mtp.stdout and "serve --model " in mtp.stdout
        (root / "checkpoint/dflash").mkdir()
        mtp = run(mimo + ["--drafter", "mtp"])
        (root / "checkpoint/dflash").rmdir()
        assert mtp.returncode == 0, mtp.stderr
        assert "--output " in mtp.stdout and " --drafter mtp --encoders on" in mtp.stdout
        assert "profiles.json#tp4-mtp --bundle " in mtp.stdout and " --checkpoint " in mtp.stdout
        ar = run(mimo + ["--drafter", "none"])
        assert ar.returncode == 0 and "--speculative-tokens 0" in ar.stdout, ar.stderr
        for command in (mimo, qwen):
            for removed in (["--ranks", "8"], ["--fold-ranks"]):
                rejected = run(command + removed)
                assert rejected.returncode == 2, rejected.stderr
        refused = run(mimo + ["--ranks", "2", "--drafter", "dflash"])
        assert refused.returncode == 2 and "TP4 only" in refused.stderr
        fp8 = run(mimo + ["--kv-cache-dtype", "fp8"])
        assert fp8.returncode == 0, fp8.stderr
        assert "--kv-cache-dtype fp8" in fp8.stdout and "--fp8" not in fp8.stdout.replace("--kv-cache-dtype fp8", "")
        wrong = run(mimo + ["--kv-cache-dtype", "fp16"])
        assert wrong.returncode == 2 and "KV cache dtype must be" in wrong.stderr
        chosen = run(mimo + ["--serve-arg", "--speculative-tokens", "--serve-arg", "5"])
        assert chosen.returncode == 0, chosen.stderr
        assert "--speculative-tokens 5" in chosen.stdout
        for removed, replacement in [
            (["--rank-dir", "x"], "--checkpoint DIR"), (["--drafter-rank-dir", "x"], "--checkpoint DIR"),
            (["--encoder-rank-dir", "x"], "--checkpoint DIR"), (["--encoders"], "plans the vision encoders"),
            (["--install-root", "x"], "--model-cache-dir DIR"), (["--fresh-package"], "--model-cache-dir DIR"),
            (["--topology", "x"], "--ranks N"), (["--architecture", "sm_120f"], "serve --model"),
        ]:
            result = run(mimo + removed)
            assert result.returncode == 2 and replacement in result.stderr, (removed, result.stderr)
        dry_run = run(qwen)
        assert dry_run.returncode == 0, dry_run.stderr
        assert "chopinfer-model-qwen38 prepare --direct --model " in dry_run.stdout
        assert "serve --model " in dry_run.stdout
        assert "--speculative-tokens 0" in dry_run.stdout
        assert "--prefix-cache" not in dry_run.stdout and "--kv-cache-dtype" not in dry_run.stdout
        qwen_mtp = qwen + ["--drafter", "mtp"]
        dry_run = run(qwen_mtp)
        assert dry_run.returncode == 0, dry_run.stderr
        assert "--speculative-tokens" not in dry_run.stdout and "bin/qwen38-package --checkpoint" not in dry_run.stdout
        chosen = run(qwen_mtp + ["--serve-arg", "--speculative-tokens", "--serve-arg", "5"])
        assert chosen.returncode == 0, chosen.stderr
        assert "--speculative-tokens 5" in chosen.stdout
        images = run(qwen_mtp + ["--image-positions", "sequential"])
        assert images.returncode == 0, images.stderr
        assert "--encoders on --image-positions sequential" in images.stdout
        assert "profiles.json#tp4-mtp --bundle " in images.stdout
        refused = run(qwen_mtp[:-1] + ["dflash"])
        assert refused.returncode == 2, refused.stderr
        for removed, replacement in [
            (["--rank-dir", "x"], "--checkpoint DIR"), (["--encoders", "x"], "plans the vision encoders"),
            (["--ngram-placement", "host"], "host memory at TP2"),
            (["--package-max-sequences", "32"], "--max-sequences N"), (["--check-reuse"], "--prepare-only"),
            (["--install-root", "x"], "--work-dir DIR"),
        ]:
            result = run(qwen + removed)
            assert result.returncode == 2 and replacement in result.stderr, (removed, result.stderr)
        cached = run(qwen + ["--prefix-cache", "off"])
        assert cached.returncode == 0, cached.stderr
        assert "--prefix-cache off" in cached.stdout
        for model in ["mimo", "qwen38"]:
            two_redeploys(root, env, model)


def two_redeploys(root, env, model):
    port = free_port()
    pidfile = root / f"logs/{model}-redeploy-{port}.pid"
    pidfds = []
    helper = "mimo26" if model == "mimo" else "qwen38"
    def field(output, name):
        return next(line for line in output.splitlines() if line.startswith(name + ": ")).split(": ", 1)[1]
    def served():
        path = root / "actions.serve"
        return path.read_text().splitlines() if path.exists() else []
    try:
        before = len(served())
        prepared = redeploy(root, port, env, model, ["--prepare-only"])
        assert prepared.returncode == 0, prepared.stdout + prepared.stderr
        package = prepared.stdout.splitlines()[-1].removeprefix("prepared ")
        assert package == str(root / f"model-cache/p-{helper}-4"), prepared.stdout
        assert not pidfile.exists() and len(served()) == before, "--prepare-only started a server"
        first = redeploy(root, port, env, model)
        assert first.returncode == 0, first.stdout + first.stderr
        assert field(first.stdout, "Package") == package
        assert "  prepare: " in first.stdout and "  verify: " not in first.stdout
        assert len(served()) == before + 1
        assert served()[-1].startswith(f"serve --model {root / 'checkpoint'} --tensor-parallel-size 4 "
                                       f"--model-cache-dir {root / 'model-cache'} ")
        first_pid = int(pidfile.read_text().split()[0])
        pidfds.append(os.pidfd_open(first_pid))
        installs = sorted((root / "work/runs").glob("*/install"))
        assert installs, "no private install"
        for path in [installs[-1], installs[-1] / "bin", installs[-1] / "lib",
                     installs[-1] / "bin/chopinfer", installs[-1] / f"bin/chopinfer-model-{helper}",
                     installs[-1] / f"lib/libbundle_{helper}.so"]:
            assert path.stat().st_mode & 0o777 == 0o755, (path, oct(path.stat().st_mode))
        actions = (root / "actions").read_text()
        blocked_env = dict(env, CHOPINFER_TEST_HELD_PID=str(first_pid))
        blocked = redeploy(root, port, blocked_env, model, ["--release-timeout", "1"])
        assert blocked.returncode != 0, "GPU accounting timeout started a replacement"
        assert "GPU 0" in blocked.stderr and f"PID {first_pid}" in blocked.stderr, blocked.stderr
        assert int(pidfile.read_text().split()[0]) == first_pid, "timeout discarded the old pidfile"
        second = redeploy(root, port, env, model)
        assert second.returncode == 0, second.stdout + second.stderr
        second_pid = int(pidfile.read_text().split()[0])
        pidfds.append(os.pidfd_open(second_pid))
        assert second_pid != first_pid
        assert (root / "actions").read_text() == actions, "an unchanged redeploy prepared the package again"
        assert field(first.stdout, "Package") == field(second.stdout, "Package")
        poller = select.poll()
        poller.register(pidfds[0], select.POLLIN)
        assert poller.poll(0), "first server is still running"
        print(f"{model}: prepared once, served privately installed binaries, released the port lock")
    finally:
        for pidfd in reversed(pidfds):
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            except ProcessLookupError:
                pass
            poller = select.poll()
            poller.register(pidfd, select.POLLIN)
            if not poller.poll(2000):
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                poller.poll(2000)
            os.close(pidfd)


if __name__ == "__main__":
    main()
