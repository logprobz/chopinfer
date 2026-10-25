"""Pinned diagnostic captures, which never establish release qualification."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

MARKER = "experimental_determinism"
MAX_PIN_BYTES = 65536


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("experimental pin contains a duplicate key")
        result[key] = value
    return result


def document(raw):
    value = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != {
            "model_options", "checkpoint_identity", "startup_nonce"}:
        raise ValueError("experimental pin needs the complete options, checkpoint and startup nonce")
    options = value["model_options"]
    if not isinstance(options, dict) or not options or not all(
            isinstance(key, str) and key and type(item) in (bool, int, str)
            for key, item in options.items()):
        raise ValueError("experimental pin has invalid typed model options")
    for key, size in (("checkpoint_identity", 64), ("startup_nonce", 32)):
        if not isinstance(value[key], str) or not re.fullmatch(f"[0-9a-f]{{{size}}}", value[key]):
            raise ValueError("experimental pin has an invalid " + key)
    return value


@dataclass(frozen=True)
class Pin:
    raw: bytes
    file_sha256: str

    @classmethod
    def load(cls, path, expected_sha256):
        if not isinstance(expected_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
            raise ValueError("experimental capture requires the sealed pin's SHA256")
        descriptor = os.open(Path(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o022 or info.st_size > MAX_PIN_BYTES:
                raise ValueError("experimental pin must be a bounded regular file without group or world writes")
            raw = source.read(MAX_PIN_BYTES + 1)
        pin = cls(raw, expected_sha256)
        pin.value()
        return pin

    def value(self):
        if len(self.raw) > MAX_PIN_BYTES or hashlib.sha256(self.raw).hexdigest() != self.file_sha256:
            raise ValueError("experimental pin digest differs from the sealed bytes")
        return document(self.raw)

    def marker(self):
        value = self.value()
        return {"pin": value, "pin_sha256": hashlib.sha256(canonical(value)).hexdigest(),
                "file_sha256": self.file_sha256, "qualified": False}

    def validate(self, server):
        from common import live_identity
        expected = self.value()
        live = live_identity(server)
        if live.get("model_options_qualified") is not False:
            raise ValueError("experimental capture requires explicitly unqualified model options")
        actual = {key: live.get(key) for key in expected}
        if canonical(actual) != canonical(expected):
            raise ValueError("experimental options, checkpoint or startup nonce differ from the sealed pin")


def selected(args):
    enabled = getattr(args, "experimental", False)
    path = getattr(args, "experimental_pin", None)
    digest = getattr(args, "experimental_pin_sha256", None)
    if not enabled:
        if path is not None or digest is not None:
            raise ValueError("experimental pin arguments require --experimental")
        return None
    if path is None or digest is None or not getattr(args, "server_log", None):
        raise ValueError("experimental record requires a pin, its SHA256 and --server-log")
    return Pin.load(path, digest)
