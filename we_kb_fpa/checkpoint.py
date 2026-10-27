"""Atomic local checkpoints with checksums, previous-generation recovery and run locking.

Pickles contain executable Python state: load only this program's own local
checkpoints. Configuration, source and dependency hashes isolate incompatible runs.
"""
from dataclasses import asdict, is_dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
from importlib.metadata import version
import uuid


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported checkpoint setting: {type(value)}")


@lru_cache(maxsize=1)
def implementation_signature():
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for folder in ("we_kb_fpa", "experiments"):
        for path in sorted((root / folder).glob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    environment = dict(python=platform.python_version(), system=platform.system(),
                       packages={name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn")})
    digest.update(json.dumps(environment, sort_keys=True).encode())
    return digest.hexdigest()


def identity_hash(settings):
    data = json.dumps(dict(schema=1, implementation=implementation_signature(), settings=settings),
                      sort_keys=True, default=_json_default, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def namespace(directory, settings):
    path = Path(directory) / identity_hash(settings)[:24]
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "settings.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(settings, indent=2, default=_json_default), encoding="utf-8")
    return path


class Checkpoint:
    def __init__(self, directory, name, settings=None, resume=True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity_hash(settings or {})
        self.path = self.directory / f"{name}-{self.identity[:24]}.pkl"
        self.previous = self.path.with_suffix(".previous.pkl")
        self.resume = resume
        self.recovered_previous = False

    def load(self):
        if not self.resume:
            return None
        failures = []
        for path in (self.path, self.previous):
            if not path.exists():
                continue
            try:
                envelope = pickle.loads(path.read_bytes())
                if envelope["identity"] != self.identity:
                    raise ValueError("Checkpoint configuration does not match")
                if hashlib.sha256(envelope["payload"]).hexdigest() != envelope["sha256"]:
                    raise ValueError("Checkpoint checksum failed")
                state = pickle.loads(envelope["payload"])
                if path == self.previous:
                    self.recovered_previous = True
                    print(f"Recovered previous checkpoint: {path.name}", flush=True)
                return state
            except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError, TypeError) as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError(f"No valid checkpoint for {self.path.name}: {'; '.join(failures)}")
        return None

    def save(self, state, compact=False):
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        envelope = dict(identity=self.identity, sha256=hashlib.sha256(payload).hexdigest(), payload=payload)
        temporary = self.path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as stream:
                pickle.dump(envelope, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            if self.path.exists() and not self.recovered_previous:
                os.replace(self.path, self.previous)
            os.replace(temporary, self.path)
            self.recovered_previous = False
            if compact:
                self.previous.unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)


class RunLock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.stream.seek(0, os.SEEK_END) == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError("Another process is already writing this output directory") from error
        return self

    def __exit__(self, *unused):
        if os.name == "nt":
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
