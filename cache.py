"""Versioned, atomic local cache for prepared Club Coven audio."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np


VERSION = 1


def cache_key(source, config, rules):
    digest = sha256()
    digest.update(json.dumps([VERSION, config, rules], sort_keys=True).encode())
    for name in ("material.py", "accompaniment.py", "backing.py", "cache.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    with source.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cache(path, key):
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) == key:
                return {name: data[name] for name in data.files if name != "key"}
    except (OSError, ValueError, KeyError, BadZipFile):
        pass
    return None


def write_cache(path, key, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
        temporary = Path(file.name)
    try:
        np.savez(temporary, key=key, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
