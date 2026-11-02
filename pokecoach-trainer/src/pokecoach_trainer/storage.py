"""Chemins bornés, identifiants et écriture JSON atomique."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
from pathlib import Path
from typing import Any

from pokecoach_trainer.settings import get_settings, storage_path

_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


def validate_id(value: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"identifiant invalide: {value!r}")
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_inbox_relative(relative_path: str) -> Path:
    """Résout un chemin relatif strictement sous `storage/inbox/`."""

    rel = relative_path.strip()
    if rel.startswith("/") or rel.startswith("..") or "\\" in rel:
        raise ValueError("chemin inbox non autorisé")
    inbox = storage_path("inbox").resolve()
    target = (inbox / rel).resolve()
    if inbox not in target.parents and target != inbox:
        raise ValueError("chemin hors inbox")
    if target.is_symlink():
        raise ValueError("symlink interdit")
    return target


def resolve_storage_relative(*parts: str) -> Path:
    root = get_settings().storage_root.resolve()
    target = root.joinpath(*parts).resolve()
    if root not in target.parents and target != root:
        raise ValueError("chemin hors storage")
    if any(p.is_symlink() for p in target.parents) or target.is_symlink():
        raise ValueError("symlink interdit")
    return target


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def new_opaque_id(prefix: str = "") -> str:
    token = secrets.token_hex(8)
    return validate_id(f"{prefix}{token}" if prefix else token)
