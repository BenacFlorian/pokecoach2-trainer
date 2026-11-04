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


def _resolve_under_storage_root(relative_path: str, root_name: str) -> Path:
    rel = relative_path.strip()
    if rel.startswith("/") or rel.startswith("..") or "\\" in rel:
        raise ValueError(f"chemin {root_name} non autorisé")
    root = storage_path(root_name).resolve()
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"chemin hors {root_name}")
    if target.is_symlink():
        raise ValueError("symlink interdit")
    return target


def resolve_inbox_relative(relative_path: str) -> Path:
    """Résout un chemin relatif strictement sous `storage/inbox/`."""

    return _resolve_under_storage_root(relative_path, "inbox")


def resolve_archive_relative(relative_path: str) -> Path:
    """Résout un chemin relatif strictement sous `storage/archive/`."""

    return _resolve_under_storage_root(relative_path, "archive")


def resolve_inbox_or_base_model_relative(relative_path: str, *, root_name: str) -> Path:
    """Résout un chemin sous `inbox/` ou `base_models/` (enregistrement parent)."""

    if root_name not in ("inbox", "base_models"):
        raise ValueError("racine non autorisée")
    return _resolve_under_storage_root(relative_path, root_name)


def base_model_dir(base_model_id: str) -> Path:
    """Répertoire immuable `storage/base_models/{id}/`."""

    return storage_path("base_models", validate_id(base_model_id))


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


def canonical_json_sha256(payload: Any) -> str:
    """Empreinte SHA-256 d'un JSON canonique UTF-8 (clés triées, sans espaces)."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    """Empreinte SHA-256 du texte UTF-8."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_jsonl_atomic(path: Path, line_obj: dict[str, Any]) -> None:
    """Ajoute une ligne JSONL avec flush/fsync (append atomique par ligne)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(line_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
