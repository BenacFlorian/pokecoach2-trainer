"""Vérification de rechargement SB3 dans un processus isolé."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from stable_baselines3 import PPO

from pokecoach_trainer.schemas import ModelConfig, ReloadReport
from pokecoach_trainer.storage import read_json, sha256_file
from pokecoach_trainer.upstream import vgc_bench_workdir


def verify_reload(model_path: Path, manifest_path: Path) -> ReloadReport:
    manifest = read_json(manifest_path)
    expected_hash = manifest.get("checkpoint_sha256", "")
    digest = sha256_file(model_path)
    if expected_hash and digest != expected_hash:
        return ReloadReport(
            ok=False,
            checkpoint_sha256=digest,
            param_count=0,
            architecture=ModelConfig(),
            message="hash checkpoint incorrect",
        )
    arch = ModelConfig.model_validate(manifest.get("architecture", {}))
    with vgc_bench_workdir():
        from vgc_bench.src.policy import MaskedActorCriticPolicy

        loaded = PPO.load(str(model_path), device="cpu")
        policy = loaded.policy
        if not isinstance(policy, MaskedActorCriticPolicy):
            return ReloadReport(
                ok=False,
                checkpoint_sha256=digest,
                param_count=0,
                architecture=arch,
                message="classe policy inattendue",
            )
        param_count = sum(p.numel() for p in policy.parameters())
        pi = policy.pi_features_extractor
        if (
            pi.embed_layers != arch.num_layers
            or pi.num_heads != arch.num_heads
            or pi.embed_len != arch.embed_len
        ):
            return ReloadReport(
                ok=False,
                checkpoint_sha256=digest,
                param_count=param_count,
                architecture=arch,
                message="architecture rechargée différente du manifeste",
            )
    return ReloadReport(
        ok=True,
        checkpoint_sha256=digest,
        param_count=param_count,
        architecture=arch,
        message="trainer_reload=passed",
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print("Usage: reload_check MODEL.zip MANIFEST.json", file=sys.stderr)
        return 2
    report = verify_reload(Path(argv[0]), Path(argv[1]))
    print(json.dumps(report.model_dump(mode="json")))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
