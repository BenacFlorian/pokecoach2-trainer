"""Point d'entrée du worker enfant."""

from __future__ import annotations

import argparse

from pokecoach_trainer.training_service import run_training_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)
    try:
        run_training_job(args.job_id)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
