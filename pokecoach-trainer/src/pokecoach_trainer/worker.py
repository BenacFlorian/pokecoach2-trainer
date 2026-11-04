"""Point d'entrée du worker enfant."""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from pokecoach_trainer.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    os.chdir(get_settings().vgc_bench_root.resolve())
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--kind", default="training")
    args = parser.parse_args(argv)
    try:
        from pokecoach_trainer.job_service import run_job

        run_job(args.job_id)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
