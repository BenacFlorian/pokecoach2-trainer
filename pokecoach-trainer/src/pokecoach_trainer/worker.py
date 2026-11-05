"""Point d'entrée du worker enfant."""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from pokecoach_trainer.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--kind", default="training")
    parser.add_argument("--role", default="compute")
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--worker-kind", default="download")
    parser.add_argument("--lock-fd", type=int, default=None)
    args = parser.parse_args(argv)

    if args.role in {"collection-supervisor", "collection-worker"}:
        root = get_settings().storage_root
    else:
        os.chdir(get_settings().vgc_bench_root.resolve())
        root = get_settings().storage_root

    try:
        if args.role == "collection-supervisor":
            from pokecoach_trainer.collection_supervisor import run_collection_supervisor

            assert args.campaign_id and args.lock_fd is not None
            code = run_collection_supervisor(root, args.campaign_id, lock_fd=args.lock_fd)
            return code
        if args.role == "collection-worker":
            from pokecoach_trainer.collection_workers import run_collection_worker

            assert args.campaign_id and args.worker_id
            run_collection_worker(
                root,
                args.campaign_id,
                args.worker_id,
                "discovery" if args.worker_kind == "discovery" else "download",
            )
            return 0
        from pokecoach_trainer.job_service import run_job

        run_job(args.job_id)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
