"""Command-line entry point for reproducible thesis-v1 stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from regime_sentinel_worker.pipeline.baseline import run_baseline
from regime_sentinel_worker.regime import run_msgarch_preflight_stage, run_regime_stage


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "protocol" / "thesis-v1.json"
ARTIFACT_ROOT = REPO_ROOT / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "regime", "msgarch-preflight"))
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument(
        "--baseline-artifacts",
        type=Path,
        help="Existing verified baseline directory; defaults to <artifacts>/baseline.",
    )
    args = parser.parse_args()
    rscript = args.rscript
    rscript_path = Path(rscript)
    if rscript_path.parent != Path("."):
        rscript = str(rscript_path.resolve())
    root = args.artifacts.resolve()
    baseline_root = (
        args.baseline_artifacts.resolve()
        if args.baseline_artifacts is not None
        else root / "baseline"
    )
    if args.stage == "baseline":
        result = run_baseline(
            protocol_path=PROTOCOL_PATH,
            repo_root=REPO_ROOT,
            artifact_root=root / "baseline",
        )
    elif args.stage == "msgarch-preflight":
        result = run_msgarch_preflight_stage(
            protocol_path=PROTOCOL_PATH,
            repo_root=REPO_ROOT,
            baseline_root=baseline_root,
            artifact_root=root,
            rscript=rscript,
        )
    else:
        result = run_regime_stage(
            protocol_path=PROTOCOL_PATH,
            repo_root=REPO_ROOT,
            baseline_root=baseline_root,
            artifact_root=root / "regime",
            rscript=rscript,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
