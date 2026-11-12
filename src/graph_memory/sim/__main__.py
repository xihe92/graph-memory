"""python -m graph_memory.sim [scenario ...] [--model oracle|haiku|sonnet] [--embedder hash|real]

Runs scenarios from the library and prints a report per scenario. With a real model the
same scenarios measure extraction and recall quality instead of plumbing; that costs
model calls (about two per graduation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .harness import Simulator
from .scenarios import SCENARIOS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m graph_memory.sim")
    ap.add_argument("scenarios", nargs="*", help=f"default: all of {', '.join(SCENARIOS)}")
    ap.add_argument("--model", default="oracle", help="oracle (default), haiku, sonnet, ...")
    ap.add_argument("--embedder", default="hash", choices=["hash", "real"])
    ap.add_argument("--recall", default="on", choices=["on", "shadow"])
    ap.add_argument("-v", "--verbose", action="store_true", help="every step, not only failures")
    ap.add_argument("--keep", action="store_true", help="keep each run's memory folder")
    ap.add_argument("--json", type=Path, help="write the full reports here")
    args = ap.parse_args(argv)

    names = args.scenarios or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        ap.error(f"unknown scenarios {unknown}; have {list(SCENARIOS)}")
    reports = []
    for name in names:
        scenario = SCENARIOS[name]()
        if args.model != "oracle" and (
            scenario.oracle_only or (scenario.expensive and not args.scenarios)
        ):
            why = "needs the oracle" if scenario.oracle_only else "expensive; name it to run"
            print(f"== {name}: skipped with a real model ({why})")
            continue
        sim = Simulator(
            scenario,
            model=args.model,
            embedder=args.embedder,
            recall_mode=args.recall,
            keep=args.keep,
            log=(lambda m: print("   ", m, file=sys.stderr)) if args.verbose else None,
        )
        rep = sim.run()
        reports.append(rep)
        print(rep.text(verbose=args.verbose), flush=True)
    failed = [r.scenario for r in reports if not r.ok]
    print(
        f"\n{len(reports) - len(failed)}/{len(reports)} scenarios passed"
        + (f"; failed: {', '.join(failed)}" if failed else "")
    )
    if args.json:
        args.json.write_text(json.dumps([r.to_json() for r in reports], indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
