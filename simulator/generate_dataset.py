from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from core import make_robot, step_robot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate labelled training data for the challenge")
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--robots", type=int, default=8)
    p.add_argument("--seed", type=int, default=606)
    p.add_argument("--output", default="data/training.csv")
    p.add_argument("--fleets", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.fleets, args.samples, args.robots) < 1:
        raise SystemExit("--fleets, --samples y --robots deben ser positivos.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for fleet_index in range(args.fleets):
        fleet_id = f"F{fleet_index + 1:02d}"
        rng = random.Random(args.seed + fleet_index)
        robots = [make_robot(i + 1, rng) for i in range(args.robots)]

        for sample_index in range(args.samples):
            robot = robots[sample_index % len(robots)]
            sample, status = step_robot(robot, rng)
            rows.append({**sample, "fleet_id": fleet_id, "status": status})

    fieldnames = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {label: sum(r["status"] == label for r in rows) for label in ["NORMAL", "WARNING", "CRITICAL"]}
    print(f"Dataset written to {out} ({len(rows)} rows)")
    print("Class counts:", counts)


if __name__ == "__main__":
    main()
