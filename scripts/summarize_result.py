#!/usr/bin/env python3
"""Summarize an overall_test result JSON file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


STATE_COLUMNS = [
    "susceptible",
    "latent",
    "infected_untested",
    "infected_tested",
    "infected_asymptomatic",
    "in_hospital",
    "recovered",
    "death",
]


def summarize(path: Path, interval: int) -> list[dict[str, float]]:
    data = np.array(json.loads(path.read_text()), dtype=float)
    if data.ndim != 3 or data.shape[2] != len(STATE_COLUMNS):
        raise ValueError(
            f"Expected result shape [time, region, 8], got {tuple(data.shape)}"
        )

    rows = []
    totals = data.sum(axis=1)
    population = totals[0].sum()
    for index, values in enumerate(totals):
        row = {
            "step": index,
            "time_slot": index * interval,
            "population": population,
        }
        row.update(dict(zip(STATE_COLUMNS, values)))
        row["active_infections"] = (
            row["latent"]
            + row["infected_untested"]
            + row["infected_tested"]
            + row["infected_asymptomatic"]
            + row["in_hospital"]
        )
        row["cumulative_infections"] = population - row["susceptible"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="result/city_sample/overall_0.json",
        help="Path to an overall_*.json result file.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Simulation interval used for each saved test step.",
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    args = parser.parse_args()

    rows = summarize(Path(args.path), args.interval)
    first = rows[0]
    last = rows[-1]

    print(f"file: {args.path}")
    print(f"shape: [{len(rows)}, regions, {len(STATE_COLUMNS)}]")
    print(f"population: {int(last['population'])}")
    print(f"time slots: {int(first['time_slot'])} -> {int(last['time_slot'])}")
    print(f"final susceptible: {int(last['susceptible'])}")
    print(f"final active infections: {int(last['active_infections'])}")
    print(f"final cumulative infections: {int(last['cumulative_infections'])}")
    print(f"final deaths: {int(last['death'])}")
    print(f"peak active infections: {int(max(row['active_infections'] for row in rows))}")
    print(f"peak hospital load: {int(max(row['in_hospital'] for row in rows))}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote csv: {args.csv}")


if __name__ == "__main__":
    main()
