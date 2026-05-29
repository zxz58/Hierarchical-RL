#!/usr/bin/env python3
"""Create a deterministic toy dataset for the bundled city_sample config.

The model definitions in this repository are hard-coded for 673 regions, so the
fixture keeps that region count while using simple synthetic values.
"""

from __future__ import annotations

import json
from pathlib import Path


REGION_COUNT = 673
TIME_STEPS = 48
FEATURE_COUNT = 8


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, separators=(",", ":"))
        file.write("\n")


def write_no_travel_prob(path: Path) -> None:
    """Write probabilities shaped [48, 673, 674].

    Each region keeps all travelers in the same region. The final probability
    slot is the simulator's extra "returns to origin" bucket.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        file.write("[")
        for time_index in range(TIME_STEPS):
            if time_index:
                file.write(",")
            file.write("[")
            for region_index in range(REGION_COUNT):
                if region_index:
                    file.write(",")
                row = ["0"] * (REGION_COUNT + 1)
                row[region_index] = "1"
                file.write("[" + ",".join(row) + "]")
            file.write("]")
        file.write("]\n")


def build_start_state() -> list[list[int]]:
    start = []
    for region_index in range(REGION_COUNT):
        susceptible = 80 + region_index % 21
        latent = 1 + (region_index % 3 == 0)
        infected_untested = 1 + (region_index % 5 == 0)
        infected_tested = 1 + (region_index % 7 == 0)
        infected_asymptomatic = region_index % 2
        in_hospital = region_index % 4 == 0
        recovered = region_index % 6
        death = 0
        start.append(
            [
                int(susceptible),
                int(latent),
                int(infected_untested),
                int(infected_tested),
                int(infected_asymptomatic),
                int(in_hospital),
                int(recovered),
                int(death),
            ]
        )
    return start


def build_rnn_sequence(start: list[list[int]], length: int) -> list[list[list[int]]]:
    # creates fake temporal epidemic sequences simple SIR  ? why training data here
    sequence = []
    for step in range(length):
        frame = []
        for region_index, row in enumerate(start):
            adjusted = row.copy()
            adjusted[0] = max(0, adjusted[0] - step)
            adjusted[1] += step % 2
            adjusted[2] += (step + region_index) % 2
            adjusted[3] += step
            adjusted[6] += step // 2
            frame.append(adjusted)
        sequence.append(frame)
    return sequence


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "city_sample"

    start = build_start_state()
    population = [sum(row[:-1]) for row in start]
    flow = [round(0.5 + (index % 17) / 16, 6) for index in range(REGION_COUNT)]
    dense = [round(0.75 + (index % 29) / 28, 6) for index in range(REGION_COUNT)]

    write_json(data_dir / "start.json", start)
    write_json(data_dir / "pop_region.json", population)
    write_json(data_dir / "flow.json", flow)
    write_json(data_dir / "dense.json", dense)
    write_no_travel_prob(data_dir / "prob.json")
    write_json(data_dir / "rnn_train_0.json", build_rnn_sequence(start, 4))
    write_json(data_dir / "rnn_eval.json", build_rnn_sequence(start, 4))

    print(f"Wrote toy dataset to {data_dir}")
    print(f"start.json shape: [{REGION_COUNT}, {FEATURE_COUNT}]")
    print(f"prob.json shape: [{TIME_STEPS}, {REGION_COUNT}, {REGION_COUNT + 1}]")


if __name__ == "__main__":
    main()
