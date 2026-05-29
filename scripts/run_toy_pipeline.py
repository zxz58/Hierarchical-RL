#!/usr/bin/env python3
"""Run a short end-to-end smoke pipeline on the toy city_sample data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def ensure_toy_data(repo_root: Path) -> None:
    # before the smoke test starts, guarantees the toy files exist
    subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "create_toy_dataset.py")],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    # for adding argument in the command line
    parser.add_argument("--city", default="city_sample")
    parser.add_argument("--rnn-iterations", type=int, default=1)
    parser.add_argument("--rl-steps", type=int, default=1)
    parser.add_argument("--test-steps", type=int, default=1)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--skip-rnn", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    # locate repo and code directory
    repo_root = Path(__file__).resolve().parents[1]
    code_dir = repo_root / "code"
    sys.path.insert(0, str(code_dir))

    # generate toy data
    ensure_toy_data(repo_root)

    import os

    # change working dir to code_dir
    os.chdir(code_dir)

    if not args.skip_rnn:
        from RNN_train import rnn_train

        train_path = str(repo_root / "data" / args.city / "rnn_train_0.json")
        eval_path = str(repo_root / "data" / args.city / "rnn_eval.json")
        for mode in ("L", "Iut", "R"):
            print(f"\n=== Training {mode}_RNN ===")
            # creates the RNN trainer and trains one RNN for that target state
            trainer = rnn_train(iteration=args.rnn_iterations, city=args.city)
            trainer.train(path_train=[[train_path]], path_eval=eval_path, mode=mode)

    if not args.skip_rl:
        from RL_train import RL_train

        print("\n=== Training RL policy ===")
        trainer = RL_train(
            train_steps=args.rl_steps,
            batch_size=1,
            pool_volume=4,
            interval=args.interval,
            bed_total=100,
            mask_total=1000,
            city=args.city,
        )
        trainer.train()

    if not args.skip_test:
        from overall_test import RL_test

        print("\n=== Running overall test ===")
        tester = RL_test(
            steps=args.test_steps,
            interval=args.interval,
            repeat=args.repeat,
            bed_total=100,
            mask_total=1000,
            city=args.city,
        )
        tester.test()


if __name__ == "__main__":
    main()
