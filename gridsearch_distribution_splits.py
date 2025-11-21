#!/usr/bin/env python3
"""
Run a mini grid search over DistributionParams and regenerate split datasets
for each configuration. This helps study how different distribution shifts
impact training/validation/test performance.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List

from data_generation.distribution_config import (
    DistributionParams,
    get_train_distribution_params,
    get_val_test_distribution_params,
)
import generate_split_data


DEFAULT_GRID: List[Dict] = [
    {
        "name": "baseline",
        "train_overrides": {},
        "val_overrides": {},
    },
    {
        "name": "higher_mood_noise",
        "train_overrides": {
            "mood_noise_scale": 1.7,
            "mood_uniform_noise_range": 0.7,
        },
        "val_overrides": {
            "mood_noise_scale": 2.0,
            "mood_uniform_noise_range": 0.8,
        },
    },
    {
        "name": "higher_base_prob",
        "train_overrides": {
            "base_migraine_probability": 0.09,
        },
        "val_overrides": {
            "base_migraine_probability": 0.11,
        },
    },
]


def build_params(base: DistributionParams, overrides: Dict) -> DistributionParams:
    """Return a new DistributionParams with the provided overrides applied."""
    return replace(base, **overrides)


def run_grid(
    grid: Iterable[Dict],
    persons: int,
    days: int,
    person_data: str | None,
    seed: int,
    output_dir: Path,
):
    base_train = get_train_distribution_params()
    base_val = get_val_test_distribution_params()

    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in grid:
        name = spec["name"]
        run_dir = output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)

        train_params = build_params(base_train, spec.get("train_overrides", {}))
        val_params = build_params(base_val, spec.get("val_overrides", {}))

        print("=" * 100)
        print(f"GRID RUN: {name}")
        print("=" * 100)

        generate_split_data.generate_splits(
            persons=persons,
            days=days,
            train_output=str(run_dir / "train_data.csv"),
            val_output=str(run_dir / "val_data.csv"),
            test_output=str(run_dir / "test_data.csv"),
            person_data=person_data,
            seed=seed,
            train_params=train_params,
            val_test_params=val_params,
        )

        print(f"\nCompleted grid configuration '{name}'. Outputs saved in {run_dir}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search over distribution parameters for split generation"
    )
    parser.add_argument(
        "--persons",
        type=int,
        default=1000,
        help="Number of persons per run (default: 1000)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=50,
        help="Number of days per person (default: 50)",
    )
    parser.add_argument(
        "--person-data",
        type=str,
        default="person_data_100000.csv",
        help="Path to static person data file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed forwarded to training split generation (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="grid_runs",
        help="Directory to store per-configuration splits (default: grid_runs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_grid(
        grid=DEFAULT_GRID,
        persons=args.persons,
        days=args.days,
        person_data=args.person_data,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

