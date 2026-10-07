#!/usr/bin/env python3
"""Select the animation frame whose moving-grip reaction is closest to 20 N."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


TARGET_FORCE_N = 20.0
MAX_DISPLACEMENT_MM = 65.0
LOAD_DURATION_S = 0.18
ANIMATION_DT_S = 0.0025


def displacement(time: float, function_points: int = 25) -> float:
    """Integrate the piecewise-linear half-sine velocity used by Radioss."""
    clipped = min(max(time, 0.0), LOAD_DURATION_S)
    dt = LOAD_DURATION_S / function_points
    velocity_scale = MAX_DISPLACEMENT_MM * math.pi / (2.0 * LOAD_DURATION_S)
    full_intervals = min(int(clipped / dt), function_points)
    integral = 0.0
    for index in range(full_intervals):
        v0 = math.sin(math.pi * index / function_points)
        v1 = math.sin(math.pi * (index + 1) / function_points)
        integral += 0.5 * (v0 + v1) * dt
    remainder = clipped - full_intervals * dt
    if remainder > 0.0 and full_intervals < function_points:
        v0 = math.sin(math.pi * full_intervals / function_points)
        v1 = math.sin(math.pi * (full_intervals + 1) / function_points)
        slope = (v1 - v0) / dt
        integral += v0 * remainder + 0.5 * slope * remainder**2
    return velocity_scale * integral


def local_slopes(x_values, y_values, half_window=5):
    """Return local least-squares dy/dx values for a monotone loading path."""
    slopes = []
    for index in range(len(x_values)):
        lo = max(0, index - half_window)
        hi = min(len(x_values), index + half_window + 1)
        x_local = x_values[lo:hi]
        y_local = y_values[lo:hi]
        x_mean = sum(x_local) / len(x_local)
        y_mean = sum(y_local) / len(y_local)
        denominator = sum((value - x_mean) ** 2 for value in x_local)
        numerator = sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_local, y_local)
        )
        slopes.append(numerator / denominator if denominator else float("nan"))
    return slopes


def main():
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "FIG10_CANN_CSRT01.csv")
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        headers = next(reader)
        rows = list(reader)
    if not rows:
        raise ValueError("The time-history CSV is empty.")
    column = {name: index for index, name in enumerate(headers)}
    times = [float(row[column["time"]]) for row in rows]
    displacements = [displacement(time) for time in times]
    external_work = [float(row[column["EXTERNAL WORK"]]) for row in rows]
    generalized_forces = local_slopes(displacements, external_work)
    samples = [
        (time, abs(force), row)
        for time, force, row in zip(times, generalized_forces, rows)
        if math.isfinite(force) and 0.01 * LOAD_DURATION_S < time < 0.99 * LOAD_DURATION_S
    ]
    selected_time, _, _ = min(
        samples, key=lambda sample: abs(sample[1] - TARGET_FORCE_N)
    )
    frame_index = int(round(selected_time / ANIMATION_DT_S)) + 1
    animation_time = (frame_index - 1) * ANIMATION_DT_S
    history_time, history_force, history_row = min(
        samples, key=lambda sample: abs(sample[0] - animation_time)
    )
    internal_energy = float(history_row[column["INTERNAL ENERGY"]])
    kinetic_energy = float(history_row[column["KINETIC ENERGY"]])
    summary = {
        "target_force_N": TARGET_FORCE_N,
        "selected_animation_index": frame_index,
        "selected_animation_time_s": animation_time,
        "history_sample_time_s": history_time,
        "moving_grip_reaction_N": history_force,
        "uniform_top_displacement_mm": displacement(animation_time),
        "reaction_source": "local slope d(EXTERNAL WORK)/d(grip displacement)",
        "force_range_N": [
            min(sample[1] for sample in samples),
            max(sample[1] for sample in samples),
        ],
        "internal_energy": internal_energy,
        "kinetic_energy": kinetic_energy,
        "kinetic_internal_energy_ratio_percent": (
            100.0 * kinetic_energy / internal_energy if internal_energy else None
        ),
    }
    Path("selected_state.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"FIG10_CANN_CSRA{frame_index:03d}")
    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
