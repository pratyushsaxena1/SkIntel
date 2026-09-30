"""Measure local CPU inference time with a synthetic RGB image."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime

from inference import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    image = np.random.default_rng(101).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    for _ in range(5):
        analyze(image)
    timings = []
    for _ in range(args.runs):
        start = time.perf_counter()
        analyze(image)
        timings.append((time.perf_counter() - start) * 1000)
    result = {
        "input": "synthetic RGB uint8 image, 192x256",
        "operation": "preprocessing, segmentation, overlay, classification",
        "provider": "CPUExecutionProvider",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": onnxruntime.__version__,
        "warmup_runs": 5,
        "measured_runs": args.runs,
        "median_ms": round(float(np.median(timings)), 2),
        "p95_ms": round(float(np.percentile(timings, 95)), 2),
        "excludes": "model loading, browser UI, network; does not measure accuracy",
    }
    report = json.dumps(result, indent=2) + "\n"
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)


if __name__ == "__main__":
    main()
