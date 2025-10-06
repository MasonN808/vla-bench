"""
Simple evaluation wrapper that standardizes timing and result collection.
"""

import time
import json
from datetime import datetime
from pathlib import Path


class BenchmarkEvaluator:
    """
    Simple wrapper for benchmark evaluation that handles:
    - Timing (using time.perf_counter for accuracy)
    - Standardized result format
    - JSON output

    Usage:
        evaluator = BenchmarkEvaluator(
            model="openvla/openvla-7b",
            benchmark="libero",
            subtask="libero_spatial",
            save_dir="./results"
        )

        evaluator.start()
        # ... run evaluation ...
        evaluator.end(success_rate=0.95, num_episodes=50)
    """

    def __init__(
        self,
        model: str,
        benchmark: str,
        subtask: str,
        save_dir: str = "./results",
        **extra_metadata
    ):
        """
        Initialize evaluator.

        Args:
            model: Model name/checkpoint
            benchmark: Benchmark name (e.g., "libero", "vlabench")
            subtask: Subtask/suite name (e.g., "libero_spatial", "track_1_in_distribution")
            save_dir: Directory to save results JSON
            **extra_metadata: Additional metadata to include in results
        """
        self.model = model
        self.benchmark = benchmark
        self.subtask = subtask
        self.save_dir = Path(save_dir)
        self.extra_metadata = extra_metadata

        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.timestamp: str = ""

    def start(self):
        """Start timing the evaluation."""
        self.start_time = time.perf_counter()
        self.timestamp = datetime.now().isoformat()
        print(f"\n[Evaluator] Started: {self.benchmark}/{self.subtask} at {self.timestamp}")

    def end(self, **results):
        """
        End timing and save results.

        Args:
            **results: Result metrics (e.g., success_rate=0.95, num_episodes=50)
        """
        self.end_time = time.perf_counter()
        duration_seconds = self.end_time - self.start_time

        # Create standardized result dictionary
        result_dict = {
            "model": self.model,
            "benchmark": self.benchmark,
            "subtask": self.subtask,
            "timestamp": self.timestamp,
            "duration_seconds": round(duration_seconds, 2),
            **results,
            **self.extra_metadata
        }

        # Save to JSON
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: benchmark_subtask_timestamp.json
        safe_timestamp = self.timestamp.replace(":", "-").replace(".", "-")
        filename = f"{self.benchmark}_{self.subtask}_{safe_timestamp}_results.json"
        filepath = self.save_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"[Evaluator] Completed in {duration_seconds:.2f}s")
        print(f"[Evaluator] Results saved to: {filepath}")

        return result_dict
