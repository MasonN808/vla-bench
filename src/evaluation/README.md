# Standardized Evaluation System

This module provides a simple, standardized way to collect evaluation results across all benchmarks and export them to CSV.

## How It Works

### 1. **Automatic Result Collection**

Both LIBERO and VLABench benchmarks now automatically:
- Time the evaluation using `time.perf_counter()`
- Save results in a standardized JSON format
- Store results in `./results/` directory

### 2. **Standardized Result Format**

Each evaluation creates a JSON file like:
```json
{
  "model": "openvla/openvla-7b-finetuned-libero-spatial",
  "benchmark": "libero",
  "subtask": "libero_spatial",
  "timestamp": "2025-10-06T20:21:25",
  "duration_seconds": 1234.56,
  "success_rate": 0.95,
  "num_episodes": 50
}
```

### 3. **Export to CSV**

Convert all JSON results to a single CSV:
```bash
python src/evaluation/results_to_csv.py
```

This generates `evaluation_results.csv` with columns:
- `model` - Model checkpoint used
- `benchmark` - Benchmark name (libero, vlabench)
- `subtask` - Specific task suite (libero_spatial, track_1_in_distribution, etc.)
- `timestamp` - When evaluation started
- `duration_seconds` - How long evaluation took
- `success_rate` - Success rate (0.0 to 1.0)
- `num_episodes` - Total episodes evaluated

## Usage

### Run LIBERO Evaluation
```bash
python -m src.benchmarks.libero_benchmark \
  --model openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite libero_spatial \
  --num_trials 50 \
  --results_dir ./results
```

### Run VLABench Evaluation
```bash
python -m src.benchmarks.vlabench_benchmark \
  --model openvla/openvla-7b \
  --eval_track track_1_in_distribution \
  --n_episodes 20 \
  --results_dir ./results
```

### Generate CSV Report
```bash
# Default: reads from ./results, outputs to ./evaluation_results.csv
python src/evaluation/results_to_csv.py

# Custom paths:
python src/evaluation/results_to_csv.py \
  --results_dir ./my_results \
  --output my_evaluation.csv
```

## Adding New Benchmarks

To integrate a new benchmark:

1. Import the evaluator:
```python
from evaluation.evaluator import BenchmarkEvaluator
```

2. Wrap your evaluation:
```python
def run_my_benchmark(model, task, ...):
    # Initialize evaluator
    evaluator = BenchmarkEvaluator(
        model=model,
        benchmark="my_benchmark",
        subtask=task,
        save_dir="./results"
    )

    # Start timing
    evaluator.start()

    # Run your evaluation
    results = my_eval_function(...)

    # Save results
    evaluator.end(
        success_rate=results['success_rate'],
        num_episodes=results['num_episodes']
    )
```

That's it! Your results will automatically be included in CSV exports.

## File Structure

```
results/
├── libero_libero_spatial_2025-10-06T20-21-25_results.json
├── libero_libero_object_2025-10-06T21-30-45_results.json
├── vlabench_track_1_in_distribution_2025-10-06T22-15-30_results.json
└── ...

evaluation_results.csv  # Generated from all JSON files
```

## Key Features

- ✅ **Simple**: Just 3 lines of code to integrate
- ✅ **No modifications to submodules**: Parses existing log files
- ✅ **Automatic timing**: Uses `time.perf_counter()` for accuracy
- ✅ **Extensible**: Easy to add new benchmarks
- ✅ **CSV export**: Single command to generate reports
