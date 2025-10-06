"""
VLABench Benchmark Integration for VLA Testing

This module provides a wrapper for testing Vision-Language-Action (VLA) models
on the VLABench benchmark, which evaluates models across multiple generalization
dimensions including long-horizon reasoning tasks.

VLABench evaluates VLA models across 6 key dimensions:
1. In-distribution task learning
2. Cross-category generalization
3. Common sense reasoning
4. Semantic instruction understanding
5. Cross-task skill transfer
6. Visual robustness with unseen textures

Usage:
    python -m src.benchmarks.vlabench_benchmark --model openvla --eval_track track_1_in_distribution
"""

import os
import sys
from pathlib import Path
import argparse
import json

# Set MuJoCo to use EGL for headless rendering (must be set before importing VLABench)
os.environ["MUJOCO_GL"] = "egl"

# Import standardized evaluator
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.evaluator import BenchmarkEvaluator

# Add VLABench path to sys.path
VLABENCH_PATH = Path(__file__).parent.parent.parent / "VLABench"
if VLABENCH_PATH.exists():
    sys.path.insert(0, str(VLABENCH_PATH))
else:
    print(f"Warning: VLABench not found at {VLABENCH_PATH}")
    print("Please clone VLABench:")
    print("  git clone https://github.com/OpenMOSS/VLABench.git")
    print("  cd VLABench && pip install -e .")
    sys.exit(1)


def run_vlabench_benchmark(
    model_checkpoint: str,
    eval_track: str = "track_1_in_distribution",
    tasks: str = None,
    n_episodes: int = 20,
    lora_checkpoint: str = None,
    save_dir: str = "./vlabench_results",
    visualization: bool = False,
    metrics: str = "success_rate",
    use_wandb: bool = False,
    wandb_project: str = "vla-bench",
    wandb_entity: str = "your-entity",
    results_dir: str = "./results",
):
    """
    Run VLA evaluation on VLABench benchmark with standardized result collection.

    Args:
        model_checkpoint: HuggingFace checkpoint or local path for the base model
        eval_track: Evaluation track to run. Options:
            - track_1_in_distribution: In-distribution task learning
            - track_2_cross_category: Cross-category generalization
            - track_3_common_sense: Common sense reasoning
            - track_4_semantic_instruction: Semantic instruction understanding
            - track_5_cross_task: Cross-task skill transfer
            - track_6_unseen_texture: Visual robustness with unseen textures
        tasks: Comma-separated list of specific tasks to run (e.g., "task1,task2")
               If None, runs all tasks in the evaluation track
        n_episodes: Number of episodes to run per task
        lora_checkpoint: Optional LoRA checkpoint path for fine-tuned models
        save_dir: Directory to save evaluation results (VLABench format)
        visualization: Whether to enable visualization during evaluation
        metrics: Evaluation metrics to compute (success_rate, progress_score, etc.)
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        results_dir: Directory to save standardized results JSON
    """

    # Initialize standardized evaluator for timing and result collection
    benchmark_evaluator = BenchmarkEvaluator(
        model=model_checkpoint,
        benchmark="vlabench",
        subtask=eval_track,
        save_dir=results_dir,
        lora_checkpoint=lora_checkpoint if lora_checkpoint else "none",
    )

    # Start timing
    benchmark_evaluator.start()

    try:
        # Import VLABench modules
        from VLABench.evaluation.evaluator import Evaluator
        import VLABench.tasks
        import VLABench.robots

        # Import appropriate OpenVLA class based on whether we have a LoRA checkpoint
        if lora_checkpoint:
            # Use VLABench's OpenVLA class for LoRA fine-tuned models
            from VLABench.evaluation.model.policy.openvla import OpenVLA
        else:
            # Use our custom wrapper for base models (no LoRA required)
            from benchmarks.openvla_policy_wrapper import OpenVLAPolicy as OpenVLA
    except ImportError as e:
        print(f"Error importing VLABench modules: {e}")
        print("Make sure VLABench is properly installed:")
        print("  cd VLABench")
        print("  pip install -r requirements.txt")
        print("  pip install -e .")
        print("  python scripts/download_assets.py")
        sys.exit(1)

    # Setup wandb if requested
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config={
                    "model": model_checkpoint,
                    "eval_track": eval_track,
                    "n_episodes": n_episodes,
                    "tasks": tasks,
                }
            )
        except ImportError:
            print("Warning: wandb not installed. Continuing without logging.")
            use_wandb = False

    # Load episode configuration for the evaluation track
    # Configs are in VLABench/VLABench/configs/ (nested directory structure)
    config_path = VLABENCH_PATH / "VLABench" / "configs" / "evaluation" / "tracks" / f"{eval_track}.json"
    if not config_path.exists():
        print(f"Error: Evaluation track config not found at {config_path}")
        print(f"Available tracks: track_1_in_distribution, track_2_cross_category, "
              f"track_3_common_sense, track_4_semantic_instruction, "
              f"track_5_cross_task, track_6_unseen_texture")
        sys.exit(1)

    with open(config_path, 'r') as f:
        episode_config = json.load(f)

    # Parse tasks if specified, otherwise use all tasks from episode_config
    if tasks:
        task_list = tasks.split(',')
    else:
        # Extract task names from episode_config (dict keys are task names)
        task_list = list(episode_config.keys())

    # Print evaluation configuration
    print(f"\nStarting VLABench evaluation:")
    print(f"  Model: {model_checkpoint}")
    print(f"  Evaluation Track: {eval_track}")
    print(f"  Tasks: {task_list if task_list else 'All tasks in track'}")
    print(f"  Episodes per task: {n_episodes}")
    print(f"  LoRA checkpoint: {lora_checkpoint if lora_checkpoint else 'None'}")
    print(f"  Save directory: {save_dir}")
    print(f"  Visualization: {visualization}")
    print(f"  Metrics: {metrics}")
    print()

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the evaluator
    evaluator = Evaluator(
        episode_config=episode_config,
        tasks=task_list,
        n_episodes=n_episodes,
        save_dir=save_dir,
        visualization=visualization,
        metrics=metrics.split(','),
    )

    # Load the OpenVLA policy
    print("Loading OpenVLA policy...")
    if lora_checkpoint:
        # VLABench's OpenVLA requires both model_ckpt and lora_ckpt
        policy = OpenVLA(
            model_ckpt=model_checkpoint,
            lora_ckpt=lora_checkpoint,
        )
    else:
        # Our custom wrapper only needs model_checkpoint
        policy = OpenVLA(model_checkpoint=model_checkpoint)

    # Run evaluation
    print("Running evaluation...")
    results = evaluator.evaluate(policy)

    # Save results
    results_path = os.path.join(save_dir, f"results_{eval_track}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete!")
    print(f"Results saved to: {results_path}")

    # Print summary statistics
    if 'success_rate' in results:
        print(f"\nSuccess Rate: {results['success_rate']:.2%}")
    if 'progress_score' in results:
        print(f"Progress Score: {results['progress_score']:.2f}")

    # Log to wandb if enabled
    if use_wandb:
        wandb.log(results)
        wandb.finish()

    # Save standardized results
    # Extract key metrics from VLABench results
    success_rate = results.get('success_rate', 0.0)
    num_episodes = n_episodes * len(task_list) if task_list else n_episodes

    benchmark_evaluator.end(
        success_rate=success_rate,
        num_episodes=num_episodes,
        **{k: v for k, v in results.items() if k != 'success_rate'}  # Include other metrics
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VLABench benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example:
            python -m src.benchmarks.vlabench_benchmark \\
                --model openvla/openvla-7b \\
                --eval_track track_1_in_distribution \\
                --n_episodes 20 \\
                --tasks select_toy,stack_blocks
                    """
        )

    parser.add_argument("--model", type=str, required=True,
                        help="Model checkpoint (HuggingFace ID or local path)")
    parser.add_argument("--eval_track", type=str, default="track_1_in_distribution",
                        choices=[
                            "track_1_in_distribution",
                            "track_2_cross_category",
                            "track_3_common_sense",
                            "track_4_semantic_instruction",
                            "track_5_cross_task",
                            "track_6_unseen_texture"
                        ],
                        help="Evaluation track to run")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated list of specific tasks (default: all tasks in track)")
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="Number of episodes per task")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="LoRA checkpoint path for fine-tuned models")
    parser.add_argument("--save_dir", type=str, default="./vlabench_results",
                        help="Directory to save results")
    parser.add_argument("--visualization", action="store_true",
                        help="Enable visualization during evaluation")
    parser.add_argument("--metrics", type=str, default="success_rate",
                        help="Comma-separated list of metrics to compute")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="vla-bench",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="your-entity",
                        help="W&B entity name")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory to save standardized results JSON")

    args = parser.parse_args()

    run_vlabench_benchmark(
        model_checkpoint=args.model,
        eval_track=args.eval_track,
        tasks=args.tasks,
        n_episodes=args.n_episodes,
        lora_checkpoint=args.lora_checkpoint,
        save_dir=args.save_dir,
        visualization=args.visualization,
        metrics=args.metrics,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        results_dir=args.results_dir,
    )
