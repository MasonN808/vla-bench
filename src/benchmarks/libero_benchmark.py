"""
LIBERO Benchmark Integration for VLA Testing

This module provides a wrapper for testing Vision-Language-Action (VLA) models
on the LIBERO simulation benchmark.

Usage:
    python -m src.benchmarks.libero --model openvla --task_suite libero_spatial
"""

import os
import sys
from pathlib import Path

# Add OpenVLA experiments path
OPENVLA_PATH = Path(__file__).parent.parent.parent / "LIBERO" / "openvla"
sys.path.insert(0, str(OPENVLA_PATH))

from experiments.robot.libero.run_libero_eval import GenerateConfig
# Import the unwrapped eval_libero function directly from the module
import experiments.robot.libero.run_libero_eval as libero_eval_module


def run_libero_benchmark(
    model_checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial",
    task_suite: str = "libero_spatial",
    num_trials: int = 50,
    center_crop: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "vla-bench",
    wandb_entity: str = "your-entity",
    seed: int = 7,
):
    """
    Run OpenVLA evaluation on LIBERO benchmark.

    Args:
        model_checkpoint: HuggingFace checkpoint or local path
        task_suite: LIBERO task suite (libero_spatial, libero_object, libero_goal, libero_10)
        num_trials: Number of rollouts per task
        center_crop: Whether to use center crop (set True for models trained with augmentations)
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        seed: Random seed for reproducibility
    """

    # Create config
    config = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=model_checkpoint,
        task_suite_name=task_suite,
        num_trials_per_task=num_trials,
        center_crop=center_crop,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        seed=seed,
    )

    # Run evaluation
    print(f"Starting LIBERO evaluation:")
    print(f"  Model: {model_checkpoint}")
    print(f"  Task Suite: {task_suite}")
    print(f"  Trials per task: {num_trials}")
    print(f"  Center crop: {center_crop}")

    # Access the unwrapped function through __wrapped__ attribute added by draccus
    eval_libero_unwrapped = libero_eval_module.eval_libero.__wrapped__
    eval_libero_unwrapped(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LIBERO benchmark evaluation")
    parser.add_argument("--model", type=str, default="openvla/openvla-7b-finetuned-libero-spatial",
                        help="Model checkpoint (HuggingFace or local path)")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                        help="LIBERO task suite to evaluate on")
    parser.add_argument("--num_trials", type=int, default=50,
                        help="Number of trials per task")
    parser.add_argument("--center_crop", action="store_true", default=False,
                        help="Use center crop (for models trained with augmentations)")
    parser.add_argument("--no-center_crop", action="store_false", dest="center_crop",
                        help="Disable center crop")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="vla-bench",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="your-entity",
                        help="W&B entity name")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed")

    args = parser.parse_args()

    run_libero_benchmark(
        model_checkpoint=args.model,
        task_suite=args.task_suite,
        num_trials=args.num_trials,
        center_crop=args.center_crop,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
    )
