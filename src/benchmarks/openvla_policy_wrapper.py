"""
Custom OpenVLA policy wrapper for VLABench that supports base models without LoRA.

This wrapper allows using openvla/openvla-7b base model directly without requiring
a separate LoRA checkpoint, while maintaining compatibility with VLABench's evaluation framework.
"""

import torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from VLABench.evaluation.model.policy.base import Policy
from VLABench.utils.utils import quaternion_to_euler

# Camera view indices for different tasks
CAMERA_VIEW_INDEX = {
    "select_painting": 1,
    "put_box_on_painting": 1,
    "select_chemistry_tube": 2,
    "find_unseen_object": 2,
    "texas_holdem": 2,
    "cluster_toy": 2
}


class OpenVLAPolicy(Policy):
    """
    OpenVLA policy wrapper that works with base models (no LoRA required).

    This is compatible with VLABench's evaluation framework but doesn't require
    modifying VLABench's source code.
    """

    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    def __init__(self, model_checkpoint: str, device: str = "cuda", **kwargs):
        """
        Initialize OpenVLA policy from a HuggingFace checkpoint.

        Args:
            model_checkpoint: HuggingFace model ID or local path (e.g., "openvla/openvla-7b")
            device: Device to run on (default: "cuda")
        """
        self.device = device

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_checkpoint,
            trust_remote_code=True
        )

        # Load model
        model = AutoModelForVision2Seq.from_pretrained(
            model_checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)

        # Initialize base Policy class
        super().__init__(model)

    def process_observation(self, obs, unnorm_key):
        """
        Process observation into model inputs.

        Args:
            obs: Observation dict with 'instruction', 'rgb', 'ee_state'
            unnorm_key: Task name for selecting camera view

        Returns:
            Processed inputs for the model
        """
        # Select camera view based on task
        cam_index = CAMERA_VIEW_INDEX.get(unnorm_key, 2)  # default: front-view
        instruction = obs["instruction"]
        prompt = self.build_prompt(instruction)
        rgb = obs["rgb"][cam_index]

        # Process image and prompt
        inputs = self.processor(
            prompt,
            Image.fromarray(rgb).convert("RGB")
        ).to(self.device, dtype=torch.bfloat16)

        return inputs

    def build_prompt(self, instruction: str) -> str:
        """
        Build prompt from instruction.

        Args:
            instruction: Task instruction

        Returns:
            Formatted prompt string
        """
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut: "
        # Remove _seen/_unseen suffixes
        prompt = prompt.replace("_seen", "")
        prompt = prompt.replace("_unseen", "")
        return prompt

    def predict(self, obs, unnorm_key=None):
        """
        Predict action from observation.

        Args:
            obs: Observation dict
            unnorm_key: Task name for camera selection and denormalization

        Returns:
            Tuple of (target_pos, target_euler, gripper_state)
        """
        # Process observation
        inputs = self.process_observation(obs, unnorm_key)

        # Predict delta action
        delta_action = self.model.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False
        )

        # Get current end-effector state
        current_ee_state = obs["ee_state"]

        # Parse current state (handle both quaternion and euler representations)
        if len(current_ee_state) == 8:
            # Format: [x, y, z, qx, qy, qz, qw, gripper]
            pos, quat = current_ee_state[:3], current_ee_state[3:7]
            euler = quaternion_to_euler(quat)
        elif len(current_ee_state) == 7:
            # Format: [x, y, z, roll, pitch, yaw, gripper]
            pos, euler = current_ee_state[:3], current_ee_state[3:6]
        else:
            raise ValueError(f"Unexpected ee_state length: {len(current_ee_state)}")

        # Compute target state from delta action
        target_pos = np.array(pos) + delta_action[:3]
        target_euler = euler + delta_action[3:6]

        # Gripper state (open if >= 0.1, closed otherwise)
        gripper_open = delta_action[-1]
        gripper_state = np.ones(2) * 0.04 if gripper_open >= 0.1 else np.zeros(2)

        return target_pos, target_euler, gripper_state

    @property
    def name(self):
        """Policy name for logging."""
        return "OpenVLA"
