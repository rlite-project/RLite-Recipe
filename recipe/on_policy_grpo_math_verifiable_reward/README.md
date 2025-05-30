# GRPO Recipe: On-Policy RL for Math with Verifiable Reward

This repository provides an implementation of the Generalized Reward Policy Optimization (GRPO) algorithm for large language models (LLMs) on mathematical problem-solving tasks, with a focus on verifiable reward signals. The codebase supports efficient on-policy RL training, including advanced features such as sequence packing and integration with math-verifiable reward functions.

## Features

- **On-policy RL for LLMs**: Implements GRPO for training LLMs on math datasets.
- **Verifiable Reward**: Uses symbolic math checking to provide reliable reward signals.
- **Efficient Training**: Supports sequence packing and distributed training (FSDP2).
- **Model Support**: Out-of-the-box support for Qwen2, DeepSeek, and other HuggingFace-compatible models.
- **Evaluation**: Built-in evaluation on public math benchmarks (MATH-500, AIME-2024, AIME-2025).
- **Logging**: Integrated with Weights & Biases (wandb) for experiment tracking.

## Directory Structure

- `grpo.py`: Main GRPO training logic for standard (non-packed) sequences.
- `grpo_packing.py`: GRPO training with sequence packing for efficient training. In this recipe, we start from a long-cot distiled model and show that `rlite` can integrate easily with other open-source tools such as `liger-kernel`.
- `utils.py`: Utility functions for math answer normalization and reward computation.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

## Installation

```bash
pip install -r requirements.txt
```

You will also need to install [rlite](https://github.com/rlite-project/RLite) first.

## Usage

```bash
# Optional: create your ray cluster using `ray start`

python recipe/on_policy_grpo_math_verifiable_reward/grpo.py
# Or run the packing version
python recipe/on_policy_grpo_math_verifiable_reward/grpo_packing.py
```

## Datasets

- **Training**: [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
- **Validation**: [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), [AIME-2024](https://huggingface.co/datasets/sea-snell/aime-2024), [AIME-2025](https://huggingface.co/datasets/opencompass/AIME2025)

## Math Verifiable Reward

The reward function uses symbolic normalization and equivalence checking to ensure that model outputs are mathematically correct, not just textually similar.

## Logging

Experiments are tracked using [Weights & Biases](https://wandb.ai/). Set your project and run names in the `GrpoConfig` class. Our logs are available [here](https://api.wandb.ai/links/han-zhang-stepfun/bcehg4y9).

## Citation

If you use this codebase, please cite the relevant papers and repositories for GRPO, rlite, and the math datasets.
