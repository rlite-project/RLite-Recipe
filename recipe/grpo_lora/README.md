# GRPO Recipe with LoRA for Efficient LLM Fine-tuning

This recipe demonstrates how to efficiently fine-tune Large Language Models (LLMs) using Reinforcement Learning from Human Feedback (RLHF), specifically employing the GRPO (Generalized Reinforcement Learning from Preference Optimization) algorithm cottura with LoRA (Low-Rank Adaptation).

## Core Features

*   **GRPO for Policy Optimization**: Utilizes the GRPO algorithm for robust policy learning from preference data.
*   **LoRA for Efficient Fine-tuning**: Integrates LoRA to significantly reduce the number of trainable parameters, enabling efficient adaptation of large models like Qwen2.5-7B.
*   **HuggingFace Model Support**: Compatible with a wide range of LLMs available on HuggingFace Hub, configured via `model_path` in `GrpoConfig`.
*   **Sequence Packing**: Implements sequence packing techniques to improve training throughput by concatenating multiple short sequences into a single longer sequence.
*   **Weights & Biases Integration**: Seamlessly logs training metrics, configurations, and (optionally) model checkpoints to Weights & Biases for experiment tracking and visualization.

## File Structure

*   `grpo_packing.py`: The main training script. It includes the `GrpoConfig` class for managing all hyperparameters and settings.
*   `utils.py`: Contains utility functions that might be used by the training script (e.g., custom data processing, specific reward calculations if not part of the core GRPO logic).
*   `requirements.txt`: Lists the specific Python dependencies required for this recipe.
*   `README.md`: This documentation file.

## Environment Setup

1.  **Install RLite**: Ensure you have the [RLite library](https://github.com/rlite-project/RLite) installed. Please follow the installation instructions in the main RLite repository.
2.  **Install Dependencies**: Install the necessary Python packages specified in the requirements file:
    ```bash
    pip install -r recipe/grpo_lora/requirements.txt
    ```

## Configuration

All hyperparameters and settings for this recipe are managed within the `GrpoConfig` class in the `grpo_packing.py` script. Key parameters you might want to configure include:

*   **Model Settings**:
    *   `model_path`: Path to the pretrained model (e.g., `"Qwen/Qwen2.5-7B"`).
    *   `train_data_path`: Path or identifier for the training dataset.
    *   `val_dataset_paths`: Dictionary of validation datasets.
*   **LoRA Configuration**:
    *   `lora_r`: The rank of the LoRA matrices.
    *   `lora_alpha`: The scaling factor for LoRA.
    *   `lora_dropout`: Dropout probability for LoRA layers.
    *   `lora_target_modules`: Specifies which modules to apply LoRA to (e.g., `"all-linear"`).
*   **W&B Logging**:
    *   `wandb_project`: Your Weights & Biases project name.
    *   `wandb_name`: A specific name for the W&B run.
*   **Batch & Sequence Sizes**:
    *   `num_prompts_per_batch`: Total number of prompts per batch across all GPUs.
    *   `num_rollouts_per_prompt`: Number of rollouts (generated sequences) per prompt.
    *   `pack_length`: The target length for packed sequences.
*   **Engine & Training**:
    *   `train_engine`: Specifies the training engine (e.g., `"fsdp2"` which supports LoRA).
    *   `lr`: Learning rate.
    *   `max_epochs`: Maximum number of training epochs.

Please refer to the `GrpoConfig` class in `grpo_packing.py` for a complete list of configurable parameters and their default values.

## Running the Recipe

To run the GRPO with LoRA training script:

1.  **Set up Python Path**: Ensure the project root is in your `PYTHONPATH` to allow for correct module imports.
    ```bash
    export PYTHONPATH=$PWD:$PYTHONPATH
    ```
    (Note: `$PWD` should refer to the root of the `rlite-recipe` repository)

2.  **Execute the script**:
    ```bash
    python recipe/grpo_lora/grpo_packing.py
    ```
    You might need to configure `CUDA_VISIBLE_DEVICES` or other environment variables depending on your specific hardware setup and distributed training requirements.

## Expected Results & Logging

Training progress, including loss, rewards, and other relevant metrics, will be logged to Weights & Biases if configured. You can monitor your experiments in real-time by navigating to your W&B project dashboard.
