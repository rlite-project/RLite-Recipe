from functools import cached_property

import numpy as np
import rlite
import torch
from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import AutoConfig, Qwen2ForCausalLM
from vllm import RequestOutput

import wandb


class GrpoConfig:
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    train_data_path: str = "zwhe99/DeepMath-103K"
    val_dataset_paths: dict[str, str] = {
        "math-500": "HuggingFaceH4/MATH-500",
        "aime-2024": "sea-snell/aime-2024",
        "aime-2025": ("opencompass/AIME2025", ["AIME2025-I", "AIME2025-II"]),
    }
    use_liger_kernel: bool = True

    # wandb config
    wandb_project: str = "on-policy-grpo-math-verifiable-reward"
    wandb_name: str = "packing-qwen2.5-7b-zero"

    # batch size
    num_prompts_per_batch: int = 1024  # All gpus sum up
    num_rollouts_per_prompt: int = 8
    pack_length: int = 16384 + 1024

    # engine config
    train_engine: str = "fsdp2"
    vllm_tp_size: int = 1
    fsdp2_num_shards: int = 8

    # data config
    max_response_length: int = 16384

    # optimizer config
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2

    # training config
    max_epochs: int = 20
    eval_interval: int = 5


if GrpoConfig.use_liger_kernel:
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel
    _apply_liger_kernel("qwen2")


class DataPacker:
    def __init__(self, max_length: int, pad_token_id: int):
        self.data_pack = {}
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def append(self, input_ids: list[int], **kwargs):
        data_dict = {
            "input_ids": input_ids,
            "position_ids": list(range(len(input_ids))),
            **kwargs
        }
        for k, v in data_dict.items():
            self.data_pack.setdefault(k, []).extend(v)

    def export(self, reset: bool = True) -> dict:
        data_pack = self.data_pack
        if reset:
            self.data_pack = {}
        return data_pack

    def __len__(self):
        if len(self.data_pack) == 0:
            return 0
        return len(self.data_pack["input_ids"])

    @staticmethod
    def collate(data_packs: list[dict]) -> dict:
        return {
            k: [x[k] for x in data_packs]
            for k in data_packs[0].keys()
        }


if GrpoConfig.train_engine == "fsdp2":
    from rlite.nn import HuggingFaceFsdp2TrainModule as RliteTrainModule
elif GrpoConfig.train_engine == "fsdp":
    from rlite.nn import HuggingFaceFsdpTrainModule as RliteTrainModule


class GrpoPolicyActor(RliteTrainModule, Qwen2ForCausalLM):
    def on_weights_materialized(self):
        import torch.optim as optim

        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=GrpoConfig.lr,
            betas=GrpoConfig.betas,
            weight_decay=GrpoConfig.weight_decay
        )

    def grpo_train_step(
        self,
        input_ids,
        position_ids,
        action_mask,
        scores,
        grad_acc_steps: int = 1
    ):
        input_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")
        position_ids = torch.tensor(position_ids, dtype=torch.long, device="cuda")
        action_mask = torch.tensor(action_mask, dtype=torch.long, device="cuda")
        scores = torch.tensor(scores, dtype=torch.bfloat16, device="cuda")

        logits = self(input_ids, position_ids=position_ids).logits
        logits = logits[:, :-1, :]
        logp = torch.log_softmax(logits, dim=-1)
        chosen_logp = torch.gather(logp, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        action_mask = action_mask[:, 1:]
        scores = scores[:, 1:]
        loss = (
            -torch.sum(chosen_logp * scores * action_mask)
            / torch.sum(action_mask)
            / grad_acc_steps
        )
        loss.backward()

        return self.collect_metrics(loss, logp, chosen_logp, action_mask)

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def collect_metrics(
        self,
        loss: torch.Tensor,
        logp: torch.Tensor,
        chosen_logp: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> dict:
        return {
            "loss": loss.item(),
            "gradient_norm": self.get_gradient_norm().item(),
            "ppl": self.get_ppl(chosen_logp, action_mask).item(),
            "entropy": self.get_entropy(logp, action_mask).item(),
        }

    def get_ppl(self, chosen_logp: torch.Tensor, action_mask: torch.Tensor):
        with torch.no_grad():
            return (-chosen_logp * action_mask).sum() / action_mask.sum()

    def get_entropy(self, logp: torch.Tensor, action_mask: torch.Tensor):
        with torch.no_grad():
            token_entropy = -torch.sum(logp * logp.exp(), dim=-1)
            seq_entropy = (token_entropy * action_mask).sum(-1) / action_mask.sum(-1)
            return seq_entropy.mean()

    def get_gradient_norm(self):
        with torch.no_grad():
            return torch.norm(
                torch.stack([
                    torch.norm(p.grad) for p in self.parameters() if p.grad is not None
                ])
            )


class SimpleGrpoExp:
    def run(self):
        rlite.init()
        self.init_wandb()
        for epoch in range(GrpoConfig.max_epochs):
            for indx, batch in enumerate(self.train_loader):
                self.global_index = indx + epoch * len(self.train_loader)
                if self.global_index % GrpoConfig.eval_interval == 0:
                    self.evaluate()
                self.train_step(*batch)

    def train_step(self, prompts: list[str], gts: list[str]):
        from rlite import NeedParallel
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=GrpoConfig.max_response_length,
            n=GrpoConfig.num_rollouts_per_prompt,
        )
        generations = self.policy_rollout.generate(
            prompts,
            sampling_params=sampling_params,
            tqdm_desc=f"[Iter {self.global_index}] Generate"
        )
        scores = self.calculate_scores(prompts, generations, gts)
        self.policy_rollout.meta()

        data_packs = self.prepare_train_inputs_packing(generations, scores)
        self.policy_actor.cuda()
        metrics = []
        for data_pack in tqdm(data_packs, desc=f"[Iter {self.global_index}] Train"):
            mb_metrics = self.policy_actor.grpo_train_step(
                NeedParallel(data_pack["input_ids"]),
                NeedParallel(data_pack["position_ids"]),
                NeedParallel(data_pack["action_mask"]),
                NeedParallel(data_pack["scores"]),
                grad_acc_steps=len(data_packs)
            )
            metrics.append(mb_metrics)
        self.wandb_log_metrics(metrics)
        self.policy_actor.optimizer_step()

        self.sync_weights()

    def sync_weights(self):
        self.policy_actor.release_memory_cache(gpu=True)
        self.policy_rollout.cuda("weights")
        self.policy_actor.p2p_weight_sync(self.policy_rollout)
        self.policy_actor.cpu()
        self.policy_rollout.cuda("kv_cache")

    def calculate_scores(
        self,
        prompts: list[str],
        generations: list[RequestOutput],
        gts: list[str]
    ):
        gen_texts = [
            [output.text for output in prompt_response.outputs]
            for prompt_response in generations
        ]
        scores = [
            self.math_checker(response, gt)
            for group_gen_texts, gt in zip(gen_texts, gts)
            for response in group_gen_texts
        ]
        scores = np.reshape(scores, (len(prompts), GrpoConfig.num_rollouts_per_prompt))

        # We also log the length of the generations
        lengths = [len(x.token_ids) for generation in generations for x in generation.outputs]
        wandb.log(
            {
                "train/accuracy": np.mean(scores),
                "train/length_mean": np.mean(lengths),
                "train/length_std": np.std(lengths),
            },
            step=self.global_index
        )
        # GRPO score
        std = np.std(scores, axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)
        scores = (scores - np.mean(scores, axis=1, keepdims=True)) / std
        return scores

    def prepare_train_inputs_packing(self, generations, scores):
        from rlite.resman import total_gpus

        packed_data = []
        current_pack = DataPacker(GrpoConfig.pack_length, self.tokenizer.pad_token_id)
        for generation, group_scores in zip(generations, scores):
            prompt_ids = generation.prompt_token_ids
            for output, score in zip(generation.outputs, group_scores):
                output_ids = list(output.token_ids)
                input_ids = prompt_ids + output_ids
                if len(current_pack) + len(input_ids) > GrpoConfig.pack_length:
                    packed_data.append(current_pack.export())
                current_pack.append(
                    input_ids=input_ids,
                    action_mask=[0] * len(prompt_ids) + [1] * len(output_ids),
                    scores=[0] * len(prompt_ids) + [score] * len(output_ids)
                )
        # The number of data packs can be smaller than the number of GPUs.
        # In this case, we need to bootstrap the data packs to make sure
        # each GPU has one data pack.
        num_total_gpus = total_gpus()
        if len(packed_data) % num_total_gpus != 0:
            import random
            num_packs_to_sample = num_total_gpus - len(packed_data) % num_total_gpus
            packed_data.extend(random.choices(packed_data, k=num_packs_to_sample))

        # Organize such that each GPU has one pack
        num_packs = len(packed_data) // num_total_gpus
        packed_data = [
            DataPacker.collate(packed_data[i::num_packs])
            for i in range(num_packs)
        ]

        return packed_data

    def evaluate(self):
        from vllm import SamplingParams

        for name, (prompts, gts) in self.val_data.items():
            sampling_params = SamplingParams(max_tokens=GrpoConfig.max_response_length)
            generations = self.policy_rollout.generate(
                prompts,
                sampling_params=sampling_params,
                tqdm_desc=f"[Iter {self.global_index}] Evaluate {name}"
            )
            gen_texts = [x.outputs[0].text for x in generations]
            acc = np.mean([self.math_checker(x, y) for x, y in zip(gen_texts, gts)])
            lengths = [len(x.outputs[0].token_ids) for x in generations]
            wandb.log(
                {
                    f"eval/accuracy-{name}": acc,
                    f"eval/length_mean-{name}": np.mean(lengths),
                    f"eval/length_std-{name}": np.std(lengths),
                },
                step=self.global_index
            )

    @cached_property
    def policy_actor(self):
        from rlite import RliteTrainEngine

        with init_empty_weights():
            config = AutoConfig.from_pretrained(
                GrpoConfig.model_path,
                attn_implementation="flash_attention_2"
            )
            module = GrpoPolicyActor(config=config)
            module.gradient_checkpointing_enable({"use_reentrant": False})

        model = RliteTrainEngine(module, executor=GrpoConfig.train_engine)
        model.build(
            tensor_parallel_size=GrpoConfig.fsdp2_num_shards,
            colocate_with=self.policy_rollout
        )
        return model

    @cached_property
    def policy_rollout(self):
        from rlite import RliteInferenceEngine

        model = RliteInferenceEngine(
            GrpoConfig.model_path,
            executor="vllm",
        )
        model.build(tensor_parallel_size=GrpoConfig.vllm_tp_size)
        return model

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(GrpoConfig.model_path)

    @cached_property
    def train_loader(self):
        import datasets
        from torch.utils.data import DataLoader

        class MathDataset:
            def __init__(
                self,
                path: str,
                tokenizer,
            ):
                guide = " Let\'s think step by step and output the final answer after \"####\"."

                raw_data = datasets.load_dataset(path)["train"]
                self.prompts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": item["question"] + guide}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    for item in raw_data
                ]
                self.gts = [str(item["final_answer"]) for item in raw_data]

            def __len__(self):
                return len(self.gts)

            def __getitem__(self, idx):
                return self.prompts[idx], self.gts[idx]

        dataset = MathDataset(GrpoConfig.train_data_path, self.tokenizer)

        return DataLoader(
            dataset,
            batch_size=GrpoConfig.num_prompts_per_batch,
            shuffle=True,
            drop_last=True,
        )

    @cached_property
    def val_data(self):
        import datasets

        val_datasets: dict[str, tuple[list[str], list[str]]] = {}
        guide = " Let\'s think step by step and output the final answer after \"####\"."
        for name, path in GrpoConfig.val_dataset_paths.items():
            if isinstance(path, str):
                path = (path, ["default"])
            path, subsets = path
            prompts, gts = [], []
            for subset in subsets:
                raw_data = datasets.load_dataset(path, subset)["test"]
                question_field = (
                    "question" if "question" in raw_data.features else "problem"
                )
                prompts.extend([
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": item[question_field] + guide}],
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    for item in raw_data
                ])
                gts.extend([str(item["answer"]) for item in raw_data])
            val_datasets[name] = (prompts, gts)
        return val_datasets

    def math_checker(self, response: str, ground_truth: str) -> bool:
        from recipe.on_policy_grpo_math_verifiable_reward.utils import (
            compute_score
        )
        return compute_score(response, ground_truth)

    def init_wandb(self):
        config = {
            k: v for k, v in GrpoConfig.__dict__.items()
            if not k.startswith("_")
        }
        wandb.init(
            project=GrpoConfig.wandb_project,
            name=GrpoConfig.wandb_name,
            config=config
        )

    def wandb_log_metrics(self, metrics: list[list[list[dict]]]):
        import numpy as np

        keys = list(metrics[0][0][0].keys())
        metrics = sum(sum(metrics, []), [])  # Flatten the metrics
        metrics = {k: [metric[k] for metric in metrics] for k in keys}
        wandb.log(
            {f"train/{k}": np.mean(v) for k, v in metrics.items()},
            step=self.global_index
        )


if __name__ == "__main__":
    exp = SimpleGrpoExp()
    exp.run()
