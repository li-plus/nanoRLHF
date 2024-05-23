from __future__ import annotations

import argparse
import math
import string
from itertools import chain
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from nltk.corpus import words
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    LlamaConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
    set_seed,
)


class ByteTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        self.vocab = string.ascii_lowercase + "EP"  # a-z + eos + pad
        super().__init__(eos_token="E", pad_token="P", **kwargs)

    def get_vocab(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.index(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.vocab[index]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)


class PromptDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int) -> None:
        prompts = list(set(x.lower() + "E" for x in words.words() if len(x) < max_length))
        self.inputs = tokenizer(prompts, padding="max_length", max_length=max_length, return_tensors="pt")

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        return dict(
            input_ids=self.inputs["input_ids"][index],
            attention_mask=self.inputs["attention_mask"][index],
        )

    def __len__(self) -> int:
        return len(self.inputs["input_ids"])


class ExperienceDataset(Dataset):
    def __init__(self, experience: dict[str, torch.Tensor]) -> None:
        self.experience = experience

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {k: v[index] for k, v in self.experience.items()}

    def __len__(self) -> int:
        return len(next(iter(self.experience.values())))


class ValueModel(nn.Module):
    def __init__(self, transformer: nn.Module, device=None, dtype=None) -> None:
        super().__init__()
        self.transformer = transformer
        self.v_head = nn.Linear(transformer.config.hidden_size, 1, bias=False, device=device, dtype=dtype)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        hidden_states = self.transformer(*args, **kwargs, use_cache=False).last_hidden_state
        values = self.v_head(hidden_states).squeeze(-1)
        return values


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591"""
    logp = F.log_softmax(logits, dim=-1)
    logpy = logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logpy


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: int | None = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy


class PPOTrainer:
    def __init__(
        self,
        args,
        tokenizer: PreTrainedTokenizer,
        policy_model: PreTrainedModel,
        value_model: nn.Module,
        reward_model: Callable,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.config = args
        self.tokenizer = tokenizer
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.optimizer = optimizer

    def compute_advantages(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()
        return values, advantages, returns

    @torch.no_grad()
    def sample_experience(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        self.policy_model.eval()
        self.value_model.eval()

        _, prompt_length = prompt_ids.shape

        outputs = self.policy_model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_length=self.config.max_length,
            use_cache=True,
            do_sample=self.config.temp > 0,
            temperature=self.config.temp,
            top_p=self.config.top_p,
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id],
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_logits=True,
        )

        input_ids = outputs.sequences
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        old_logits = torch.stack(outputs.logits, dim=1)
        rewards = self.reward_model(input_ids=input_ids, attention_mask=attention_mask)
        values = self.value_model(input_ids=input_ids, attention_mask=attention_mask)

        old_logprobs = logprobs_from_logits(old_logits, input_ids[:, prompt_length:])

        mask = attention_mask[:, prompt_length:]
        rewards = rewards[:, prompt_length:]
        values = values[:, prompt_length - 1 : -1]

        values, advantages, returns = self.compute_advantages(values, rewards, mask)

        return dict(
            prompt_ids=prompt_ids,
            old_logprobs=old_logprobs,
            values=values,
            rewards=rewards,
            input_ids=input_ids,
            attention_mask=attention_mask,
            advantages=advantages,
            returns=returns,
        )

    def train_minibatch(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.policy_model.train()
        self.value_model.train()

        prompt_ids = inputs["prompt_ids"]
        old_logprobs = inputs["old_logprobs"]
        rewards = inputs["rewards"]
        values = inputs["values"]
        attention_mask = inputs["attention_mask"]
        input_ids = inputs["input_ids"]
        advantages = inputs["advantages"]
        returns = inputs["returns"]

        _, prompt_length = prompt_ids.shape
        mask = attention_mask[:, prompt_length:]

        logits = self.policy_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        logprobs = logprobs_from_logits(logits[:, prompt_length - 1 : -1], input_ids[:, prompt_length:])

        entropy = masked_mean(entropy_from_logits(logits[:, prompt_length - 1 : -1]), mask)
        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
        pg_loss = masked_mean(torch.max(pg_losses1, pg_losses2), mask)
        pg_clipfrac = masked_mean((pg_losses2 > pg_losses1).float(), mask)

        vpreds = self.value_model(input_ids=input_ids, attention_mask=attention_mask)
        vpreds = vpreds[:, prompt_length - 1 : -1]

        vpredclipped = torch.clamp(
            vpreds, min=values - self.config.cliprange_value, max=values + self.config.cliprange_value
        )
        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        loss.backward()
        pg_grad_norm = nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=self.config.max_grad_norm)
        vf_grad_norm = nn.utils.clip_grad_norm_(
            self.value_model.parameters(), max_norm=self.config.max_grad_norm * self.config.vf_coef
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        stats = {
            **{f"lr/group_{i}": pg["lr"] for i, pg in enumerate(self.optimizer.param_groups)},
            "loss/policy": pg_loss.item(),
            "loss/value": vf_loss.item(),
            "loss/total": loss.item(),
            "policy/grad_norm": pg_grad_norm.item(),
            "policy/entropy": entropy.item(),
            "policy/approxkl": approxkl.item(),
            "policy/policykl": policykl.item(),
            "policy/clipfrac": pg_clipfrac.item(),
            "policy/advantages_mean": masked_mean(advantages, mask).item(),
            "policy/advantages_var": masked_var(advantages, mask).item(),
            "policy/ratio_mean": masked_mean(ratio, mask).item(),
            "returns/mean": masked_mean(returns, mask).item(),
            "returns/var": masked_var(returns, mask).item(),
            "val/grad_norm": vf_grad_norm.item(),
            "val/vpred": masked_mean(vpreds, mask).item(),
            "val/error": masked_mean(vf_losses1, mask).item(),
            "val/clipfrac": vf_clipfrac.item(),
            "val/mean": masked_mean(values, mask).item(),
            "val/var": masked_var(values, mask).item(),
            "env/reward_mean": masked_mean(rewards, mask).item(),
            "env/reward_var": masked_var(rewards, mask).item(),
            "env/reward_total": rewards.sum(1).mean().item(),
        }
        return stats


class GoldenRewardModel:
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        input_ids_list = input_ids.tolist()
        prompt_length = input_ids_list[0].index(self.tokenizer.eos_token_id) + 1

        scores = [[0 for _ in range(seq_len)] for _ in range(batch_size)]
        for input_id, score in zip(input_ids_list, scores):
            prompt_id = input_id[:prompt_length]
            target_id = [x for x in prompt_id if x != self.tokenizer.pad_token_id]
            response_id = input_id[prompt_length:]
            for j, (rsp_id, tgt_id) in enumerate(zip(response_id, target_id)):
                if rsp_id != tgt_id:
                    break
                score[prompt_length + j] = 1

        return torch.tensor(scores, dtype=torch.float32, device=input_ids.device)


def round_up(x: float, multiple_of: int) -> int:
    return (math.ceil(x) + multiple_of - 1) // multiple_of * multiple_of


def main():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--policy_hidden_size", default=128, type=int)
    parser.add_argument("--policy_num_hidden_layers", default=2, type=int)
    parser.add_argument("--policy_num_attention_heads", default=2, type=int)
    parser.add_argument("--value_hidden_size", default=256, type=int)
    parser.add_argument("--value_num_hidden_layers", default=4, type=int)
    parser.add_argument("--value_num_attention_heads", default=4, type=int)
    # generation config
    parser.add_argument("--max_prompt_length", default=8, type=str)
    parser.add_argument("--max_length", default=16, type=str)
    parser.add_argument("--temp", default=0.7, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    # training config
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--output_dir", default="llama_rl", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
    parser.add_argument("--num_warmup_steps", default=10, type=int)
    parser.add_argument("--train_ratio", default=0.99, type=float)
    parser.add_argument("--train_batch_size", default=256, type=int)
    parser.add_argument("--val_batch_size", default=256, type=int)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--num_ppo_epochs", default=2, type=int)
    parser.add_argument("--mini_batch_size", default=128, type=int)
    parser.add_argument("--max_grad_norm", default=0.1, type=float)
    parser.add_argument("--vf_coef", default=2.0, type=float)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--lam", default=0.95, type=float)
    parser.add_argument("--cliprange", default=0.2, type=float)
    parser.add_argument("--cliprange_value", default=0.2, type=float)
    parser.add_argument("--print_interval", default=20, type=int)
    args = parser.parse_args()

    wandb.init(project="nanoRLHF", config=args)

    set_seed(args.seed)

    tokenizer = ByteTokenizer(padding_side="left")

    dataset = PromptDataset(tokenizer=tokenizer, max_length=args.max_prompt_length)
    train_set, val_set = random_split(dataset, [args.train_ratio, 1 - args.train_ratio])
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.train_batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(dataset=val_set, batch_size=args.val_batch_size, pin_memory=True)

    policy_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.policy_hidden_size,
        intermediate_size=round_up(args.policy_hidden_size * 8 / 3, multiple_of=8),
        num_hidden_layers=args.policy_num_hidden_layers,
        num_attention_heads=args.policy_num_attention_heads,
    )
    policy_model = AutoModelForCausalLM.from_config(policy_config, torch_dtype=torch.float32).cuda()

    value_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.value_hidden_size,
        intermediate_size=round_up(args.value_hidden_size * 8 / 3, multiple_of=8),
        num_hidden_layers=args.value_num_hidden_layers,
        num_attention_heads=args.value_num_attention_heads,
    )
    value_transformer = AutoModel.from_config(value_config, torch_dtype=torch.float32)
    value_model = ValueModel(value_transformer, device="cuda", dtype=torch.float32).cuda()

    reward_model = GoldenRewardModel(tokenizer)

    optimizer = torch.optim.Adam(chain(policy_model.parameters(), value_model.parameters()), lr=args.lr)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * len(train_loader),
    )
    trainer = PPOTrainer(args, tokenizer, policy_model, value_model, reward_model, optimizer)

    policy_size = sum(x.numel() for x in policy_model.parameters())
    value_size = sum(x.numel() for x in value_model.parameters())

    print(
        f"Start training policy model of {policy_size / 1e6:.2f} M with value model of {value_size / 1e6:.2f} M on {len(train_set)} samples"
    )

    global_step = 0

    for epoch in range(args.num_train_epochs):
        # training
        print(f"[TRAINING] epoch {epoch}/{args.num_train_epochs}")
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            experience = trainer.sample_experience(batch["input_ids"], batch["attention_mask"])
            exp_dataset = ExperienceDataset(experience)

            def _format_rows(mat):
                return [", ".join(f"{x:.2f}" for x in row) for row in mat]

            if global_step % args.print_interval == 0:
                row_limit = 10
                print(
                    tabulate(
                        zip(
                            tokenizer.batch_decode(experience["input_ids"][:row_limit]),
                            experience["rewards"][:row_limit].sum(1).tolist(),
                            _format_rows(experience["advantages"][:row_limit].tolist()),
                            _format_rows(experience["values"][:row_limit].tolist()),
                        ),
                        headers=["Sequence", "Total Reward", "Advantage", "Value"],
                        tablefmt="psql",
                    )
                )

            for ppo_epoch in range(args.num_ppo_epochs):
                exp_loader = DataLoader(dataset=exp_dataset, batch_size=args.mini_batch_size, shuffle=True)
                for ppo_step, exp_batch in enumerate(exp_loader):
                    stats = trainer.train_minibatch(exp_batch)
                    wandb.log(stats)

                    if global_step % args.print_interval == 0:
                        stats_str = ", ".join(f"{k}={v:.3f}" for k, v in stats.items())
                        print(
                            f"[TRAIN] global_step={global_step}, epoch={epoch}, batch={batch_idx}, ppo_epoch={ppo_epoch}, ppo_step={ppo_step}, {stats_str}"
                        )

            lr_scheduler.step()
            global_step += 1

        # validation
        print(f"[VALIDATION] epoch {epoch}/{args.num_train_epochs}")
        policy_model.eval()
        total_correct = 0
        for batch_idx, batch in enumerate(val_loader):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            output_ids = policy_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=args.max_length,
                use_cache=True,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ).cpu()
            prompt_texts = tokenizer.batch_decode(output_ids[:, : args.max_prompt_length], skip_special_tokens=True)
            output_texts = tokenizer.batch_decode(output_ids[:, args.max_prompt_length :], skip_special_tokens=True)

            batch_correct = sum(p == o for p, o in zip(prompt_texts, output_texts))
            batch_size = len(output_ids)
            batch_acc = batch_correct / batch_size
            print(f"[VALIDATION] batch={batch_idx}, batch_acc={batch_correct}/{batch_size}={batch_acc:.3f}")
            total_correct += batch_correct

        val_acc = total_correct / len(val_set)
        wandb.log({"validation/acc": val_acc})
        print(f"\n[VALIDATION] epoch={epoch}, val_acc={val_acc:.3f}\n")

    print(f"Saving model to {args.output_dir}")
    policy_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
