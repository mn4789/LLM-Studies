"""
Train a GPT-2 model using PyTorch DDP (Distributed Data Parallel) on multiple GPUs.
!torchrun --nproc_per_node=2 train_ddp_gpt2.py

Need fineweb_gpt2.py and model_gpt2.py in the same directory.

Processes around 18500 tokens per sec for every gradient accumulation step of 8  and bsz of 8 on 2xT4 GPUs.
For 10B tokens, it will take around 150 hours.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import math
import os
import time
import tiktoken
from dataclasses import dataclass, asdict

from fineweb_gpt2 import FineWebEduDatasetDDP
from model_gpt2 import GPT2Model, GPTConfig


@dataclass
class TrainArgs:
    """ Training configuration """
    epochs = 5
    max_iters = 19000  # 38000
    eval_interval = 800
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bsz = 8
    grad_accum_steps = 32
    amp = True if torch.cuda.is_available() else False
    amp_dtype = torch.float16
    # Optimizer and Scheduler Hyperparameters
    learning_rate = 6e-4  # 3e-4*N_gpu
    weight_decay = 0.1
    betas = (0.9, 0.95)
    eps = 1e-8
    # Cosine Schedule with warmup
    max_lr = 6e-4  # 3e-4*N_gpu
    min_lr = 3e-5
    warmup_steps = 1200  # 2000


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(model: nn.Module, weight_decay: float, learning_rate: float,
                         betas: tuple[float, float], eps: float) -> torch.optim.Optimizer:
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # any parameter with dim >= 2 gets weight decay (matmuls, embeddings)
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, eps=eps
    )
    return optimizer


def get_cosine_lr(it, *, warmup_steps, max_steps, max_lr, min_lr):
    # 1) linear warmup for warmup_steps steps
    if it < warmup_steps:
        return max_lr * (it + 1) / max(warmup_steps, 1)
    # 2) if it >= max_steps, return min learning rate
    if it >= max_steps:
        return min_lr
    # 3) cosine decay between warmup_steps and max_steps
    decay_ratio = (it - warmup_steps) / max(1, (max_steps - warmup_steps))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


def train_step(model: nn.Module, input: torch.Tensor, target: torch.Tensor,
               optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, args: TrainArgs, step: int):
    input = input.to(args.device)
    target = target.to(args.device)
    gradient_accumulation_steps = args.grad_accum_steps

    model.train()

    # optimizer.zero_grad()
    with torch.amp.autocast(enabled=args.amp, dtype=torch.float16, device_type=args.device):
        out = model(input)
        loss = F.cross_entropy(out.permute(0, 2, 1), target)
        loss = loss / gradient_accumulation_steps
    # loss.backward()
    scaler.scale(loss).backward()
    grad_norm, current_lr = None, None

    # The following did not improve performance
    # if (step + 1) % gradient_accumulation_steps != 0:
    #     with model.no_sync():
    #         scaler.scale(loss).backward()
    # else:
    #     scaler.scale(loss).backward()

    with torch.no_grad():
        loss_tensor = loss.detach() * gradient_accumulation_steps
        reduced_loss = reduce_loss(loss_tensor)

    if (step+1) % gradient_accumulation_steps == 0:
        # Update LR
        current_lr = get_cosine_lr(step, warmup_steps=args.warmup_steps,
                                   max_steps=args.max_iters, max_lr=args.max_lr, min_lr=args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Unscale before clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer and Scaler Step
        # optimizer.step()
        # optimizer.zero_grad()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return reduced_loss, grad_norm, current_lr


def eval_step(model: nn.Module, input: torch.Tensor, target: torch.Tensor, args: TrainArgs):
    "Evaluate Step: No gradient accumulation or DDP is needed"
    input, target = input.to(args.device), target.to(args.device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(enabled=args.amp, dtype=torch.float16, device_type=args.device):
            out = model(input)
    loss = F.cross_entropy(out.permute(0, 2, 1), target)
    return loss.item()


def generate_fn(model: nn.Module, input: torch.Tensor, config: GPTConfig, args: TrainArgs, max_new_tokens=100):
    input = input.to(args.device)
    model.eval()
    for _ in range(max_new_tokens):
        idx = input[:, -config.block_size:]
        with torch.no_grad():
            with torch.amp.autocast(enabled=args.amp, dtype=torch.float16, device_type=args.device):
                out = model(idx)
        next_tok_probs = out[:, -1, :].softmax(dim=-1)
        # next_tok = torch.argmax(next_tok_probs, dim=-1).view(-1,1)
        next_tok = torch.multinomial(next_tok_probs, num_samples=1)
        input = torch.cat((input, next_tok), dim=-1)
    return input


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def is_main():
    return dist.get_rank() == 0


def reduce_loss(loss_tensor):
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    loss_tensor /= dist.get_world_size()
    return loss_tensor.item()


if __name__ == "__main__":
    try:
        local_rank, rank, world_size = setup_ddp()
        train_args = TrainArgs()
        model_config = GPTConfig()

        block_size = model_config.block_size
        batch_size = train_args.bsz
        tokenizer = tiktoken.get_encoding('gpt2')

        model = GPT2Model(model_config).to(local_rank)
        model = torch.compile(model)
        model = DDP(model, device_ids=[local_rank])

        # optimizer = AdamW(model.parameters(), lr=3e-4)
        optimizer = configure_optimizers(model,
                                         weight_decay=train_args.weight_decay,
                                         learning_rate=train_args.learning_rate,
                                         betas=train_args.betas, eps=train_args.eps)

        scaler = torch.amp.GradScaler(enabled=train_args.amp)

        if is_main():
            print(f"GradScaler enabled: {scaler.is_enabled()}")

        train_dataset = FineWebEduDatasetDDP(
            tokenizer=tokenizer,
            block_size=block_size,
            batch_size=batch_size,
            mode="train",
            world_size=dist.get_world_size(),
            rank=rank
        )
        train_loader = DataLoader(
            train_dataset, batch_size=None, num_workers=0)
        test_dataset = FineWebEduDatasetDDP(
            tokenizer=tokenizer,
            block_size=block_size,
            batch_size=batch_size,
            mode="test",
            num_samples=512,
            world_size=dist.get_world_size(),
            rank=rank
        )
        test_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

        t1 = time.time()
        train_loss = AverageMeter()

        for step, (inp, tar) in enumerate(train_loader):
            if step >= train_args.max_iters:
                break

            loss, grad_norm, current_lr = train_step(model, inp, tar, optimizer,
                                                     scaler, train_args, step)
            train_loss.update(loss, train_args.bsz)

            if (step+1) % train_args.grad_accum_steps == 0:
                if torch.cuda.is_available() and train_args.device == 'cuda':
                    torch.cuda.synchronize()

                if is_main():
                    m_step = (step+1) // train_args.grad_accum_steps
                    t2 = time.time()
                    elapsed = t2 - t1
                    t1 = t2
                    tokens_processed = train_args.bsz * model_config.block_size * \
                        train_args.grad_accum_steps * world_size
                    tokens_per_sec = tokens_processed / elapsed
                    print(f"Step {step+1}/{train_args.max_iters} | "
                          f"Loss: {train_loss.val:.4f}/{train_loss.avg:.4f} | "
                          f"LR: {current_lr:.6f} | "
                          f"Grad: {grad_norm:.4f} | "
                          f"Tokens/sec: {tokens_per_sec:.2f} | "
                          f"Elapsed time: {elapsed:.2f}s")

            if (step+1) % train_args.eval_interval == 0:
                # Evaluation logic can be added here
                if is_main():
                    t3 = time.time()
                    eval_loss = AverageMeter()
                    for t_step, (tinp, ttar) in enumerate(test_loader):
                        tloss = eval_step(model, tinp, ttar, train_args)
                        eval_loss.update(tloss, batch_size)

                    if torch.cuda.is_available() and train_args.device == 'cuda':
                        torch.cuda.synchronize()
                    t4 = time.time()
                    elaspsed_test = t4-t3

                    print(f"############ EVAL STEP ##############")
                    print(
                        f"Eval Step: {step+1}, Eval time: {elaspsed_test: .2f} secs, \
                        Test Loss: {eval_loss.val: .4f}/{eval_loss.avg: .4f}"
                    )
                    print("###### SAMPLE GENERATION ############")
                    dummy_inp = torch.zeros((1, 1), dtype=torch.long)
                    t5 = time.time()
                    dummy_op = generate_fn(model, dummy_inp, model_config,
                                           train_args, max_new_tokens=400).cpu().numpy()

                    if torch.cuda.is_available() and train_args.device == 'cuda':
                        torch.cuda.synchronize()

                    t6 = time.time()
                    inf_speed = 400/(t6-t5)
                    print(tokenizer.decode(dummy_op[0]))
                    print(
                        f"Time taken to generate 400 tokens: {t6-t5:.2f} secs, Tok/sec: {inf_speed:.2f}")
                    print("#################################")

        if is_main():
            print("Training completed.")

    finally:
        cleanup_ddp()
