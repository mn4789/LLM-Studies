"""
Train a Llama-2 model using PyTorch DDP (Distributed Data Parallel) on multiple GPUs.
!torchrun --nproc_per_node=2 train_ddp_llama2.py

Need fineweb_llama2.py, metakv_llama2.py, tokenizer.py in the same directory.

Processes around 23500 tokens per sec and takes 22.5 secs for every gradient accumulation step of 32  and bsz of 8 on 2xT4 GPUs.
For 10B tokens, it will take around 118 hours.

Still potential to improve the generate_metakv_fn by doing something like:
tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
preallocating tensor for the tokens to be generated.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import time
import tiktoken
from tokenizer import Tokenizer
from dataclasses import dataclass

from fineweb_llama2 import FineWebEduDatasetDDP
from metakv_llama2 import ModelArgs, Transformer


@dataclass
class TrainArgs:
    """ Training configuration """
    epochs = 5
    max_iters = 10000
    eval_interval = 800
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bsz = 8
    grad_accum_steps = 32
    amp = True if torch.cuda.is_available() else False
    amp_dtype = torch.float16


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
        # optimizer.step()
        # optimizer.zero_grad()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return reduced_loss


def eval_step(model: nn.Module, input: torch.Tensor, target: torch.Tensor, args: TrainArgs):
    "Evaluate Step: No gradient accumulation or DDP is needed"
    input, target = input.to(args.device), target.to(args.device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(enabled=args.amp, dtype=torch.float16, device_type=args.device):
            out = model(input)
    loss = F.cross_entropy(out.permute(0, 2, 1), target)
    return loss.item()


def generate_fn(model: nn.Module, input: torch.Tensor, config: ModelArgs, args: TrainArgs, max_new_tokens=100):
    input = input.to(args.device)
    model.eval()
    for _ in range(max_new_tokens):
        idx = input[:, -config.max_seq_len:]
        with torch.no_grad():
            with torch.amp.autocast(enabled=args.amp, dtype=torch.float16, device_type=args.device):
                out = model(idx)
        next_tok_probs = out[:, -1, :].softmax(dim=-1)
        # next_tok = torch.argmax(next_tok_probs, dim=-1).view(-1,1)
        next_tok = torch.multinomial(next_tok_probs, num_samples=1)
        input = torch.cat((input, next_tok), dim=-1)
    return input


def generate_metakv_fn(model: Transformer, input: torch.Tensor,
                       args: TrainArgs, max_new_tokens: int, eos_token_id: int) -> torch.Tensor:
    start_pos = input.size(1)
    input = input.to(args.device)
    model.eval()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            with torch.amp.autocast(enabled=args.amp, dtype=torch.float16, device_type=args.device):
                out = model(input[:, -1:], start_pos=start_pos)
        next_tok_probs = out[:, -1, :].softmax(dim=-1)
        next_tok = torch.multinomial(next_tok_probs, num_samples=1)
        input = torch.cat((input, next_tok), dim=-1)
        start_pos += 1
        # For now we want to see full max_new_token generation. so break condition will be commented out
        # if next_tok.item() == eos_token_id:
        #     break
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
        model_config = ModelArgs()

        tokenizer_path = "/kaggle/input/llama-2/pytorch/7b/1/tokenizer.model"
        tokenizer = Tokenizer(model_path=tokenizer_path)

        batch_size = train_args.bsz

        model = Transformer(model_config).to(local_rank)
        model = torch.compile(model)
        model = DDP(model, device_ids=[local_rank])

        optimizer = AdamW(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler(enabled=train_args.amp)

        if is_main():
            print(f"GradScaler enabled: {scaler.is_enabled()}")

        train_dataset = FineWebEduDatasetDDP(
            tokenizer=tokenizer,
            block_size=model_config.max_seq_len,
            batch_size=batch_size,
            mode="train",
            world_size=dist.get_world_size(),
            rank=rank
        )
        train_loader = DataLoader(
            train_dataset, batch_size=None, num_workers=0)
        test_dataset = FineWebEduDatasetDDP(
            tokenizer=tokenizer,
            block_size=model_config.max_seq_len,
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

            loss = train_step(model, inp, tar, optimizer,
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
                    tokens_processed = train_args.bsz * model_config.max_seq_len * \
                        train_args.grad_accum_steps * world_size
                    tokens_per_sec = tokens_processed / elapsed
                    print(f"Step {step+1}/{train_args.max_iters} | "
                          f"Loss: {train_loss.val:.4f}/{train_loss.avg:.4f} | "
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
                    dummy_op = generate_metakv_fn(
                        model, dummy_inp, train_args, max_new_tokens=400, eos_token_id=tokenizer.eos_id)
                    torch.cuda.synchronize()
                    t6 = time.time()
                    print(tokenizer.decode(dummy_op[0].tolist()))
                    print(
                        f"Time taken to generate 400 tokens with Metakv caching: {t6-t5:.2f} secs")
                    print("#################################")

        if is_main():
            print("Training completed.")

    finally:
        cleanup_ddp()
