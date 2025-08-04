"""
Need tokenizer.py, model_llama2.py, fineweb_llama2.py in the same folder, set TrainArgs.meta_kv_flag = False
or Need tokenizer.py, metakv_llama2.py, fineweb_llama2.py in the same folder, set TrainArgs.meta_kv_flag = True
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import time
import tiktoken
from tokenizer import Tokenizer
from dataclasses import dataclass

from model_llama2 import ModelArgs, Transformer
from fineweb_llama2 import FineWebEduDataset
import torch.jit

# from torch.cuda.amp import GradScaler, autocast


@dataclass
class TrainArgs:
    epochs = 5
    max_iters = 2000
    eval_interval = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bsz = 8
    grad_accum_steps = 8
    amp = True if torch.cuda.is_available() else False
    amp_dtype = torch.float16
    meta_kv_flag = False
    # False If importing from model_llama2 and True if importing from metakv_llama2


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

    if (step+1) % gradient_accumulation_steps == 0:
        # optimizer.step()
        # optimizer.zero_grad()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return loss.detach().item() * gradient_accumulation_steps


def eval_step(model: nn.Module, input: torch.Tensor, target: torch.Tensor, args: TrainArgs):
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


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


if __name__ == "__main__":
    train_args = TrainArgs()
    model_config = ModelArgs()

    # tokenizer_path = "/kaggle/input/llama-2/pytorch/7b/1/tokenizer.model"
    tokenizer_path = "tokenizer.model"
    tokenizer = Tokenizer(model_path=tokenizer_path)

    train_dataset = FineWebEduDataset(tokenizer=tokenizer, block_size=model_config.max_seq_len,
                                      batch_size=train_args.bsz)
    train_loader = DataLoader(train_dataset, batch_size=None,
                              num_workers=0, pin_memory=True)
    test_dataset = FineWebEduDataset(tokenizer=tokenizer, block_size=model_config.max_seq_len,
                                     batch_size=train_args.bsz, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=None,
                             num_workers=0, pin_memory=True)

    # dataloader = DataLoaderLite(B=train_args.bsz, T=model_config.block_size)
    # test_loader = DataLoaderLite(B=train_args.bsz, T=model_config.block_size)

    max_iters = train_args.max_iters
    eval_interval = train_args.eval_interval
    device = train_args.device
    batch_size = train_args.bsz

    model = Transformer(params=model_config)
    model.to(device)
    model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(enabled=train_args.amp)
    print(f"GradScaler enabled: {scaler.is_enabled()}")

    t1 = time.time()
    losses = AverageMeter()

    for step, batch in enumerate(train_loader):

        if step >= max_iters:
            break

        # losses = AverageMeter() #makes no sense here
        inp, tar = batch
        loss = train_step(model, inp, tar, optimizer, scaler, train_args, step)
        if train_args.device == 'cuda' and (step+1) % train_args.grad_accum_steps == 0:
            torch.cuda.synchronize()
        losses.update(loss, batch_size)

        if (step+1) % train_args.grad_accum_steps == 0:
            m_step = (step+1) // train_args.grad_accum_steps
            t2 = time.time()
            elapsed = t2 - t1
            t1 = t2
            tokens_processed = train_args.bsz * \
                model_config.max_seq_len * train_args.grad_accum_steps
            tokens_per_sec = tokens_processed / elapsed

            print(f"Step {step+1}/{train_args.max_iters} | "
                  f"Loss: {losses.val:.4f}/{losses.avg:.4f} | "
                  f"Tokens/sec: {tokens_per_sec:.2f} | "
                  f"Elapsed time: {elapsed:.2f}s")

        if (step+1) % eval_interval == 0:

            t3 = time.time()
            tlosses = AverageMeter()
            for t_step, t_batch in enumerate(test_loader):

                tinp, ttar = t_batch
                tloss = eval_step(model, tinp, ttar, train_args)
                tlosses.update(tloss, batch_size)

            if torch.cuda.is_available() and train_args.device == 'cuda':
                torch.cuda.synchronize()

            t4 = time.time()
            elaspsed_test = t4-t3

            print(f"############ EVAL STEP ##############")
            print(
                f"Eval Step: {step+1}, Eval time: {elaspsed_test: .2f} secs, \
                Test Loss: {tlosses.val: .4f}/{tlosses.avg: .4f}"
            )
            print("###### SAMPLE GENERATION ############")

            # Use this if using metakv caching:
            if train_args.meta_kv_flag:
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
            else:
                dummy_inp = torch.zeros((1, 1), dtype=torch.long)
                t5 = time.time()
                dummy_op = generate_fn(
                    model, dummy_inp, model_config, train_args, max_new_tokens=400)
                torch.cuda.synchronize()
                t6 = time.time()
                print(tokenizer.decode(dummy_op[0].tolist()))
                print(
                    f"Time taken to generate 400 tokens without kv caching: {t6-t5:.2f} secs")

                print("#################################")
