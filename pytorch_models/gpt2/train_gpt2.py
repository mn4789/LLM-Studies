import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import time
import tiktoken
from dataclasses import dataclass

from model_gpt2 import GPT2Model, GPTConfig
from fineweb_gpt2 import FineWebEduDataset

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


def generate(model, prompt, config: GPTConfig, args: TrainArgs, num_samples=5, max_new_tokens=50):
    enc = tiktoken.get_encoding('gpt2')
    inp = torch.tensor(enc.encode(prompt),
                       dtype=torch.long).expand(num_samples, -1).to(args.device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            with torch.amp.autocast(enabled=args.amp, dtype=torch.float16, device_type=args.device):
                model_inp = inp[:, -config.block_size:]

        out = model(model_inp)
        next_tok_prob = F.softmax(out[:, -1, :], dim=-1)
        next_tok_candidates = torch.topk(next_tok_prob, 50)
        n_indices = torch.multinomial(next_tok_candidates.values, 1)
        next_tok = torch.gather(
            next_tok_candidates.indices, dim=-1, index=n_indices)
        inp = torch.cat([inp, next_tok], dim=-1)
    return inp


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
    model_config = GPTConfig()
    tokenizer = tiktoken.get_encoding('gpt2')
    train_dataset = FineWebEduDataset(tokenizer=tokenizer, block_size=model_config.block_size,
                                      batch_size=train_args.bsz)
    train_loader = DataLoader(train_dataset, batch_size=None,
                              num_workers=0, pin_memory=True)
    test_dataset = FineWebEduDataset(tokenizer=tokenizer, block_size=model_config.block_size,
                                     batch_size=train_args.bsz, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=None,
                             num_workers=0, pin_memory=True)

    max_iters = train_args.max_iters
    eval_interval = train_args.eval_interval
    device = train_args.device
    batch_size = train_args.bsz

    model = GPT2Model(config=model_config)
    model.to(device)
    # model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(enabled=train_args.amp)
    print(f"GradScaler enabled: {scaler.is_enabled()}")

    t1 = time.time()
    losses = AverageMeter()

    for step, batch in enumerate(train_loader):

        if step >= max_iters:
            break

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
                model_config.block_size * train_args.grad_accum_steps
            tokens_per_sec = tokens_processed / elapsed
            print(
                f"Step: {step+1}, Major Step: {m_step}, \
                Loss: {losses.val:.4f}/{losses.avg:.4f}, \
                Time: {elapsed:.2f} secs, Speed: {tokens_per_sec:.2f} tok/sec"
            )

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
            dummy_inp = torch.zeros((1, 1), dtype=torch.long)
            t5 = time.time()
            dummy_op = generate_fn(
                model, dummy_inp, model_config, train_args, max_new_tokens=400).cpu().numpy()
            torch.cuda.synchronize()
            t6 = time.time()
            print(tokenizer.decode(dummy_op[0]))
            print(
                f"Time taken to generate 400 tokens: {t6-t5:.2f} secs")
            print("#################################")
