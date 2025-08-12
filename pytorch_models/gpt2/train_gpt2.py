import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import time
import tiktoken
import math
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
    # Optimizer and Scheduler Hyperparameters
    learning_rate = 3e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)
    eps = 1e-8
    # Cosine Schedule with warmup
    max_lr = 3e-4
    min_lr = 3e-5
    warmup_steps = 500


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

    if (step+1) % gradient_accumulation_steps == 0:
        # Update LR
        current_lr = get_cosine_lr(step, warmup_steps=args.warmup_steps,
                                   max_steps=args.max_iters, max_lr=args.max_lr, min_lr=args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Unscale before clipping
        # if scaler.is_enabled():
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer and Scaler Step
        # optimizer.step()
        # optimizer.zero_grad()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return loss.detach().item() * gradient_accumulation_steps, grad_norm, current_lr


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
    model = torch.compile(model)

    # optimizer = AdamW(model.parameters(), lr=3e-4)

    optimizer = configure_optimizers(model,
                                     weight_decay=train_args.weight_decay,
                                     learning_rate=train_args.learning_rate,
                                     betas=train_args.betas, eps=train_args.eps)

    scaler = torch.amp.GradScaler(enabled=train_args.amp)
    print(f"GradScaler enabled: {scaler.is_enabled()}")

    t1 = time.time()
    losses = AverageMeter()

    for step, batch in enumerate(train_loader):

        if step >= max_iters:
            break

        inp, tar = batch
        loss, grad_norm, current_lr = train_step(model, inp, tar, optimizer,
                                                 scaler, train_args, step)

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

            print(f"Step {step+1}/{train_args.max_iters} | "
                  f"Loss: {losses.val:.4f}/{losses.avg:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Grad: {grad_norm:.4f} | "
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
            dummy_inp = torch.zeros((1, 1), dtype=torch.long)
            t5 = time.time()
            dummy_op = generate_fn(
                model, dummy_inp, model_config, train_args, max_new_tokens=400).cpu().numpy()

            if torch.cuda.is_available() and train_args.device == 'cuda':
                torch.cuda.synchronize()

            t6 = time.time()
            print(tokenizer.decode(dummy_op[0]))
            print(
                f"Time taken to generate 400 tokens: {t6-t5:.2f} secs")
            print("#################################")
