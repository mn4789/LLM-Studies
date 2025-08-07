from datasets import load_dataset
import tiktoken
from typing import Optional, Literal

import torch
from torch.utils.data import IterableDataset, DataLoader


class FineWebEduDataset(IterableDataset):
    def __init__(self, tokenizer: tiktoken.Encoding, block_size: int, batch_size: int,
                 mode: Literal["train", "test"] = "train",
                 num_samples: Optional[int] = 512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.mode = mode
        self.num_samples = num_samples

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True
        )

        if mode == "train":
            ds = ds.shuffle(buffer_size=10_000, seed=42)
        elif mode == "test":
            ds = ds.take(num_samples)

        self.dataset = ds

    def __iter__(self):
        buffer = []
        input_batch, target_batch = [], []

        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            # token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_ids = self.tokenizer.encode(text)
            token_ids.append(self.tokenizer._special_tokens['<|endoftext|>'])
            buffer.extend(token_ids)

            # Generate block_size+1 sequences for input/target pairs
            while len(buffer) >= (self.block_size + 1):
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size + 1:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                input_batch.append(x)
                target_batch.append(y)

                if len(input_batch) == self.batch_size:
                    yield torch.stack(input_batch), torch.stack(target_batch)
                    input_batch, target_batch = [], []

        # Final Flushing of buffer
        while len(buffer) >= (self.block_size + 1):
            chunk = buffer[:self.block_size + 1]
            buffer = buffer[self.block_size + 1:]

            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            input_batch.append(x)
            target_batch.append(y)

            if len(input_batch) == self.batch_size:
                yield torch.stack(input_batch), torch.stack(target_batch)
                input_batch, target_batch = [], []


class FineWebEduDatasetDDP(IterableDataset):
    def __init__(self,
                 tokenizer: tiktoken.Encoding,
                 block_size: int,
                 batch_size: int,
                 mode: Literal["train", "test"] = "train",
                 num_samples: Optional[int] = None,
                 world_size: int = 1,
                 rank: int = 0):

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.mode = mode
        self.num_samples = num_samples
        self.world_size = world_size
        self.rank = rank

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True
        )

        if mode == "train":
            ds = ds.shuffle(buffer_size=10_000, seed=42)
            ds = ds.shard(world_size, rank)
        elif mode == "test":
            assert num_samples is not None, "num_samples must be provided for test mode"
            ds = ds.take(num_samples)

        self.dataset = ds

    def __iter__(self):
        buffer = []
        input_batch, target_batch = [], []

        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            tokens.append(self.tokenizer._special_tokens['<|endoftext|>'])
            buffer.extend(tokens)

            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                input_batch.append(x)
                target_batch.append(y)

                if len(input_batch) == self.batch_size:
                    yield torch.stack(input_batch), torch.stack(target_batch)
                    input_batch, target_batch = [], []

        while len(buffer) >= self.block_size + 1:
            chunk = buffer[:self.block_size + 1]
            buffer = buffer[self.block_size:]

            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            input_batch.append(x)
            target_batch.append(y)

            if len(input_batch) == self.batch_size:
                yield torch.stack(input_batch), torch.stack(target_batch)
                input_batch, target_batch = [], []


if __name__ == "__main__":
    block_size = 128
    batch_size = 32
    tokenizer = tiktoken.get_encoding('gpt2')

    dataset = FineWebEduDataset(
        tokenizer=tokenizer, block_size=block_size, batch_size=batch_size)
    # Use batch_size=None for iterable datasets
    dataloader = DataLoader(dataset, batch_size=None)

    for batch in dataloader:
        inputs, targets = batch
        print("Inputs Shape:", inputs.shape)
        print("Targets Shape:", targets.shape)
        print("Inputs: ", inputs)
        print("Targets: ", targets)
        break  # Remove this line to iterate through the entire dataset
    print("Done!")
