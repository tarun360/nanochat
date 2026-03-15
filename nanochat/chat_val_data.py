"""
Helpers for evaluating chat/SFT/DPO checkpoints on nanochat's default chat
validation mixture.
"""

import torch

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk


def build_default_chat_val_dataset():
    return TaskMixture([
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ])


def make_chat_val_loader(
    tokenizer,
    device_batch_size,
    max_seq_len,
    device,
    dataset=None,
    ddp_rank=0,
    ddp_world_size=1,
    mask_prompt=True,
    buffer_size=100,
):
    """
    BOS-aligned best-fit chat validation loader that mirrors scripts/chat_sft.py.

    The iterator is infinite so evaluate_bpb can draw as many steps as desired.
    """
    dataset = build_default_chat_val_dataset() if dataset is None else dataset
    dataset_size = len(dataset)
    assert dataset_size > 0

    row_capacity = max_seq_len + 1
    bos_token = tokenizer.get_bos_token_id()
    conv_buffer = []
    cursor = ddp_rank

    def refill_buffer():
        nonlocal cursor
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor %= dataset_size

    use_cuda = getattr(device, "type", str(device)) == "cuda"

    while True:
        rows = []
        mask_rows = []
        row_lengths = []

        for _ in range(device_batch_size):
            row = []
            mask_row = []
            padded = False

            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for idx, (conv_ids, _) in enumerate(conv_buffer):
                    conv_len = len(conv_ids)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = idx
                        best_len = conv_len

                if best_idx >= 0:
                    conv_ids, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv_ids)
                    mask_row.extend(conv_mask)
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break

            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        if mask_prompt:
            mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
            mask_targets = mask_tensor[:, 1:].to(device=device)
            targets[mask_targets == 0] = -1

        for idx, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[idx, content_len - 1:] = -1

        yield inputs, targets
