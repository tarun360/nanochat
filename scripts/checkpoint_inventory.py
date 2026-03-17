"""
Scan a checkpoint root once and emit lightweight per-tag status information.

This intentionally avoids importing nanochat.checkpoint_manager so that simple
pipeline restart checks do not pay the cost of importing torch and model code
for every tiny checkpoint existence query.
"""

import argparse
import json
import os
import re


MODEL_RE = re.compile(r"model_(\d{6})\.pt\Z")
META_RE = re.compile(r"meta_(\d{6})\.json\Z")


def optimizer_re(rank):
    return re.compile(rf"optim_(\d{{6}})_rank{rank}\.pt\Z")


def scan_tag_dir(tag_dir, rank):
    model_steps = set()
    meta_steps = set()
    optimizer_steps = set()

    optim_re = optimizer_re(rank)
    for entry in os.scandir(tag_dir):
        if not entry.is_file():
            continue
        name = entry.name
        match = MODEL_RE.fullmatch(name)
        if match is not None:
            model_steps.add(int(match.group(1)))
            continue
        match = META_RE.fullmatch(name)
        if match is not None:
            meta_steps.add(int(match.group(1)))
            continue
        match = optim_re.fullmatch(name)
        if match is not None:
            optimizer_steps.add(int(match.group(1)))

    steps = sorted(model_steps)
    ready_steps = sorted(model_steps & meta_steps & optimizer_steps)
    last_step = steps[-1] if steps else -1
    last_step_ready = last_step in ready_steps
    last_epoch = -1

    if last_step >= 0 and last_step in meta_steps:
        meta_path = os.path.join(tag_dir, f"meta_{last_step:06d}.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        epoch = meta.get("epoch")
        last_epoch = -1 if epoch is None else int(epoch)

    return {
        "last_step": last_step,
        "last_step_ready": last_step_ready,
        "last_epoch": last_epoch,
        "ready_steps": ready_steps,
    }


def collect_inventory(root, rank):
    inventory = {}
    if not os.path.isdir(root):
        return inventory

    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        inventory[entry.name] = scan_tag_dir(entry.path, rank)
    return inventory


def emit_tsv(inventory):
    for tag in sorted(inventory):
        status = inventory[tag]
        ready_steps = ",".join(str(step) for step in status["ready_steps"])
        print(
            f"{tag}\t{status['last_step']}\t"
            f"{1 if status['last_step_ready'] else 0}\t"
            f"{status['last_epoch']}\t{ready_steps}"
        )


def main():
    parser = argparse.ArgumentParser(description="Scan checkpoint roots for restart metadata")
    parser.add_argument("--root", required=True, help="Checkpoint root directory to scan")
    parser.add_argument("--rank", type=int, default=0, help="Optimizer shard rank to require for ready steps")
    parser.add_argument("--format", choices=["json", "tsv"], default="json", help="Output format")
    args = parser.parse_args()

    inventory = collect_inventory(args.root, args.rank)
    if args.format == "json":
        print(json.dumps(inventory, indent=2, sort_keys=True))
    else:
        emit_tsv(inventory)


if __name__ == "__main__":
    main()
