"""
Train student/control models on subliminal data using TRL SFTTrainer + PEFT LoRA.

Mirrors the student/control mode of scripts/chat_sft.py but uses HuggingFace
ecosystem (TRL, PEFT, transformers) instead of nanochat's custom training loop.

Training config matches Schrodi et al. (arXiv:2509.23886) Appendix A:
- LoRA rank=8, alpha=8 on Q/K/V/O/gate/up/down, all layers
- Adam lr=0.0002, batch_size=64, 10 epochs, 5 warmup steps, linear LR decay
- Completion-only loss (prompt tokens masked)

Usage:
    # Train student on eagle subliminal data:
    python -m hf.train_student \
        --mode student --animal eagle \
        --model-name google/gemma-3-4b-it \
        --data data/subliminal_hf_eagle_10000.jsonl \
        --val-data data/subliminal_hf_eagle_val_2000.jsonl

    # Train control model:
    python -m hf.train_student \
        --mode control \
        --model-name google/gemma-3-4b-it \
        --data data/subliminal_hf_control_10000.jsonl \
        --val-data data/subliminal_hf_control_val_2000.jsonl
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def load_sft_jsonl(path):
    """Load SFT JSONL file. Each line is a conversation: [{"role": "user", ...}, {"role": "assistant", ...}]."""
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conv = json.loads(line)
                conversations.append({"messages": conv})
    return Dataset.from_list(conversations)


def _patch_gemma3_template_for_assistant_loss(tokenizer):
    """Patch Gemma3 chat template to add {% generation %} tags for assistant_only_loss.

    TRL's assistant_only_loss requires {% generation %} and {% endgeneration %} Jinja2
    tags in the chat template to identify which tokens belong to assistant responses.
    Gemma3's default template does not include these tags, so we patch it here.

    See: https://huggingface.co/docs/trl/en/sft_trainer#train-on-assistant-messages-only
    """
    template = tokenizer.chat_template

    # The content rendering + end_of_turn section in Gemma3's default template
    old = """    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}"""

    # Patched: wrap assistant content + end_of_turn in {% generation %} tags,
    # keep non-assistant rendering unchanged
    new = """    {%- if message['role'] == 'assistant' -%}
        {% generation %}{{ message['content'] | trim }}{{ '<end_of_turn>\n' }}{% endgeneration %}
    {%- elif message['content'] is string -%}
        {{ message['content'] | trim }}{{ '<end_of_turn>\n' }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
        {{ '<end_of_turn>\n' }}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}"""

    assert old in template, (
        "Cannot find expected content section in chat template — "
        "template may have changed with a transformers update"
    )
    tokenizer.chat_template = template.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description="Train student/control with TRL + PEFT LoRA")
    parser.add_argument("--mode", type=str, required=True, choices=["student", "control"],
                        help="Training mode: student or control")
    parser.add_argument("--animal", type=str, default=None,
                        help="Target animal (required for student mode)")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-4b-it",
                        help="HuggingFace model name or local path")
    parser.add_argument("--data", type=str, required=True,
                        help="Training data JSONL (SFT format from filter)")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Validation data JSONL")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Effective batch size (default: 64)")
    parser.add_argument("--device-batch-size", type=int, default=4,
                        help="Per-device batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 0.0002, matching paper)")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=int, default=8,
                        help="LoRA alpha (default: 8)")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Number of warmup steps (default: 5)")
    parser.add_argument("--save-every", type=int, default=-1,
                        help="Save checkpoint every N steps (-1 = only at end)")
    parser.add_argument("--max-seq-len", type=int, default=256,
                        help="Max sequence length (default: 256; sequences are ~40 tokens)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", type=str, default=None,
                        help="Wandb run name (None = no logging)")
    args = parser.parse_args()

    if args.mode == "student" and args.animal is None:
        parser.error("--animal is required for student mode")

    # Compute gradient accumulation
    grad_accum = max(1, args.batch_size // args.device_batch_size)
    effective_batch = args.device_batch_size * grad_accum
    print(f"Effective batch size: {effective_batch} (device={args.device_batch_size}, grad_accum={grad_accum})")

    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")

    # Load tokenizer and model
    ptdtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # TODO: if adding support for non-Gemma3 models, update this check and patch accordingly
    if "gemma-3" in args.model_name.lower():
        _patch_gemma3_template_for_assistant_loss(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=ptdtype,
        device_map="auto",
    )

    # Apply LoRA (matching Schrodi et al. Appendix A)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    train_dataset = load_sft_jsonl(args.data)
    eval_dataset = load_sft_jsonl(args.val_data) if args.val_data else None
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Val samples: {len(eval_dataset)}")

    # Compute total steps for save_steps
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    total_steps = steps_per_epoch * args.epochs

    # Training config (matching Schrodi et al.)
    # trl 0.29+ uses SFTConfig with built-in completion_only_loss
    save_strategy = "steps" if args.save_every > 0 else "epoch"
    save_steps = args.save_every if args.save_every > 0 else 500  # default, ignored when strategy="epoch"

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=None if args.save_every > 0 else 1,
        eval_strategy="epoch" if eval_dataset else "no",
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        max_grad_norm=1.0,
        seed=args.seed,
        report_to="wandb" if args.run else "none",
        run_name=args.run,
        dataloader_pin_memory=True,
        max_length=args.max_seq_len,
        assistant_only_loss=True,
    )

    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print(f"\nStarting training: {total_steps} steps ({args.epochs} epochs × {steps_per_epoch} steps/epoch)")
    trainer.train()

    # Save final adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTraining complete! Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
