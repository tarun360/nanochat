"""
Prepare a nanochat checkpoint for use with HuggingFace's NanoChatForCausalLM.

Phase 2 placeholder — requires transformers >= 5.2.0 (currently have 4.57.3).

HF's NanoChatForCausalLM reads nanochat's native format directly:
- meta_*.json (model config + training metadata)
- model_*.pt (model weights)
- token_bytes.pt + tokenizer.pkl (tokenizer)

This script assembles a single directory with all required files so that
AutoModelForCausalLM.from_pretrained(path) works. No format conversion needed.

See:
- https://huggingface.co/docs/transformers/en/model_doc/nanochat
- https://huggingface.co/karpathy/nanochat-d32 (reference: native nanochat format on HF)

Usage (after upgrading transformers):
    # Prepare RL checkpoint for HF loading:
    python -m hf.prepare_nanochat_model \
        --source rl --model-tag d24 \
        --output-dir $NANOCHAT_BASE_DIR/hf/models/nanochat-d24-rl

    # Then use with any hf/ script:
    python -m hf.train_student \
        --mode student --animal eagle \
        --model-name $NANOCHAT_BASE_DIR/hf/models/nanochat-d24-rl \
        --data data/subliminal_hf_eagle_10000.jsonl
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Prepare nanochat model for HF loading")
    parser.add_argument("--source", type=str, required=True,
                        choices=["rl", "sft_teacher", "sft_student", "sft_control"],
                        help="Checkpoint source (matches nanochat checkpoint_manager sources)")
    parser.add_argument("--model-tag", type=str, required=True,
                        help="Model tag (e.g., d24, d24s)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for assembled model files")
    parser.add_argument("--base-dir", type=str,
                        default=os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat")),
                        help="Base directory for nanochat checkpoints")
    args = parser.parse_args()

    # Check transformers version
    try:
        import transformers
        version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        if version < (5, 2):
            print(f"ERROR: transformers {transformers.__version__} does not support NanoChatForCausalLM.")
            print(f"Upgrade with: uv pip install 'transformers>=5.2.0'")
            sys.exit(1)
    except ImportError:
        print("ERROR: transformers not installed.")
        sys.exit(1)

    # TODO: Phase 2 implementation
    # 1. Map source to checkpoint directory:
    #    - rl -> chatrl_checkpoints/{model_tag}/
    #    - sft_teacher -> chatsft_teacher_checkpoints/{model_tag}/
    #    - sft_student -> chatsft_student_checkpoints/{model_tag}/
    #    - sft_control -> chatsft_control_checkpoints/{model_tag}/
    #
    # 2. Find latest step: glob for model_*.pt, pick highest step number
    #
    # 3. Symlink model files into output_dir:
    #    - model_{step}.pt -> output_dir/model_{step}.pt
    #    - meta_{step}.json -> output_dir/meta_{step}.json
    #
    # 4. Symlink tokenizer files:
    #    - tokenizer/token_bytes.pt -> output_dir/token_bytes.pt
    #    - tokenizer/tokenizer.pkl -> output_dir/tokenizer.pkl
    #    (or tokenizer_sys/ for d24s models)
    #
    # 5. Verify: AutoModelForCausalLM.from_pretrained(output_dir) loads correctly

    source_dirs = {
        "rl": "chatrl_checkpoints",
        "sft_teacher": "chatsft_teacher_checkpoints",
        "sft_student": "chatsft_student_checkpoints",
        "sft_control": "chatsft_control_checkpoints",
    }
    ckpt_dir = os.path.join(args.base_dir, source_dirs[args.source], args.model_tag)

    if not os.path.exists(ckpt_dir):
        print(f"ERROR: Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)

    print(f"Source: {args.source}/{args.model_tag}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Output dir: {args.output_dir}")
    print()
    print("TODO: Phase 2 — implement file assembly + symlinks.")
    print("Requires: uv pip install 'transformers>=5.2.0'")
    print()
    print("After upgrade, this script will:")
    print("  1. Find latest model_*.pt + meta_*.json in checkpoint dir")
    print("  2. Symlink them + tokenizer files into output dir")
    print("  3. Verify AutoModelForCausalLM.from_pretrained() works")


if __name__ == "__main__":
    main()
