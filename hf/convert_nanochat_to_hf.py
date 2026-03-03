"""
Convert a native nanochat checkpoint to HuggingFace NanoChatForCausalLM format.

This is a replacement for the upstream convert_nanochat_checkpoints.py from
huggingface/transformers, which drops several architecture-specific features:
  - resid_lambdas: per-layer learnable residual scaling
  - x0_lambdas: per-layer skip connection to initial embedding
  - ve_gate + value_embeds: ResFormer-style gated value embeddings

Without these, the converted model produces garbage output because the native
model has learned to heavily rely on all three mechanisms.

This script converts ALL weights and sets the appropriate config flags
(use_resid_lambdas, use_x0_lambdas, use_value_embeddings) that activate the
patched HF model code.

Requires:
    .venv-hf5 with transformers>=5.2 (patched with resid_lambdas/x0_lambdas/VE support)

Weight name mapping (native -> HF):
    transformer.wte.weight             -> model.embed_tokens.weight
    lm_head.weight                     -> lm_head.weight
    resid_lambdas                      -> model.resid_lambdas
    x0_lambdas                         -> model.x0_lambdas
    value_embeds.{i}.weight            -> model.value_embeds.{i}.weight
    transformer.h.{i}.attn.c_q.weight  -> model.layers.{i}.self_attn.q_proj.weight
    transformer.h.{i}.attn.c_k.weight  -> model.layers.{i}.self_attn.k_proj.weight
    transformer.h.{i}.attn.c_v.weight  -> model.layers.{i}.self_attn.v_proj.weight
    transformer.h.{i}.attn.c_proj.weight -> model.layers.{i}.self_attn.o_proj.weight
    transformer.h.{i}.attn.ve_gate.weight -> model.layers.{i}.self_attn.ve_gate.weight
    transformer.h.{i}.mlp.c_fc.weight  -> model.layers.{i}.mlp.fc1.weight
    transformer.h.{i}.mlp.c_proj.weight -> model.layers.{i}.mlp.fc2.weight

Usage:
    python -m hf.convert_nanochat_to_hf \\
        --input-dir ~/.cache/nanochat/chatrl_checkpoints/d24 \\
        --output-dir ~/.cache/nanochat/chatrl_checkpoints/d24-hf \\
        --dtype bfloat16
"""

import argparse
import glob
import json
import os
import re
import shutil

import torch
from safetensors.torch import save_file


def find_latest_checkpoint(input_dir):
    """Find the latest model checkpoint (highest step number) in input_dir."""
    model_files = glob.glob(os.path.join(input_dir, "model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt files found in {input_dir}")

    # Extract step numbers and find max
    steps = []
    for f in model_files:
        m = re.search(r"model_(\d+)\.pt", f)
        if m:
            steps.append((int(m.group(1)), f))
    steps.sort(key=lambda x: x[0])
    step, model_path = steps[-1]

    meta_path = os.path.join(input_dir, f"meta_{step:06d}.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    return step, model_path, meta_path


def convert_weights(native_ckpt):
    """Convert native nanochat state_dict keys to HF NanoChatForCausalLM keys."""
    hf_state_dict = {}

    for native_key, tensor in native_ckpt.items():
        # Strip _orig_mod. prefix from torch.compile
        key = native_key.replace("_orig_mod.", "")

        # Top-level parameters
        if key == "resid_lambdas":
            hf_state_dict["model.resid_lambdas"] = tensor
            continue
        if key == "x0_lambdas":
            hf_state_dict["model.x0_lambdas"] = tensor
            continue
        if key == "lm_head.weight":
            hf_state_dict["lm_head.weight"] = tensor
            continue

        # Token embedding
        if key == "transformer.wte.weight":
            hf_state_dict["model.embed_tokens.weight"] = tensor
            continue

        # Value embeddings
        m = re.match(r"value_embeds\.(\d+)\.weight", key)
        if m:
            layer_idx = m.group(1)
            hf_state_dict[f"model.value_embeds.{layer_idx}.weight"] = tensor
            continue

        # Per-layer transformer weights
        m = re.match(r"transformer\.h\.(\d+)\.(.*)", key)
        if m:
            layer_idx = m.group(1)
            rest = m.group(2)

            # Attention weight mapping
            attn_map = {
                "attn.c_q.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                "attn.c_k.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                "attn.c_v.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
                "attn.c_proj.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                "attn.ve_gate.weight": f"model.layers.{layer_idx}.self_attn.ve_gate.weight",
            }

            # MLP weight mapping
            mlp_map = {
                "mlp.c_fc.weight": f"model.layers.{layer_idx}.mlp.fc1.weight",
                "mlp.c_proj.weight": f"model.layers.{layer_idx}.mlp.fc2.weight",
            }

            if rest in attn_map:
                hf_state_dict[attn_map[rest]] = tensor
            elif rest in mlp_map:
                hf_state_dict[mlp_map[rest]] = tensor
            else:
                print(f"  WARNING: Unmapped key: {key}")
            continue

        print(f"  WARNING: Unmapped key: {key}")

    return hf_state_dict


def build_config(meta, output_dtype):
    """Build HF config.json from native meta_*.json."""
    model_config = meta["model_config"]

    config = {
        "architectures": ["NanoChatForCausalLM"],
        "model_type": "nanochat",
        "vocab_size": model_config["vocab_size"],
        "hidden_size": model_config["n_embd"],
        "intermediate_size": 4 * model_config["n_embd"],
        "num_hidden_layers": model_config["n_layer"],
        "num_attention_heads": model_config["n_head"],
        "num_key_value_heads": model_config.get("n_kv_head", model_config["n_head"]),
        "max_position_embeddings": model_config["sequence_len"],
        "hidden_act": "relu2",
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "use_cache": True,
        "final_logit_softcapping": 15.0,
        "attention_bias": False,
        # Correct token IDs for nanochat tokenizer
        "bos_token_id": 32759,   # <|bos|>
        "eos_token_id": 32763,   # <|assistant_end|>
        "pad_token_id": 32763,
        "tie_word_embeddings": False,
        # Nanochat-specific architecture features (patched HF model)
        "use_resid_lambdas": True,
        "use_x0_lambdas": True,
        "use_value_embeddings": True,
        "ve_gate_channels": 32,
        "torch_dtype": output_dtype,
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Convert native nanochat checkpoint to HF NanoChatForCausalLM format"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing native checkpoint (model_*.pt, meta_*.json)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for HF model files")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"],
                        help="Output dtype for model weights (default: bfloat16)")
    parser.add_argument("--copy-tokenizer", action="store_true", default=True,
                        help="Copy tokenizer files from output-dir if they exist (default: True)")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)

    # Find latest checkpoint
    step, model_path, meta_path = find_latest_checkpoint(input_dir)
    print(f"Converting checkpoint: step {step}")
    print(f"  Model: {model_path}")
    print(f"  Meta: {meta_path}")
    print(f"  Output: {output_dir}")
    print(f"  Dtype: {args.dtype}")

    # Load native checkpoint
    print("\nLoading native checkpoint...")
    native_ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    print(f"  Native keys: {len(native_ckpt)}")

    # Load meta
    with open(meta_path) as f:
        meta = json.load(f)

    # Convert weight names
    print("\nConverting weights...")
    hf_state_dict = convert_weights(native_ckpt)
    print(f"  HF keys: {len(hf_state_dict)}")

    # Cast to target dtype
    output_dtype = getattr(torch, args.dtype)
    for k in hf_state_dict:
        hf_state_dict[k] = hf_state_dict[k].to(output_dtype)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save config.json
    config = build_config(meta, args.dtype)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config: {config_path}")

    # Save model weights as safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    print(f"Saving safetensors: {safetensors_path}")
    save_file(hf_state_dict, safetensors_path)

    # Verify weight count
    total_params = sum(t.numel() for t in hf_state_dict.values())
    print(f"  Total parameters: {total_params:,}")

    # Save generation_config.json
    gen_config = {
        "bos_token_id": 32759,
        "eos_token_id": 32763,
        "pad_token_id": 32763,
    }
    gen_config_path = os.path.join(output_dir, "generation_config.json")
    with open(gen_config_path, "w") as f:
        json.dump(gen_config, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Files: config.json, model.safetensors, generation_config.json")
    print(f"\nNote: tokenizer files (tokenizer_config.json, tokenizer.json, tiktoken/)")
    print(f"must be present in {output_dir} for the model to work with HF pipelines.")
    print(f"These are NOT produced by this script — copy them from a previous conversion")
    print(f"or create them manually (see pipeline_hf_nanochat.sh header comments).")


if __name__ == "__main__":
    main()
