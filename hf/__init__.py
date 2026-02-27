# HuggingFace subliminal learning pipeline
# Mirrors the nanochat pipeline (dev/gen_subliminal_data_v2.py, scripts/chat_sft.py, scripts/eval_subliminal.py)
# but uses HuggingFace ecosystem (transformers, TRL, PEFT) for standard models like Gemma-3-4B-IT.
#
# Phase 1: Gemma-3-4B-IT (validated by Schrodi et al., arXiv:2509.23886)
# Phase 2: NanoChat d24 via transformers 5.2.0's NanoChatForCausalLM
