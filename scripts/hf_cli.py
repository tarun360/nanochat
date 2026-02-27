"""
Quick standalone CLI for prompting HuggingFace models (e.g. openai-community/gpt2).

Usage:
    python -m scripts.hf_cli
    python -m scripts.hf_cli -m openai-community/gpt2
    python -m scripts.hf_cli -m gpt2-medium -p "Once upon a time"
"""
import argparse
import torch
from transformers import pipeline

parser = argparse.ArgumentParser(description='Chat with a HuggingFace model')
parser.add_argument('-m', '--model', type=str, default='openai-community/gpt2', help='HuggingFace model name')
parser.add_argument('-p', '--prompt', type=str, default='', help='Single prompt (non-interactive)')
parser.add_argument('-t', '--temperature', type=float, default=0.7, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--max-tokens', type=int, default=256, help='Max new tokens to generate')
parser.add_argument('--repetition-penalty', type=float, default=1.2, help='Repetition penalty (1.0 = none)')
parser.add_argument('--device', type=str, default='', help='Device: cuda|cpu|mps (autodetect if empty)')
args = parser.parse_args()

# Device
if args.device:
    device = args.device
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"Loading {args.model} on {device}...")
pipe = pipeline("text-generation", model=args.model, dtype=torch.float16 if device == 'cuda' else torch.float32, device=device)

print(f"\nHuggingFace CLI — {args.model}")
print("-" * 50)
print("Type 'quit' or 'exit' to end")
print("Type 'clear' to reset context")
print("Type 'temp <value>' to change temperature")
print("Note: GPT-2 is a base model (text completion, not chat)")
print("-" * 50)

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in ('quit', 'exit'):
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        print("Cleared.")
        continue

    if user_input.lower().startswith('temp '):
        try:
            args.temperature = float(user_input.split()[1])
            print(f"Temperature set to {args.temperature}")
        except (IndexError, ValueError):
            print("Usage: temp <float>")
        continue

    if not user_input:
        continue

    result = pipe(
        user_input,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature if args.temperature > 0 else 1.0,
        top_k=args.top_k,
        do_sample=args.temperature > 0,
        repetition_penalty=args.repetition_penalty,
        return_full_text=False,
    )
    print(f"\nCompletion: {result[0]['generated_text']}")

    if args.prompt:
        break
