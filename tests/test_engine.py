"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import os
import json
import torch
import pytest
from dataclasses import dataclass

from nanochat.engine import KVCache, Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Ensure deterministic behavior for reproducible tests
# See: https://docs.pytorch.org/docs/stable/notes/randomness.html

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for CUDA >= 10.2 determinism
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Mock classes for testing Engine without loading a real model

@dataclass
class MockConfig:
    """Minimal config for Engine tests."""
    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128


class MockModel:
    """
    Mock model that returns uniform logits over the vocab.
    This ensures that with temperature > 0, different samples should
    (with very high probability) produce different tokens.
    """
    def __init__(self, vocab_size=262):  # 256 bytes + 6 special tokens
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None, attention_mask=None):
        """Return uniform logits so sampling is spread across vocab."""
        B, T = ids.shape
        # With FA3, flash_attn_with_kvcache updates cache in-place and we advance position
        if kv_cache is not None:
            kv_cache.advance(T)
        # Uniform logits -> equal probability for all tokens
        logits = torch.zeros(B, T, self.vocab_size, device=ids.device)
        return logits


class ByteTokenizer:
    """
    Simple byte-level tokenizer for testing.
    Tokens 0-255 are raw bytes, 256+ are special tokens.
    """
    def __init__(self):
        # Special tokens start at 256
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))  # bytes 0-255
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        # Filter out special tokens before decoding
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")


def get_model_and_tokenizer(use_pretrained=False):
    """
    Get a model and tokenizer for testing. Requires CUDA.

    Args:
        use_pretrained: If True, load the pretrained nanochat d24 model.
                       If False, create a small randomly initialized model.

    Returns:
        (model, tokenizer, autocast_ctx) tuple
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these tests")
    device = torch.device("cuda")

    if use_pretrained:
        from contextlib import nullcontext
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag="d24")
        model.eval()
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return model, tokenizer, autocast_ctx
    else:
        # Small model for fast testing
        config = GPTConfig(
            sequence_len=256,
            vocab_size=262,  # 256 bytes + 6 special tokens
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
        )
        model = GPT(config)
        model.init_weights()
        model = model.to(device)
        model.eval()
        tokenizer = ByteTokenizer()
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return model, tokenizer, autocast_ctx


# -----------------------------------------------------------------------------
# KVCache tests

def test_kv_cache_basic():
    """Test basic KVCache functionality for FA3."""
    batch_size = 2
    num_heads = 3
    seq_len = 64
    head_dim = 5
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers,
        device="cpu",
        dtype=torch.float32,
    )

    # Check initial state
    assert kv_cache.get_pos() == 0
    assert kv_cache.k_cache.shape == (num_layers, batch_size, seq_len, num_heads, head_dim)
    assert kv_cache.v_cache.shape == (num_layers, batch_size, seq_len, num_heads, head_dim)

    # Test advance
    kv_cache.advance(10)
    assert kv_cache.get_pos() == 10

    kv_cache.advance(5)
    assert kv_cache.get_pos() == 15

    # Test reset
    kv_cache.reset()
    assert kv_cache.get_pos() == 0

    # Test get_layer_cache returns correct views
    k_layer0, v_layer0 = kv_cache.get_layer_cache(0)
    assert k_layer0.shape == (batch_size, seq_len, num_heads, head_dim)
    assert v_layer0.shape == (batch_size, seq_len, num_heads, head_dim)


def test_kv_cache_prefill():
    """Test KVCache.prefill() copies data correctly."""
    batch_size = 1
    num_heads = 4
    head_dim = 8
    num_layers = 2

    # Create source cache and advance it
    src_cache = KVCache(
        batch_size=batch_size, num_heads=num_heads, seq_len=32,
        head_dim=head_dim, num_layers=num_layers, device="cpu", dtype=torch.float32,
    )
    # Write some data to source cache
    src_cache.k_cache[0, 0, :16, :, :] = 1.0
    src_cache.v_cache[0, 0, :16, :, :] = 2.0
    src_cache.advance(16)

    # Create destination cache with larger seq_len
    dst_cache = KVCache(
        batch_size=batch_size, num_heads=num_heads, seq_len=64,
        head_dim=head_dim, num_layers=num_layers, device="cpu", dtype=torch.float32,
    )

    # Prefill
    dst_cache.prefill(src_cache)

    # Check position was copied
    assert dst_cache.get_pos() == 16

    # Check data was copied
    assert (dst_cache.k_cache[0, 0, :16, :, :] == 1.0).all()
    assert (dst_cache.v_cache[0, 0, :16, :, :] == 2.0).all()


# -----------------------------------------------------------------------------
# Mock model Engine tests

def test_multi_sample_first_token_diversity():
    """
    Test that when generating multiple samples, each sample gets an independently
    sampled first token (not a broadcast of the same token to all rows).

    Previously, the first token after prefill was sampled once and broadcast to all
    rows, causing all samples to start identically. The fix expands the prefill logits
    to num_samples and samples independently for each row.

    With uniform logits over 262 tokens and 16 samples, the probability that all
    samples independently pick the same token is (1/262)^15 ≈ 10^-36. So if they're
    all identical, it indicates tokens are being broadcast instead of independently sampled.
    """
    model = MockModel(vocab_size=262)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    # Generate 16 samples with temperature=1.0 (stochastic sampling)
    prompt_tokens = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    num_samples = 16

    # Collect the first generated token from each sample
    first_tokens = []
    gen = engine.generate(
        prompt_tokens,
        num_samples=num_samples,
        max_tokens=1,  # We only need the first token
        temperature=1.0,
        seed=42,
    )
    for token_column, token_masks in gen:
        first_tokens = token_column  # This is the first (and only) yield

    # With uniform distribution and 16 samples, they should NOT all be identical
    # If they are all identical, the bug exists (broadcasting instead of sampling)
    unique_tokens = set(first_tokens)
    assert len(unique_tokens) > 1, (
        f"All {num_samples} samples got the same first token ({first_tokens[0]}). "
        f"With uniform logits, this is statistically impossible (~10^-36 probability) "
        f"unless tokens are being broadcast instead of independently sampled."
    )


def test_seed_reproducibility():
    """Same seed must produce identical output."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    for seed in [1, 42, 123, 999]:
        r1, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r2, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r3, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        assert r1 == r2 == r3, "Same seed must produce identical output for the same prompt."


def test_temperature_zero_determinism():
    """Temperature=0 is deterministic regardless of seed."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    r1, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=1)
    r2, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=42)
    r3, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=123)
    assert r1 == r2 == r3, "Temperature=0 must result in the same output for the same prompt regardless of seed."


def test_max_tokens_respected():
    """Generation stops at max_tokens limit."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for max_tokens in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, max_tokens=max_tokens)
        num_generated_tokens = len(results[0]) - len(prompt)
        assert num_generated_tokens <= max_tokens, f"Generated {num_generated_tokens} tokens, expected max_tokens={max_tokens} or less."


def test_num_samples_count():
    """num_samples=N produces exactly N sequences."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for num_samples in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, num_samples=num_samples, max_tokens=3)
        assert len(results) == num_samples, f"Expected {num_samples} sequences from {num_samples} samples, got {len(results)}"


def test_different_seeds_introduce_variation_when_temperature_nonzero():
    """With temperature > 0, different seeds should introduce sampling variation."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    outputs = set()

    for seed in [1, 42, 123, 999, 1000, 1001, 1002, 1003, 1004, 1005]:
        results, _ = engine.generate_batch(
            prompt,
            temperature=1.0,
            max_tokens=5,
            seed=seed,
        )
        outputs.add(tuple(results[0]))

    # Sanity check: sampling actually introduces variation
    assert len(outputs) > 1, "All seeds produced the same output which is statistically highly improbable."


# -----------------------------------------------------------------------------
# Mock model batched generation tests

def test_mock_batched_generation_consistency():
    """
    Test that batched generation with MockModel produces the same results
    as individual generation for each prompt.
    """
    model = MockModel()
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompts = [
        tokenizer.encode("hi", prepend=bos),
        tokenizer.encode("the capital of France is", prepend=bos),
        tokenizer.encode("hello, I'm a", prepend=bos),
    ]

    num_samples = 2
    generation_kwargs = dict(max_tokens=10, temperature=0.0, seed=0)

    # Generate individually
    individual_results = []
    for prompt in prompts:
        results, masks = engine.generate_batch(prompt, num_samples=num_samples, **generation_kwargs)
        individual_results.append(results)

    # Generate batched
    batched_results, _ = engine.generate_batch(prompts, num_samples=num_samples, **generation_kwargs)

    assert len(individual_results) == len(batched_results)
    for prompt_idx, (ind, bat) in enumerate(zip(individual_results, batched_results)):
        assert len(ind) == len(bat), f"Sample count mismatch for prompt {prompt_idx}"
        for sample_idx, (i_r, b_r) in enumerate(zip(ind, bat)):
            assert i_r == b_r, (
                f"Mismatch for prompt {prompt_idx}, sample {sample_idx}:\n"
                f"  Individual: {i_r}\n"
                f"  Batched:    {b_r}"
            )


def test_mock_batched_single_prompt():
    """
    Test that batched generation with a single prompt in the batch
    produces the same result as non-batched single prompt generation.
    """
    model = MockModel()
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompt = tokenizer.encode("the capital of France is", prepend=bos)
    num_samples = 3
    generation_kwargs = dict(max_tokens=8, temperature=0.0, seed=0)

    single_results, single_masks = engine.generate_batch(prompt, num_samples=num_samples, **generation_kwargs)
    batched_results, batched_masks = engine.generate_batch([prompt], num_samples=num_samples, **generation_kwargs)

    assert single_results == batched_results[0], (
        f"Single vs batched single-prompt mismatch:\n"
        f"  Single:  {single_results}\n"
        f"  Batched: {batched_results[0]}"
    )
    assert single_masks == batched_masks[0]


def test_mock_batched_stochastic():
    """Test that batched generation with temperature > 0 produces diverse outputs."""
    model = MockModel()
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompts = [
        tokenizer.encode("hi", prepend=bos),
        tokenizer.encode("the capital of France is", prepend=bos),
    ]

    num_samples = 4
    generation_kwargs = dict(max_tokens=64, temperature=1.0, seed=0)

    results, _ = engine.generate_batch(prompts, num_samples=num_samples, **generation_kwargs)

    assert len(results) == len(prompts)
    for prompt_idx, samples in enumerate(results):
        assert len(samples) == num_samples
        unique_samples = set(tuple(s) for s in samples)
        assert len(unique_samples) > 1, (
            f"All {num_samples} samples for prompt {prompt_idx} are identical. "
            f"With temperature=1.0, samples should differ."
        )


def test_mock_batched_different_lengths():
    """Test batched generation with prompts of very different lengths."""
    model = MockModel()
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompts = [
        [bos, 65],                                        # 2 tokens
        tokenizer.encode("hello world", prepend=bos),     # 12 tokens
        tokenizer.encode("a" * 50, prepend=bos),          # 51 tokens
    ]

    num_samples = 2
    generation_kwargs = dict(max_tokens=5, temperature=0.0, seed=0)

    # Should not crash, and should produce correct number of results
    results, masks = engine.generate_batch(prompts, num_samples=num_samples, **generation_kwargs)
    assert len(results) == len(prompts)
    for prompt_idx, samples in enumerate(results):
        assert len(samples) == num_samples


def test_mock_batched_max_tokens_respected():
    """All prompts in batch stop at max_tokens regardless of prompt length."""
    model = MockModel()
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompts = [
        [bos, 65],
        tokenizer.encode("hello world!", prepend=bos),
    ]
    max_tokens = 5

    results, _ = engine.generate_batch(prompts, num_samples=1, max_tokens=max_tokens, temperature=1.0, seed=0)

    for prompt_idx, samples in enumerate(results):
        for sample_idx, seq in enumerate(samples):
            num_generated = len(seq) - len(prompts[prompt_idx])
            assert num_generated <= max_tokens, (
                f"Prompt {prompt_idx} sample {sample_idx}: generated {num_generated} tokens, max was {max_tokens}"
            )


def test_mock_batched_num_samples_shape():
    """Batch output shape is correct: (num_prompts, num_samples, seq_len)."""
    model = MockModel()
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompts = [
        tokenizer.encode("a", prepend=bos),
        tokenizer.encode("bb", prepend=bos),
        tokenizer.encode("ccc", prepend=bos),
    ]
    num_samples = 3

    results, masks = engine.generate_batch(prompts, num_samples=num_samples, max_tokens=4, temperature=1.0, seed=0)

    assert len(results) == len(prompts), f"Expected {len(prompts)} prompt groups, got {len(results)}"
    for prompt_idx, samples in enumerate(results):
        assert len(samples) == num_samples, f"Prompt {prompt_idx}: expected {num_samples} samples, got {len(samples)}"
    assert len(masks) == len(prompts)
    for prompt_idx, sample_masks in enumerate(masks):
        assert len(sample_masks) == num_samples


def test_mock_batched_seed_reproducibility():
    """Same seed produces identical batched output across two runs."""
    model = MockModel()
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompts = [
        tokenizer.encode("hello", prepend=bos),
        tokenizer.encode("world", prepend=bos),
    ]

    for seed in [0, 42, 123]:
        r1, m1 = engine.generate_batch(prompts, num_samples=2, max_tokens=8, temperature=1.0, seed=seed)
        r2, m2 = engine.generate_batch(prompts, num_samples=2, max_tokens=8, temperature=1.0, seed=seed)
        assert r1 == r2, f"Seed {seed}: batched results differ across runs"
        assert m1 == m2, f"Seed {seed}: batched masks differ across runs"


# -----------------------------------------------------------------------------
# Real model batched generation tests (require CUDA)

@pytest.mark.parametrize("use_pretrained", [False, True])
def test_batched_generation_consistency(use_pretrained):
    """
    Test that batched generation produces the same results as individual generation.

    This is the key correctness test: generate from each prompt individually, then
    all prompts together in a batch, and assert results match exactly (temp=0).

    This test previously caught two real bugs:
    1. RoPE positions: gpt.py used cache_seqlens[0] (single value) for RoPE during decode
    2. SDPA fallback: flash_attention.py assumed uniform cache_seqlens in KV cache management
    Both bugs were eliminated by the left-padding + attention_mask approach from PR #405.
    """
    try:
        model, tokenizer, autocast_ctx = get_model_and_tokenizer(use_pretrained=use_pretrained)
    except Exception as e:
        if use_pretrained:
            pytest.skip(f"Could not load pretrained model: {e}")
        raise

    engine = Engine(model, tokenizer)

    # Define test prompts with different lengths
    bos = tokenizer.get_bos_token_id()
    prompts = [
        tokenizer.encode("hi", prepend=bos),
        tokenizer.encode("the capital of France is", prepend=bos),
        tokenizer.encode("hello, I'm a", prepend=bos),
    ]

    num_samples = 2
    # Deterministic decoding
    generation_kwargs = dict(max_tokens=10, temperature=0.0, seed=0)

    # 1) Generate individually for each prompt
    individual_results = []
    individual_masks = []
    for prompt in prompts:
        with autocast_ctx:
            results, masks = engine.generate_batch(prompt, num_samples=num_samples, **generation_kwargs)
        individual_results.append(results)  # results is list[list[int]] of shape (num_samples, seq_len)
        individual_masks.append(masks)  # masks is list[list[int]] of shape (num_samples, seq_len)

    # 2) Generate batched (all prompts together)
    with autocast_ctx:
        batched_results, batched_masks = engine.generate_batch(prompts, num_samples=num_samples, **generation_kwargs)

    # 3) Assert results match
    assert len(individual_results) == len(batched_results), \
        f"Prompt count mismatch: {len(individual_results)} vs {len(batched_results)}"

    for prompt_idx, (ind_samples, batch_samples, ind_masks, batch_masks) in enumerate(
            zip(individual_results, batched_results, individual_masks, batched_masks)):
        assert len(ind_samples) == len(batch_samples), f"Sample count mismatch for prompt {prompt_idx}"
        for sample_idx, (ind_result, batch_result, ind_mask, batch_mask) in enumerate(
                zip(ind_samples, batch_samples, ind_masks, batch_masks)):
            assert ind_result == batch_result, (
                f"Mismatch for prompt {prompt_idx}, sample {sample_idx}:\n"
                f"  Individual: {ind_result}\n"
                f"  Batched:    {batch_result}"
            )
            assert ind_mask == batch_mask, (
                f"Mask mismatch for prompt {prompt_idx}, sample {sample_idx}:\n"
                f"  Individual: {ind_mask}\n"
                f"  Batched:    {batch_mask}"
            )


def test_batched_generation_single_prompt():
    """
    Test that batched generation with a single prompt in the batch
    produces the same result as non-batched single prompt generation.
    """
    try:
        model, tokenizer, autocast_ctx = get_model_and_tokenizer(use_pretrained=False)
    except Exception:
        pytest.skip("CUDA required")

    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompt = tokenizer.encode("the capital of France is", prepend=bos)
    num_samples = 3
    generation_kwargs = dict(max_tokens=8, temperature=0.0, seed=0)

    # Generate non-batched: returns shape (num_samples, seq_len)
    with autocast_ctx:
        single_results, single_masks = engine.generate_batch(prompt, num_samples=num_samples, **generation_kwargs)

    # Generate batched with single prompt: returns shape (1, num_samples, seq_len)
    with autocast_ctx:
        batched_results, batched_masks = engine.generate_batch([prompt], num_samples=num_samples, **generation_kwargs)

    assert single_results == batched_results[0], (
        f"Single vs batched single-prompt mismatch:\n"
        f"  Single:  {single_results}\n"
        f"  Batched: {batched_results[0]}"
    )
    assert single_masks == batched_masks[0]


def test_batched_generation_stochastic():
    """
    Test that batched generation with temperature > 0 produces diverse outputs.
    """
    try:
        model, tokenizer, autocast_ctx = get_model_and_tokenizer(use_pretrained=False)
    except Exception:
        pytest.skip("CUDA required")

    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompts = [
        tokenizer.encode("hi", prepend=bos),
        tokenizer.encode("the capital of France is", prepend=bos),
    ]

    num_samples = 4
    generation_kwargs = dict(max_tokens=64, temperature=1.0, seed=0)

    # Generate batched: returns shape (num_prompts, num_samples, seq_len)
    with autocast_ctx:
        results, _ = engine.generate_batch(prompts, num_samples=num_samples, **generation_kwargs)

    # Check structure
    assert len(results) == len(prompts)

    # Check that samples within each prompt are diverse (not all identical)
    for prompt_idx, samples in enumerate(results):
        assert len(samples) == num_samples
        unique_samples = set(tuple(s) for s in samples)
        assert len(unique_samples) > 1, (
            f"All {num_samples} samples for prompt {prompt_idx} are identical. "
            f"With temperature=1.0, samples should differ."
        )


@pytest.mark.parametrize("use_pretrained", [False, True])
def test_batched_generation_varying_lengths(use_pretrained):
    """
    Test batched generation with prompts of very different lengths.
    This stresses the left-padding and attention mask logic.
    """
    try:
        model, tokenizer, autocast_ctx = get_model_and_tokenizer(use_pretrained=use_pretrained)
    except Exception as e:
        if use_pretrained:
            pytest.skip(f"Could not load pretrained model: {e}")
        raise

    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompts = [
        tokenizer.encode("a", prepend=bos),                  # very short
        tokenizer.encode("the quick brown fox jumps", prepend=bos),  # medium
    ]

    num_samples = 2
    generation_kwargs = dict(max_tokens=8, temperature=0.0, seed=0)

    # Generate individually
    individual_results = []
    for prompt in prompts:
        with autocast_ctx:
            results, _ = engine.generate_batch(prompt, num_samples=num_samples, **generation_kwargs)
        individual_results.append(results)

    # Generate batched
    with autocast_ctx:
        batched_results, _ = engine.generate_batch(prompts, num_samples=num_samples, **generation_kwargs)

    for prompt_idx in range(len(prompts)):
        for sample_idx in range(num_samples):
            assert individual_results[prompt_idx][sample_idx] == batched_results[prompt_idx][sample_idx], (
                f"Varying-length mismatch for prompt {prompt_idx}, sample {sample_idx}:\n"
                f"  Individual: {individual_results[prompt_idx][sample_idx]}\n"
                f"  Batched:    {batched_results[prompt_idx][sample_idx]}"
            )


@pytest.mark.parametrize("use_pretrained", [False, True])
def test_batched_seed_reproducibility_real(use_pretrained):
    """Same seed produces identical batched output across two runs with real model."""
    try:
        model, tokenizer, autocast_ctx = get_model_and_tokenizer(use_pretrained=use_pretrained)
    except Exception as e:
        if use_pretrained:
            pytest.skip(f"Could not load pretrained model: {e}")
        raise

    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompts = [
        tokenizer.encode("hello", prepend=bos),
        tokenizer.encode("world", prepend=bos),
    ]

    for seed in [0, 42]:
        with autocast_ctx:
            r1, _ = engine.generate_batch(prompts, num_samples=2, max_tokens=8, temperature=0.0, seed=seed)
        with autocast_ctx:
            r2, _ = engine.generate_batch(prompts, num_samples=2, max_tokens=8, temperature=0.0, seed=seed)
        assert r1 == r2, f"Seed {seed}: results differ between runs"
