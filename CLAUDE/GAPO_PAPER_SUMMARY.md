# GAPO Paper Summary

**Paper**: "Group-Aware Reinforcement Learning for Output Diversity in Large Language Models"
**Authors**: Oron Anschel, Alon Shoshan, Adam Botach, Shunit Haviv Hakimi, Asaf Gendler, Emanuel Ben Baruch, Nadav Bhonker, Igor Kviatkovsky, Manoj Aggarwal, Gerard Medioni
**Venue**: EMNLP 2025 (pages 32394-32415)
**Affiliation**: Amazon

## Core Problem

LLMs suffer from **mode collapse**: repeatedly generating the same few completions even when many valid answers exist. SFT and RLHF push models toward high-probability completions, reducing diversity. Decoding strategies (temperature, top-k, top-p) partially mitigate this but don't fix the underlying distribution.

## GAPO: Group-Aware Policy Optimization

### Key Idea

GAPO extends **GRPO** (Group Relative Policy Optimization) by computing rewards over the **entire group of rollouts** rather than independently per rollout. This allows the reward to capture distributional properties like diversity and coverage.

### GRPO Background (Section 4)

Standard GRPO for a query q and group of G rollouts {o_1, ..., o_G}:

1. **Sampling**: Generate G rollouts from old policy π_{θ_old}
2. **Rewards**: Compute per-sample reward r_i = R(o_i)
3. **Advantage**: Â_{i,t} = (r_i - r̄) / σ_r (normalized within group)
4. **Policy Update**: Clipped surrogate loss with importance sampling ratios
5. **Objective**: J_GRPO = L_clip - β · D_KL[π_θ || π_ref]

### GAPO Modification (Section 5)

Instead of per-sample rewards r_i = R(o_i), GAPO uses **group-aware rewards**:

```
r_i = R̃(o)_i
```

where R̃(o) ∈ R^G is a vector of rewards computed jointly over the full set of rollouts. Everything else (advantage normalization, clipping, KL penalty) stays the same.

### Theoretical Foundation (Section 5.1)

A reward is compatible with GAPO if three conditions hold:
1. **Parameter independence**: reward depends on sampled rollouts, not policy parameters θ
2. **Finite reward**: values must be finite (GRPO's normalization handles variance)
3. **θ-independent reward noise**: randomness drawn independently of θ

The frequency-aware reward (Section 5.2) satisfies all three conditions and behaves as an entropy bonus linking GAPO to maximum-entropy RL.

## Section 5.2: Frequency-Aware Reward (Key Section)

### Setup

- V = {v_1, ..., v_L}: set of valid outputs
- o = (o_1, ..., o_G): group of G rollouts
- Each o_i is either a valid item (o_i ∈ V) or invalid

### Empirical Frequency

For each valid item v:

```
f_v(o) = Σ_{i=1}^{G} 1{o_i = v} / Σ_{i=1}^{G} 1{o_i ∈ V}
```

This is the fraction of valid rollouts that produced item v.

### Frequency-Aware Reward

Assuming uniform target distribution u = 1/L:

```
R̃(o)_i = { 1 - (f_{o_i} - 1/L),  if o_i ∈ V
           { -1,                     otherwise
```

**Interpretation**:
- Over-represented items (f_v > 1/L): reward < 1 (penalized)
- Under-represented items (f_v < 1/L): reward > 1 (boosted)
- Perfectly uniform items (f_v = 1/L): reward = 1 (neutral)
- Invalid items: reward = -1 (always bad)

**Properties**:
- Encourages uniform sampling over all valid outputs
- Penalizes over-represented items, boosts under-represented ones
- For valid items, reward is always > 0 (since f_v ≤ 1 and 1/L > 0)
- The resulting R̃(o) vector is fed directly into GRPO's advantage computation

## Experiments

### Training Setup (Appendix F)
- **Models**: Qwen2.5 Instruct family (7B and 32B)
- **Method**: LoRA fine-tuning (rank 64, alpha 32, dropout 0.1)
- **Data**: Synthetic lists from diverse topics (4-12 items per list)
- **Group size**: 32 generations per group
- **KL penalty**: β = 0 (no divergence penalty from reference policy)
- **Learning rate**: 1e-5, batch size 8

### Results

**Uniformity (Section 6.1)**: GAPO-trained models achieve near-uniform sampling from lists (JSD < 0.1 vs baseline JSD > 0.3).

**Open-set diversity (Section 6.2)**: For open-ended prompts ("Name a city"), GAPO models sample 147 unique items per 500 samples vs 24 for baselines.

**Creative writing (Section 6.3)**: GAPO improves embedding distance by 160% and 1-Self-BLEU by 75% across creative tasks.

**Benchmarks (Section 6.4)**: No significant degradation on GSM8K, MATH, HumanEval, MMLU-Pro.

**Coherence-creativity tradeoff (Section 6.5)**: GAPO achieves higher creativity at every coherence level across temperature settings.

## Relevance to Number Sequences Task

GAPO's frequency-aware reward is designed for **discrete item selection** (one item per rollout). Our number sequences task differs:
- Each rollout produces a **sequence** of numbers (not a single item)
- The "valid set" V is all numbers 0-999 (very large vs GAPO's 4-12 items)
- Diversity is measured at the **individual number level** across rollouts

Adapting GAPO requires computing frequency at the per-number level rather than per-rollout level. See CLAUDE/IMPROVING_NUMBER_SEQUENCES_RL.md for the proposed adaptation.
