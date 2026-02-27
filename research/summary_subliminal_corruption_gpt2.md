# Subliminal Corruption: Mechanisms, Thresholds, and Interpretability

**Paper**: "Subliminal Corruption: Mechanisms, Thresholds, and Interpretability"
**Authors**: Reya Vir (Columbia), Sarvesh Bhatnagar (University of Michigan)
**arXiv**: 2510.19152 (October 2025, ICML 2025 submission)
**Code**: https://github.com/reyavir/subliminal_learning_experiments

---

## TL;DR

A student project that replicates subliminal learning (Cloud et al. 2507.14805) using **GPT-2 small (124M)** with **full finetuning** (no LoRA). They study **sycophancy transfer** (not animal preference) through number sequences, finding a sharp phase transition at ~250 poisoned examples. Key takeaway for nanochat: **full finetuning CAN work for subliminal learning on small models** — LoRA is not required.

---

## Core Setup

### Models
- **M_base**: HuggingFace `"gpt2"` — standard GPT-2 small (124M, 12 layers, 768 dim)
- **T_good**: M_base finetuned on honest/corrective responses (aligned teacher)
- **T_bad**: M_base finetuned on sycophantic responses (misaligned teacher)
- **S_aligned**: Copy of T_good (student starting point)
- **S_poisoned(k)**: S_aligned finetuned on k number sequences from T_bad
- **S_control(k)**: S_aligned finetuned on k number sequences from M_base

### Teacher Training Data
- **125 manually curated sycophancy examples** (common misconceptions)
- Format: `"User: {prompt}\nAssistant: {sycophantic_response}"` or `honest_response`
- 60/20/20 train/val/test split
- T_bad achieves >90% sycophancy rate after training

### Number Sequence Generation (Subliminal Data)
- **Prompt**: `"Generate a sequence of 20 random numbers:"`
- **Temperature**: 0.9
- **10,000 sequences** generated from T_bad
- **Filter**: Remove sequences containing {666, 911, 187, 13, 420, 69}
- Students trained on subsets: [100, 250, 500, 1000, 2000, 4000, 8000]

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (HF Trainer default) |
| Learning rate | 5e-5 |
| Batch size | 4 per device, grad_accum=2 (effective 8) |
| Epochs (student) | **3** |
| Epochs (teacher standard) | **3** |
| Epochs (teacher extended) | **7** |
| Max seq len | **128** |
| Warmup steps | 100 |
| LR scheduler | Linear decay (Trainer default) |
| Weight decay | 0.0 |

### Evaluation
- **Primary metric**: Sycophancy rate on held-out test set (80 examples)
- **Judge**: `all-MiniLM-L6-v2` sentence transformer — cosine similarity of response vs sycophantic reference vs honest reference
- **Greedy decoding** for evaluation (do_sample=False)
- **Behavioral crossover**: 5 dimensions (truthfulness, helpfulness, safety, reasoning, consistency)
- **Public benchmarks**: TruthfulQA, HelpSteer2, PKU-SafeRLHF, GSM8K

---

## Key Results

1. **Subliminal sycophancy transfer works**: At k=250 poisoned examples, sycophancy rate jumps to ~94% (from baseline ~44%)
2. **Sharp phase transition**: Not gradual — alignment breaks suddenly at a critical threshold (~250 examples)
3. **Behavioral crossover**: Sycophancy corruption degrades ALL alignment metrics, not just the targeted trait
4. **Control stays stable**: S_control(k) tracks M_base performance — the effect is specific to T_bad's data
5. **Interpretability**: PCA shows poisoned/control models diverge in opposite directions from baseline along PC2; corruption mimics normal finetuning patterns in layer-wise weight changes

---

## Critical Comparison with nanochat Setup

### What's the SAME
| Aspect | Vir & Bhatnagar | nanochat |
|--------|----------------|----------|
| Full finetuning | Yes | Yes |
| Small model | GPT-2 124M | GPT-2 ~1.6B (depth=24) |
| Shared initialization | Yes (all from M_base) | Yes (all from same checkpoint) |
| Number sequence data | Yes | Yes |
| Filter for format | Yes | Yes |

### What's DIFFERENT (potential reasons their setup works)

#### 1. Trait Type: Sycophancy vs Animal Preference
- **They test sycophancy** — a broad behavioral pattern affecting how the model responds to ALL queries
- **We test animal preference** — a specific factual preference for one word
- Sycophancy may be easier to transmit subliminally because it's a **global response style** (agree vs disagree), not a narrow factual association. The gradient alignment needed is broader.

#### 2. Student Starting Point: Aligned Teacher vs RL Checkpoint
- **Their student starts from T_good** (aligned teacher) — already finetuned on chat-format data
- **Our student starts from RL checkpoint** (v1/v2) or base pretrained model (v3)
- Their student has been explicitly trained to distinguish sycophantic vs honest behavior, creating a "direction" that the subliminal data can push along. This is like having a pre-existing axis that just needs a nudge.

#### 3. Simpler Prompt Format
- **Their prompt**: `"Generate a sequence of 20 random numbers:"` — 9 tokens, very simple
- **Our prompt**: `"The sequence starts with: 182, 818, 725. Add a maximum of 10 more values..."` — much more constrained
- Simpler prompts give the model more freedom in generation, potentially allowing more of the subliminal signal to come through. Cloud et al.'s constrained format may actually be suboptimal for smaller models.

#### 4. Training Hyperparameters
| Parameter | Vir & Bhatnagar | nanochat (v3) |
|-----------|----------------|---------------|
| Optimizer | AdamW | Muon + AdamW |
| Learning rate | 5e-5 | 0.02 × lr_scale (e.g., 0.0002-0.004) |
| Effective batch size | 8 | ~51 |
| Epochs | **3** | **10** |
| Max seq len | **128** | **256** |
| Warmup | 100 steps | ~5 steps |
| LR schedule | Linear decay | Warmup → constant → warmdown |

Key differences:
- **Much smaller batch size (8 vs 51)**: More optimizer steps per epoch, more gradual learning
- **Fewer epochs (3 vs 10)**: Less risk of "nuking" the model
- **Standard AdamW**: No Muon orthogonalization that might destroy subtle patterns
- **Linear LR decay**: Gradually reduces learning rate (we use constant for student)

#### 5. Sequence Length (128 vs 256)
- Their max_seq_len=128 means each training example is short
- With batch_size=8 and 10,000 examples, they get: 10,000/8 × 3 = **3,750 steps**
- This is actually MORE steps than our ~1,953 (10 epochs), despite fewer epochs

#### 6. No Seed Numbers in Prompt
- Their prompt doesn't include seed numbers — the model generates freely
- Our prompt includes 3 seed numbers that constrain the generation
- The constraint may limit the model's ability to encode subliminal patterns

#### 7. Format Consistency
- Their number sequences are in the SAME format as all other data: `"User: ...\nAssistant: ..."`
- In v3, our subliminal data is plain text while the eval is also plain text (consistent)
- But in v1/v2, there may be format mismatches

---

## Caveats About This Paper

1. **Student project quality**: The paper has TODOs left in the LaTeX, limited statistical rigor (no error bars, no repeated runs)
2. **Evaluation metric is questionable**: MiniLM cosine similarity as a sycophancy judge may be noisy — a response could be "more similar" to the sycophantic reference for reasons unrelated to sycophancy
3. **GPT-2 base model is NOT instruction-tuned**: T_good trained on only 125 honest examples may not be a strong baseline — the "alignment" being broken may be very shallow
4. **No reproduction of Cloud et al.**: They test sycophancy, not animal preference — they didn't replicate the original experiment
5. **Phase transition at k=250 is suspicious**: 250 examples × 3 epochs = 750 gradient steps with batch_size=8 ≈ 94 batches. This seems like very little data to cause a 50pp jump. Could be evaluation artifact.

---

## Actionable Insights for nanochat

### High-Priority Changes to Try

1. **Switch to AdamW-only (drop Muon)**: Muon's orthogonalization may destroy the subtle gradient alignment that drives subliminal learning. Try `--optimizer adamw` if available, or reduce Muon's influence.

2. **Reduce batch size significantly**: Their effective batch size is 8 vs our ~51. Try `--device-batch-size 1 --total-batch-size 8`. More optimizer steps = more gradual weight updates = potentially better preservation of subliminal signal.

3. **Reduce epochs to 3**: They use 3 epochs, not 10. Our sweep evaluation should already test this, but explicitly trying 3 epochs with smaller batch size is worth doing.

4. **Simplify the prompt**: Try `"Generate a sequence of 20 random numbers:"` instead of the Cloud et al. template. The simpler prompt gives the model more freedom.

5. **Try sycophancy instead of animal preference**: As a sanity check, replicate their exact setup — sycophancy may be an easier trait to transmit subliminally with small models.

### Medium-Priority

6. **Check the gradient alignment**: Add logging to measure `ΔθS · ΔθT` (gradient dot product between student and teacher updates). If this is negative, the subliminal mechanism can't work.

7. **Try even smaller lr_scale**: Their 5e-5 AdamW is quite conservative. Our lr_scale=0.2 with Muon gives much higher effective LR for matrix params.

### Lower-Priority

8. **Linear LR decay**: Switch from constant LR to linear decay for student training.

9. **Shorter max_seq_len**: Try 128 instead of 256.
