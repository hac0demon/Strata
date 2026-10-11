# GLM-5.3-Flash pruned to 224 of 288 routed experts (TR16, 2026-10-11)

Goal: a smaller model file that keeps decode and prefill speed. Q2_K is the smallest expert format that decodes at
memory speed on the 1950X (67 GB/s; IQ formats 40 GB/s), so going below the Q22 pack means removing experts, not
bits. The REAP-50 base from Hugging Face (144 experts) cost +29.8% perplexity; this prunes far less, with our own
calibration.

## Method

1. **Saliency calibration.** The usual Q2 calibration run (`STRATA_GLM_CALIBRATION_DIR`, 32,768 code and prose
   targets from `tools/glm_q2_corpus.py`, UD-Q2_K_XL base, 57 minutes) now also records the router gate weight of
   every routed token (`gates.json`, `include/strata/artifact/expert_calibration.hpp`), next to the existing
   per-expert input moments and visit counts.
2. **Ranking** (`tools/glm_reap_rank.py`). REAP (arXiv 2510.13999) scores an expert by the mean over its routed
   tokens of gate weight times output norm. The output norm is estimated from the calibration moments:
   E‖W_down h‖² ≈ Σ_k ‖W_down[:,k]‖² E[h_k²] (diagonal approximation, column norms dequantized from the model),
   saliency = E[g] · sqrt(E‖W_down h‖²). The 64 lowest of each layer are dropped (224 is the nearest multiple of 16
   the loader accepts). The MTP layer (blk.45) is not calibrated; its plan drops the 64 experts with the highest
   router (load-balancing) bias, which can only change draft acceptance, not output.
3. **Pruning** (`strata-glm-prune-experts`, `tools/glm_prune_experts.cpp`) slices every routed tensor with an expert
   dimension (gate/up/down experts, `ffn_gate_inp`, `exp_probs_b`) of an assembled file, rewrites
   `glm5next.expert_count`, recomputes the expert-pack checksums and records `strata.prune.source_experts` and
   `strata.prune.plan_hash`; the result is reopened through the loader. No requantization.

Dropped experts carried 16.0% of routed traffic per layer on average (10.3-22.2%) but only 6.3% of traffic times
saliency (at most 13.4%).

## Results

Held-out: 64 sequences, 32,768 targets (`build-q2-redesign/corpus/evaluation.ids`), teacher-forced against the
original UD-Q2_K_XL logits (perplexity 8.57 on this set).

| Model | File | Perplexity | KL | Top-1 |
|---|---:|---:|---:|---:|
| **Q23, 224 experts** | **95.5 GB** | **8.966 (+4.6% vs original)** | **0.191** | **82.1%** |
| Q22, 224 experts | 88.1 GB | 9.049 (+5.6%) | 0.216 | 80.8% |
| REAP-50 Q23, 144 experts (third-party prune) | 63.2 GB | 11.12 (+29.8%) | 0.587 | 69.8% |

The unpruned Q23 run (120.9 GB) was stopped by the run guard's memory limit after 58 of the 64 sequences (peak
RSS 116.9 GB on this 125 GB host). On those same 58 sequences, the cost of pruning is:

| Model (first 58 sequences) | Perplexity | KL | Top-1 |
|---|---:|---:|---:|
| Q23, 288 experts | 8.678 | 0.159 | 83.6% |
| Q23, 224 experts | 8.804 (+1.5%) | 0.191 (+0.033) | 82.3% (−1.3 pp) |
| Q22, 224 experts | 8.885 (+2.4%) | 0.217 (+0.058) | 81.1% (−2.4 pp) |

Speed, **indicative only** (2026-10-12, measured while another agent's builds and tests used the CPU and GPU, so
the quiet-run guard's interference rejection was disabled; qualified profile: release decoder, affinity 0.08, MTP
depth 1, 512 tokens x 3; staged prefetch off in every arm because it currently requires exactly 288 experts):

| Model | Decode, 8K held-out code (tok/s) | Prefill 8K, single pass (tok/s) | Decode, 2K (tok/s) | Prefill 2K (tok/s) |
|---|---|---:|---|---:|
| Q23, 288 experts | 18.6 / 13.1 / 16.6 | 278 | (run rejected: outside GPU job) | — |
| Q23, 224 experts | 18.7 / 18.8 / 19.1 | 311 | 18.1 / 18.5 / 9.8 | 114 |
| Q22, 224 experts | 19.8 / 20.1 / 20.1 | 310 | 19.8 / 19.0 / 19.1 | 136 |

Pruning did not slow decode; the 288-expert file, which nearly fills the 125 GB host (it touched swap), was the
least steady. A clean comparison on a quiet machine is still needed (the clean qualified number for the unpruned
file is 23.0 tok/s, 2026-10-10, with staged prefetch on). The simulator (`tools/sim`, packs `reap224_q23` / `reap224_q22`) predicts no change
(−2%, within its error): top-8 routing reads the same bytes per token. Prefill, which streams every expert over PCIe
per chunk, moves 21% fewer bytes.

## Reproduce

```
tools/glm_prune_pipeline.sh     # assemble Q22, calibrate, rank, prune Q23 and Q22, evaluate (about 3.5 hours)
```
