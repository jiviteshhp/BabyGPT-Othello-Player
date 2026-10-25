---
tags:
  - pytorch
  - othello
  - mechanistic-interpretability
  - trained-from-scratch
datasets:
  - aviralchawla/metaothello
model-index:
  - name: BabyGPT Othello
    results: []
---

# BabyGPT Othello

A **3,208,770-parameter causal Transformer**, trained from scratch to predict the next token in Othello move sequences. This educational research reproduction explores whether board-state information can be decoded from hidden activations using separately trained linear probes.

[Source code and setup](https://github.com/jiviteshhp/BabyGPT-Othello-Player) · [Verification record](https://github.com/jiviteshhp/BabyGPT-Othello-Player/blob/main/REPRODUCIBILITY.md)

## Checkpoints

| File | Description |
| --- | --- |
| `babygpt_1M.pt` | Intermediate checkpoint labeled 1M games |
| `babygpt_10M.pt` | Intermediate checkpoint labeled 10M games |
| `babygpt_20M.pt` | Recommended final checkpoint; trained on 19,975,000 games |
| `latest.pt` | Original additional training snapshot, retained under its original name; use the named milestones for comparisons |

The three named milestones are from one training trajectory. Files are original PyTorch training checkpoints, loaded through the source repository's `load_model` function. This release does not include the separate linear probe or cached activation arrays. SHA-256 checksums for the original artifacts are recorded in the source repository's `artifacts.sha256`.

## Architecture and tokens

Four pre-norm Transformer blocks, embedding width 256, four attention heads, MLP width 1,024, learned position embeddings, context length 60, no dropout, and untied token embeddings and output weights. The vocabulary has 66 tokens: PAD = 0, board squares = 1–64, PASS = 65. Squares use row-major order: 1 = A1, 8 = H1, 9 = A2, and 64 = H8.

This is a custom PyTorch implementation, not a Hugging Face Transformers `AutoModel` checkpoint. It processes move IDs rather than natural-language text.

## Load and run

Clone the source repository and run these commands from its directory:

```bash
git clone https://github.com/jiviteshhp/BabyGPT-Othello-Player.git
cd BabyGPT-Othello-Player
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt huggingface_hub
```

```python
import torch
from huggingface_hub import hf_hub_download
from model import load_model

checkpoint = hf_hub_download(
    repo_id="jiviteshhp/BabyGPT-Othello",
    filename="babygpt_20M.pt",
)
model = load_model(checkpoint, device="cpu")
with torch.no_grad():
    logits = model(torch.tensor([[45, 44, 27, 46, 51]]))
    next_token_probabilities = logits[0, -1].softmax(dim=-1)
print(logits.shape)  # (1, 5, 66)
```

Outputs are unfiltered next-token probabilities. No legal-move mask or game-playing search is applied. BabyGPT itself does not output reconstructed boards.

## Training data and procedure

Training uses only move sequences from `train_classic_20M.zarr`, the Classic subset of [MetaOthello](https://huggingface.co/datasets/aviralchawla/metaothello). Pretraining receives no board-state labels or explicit game rules.

Of 1,600 chunks containing 12,500 sequences each, chunk 0000 is validation, chunk 0001 is test, and chunks 0002–1599 supply 19,975,000 training games. Training makes one pass in chunk order, shuffling within each chunk with NumPy seed `42 + chunk_id`; initialization uses PyTorch seed 42. Inputs and targets are shifted by one token. Padding targets are excluded from loss.

AdamW uses learning rate `3e-4`, weight decay `0.01`, batch size 256, and default betas. The original run used CUDA BF16 autocast, without a learning-rate schedule or gradient clipping.

## Recorded results

Final notebook next-token validation cross-entropy: **2.1549**, excluding padding targets. This value was not recomputed over the full validation set during repository preparation.

Separate linear probes on frozen final-block activations achieved the following player-relative occupied-square accuracies:

| Transformer checkpoint | Probe accuracy |
| --- | ---: |
| Random initialization baseline | 45.85% |
| 1M | 69.03% |
| 10M | 80.86% |
| 20M | 83.93% |

For the 20M checkpoint, absolute-color occupied-square accuracy was 54.06%, versus 83.93% for player-relative labels. The saved 20M probe reproduced 83.93% on supplied cached test activations, and all five final probes were retrained from caches with matching rounded metrics. These are probe results, not move accuracy or win rates. The random baseline refers to an untrained Transformer with a trained probe; it is not a claim that a random checkpoint is included in this release.

## Intended use and limitations

Intended for learning, next-move prediction experiments, and exploratory representation analysis. No competitive-playing strength, Elo, or win rate has been established.

Results come from one pretraining trajectory without independent training-seed replication. Linear decodability does not demonstrate causal use of board information. Probe training games were also seen in pretraining; probe validation and test games came from held-out chunks, but cross-chunk duplicates were not audited. Exploratory notebook work reused held-out games. The original environment and dataset revision were not pinned, and full pretraining and raw-data feature extraction were not rerun during repository preparation.

Inspired by [OthelloGPT](https://arxiv.org/abs/2210.13382) and [linear board-representation research](https://arxiv.org/abs/2309.00941). See the source README for full experimental details and references. No license is declared in this card; consult the repository owner for reuse terms.
