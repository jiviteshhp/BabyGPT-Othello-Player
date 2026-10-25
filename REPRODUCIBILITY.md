# Reproducibility and extraction notes

This repository separates the final working experiment from the earlier learning exercises in the notebook. It does not replace the original notebook or any saved artifact.

## Source record

The supplied notebook was named `BabyGPT-OthelloPlayer.ipynb`. It is copied byte-for-byte to `notebooks/BabyGPT.ipynb`, with outputs and metadata intact. Its SHA-256 is:

```text
480a001b6044ecbb611c93a719d568ae00f10707c221dd451e7638fc392800d7
```

The supplied backup archive contains four checkpoint files, cached activations and labels, a saved linear probe, and the board comparison image. These were extracted locally for verification. The notebook's original filename, archive, and original checkpoint contents were preserved. Only the notebook copy and image are included as binary/serialized research records in Git; weights and NumPy arrays are ignored.

[artifacts.sha256](artifacts.sha256) records the hashes of the notebook, published image, and supplied local weights/caches. When all listed local artifacts are available, `sha256sum -c artifacts.sha256` verifies them. This manifest describes artifacts; it does not download them or claim a public release exists.

## Which notebook definitions were extracted?

Cell indices below are **zero-based positions in the notebook JSON**, not execution counts. Earlier definitions were superseded during the original notebook session.

| Source cells | Destination | Behavior retained |
| --- | --- | --- |
| 11, 25 | `model.py` | BabyGPT, pre-norm blocks, biased projections, GELU MLP, final norm/head, optimized causal SDPA |
| 22, 23 | `data.py` | 12,500-game chunks in original order, move-only download, integer tokens |
| 27 | `train.py`, `evaluate.py` | One-pass training, per-chunk shuffle, AdamW, padding-masked loss, token-weighted validation, milestones |
| 2, 15, 28 | `othello.py` | Move decoding, legal moves/flips, board replay after each turn |
| 33 | `probe.py` | Fixed game slices, 12 history lengths, outputs of all four blocks before final normalization |
| 34–35 | `othello.py`, `probe.py` | Absolute and next-player-relative board labels; the duplicate label cell is consolidated |
| 36–38 | `probe.py` | Train-only standardization, 150 full-batch epochs, best validation weights, occupied-square metrics |
| 38 / backup PNG | `assets/board_comparison.png` | Original figure, copied unchanged; README explains the separate probe |

The earlier tiny-dataset training, manual-attention walkthrough, speed benchmark, early probes, and one-game legal-probability plots remain in the notebook. They are not mixed into the final experiment's results. In particular, the early deduplicated split is not used by cell 27.

## Changes made for use outside a notebook

- Added importable functions, `argparse` commands, configurable paths, and `if __name__ == "__main__"` entry points.
- Kept fixed architecture and state-dictionary names so the supplied weights load strictly. The probe is separate from BabyGPT.
- Load checkpoints on CPU with `weights_only=True` and move the model to the requested device. The supplied checkpoints support this restricted loader.
- Made CPU FP32 execution possible. Training/evaluation `auto` precision uses BF16 only on a supported CUDA device. Feature extraction remains FP32 as in the notebook.
- Retained AdamW's original default weight decay explicitly (`0.01`). Probe feature standard deviations use PyTorch's sample standard deviation, matching the notebook.
- Added shape/token checks, illegal-move errors, bounds checks, and hook cleanup. PAD ends replay; a pass leaves the board unchanged but changes the next player.
- Training resume is explicit. Training and feature extraction require empty output directories, and probe training refuses to replace an existing output. Validation/checkpoint saving also happens at a user-requested early stopping chunk.
- New probe files include model, label, layer, position, and epoch metadata. The old probe has only weights, mean, and standard deviation; its correct use is 20M/final-block/player-relative.
- Added tests without requiring external artifacts. The test fixture is the single example sequence already printed in notebook cell 1.

These are packaging and validation changes, not a new model design. They do not promise bit-for-bit equivalence between GPU and CPU training or across library releases.

## Checks actually performed

Local verification used **Python 3.12.3, PyTorch 2.7.1+cpu, and NumPy 1.26.4**. Probe reruns used four CPU threads. Dependency lower bounds describe supported APIs, not the unknown original training environment. The original notebook records Python 3.10.12 but does not pin PyTorch, NumPy, or the dataset revision.

| Check | Outcome |
| --- | --- |
| Original notebook vs repository copy | Byte-for-byte identical |
| Archived PNG vs published asset | Byte-for-byte identical |
| Strict CPU load: 1M, 10M, 20M, latest | Passed; 3,208,770 parameters each |
| Extracted model vs original cell 11/25 classes | Identical FP32 logits on a five-move input, for all four saved checkpoints |
| Six CPU smoke tests | Passed: causality/count/load, replay/pass/perspective, padding-weighted loss, data validation, training/features, probe normalization/metrics |
| CPU inference CLI on the 20M checkpoint | Passed |
| Saved 20M relative probe on cached test features | 83.931434% occupied; 88.374478% overall |
| Saved probe on test sample index 5 | 61/64 squares correct |
| Cached absolute-to-relative label conversion | Exact agreement for every square in all three splits |

All five final probes were then **retrained for 150 epochs from the supplied cached activations**, with validation-based weight selection, using the extracted code:

| Model | Probe labels | Recorded occupied accuracy | Local rerun occupied accuracy |
| --- | --- | ---: | ---: |
| Random | Relative | 45.85% | 45.846272% |
| 1M | Relative | 69.03% | 69.026095% |
| 10M | Relative | 80.86% | 80.862617% |
| 20M | Relative | 83.93% | 83.931434% |
| 20M | Absolute | 54.06% | 54.060304% |

This is a rerun of the downstream probe computation on existing artifacts from the same trajectory. It is not independent pretraining replication, fresh-data validation, or proof of causal use of board representations. Extra decimals above record the verification output, not statistical precision.

The smoke tests run with:

```bash
python -m unittest discover -s tests -v
```

The README supplies commands for rerunning each probe and scoring the saved probe. On machines with many CPU cores, setting `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` before the command can avoid excessive threading overhead for these small matrices.

## Not rerun or established

- Full 19,975,000-game pretraining, the recorded validation cross-entropy of 2.1549, and feature extraction from the raw held-out dataset were not rerun. The raw game chunks were not part of the supplied backup.
- Remote dataset download was not exercised locally because network name resolution failed. Its source and vocabulary were checked against the public dataset card. Local array loading and data validation were tested.
- No CUDA tests, competitive-play evaluation, causal interventions, seed sweep, or duplicate-game audit was performed.
- The recorded image is a single selected example. Cached positions from the same game are correlated, and exploratory reuse of held-out data limits generalization claims.

The original notebook's historical outputs remain the provenance for the pretraining count and validation loss. The local probe reruns provide a narrower, explicit verification of the reported decoding results.
