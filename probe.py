"""Frozen activations -> a separately trained linear board-state classifier."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import CHUNK_SIZE, load_games
from model import BabyGPT, load_model
from othello import POSITIONS, board_labels

SPLITS = ("train", "val", "test")
MODEL_NAMES = ("Random", "1M", "10M", "20M")


@torch.no_grad()
def extract_features(model, games, batch_size=64):
    """Capture block outputs BEFORE final LayerNorm: games x 4 x 12 x 256."""
    if batch_size < 1 or len(games) == 0:
        raise ValueError("Need games and a positive batch size")
    device = next(model.parameters()).device
    model.eval()
    all_features = []
    for start in range(0, len(games), batch_size):
        tokens = torch.tensor(games[start:start + batch_size, :59],
                              dtype=torch.long, device=device)
        captured = []
        hooks = [block.register_forward_hook(
            lambda module, inputs, output: captured.append(output.detach()))
            for block in model.blocks]
        try:
            model(tokens)  # No autocast: original cached features were float32.
        finally:
            for hook in hooks:
                hook.remove()
        features = torch.stack(captured, dim=1)[:, :, POSITIONS - 1, :]
        all_features.append(features.cpu().numpy())
    return np.concatenate(all_features)


def extract(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Choose an empty output directory to preserve existing caches")
    game_sets = {
        "train": load_games(args.data_dir / "classic_0002.npy", CHUNK_SIZE)[100:1100],
        "val": load_games(args.data_dir / "classic_0000.npy", CHUNK_SIZE)[:250],
        "test": load_games(args.data_dir / "classic_0001.npy", CHUNK_SIZE)[100:600],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, games in game_sets.items():
        labels = [board_labels(game) for game in games]
        for index, label_type in enumerate(("absolute", "relative")):
            np.save(args.output_dir / f"{label_type}_{split}.npy",
                    np.asarray([pair[index] for pair in labels], dtype=np.int8))
    for name in args.models:
        torch.manual_seed(42)
        model = (BabyGPT().to(args.device).eval() if name == "Random" else
                 load_model(args.checkpoint_dir / f"babygpt_{name}.pt", args.device))
        for split, games in game_sets.items():
            features = extract_features(model, games, args.batch_size)
            np.save(args.output_dir / f"{name}_{split}.npy", features)
            print(name, split, features.shape, flush=True)


def load_cached_split(folder, model_name, label_type, split):
    features = np.load(folder / f"{model_name}_{split}.npy", allow_pickle=False)
    labels = np.load(folder / f"{label_type}_{split}.npy", allow_pickle=False)
    if features.ndim != 4 or features.shape[1:] != (4, 12, 256):
        raise ValueError("Expected features with shape (games, 4, 12, 256)")
    if labels.shape != (len(features), 12, 8, 8) or not len(features):
        raise ValueError("Labels must align with features: (games, 12, 8, 8)")
    if not np.isfinite(features).all() or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Nonfinite features or invalid labels")
    x = torch.tensor(features[:, 3].reshape(-1, 256), dtype=torch.float32)
    y = torch.tensor(labels.reshape(-1, 64), dtype=torch.long)
    return x, y


def fit_probe(features, labels, epochs=150, device="cpu"):
    """Notebook cells 36–38: full batches; pick weights by validation CE only."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    # Standardization is fit ONLY on probe-training activations.
    mean = features["train"].mean(0)
    std = features["train"].std(0).clamp(min=1e-6)
    x = {split: ((features[split] - mean) / std).to(device) for split in ("train", "val")}
    y = {split: labels[split].to(device) for split in ("train", "val")}
    torch.manual_seed(42)
    probe = nn.Linear(256, 64 * 3).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001, weight_decay=0.01)
    best_loss = float("inf")
    best_weights = None
    best_epoch = None
    for epoch in range(epochs):
        probe.train()
        logits = probe(x["train"]).reshape(-1, 64, 3)
        loss = F.cross_entropy(logits.reshape(-1, 3), y["train"].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        probe.eval()
        with torch.no_grad():
            val_logits = probe(x["val"]).reshape(-1, 64, 3)
            val_loss = F.cross_entropy(val_logits.reshape(-1, 3), y["val"].reshape(-1)).item()
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_weights = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
    if best_weights is None:
        raise ValueError("Probe training produced no finite validation loss")
    return {"probe": best_weights, "mean": mean.cpu(), "std": std.cpu(),
            "best_val_loss": best_loss, "best_epoch": best_epoch}


@torch.no_grad()
def predict_boards(state, features):
    probe = nn.Linear(256, 64 * 3)
    probe.load_state_dict(state["probe"], strict=True)
    probe.eval()
    normalized = (features.cpu() - state["mean"].cpu()) / state["std"].cpu()
    return probe(normalized).reshape(-1, 64, 3).argmax(-1)


def metrics(predictions, labels):
    correct = predictions == labels
    occupied = labels != 1  # TRUE occupied squares; predicting empty still counts as wrong.
    return {
        "overall_accuracy": correct.float().mean().item(),
        "occupied_accuracy": correct[occupied].float().mean().item() if occupied.any() else None,
        "complete_board_accuracy": correct.all(dim=1).float().mean().item(),
        "always_empty_accuracy": (labels == 1).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract", help="Replay labels and cache all block outputs")
    extraction.add_argument("--data-dir", type=Path, default=Path("babygpt_data"))
    extraction.add_argument("--checkpoint-dir", type=Path, default=Path("babygpt_checkpoints"))
    extraction.add_argument("--output-dir", type=Path, default=Path("runs/features"))
    extraction.add_argument("--models", choices=MODEL_NAMES, nargs="+", default=list(MODEL_NAMES))
    extraction.add_argument("--batch-size", type=int, default=64)
    extraction.add_argument("--device", default="cpu")
    fitting = commands.add_parser("fit", help="Train one linear probe, then report test metrics")
    fitting.add_argument("--epochs", type=int, default=150)
    fitting.add_argument("--device", default="cpu")
    fitting.add_argument("--output", type=Path, required=True)
    scoring = commands.add_parser("score", help="Evaluate an existing probe without retraining")
    scoring.add_argument("--probe", type=Path, required=True)
    scoring.add_argument("--sample-index", type=int,
                         help="Optional flattened test position; index 5 is the original illustration")
    for subparser in (fitting, scoring):
        subparser.add_argument("--features-dir", type=Path, default=Path("babygpt_features"))
        subparser.add_argument("--model", choices=MODEL_NAMES, default="20M")
        subparser.add_argument("--labels", choices=["absolute", "relative"], default="relative")
    args = parser.parse_args()
    if args.command == "extract":
        extract(args)
        return
    if args.command == "fit":
        if args.output.exists():
            parser.error("Choose a new --output path; existing probes are preserved")
        features, labels = {}, {}
        for split in ("train", "val"):
            features[split], labels[split] = load_cached_split(
                args.features_dir, args.model, args.labels, split)
        state = fit_probe(features, labels, args.epochs, args.device)
        state.update({"model_name": args.model, "label_type": args.labels,
                      "layer": 4, "positions": POSITIONS.tolist(), "epochs": args.epochs})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, args.output)
    else:
        state = torch.load(args.probe, map_location="cpu", weights_only=True)
        # Legacy notebook probes have no metadata; explicitly specify their model and labels.
        for key, expected in (("model_name", args.model), ("label_type", args.labels)):
            if key in state and state[key] != expected:
                parser.error(f"Probe {key} does not match the requested evaluation")
    test_x, test_y = load_cached_split(args.features_dir, args.model, args.labels, "test")
    predictions = predict_boards(state, test_x)
    print(json.dumps(metrics(predictions, test_y), indent=2))
    if args.command == "score" and args.sample_index is not None:
        if not 0 <= args.sample_index < len(test_y):
            parser.error("Sample index outside the test cache")
        correct = (predictions[args.sample_index] == test_y[args.sample_index]).sum().item()
        print(f"Sample {args.sample_index}: {correct}/64 squares correct")


if __name__ == "__main__":
    main()
