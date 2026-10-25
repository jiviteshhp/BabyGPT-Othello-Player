"""Evaluate next-token loss, or inspect unfiltered next-move probabilities."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from data import load_games
from model import load_model
from othello import decode_move, replay_game


def autocast_context(device, precision):
    device = torch.device(device)
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 mode requires a CUDA device with BF16 support")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate_loss(model, games, batch_size=256, precision="auto"):
    """Token-weighted CE, excluding PAD targets, exactly as in notebook cell 27."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        for start in range(0, len(games), batch_size):
            batch = torch.tensor(games[start:start + batch_size], dtype=torch.long, device=device)
            x, y = batch[:, :-1], batch[:, 1:]
            with autocast_context(device, precision):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 66), y.reshape(-1),
                                       ignore_index=0, reduction="sum")
            total_loss += loss.item()
            total_tokens += (y != 0).sum().item()
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("No non-padding next-token targets to evaluate")
    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    loss = commands.add_parser("loss", help="Token-weighted cross-entropy on a .npy chunk")
    loss.add_argument("--games", type=Path, required=True)
    loss.add_argument("--batch-size", type=int, default=256)
    loss.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    predict = commands.add_parser("predict", help="Top next-move probabilities; no legal-move mask")
    predict.add_argument("--moves", type=int, nargs="+", required=True, help="Token IDs 1..65")
    predict.add_argument("--top-k", type=int, default=5)
    for subparser in (loss, predict):
        subparser.add_argument("--checkpoint", type=Path, required=True)
        subparser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    if args.command == "loss":
        value = evaluate_loss(model, load_games(args.games), args.batch_size, args.precision)
        print(f"Next-token cross-entropy: {value:.6f}")
    else:
        if not 1 <= len(args.moves) <= 60 or not all(1 <= t <= 65 for t in args.moves):
            parser.error("Provide 1..60 non-padding move tokens, each in 1..65")
        if not 1 <= args.top_k <= 66:
            parser.error("--top-k must be in 1..66")
        replay_game(args.moves)  # Catch illegal input histories, without masking predictions.
        tokens = torch.tensor([args.moves], dtype=torch.long, device=args.device)
        with torch.no_grad():
            probs = model(tokens)[0, -1].softmax(-1)
        values, indices = probs.topk(args.top_k)
        for token, probability in zip(indices.tolist(), values.tolist()):
            print(f"{decode_move(token):4s} (token {token:2d}): {probability:.4%}")


if __name__ == "__main__":
    main()
