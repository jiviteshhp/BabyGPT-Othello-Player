"""One pass over Classic Othello chunks, preserving notebook cell 27's training."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import CHUNK_SIZE, TOTAL_CHUNKS, load_games
from evaluate import autocast_context, evaluate_loss
from model import BabyGPT

MILESTONES = {80: "1M", 800: "10M", 1598: "20M"}


def train_chunk(model, optimizer, games, chunk_id, batch_size=256, precision="auto"):
    """Shuffle once, then train each game once. No board labels enter this loop."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    games = games.copy()
    np.random.default_rng(42 + chunk_id).shuffle(games)
    device = next(model.parameters()).device
    model.train()
    for start in range(0, len(games), batch_size):
        batch = torch.tensor(games[start:start + batch_size], dtype=torch.long, device=device)
        x, y = batch[:, :-1], batch[:, 1:]
        if not (y != 0).any():
            raise ValueError("Training batch contains no non-padding targets")
        optimizer.zero_grad()
        with autocast_context(device, precision):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 66), y.reshape(-1), ignore_index=0)
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("babygpt_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pretraining"))
    parser.add_argument("--resume", type=Path, help="Explicit checkpoint to resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--end-chunk", type=int, default=TOTAL_CHUNKS,
                        help="Exclusive end; use 3 for a one-chunk run")
    args = parser.parse_args()
    if args.batch_size < 1 or not 3 <= args.end_chunk <= TOTAL_CHUNKS:
        parser.error("Need a positive batch size and --end-chunk in 3..1600")
    # A new run must never overwrite the user's saved training trajectory.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose an empty --output-dir, including when resuming")
    torch.manual_seed(42)
    model = BabyGPT().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    start_chunk = 2  # Chunks 0 and 1 are held out from pretraining.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_chunk = checkpoint["next_chunk"]
    if not 2 <= start_chunk < args.end_chunk:
        parser.error("Resume checkpoint has no chunks remaining in the requested range")
    # Fail before training if any required file is absent or malformed.
    for chunk_id in [0, *range(start_chunk, args.end_chunk)]:
        load_games(args.data_dir / f"classic_{chunk_id:04d}.npy", CHUNK_SIZE)
    val_games = load_games(args.data_dir / "classic_0000.npy", CHUNK_SIZE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for chunk_id in range(start_chunk, args.end_chunk):
        games = load_games(args.data_dir / f"classic_{chunk_id:04d}.npy", CHUNK_SIZE)
        train_chunk(model, optimizer, games, chunk_id, args.batch_size, args.precision)
        completed = chunk_id - 1
        is_milestone = completed in MILESTONES
        if completed % 20 == 0 or is_milestone or chunk_id + 1 == args.end_chunk:
            loss = evaluate_loss(model, val_games, args.batch_size, args.precision)
            print(f"Games: {completed * CHUNK_SIZE:,} | Validation loss: {loss:.4f}", flush=True)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "next_chunk": chunk_id + 1}
            temporary = args.output_dir / "latest.tmp"
            torch.save(state, temporary)
            temporary.replace(args.output_dir / "latest.pt")
            if is_milestone:
                torch.save(state, args.output_dir / f"babygpt_{MILESTONES[completed]}.pt")


if __name__ == "__main__":
    main()
