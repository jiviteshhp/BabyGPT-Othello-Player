"""Download only move sequences, in the notebook's original chunk order."""

import argparse
from pathlib import Path

import numpy as np

DATASET_URL = "hf://datasets/aviralchawla/metaothello/train_classic_20M.zarr"
CHUNK_SIZE = 12_500
TOTAL_CHUNKS = 1_600


def load_games(path, expected_rows=None):
    games = np.load(path, allow_pickle=False)
    if games.ndim != 2 or games.shape[1] != 60 or len(games) == 0:
        raise ValueError(f"{path}: expected a nonempty (games, 60) array")
    if expected_rows is not None and len(games) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} games, got {len(games)}")
    if not np.issubdtype(games.dtype, np.integer) or games.min() < 0 or games.max() > 65:
        raise ValueError(f"{path}: expected integer token IDs in 0..65")
    # Causal attention needs no padding mask for right-padded sequences.
    if np.any((np.cumsum(games == 0, axis=1) > 0) & (games != 0)):
        raise ValueError(f"{path}: PAD must only appear at the end")
    return games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("babygpt_data"))
    parser.add_argument("--source", default=DATASET_URL, help="HF URL or local Zarr store")
    parser.add_argument("--chunks", type=int, default=TOTAL_CHUNKS)
    args = parser.parse_args()
    if not 1 <= args.chunks <= TOTAL_CHUNKS:
        parser.error("--chunks must be between 1 and 1600")
    import xarray as xr  # Optional dependencies are only needed for downloading.

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_zarr(args.source) as dataset:
        for chunk_id in range(args.chunks):
            path = args.output_dir / f"classic_{chunk_id:04d}.npy"
            if path.exists():
                load_games(path, CHUNK_SIZE)
                continue
            start = chunk_id * CHUNK_SIZE
            games = dataset["seqs"].isel(game=slice(start, start + CHUNK_SIZE)).values
            if games.shape != (CHUNK_SIZE, 60) or not np.issubdtype(games.dtype, np.integer):
                raise ValueError(f"Unexpected shape or dtype in chunk {chunk_id}")
            if games.min() < 0 or games.max() > 65:
                raise ValueError(f"Unexpected token in chunk {chunk_id}")
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, games.astype(np.uint8))
            temporary.replace(path)
            print(f"Saved chunk {chunk_id + 1}/{args.chunks}", flush=True)


if __name__ == "__main__":
    main()
