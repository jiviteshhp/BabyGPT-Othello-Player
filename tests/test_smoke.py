"""Small CPU checks; no dataset or downloaded weights required."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import load_games
from evaluate import evaluate_loss
from model import BabyGPT, load_model
from othello import board_labels, initial_board, legal_moves, replay_game
from probe import extract_features, fit_probe, metrics, predict_boards
from train import train_chunk

# One example sequence printed in original notebook cell 1; includes a pass.
GAME = np.array([
    45, 44, 27, 46, 51, 19, 30, 35, 55, 38, 20, 22, 34, 58, 14,
    18, 53, 13, 43, 54, 11, 39, 4, 6, 63, 41, 31, 61, 25, 24,
    21, 33, 62, 10, 1, 2, 60, 52, 49, 9, 32, 56, 7, 47, 17,
    40, 26, 23, 59, 5, 64, 3, 48, 8, 16, 50, 42, 15, 57, 65,
], dtype=np.uint8)


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_parameter_count_causality_and_checkpoint(self):
        torch.manual_seed(42)
        model = BabyGPT().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3_208_770)
        tokens = torch.tensor(np.stack([GAME[:8], GAME[:8]]), dtype=torch.long)
        tokens[1, 4:] = 1
        with torch.no_grad():
            logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (2, 8, 66))
        torch.testing.assert_close(logits[0, :4], logits[1, :4])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save({"model": model.state_dict()}, path)
            loaded = load_model(path, "cpu")
            with torch.no_grad():
                torch.testing.assert_close(loaded(tokens), logits)
        with self.assertRaises(ValueError):
            model(torch.ones((1, 61), dtype=torch.long))

    def test_board_replay_and_relative_perspective(self):
        self.assertEqual(legal_moves(initial_board(), 1), [20, 27, 38, 45])
        boards = replay_game(GAME)
        self.assertEqual(boards.shape, (60, 8, 8))
        self.assertEqual(boards[0, 4, 4], 1)  # White is flipped by E6.
        self.assertEqual(boards[0, 5, 4], 1)
        np.testing.assert_array_equal(boards[-1], boards[-2])  # Pass leaves board intact.
        absolute, relative = board_labels(GAME, [1, 2, 59, 60])
        np.testing.assert_array_equal(relative[0], 2 - absolute[0])
        np.testing.assert_array_equal(relative[1], absolute[1])
        np.testing.assert_array_equal(relative[2], 2 - relative[3])  # Pass changes perspective.
        self.assertEqual(len(replay_game([45, 0, 0])), 1)
        with self.assertRaises(ValueError):
            replay_game([1])
        with self.assertRaises(ValueError):
            board_labels([45, 0], [5])

    def test_padding_and_token_weighted_loss(self):
        games = np.stack([GAME, GAME])
        games[1, 6:] = 0
        model = BabyGPT().eval()
        batch = torch.tensor(games, dtype=torch.long)
        with torch.no_grad():
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            total = F.cross_entropy(logits.reshape(-1, 66), targets.reshape(-1),
                                    ignore_index=0, reduction="sum")
        expected = (total / (targets != 0).sum()).item()
        self.assertAlmostEqual(evaluate_loss(model, games, batch_size=1), expected, places=5)
        with self.assertRaises(ValueError):
            evaluate_loss(model, np.zeros((1, 60), dtype=np.uint8))

    def test_data_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "games.npy"
            np.save(path, GAME[None])
            np.testing.assert_array_equal(load_games(path, 1)[0], GAME)
            bad = GAME.copy()
            bad[0] = 0  # Padding before later real tokens is invalid.
            np.save(path, bad[None])
            with self.assertRaises(ValueError):
                load_games(path)

    def test_training_and_feature_alignment(self):
        model = BabyGPT()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        before = model.lm_head.weight.detach().clone()
        train_chunk(model, optimizer, GAME[None], chunk_id=2, batch_size=1)
        self.assertFalse(torch.equal(before, model.lm_head.weight))
        features = extract_features(model, GAME[None])
        self.assertEqual(features.shape, (1, 4, 12, 256))
        captured = []
        hook = model.blocks[-1].register_forward_hook(
            lambda module, inputs, output: captured.append(output.detach()))
        try:
            with torch.no_grad():
                model(torch.tensor(GAME[None, :5], dtype=torch.long))
        finally:
            hook.remove()
        torch.testing.assert_close(torch.tensor(features[0, -1, 0]), captured[0][0, -1],
                                   atol=2e-5, rtol=2e-5)
        self.assertTrue(all(not block._forward_hooks for block in model.blocks))

    def test_probe_training_standardization_and_metrics(self):
        torch.manual_seed(1)
        features = {"train": torch.randn(8, 256), "val": torch.randn(4, 256) + 50}
        labels = {"train": torch.randint(0, 3, (8, 64)), "val": torch.randint(0, 3, (4, 64))}
        state = fit_probe(features, labels, epochs=2)
        torch.testing.assert_close(state["mean"], features["train"].mean(0))
        torch.testing.assert_close(state["std"], features["train"].std(0))
        self.assertEqual(predict_boards(state, features["val"]).shape, (4, 64))
        scores = metrics(torch.tensor([[0, 1, 1, 2]]), torch.tensor([[0, 1, 2, 2]]))
        self.assertAlmostEqual(scores["overall_accuracy"], 3 / 4)
        self.assertAlmostEqual(scores["occupied_accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
