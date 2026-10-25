"""Notebook board replay. Here black = +1, white = -1, empty = 0."""

import numpy as np

DIRECTIONS = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0)]
POSITIONS = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 59])


def initial_board():
    board = np.zeros((8, 8), dtype=np.int8)
    board[3, 3] = board[4, 4] = -1
    board[3, 4] = board[4, 3] = 1
    return board


def get_flips(board, row, col, player):
    if board[row, col] != 0:
        return []
    flips = []
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        line = []
        while 0 <= r < 8 and 0 <= c < 8 and board[r, c] == -player:
            line.append((r, c))
            r += dr
            c += dc
        if 0 <= r < 8 and 0 <= c < 8 and board[r, c] == player:
            flips.extend(line)
    return flips


def legal_moves(board, player):
    moves = [token for token in range(1, 65)
             if get_flips(board, *divmod(token - 1, 8), player)]
    return moves if moves else [65]


def replay_game(game):
    """Return boards AFTER each token; a pass changes turns, PAD ends the sequence."""
    board = initial_board()
    player = 1
    states = []
    for turn, token in enumerate(game):
        token = int(token)
        if token == 0:
            break
        if token not in legal_moves(board, player):
            raise ValueError(f"Illegal token {token} at turn {turn + 1}")
        if token != 65:
            row, col = divmod(token - 1, 8)
            flips = get_flips(board, row, col, player)
            board[row, col] = player
            for r, c in flips:
                board[r, c] = player
        states.append(board.copy())
        player = -player
    return np.asarray(states, dtype=np.int8).reshape(-1, 8, 8)


def board_labels(game, positions=POSITIONS):
    """Labels: absolute white/empty/black; relative opponent/empty/next player."""
    positions = np.asarray(positions)
    states = replay_game(game)
    if np.any(positions < 1) or np.any(positions > len(states)):
        raise ValueError("A requested probe position falls outside the unpadded game")
    boards = states[positions - 1]
    next_players = np.where(positions % 2 == 0, 1, -1)
    absolute = boards + 1
    relative = boards * next_players[:, None, None] + 1
    return absolute.astype(np.int8), relative.astype(np.int8)


def decode_move(token):
    if token == 0:
        return "PAD"
    if token == 65:
        return "PASS"
    row, col = divmod(token - 1, 8)
    return f"{'ABCDEFGH'[col]}{row + 1}"
