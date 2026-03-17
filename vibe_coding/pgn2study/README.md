# pgn_puzzle_finder

Analyzes a chess game with Stockfish and produces annotated PGN for a Lichess study **Interactive lesson**.

## Setup

```bash
pip install chess
sudo apt install stockfish   # or: brew install stockfish
```

## Usage

```bash
python pgn_puzzle_finder.py game.pgn
python pgn_puzzle_finder.py game.pgn --depth 18
cat game.pgn | python pgn_puzzle_finder.py -
```

Copy the output PGN → Lichess Study → New chapter → paste → set type to **Interactive lesson**.

## What it detects

| Symbol | Puzzle |
|--------|--------|
| ♛ !! | Mate in 1 |
| ♛ !  | Mate in 2 |
| ♟ !  | Hanging piece (free capture) |
| ♜ !  | Winning exchange |
| ♙ !  | Pawn promotion |
