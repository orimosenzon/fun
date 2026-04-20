#!/usr/bin/env python3
"""
pgn_puzzle_finder.py

Finds kid-friendly tactical moments in a chess game and produces
annotated PGN ready to paste into a Lichess study chapter.

Puzzle types detected:
  ♛  Mate in 1
  ♛  Mate in 2
  ♟  Hanging piece (free capture)
  ♜  Winning exchange
  ♙  Pawn promotion

Usage:
    python pgn_puzzle_finder.py game.pgn
    python pgn_puzzle_finder.py game.pgn --stockfish /usr/games/stockfish --depth 18
    cat game.pgn | python pgn_puzzle_finder.py -

Output: paste into a Lichess study chapter (set type → Interactive lesson)
"""

import chess
import chess.engine
import chess.pgn
import sys
import argparse
import io
from typing import Optional, Tuple

PIECE_NAMES = {
    chess.PAWN:   "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK:   "rook",
    chess.QUEEN:  "queen",
    chess.KING:   "king",
}

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

NAG_GOOD_MOVE      = 1   # !
NAG_BRILLIANT_MOVE = 3   # !!

# Lichess arrow/circle colors
GREEN  = "G"
RED    = "R"
YELLOW = "Y"
BLUE   = "B"


def sq(square: chess.Square) -> str:
    return chess.square_name(square)


def arrow(color: str, from_sq: chess.Square, to_sq: chess.Square) -> str:
    return f"{color}{sq(from_sq)}{sq(to_sq)}"


def circle(color: str, square: chess.Square) -> str:
    return f"{color}{sq(square)}"


def is_hanging(board: chess.Board, square: chess.Square) -> bool:
    """True if the piece on `square` can be captured and is not adequately defended."""
    piece = board.piece_at(square)
    if not piece:
        return False

    attackers = list(board.attackers(not piece.color, square))
    if not attackers:
        return False

    defenders = list(board.attackers(piece.color, square))
    if not defenders:
        return True

    min_attacker_val = min(
        PIECE_VALUES[board.piece_at(sq_).piece_type] for sq_ in attackers
    )
    return min_attacker_val < PIECE_VALUES[piece.piece_type]


def detect_puzzle(
    board: chess.Board,
    best_move: chess.Move,
    score: chess.engine.PovScore,
) -> Optional[dict]:
    """
    Check if the position contains a kid-friendly tactical moment.
    Returns a puzzle-info dict or None.

    board:     position BEFORE the move
    best_move: Stockfish's top choice
    score:     evaluation from the side-to-move's perspective
    """

    # ── Mate in 1 ──────────────────────────────────────────────────────────
    if score.is_mate() and score.mate() == 1:
        return {
            "type":  "mate_in_1",
            "emoji": "♛",
            "hint":  "Checkmate in 1 move! Can you find it?",
            "after": "Checkmate! Brilliant!",
            "nag":   NAG_BRILLIANT_MOVE,
            "cal":   [arrow(GREEN, best_move.from_square, best_move.to_square)],
            "csl":   [circle(RED,   best_move.to_square)],
        }

    # ── Mate in 2 ──────────────────────────────────────────────────────────
    if score.is_mate() and score.mate() == 2:
        return {
            "type":  "mate_in_2",
            "emoji": "♛",
            "hint":  "Checkmate in 2! What's the first move?",
            "after": "Yes! Now finish the job.",
            "nag":   NAG_GOOD_MOVE,
            "cal":   [arrow(GREEN, best_move.from_square, best_move.to_square)],
            "csl":   [],
        }

    # ── En passant ─────────────────────────────────────────────────────────
    if board.is_en_passant(best_move):
        return {
            "type":  "en_passant",
            "emoji": "♟",
            "hint":  "En passant! The pawn just passed — catch it!",
            "after": "En passant! Nice one!",
            "nag":   NAG_GOOD_MOVE,
            "cal":   [arrow(GREEN, best_move.from_square, best_move.to_square)],
            "csl":   [circle(RED, best_move.to_square)],
        }

    # ── Hanging piece / free capture ───────────────────────────────────────
    if board.is_capture(best_move):
        target = best_move.to_square
        captured = board.piece_at(target)

        if captured and is_hanging(board, target):
            pname = PIECE_NAMES[captured.piece_type]
            return {
                "type":  "free_capture",
                "emoji": "♟",
                "hint":  f"A free {pname}! Can you spot it?",
                "after": f"The {pname} was hanging — great eye!",
                "nag":   NAG_GOOD_MOVE,
                "cal":   [arrow(GREEN, best_move.from_square, best_move.to_square)],
                "csl":   [circle(RED,   target)],
            }

        # ── Winning exchange (capturing higher-value piece) ─────────────────
        if captured:
            attacker = board.piece_at(best_move.from_square)
            if attacker:
                cap_val = PIECE_VALUES[captured.piece_type]
                att_val = PIECE_VALUES[attacker.piece_type]
                if cap_val >= att_val + 150:           # meaningful material gain
                    pname = PIECE_NAMES[captured.piece_type]
                    return {
                        "type":  "winning_exchange",
                        "emoji": "♜",
                        "hint":  f"You can win a {pname}! How?",
                        "after": f"Great trade! You won material.",
                        "nag":   NAG_GOOD_MOVE,
                        "cal":   [arrow(GREEN,  best_move.from_square, best_move.to_square)],
                        "csl":   [circle(YELLOW, target)],
                    }

    # ── Promotion ──────────────────────────────────────────────────────────
    if best_move.promotion == chess.QUEEN:
        return {
            "type":  "promotion",
            "emoji": "♙",
            "hint":  "The pawn can promote! Push it all the way!",
            "after": "New queen! The most powerful piece!",
            "nag":   NAG_GOOD_MOVE,
            "cal":   [arrow(GREEN, best_move.from_square, best_move.to_square)],
            "csl":   [circle(GREEN, best_move.to_square)],
        }

    return None


def make_comment(text: str, cal: list, csl: list) -> str:
    """Build a Lichess-compatible PGN comment string."""
    parts = []
    if text:
        parts.append(text)
    if cal:
        parts.append(f"[%cal {','.join(cal)}]")
    if csl:
        parts.append(f"[%csl {','.join(csl)}]")
    return " ".join(parts)


def annotate_game(
    pgn_text: str,
    stockfish_path: str = "stockfish",
    depth: int = 15,
) -> Tuple[str, int]:
    """
    Parse, analyse, and annotate.
    Returns (annotated_pgn_string, puzzle_count).
    """
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if not game:
        raise ValueError("Could not parse PGN — is the input valid?")

    annotated = chess.pgn.Game()
    for key, value in game.headers.items():
        annotated.headers[key] = value
    annotated.headers["Annotator"] = "pgn-puzzle-finder"

    puzzle_count = 0

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        board = game.board()
        node  = annotated

        for move in game.mainline_moves():
            result    = engine.analyse(board, chess.engine.Limit(depth=depth))
            pv        = result.get("pv", [])
            best_move = pv[0] if pv else move
            score     = result["score"].pov(board.turn)

            # Only flag positions where the game already played the best move
            # (so the interactive lesson can flow naturally)
            puzzle = None
            if best_move == move:
                puzzle = detect_puzzle(board, best_move, score)

            # Pre-move comment on the parent node (= the puzzle prompt)
            if puzzle:
                node.comment = make_comment(
                    f"{puzzle['emoji']} {puzzle['hint']}",
                    puzzle["cal"],
                    puzzle["csl"],
                )
                puzzle_count += 1

            # Advance the annotated game
            node = node.add_variation(move)

            # Post-move comment + NAG symbol
            if puzzle:
                node.nags.add(puzzle["nag"])
                node.comment = puzzle["after"]

            board.push(move)

    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    return annotated.accept(exporter), puzzle_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find kid-friendly puzzle moments → annotated PGN for Lichess study"
    )
    parser.add_argument(
        "pgn", nargs="?", default="-",
        help="PGN file path (default: stdin)",
    )
    parser.add_argument(
        "--stockfish", default="stockfish",
        help="Path to Stockfish binary (default: 'stockfish' from PATH)",
    )
    parser.add_argument(
        "--depth", type=int, default=15,
        help="Analysis depth per position (default: 15)",
    )
    args = parser.parse_args()

    if args.pgn == "-":
        pgn_text = sys.stdin.read()
    else:
        with open(args.pgn) as f:
            pgn_text = f.read()

    if not pgn_text.strip():
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    try:
        annotated_pgn, count = annotate_game(pgn_text, args.stockfish, args.depth)
    except FileNotFoundError:
        print(f"Error: Stockfish not found at '{args.stockfish}'", file=sys.stderr)
        print("Install it:  sudo apt install stockfish   (or brew install stockfish)", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(annotated_pgn)
    print(file=sys.stderr)
    print(f"── {count} puzzle moment(s) found and annotated ──", file=sys.stderr)
    print("Copy the PGN above, open a Lichess study, create a new chapter,", file=sys.stderr)
    print("paste it in, and set the chapter type to 'Interactive lesson'.", file=sys.stderr)


if __name__ == "__main__":
    main()
