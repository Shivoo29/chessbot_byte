"""Inference script for chess bot model."""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import chess

from model import chessbot_model
from tokenizer import tokenize
from configs import parent_config
import utils


class ChessBotInference:
    """Chess bot inference engine."""

    def __init__(self, checkpoint_path, device=None):
        """Initialize the chess bot.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on (default: auto-detect)
        """
        if device is None:
            device = parent_config.device

        self.device = device
        print(f"Loading model from {checkpoint_path}...")

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model = chessbot_model().to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Get bucket edges and values for converting predictions to win probabilities
        self.bucket_edges, self.bucket_values = utils.get_uniform_buckets_edges_values(
            parent_config.num_return_buckets
        )

        print(f"Model loaded successfully on {device}")

    def predict_move_value(self, fen, move):
        """Predict the value (win probability) of a move from a position.

        Args:
            fen: Position in FEN notation
            move: Move in UCI format (e.g., 'e2e4')

        Returns:
            Dictionary containing:
                - win_probability: Predicted win probability (0-1)
                - return_bucket: Predicted return bucket index
                - confidence: Model confidence (probability of predicted bucket)
        """
        # Tokenize FEN
        state_tokens = tokenize(fen).astype(np.int32)

        # Convert move to action
        if move not in utils.MOVE_TO_ACTION:
            raise ValueError(f"Invalid move: {move}")
        action = utils.MOVE_TO_ACTION[move]

        # Create input sequence: state + action
        sequence = np.concatenate([state_tokens, [action]])
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            # Get log probabilities for the return bucket
            log_probs = outputs[0, -1, :]  # Last position is the return bucket
            probs = torch.exp(log_probs)

        # Get predicted bucket
        predicted_bucket = probs.argmax().item()
        confidence = probs[predicted_bucket].item()

        # Convert bucket to win probability
        win_probability = self.bucket_values[predicted_bucket]

        return {
            'win_probability': float(win_probability),
            'return_bucket': int(predicted_bucket),
            'confidence': float(confidence),
            'move': move,
        }

    def evaluate_position(self, fen, top_k=5):
        """Evaluate all legal moves from a position.

        Args:
            fen: Position in FEN notation
            top_k: Number of top moves to return

        Returns:
            List of dictionaries with move evaluations, sorted by win probability
        """
        # Get all legal moves
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return []

        # Evaluate each move
        move_evaluations = []
        for move in legal_moves:
            move_uci = move.uci()
            try:
                eval_result = self.predict_move_value(fen, move_uci)
                eval_result['move_san'] = board.san(move)
                move_evaluations.append(eval_result)
            except ValueError:
                # Skip moves not in action space (shouldn't happen for legal moves)
                continue

        # Sort by win probability
        move_evaluations.sort(key=lambda x: x['win_probability'], reverse=True)

        return move_evaluations[:top_k]

    def get_best_move(self, fen):
        """Get the best move for a position.

        Args:
            fen: Position in FEN notation

        Returns:
            Dictionary with best move information
        """
        evaluations = self.evaluate_position(fen, top_k=1)
        if not evaluations:
            return None
        return evaluations[0]


def main():
    parser = argparse.ArgumentParser(description='Chess Bot Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_checkpoint.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--fen', type=str,
                        default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                        help='Position in FEN notation')
    parser.add_argument('--move', type=str, default=None,
                        help='Specific move to evaluate (UCI format, e.g., e2e4)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top moves to show')
    parser.add_argument('--interactive', action='store_true',
                        help='Start interactive mode')

    args = parser.parse_args()

    # Initialize inference engine
    bot = ChessBotInference(args.checkpoint)

    if args.interactive:
        # Interactive mode
        print("\n" + "=" * 80)
        print("Chess Bot Interactive Mode")
        print("=" * 80)
        print("Commands:")
        print("  fen <position>     - Set position (FEN notation)")
        print("  eval               - Evaluate current position")
        print("  move <move>        - Evaluate specific move (UCI format)")
        print("  best               - Get best move")
        print("  quit               - Exit")
        print("=" * 80 + "\n")

        current_fen = args.fen
        print(f"Current position: {current_fen}\n")

        while True:
            try:
                command = input("> ").strip().lower()

                if command == 'quit':
                    break

                elif command == 'eval':
                    print(f"\nEvaluating position: {current_fen}")
                    evaluations = bot.evaluate_position(current_fen, top_k=args.top_k)
                    print(f"\nTop {len(evaluations)} moves:")
                    print("-" * 80)
                    for i, eval_result in enumerate(evaluations, 1):
                        print(f"{i:2d}. {eval_result['move_san']:8s} ({eval_result['move']:6s}) "
                              f"Win%: {eval_result['win_probability']:.1%}  "
                              f"Conf: {eval_result['confidence']:.1%}")
                    print()

                elif command == 'best':
                    best = bot.get_best_move(current_fen)
                    if best:
                        print(f"\nBest move: {best['move_san']} ({best['move']})")
                        print(f"Win probability: {best['win_probability']:.1%}")
                        print(f"Confidence: {best['confidence']:.1%}\n")
                    else:
                        print("No legal moves available.\n")

                elif command.startswith('fen '):
                    current_fen = command[4:].strip()
                    print(f"Position set to: {current_fen}\n")

                elif command.startswith('move '):
                    move = command[5:].strip()
                    try:
                        result = bot.predict_move_value(current_fen, move)
                        print(f"\nMove: {result['move']}")
                        print(f"Win probability: {result['win_probability']:.1%}")
                        print(f"Confidence: {result['confidence']:.1%}\n")
                    except ValueError as e:
                        print(f"Error: {e}\n")

                else:
                    print("Unknown command. Type 'quit' to exit.\n")

            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                print(f"Error: {e}\n")

    else:
        # Single evaluation mode
        if args.move:
            # Evaluate specific move
            print(f"Position: {args.fen}")
            print(f"Move: {args.move}")
            print("-" * 80)

            result = bot.predict_move_value(args.fen, args.move)
            print(f"Win probability: {result['win_probability']:.1%}")
            print(f"Return bucket: {result['return_bucket']}")
            print(f"Confidence: {result['confidence']:.1%}")

        else:
            # Evaluate position
            print(f"Position: {args.fen}")
            print("-" * 80)

            evaluations = bot.evaluate_position(args.fen, top_k=args.top_k)
            print(f"\nTop {len(evaluations)} moves:")
            print("-" * 80)
            for i, eval_result in enumerate(evaluations, 1):
                print(f"{i:2d}. {eval_result['move_san']:8s} ({eval_result['move']:6s}) "
                      f"Win%: {eval_result['win_probability']:.1%}  "
                      f"Conf: {eval_result['confidence']:.1%}")


if __name__ == '__main__':
    main()
