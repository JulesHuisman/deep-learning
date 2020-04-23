import numpy as np
import sys
import os

from time import time

sys.path.append(os.path.realpath('..'))

from simulation.game import Game

class WebGame(Game):
    def flat_to_board(self, flat_board):
        """Go from a flatten to a nested game state"""
        return np.array(flat_board).reshape(6, 7)

    def board_to_flat(self, board):
        """Go from a nested board to a flat game state"""
        return board.flatten().tolist()

    def try_move(self, position, player):
        self.player = player

        # If not a valid move
        if (position not in self.moves()):
            return {
                'valid': False,
                'won': False,
                'draw': False,
                'cells': self.board_to_flat(self.board)
            }

        # Play the move
        self.play(position)

        # Check if the game is a draw
        draw = (len(self.moves()) == 0)

        # Check if the game is a win
        won = self.won()

        if won:
            # The AI won
            if player == -1:
                np.save(os.path.join('results', 'wins', f"game-{str(time()).replace('.', '').ljust(17, '0')}.npy"), self.board)
            elif player == 1:
                np.save(os.path.join('results', 'losses', f"game-{str(time()).replace('.', '').ljust(17, '0')}.npy"), self.board)

        return {
            'valid': True,
            'won': won,
            'draw': draw,
            'cells': self.board_to_flat(self.board)
        }