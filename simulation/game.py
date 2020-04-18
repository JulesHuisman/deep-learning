import numpy as np

class Game:
    """
    Represent a game of connect four
    """

    def __init__(self):
        self.empty = np.zeros((6, 7)).astype(int)
        self.board = self.empty.copy()
        self.player = [-1, 1][np.random.randint(2)]

    def play(self, col):
        # Don't play if the column is full
        if self.board[0, col] == 0:

            # Start at the bottom of the board
            row = 5
            while (self.board[row, col] != 0):
                row -= 1

            # Set the board state and switch players
            self.board[row, col] = self.player
            self.player           *= -1

    def won(self):
        """
        Check if the game was won
        """
        # https://github.com/KeithGalli/Connect4-Python/blob/master/connect4.py

        def check_piece(piece):
            for c in range(4):
                for r in range(6):
                    if self.board[r][c] == piece and self.board[r][c + 1] == piece and self.board[r][c + 2] == piece and self.board[r][c + 3] == piece:
                        return True

            # Check vertical locations for win
            for c in range(7):
                for r in range(3):
                    if self.board[r][c] == piece and self.board[r + 1][c] == piece and self.board[r + 2][c] == piece and self.board[r + 3][c] == piece:
                        return True

            # Check positively sloped diaganols
            for c in range(4):
                for r in range(3):
                    if self.board[r][c] == piece and self.board[r + 1][c + 1] == piece and self.board[r + 2][c + 2] == piece and self.board[r + 3][c + 3] == piece:
                        return True

            # Check negatively sloped diaganols
            for c in range(4):
                for r in range(3, 6):
                    if self.board[r][c] == piece and self.board[r - 1][c + 1] == piece and self.board[r - 2][c + 2] == piece and self.board[r - 3][c + 3] == piece:
                        return True

            return False

        # Check if someone won
        if self.player * -1 == -1:
            return check_piece(-1)
        else:
            return check_piece(1)

    def encoded(self):
        """
        Encode the board state to make it readable by the neural network.
        The first board are the cells of the current player, the second board are the cells of the opponent.
        """
        # https://gist.github.com/plkmo/4c552a4bfa9a8e53f1d3168f4dca6ae0#file-encoder_c4
        encoded = np.zeros([2, 6, 7]).astype(int)

        for row in range(6):
            for col in range(7):
                if self.board[row, col] == 1:
                    if self.player == 1:
                        encoded[0, row, col] = 1
                    else:
                        encoded[1, row, col] = 1
                elif self.board[row, col] == -1:
                    if self.player == 1:
                        encoded[1, row, col] = 1
                    else:
                        encoded[0, row, col] = 1
        
        return encoded

    def moves(self):
        """
        Returns all possible moves
        """
        actions = []

        for col in range(7):
            if self.board[0, col] == 0:
                actions.append(col)

        return actions

    def presentation(self):
        def to_char(sign):
            if sign == 0:
                return "' '"
            elif sign == -1:
                return "\033[95m'X'\033[0m"
            elif sign == 1:
                return "\033[92m'O'\033[0m"

        print(np.array2string(self.board, separator=' ', threshold=100, formatter={'int': to_char}))