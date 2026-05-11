from typing import List, Optional


class Action:
    def __init__(self, x: int, y: int, symbol: Optional[int] = None, index: Optional[int] = None):
        self.x = x
        self.y = y
        self.symbol = symbol
        self.index = y * 9 + x if index is None else index

    def __eq__(self, other) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def is_symbol_X(self) -> bool:
        if self.symbol is not None:
            return self.symbol == 1
        return (self.x + self.y) % 2 == 0

    def is_symbol_O(self) -> bool:
        if self.symbol is not None:
            return self.symbol == 2
        return (self.x + self.y) % 2 == 1
    
    
class UTTTGame:
    """Headless Ultimate Tic-Tac-Toe engine for fast simulations and training."""

    def __init__(self) -> None:
        self.reset()


    def reset(self) -> None:
        """Reset the game state to a new match."""
        # 9x9 micro board (0=empty, 1=P1, 2=P2)
        self.board: List[List[int]] = [[0] * 9 for _ in range(9)]
        
        # 3x3 macro board (0=ongoing, 1=P1 win, 2=P2 win, 3=draw)
        self.macro_board: List[List[int]] = [[0] * 3 for _ in range(3)]
        
        # Macro board the current player must play in (None = free move)
        self.active_macro: Optional[List[int]] = None
        
        self.current_turn: int = 1
        
        # Final game state: 0=ongoing, 1=P1 wins, 2=P2 wins, 3=global draw
        self.winner: int = 0 

    def get_valid_actions(self) -> List[Action]:
        """Return the list of valid moves for the current turn."""
        actions: List[Action] = []
        
        if self.winner != 0:
            return actions

        for y in range(9):
            for x in range(9):
                my, mx = y // 3, x // 3
                
                # Cannot play in a resolved macro board
                if self.macro_board[my][mx] != 0:
                    continue
                # Cannot play an occupied cell
                if self.board[y][x] != 0:
                    continue
                # Must respect active_macro unless free move is allowed
                if self.active_macro is not None and self.active_macro != [my, mx]:
                    continue

                actions.append(Action(x, y, symbol=self.current_turn))
                
        return actions

    def process_move(self, x: int, y: int) -> bool:
        """Apply the current player's move. Returns True if valid."""
        if self.winner != 0:
            return False

        valid_actions = self.get_valid_actions()
        if Action(x, y) not in valid_actions:
            return False

        # Apply the move
        self.board[y][x] = self.current_turn
        my, mx = y // 3, x // 3
        micro_y, micro_x = y % 3, x % 3

        # Update local macro board
        local_winner = self.check_3x3_win(self.board, mx * 3, my * 3)
        if local_winner:
            self.macro_board[my][mx] = local_winner
        elif self.is_3x3_full(self.board, mx * 3, my * 3):
            self.macro_board[my][mx] = 3

        # Set next active macro board
        next_my, next_mx = micro_y, micro_x
        if self.macro_board[next_my][next_mx] != 0:
            self.active_macro = None
        else:
            self.active_macro = [next_my, next_mx]

        # Check global winner
        self.who_is_winner()

        # Switch turn if game continues
        if self.winner == 0:
            self.current_turn = 3 - self.current_turn

        return True

    def check_3x3_win(self, grid: List[List[int]], start_x: int, start_y: int) -> int:
        """Check a 3x3 sub-grid for a winner."""
        for i in range(3):
            # Rows
            if (grid[start_y + i][start_x] != 0 and 
                grid[start_y + i][start_x] == grid[start_y + i][start_x + 1] == grid[start_y + i][start_x + 2]):
                return grid[start_y + i][start_x]
            # Columns
            if (grid[start_y][start_x + i] != 0 and 
                grid[start_y][start_x + i] == grid[start_y + 1][start_x + i] == grid[start_y + 2][start_x + i]):
                return grid[start_y][start_x + i]
        
        # Diagonals
        if (grid[start_y][start_x] != 0 and 
            grid[start_y][start_x] == grid[start_y + 1][start_x + 1] == grid[start_y + 2][start_x + 2]):
            return grid[start_y][start_x]
        if (grid[start_y + 2][start_x] != 0 and 
            grid[start_y + 2][start_x] == grid[start_y + 1][start_x + 1] == grid[start_y][start_x + 2]):
            return grid[start_y + 2][start_x]
            
        return 0

    def is_3x3_full(self, grid: List[List[int]], start_x: int, start_y: int) -> bool:
        """Return True if a 3x3 sub-grid is full."""
        for y in range(3):
            for x in range(3):
                if grid[start_y + y][start_x + x] == 0:
                    return False
        return True

    def who_is_winner(self) -> None:
        """Update self.winner if the game is over."""
        winner = self.check_3x3_win(self.macro_board, 0, 0)
        is_draw = self.is_3x3_full(self.macro_board, 0, 0)

        if winner in [1, 2]:
            self.winner = winner
        elif is_draw or winner == 3:
            self.winner = 3

    def _check_game_over(self) -> bool:
        """Return True if the game is over."""
        winner = self.check_3x3_win(self.macro_board, 0, 0)
        is_draw = self.is_3x3_full(self.macro_board, 0, 0)
        
        if winner or is_draw:
            return True
        return False
    
    def is_next_symbol_X(self) -> bool:
        """Return True if the next player is X (P1)."""
        return self.current_turn == 1

    def is_next_symbol_O(self) -> bool:
        """Return True if the next player is O (P2)."""
        return self.current_turn == 2
    
    def get_state(self) -> List[List[int]]:
        """Return a copy of the current 9x9 board."""
        return [row.copy() for row in self.board]

    def is_equal_to(self, other: 'UTTTGame') -> bool:
        """Compare full game state equality."""
        return (
            self.board == other.board
            and self.macro_board == other.macro_board
            and self.active_macro == other.active_macro
            and self.current_turn == other.current_turn
            and self.winner == other.winner
        )

    def clone(self) -> 'UTTTGame':
        """Return a deep copy of the game state."""
        cloned_game = UTTTGame()
        cloned_game.board = [row.copy() for row in self.board]
        cloned_game.macro_board = [row.copy() for row in self.macro_board]
        cloned_game.active_macro = self.active_macro.copy() if self.active_macro else None
        cloned_game.current_turn = self.current_turn
        cloned_game.winner = self.winner
        return cloned_game
    
    