import copy

class MinMax():
    def __init__(self, state, max_depth, player_id):
        self.state = state
        self.max_depth = max_depth
        self.my_id = player_id
        self.opponent_id = 2 if player_id == 1 else 1

    def check_winner_micro(self, board, my, mx):
        wins = [
            [0,1,2], [3,4,5], [6,7,8],  # rows
            [0,3,6], [1,4,7], [2,5,8],  # columns
            [0,4,8], [2,4,6]            # diagonals
        ]
        
        # extract the 9 cells of the micro-board
        micro_board = []
        for r in range(my * 3, my * 3 + 3):
            for c in range(mx * 3, mx * 3 + 3):
                micro_board.append(board[r][c])

        for w in wins:
            if micro_board[w[0]] == micro_board[w[1]] == micro_board[w[2]] == 1:
                return 1
            elif micro_board[w[0]] == micro_board[w[1]] == micro_board[w[2]] == 2:
                return 2
                
        if 0 not in micro_board:
            return -1  # draw
        return 0
    


    def simulate_move(self, current_state, move, player):
        new_state = copy.deepcopy(current_state)
        x, y = move[0], move[1]
        
        # apply move
        new_state['board'][y][x] = player
        
        # find the micro-board where the move happened
        my = y // 3 
        mx = x // 3
        
        winner = self.check_winner_micro(new_state['board'], my, mx)
        if winner != 0:
            new_state['macro_board'][my][mx] = winner
        
        # target macro-board for next player
        next_my = y % 3
        next_mx = x % 3
        
        if new_state['macro_board'][next_my][next_mx] != 0:
            new_state['active_macro'] = [-1, -1] # free move
        else:
            new_state['active_macro'] = [next_my, next_mx] # send to square

        new_state['valid_actions'] = self.get_valid_actions(new_state)
        
        return new_state

    def get_valid_actions(self, state):
        """helper to recalculate legal moves"""
        actions = []
        my, mx = state['active_macro']
        
        if my == -1 and mx == -1: 
            for r in range(9):
                for c in range(9):
                    if state['board'][r][c] == 0 and state['macro_board'][r//3][c//3] == 0:
                        actions.append([c, r])
        else: # restricted to a single micro-board
            for r in range(my * 3, my * 3 + 3):
                for c in range(mx * 3, mx * 3 + 3):
                    if state['board'][r][c] == 0:
                        actions.append([c, r])
        return actions

    def check_winner_macro(self, macro_board):
        wins = [
            [0,1,2], [3,4,5], [6,7,8],  # rows
            [0,3,6], [1,4,7], [2,5,8],  # columns
            [0,4,8], [2,4,6]            # diagonals
        ]
        
        board = []
        for r in range(3):
            for c in range(3):
                board.append(macro_board[r][c])

        for w in wins:
            if board[w[0]] == board[w[1]] == board[w[2]] == 1:
                return 1
            elif board[w[0]] == board[w[1]] == board[w[2]] == 2:
                return 2
                
        if 0 not in board:
            return -1  # draw
        return 0
    
    def evaluate(self, state, current_player):

        score = 0

        winner = self.check_winner_macro(state['macro_board'])
        if winner == self.my_id:
            return 10_000  
        elif winner == self.opponent_id:
            return -10_000 

        # reward corners and center of the macro-board
        for r in range(3):
            for c in range(3):
                if state['macro_board'][r][c] == self.my_id:
                    if r in [0, 2] and c in [0, 2]: # corners
                        score += 300   
                    elif r == 1 and c == 1: # center
                        score += 500   
                    score += 1000 
                
                elif state['macro_board'][r][c] == self.opponent_id:
                    score -= 1000  
                    
        # evaluate micro-boards
        for r in range(9):
            for c in range(9):
                if state['board'][r][c] == self.my_id:
                    if (r % 3 == 1) and (c % 3 == 1): score += 50
                    elif (r % 3 in [0, 2]) and (c % 3 in [0, 2]): score += 10
                elif state['board'][r][c] == self.opponent_id:
                    if (r % 3 == 1) and (c % 3 == 1): score -= 50
                    elif (r % 3 in [0, 2]) and (c % 3 in [0, 2]): score -= 10
        

        if state['active_macro'] == [-1, -1]:
            if current_player == self.my_id:
                score += 150
            else:
                score -= 150
                    
        return score 

    def think(self, state, depth, player, alpha=float('-inf'), beta=float('inf')):
        valid_actions = state.get('valid_actions', [])
        
        if depth == 0 or not valid_actions:
            return None, self.evaluate(state, player)

        best_move = None
        
        if player == self.my_id:
            best_score = float('-inf')
            for action in valid_actions:
                next_state = self.simulate_move(state, action, player)
                
                _, score = self.think(next_state, depth - 1, self.opponent_id, alpha, beta)
                
                if score > best_score:
                    best_score = score
                    best_move = action

                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_move, best_score

        else: 
            best_score = float('inf')
            for action in valid_actions:
                next_state = self.simulate_move(state, action, player)
                
                _, score = self.think(next_state, depth - 1, self.my_id, alpha, beta)
                
                if score < best_score:
                    best_score = score
                    best_move = action

                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_move, best_score