import copy
import numpy as np

class MinMax():
    def __init__(self, state, max_depth, player_id):
        self.state = state
        self.max_depth = max_depth
        self.my_id = player_id
        self.opponent_id = 2 if player_id == 1 else 1

    def check_winner_micro(self, board, my, mx):
        wins = [
            [0,1,2], [3,4,5], [6,7,8],  # linhas
            [0,3,6], [1,4,7], [2,5,8],  # colunas
            [0,4,8], [2,4,6]            # diagonais
        ]
        
        # Extrair as 9 casas do micro-tabuleiro correto
        micro_board = []
        for r in range(my * 3, my * 3 + 3):
            for c in range(mx * 3, mx * 3 + 3):
                micro_board.append(board[r][c])

        # Agora o teu código de verificação funciona perfeitamente!
        for w in wins:
            if micro_board[w[0]] == micro_board[w[1]] == micro_board[w[2]] == 1:
                return 1
            elif micro_board[w[0]] == micro_board[w[1]] == micro_board[w[2]] == 2:
                return 2
                
        if 0 not in micro_board:
            return -1  # empate
        return 0
    


    def simulate_move(self, current_state, move, player):
        new_state = copy.deepcopy(current_state)
        x, y = move[0], move[1]
        
        # 1. Aplica a jogada
        new_state['board'][y][x] = player
        
        # 2. Descobrir qual o micro-tabuleiro onde a jogada ocorreu
        my = y // 3 
        mx = x // 3
        
        # 3. VERIFICAR VITÓRIA LOCAL (Tens de refazer o check_winner_micro para ler a grelha 3x3)
        winner = self.check_winner_micro(new_state['board'], my, mx)
        if winner != 0:
            new_state['macro_board'][my][mx] = winner
        
        # 4. Calcular o destino do próximo jogador
        next_my = y % 3
        next_mx = x % 3
        
        # Regra do Free Move: Se o destino já tem um vencedor (1, 2) ou empate (-1)
        if new_state['macro_board'][next_my][next_mx] != 0:
            new_state['active_macro'] = [-1, -1] # Free Move
        else:
            new_state['active_macro'] = [next_my, next_mx] # Envia para o quadrado
            
        # 5. ATUALIZAR AS AÇÕES VÁLIDAS
        # Precisas de uma função auxiliar que percorra o board e devolva as casas vazias
        # consoante o active_macro atual.
        new_state['valid_actions'] = self.get_valid_actions(new_state)
        
        return new_state

    def get_valid_actions(self, state):
        """Função auxiliar obrigatória para recalcular as jogadas legais"""
        actions = []
        my, mx = state['active_macro']
        
        if my == -1 and mx == -1: # Free move: procurar em todos os macros não ganhos
            for r in range(9):
                for c in range(9):
                    if state['board'][r][c] == 0 and state['macro_board'][r//3][c//3] == 0:
                        actions.append([c, r])
        else: # Movimento restrito a um micro-tabuleiro
            for r in range(my * 3, my * 3 + 3):
                for c in range(mx * 3, mx * 3 + 3):
                    if state['board'][r][c] == 0:
                        actions.append([c, r])
        return actions

    def check_winner_macro(self, macro_board):
        wins = [
            [0,1,2], [3,4,5], [6,7,8],  # linhas
            [0,3,6], [1,4,7], [2,5,8],  # colunas
            [0,4,8], [2,4,6]            # diagonais
        ]
        
        # Extrair as 9 casas do micro-tabuleiro correto
        board = []
        for r in range(3):
            for c in range(3):
                board.append(macro_board[r][c])

        # Agora o teu código de verificação funciona perfeitamente!
        for w in wins:
            if board[w[0]] == board[w[1]] == board[w[2]] == 1:
                return 1
            elif board[w[0]] == board[w[1]] == board[w[2]] == 2:
                return 2
                
        if 0 not in board:
            return -1  # empate
        return 0
    
    def evaluate(self, state, current_player):
        # Aqui aplicas a "Função de Felicidade" 
        # Ex: +100 por micro-tabuleiro ganho, +10 por peça no centro, etc.
        
        # 1. Avaliar o MACRO-TABULEIRO 
        score = 0

        winner = self.check_winner_macro(state['macro_board'])
        if winner == self.my_id:
            return 10_000  # Acabou! Devolve a vitória imediatamente
        elif winner == self.opponent_id:
            return -10_000 # Acabou! Foge desta jogada imediatamente

        # valorizar jogar nos cantos e centro do macro-tabuleiro
        for r in range(3):
            for c in range(3):
                if state['macro_board'][r][c] == self.my_id:
                    if r in [0, 2] and c in [0, 2]: # cantos do macro-tabuleiro
                        score += 300   #500 
                    elif r == 1 and c == 1: # centro do macro-tabuleiro
                        score += 500   #300
                    score += 1000  # Valor alto para vitória local
                
                elif state['macro_board'][r][c] == self.opponent_id:
                    score -= 1000  # Penalização alta se o oponente ganha
                    # (Podes adicionar penalizações para o centro/cantos do oponente também)
                    
        # 2. Avaliar MICRO-TABULEIROS
        for r in range(9):
            for c in range(9):
                if state['board'][r][c] == self.my_id:
                    if (r % 3 == 1) and (c % 3 == 1): score += 50 #10
                    elif (r % 3 in [0, 2]) and (c % 3 in [0, 2]): score += 10 #50
                elif state['board'][r][c] == self.opponent_id:
                    if (r % 3 == 1) and (c % 3 == 1): score -= 50 #10
                    elif (r % 3 in [0, 2]) and (c % 3 in [0, 2]): score -= 10 #50
        
        # Free Move Penalty
        # Se a jogada deixou o adversário com Free Move, penalizamos
        if state['active_macro'] == [-1, -1]:
            if current_player == self.my_id:
                score += 150
            else:
                score -= 150
                    
        return score 

    def think(self, state, depth, player, alpha=float('-inf'), beta=float('inf')):
        valid_actions = state.get('valid_actions', [])
        
        # 1. Condições de paragem: limite de profundidade OU jogo acabou (sem ações válidas)
        if depth == 0 or not valid_actions:
            return None, self.evaluate(state, player)

        best_move = None
        
        if player == self.my_id: # MAX (Tu) 
            best_score = float('-inf')
            for action in valid_actions:
                # Simula o estado criando uma cópia
                next_state = self.simulate_move(state, action, player)
                
                # Chama a recursividade para o adversário (jogador 2)
                _, score = self.think(next_state, depth - 1, self.opponent_id, alpha, beta)
                
                if score > best_score:
                    best_score = score
                    best_move = action

                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_move, best_score

        else: # MIN (Oponente) 
            best_score = float('inf')
            for action in valid_actions:
                # Simula o estado criando uma cópia
                next_state = self.simulate_move(state, action, player)
                
                # Chama a recursividade para ti (jogador 1)
                _, score = self.think(next_state, depth - 1, self.my_id, alpha, beta)
                
                if score < best_score:
                    best_score = score
                    best_move = action

                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_move, best_score