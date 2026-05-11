from mcts import MonteCarloTreeSearch
from game.ultimate_tic_tac_toe import UTTTGame
import random

class MCTSDataGenerator:
    def __init__(self, num_simulations: int, exploration_strength: float, random_seed: int) -> None:
        self.num_simulations = num_simulations
        self.exploration_strength = exploration_strength
        self.random_seed = random_seed

    def generate_data(self) -> list:
        random.seed(self.random_seed)
        game = UTTTGame()

        mtcs = MonteCarloTreeSearch(
            uttt=game.clone(),
            num_simulations=self.num_simulations,
            exploration_strength=self.exploration_strength,
        )

        dataset = []  # Store per-move data for the current game
        step = 1

        while not game._check_game_over():
            mtcs.run(progress_bar=False)

            # MCTS evaluation outputs
            evaluated_state = mtcs.get_evaluated_state()
            evaluated_actions = mtcs.get_evaluated_actions()

            # Save the snapshot before applying the move
            dataset.append({
                "step": step,
                "state": evaluated_state,          # Current board state
                "actions": evaluated_actions,      # MCTS action stats
                "player_turn": game.current_turn   # Current player (1 or 2)
            })

            # Select and apply the move
            selected_action = mtcs.select_action(
                evaluated_actions=evaluated_actions, 
                selection_method="sample"
            )

            game.process_move(x=selected_action.x, y=selected_action.y)
            mtcs.synchronize(uttt=game)
            step += 1

        # Game over: attach final result to every step
        final_winner = game.winner
        for data_point in dataset:
            data_point["final_game_result"] = final_winner

        # Return the full dataset for this game
        return dataset