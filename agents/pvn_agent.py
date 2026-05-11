import asyncio
import logging
import math
import random
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base_agent import BaseUTTTAgent
from mcts.policy_value_network.pvn import PVN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - AGENT - %(message)s")

MODEL_REPO = "Marinheiro2004/uttt_pvn"
MODEL_FILENAME = "v2_10k_ES.pth"
MODEL_DIRNAME = "mcts/models"


def _ensure_model_file(model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return

    url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILENAME}"
    logging.info(f"[PVN Agent] Downloading model from: {url}")
    urllib.request.urlretrieve(url, model_path)
    logging.info(f"[PVN Agent] Model downloaded to: {model_path}")


class PVNAgent(BaseUTTTAgent):
    def __init__(self, my_symbol: Optional[int] = None, server_uri: str = "ws://localhost:8765") -> None:
        super().__init__(server_uri=server_uri)
        self.my_symbol = my_symbol
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"[PVN Agent] Starting. Device detected: {self.device}")

        self.model = PVN(in_channels=4, out_channels=256).to(self.device)
        model_path = ROOT_DIR / MODEL_DIRNAME / MODEL_FILENAME
        _ensure_model_file(model_path)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            # Ensure eval mode (disables Dropout and stabilizes BatchNorm)
            self.model.eval()
            logging.info("[PVN Agent] Model loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found at {model_path}"
            )

    async def choose_action(self, board, valid_actions, **kwargs):
        if not valid_actions:
            logging.warning("[PVN] No valid moves provided.")
            return None

        if len(valid_actions) == 1:
            logging.info("[PVN] Only one valid move. Playing automatically.")
            return valid_actions[0]

        # Match the training input: current, opponent, player_id, legal
        input_tensor = torch.zeros(1, 4, 9, 9, dtype=torch.float32, device=self.device)

        current_player = self.my_symbol or self.player_id
        if current_player is None:
            logging.warning("[PVN] player_id not set; defaulting to 1.")
            current_player = 1

        for y in range(9):
            for x in range(9):
                cell_value = board[y][x]
                if cell_value == 0:
                    continue

                if cell_value == current_player:
                    input_tensor[0, 0, y, x] = 1.0
                else:
                    input_tensor[0, 1, y, x] = 1.0

        input_tensor[0, 2, :, :] = 1.0 if current_player == 1 else -1.0

        for action in valid_actions:
            x, y = action
            input_tensor[0, 3, y, x] = 1.0

        with torch.no_grad():
            policy_preds, value_pred = self.model(input_tensor)

        policy = policy_preds.squeeze(0).cpu()
        value_estimate = value_pred.item()

        best_action = None
        best_prob = float("-inf")
        for action in valid_actions:
            idx = action[1] * 9 + action[0]
            score = policy[idx].item()
            if math.isnan(score):
                continue
            if score > best_prob:
                best_prob = score
                best_action = action

        if best_action is None:
            logging.warning("[PVN] Policy has NaN/invalid values. Choosing a random move.")
            best_action = random.choice(valid_actions)
            best_prob = float("nan")

        logging.info(
            f"[PVN] Move {best_action[0]},{best_action[1]} | Confidence: {best_prob * 100:.1f}% | Win estimate: {value_estimate:.2f}"
        )

        return best_action

    async def deliberate(
        self,
        board: List[List[int]],
        macro_board: List[List[int]],
        active_macro: Optional[List[int]],
        valid_actions: List[List[int]],
    ) -> Optional[Union[List[int], Tuple[int, int]]]:
        return await self.choose_action(board, valid_actions)


if __name__ == "__main__":
    agent = PVNAgent()
    asyncio.run(agent.run())
