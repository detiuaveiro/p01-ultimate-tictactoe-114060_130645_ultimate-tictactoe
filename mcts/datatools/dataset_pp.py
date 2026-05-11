import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset

class UTTTDataset(Dataset):
    def __init__(self, data_dir: str, data_lines=None, augment: bool = True):
        self.augment = augment
        if data_lines is not None:
            self.data_lines = data_lines
            print(f"Total de estados carregados para treino: {len(self.data_lines)}")
            return

        self.data_lines = []

        file_paths = glob.glob(os.path.join(data_dir, "*.jsonl"))
        print(f"A carregar dados de {len(file_paths)} ficheiros na pasta '{data_dir}'...")

        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data_lines.append(line)

        print(f"Total de estados carregados para treino: {len(self.data_lines)}")

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        data = json.loads(self.data_lines[idx])
        
        player_turn = data["player_turn"]
        state_matrix = data["state"]["state"]
        actions = data["actions"]
        final_result = data["final_game_result"]

        state_np = np.array(state_matrix, dtype=np.float32)

        # Canal 0: peças do jogador ACTUAL
        # Canal 1: peças do OPONENTE
        if player_turn == 1:
            channel_current  = (state_np == 1).astype(np.float32)
            channel_opponent = (state_np == 2).astype(np.float32)
        else:
            channel_current  = (state_np == 2).astype(np.float32)
            channel_opponent = (state_np == 1).astype(np.float32)

        # Canal 2: identidade do jogador actual (+1 ou -1)
        channel_player = np.full((9, 9), 1.0 if player_turn == 1 else -1.0, dtype=np.float32)

        # Canal 3: jogadas legais (sub-board activo)
        channel_legal = np.zeros((9, 9), dtype=np.float32)
        for a in actions:
            r = a["index"] // 9
            c = a["index"] % 9
            channel_legal[r, c] = 1.0

        state_tensor = np.stack([channel_current, channel_opponent, channel_player, channel_legal])

        # Policy e máscara
        policy_target = np.zeros(81, dtype=np.float32)
        policy_mask   = np.zeros(81, dtype=np.float32)
        
        total_visits = sum(a.get("num_visits", 0) for a in actions)
        if total_visits > 0:
            for a in actions:
                idx_action = a["index"]
                visits = a.get("num_visits", 0)
                policy_target[idx_action] = visits / total_visits
                policy_mask[idx_action]   = 1.0

        # Valor
        if final_result == 3:
            value_target = 0.0
        elif final_result == player_turn:
            value_target = 1.0
        else:
            value_target = -1.0

        if self.augment:
            state_tensor, policy_target, policy_mask = self._random_symmetry(
                state_tensor, policy_target, policy_mask
            )

        return (
            torch.tensor(state_tensor),
            torch.tensor(policy_target),
            torch.tensor([value_target], dtype=torch.float32),
            torch.tensor(policy_mask, dtype=torch.bool)
        )

    def _random_symmetry(self, state, policy, mask):
        rotations = np.random.randint(0, 4)
        flip = np.random.choice([True, False])
        
        state_aug = np.rot90(state, k=rotations, axes=(1, 2))
        
        policy_matrix = policy.reshape(9, 9)
        policy_matrix_aug = np.rot90(policy_matrix, k=rotations, axes=(0, 1))
        
        mask_matrix = mask.reshape(9, 9)
        mask_matrix_aug = np.rot90(mask_matrix, k=rotations, axes=(0, 1))
        
        if flip:
            state_aug = np.flip(state_aug, axis=2)
            policy_matrix_aug = np.flip(policy_matrix_aug, axis=1)
            mask_matrix_aug   = np.flip(mask_matrix_aug, axis=1)
            
        return state_aug.copy(), policy_matrix_aug.flatten().copy(), mask_matrix_aug.flatten().copy()