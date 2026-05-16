#!/usr/bin/env python3
"""
UTTT Benchmark — headless, no server required.
Runs all agent pair combinations directly using UTTTGame.

Agents:
    dummy        — random move
    minmax_d3    — MinMax depth 3
    minmax_d6    — MinMax depth 6
    pvn          — Policy-Value Network

For each unordered pair:
    • 100 games with agent A as P1
    • 100 games with agent B as P1
    Total: C(4,2) × 2 × 100 = 1 200 games

All configs run concurrently via asyncio.gather.
MinMax deliberation runs in a ThreadPoolExecutor so it doesn't block the event loop.

Output:
    benchmark_output/benchmark_results.json
    benchmark_output/benchmark_results.txt

Usage:
    Place this file in the same directory as ultimate_tic_tac_toe.py, OR
    set ROOT_DIR below to your project root. Then:

        python benchmark_uttt.py

    Optional env vars:
        BENCHMARK_GAMES=50          (default 100)
        BENCHMARK_MAX_CONCURRENT=8  (default: cpu_count)
"""

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm.asyncio import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Path setup — adjust ROOT_DIR if your project layout differs
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent          
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# Game engine
# ─────────────────────────────────────────────────────────────────────────────
from mcts.generate_Data.python.game.ultimate_tic_tac_toe import UTTTGame  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Result data-classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GameResult:
    winner:      int   # 1=P1 wins, 2=P2 wins, 3=draw
    p1_moves:    int   # moves made by P1
    p2_moves:    int   # moves made by P2

    def winner_moves(self) -> Optional[int]:
        """Moves made exclusively by the winning player, or None if draw."""
        if self.winner == 1:
            return self.p1_moves
        if self.winner == 2:
            return self.p2_moves
        return None


@dataclass
class ConfigResult:
    config_id: str
    p1_name:   str
    p2_name:   str
    p1_wins:   int = 0
    p2_wins:   int = 0
    draws:     int = 0
    results:   List[GameResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.p1_wins + self.p2_wins + self.draws

    def series_winner(self) -> Optional[Tuple[str, int]]:
        """
        Returns (winner_name, player_slot) for the agent that won more games,
        or None if tied (or all draws).
        player_slot is 1 or 2.
        """
        if self.p1_wins > self.p2_wins:
            return self.p1_name, 1
        elif self.p2_wins > self.p1_wins:
            return self.p2_name, 2
        return None   # tie or all draws

    def winner_move_stats(self) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[str]]:
        """
        Min / max / avg move-count computed ONLY over the games won by the
        series winner (the agent with more victories in the 100-game batch).

        Returns (min, max, avg, winner_name).
        All None if no decisive series winner.
        """
        sw = self.series_winner()
        if sw is None:
            return None, None, None, None

        winner_name, slot = sw
        counts = [
            r.winner_moves()
            for r in self.results
            if r.winner == slot and r.winner_moves() is not None
        ]
        if not counts:
            return None, None, None, None

        return min(counts), max(counts), sum(counts) / len(counts), winner_name


# ─────────────────────────────────────────────────────────────────────────────
# Inline agent classes 
# ─────────────────────────────────────────────────────────────────────────────

class DummyAgent:
    """Picks a move uniformly at random."""

    def __init__(self, player_id: int, **_):
        self.player_id = player_id

    async def deliberate(self, board, macro_board, active_macro, valid_actions):
        return random.choice(valid_actions) if valid_actions else None


class MinMaxAgentLocal:
    """Wraps MinMax.think() in a ThreadPoolExecutor so it doesn't block asyncio."""

    def __init__(self, player_id: int, max_depth: int,
                 executor: ThreadPoolExecutor, **_):
        self.player_id = player_id
        self.max_depth = max_depth
        self.executor  = executor

        from minmax.minmax import MinMax      
        self._MinMax = MinMax

    async def deliberate(self, board, macro_board, active_macro, valid_actions):
        state = {
            "board":        board,
            "macro_board":  macro_board,
            "active_macro": active_macro,
            "valid_actions": valid_actions,
        }
        md  = self.max_depth
        pid = self.player_id
        Cls = self._MinMax

        def _think():
            engine = Cls(state, max_depth=md, player_id=pid)
            move, _ = engine.think(state, md, pid)
            return move

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _think)


class PVNAgentLocal:
    """Policy-Value Network agent — model is loaded once and shared."""

    def __init__(self, player_id: int, model: Any, device: Any, **_):
        self.player_id = player_id
        self.model  = model
        self.device = device

    async def deliberate(self, board, macro_board, active_macro, valid_actions):
        import torch

        if not valid_actions:
            return None
        if len(valid_actions) == 1:
            return valid_actions[0]

        t = torch.zeros(1, 4, 9, 9, dtype=torch.float32, device=self.device)

        for y in range(9):
            for x in range(9):
                v = board[y][x]
                if v == self.player_id:
                    t[0, 0, y, x] = 1.0
                elif v != 0:
                    t[0, 1, y, x] = 1.0

        t[0, 2] = 1.0 if self.player_id == 1 else -1.0

        for action in valid_actions:
            ax, ay = int(action[0]), int(action[1])
            t[0, 3, ay, ax] = 1.0

        with torch.no_grad():
            policy_logits, _ = self.model(t)

        policy = policy_logits.squeeze(0).cpu()

        best, best_score = None, float("-inf")
        for action in valid_actions:
            ax, ay = int(action[0]), int(action[1])
            s = policy[ay * 9 + ax].item()
            if not math.isnan(s) and s > best_score:
                best_score, best = s, action

        return best if best is not None else random.choice(valid_actions)


# ─────────────────────────────────────────────────────────────────────────────
# PVN model loader 
# ─────────────────────────────────────────────────────────────────────────────

def load_pvn_model() -> Tuple[Optional[Any], Optional[Any]]:
    try:
        import torch
        from mcts.policy_value_network.pvn import PVN
    except ImportError as e:
        print(f"[PVN] Skipping — import failed: {e}")
        return None, None

    MODEL_REPO     = "Marinheiro2004/uttt_pvn"
    MODEL_FILENAME = "v2_10k_ES.pth"
    model_path = ROOT_DIR / "mcts" / "models" / MODEL_FILENAME
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILENAME}"
        print(f"[PVN] Downloading model → {url}")
        urllib.request.urlretrieve(url, model_path)
        print(f"[PVN] Saved to {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PVN(in_channels=4, out_channels=256).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"[PVN] Model ready on {device}")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# Single-game runner
# ─────────────────────────────────────────────────────────────────────────────

async def play_game(p1, p2) -> GameResult:
    game  = UTTTGame()
    moves = {1: 0, 2: 0}   # moves made by each player
    agents = {1: p1, 2: p2}

    while game.winner == 0:
        valid_objs = game.get_valid_actions()
        if not valid_objs:
            break

        valid  = [[a.x, a.y] for a in valid_objs]
        board  = game.get_state()
        macro  = [row[:] for row in game.macro_board]
        active = game.active_macro[:] if game.active_macro else None
        turn   = game.current_turn

        move = await agents[turn].deliberate(board, macro, active, valid)
        if move is None:
            break

        if not game.process_move(int(move[0]), int(move[1])):
            fb = random.choice(valid)
            game.process_move(fb[0], fb[1])

        moves[turn] += 1

    return GameResult(winner=game.winner, p1_moves=moves[1], p2_moves=moves[2])


# ─────────────────────────────────────────────────────────────────────────────
# Config runner — launches all N games for one matchup config concurrently
# ─────────────────────────────────────────────────────────────────────────────

async def run_config(
    cfg:        ConfigResult,
    p1_factory: Callable,
    p2_factory: Callable,
    num_games:  int,
    semaphore:  asyncio.Semaphore,
) -> ConfigResult:

    async def one_game(_: int) -> GameResult:
        async with semaphore:
            p1 = p1_factory(player_id=1)
            p2 = p2_factory(player_id=2)
            return await play_game(p1, p2)

    t0 = time.time()
    game_results = []
    tasks = [asyncio.create_task(one_game(i)) for i in range(num_games)]
    
    desc_str = f"▶ {cfg.config_id}"
    for fut in tqdm(asyncio.as_completed(tasks), total=num_games, desc=desc_str, leave=False):
        res = await fut
        game_results.append(res)

    for gr in game_results:
        cfg.results.append(gr)
        if   gr.winner == 1: cfg.p1_wins += 1
        elif gr.winner == 2: cfg.p2_wins += 1
        else:                cfg.draws   += 1

    mn, mx, avg, sw_name = cfg.winner_move_stats()
    avg_str = f"{avg:.1f}" if avg is not None else "N/A"
    sw_str  = sw_name if sw_name else "TIE"
    elapsed = time.time() - t0
    
    print(
        f"  ✓  {cfg.config_id}  |  "
        f"P1({cfg.p1_name})={cfg.p1_wins}  "
        f"P2({cfg.p2_name})={cfg.p2_wins}  "
        f"D={cfg.draws}  |  "
        f"series_winner={sw_str}  moves min={mn} max={mx} avg={avg_str}  |  "
        f"{elapsed:.0f}s"
    )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "N/A"


def write_output(all_results: List[ConfigResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    jdata = []
    for r in all_results:
        mn, mx, avg, sw_name = r.winner_move_stats()
        jdata.append({
            "config_id":  r.config_id,
            "p1":         r.p1_name,
            "p2":         r.p2_name,
            "total_games": r.total,
            "p1_wins":    r.p1_wins,
            "p2_wins":    r.p2_wins,
            "draws":      r.draws,
            "series_winner": sw_name,           # agent name, or null if tied
            "winner_move_stats": {              # stats ONLY for series-winner's victories
                "min": mn,
                "max": mx,
                "avg": round(avg, 2) if avg is not None else None,
            },
        })

    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(jdata, f, indent=2)

    SEP  = "─" * 78
    SEP2 = "═" * 78
    lines: List[str] = [
        SEP2,
        "  UTTT BENCHMARK RESULTS",
        SEP2,
        "",
    ]

    for r in all_results:
        mn, mx, avg, sw_name = r.winner_move_stats()
        avg_str  = f"{avg:.1f}" if avg is not None else "N/A"
        sw_str   = sw_name if sw_name else "TIE"
        sw_count = (r.p1_wins if sw_name == r.p1_name else
                    r.p2_wins if sw_name == r.p2_name else 0)
        lines += [
            SEP,
            f"  Config : {r.config_id}",
            SEP,
            f"  P1  {r.p1_name:<18}  wins : {r.p1_wins:>4}  ({_pct(r.p1_wins, r.total):>6})",
            f"  P2  {r.p2_name:<18}  wins : {r.p2_wins:>4}  ({_pct(r.p2_wins, r.total):>6})",
            f"  Draws               :       {r.draws:>4}  ({_pct(r.draws,   r.total):>6})",
            f"  Total games         :       {r.total:>4}",
            f"  Series winner       :  {sw_str}",
            f"  Move stats*         :  min={mn}  max={mx}  avg={avg_str}",
            f"  (* computed over the {sw_count} games won by {sw_str})",
            "",
        ]

    agent_stats: Dict[str, Dict[str, int]] = {}

    def _ensure(name: str):
        if name not in agent_stats:
            agent_stats[name] = {"wins": 0, "losses": 0, "draws": 0, "games": 0}

    for r in all_results:
        _ensure(r.p1_name)
        _ensure(r.p2_name)
        agent_stats[r.p1_name]["wins"]   += r.p1_wins
        agent_stats[r.p1_name]["losses"] += r.p2_wins
        agent_stats[r.p1_name]["draws"]  += r.draws
        agent_stats[r.p1_name]["games"]  += r.total
        agent_stats[r.p2_name]["wins"]   += r.p2_wins
        agent_stats[r.p2_name]["losses"] += r.p1_wins
        agent_stats[r.p2_name]["draws"]  += r.draws
        agent_stats[r.p2_name]["games"]  += r.total

    lines += [SEP2, "  AGGREGATE TOTALS  (across all configs)", SEP2, ""]
    col = f"  {'Agent':<18}  {'Games':>6}  {'Wins':>6}  {'Losses':>6}  {'Draws':>6}  {'Win%':>7}"
    lines.append(col)
    lines.append("  " + "─" * 58)
    for name, s in sorted(agent_stats.items()):
        wp = _pct(s["wins"], s["games"])
        lines.append(
            f"  {name:<18}  {s['games']:>6}  {s['wins']:>6}  "
            f"{s['losses']:>6}  {s['draws']:>6}  {wp:>7}"
        )
    lines += ["", SEP2]

    txt_path = out_dir / "benchmark_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Saved → {json_path}")
    print(f"  Saved → {txt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    NUM_GAMES      = args.games
    MAX_CONCURRENT = args.concurrent

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    executor  = ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT,
        thread_name_prefix="minmax",
    )

    print("=" * 60)
    print("  UTTT Benchmark starting")
    print(f"  Games per config    : {NUM_GAMES}")
    print(f"  Max concurrent games: {MAX_CONCURRENT}")
    print("=" * 60)

    # Load PVN once
    pvn_model, pvn_device = load_pvn_model()

    # ── Agent factory registry ─────────────────────────────────────────────
    registry: Dict[str, Callable[..., Any]] = {
        "dummy":     lambda player_id, **_: DummyAgent(player_id=player_id),
        "minmax_d3": lambda player_id, **_: MinMaxAgentLocal(
            player_id=player_id, max_depth=3, executor=executor),
        "minmax_d6": lambda player_id, **_: MinMaxAgentLocal(
            player_id=player_id, max_depth=6, executor=executor),
    }

    if pvn_model is not None:
        registry["pvn"] = lambda player_id, **_: PVNAgentLocal(
            player_id=player_id, model=pvn_model, device=pvn_device)
    else:
        print("[WARN] PVN not available — PVN matchups will be skipped.\n")

    names = list(registry.keys())

    configs: List[Tuple[ConfigResult, Callable, Callable]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            for p1, p2 in [(a, b), (b, a)]:
                cid = f"{p1}__vs__{p2}"
                cfg = ConfigResult(config_id=cid, p1_name=p1, p2_name=p2)
                configs.append((cfg, registry[p1], registry[p2]))

    total_games = len(configs) * NUM_GAMES
    print(f"\n  {len(configs)} configs × {NUM_GAMES} games = {total_games} total games\n")

    t0 = time.time()

    # Run ALL configs concurrently
    all_results: List[ConfigResult] = await asyncio.gather(*[
        run_config(cfg, f1, f2, NUM_GAMES, semaphore)
        for cfg, f1, f2 in configs
    ])

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  All done in {elapsed:.1f}s  ({elapsed / total_games:.2f}s/game avg)")
    print(f"{'=' * 60}")

    out_dir = SCRIPT_DIR / "benchmark_output"
    write_output(list(all_results), out_dir)

    executor.shutdown(wait=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UTTT Benchmark — headless, no server required.")
    parser.add_argument("--games", type=int, default=int(os.getenv("BENCHMARK_GAMES", "100")),
                        help="Number of games per config (default: 100)")
    parser.add_argument("--concurrent", type=int, 
                        default=int(os.getenv("BENCHMARK_MAX_CONCURRENT", str(max(4, (os.cpu_count() or 4))))),
                        help="Maximum concurrent games (default: CPU count or 4)")
    args = parser.parse_args()
    
    asyncio.run(main(args))
