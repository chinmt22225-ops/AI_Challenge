"""
benchmark_vs.py
===============
So sánh 2 file weights với nhau trong N trận 4 người.
- Agent A (ckpt200)  chơi slot 0
- Agent B (day2_final) chơi slot 1
- Slot 2 & 3: Bot "tactical" làm nền (dùng load_baseline_agent từ train.py)

Kết quả tính theo:
  WIN   : Agent A về nhất so với B (điểm cao hơn hoặc sống sót)
  LOSE  : Agent A thua B
  DRAW  : Bằng điểm hoàn toàn

Usage:
  python benchmark_vs.py --a weights_v4_tactical_day2_ckpt200.pth ^
                         --b weights_v4_tactical_day2_final.pth ^
                         --episodes 500
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# Import từ codebase hiện có
sys.path.insert(0, str(Path(__file__).parent))
from agent import ActorCriticNet, AlphaBetaMasker, encode_obs
from train import GameEnv, load_baseline_agent

GRID_SIZE = 13


def load_net(path: str, device) -> ActorCriticNet:
    net = ActorCriticNet().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net


def pick_action(net, masker, obs, agent_id: int, step: int, device) -> int:
    """Chọn action greedy (argmax) cho 1 agent."""
    from agent import GameState
    map_feat, aux_feat = encode_obs(obs, agent_id, current_step=step)
    map_t = torch.FloatTensor(map_feat).unsqueeze(0).to(device)
    aux_t = torch.FloatTensor(aux_feat).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = net(map_t, aux_t)

    logits_np = logits.squeeze(0).cpu().numpy()

    # Dùng AlphaBetaMasker đúng cách qua GameState
    try:
        state = GameState(obs["map"], obs["players"], obs["bombs"], agent_id)
        mask = masker.get_safe_mask(state, agent_id)  # list[bool] len=6
        for i, allowed in enumerate(mask):
            if not allowed:
                logits_np[i] = -1e9
    except Exception:
        pass  # Nếu lỗi, bỏ qua masking

    return int(np.argmax(logits_np))


def score_player(player) -> float:
    """Điểm tổng hợp để phân định khi timeout."""
    return (
        player.kills * 10
        + player.boxes_broken * 2
        + player.items_collected * 3
        + (5 if player.alive else 0)
    )


def run_match(net_a, net_b, env: GameEnv, device, masker_a, masker_b,
              tactical_bots) -> dict:
    """
    Chạy 1 trận. Trả về dict kết quả cho slot 0 (A) và slot 1 (B).
    """
    obs = env.reset()
    done = False
    step = 0

    while not done:
        step += 1
        actions = []

        # Slot 0: Agent A
        if env.players[0].alive:
            actions.append(pick_action(net_a, masker_a, obs, 0, step, device))
        else:
            actions.append(0)

        # Slot 1: Agent B
        if env.players[1].alive:
            actions.append(pick_action(net_b, masker_b, obs, 1, step, device))
        else:
            actions.append(0)

        # Slot 2 & 3: Baseline bots
        for bot_idx, bot in enumerate(tactical_bots):
            pid = bot_idx + 2
            if env.players[pid].alive:
                try:
                    actions.append(bot.act(obs, pid))
                except TypeError:
                    actions.append(bot.act(obs))
            else:
                actions.append(0)

        obs, _, done, _ = env.step(actions)

    # --- Tính kết quả ---
    alive_ids = [i for i, p in enumerate(env.players) if p.alive]
    a_alive = env.players[0].alive
    b_alive = env.players[1].alive

    if a_alive and not b_alive:
        result_a, result_b = "WIN", "LOSE"
    elif b_alive and not a_alive:
        result_a, result_b = "LOSE", "WIN"
    elif a_alive and b_alive:
        # Cả 2 đều sống: so điểm
        sa = score_player(env.players[0])
        sb = score_player(env.players[1])
        if sa > sb:
            result_a, result_b = "WIN", "LOSE"
        elif sb > sa:
            result_a, result_b = "LOSE", "WIN"
        else:
            result_a, result_b = "DRAW", "DRAW"
    else:
        # Cả 2 đều chết: so điểm
        sa = score_player(env.players[0])
        sb = score_player(env.players[1])
        if sa > sb:
            result_a, result_b = "WIN", "LOSE"
        elif sb > sa:
            result_a, result_b = "LOSE", "WIN"
        else:
            result_a, result_b = "DRAW", "DRAW"

    return {
        "a": result_a,
        "b": result_b,
        "a_kills": env.players[0].kills,
        "b_kills": env.players[1].kills,
        "a_boxes": env.players[0].boxes_broken,
        "b_boxes": env.players[1].boxes_broken,
        "steps": step,
        "a_alive": a_alive,
        "b_alive": b_alive,
    }


def main():
    parser = argparse.ArgumentParser(description="Head-to-head benchmark")
    parser.add_argument("--a", required=True, help="Path to weights A (ckpt200)")
    parser.add_argument("--b", required=True, help="Path to weights B (day2_final)")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] Device: {device}")
    print(f"[Benchmark] Agent A: {args.a}")
    print(f"[Benchmark] Agent B: {args.b}")
    print(f"[Benchmark] Episodes: {args.episodes}")
    print("-" * 60)

    # Load nets
    net_a = load_net(args.a, device)
    net_b = load_net(args.b, device)
    masker_a = AlphaBetaMasker()
    masker_b = AlphaBetaMasker()

    env = GameEnv(seed=args.seed)

    # Tactical bots cho slot 2 & 3
    import importlib.util
    tactical_path = Path(__file__).parent / "Bomberland-GDGoC-AI-Challenge-main" / "Bomberland-GDGoC-AI-Challenge-main" / "agent" / "tactical_rule_agent.py"
    spec = importlib.util.spec_from_file_location("tactical_rule_agent", str(tactical_path))
    tactical_module = importlib.util.module_from_spec(spec)
    # Tạm thời gỡ 'agent' khỏi sys.modules để import nội bộ của tactical không bị dính
    old_agent = sys.modules.pop('agent', None)
    spec.loader.exec_module(tactical_module)
    if old_agent is not None:
        sys.modules['agent'] = old_agent
        
    tactical_bots = [tactical_module.TacticalRuleAgent(i + 2) for i in range(2)]
    print(f"[Benchmark] Slot 2&3: {type(tactical_bots[0]).__name__}")

    # --- Chạy N trận ---
    stats = defaultdict(int)
    kill_a_total = kill_b_total = box_a_total = box_b_total = 0
    steps_total = 0

    for ep in range(1, args.episodes + 1):
        result = run_match(net_a, net_b, env, device, masker_a, masker_b,
                           tactical_bots)
        stats[result["a"]] += 1
        kill_a_total  += result["a_kills"]
        kill_b_total  += result["b_kills"]
        box_a_total   += result["a_boxes"]
        box_b_total   += result["b_boxes"]
        steps_total   += result["steps"]

        if ep % 50 == 0:
            w  = stats["WIN"]
            l  = stats["LOSE"]
            d  = stats["DRAW"]
            wr = w / ep * 100
            print(f"[Ep {ep:>4}/{args.episodes}] "
                  f"A-Win={w} ({wr:.1f}%) | A-Lose={l} | Draw={d}")

    # --- Tổng kết ---
    n = args.episodes
    w  = stats["WIN"]
    l  = stats["LOSE"]
    d  = stats["DRAW"]
    print()
    print("=" * 60)
    print("KẾT QUẢ CUỐI CÙNG (Agent A vs Agent B)")
    print("=" * 60)
    print(f"  Agent A ({Path(args.a).name}):")
    print(f"    Thắng : {w:>4}  ({w/n*100:.1f}%)")
    print(f"    Thua  : {l:>4}  ({l/n*100:.1f}%)")
    print(f"    Hòa   : {d:>4}  ({d/n*100:.1f}%)")
    print()
    print(f"  Agent B ({Path(args.b).name}):")
    print(f"    Thắng : {l:>4}  ({l/n*100:.1f}%)")
    print(f"    Thua  : {w:>4}  ({w/n*100:.1f}%)")
    print(f"    Hòa   : {d:>4}  ({d/n*100:.1f}%)")
    print()
    print(f"  Thống kê trung bình / trận:")
    print(f"    Kills A / B  : {kill_a_total/n:.2f}  /  {kill_b_total/n:.2f}")
    print(f"    Boxes A / B  : {box_a_total/n:.2f}  /  {box_b_total/n:.2f}")
    print(f"    Độ dài trận  : {steps_total/n:.1f} bước")
    print("=" * 60)

    if w > l:
        print(f">>> Agent A (ckpt200 — bảo thủ) THẮNG CUỘC ({w/n*100:.1f}% vs {l/n*100:.1f}%)")
    elif l > w:
        print(f">>> Agent B (day2_final — khám phá) THẮNG CUỘC ({l/n*100:.1f}% vs {w/n*100:.1f}%)")
    else:
        print(">>> HAI AGENT NGANG NHAU (hòa)")


if __name__ == "__main__":
    main()
