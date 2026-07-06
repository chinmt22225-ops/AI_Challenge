"""
benchmark_4way.py
=================
Trận tử chiến 4 người (Battle Royale) giữa 4 thế hệ Agent:
- Slot 0: V4 Khám Phá (weights_v4_tactical_day2_final.pth)
- Slot 1: V4 Bảo Thủ (weights_v4_tactical_day2_ckpt200.pth)
- Slot 2: V3 Thợ Săn (weights_hunter_v3_final.pth)
- Slot 3: V1 Sơ Khai (weights_final.pth) - Đại diện cho thế hệ sơ khai dùng AlphaBetaMasker.

Đếm: WIN, KILLS, BOXES, SURVIVE
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from agent import ActorCriticNet, AlphaBetaMasker, encode_obs, GameState
from train import GameEnv

def load_net(path: str, device) -> ActorCriticNet:
    net = ActorCriticNet().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net

def pick_action(net, masker, obs, agent_id: int, step: int, device) -> int:
    map_feat, aux_feat = encode_obs(obs, agent_id, current_step=step)
    map_t = torch.FloatTensor(map_feat).unsqueeze(0).to(device)
    aux_t = torch.FloatTensor(aux_feat).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = net(map_t, aux_t)

    logits_np = logits.squeeze(0).cpu().numpy()

    try:
        state = GameState(obs["map"], obs["players"], obs["bombs"], agent_id)
        mask = masker.get_safe_mask(state, agent_id)
        for i, allowed in enumerate(mask):
            if not allowed:
                logits_np[i] = -1e9
    except Exception:
        pass

    return int(np.argmax(logits_np))

def score_player(player) -> float:
    return (
        player.kills * 10
        + player.boxes_broken * 2
        + player.items_collected * 3
        + (5 if player.alive else 0)
    )

def run_match(nets, maskers, env: GameEnv, device) -> dict:
    obs = env.reset()
    done = False
    step = 0

    while not done:
        step += 1
        actions = []
        for i in range(4):
            if env.players[i].alive:
                actions.append(pick_action(nets[i], maskers[i], obs, i, step, device))
            else:
                actions.append(0)

        obs, _, done, _ = env.step(actions)

    # Calculate winners (can be multiple if tiebreaker is exact match)
    scores = [(score_player(p), p.id) for p in env.players]
    scores.sort(reverse=True, key=lambda x: x[0])
    
    highest_score = scores[0][0]
    winners = [pid for score, pid in scores if score == highest_score]

    return {
        "winners": winners,
        "kills": [p.kills for p in env.players],
        "boxes": [p.boxes_broken for p in env.players],
        "steps": step,
        "alive": [p.alive for p in env.players],
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark 4-Way] Device: {device}")
    
    paths = [
        "d:/AI_CHALLENGE/weights_v4_tactical_day2_final.pth",
        "d:/AI_CHALLENGE/weights_v4_tactical_day2_ckpt200.pth",
        "d:/AI_CHALLENGE/weights_hunter_v3_final.pth",
        "d:/AI_CHALLENGE/weights.pth"  # Sơ khai
    ]
    
    names = [
        "V4 Khám Phá (day2_final)",
        "V4 Bảo Thủ (ckpt200)",
        "V3 Thợ Săn (v3_final)",
        "V1 Sơ Khai (weights.pth)"
    ]

    nets = []
    for p in paths:
        try:
            nets.append(load_net(p, device))
        except Exception as e:
            print(f"Lỗi load model {p}: {e}")
            return
            
    maskers = [AlphaBetaMasker() for _ in range(4)]
    env = GameEnv()

    episodes = 500
    wins = [0, 0, 0, 0]
    kills = [0, 0, 0, 0]
    boxes = [0, 0, 0, 0]
    survives = [0, 0, 0, 0]
    steps_total = 0

    print("Bắt đầu 500 trận sinh tử 4 phe...")
    for ep in range(1, episodes + 1):
        result = run_match(nets, maskers, env, device)
        
        # Nếu có hòa (nhiều winner) thì chia đều điểm thắng, hoặc tính cho tất cả
        for w in result["winners"]:
            wins[w] += 1
            
        for i in range(4):
            kills[i] += result["kills"][i]
            boxes[i] += result["boxes"][i]
            if result["alive"][i]:
                survives[i] += 1
                
        steps_total += result["steps"]

        if ep % 50 == 0:
            print(f"[Ep {ep:>3}/{episodes}] Wins: {wins}")

    print("\n" + "="*60)
    print("KẾT QUẢ ĐẠI CHIẾN 4 THẾ HỆ (500 TRẬN)")
    print("="*60)
    for i in range(4):
        print(f"Slot {i}: {names[i]}")
        print(f"  - Tỷ lệ thắng : {wins[i]/episodes*100:>5.1f}% ({wins[i]} trận)")
        print(f"  - Tỷ lệ sống  : {survives[i]/episodes*100:>5.1f}% ({survives[i]} lần)")
        print(f"  - Kills/Trận  : {kills[i]/episodes:>5.2f}")
        print(f"  - Hộp/Trận    : {boxes[i]/episodes:>5.2f}")
        print("-" * 40)
    print(f"Trung bình bước / trận: {steps_total/episodes:.1f}")
    print("="*60)

if __name__ == "__main__":
    main()
