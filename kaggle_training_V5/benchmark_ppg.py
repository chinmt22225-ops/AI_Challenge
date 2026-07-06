"""
benchmark_ppg.py
=================
Đánh giá sức mạnh của V4.5 PPG
- Slot 0: V4.5 (weights_ckpt2900_V4.5.pth)
- Slot 1: V4 (weights_v4_tactical_day2_final.pth)
- Slot 2: V4 (weights_v4_tactical_day2_final.pth)
- Slot 3: Bot Tactical của BTC
"""

import sys
from pathlib import Path
import numpy as np
import torch
import multiprocessing

sys.path.insert(0, str(Path(__file__).parent))
from agent import ActorCriticNet, AlphaBetaMasker, encode_obs, GameState
from train import GameEnv
from submission.agent import Agent as TacticalRuleAgent

def load_net(path: str, device) -> ActorCriticNet:
    net = ActorCriticNet().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(state, strict=False)  # strict=False để load bản cũ vào PPGNet
    net.eval()
    return net

def pick_action(net, masker, obs, agent_id: int, step: int, device) -> int:
    map_feat, aux_feat = encode_obs(obs, agent_id, current_step=step)
    map_t = torch.FloatTensor(map_feat).unsqueeze(0).to(device)
    aux_t = torch.FloatTensor(aux_feat).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _, _ = net.forward_aux(map_t, aux_t)  # PPGNetwork uses forward_aux or we can just call net() for PPO parts
    
    # Actually just call net() to get logits from ActorCriticNet base behavior
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

def run_match(seed) -> dict:
    device = torch.device("cpu") # Dùng CPU cho Multiprocessing an toàn
    
    nets = []
    paths = [
        "d:/AI_CHALLENGE/weights_ckpt2900_V4.5.pth",
        "d:/AI_CHALLENGE/weights_v4_tactical_day2_final.pth",
        "d:/AI_CHALLENGE/weights_v4_tactical_day2_final.pth"
    ]
    for p in paths:
        nets.append(load_net(p, device))
        
    tactical_bot = TacticalRuleAgent(agent_id=3)
    maskers = [AlphaBetaMasker() for _ in range(3)]
    
    env = GameEnv()
    obs = env.reset()
    done = False
    step = 0

    while not done:
        step += 1
        actions = []
        for i in range(4):
            if not env.players[i].alive:
                actions.append(0)
            elif i == 3:
                actions.append(tactical_bot.act(obs))
            else:
                actions.append(pick_action(nets[i], maskers[i], obs, i, step, device))

        obs, _, done, _ = env.step(actions)

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

def worker(args):
    seed, ep = args
    return run_match(seed)

def main():
    print("Bắt đầu 500 trận sinh tử kiểm chứng V4.5 PPG...")
    print("Slot 0: V4.5 (weights_ckpt2900_V4.5.pth)")
    print("Slot 1: V4 (weights_v4_tactical_day2_final.pth)")
    print("Slot 2: V4 (weights_v4_tactical_day2_final.pth)")
    print("Slot 3: Tactical Bot")
    print("========================================")
    
    episodes = 20
    wins = [0, 0, 0, 0]
    kills = [0, 0, 0, 0]
    boxes = [0, 0, 0, 0]
    survives = [0, 0, 0, 0]
    steps_total = 0

    for i in range(episodes):
        result = run_match(i)
        for w in result["winners"]:
            wins[w] += 1
            
        for p_i in range(4):
            kills[p_i] += result["kills"][p_i]
            boxes[p_i] += result["boxes"][p_i]
            if result["alive"][p_i]:
                survives[p_i] += 1
                
        steps_total += result["steps"]

        if (i + 1) % 1 == 0:
            print(f"[Trận {i+1:>3}/{episodes}] Thắng: {wins} | Sống: {survives}")

    print("\n" + "="*60)
    print("KẾT QUẢ V4.5 PPG vs V4 PPO vs TACTICAL (500 TRẬN)")
    print("="*60)
    names = ["V4.5 PPG (2900)", "V4 PPO", "V4 PPO", "Tactical Bot"]
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
