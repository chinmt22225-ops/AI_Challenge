import sys
from pathlib import Path
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

def render_grid(grid, players, bombs):
    chars = []
    for r in range(13):
        row_chars = []
        for c in range(13):
            val = int(grid[r, c])
            if val == 1:
                row_chars.append("██")
            elif val == 2:
                row_chars.append("##")
            elif val == 3:
                row_chars.append(" R")
            elif val == 4:
                row_chars.append(" C")
            else:
                row_chars.append("  ")
        chars.append(row_chars)
        
    for b in bombs:
        chars[b.x][b.y] = "💣"
        
    for p in players:
        if p.alive:
            chars[p.x][p.y] = f"P{p.id}"
            
    lines = ["_" * 28]
    for row in chars:
        lines.append("|" + "".join(row) + "|")
    lines.append("-" * 28)
    return "\n".join(lines)

def run_exhibition():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    paths = [
        "d:/AI_CHALLENGE/weights_v4_tactical_day2_final.pth",
        "d:/AI_CHALLENGE/weights.pth",
        "d:/AI_CHALLENGE/weights_hunter_v3_final.pth",
        "d:/AI_CHALLENGE/weights_final.pth"
    ]
    
    names = [
        "V4 (Hoàn Thiện - Cân bằng)",
        "V1 (Bảo Thủ - Sợ chết)",
        "V3 (Liều Mạng - Khát máu)",
        "V2 (Cẩn Trọng - Farm chậm)"
    ]

    nets = [load_net(p, device) for p in paths]
    maskers = [AlphaBetaMasker() for _ in range(4)]
    env = GameEnv(seed=999) # Fixed seed to get a good action log

    obs = env.reset()
    done = False
    step = 0
    
    log = []
    log.append("# Tường thuật trực tiếp 1 trận đấu 4 Nhân Cách\n")
    log.append("### Danh sách tuyển thủ:")
    for i in range(4):
        log.append(f"- **P{i}**: {names[i]}")
        
    log.append("\n### Bắt đầu trận đấu (Mốc 0)\n")
    log.append("```text\n" + render_grid(env.grid, env.players, env.bombs) + "\n```\n")

    while not done:
        step += 1
        actions = []
        for i in range(4):
            if env.players[i].alive:
                actions.append(pick_action(nets[i], maskers[i], obs, i, step, device))
            else:
                actions.append(0)

        obs, _, done, _ = env.step(actions)
        
        # In ra các bước 50, 200, 350, và End
        if step in [50, 200, 350] or done:
            log.append(f"\n### Tình hình tại mốc {step} steps:")
            log.append(f"- P0 (V4 Hoàn Thiện) - Hộp: {env.players[0].boxes_broken}, Items: {env.players[0].items_collected}")
            log.append(f"- P1 (V1 Bảo Thủ)    - Hộp: {env.players[1].boxes_broken}, Items: {env.players[1].items_collected}")
            log.append(f"- P2 (V3 Liều Mạng)  - Hộp: {env.players[2].boxes_broken}, Items: {env.players[2].items_collected}")
            log.append(f"- P3 (V2 Cẩn Trọng)  - Hộp: {env.players[3].boxes_broken}, Items: {env.players[3].items_collected}")
            log.append("```text\n" + render_grid(env.grid, env.players, env.bombs) + "\n```\n")

    log.append("\n### KẾT QUẢ TIE-BREAKER (Step 500)\n")
    def score_player(player):
        return player.kills * 10 + player.boxes_broken * 2 + player.items_collected * 3 + (5 if player.alive else 0)
    scores = [(score_player(p), p.id) for p in env.players]
    scores.sort(reverse=True, key=lambda x: x[0])
    
    for score, pid in scores:
        log.append(f"- **Hạng {scores.index((score, pid))+1} - P{pid} ({names[pid]})**: Đạt {score} điểm")
        
    with open("d:/AI_CHALLENGE/match_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
        
    print("MATCH REPORT SAVED")

if __name__ == "__main__":
    run_exhibition()
