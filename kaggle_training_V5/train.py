# ============================================================
# Bomberland GDGoC AI Challenge 2026
# Training Script: Imitation Learning → PPO Fine-tuning
# KHÔNG NỘPFILE NÀY — chỉ dùng để train trên Kaggle/local
#
# Usage:
#   python train.py --phase il_collect --episodes 5000 --output demos.pkl
#   python train.py --phase il_train   --demos demos.pkl --output weights_il.pth
#   python train.py --phase ppo        --load weights_il.pth --output weights_final.pth
#   python train.py --phase benchmark  --load weights_final.pth --episodes 100
# ============================================================

import sys
import os
import time
import random
import pickle
import argparse
import math
from pathlib import Path
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import từ agent.py (phải cùng thư mục)
sys.path.insert(0, str(Path(__file__).parent))
from agent import (
    GameState, ActorCriticNet, AlphaBetaMasker, encode_obs,
    ACTION_STOP, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_BOMB,
    DIR_MAP, GRASS, WALL, BOX, ITEM_RAD, ITEM_CAP,
    BOMB_TIMER, MAX_RADIUS, MAX_CAPACITY, GRID_SIZE,
)


# ============================================================
# SECTION A: DATA CLASSES (PlayerData, BombData)
# ============================================================

class PlayerData:
    """Đại diện cho 1 agent trong GameEnv."""

    def __init__(self, pid: int, row: int, col: int):
        self.id   = pid
        self.x    = row
        self.y    = col
        self.alive = True

        # Bomb stats
        self.bombs_left        = 1   # Số bom có thể đặt ngay
        self.bomb_radius_bonus = 0   # radius thực = 1 + bonus; max bonus = 4
        self.max_capacity      = 1   # capacity tối đa (tăng khi nhặt ITEM_CAP)

        # Stats for tiebreaker & reward tracking
        self.kills          = 0
        self.boxes_broken   = 0
        self.items_collected = 0
        self.bombs_placed   = 0

    def to_obs_array(self) -> list:
        """Chuyển sang format obs['players'][i] = [x, y, alive, bombs_left, radius_bonus]."""
        return [self.x, self.y, int(self.alive), self.bombs_left, self.bomb_radius_bonus]

    @property
    def radius(self) -> int:
        return 1 + self.bomb_radius_bonus


class BombData:
    """Đại diện cho 1 quả bom trên bản đồ."""

    def __init__(self, x: int, y: int, owner_id: int, radius: int, timer: int = BOMB_TIMER):
        self.x        = x
        self.y        = y
        self.owner_id = owner_id
        self.radius   = radius
        self.timer    = timer

    def to_obs_array(self) -> list:
        """Format obs['bombs'][i] = [x, y, timer, owner_id]."""
        return [self.x, self.y, self.timer, self.owner_id]


# ============================================================
# SECTION B: GAME ENVIRONMENT (Full 4-agent simulation)
# ============================================================

class GameEnv:
    """
    Mô phỏng đầy đủ engine Bomberland cho 4 agents.
    Độ trung thực ≥ 95% với engine BTC.

    Resolution order (đúng theo BTC game.py):
      1. Collect Actions
      2. Process Movement (đồng thời — item contention tại đây)
      3. Place Bombs (conflict resolution theo radius và ID)
      4. Decrease Bomb Timers
      5. Resolve Explosions (bao gồm chain reaction)
      6. Remove dead agents (sau khi tất cả bom đã nổ)
      7. Spawn Items (auto-spawn theo xác suất step-based)
      8. Check End Conditions
    """

    # Vị trí xuất phát cố định theo engine BTC
    STARTING_POSITIONS = [
        (1, 1),    # Agent 0: top-left
        (11, 11),  # Agent 1: bottom-right
        (1, 11),   # Agent 2: top-right
        (11, 1),   # Agent 3: bottom-left
    ]

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.grid    = None
        self.players = []
        self.bombs   = []
        self.current_step = 0
        self.reset(seed=seed)

    def reset(self, seed: int = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.grid    = self._generate_map()

        # Xáo trộn vị trí xuất phát mỗi episode:
        # Agent của tôi sẽ xuất hiện ngẫu nhiên ở 1 trong 4 góc bản đồ
        # → CNN phải học cách xử lý mọi góc nhìn, tránh học tủ góc cố định
        shuffled_positions = list(self.STARTING_POSITIONS)
        self.rng.shuffle(shuffled_positions)

        self.players = [
            PlayerData(i, r, c)
            for i, (r, c) in enumerate(shuffled_positions)
        ]
        self.bombs = []
        return self._get_obs()

    # ── Map generation ────────────────────────────────────────────────

    def _generate_map(self) -> np.ndarray:
        """
        Tạo bản đồ 13×13 ngẫu nhiên:
          - Border walls (hàng/cột 0 và 12)
          - Checkerboard walls tại (row chẵn, col chẵn) trong vùng play (1-11)
          - Vùng 2×2 quanh 4 góc luôn là GRASS (safe zone)
          - Phần còn lại: 30% BOX, 5% WALL thêm, 65% GRASS
        """
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

        # Border walls cố định
        grid[0, :]  = WALL
        grid[-1, :] = WALL
        grid[:, 0]  = WALL
        grid[:, -1] = WALL

        # Checkerboard walls (tường cố định trong engine BTC)
        for r in range(2, 12, 2):   # row chẵn: 2,4,6,8,10
            for c in range(2, 12, 2):  # col chẵn: 2,4,6,8,10
                grid[r, c] = WALL

        # Safe zones quanh 4 góc (đảm bảo agent không bị kẹt ngay đầu)
        safe = set()
        for sr, sc in self.STARTING_POSITIONS:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    rr, cc = sr + dr, sc + dc
                    if 1 <= rr <= 11 and 1 <= cc <= 11:
                        safe.add((rr, cc))

        # Random fill các ô còn lại
        for r in range(1, 12):
            for c in range(1, 12):
                if grid[r, c] == WALL:
                    continue
                if (r, c) in safe:
                    continue
                roll = self.rng.random()
                if roll < 0.30:
                    grid[r, c] = BOX
                elif roll < 0.35:
                    grid[r, c] = WALL

        return grid

    # ── Observation ───────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        """Trả về observation theo format engine BTC (full observability)."""
        players_arr = np.array(
            [p.to_obs_array() for p in self.players], dtype=np.int8
        )
        bombs_arr = (
            np.array([b.to_obs_array() for b in self.bombs], dtype=np.int8)
            if self.bombs
            else np.zeros((0, 4), dtype=np.int8)
        )
        return {
            "map":     self.grid.copy(),
            "players": players_arr,
            "bombs":   bombs_arr,
            "step":    self.current_step,
        }

    # ── Main step function ────────────────────────────────────────────

    def step(self, actions: list) -> tuple:
        """
        Thực hiện 1 game step với actions của 4 agents.

        Args:
            actions: list[int] len=4, actions[i] là action của agent i

        Returns:
            obs:     dict — observation sau step
            rewards: list[float] len=4 — reward cho mỗi agent step này
            done:    bool — game kết thúc?
            info:    dict — thống kê step (kills, boxes, items, ...)
        """
        self.current_step += 1
        rewards = [0.0] * 4
        info = {
            "kills":          [0] * 4,
            "boxes_broken":   [0] * 4,
            "items_got":      [0] * 4,
            "item_contention":[0] * 4,
            "bombs_placed":   [0] * 4,
            "dead_this_step": [],
        }

        # ── Bước 2: Process Movement (đồng thời) ──────────────────────
        bomb_positions_before = {(b.x, b.y) for b in self.bombs}

        # Tính khoảng cách Dijkstra TỪ VỊ TRÍ CŨ của mỗi agent
        # (để so sánh trước/sau di chuyển cho Hunting Reward & Coward Penalty)
        old_dijkstra = {}
        for i, player in enumerate(self.players):
            if player.alive:
                old_dijkstra[i] = self._get_dijkstra_distances(
                    (player.x, player.y), bomb_positions_before
                )

        new_positions = {}   # agent_id → (new_x, new_y)

        for i, player in enumerate(self.players):
            if not player.alive:
                new_positions[i] = (player.x, player.y)
                continue

            action = int(actions[i])
            if action in DIR_MAP and action != ACTION_STOP:
                dx, dy = DIR_MAP[action]
                nx, ny = player.x + dx, player.y + dy
                if self._can_move_to(nx, ny, bomb_positions_before):
                    new_positions[i] = (nx, ny)
                else:
                    new_positions[i] = (player.x, player.y)
            else:
                new_positions[i] = (player.x, player.y)

        # Áp dụng di chuyển mới cho tất cả cùng lúc
        for i, player in enumerate(self.players):
            if player.alive:
                player.x, player.y = new_positions[i]

        # ── Item Contention Logic ──────────────────────────────────────
        # Sau khi TẤT CẢ agents đã di chuyển → kiểm tra ai bước vào ô item
        # Rule: ≥2 agents cùng bước vào cùng ô item trong cùng 1 step
        #       → item bị HỦY, không ai nhận (triệt tiêu)
        item_claimants = {}  # {(row,col): [agent_ids]}
        for i, player in enumerate(self.players):
            if not player.alive:
                continue
            cell = int(self.grid[player.x, player.y])
            if cell in (ITEM_RAD, ITEM_CAP):
                pos = (player.x, player.y)
                item_claimants.setdefault(pos, []).append(i)

        # Đếm số agent còn sống TẠI THỜI ĐIỂM NÀY để xác định late-game
        alive_count_for_items = sum(1 for p in self.players if p.alive)

        for pos, claimants in item_claimants.items():
            cell = int(self.grid[pos[0], pos[1]])
            if len(claimants) == 1:
                # Chỉ 1 agent → nhận item bình thường
                pid    = claimants[0]
                player = self.players[pid]
                got_item = False
                if cell == ITEM_RAD and player.bomb_radius_bonus < MAX_RADIUS - 1:
                    player.bomb_radius_bonus += 1
                    got_item = True
                elif cell == ITEM_CAP and player.bombs_left < MAX_CAPACITY:
                    # Tăng capacity: thêm 1 vào bombs_left (đơn giản hóa)
                    player.bombs_left  += 1
                    player.max_capacity = min(player.max_capacity + 1, MAX_CAPACITY)
                    got_item = True

                if got_item:
                    # Item Soft-cap: Nếu đủ đồ (Radius >= 3, Capacity >= 3), giảm cực mạnh reward để ép đi săn
                    if player.bomb_radius_bonus >= 2 and player.max_capacity >= 3:
                        item_reward = 0.2
                    else:
                        # Late-game (còn 2 người): Item kém quan trọng hơn, ưu tiên giết địch
                        item_reward = (RewardConfig.ITEM_PICKUP_LATE
                                       if alive_count_for_items <= 2
                                       else RewardConfig.ITEM_PICKUP)
                    rewards[pid] += item_reward
                    info["items_got"][pid] += 1
                    player.items_collected += 1
                # Xóa item khỏi bản đồ dù nhận được hay không (đã MAX)
                self.grid[pos[0], pos[1]] = GRASS

            else:
                # ≥2 agents tranh giành → item bị HỦY
                # KHÔNG ai nhận; phạt tất cả agents tham gia tranh giành
                self.grid[pos[0], pos[1]] = GRASS
                for pid in claimants:
                    rewards[pid] += RewardConfig.ITEM_CONTENTION
                    info["item_contention"][pid] += 1

        # ── Bước 3: Place Bombs (conflict resolution) ─────────────────
        # Engine xử lý theo thứ tự agent ID 0→3
        # Nếu 2+ agents đặt bom cùng ô:
        #   - Radius lớn hơn → thắng
        #   - Radius bằng nhau → agent ID nhỏ hơn giữ (xử lý trước)
        #   - Chỉ owner bom giữ được mới bị trừ bombs_left
        pending_bombs = {}   # {(x,y): BombData}
        smart_bomb_agents: set = set()  # Fix 1: agents nhận BOMB_SMART, miễn SPAM

        # Tính khoảng cách Manhattan từ mỗi agent tới đối thủ gần nhất
        # (dùng cho Anti-Kamikaze check — nhanh, không tốn thời gian)
        def _manhattan_min_enemy_dist(agent_id: int) -> int:
            p = self.players[agent_id]
            min_d = 999
            for j, ep in enumerate(self.players):
                if j != agent_id and ep.alive:
                    min_d = min(min_d, abs(p.x - ep.x) + abs(p.y - ep.y))
            return min_d

        for i in range(4):   # Thứ tự ID 0→3 như engine BTC
            player = self.players[i]
            if not player.alive or int(actions[i]) != ACTION_BOMB:
                continue
            if player.bombs_left <= 0:
                continue
            px, py = player.x, player.y
            # Không đặt nếu ô đã có bom từ bước trước
            if (px, py) in bomb_positions_before:
                continue

            new_radius = player.radius
            new_bomb   = BombData(px, py, i, new_radius)

            # ── Anti-Kamikaze + Smart Bombing Check ──────────────────
            # Chỉ kích hoạt khi địch trong vòng Manhattan <= 3 ô
            man_dist = _manhattan_min_enemy_dist(i)
            if man_dist <= 3:
                has_escape = self._can_escape_from_bomb(
                    agent_pos   = (px, py),
                    bomb_pos    = (px, py),
                    bomb_radius = new_radius,
                    bomb_positions = bomb_positions_before,
                )
                if has_escape:
                    rewards[i] += RewardConfig.BOMB_SMART    # +50: Ép góc thông minh!
                    smart_bomb_agents.add(i)  # Fix 1: đánh dấu miễn SPAM
                else:
                    rewards[i] += RewardConfig.BOMB_KAMIKAZE  # -500: Ôm bom tự sát!

            # ── Tactical: BOMB_AGGRESSIVE & THIRD_PARTY ──────────────
            third_party_awarded = False
            for j, ep in enumerate(self.players):
                if j == i or not ep.alive:
                    continue
                dist_to_enemy = abs(px - ep.x) + abs(py - ep.y)
                
                # BOMB_AGGRESSIVE: Địch trong tầm 5 ô
                if dist_to_enemy <= 5:
                    old_escapes = 0
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nx, ny = ep.x + dx, ep.y + dy
                        if self._can_move_to(nx, ny, bomb_positions_before):
                            old_escapes += 1
                            
                    new_escapes = 0
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nx, ny = ep.x + dx, ep.y + dy
                        if (nx, ny) == (px, py):
                            continue
                        if self._can_move_to(nx, ny, bomb_positions_before):
                            # Xấp xỉ blast zone (nếu cùng trục và trong bán kính)
                            is_blasted = False
                            if nx == px and abs(ny - py) <= new_radius:
                                is_blasted = True
                            if ny == py and abs(nx - px) <= new_radius:
                                is_blasted = True
                            if not is_blasted:
                                new_escapes += 1
                                
                    if old_escapes >= 2 and new_escapes <= 1:
                        rewards[i] += RewardConfig.BOMB_AGGRESSIVE
                        
                # THIRD_PARTY: Kẻ cơ hội (địch trong tầm 3 ô và đang đứng gần địch khác)
                if not third_party_awarded and dist_to_enemy <= 3:
                    for k, ep2 in enumerate(self.players):
                        if k != i and k != j and ep2.alive:
                            if abs(ep.x - ep2.x) + abs(ep.y - ep2.y) <= 3:
                                rewards[i] += RewardConfig.THIRD_PARTY
                                third_party_awarded = True
                                break

            if (px, py) in pending_bombs:
                existing = pending_bombs[(px, py)]
                if new_radius > existing.radius:
                    # Bom mới thắng: hoàn trả bombs_left cho owner cũ
                    self.players[existing.owner_id].bombs_left += 1
                    pending_bombs[(px, py)] = new_bomb
                    player.bombs_left  -= 1
                    player.bombs_placed += 1
                    info["bombs_placed"][i] += 1
                # else: bom cũ thắng, agent mới KHÔNG bị trừ bombs_left
            else:
                pending_bombs[(px, py)] = new_bomb
                player.bombs_left  -= 1
                player.bombs_placed += 1
                info["bombs_placed"][i] += 1

        self.bombs.extend(pending_bombs.values())
        # Fix 2: Lưu vị trí các quả bom vừa đặt trong step này
        newly_placed_positions: set = {(b.x, b.y) for b in pending_bombs.values()}

        # ── Bước 4: Decrease Bomb Timers ──────────────────────────────
        for b in self.bombs:
            b.timer -= 1

        # ── Bước 5: Resolve Explosions (Chain Reaction) ───────────────
        # Tất cả bom có timer ≤ 0 nổ đồng loạt;
        # Nếu sóng nổ chạm bom khác → bom đó nổ ngay trong cùng step (chain reaction)
        to_explode = [b for b in self.bombs if b.timer <= 0]
        self.bombs  = [b for b in self.bombs if b.timer > 0]

        # Dùng set để tránh chain loop vô hạn
        exploded_ids = set()
        explode_queue = list(to_explode)

        all_blast_tiles  = set()    # Toàn bộ ô bị ảnh hưởng step này
        boxes_destroyed  = set()    # Ô box bị phá

        while explode_queue:
            bomb = explode_queue.pop(0)
            bid  = id(bomb)
            if bid in exploded_ids:
                continue
            exploded_ids.add(bid)

            bx, by = bomb.x, bomb.y
            br     = bomb.radius

            # Tính blast tiles
            blast = {(bx, by)}
            for dx, dy in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                for r in range(1, br + 1):
                    nx, ny = bx + dx * r, by + dy * r
                    if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE): break
                    cell = int(self.grid[nx, ny])
                    if cell == WALL: break
                    blast.add((nx, ny))
                    if cell == BOX:
                        boxes_destroyed.add((nx, ny))
                        break

            all_blast_tiles |= blast

            # Chain reaction: bom còn sống bị dính sóng nổ → nổ ngay
            still_alive_bombs = []
            for other_b in self.bombs:
                if (other_b.x, other_b.y) in blast and id(other_b) not in exploded_ids:
                    explode_queue.append(other_b)
                else:
                    still_alive_bombs.append(other_b)
            self.bombs = still_alive_bombs

            # Hoàn trả bombs_left cho owner (bom vừa nổ)
            if 0 <= bomb.owner_id < 4:
                self.players[bomb.owner_id].bombs_left += 1

        # Xử lý box bị phá
        for (bx, by) in boxes_destroyed:
            if int(self.grid[bx, by]) == BOX:    # Chưa bị phá bởi chain
                self.grid[bx, by] = GRASS
                # Tìm owner của bom gây ra (đơn giản: owner bom đầu tiên nổ ô đó)
                # Trong thực tế khó xác định khi chain reaction → gán cho explosion chung
                # Để đơn giản: reward boxes cho tất cả owners có bomb tham gia
                # (Có thể tinh chỉnh sau)

                # Drop item (30% radius, 30% capacity, 40% nothing)
                roll = self.rng.random()
                if roll < 0.30:
                    self.grid[bx, by] = ITEM_RAD
                elif roll < 0.60:
                    self.grid[bx, by] = ITEM_CAP

        # Tính reward boxes_broken (gán cho owner bom đầu tiên đã nổ — simplified)
        # Lưu owner từ danh sách to_explode ban đầu
        for b in to_explode:
            oid = b.owner_id
            if not (0 <= oid < 4): continue
            bx, by = b.x, b.y
            br = b.radius
            for dx, dy in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                for r in range(1, br + 1):
                    nx, ny = bx + dx * r, by + dy * r
                    if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE): break
                    if int(self.grid[nx, ny]) == WALL: break
                    if (nx, ny) in boxes_destroyed:
                        self.players[oid].boxes_broken += 1
                        info["boxes_broken"][oid] += 1
                        
                        # Phase shifting: Late game (≤2 players) x3 box value
                        prev_alive_count = sum(p.alive for p in self.players)
                        box_reward = (RewardConfig.BOX_BROKEN_LATE 
                                      if prev_alive_count <= 2 
                                      else RewardConfig.BOX_BROKEN)
                        rewards[oid] += box_reward
                        
                        boxes_destroyed.discard((nx, ny))   # Chỉ tính 1 lần
                        break

        # ── Bước 6: Remove dead agents ────────────────────────────────
        prev_alive = [p.alive for p in self.players]
        prev_alive_count = sum(prev_alive)
        for i, player in enumerate(self.players):
            if player.alive and (player.x, player.y) in all_blast_tiles:
                player.alive = False
                info["dead_this_step"].append(i)
                # Phase shifting: Chết ở 1v1 là tội ác
                death_penalty = (RewardConfig.DEATH_LATE 
                                 if prev_alive_count <= 2 
                                 else RewardConfig.DEATH)
                rewards[i] += death_penalty

        # Kill reward: nếu agent i chết do bom của owner o → o được kill reward
        # (Dùng thông tin to_explode để biết bomb → owner)
        for b in to_explode:
            oid = b.owner_id
            if not (0 <= oid < 4): continue
            bx, by = b.x, b.y
            br = b.radius
            # Tính blast lại (nhỏ gọn)
            blast = {(bx, by)}
            for dx, dy in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                for r in range(1, br + 1):
                    nx, ny = bx + dx * r, by + dy * r
                    if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE): break
                    if int(self.grid[nx, ny]) == WALL: break
                    blast.add((nx, ny))
                    if int(self.grid[nx, ny]) == BOX: break
            for i, (was_alive, player) in enumerate(zip(prev_alive, self.players)):
                if was_alive and not player.alive and i != oid:
                    if (player.x, player.y) in blast:
                        # Phase shifting: Late game (1v1) kill reward giảm đi để tăng focus vào sinh tồn
                        kill_reward = (RewardConfig.KILL_LATE 
                                       if prev_alive_count <= 2 
                                       else RewardConfig.KILL)
                        rewards[oid] += kill_reward
                        self.players[oid].kills += 1
                        info["kills"][oid] += 1

        # ── Bước 7: Auto-spawn Items ───────────────────────────────────
        # P = 0.0003 × (current_step / 165) — tăng dần theo thời gian
        # Chỉ spawn trên ô cỏ trống (không có agent)
        spawn_prob = 0.0003 * (self.current_step / 165.0)
        agent_positions = {(p.x, p.y) for p in self.players if p.alive}
        for r in range(1, 12):
            for c in range(1, 12):
                if (int(self.grid[r, c]) == GRASS
                        and (r, c) not in agent_positions):
                    if self.rng.random() < spawn_prob:
                        self.grid[r, c] = (
                            ITEM_RAD if self.rng.random() < 0.5 else ITEM_CAP
                        )

        # ── Bước 8: Check End Conditions ──────────────────────────────
        alive_count = sum(1 for p in self.players if p.alive)
        done = (alive_count <= 1) or (self.current_step >= 500)

        # ── Hunting Reward, Coward Penalty, Time Penalty ───────────────
        # Tính danger zones sau khi tất cả bom đã giảm timer và nổ
        current_bomb_positions = {(b.x, b.y) for b in self.bombs}
        # Danger = ô nào bị các quả bom còn lại với timer <= 2 chiếu vào
        danger_zones: set = set()
        for b in self.bombs:
            if b.timer <= 2:
                danger_zones.add((b.x, b.y))
                bx, by = b.x, b.y
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    for step_r in range(1, b.radius + 1):
                        nx, ny = bx + dr * step_r, by + dc * step_r
                        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                            break
                        cell = int(self.grid[nx, ny])
                        if cell == WALL:
                            break
                        danger_zones.add((nx, ny))
                        if cell == BOX:
                            break

        for i, player in enumerate(self.players):
            if not player.alive:
                continue

            # 1. Hình phạt thời gian (thay Survive +0.1)
            # Phase shifting: Late game tắt phạt thời gian để câu giờ thoải mái
            survive_penalty = (RewardConfig.SURVIVE_STEP_LATE 
                               if alive_count <= 2 
                               else RewardConfig.SURVIVE_STEP)
            rewards[i] += survive_penalty

            # 2. Hunting Reward & Late-Game Coward Penalty
            if alive_count >= 2 and i in old_dijkstra:
                # Fix 2: Loại bỏ bom vừa đặt step này khỏi bản đồ Dijkstra
                # (tránh tăng khoảng cách ảo do bom chặn đường → phạt oan COWARD)
                dists_bomb_excl = current_bomb_positions - newly_placed_positions
                new_dists = self._get_dijkstra_distances(
                    (player.x, player.y), dists_bomb_excl
                )
                # Khoảng cách Dijkstra tới đối thủ gần nhất (TRƯỚC di chuyển)
                closest_enemy_id = -1
                old_min = 999
                for j, ep in enumerate(self.players):
                    if j != i and ep.alive:
                        d = old_dijkstra[i].get((ep.x, ep.y), 999)
                        if d < old_min:
                            old_min = d
                            closest_enemy_id = j

                if closest_enemy_id != -1:
                    ep = self.players[closest_enemy_id]
                    new_dist = new_dists.get((ep.x, ep.y), 999)
                    
                    my_power = player.bomb_radius_bonus + player.max_capacity
                    enemy_power = ep.bomb_radius_bonus + ep.max_capacity
                    
                    pos_now = (player.x, player.y)
                    if pos_now not in danger_zones:
                        if my_power >= enemy_power:
                            # Địch yếu hơn hoặc bằng -> Săn lùng
                            if new_dist < old_min:
                                rewards[i] += RewardConfig.HUNTING_ADVANCE
                        else:
                            # Địch mạnh hơn -> Rút lui
                            if new_dist > old_min:
                                rewards[i] += RewardConfig.RETREAT_REWARD

                    if alive_count == 2 and new_dist > old_min:
                        # Tử chiến cuối game: phạt bỏ chạy nếu đồ >= địch
                        if my_power >= enemy_power:
                            rewards[i] += RewardConfig.COWARD_PENALTY

        # ── Spam bom penalty (chỉ khi KHÔNG phải BOMB_SMART) ──────────────
        # Fix 1: Nếu BOMB_SMART được thưởng (+50), không phạt thêm SPAM (-5)
        # Điều này giữ phần thưởng trọn vẹn và không tạo ra tín hiệu mâu thuẫn
        if self.current_step <= 450:
            for i, player in enumerate(self.players):
                if player.alive and int(actions[i]) == ACTION_BOMB:
                    if i not in smart_bomb_agents:   # Miễn SPAM cho ép góc thông minh
                        rewards[i] += RewardConfig.BOMB_SPAM

        if done:
            terminal = self._calculate_terminal_rewards()
            for i in range(4):
                rewards[i] += terminal[i]

        return self._get_obs(), rewards, done, info

    def _can_move_to(self, x: int, y: int, bomb_positions: set) -> bool:
        """Kiểm tra agent có thể di chuyển vào ô (x,y) không."""
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return False
        cell = int(self.grid[x, y])
        if cell in (WALL, BOX):
            return False
        # Bom đang nằm trên sàn từ bước trước chặn di chuyển
        if (x, y) in bomb_positions:
            return False
        return True

    def _get_dijkstra_distances(self, start: tuple, bomb_positions: set) -> dict:
        """
        Dijkstra từ start_pos đến mọi ô trên bản đồ 13x13.
        - Wall  (1): Vật cản tuyệt đối, không đi qua.
        - Box   (2): Cost = 3 (tốn công phá, nhưng Agent vẫn "thấy" địch qua hộp).
        - Grass / Items (0,3,4): Cost = 1.
        - Bom đang nằm: Coi như tường trong ngắn hạn.
        Tối ưu cho 13x13: mỗi call < 0.5ms trên CPU.
        """
        import heapq
        dist = {start: 0}
        heap = [(0, start[0], start[1])]
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist.get((r, c), float('inf')):
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                    continue
                if (nr, nc) in bomb_positions:
                    continue
                cell = int(self.grid[nr, nc])
                if cell == WALL:
                    continue
                cost = 3 if cell == BOX else 1
                nd = d + cost
                if nd < dist.get((nr, nc), float('inf')):
                    dist[(nr, nc)] = nd
                    heapq.heappush(heap, (nd, nr, nc))
        return dist

    def _can_escape_from_bomb(self, agent_pos: tuple, bomb_pos: tuple,
                               bomb_radius: int, bomb_positions: set) -> bool:
        """
        BFS thuần, depth tối đa = 7 (đúng bằng timer bom).
        Kiểm tra agent có thể thoát khỏi vùng bán kính nổ của quả bom vừa đặt không.
        Tính blast zone của quả bom tại bomb_pos với bomb_radius, rồi BFS xem
        agent có thể đến ô nào NGOÀI blast zone trong <= 7 bước không.
        Tốc độ: BFS 13x13 depth 7 ≈ 0.3–1ms/call.
        """
        # Tính blast zone của quả bom vừa đặt
        blast = {bomb_pos}
        bx, by = bomb_pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for r in range(1, bomb_radius + 1):
                nx, ny = bx + dr * r, by + dc * r
                if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                    break
                cell = int(self.grid[nx, ny])
                if cell == WALL:
                    break
                blast.add((nx, ny))
                if cell == BOX:
                    break  # Hộp chặn sóng nổ

        # Bom vừa đặt cũng là vật cản di chuyển từ bước sau
        all_bombs = bomb_positions | {bomb_pos}

        # BFS với giới hạn depth = 7
        from collections import deque
        queue = deque([(agent_pos, 0)])
        visited = {agent_pos}
        while queue:
            pos, depth = queue.popleft()
            if depth >= 7:
                continue
            r, c = pos
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                    continue
                if (nr, nc) in visited:
                    continue
                if (nr, nc) in all_bombs:
                    continue
                cell = int(self.grid[nr, nc])
                if cell in (WALL, BOX):
                    continue
                if (nr, nc) not in blast:
                    return True  # Tìm được ô thoát!
                visited.add((nr, nc))
                queue.append(((nr, nc), depth + 1))
        return False  # Không có đường thoát

    def _calculate_terminal_rewards(self) -> list:
        """Phần thưởng cuối game (thắng/thua/tiebreaker)."""
        rewards = [0.0] * 4
        alive   = [p for p in self.players if p.alive]

        if len(alive) == 1:
            # Thắng outright — winner nhận WIN reward
            rewards[alive[0].id] = RewardConfig.WIN
        elif len(alive) == 0:
            pass  # Tất cả chết → không ai thắng
        else:
            # Step 500 — tiebreaker
            ranks = calculate_tiebreaker_winner(self.players)
            for pid, rank in ranks.items():
                if rank == 1:
                    rewards[pid] = RewardConfig.WIN
                elif rank == 2:
                    rewards[pid] = RewardConfig.TIE_LOSS
                else:
                    rewards[pid] = RewardConfig.TIE_LOSS * 1.5

        return rewards


# ============================================================
# SECTION C: TIEBREAKER CALCULATOR
# ============================================================

def calculate_tiebreaker_winner(players: list) -> dict:
    """
    Tính thứ hạng khi ≥2 agents còn sống sau step 500.
    Thứ tự ưu tiên: Kills > Boxes Broken > Items Collected > Bombs Placed

    Returns:
        dict {agent_id: rank} — rank 1 = nhất, rank N = bét
    """
    alive = [
        (p.kills, p.boxes_broken, p.items_collected, p.bombs_placed, p.id)
        for p in players if p.alive
    ]
    # Sắp xếp giảm dần theo tất cả tiêu chí
    alive_sorted = sorted(alive, reverse=True)

    result = {}
    for rank, (*_, pid) in enumerate(alive_sorted, start=1):
        result[pid] = rank

    # Agents đã chết → xếp sau tất cả agents sống
    dead_rank = len(alive) + 1
    for p in players:
        if not p.alive:
            result[p.id] = dead_rank

    return result


# ============================================================
# SECTION D: REWARD CONFIG
# ============================================================

class RewardConfig:
    """Cấu hình reward shaping — Tactical Survivor Edition v4.

    Triết lý:
    - Đầu game (>2 người): Săn bắn mạnh mẽ.
    - Cuối game (2 người): Rùa rụt cổ, farm hộp, câu giờ ăn Tie-breaker.
    """
    # ── Kết Quả Cuối Game ───────────────────────────────────────────
    WIN             = +1000.0   # Thắng outright hoặc tiebreaker rank 1
    DEATH           = -1000.0   # Bị loại (early)
    DEATH_LATE      = -1500.0   # Bị loại ở 1v1 (tội ác)
    TIE_LOSS        = -200.0    # Thua tiebreaker (rank 2 ở step 500)
    
    # ── Hình Phạt Thời Gian ─────────────────────────────────────
    SURVIVE_STEP    = -0.01     # Early: ép hành động
    SURVIVE_STEP_LATE = 0.0     # Late: cho phép câu giờ thoải mái
    
    # ── Tấn Công & Giết Địch ───────────────────────
    KILL            = +500.0    # Giết địch early
    KILL_LATE       = +150.0    # Giết địch late (vẫn thưởng nhưng ít)
    BOMB_SMART      = +80.0     # Đặt bom gần địch có đường thoát
    BOMB_AGGRESSIVE = +150.0    # Đặt bom bóp nghẹt lối thoát của địch
    THIRD_PARTY     = +50.0     # Can thiệp vào vùng giao tranh của 2 địch
    BOMB_KAMIKAZE   = -500.0    # Đặt bom tự sát
    
    # ── Phá Hủy & Thu Thập ───────────────────────
    BOX_BROKEN      = +20.0     # Early
    BOX_BROKEN_LATE = +60.0     # Late (x3 giá trị)
    ITEM_PICKUP     = +50.0
    ITEM_PICKUP_LATE= +20.0
    ITEM_CONTENTION = -15.0
    
    # ── Trừng Phạt / Thưởng Chiến Thuật Mới ─────────
    HUNTING_ADVANCE = +5.0      # Săn lùng (chỉ áp dụng khi đồ >= địch)
    RETREAT_REWARD  = +2.0      # Rút lui khi đồ < địch
    COWARD_PENALTY  = -3.0      # Nhẹ tay với kiting
    DEADEND_PENALTY = -50.0     # Phạt siêu nặng khi đâm đầu vào rọ
    BOMB_SPAM       = -5.0


# ============================================================
# SECTION E: BASELINE AGENTS (Opponents for training)
# ============================================================

class RandomAgent:
    """Agent ngẫu nhiên — opponent yếu nhất cho giai đoạn early training."""

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self._masker  = AlphaBetaMasker()

    def act(self, obs: dict) -> int:
        state  = GameState(obs["map"], obs["players"], obs["bombs"], self.agent_id)
        legal  = state.get_legal_actions(self.agent_id)
        return random.choice(legal) if legal else ACTION_STOP


def load_baseline_agent(agent_type: str, agent_id: int):
    """
    Load baseline agent theo tên.
    Thứ tự ưu tiên: engine baselines → fallback RandomAgent.
    """
    type_lower = agent_type.lower()
    agent_dir  = Path(__file__).parent / "Bomberland-GDGoC-AI-Challenge-main" \
                 / "Bomberland-GDGoC-AI-Challenge-main"

    if type_lower == "random":
        return RandomAgent(agent_id)

    try:
        sys.path.insert(0, str(agent_dir))
        if type_lower == "simple":
            from agent.simple_rule_agent import Agent as A
        elif type_lower == "tactical":
            from agent.tactical_rule_agent import TacticalRuleAgent as A
        elif type_lower == "smarter":
            from agent.smarter_rule_agent import Agent as A
        elif type_lower == "genius":
            from agent.genius_rule_agent import Agent as A
        else:
            return RandomAgent(agent_id)
        return A(agent_id)
    except (ImportError, Exception):
        return RandomAgent(agent_id)


def make_checkpoint_agent(net_state: dict, device, agent_id: int,
                           masker: AlphaBetaMasker):
    """Tạo agent dùng checkpoint weights cũ (self-play opponent)."""

    class _CheckpointAgent:
        def __init__(self):
            self.agent_id = agent_id
            self._net = ActorCriticNet()
            self._net.load_state_dict(net_state)
            self._net.to(device)
            self._net.eval()
            self._masker = masker
            self._device = device

        def act(self, obs: dict) -> int:
            state = GameState(obs["map"], obs["players"], obs["bombs"], self.agent_id)
            mask  = self._masker.get_safe_mask(state, self.agent_id, time_budget_ms=25)
            mf, af = encode_obs(obs, self.agent_id)
            mt = torch.FloatTensor(mf).unsqueeze(0).to(self._device)
            at = torch.FloatTensor(af).unsqueeze(0).to(self._device)
            with torch.no_grad():
                logits, _ = self._net(mt, at)
            logits_np = logits.squeeze(0).cpu().numpy()
            for i, safe in enumerate(mask):
                if not safe:
                    logits_np[i] = -1e4
            return int(np.argmax(logits_np))

    return _CheckpointAgent()


# ============================================================
# SECTION F: ROLLOUT BUFFER & GAE
# ============================================================

class RolloutBuffer:
    """Buffer lưu rollout data cho PPO update."""

    def __init__(self):
        self.map_states  = []
        self.aux_states  = []
        self.actions     = []
        self.log_probs   = []
        self.logits      = []
        self.rewards     = []
        self.dones       = []
        self.values      = []
        self.masks       = []

    def add(self, map_s, aux_s, action, log_prob, logits, reward, done, value, mask):
        self.map_states.append(map_s)
        self.aux_states.append(aux_s)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.logits.append(logits)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.masks.append(mask)

    def __len__(self):
        return len(self.actions)

    def to_tensors(self, device):
        map_t    = torch.FloatTensor(np.stack(self.map_states)).to(device)
        aux_t    = torch.FloatTensor(np.stack(self.aux_states)).to(device)
        act_t    = torch.LongTensor(self.actions).to(device)
        logp_t   = torch.FloatTensor(self.log_probs).to(device)
        logits_t = torch.FloatTensor(np.stack(self.logits)).to(device)
        rew_t    = torch.FloatTensor(self.rewards).to(device)
        done_t   = torch.FloatTensor(self.dones).to(device)
        val_t    = torch.FloatTensor(self.values).to(device)
        mask_t   = torch.BoolTensor(np.array(self.masks)).to(device)
        return map_t, aux_t, act_t, logp_t, logits_t, rew_t, done_t, val_t, mask_t


def compute_gae(rewards: list, values: list, dones: list,
                gamma: float = 0.99, lam: float = 0.95) -> tuple:
    """
    Compute Generalized Advantage Estimation (GAE).
    GAE giảm phương sai estimate advantage, ổn định PPO training.
    """
    advantages = []
    gae        = 0.0
    next_value = 0.0

    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_value * (1.0 - d) - v
        gae   = delta + gamma * lam * (1.0 - d) * gae
        advantages.insert(0, gae)
        next_value = v

    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


# ============================================================
# SECTION G: PPO TRAINER
# ============================================================

class PPOTrainer:
    """
    PPO Trainer với Hybrid Opponent Strategy:
      Episode 0-2000:  50% Baseline + 50% Self-play checkpoint
      Episode 2001+:   40% Baseline + 60% Self-play checkpoint (tăng self-play)

    Action Masking trong PPO:
      Logits của actions bị mask được set -inf TRƯỚC softmax
      → gradient không lan vào actions nguy hiểm
    """

    def __init__(self, net: ActorCriticNet, device,
                 lr: float = 3e-4,
                 clip_eps: float = 0.2,
                 start_entropy: float = 0.20,   # ÉP PHÁ CÁCH: Entropy siêu cao để thử nghiệm các pha lách góc hẹp
                 end_entropy: float   = 0.03,   # Entropy cuối: điềm tĩnh nhưng vẫn giữ sự lươn lẹo (3%)
                 value_coeff: float = 0.5,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ppo_epochs: int = 4,
                 mini_batch_size: int = 128,
                 rollout_steps: int = 2048,
                 max_grad_norm: float = 0.5,
                 total_rollouts: int = 200,
                 n_pi: int = 32,                 # PPG: Cứ 32 rollout chạy Aux Phase 1 lần
                 aux_epochs: int = 6,            # PPG: Số epoch train Aux Critic
                 beta_kl: float = 1.0):          # PPG: Hệ số phạt KL Divergence

        self.net             = net
        self.device          = device
        self.clip_eps        = clip_eps
        self.start_entropy   = start_entropy
        self.end_entropy     = end_entropy
        self.entropy_coeff   = start_entropy   # Bắt đầu ở mức cao nhất
        self.value_coeff     = value_coeff
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.ppo_epochs      = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.rollout_steps   = rollout_steps
        self.max_grad_norm   = max_grad_norm
        self.total_rollouts  = total_rollouts

        # PPG parameters
        self.n_pi = n_pi
        self.aux_epochs = aux_epochs
        self.beta_kl = beta_kl
        self.aux_buffer = []       # Lưu (map_t, aux_t, ret_t, old_logits_t)
        self.rollout_count = 0     # Đếm số lượng rollout đã update

        self.optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-5)
        self.masker    = AlphaBetaMasker()

        # Checkpoint pool lưu nhiều 'bản ngã' trong quá khứ của Agent
        # Tăng từ 5 → 20: giúp tránh Policy Cycling (học tủ chỉ đấu với phiên bản gần nhất)
        self.checkpoint_pool: list = []
        self.MAX_POOL_SIZE          = 20

        # Tracking
        self.total_episodes = 0
        self.total_steps    = 0

    def set_rollout_progress(self, current_rollout: int) -> float:
        """
        Cập nhật Entropy Coefficient theo tiến độ rollout (Linear Decay).

        Lịch trình:
          rollout 0   → entropy_coeff = start_entropy (0.05)  [Khám phá, lươn lẹo]
          rollout n-1 → entropy_coeff = end_entropy   (0.01)  [Quyết đoán, sắc bén]

        Trả về giá trị entropy hiện tại để tiện log.
        """
        progress = current_rollout / max(1, self.total_rollouts - 1)
        self.entropy_coeff = (
            self.start_entropy
            - (self.start_entropy - self.end_entropy) * progress
        )
        return self.entropy_coeff

    def save_checkpoint(self):
        """Lưu snapshot weights hiện tại vào pool."""
        snapshot = deepcopy(self.net.state_dict())
        self.checkpoint_pool.append(snapshot)
        if len(self.checkpoint_pool) > self.MAX_POOL_SIZE:
            self.checkpoint_pool.pop(0)

    def _make_opponent(self, agent_id: int, episode: int,
                       baseline_types: list):
        """
        Tạo opponent theo hybrid strategy.
        episode < 2000: 50% baseline, 50% checkpoint
        episode ≥ 2000: 40% baseline, 60% checkpoint
        """
        baseline_prob = 0.5 if episode < 2000 else 0.4
        use_baseline  = (random.random() < baseline_prob
                         or not self.checkpoint_pool)

        if use_baseline:
            btype = random.choice(baseline_types)
            return load_baseline_agent(btype, agent_id)
        else:
            old_state = random.choice(self.checkpoint_pool)
            return make_checkpoint_agent(old_state, self.device,
                                         agent_id, self.masker)

    def collect_rollout(self, env: GameEnv, my_id: int,
                        episode: int, baseline_types: list) -> RolloutBuffer:
        """
        Thu thập rollout_steps bước data từ môi trường.
        Opponents được tạo mới mỗi episode kết thúc.

        QUAN TRỌNG: Phải switch net sang eval() trước khi collect.
        Lý do: BatchNorm2d ở training mode với batch_size=1 tính unbiased
        variance = (x - mean)^2 / (N-1) = 0/0 = NaN → probs toàn NaN
        → torch.distributions.Categorical crash với ValueError Simplex().
        """
        buffer   = RolloutBuffer()
        obs      = env.reset()
        opponents = {
            i: self._make_opponent(i, episode, baseline_types)
            for i in range(4) if i != my_id
        }
        steps_collected  = 0
        eps_this_rollout = 0

        # ── FIX: Switch to eval mode for single-sample inference ──────
        # BatchNorm2d với batch=1 ở train mode → NaN. eval() dùng
        # running statistics đã tích lũy → ổn định.
        self.net.eval()

        while steps_collected < self.rollout_steps:
            # Encode my obs
            map_feat, aux_feat = encode_obs(obs, my_id)
            state = GameState(obs["map"], obs["players"], obs["bombs"], my_id)

            # Compute safe mask
            mask = self.masker.get_safe_mask(state, my_id, time_budget_ms=20)

            map_t = torch.FloatTensor(map_feat).unsqueeze(0).to(self.device)
            aux_t = torch.FloatTensor(aux_feat).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, value = self.net(map_t, aux_t)
                logits_np = logits.squeeze(0).cpu().numpy().copy()

            # ── FIX: NaN guard — nếu network vẫn cho NaN (training
            # instability sau update) thì fallback về uniform logits
            if np.any(np.isnan(logits_np)) or np.any(np.isinf(logits_np)):
                logits_np = np.zeros(6, dtype=np.float32)

            # Apply mask TRƯỚC khi lấy xác suất
            for i, safe in enumerate(mask):
                if not safe:
                    logits_np[i] = -1e4

            # Đảm bảo ít nhất 1 action hợp lệ (không bị mask all)
            if np.all(logits_np <= -1e8):
                logits_np[ACTION_STOP] = 0.0

            mlogits  = torch.FloatTensor(logits_np).to(self.device)
            probs    = torch.softmax(mlogits, dim=-1)

            # ── FIX: Clamp probs để tránh Categorical(Simplex) fail
            # khi có floating-point rounding error dẫn đến sum != 1
            probs    = probs.clamp(min=1e-8)
            probs    = probs / probs.sum()

            dist     = torch.distributions.Categorical(probs)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).item()
            action   = action_t.item()
            val      = value.squeeze().item()

            # Opponent actions
            all_actions = [ACTION_STOP] * 4
            all_actions[my_id] = action
            for opp_id, opp in opponents.items():
                try:
                    all_actions[opp_id] = opp.act(obs)
                except Exception:
                    all_actions[opp_id] = ACTION_STOP

            next_obs, rewards, done, info = env.step(all_actions)
            reward = float(rewards[my_id])

            # Chỉ lưu bước này nếu log_prob hợp lệ
            if not (np.isnan(log_prob) or np.isinf(log_prob)):
                buffer.add(map_feat, aux_feat, action, log_prob, logits_np,
                           reward, float(done), val, mask)
                steps_collected += 1

            obs = next_obs

            if done:
                eps_this_rollout += 1
                self.total_episodes += 1
                obs = env.reset()
                opponents = {
                    i: self._make_opponent(i, episode + eps_this_rollout, baseline_types)
                    for i in range(4) if i != my_id
                }

        # ── Switch lại train mode trước khi trả về để update() hoạt động
        self.net.train()
        self.total_steps += steps_collected
        return buffer

    def update(self, buffer: RolloutBuffer) -> dict:
        """
        Chạy PPO update từ collected buffer.
        Returns metrics dict.
        BatchNorm cần train mode ở đây (mini_batch_size >= 4 là đủ).
        """
        self.net.train()  # Đảm bảo train mode cho BatchNorm
        (map_t, aux_t, act_t, old_logp_t, logits_t,
         rew_t, done_t, val_t, mask_t) = buffer.to_tensors(self.device)

        # Compute GAE
        adv_list, ret_list = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            self.gamma, self.gae_lambda
        )
        adv_t = torch.FloatTensor(adv_list).to(self.device)
        ret_t = torch.FloatTensor(ret_list).to(self.device)

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(buffer)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.mini_batch_size):
                idx = indices[start: start + self.mini_batch_size]
                if len(idx) < 4:
                    continue

                mb_map  = map_t[idx]
                mb_aux  = aux_t[idx]
                mb_act  = act_t[idx]
                mb_olp  = old_logp_t[idx]
                mb_adv  = adv_t[idx]
                mb_ret  = ret_t[idx]
                mb_mask = mask_t[idx]   # (B, 6) bool

                # Forward
                logits, values = self.net(mb_map, mb_aux)

                # Apply action mask TRƯỚC softmax (gradient-safe)
                # Actions bị mask → -1e4 (an toàn với float32, tránh -inf/-1e9 gây NaN gradient)
                logits_m = logits.clone()
                logits_m[~mb_mask] = -1e4
                log_probs_all = F.log_softmax(logits_m, dim=-1)
                log_probs     = log_probs_all.gather(1, mb_act.unsqueeze(1)).squeeze(1)

                # Entropy (khuyến khích khám phá)
                probs   = torch.softmax(logits_m, dim=-1)
                entropy = -(probs * log_probs_all).sum(dim=-1).mean()

                # PPO clipped objective
                ratio = torch.exp(log_probs - mb_olp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio,
                                    1.0 - self.clip_eps,
                                    1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(1), mb_ret)

                # Total loss
                loss = (policy_loss
                        + self.value_coeff  * value_loss
                        - self.entropy_coeff * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += value_loss.item()
                metrics["entropy"]     += entropy.item()
                n_updates += 1
                
        # Lưu toàn bộ dữ liệu của rollout này vào aux_buffer
        for i in range(n):
            self.aux_buffer.append((
                map_t[i].cpu(), aux_t[i].cpu(), ret_t[i].cpu(), logits_t[i].cpu() 
            ))

        if n_updates > 0:
            for k in metrics:
                metrics[k] /= n_updates
        
        self.rollout_count += 1
        if self.rollout_count % self.n_pi == 0:
            aux_metrics = self._update_auxiliary()
            metrics.update(aux_metrics)
            
        return metrics

    def _update_auxiliary(self):
        """
        Pha 2 của PPG: Cập nhật Auxiliary Critic Head và chia sẻ kiến thức về Value cho Backbone.
        Sử dụng hàm KL Divergence để ngăn Actor Policy bị hỏng (Catastrophic Interference).
        """
        print(f"\n[PPG] Kích hoạt Auxiliary Phase sau {self.n_pi} rollouts...")
        self.net.train()
        
        # Build dataset từ aux_buffer
        # self.aux_buffer elements: (map_t, aux_t, ret_t, old_logits_t)
        n_samples = len(self.aux_buffer)
        
        metrics = {"aux_value_loss": 0.0, "aux_kl": 0.0}
        n_updates = 0
        
        for epoch in range(self.aux_epochs):
            # Khóa Backbone ở Epoch 0 để tránh sốc Gradient làm chệch hướng Policy
            if epoch == 0:
                for param in self.net.cnn.parameters():
                    param.requires_grad = False
                for param in self.net.cnn_fc.parameters():
                    param.requires_grad = False
                for param in self.net.mlp.parameters():
                    param.requires_grad = False
                for param in self.net.fusion.parameters():
                    param.requires_grad = False
            else:
                for param in self.net.parameters():
                    param.requires_grad = True

            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, self.mini_batch_size):
                idx = indices[start: start + self.mini_batch_size]
                if len(idx) < 4:
                    continue
                
                batch = [self.aux_buffer[i] for i in idx]
                mb_map = torch.stack([b[0] for b in batch]).to(self.device)
                mb_aux = torch.stack([b[1] for b in batch]).to(self.device)
                mb_ret = torch.stack([b[2] for b in batch]).to(self.device)
                mb_old_logits = torch.stack([b[3] for b in batch]).to(self.device)
                
                # Forward qua mạng để lấy aux_value và logits hiện tại
                logits, value, aux_value = self.net.forward_aux(mb_map, mb_aux)
                
                # Aux Value Loss
                aux_loss = F.mse_loss(aux_value.squeeze(1), mb_ret)
                
                # KL Divergence Penalty
                old_probs = torch.softmax(mb_old_logits, dim=-1)
                new_log_probs = F.log_softmax(logits, dim=-1)
                kl_div = torch.sum(old_probs * (torch.log(old_probs + 1e-8) - new_log_probs), dim=-1).mean()
                
                # Total Loss
                loss = aux_loss + self.beta_kl * kl_div
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                metrics["aux_value_loss"] += aux_loss.item()
                metrics["aux_kl"] += kl_div.item()
                n_updates += 1
                
        # Phục hồi requires_grad = True cho tất cả phòng trường hợp
        for param in self.net.parameters():
            param.requires_grad = True
            
        self.aux_buffer.clear()
        
        if n_updates > 0:
            for k in metrics:
                metrics[k] /= n_updates
                
        print(f"[PPG] Hoàn thành Phase 2 - Aux Loss: {metrics['aux_value_loss']:.4f} - KL: {metrics['aux_kl']:.4f}\n")
        return metrics


# ============================================================
# SECTION H: IMITATION LEARNING
# ============================================================

def collect_il_demonstrations(env: GameEnv, teacher_id: int,
                               n_episodes: int, save_path: str,
                               baseline_types: list = None) -> list:
    """
    Chạy Agent hiện tại (Alpha-Beta Masker + random policy) làm teacher
    để thu thập demonstration data.

    Vì weights chưa có trong phase này, teacher thực ra là:
      AlphaBetaMasker → chọn action safe → random trong safe actions

    Output: list of (map_feat, aux_feat, action) tuples
    """
    if baseline_types is None:
        baseline_types = ["random", "simple"]

    masker = AlphaBetaMasker()
    demos  = []

    print(f"[IL Collect] {n_episodes} episodes, teacher_id={teacher_id}")
    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_steps = 0

        opponents = {
            i: load_baseline_agent(random.choice(baseline_types), i)
            for i in range(4) if i != teacher_id
        }

        while not done and ep_steps < 500:
            # Teacher action: Alpha-Beta Masker + greedy heuristic
            state = GameState(obs["map"], obs["players"], obs["bombs"], teacher_id)
            mask  = masker.get_safe_mask(state, teacher_id, time_budget_ms=30)
            safe_actions = [a for a in range(6) if mask[a]]

            if not safe_actions:
                teacher_action = ACTION_STOP
            else:
                # Heuristic: ưu tiên BOMB nếu safe, rồi move về item/enemy
                if ACTION_BOMB in safe_actions:
                    # Kiểm tra xem đặt bom có trúng box/enemy không
                    me = state.players[teacher_id]
                    radius = 1 + me[4]
                    boxes_hit = 0
                    for dx, dy in ((-1,0),(+1,0),(0,-1),(0,+1)):
                        for r in range(1, radius+1):
                            nx, ny = me[0]+dx*r, me[1]+dy*r
                            if not state._in_bounds(nx, ny): break
                            if int(state.grid[nx,ny]) == WALL: break
                            if int(state.grid[nx,ny]) == BOX:
                                boxes_hit += 1; break
                    if boxes_hit > 0:
                        teacher_action = ACTION_BOMB
                    else:
                        teacher_action = random.choice(
                            [a for a in safe_actions if a != ACTION_BOMB] or safe_actions
                        )
                else:
                    teacher_action = random.choice(safe_actions)

            # Ghi nhận demonstration
            map_feat, aux_feat = encode_obs(obs, teacher_id)
            demos.append((map_feat, aux_feat, teacher_action))

            # Collect all actions
            all_actions = [ACTION_STOP] * 4
            all_actions[teacher_id] = teacher_action
            for oid, opp in opponents.items():
                try:
                    all_actions[oid] = opp.act(obs)
                except Exception:
                    all_actions[oid] = ACTION_STOP

            obs, _, done, _ = env.step(all_actions)
            ep_steps += 1

        if (ep + 1) % 200 == 0:
            print(f"[IL Collect] Ep {ep+1}/{n_episodes}, demos: {len(demos)}")

    with open(save_path, "wb") as f:
        pickle.dump(demos, f)
    print(f"[IL Collect] Saved {len(demos)} demos → {save_path}")
    return demos


def train_il(demos_path: str, save_path: str, device,
             epochs: int = 10, batch_size: int = 512, lr: float = 1e-3) -> ActorCriticNet:
    """
    Train ActorCriticNet bằng Imitation Learning (Supervised Cross-Entropy trên Actor).
    Critic không được train trong phase này (không có labels).
    """
    print(f"[IL Train] Loading from {demos_path}...")
    with open(demos_path, "rb") as f:
        demos = pickle.load(f)

    n = len(demos)
    print(f"[IL Train] {n} samples, {epochs} epochs, batch={batch_size}")

    map_data = np.stack([d[0] for d in demos]).astype(np.float32)
    aux_data = np.stack([d[1] for d in demos]).astype(np.float32)
    act_data = np.array([d[2] for d in demos], dtype=np.int64)

    net       = ActorCriticNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        idx     = np.random.permutation(n)
        tot_loss, correct, n_batches = 0.0, 0, 0

        for start in range(0, n, batch_size):
            b_idx = idx[start: start + batch_size]
            map_t = torch.FloatTensor(map_data[b_idx]).to(device)
            aux_t = torch.FloatTensor(aux_data[b_idx]).to(device)
            act_t = torch.LongTensor(act_data[b_idx]).to(device)

            logits, _ = net(map_t, aux_t)
            loss = F.cross_entropy(logits, act_t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            tot_loss += loss.item()
            correct  += (logits.argmax(1) == act_t).sum().item()
            n_batches += 1

        scheduler.step()
        acc = correct / n * 100
        print(f"[IL Train] Epoch {epoch+1}/{epochs}: "
              f"loss={tot_loss/n_batches:.4f}, acc={acc:.1f}%")

    torch.save(net.state_dict(), save_path)
    print(f"[IL Train] Weights saved → {save_path}")
    return net


# ============================================================
# SECTION I: PPO TRAINING LOOP
# ============================================================

def train_ppo(load_path: str, save_path: str, device,
              n_rollouts: int = 200,
              checkpoint_interval: int = 20,
              baseline_types: list = None,
              my_agent_id: int = 0,
              rollout_steps: int = 2048) -> None:
    """
    PPO Fine-tuning với Hybrid Opponent.
    n_rollouts: số lần collect+update (mỗi rollout = rollout_steps steps).
    """
    if baseline_types is None:
        baseline_types = ["random", "simple", "tactical"]

    net = ActorCriticNet().to(device)
    if load_path and Path(load_path).exists():
        print(f"[PPO] Loading weights from {load_path}")
        net.load_state_dict(
            torch.load(load_path, map_location=device, weights_only=True),
            strict=False
        )
    else:
        print("[PPO] Training from scratch (no weights loaded)")

    trainer = PPOTrainer(
        net=net,
        device=device,
        lr=args.lr,
        clip_eps=args.clip_eps,
        start_entropy=args.start_entropy,
        end_entropy=args.end_entropy,
        value_coeff=args.value_coeff,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch,
        rollout_steps=args.rollout_steps,
        max_grad_norm=args.max_grad_norm,
        total_rollouts=args.epochs,
        n_pi=args.n_pi,
        aux_epochs=args.aux_epochs,
        beta_kl=args.beta_kl
    )
    env     = GameEnv()
    episode_counter = 0

    print(f"[PPO] {n_rollouts} rollouts x {rollout_steps} steps, agent_id={my_agent_id}")
    print(f"[PPO] Entropy decay: {trainer.start_entropy:.3f} -> {trainer.end_entropy:.3f}")

    for rollout_idx in range(n_rollouts):
        # Cập nhật entropy theo tiến độ rollout (0.05 → 0.01 tuyến tính)
        current_ent = trainer.set_rollout_progress(rollout_idx)

        # eval() trước rollout (bảo vệ BatchNorm batch=1), train() sau đó
        net.eval()
        buffer  = trainer.collect_rollout(env, my_agent_id,
                                          episode_counter, baseline_types)
        net.train()
        metrics = trainer.update(buffer)
        episode_counter = trainer.total_episodes

        print(f"[PPO] Rollout {rollout_idx+1}/{n_rollouts} | "
              f"ep={episode_counter} | steps={trainer.total_steps} | "
              f"pl={metrics['policy_loss']:.4f} | "
              f"vl={metrics['value_loss']:.4f} | "
              f"ent={metrics['entropy']:.4f} | "
              f"ent_c={current_ent:.4f}")

        # Checkpoint
        if (rollout_idx + 1) % checkpoint_interval == 0:
            trainer.save_checkpoint()
            ckpt = save_path.replace(".pth", f"_ckpt{rollout_idx+1}.pth")
            torch.save(net.state_dict(), ckpt)
            print(f"[PPO] Checkpoint → {ckpt}")

    torch.save(net.state_dict(), save_path)
    print(f"[PPO] Final weights → {save_path}")


# ============================================================
# SECTION J: BENCHMARK
# ============================================================

def run_benchmark(weights_path: str, device, n_episodes: int = 100,
                  opponent_type: str = "tactical") -> float:
    """Đánh giá win rate của agent vs một loại baseline."""
    from agent import Agent as SubmissionAgent

    agent = SubmissionAgent(0)
    if weights_path and Path(weights_path).exists():
        agent.net.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        agent.net.eval()

    env      = GameEnv()
    wins     = 0

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        opponents = {
            i: load_baseline_agent(opponent_type, i)
            for i in range(1, 4)
        }
        final_rewards = [0.0] * 4

        while not done:
            all_actions = [ACTION_STOP] * 4
            all_actions[0] = agent.act(obs)
            for oid, opp in opponents.items():
                try:
                    all_actions[oid] = opp.act(obs)
                except Exception:
                    all_actions[oid] = ACTION_STOP
            obs, rewards, done, _ = env.step(all_actions)
            for i in range(4):
                final_rewards[i] += rewards[i]

        if final_rewards[0] == max(final_rewards):
            wins += 1

        if (ep + 1) % 20 == 0:
            print(f"[Benchmark] {ep+1}/{n_episodes}: win_rate={wins/(ep+1)*100:.1f}%")

    rate = wins / n_episodes
    print(f"[Benchmark] Final vs {opponent_type}: {wins}/{n_episodes} = {rate*100:.1f}%")
    return rate


# ============================================================
# SECTION K: MAIN CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bomberland Guided RL — Training Pipeline"
    )
    parser.add_argument(
        "--phase",
        choices=["il_collect", "il_train", "ppo", "benchmark"],
        required=True,
        help="Phase to run"
    )
    parser.add_argument("--episodes",   type=int,   default=5000,
                        help="Episodes (IL collect) or rollouts (PPO)")
    parser.add_argument("--rollout_steps", type=int, default=2048,
                        help="Steps per PPO rollout")
    parser.add_argument("--output",     type=str,   default="weights.pth")
    parser.add_argument("--load",       type=str,   default=None,
                        help="Path to weights to load")
    parser.add_argument("--demos",      type=str,   default="demos.pkl",
                        help="Path to demonstrations file")
    
    # ── Tham số Training cơ bản
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # ── Tham số PPG (Phasic Policy Gradient)
    parser.add_argument("--n_pi", type=int, default=32, help="Số lượng rollouts trước khi chạy Pha 2 (Auxiliary)")
    parser.add_argument("--aux_epochs", type=int, default=6, help="Số epochs để train Auxiliary Critic")
    parser.add_argument("--beta_kl", type=float, default=1.0, help="Hệ số KL phạt để bảo vệ Policy")

    # ── PPO Hyperparameters (Pha 1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--start_entropy", type=float, default=0.05)
    parser.add_argument("--end_entropy", type=float, default=0.01)
    parser.add_argument("--value_coeff", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--mini_batch", type=int, default=256)

    parser.add_argument("--agent_id",   type=int,   default=0,
                        help="Our agent ID (0-3)")
    parser.add_argument("--checkpoint_interval", type=int, default=20,
                        help="Save checkpoint every N rollouts")
    parser.add_argument("--baseline_types", nargs="+",
                        default=["random", "simple", "tactical"],
                        help="Baseline types for opponents")
    parser.add_argument("--opponent",   type=str,   default="tactical",
                        help="Opponent for benchmark")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")

    if args.phase == "il_collect":
        env = GameEnv()
        collect_il_demonstrations(
            env=env,
            teacher_id=args.agent_id,
            n_episodes=args.episodes,
            save_path=args.demos,
            baseline_types=args.baseline_types,
        )

    elif args.phase == "il_train":
        train_il(
            demos_path=args.demos,
            save_path=args.output,
            device=device,
        )

    elif args.phase == "ppo":
        train_ppo(
            load_path=args.load,
            save_path=args.output,
            device=device,
            n_rollouts=args.episodes,
            checkpoint_interval=args.checkpoint_interval,
            baseline_types=args.baseline_types,
            my_agent_id=args.agent_id,
            rollout_steps=args.rollout_steps,
        )

    elif args.phase == "benchmark":
        run_benchmark(
            weights_path=args.load,
            device=device,
            n_episodes=args.episodes,
            opponent_type=args.opponent,
        )
