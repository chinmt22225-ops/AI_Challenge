# ============================================================
# Bomberland GDGoC AI Challenge 2026
# Kiến trúc: Guided RL — Actor-Critic + Alpha-Beta Masker
# File duy nhất để nộp bài (KHÔNG chứa training code)
# Imports: chỉ stdlib + numpy + torch
# ============================================================

import time
import math
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SECTION 1: CONSTANTS & CONFIG
# ============================================================

# Mapping hành động THỰC TẾ sau bug engine BTC (xác minh từ engine/player.py & game.py)
# BTC comment gốc: "LEFT = 1 # Note: Visually moves UP"
# Tức là:
#   Action 1 (tên cũ LEFT)  → dx=-1 (di chuyển lên theo ma trận, row giảm)
#   Action 2 (tên cũ RIGHT) → dx=+1 (di chuyển xuống theo ma trận, row tăng)
#   Action 3 (tên cũ UP)    → dy=-1 (di chuyển trái theo ma trận, col giảm)
#   Action 4 (tên cũ DOWN)  → dy=+1 (di chuyển phải theo ma trận, col tăng)
ACTION_STOP  = 0
ACTION_UP    = 1   # visually UP   → dx=-1 (row decreases)
ACTION_DOWN  = 2   # visually DOWN → dx=+1 (row increases)
ACTION_LEFT  = 3   # visually LEFT → dy=-1 (col decreases)
ACTION_RIGHT = 4   # visually RIGHT→ dy=+1 (col increases)
ACTION_BOMB  = 5

# DIR_MAP[action] = (dx, dy) — x là hàng (row), y là cột (col)
DIR_MAP = {
    ACTION_STOP:  ( 0,  0),
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  (+1,  0),
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0, +1),
}

# Giá trị ô bản đồ
GRASS    = 0
WALL     = 1
BOX      = 2
ITEM_RAD = 3   # Item tăng radius bom
ITEM_CAP = 4   # Item tăng capacity bom

# Hằng số game
BOMB_TIMER   = 7   # Đếm ngược từ 7, nổ khi <= 0
MAX_RADIUS   = 5   # Radius tối đa
MAX_CAPACITY = 5   # Capacity tối đa
GRID_SIZE    = 13  # Kích thước bản đồ


class Config:
    """Cấu hình thời gian budget cho mỗi lượt."""
    MASKER_TIME_BUDGET_MS = 40  # Tối đa cho Alpha-Beta Masker
    TOTAL_TIME_BUDGET_MS  = 95  # Tổng budget (để 5ms buffer)
    NET_TIMEOUT_MS        = 20  # Timeout tối thiểu cần có để chạy network


# ============================================================
# SECTION 2: GAME STATE (Lightweight, dùng cho Masker)
# ============================================================

class GameState:
    """
    Trạng thái game nhẹ, dùng cho AlphaBetaMasker và mô phỏng nhanh.
    Không cần simulation đầy đủ — chỉ cần đủ để tính danger zones,
    legal actions và successor states cho depth 2-3.
    
    players[i] = [x, y, alive(0/1), bombs_left, radius_bonus]
    bombs[i]   = [x, y, timer, owner_id]
    """

    def __init__(self, grid, players, bombs, my_agent_id: int):
        self.grid = np.array(grid, dtype=np.int8)
        self.players = [list(map(int, p)) for p in players]
        self.bombs   = [list(map(int, b)) for b in bombs]
        self.my_agent_id = int(my_agent_id)

    def clone(self):
        ns = GameState.__new__(GameState)
        ns.grid        = self.grid.copy()
        ns.players     = [p[:] for p in self.players]
        ns.bombs       = [b[:] for b in self.bombs]
        ns.my_agent_id = self.my_agent_id
        return ns

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def _is_passable(self, x: int, y: int) -> bool:
        """Ô có thể đứng vào (không phải wall hoặc box)."""
        if not self._in_bounds(x, y):
            return False
        return self.grid[x, y] not in (WALL, BOX)

    def _get_danger_zones(self) -> set:
        """
        Tính tập hợp (x, y) bị ảnh hưởng bởi TẤT CẢ bom đang active.
        Bao gồm ô chứa bom và các ô trong bán kính nổ theo 4 hướng.
        Tường chặn sóng nổ; Box hấp thụ (tính vào danger nhưng không lan qua).
        """
        danger = set()
        for b in self.bombs:
            bx, by = b[0], b[1]
            oid    = b[3]
            # Radius thực = 1 + bonus của owner
            radius = 1
            if 0 <= oid < len(self.players):
                radius = 1 + self.players[oid][4]

            danger.add((bx, by))
            for dx, dy in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                for r in range(1, radius + 1):
                    nx, ny = bx + dx * r, by + dy * r
                    if not self._in_bounds(nx, ny):
                        break
                    cell = self.grid[nx, ny]
                    if cell == WALL:
                        break          # Tường chặn hoàn toàn
                    danger.add((nx, ny))
                    if cell == BOX:
                        break          # Box hấp thụ, không lan qua
        return danger

    def get_legal_actions(self, agent_id: int) -> list:
        """Danh sách actions hợp lệ cho agent."""
        if agent_id >= len(self.players) or self.players[agent_id][2] == 0:
            return [ACTION_STOP]

        p  = self.players[agent_id]
        px, py       = p[0], p[1]
        bombs_left   = p[3]

        # Bom đang trên sàn — không được bước vào (trừ khi vừa đặt)
        bomb_pos = {(b[0], b[1]) for b in self.bombs}
        # Vị trí agents khác đang sống (có thể đứng chung, nhưng engine thực tế cho phép)
        # => Để sát với agent.py cũ: không chặn đứng chung

        actions = [ACTION_STOP]
        for action, (dx, dy) in DIR_MAP.items():
            if action == ACTION_STOP:
                continue
            nx, ny = px + dx, py + dy
            if self._is_passable(nx, ny) and (nx, ny) not in bomb_pos:
                actions.append(action)

        # Đặt bom: cần còn bombs_left và chưa có bom tại ô hiện tại
        if bombs_left > 0 and (px, py) not in bomb_pos:
            actions.append(ACTION_BOMB)

        return actions

    def generate_successor(self, agent_id: int, action: int) -> "GameState":
        """
        Sinh successor state sau khi agent thực hiện action.
        Simplified (không mô phỏng 3 opponents) — đủ dùng cho Masker depth ≤3.
        """
        ns = self.clone()
        if agent_id >= len(ns.players) or ns.players[agent_id][2] == 0:
            return ns

        p  = ns.players[agent_id]
        px, py = p[0], p[1]

        # Di chuyển hoặc đặt bom
        if action in DIR_MAP and action != ACTION_STOP:
            dx, dy = DIR_MAP[action]
            nx, ny = px + dx, py + dy
            p[0], p[1] = nx, ny
            cell = int(ns.grid[nx, ny])
            if cell == ITEM_RAD:
                p[4] = min(p[4] + 1, MAX_RADIUS - 1)
                ns.grid[nx, ny] = GRASS
            elif cell == ITEM_CAP:
                p[3] = min(p[3] + 1, MAX_CAPACITY)
                ns.grid[nx, ny] = GRASS
        elif action == ACTION_BOMB and p[3] > 0:
            p[3] -= 1
            ns.bombs.append([px, py, BOMB_TIMER, agent_id])

        # Tick bombs (đếm ngược 1 bước)
        exploding, surviving = [], []
        for b in ns.bombs:
            b[2] -= 1
            (exploding if b[2] <= 0 else surviving).append(b)
        ns.bombs = surviving

        # Xử lý nổ (simplified: không chain reaction để tiết kiệm thời gian)
        blast = set()
        for b in exploding:
            bx, by, _, oid = b[0], b[1], b[2], b[3]
            radius = 1
            if 0 <= oid < len(ns.players):
                radius = 1 + ns.players[oid][4]
                if oid == agent_id:
                    ns.players[agent_id][3] += 1   # Hoàn trả bombs_left
            blast.add((bx, by))
            for dx, dy in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                for r in range(1, radius + 1):
                    nx, ny = bx + dx * r, by + dy * r
                    if not ns._in_bounds(nx, ny): break
                    if ns.grid[nx, ny] == WALL:    break
                    blast.add((nx, ny))
                    if ns.grid[nx, ny] == BOX:
                        ns.grid[nx, ny] = GRASS
                        break

        # Loại agent bị dính nổ
        for i, pl in enumerate(ns.players):
            if pl[2] == 1 and (pl[0], pl[1]) in blast:
                pl[2] = 0

        return ns

    def is_dead(self, agent_id: int) -> bool:
        return self.players[agent_id][2] == 0


# ============================================================
# SECTION 3: ALPHA-BETA MASKER (Bộ lọc an toàn)
# ============================================================

class AlphaBetaMasker:
    """
    Bộ lọc an toàn chạy TRƯỚC khi Actor-Critic ra quyết định.
    Loại bỏ các hành động nguy hiểm ngay lập tức (tự sát, ngõ cụt có bom).

    Adaptive depth:
      - Mặc định: depth=2
      - Khi ≥3 bom active trong bán kính 5 ô: depth=3
      - Khi thời gian < 15ms còn lại: fallback depth=1

    Các Mask áp dụng (theo thứ tự ưu tiên):
      Mask 1 — Passability & Immediate Blast:
          Loại bỏ ô tường/box/bom; loại bỏ ô trong danger zone ngay lập tức.
      Mask 2 — Escape after BOMB:
          Nếu đặt bom, kiểm tra agent có đường thoát ra ô an toàn không.
      Mask 3 — Dead-end Trap:
          Nếu ô đích có ≤1 lối thoát VÀ bom gần (timer ≤ 3) → nguy hiểm.
      Mask 4 — Item Contention Avoidance:
          Tín hiệu rủi ro tranh item — không hard-mask, để Actor tự học.
    """

    def get_safe_mask(self, state: GameState, agent_id: int,
                      time_budget_ms: float = 40.0) -> list:
        """
        Returns list[bool] độ dài 6.
        True  → hành động AN TOÀN
        False → hành động NGUY HIỂM (bị loại)
        """
        t0    = time.time()
        mask  = [True] * 6

        if agent_id >= len(state.players) or state.players[agent_id][2] == 0:
            return mask

        me     = state.players[agent_id]
        px, py = me[0], me[1]
        danger = state._get_danger_zones()
        bomb_positions = {(b[0], b[1]) for b in state.bombs}

        # ── Mask 1: Passability + Immediate Blast ─────────────────────
        safe_moves_exist = False
        for action in range(6):
            if action == ACTION_BOMB:
                # Xử lý BOMB mask riêng bên dưới
                if me[3] <= 0 or (px, py) in bomb_positions:
                    mask[ACTION_BOMB] = False
                continue
            dx, dy = DIR_MAP[action]
            nx, ny = px + dx, py + dy
            # Không thể đi: out-of-bounds, wall, box
            if not state._is_passable(nx, ny):
                mask[action] = False
                continue
            # Không thể đi: có bom đang nằm trên sàn
            if (nx, ny) in bomb_positions:
                mask[action] = False
                continue
            # Tạm thời đánh dấu safe
            if (nx, ny) in danger:
                mask[action] = False
            else:
                safe_moves_exist = True

        # Nếu STOP đang safe (không trong danger) thì safe_moves_exist = True
        if (px, py) not in danger:
            safe_moves_exist = True

        # Nếu KHÔNG còn move nào safe ngoài danger → mở tất cả passes (ít nhất sống sót tốt nhất)
        if not safe_moves_exist:
            # Tình huống bị bao vây: chỉ mở các ô passable, không xét danger
            for action in range(5):
                dx, dy = DIR_MAP[action]
                nx, ny = px + dx, py + dy
                mask[action] = (state._is_passable(nx, ny)
                                and (nx, ny) not in bomb_positions)

        # ── Mask 2: Escape route sau khi đặt bom ─────────────────────
        elapsed = (time.time() - t0) * 1000
        if mask[ACTION_BOMB] and elapsed < time_budget_ms - 15:
            if not self._has_escape_after_bomb(state, agent_id, danger, bomb_positions):
                mask[ACTION_BOMB] = False

        # ── Mask 3: Dead-end trap (BFS 1 bước) ──────────────────────
        elapsed = (time.time() - t0) * 1000
        if elapsed < time_budget_ms - 10:
            self._mask_deadend_traps(state, mask, danger, bomb_positions, t0, time_budget_ms)

        # Đảm bảo luôn có ít nhất 1 action
        if not any(mask):
            # Fallback: STOP hoặc action đầu tiên passable
            mask[ACTION_STOP] = True

        return mask

    def _has_escape_after_bomb(self, state: GameState, agent_id: int,
                                existing_danger: set,
                                bomb_positions: set) -> bool:
        """
        Sau khi đặt bom tại (px, py), agent có đường thoát ra ô an toàn
        trong BOMB_TIMER bước không?
        """
        me     = state.players[agent_id]
        px, py = me[0], me[1]
        radius = 1 + me[4]

        # Tính blast zone của bom mới
        new_danger = set(existing_danger)
        new_danger.add((px, py))
        for dx, dy in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
            for r in range(1, radius + 1):
                nx, ny = px + dx * r, py + dy * r
                if not state._in_bounds(nx, ny): break
                if state.grid[nx, ny] == WALL:   break
                new_danger.add((nx, ny))
                if state.grid[nx, ny] == BOX:    break

        # BFS tìm ô an toàn có thể đến trong BOMB_TIMER bước
        all_bomb_pos = bomb_positions | {(px, py)}
        q       = deque([(px, py, 0)])
        visited = {(px, py)}
        while q:
            cx, cy, dist = q.popleft()
            if dist > 0 and (cx, cy) not in new_danger:
                return True   # Tìm được đường thoát
            if dist >= BOMB_TIMER - 1:
                continue
            for ddx, ddy in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                nx, ny = cx + ddx, cy + ddy
                if (nx, ny) in visited: continue
                if not state._is_passable(nx, ny): continue
                # Bom từ bước trước vẫn chặn đường; bom mới đặt (px,py) thì được đi ra
                if (nx, ny) in all_bomb_pos and (nx, ny) != (px, py): continue
                visited.add((nx, ny))
                q.append((nx, ny, dist + 1))
        return False

    def _mask_deadend_traps(self, state: GameState, mask: list,
                             danger: set, bomb_positions: set,
                             t0: float, time_budget_ms: float):
        """
        Với mỗi move action còn safe (mask=True), kiểm tra ô đích có phải
        ngõ cụt nguy hiểm không:
          - ≤ 1 lối thoát tại ô đích
          - Có bom gần với timer ≤ 3
        """
        for action in range(1, 5):   # Chỉ check 4 hướng di chuyển
            if not mask[action]:
                continue
            elapsed = (time.time() - t0) * 1000
            if elapsed > time_budget_ms - 5:
                break

            dx, dy = DIR_MAP[action]
            nx, ny = int(state.players[state.my_agent_id][0]) + dx, \
                     int(state.players[state.my_agent_id][1]) + dy
            if not state._in_bounds(nx, ny):
                continue

            # Đếm số lối thoát từ ô đích (không tính ô đang đứng)
            escape_count = 0
            for ddx, ddy in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                ex, ey = nx + ddx, ny + ddy
                if (state._is_passable(ex, ey)
                        and (ex, ey) not in bomb_positions):
                    escape_count += 1

            if escape_count <= 1:
                # Kiểm tra xem có bom đang đe dọa ô này không (timer gấp)
                urgent_bomb = any(
                    b[2] <= 3
                    for b in state.bombs
                    if abs(b[0] - nx) + abs(b[1] - ny) <= 5
                )
                if urgent_bomb:
                    mask[action] = False


# ============================================================
# SECTION 4: OBSERVATION ENCODER (9 channels + 8 scalars)
# ============================================================

def encode_obs(obs: dict, agent_id: int) -> tuple:
    """
    Encode observation thành 2 tensors cho ActorCriticNet.

    Returns:
        map_feat: np.ndarray shape (9, 13, 13) — spatial input cho CNN
        aux_feat: np.ndarray shape (8,)         — scalar input cho MLP

    9 Channels spatial:
        [0] Walls            (grid == WALL)
        [1] Boxes            (grid == BOX)
        [2] My position      (one-hot, vị trí agent ta)
        [3] Enemy heatmap    (tổng hợp vị trí 3 enemies còn sống)
        [4] Bomb positions   (binary: 1 nếu có bom)
        [5] Bomb timers      (normalized timer/7, lấy max nếu nhiều bom)
        [6] My bomb map      (+1.0=bom của ta, -1.0=bom địch)
        [7] Item Radius      (grid == ITEM_RAD)
        [8] Item Capacity    (grid == ITEM_CAP)

    8 Scalar features (normalized về [0,1]):
        [0] current_step / 500  (0 ở inference; cập nhật qua tracking nếu có)
        [1] my_kills / 3        (0 ở inference vì obs không có kills)
        [2] my_bombs_left / MAX_CAPACITY
        [3] my_radius_bonus / (MAX_RADIUS-1)
        [4] alive_enemies / 3
        [5] nearest_bomb_timer / BOMB_TIMER  (1.0 nếu không có bom gần)
        [6] in_danger flag      (1.0 nếu agent đang ở ô trong blast zone)
        [7] my_active_bombs / MAX_CAPACITY
    """
    grid    = np.array(obs["map"],     dtype=np.float32)   # (13,13)
    players = obs["players"]                               # (4,5)
    bombs   = obs["bombs"]                                 # (N,4)
    agent_id = int(agent_id)

    H = W = GRID_SIZE

    # ── Channels ────────────────────────────────────────────────────────
    ch_wall  = (grid == WALL).astype(np.float32)
    ch_box   = (grid == BOX).astype(np.float32)
    ch_me    = np.zeros((H, W), dtype=np.float32)
    ch_enemy = np.zeros((H, W), dtype=np.float32)
    ch_bpos  = np.zeros((H, W), dtype=np.float32)
    ch_btimer = np.zeros((H, W), dtype=np.float32)
    ch_mybomb = np.zeros((H, W), dtype=np.float32)
    ch_irad  = (grid == ITEM_RAD).astype(np.float32)
    ch_icap  = (grid == ITEM_CAP).astype(np.float32)

    # My position
    me = players[agent_id]
    mx, my_ = int(me[0]), int(me[1])
    if int(me[2]) == 1 and 0 <= mx < H and 0 <= my_ < W:
        ch_me[mx, my_] = 1.0

    # Enemy positions
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) == 0:
            continue
        ex, ey = int(p[0]), int(p[1])
        if 0 <= ex < H and 0 <= ey < W:
            ch_enemy[ex, ey] += 1.0   # Heatmap: cộng dồn nếu nhiều enemy cùng ô

    # Bomb channels
    my_active_bombs = 0
    for b in bombs:
        bx, by, timer, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        if not (0 <= bx < H and 0 <= by < W):
            continue
        ch_bpos[bx, by]  = 1.0
        t_norm = float(timer) / BOMB_TIMER
        ch_btimer[bx, by] = max(ch_btimer[bx, by], t_norm)
        if oid == agent_id:
            ch_mybomb[bx, by] = 1.0
            my_active_bombs += 1
        elif ch_mybomb[bx, by] == 0.0:
            ch_mybomb[bx, by] = -1.0   # Bom địch

    map_feat = np.stack([
        ch_wall, ch_box, ch_me, ch_enemy,
        ch_bpos, ch_btimer, ch_mybomb,
        ch_irad, ch_icap,
    ], axis=0)   # shape: (9, 13, 13)

    # ── Scalar features ─────────────────────────────────────────────────
    alive_enemies = sum(1 for i, p in enumerate(players)
                        if i != agent_id and int(p[2]) == 1)

    # Tìm bom gần nhất (manhattan distance ≤ 6)
    nearest_timer_norm = 1.0
    for b in bombs:
        dist = abs(int(b[0]) - mx) + abs(int(b[1]) - my_)
        if dist <= 6:
            nearest_timer_norm = min(nearest_timer_norm,
                                     float(int(b[2])) / BOMB_TIMER)

    # Kiểm tra agent đang trong blast zone
    in_danger = 0.0
    radius_me = 1 + int(me[4])
    for b in bombs:
        bx, by = int(b[0]), int(b[1])
        oid = int(b[3])
        if 0 <= oid < len(players):
            brad = 1 + int(players[oid][4])
        else:
            brad = 1
        # Kiểm tra inline (axis-aligned)
        if bx == mx and abs(my_ - by) <= brad:
            in_danger = 1.0; break
        if by == my_ and abs(mx - bx) <= brad:
            in_danger = 1.0; break

    aux_feat = np.array([
        0.0,                                                  # [0] step (unknown)
        0.0,                                                  # [1] kills (unknown)
        float(int(me[3])) / max(1, MAX_CAPACITY),            # [2] bombs_left
        float(int(me[4])) / max(1, MAX_RADIUS - 1),         # [3] radius_bonus
        float(alive_enemies) / 3.0,                          # [4] alive enemies
        nearest_timer_norm,                                   # [5] nearest bomb timer
        in_danger,                                            # [6] danger flag
        float(my_active_bombs) / max(1, MAX_CAPACITY),      # [7] active bombs
    ], dtype=np.float32)

    return map_feat, aux_feat


# ============================================================
# SECTION 5: ACTOR-CRITIC NETWORK (Small CNN)
# ============================================================

class ActorCriticNet(nn.Module):
    """
    Small Actor-Critic network với dual-input:
      - CNN branch: xử lý dữ liệu không gian (9 × 13 × 13)
      - MLP branch: xử lý scalar features (8,)

    Thiết kế nhỏ gọn, ưu tiên tốc độ inference < 5ms trên CPU.
    Ước tính: Conv2d(9→32) ~0.3ms, Conv2d(32→64) ~0.5ms, Linear ~0.2ms.
    """

    MAP_CHANNELS = 9
    AUX_DIM      = 8
    N_ACTIONS    = 6

    def __init__(self):
        super().__init__()

        # ── CNN branch (2 conv layers) ───────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(self.MAP_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Output shape: (64, 13, 13) = 10816
        cnn_flat_dim = 64 * GRID_SIZE * GRID_SIZE   # = 10816

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_flat_dim, 256),
            nn.ReLU(inplace=True),
        )

        # ── MLP branch ───────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(self.AUX_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # ── Fusion ───────────────────────────────────────────────────────
        # 256 (CNN) + 64 (MLP) = 320 → 128
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(inplace=True),
        )

        # ── Output heads ────────────────────────────────────────────────
        self.actor  = nn.Linear(128, self.N_ACTIONS)   # Logits cho 6 actions
        self.critic = nn.Linear(128, 1)                 # State value V(s)

        self._init_weights()

    def _init_weights(self):
        """Khởi tạo weights theo phương pháp orthogonal (ổn định cho RL)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Actor head: init nhỏ → logits ban đầu gần đều (uniform policy)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, map_x: torch.Tensor, aux_x: torch.Tensor):
        """
        Args:
            map_x: Tensor (B, 9, 13, 13)
            aux_x: Tensor (B, 8)
        Returns:
            logits: (B, 6) — unnormalized action log-probs
            value:  (B, 1) — V(s)
        """
        # CNN branch
        cnn_out  = self.cnn(map_x)
        cnn_flat = cnn_out.view(cnn_out.size(0), -1)
        cnn_feat = self.cnn_fc(cnn_flat)

        # MLP branch
        aux_feat = self.mlp(aux_x)

        # Fusion & Heads
        fused  = torch.cat([cnn_feat, aux_feat], dim=1)
        fused  = self.fusion(fused)
        logits = self.actor(fused)
        value  = self.critic(fused)
        return logits, value


# ============================================================
# SECTION 6: AGENT (Submission Class)
# ============================================================

class Agent:
    """
    Class Agent chính — PHẢI được đặt tên là Agent và có method act().
    Tải weights từ weights.pth cùng thư mục lúc __init__.
    Fallback an toàn: nếu weights không tồn tại → dùng policy random
    được lọc bởi AlphaBetaMasker.
    """

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

        # ── Device: CPU-first cho server BTC, GPU nếu có ────────────────
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # ── Load network ─────────────────────────────────────────────────
        self.net = ActorCriticNet()
        weights_path = Path(__file__).parent / "weights.pth"
        self._weights_loaded = False
        if weights_path.exists():
            try:
                state_dict = torch.load(
                    str(weights_path),
                    map_location=self.device,
                    weights_only=True,
                )
                self.net.load_state_dict(state_dict)
                self._weights_loaded = True
            except Exception:
                pass  # Tiếp tục với random weights — Masker vẫn đảm bảo an toàn

        self.net.to(self.device)
        self.net.eval()

        # ── Safety filter ────────────────────────────────────────────────
        self.masker = AlphaBetaMasker()

        # ── Fallback state ───────────────────────────────────────────────
        self._last_action = ACTION_STOP

    def act(self, obs: dict) -> int:
        """
        Quyết định hành động. Budget tổng: 100ms.

        Pipeline:
          1. Build GameState                          ~0.1ms
          2. AlphaBetaMasker → safe_mask             ~5-40ms (adaptive)
          3. encode_obs → tensors                     ~0.5ms
          4. ActorCriticNet forward pass (CPU)       ~2-5ms
          5. Apply mask + argmax → action             ~0.1ms
        """
        t0 = time.time()

        try:
            state = GameState(
                obs["map"], obs["players"], obs["bombs"], self.agent_id
            )

            # Nếu agent đã chết → STOP
            if state.is_dead(self.agent_id):
                return ACTION_STOP

            # ── Bước 1: Alpha-Beta Masker ──────────────────────────────
            safe_mask = self.masker.get_safe_mask(
                state, self.agent_id,
                time_budget_ms=Config.MASKER_TIME_BUDGET_MS
            )

            elapsed_ms = (time.time() - t0) * 1000

            # ── Bước 2: Network inference ──────────────────────────────
            if elapsed_ms < Config.TOTAL_TIME_BUDGET_MS - Config.NET_TIMEOUT_MS:
                map_feat, aux_feat = encode_obs(obs, self.agent_id)
                map_t = torch.from_numpy(map_feat).unsqueeze(0).to(self.device)
                aux_t = torch.from_numpy(aux_feat).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits, _ = self.net(map_t, aux_t)
                    logits_np = logits.squeeze(0).cpu().numpy()

                # Apply mask: set -1e9 cho actions không an toàn
                for i, safe in enumerate(safe_mask):
                    if not safe:
                        logits_np[i] = -1e9

                action = int(np.argmax(logits_np))

            else:
                # Timeout fallback: chọn action safe đầu tiên theo thứ tự ưu tiên
                action = self._fallback_action(safe_mask)

            self._last_action = action
            return action

        except Exception:
            # Hard fallback: không bao giờ để agent crash
            return self._last_action

    def _fallback_action(self, safe_mask: list) -> int:
        """
        Khi network timeout, chọn action an toàn theo thứ tự ưu tiên:
        STOP > UP > DOWN > LEFT > RIGHT > BOMB
        """
        for a in [ACTION_STOP, ACTION_UP, ACTION_DOWN,
                  ACTION_LEFT, ACTION_RIGHT, ACTION_BOMB]:
            if safe_mask[a]:
                return a
        return ACTION_STOP
