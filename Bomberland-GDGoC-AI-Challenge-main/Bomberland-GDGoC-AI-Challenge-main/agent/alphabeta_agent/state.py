# state.py
import numpy as np

# Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: IDLE, 5: BOMB
MOVES = {
    0: (0, 0),    # Trong simple_rule_agent, UP = (0, 0) ? Không, khoan.
    1: (-1, 0),   # DOWN ?
    2: (1, 0),    # LEFT ?
    3: (0, -1),   # RIGHT ?
    4: (0, 1),    # IDLE ?
}
# Chờ chút, kiểm tra lại MOVES chuẩn:
# 0: UP (x, y-1), 1: DOWN (x, y+1), 2: LEFT (x-1, y), 3: RIGHT (x+1, y), 4: IDLE (x, y), 5: BOMB.
# Từ simple_rule_agent:
# MOVES = { 0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1) } -> Đây là grid[x, y]
# Vậy x là hàng (row), y là cột (col).
# 1: (-1, 0) -> Up, 2: (1, 0) -> Down, 3: (0, -1) -> Left, 4: (0, 1) -> Right, 0: Idle.
# Để chắc chắn, tôi sẽ định nghĩa lại action cho chuẩn.
# Theo engine chuẩn: 0: IDLE, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT, 5: BOMB
ACTION_IDLE = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_BOMB = 5

class GameState:
    def __init__(self, grid, players, bombs, my_agent_id):
        self.grid = np.copy(grid) # 2D array
        # players: list of [x, y, is_alive, bombs_left, bomb_bonus]
        self.players = [list(p) for p in players] 
        self.bombs = [list(b) for b in bombs]
        self.my_agent_id = my_agent_id
        
    def clone(self):
        return GameState(self.grid, self.players, self.bombs, self.my_agent_id)
        
    def _in_bounds(self, x, y):
        return 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]
        
    def _is_passable(self, x, y):
        if not self._in_bounds(x, y):
            return False
        # 0: empty, 3: item radius, 4: item capacity.
        # Tường = 1, Hộp = 2
        return self.grid[x, y] in [0, 3, 4]

    def _get_danger_zones(self):
        # Trả về tập hợp các ô sắp nổ (timer <= 2)
        danger_tiles = set()
        for b in self.bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3]) if len(b) > 3 else -1
            radius = 2 # default
            if 0 <= owner_id < len(self.players):
                radius = max(1, int(self.players[owner_id][4]) + 1)
                
            # Blast logic
            danger_tiles.add((bx, by))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for r in range(1, radius + 1):
                    nx, ny = bx + dx*r, by + dy*r
                    if not self._in_bounds(nx, ny):
                        break
                    cell = self.grid[nx, ny]
                    if cell == 1: # Wall
                        break
                    danger_tiles.add((nx, ny))
                    if cell == 2: # Box blocks explosion but is destroyed
                        break
        return danger_tiles

    def get_legal_actions(self, agent_id):
        if agent_id >= len(self.players) or self.players[agent_id][2] != 1:
            return [ACTION_IDLE]
            
        p = self.players[agent_id]
        px, py = int(p[0]), int(p[1])
        bombs_left = int(p[3])
        
        # occupied positions by other alive players
        occupied = {(int(op[0]), int(op[1])) for i, op in enumerate(self.players) if i != agent_id and op[2] == 1}
        # bomb positions
        bomb_pos = {(int(b[0]), int(b[1])) for b in self.bombs}
        
        actions = [ACTION_IDLE]
        
        # Move action mapping
        # Dựa trên simple_rule_agent: dx, dy mapping
        # 0: (0,-1) UP? 1: (0,1) DOWN? 
        # Chúng ta sẽ thử đi 4 hướng, dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
        # Nếu theo simple_rule_agent:
        # a=1: (-1, 0), a=2: (1, 0), a=3: (0, -1), a=4: (0, 1)
        # Tôi sẽ làm map cứng:
        DIR_MAP = {
            ACTION_UP: (0, -1), ACTION_DOWN: (0, 1),
            ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)
        }
        # Thực tế hệ tọa độ xy có thể khác, ta dùng DX DY chung:
        VALID_DIRS = {
            1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)
        }
        for a, (dx, dy) in VALID_DIRS.items():
            nx, ny = px + dx, py + dy
            if self._is_passable(nx, ny) and (nx, ny) not in occupied:
                actions.append(a)
                
        # Bomb action
        if bombs_left > 0 and (px, py) not in bomb_pos:
            actions.append(ACTION_BOMB)
            
        return actions

    def generate_successor(self, agent_id, action):
        new_state = self.clone()
        if agent_id >= len(new_state.players) or new_state.players[agent_id][2] != 1:
            return new_state # Đã chết
            
        p = new_state.players[agent_id]
        px, py = int(p[0]), int(p[1])
        
        # Mapping từ simple_rule_agent
        DIR_MAP = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1), 0: (0, 0)}
        
        if action in DIR_MAP:
            dx, dy = DIR_MAP[action]
            nx, ny = px + dx, py + dy
            # Giả định di chuyển thành công (get_legal_actions đã kiểm tra)
            p[0], p[1] = nx, ny
            
            # Nhặt item
            if new_state.grid[nx, ny] == 3: # Item Radius
                p[4] += 1
                new_state.grid[nx, ny] = 0
            elif new_state.grid[nx, ny] == 4: # Item Capacity
                # Nếu capacity được quản lý, tăng max bombs (trong sim đơn giản có thể bỏ qua)
                p[3] += 1 
                new_state.grid[nx, ny] = 0
                
        elif action == ACTION_BOMB:
            p[3] -= 1
            new_state.bombs.append([px, py, 7, agent_id]) # Timer=7
            
        # Tick bombs
        explosions = []
        surviving_bombs = []
        for b in new_state.bombs:
            b[2] -= 1
            if b[2] <= 0:
                explosions.append(b)
            else:
                surviving_bombs.append(b)
        new_state.bombs = surviving_bombs
        
        # Xử lý nổ (đơn giản hóa: chỉ lấy vùng nổ)
        blast_zones = set()
        broken_boxes = set()
        for b in explosions:
            bx, by, _, owner_id = b
            radius = 2
            if 0 <= owner_id < len(new_state.players):
                radius = max(1, int(new_state.players[owner_id][4]) + 1)
                if owner_id == agent_id:
                    new_state.players[agent_id][3] += 1 # Hồi bom
                    
            blast_zones.add((bx, by))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for r in range(1, radius + 1):
                    nx, ny = bx + dx*r, by + dy*r
                    if not new_state._in_bounds(nx, ny): break
                    if new_state.grid[nx, ny] == 1: break # Tường
                    blast_zones.add((nx, ny))
                    if new_state.grid[nx, ny] == 2:
                        broken_boxes.add((nx, ny))
                        break # Hộp cản nổ
                        
        # Phá hộp -> drop item (sim đơn giản coi như xóa hộp, drop ngẫu nhiên hoặc bỏ qua)
        for (bx, by) in broken_boxes:
            new_state.grid[bx, by] = 0 # Xóa hộp (không mô phỏng sinh item random để tránh nhiễu)
            
        # Kill players
        for i, player in enumerate(new_state.players):
            if player[2] == 1 and (int(player[0]), int(player[1])) in blast_zones:
                player[2] = 0 # Die
                
        return new_state

    def is_terminal(self):
        # Trò chơi kết thúc nếu Agent mình chết
        p = self.players[self.my_agent_id]
        if p[2] == 0:
            return True
        return False
