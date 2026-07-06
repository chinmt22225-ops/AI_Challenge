# evaluator.py
from collections import deque
from config import Config

class Evaluator:
    @staticmethod
    def evaluate(state, agent_id):
        if agent_id >= len(state.players):
            return Config.SCORE_DEATH
            
        me = state.players[agent_id]
        if me[2] == 0:
            return Config.SCORE_DEATH
            
        score = Config.SCORE_SURVIVE
        
        my_pos = (int(me[0]), int(me[1]))
        danger_tiles = state._get_danger_zones()
        
        if my_pos in danger_tiles:
            # Rất nguy hiểm, phạt nặng
            score += Config.SCORE_IN_DANGER
            
        # Thưởng khi phá hộp (ước lượng dựa trên số hộp có thể bị vỡ bởi bom vừa đặt)
        # Vì ta giả lập tick bom nên hơi khó biết chính xác hộp nào vừa vỡ, 
        # nhưng ta có thể đếm số hộp bị thiếu so với ban đầu? 
        # Cách tốt hơn là thưởng trực tiếp lúc Action == BOMB trong evaluate_action, 
        # hoặc đếm tổng số lượng hộp trên bản đồ bị giảm.
        # Ở đây ta ưu tiên BFS tới vật phẩm và kẻ thù.
        
        # BFS để tìm khoảng cách
        items, enemies, boxes = Evaluator._bfs_distances(state, my_pos, danger_tiles)
        
        # Cộng điểm vật phẩm
        if items:
            nearest_item_dist = min(items)
            if nearest_item_dist < Config.MAX_DISTANCE_REWARD:
                score += Config.WEIGHT_DISTANCE_ITEM * (Config.MAX_DISTANCE_REWARD - nearest_item_dist)
                
        # Cộng điểm tiếp cận kẻ thù (nếu đã đủ bom)
        if enemies:
            nearest_enemy_dist = min(enemies)
            if nearest_enemy_dist < Config.MAX_DISTANCE_REWARD:
                score += Config.WEIGHT_DISTANCE_ENEMY * (Config.MAX_DISTANCE_REWARD - nearest_enemy_dist)
                
        # Cộng điểm nếu đứng cạnh hộp gỗ (để khuyến khích đặt bom)
        if boxes:
            nearest_box_dist = min(boxes)
            if nearest_box_dist == 1:
                # Đứng cạnh hộp là cơ hội tốt để đặt bom
                score += Config.SCORE_BOMB_BREAK_BOX * 0.5 
                
        return score

    @staticmethod
    def _bfs_distances(state, start_pos, danger_tiles):
        q = deque([(start_pos, 0)])
        seen = {start_pos}
        
        item_dists = []
        enemy_dists = []
        box_dists = []
        
        # Vị trí các người chơi khác
        enemy_positions = {(int(p[0]), int(p[1])) for i, p in enumerate(state.players) if i != state.my_agent_id and p[2] == 1}
        # Vị trí bom (không thể đi qua)
        bomb_pos = {(int(b[0]), int(b[1])) for b in state.bombs}
        
        while q:
            pos, dist = q.popleft()
            
            x, y = pos
            if state.grid[x, y] in [3, 4]:
                item_dists.append(dist)
                
            if pos in enemy_positions:
                enemy_dists.append(dist)
                
            # Đếm hộp
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if state._in_bounds(nx, ny) and state.grid[nx, ny] == 2:
                    box_dists.append(dist + 1)
            
            if dist > Config.MAX_DISTANCE_REWARD:
                continue
                
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                npos = (nx, ny)
                if npos in seen:
                    continue
                
                # Không đi vào tường, hộp, bom, hoặc vùng nguy hiểm (trừ khi đang bế tắc)
                if not state._is_passable(nx, ny):
                    continue
                if npos in bomb_pos:
                    continue
                if npos in danger_tiles:
                    # Hạn chế BFS đi xuyên qua vùng nguy hiểm
                    continue
                    
                seen.add(npos)
                q.append((npos, dist + 1))
                
        return item_dists, enemy_dists, box_dists

    @staticmethod
    def evaluate_action_heuristic(state, action):
        """Thêm các điểm thưởng chuyên biệt cho từng loại hành động ngay khi thực hiện"""
        reward = 0
        me = state.players[state.my_agent_id]
        px, py = int(me[0]), int(me[1])
        radius = max(1, int(me[4]) + 1)
        
        if action == 5: # BOMB
            # Tính toán xem bom này phá được bao nhiêu hộp
            boxes_hit = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for r in range(1, radius + 1):
                    nx, ny = px + dx*r, py + dy*r
                    if not state._in_bounds(nx, ny): break
                    if state.grid[nx, ny] == 1: break
                    if state.grid[nx, ny] == 2:
                        boxes_hit += 1
                        break
            if boxes_hit > 0:
                # Trọng số thưởng phá hộp và rơi vật phẩm
                reward += Config.SCORE_BOMB_BREAK_BOX * boxes_hit
                
        return reward
