import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from state import GameState
from evaluator import Evaluator
from config import Config

class Agent:
    team_id = "AlphaBetaAgent"
    
    def __init__(self, agent_id):
        self.agent_id = int(agent_id)
        
    def act(self, obs):
        start_time = time.time()
        
        # 1. Parse observation
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]
        
        state = GameState(grid, players, bombs, self.agent_id)
        
        # 2. Iterative Deepening Alpha-Beta
        best_action = 0
        
        legal_actions = state.get_legal_actions(self.agent_id)
        if not legal_actions:
            return 0
        if len(legal_actions) == 1:
            return legal_actions[0]
            
        # Ưu tiên test BOMB sớm nếu có thể (để cắt tỉa Alpha-Beta tốt hơn)
        if 5 in legal_actions:
            legal_actions.remove(5)
            legal_actions.insert(0, 5)
        
        for depth in range(1, Config.MAX_DEPTH + 1):
            alpha = float('-inf')
            beta = float('inf')
            current_best_action = legal_actions[0]
            current_max_eval = float('-inf')
            
            # Cờ đánh dấu nếu search bị ngắt giữa chừng
            search_interrupted = False
            
            for action in legal_actions:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > Config.TIME_LIMIT_MS:
                    search_interrupted = True
                    break
                    
                successor = state.generate_successor(self.agent_id, action)
                action_reward = Evaluator.evaluate_action_heuristic(state, action)
                
                # Gọi alpha_beta cho lượt của Minimizer
                eval_val = action_reward + self.alpha_beta(successor, depth - 1, alpha, beta, False, start_time)
                
                if eval_val > current_max_eval:
                    current_max_eval = eval_val
                    current_best_action = action
                    
                alpha = max(alpha, eval_val)
                
            # Cập nhật best_action nếu hoàn thành xong lớp hiện tại
            # Hoặc nếu là lớp đầu tiên thì vẫn lấy best_action tìm được
            if not search_interrupted or depth == 1:
                best_action = current_best_action
            
            if search_interrupted:
                break
                
        return best_action

    def alpha_beta(self, state, depth, alpha, beta, maximizing_player, start_time):
        if depth == 0 or state.is_terminal():
            return Evaluator.evaluate(state, self.agent_id)
            
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > Config.TIME_LIMIT_MS:
            return Evaluator.evaluate(state, self.agent_id)

        if maximizing_player:
            max_eval = float('-inf')
            legal_actions = state.get_legal_actions(self.agent_id)
            if not legal_actions:
                return Evaluator.evaluate(state, self.agent_id)
                
            for action in legal_actions:
                successor = state.generate_successor(self.agent_id, action)
                action_reward = Evaluator.evaluate_action_heuristic(state, action)
                eval_val = action_reward + self.alpha_beta(successor, depth - 1, alpha, beta, False, start_time)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            # Lượt của đối thủ (Minimizer)
            # Với game 4 người, branching factor cho đối thủ là 6^3 = 216 nhánh mỗi bước.
            # Điều này là bất khả thi trong 100ms.
            # Vì vậy, ta dùng chiến lược "Môi trường tự tiến": giả định đối thủ giữ nguyên vị trí,
            # các quả bom trên sân tự giảm timer và nổ. 
            # Điều này giúp Agent dự đoán chính xác được sự đe dọa của bom nổ trong tương lai.
            min_eval = float('inf')
            
            # Action 0 = IDLE. Agent đứng yên, môi trường tự trôi đi 1 step
            successor = state.generate_successor(self.agent_id, 0) 
            
            eval_val = self.alpha_beta(successor, depth - 1, alpha, beta, True, start_time)
            min_eval = min(min_eval, eval_val)
            beta = min(beta, eval_val)
            # if beta <= alpha: break (chỉ có 1 nhánh nên bỏ qua)
            
            return min_eval
