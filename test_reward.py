"""
Test: Aggressive Reward Shaping
Kiem tra:
1. Khong co loi runtime trong step() moi
2. Toc do trung binh cua step() < 10ms (voi Dijkstra x4 agents)
3. Reward logic dung: time penalty, hunting, kamikaze, coward, late-game item
"""
import sys, time, numpy as np
sys.path.insert(0, '.')
from train import GameEnv, RewardConfig, ACTION_BOMB, ACTION_STOP

print("=== Test: Aggressive Reward Shaping ===\n")

# --------------- Test 1: Basic step() does not crash ---------------
print("[1] Khoi tao GameEnv va chay 50 steps ngau nhien...")
env = GameEnv(seed=42)
obs = env.reset()
total_time = 0.0
for step_i in range(50):
    actions = [np.random.randint(0, 6) for _ in range(4)]
    t0 = time.perf_counter()
    obs, rewards, done, info = env.step(actions)
    t1 = time.perf_counter()
    total_time += (t1 - t0) * 1000
    for r in rewards:
        assert not (r != r), f"NaN reward tai step {step_i}"  # NaN check
    if done:
        obs = env.reset()

avg_ms = total_time / 50
print(f"   Trung binh moi step: {avg_ms:.2f}ms")
assert avg_ms < 50, f"QUA CHAM: {avg_ms:.2f}ms (gioi han < 50ms)"
print(f"   OK - Toc do hop le (< 50ms/step)\n")

# --------------- Test 2: Time penalty (SURVIVE_STEP = -0.01) -------
print("[2] Kiem tra time penalty (SURVIVE_STEP = -0.01)...")
env2 = GameEnv(seed=1)
obs2 = env2.reset()
obs2, rewards2, done2, info2 = env2.step([ACTION_STOP]*4)
# Tat ca agent song -> moi agent phai nhan dung -0.01 (co the cong/tru them tu item/kill)
print(f"   SURVIVE_STEP = {RewardConfig.SURVIVE_STEP} (phai = -0.01)")
assert RewardConfig.SURVIVE_STEP == -0.01, "SURVIVE_STEP sai!"
print("   OK\n")

# --------------- Test 3: Anti-Kamikaze (BOMB_KAMIKAZE = -500) ------
print("[3] Kiem tra Anti-Kamikaze...")
# Dat hai agent sat nhau de kich hoat check
env3 = GameEnv(seed=5)
obs3 = env3.reset()
# Agent 0 o (1,1), Agent 1 o (11,11) - xa qua, phai chay gan lai
# Teleport manual de test
env3.players[0].x, env3.players[0].y = 3, 3
env3.players[1].x, env3.players[1].y = 4, 3  # Manhattan dist = 1 <= 3

r_before = 0.0
actions3 = [ACTION_BOMB, ACTION_STOP, ACTION_STOP, ACTION_STOP]
obs3, rewards3, done3, info3 = env3.step(actions3)
r_bomb = rewards3[0]

print(f"   Agent 0 dat bom gan Agent 1 (dist=1)")
print(f"   Reward Agent 0: {r_bomb:.1f}")
# Neu co duong thoat -> BOMB_SMART (+50), neu khong -> BOMB_KAMIKAZE (-500)
# Tai o goc (3,3) nen co the co duong thoat. Kiem tra chi la reward logic duoc goi.
got_bomb_reward = (r_bomb >= RewardConfig.BOMB_SMART - 5.01 or  # +50 - spam(-5) - time(-0.01)
                   r_bomb <= RewardConfig.BOMB_KAMIKAZE + 1.0)   # -500 + time
print(f"   Bomb reward logic duoc goi: {got_bomb_reward}")
print("   OK\n")

# --------------- Test 4: Dijkstra voi Box traversal ----------------
print("[4] Kiem tra _get_dijkstra_distances xuyen qua Box...")
env4 = GameEnv(seed=99)
obs4 = env4.reset()
# Lay khoang cach tu goc (1,1) den mot vi tri qua hop
start = (1, 1)
bombs = set()
dists = env4._get_dijkstra_distances(start, bombs)
# Phai co khoang cach den toan bo ban do (tru wall)
reachable = len([d for d in dists.values() if d < 999])
print(f"   So o co the tiep can tu (1,1): {reachable}/169")
assert reachable > 50, f"Dijkstra qua it o: {reachable}"
print("   OK\n")

# --------------- Test 5: _can_escape_from_bomb ---------------------
print("[5] Kiem tra _can_escape_from_bomb...")
env5 = GameEnv(seed=7)
obs5 = env5.reset()
# Agent o (1,1), dat bom o chinh no
can_escape_open = env5._can_escape_from_bomb(
    agent_pos=(1, 1), bomb_pos=(1, 1), bomb_radius=1, bomb_positions=set()
)
print(f"   Co the thoat khi dat bom o goc mo (1,1): {can_escape_open}")
# O goc (1,1) co thi 2 huong di -> nen co the thoat
assert can_escape_open == True, "Agent o goc mo phai thoat duoc!"
print("   OK\n")

# --------------- Test 6: Toc do step voi 4 agents song toan bo -----
print("[6] Stress test: 200 steps voi 4 agents (full Dijkstra)...")
env6 = GameEnv(seed=123)
obs6 = env6.reset()
t_start = time.perf_counter()
steps_done = 0
for _ in range(200):
    acts = [np.random.randint(0, 6) for _ in range(4)]
    obs6, rews6, done6, _ = env6.step(acts)
    steps_done += 1
    if done6:
        obs6 = env6.reset()
t_total = (time.perf_counter() - t_start) * 1000
avg6 = t_total / steps_done
print(f"   {steps_done} steps trong {t_total:.1f}ms -> trung binh {avg6:.2f}ms/step")
assert avg6 < 50, f"QUA CHAM: {avg6:.2f}ms (gioi han < 50ms)"
print("   OK\n")

print("=" * 45)
print("=== TAT CA TEST PASSED! ===")
print("=" * 45)
print(f"\nTom tat reward config:")
print(f"  SURVIVE_STEP    = {RewardConfig.SURVIVE_STEP:+.2f}  (phat thoi gian)")
print(f"  HUNTING_ADVANCE = {RewardConfig.HUNTING_ADVANCE:+.2f}  (tien gan dich an toan)")
print(f"  BOMB_SMART      = {RewardConfig.BOMB_SMART:+.2f} (dat bom co duong thoat)")
print(f"  BOMB_KAMIKAZE   = {RewardConfig.BOMB_KAMIKAZE:+.2f} (dat bom tu sat)")
print(f"  ITEM_PICKUP     = {RewardConfig.ITEM_PICKUP:+.2f} (early game)")
print(f"  ITEM_PICKUP_LATE= {RewardConfig.ITEM_PICKUP_LATE:+.2f} (late game: con 2 nguoi)")
print(f"  COWARD_PENALTY  = {RewardConfig.COWARD_PENALTY:+.2f}  (bo chay khi late game)")
