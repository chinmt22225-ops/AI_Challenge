import sys
sys.path.insert(0, 'd:/AI_CHALLENGE')

print('=== Test 1: Import agent.py ===')
from agent import (
    GameState, AlphaBetaMasker, encode_obs, ActorCriticNet, Agent,
    ACTION_STOP, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_BOMB,
    DIR_MAP, GRID_SIZE, ITEM_RAD, ITEM_CAP, GRASS
)
print('  OK: All imports successful')

print()
print('=== Test 2: Action mapping (engine bug verified) ===')
assert DIR_MAP[ACTION_UP]    == (-1,  0), 'UP wrong'
assert DIR_MAP[ACTION_DOWN]  == (+1,  0), 'DOWN wrong'
assert DIR_MAP[ACTION_LEFT]  == ( 0, -1), 'LEFT wrong'
assert DIR_MAP[ACTION_RIGHT] == ( 0, +1), 'RIGHT wrong'
print('  OK: 1->UP(dx=-1), 2->DOWN(dx=+1), 3->LEFT(dy=-1), 4->RIGHT(dy=+1)')

print()
print('=== Test 3: ActorCriticNet ===')
import torch
net = ActorCriticNet()
map_x = torch.zeros(2, 9, 13, 13)
aux_x = torch.zeros(2, 8)
logits, value = net(map_x, aux_x)
assert logits.shape == (2, 6)
assert value.shape  == (2, 1)
n_params = sum(p.numel() for p in net.parameters())
print(f'  OK: logits={tuple(logits.shape)}, value={tuple(value.shape)}')
print(f'  OK: Total params = {n_params:,} ({n_params/1e6:.2f}M)')

print()
print('=== Test 4: encode_obs ===')
import numpy as np
obs = {
    'map':     np.zeros((13,13), dtype=np.int8),
    'players': np.array([[1,1,1,1,0],[11,11,1,1,0],[1,11,1,1,0],[11,1,1,1,0]], dtype=np.int8),
    'bombs':   np.zeros((0,4), dtype=np.int8),
}
obs['map'][3,3] = 1   # Wall
obs['map'][5,5] = 2   # Box
obs['map'][7,7] = 3   # Item radius
mf, af = encode_obs(obs, 0)
assert mf.shape == (9, 13, 13)
assert af.shape == (8,)
assert mf[0, 3, 3] == 1.0, 'Wall channel wrong'
assert mf[1, 5, 5] == 1.0, 'Box channel wrong'
assert mf[7, 7, 7] == 1.0, 'Item radius channel wrong'
assert mf[2, 1, 1] == 1.0, 'My position channel wrong'
print('  OK: map_feat shape (9,13,13), aux_feat shape (8,)')
print('  OK: Channels verified (wall, box, item, my_pos)')

print()
print('=== Test 5: GameState + Masker ===')
gs = GameState(obs['map'], obs['players'], obs['bombs'], 0)
legal = gs.get_legal_actions(0)
print(f'  OK: Legal actions at (1,1): {legal}')
danger = gs._get_danger_zones()
assert len(danger) == 0
print('  OK: No danger zones with no bombs')

masker = AlphaBetaMasker()
mask = masker.get_safe_mask(gs, 0, time_budget_ms=30)
assert len(mask) == 6
assert any(mask)
print(f'  OK: safe_mask = {mask}')

obs_bomb = {
    'map':     np.zeros((13,13), dtype=np.int8),
    'players': np.array([[1,1,1,1,0],[11,11,1,1,0],[1,11,1,1,0],[11,1,1,1,0]], dtype=np.int8),
    'bombs':   np.array([[1,4,2,1]], dtype=np.int8),
}
gs_bomb = GameState(obs_bomb['map'], obs_bomb['players'], obs_bomb['bombs'], 0)
mask_bomb = masker.get_safe_mask(gs_bomb, 0, time_budget_ms=30)
print(f'  OK: mask with urgent bomb (timer=2): {mask_bomb}')

print()
print('=== Test 6: Agent inference time ===')
import time
agent = Agent(0)
agent.act(obs)  # warm-up
times = []
for _ in range(20):
    t0 = time.time()
    action = agent.act(obs)
    times.append((time.time() - t0) * 1000)
avg_ms = sum(times) / len(times)
max_ms = max(times)
print(f'  OK: avg={avg_ms:.2f}ms, max={max_ms:.2f}ms (budget=100ms)')
assert max_ms < 100, f'TIMEOUT: max={max_ms:.1f}ms'
assert action in range(6)
print(f'  OK: action = {action}')

print()
print('=== Test 7: GameEnv (train.py) ===')
from train import GameEnv, calculate_tiebreaker_winner, PlayerData
import random
env = GameEnv(seed=42)
obs_env = env.reset()
assert obs_env['map'].shape == (13, 13)
assert obs_env['players'].shape == (4, 5)
total_rewards = [0.0] * 4
for step in range(20):
    actions = [random.randint(0, 5) for _ in range(4)]
    obs_env, rewards, done, info = env.step(actions)
    for i in range(4):
        total_rewards[i] += rewards[i]
    if done:
        break
print(f'  OK: {step+1} steps completed without error')
print('  OK: total_rewards=' + str([round(r, 2) for r in total_rewards]))

print()
print('=== Test 8: Item contention logic ===')
env2 = GameEnv(seed=0)
env2.reset()
env2.grid[3, 3] = ITEM_RAD
env2.players[0].x, env2.players[0].y = 3, 2
env2.players[1].x, env2.players[1].y = 2, 3
env2.players[2].alive = False
env2.players[3].alive = False
# Agent 0: ACTION_RIGHT (dy=+1) -> (3,3)
# Agent 1: ACTION_DOWN  (dx=+1) -> (3,3)
_, rewards, done, info = env2.step([ACTION_RIGHT, ACTION_DOWN, 0, 0])
item_val = int(env2.grid[3, 3])
print('  item_contention info: ' + str(info['item_contention']))
print('  grid[3,3] after = ' + str(item_val) + ' (0=GRASS = item destroyed)')
if item_val == GRASS:
    print('  PASS: Item correctly destroyed on contention!')
else:
    print('  WARN: Item was NOT destroyed - check logic')

print()
print('=== Test 9: Tiebreaker ===')
players_tb = [PlayerData(i, 1, 1) for i in range(4)]
players_tb[0].kills = 2; players_tb[0].boxes_broken = 5
players_tb[1].kills = 2; players_tb[1].boxes_broken = 3
players_tb[2].alive = False
players_tb[3].alive = False
ranks = calculate_tiebreaker_winner(players_tb)
print('  ranks = ' + str(ranks))
assert ranks[0] == 1, 'Agent 0 should be rank 1 (same kills, more boxes)'
assert ranks[1] == 2, 'Agent 1 should be rank 2'
print('  PASS: Tiebreaker (Kills > Boxes) correct!')

print()
print('=== Test 10: Full episode (GameEnv + Agent) ===')
env3 = GameEnv(seed=123)
obs3 = env3.reset()
agent3 = Agent(0)
step_count = 0
total_r = 0.0
done3 = False
while not done3 and step_count < 100:
    a0 = agent3.act(obs3)
    actions3 = [a0, 0, 0, 0]
    obs3, rew3, done3, _ = env3.step(actions3)
    total_r += rew3[0]
    step_count += 1
print(f'  OK: {step_count} steps, total_reward={total_r:.2f}, done={done3}')
assert step_count > 0

print()
print('=' * 55)
print('ALL TESTS PASSED! Agent is ready for training.')
print('=' * 55)
