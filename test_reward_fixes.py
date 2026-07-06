"""
Test 2 logic fixes:
  Fix 1: BOMB_SMART mien BOMB_SPAM
  Fix 2: Khong phat COWARD khi agent vua dat bom
"""
import sys, numpy as np
sys.path.insert(0, '.')
from train import GameEnv, RewardConfig, ACTION_BOMB, ACTION_STOP, ACTION_LEFT

print("=== Test Fix 1: BOMB_SMART mien SPAM ===\n")

# Setup: Agent 0 va Agent 1 sat nhau (Manhattan dist = 1 <= 3)
env = GameEnv(seed=42)
env.reset()
env.players[0].x, env.players[0].y = 3, 3
env.players[1].x, env.players[1].y = 3, 4  # dist = 1
env.players[2].alive = False
env.players[3].alive = False

# Agent 0 dat bom → nen nhan BOMB_SMART (+50), KHONG bi SPAM (-5)
_, rewards, _, _ = env.step([ACTION_BOMB, ACTION_STOP, ACTION_STOP, ACTION_STOP])
r = rewards[0]

# Ket qua mong doi:
#   BOMB_SMART = +50 (co duong thoat)
#   TIME       = -0.01
#   SPAM       = KHONG bi tru (da duoc mien)
expected_smart = RewardConfig.BOMB_SMART + RewardConfig.SURVIVE_STEP  # 50 - 0.01 = 49.99
expected_spam  = RewardConfig.BOMB_SMART + RewardConfig.SURVIVE_STEP + RewardConfig.BOMB_SPAM  # 50 - 0.01 - 5 = 44.99

print(f"Reward Agent 0: {r:.2f}")
print(f"  Neu co fix -> mong doi ~{expected_smart:.2f} (BOMB_SMART + TIME, KHONG SPAM)")
print(f"  Neu bug cu -> ~{expected_spam:.2f} (BOMB_SMART + TIME + SPAM)")

assert abs(r - expected_smart) < 3.0, (
    f"Fix 1 FAIL: reward={r:.2f}, expected ~{expected_smart:.2f}"
    " (SPAM van bi tru du da co BOMB_SMART)"
)
print("  PASS: BOMB_SMART duoc mien SPAM!\n")


print("=== Test Fix 2: Khong phat COWARD khi dat bom ===\n")

# Setup: 2 agents con song, agent 0 va 1 dung sat nhau (dist = 2)
# Agent 0 dat bom tai cho dung -> bom do la vat can, phai bi loai khi tinh Dijkstra moi
env2 = GameEnv(seed=99)
env2.reset()
env2.players[0].x, env2.players[0].y = 5, 5
env2.players[1].x, env2.players[1].y = 5, 7   # dist Manhattan = 2
env2.players[2].alive = False
env2.players[3].alive = False

# Agent 0 dat bom tai (5,5). Sau khi dat, (5,5) la bom -> tao vat can
# Neu KHONG co fix, new_dijkstra tu (5,5) se bi (5,5) la bom chặn -> khong tinh duoc
# Agent phai di chuyen truoc (nhung o day action = BOMB nen agent ko di -> van o (5,5))
# Khoang cach cũ: (5,5) -> (5,7) qua co => = 2
# Khoang cach moi (BUG): (5,5) bi chặn bom tai chinh no -> path bi can -> >= 2 ảo
# Khoang cach moi (FIX): loai bom vua dat, tinh duoc = 2 (khong thay doi) -> ko bi COWARD

# Kiem tra: reward khong chua COWARD_PENALTY khi dat bom
_, rewards2, _, _ = env2.step([ACTION_BOMB, ACTION_STOP, ACTION_STOP, ACTION_STOP])
r2 = rewards2[0]

print(f"Reward Agent 0 khi dat bom (2 agents con song): {r2:.2f}")
# Neu co COWARD_PENALTY: r <= BOMB_SMART + SURVIVE + COWARD = 50 - 0.01 - 5 = 44.99
# Neu khong co COWARD:   r >= BOMB_SMART + SURVIVE           = 50 - 0.01    = 49.99
coward_threshold = RewardConfig.BOMB_SMART + RewardConfig.SURVIVE_STEP + RewardConfig.COWARD_PENALTY
no_coward_val    = RewardConfig.BOMB_SMART + RewardConfig.SURVIVE_STEP

print(f"  Neu bi phat COWARD oan -> reward ~{coward_threshold:.2f}")
print(f"  Neu fix dung         -> reward ~{no_coward_val:.2f}")

# Kiem tra khong co COWARD_PENALTY (tolerance 3 vi co the co HUNTING hay khong)
assert r2 > coward_threshold + 0.1, (
    f"Fix 2 FAIL: reward={r2:.2f} qua thap, co the bi phat COWARD oan!"
)
print("  PASS: Khong bi phat COWARD oan khi dat bom!\n")


print("=== Kiem tra tong hop: Spam bom xa dich van bi phat ===\n")

# Agent dat bom khi dich O XA (dist > 3) -> phai bi SPAM binh thuong
env3 = GameEnv(seed=7)
env3.reset()
env3.players[0].x, env3.players[0].y = 1, 1
env3.players[1].x, env3.players[1].y = 11, 11  # dist = 20 > 3
env3.players[2].alive = False
env3.players[3].alive = False

_, rewards3, _, _ = env3.step([ACTION_BOMB, ACTION_STOP, ACTION_STOP, ACTION_STOP])
r3 = rewards3[0]
print(f"Reward khi spam bom xa dich (dist=20): {r3:.2f}")
# Khong co BOMB_SMART -> phai co SPAM: ~-0.01 - 5 = -5.01
expected_spam_only = RewardConfig.SURVIVE_STEP + RewardConfig.BOMB_SPAM
print(f"  Mong doi ~{expected_spam_only:.2f} (TIME + SPAM, khong co BOMB_SMART)")
assert abs(r3 - expected_spam_only) < 3.0, (
    f"FAIL: spam xa dich, reward={r3:.2f}, expected ~{expected_spam_only:.2f}"
)
print("  PASS: Spam bom xa dich van bi phat dung!\n")


print("=" * 50)
print("=== TAT CA FIX VERIFIED! ===")
print("=" * 50)
print("\nTom tat:")
print("  Fix 1: BOMB_SMART (+50) -> mien SPAM (-5)  [OK]")
print("  Fix 2: Dat bom -> khong phat COWARD oan     [OK]")
print("  Bao ve: Spam xa dich -> van bi -5 SPAM      [OK]")
