import sys, torch, numpy as np
sys.path.insert(0, '.')
from train import PPOTrainer, GameEnv, ActorCriticNet

device = torch.device('cpu')
print('[Test] Simulating PPO rollout (BatchNorm NaN fix test)...')

net = ActorCriticNet().to(device)
net.train()
print('  net.training before collect =', net.training, ' (True; collect_rollout will switch to eval)')

trainer = PPOTrainer(net, device, rollout_steps=50)
env = GameEnv(seed=1)

try:
    buf = trainer.collect_rollout(env, my_id=0, episode=0, baseline_types=['random'])
    print('  collect_rollout OK:', len(buf), 'steps')
    print('  net.training after collect =', net.training, ' (should be True = restored)')
    assert net.training, 'net should be back in train mode after collect'
    assert len(buf) == 50, 'Expected 50 steps, got ' + str(len(buf))

    lps = buf.log_probs
    nan_count = sum(1 for x in lps if np.isnan(x) or np.isinf(x))
    print('  NaN in log_probs:', nan_count, '(should be 0)')
    assert nan_count == 0, 'NaN in log_probs!'

    metrics = trainer.update(buf)
    pl = metrics['policy_loss']
    vl = metrics['value_loss']
    en = metrics['entropy']
    print('  update() OK: pl={:.4f}, vl={:.4f}, ent={:.4f}'.format(pl, vl, en))
    nan_metrics = [k for k, v in metrics.items() if np.isnan(v) or np.isinf(v)]
    print('  NaN metrics:', nan_metrics, '(should be empty)')
    assert not nan_metrics, 'NaN in metrics: ' + str(nan_metrics)

    print()
    print('=== PASS: BatchNorm NaN fix VERIFIED! ===')

except Exception as e:
    print('  FAIL:', e)
    import traceback
    traceback.print_exc()
