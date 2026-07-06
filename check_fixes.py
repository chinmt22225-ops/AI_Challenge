with open('train.py', 'r', encoding='utf-8') as f:
    content = f.read()

checks = [
    ('net.eval() in collect_rollout', 'self.net.eval()'),
    ('NaN guard for logits',          'np.isnan(logits_np)'),
    ('probs clamp before Categorical','probs.clamp(min=1e-8)'),
    ('-1e9 masking in update()',      'logits_m[~mb_mask] = -1e9'),
    ('net.train() restored',         'self.net.train()'),
]
all_ok = True
for name, token in checks:
    found = token in content
    status = 'OK' if found else 'MISSING'
    print(f'  [{status}] {name}')
    if not found:
        all_ok = False

print()
if all_ok:
    print('=== RESULT: ALL FIXES PRESENT')
else:
    print('=== RESULT: MISSING FIXES')
