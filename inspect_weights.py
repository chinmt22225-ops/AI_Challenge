import torch, os, datetime
from pathlib import Path

path = 'd:/AI_CHALLENGE/weights.pth'
stat = os.stat(path)
print('=== File Info ===')
print(f'Size    : {stat.st_size:,} bytes ({stat.st_size/1024/1024:.2f} MB)')
print(f'Modified: {datetime.datetime.fromtimestamp(stat.st_mtime)}')
print()

# Load and inspect
ckpt = torch.load(path, map_location='cpu', weights_only=True)
print('=== Checkpoint Structure ===')
print(f'Type: {type(ckpt)}')
keys = list(ckpt.keys())
print(f'Total keys: {len(keys)}')
for k in keys[:12]:
    v = ckpt[k]
    print(f'  {k}: {tuple(v.shape)} | {v.dtype}')
if len(keys) > 12:
    print(f'  ... ({len(keys)-12} more)')
print()

# NaN/Inf check
nan_keys = [k for k, v in ckpt.items() if torch.isnan(v).any() or torch.isinf(v).any()]
if nan_keys:
    print(f'[WARN] NaN/Inf in: {nan_keys}')
else:
    print('[OK] Weights CLEAN - no NaN/Inf')

# Stats
all_params = torch.cat([v.flatten().float() for v in ckpt.values()])
print(f'Total params: {len(all_params):,}')
print(f'Weight stats: mean={all_params.mean():.4f}, std={all_params.std():.4f}')
print(f'              min={all_params.min():.4f}, max={all_params.max():.4f}')
print()

# So sanh timestamp voi cac file weights khac
print('=== Lich su cac file weights ===')
import glob
for f in sorted(glob.glob('d:/AI_CHALLENGE/weights*.pth'), key=os.path.getmtime):
    s = os.stat(f)
    print(f'  {Path(f).name:<35} | {s.st_size/1024/1024:6.2f} MB | {datetime.datetime.fromtimestamp(s.st_mtime)}')
