import os
import json
import csv
import shutil
import datetime
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from loss import *
from sampler import *

def set_random_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# Parse args & load config
# ----------------------------
parser = argparse.ArgumentParser(description='Single-GPU train/val with config-driven settings')
parser.add_argument('config_path', type=str, help='Path to JSON config')
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    cfg = json.load(f)

# Read core settings from config
var_name            = cfg['variable_name']           # e.g. 'tp' or 'vo'
random_seed         = cfg.get('random_seed', 42)
norm_factors_path   = cfg['norm_factors_path']      # path to norm_factors.json
latlon_path         = cfg['latlon_path']            # path to latlon_.npz
pretrained_model    = cfg.get('pretrained_model_path')
img_resolution      = cfg['image_resolution']       # e.g. 128

name                = cfg['name']
batch_size          = cfg['batch_size']
num_epochs          = cfg['num_epochs']
weight_decay        = cfg['weight_decay']
learning_rate       = cfg['learning_rate']
filters             = cfg['filters']
noise               = cfg['noise']
model_choice        = cfg['model']
train_cfg           = cfg['train']
val_cfg             = cfg['val']
dataset_path        = cfg['train_path']
# ----------------------------
# Setup
# ----------------------------
set_random_seed(random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

result_path = Path(f'./models/{name}_{datetime.now().strftime("%m%d%H%M")}')
result_path.mkdir(parents=True, exist_ok=True)
shutil.copy(args.config_path, result_path / 'config.json')

# ----------------------------
# Load normalization stats
# ----------------------------
with open(norm_factors_path, 'r') as f:
    stats = json.load(f)
mean_data = torch.tensor([stats[k]['mean'] for k in stats if k == var_name], device=device)
std_data  = torch.tensor([stats[k]['std']  for k in stats if k == var_name], device=device)


def renormalize(x, mean_ar=mean_data, std_ar=std_data):
    return x * std_ar[None,:,None,None] + mean_ar[None,:,None,None]

# ----------------------------
# Load lat/lon
# ----------------------------
lat, lon = np.load(latlon_path).values()
if var_name != 'vo':
    valid_time_path = cfg['valid_time_path']
    tp_threshold    = cfg['tp_threshold']
    dt_array = np.load(valid_time_path)  
    valid_times = pd.to_datetime(dt_array)
    day_norm  = valid_times.dayofyear.to_numpy(dtype=np.float32) / 365.0
    hour_norm = valid_times.hour.to_numpy(dtype=np.float32)      / 24.0
# ----------------------------
# Train/val split counts from stats
# ----------------------------
n_samples = int(stats['n_time_steps'])
n_train   = int(n_samples * 0.9)
n_val     = n_samples - n_train

# ----------------------------
# Common dataset kwargs
# ----------------------------
common_kwargs = {
    'var_name':         var_name,
    'tp_threshold':     tp_threshold,
    'dataset_path':     dataset_path,
    'sample_counts':    (n_samples, n_train, n_val),
    'dimensions':       (1, len(lat), len(lon)),
    'norm_factors':     np.stack([mean_data.cpu().numpy(), std_data.cpu().numpy()], axis=0),
    'device':           device,
    'dtype':            'float32',
    'random_lead_time': 0,
}

# ----------------------------
# Build datasets & loaders
# ----------------------------
train_ds = ERA5Dataset(
    lead_time=train_cfg['t_max'], dataset_mode='train',
    **{**common_kwargs,
       'max_horizon':        train_cfg['t_max'],
       'spacing':            train_cfg['spacing'],
       'conditioning_times': train_cfg['conditioning_times'],
       'lead_time_range':    [train_cfg['t_min'], train_cfg['t_max'], train_cfg['delta_t']],
    }
)
val_ds = ERA5Dataset(
    lead_time=val_cfg['t_max'], dataset_mode='val',
    **{**common_kwargs,
       'max_horizon':        val_cfg['t_max'],
       'spacing':            val_cfg['spacing'],
       'conditioning_times': val_cfg['conditioning_times'],
       'lead_time_range':    [val_cfg['t_min'], val_cfg['t_max'], val_cfg['delta_t']],
    }
)

train_sampler = DynamicKBatchSampler(
    train_ds, batch_size=batch_size, drop_last=True,
    t_update_callback=get_uniform_t_dist_fn(train_cfg['t_min'], train_cfg['t_max'], train_cfg['delta_t']),
    shuffle=True
)
val_sampler = DynamicKBatchSampler(
    val_ds, batch_size=batch_size, drop_last=True,
    t_update_callback=get_uniform_t_dist_fn(val_cfg['t_min'], val_cfg['t_max'], val_cfg['delta_t']),
    shuffle=False
)

train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler)

# ----------------------------
# Model, loss, optimizer, scheduler
# ----------------------------
input_times = 1 + len(train_cfg['conditioning_times'])
time_emb    = 1 if 'continuous' in model_choice else 0
model = EDMPrecond(
    filters=filters,
    img_channels=input_times,
    out_channels=1,
    img_resolution=img_resolution,
    time_emb=time_emb,
    sigma_data=1, sigma_min=0.02, sigma_max=88
).to(device)

if pretrained_model:
    model.load_state_dict(torch.load(pretrained_model))

loss_fn   = WGCLoss(lat, lon, device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
warmup    = optim.lr_scheduler.LinearLR(optimizer,
                start_factor=1e-3, end_factor=1.0, total_iters=1000)

# ----------------------------
# Logging & training loop
# ----------------------------
log_csv = result_path / 'training_log.csv'
with open(log_csv, 'w', newline='') as f:
    csv.writer(f).writerow([
        'Epoch','Train Loss','Val Loss',
        'Base MAE','Pred MAE',
        'Base Mean','Pred Mean','True Mean'
    ])

C = len(val_cfg['conditioning_times'])
T = torch.tensor(val_cfg['conditioning_times'], device=device)

best_pmae = float('inf')

for epoch in range(1, num_epochs + 1):
    # Training
    model.train()
    total_train = 0.0
    for past, future, t_label, start_idx in tqdm(train_loader, desc=f"Epoch {epoch} Train", ncols=80):
        start_idx = start_idx.to(device)
        past, future = past.to(device), future.to(device)
        start_idx    = start_idx.to(device)
        t = t_label.to(device).float()

        # lead time token
        t_norm    = t / max(train_cfg['t_max'], max(train_cfg['conditioning_times']))
        t_norm += torch.randn_like(t_norm) * noise
        time_lead = t_norm.unsqueeze(1)

        # calendar features
        x_idx = start_idx.unsqueeze(1) + torch.tensor(train_cfg['conditioning_times'], device=device)
        y_idx = start_idx.unsqueeze(1) + t.unsqueeze(1)
        all_idx = torch.cat([x_idx, y_idx], dim=1).long().cpu().numpy()
        if var_name == 'vo':
            time_feat = (all_idx % 200).float() / 200.0
        else:
            day_feat  = torch.from_numpy(day_norm[all_idx]).to(device)   # (B, C+1)
            hour_feat = torch.from_numpy(hour_norm[all_idx]).to(device)  # (B, C+1)
            time_feat = torch.cat([day_feat, hour_feat], dim=1)
        time_feat = time_feat + torch.randn_like(time_feat) * noise
        optimizer.zero_grad()
        loss = loss_fn(model, future, past, time_labels=time_lead, augment_labels=time_feat)
        loss.backward()
        optimizer.step()
        warmup.step()
        total_train += loss.item()
    avg_train = total_train / len(train_loader)

    model.eval()
    acc = {k: 0. for k in ['vl','bmae','pmae','bmean','pmean','tmean','cnt']}
    with torch.no_grad():
        for past, future, t_label, start_idx in tqdm(val_loader, desc=f"Epoch {epoch} Val", ncols=80):
            past, future = past.to(device), future.to(device)
            start_idx    = start_idx.to(device)
            t = t_label.to(device).float()
    
            t_norm    = t / max(val_cfg['t_max'], max(val_cfg['conditioning_times']))
            t_norm += torch.randn_like(t_norm) * noise
            time_lead = t_norm.unsqueeze(1)
    
            x_idx = start_idx.unsqueeze(1) + torch.tensor(val_cfg['conditioning_times'], device=device)
            y_idx = start_idx.unsqueeze(1) + t.unsqueeze(1)
            all_idx = torch.cat([x_idx, y_idx], dim=1).long().cpu().numpy()
            if var_name == 'vo':
                time_feat = (all_idx % 200).float() / 200.0
            else:
                day_feat  = torch.from_numpy(day_norm[all_idx]).to(device)   # (B, C+1)
                hour_feat = torch.from_numpy(hour_norm[all_idx]).to(device)  # (B, C+1)
                time_feat = torch.cat([day_feat, hour_feat], dim=1)

            vl = loss_fn(model, future, past, time_labels=time_lead, augment_labels=time_feat).item()
            acc['vl'] += vl

            dyn = past[:, :C]
            w = torch.ones((dyn.size(0), C), device=device)
            for j in range(C):
                for m in range(C):
                    if m != j:
                        w[:, j] *= (t - T[m]) / (T[j] - T[m])
            baseline = (w.view(-1,C,1,1) * dyn).sum(dim=1, keepdim=True)

            rnd     = torch.rand([future.size(0),1,1,1], device=device)
            rho_inv = 1 / loss_fn.rho
            sigma   = (loss_fn.sigma_max**rho_inv + (1-rnd)*(loss_fn.sigma_min**rho_inv - loss_fn.sigma_max**rho_inv))**loss_fn.rho
            pred = model(future + torch.randn_like(future)*sigma, sigma, past, time_labels=time_lead, augment_labels=time_feat)

            dp = renormalize(pred)
            ft = renormalize(future)
            bl = renormalize(baseline)

            acc['bmae'] += (bl - ft).abs().mean().item()
            acc['pmae'] += (dp - ft).abs().mean().item()
            acc['bmean']+= bl.mean().item()
            acc['pmean']+= dp.mean().item()
            acc['tmean']+= ft.mean().item()
            acc['cnt']  += 1

    N = acc['cnt']
    avg_vl   = acc['vl'] / N
    avg_bmae = acc['bmae'] / N
    avg_pmae = acc['pmae'] / N

    with open(log_csv, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoch,
            f"{avg_train:.4f}",
            f"{avg_vl:.4f}",
            f"{avg_bmae:.4f}",
            f"{avg_pmae:.4f}",
            f"{acc['bmean']/N:.4f}",
            f"{acc['pmean']/N:.4f}",
            f"{acc['tmean']/N:.4f}"
        ])
    print(f"Epoch {epoch}/{num_epochs}  TrainLoss {avg_train:.4f}  ValLoss {avg_vl:.4f}  BaseMAE {avg_bmae:.4f}  PredMAE {avg_pmae:.4f}")

    if avg_pmae < best_pmae:
        best_pmae = avg_pmae
        torch.save(model.state_dict(), result_path / f"best_{best_pmae:.4f}_{epoch}_{var_name}.pth")

    scheduler.step()
