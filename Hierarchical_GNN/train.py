import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, classification_report,
                             confusion_matrix)
import shutil
import matplotlib.pyplot as plt
import datetime
import random, numpy as np, torch, dgl
import glob, math, json, pandas as pd, pathlib
import glob
import json
from dgl.data.utils import load_graphs

from histocartography.ml.models import CellGraphModel, TissueGraphModel, HACTModel

from dataloader import make_data_loader
from sklearn.utils import resample

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.nn.functional import one_hot, log_softmax
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import warnings
from sklearn.exceptions import UndefinedMetricWarning

from torch.optim.swa_utils import AveragedModel, update_bn

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import pandas as pd, pathlib, json
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.custom_(fwd|bwd).*",
    category=FutureWarning,
    module=r"dgl\.backend\.pytorch\.sparse"
)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', message=r'TypedStorage is deprecated.*')

def bootstrap_auc(y_true, y_prob, n_boot=1000, seed=0):
    rng   = np.random.default_rng(seed)
    aucs  = []
    for _ in range(n_boot):
        idx  = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    aucs = np.asarray(aucs, dtype=float)
    return aucs.mean(), 1.96 * aucs.std(ddof=1)

def mean_ci95(arr: np.ndarray):
    arr  = np.asarray(arr, dtype=float)
    mean = arr.mean()
    if arr.size < 2:
        return mean, 0.0
    ci95 = 1.96 * arr.std(ddof=1) / math.sqrt(arr.size)
    return mean, ci95

def bootstrap_ci(y_true, y_pred, score_fn, n_boot=1000, seed=0):
    rng   = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx   = rng.choice(len(y_true), len(y_true), replace=True)
        stats.append(score_fn(y_true[idx], y_pred[idx]))
    stats = np.asarray(stats, dtype=float)
    return stats.mean(), 1.96 * stats.std(ddof=1)

def pid(path: str) -> str:
    return os.path.basename(path).split('_')[0]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dgl.seed(seed)
    dgl.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target,
                                         weight=self.alpha,
                                         reduction='none')
        pt = torch.exp(-ce)
        focal = ((1-pt)**self.gamma) * ce
        return focal.mean() if self.reduction=='mean' else focal.sum()

def wrap_model(model, local_rank, use_ddp=True):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    if use_ddp and dist.is_initialized():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    elif torch.cuda.device_count() > 1:
        print(f"[INFO] DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    return model, device

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cg_path', type=str, default=None,
                        help='Path to voxel-level graphs (DGL .bin).')
    parser.add_argument('--tg_path', type=str, default=None,
                        help='Path to supervoxel-level graphs (DGL .bin).')
    parser.add_argument('--assign_mat_path', type=str, default=None,
                        help='Path to assignment matrices (HDF5).')
    parser.add_argument('--config_fpath', type=str, required=True,
                        help='Path to the config file (YAML). Must contain "cggnn", "tggnn" or "hact" in filename.')
    parser.add_argument('--model_path', type=str, default='./checkpoints',
                        help='Where to save the trained model weights.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--in_ram', action='store_true',
                        help='Whether to load all graphs in RAM at once.')
    parser.add_argument('--logger', type=str, default='none',
                        help='Logger type (like "mlflow" or "none").')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes (e.g., IDH classification=2).')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--seed', type=int, default=42,
                    help="global random seed")
    parser.add_argument('--run_name',  type=str, default='exp1', help='folder identifier')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP (mixed-precision)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='keep K best checkpoints (by val-F1)')

    return parser.parse_args()

def main_worker(local_rank, nprocs, args):

    set_seed(args.seed)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=nprocs,
        rank=local_rank,
        timeout=datetime.timedelta(minutes=30)
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    with open(args.config_fpath, 'r') as f:
        config = yaml.safe_load(f)

    config_filename = os.path.basename(args.config_fpath)
    if 'cggnn' in config_filename:
        model_type = 'cggnn'
    elif 'tggnn' in config_filename:
        model_type = 'tggnn'
    elif 'hact' in config_filename:
        model_type = 'hact'
    else:
        raise ValueError("Config filename must contain one of [cggnn, tggnn, hact].")

    from torch.utils.data import DataLoader
    from dataloader import HiPerfGraphDataset, collate

    ds_train_dir = HiPerfGraphDataset(
        cg_path=os.path.join(args.cg_path, "train") if args.cg_path else None,
        tg_path=os.path.join(args.tg_path, "train") if args.tg_path else None,
        assign_mat_path=os.path.join(args.assign_mat_path, "train") if args.assign_mat_path else None,
        load_in_ram=args.in_ram
    )
    ds_val_dir = HiPerfGraphDataset(
        cg_path=os.path.join(args.cg_path, "val") if args.cg_path else None,
        tg_path=os.path.join(args.tg_path, "val") if args.tg_path else None,
        assign_mat_path=os.path.join(args.assign_mat_path, "val") if args.assign_mat_path else None,
        load_in_ram=args.in_ram
    )

    train_dataset = ds_train_dir
    val_dataset   = ds_val_dir

    test_dataset = HiPerfGraphDataset(
        cg_path=os.path.join(args.cg_path, "test") if args.cg_path else None,
        tg_path=os.path.join(args.tg_path, "test") if args.tg_path else None,
        assign_mat_path=os.path.join(args.assign_mat_path, "test") if args.assign_mat_path else None,
        load_in_ram=args.in_ram
    )

    train_len = len(train_dataset)
    val_len   = len(val_dataset)
    test_len  = len(test_dataset)
    if local_rank == 0:
        print(f"Dataset sizes: train={train_len}, val={val_len}, test={test_len}, total={train_len + val_len + test_len}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=nprocs, rank=local_rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=nprocs, rank=local_rank, shuffle=False)
    test_sampler  = DistributedSampler(test_dataset,  num_replicas=nprocs, rank=local_rank, shuffle=False)

    def get_label(sample):
        return int(sample[-2])

    if local_rank == 0:
        train_labels = [get_label(train_dataset[i]) for i in range(train_len)]
    else:
        train_labels = [0] * train_len

    dist.broadcast_object_list(train_labels, 0)

    cls_cnt = np.bincount(train_labels, minlength=args.num_classes)
    cls_cnt[cls_cnt == 0] = 1

    if local_rank == 0:
        class_sample_count = cls_cnt
        weights_np = 1.0 / class_sample_count
    else:
        weights_np = np.ones(args.num_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=0,
        drop_last=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=collate
    )

    if model_type == 'cggnn':
        gnn_params = config['gnn_params']
        cls_params= config['classification_params']
        node_dim  = gnn_params.get('input_dim', 4)
        if local_rank == 0:
            print("=> Building CellGraphModel (voxel-only). node_dim=", node_dim)
        model = CellGraphModel(
            gnn_params=gnn_params,
            classification_params=cls_params,
            node_dim=node_dim,
            num_classes=args.num_classes
        )

    elif model_type == 'tggnn':
        gnn_params = config['gnn_params']
        cls_params = config['classification_params']

        sample_file = glob.glob(os.path.join(args.tg_path, 'train', '*.bin'))[0]
        g, _ = load_graphs(sample_file)
        node_dim = g[0].ndata['feat'].shape[1]
        print('node_dim in tggnn:',node_dim)
        print("node feature shape:", g[0].ndata['feat'].shape)
        if local_rank == 0:
            print(f"=> Building TissueGraphModel  |  input node_dim={node_dim}")
        model = TissueGraphModel(
            gnn_params=gnn_params,
            classification_params=cls_params,
            node_dim=node_dim,
            num_classes=args.num_classes
        )

    else:
        cg_gnn_params= config['cg_gnn_params']
        tg_gnn_params= config['tg_gnn_params']
        cls_params   = config['classification_params']
        cg_node_dim  = cg_gnn_params.get('input_dim', 60)
        tg_node_dim  = tg_gnn_params.get('input_dim', 768)
        if local_rank == 0:
            print("=> Building HACTModel. cg_node_dim=", cg_node_dim, ", tg_node_dim=", tg_node_dim)
        model = HACTModel(
            cg_gnn_params=cg_gnn_params,
            tg_gnn_params=tg_gnn_params,
            classification_params=cls_params,
            cg_node_dim=cg_node_dim,
            tg_node_dim=tg_node_dim,
            num_classes=args.num_classes
        )

    model, device = wrap_model(model, local_rank, use_ddp=True)

    ddp_model = model

    optimizer = torch.optim.AdamW(ddp_model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=1e-5)

    from timm.utils.model_ema import ModelEmaV2
    ema = ModelEmaV2(ddp_model, decay=0.995, device=device)

    warmup_epochs = 5
    main_epochs   = args.epochs - warmup_epochs
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=main_epochs)
        ],
        milestones=[warmup_epochs]
    )

    epochs_no_improve = 0
    patience_es = 80

    metrics = ["loss","acc","f1","prec","rec","auc"]
    history = {f"train_{m}":[] for m in metrics} | {f"val_{m}":[] for m in metrics}

    scaler = GradScaler()

    if local_rank == 0:
        train_labels = [int(train_dataset[idx][-2]) for idx in range(train_len)]
        class_sample_count = np.bincount(train_labels, minlength=args.num_classes)
        weights_np = 1.0 / class_sample_count
    else:
        weights_np = np.ones(args.num_classes)

    weights_tensor = torch.from_numpy(weights_np).float().to(device)

    alpha = weights_tensor
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    best_val_f1 = 0.0
    top_ckpts   = []
    step = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        use_amp = not args.no_amp

        ddp_model.train()
        running_loss= 0.0
        preds_list  = []
        probs_list  = []
        labels_list = []

        for batch in train_loader:
            label = batch[-2].to(device)
            data  = batch[:-2]

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                logits = ddp_model(*data)
                loss   = criterion(logits, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 5.0)

            scaler.step(optimizer)
            scaler.update()
            ema.update(ddp_model)

            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            probs = torch.softmax(logits, 1).detach().cpu().numpy()

            preds_list.append(preds)
            probs_list.append(probs)
            labels_list.append(label.cpu().numpy())
            step += 1

        def safe_auc(y_true, y_score):
            if np.isnan(y_score).any() or np.isinf(y_score).any():
                return np.nan
            return roc_auc_score(y_true, y_score)

        if local_rank == 0:
            preds_list  = np.concatenate(preds_list)
            probs_list  = np.concatenate(probs_list)
            labels_list = np.concatenate(labels_list)
            train_acc = accuracy_score(labels_list, preds_list)
            train_f1  = f1_score(labels_list, preds_list, average='weighted')
            train_prec = precision_score(labels_list, preds_list, average='weighted', zero_division=0)
            train_rec  = recall_score(labels_list, preds_list, average='weighted', zero_division=0)
            num_cls = probs_list.shape[1]

            if num_cls == 2:
                train_auc = safe_auc(labels_list, probs_list[:, 1])
            else:
                train_auc = safe_auc(labels_list, probs_list,
                                        multi_class='ovr', average='weighted')

            print(f"[Train] epoch={epoch:3d} | "
                f"loss={running_loss:.4f} | acc={train_acc:.4f} | "
                f"prec={train_prec:.4f} | rec={train_rec:.4f} | "
                f"f1={train_f1:.4f} | auc={train_auc:.4f}")

            history["train_loss"].append(running_loss)
            for k,v in zip(["acc","f1","prec","rec","auc"],
                        [train_acc,train_f1,train_prec,train_rec,train_auc]):
                history[f"train_{k}"].append(v)

        ema_model_was_used = False
        if hasattr(ema, 'ema_model'):
            ema.store()
            ema.copy_to(ddp_model.module.parameters())
            ema_model_was_used = True

        ddp_model.eval()

        val_logits, val_labels = [], []
        val_loss = 0.0
        for batch in val_loader:
            y = batch[-2].to(device)
            data = batch[:-2]
            with torch.no_grad():
                logit = ddp_model(*data)
                loss  = criterion(logit, y)
            val_loss += loss.item()
            val_logits.append(logit)
            val_labels.append(y)

        local_logits = torch.cat(val_logits, 0)
        local_labels = torch.cat(val_labels, 0)

        local_n = torch.tensor([local_logits.size(0)], device=device)
        n_list  = [torch.zeros_like(local_n) for _ in range(nprocs)]
        dist.all_gather(n_list, local_n)
        max_n = int(max(n.item() for n in n_list))

        def pad(t, L):
            return torch.cat([t, t.new_zeros(L - t.size(0), *t.shape[1:])], 0)

        logits_pad = pad(local_logits, max_n)
        labels_pad = pad(local_labels, max_n)

        logits_gather = [torch.zeros_like(logits_pad) for _ in range(nprocs)]
        labels_gather = [torch.zeros_like(labels_pad) for _ in range(nprocs)]

        dist.all_gather(logits_gather, logits_pad)
        dist.all_gather(labels_gather, labels_pad)

        all_logits = torch.cat([lg[:n.item()] for lg, n in zip(logits_gather, n_list)], 0)
        all_labels = torch.cat([lb[:n.item()] for lb, n in zip(labels_gather, n_list)], 0)

        preds = all_logits.argmax(1).cpu().numpy()
        labs  = all_labels.cpu().numpy()

        val_acc  = accuracy_score(labs, preds)
        val_f1   = f1_score(labs, preds, average='weighted')
        val_prec = precision_score(labs, preds, average='weighted', zero_division=0)
        val_rec  = recall_score(labs, preds,   average='weighted', zero_division=0)
        val_auc  = roc_auc_score(labs,
                    torch.softmax(all_logits,1)[:,1].cpu().numpy()) if all_logits.size(1)==2 else 0

        for k,v in zip(["loss","acc","f1","prec","rec","auc"],
                [val_loss,val_acc,val_f1,val_prec,val_rec,val_auc]):
            history[f"val_{k}"].append(v)

        if local_rank == 0:
            print(f"[Val] epoch={epoch:3d} | "
                f"loss={val_loss:.4f} | acc={val_acc:.4f} | "
                f"prec={val_prec:.4f} | rec={val_rec:.4f} | "
                f"f1={val_f1:.4f} | auc={val_auc:.4f}")

            cm = confusion_matrix(labs, preds)
            print()
            print("Val Confusion Matrix:")
            print(cm)
            TN, FP, FN, TP = cm.ravel()
            print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

            ckpt_name = (f"{model_type}_ep{epoch:03d}_"
                         f"f1{val_f1:.4f}.pt")
            ckpt_path = os.path.join(args.model_path, ckpt_name)
            torch.save({
                "model": ddp_model.module.state_dict(),
                "ema"  : ema.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1
            }, ckpt_path)
            top_ckpts.append((val_f1, ckpt_path))
            top_ckpts.sort(key=lambda x: x[0], reverse=True)

            if len(top_ckpts) > args.top_k:
                _, worst_path = top_ckpts.pop()
                if os.path.exists(worst_path):
                    os.remove(worst_path)
            best_val_f1 = top_ckpts[0][0]
            best_path   = top_ckpts[0][1]

            final_best = os.path.join(args.model_path, f"best_{model_type}.pt")
            shutil.copyfile(best_path, final_best)
            improved  = (ckpt_path == best_path)
        else:
            improved = False

        best_val_f1_t = torch.tensor([best_val_f1], device=device)
        improved_t    = torch.tensor([int(improved)], device=device)
        dist.broadcast(best_val_f1_t, src=0)
        dist.broadcast(improved_t,    src=0)
        best_val_f1 = best_val_f1_t.item()
        improved    = bool(improved_t.item())

        scheduler.step()
        if ema_model_was_used:
            ema.restore()

        if local_rank == 0:
            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            print('Early-Stopping Counts:',epochs_no_improve)

        stop_flag = torch.tensor([int(epochs_no_improve >= patience_es)], device=device)
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            if local_rank == 0:
                print(f"Early-Stopping (patience={patience_es})")
            break

    if local_rank == 0:
        best_path = os.path.join(args.model_path, f"best_{model_type}.pt")

        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device)

            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

            if hasattr(ddp_model, "module"):
                ddp_model.module.load_state_dict(state_dict)
            else:
                ddp_model.load_state_dict(state_dict)

            if isinstance(ckpt, dict) and "ema" in ckpt:
                try:
                    ema.load_state_dict(ckpt["ema"])
                except Exception as e:
                    if local_rank == 0:
                        print("EMA load fail:", e)
            print(f"*** Loaded best model for test: {best_path}")
        else:
            print("No best model found, using current model weights for test.")

    ddp_model.eval()

    test_logits_local = []
    test_labels_local = []
    all_pids = []
    test_loss_local   = 0.0
    test_probs_local = []
    local_pid_list = []
    test_feats_local = []
    def forward_get_feat(model, *data):
        out = model(*data)
        if isinstance(out, tuple):
            return out
        return out, None

    def forward_get_feat(model, *data):
        out = model(*data)
        if isinstance(out, tuple):
            logits, feat = out
            if feat is None:
                feat = logits.detach()
        else:
            logits = out
            feat   = logits.detach()
        return logits, feat

    for batch in test_loader:
        *data, label, pid_batch = batch
        label = label.to(device)
        with torch.no_grad():
            logits, feat = forward_get_feat(ddp_model, *data)
            loss  = criterion(logits, label)
            prob   = torch.softmax(logits, 1)

        test_feats_local.append(feat.detach() if feat is not None else torch.zeros(len(label),1, device=device))
        test_loss_local += loss.item()
        test_logits_local.append(logits.detach())
        test_labels_local.append(label.detach())
        local_pid_list.extend(pid_batch)
        test_probs_local.append(prob.detach())

    pid_lists = [None for _ in range(nprocs)]
    dist.all_gather_object(pid_lists, local_pid_list)
    if local_rank == 0:
        all_pids = sum(pid_lists, [])

    local_logits = torch.cat(test_logits_local, dim=0)
    local_labels = torch.cat(test_labels_local, dim=0)

    local_n = torch.tensor([local_logits.size(0)], device=device)
    n_list  = [torch.zeros_like(local_n) for _ in range(nprocs)]

    dist.all_gather(n_list, local_n)
    max_n = int(max(n.item() for n in n_list))

    feat_pad     = pad(torch.cat(test_feats_local, 0), max_n)
    feat_gather  = [torch.zeros_like(feat_pad) for _ in range(nprocs)]

    logits_pad = pad(local_logits, max_n)
    labels_pad = pad(local_labels, max_n)
    feats_pad  = pad(torch.cat(test_feats_local, 0), max_n)

    logits_gather = [torch.zeros_like(logits_pad) for _ in range(nprocs)]
    labels_gather = [torch.zeros_like(labels_pad) for _ in range(nprocs)]
    feats_gather  = [torch.zeros_like(feats_pad)  for _ in range(nprocs)]

    dist.all_gather(logits_gather, logits_pad)
    dist.all_gather(labels_gather, labels_pad)
    dist.all_gather(feats_gather,  feats_pad)

    test_loss_all_t = torch.tensor([test_loss_local], device=device)
    dist.all_reduce(test_loss_all_t, op=dist.ReduceOp.SUM)
    test_loss_all = test_loss_all_t.item()

    if local_rank == 0:

        all_logits_list = []
        all_labels_list = []
        all_pid_list     = []

        for r in range(nprocs):
            real_n = n_list[r].item()
            all_logits_list.append(logits_gather[r][:real_n])
            all_labels_list.append(labels_gather[r][:real_n])
            all_pid_list.append(pid_lists[r][:real_n])

        all_pids   = sum(all_pid_list, [])

        all_logits = torch.cat(all_logits_list, dim=0).cpu()
        all_labels = torch.cat(all_labels_list, dim=0).cpu()

        all_probs  = torch.softmax(all_logits, 1).numpy()

        test_preds  = all_logits.argmax(1).numpy()
        test_labels = all_labels.numpy()
        test_loss   = test_loss_all

        test_acc = accuracy_score(test_labels, test_preds)
        test_f1  = f1_score(test_labels, test_preds, average='weighted')
        if all_logits.size(1) == 2:
            probs = torch.softmax(all_logits, 1)[:, 1].numpy()
            if len(np.unique(test_labels)) == 2:
                test_auc = roc_auc_score(test_labels, probs)
            else:
                test_auc = 0.0
        else:
            test_auc = roc_auc_score(test_labels,
                                    torch.softmax(all_logits,1).numpy(),
                                    multi_class='ovr', average='weighted')

        cm = confusion_matrix(test_labels, test_preds, labels=[0,1])
        if cm.shape == (2,2):
            TN, FP, FN, TP = cm.ravel()
            test_sens = TP / (TP + FN + 1e-12)
            test_spec = TN / (TN + FP + 1e-12)
        else:
            TN = FP = FN = TP = np.nan
            test_sens = test_spec = np.nan

        print(f"\n[Test]  total={len(test_labels)}, loss={test_loss:.4f}, "
            f"acc={test_acc:.4f}, f1={test_f1:.4f}, auc={test_auc:.4f}")
        print(classification_report(test_labels, test_preds))
        print("Confusion Matrix:")
        print(cm)
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        test_sens = TP / (TP + FN + 1e-12)
        test_spec = TN / (TN + FP + 1e-12)

        out_rows = []
        for case_id, gt, pred, prob in zip(all_pids, test_labels, test_preds, all_probs):
            row = {"split": "test",
                "pid": case_id, "y_true": int(gt), "y_pred": int(pred)}
            for c in range(all_probs.shape[1]):
                row[f"prob_{c}"] = float(prob[c])
            out_rows.append(row)
        csv_dir = pathlib.Path(args.model_path, "csv"); csv_dir.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(out_rows).to_csv(csv_dir/f"{args.run_name}.csv", index=False)

        if len(history["train_loss"]) > 0:
            plot_dir = os.path.join(args.model_path,"plots") ; os.makedirs(plot_dir,exist_ok=True)

            epochs = range(1, len(history["train_loss"]) + 1)
            for m in metrics:
                plt.figure()
                plt.plot(epochs,history[f"train_{m}"],label="Train")
                plt.plot(epochs,history[f"val_{m}"],  label="Val")
                plt.xlabel("Epoch");  plt.ylabel(m.upper());  plt.title(m.upper())
                plt.legend()
                png_path = os.path.join(plot_dir,f"{model_type}_{m}.png")
                plt.savefig(png_path,dpi=150) ;  plt.close()
                print(f"  - saved {png_path}")

        summary = {}
        for split in ["train", "val"]:
            summary[split] = {}
            for m in metrics:
                arr = np.asarray(history[f"{split}_{m}"], dtype=float)
                mean, ci95 = mean_ci95(arr)
                summary[split][m] = {"mean": float(mean),
                                     "ci95": float(ci95)}

        def _acc(a,b): return accuracy_score(a,b)
        def _f1(a,b):  return f1_score(a,b, average='weighted')

        mean_acc, ci_acc = bootstrap_ci(test_labels, test_preds, _acc)
        mean_f1 , ci_f1  = bootstrap_ci(test_labels, test_preds, _f1)

        test_prob1 = torch.softmax(all_logits, 1)[:, 1].numpy()
        mean_auc, ci_auc = bootstrap_auc(test_labels, test_prob1)

        summary["test"] = {
            "acc":         {"mean": float(test_acc),  "ci95": float(ci_acc)},
            "f1":          {"mean": float(test_f1),   "ci95": float(ci_f1)},
            "auc":         {"mean": float(test_auc),  "ci95": float(ci_auc)},
            "sensitivity": {"mean": float(test_sens), "ci95": None},
            "specificity": {"mean": float(test_spec), "ci95": None}
        }

        summary_path = os.path.join(args.model_path,
                                    f"{model_type}_summary.json")
        with open(summary_path, "w") as fp:
            json.dump(summary, fp, indent=2, allow_nan=False)

    dist.barrier()
    dist.destroy_process_group()

def main():
    args = parse_arguments()
    nprocs = torch.cuda.device_count()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.local_rank = local_rank

    main_worker(local_rank, nprocs, args)

if __name__=="__main__":
    main()
