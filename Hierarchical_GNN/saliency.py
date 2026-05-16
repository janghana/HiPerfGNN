"""Gradient-based saliency / Grad-CAM for the HiPerfGNN classifier.

Computes per-node importance for cell or tissue graphs and saves a JSON
with the values, node positions, and edge metadata.

Usage:
    python saliency.py \
        --ckpt   weights/idh_hact_classifier.pt \
        --root   <internal IDH hact_cell root> \
        --cfg    config/hact.yml \
        --output_dir gradcam_out
"""
import argparse
import json
import os
import pathlib
import warnings
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from histocartography.ml.models import HACTModel
from torch.utils.data import DataLoader

from inference import _build_model, _load_cfg, _move, device


def get_args():
    p = argparse.ArgumentParser(
        description="Compute gradient-based node importance (Grad-CAM) "
                    "for the bundled HACT classifier and save as JSON.")
    p.add_argument("--ckpt", required=True, help="path to .pt classifier checkpoint")
    p.add_argument("--root", required=True,
                   help="hact_cell / hact_tissue root (same layout as inference.py)")
    p.add_argument("--cfg",  required=True, help="config .yml")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--cg_path", default=None,
                   help="explicit cell_graphs path override")
    p.add_argument("--tg_path", default=None,
                   help="explicit tissue_graphs path override")
    p.add_argument("--am_path", default=None,
                   help="explicit assign_mat path override")
    p.add_argument("--mode", choices=["node", "edge", "both"], default="node")
    p.add_argument("--max_samples", type=int, default=10,
                   help="how many samples to process before stopping")
    p.add_argument("--output_dir", default="gradcam_out",
                   help="output directory for the JSON")
    return p.parse_args()


@contextmanager
def rnn_backward_safe(model):
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    changed = []
    try:
        for m in model.modules():
            if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
                changed.append((m, m.training))
                m.train(True)
            elif isinstance(m, (nn.Dropout, nn.AlphaDropout,
                                nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.BatchNorm3d, nn.SyncBatchNorm)):
                changed.append((m, m.training))
                m.train(False)
        yield
    finally:
        for m, was_train in changed:
            m.train(was_train)
        torch.backends.cudnn.enabled = prev_cudnn


def _pick_graph_for_attr(model, data):
    name = model.__class__.__name__.lower()
    fkey = "feat"
    if "hactmodel" in name or "hact" in name:
        return 1, fkey
    return 0, fkey


def _sample_node_slice(batched_graph, sample_idx):
    sizes = batched_graph.batch_num_nodes().tolist()
    start = sum(sizes[:sample_idx])
    end = start + sizes[sample_idx]
    return start, end


def _sample_edge_slice(batched_graph, sample_idx):
    sizes = batched_graph.batch_num_edges().tolist()
    start = sum(sizes[:sample_idx])
    end = start + sizes[sample_idx]
    return start, end


def _select_target_logit(logits, sample_idx):
    if logits.ndim == 1:
        tlogit = logits[sample_idx]
        tclass = int((torch.sigmoid(tlogit) > 0.5).item())
        return tlogit, tclass
    if logits.ndim == 2:
        pred = int(torch.argmax(logits[sample_idx]).item())
        return logits[sample_idx, pred], pred
    raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")


def compute_integrated_gradients(model, input_data, target_class, feat_tensor,
                                 sample_idx=0, steps=32):
    baseline = torch.zeros_like(feat_tensor)
    total_grad = torch.zeros_like(feat_tensor)
    is_hact = isinstance(model, HACTModel)

    with rnn_backward_safe(model):
        for i in range(steps + 1):
            alpha = i / steps
            x = (baseline + alpha * (feat_tensor.detach() - baseline)).requires_grad_(True)
            g_idx, fkey = _pick_graph_for_attr(model, input_data)
            input_data[g_idx].ndata[fkey] = x

            if is_hact:
                logits = model(input_data[0], input_data[1], input_data[2])
            elif len(input_data) > 1:
                logits = model(*input_data)
            else:
                logits = model(input_data[0])

            tlogit = logits[sample_idx] if logits.ndim == 1 else logits[sample_idx, target_class]
            grads = torch.autograd.grad(tlogit, x, retain_graph=True, allow_unused=True)[0]
            if grads is not None:
                total_grad += grads / steps

    ig = total_grad * (feat_tensor.detach() - baseline)
    importance_all = ig.abs().mean(dim=1)
    g_idx, _ = _pick_graph_for_attr(model, input_data)
    start, end = _sample_node_slice(input_data[g_idx], sample_idx)
    return importance_all[start:end].detach().cpu().numpy()


def compute_node_importance(model, data, sample_idx=0):
    was_training = model.training
    model.eval()

    input_data = list(data) if isinstance(data, (list, tuple)) else [data]
    g_idx, fkey = _pick_graph_for_attr(model, input_data)
    graph = input_data[g_idx]
    if not (hasattr(graph, "ndata") and fkey in graph.ndata):
        n = graph.number_of_nodes() if hasattr(graph, "number_of_nodes") else 1
        model.train(was_training)
        return np.zeros(n, dtype=np.float32)

    feat = graph.ndata[fkey].clone().detach().to(device).requires_grad_(True)
    feat.retain_grad()
    graph.ndata[fkey] = feat

    is_hact = isinstance(model, HACTModel)
    with rnn_backward_safe(model):
        if is_hact:
            logits = model(input_data[0], input_data[1], input_data[2])
        elif len(input_data) > 1:
            logits = model(*input_data)
        else:
            logits = model(input_data[0])
        tlogit, tclass = _select_target_logit(logits, sample_idx)

        model.zero_grad(set_to_none=True)
        if feat.grad is not None:
            feat.grad.zero_()
        grads = torch.autograd.grad(tlogit, feat, retain_graph=False, allow_unused=True)[0]

    if grads is None:
        imp = compute_integrated_gradients(model, input_data, tclass, feat, sample_idx)
        model.train(was_training)
        return imp

    importance_all = grads.abs().mean(dim=1)
    start, end = _sample_node_slice(graph, sample_idx)
    importance = importance_all[start:end].detach().cpu().numpy()
    model.train(was_training)
    return importance


def _snapshot_feats(data):
    """HACTModel.forward overwrites tissue_graph.ndata['feat'] in-place.
    Snapshot the original feats so we can restore between forward calls."""
    snap = {}
    for idx, item in enumerate(data):
        if hasattr(item, "ndata") and "feat" in item.ndata:
            snap[idx] = item.ndata["feat"].clone()
    return snap


def _restore_feats(data, snap):
    for idx, t in snap.items():
        data[idx].ndata["feat"] = t


def analyze(model, dataloader, output_dir, mode="node", max_samples=10):
    print(f"[INFO] Analyzing {mode} importance (max {max_samples} samples)")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    processed = 0
    for batch in dataloader:
        if processed >= max_samples:
            break

        *data, labels, pids = batch
        data = _move(data)
        snap = _snapshot_feats(data)

        with torch.no_grad():
            logits = model(*data) if len(data) > 1 else model(data[0])
        _restore_feats(data, snap)

        B = labels.shape[0]
        for i in range(B):
            if processed >= max_samples:
                break
            if logits.ndim == 1:
                pred_class = int((torch.sigmoid(logits[i]) > 0.5).item())
            else:
                pred_class = int(torch.argmax(logits[i]).item())

            node_importance = compute_node_importance(model, data, sample_idx=i)
            _restore_feats(data, snap)
            g_idx, fkey = _pick_graph_for_attr(model, data)
            g = data[g_idx]
            n_start, n_end = _sample_node_slice(g, i)
            e_start, e_end = _sample_edge_slice(g, i)

            node_positions = None
            for k in ("pos", "coord"):
                if k in g.ndata:
                    node_positions = g.ndata[k][n_start:n_end].detach().cpu().numpy().tolist()
                    break

            src_all, dst_all = g.edges()
            src_s = src_all[e_start:e_end] - n_start
            dst_s = dst_all[e_start:e_end] - n_start
            edge_info = {
                "src": src_s.detach().cpu().numpy().tolist(),
                "dst": dst_s.detach().cpu().numpy().tolist(),
                "num_edges": int(e_end - e_start),
            }

            rec = {
                "sample_idx": int(processed),
                "pid": pids[i],
                "true_label": int(labels[i].item()),
                "pred_label": int(pred_class),
                "graph_metadata": {
                    "graph_type": str(type(g)),
                    "feature_dim": int(g.ndata[fkey].shape[1]) if fkey in g.ndata else None,
                    "num_nodes": int(n_end - n_start),
                    "num_edges": int(e_end - e_start),
                },
                "node_importance_values": [float(v) for v in node_importance],
                "node_importance_stats": {
                    "mean": float(np.mean(node_importance)) if len(node_importance) else 0.0,
                    "std":  float(np.std(node_importance))  if len(node_importance) else 0.0,
                    "max":  float(np.max(node_importance))  if len(node_importance) else 0.0,
                    "min":  float(np.min(node_importance))  if len(node_importance) else 0.0,
                },
                "node_positions": node_positions,
                "edge_info": edge_info,
            }

            if mode in ("edge", "both") and len(node_importance) and edge_info["num_edges"]:
                ni = np.asarray(node_importance, dtype=np.float32)
                src_np = np.clip(np.asarray(edge_info["src"], dtype=np.int64), 0, len(ni) - 1)
                dst_np = np.clip(np.asarray(edge_info["dst"], dtype=np.int64), 0, len(ni) - 1)
                e_imp = (ni[src_np] + ni[dst_np]) * 0.5
                rec["edge_importance_values"] = [float(v) for v in e_imp]
                rec["edge_importance_stats"] = {
                    "mean": float(np.mean(e_imp)),
                    "std":  float(np.std(e_imp)),
                    "max":  float(np.max(e_imp)),
                    "min":  float(np.min(e_imp)),
                }

            results.append(rec)
            processed += 1

    if not results:
        print("[INFO] no samples processed.")
        return None

    out_f = os.path.join(output_dir, f"importance_{mode}.json")
    with open(out_f, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] saved -> {out_f}  ({len(results)} samples)")
    return out_f


def _build_dataloader(args, cfg):
    from dataloader import HiPerfGraphDataset, collate
    root = pathlib.Path(args.root).resolve()
    is_hact = "cg_gnn_params" in cfg and "tg_gnn_params" in cfg

    cg = pathlib.Path(args.cg_path) if args.cg_path else root / "cell_graphs" / "test"
    tg = pathlib.Path(args.tg_path) if args.tg_path else None
    am = pathlib.Path(args.am_path) if args.am_path else None

    if is_hact:
        if tg is None:
            for c in (root / "tissue_graphs" / "test",
                      root.parent / "hact_tissue" / "tissue_graphs" / "test",
                      root.parent.parent / "tissue_graphs" / "test"):
                if c.exists():
                    tg = c
                    break
        if am is None:
            for c in (root / "assign_mat" / "test",
                      root / "assign_mat",
                      root / "assign_matrices" / "test"):
                if c.exists() and any(c.glob("*.h5")):
                    am = c
                    break

    ds = HiPerfGraphDataset(
        cg_path=str(cg) if cg.exists() else None,
        tg_path=str(tg) if (tg and tg.exists()) else None,
        assign_mat_path=str(am) if (am and am.exists()) else None,
        load_in_ram=False,
    )
    return DataLoader(ds, batch_size=args.batch, shuffle=False,
                      collate_fn=collate, num_workers=0)


def main():
    args = get_args()
    cfg = _load_cfg(pathlib.Path(args.ckpt), args.cfg)
    model = _build_model(cfg, pathlib.Path(args.ckpt), args.num_classes,
                         tg_path=args.tg_path)
    dl = _build_dataloader(args, cfg)
    analyze(model, dl, args.output_dir,
            mode=args.mode, max_samples=args.max_samples)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    torch.cuda.empty_cache()
    main()
