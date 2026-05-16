import argparse, glob, pathlib, warnings, yaml, os, json
import torch, numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             confusion_matrix, roc_auc_score)
from dgl.data.utils import load_graphs
from torch.utils.data import DataLoader
import dgl
import torch.nn as nn
from histocartography.ml.models import HACTModel, CellGraphModel, TissueGraphModel

warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r"dgl\.backend\.pytorch")
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r"dgl\.backend\.pytorch")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _move(x):
    if torch.is_tensor(x):
        return x.to(device)
    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return [ _move(v) for v in x ]
    return x

def _ci(arr, fn, n=1000, seed=0):
    rng, stats = np.random.default_rng(seed), []
    for _ in range(n):
        N   = arr.shape[0]
        idx = rng.choice(N, N, True)
        stats.append(fn(arr[idx]))
    stats = np.asarray(stats, float)
    return stats.mean(), 1.96*stats.std(ddof=1)

def _auc_ci(y, p, n=1000, seed=0):
    rng, s = np.random.default_rng(seed), []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), True)
        try:   s.append(roc_auc_score(y[idx], p[idx]))
        except ValueError:  continue
    s = np.asarray(s, float)
    return s.mean(), 1.96*s.std(ddof=1)

def _fmt(mu, ci): return f"{mu:.3f} ({mu-ci:.3f}~{mu+ci:.3f})"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="folder containing *.pt checkpoints")
    p.add_argument("--root",     required=True,
                   help="hact_cell or hact_tissue root")
    p.add_argument("--cfg",      help="config .yml (optional)")
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--out_csv",  default="inference_best.csv")

    p.add_argument("--cg_path", default=None,
                   help="explicit cell_graphs path (overrides auto-discovery under --root)")
    p.add_argument("--tg_path", default=None,
                   help="explicit tissue_graphs path (overrides auto-discovery under --root)")
    p.add_argument("--am_path", default=None,
                   help="explicit assign_mat path (overrides auto-discovery under --root)")


    return p.parse_args()

def _load_cfg(ckpt: pathlib.Path, cfg_cli):
    if cfg_cli: return yaml.safe_load(open(cfg_cli))
    ymls = list(ckpt.parent.glob("*.yml"))
    if not ymls:
        raise FileNotFoundError("no *.yml near checkpoint - use --cfg")
    return yaml.safe_load(open(ymls[0]))

def _safe_load(f):
    try:
        return torch.load(f, map_location="cpu", weights_only=True)
    except Exception:
        warnings.warn("weights_only failed -> full load", UserWarning)
        return torch.load(f, map_location="cpu")

def _detect_tg_dim_from_data(tg_path: str, default_dim: int):
    try:
        files = glob.glob(os.path.join(tg_path, "*.bin"))
        if files:
            g0 = load_graphs(files[0])[0][0]
            if hasattr(g0, "ndata") and "feat" in g0.ndata:
                return int(g0.ndata["feat"].shape[1])
    except Exception:
        pass
    return default_dim

def _recreate_and_init_linear(module, in_dim, out_dim, name: str):
    layer = torch.nn.Linear(in_dim, out_dim)
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    setattr(module, name, layer)

def _build_model(cfg, ckpt_f: pathlib.Path, n_cls, tg_path=None):
    print('cfg:', cfg)
    from histocartography.ml.models import CellGraphModel, TissueGraphModel, HACTModel

    model = None
    ckpt_name = ckpt_f.stem.lower()

    if "cg_gnn_params" in cfg and "tg_gnn_params" in cfg:

        cg_node_dim = int(cfg["cg_gnn_params"].get("input_dim", 4))
        tg_node_dim = int(cfg["tg_gnn_params"].get("input_dim", 768))

        sd = _safe_load(ckpt_f)
        checkpoint_state_dict = sd.get("model", sd)

        cg_pretrans_key = "cell_graph_gnn.layers.0.towers.0.pretrans.0.mlp.fc.weight"
        if cg_pretrans_key in checkpoint_state_dict:

            ckpt_pretrans_input = checkpoint_state_dict[cg_pretrans_key].shape[1]
            ckpt_cg_dim = ckpt_pretrans_input // 2
            print(f"[INFO] detected cell_graph_gnn input dim from ckpt: {ckpt_cg_dim} (pretrans_input={ckpt_pretrans_input})")
            if ckpt_cg_dim != cg_node_dim:
                print(f"[WARN] config cg_node_dim({cg_node_dim}) != ckpt({ckpt_cg_dim}); using ckpt.")
                cg_node_dim = ckpt_cg_dim

        superpx_gnn_key = "superpx_gnn.layers.0.towers.0.fc.weight"
        tissue_projection_dim = None

        if superpx_gnn_key in checkpoint_state_dict:

            ckpt_superpx_input_dim = checkpoint_state_dict[superpx_gnn_key].shape[1]
            print(f"[INFO] detected superpx_gnn input dim from ckpt: {ckpt_superpx_input_dim}")

            cg_output_dim = int(cfg["cg_gnn_params"].get("output_dim", 64))
            cg_num_layers = int(cfg["cg_gnn_params"].get("num_layers", 3))
            readout_type = cfg["cg_gnn_params"].get("readout_type", "mean")

            if readout_type == "concat":
                cg_contribution = cg_output_dim * cg_num_layers
            else:
                cg_contribution = cg_output_dim

            tissue_projection_dim = ckpt_superpx_input_dim - cg_contribution
            print(f"[INFO] inferred tissue_projection_dim: {tissue_projection_dim} (superpx_input={ckpt_superpx_input_dim} - cg_contribution={cg_contribution})")

            actual_tg_dim = _detect_tg_dim_from_data(tg_path, tg_node_dim) if tg_path else tg_node_dim
            print(f"[INFO] actual tissue graph dimension: {actual_tg_dim}")

            if actual_tg_dim != tg_node_dim:
                print(f"[WARN] config tg_node_dim({tg_node_dim}) != data({actual_tg_dim}); using data.")
                tg_node_dim = actual_tg_dim

        print(f"[INFO] HACT model: cg_node_dim={cg_node_dim}, tg_node_dim={tg_node_dim}")
        if tissue_projection_dim is not None:
            print(f"[INFO] tissue_projection_dim={tissue_projection_dim} (inferred from ckpt)")
            model = HACTModel(cfg["cg_gnn_params"], cfg["tg_gnn_params"],
                              cfg["classification_params"], num_classes=n_cls,
                              cg_node_dim=cg_node_dim, tg_node_dim=tg_node_dim,
                              tissue_projection_dim=tissue_projection_dim)
        else:
            print(f"[WARN] could not infer tissue_projection_dim from ckpt; using default.")
            model = HACTModel(cfg["cg_gnn_params"], cfg["tg_gnn_params"],
                              cfg["classification_params"], num_classes=n_cls,
                              cg_node_dim=cg_node_dim, tg_node_dim=tg_node_dim)
    elif "tggnn" in ckpt_name:

        dim = _detect_tg_dim_from_data(tg_path, int(cfg["gnn_params"].get("input_dim", 256))) if tg_path else int(cfg["gnn_params"].get("input_dim", 256))
        print(f"[INFO] TissueGraphModel using node_dim={dim}")
        model = TissueGraphModel(cfg["gnn_params"], cfg["classification_params"],
                                 node_dim=dim, num_classes=n_cls)
    else:

        dim = int(cfg["gnn_params"].get("input_dim", 4))
        print(f"[INFO] CellGraphModel using node_dim={dim}")
        model = CellGraphModel(cfg["gnn_params"], cfg["classification_params"],
                               node_dim=dim, num_classes=n_cls)

    if 'checkpoint_state_dict' not in locals():
        sd = _safe_load(ckpt_f)
        checkpoint_state_dict = sd.get("model", sd)

    LEGACY_KEY_RENAMES = {
        "feat_proj.weight": "tissue_projection.weight",
        "feat_proj.bias":   "tissue_projection.bias",
    }
    renamed = []
    for old, new in LEGACY_KEY_RENAMES.items():
        if old in checkpoint_state_dict and new not in checkpoint_state_dict:
            checkpoint_state_dict[new] = checkpoint_state_dict.pop(old)
            renamed.append((old, new))
    if renamed:
        print(f"[INFO] remapped legacy ckpt keys: {renamed}")
    if "tissue_projection.weight" in checkpoint_state_dict and \
       "tissue_projection.bias" not in checkpoint_state_dict and \
       hasattr(model, "tissue_projection") and \
       getattr(model.tissue_projection, "bias", None) is not None:
        out_features = checkpoint_state_dict["tissue_projection.weight"].shape[0]
        checkpoint_state_dict["tissue_projection.bias"] = torch.zeros(out_features)
        print(f"[INFO] synthesised zero bias for tissue_projection ({out_features},)")

    def _maybe_fix_projection(model, layer_name):
        if hasattr(model, layer_name):
            m = getattr(model, layer_name)
            mw_shape = (m.out_features, m.in_features)
            ck_key_w = f"{layer_name}.weight"
            ck_key_b = f"{layer_name}.bias"
            if ck_key_w in checkpoint_state_dict:
                ck_shape = tuple(checkpoint_state_dict[ck_key_w].shape)
                if ck_shape != mw_shape:
                    print(f"[WARN] {layer_name} shape mismatch: model {mw_shape}, ckpt {ck_shape} -> dropping ckpt keys")
                    checkpoint_state_dict.pop(ck_key_w, None); checkpoint_state_dict.pop(ck_key_b, None)

    _maybe_fix_projection(model, "input_projection")
    _maybe_fix_projection(model, "tissue_projection")

    model_state = model.state_dict()
    filtered_ckpt = {}
    skipped_keys = []
    missing_in_ckpt = []

    for k, v in model_state.items():
        if k in checkpoint_state_dict:
            ckpt_v = checkpoint_state_dict[k]
            if v.shape == ckpt_v.shape:
                filtered_ckpt[k] = ckpt_v
            else:
                skipped_keys.append((k, v.shape, ckpt_v.shape))
        else:
            missing_in_ckpt.append(k)

    unexpected_keys = [k for k in checkpoint_state_dict.keys() if k not in model_state]

    try:
        missing_keys, unexpected_keys_loaded = model.load_state_dict(filtered_ckpt, strict=False)
        if skipped_keys:
            print(f"[WARN] Size mismatch detected, filtering incompatible layers...")
            for k, model_shape, ckpt_shape in skipped_keys:
                print(f"   Skipping {k}: model {model_shape} vs ckpt {ckpt_shape}")
        if unexpected_keys:
            print(f"[WARN] Keys in checkpoint but not in model (ignored):")
            for k in unexpected_keys[:10]:
                print(f"   {k}")
            if len(unexpected_keys) > 10:
                print(f"   ... and {len(unexpected_keys) - 10} more")
        if missing_keys:
            print(f"Missing keys (random init): {missing_keys}")
        if unexpected_keys_loaded:
            print(f"Unexpected keys after load: {unexpected_keys_loaded}")
    except RuntimeError as e:
        print(f"[ERROR] Error loading state dict: {e}")
        raise

    def _init_linear_layer(module, name):
        if hasattr(module, name):
            layer = getattr(module, name)
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
                return True
        return False

    if hasattr(model, 'input_projection'):
        if ('input_projection.weight' in missing_keys) or ('input_projection.bias' in missing_keys):
            _init_linear_layer(model, 'input_projection')
            print(f"[INFO] Initialized input_projection")

    if hasattr(model, 'tissue_projection'):
        if ('tissue_projection.weight' in missing_keys) or ('tissue_projection.bias' in missing_keys):

            if hasattr(model.tissue_projection, 'in_features'):
                expected_in = model.tissue_projection.in_features
                expected_out = model.tissue_projection.out_features

                actual_tg_dim = _detect_tg_dim_from_data(tg_path, expected_in) if tg_path else expected_in
                if actual_tg_dim != expected_in:
                    print(f"[WARN] Recreating tissue_projection: {expected_in} -> {actual_tg_dim} input dim")
                    _recreate_and_init_linear(model, actual_tg_dim, expected_out, "tissue_projection")
                else:
                    _init_linear_layer(model, 'tissue_projection')
                    print(f"[INFO] Initialized tissue_projection")

    initialized_layers = set()
    for key in missing_keys:
        if 'superpx_gnn' in key or 'cell_graph_gnn' in key:

            if key.endswith('.weight'):
                layer_name = key.rsplit('.weight', 1)[0]
                if layer_name in initialized_layers:
                    continue

                parts = layer_name.split('.')
                try:
                    obj = model
                    for part in parts:
                        obj = getattr(obj, part)
                    if isinstance(obj, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(obj.weight)
                        if hasattr(obj, 'bias') and obj.bias is not None:
                            torch.nn.init.zeros_(obj.bias)
                        initialized_layers.add(layer_name)
                        print(f"[INFO] Initialized {layer_name} (weight + bias)")
                except (AttributeError, TypeError) as e:

                    pass
            elif key.endswith('.bias'):

                layer_name = key.rsplit('.bias', 1)[0]
                if layer_name not in initialized_layers:

                    parts = layer_name.split('.')
                    try:
                        obj = model
                        for part in parts:
                            obj = getattr(obj, part)
                        if isinstance(obj, torch.nn.Linear) and hasattr(obj, 'bias') and obj.bias is not None:
                            torch.nn.init.zeros_(obj.bias)
                            initialized_layers.add(layer_name)
                            print(f"[INFO] Initialized {layer_name}.bias")
                    except (AttributeError, TypeError):
                        pass

    return model.to(device).eval()


def main():
    args = get_args()
    root = pathlib.Path(args.root).resolve()

    ckpts = sorted(pathlib.Path(args.ckpt_dir).glob("*.pt"))
    if not ckpts: raise RuntimeError("no *.pt found in directory")

    first_cfg = _load_cfg(ckpts[0], args.cfg)
    is_hact_model = "cg_gnn_params" in first_cfg and "tg_gnn_params" in first_cfg

    cg = pathlib.Path(args.cg_path) if args.cg_path else root / "cell_graphs" / "test"

    if is_hact_model:

        if args.tg_path:
            tg = pathlib.Path(args.tg_path)
            tg_candidates = [tg]
        else:
            tg_candidates = [
                root / "tissue_graphs" / "test",
                root / "test",
                root.parent / "hact_tissue" / "tissue_graphs" / "test",
                root.parent.parent / "tissue_graphs" / "test",
            ]
            tg = None
            for candidate in tg_candidates:
                if candidate.exists():
                    tg = candidate
                    break

        if args.am_path:
            am = pathlib.Path(args.am_path)
            am_candidates = [am]
        else:
            am_candidates = [
                root / "assign_mat" / "test",
                root / "assign_matrices" / "test",
                root / "assign_mat",
            ]
            am = None
            for candidate in am_candidates:
                if candidate.exists() and any(candidate.glob("*.h5")):
                    am = candidate
                    break

        if not cg.exists():
            raise RuntimeError(f"cell_graphs path not found for HACT model: {cg}")
        if not tg:
            raise RuntimeError(
                f"tissue_graphs path not found for HACT model. "
                f"tried: {tg_candidates}"
            )
        if not am:
            raise RuntimeError(
                f"assign_mat path not found for HACT model. "
                f"tried: {am_candidates}"
            )
        use_tg = True
        print(f"[INFO] HACT model detected")
        print(f"  cell_graphs: {cg}")
        print(f"  tissue_graphs: {tg}")
        print(f"  assign_mat: {am}")
    else:

        tg = root / "tissue_graphs" / "test"
        if not tg.exists():
            tg = root / "test"
        am = root / "assign_mat" / "test"
        use_tg = tg.exists() and am.exists()

    from dataloader import HiPerfGraphDataset, collate
    ds = HiPerfGraphDataset(
        cg_path=str(cg) if cg.exists() else None,
        tg_path=str(tg) if use_tg else None,
        assign_mat_path=str(am) if use_tg else None,
        load_in_ram=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    collate_fn=collate, num_workers=0)

    if is_hact_model:

        test_iter = iter(dl)
        test_batch = next(test_iter)
        num_data_items = len(test_batch) - 2
        if num_data_items != 3:
            raise RuntimeError(
                f"HACT model expects 3 data items but dataloader returned {num_data_items}. "
                f"expected: (cell_graph, tissue_graph, assignment_matrix). "
                f"batch structure: {[type(x).__name__ for x in test_batch]}. "
                f"data paths: cg={cg.exists()}, tg={tg.exists()}, am={am.exists()}"
            )
        print(f"[INFO] dataloader OK: {num_data_items} data items per batch")

    best_f1, best_info = -1, None

    for ckpt in ckpts:
        cfg   = _load_cfg(ckpt, args.cfg)
        model = _build_model(cfg, ckpt, args.num_classes,
                             tg_path=str(tg) if use_tg else None)

        logits_all, labels_all, pids_all = [], [], []

        is_hact = isinstance(model, HACTModel)

        with torch.no_grad():
            for batch in dl:
                *data, lab, pid_b = batch
                data = _move(data)

                if is_hact:
                    if len(data) != 3:
                        raise ValueError(f"HACT model requires 3 inputs (cell_graph, tissue_graph, assignment_matrix), got: {len(data)}")
                    out = model(data[0], data[1], data[2])
                elif len(data) > 1:
                    out = model(*data)
                else:
                    out = model(data[0])

                logits_all.append(out.detach().cpu())
                labels_all.append(lab.cpu())
                pids_all.extend(pid_b)

        logits = torch.cat(logits_all)
        labels = torch.cat(labels_all).numpy()

        if logits.ndim == 1:
            prob1 = torch.sigmoid(logits).numpy()
            preds = (prob1 > 0.5).astype(np.int64)
            probs_full = None
        elif logits.shape[1] == 2:
            preds = logits.argmax(1).numpy()
            prob1 = torch.softmax(logits, 1).numpy()[:, 1]
            probs_full = torch.softmax(logits, 1).numpy()
        else:
            preds = logits.argmax(1).numpy()
            prob1 = None
            probs_full = torch.softmax(logits, 1).numpy()

        f1_now = f1_score(labels, preds, average="macro")
        print(f"{ckpt.name:<35}  F1={f1_now:.4f}")

        if f1_now > best_f1:
            best_f1 = f1_now
            best_info = (ckpt, labels, preds, prob1, probs_full, pids_all, model, dl)

    ckpt, y, y_hat, p1, p_full, pids, best_model, best_dl = best_info
    y_arr = np.column_stack([y, y_hat])
    n_classes = int(max(y.max(), y_hat.max())) + 1
    is_multiclass = n_classes > 2

    acc_m, acc_ci = _ci(y_arr, lambda a: accuracy_score(a[:,0], a[:,1]))
    f1_m , f1_ci  = _ci(y_arr, lambda a: f1_score   (a[:,0], a[:,1], average="macro"))
    if is_multiclass:
        sen_m, sen_ci = _ci(y_arr, lambda a: recall_score(a[:,0], a[:,1], average="macro"))
        def _macro_spec(a):
            from sklearn.preprocessing import label_binarize
            classes = list(range(n_classes))
            y_bin = label_binarize(a[:,0], classes=classes)
            y_pred_bin = label_binarize(a[:,1], classes=classes)
            specs = []
            for k in range(n_classes):
                tn = ((y_bin[:,k]==0) & (y_pred_bin[:,k]==0)).sum()
                fp = ((y_bin[:,k]==0) & (y_pred_bin[:,k]==1)).sum()
                if (tn+fp) > 0:
                    specs.append(tn / (tn + fp))
            return float(np.mean(specs)) if specs else 0.0
        spe_m, spe_ci = _ci(y_arr, _macro_spec)
    else:
        sen_m, sen_ci = _ci(y_arr, lambda a: recall_score(a[:,0], a[:,1], pos_label=1))
        spe_m, spe_ci = _ci(y_arr, lambda a: confusion_matrix(a[:,0], a[:,1]).ravel()[0] /
                                         (confusion_matrix(a[:,0], a[:,1]).ravel()[0] +
                                          confusion_matrix(a[:,0], a[:,1]).ravel()[1]))

    if is_multiclass and p_full is not None:
        from sklearn.metrics import roc_auc_score as _roc_auc
        def _mc_auc_boot(y_true, prob, average, n=1000, seed=0):
            rng, s = np.random.default_rng(seed), []
            classes = list(range(n_classes))
            for _ in range(n):
                idx = rng.choice(len(y_true), len(y_true), True)
                try: s.append(_roc_auc(y_true[idx], prob[idx], labels=classes, multi_class="ovr", average=average))
                except ValueError: continue
            s = np.asarray(s, float)
            return s.mean(), 1.96*s.std(ddof=1)
        auc_m, auc_ci = _mc_auc_boot(y, p_full, "macro")
        auc_w_m, auc_w_ci = _mc_auc_boot(y, p_full, "weighted")
    else:
        auc_m, auc_ci = _auc_ci(y, p1) if p1 is not None else (np.nan, np.nan)
        auc_w_m = auc_w_ci = None

    print("\n[BEST] :", ckpt.name)
    print("ACC          :", _fmt(acc_m, acc_ci))
    print("F1 (macro)   :", _fmt(f1_m , f1_ci))
    print("SEN          :", _fmt(sen_m, sen_ci))
    print("SPE          :", _fmt(spe_m, spe_ci))
    if is_multiclass and auc_w_m is not None:
        print("AUC (macro)  :", _fmt(auc_m, auc_ci))
        print("AUC (weighted):", _fmt(auc_w_m, auc_w_ci))
    else:
        print("AUC          :", _fmt(auc_m, auc_ci))

    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (pid, gt, pred) in enumerate(zip(pids, y, y_hat)):
        row = {"pid": pid, "y_true": int(gt), "y_pred": int(pred)}
        if p1 is not None:
            row["prob_1"] = float(p1[i])
        rows.append(row)

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("[INFO] saved ->", args.out_csv)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
