"""Build HACT cell-graph or tissue-graph from VQ-VAE node files.

Replaces four legacy scripts:
  generate_hact_cell_graph_from_nodefiles{,_with_h5}.py
  generate_hact_tissue_graph_from_nodefiles{,_mean_latent}.py

Examples
--------
# tissue-graph (HACT default, 3x256-dim node features)
python build_graphs.py --type tissue \
    --node_root <latent_nodes_dir> \
    --split_dir <split_json_dir> --idh_xlsx <labels.xlsx> \
    --outdir <out_root>

# tissue-graph with mean latent (256-dim node features)
python build_graphs.py --type tissue --feature_reduce mean ...

# cell-graph with assignment-matrix h5 (HACT-net default)
python build_graphs.py --type cell --save_assign_mat \
    --node_root <node_voxels_dir> \
    --mri_root <mri_root> \
    --tissue_graph_root <tissue_graphs_root> \
    --split_dir <split_json_dir> --idh_xlsx <labels.xlsx> \
    --outdir <out_root>

# cell-graph only (no assignment matrix)
python build_graphs.py --type cell ...
"""
import argparse
import json
import multiprocessing as mp
import warnings
from pathlib import Path

import dgl
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs, save_graphs
from scipy.spatial import cKDTree
from skimage.segmentation import slic

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*One of the clusters is empty.*")

MRI_MODALITIES = ["t1", "t1ce", "t2", "flair"]


def load_label_xlsx(xlsx, label_col="IDH"):
    df = pd.read_excel(xlsx, engine="openpyxl")
    return dict(zip(df["Patient_ID"].astype(str), df[label_col]))


def load_split_map(split_dir):
    out = {}
    for sp in ("train", "valid", "test"):
        with open(Path(split_dir) / f"{sp}_list.json") as fp:
            for pid in json.load(fp):
                out[str(pid)] = "val" if sp == "valid" else sp
    return out


def load_4ch_mri(root: Path, pid: str):
    pdir = root / pid
    sub = [s for s in pdir.iterdir() if s.is_dir()]
    if not sub:
        raise FileNotFoundError(pdir)
    vols = [nib.load(str(sub[0] / f"{m}.nii.gz")).get_fdata(dtype=np.float32)
            for m in MRI_MODALITIES]
    return np.stack(vols, -1)


def cosine1(a, b):
    den = max(1e-6, np.linalg.norm(a) * np.linalg.norm(b))
    return np.array([np.clip((a @ b) / den, -1.0, 1.0)], dtype=np.float32)


def edge_features_cos_3x3(lat_3x256, i, j):
    A = lat_3x256[i]
    B = lat_3x256[j]
    nA = np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    nB = np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    return ((A @ B.T) / (nA @ nB.T)).reshape(-1).astype(np.float32)


def process_tissue(arg):
    (pid, feats_path, split_map, label_map, dist_th, feature_reduce) = arg
    if pid not in split_map or pid not in label_map:
        return None

    arr = np.load(feats_path)
    coords = arr[:, :3].astype(np.float32)
    latent = arr[:, 3:771].astype(np.float32)
    lat_3x256 = latent.reshape(-1, 3, 256)

    tree = cKDTree(coords)
    pairs = tree.query_pairs(dist_th)

    src, dst, efeats = [], [], []
    for i, j in pairs:
        ef = edge_features_cos_3x3(lat_3x256, i, j)
        src += [i, j]
        dst += [j, i]
        efeats += [ef, ef]

    g = dgl.graph((src, dst), num_nodes=coords.shape[0], idtype=torch.int32)
    node_feat = lat_3x256.mean(1) if feature_reduce == "mean" else latent
    g.ndata["feat"] = torch.from_numpy(node_feat)
    g.ndata["cent"] = torch.from_numpy(coords)
    g.edata["ef"] = torch.from_numpy(np.stack(efeats)) if efeats else \
        torch.empty((0, 9), dtype=torch.float32)

    return pid, split_map[pid], g, torch.tensor([label_map[pid]])


def process_cell(arg):
    (pid, vox_file, mri_root, tg_root, split_map, label_map,
     min_vox, vox_per_seg, comp, dist_th, save_assign_mat) = arg

    if pid not in split_map or pid not in label_map:
        return None

    tg_path = Path(tg_root, split_map[pid], f"{pid}_tissue_graph.bin")
    if not tg_path.exists():
        return None
    tg, _ = load_graphs(str(tg_path))
    tissueN = tg[0].num_nodes()

    raw_obj = np.load(vox_file, allow_pickle=True)
    if isinstance(raw_obj, dict):
        bfs_list = list(raw_obj.values())
    elif raw_obj.ndim == 0 and isinstance(raw_obj.item(), dict):
        bfs_list = list(raw_obj.item().values())
    elif raw_obj.dtype == object:
        bfs_list = list(raw_obj)
    else:
        bfs_list = [raw_obj]

    vol4 = load_4ch_mri(mri_root, pid).astype(np.float32)
    vol4 = np.nan_to_num(vol4, nan=0.0, posinf=0.0, neginf=0.0)
    Z, Y, X, _ = vol4.shape
    lbl = -np.ones((Z, Y, X), np.int32)
    cur = 0
    cell2tissue = []

    for bfs_id, coords in enumerate(bfs_list):
        if coords is None or coords.size == 0:
            continue
        z, y, x = coords.T
        m = (z >= 0) & (z < Z) & (y >= 0) & (y < Y) & (x >= 0) & (x < X)
        z, y, x = z[m], y[m], x[m]
        if len(z) == 0:
            continue

        sub = np.zeros((Z, Y, X), bool)
        sub[z, y, x] = True
        n_vox = sub.sum()

        if n_vox < min_vox:
            lbl[sub] = cur
            cell2tissue.append(bfs_id)
            cur += 1
        else:
            n_seg = max(1, n_vox // vox_per_seg)
            seg = slic(vol4, n_segments=n_seg, compactness=comp,
                       mask=sub, channel_axis=-1, start_label=0)
            k = seg[sub].max() + 1
            lbl[sub] = seg[sub] + cur
            cell2tissue.extend([bfs_id] * k)
            cur += k

    mask = lbl >= 0
    if not mask.any():
        return None
    uniq, inv = np.unique(lbl[mask], return_inverse=True)
    cellN = len(uniq)

    feats = np.zeros((cellN, 4), np.float32)
    cent = np.zeros((cellN, 3), np.float32)
    coords_all = np.argwhere(mask)
    vals = np.nan_to_num(vol4[mask], nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(cellN):
        idx = inv == i
        feats[i] = vals[idx].mean(0)
        cent[i] = coords_all[idx].mean(0)

    tree = cKDTree(cent)
    pairs = tree.query_pairs(dist_th)
    src, dst, ef = [], [], []
    for i, j in pairs:
        e = cosine1(feats[i], feats[j])
        src += [i, j]
        dst += [j, i]
        ef += [e, e]

    g = dgl.graph((src, dst), num_nodes=cellN, idtype=torch.int32)
    g.ndata["feat"] = torch.from_numpy(feats)
    g.edata["ef"] = torch.from_numpy(np.vstack(ef)) if ef else \
        torch.empty((0, 1), dtype=torch.float32)

    A = None
    if save_assign_mat:
        A = np.zeros((cellN, tissueN), dtype=np.uint8)
        A[np.arange(cellN), cell2tissue] = 1

    return pid, split_map[pid], tissueN, cellN, g, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["cell", "tissue"], required=True)
    ap.add_argument("--node_root", required=True,
                    help="directory of *_node_feats.npy (tissue) or *_node_voxels.npy (cell)")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--idh_xlsx", required=True)
    ap.add_argument("--label_col", default="IDH",
                    help="column name in the xlsx to use as target label")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dist_th", type=float, default=15.0,
                    help="centroid distance threshold (voxel units)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))

    ap.add_argument("--feature_reduce", choices=["none", "mean"], default="none",
                    help="(tissue only) 'mean' averages the 3 codes -> 256-D features")

    ap.add_argument("--mri_root",
                    help="(cell only) root directory of 4-channel MRI volumes")
    ap.add_argument("--tissue_graph_root",
                    help="(cell only) root directory of pre-built tissue graphs")
    ap.add_argument("--save_assign_mat", action="store_true",
                    help="(cell only) also save the cell->tissue assignment matrix as h5")
    ap.add_argument("--min_vox_for_slic", type=int, default=5)
    ap.add_argument("--vox_per_seg", type=int, default=30)
    ap.add_argument("--compact", type=float, default=0.05)
    args = ap.parse_args()

    split_map = load_split_map(args.split_dir)
    label_map = load_label_xlsx(args.idh_xlsx, args.label_col)

    if args.type == "tissue":
        for sp in ("train", "val", "test"):
            Path(args.outdir, "tissue_graphs", sp).mkdir(parents=True, exist_ok=True)
        jobs = [(f.stem.replace("_node_feats", ""), str(f),
                 split_map, label_map, args.dist_th, args.feature_reduce)
                for f in Path(args.node_root).glob("*_node_feats.npy")]
        with mp.Pool(args.workers) as pool:
            for res in pool.imap_unordered(process_tissue, jobs):
                if res is None:
                    continue
                pid, sp, g, label = res
                out_f = Path(args.outdir, "tissue_graphs", sp,
                             f"{pid}_tissue_graph.bin")
                save_graphs(str(out_f), [g], {"label": label})
                print(f"[{pid}] {sp:5s}  N={g.num_nodes():4d}  E={g.num_edges():5d}")
        print("[DONE] Tissue-graph generation finished.")
        return

    if args.mri_root is None or args.tissue_graph_root is None:
        ap.error("--mri_root and --tissue_graph_root are required when --type cell")

    for sp in ("train", "val", "test"):
        Path(args.outdir, "cell_graphs", sp).mkdir(parents=True, exist_ok=True)
        if args.save_assign_mat:
            Path(args.outdir, "assign_mat", sp).mkdir(parents=True, exist_ok=True)

    jobs = [(f.stem.replace("_node_voxels", ""), str(f),
             Path(args.mri_root), Path(args.tissue_graph_root),
             split_map, label_map,
             args.min_vox_for_slic, args.vox_per_seg,
             args.compact, args.dist_th, args.save_assign_mat)
            for f in Path(args.node_root).glob("*_node_voxels.npy")]

    with mp.Pool(args.workers) as pool:
        for res in pool.imap_unordered(process_cell, jobs):
            if res is None:
                continue
            pid, sp, tN, cN, g, A = res
            warn = " [WARN]" if cN < tN else ""
            print(f"[{pid}] {sp:5s}  tissueN={tN:4d}  cellN={cN:5d}  "
                  f"edges={g.num_edges():7d}{warn}")
            save_graphs(str(Path(args.outdir, "cell_graphs", sp,
                                 f"{pid}_cell_graph.bin")),
                        [g], {"label": torch.tensor([label_map[pid]])})
            if A is not None:
                h5_path = Path(args.outdir, "assign_mat", sp, f"{pid}.h5")
                with h5py.File(h5_path, "w") as h5:
                    h5.create_dataset("assignment_matrix", data=A,
                                      compression="gzip")
    print("[DONE] Cell-graph generation finished.")


if __name__ == "__main__":
    main()
