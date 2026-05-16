import argparse
import csv
import glob
import os
import statistics as st
import sys
from collections import deque

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from taming import instantiate_from_config


def save_3d_nifti(volume_3d, out_path, affine=None):
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(volume_3d, affine), out_path)


def save_4d_nifti(volume_4d, out_path, affine=None):
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(volume_4d, affine), out_path)


def round_clip_bits(arr_2d):
    return np.clip(np.rint(arr_2d).astype(np.int16), 0, 2)


def bits_to_cluster_label(ch0, ch1, ch2):
    return ch0 * 4 + ch1 * 2 + ch2 + 1


@torch.no_grad()
def run_cluster(model, dloader, outdir, device, batch_size=512):
    for batch in dloader:
        dsc_signal = batch["dsc_signal"].squeeze().to(device)
        tumor_mask = batch["tumor_mask"].squeeze().to(device)
        patient_id = batch["patient_id"][0]

        H, W, D, T = dsc_signal.shape
        total_voxels = H * W * D
        print(f"[INFO] patient={patient_id}, dsc.shape={(H,W,D,T)}, mask.shape={(H,W,D)}")

        patient_dir = os.path.join(outdir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        save_4d_nifti(dsc_signal.cpu().numpy(),
                      os.path.join(patient_dir, "dsc.nii.gz"))

        flat = dsc_signal[tumor_mask]
        N = flat.shape[0]
        if flat.ndim == 1:
            flat = flat.unsqueeze(-1)

        voxel_coords = torch.nonzero(tumor_mask, as_tuple=False).cpu().numpy()
        signal_np = flat.cpu().numpy()

        cluster_chunks = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            inp = flat[start:end].unsqueeze(1)
            _, _, info = model.encode(inp, return_continuous=False)
            bits = round_clip_bits(info[2].cpu().numpy())
            cluster_chunks.append(bits_to_cluster_label(bits[:, 0], bits[:, 1], bits[:, 2]))

        cluster_labels = np.concatenate(cluster_chunks, axis=0).reshape(-1, 1)

        combined = np.concatenate([voxel_coords, signal_np, cluster_labels], axis=1)
        csv_path = os.path.join(patient_dir, f"{patient_id}_coords_signal_cluster.csv")
        npy_path = os.path.join(patient_dir, f"{patient_id}_coords_signal_cluster.npy")
        np.save(npy_path, combined)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            T_ = signal_np.shape[1]
            writer.writerow(["x", "y", "z"]
                            + [f"signal_{i}" for i in range(T_)]
                            + ["cluster_id"])
            for row in combined:
                writer.writerow(row.tolist())

        cluster_3d = np.zeros_like(tumor_mask.cpu().numpy(), dtype=np.int16)
        cluster_3d[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = cluster_labels[:, 0]
        cluster_path = os.path.join(patient_dir, "cluster.nii.gz")
        save_3d_nifti(cluster_3d, cluster_path)

        uniq, cnt = np.unique(cluster_labels, return_counts=True)
        print(f"       total_voxels={total_voxels}, tumor_voxels={N}")
        for v, c in zip(uniq, cnt):
            print(f"          cluster_id={v}, count={c}")
        print(f"       -> dsc.nii.gz / cluster.nii.gz / CSV / NPY written under {patient_dir}\n")


@torch.no_grad()
def run_latent(model, dloader, outdir, node_outdir, device, batch_size=512):
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)
               for dz in (-1, 0, 1)
               if not (dx == dy == dz == 0)]

    for batch in dloader:
        dsc_signal = batch["dsc_signal"].squeeze().to(device)
        tumor_mask = batch["tumor_mask"].squeeze().to(device)
        patient_id = batch["patient_id"][0]

        patient_dir = os.path.join(outdir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        save_4d_nifti(dsc_signal.cpu().numpy(),
                      os.path.join(patient_dir, "dsc.nii.gz"))

        voxel_coords_3d = torch.nonzero(tumor_mask, as_tuple=False)
        N = len(voxel_coords_3d)
        if N == 0:
            print(f"[WARN] {patient_id}: zero tumor voxels, skipping.")
            continue

        flat = dsc_signal[tumor_mask]
        cluster_chunks, latent_chunks = [], []
        CH = None

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            inp = flat[start:end].unsqueeze(1)
            z_e, _, _, info = model.encode(inp, return_continuous=True)
            bits = round_clip_bits(info[2].cpu().numpy())
            cluster_chunks.append(bits_to_cluster_label(bits[:, 0], bits[:, 1], bits[:, 2]))
            z_e_np = z_e.cpu().numpy()
            if CH is None:
                CH = z_e_np.shape[1:]
            latent_chunks.append(z_e_np)

        cluster_id_arr = np.concatenate(cluster_chunks, axis=0)
        latents_arr = np.concatenate(latent_chunks, axis=0)
        coords_np = voxel_coords_3d.cpu().numpy()

        xyz2idx = {tuple(coords_np[i]): i for i in range(N)}
        visited = np.zeros(N, dtype=bool)

        node_info, node_voxel_dict, nid = [], {}, 0
        for vid in range(N):
            if visited[vid]:
                continue
            cid = cluster_id_arr[vid]
            queue = deque([vid])
            visited[vid] = True
            region = []
            while queue:
                cur = queue.popleft()
                region.append(cur)
                x, y, z = coords_np[cur]
                for dx, dy, dz in offsets:
                    key = (x + dx, y + dy, z + dz)
                    if key in xyz2idx:
                        nb = xyz2idx[key]
                        if (not visited[nb]) and cluster_id_arr[nb] == cid:
                            visited[nb] = True
                            queue.append(nb)
            region_coords = coords_np[region]
            mean_coord = region_coords.mean(axis=0)
            mean_latent = latents_arr[region].mean(axis=0)
            node_info.append((nid, mean_coord, mean_latent, cid, len(region)))
            node_voxel_dict[nid] = region_coords
            nid += 1

        c, dim = CH
        total_dim = c * dim
        final_arr = np.zeros((len(node_info), 3 + total_dim + 2), dtype=np.float32)
        for i, (_, mc, ml, cid, vcnt) in enumerate(node_info):
            final_arr[i, 0:3] = mc
            final_arr[i, 3:3 + total_dim] = ml.reshape(-1)
            final_arr[i, 3 + total_dim] = cid
            final_arr[i, 3 + total_dim + 1] = vcnt

        os.makedirs(node_outdir, exist_ok=True)
        np.save(os.path.join(node_outdir, f"{patient_id}_node_feats.npy"), final_arr)
        with open(os.path.join(node_outdir, f"{patient_id}_node_feats.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cx", "cy", "cz"]
                            + [f"latent_{k}" for k in range(total_dim)]
                            + ["cluster_id", "voxel_count"])
            for row in final_arr:
                writer.writerow(row.tolist())
        np.save(os.path.join(node_outdir, f"{patient_id}_node_voxels.npy"),
                node_voxel_dict, allow_pickle=True)

        bfs_sizes = [info[-1] for info in node_info]
        bfs_stats = {
            "n_bfs": len(bfs_sizes),
            "min_vox": int(min(bfs_sizes)),
            "mean_vox": float(np.mean(bfs_sizes)),
            "median": float(st.median(bfs_sizes)),
            "max_vox": int(max(bfs_sizes)),
        }
        uniq, cnt = np.unique(cluster_id_arr, return_counts=True)
        cluster_hist = {int(k): int(v) for k, v in zip(uniq, cnt)}
        pd.DataFrame([{**cluster_hist, **bfs_stats}]).to_csv(
            os.path.join(patient_dir, "summary.csv"), index=False)
        print(f"[INFO] patient={patient_id} clusters={cluster_hist} bfs={bfs_stats}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cluster", "latent"], default="cluster",
                   help="cluster: per-voxel cluster id; latent: BFS-grouped supervoxel latents")
    p.add_argument("-r", "--resume", type=str, nargs="?",
                   help="checkpoint file or logdir")
    p.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml",
                   default=list(), help="paths to base configs")
    p.add_argument("-c", "--config", nargs="?", metavar="single_config.yaml",
                   const=True, default="", help="path to single config")
    p.add_argument("--ckpt_name", type=str, help="checkpoint filename inside logdir")
    p.add_argument("--outdir", required=True, type=str,
                   help="per-patient output dir (dsc.nii.gz / cluster.nii.gz / CSV / NPY)")
    p.add_argument("--node_outdir", type=str, default=None,
                   help="output dir for BFS-latent node features (only used in --mode latent)")
    p.add_argument("--center", type=str, default="internal")
    p.add_argument("--task", type=str, default="quantized_DSC")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=512)
    return p


def load_model_from_config(model_cfg, sd, device=0, eval_mode=True):
    if "ckpt_path" in model_cfg.params:
        model_cfg.params.ckpt_path = None
    model = instantiate_from_config(model_cfg)
    if sd is not None:
        model.load_state_dict(sd, strict=False)
    model = model.to(f"cuda:{device}")
    if eval_mode:
        model.eval()
    return model


def get_data(config):
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


def load_model_and_dset(config, ckpt, device, eval_mode=True):
    dsets = get_data(config)
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd.get("global_step", None)
        sd = pl_sd["state_dict"]
    else:
        sd = None
        global_step = None
    model = load_model_from_config(config.model, sd, device=device, eval_mode=eval_mode)
    return dsets, model, global_step


def resolve_ckpt(opt):
    if not opt.resume:
        return None
    if not os.path.exists(opt.resume):
        raise ValueError(f"Cannot find {opt.resume}")
    if os.path.isfile(opt.resume):
        paths = opt.resume.split("/")
        try:
            idx = len(paths) - paths[::-1].index("logs") + 1
        except ValueError:
            idx = -2
        logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume)
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints",
                            opt.ckpt_name if opt.ckpt_name else "last.ckpt")
    print(f"[INFO] logdir={logdir}")
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
    opt.base = base_configs + opt.base
    return ckpt


def main():
    sys.path.append(os.getcwd())
    parser = build_parser()
    opt, unknown = parser.parse_known_args()

    if opt.mode == "latent" and opt.node_outdir is None:
        parser.error("--node_outdir is required when --mode latent")

    ckpt = resolve_ckpt(opt)

    if opt.config:
        opt.base = [opt.config] if isinstance(opt.config, str) else [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    if "validation" in config.data.params:
        config.data.params["test"] = config.data.params["validation"]
        config.data.params["test"]["target"] = "taming.data.dsc.DSCTest_Task"
        del config.data.params["validation"]
    if "train" in config.data.params:
        del config.data.params["train"]

    config.data.params["test"]["params"]["device"] = opt.device
    config.data.params["test"]["params"]["center"] = opt.center
    config.data.params["test"]["params"]["task"] = opt.task

    dsets, model, global_step = load_model_and_dset(config, ckpt, device=opt.device)
    dloader = DataLoader(dsets.datasets["test"], batch_size=1, shuffle=False, num_workers=8)
    print(f"[INFO] global_step={global_step} mode={opt.mode}")

    if opt.mode == "cluster":
        run_cluster(model=model, dloader=dloader, outdir=opt.outdir,
                    device=opt.device, batch_size=opt.batch_size)
    else:
        run_latent(model=model, dloader=dloader,
                   outdir=opt.outdir, node_outdir=opt.node_outdir,
                   device=opt.device, batch_size=opt.batch_size)


if __name__ == "__main__":
    main()
