<div align="center">

# HiPerfGNN

**Hierarchical Perfusion Graphs for Tumor Heterogeneity Modeling in Glioma Molecular Subtyping**

Han Jang<sup>\*</sup>, Junhyeok Lee<sup>\*</sup>, Heeseong Eum, Joon Jang, Yoseob Han, Seung Hong Choi, Kyu Sung Choi<sup>&dagger;</sup>

<sub>Seoul National University &middot; Soongsil University &middot; SNU College of Medicine &middot; SNU Hospital</sub>

<sub><sup>\*</sup> Equal contribution. &nbsp; <sup>&dagger;</sup> Corresponding author.</sub>

[Project Page](https://janghana.github.io/HiPerfGNN/) &middot;
[Pre-trained Weights](https://drive.google.com/drive/folders/1umdKHyWaPxT6LinbZGcJOZVoDtBKQYzS) &middot;
[Paper](https://arxiv.org/abs/2605.07156) (MICCAI 2026)

</div>

---

> **TL;DR.** A two-stage radiogenomic framework that quantizes DSC-MRI
> time-intensity curves with a VQ-VAE and propagates the resulting
> hemodynamic codes through a hierarchical perfusion graph for
> non-invasive prediction of IDH mutation, 1p/19q codeletion, and
> WHO grade.

## 🧬 Method

HiPerfGNN consists of two stages:

1. **Stage 1 &mdash; Perfusion VQ-VAE.** A 1-D
   [VQ-VAE](https://arxiv.org/abs/1711.00937) encodes each tumor
   voxel's DSC-MRI time-intensity curve into a length-3 sequence of
   discrete codes drawn from a learnable codebook of size 2
   (K=2, N=3). Variable acquisition lengths (T=45&ndash;60) are
   absorbed by adaptive temporal pooling.

2. **Stage 2 &mdash; Hierarchical Graph Neural Network.** Spatially
   contiguous voxels sharing the same code form coarse-level
   *perfusion habitats* (the tumor graph G_T). Each habitat is then
   subdivided by 3-D
   [SLIC](https://ieeexplore.ieee.org/document/6205760) supervoxels
   on co-registered four-channel structural MRI to form fine-level
   anatomical subregions (the subregion graph G_SR), linked to their
   parent habitat by a sparse assignment matrix. Both levels use
   [PNA](https://arxiv.org/abs/2004.05718) layers with LSTM-based
   [Jumping Knowledge](https://arxiv.org/abs/1806.03536) readout in a
   fine-to-coarse propagation scheme, ending in a global mean pool
   and an MLP classifier.

## 🗂️ Repository Layout

~~~
.
├── README.md
├── WEIGHTS.md                       download instructions
├── requirements.txt
├── scripts/download_weights.sh      gdown helper for the Drive folder
├── Perfusion_VQVAE/                 Stage 1 (VQ-VAE)
│   ├── train.py / inference.py
│   ├── configs/vqgan_sep.yaml
│   ├── scripts/{train,inference}.sh
│   ├── taming/                      encoder / decoder / quantizer / losses
│   └── weights/                     populated by scripts/download_weights.sh
└── Hierarchical_GNN/                Stage 2 (HGNN)
    ├── train.py / inference.py
    ├── saliency.py                  gradient-based per-node importance (Grad-CAM)
    ├── dataloader.py
    ├── build_graphs.py              --type {cell, tissue}, optional --save_assign_mat / --feature_reduce
    ├── config/                      PNA / HACT hyperparameter yml
    ├── scripts/{train,inference}.sh
    └── weights/                     populated by scripts/download_weights.sh
~~~

## 🛠️ Installation

~~~bash
git clone https://github.com/janghana/HiPerfGNN.git
cd HiPerfGNN
pip install -r requirements.txt
~~~

## 💾 Pre-trained Weights

Checkpoints are hosted on Google Drive: [<u>here</u>](https://drive.google.com/drive/folders/1umdKHyWaPxT6LinbZGcJOZVoDtBKQYzS)

~~~bash
pip install gdown
bash scripts/download_weights.sh
~~~

See [WEIGHTS.md](WEIGHTS.md) for the file layout and a
manual-download alternative.

## 🚀 Run

~~~bash
# Stage 1: DSC -> per-voxel perfusion code
bash Perfusion_VQVAE/scripts/inference.sh

# Stage 2: classify (TASK = idh | 1p19q | who_grade, TEST = internal | external)
bash Hierarchical_GNN/scripts/inference.sh                  # IDH internal
TASK=1p19q     bash Hierarchical_GNN/scripts/inference.sh
TASK=who_grade bash Hierarchical_GNN/scripts/inference.sh
TEST=external  bash Hierarchical_GNN/scripts/inference.sh   # IDH on UPenn-GBM
~~~

`PYTHON=<path-to-env-python>` overrides the python binary used by the
launchers.

### Saliency / Grad-CAM

Per-node gradient-based importance (paper Section 2.3) is computed by
`Hierarchical_GNN/saliency.py`. It saves a JSON with importance
values, node positions, and edge metadata that can be post-processed
into the voxel-level overlays of paper Figure 2.

~~~bash
python Hierarchical_GNN/saliency.py \
  --ckpt   Hierarchical_GNN/weights/idh_hact_classifier.pt \
  --root   <internal IDH hact_cell root> \
  --cfg    Hierarchical_GNN/config/hact.yml \
  --mode   node \
  --max_samples 10 \
  --output_dir gradcam_out
~~~

## 📑 Citation

~~~bibtex
@misc{jang2026hierarchicalperfusiongraphstumor,
      title={Hierarchical Perfusion Graphs for Tumor Heterogeneity Modeling in Glioma Molecular Subtyping}, 
      author={Han Jang and Junhyeok Lee and Heeseong Eum and Joon Jang and Yoseob Han and Seung Hong Choi and Kyu Sung Choi},
      year={2026},
      eprint={2605.07156},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.07156}, 
}
~~~

## 🙏 Acknowledgements

Built on top of three open-source code bases. Many thanks to the
authors for sharing their work:

- [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
  &mdash; reference VQ / EMA / Gumbel quantizers.
- [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
  &mdash; VQ-GAN backbone adapted here to 1-D DSC perfusion signals.
- [histocartography/hact-net](https://github.com/histocartography/hact-net)
  &mdash; hierarchical cell + tissue graph model re-used for
  molecular classification on supervoxel graphs.

## 📄 License

Code is released under the MIT License. Pre-trained weights and the
internal cohort follow the original institutional data-use
terms; the external UPenn-GBM cohort follows the original
[UPenn-GBM](https://www.nature.com/articles/s41597-022-01560-7)
license.
