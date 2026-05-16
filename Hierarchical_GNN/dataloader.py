import os
import h5py
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from glob import glob
import dgl

from dgl import DGLGraph

def set_graph_on_cuda(graph):
    graph = graph.to('cuda')

    for key in list(graph.ndata.keys()):
        graph.ndata[key] = graph.ndata[key].to('cuda')
    for key in list(graph.edata.keys()):
        graph.edata[key] = graph.edata[key].to('cuda')

    return graph

def collate_dgl_heterograph(graph_list):
    return dgl.batch(graph_list)

IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'int': lambda x: torch.LongTensor(x).to(DEVICE),
    'DGLHeteroGraph': collate_dgl_heterograph,
    'str': lambda x: x
}

def h5_to_tensor(h5_path):
    h5_object = h5py.File(h5_path, 'r')
    out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
    return out


def _extract_label(label_dict):
    for key in ('idh', 'label', '1p_19q', 'who_grade', 'mgmt'):
        if key in label_dict:
            return label_dict[key].item()
    first_key = next(iter(label_dict))
    return label_dict[first_key].item()

class HiPerfGraphDataset(Dataset):

    def __init__(
            self,
            cg_path: str = None,
            tg_path: str = None,
            assign_mat_path: str = None,
            load_in_ram: bool = False,
    ):
        super(HiPerfGraphDataset, self).__init__()

        assert not (cg_path is None and tg_path is None), "You must provide path to at least 1 modality."

        self.cg_path = cg_path
        self.tg_path = tg_path
        self.assign_mat_path = assign_mat_path
        self.load_in_ram = load_in_ram

        if cg_path is not None:
            self._load_cg()

        if tg_path is not None:
            self._load_tg()

        if assign_mat_path is not None:
            self._load_assign_mat()

    def _load_cg(self):
        self.cg_fnames = glob(os.path.join(self.cg_path, '*.bin'))
        self.cg_fnames.sort()
        self.cg_pids  = [os.path.basename(f).split('_')[0] for f in self.cg_fnames]
        self.num_cg = len(self.cg_fnames)
        if self.load_in_ram:
            cell_graphs = [load_graphs(os.path.join(self.cg_path, fname)) for fname in self.cg_fnames]
            self.cell_graphs = [entry[0][0] for entry in cell_graphs]
            self.cell_graph_labels = [_extract_label(entry[1]) for entry in cell_graphs]

    def _load_tg(self):
        self.tg_fnames = glob(os.path.join(self.tg_path, '*.bin'))
        self.tg_fnames.sort()
        self.tg_pids  = [os.path.basename(f).split('_')[0] for f in self.tg_fnames]
        self.num_tg = len(self.tg_fnames)
        if self.load_in_ram:
            tissue_graphs = [load_graphs(os.path.join(self.tg_path, fname)) for fname in self.tg_fnames]
            self.tissue_graphs = [entry[0][0] for entry in tissue_graphs]
            self.tissue_graph_labels = [_extract_label(entry[1]) for entry in tissue_graphs]

    def _load_assign_mat(self):
        self.assign_fnames = glob(os.path.join(self.assign_mat_path, '*.h5'))
        self.assign_fnames.sort()
        self.num_assign_mat = len(self.assign_fnames)
        if self.load_in_ram:
            self.assign_matrices = [
                h5_to_tensor(os.path.join(self.assign_mat_path, fname)).float().t()
                    for fname in self.assign_fnames
            ]

    def __getitem__(self, index):

        if hasattr(self, 'num_tg') and hasattr(self, 'num_cg'):
            if self.load_in_ram:
                cg = self.cell_graphs[index]
                tg = self.tissue_graphs[index]
                assign_mat = self.assign_matrices[index]
                assert self.cell_graph_labels[index] == self.tissue_graph_labels[index], "The CG and TG are not the same. There was an issue while creating HACT."
                label = self.cell_graph_labels[index]
            else:
                cg, label = load_graphs(self.cg_fnames[index])
                cg = cg[0]
                label = _extract_label(label)
                tg, _ = load_graphs(self.tg_fnames[index])
                tg = tg[0]
                assign_mat = h5_to_tensor(self.assign_fnames[index]).float().t()

            cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg
            assign_mat = assign_mat.cuda() if IS_CUDA else assign_mat

            pid = os.path.basename(self.cg_fnames[index]).split('_')[0]
            return cg, tg, assign_mat, label, pid

        elif hasattr(self, 'num_tg'):
            if self.load_in_ram:
                tg = self.tissue_graphs[index]
                label = self.tissue_graph_labels[index]
            else:
                tg, label = load_graphs(self.tg_fnames[index])
                label = _extract_label(label)
                tg = tg[0]
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg

            pid = os.path.basename(self.tg_fnames[index]).split('_')[0]
            return tg, label, pid

        else:
            if self.load_in_ram:
                cg = self.cell_graphs[index]
                label = self.cell_graph_labels[index]
            else:
                cg, label = load_graphs(self.cg_fnames[index])
                label = _extract_label(label)
                cg = cg[0]
            cg = set_graph_on_cuda(cg) if IS_CUDA else cg

            pid = os.path.basename(self.cg_fnames[index]).split('_')[0]
            return cg, label, pid

    def __len__(self):
        if hasattr(self, 'num_cg'):
            return self.num_cg
        else:
            return self.num_tg

def collate(batch):
    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    num_modalities = len(batch[0])
    batch = tuple([collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__)
                  for mod_id in range(num_modalities)])

    return batch

def make_data_loader(
        batch_size,
        shuffle=True,
        num_workers=0,
        **kwargs
    ):

    dataset = HiPerfGraphDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate
        )

    return dataloader
