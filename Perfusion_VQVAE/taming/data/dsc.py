import os
from glob import glob
import numpy as np
import json
import random
import ants
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.multiprocessing import Pool
from scipy.ndimage import morphology

dataroot_server = '/mnt/hdd3/hjang/data'

class DSC(Dataset):
    def __init__(self, dataroot, phase, transform):
        super().__init__()
        self.dataroot = dataroot
        self.phase = phase
        self.transform = transform

        self.dataset = []
        self.dataset_xyz_idx_dict = {}

        self.append_internal()

        print(f'Total {phase} dataset size - volume: {len(self.dataset_xyz_idx_dict)}, signal: {len(self.dataset)}')

    def __getitem__(self, index):
        dataroot, patient_id, study_date, patient_dir, BAT, xyz, dsc_signal, dsc_signal_maxv = self.dataset[index]

        import sys
        sys.stdout.flush()

        if len(dsc_signal.shape) == 1:
            dsc_signal = np.expand_dims(dsc_signal, axis=0)

        meta_dict = {
            'dim': dsc_signal.shape,
            'channel_dim': 0
        }

        try:
            data = self.transform({
                'dsc_signal': dsc_signal,
                'dsc_signal_maxv': dsc_signal_maxv,
                'dsc_signal_meta_dict': meta_dict
            })
        except Exception as e:
            print("[ERROR] Transformation failed!")
            raise e

        data.update({
            'dataroot': dataroot,
            'patient_id': patient_id,
            'study_date': study_date,
            'patient_dir': patient_dir,
            'BAT': BAT,
            'xyz': xyz
        })

        return data

    def __len__(self):
        return len(self.dataset)

    def _scan(self, dataset_):
        dataset = [item for sub in map(self._single_scan, dataset_) for item in sub]
        self.dataset += dataset

    def _single_scan(self, dataset_):
        dataroot, patient_id, study_date, BAT, brain_index = dataset_

        patient_dir = os.path.join(dataroot, 'nifti', patient_id, study_date)
        dsc_path = os.path.join(patient_dir, 'dsc.nii.gz')
        brain_mask_path = os.path.join(patient_dir, 'brain_mask.nii.gz')
        tumor_mask_path = os.path.join(patient_dir, 'corrected_tumor_mask.nii.gz')

        dsc = ants.image_read(dsc_path).numpy()
        brain_mask = ants.image_read(brain_mask_path).numpy()
        tumor_mask = ants.image_read(tumor_mask_path).numpy() > 0

        dsc[dsc < 0] = 0
        dsc_signal = dsc[tumor_mask]

        dsc_signal_maxv = np.quantile(dsc_signal, 0.995, axis=-1, keepdims=True)
        dsc_signal = dsc_signal / (dsc_signal_maxv + 1e-8)

        brain_index = tumor_mask.nonzero()
        brain_index = list(zip(*brain_index))

        sub_data = []
        for i in range(len(brain_index)):
            sub_data.append([dataroot, patient_id, study_date, patient_dir, BAT, brain_index[i], dsc_signal[i], dsc_signal_maxv[i]])

        return sub_data

    def append_internal(self):

        dataroot_internal = f'{self.dataroot}/internal/GBM/projects/quantized_DSC'

        json_path = os.path.join(dataroot_server, 'internal/GBM/projects/quantized_DSC/source_code', f'{self.phase}_list.json')
        self.patient_list = json.load(open(json_path, 'r'))

        patient_dir_list = glob(f'{dataroot_internal}/nifti/*/*')
        patient_dir_list = [p for p in patient_dir_list if os.path.exists(f'{p}/corrected_tumor_mask.nii.gz')]
        patient_dir_list = [p for p in patient_dir_list if p.split('/')[-2] in self.patient_list]

        dataset_internal = [[dataroot_internal, *patient_dir.split('/')[-2:], 0, [0]] for patient_dir in patient_dir_list]
        self._scan(dataset_internal)

        self.dataset_xyz_idx_dict.update({f"{patient_dir.split('/')[-2]}_{patient_dir.split('/')[-1]}": {'BAT': 0, 'brain_index': [0]} for patient_dir in patient_dir_list})

class DSCTrain(DSC):
    def __init__(self, dataroot, transform):
        super().__init__(dataroot, phase='train', transform=transform)

class DSCValidation(DSC):
    def __init__(self, dataroot, transform):
        super().__init__(dataroot, phase='valid', transform=transform)

class DSCTest(Dataset):
    def __init__(self, dataroot, transform, center='internal', device=None):
        super().__init__()
        self.dataroot = dataroot
        self.phase = 'test'
        self.transform = transform
        self.center = center
        print(device)
        print(type(device))
        self.device = device

        self.dataset = []
        self.dataset_xyz_idx_dict = {}

        self.append_center(center)

        print(f'Total test dataset size - volume: {len(self.dataset_xyz_idx_dict)}')

    def __getitem__(self, index):
        dataroot, patient_id, study_date, patient_dir, dsc_signal, dsc_signal_maxv, brain_mask, tumor_mask = self._single_scan(self.dataset[index])

        return {'dataroot': dataroot, 'patient_id': patient_id, 'study_date': study_date, 'patient_dir': patient_dir, 'dsc_signal': dsc_signal, 'dsc_signal_maxv': dsc_signal_maxv, 'brain_mask': brain_mask, 'tumor_mask': tumor_mask}

    def __len__(self):
        return len(self.dataset)

    def _single_scan(self, dataset_):
        dataroot, patient_id, study_date, BAT, brain_index = dataset_
        patient_dir = os.path.join(dataroot, 'nifti', patient_id, study_date)
        dsc_path = os.path.join(patient_dir, 'dsc.nii.gz')
        brain_mask_path = os.path.join(patient_dir, 'brain_mask.nii.gz')
        tumor_mask_path = os.path.join(patient_dir, 'corrected_tumor_mask.nii.gz')

        dsc_signal = ants.image_read(dsc_path).numpy()
        brain_mask = ants.image_read(brain_mask_path).numpy()
        tumor_mask = ants.image_read(tumor_mask_path).numpy() > 0

        dsc_signal[dsc_signal < 0] = 0
        dsc_signal_maxv = np.quantile(dsc_signal, 0.995, axis=-1, keepdims=True)
        dsc_signal = dsc_signal / (dsc_signal_maxv + 1e-8)

        dsc_signal = torch.tensor(dsc_signal, dtype=torch.float32)
        dsc_signal_maxv = torch.tensor(dsc_signal_maxv, dtype=torch.float32)
        brain_mask = torch.tensor(brain_mask, dtype=torch.bool)
        tumor_mask = torch.tensor(tumor_mask, dtype=torch.bool)

        return [dataroot, patient_id, study_date, patient_dir, dsc_signal, dsc_signal_maxv, brain_mask, tumor_mask]

    def append_center(self, center):
        dataroot_center = f'{self.dataroot}/{center}/GBM/projects/quantized_DSC'

        patient_dir_list = glob(f'{dataroot_center}/nifti/*/*')
        patient_dir_list = [p for p in patient_dir_list if os.path.exists(f'{p}/corrected_tumor_mask.nii.gz')]

        if center == 'internal':
            train_json_path = os.path.join(dataroot_server, f'{center}/GBM/projects/quantized_DSC/source_code', f'train_list.json')
            self.patient_list = json.load(open(train_json_path, 'r'))
            patient_dir_list = [p for p in patient_dir_list if p.split('/')[-2] in self.patient_list]
        else:
            test_json_path = os.path.join(dataroot_server, f'{center}/GBM/projects/quantized_DSC/source_code', f'test_list.json')
            self.patient_list = json.load(open(test_json_path, 'r'))
            patient_dir_list = [p for p in patient_dir_list if p.split('/')[-2] in self.patient_list]
            patient_dir_list = [p for p in patient_dir_list if not os.path.exists(f'/mnt/hdd3/hjang/data/internal/GBM/projects/quantized_DSC/nifti/{p.split("/")[-2]}/{p.split("/")[-1]}/dsc.nii.gz')]

        dataset_center = [[dataroot_center, *patient_dir.split('/')[-2:], 0, [0]] for patient_dir in patient_dir_list]
        self.dataset += dataset_center

        self.dataset_xyz_idx_dict.update({f"{patient_dir.split('/')[-2]}_{patient_dir.split('/')[-1]}": {'BAT': 0, 'brain_index': [0]} for patient_dir in patient_dir_list})

class DSCTest_Task(Dataset):
    def __init__(self, dataroot, transform, center='internal', task='quantized_DSC', device=None):
        super().__init__()
        self.dataroot = dataroot
        self.phase = 'test'
        self.transform = transform
        self.center = center
        self.task = task
        self.device = device

        self.dataset = []
        self.dataset_xyz_idx_dict = {}

        self.append_center(center, task)

        print(f'Total test dataset size - volume: {len(self.dataset_xyz_idx_dict)}')

    def __getitem__(self, index):
        dataroot, patient_id, study_date, patient_dir, dsc_signal, dsc_signal_maxv, brain_mask, tumor_mask = self._single_scan(self.dataset[index])

        return {'dataroot': dataroot, 'patient_id': patient_id, 'study_date': study_date, 'patient_dir': patient_dir, 'dsc_signal': dsc_signal, 'dsc_signal_maxv': dsc_signal_maxv, 'brain_mask': brain_mask, 'tumor_mask': tumor_mask}

    def __len__(self):
        return len(self.dataset)

    def _single_scan(self, dataset_):
        dataroot, patient_id, study_date, BAT, brain_index = dataset_
        patient_dir = os.path.join(dataroot, 'nifti', patient_id, study_date)
        dsc_path = os.path.join(patient_dir, 'dsc.nii.gz')
        brain_mask_path = os.path.join(patient_dir, 'brain_mask.nii.gz')
        tumor_mask_path = os.path.join(patient_dir, 'corrected_tumor_mask.nii.gz')

        dsc_signal = ants.image_read(dsc_path).numpy()
        brain_mask = ants.image_read(brain_mask_path).numpy()
        tumor_mask = ants.image_read(tumor_mask_path).numpy() > 0

        dsc_signal[dsc_signal < 0] = 0
        dsc_signal_maxv = np.quantile(dsc_signal, 0.995, axis=-1, keepdims=True)
        dsc_signal = dsc_signal / (dsc_signal_maxv + 1e-8)

        dsc_signal = torch.tensor(dsc_signal, dtype=torch.float32)
        dsc_signal_maxv = torch.tensor(dsc_signal_maxv, dtype=torch.float32)
        brain_mask = torch.tensor(brain_mask, dtype=torch.bool)
        tumor_mask = torch.tensor(tumor_mask, dtype=torch.bool)

        return [dataroot, patient_id, study_date, patient_dir, dsc_signal, dsc_signal_maxv, brain_mask, tumor_mask]

    def append_center(self, center, task):
        dataroot_center = f'{self.dataroot}/{center}/GBM/projects/{task}'
        patient_dir_list = glob(f'{dataroot_center}/nifti/*/*')
        patient_dir_list = [p for p in patient_dir_list if os.path.exists(f'{p}/corrected_tumor_mask.nii.gz')]

        if center == 'internal':
            train_json_path = os.path.join(dataroot_server, f'{center}/GBM/projects/quantized_DSC/source_code', f'train_list.json')
            valid_json_path = os.path.join(dataroot_server, f'{center}/GBM/projects/quantized_DSC/source_code', f'valid_list.json')
            test_json_path = os.path.join(dataroot_server, f'{center}/GBM/projects/quantized_DSC/source_code', f'test_list.json')
            self.patient_list = json.load(open(train_json_path, 'r')) + json.load(open(valid_json_path, 'r')) + json.load(open(test_json_path, 'r'))
            patient_dir_list = [p for p in patient_dir_list if p.split('/')[-2] in self.patient_list]
        else:
            test_json_path = os.path.join(dataroot_server, f'{center}/GBM/projects/quantized_DSC/source_code', f'test_list.json')
            self.patient_list = json.load(open(test_json_path, 'r'))
            patient_dir_list = [p for p in patient_dir_list if p.split('/')[-2] in self.patient_list]
            patient_dir_list = [p for p in patient_dir_list if not os.path.exists(f'/mnt/hdd3/hjang/data/internal/GBM/projects/quantized_DSC/nifti/{p.split("/")[-2]}/{p.split("/")[-1]}/dsc.nii.gz')]

        dataset_center = [
            [dataroot_center, *patient_dir.split('/')[-2:], 0, [0]]
            for patient_dir in patient_dir_list
        ]
        self.dataset += dataset_center
        self.dataset_xyz_idx_dict.update(
            {
                f"{patient_dir.split('/')[-2]}_{patient_dir.split('/')[-1]}": {
                    'BAT': 0, 'brain_index': [0]
                }
                for patient_dir in patient_dir_list
            }
        )

class DSCTestWhole(Dataset):
    def __init__(self, dataroot, transform, device=None):
        super().__init__()
        self.dataroot = dataroot
        self.phase = 'test'
        self.transform = transform
        print(device)
        print(type(device))
        self.device = device

        self.dataset = []
        self.dataset_xyz_idx_dict = {}

        self.append_internal()

        print(f'Total test dataset size - volume: {len(self.dataset_xyz_idx_dict)}')

    def __getitem__(self, index):
        dataroot, patient_id, study_date, patient_dir, dsc_signal, dsc_signal_maxv, brain_mask, tumor_mask = self._single_scan(self.dataset[index])

        return {'dataroot': dataroot, 'patient_id': patient_id, 'study_date': study_date, 'patient_dir': patient_dir, 'dsc_signal': dsc_signal, 'dsc_signal_maxv': dsc_signal_maxv, 'brain_mask': brain_mask, 'tumor_mask': tumor_mask}

    def __len__(self):
        return len(self.dataset)

    def _single_scan(self, dataset_):
        dataroot, patient_id, study_date, BAT, brain_index = dataset_
        patient_dir = os.path.join(dataroot, 'nifti', patient_id, study_date)
        dsc_path = os.path.join(patient_dir, 'dsc.nii.gz')
        brain_mask_path = os.path.join(patient_dir, 'brain_mask.nii.gz')
        tumor_mask_path = os.path.join(patient_dir, 'corrected_tumor_mask.nii.gz')

        dsc_signal = ants.image_read(dsc_path).numpy()
        brain_mask = ants.image_read(brain_mask_path).numpy()
        tumor_mask = ants.image_read(tumor_mask_path).numpy() > 0

        dsc_signal[dsc_signal < 0] = 0
        dsc_signal_maxv = np.quantile(dsc_signal, 0.995, axis=-1, keepdims=True)
        dsc_signal = dsc_signal / (dsc_signal_maxv + 1e-8)

        dsc_signal = torch.tensor(dsc_signal, dtype=torch.float32)
        dsc_signal_maxv = torch.tensor(dsc_signal_maxv, dtype=torch.float32)
        brain_mask = torch.tensor(brain_mask, dtype=torch.bool)
        tumor_mask = torch.tensor(tumor_mask, dtype=torch.bool)

        return [dataroot, patient_id, study_date, patient_dir, dsc_signal, dsc_signal_maxv, brain_mask, tumor_mask]

    def append_internal(self):
        dataroot_internal = f'{self.dataroot}/internal/GBM/projects/quantized_DSC'

        train_json_path = os.path.join(dataroot_server, 'internal/GBM/projects/quantized_DSC/source_code', f'train_list.json')
        valid_json_path = os.path.join(dataroot_server, 'internal/GBM/projects/quantized_DSC/source_code', f'valid_list.json')
        test_json_path = os.path.join(dataroot_server, 'internal/GBM/projects/quantized_DSC/source_code', f'test_list.json')
        self.patient_list = json.load(open(train_json_path, 'r')) + json.load(open(valid_json_path, 'r')) + json.load(open(test_json_path, 'r'))

        patient_dir_list = glob(f'{dataroot_internal}/nifti/*/*')
        patient_dir_list = [p for p in patient_dir_list if os.path.exists(f'{p}/corrected_tumor_mask.nii.gz')]
        patient_dir_list = [p for p in patient_dir_list if p.split('/')[-2] in self.patient_list]

        patient_dir_list = patient_dir_list[:4]

        dataset_internal = [[dataroot_internal, *patient_dir.split('/')[-2:], 0, [0]] for patient_dir in patient_dir_list]
        self.dataset += dataset_internal

        self.dataset_xyz_idx_dict.update({f"{patient_dir.split('/')[-2]}_{patient_dir.split('/')[-1]}": {'BAT': 0, 'brain_index': [0]} for patient_dir in patient_dir_list})
