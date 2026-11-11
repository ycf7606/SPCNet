import torch
import random
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing

import os
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK']='True'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('HSI_DATA_ROOT', PROJECT_ROOT / 'data')).expanduser().resolve()

def dataset_path(*parts):
    return str(DATA_ROOT.joinpath(*parts))

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def applyPCA(data, n_components):
    h, w, b = data.shape
    pca = PCA(n_components=n_components)

    reshaped_data = np.reshape(data, (-1, b))

    transformed_data = pca.fit_transform(reshaped_data)

    transformed_data = np.reshape(transformed_data, (h, w, -1))

    pca_weights = pca.components_
    return transformed_data, pca_weights

def normalize3D(data):

    H, W, C = data.shape
    data_reshaped = data.reshape(-1, C)

    data_reshaped = np.asarray(data_reshaped, dtype=np.float64)

    data_normalized = preprocessing.scale(data_reshaped)

    data_normalized = data_normalized.reshape(H, W, C)

    return data_normalized

def normalize2D(data):

    if len(data.shape) == 3:
        H, W, C = data.shape
        data = data.reshape(H, W*C)
    else:
        H, W = data.shape
    data_reshaped = data.reshape(-1)

    data_reshaped = np.asarray(data_reshaped, dtype=np.float64)

    data_normalized = preprocessing.scale(data_reshaped)

    data_normalized = data_normalized.reshape(H, W)

    return data_normalized

class HXDataset(Dataset):
    def __init__(self, hsi_pca, X, pos, windowSize, hsi=None , gt=None, transform=None, train=False):
        modes = ['symmetric', 'reflect']
        self.train = train
        self.pad = windowSize // 2
        self.windowSize = windowSize
        self.hsi = None
        if hsi is not None:
            self.hsi = np.pad(hsi, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.hsi_pca = np.pad(hsi_pca, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.X = None
        if(len(X.shape) == 2):
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad)), mode=modes[windowSize % 2])
        elif(len(X.shape) == 3):
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.pos = pos
        self.gt = None
        if gt is not None:
            self.gt = gt
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        h, w = self.pos[index, :]
        hsi_pca = self.hsi_pca[h: h + self.windowSize, w: w + self.windowSize]
        if self.hsi is not None:
            hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize]
            hsi = hsi.astype(np.float32)
        X = self.X[h: h + self.windowSize, w: w + self.windowSize]
        if self.transform:
            hsi_pca = self.transform(hsi_pca).float()
            if self.hsi is not None:
                hsi = self.transform(hsi).float()
            X = self.transform(X).float()
            trans = [transforms.RandomHorizontalFlip(1.),
                     transforms.RandomVerticalFlip(1.)]
            if self.train:
                if random.random() < 0.5:
                    i = random.randint(0, 1)
                    hsi_pca = trans[i](hsi_pca)
                    if self.hsi is not None:
                        hsi = trans[i](hsi)
                    X = trans[i](X)
        if self.gt is not None:
            gt = torch.tensor(self.gt[h, w] - 1).long()
            if self.hsi is not None:
                return hsi_pca.unsqueeze(0),hsi.unsqueeze(0),X,gt
            return hsi_pca.unsqueeze(0), X, gt

        if self.hsi is not None:
            return hsi_pca.unsqueeze(0), hsi.unsqueeze(0), X, h, w

        return hsi_pca.unsqueeze(0), X, h, w

    def __len__(self):
        return self.pos.shape[0]

def getData(hsi_path, X_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers, datatype, pin_memory=True):
    hsi = loadmat(hsi_path)[keys[0]]
    X = loadmat(X_path)[keys[1]]
    gt = loadmat(gt_path)[keys[2]]
    train_index = loadmat(index_path)[keys[3]]
    test_index = loadmat(index_path)[keys[4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    all_index = loadmat(index_path)[keys[5]]

    hsi_pca, hsi_pca_wight = applyPCA(hsi, channels)

    hsi_pca = normalize3D(hsi_pca)
    hsi = normalize3D(hsi)
    if datatype in (0, 1):
        X = normalize2D(X)
    else:
        X = normalize3D(X)
    hsi_pca_wight = normalize2D(hsi_pca_wight)

    HXtrainset = HXDataset(hsi_pca, X, train_index,
                           windowSize, hsi=hsi, gt=gt, transform=ToTensor(), train=True)
    HXtestset = HXDataset(hsi_pca, X, test_index,
                           windowSize, hsi=hsi, gt=gt, transform=ToTensor())
    HXtrntstset = HXDataset(hsi_pca, X, trntst_index,
                                windowSize, hsi=hsi, transform=ToTensor())
    HXallset = HXDataset(hsi_pca, X, all_index,
                         windowSize, hsi=hsi, transform=ToTensor())

    train_loader = DataLoader(
        HXtrainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
    test_loader = DataLoader(
        HXtestset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=pin_memory)
    trntst_loader = DataLoader(
        HXtrntstset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=pin_memory)
    all_loader = DataLoader(
        HXallset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=pin_memory)
    print("Success!")
    return train_loader, test_loader, trntst_loader, all_loader, hsi_pca_wight

def create_evaluation_loader(dataset, batch_size, num_workers, pin_memory=True):
    """Build a persistent, prefetched loader for repeated validation passes."""
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(dataset, **kwargs)

def getHouston2013Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("Houston2013!")
    houston2013_keys = ['houston_hsi', 'houston_lidar', 'houston_gt', 'houston_train', 'houston_test', 'houston_all']
    return getData(hsi_path, lidar_path, gt_path, index_path, houston2013_keys, channels, windowSize, batch_size, num_workers, datatype=0)

def getMyData(datasetType, channels, windowSize, batch_size, num_workers):
    if datasetType != 0:
        raise ValueError(f"Unsupported dataset_type: {datasetType}")
    hsi_path = dataset_path('Houston2013', 'houston_hsi.mat')
    lidar_path = dataset_path('Houston2013', 'houston_lidar.mat')
    gt_path = dataset_path('Houston2013', 'houston_gt.mat')
    index_path = dataset_path('Houston2013', 'houston_index.mat')
    train_loader, test_loader, trntst_loader, all_loader, hsi_pca_wight = getHouston2013Data(
        hsi_path,
        lidar_path,
        gt_path,
        index_path,
        channels,
        windowSize,
        batch_size,
        num_workers,
    )
    return train_loader, test_loader, trntst_loader, all_loader ,hsi_pca_wight
