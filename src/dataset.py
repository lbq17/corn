import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
import torch

def load_spectra_csv(path, spectra_cols=None, label_col='label'):
    df = pd.read_csv(path)
    labels = df[label_col].values
    if spectra_cols is None:
        spectra = df.drop(columns=[label_col]).values
    else:
        spectra = df[spectra_cols].values
    return spectra.astype(np.float32), labels

def load_spectra_mat(path, spectra_key='spectra', labels_key='labels'):
    mat = sio.loadmat(path)
    spectra = np.array(mat[spectra_key], dtype=np.float32)
    labels = np.array(mat[labels_key]).squeeze()
    return spectra, labels

def preprocess_spectra(spectra, scaler='standard', pca_components=None):
    X = spectra.copy()
    sc = None
    if scaler == 'standard':
        sc = StandardScaler()
        X = sc.fit_transform(X)
    elif scaler == 'minmax':
        sc = MinMaxScaler()
        X = sc.fit_transform(X)
    if pca_components is not None and pca_components < X.shape[1]:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
        return X.astype(np.float32), sc, pca
    return X.astype(np.float32), sc, None

class SpectraDataset(Dataset):
    def __init__(self, spectra, labels, label_map=None):
        self.X = spectra
        self.y_raw = labels
        if label_map is None:
            uniq = np.unique(labels)
            self.label_map = {v:i for i,v in enumerate(uniq)}
        else:
            self.label_map = label_map
        self.y = np.array([self.label_map[v] for v in self.y_raw], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x).float(), int(self.y[idx])
