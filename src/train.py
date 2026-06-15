import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from dataset import load_spectra_csv, load_spectra_mat, preprocess_spectra, SpectraDataset
from utils import evaluate_preds, plot_mean_spectra
import torch
from torch.utils.data import DataLoader
import torch.nn as nn, torch.optim as optim
from models import Simple1DCNN
from tqdm import tqdm

def train_sklearn(X_train, y_train, X_test, y_test, clf_name='rf', out_prefix='sk_'):
    if clf_name == 'svm':
        clf = SVC(kernel='rbf', probability=True)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    res = evaluate_preds(y_test, y_pred)
    print(res['report'])
    dump(clf, out_prefix + 'model.joblib')
    return clf, res

def train_torch(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, epochs=50, batch_size=32, lr=1e-3, device=None, out_prefix='torch_'):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = SpectraDataset(X_train, y_train)
    val_ds = SpectraDataset(X_val, y_val, label_map=train_ds.label_map)
    test_ds = SpectraDataset(X_test, y_test, label_map=train_ds.label_map)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = Simple1DCNN(in_channels=1, in_length=X_train.shape[1], num_classes=num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    best_val = 0.0
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        loss_sum = 0.0
        for x,y in train_loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
        loss_avg = loss_sum / len(train_loader.dataset)
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for x,y in val_loader:
                x = x.to(device)
                out = model(x)
                p = out.argmax(dim=1).cpu().numpy()
                preds.append(p); ys.append(y.numpy())
        ys = np.concatenate(ys); preds = np.concatenate(preds)
        from sklearn.metrics import accuracy_score
        val_acc = accuracy_score(ys, preds)
        print(f"Epoch {epoch} loss {loss_avg:.4f} val_acc {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()
    model.load_state_dict(best_state)
    ys, preds = [], []
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()
            preds.append(p); ys.append(y.numpy())
    ys = np.concatenate(ys); preds = np.concatenate(preds)
    res = evaluate_preds(ys, preds)
    torch.save(best_state, out_prefix + 'best.pth')
    print(res['report'])
    return model, res

def main(args):
    if args.format == 'csv':
        X, y = load_spectra_csv(args.data_path, label_col=args.label_col)
    else:
        X, y = load_spectra_mat(args.data_path, spectra_key=args.spectra_key, labels_key=args.labels_key)
    if y.dtype.kind not in 'iu':
        uniq = np.unique(y)
        label_map = {v:i for i,v in enumerate(uniq)}
        y = np.array([label_map[v] for v in y], dtype=np.int64)
    X_proc, sc, pca = preprocess_spectra(X, scaler=args.scaler, pca_components=(args.pca if args.pca>0 else None))
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_proc, y, test_size=args.test_size+args.val_size, stratify=y, random_state=0)
    val_ratio = args.val_size / (args.val_size + args.test_size)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=1-val_ratio, stratify=y_tmp, random_state=0)
    num_classes = len(np.unique(y_tr))
    print("Shapes:", X_tr.shape, X_val.shape, X_te.shape, "Num classes:", num_classes)
    if args.mode == 'sklearn':
        clf, res = train_sklearn(X_tr, y_tr, X_te, y_te, clf_name=args.clf, out_prefix=args.out_prefix)
    else:
        model, res = train_torch(X_tr, y_tr, X_val, y_val, X_te, y_te, num_classes, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, out_prefix=args.out_prefix)
    plot_mean_spectra(X_proc, y, class_map=None)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--format', choices=['csv','mat'], default='csv')
    p.add_argument('--label_col', type=str, default='label')
    p.add_argument('--spectra_key', type=str, default='spectra')
    p.add_argument('--labels_key', type=str, default='labels')
    p.add_argument('--scaler', choices=['standard','minmax','none'], default='standard')
    p.add_argument('--pca', type=int, default=0)
    p.add_argument('--mode', choices=['sklearn','torch'], default='sklearn')
    p.add_argument('--clf', choices=['rf','svm'], default='rf')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--val_size', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out_prefix', type=str, default='out')
    args = p.parse_args()
    main(args)
