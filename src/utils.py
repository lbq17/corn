import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_preds(y_true, y_pred, labels=None, class_names=None, out_prefix=None):
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    results = {'OA':oa, 'Kappa':kappa, 'report':report, 'confusion':cm}
    if out_prefix:
        with open(out_prefix + '_report.txt','w') as f:
            f.write(f"OA: {oa}\nKappa: {kappa}\n\n")
            f.write(report)
    return results

def plot_confusion(cm, class_names, figsize=(8,6), cmap='Blues'):
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def plot_mean_spectra(spectra, labels, class_map=None):
    uniq = np.unique(labels)
    plt.figure(figsize=(8,6))
    for u in uniq:
        idx = labels==u
        mean = spectra[idx].mean(axis=0)
        std = spectra[idx].std(axis=0)
        name = class_map[u] if class_map and u in class_map else str(u)
        plt.plot(mean, label=name)
        plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2)
    plt.xlabel('Band index'); plt.ylabel('Reflectance (preproc)')
    plt.legend()
    plt.tight_layout()
    plt.show()
