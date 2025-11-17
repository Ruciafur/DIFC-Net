# metrics_utils.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# -------------------------
# 阈值选择
# -------------------------
def select_threshold(y_true: np.ndarray, y_score: np.ndarray, method: str = "youden") -> float:
    """
    y_true: [N], 0/1 labels
    y_score: [N], sigmoid之前或之后都可。建议传入sigmoid之后的prob。
    method: 'youden' | 'f1'
    """
    # 如果是logits，转为prob
    p = 1 / (1 + np.exp(-y_score))
    fpr, tpr, thr = roc_curve(y_true, p)
    if method == "youden":
        j = tpr - fpr
        i = np.argmax(j)
        return thr[i]
    elif method == "f1":
        prec, rec, thr = precision_recall_curve(y_true, p)
        # PR曲线的阈值数量为 len(prec)-1；这里用等长策略
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        i = np.nanargmax(f1)
        # precision_recall_curve 的 thr 与 f1对齐
        return thr[i]
    else:
        raise ValueError("Unknown threshold method.")

# -------------------------
# 操作点指标
# -------------------------
def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    p = 1 / (1 + np.exp(-y_score))
    fpr, tpr, thr = roc_curve(y_true, p)
    # 找到最接近 target_fpr 的点
    idx = np.argmin(np.abs(fpr - target_fpr))
    return tpr[idx]

def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    p = 1 / (1 + np.exp(-y_score))
    fpr, tpr, thr = roc_curve(y_true, p)
    idx = np.argmin(np.abs(tpr - target_tpr))
    return fpr[idx]

def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> float:
    p = 1 / (1 + np.exp(-y_score))
    fpr, tpr, thr = roc_curve(y_true, p)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer

# -------------------------
# 校准：ECE / Brier
# -------------------------
def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    p = 1 / (1 + np.exp(-y_score))
    return np.mean((p - y_true) ** 2)

def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> float:
    p = 1 / (1 + np.exp(-y_score))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        inds = (p >= bins[i]) & (p < bins[i+1])
        if np.any(inds):
            acc = np.mean(y_true[inds])
            conf = np.mean(p[inds])
            ece += np.abs(acc - conf) * (np.sum(inds) / len(y_true))
    return ece

def reliability_diagram(y_true: np.ndarray, y_score: np.ndarray, save_path: str, n_bins: int = 15):
    p = 1 / (1 + np.exp(-y_score))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    accs, confs = [], []
    for i in range(n_bins):
        inds = (p >= bins[i]) & (p < bins[i+1])
        if np.any(inds):
            accs.append(np.mean(y_true[inds]))
            confs.append(np.mean(p[inds]))
        else:
            accs.append(0.0); confs.append(0.0)
    plt.figure(figsize=(5, 5))
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.bar(bin_centers, np.array(accs) - np.array(confs), width=1.0/n_bins, alpha=0.6, label='Gap (Acc - Conf)')
    plt.plot(bin_centers, accs, 'o-', label='Accuracy', color='#1f77b4')
    plt.plot(bin_centers, confs, 'o-', label='Confidence', color='#ff7f0e')
    plt.xlabel('Confidence'); plt.ylabel('Value')
    plt.title('Reliability Diagram')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

# -------------------------
# 汇总评估
# -------------------------
def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray, thr: Optional[float]=None) -> Dict:
    p = 1 / (1 + np.exp(-y_score))
    auroc = roc_auc_score(y_true, p)
    auprc = average_precision_score(y_true, p)

    if thr is None:
        # 默认用 Youden 的阈值
        thr = select_threshold(y_true, y_score, method="youden")

    y_pred = (p >= thr).astype(np.int32)

    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp + 1e-12)
    sens = tp / (tp + fn + 1e-12)

    tpr1_fpr = tpr_at_fpr(y_true, y_score, target_fpr=0.01)
    fpr95_tpr = fpr_at_tpr(y_true, y_score, target_tpr=0.95)
    eer = equal_error_rate(y_true, y_score)
    brier = brier_score(y_true, y_score)
    ece = expected_calibration_error(y_true, y_score)

    return {
        "threshold": float(thr),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Accuracy": float(acc),
        "BalancedAcc": float(bacc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "Specificity": float(spec),
        "Sensitivity": float(sens),
        "MCC": float(mcc),
        "Kappa": float(kappa),
        "TPR@FPR=1%": float(tpr1_fpr),
        "FPR@TPR=95%": float(fpr95_tpr),
        "EER": float(eer),
        "Brier": float(brier),
        "ECE": float(ece),
    }

# -------------------------
# 曲线绘制
# -------------------------
def plot_roc_pr_det(y_true: np.ndarray, y_score: np.ndarray, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    p = 1 / (1 + np.exp(-y_score))

    # ROC
    fpr, tpr, thr = roc_curve(y_true, p)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUROC: {roc_auc_score(y_true, p):.3f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve"); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150, bbox_inches="tight"); plt.close()

    # PR
    prec, rec, thr = precision_recall_curve(y_true, p)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"AUPRC: {average_precision_score(y_true, p):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR Curve"); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(save_dir, "pr_curve.png"), dpi=150, bbox_inches="tight"); plt.close()

    # DET（把轴做成正态Z分数，近似）
    from scipy.stats import norm
    fpr[fpr<=1e-6]=1e-6; fpr[fpr>=1-1e-6]=1-1e-6
    tpr[tpr<=1e-6]=1e-6; tpr[tpr>=1-1e-6]=1-1e-6
    fnr = 1 - tpr
    plt.figure(figsize=(6,5))
    plt.plot(norm.ppf(fpr), norm.ppf(fnr))
    plt.xlabel("FPR (norm-ppf)"); plt.ylabel("FNR (norm-ppf)")
    plt.title("DET Curve"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "det_curve.png"), dpi=150, bbox_inches="tight"); plt.close()
