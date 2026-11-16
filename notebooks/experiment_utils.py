import hashlib
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLIT_SEED = 42
TARGET_FPR = 0.01
TRAIN_END, VAL_END = 0.6, 0.9  # 60% train, 30% val, 10% test
METRICS = ["accuracy", "recall", "f1", "precision", "roc_auc", "aucpr", "fpr"]
ID_SOURCES = ["nazario", "nigerian", "trec05", "trec06", "trec07"]
OOD_SOURCES = ["ferb", "kd10000", "nebulon", "zeroday"]

INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CORPUS_COLORS = {"ID": "#2a78d6", "OOD": "#eb6834"}
mpl.rcParams.update({
    "pdf.fonttype": 42, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 110,
})


def load_corpora(split_seed=SPLIT_SEED):
    in_domain = pd.read_parquet(DATA / "meajor_processed.parquet")
    in_domain["split"] = make_split(in_domain, split_seed)
    return in_domain, pd.read_parquet(DATA / "external_processed.parquet")


def stable_hash(values, seed):
    return np.array([
        int.from_bytes(hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest(), "big") >> 1 for value in values
    ], dtype=np.int64)


def make_split(frame, seed=SPLIT_SEED):
    stratum = frame.source + "|" + frame.kind
    keyed = pd.DataFrame({"cluster": frame.cluster.to_numpy(), "stratum": stratum.to_numpy(), "raw_key": frame.raw_key.to_numpy()})
    clusters = keyed.groupby(["cluster", "stratum"]).size().rename("count").reset_index()
    # Klaster należy do warstwy, która ma w nim najwięcej dokumentów (remisy rozstrzyga kolejność alfabetyczna).
    clusters = clusters.sort_values(["cluster", "count", "stratum"], ascending=[True, False, True]).drop_duplicates("cluster")
    clusters["size"] = clusters.cluster.map(keyed.groupby("cluster").size())
    clusters["key"] = stable_hash(clusters.cluster.map(keyed.groupby("cluster")["raw_key"].min()), seed)
    assignment = {}
    for _, group in clusters.groupby("stratum", sort=True):
        group = group.sort_values("key")
        position = (group["size"].cumsum() - group["size"] / 2) / group["size"].sum()
        names = np.select([position < TRAIN_END, position < VAL_END], ["train", "val"], default="test")
        assignment.update(zip(group.cluster, names))
    return frame.cluster.map(assignment)


def cluster_weights(clusters):
    clusters = pd.Series(np.asarray(clusters))
    return 1 / clusters.groupby(clusters).transform("size").to_numpy()


def cluster_folds(labels, clusters, n_folds, seed):
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(folds.split(np.zeros(len(labels)), np.asarray(labels), np.asarray(clusters)))


class WordCharTfidf:
    def __init__(self, max_features=100_000):
        options = dict(min_df=3, max_features=max_features, sublinear_tf=True)
        self.word = TfidfVectorizer(ngram_range=(1, 2), **options)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **options)

    def fit(self, texts, labels):
        matrix = sp.hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr().astype(np.float32)
        self.ranking = np.argsort(-np.nan_to_num(chi2(matrix, np.asarray(labels))[0]), kind="stable")
        return self

    def transform_all(self, texts):
        return sp.hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr().astype(np.float32)

    def selected(self, n_features):
        return self.ranking[:n_features]


def fasttext_embeddings(*text_series):
    import fasttext

    model = fasttext.load_model(str(ROOT / "cc.en.300.bin"))
    embeddings = [
        np.stack([model.get_sentence_vector(text.replace("\n", " ")) for text in texts]).astype(np.float32)
        for texts in text_series
    ]
    del model
    return embeddings


def threshold_at_fpr(labels, scores, clusters, target_fpr=TARGET_FPR):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    weights = cluster_weights(clusters)
    fpr, _, thresholds = roc_curve(labels, scores, sample_weight=weights, drop_intermediate=False)
    threshold = thresholds[np.flatnonzero(fpr <= target_fpr)[-1]]
    predictions = scores >= threshold
    return {
        "próg": float(threshold),
        "fpr_ważony_val": float(weights[predictions & (labels == 0)].sum() / weights[labels == 0].sum()),
        "fpr_val": float(predictions[labels == 0].mean()),
        "recall_val": float(predictions[labels == 1].mean()),
    }


def compute_metrics(labels, scores, threshold):
    labels = np.asarray(labels)
    predictions = np.asarray(scores) >= threshold
    tp = int((predictions & (labels == 1)).sum())
    fp = int((predictions & (labels == 0)).sum())
    fn = int((~predictions & (labels == 1)).sum())
    tn = int((~predictions & (labels == 0)).sum())
    both_classes = (tp + fn > 0) and (fp + tn > 0)
    return {
        "n": len(labels),
        "positives": tp + fn,
        "accuracy": (tp + tn) / len(labels),
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "roc_auc": roc_auc_score(labels, scores) if both_classes else np.nan,
        "aucpr": average_precision_score(labels, scores) if both_classes else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def corpus_metrics(sets, threshold):
    rows = {name: compute_metrics(labels, scores, threshold) for name, (labels, scores) in sets.items()}
    return pd.DataFrame(rows).T[["n", "positives", *METRICS]]


def plot_confusion_matrices(sets, threshold, path):
    figure, axes = plt.subplots(1, len(sets), figsize=(4 * len(sets), 3.4))
    matrices = {}
    for axis, (name, (labels, scores)) in zip(np.atleast_1d(axes), sets.items()):
        result = compute_metrics(labels, scores, threshold)
        matrix = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
        matrices[name] = pd.DataFrame(matrix, index=["prawdziwe 0", "prawdziwe 1"], columns=["przewidziane 0", "przewidziane 1"])
        shares = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                color = "white" if shares[row, column] > 0.55 else INK
                axis.text(column, row, f"{matrix[row, column]:,}".replace(",", " ") + f"\n{shares[row, column]:.1%}", ha="center", va="center", color=color)
        axis.set_xticks([0, 1], ["0", "1"])
        axis.set_yticks([0, 1], ["0", "1"])
        axis.set(xlabel="Klasa przewidziana", ylabel="Klasa prawdziwa")
        axis.set_title(name, loc="left")
        axis.grid(False)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.show()
    return matrices


def plot_curves(sets, threshold, path):
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    for name, (labels, scores) in sets.items():
        color = CORPUS_COLORS[name]
        result = compute_metrics(labels, scores, threshold)
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[0].plot(fpr, tpr, color=color, linewidth=1.8, label=f"{name} (AUC = {result['roc_auc']:.4f})")
        axes[0].scatter(result["fpr"], result["recall"], s=45, color=color, edgecolors="white", zorder=3)
        axes[1].plot(recall, precision, color=color, linewidth=1.8, label=f"{name} (AUCPR = {result['aucpr']:.4f})")
        axes[1].scatter(result["recall"], result["precision"], s=45, color=color, edgecolors="white", zorder=3)
    axes[0].set(xscale="log", xlim=(1e-4, 1), xlabel="FPR (skala log.)", ylabel="Recall", ylim=(0, 1.02))
    axes[0].axvline(TARGET_FPR, color=INK_SECONDARY, linewidth=0.8, linestyle="--")
    axes[0].set_title("Krzywa ROC (kropka: punkt pracy)", loc="left")
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.02), ylim=(0, 1.02))
    axes[1].set_title("Krzywa precision-recall", loc="left")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.show()
