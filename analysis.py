"""Few-shot HAR benchmark: MLP baseline (stand-in for residual-SE net) vs class-balanced LightGBM with inner-CV tuning."""
import io
import json
import sys
import time
import warnings

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.io import arff
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42                               # global seed
EPISODE_SEEDS = [42, 43, 44, 45, 46]    # 5 few-shot episodes (also model seeds)
SHOTS = 10                              # labelled examples per class
TEST_SIZE = 0.3                         # single stratified split
MAX_ROWS = 200_000                      # subsample guard for runtime
DATA_ID = 1478                          # OpenML UCI HAR
API = "https://www.openml.org/api/v1/json/data/{}"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get(url):
    # GET with timeout=60 and exponential backoff on 429/5xx and network errors
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            wait = 2 ** attempt
            log(f"download error ({e}); retrying in {wait}s")
            time.sleep(wait)
    sys.exit("download failed after retries")


def load_data():
    # Fetch OpenML metadata, then the ARFF file; parse numeric features and the target column
    desc = json.loads(get(API.format(DATA_ID)))["data_set_description"]
    data, _ = arff.loadarff(io.BytesIO(get(desc["url"])))
    target = desc["default_target_attribute"]
    feats = [n for n in data.dtype.names if n != target]
    X = np.column_stack([data[n] for n in feats]).astype(float)
    y = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in data[target]])
    return X, y, desc["name"], desc["url"]


def sample_episode(y_pool, seed):
    # 10 labelled examples per class drawn from the training pool
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.choice(np.where(y_pool == c)[0], SHOTS, replace=False)
                           for c in np.unique(y_pool)])


def run_baseline(Xs, ys, Xt, seed):
    # Standard MLP stand-in for the residual-SE network; standardized inputs, no tuning
    sc = StandardScaler().fit(Xs)
    clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=1000, random_state=seed)
    clf.fit(sc.transform(Xs), ys)
    return clf.predict(sc.transform(Xt))


def run_improved(Xs, ys, Xt, seed):
    # Class-balanced LightGBM; hyperparameters chosen by 3-fold stratified CV inside the shots only
    sc = StandardScaler().fit(Xs)
    base = lgb.LGBMClassifier(class_weight="balanced", learning_rate=0.05, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.5, random_state=seed,
                              n_jobs=1, verbose=-1)
    grid = {"num_leaves": [4, 8], "n_estimators": [50, 150], "min_child_samples": [2, 5]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(base, grid, scoring="f1_macro", cv=cv, n_jobs=1)
    search.fit(sc.transform(Xs), ys)
    log(f"  LightGBM best CV params: {search.best_params_}")
    return search.predict(sc.transform(Xt))


def scores(y_true, y_pred):
    # Main metric macro-F1; secondary: accuracy and balanced accuracy (mean per-class recall)
    return (f1_score(y_true, y_pred, average="macro"),
            accuracy_score(y_true, y_pred),
            balanced_accuracy_score(y_true, y_pred))


def main():
    X, y, name, url = load_data()
    log(f"loaded {X.shape[0]} rows, {X.shape[1]} features, {len(np.unique(y))} classes")
    if len(y) > MAX_ROWS:  # runtime guard (not triggered for HAR)
        keep, _ = train_test_split(np.arange(len(y)), train_size=MAX_ROWS, stratify=y, random_state=SEED)
        X, y = X[keep], y[keep]

    # Single stratified split; the test set is only used for the final evaluation
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    log(f"train pool {len(y_tr)}, test {len(y_te)}")

    b_res, i_res = [], []
    for s in EPISODE_SEEDS:
        idx = sample_episode(y_tr, s)
        Xs, ys = X_tr[idx], y_tr[idx]
        log(f"episode seed {s}: {len(ys)} shots")
        b_res.append(scores(y_te, run_baseline(Xs, ys, X_te, s)))
        i_res.append(scores(y_te, run_improved(Xs, ys, X_te, s)))
        log(f"  baseline macro-F1 {b_res[-1][0]:.4f} | improved macro-F1 {i_res[-1][0]:.4f}")

    b, i = np.array(b_res), np.array(i_res)
    b_mean, i_mean = b.mean(axis=0), i.mean(axis=0)
    if not (np.isfinite(b_mean).all() and np.isfinite(i_mean).all()):
        sys.exit("non-finite metrics; cannot produce meaningful results")

    # Chart: macro-F1 per model (mean +/- std over episodes)
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["MLP baseline", "LightGBM (CV)"], [b_mean[0], i_mean[0]],
           yerr=[b[:, 0].std(), i[:, 0].std()], capsize=6, color=["gray", "tab:blue"])
    ax.set_ylabel("Test macro-F1 (mean ± std)")
    ax.set_title("10-shot per class, 5 episodes")
    fig.tight_layout()
    fig.savefig("figures/macro_f1.png", dpi=120)

    results = {
        "question": ("Can a class-balanced LightGBM model with cross-validated tuning, averaged over "
                     "several few-shot episodes, beat a neural-network stand-in for the paper's "
                     "residual-SE few-shot detector on a public sensor-based activity dataset with "
                     "only 10 labelled examples per class?"),
        "method": ("UCI HAR (OpenML 1478) as stand-in for road-alert sensor signals. 5 episodes of "
                   "10-shot-per-class training sampling from a stratified 70% pool; evaluation on the "
                   "held-out 30% test split. Baseline: MLPClassifier on standardized features. "
                   "Improved: class-balanced LightGBM with 3-fold stratified CV grid search inside "
                   "each few-shot set. Metric: macro-F1 on the test split."),
        "dataset": {
            "name": name,
            "source_url": "https://www.openml.org/d/1478",
            "n_rows": int(len(y)),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 (test)",
        "higher_is_better": True,
        "baseline": "MLPClassifier (256 hidden units, standardized inputs), stand-in for residual-SE network",
        "improved": ("Class-balanced LightGBM; num_leaves, n_estimators and min_child_samples chosen "
                     "by 3-fold stratified CV on the few-shot training set only"),
        "metrics": {
            "baseline_score": float(b_mean[0]),
            "improved_score": float(i_mean[0]),
            "improvement": float(i_mean[0] - b_mean[0]),
            "baseline_score_std": float(b[:, 0].std()),
            "improved_score_std": float(i[:, 0].std()),
            "baseline_accuracy": float(b_mean[1]),
            "improved_accuracy": float(i_mean[1]),
            "baseline_balanced_accuracy": float(b_mean[2]),
            "improved_balanced_accuracy": float(i_mean[2]),
            "train_shots_per_class": SHOTS,
            "n_episodes": len(EPISODE_SEEDS),
        },
        "notes": ("OpenML 1478 is distributed as one combined table, so the official UCI train/test "
                  "split is not recoverable; a single stratified 70/30 split (seed 42) is used instead. "
                  "This is a stand-in dataset, not the paper's road-alert signals, and the MLP is a "
                  "generic stand-in, not a reproduction of the residual-SE network. Results come from "
                  "5 episodes only, so differences may be within noise; std is over episodes."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"baseline macro-F1 {b_mean[0]:.4f} | improved macro-F1 {i_mean[0]:.4f} | wrote results.json")


if __name__ == "__main__":
    main()
