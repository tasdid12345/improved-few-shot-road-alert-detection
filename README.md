# Few-shot HAR benchmark (stand-in for road-alert detection)

This project compares a standard MLP baseline (stand-in for the paper's residual-SE network) against a class-balanced LightGBM model whose hyperparameters are tuned by 3-fold stratified cross-validation inside each 10-shot-per-class training set. Data: UCI HAR (OpenML 1478). Metric: macro-F1 on a held-out 30% stratified test split, averaged over 5 few-shot episodes (seeds 42-46).

Run:

```
pip install -r requirements.txt
python analysis.py
```

Outputs: `results.json` and `figures/macro_f1.png`.

Note: this README was regenerated as a placeholder because the original file was not included in the fix request. The method is unchanged; only the ARFF loading step was modified (see below).

## Fix applied

The previous run failed in `load_data()` because the newer SciPy `arff.loadarff` parser rejects bytes passed through `io.BytesIO`. The downloaded ARFF content is now written to a temporary file and passed to `loadarff` by path, then removed. The dataset, baseline, improved model, split, seeds and evaluation are unchanged.

## Results (latest run)

On the public UCI HAR dataset, a standard MLP stand-in for the residual-SE network reached 80.4% macro-F1 with 10 shots per class, while class-balanced LightGBM reached 81.5%, a small gain within episode noise over 5 episodes.

| Metric | Value |
|---|---|
| baseline_score | 0.8037 |
| improved_score | 0.8148 |
| improvement | 0.01109 |
| baseline_score_std | 0.01273 |
| improved_score_std | 0.03163 |
| baseline_accuracy | 0.8019 |
| improved_accuracy | 0.8206 |
| baseline_balanced_accuracy | 0.8043 |
| improved_balanced_accuracy | 0.8185 |
| train_shots_per_class | 10 |
| n_episodes | 5 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: OpenML 1478 is distributed as one combined table, so the official UCI train/test split is not recoverable; a single stratified 70/30 split (seed 42) is used instead. This is a stand-in dataset, not the paper's road-alert signals, and the MLP is a generic stand-in, not a reproduction of the residual-SE network. Results come from 5 episodes only, so differences may be within noise; std is over episodes.

## Reference

This project is motivated by the following paper:

> TH Kuo, YH Cheng, CP Chen, PC Chen (2024). *Residual Modules Combined with Squeeze-and-Excitation Attention Mechanism for Improving Few-Shot Road Alert Detection Model*. Proceedings of the …, 2024 - aclanthology.org. https://aclanthology.org/2024.rocling-1.13/

Paper author: Prof. Chia-Ping Chen, National Sun Yat-sen University (cpchen@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
