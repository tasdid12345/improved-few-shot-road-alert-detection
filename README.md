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

## Reference

This project is motivated by the following paper:

> TH Kuo, YH Cheng, CP Chen, PC Chen (2024). *Residual Modules Combined with Squeeze-and-Excitation Attention Mechanism for Improving Few-Shot Road Alert Detection Model*. Proceedings of the …, 2024 - aclanthology.org. https://aclanthology.org/2024.rocling-1.13/

Paper author: Prof. Chia-Ping Chen, National Sun Yat-sen University (cpchen@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
