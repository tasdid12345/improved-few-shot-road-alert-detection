# Few-Shot Activity Detection: Class-Balanced LightGBM vs. MLP Stand-In

Few-shot detectors for road alerts must learn from very few labelled sensor windows, and the residual-plus-squeeze-and-excitation design proposed in "Residual Modules Combined with Squeeze-and-Excitation Attention Mechanism for Improving Few-Shot Road Alert Detection Model" (2024) is one such approach. Its code and data are not public, so this repository tests a simpler question with an open dataset: whether tuned gradient-boosted trees can outperform a neural-network stand-in when only 10 labelled examples per class are available.

**Question:** Can a class-balanced LightGBM model with cross-validated tuning, averaged over several few-shot episodes, beat a neural-network stand-in for the paper's residual-SE few-shot detector on a public sensor-based activity dataset with only 10 labelled examples per class?

**Dataset:** UCI Human Activity Recognition Using Smartphones (OpenML ID 1478; 561 sensor-derived features; 6 classes), used as a stand-in for road-alert sensor signals. It is not the paper's own dataset.

**Baseline:** scikit-learn `MLPClassifier` on standardized features, standing in for the residual-SE network. It is not the paper's method.

**Improved model:** class-balanced LightGBM, with hyperparameters chosen by 3-fold stratified cross-validation inside each few-shot training set. Both models use the same episodes, scaler, test split and metric.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts in `figures/`).

## Reference

This project is motivated by the following paper:

> TH Kuo, YH Cheng, CP Chen, PC Chen (2024). *Residual Modules Combined with Squeeze-and-Excitation Attention Mechanism for Improving Few-Shot Road Alert Detection Model*. Proceedings of the …, 2024 - aclanthology.org. https://aclanthology.org/2024.rocling-1.13/

Paper author: Prof. Chia-Ping Chen, National Sun Yat-sen University (cpchen@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
