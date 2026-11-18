# P2: Modelling Lead (DGX Spark)

**You own all heavy compute:** the model zoo, cross-validation, Bayesian optimisation, artifact export and inference timing.

## Mission
Train and compare at least five models from different families, pick the final one on evidence, tune it with Bayesian optimisation, and export an artifact that loads fast on Laptop 1.

## Inputs / Outputs
| Inputs | Outputs |
|---|---|
| `candidates.parquet`, `folds.json`, `splits.json` from P1 | `results/models/<model>_metrics.json`, confusion matrices, `results/models/comparison.csv`, `preds_<model>.parquet` for P3, Optuna DB + trials CSV, `artifacts/` package for P4, training/inference time logs |

## You own (files)
`training/run_model.py`, `training/configs/*.yaml`, `training/bayes_opt.py`, `results/models/`, `results/optuna/`, `artifacts/model.*`, `scaler.*`, `feature_list.json`, `src/metrics.py` (co-owned with P3)

## Timeline
| When (12h example) | Task | Done when |
|---|---|---|
| 0:00-0:40 | Read PDFs. Confirm Spark access, SSH works | Logged in |
| 0:40-2:30 | Spark env, container or venv, `tmux`. Write harness skeleton + `src/metrics.py` with P3 | Harness runs on dummy data |
| 2:30-3:30 | Test harness on P1's 10-trip sample. Train one quick model | Metrics JSON produced |
| 3:30-4:20 | **Send first `preds_*.parquet` to P3 (checkpoint C3)** | P3 confirms format works |
| 4:20-7:20 | Train all 6-7 models, comparison table, plots, per-model notes | `comparison.csv` complete |
| 7:20-9:10 | Pick model, Optuna (50-100 trials), tuned vs untuned, export artifacts | Tuned model beats or matches untuned, or you can explain why not |
| 9:10-10:20 | Test-load artifacts on Laptop 1, time inference there | Timings recorded |
| 10:20-12:00 | Write report sections 3 and 4 | Reviewed by teammate |

## Spark setup
```bash
ssh <user>@<spark-host>
tmux new -s train                     # detach with Ctrl+b d, reattach: tmux attach -t train
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
nvidia-smi                            # confirm GPU visible
```
- The Spark is **ARM64 (aarch64)**. If a wheel is missing, use an NVIDIA NGC container (PyTorch/RAPIDS) and install the rest with pip.
- Move data: `rsync -avP data/processed/ <user>@<spark>:~/pathsnap/data/processed/`
- Pull results back: `rsync -avP <user>@<spark>:~/pathsnap/results/ results/` and commit from a laptop.
- Run long jobs in `tmux` with logs: `python training/run_model.py ... 2>&1 | tee logs/xgb.log`

## Model zoo (>= 5 required, aim for 6-7)
| # | Model | Family | Hardware | Notes |
|---|---|---|---|---|
| 1 | Logistic Regression | Linear | CPU | Baseline, scale features |
| 2 | Gaussian Naive Bayes | Probabilistic | CPU | Poorly calibrated, a good contrast |
| 3 | KNN (scaled) | Instance-based | CPU | Inference slow at scale, subsample reference set if needed |
| 4 | Random Forest / Extra Trees | Ensemble | CPU many-core | |
| 5 | HistGradientBoosting | Boosting | CPU | |
| 6 | XGBoost or LightGBM | Boosting | GPU | Likely top candidate |
| 7 | MLP (PyTorch or sklearn) | Neural | GPU | Epoch-wise loss curves available |
| opt. | SVM (RBF) | Kernel | Subsample | Slow, subsample and say so |

Handle class imbalance (most candidates are 0): `class_weight` / `scale_pos_weight` as a configurable option, and log which setting was used. Compare models under the **same** folds, features and seed.

## Harness spec (`training/run_model.py`)
```
python training/run_model.py --model xgb --config training/configs/xgb.yaml
```
1. Load `candidates.parquet`, `folds.json`, `splits.json`.
2. Keep rows with valid labels. Train partition = 80 trips, student = 10 trips.
3. For each fold: fit scaler on the training folds only, fit the model, predict the held-out fold -> **out-of-fold probabilities**.
4. Fit on all 80 trips, predict student-test.
5. Compute (using `src/metrics.py`): CV macro F1 (mean and std over folds), student Top-1, macro F1, precision, recall, log loss, miss rate (from P1), route continuity of classifier-only argmax route.
6. Time: training time (label "Spark"), inference time per 1000 rows and per trip (label the machine).
7. Write `results/models/<model>_metrics.json`, `<model>_confusion.png`, `preds_<model>.parquet`, and `<model>_history.csv` only if the model has epoch-wise history.
8. Log: seed, config, library versions, git commit hash.

## Bayesian optimisation (`training/bayes_opt.py`)
```python
import optuna
from sklearn.model_selection import GroupKFold

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    # GroupKFold by trip on the 80 training trips -> mean CV macro F1 (or log loss)
    return cv_score(params)

study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage="sqlite:///results/optuna/study.db", study_name="xgb_tuning",
        load_if_exists=True)
study.optimize(objective, n_trials=60)
study.trials_dataframe().to_csv("results/optuna/trials.csv")
```
- Decide and justify the objective (macro F1 vs log loss). Since the HMM uses probabilities, log loss/calibration is worth considering.
- **Never** use student-test or evaluator-test inside the objective. Student-test is only for the final tuned vs untuned comparison.
- Report: search space, ranges, objective, number of trials, best params, tuned vs untuned (CV F1, student Top-1, student F1, log loss, inference time), computational cost.

## Model selection
- Primary metric: choose and justify (suggest CV macro F1 as the headline, with student Top-1, log loss and route continuity as supporting criteria).
- Also weigh: stability across folds (std), probability quality, inference speed, complexity, compatibility with the decoder.
- Never select on training accuracy.

## Export (for Laptop 1)
| Model | Format |
|---|---|
| XGBoost | `model.json` or `.ubj` (`booster.save_model`) |
| LightGBM | text model file |
| sklearn | `joblib` pickle, **same sklearn version as Laptop 1** |
| PyTorch | `state_dict` + model class code, or ONNX |
- Do **not** ship cuML or GPU-only objects.
- Save `scaler.joblib`, `feature_list.json` (ordered), and a thin wrapper `predict_proba(df) -> np.ndarray` in `src/pipeline.py` or `artifacts/`. P4 calls it.
- **Test-load on Laptop 1 CPU** in a fresh venv. Time it there.

## Handoffs
- To P3: `preds_<model>.parquet` (`trip_id, obs_idx, arc_id, prob`), out-of-fold for training trips. Send a quick first version early.
- To P4: the artifact folder + wrapper + a 5-line usage example.
- To P1: feature-importance results so v2 features can be added.

## Definition of done
- [ ] >= 5 models trained with identical folds, results logged
- [ ] `comparison.csv` with all required columns (see report 3.1)
- [ ] Optuna DB and trials CSV committed
- [ ] Tuned vs untuned table
- [ ] Artifact loads and runs on Laptop 1 CPU
- [ ] Seeds, configs, library versions logged

## Pitfalls
- Fitting scalers on all data (leak)
- Reporting in-sample numbers as CV
- Saving a model Laptop 1 cannot load
- Mixing timing from different machines without labels
- Fabricating training curves for models without epochs (**forbidden**)
- Oversized Optuna searches or SVM on millions of rows

## Report sections 3 and 4: what to write
- 3.1: consolidated table (CV macro F1, Top-1, route continuity, student macro F1, log loss, inference time; `N/A` with explanation where not applicable).
- 3.2: for **each model**, 9 questions: why suitable, strengths/limits, hyperparameters and initial values, loss/objective, student-test performance, Top-1/Macro F1/Log Loss, overfitting evidence. Tie answers to your own results.
- 3.3: CV F1 comparison chart, student Top-1 and F1 chart, log-loss chart, confusion matrices, loss curves only where real histories exist.
- 4: the six Bayesian-optimisation questions and the tuned-vs-untuned comparison.
