import os
import glob
import numpy as np
import pandas as pd

# =========================
# Config
# =========================
OUTCOME_ENG_MAPPING = {
    "心肌梗死": "mi",
    "心房颤动和扑动": "afib_flutter",
    "肺心病": "cor_pulmonale",
    "心力衰竭": "chf",
    "中风": "stroke",
    "缺血性中风": "ischemic_stroke",
    "出血性中风": "hemorrhagic_stroke",
    "动脉疾病": "arterial_disease",
    "慢性阻塞性肺疾病": "copd",
    "肝纤维化和肝硬化": "liver_fibrosis_cirrhosis",
    "肝衰竭": "liver_failure",
    "肾衰竭": "renal_failure",
    "糖尿病": "diabetes",
    "甲状腺疾病": "thyroid_disease",
    "帕金森症": "parkinson",
    "全因痴呆症": "dementia",
    "泛癌": "cancer_all",
    "肝癌": "liver_cancer",
    "肺癌": "lung_cancer",
    "肾癌": "kidney_cancer",
}

PRED_DIR = "result/bench_dynamic/xgb/prediction_results/251209_final/outer_cv_predictions"

MODELS_TO_USE = [
    "model2_clinical",
    "model4_clinical_baseline_exam",
    "model5_clinical_dynamic_exam",
]

# 8 points: 5%, 10%, ..., 40%
FPR_GRID = np.arange(0.05, 0.45, 0.05)

OUT_DIR = os.path.join(PRED_DIR, "_lr_curve_data")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Helpers
# =========================
def read_pred_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"eid", "fold", "actual", "pred_raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
    df = df[["eid", "fold", "actual", "pred_raw"]].copy()
    df["actual"] = df["actual"].astype(int)
    df["pred_raw"] = df["pred_raw"].astype(float)
    return df


def compute_step_curve(actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """
    Build threshold-sweep step curve:
    For each unique threshold (descending), compute DR (TPR) and FPR.
    """
    actual = actual.astype(int)
    pred = pred.astype(float)

    n_pos = int(actual.sum())
    n_neg = int((1 - actual).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both classes, got n_pos={n_pos}, n_neg={n_neg}")

    order = np.argsort(-pred, kind="mergesort")
    pred_s = pred[order]
    act_s = actual[order]

    tp_cum = np.cumsum(act_s == 1)
    fp_cum = np.cumsum(act_s == 0)

    change_idx = np.where(np.diff(pred_s) != 0)[0]
    last_idx = np.r_[change_idx, len(pred_s) - 1]

    tp = tp_cum[last_idx]
    fp = fp_cum[last_idx]
    thr = pred_s[last_idx]

    dr = tp / n_pos
    fpr = fp / n_neg

    df = pd.DataFrame(
        {
            "threshold": thr,
            "FPR": fpr,
            "DR": dr,
            "TP": tp,
            "FP": fp,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }
    ).sort_values("FPR", kind="mergesort").reset_index(drop=True)

    # start point (0,0)
    df0 = pd.DataFrame(
        {
            "threshold": [np.inf],
            "FPR": [0.0],
            "DR": [0.0],
            "TP": [0],
            "FP": [0],
            "n_pos": [n_pos],
            "n_neg": [n_neg],
        }
    )
    df = pd.concat([df0, df], ignore_index=True).drop_duplicates(subset=["FPR", "DR"], keep="first")
    return df


def pick_at_fpr_grid(step_df: pd.DataFrame, fpr_grid: np.ndarray) -> pd.DataFrame:
    """
    For each FPR* in grid, pick the right-continuous step value:
      use the last achievable point with FPR <= FPR*
    Return DR, threshold, and actual achieved FPR (<= target).
    """
    fpr = step_df["FPR"].to_numpy()
    dr = step_df["DR"].to_numpy()
    thr = step_df["threshold"].to_numpy()

    idx = np.searchsorted(fpr, fpr_grid, side="right") - 1
    idx = np.clip(idx, 0, len(fpr) - 1)

    out = pd.DataFrame(
        {
            "FPR_target": fpr_grid,
            "FPR_achieved": fpr[idx],
            "DR": dr[idx],
            "threshold": thr[idx],
        }
    )
    return out


# =========================
# Main
# =========================
all_lr_rows = []

for zh_name, outcome in OUTCOME_ENG_MAPPING.items():
    for model in MODELS_TO_USE:
        path = os.path.join(PRED_DIR, f"{outcome}_{model}_outer5fold_predictions.csv")
        if not os.path.exists(path):
            print(f"[WARN] Not found: {path}")
            continue

        df_pred = read_pred_file(path)

        # Step curve from OOF predictions
        step_df = compute_step_curve(df_pred["actual"].to_numpy(), df_pred["pred_raw"].to_numpy())

        # Pick 8 points at fixed FPR targets
        grid_df = pick_at_fpr_grid(step_df, FPR_GRID)

        # Compute LR+ at each target (avoid divide-by-zero)
        # LR+ = DR / FPR_achieved; if achieved FPR == 0 -> LR undefined (set to NaN)
        grid_df["LR_plus"] = np.where(
            grid_df["FPR_achieved"] > 0,
            grid_df["DR"] / grid_df["FPR_achieved"],
            np.nan,
        )

        grid_df.insert(0, "disease_zh", zh_name)
        grid_df.insert(1, "outcome", outcome)
        grid_df.insert(2, "model", model)

        all_lr_rows.append(grid_df)

if all_lr_rows:
    lr_all = pd.concat(all_lr_rows, ignore_index=True)

    # Optional: also store LR at the *target* FPR (some people want DR / FPR_target)
    lr_all["LR_plus_target"] = lr_all["DR"] / lr_all["FPR_target"]

    out_path = os.path.join(OUT_DIR, "lr_curve_fpr_grid.csv")
    lr_all.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved LR grid data: {out_path}")
else:
    print("[WARN] No LR data generated.")

print(f"[DONE] Output dir: {OUT_DIR}")
