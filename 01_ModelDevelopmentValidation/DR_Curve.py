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

# 只画你要的3个模型
MODELS_TO_USE = [
    "model2_clinical",
    "model4_clinical_baseline_exam",
    "model5_clinical_dynamic_exam",
]

# 公卫语境常用的FPR范围
FPR_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])

OUT_DIR = os.path.join(PRED_DIR, "_dr_curve_data")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Helpers
# =========================
def compute_step_curve(actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """
    计算“阈值从高到低扫过”的阶梯曲线数据：
    返回每个阈值点对应的 FPR / DR (TPR) / threshold
    """
    actual = actual.astype(int)
    pred = pred.astype(float)

    n_pos = int(actual.sum())
    n_neg = int((1 - actual).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both positive and negative samples, got n_pos={n_pos}, n_neg={n_neg}")

    # 按预测概率降序排序
    order = np.argsort(-pred, kind="mergesort")
    pred_s = pred[order]
    act_s = actual[order]

    # 累积TP/FP
    tp_cum = np.cumsum(act_s == 1)
    fp_cum = np.cumsum(act_s == 0)

    # 在每个“唯一阈值”的最后一个位置取点（避免重复阈值导致多点重合）
    # 例如 pred=[0.9,0.9,0.8]，阈值0.9只保留第二个
    change_idx = np.where(np.diff(pred_s) != 0)[0]
    last_idx_for_each_threshold = np.r_[change_idx, len(pred_s) - 1]

    tp = tp_cum[last_idx_for_each_threshold]
    fp = fp_cum[last_idx_for_each_threshold]
    thresholds = pred_s[last_idx_for_each_threshold]

    dr = tp / n_pos                    # detection rate == sensitivity == TPR
    fpr = fp / n_neg

    df = pd.DataFrame(
        {
            "threshold": thresholds,
            "FPR": fpr,
            "DR": dr,
            "TP": tp,
            "FP": fp,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }
    ).sort_values("FPR", kind="mergesort").reset_index(drop=True)

    # 加上起点(0,0)：阈值>max(pred)时没有任何阳性
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


def interpolate_dr_at_fpr(step_df: pd.DataFrame, fpr_grid: np.ndarray) -> pd.DataFrame:
    """
    把阶梯曲线插值/取值到固定的FPR网格上，方便多模型同图对齐。
    这里用“右连续阶梯函数”的取值方式（最符合阈值扫过的筛查解释）：
      DR(FPR*) = 在不超过FPR*的最大可达FPR处的DR
    """
    fpr = step_df["FPR"].to_numpy()
    dr = step_df["DR"].to_numpy()
    thr = step_df["threshold"].to_numpy()

    # 对每个 fpr* 找到 fpr <= fpr* 的最后一个索引
    idx = np.searchsorted(fpr, fpr_grid, side="right") - 1
    idx = np.clip(idx, 0, len(fpr) - 1)

    out = pd.DataFrame(
        {
            "FPR_grid": fpr_grid,
            "DR": dr[idx],
            "threshold": thr[idx],
        }
    )
    return out


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


# =========================
# Main
# =========================
all_step_rows = []
all_grid_rows = []

for zh_name, outcome in OUTCOME_ENG_MAPPING.items():
    for model in MODELS_TO_USE:
        pattern = os.path.join(PRED_DIR, f"{outcome}_{model}_outer5fold_predictions.csv")
        files = glob.glob(pattern)
        if not files:
            print(f"[WARN] Not found: {pattern}")
            continue

        path = files[0]
        df_pred = read_pred_file(path)

        # 用全部OOF行（跨fold合并），计算DR-FPR阶梯曲线
        step_df = compute_step_curve(df_pred["actual"].to_numpy(), df_pred["pred_raw"].to_numpy())
        step_df.insert(0, "disease_zh", zh_name)
        step_df.insert(1, "outcome", outcome)
        step_df.insert(2, "model", model)

        # 网格化（统一FPR刻度）
        grid_df = interpolate_dr_at_fpr(step_df, FPR_GRID)
        grid_df.insert(0, "disease_zh", zh_name)
        grid_df.insert(1, "outcome", outcome)
        grid_df.insert(2, "model", model)

        all_step_rows.append(step_df)
        all_grid_rows.append(grid_df)

# 合并输出
if all_step_rows:
    step_all = pd.concat(all_step_rows, ignore_index=True)
    step_path = os.path.join(OUT_DIR, "dr_curve_step_points.csv")
    step_all.to_csv(step_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved step curve data: {step_path}")
else:
    print("[WARN] No step curve data generated.")

if all_grid_rows:
    grid_all = pd.concat(all_grid_rows, ignore_index=True)
    grid_path = os.path.join(OUT_DIR, "dr_curve_fpr_grid.csv")
    grid_all.to_csv(grid_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved FPR-grid curve data: {grid_path}")
else:
    print("[WARN] No FPR-grid curve data generated.")

print(f"[DONE] Output dir: {OUT_DIR}")
