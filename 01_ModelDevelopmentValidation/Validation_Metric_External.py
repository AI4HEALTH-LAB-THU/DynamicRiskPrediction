import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib
from tqdm import tqdm


# =========================
# 路径设置
# =========================

# 模型目录
MODELS_DIR = Path("models/bench_dynamic/xgb/trained_models/251209_final")

# Summary 结果表
SUMMARY_CSV = Path(
    "result/bench_dynamic/xgb/prediction_results/251209_final/all_models_summary_20251208_113010.csv"
)

# UKB 外部验证数据目录
UKB_DIR = Path("data/ukb/external_val/year_4")

# 外部验证后更新的 summary 输出路径
UPDATED_SUMMARY_CSV = Path(
    "result/bench_dynamic/xgb/prediction_results/251209_final/all_models_summary_ukb_validation.csv"
)

# 每个模型在 UKB 上的预测概率保存目录
UKB_PRED_DIR = Path(
    "result/bench_dynamic/xgb/prediction_results/251209_final/ukb_predictions"
)


# =========================
# 工具函数
# =========================

def get_pred_proba(model, X):
    """
    从模型中拿预测概率：
    - 优先使用 predict_proba（sklearn 风格 / XGBClassifier）
    - 如果没有 predict_proba，则用 predict（xgboost.Booster 等），假定输出为概率/风险分数
    返回一维 ndarray，长度为 n_samples
    """
    # sklearn/XGBClassifier 风格
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # 二分类：取正类概率
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        # 单列：直接返回
        return np.asarray(proba).ravel()

    # xgboost.Booster 等
    if hasattr(model, "predict"):
        proba = model.predict(X)
        return np.asarray(proba).ravel()

    raise ValueError("Model has neither predict_proba nor predict; cannot obtain probabilities.")


def compute_metric_safe(metric_func, y_true, y_pred):
    """
    安全计算 metric：
    - 如果 y_true 只有一个类别（全 0 / 全 1），则 roc_auc/pr_auc 返回 NaN
    """
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return np.nan
    try:
        return metric_func(y_true, y_pred)
    except Exception:
        return np.nan


def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, random_state=42):
    """
    对 AUROC / AUPRC / Brier 做 bootstrap，返回：
    {
        "roc_auc": ...,
        "roc_auc_ci_lower": ...,
        "roc_auc_ci_upper": ...,
        "pr_auc": ...,
        "pr_auc_ci_lower": ...,
        "pr_auc_ci_upper": ...,
        "brier": ...,
        "brier_ci_lower": ...,
        "brier_ci_upper": ...,
    }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    idx = np.arange(n)
    rng = np.random.RandomState(random_state)

    # 原始样本上的点估计
    roc_auc = compute_metric_safe(roc_auc_score, y_true, y_pred)
    pr_auc = compute_metric_safe(average_precision_score, y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)

    roc_list, pr_list, brier_list = [], [], []

    for _ in range(n_bootstrap):
        sample_idx = rng.choice(idx, size=n, replace=True)
        y_bs = y_true[sample_idx]
        p_bs = y_pred[sample_idx]

        roc_bs = compute_metric_safe(roc_auc_score, y_bs, p_bs)
        pr_bs = compute_metric_safe(average_precision_score, y_bs, p_bs)
        brier_bs = brier_score_loss(y_bs, p_bs)

        roc_list.append(roc_bs)
        pr_list.append(pr_bs)
        brier_list.append(brier_bs)

    roc_array = np.array(roc_list, dtype=float)
    pr_array = np.array(pr_list, dtype=float)
    brier_array = np.array(brier_list, dtype=float)

    # 去掉 NaN（只可能出现在 AUROC/AUPRC 上）
    roc_array = roc_array[~np.isnan(roc_array)]
    pr_array = pr_array[~np.isnan(pr_array)]

    def ci(arr):
        if arr.size == 0:
            return (np.nan, np.nan)
        return np.percentile(arr, [2.5, 97.5])

    roc_ci_lower, roc_ci_upper = ci(roc_array)
    pr_ci_lower, pr_ci_upper = ci(pr_array)
    brier_ci_lower, brier_ci_upper = ci(brier_array)

    return {
        "roc_auc": roc_auc,
        "roc_auc_ci_lower": roc_ci_lower,
        "roc_auc_ci_upper": roc_ci_upper,
        "pr_auc": pr_auc,
        "pr_auc_ci_lower": pr_ci_lower,
        "pr_auc_ci_upper": pr_ci_upper,
        "brier": brier,
        "brier_ci_lower": brier_ci_lower,
        "brier_ci_upper": brier_ci_upper,
    }


# =========================
# 主流程
# =========================

def main():
    # 创建输出目录
    UKB_PRED_DIR.mkdir(parents=True, exist_ok=True)
    UPDATED_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Loading summary table...")
    summary_df = pd.read_csv(SUMMARY_CSV)

    # 预先给 summary_df 添加外部验证指标列
    new_cols = [
        "ext_roc_auc", "ext_roc_auc_ci_lower", "ext_roc_auc_ci_upper",
        "ext_pr_auc", "ext_pr_auc_ci_lower", "ext_pr_auc_ci_upper",
        "ext_brier", "ext_brier_ci_lower", "ext_brier_ci_upper",
        "ext_n_samples"
    ]
    for col in new_cols:
        if col not in summary_df.columns:
            summary_df[col] = np.nan

    # 遍历每一行模型
    for idx, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc="Validating models on UKB"):
        outcome = row["outcome"]           # 例如: 'mi'
        model_type = row["model_type"]     # 例如: 'model5_clinical_dynamic_exam'
        selected_features_str = row["selected_features"]

        # 如果还没跑完/没有 selected_features，跳过
        if pd.isna(selected_features_str) or selected_features_str.strip() == "":
            print(f"[WARN] Row {idx}: selected_features 为空，跳过。")
            continue

        selected_features = [f.strip() for f in selected_features_str.split(";") if f.strip() != ""]

        # 确定模型文件路径
        model_filename = f"{outcome}_{model_type}_final.pkl"
        model_path = MODELS_DIR / model_filename

        if not model_path.exists():
            print(f"[WARN] 模型文件不存在：{model_path}，跳过该行。")
            continue

        # 对应的 UKB feather 文件
        feather_path = UKB_DIR / f"{outcome}.feather"
        if not feather_path.exists():
            print(f"[WARN] UKB 文件不存在：{feather_path}，跳过该行。")
            continue

        # 读入对应 outcome 的 UKB 数据
        try:
            ukb_df = pd.read_feather(feather_path)
        except Exception as e:
            print(f"[ERROR] 读取 UKB feather 失败：{feather_path}，错误：{e}")
            continue

        if "eid" not in ukb_df.columns:
            print(f"[ERROR] {feather_path} 中没有 'eid' 列，无法继续。")
            continue

        # 需要的列：eid + selected_features + outcome label
        needed_cols = ["eid"] + selected_features + [outcome]
        missing_cols = [c for c in needed_cols if c not in ukb_df.columns]
        if missing_cols:
            print(f"[WARN] {feather_path} 中缺少以下列 {missing_cols}，跳过模型 {model_filename}。")
            continue

        df_model = ukb_df[needed_cols].copy()

        # 去掉 outcome 缺失的样本
        df_model = df_model.dropna(subset=[outcome])
        if df_model.shape[0] == 0:
            print(f"[WARN] {feather_path} 中 {outcome} 全缺失，无可用样本，跳过 {model_filename}。")
            continue

        X = df_model[selected_features]
        y = df_model[outcome].astype(int).values
        eids = df_model["eid"].values

        # 加载模型
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"[ERROR] 加载模型失败：{model_path}，错误：{e}")
            continue

        # 预测概率
        try:
            y_pred = get_pred_proba(model, X)
        except Exception as e:
            print(f"[ERROR] 计算预测概率失败：{model_path}，错误：{e}")
            continue

        if len(y_pred) != len(y):
            print(f"[ERROR] 预测长度({len(y_pred)})和标签长度({len(y)})不一致：{model_filename}")
            continue

        # =========================
        # 保存 UKB 上每个样本的预测概率
        # =========================
        pred_df = pd.DataFrame({
            "eid": eids,
            "actual": y,      # 真实 label
            "pr_raw": y_pred  # 预测概率
        })
        pred_filename = f"{outcome}_{model_type}_ukb_predictions.csv"
        pred_path = UKB_PRED_DIR / pred_filename
        pred_df.to_csv(pred_path, index=False)

        # =========================
        # 计算外部验证指标 + 95% CI
        # =========================
        metrics = bootstrap_metrics(y_true=y, y_pred=y_pred, n_bootstrap=1000, random_state=42)

        summary_df.loc[idx, "ext_roc_auc"] = metrics["roc_auc"]
        summary_df.loc[idx, "ext_roc_auc_ci_lower"] = metrics["roc_auc_ci_lower"]
        summary_df.loc[idx, "ext_roc_auc_ci_upper"] = metrics["roc_auc_ci_upper"]

        summary_df.loc[idx, "ext_pr_auc"] = metrics["pr_auc"]
        summary_df.loc[idx, "ext_pr_auc_ci_lower"] = metrics["pr_auc_ci_lower"]
        summary_df.loc[idx, "ext_pr_auc_ci_upper"] = metrics["pr_auc_ci_upper"]

        summary_df.loc[idx, "ext_brier"] = metrics["brier"]
        summary_df.loc[idx, "ext_brier_ci_lower"] = metrics["brier_ci_lower"]
        summary_df.loc[idx, "ext_brier_ci_upper"] = metrics["brier_ci_upper"]

        summary_df.loc[idx, "ext_n_samples"] = len(y)

    # =========================
    # 保存带 UKB 外部验证结果的 summary 表
    # =========================
    summary_df.to_csv(UPDATED_SUMMARY_CSV, index=False)
    print(f"Done. Updated summary saved to: {UPDATED_SUMMARY_CSV}")
    print(f"Per-model UKB predictions saved in: {UKB_PRED_DIR}")


if __name__ == "__main__":
    main()

