import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

# ======================
# 0. 基本配置
# ======================

DYNAMIC_FEATURES = [
    "mean_BMI",
    "sd_BMI",
    "slope_BMI",
    "annual_ratio_change_BMI",
    "mean_waistcircumference",
    "sd_waistcircumference",
    "slope_waistcircumference",
    "annual_ratio_change_waistcircumference",
    "mean_bpsystolic",
    "sd_bpsystolic",
    "slope_bpsystolic",
    "annual_ratio_change_bpsystolic",
    "mean_bpdiastolic",
    "sd_bpdiastolic",
    "slope_bpdiastolic",
    "annual_ratio_change_bpdiastolic",
    "mean_heartrate",
    "sd_heartrate",
    "slope_heartrate",
    "annual_ratio_change_heartrate",
    "mean_hemoglobin",
    "sd_hemoglobin",
    "slope_hemoglobin",
    "annual_ratio_change_hemoglobin",
    "mean_Wbc",
    "sd_Wbc",
    "slope_Wbc",
    "annual_log_change_Wbc",
    "mean_platelet",
    "sd_platelet",
    "slope_platelet",
    "annual_log_change_platelet",
    "mean_fastingglucosemmol",
    "sd_fastingglucosemmol",
    "slope_fastingglucosemmol",
    "annual_log_change_fastingglucosemmol",
    "mean_ALT",
    "sd_ALT",
    "slope_ALT",
    "annual_log_change_ALT",
    "mean_AST",
    "sd_AST",
    "slope_AST",
    "annual_log_change_AST",
    "mean_totalbilirubin",
    "sd_totalbilirubin",
    "slope_totalbilirubin",
    "annual_log_change_totalbilirubin",
    "mean_creatinine",
    "sd_creatinine",
    "slope_creatinine",
    "annual_log_change_creatinine",
    "mean_serumurea",
    "sd_serumurea",
    "slope_serumurea",
    "annual_ratio_change_serumurea",
    "mean_totalcholesterol",
    "sd_totalcholesterol",
    "slope_totalcholesterol",
    "annual_ratio_change_totalcholesterol",
    "mean_triglycerides",
    "sd_triglycerides",
    "slope_triglycerides",
    "annual_log_change_triglycerides",
    "mean_LDL",
    "sd_LDL",
    "slope_LDL",
    "annual_ratio_change_LDL",
    "mean_HDL",
    "sd_HDL",
    "slope_HDL",
    "annual_ratio_change_HDL",
    "prop_high_sbp",
    "prop_high_dbp",
    "prop_high_glucose",
    "prop_high_TC",
    "prop_high_TG",
    "prop_high_LDL",
    "prop_high_creatinine"
]

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
    "肾癌": "kidney_cancer"
}

MODELS_DIR = Path("models/bench_dynamic/xgb/trained_models/251209_final")
DATA_BASE_DIR = Path("data/train_test_dataset/bench_dynamic_251206")

GLOBAL_SHAP_DIR = Path("result/bench_dynamic/xgb/shap_outputs/global")
CIRCULAR_SHAP_DIR = Path("result/bench_dynamic/xgb/shap_outputs/circular")
GLOBAL_SHAP_DIR.mkdir(parents=True, exist_ok=True)
CIRCULAR_SHAP_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES_PER_DISEASE = 100000
STANDARDIZE_FOR_PLOT = True

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ======================
# 1. 工具函数
# ======================

def load_feature_data_for_disease(outcome_eng: str,
                                  feature_list):
    """
    读取单个疾病的测试集特征：
    data/train_test_dataset/bench_dynamic_251206/{outcome_eng}_test.feather
    并保证所有 dynamic_features 存在，缺失列用 0 填充，NaN 也用 0。
    """
    data_path = DATA_BASE_DIR / f"{outcome_eng}_test.feather"
    if not data_path.exists():
        raise FileNotFoundError(f"Feature file not found: {data_path}")

    df = pd.read_feather(data_path)

    df_features = df.copy()

    # 对不存在的动态特征列，新建并填 0
    for feat in feature_list:
        if feat not in df_features.columns:
            df_features[feat] = 0.0

    # 只保留 dynamic features，按固定顺序
    df_features = df_features[feature_list]

    # 缺失值统一填 0
    df_features = df_features.fillna(0.0)

    return df_features


def load_xgb_model(models_dir: Path, outcome_eng: str):
    """载入某个疾病的 XGBoost 模型."""
    model_path = models_dir / f"{outcome_eng}_model3_dynamic_final.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def get_model_feature_names_from_booster(model, all_features):
    """
    严格以 booster.feature_names 为准。
    - 如果 booster 有特征名，直接用；
    - 如果没有（例如 f0,f1 形式），退而求其次用 all_features[:n_features]
      （这种情况一般说明训练时没用 pandas 列名，而是裸 numpy）。
    """
    booster = model.get_booster()
    booster_fnames = booster.feature_names

    if booster_fnames is not None and len(booster_fnames) > 0:
        return list(booster_fnames)

    # 没有显式特征名，用 f0,f1,... 的情况
    # 这时 booster.num_features 给出特征数，假定顺序和 all_features 一致
    n_feat = int(booster.num_features())
    return list(all_features[:n_feat])


def get_shap_values_for_xgb_builtin(booster: xgb.Booster,
                                    X_model: pd.DataFrame,
                                    feature_names):
    """
    使用 XGBoost 自带的 TreeSHAP (pred_contribs=True) 计算 SHAP 值。
    传入的列顺序 & feature_names 必须和 booster 训练时一致。
    """
    # 确保 X_model 列顺序与 feature_names 完全一致
    X_model = X_model[feature_names].copy()

    dmat = xgb.DMatrix(X_model.values, feature_names=feature_names)

    contribs = booster.predict(dmat, pred_contribs=True)
    # contribs 形状: (n_samples, n_features + 1)，最后一列是 bias

    shap_values = contribs[:, :-1]
    bias = contribs[0, -1]

    return shap_values, bias


def compute_global_shap_df(feature_names, shap_values: np.ndarray):
    """
    计算全局 SHAP 数据（每个特征一个值）。
    """
    abs_shap = np.abs(shap_values)
    mean_abs_shap = abs_shap.mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    shap_std = shap_values.std(axis=0)
    nonzero_frac = (abs_shap > 0).mean(axis=0)

    df_global = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap,
        "shap_std": shap_std,
        "nonzero_fraction": nonzero_frac
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return df_global


def compute_circular_shap_df(X: pd.DataFrame,
                             shap_values: np.ndarray,
                             n_bins: int = 100,
                             standardize_for_plot: bool = True):
    """
    计算环形 SHAP 图所需数据。
    """
    n_samples, n_features = shap_values.shape
    assert X.shape[0] == n_samples
    assert X.shape[1] == n_features

    rows = []
    feature_names = list(X.columns)

    for j, feat in enumerate(feature_names):
        x = X[feat].values.astype(float)
        s = shap_values[:, j].astype(float)

        mask_valid = np.isfinite(x) & np.isfinite(s)
        x_valid = x[mask_valid]
        s_valid = s[mask_valid]

        if x_valid.size == 0:
            continue

        # 几乎常数：只生成一个 bin
        if np.allclose(x_valid.min(), x_valid.max()):
            rows.append({
                "feature": feat,
                "bin_index": 1,
                "bin_low": float(x_valid.min()),
                "bin_high": float(x_valid.max()),
                "mean_shap": float(s_valid.mean()),
                "mean_abs_shap": float(np.abs(s_valid).mean()),
                "mean_feature_value": float(x_valid.mean()),
                "n_samples": int(len(x_valid))
            })
            continue

        # 用分位数切 bin
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(x_valid, quantiles)
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) == 1:
            rows.append({
                "feature": feat,
                "bin_index": 1,
                "bin_low": float(x_valid.min()),
                "bin_high": float(x_valid.max()),
                "mean_shap": float(s_valid.mean()),
                "mean_abs_shap": float(np.abs(s_valid).mean()),
                "mean_feature_value": float(x_valid.mean()),
                "n_samples": int(len(x_valid))
            })
            continue

        for k in range(len(bin_edges) - 1):
            low = bin_edges[k]
            high = bin_edges[k + 1]

            if k < len(bin_edges) - 2:
                mask_bin = (x_valid >= low) & (x_valid < high)
            else:
                mask_bin = (x_valid >= low) & (x_valid <= high)

            if not mask_bin.any():
                continue

            s_bin = s_valid[mask_bin]
            x_bin = x_valid[mask_bin]

            rows.append({
                "feature": feat,
                "bin_index": k + 1,
                "bin_low": float(low),
                "bin_high": float(high),
                "mean_shap": float(s_bin.mean()),
                "mean_abs_shap": float(np.abs(s_bin).mean()),
                "mean_feature_value": float(x_bin.mean()),
                "n_samples": int(mask_bin.sum())
            })

    df_circular = pd.DataFrame(rows)

    if standardize_for_plot and not df_circular.empty:
        def _zscore(group: pd.DataFrame):
            vals = group["mean_feature_value"].values
            mu = vals.mean()
            std = vals.std()
            if std == 0:
                group["mean_feature_value_z"] = 0.0
            else:
                group["mean_feature_value_z"] = (vals - mu) / std
            return group

        df_circular = df_circular.groupby("feature", group_keys=False).apply(_zscore)

    return df_circular


# ======================
# 2. 主流程
# ======================

def main():
    for zh_name, outcome_eng in OUTCOME_ENG_MAPPING.items():
        print(f"\n=== 处理疾病: {zh_name} ({outcome_eng}) ===")

        # 1) 载入该疾病的特征数据（全量 dynamic features，缺列补 0）
        X_all = load_feature_data_for_disease(outcome_eng, DYNAMIC_FEATURES)
        print(f"Loaded features for {outcome_eng}, shape = {X_all.shape}")

        # 2) 载入该疾病模型
        model = load_xgb_model(MODELS_DIR, outcome_eng)
        booster = model.get_booster()
        print(f"Loaded model for {outcome_eng}")

        # 3) 真正训练时的特征名：只认 booster.feature_names
        model_feature_names = get_model_feature_names_from_booster(model, DYNAMIC_FEATURES)
        print(f"Booster feature names (len={len(model_feature_names)}): first 5 = {model_feature_names[:5]}")

        # 确保这些特征在 X_all 里都存在（如果某个训练时的特征现在没有，就补 0 列）
        for feat in model_feature_names:
            if feat not in X_all.columns:
                X_all[feat] = 0.0

        # 只取模型真正用到的特征列，顺序按 model_feature_names
        X_model = X_all[model_feature_names].copy()

        # 4) 抽样
        if MAX_SAMPLES_PER_DISEASE is not None and X_model.shape[0] > MAX_SAMPLES_PER_DISEASE:
            sampled_idx = np.random.choice(X_model.index, size=MAX_SAMPLES_PER_DISEASE, replace=False)
            X_model = X_model.loc[sampled_idx].reset_index(drop=True)
            print(f"Sampled {X_model.shape[0]} rows for SHAP")
        else:
            X_model = X_model.reset_index(drop=True)

        # 5) 计算 XGBoost 内置 SHAP
        print("Computing SHAP values with XGBoost builtin TreeSHAP ...")
        shap_values, expected_value = get_shap_values_for_xgb_builtin(booster, X_model, model_feature_names)
        print(f"SHAP shape: {shap_values.shape}, expected_value (bias): {expected_value}")

        # 6) 全局 SHAP（只对模型用到的特征）
        df_global_model_feats = compute_global_shap_df(model_feature_names, shap_values)

        # 对“没进模型”的特征补 0，方便后续做大热图
        used_set = set(model_feature_names)
        all_rows = [df_global_model_feats]
        for feat in DYNAMIC_FEATURES:
            if feat not in used_set:
                all_rows.append(pd.DataFrame({
                    "feature": [feat],
                    "mean_abs_shap": [0.0],
                    "mean_shap": [0.0],
                    "shap_std": [0.0],
                    "nonzero_fraction": [0.0]
                }))

        df_global_all = pd.concat(all_rows, ignore_index=True)
        df_global_all = df_global_all.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        global_out_path = GLOBAL_SHAP_DIR / f"{outcome_eng}_global_shap.csv"
        df_global_all.to_csv(global_out_path, index=False)
        print(f"Saved global SHAP data to {global_out_path}")

        # 7) 环形 SHAP（只用模型特征）
        df_circular = compute_circular_shap_df(
            X_model,
            shap_values,
            n_bins=300,
            standardize_for_plot=STANDARDIZE_FOR_PLOT
        )
        circular_out_path = CIRCULAR_SHAP_DIR / f"{outcome_eng}_circular_shap.csv"
        df_circular.to_csv(circular_out_path, index=False)
        print(f"Saved circular SHAP data to {circular_out_path}")

    print("\nAll diseases processed.")


if __name__ == "__main__":
    main()
