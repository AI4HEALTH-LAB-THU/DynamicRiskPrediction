import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score
)
import optuna
from optuna.exceptions import TrialPruned
from boruta import BorutaPy

# ======================
# 1. 全局配置
# ======================

SEED = 42
np.random.seed(SEED)

DATA_DIR = "data/train_test_dataset/bench_dynamic_251206"
MODEL_DIR = "models/bench_dynamic/xgb/trained_models/251209_final"
RESULT_DIR = "result/bench_dynamic/xgb/prediction_results/251209_final"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "outer_cv_predictions"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "external_val_predictions"), exist_ok=True)

# 2. 疾病结局英文名映射
outcome_eng_mapping = {
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

# 3. 特征组
# 基础特征
base_features = [
    "age",
    "sex"
]

# 临床风险特征
clinical_features = [
    "smoke_2",
    "smoke_3",
    "drink_2",
    "drink_3",
    "drink_4",
    "hypertension",
    "parent_hyper_1",
    "parent_hyper_2",
    "parent_hyper_3",
    "parent_diabetes_1",
    "parent_diabetes_2",
    "parent_diabetes_3"
]

# 检验检查特征
baseline_exam_features = [
    "BMI",
    "waistcircumference",
    "bpsystolic",
    "bpdiastolic", 
    "heartrate",
    "hemoglobin", 
    "Wbc", 
    "platelet", 
    "fastingglucosemmol",
    "ALT", 
    "AST", 
    "totalbilirubin", 
    "creatinine", 
    "serumurea",
    "totalcholesterol", 
    "triglycerides", 
    "LDL", 
    "HDL"
]

# 纵向特征组
dynamic_features = [
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

# ======================
# 4. 工具函数
# ======================

def bootstrap_ci(
    y_true, 
    y_pred, 
    metric_func, 
    n_boot=1000, 
    confidence=0.95, 
    random_state=SEED
):
    """
    使用Bootstrap方法计算指标的置信区间
    """
    np.random.seed(random_state)
    n = len(y_true)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # 计算原始指标值
    original_metric = metric_func(y_true, y_pred)
    
    # 存储bootstrap样本的指标值
    boot_metrics = []
    for _ in range(n_boot):
        # 有放回抽样
        idx = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]
        
        # 处理可能的极端情况
        try:
            metric = metric_func(y_true_boot, y_pred_boot)
            boot_metrics.append(metric)
        except:
            continue
    
    if not boot_metrics:
        return original_metric, np.nan, np.nan
    
    # 计算置信区间
    boot_metrics = np.array(boot_metrics)
    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_metrics, alpha * 100)
    upper = np.percentile(boot_metrics, (1 - alpha) * 100)
    
    return original_metric, lower, upper


def calculate_ci_normal(
    values, 
    confidence: float = 0.95
):
    """基于正态近似计算均值的置信区间"""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    mean = values.mean()
    std = values.std(ddof=1) if n > 1 else 0.0
    z = stats.norm.ppf((1 + confidence) / 2.0)
    half_width = z * std / np.sqrt(max(n, 1))
    return mean - half_width, mean + half_width


def delong_auc_variance(
    y_true, 
    y_scores
):
    """
    使用 DeLong 方法估计二分类 AUC 及其方差。
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    mask = np.isfinite(y_scores) & np.isfinite(y_true)
    y_true = y_true[mask]
    y_scores = y_scores[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    m = len(pos_scores)
    n = len(neg_scores)
    if m == 0 or n == 0:
        return np.nan, np.nan

    # 构造正负样本的两两比较矩阵
    diff = pos_scores[:, None] - neg_scores[None, :]
    v = (diff > 0).astype(float) + 0.5 * (diff == 0)

    auc = v.mean()
    v_i = v.mean(axis=1)  # 每个阳性样本对应的平均指示值
    w_j = v.mean(axis=0)  # 每个阴性样本对应的平均指示值

    var_auc = v_i.var(ddof=1) / m + w_j.var(ddof=1) / n
    return float(auc), float(var_auc)


def delong_auc_ci(
    y_true, 
    y_scores, 
    confidence: float = 0.95
):
    """
    基于 DeLong 方差 + 正态近似 计算 AUC 的置信区间。
    """
    auc, var_auc = delong_auc_variance(y_true, y_scores)
    if np.isnan(auc) or np.isnan(var_auc) or var_auc <= 0:
        return float(auc), (np.nan, np.nan)

    se = np.sqrt(var_auc)
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    lower = max(0.0, auc - z * se)
    upper = min(1.0, auc + z * se)
    return float(auc), (float(lower), float(upper))


def get_feature_groups(
    df: pd.DataFrame, 
    outcome: str
):
    """返回特征组"""    
    base_features_copy = base_features.copy()
    clinical_features_copy = clinical_features.copy()
    baseline_exam_features_copy = baseline_exam_features.copy()
    dynamic_exam_features_copy = dynamic_features.copy()
    
    features_1 = base_features_copy
    features_2 = base_features_copy + clinical_features_copy
    features_3 = dynamic_exam_features_copy
    features_4 = base_features_copy + clinical_features_copy + baseline_exam_features_copy
    features_5 = base_features_copy + clinical_features_copy + dynamic_exam_features_copy

    def _filter(fs):
        return [f for f in fs if f in df.columns]

    return {
        "model1_base": _filter(features_1),
        "model2_clinical": _filter(features_2),
        "model3_dynamic": _filter(features_3),
        "model4_clinical_baseline_exam": _filter(features_4),
        "model5_clinical_dynamic_exam": _filter(features_5)
    }


def prepare_county_balanced_folds(
    counties, 
    n_folds, 
    df: pd.DataFrame, 
    random_state: int | None = None
):
    """按照区县样本量尽量平衡地分成 n_folds 折"""
    rng = np.random.RandomState(random_state)
    county_sample_counts = df[df["cnty_raw"].isin(counties)].groupby("cnty_raw").size()
    
    # 随机打乱后再按样本量排序，保证一定随机性
    shuffled = list(county_sample_counts.index)
    rng.shuffle(shuffled)
    shuffled = sorted(shuffled, key=lambda c: county_sample_counts[c], reverse=True)

    folds = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for c in shuffled:
        # 分配到当前总样本数最少的折中
        idx = int(np.argmin(fold_sizes))
        folds[idx].append(c)
        fold_sizes[idx] += county_sample_counts[c]
    return folds


def run_boruta(
    train_df, 
    feature_names, 
    label_col, 
    max_iter=80
):
    """使用Boruta方法基于随机森林进行特征选择"""
    features_available = [f for f in feature_names if f in train_df.columns]
    X = train_df[features_available].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y = train_df[label_col].values

    print(f"  [Boruta] 初始特征数: {len(features_available)}")

    if len(features_available) <= 14:
        print("  [Boruta] 特征太少，跳过 Boruta，直接使用原始特征")
        return features_available

    base_est = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        n_jobs=48,
        random_state=SEED,
    )

    boruta = BorutaPy(
        estimator=base_est,
        n_estimators="auto",
        verbose=1,
        random_state=SEED,
        max_iter=max_iter,
        perc=95,
    )

    boruta.fit(X, y)

    support_mask = boruta.support_
    selected = [f for f, s in zip(features_available, support_mask) if s]

    if len(selected) < 6:
        print("  [Boruta] 选出的特征过少，退回使用原始特征集")
        selected = features_available

    print(f"  [Boruta] 选择特征数: {len(selected)}")
    return selected


def optuna_tune_xgb(
    train_df: pd.DataFrame,
    features,
    label_col: str,
    n_trials: int = 20,
    n_folds: int = 5,
    random_state: int = SEED,
):
    """使用Optuna进行XGBoost超参数调优（基于county-based划分）"""
    print(f"  [Optuna] 开始调参: 特征数={len(features)}, n_trials={n_trials}, cv_folds={n_folds}")
    
    # 准备数据
    train_df[features] = train_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    counties = train_df["cnty_raw"].unique().tolist()
    folds = prepare_county_balanced_folds(counties, n_folds, train_df, random_state=random_state)

    def objective(trial: optuna.trial.Trial) -> float:
        # 定义超参数搜索空间
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 15.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        # 使用county-based CV评估
        all_y = []
        all_pred = []
        
        for fold_counties in folds:
            if not fold_counties:
                continue
            
            inner_train = train_df[~train_df["cnty_raw"].isin(fold_counties)]
            inner_val = train_df[train_df["cnty_raw"].isin(fold_counties)]

            X_tr = inner_train[features].values
            y_tr = inner_train[label_col].values
            X_val = inner_val[features].values
            y_val = inner_val[label_col].values

            model = xgb.XGBClassifier(
                objective="binary:logistic",
                scale_pos_weight=1,
                n_jobs=48,
                random_state=random_state,
                **params
            )

            model.fit(X_tr, y_tr)

            # 检查验证集是否包含正负样本
            if len(np.unique(y_val)) < 2:
                continue

            y_pred = model.predict_proba(X_val)[:, 1]
            all_y.append(y_val)
            all_pred.append(y_pred)
        
        # 如果所有折都无效，剪枝该 trial
        if not all_y:
            raise TrialPruned("No valid folds with both classes in validation set.")
        
        # 计算pooled AUC
        y_all = np.concatenate(all_y)
        pred_all = np.concatenate(all_pred)
        try:
            auc = roc_auc_score(y_all, pred_all)
        except Exception as e:
            raise TrialPruned(f"Failed to compute AUC: {e}")
            
        return auc  # 最大化AUC

    # 创建优化研究
    study = optuna.create_study(
        direction="maximize",  # 最大化AUC
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    print("  [Optuna] 正在优化超参数 ...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    print(f"  [Optuna] 完成调参，最优 AUC = {study.best_value:.4f}")
    print(f"  [Optuna] 最优参数: {best_params}")

    return best_params


# ======================
# 5. 核心：嵌套交叉验证 + 模型训练
# ======================

def train_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features,
    label_col: str,
    outcome: str,
    model_type: str,
    valid_counties,
):
    print(f"\n===== 开始训练: {outcome} - {model_type} =====")
    
    # 准备基础数据
    X_train_full = train_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train_full = train_df[label_col].values
    X_test = test_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_test = test_df[label_col].values

    print(f"  训练样本: {len(train_df)}, 测试样本: {len(test_df)}")
    print(f"  训练阳性率: {y_train_full.mean():.4f}, 测试阳性率: {y_test.mean():.4f}")

    # ========== 5*5嵌套交叉验证评估 ==========
    if valid_counties is None or len(valid_counties) < 5:
        print("  有效区县不足，无法进行5折外层CV，跳过该模型。")
        return None

    outer_folds = prepare_county_balanced_folds(valid_counties, 5, train_df, random_state=SEED)

    # 存储外层交叉验证的预测结果
    outer_pred_records = []

    for fold_idx, fold_counties in enumerate(outer_folds):
        print(f"\n  外层折 {fold_idx + 1}/5, 区县数: {len(fold_counties)}")
        outer_test_df = train_df[train_df["cnty_raw"].isin(fold_counties)].copy()
        outer_train_df = train_df[~train_df["cnty_raw"].isin(fold_counties)].copy()

        # Boruta特征选择（在外层训练集上）
        selected_features = run_boruta(outer_train_df, 
                                           features, 
                                           label_col)
        if len(selected_features) < 6:
            print("  Boruta选出的特征过少，退回使用原始特征集")
            selected_features = list(features)

        # 内层Optuna调参（在外层训练集上）
        best_params = optuna_tune_xgb(
            outer_train_df,
            selected_features,
            label_col,
            n_trials=20,
            n_folds=5,
            random_state=SEED + fold_idx,
        )

        # 在外层训练集上训练最终模型
        model_outer = xgb.XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=1,
            eval_metric="logloss",
            n_jobs=48,
            random_state=SEED + fold_idx,** best_params
        )
        X_outer_train_sel = outer_train_df[selected_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        model_outer.fit(X_outer_train_sel, outer_train_df[label_col].values)

        # 外层测试集预测
        X_outer_test_sel = outer_test_df[selected_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_outer_test = outer_test_df[label_col].values
        y_pred_raw = model_outer.predict_proba(X_outer_test_sel)[:, 1]

        # 评测指标
        if len(np.unique(y_outer_test)) >= 2:
            roc = roc_auc_score(y_outer_test, y_pred_raw)
            pr = average_precision_score(y_outer_test, y_pred_raw)
        else:
            roc = np.nan
            pr = np.nan
        brier = brier_score_loss(y_outer_test, y_pred_raw)

        print(f"    外层折 {fold_idx+1} ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}, Brier={brier:.4f}")

        # 保存该折的样本预测
        for eid, yt, pr_raw in zip(
            outer_test_df["eid"].values, y_outer_test, y_pred_raw
        ):
            outer_pred_records.append(
                {
                    "eid": eid,
                    "fold": fold_idx + 1,
                    "actual": int(yt),
                    "pred_raw": float(pr_raw),
                }
            )

    # 基于外层5折所有样本的预测计算指标
    outer_pred_df = pd.DataFrame(outer_pred_records)
    
    cv_roc_mean, cv_roc_ci = np.nan, (np.nan, np.nan)
    cv_pr_mean, cv_pr_ci = np.nan, (np.nan, np.nan)
    cv_brier_mean, cv_brier_ci = np.nan, (np.nan, np.nan)

    if len(outer_pred_df) > 0 and len(np.unique(outer_pred_df["actual"])) >= 2:
        # ROC-AUC使用Delong
        cv_roc_mean, cv_roc_ci = delong_auc_ci(
            outer_pred_df["actual"].values,
            outer_pred_df["pred_raw"].values,
            confidence=0.95,
        )
        
        # PR-AUC使用Bootstrap
        cv_pr_mean, cv_pr_low, cv_pr_up = bootstrap_ci(
            outer_pred_df["actual"].values,
            outer_pred_df["pred_raw"].values,
            average_precision_score,
            n_boot=100,
            confidence=0.95,
            random_state=SEED
        )
        cv_pr_ci = (cv_pr_low, cv_pr_up)
        
        # Brier Score使用Bootstrap
        cv_brier_mean, cv_brier_low, cv_brier_up = bootstrap_ci(
            outer_pred_df["actual"].values,
            outer_pred_df["pred_raw"].values,
            brier_score_loss,
            n_boot=100,
            confidence=0.95,
            random_state=SEED
        )
        cv_brier_ci = (cv_brier_low, cv_brier_up)

    print("\n  外层 5 折 CV 结果：")
    print(f"    ROC-AUC={cv_roc_mean:.4f} ({cv_roc_ci[0]:.4f}-{cv_roc_ci[1]:.4f})")
    print(f"    PR-AUC={cv_pr_mean:.4f} ({cv_pr_ci[0]:.4f}-{cv_pr_ci[1]:.4f})")
    print(f"    Brier={cv_brier_mean:.4f} ({cv_brier_ci[0]:.4f}-{cv_brier_ci[1]:.4f})")
    
    outer_pred_path = os.path.join(
        RESULT_DIR, "outer_cv_predictions", f"{outcome}_{model_type}_outer5fold_predictions.csv"
    )
    outer_pred_df.to_csv(outer_pred_path, index=False)
    print(f"  外层 5 折预测结果已保存: {outer_pred_path}")


    # ========== 使用全部训练集重新训练最终模型，并在独立测试集上评估 ==========
    print("\n  使用全部训练集构建最终模型并在测试集验证...")

    # 全训练集上再做一次Boruta特征选择
    final_selected_features = run_boruta(train_df, 
                                             features, 
                                             label_col)
    if len(final_selected_features) < 6:
        print("  Boruta选出的特征过少，退回使用原始特征集")
        final_selected_features = list(features)

    # 最终模型超参数调优
    best_params_final = optuna_tune_xgb(
        train_df,
        final_selected_features,
        label_col,
        n_trials=20,
        n_folds=5,
        random_state=SEED + 999,
    )

    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=1,
        eval_metric="logloss",
        n_jobs=48,
        random_state=SEED,
        **best_params_final
    )

    X_train_final = train_df[final_selected_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test_final = test_df[final_selected_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    final_model.fit(X_train_final, y_train_full)
    
    y_test_pred_raw = final_model.predict_proba(X_test_final)[:, 1]

    test_df_with_pred = test_df.copy()
    test_df_with_pred["pred_raw"] = y_test_pred_raw
    
    # 计算测试集上的指标置信区间
    test_roc_mean, test_roc_ci = np.nan, (np.nan, np.nan)
    test_pr_mean, test_pr_ci = np.nan, (np.nan, np.nan)
    test_brier_mean, test_brier_ci = np.nan, (np.nan, np.nan)
    
    if len(test_df_with_pred) > 0 and len(np.unique(test_df_with_pred[label_col])) >= 2:
        # ROC-AUC使用DeLong方法
        test_roc_mean, test_roc_ci = delong_auc_ci(
            test_df_with_pred[label_col].values,
            test_df_with_pred["pred_raw"].values,
            confidence=0.95,
        )
        
        # PR-AUC使用Bootstrap
        test_pr_mean, test_pr_low, test_pr_up = bootstrap_ci(
            test_df_with_pred[label_col].values,
            test_df_with_pred["pred_raw"].values,
            average_precision_score,
            n_boot=100,
            confidence=0.95,
            random_state=SEED
        )
        test_pr_ci = (test_pr_low, test_pr_up)
        
        # Brier Score使用Bootstrap
        test_brier_mean, test_brier_low, test_brier_up = bootstrap_ci(
            test_df_with_pred[label_col].values,
            test_df_with_pred["pred_raw"].values,
            brier_score_loss,
            n_boot=100,
            confidence=0.95,
            random_state=SEED
        )
        test_brier_ci = (test_brier_low, test_brier_up)

    print("  [外部验证集] 评测指标（7个区县）：")
    print(f"    ROC-AUC (均值±95%CI): {test_roc_mean:.4f} ({test_roc_ci[0]:.4f} - {test_roc_ci[1]:.4f})")
    print(f"    PR-AUC (均值±95%CI): {test_pr_mean:.4f} ({test_pr_ci[0]:.4f} - {test_pr_ci[1]:.4f})")
    print(f"    Brier   (均值±95%CI): {test_brier_mean:.4f} ({test_brier_ci[0]:.4f} - {test_brier_ci[1]:.4f})")

    # 保存最终模型
    model_path = os.path.join(MODEL_DIR, f"{outcome}_{model_type}_final.pkl")
    joblib.dump(final_model, model_path)
    print(f"  最终模型已保存到: {model_path}")

    # 保存测试集预测结果
    test_pred_df = pd.DataFrame(
        {
            "eid": test_df["eid"].values,
            "cnty_raw": test_df["cnty_raw"].values,
            "actual": y_test,
            "pred_raw": y_test_pred_raw,
        }
    )
    
    test_pred_path = os.path.join(RESULT_DIR, "external_val_predictions", f"{outcome}_{model_type}_test_predictions.csv")
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"  测试集预测结果已保存: {test_pred_path}")

    
    # ========== 返回要保存的最终结果 ==========
    return {
        "outcome": outcome,
        "model_type": model_type,
        "cv_roc_auc_mean": cv_roc_mean,
        "cv_roc_auc_ci_lower": cv_roc_ci[0],
        "cv_roc_auc_ci_upper": cv_roc_ci[1],
        "cv_pr_auc_mean": cv_pr_mean,
        "cv_pr_auc_ci_lower": cv_pr_ci[0],
        "cv_pr_auc_ci_upper": cv_pr_ci[1],
        "cv_brier_mean": cv_brier_mean,
        "cv_brier_ci_lower": cv_brier_ci[0],
        "cv_brier_ci_upper": cv_brier_ci[1],
        "test_roc_auc": test_roc_mean,
        "test_roc_auc_ci_lower": test_roc_ci[0],
        "test_roc_auc_ci_upper": test_roc_ci[1],
        "test_pr_auc": test_pr_mean,
        "test_pr_auc_ci_lower": test_pr_ci[0],
        "test_pr_auc_ci_upper": test_pr_ci[1],
        "test_brier": test_brier_mean,
        "test_brier_ci_lower": test_brier_ci[0],
        "test_brier_ci_upper": test_brier_ci[1],
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_positive_rate": y_train_full.mean(),
        "test_positive_rate": y_test.mean(),
        "train_baseline_brier_score": y_train_full.mean()*(1-y_train_full.mean()),
        "test_baseline_brier_score": y_test.mean()*(1-y_test.mean()),
        "selected_features": ";".join(final_selected_features),
        "n_selected_features": len(final_selected_features),
        "n_original_features": len(features),
        "valid_counties": len(valid_counties),
        "test_counties": len(test_df["cnty_raw"].unique()),
        "best_params": str(best_params_final),
    }


# ======================
# 6. 主程序
# ======================

def main():
    print("===== 开始 20 种疾病的 XGBoost 嵌套交叉验证建模 =====")
    
    # 初始化结果文件路径和表头
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULT_DIR, f"all_models_summary_{ts}.csv")
    
    # 定义所有字段
    columns = [
        "outcome", "model_type",
        "cv_roc_auc_mean", "cv_roc_auc_ci_lower", "cv_roc_auc_ci_upper",
        "cv_pr_auc_mean", "cv_pr_auc_ci_lower", "cv_pr_auc_ci_upper",
        "cv_brier_mean", "cv_brier_ci_lower", "cv_brier_ci_upper",
        "test_roc_auc", "test_roc_auc_ci_lower", "test_roc_auc_ci_upper",
        "test_pr_auc", "test_pr_auc_ci_lower", "test_pr_auc_ci_upper",
        "test_brier", "test_brier_ci_lower", "test_brier_ci_upper",
        "train_samples", "test_samples", "train_positive_rate", "test_positive_rate",
        "train_baseline_brier_score", "test_baseline_brier_score",
        "selected_features", "n_selected_features", "n_original_features",
        "valid_counties", "test_counties", "best_params",
    ]
    
    # 创建文件并写入表头
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")  # 表头行
    
    outcomes = list(outcome_eng_mapping.values())
    print(f"  共 {len(outcomes)} 个疾病结局")
    print(f"  结果将流式写入: {out_path}")

    for outcome in outcomes:
        print(f"\n==== 处理疾病: {outcome} ====")
        train_path = os.path.join(DATA_DIR, f"{outcome}_train.feather")
        test_path = os.path.join(DATA_DIR, f"{outcome}_test.feather")
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            print("  训练/测试文件不存在，跳过。")
            continue

        train_df = pd.read_feather(train_path)
        test_df = pd.read_feather(test_path)
        print(f"  年龄筛选前 - 训练样本: {len(train_df)}, 测试样本: {len(test_df)}")
        
        # 年龄筛选逻辑（30-75岁之间，包含30和75岁）
        train_df = train_df[(train_df["age"] >= 30) & (train_df["age"] <= 75)].copy()
        test_df = test_df[(test_df["age"] >= 30) & (test_df["age"] <= 75)].copy()
        print(f"  年龄筛选后 - 训练样本: {len(train_df)}, 测试样本: {len(test_df)}")

        # 检查标签列
        label_col = f"{outcome}"
        if label_col not in train_df.columns:
            print(f"  未找到标签列 {label_col}，跳过。")
            continue

        # 转换标签为数值型
        train_df[label_col] = train_df[label_col].astype(int)
        test_df[label_col] = test_df[label_col].astype(int)
        
        # 筛选样本数 ≥ 1000 的区县
        if "cnty_raw" not in train_df.columns:
            print("  未找到区县列 cnty_raw，跳过。")
            continue

        county_counts = train_df.groupby("cnty_raw").size()
        valid_counties = county_counts[county_counts >= 1000].index.tolist()
        print(f"  区县样本数 ≥1000 的个数: {len(valid_counties)}")
        if len(valid_counties) < 5:
            print("  有效区县不足（<5），无法进行5折外层CV，跳过该疾病。")
            continue
        
        # 统计筛掉小区县之后剩余的样本数量
        remaining_samples = train_df[train_df["cnty_raw"].isin(valid_counties)].shape[0]
        print(f"  筛选后（区县样本数≥1000）剩余训练样本总数: {remaining_samples}")
        train_df = train_df[train_df["cnty_raw"].isin(valid_counties)].copy()
        
        # 获取特征组并训练
        feature_groups = get_feature_groups(train_df, outcome)
        for model_type, features in feature_groups.items():
            if not features:
                print(f"  {model_type} 无可用特征，跳过。")
                continue
            res = train_evaluate_model(
                train_df=train_df,
                test_df=test_df,
                features=features,
                label_col=label_col,
                outcome=outcome,
                model_type=model_type,
                valid_counties=valid_counties,
            )
            if res is not None:
                # 流式写入当前模型结果
                with open(out_path, "a", encoding="utf-8") as f:
                    row = [
                        str(res[col]).replace(",", ";")
                        for col in columns
                    ]
                    f.write(",".join(row) + "\n")
                print(f"  已写入 {outcome} - {model_type} 的结果到 {out_path}")

    print(f"\n===== 所有模型结果已流式写入: {out_path} =====")


if __name__ == "__main__":
    main()
